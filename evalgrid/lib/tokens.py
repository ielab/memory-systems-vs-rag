"""
evalgrid/lib/tokens.py
----------------------
Token counting and the fixed answering-token budget (``--context-budget``).

Why this exists. Matching two systems at the same top-k does not match them at
the same amount of evidence: a `pair_turns` doc is ~2x an `all_turns` doc, a
LightMem memory is a fraction of either, and a hypermem fact is one sentence.
The old matched-token rows were built by hand-pairing k values (LightMem
top-6/10/20 against Naive RAG top-3/5/10) and labelling the cell with the mean
context length that pairing happened to produce. That works for exactly two
corpora whose mean unit length sits at 2:1; it does not generalise to six
systems.

A budget generalises: fill the prompt in rank order until the next fact would
cross T tokens, and every system is compared at the same evidence volume
whatever its unit size.

Three decisions the numbers depend on, all deliberate:

* **Append in rank order while whole facts fit, and never skip-and-continue.**
  Pulling in a later short fact past one that did not fit would break the
  rank-prefix property: the context would no longer be "the top of the ranking",
  and short-unit systems would silently be read to a greater depth than
  long-unit ones. What lands in the prompt is always a prefix of the ranking.
* **When nothing has been kept yet and the head fact alone exceeds T, CUT IT to
  T and keep the fragment.** The two alternatives are both worse. Admitting the
  fact whole reports 2,537 tokens under a 200-token label, which is not a
  matched cell; dropping it answers from an empty context, which measures
  nothing and scores as noise. The cut is not hidden — ``context_unit_truncated``
  flags it and ``context_truncated_tokens`` says how much was dropped — and it
  is what makes small budgets measurable for the coarse systems at all
  (MemoryBank's median unit is 461 tokens, its p90 717).

  A truncated fact is a fragment: for MemoryBank it is a mid-conversation cut,
  and the answer-bearing turn may lie past it. That is a real cost, and it is
  the opposite trade from overflow — comparable in tokens, possibly destroyed as
  evidence. The flag is what lets a reader see the price, so a row whose system
  is truncated on most queries must be reported as fragment-fed, not averaged in
  silently. Truncation needs a ``truncate`` callable; without one this falls
  back to the old keep-whole behaviour and says so with ``context_overflow``.
* **Once something has been kept, stop — do not fragment to top up.** A partial
  fact appended after whole facts adds noise rather than evidence, and the
  residual is bounded by one unit size. Realised tokens are recorded per
  question, so an under-fill of a few percent is measured rather than assumed.
* **Count the rendered fact string**, i.e. what ``answer.format_fact`` produced,
  timestamp prefix included — that is what the generator actually pays for.
  Counting raw ``doc["text"]`` would charge each system differently depending on
  whether its units carry timestamps. The prompt template and the question are
  excluded: they are identical across systems, so they shift every row by the
  same constant.

Cutting on a MEMORY boundary rather than a token one is the caller's job, not
this module's: ``pipeline/answer.install_replay`` splits MemoryBank's blocks
into their member memories before budgeting, so what arrives here is already one
memory per element and a token cut only happens when a single memory exceeds T.

The tokenizer is the ANSWERING model's, not a generic one — a budget expressed
in tokens some other model would have counted is not a budget.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

#: Tokens charged for the newline joining two facts in the prompt
#: (``shared_answer_prompt`` joins with "\n"). Counting per fact and adding one
#: separator is within a token or two of tokenising the joined string — BPE
#: merges rarely cross a newline — and it keeps fitting O(n) with a per-doc
#: memo, which matters because the full-context and oracle paths would otherwise
#: re-tokenise the whole corpus once per question.
SEPARATOR_TOKENS = 1


@lru_cache(maxsize=8)
def _hf_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


@lru_cache(maxsize=8)
def _tiktoken_encoding(model: str):
    import tiktoken
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown OpenAI model name — o200k_base is what the current families
        # use. Named explicitly rather than guessed per call.
        return tiktoken.get_encoding("o200k_base")


def _hf_tokenizer_or_exit(model: str):
    """``_hf_tokenizer`` with the actionable failure message.

    Failure to load raises rather than falling back to a word-count heuristic:
    a budget counted with the wrong tokenizer is wrong by a systematic factor
    that no downstream reader could detect.
    """
    try:
        return _hf_tokenizer(model)
    except Exception as e:  # network-off nodes, gated repos, typos
        raise SystemExit(
            f"Could not load a tokenizer for {model!r}: {e}\n"
            f"--context-budget needs the answering model's own tokenizer. On an "
            f"offline machine, warm the HF cache first:\n"
            f"  python -c \"from transformers import AutoTokenizer; "
            f"AutoTokenizer.from_pretrained('{model}')\""
        )


@lru_cache(maxsize=8)
def token_counter(model: str) -> Callable[[str], int]:
    """Return ``text -> token count`` under ``model``'s own tokenizer.

    A model id containing "/" is treated as a HuggingFace repo and loaded with
    AutoTokenizer (Qwen/Qwen3-30B-A3B-Instruct-2507, ...); anything else goes to
    tiktoken (gpt-4o-mini, gpt-5.5, ...).

    Special-token markers in the text are counted as ORDINARY TEXT
    (``disallowed_special=()``). tiktoken defaults to raising on a literal
    "<|endoftext|>", which is the right default for a prompt template and the
    wrong one for benchmark content: longmemeval_s_cleaned.json contains that
    string inside a conversation turn, and the default turns counting the turn
    into a ValueError several frames below the caller. The HF branch already
    behaves this way via ``add_special_tokens=False``.
    """
    if "/" in model:
        tok = _hf_tokenizer_or_exit(model)
        return lambda text: len(tok.encode(text, add_special_tokens=False))
    enc = _tiktoken_encoding(model)
    return lambda text: len(enc.encode(text, disallowed_special=()))


@lru_cache(maxsize=8)
def _counter_or_none(model: str) -> Callable[[str], int] | None:
    # Cached either way, so a node without the tokenizer tries to load it once,
    # not once per call.
    try:
        return token_counter(model)
    except (Exception, SystemExit):
        return None


def try_count_tokens(model: str, text: str) -> int | None:
    """``token_counter(model)(text)``, or None when no tokenizer can be loaded.

    For bookkeeping that must never stop a run. ``token_counter`` exits on
    failure, which is right for a budget and wrong for a statistic.
    """
    count = _counter_or_none(model)
    return count(text) if count is not None else None


@lru_cache(maxsize=8)
def prompt_opens_think(model: str) -> bool | None:
    """Whether ``model``'s chat template ends the generation prompt inside ``<think>``.

    True for a thinking model like Qwen3.8-27B, whose completion then carries
    only ``</think>``. The generator needs this to tell a completion that was cut
    off mid-reasoning from a plain answer (models/generators/vllm_gen.py,
    ``split_reasoning``). None when it cannot be known: an OpenAI model, or no
    tokenizer or chat template available. Never raises, because a caller must
    still be able to generate without it.
    """
    if "/" not in model:
        return None
    try:
        tok = _hf_tokenizer(model)
        prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                         tokenize=False, add_generation_prompt=True)
    except Exception:
        return None
    return prompt.rstrip().endswith("<think>")


@lru_cache(maxsize=8)
def token_truncator(model: str) -> Callable[[str, int], tuple[str, int]]:
    """Return ``(text, n) -> (prefix, its real token count)`` under ``model``.

    The companion to :func:`token_counter`, used by :func:`fit_to_budget` for
    rule 2 — cut the head fact rather than overflow the budget with it.

    **It returns the realised count rather than ``n``, because decode(encode(x))
    is not a round trip.** Cutting a token list mid-word and decoding it can
    re-encode to a different length, so a cell that assumed ``n`` would report a
    budget it did not actually deliver. One extra encode of at most ``n`` tokens,
    at most once per question, buys an exact number.
    """
    if "/" in model:
        tok = _hf_tokenizer_or_exit(model)

        def cut(text: str, n: int) -> tuple[str, int]:
            prefix = tok.decode(tok.encode(text, add_special_tokens=False)[:n],
                                skip_special_tokens=True)
            return prefix, len(tok.encode(prefix, add_special_tokens=False))
        return cut

    enc = _tiktoken_encoding(model)

    def cut(text: str, n: int) -> tuple[str, int]:
        prefix = enc.decode(enc.encode(text, disallowed_special=())[:n])
        return prefix, len(enc.encode(prefix, disallowed_special=()))
    return cut


def fit_to_budget(facts: list[str], sizes: list[int], budget: int | None,
                  *, truncate: Callable[[str, int], tuple[str, int]] | None = None,
                  ) -> tuple[list[str], dict]:
    """Cut ``facts`` (in rank order, with their token ``sizes``) to ``budget``.

    ``budget=None`` keeps every fact and only measures it, so a plain top-k run
    still records ``context_tokens`` and can be placed on the same cost axis as
    a budget run.

    ``truncate`` is :func:`token_truncator` for the answering model. It is what
    implements rule 2 of the module docstring — cutting a head fact that alone
    exceeds ``budget`` instead of admitting it whole. Pass it whenever a budget
    is passed; the fallback exists only so a caller that has no tokenizer at
    hand degrades LOUDLY (``context_overflow``) rather than silently.

    Returns ``(kept_facts, stats)``. The stats are the fields that make a budget
    cell auditable — without them "matched at 500 tokens" is a claim, not a
    measurement:

    ===========================  ==============================================
    ``context_items``            facts that reached the prompt
    ``context_tokens``           their token total, separators included
    ``context_budget``           T (absent when unbudgeted)
    ``context_truncated``        the ranking had more, and it was cut
    ``context_underfilled``      the ranking ran out with room left — this
                                 system could not fill T, which is a result
                                 about the system, not a bug
    ``context_unit_truncated``   the head fact alone exceeded T and was CUT to
                                 it; what reached the prompt is a fragment
    ``context_truncated_tokens`` how many tokens that cut dropped
    ``context_overflow``         no ``truncate`` was available, so the head fact
                                 was kept whole and the cell is over budget.
                                 Never set when ``truncate`` is passed
    ===========================  ==============================================
    """
    if budget is None:
        total = sum(sizes) + SEPARATOR_TOKENS * max(0, len(facts) - 1)
        return list(facts), {"context_items": len(facts), "context_tokens": total}

    kept: list[str] = []
    total = 0
    unit_truncated = dropped = 0
    overflow = False
    for fact, size in zip(facts, sizes):
        # Rule 2, and it can only ever fire on the first fact: once something is
        # kept, rule 3 stops the fill instead of fragmenting to top up.
        if not kept and size > budget:
            if truncate is None:
                kept.append(fact)
                total = size
                overflow = True
            else:
                prefix, realised = truncate(fact, budget)
                kept.append(prefix)
                total = realised
                unit_truncated = 1
                dropped = size - realised
                # decode(encode(x)) is not a round trip, so a cut can in
                # principle re-encode LONGER than the budget it was cut to
                # (an incomplete UTF-8 sequence at the boundary is the way it
                # happens). Vanishingly rare, and marked rather than hidden if
                # it does: an over-budget cell is not a matched cell.
                overflow = realised > budget
            break
        step = size + (SEPARATOR_TOKENS if kept else 0)
        # Rule 1: stop, never skip-and-continue — what lands in the prompt is
        # always a prefix of the ranking.
        if kept and total + step > budget:
            break
        kept.append(fact)
        total += step

    stats = {
        "context_items": len(kept),
        "context_tokens": total,
        "context_budget": budget,
        "context_truncated": len(kept) < len(facts),
        # A cut head fact fills the budget exactly, so it is never also an
        # under-fill; the `not unit_truncated` guard keeps the two disjoint even
        # if a tokenizer round-trip lands a token or two short.
        "context_underfilled": (len(kept) == len(facts) and total < budget
                                and not unit_truncated),
        "context_unit_truncated": bool(unit_truncated),
    }
    if unit_truncated:
        stats["context_truncated_tokens"] = dropped
    if overflow:
        stats["context_overflow"] = True
    return kept, stats


def fit_to_window(facts: list[str], sizes: list[int], window: int,
                  *, keep: str = "top",
                  truncate: Callable[[str, int], tuple[str, int]] | None = None,
                  ) -> tuple[list[str], dict]:
    """Cut ``facts`` to a serving WINDOW — a hard cap, not a comparison axis.

    This is the companion to :func:`fit_to_budget` and the two are not the same
    thing, which is why they are two functions:

    ``fit_to_budget``   a chosen experimental condition. It defines the cell
                        ("every system at 500 tokens of evidence"), it names the
                        output directory (``budget<T>/``), and it is meant to
                        bind on every question.
    ``fit_to_window``   a property of the SERVER. The generator is up at some
                        ``--max-model-len``; a prompt past it is not a smaller
                        answer, it is an HTTP 400 and a hole in the results
                        table. It binds only when the prompt would not fit, and
                        it does not name the directory, because it is not what
                        the run is about.

    The row it exists for is the full-context baseline. A rendered ``all_turns``
    LongMemEval-S haystack measures ~110k tokens (o200k) / ~112k (cl100k), which
    clears a 131,072-token window with ~15k to spare once the answer reservation
    is subtracted, and overflows a 65,536 one by ~50k. So at the window this repo
    serves at, this function is a GUARD that should never fire; at the documented
    fallback window it halves every full-context prompt. Either way the outcome is
    now deterministic and counted, where before an over-long prompt failed with a
    400 and was written to errors.jsonl — a silently shortened denominator, which
    is worse than a stated cut.

    ``keep`` says WHICH END SURVIVES, and it must follow the order the caller's
    list is in — this is the whole correctness content of the function:

    ``"top"``     (default) the list is a RANKING; drop from the BACK. The
                  lowest-ranked facts go first, so what reaches the prompt stays
                  a prefix of the ranking, exactly as ``fit_to_budget`` leaves
                  it. Used for every retrieval path and for oracle.
    ``"newest"``  the list is CHRONOLOGICAL; drop from the FRONT. The oldest
                  turns go first and the most recent survive. Used for
                  ``--retriever full``, for two reasons: it is what a real agent
                  with a finite window does (eviction is oldest-first), and
                  LongMemEval's knowledge-update questions are answered by the
                  LATEST mention of a fact, so dropping the newest end would
                  systematically destroy the evidence for a whole question type
                  while dropping the oldest merely loses coverage.

    Neither end is free, and the cost is reported rather than assumed:
    ``context_dropped_items`` / ``context_dropped_tokens`` say how much of the
    conversation never reached the prompt, per question. A full-context row
    where those are non-zero on most questions is a TRUNCATED full-context row
    and has to be labelled as one — it is no longer an upper bound on "the whole
    haystack in the window".

    ``window`` is the token budget for the FACTS ALONE. The caller subtracts the
    prompt template, the question and the answer reservation before calling —
    see ``answer._facts_window``.

    Returns ``(kept_facts, stats)``, whose keys are merged over
    :func:`fit_to_budget`'s by the caller. ``context_items`` and
    ``context_tokens`` are recomputed here and REPLACE the budget pass's values;
    ``context_unit_truncated`` / ``context_truncated_tokens`` are shared with
    that pass (they can only be set by whichever pass actually cut a unit, and a
    window cut is applied second, so its value wins).
    """
    if keep not in ("top", "newest"):
        raise ValueError(f"keep must be 'top' or 'newest', got {keep!r}")
    total = sum(sizes) + SEPARATOR_TOKENS * max(0, len(facts) - 1)
    if total <= window:
        # The common case, including every top-k run: measure, change nothing.
        return list(facts), {"context_items": len(facts), "context_tokens": total,
                             "context_window": window, "context_window_truncated": False,
                             "context_dropped_items": 0, "context_dropped_tokens": 0}

    # Walk from the surviving end inward, keeping whole facts while they fit.
    order = range(len(facts) - 1, -1, -1) if keep == "newest" else range(len(facts))
    kept_idx: list[int] = []
    kept_tokens = 0
    for i in order:
        step = sizes[i] + (SEPARATOR_TOKENS if kept_idx else 0)
        if kept_idx and kept_tokens + step > window:
            break
        if not kept_idx and sizes[i] > window:
            # One fact alone is bigger than the whole window. Only reachable
            # with a pathological corpus (a single enormous document) or an
            # absurdly small window, but answering from nothing measures
            # nothing, so cut it rather than drop it — same trade as
            # fit_to_budget's rule 2, and flagged the same way.
            stats = {"context_window": window, "context_window_truncated": True,
                     "context_dropped_items": len(facts) - 1}
            if truncate is None:
                # No tokenizer to cut with: keep it whole and say so. This
                # prompt WILL be rejected by the server — context_overflow is
                # the record of why, in the same field fit_to_budget uses.
                stats.update({"context_items": 1, "context_tokens": sizes[i],
                              "context_dropped_tokens": total - sizes[i],
                              "context_overflow": True})
                return [facts[i]], stats
            prefix, realised = truncate(facts[i], window)
            stats.update({"context_items": 1, "context_tokens": realised,
                          "context_dropped_tokens": total - realised,
                          "context_unit_truncated": True,
                          "context_truncated_tokens": sizes[i] - realised,
                          # A round-trip can re-encode longer than the cut; an
                          # over-window prompt is still an over-window prompt.
                          **({"context_overflow": True} if realised > window else {})})
            return [prefix], stats
        kept_idx.append(i)
        kept_tokens += step

    kept_idx.sort()   # restore corpus / rank order whichever end we walked from
    kept_facts = [facts[i] for i in kept_idx]
    return kept_facts, {
        "context_items": len(kept_facts),
        "context_tokens": kept_tokens,
        "context_window": window,
        "context_window_truncated": True,
        "context_dropped_items": len(facts) - len(kept_facts),
        "context_dropped_tokens": total - kept_tokens,
    }
