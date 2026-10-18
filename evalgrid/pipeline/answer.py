"""
evalgrid/pipeline/answer.py
------------------------
Generates the final answer for each query, by either of the two ways this repo
can put memory in front of a generator:

  the SHARED path (default)
      Take the top-k (or budget-worth of) facts from a retrieval run, render
      them as a flat corpus, and answer with the one prompt every system is
      compared under. Holds retriever and prompt fixed so the comparison is
      about memory representation. No memory system is loaded.

  --retriever native --prompt native
      Answer with the system's OWN retrieval trace and its OWN prompt — the
      system as published. The system is loaded and driven; retrieval either
      runs live or is replayed from a saved `evalgrid.pipeline.retrieve
      --retriever native` run (see --from-retrieval).

Those two, plus the cells between them, are the point of one module: with
`--retriever` and `--prompt` as independent axes you can hold a system's
retrieval and vary only its prompt, or the reverse.

  --retriever bm25   --prompt shared   flat corpus,      shared prompt
  --retriever bm25   --prompt native   flat corpus,      the system's prompt
  --retriever native --prompt shared   its own ranking,  shared prompt
  --retriever native --prompt native   the system, as published

Only the last needs the system loaded — the other three answer from the corpus
and a run.tsv, and `--prompt native` there is just `SystemPlugin.answer_prompt`,
a template.

Reads:
  shared  outputs/_qrels/<system>/<dataset>/<mem-seg>/<labeler-seg>/<sample>/{corpus.jsonl,queries.jsonl}
          outputs/_retrieval/<system>/<dataset>/<mem-seg>/<retriever>/<sample>/run.tsv
  native  outputs/_memory/<system>/<dataset>/<mem-seg>/<sample>/       (the store)
          + optionally a saved retrieval dir (--from-retrieval)
Writes:
  outputs/_answers/<system>/<dataset>/<mem-model>/<retriever>/<gen-model>/top<k>/predictions.jsonl
  native additionally writes traces.jsonl, token_usage.json, sample_usage.jsonl
  and a run.tsv of the units its own retrieval touched

Retriever answers carry no <labeler-segment>: they're built from the retrieval
run + corpus + generator, none of which depend on the labeler, so one answer
set is shared across labeling experiments.

--retriever oracle answers from the ground truth instead of a retrieval run:
the facts are every corpus doc the labeler judged relevant (the gold file for
--gold,
score > 0), with no retrieval, no run.tsv, and no top-k cap. This is the
upper-bound baseline — how well the generator does given perfect retrieval.
Because oracle facts ARE the labeler's output, its predictions are keyed by
labeler under .../oracle/<labeler-segment>/<gen-model>/all/predictions.jsonl.

Two ways to decide how much context the prompt gets, and they are alternatives:

  --top-k K            the top K ranked facts            -> .../top<k>/
  --context-budget T   facts in rank order until the next
                       would cross T tokens              -> .../budget<T>/

A THIRD knob, --context-window W, is not an alternative to those and does not
name a directory: it is the generator's serving --max-model-len, and it only
ever fires when a prompt would not fit. The row it exists for is --retriever
full on all_turns, whose prompt is the whole rendered haystack: measured on
LongMemEval-S that is ~110k tokens (o200k) / ~112k (cl100k), so it CLEARS a
131,072 window with ~15k to spare and does NOT clear 65,536 by ~50k. Without W a
prompt past the window is not a shorter answer, it is a 400: the question lands
in errors.jsonl rather than predictions.jsonl and the denominator silently
shrinks. With it the prompt is cut to fit (oldest turns first, since the corpus
is chronological and knowledge-update questions need the LATEST mention) and the
cost is recorded per question as context_dropped_tokens. See lib/tokens.py.

The budget is the axis that generalises across systems. Equal k is not equal
evidence when one system's unit is twice another's — a pair doc vs a turn doc vs
a LightMem memory — which is why the matched-token rows used to be built by
hand-pairing k values per corpus. See evalgrid/lib/tokens.py for the three
decisions the numbers depend on (stop at first overflow, always keep the top-1
fact, count the rendered fact string).

predictions.jsonl rows are {"question_id", "hypothesis"} — enough to be scored
directly with evalgrid/pipeline/judge.py. The depth/budget is its own path
segment so sweeping it produces separate, directly comparable predictions files
instead of colliding (and silently being skipped by the resumability check
below). answer_stats.jsonl is written beside predictions.jsonl with per-question
latency, generator token usage, the context measurement (context_tokens /
context_items / whether the budget was filled) and the query-time LLM cost
(evalgrid/lib/cost.py); answer_summary.json holds aggregate totals.

Resumable: question_ids already in predictions.jsonl are skipped (use
--overwrite to regenerate everything; new answers are appended otherwise). A
failing query is recorded to errors.jsonl rather than aborting the batch, so a
re-run retries exactly the questions that errored. When a cell turns out to owe
NOTHING, the run is a no-op on disk as well: config.json, manifest.json and
answer_summary.json are left as the invocation that generated the answers wrote
them, rather than restamped with this pass's flags and a ~0 wall time. The
native path is resumable per SAMPLE and additionally shardable.

**The two halves take disjoint flags, and that is enforced rather than ignored**
— `--emb-model`/`--concurrency`/`--retrieval-dir` configure the shared path,
`--mem-dir`/`--mem-base-url`/`--config`/`--from-retrieval` the native one.
(`--qrels-dir` is read by BOTH: the shared path answers from the corpus it
holds, and a native replay reads the same corpus to split grouped units into
memories — see install_replay.) The
check is against flags given on the COMMAND LINE only, so a shared `--exp` spec
carrying values for the other half stays legal. Same rule as
evalgrid/pipeline/retrieve.py.

**--top-k means one of two things, depending on whether retrieval has already
happened.** Both knobs — depth and token budget — are answer-time cuts of a
finished ranking wherever that ranking already exists:

  shared path                  post-hoc. Retrieval saved the FULL ranking; the
                               prompt takes the first k (or a budget's worth).
                               One retrieval serves every k -> `top<k>/`.
  native + --from-retrieval    post-hoc, identically. Retrieval already ran, so
                               k can only mean "how many saved units reach the
                               prompt" -> `top<k>/` under the retriever segment
                               the saved run was produced with.
  native, live (no replay)     NOT post-hoc. k is pushed into the system's own
                               retriever before it runs, because there k sits
                               inside the method — A-Mem's link expansion,
                               MemoryBank's regrouping and HyperMem's funnel all
                               operate on it, so a deep run truncated back is a
                               DIFFERENT retrieval, not a prefix of a shallow
                               one. That makes k retriever *identity*, so it
                               lives in the retriever segment
                               (`native-emb-X-top20`) and `<topk-seg>` is "all".

So the affordable sweep is: one deep `retrieve --retriever native --top-k N`,
then answer from it repeatedly, varying `--top-k` and `--context-budget` freely
with no re-retrieval. Defaults differ: 10 on the shared path, "each config's own
depth" on the native one.

Usage:
  # shared prompt over a flat retrieval run
  python -m evalgrid.pipeline.answer --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --retriever bm25 \\
      --top-k 10 --gen-model gpt-4o-mini

  # the system as published, replaying a saved native retrieval
  python -m evalgrid.pipeline.answer --exp experiments/simplemem/longmemeval.yaml \\
      --retriever native --prompt native --gen-model ... \\
      --from-retrieval <dir written by retrieve --retriever native> --context-budget 2000
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from time import perf_counter
from typing import Optional

from evalgrid.lib.cli import (
    add_candidate_source_arg,
    add_concurrency_arg,
    add_dataset_arg,
    add_emb_model_arg,
    add_exp_arg,
    add_labeler_model_arg,
    add_limit_arg,
    add_mem_model_arg,
    add_mem_variant_arg,
    add_overwrite_arg,
    add_retriever_arg,
    add_shard_args,
    add_system_arg,
    add_trace_args,
    apply_spec,
    build_recorder_from_args,
    require_resolved,
)
# Private, deliberately: this is one of the two stages whose flags are only
# valid for half its invocations, so it needs to distinguish "the user typed
# this" from "a shared --exp supplied it". See pipeline/retrieve.py.
from evalgrid.lib.cli import _cli_provided_dests
from evalgrid.lib.manifest import write_manifest
from evalgrid.lib.paths import (
    PROMPTS,
    ExperimentPaths,
    answer_depth_segment,
    labeler_segment,
    model_slug,
    native_segment,
    retriever_name,
)
from evalgrid.models.factory import _OPENAI_GEN_PREFIXES

from tqdm import tqdm

from evalgrid.benchmarks.gold import (ALL_GOLD_LEVELS, EVAL_GOLD_LEVEL,
                                      gold_filename, gold_suffix)
from evalgrid.benchmarks.retrieval import read_jsonl, read_qrels, read_run
from evalgrid.lib.cost import query_cost, sum_cost
from evalgrid.lib.llm import DEFAULT_SEED, DEFAULT_TEMPERATURE, build_generator
from evalgrid.lib.tokens import (fit_to_budget, fit_to_window, token_counter,
                                 token_truncator)
from evalgrid.systems.base import shared_answer_prompt
from evalgrid.systems.registry import get_plugin


# Canonical timestamp rendering for facts. LongMemEval's native shape, which is
# also what the question time uses in the answer prompt — so facts and the question
# read the same. Both corpus shapes are normalized to this single format.
_TS_RENDER = "%Y/%m/%d (%a) %H:%M"
# Non-ISO source shapes to try with strptime (ISO is handled via fromisoformat).
_TS_PARSE_FORMATS = ("%Y/%m/%d (%a) %H:%M",)


def _normalize_timestamp(time_stamp: str) -> str:
    """Render any supported timestamp shape as `_TS_RENDER`.

    Corpus timestamps arrive in two shapes — lightmem memories store an ISO
    stamp ('2023-05-20T01:33:00.000'), user_turns store LongMemEval's native
    '2023/06/23 (Fri) 06:57' — so we parse both and re-render them identically,
    deriving the weekday from the parsed date. A stamp that parses as neither is
    returned unchanged (defensive — never drop the temporal cue)."""
    try:
        return datetime.fromisoformat(time_stamp).strftime(_TS_RENDER)
    except ValueError:
        pass
    for fmt in _TS_PARSE_FORMATS:
        try:
            return datetime.strptime(time_stamp, fmt).strftime(_TS_RENDER)
        except ValueError:
            continue
    return time_stamp


def format_fact(doc: dict) -> str:
    """Prefix a corpus doc with its bracketed, normalized timestamp — close to
    LightMem's retrieve(), which returns '<time_stamp> <weekday> <memory>'
    strings that LightMemLongMemEval.answer() feeds straight to the generator,
    but with the stamp normalized (see _normalize_timestamp) and wrapped as
    '[<time_stamp>] <text>' so the temporal cue is uniform across corpus shapes
    and clearly delimited from the fact text.

    Both corpus shapes keep the text clean and carry the timestamp in metadata
    (time_stamp); whatever its source shape, it's rendered to one format here:
      * lightmem memory: ISO '2021-05-03T19:18:04.000' -> '[2021/05/03 (Mon) 19:18] <memory>'
      * user_turns:      '2023/06/23 (Fri) 06:57'      -> '[2023/06/23 (Fri) 06:57] <text>'
    A doc with no timestamp is returned unchanged. The 'time_stamp in text'
    guard is a defensive no-duplication check (no current path embeds the stamp
    in text)."""
    text = doc["text"]
    meta = doc.get("metadata") or {}
    time_stamp = meta.get("time_stamp") or ""
    if not time_stamp or time_stamp in text:
        return text
    return f"[{_normalize_timestamp(time_stamp)}] {text}"


#: Tokens held back from --context-window, on top of the measured prompt text
#: and --max-tokens, to cover what the counter cannot see: the chat template's
#: role markers and special tokens (``<|im_start|>assistant`` and friends), which
#: token_counter deliberately does not add (``add_special_tokens=False``) because
#: a --context-budget must not be charged for them. A window cut may not be off
#: by even one token in the direction that 400s, so it pays for them here. ~30
#: tokens is the real figure for the Qwen/Llama templates; 128 is that with room
#: for a more verbose template and is negligible against any real window.
CHAT_TEMPLATE_RESERVE = 128


def _facts_window(build_prompt, question: str, question_date: Optional[str],
                  count_tokens, window: int, max_tokens: int) -> int:
    """Tokens left for the FACTS, once everything else in the request is paid for.

    ``fit_to_window`` budgets facts, but a server rejects a REQUEST: prompt plus
    the reserved completion. So the fact budget is the window minus the answer
    reservation, minus this question's own prompt overhead — the template, the
    question and its date — measured by rendering the prompt with an empty fact
    list. That overhead is a couple of hundred tokens and varies per question, so
    it is measured rather than assumed constant.

    Slightly conservative on purpose: the empty-fact render includes the "(no
    relevant memories found)" placeholder that a real prompt will not carry, and
    CHAT_TEMPLATE_RESERVE is rounded up. Erring small costs a few tokens of
    evidence; erring large costs the whole answer.
    """
    overhead = sum(count_tokens(m["content"])
                   for m in build_prompt(question, question_date, []))
    return window - overhead - max_tokens - CHAT_TEMPLATE_RESERVE



def resolve_prompt(system: str, prompt: str = "shared"):
    """Return the prompt builder for ``system`` under prompt mode ``prompt``.

    "shared" (default) is evalgrid.systems.base's prompt for every system —
    holding it fixed is what makes the comparison about memory representation
    rather than prompt engineering, and it is what every run in outputs/ was
    produced under.

    "native" asks the system to be prompted its own way, via
    SystemPlugin.answer_prompt. It is an explicit flag rather than the implicit
    default because a plugin that overrides answer_prompt would otherwise change
    the meaning of every existing answers dir without changing its path; the two
    modes are separated on disk by gen_segment's suffix.

    Corpus sources that are not memory systems (the raw-turn baselines,
    --system all_turns/user_turns/pair_turns) have no plugin, so only "shared"
    is defined for them.
    """
    if prompt == "shared":
        return shared_answer_prompt
    try:
        return get_plugin(system).answer_prompt
    except KeyError:
        raise SystemExit(
            f"--prompt native is not available for {system!r}: it is a raw-corpus "
            f"baseline with no plugin, so it has no native prompt. Use --prompt shared."
        )


async def _answer_one(
    sem: asyncio.Semaphore, generator, max_tokens: int, build_prompt,
    question_id: str, question: str, question_date: Optional[str], facts: list[str],
    ctx: dict,
) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """Returns (prediction_record, stats_record, error_record).

    ``ctx`` is the context measurement from lib/tokens.fit_to_budget — how much
    evidence this question's prompt actually carried, and whether the budget was
    filled. It is recorded whether or not --context-budget was passed, so a
    plain top-k run lands on the same cost axis as a budget one.
    """
    # Started INSIDE the semaphore, not before it: every question's coroutine is
    # created up front by the gather, so a timer started here would run while the
    # call sits in the concurrency queue and latency_sec would come out as
    # elapsed-since-run-start, not call duration. Runs written before this fix
    # have that inflation in answer_stats.jsonl and in answer_summary.json's
    # latency.sum_sec / latency.avg_sec; latency.wall_sec_this_invocation is sound.
    started = perf_counter()
    try:
        messages = build_prompt(question, question_date, facts)
        async with sem:
            started = perf_counter()
            result = await generator.generate_with_usage(messages, max_tokens=max_tokens)
        latency_sec = perf_counter() - started
        stats = {
            "question_id": question_id,
            "latency_sec": latency_sec,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            # Included in output_tokens, not on top of it; None = not measured.
            "reasoning_tokens": result.reasoning_tokens,
            **ctx,
            # Query-time LLM cost. Retrieval is zero by construction on this
            # path: the ranking was produced offline by stage 3 with an embedder
            # or BM25, not by an LLM call per question. That zero is the point of
            # recording it — it is what a native row's non-zero is compared with.
            **query_cost(answer=(result.input_tokens, result.output_tokens, 1)),
        }
        return {"question_id": question_id, "hypothesis": result.text}, stats, None
    except Exception as e:
        latency_sec = perf_counter() - started
        tqdm.write(f"\n[ERROR] question {question_id!r}: {e}")
        tqdm.write(traceback.format_exc())
        return None, None, {
            "question_id": question_id,
            "latency_sec": latency_sec,
            "error": str(e),
            "traceback": traceback.format_exc(),
            # What the call spent before it failed. Only OutputCapError knows;
            # a transport error spent nothing we can see, and records zeros.
            "input_tokens": getattr(e, "input_tokens", 0),
            "output_tokens": getattr(e, "output_tokens", 0),
            "reasoning_tokens": getattr(e, "reasoning_tokens", None),
        }


async def _run(generator, concurrency: int, max_tokens: int, build_prompt,
               items: list[tuple]) -> list[tuple]:
    sem = asyncio.Semaphore(concurrency)
    with tqdm(total=len(items), desc="queries", unit="query") as bar:
        async def _wrapped(question_id, question, question_date, facts, ctx):
            result = await _answer_one(sem, generator, max_tokens, build_prompt,
                                       question_id, question, question_date, facts, ctx)
            bar.update(1)
            return result

        return await asyncio.gather(*(_wrapped(*item) for item in items))


def _context_summary(rows: list[dict]) -> dict:
    """Aggregate the per-question context measurements.

    ``avg_tokens`` is the number a matched-budget table is actually matched on
    — the budget is a ceiling, not the realised context — and the counts below
    it say whether the cell is comparable at all. A budget that a system
    under-fills on a large fraction of questions is a budget that system cannot
    reach; one where its head unit had to be CUT to fit is a budget below that
    system's unit size, and those questions were answered from a fragment.
    Either way the cell must be marked rather than averaged into a row.
    """
    count = len(rows)
    tokens = [int(r.get("context_tokens") or 0) for r in rows]
    items = [int(r.get("context_items") or 0) for r in rows]
    return {
        "avg_tokens": sum(tokens) / count if count else 0.0,
        "avg_items": sum(items) / count if count else 0.0,
        "max_tokens": max(tokens) if tokens else 0,
        "truncated": sum(1 for r in rows if r.get("context_truncated")),
        "underfilled": sum(1 for r in rows if r.get("context_underfilled")),
        # Questions whose head unit alone exceeded T and was cut to it. This is
        # the rate to report, not a footnote: those cells were answered from a
        # fragment of one unit, and a system cut on most of its questions is
        # being measured on fragments at that budget.
        "unit_truncated": sum(1 for r in rows if r.get("context_unit_truncated")),
        # Only ever non-zero on the degraded path (no tokenizer to cut with).
        "overflow": sum(1 for r in rows if r.get("context_overflow")),
        # --context-window: questions whose prompt did not fit the SERVER and
        # was cut to it, and the evidence that cost. Distinct from "truncated"
        # above, which is a budget cut (a chosen condition). A full-context row
        # with window_truncated > 0 is a truncated full-context row: it is no
        # longer "the whole haystack in the window" and must be labelled with
        # this count and avg_dropped_tokens, not averaged in silently.
        "window_truncated": sum(1 for r in rows if r.get("context_window_truncated")),
        "avg_dropped_tokens": (sum(int(r.get("context_dropped_tokens") or 0) for r in rows)
                               / count if count else 0.0),
        "max_dropped_tokens": max((int(r.get("context_dropped_tokens") or 0) for r in rows),
                                  default=0),
    }


def _sum_reasoning(rows: list[dict]) -> int | None:
    """Total reasoning tokens over the rows that measured it; None if none did.

    None rather than 0, because rows written before the field existed, or by a
    backend that does not look, have no trace to count, and 0 would read as a
    measured "no reasoning".
    """
    measured = [int(r["reasoning_tokens"]) for r in rows
                if r.get("reasoning_tokens") is not None]
    return sum(measured) if measured else None


def _write_answer_summary(out_run_dir: str, stats_path: str, generator, run_wall_sec: float) -> None:
    rows = read_jsonl(stats_path) if os.path.isfile(stats_path) else []
    errors_path = os.path.join(out_run_dir, "errors.jsonl")
    failed = read_jsonl(errors_path) if os.path.isfile(errors_path) else []
    count = len(rows)
    total_latency = sum(float(r.get("latency_sec") or 0.0) for r in rows)
    total_input = sum(int(r.get("input_tokens") or 0) for r in rows)
    total_output = sum(int(r.get("output_tokens") or 0) for r in rows)
    failed_input = sum(int(r.get("input_tokens") or 0) for r in failed)
    failed_output = sum(int(r.get("output_tokens") or 0) for r in failed)
    summary = {
        "answered": count,
        "latency": {
            "sum_sec": total_latency,
            "avg_sec": total_latency / count if count else 0.0,
            "wall_sec_this_invocation": run_wall_sec,
        },
        "tokens": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            # Part of output_tokens, not added to it.
            "reasoning_tokens": _sum_reasoning(rows),
        },
        # Calls that were paid for and produced no answer, such as a thinking
        # model cut off by max_tokens before </think>. They are kept out of
        # "tokens" and "query_cost" above, which stay per ANSWERED question,
        # and must be added to them for the cell's true spend. errors.jsonl is
        # append-only, so this includes failures a later pass re-answered:
        # those were spent too. Records written before the error record
        # carried usage count as zero.
        "failed_calls": {
            "calls": len(failed),
            "input_tokens": failed_input,
            "output_tokens": failed_output,
            "total_tokens": failed_input + failed_output,
            "reasoning_tokens": _sum_reasoning(failed),
        },
        "context": _context_summary(rows),
        # C2 for this run. Read it against outputs/_memory/**/ingest_usage.json
        # (C1) — on LongMemEval-S the two add up per question, one haystack each.
        "query_cost": sum_cost(rows),
        "generator_cumulative_tokens_this_invocation": {
            "input_tokens": generator.token_usage.input_tokens,
            "output_tokens": generator.token_usage.output_tokens,
            "total_tokens": generator.token_usage.total_tokens,
            "calls": generator.token_usage.calls,
        },
    }
    with open(os.path.join(out_run_dir, "answer_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# --retriever native --prompt native: the system's own loop and its own prompt
# ---------------------------------------------------------------------------

#: The retriever segment a native run writes under, in both output trees. It is
#: a retriever name like "bm25" so eval_retrieval, judge and the answer-path
#: reconstruction in run_all_judge.sh need no special case.
NATIVE = "native"


def ranked_from_trace(plugin, trace: dict) -> list[tuple[str, float]]:
    """One query's trace -> the (unit_id, score) list write_run expects.

    De-duplicates keeping first occurrence: A-Mem's link expansion can surface
    the same note twice (once as a hit, once as another hit's neighbour), and a
    run.tsv with a repeated corpus-id is malformed. First occurrence is the
    right one to keep — it is the best rank the system gave that unit.

    Scores are ``1/rank``, not similarities — a native loop returns a set or an
    ordered list, not calibrated scores. Monotone in rank, so correct for
    recall@k / NDCG@k / MRR; do not read them as similarities.
    """
    seen: set[str] = set()
    ranked: list[tuple[str, float]] = []
    for unit_id in plugin.native_unit_ids(trace):
        if unit_id in seen:
            continue
        seen.add(unit_id)
        ranked.append((unit_id, 1.0 / (len(ranked) + 1)))
    return ranked


def memory_atoms(trace: dict, corpus: dict[str, str] | None) -> tuple[list[str], list[str] | None]:
    """One query's saved retrieval as ``(texts, ids)`` — one element per MEMORY.

    **This is the exchange rate both answering budgets are cut on.** ``--top-k``
    counts memories and ``--context-budget`` counts the tokens of memories, so
    both need the ranking expressed in memories rather than in whatever the
    system happens to return as a "unit".

    For five of six systems those are the same thing and this is a passthrough.
    **MemoryBank is the exception**: ``_regroup`` concatenates hits that share a
    date bucket, so one unit is ~1.5 memories, and cutting whole units puts a
    different amount of evidence in front of the generator than the same k does
    everywhere else — k=3 and k=5 are not even reachable, because the top block
    routinely holds more memories than that.

    The member texts come from **the sample's ``corpus.jsonl``**, joined by the
    ids in ``unit_id_groups``: a block IS its member memories joined by "\n" in
    exactly that order, so this reconstructs it byte-for-byte. Deliberately a
    read of an existing artefact rather than a new field on the trace — every
    saved retrieval already on disk works unchanged, no system's own code moves,
    and MemoryBank's ``retrieval.jsonl`` does not grow by ~50% carrying a second
    copy of every block. A system that ever needs to be corpus-independent can
    record its members as ``unit_texts`` and read them here instead.

    Splitting the block string on "\n" is NOT an option, and cannot be: a single
    memory contains newlines of its own (an assistant turn is many lines), so the
    split would land in the middle of a memory and hand the generator fragments
    while reporting whole ids.

    Passing the memories to the prompt separately instead of as one block is
    not a change of prompt: ``shared_answer_prompt`` joins facts with "\n",
    which is the same separator ``_regroup`` used, so the rendered string is
    byte-identical.
    """
    units = trace.get("retrieved_units") or []
    groups = trace.get("unit_id_groups")
    if not (isinstance(groups, list) and len(groups) == len(units)):
        # Not a grouping system: one unit is one memory. Ids come along only if
        # they line up, which is every system except HyperMem (whose
        # retrieved_unit_ids is a per-layer dict, not a flat list).
        ids = trace.get("retrieved_unit_ids")
        return list(units), list(ids) if isinstance(ids, list) and len(ids) == len(units) else None

    texts: list[str] = []
    ids: list[str] = []
    saved_texts = trace.get("unit_texts")
    for index, group in enumerate(groups):
        if isinstance(saved_texts, list) and len(saved_texts) == len(groups) \
                and len(saved_texts[index]) == len(group):
            members = list(saved_texts[index])     # a system that records its own
        elif corpus is not None and all(i in corpus for i in group):
            members = [corpus[i] for i in group]
        else:
            # Loud, not a fall back to whole-unit cutting: that would report k
            # memories and hand the generator more.
            raise RuntimeError(
                f"this retrieval groups {len(group)} memories into one unit, and cutting "
                f"it needs the members — but the sample's corpus.jsonl "
                f"{'was not found' if corpus is None else 'does not hold every id'}.\n"
                f"Build the qrels for this sample (they write corpus.jsonl), then pass "
                f"--qrels-dir if it is not in the default tree:\n"
                f"  python -m evalgrid.pipeline.qrels_has_answer --exp <spec>"
            )
        texts.extend(members)
        ids.extend(group)
    return texts, ids


def rank_order(trace: dict, ids: list[str] | None) -> list[int] | None:
    """Positions into :func:`memory_atoms`' output, best-ranked first.

    ``memory_atoms`` returns memories in READING order — for a grouping system
    that is the order the generator sees them in, which is what the prompt
    needs and is NOT the order they were ranked in. Every cut below is defined
    on the ranking instead: ``--top-k`` means "the k best", and
    ``lib/tokens.fit_to_budget`` takes its facts "in rank order" and fills
    greedily from the front. Handing either one reading order silently turns it
    into "the k that sort first by bucket" — for MemoryBank an alphabetical
    slice, which is how its whole retrieval column once scored 0.000 recall
    with the gold sitting at reading-position 45 of 100.

    ``trace["retrieved_unit_ids"]`` is the ranking (systems/base.native_unit_ids:
    "rank order is significant"). Returns ``None`` when there is nothing to
    reorder — no ids to key on, or a trace predating the field — and the caller
    keeps reading order, which is what every non-grouping system wants anyway
    because for them the two orders are the same list.
    """
    if not ids:
        return None
    ranking = trace.get("retrieved_unit_ids")
    if not isinstance(ranking, list) or not ranking:
        return None
    rank = {}
    for position, unit_id in enumerate(ranking):
        rank.setdefault(unit_id, position)     # first occurrence = best rank
    if not any(i in rank for i in ids):
        return None                            # disjoint id spaces: do no harm
    # Unranked ids sort last but keep their reading order among themselves, so
    # a partial ranking degrades to "ranked ones first" rather than shuffling.
    unranked = len(ranking)
    return sorted(range(len(ids)), key=lambda p: (rank.get(ids[p], unranked), p))


def install_replay(system, retrieval_dir: str, *, budget, top_k, gen_model: str,
                   qrels_dir: str | None = None) -> None:
    """Make ``system.answer`` replay saved retrieval instead of retrieving again.

    Instance rebinding, not a subclass: the system is already built and the
    runner drives it through ``reset`` / ``load_memory`` / ``answer``, so the
    two methods that need to change are changed on the object. Same pattern as
    ``evalgrid.lib.trace.record_calls``.

    ``load_memory`` is wrapped because it is the only place the sample identity
    reaches the system — the runner passes the per-sample memory dir, whose
    basename is the sample id, which is what names the saved retrieval file and
    the sample's ``corpus.jsonl`` under ``qrels_dir``.

    **Both depth knobs are applied HERE, both are post-hoc, and both cut the
    same list of MEMORIES** (:func:`memory_atoms`). Retrieval has already
    happened, so on this path ``--top-k`` cannot mean "how deep to retrieve" —
    it can only mean "how many of the saved memories reach the prompt", which is
    a cut of a fixed ranking exactly like ``--context-budget``'s cut by tokens.
    That is the same meaning both knobs have on the shared path, and it is what
    lets one deep retrieval serve a sweep over BOTH axes without re-retrieving.
    (On a LIVE native run ``--top-k`` is the opposite: it goes into the system's
    own retriever before it runs, because there k sits inside the method.)

    A budget supersedes k when both are given, mirroring the shared path: under
    a budget the whole saved ranking is the candidate pool, since k would cap it
    at an arbitrary point before the budget ever bound.
    """
    from evalgrid.benchmarks.runner import _sample_mem_path
    from evalgrid.lib.tokens import fit_to_budget, token_counter, token_truncator
    from evalgrid.pipeline.retrieve import RETRIEVAL_FILE

    count = token_counter(gen_model)
    # Only built when a budget can actually fire, so a plain top-k sweep does
    # not pay for a second tokenizer handle it never calls.
    cut = token_truncator(gen_model) if budget is not None else None
    original_load = system.load_memory
    saved: dict[str, dict] = {}
    corpus: dict[str, str] = {}

    def load_memory(path: str) -> None:
        original_load(path)
        saved.clear()
        corpus.clear()
        sample_id = os.path.basename(os.path.normpath(path))
        f = os.path.join(_sample_mem_path(retrieval_dir, sample_id), RETRIEVAL_FILE)
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"--from-retrieval is set but {f} does not exist. Run "
                f"evalgrid.pipeline.retrieve --retriever native for this sample first, "
                f"or drop --from-retrieval to retrieve inline."
            )
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    trace = json.loads(line)
                    saved[str(trace.get("query_id"))] = trace
        # Read once per SAMPLE, not once per query: the same few hundred docs
        # serve every question of the sample, and only a grouping system ever
        # asks for them (memory_atoms falls through without touching it).
        if qrels_dir:
            docs = os.path.join(qrels_dir, sample_id, "corpus.jsonl")
            if os.path.isfile(docs):
                for doc in read_jsonl(docs):
                    corpus[str(doc.get("_id"))] = doc.get("text") or ""

    def answer(query) -> str:
        trace = saved.get(str(query.query_id))
        if trace is None:
            raise KeyError(
                f"no saved retrieval for query {query.query_id!r} in {retrieval_dir}. "
                f"The retrieve run covered a different set of samples."
            )
        trace = dict(trace)
        units = trace.get("retrieved_units")
        if budget is not None or top_k is not None:
            if units is None:
                knob = "--context-budget" if budget is not None else "--top-k"
                raise RuntimeError(
                    f"{knob} needs per-unit texts, and this saved retrieval has "
                    f"none (retrieved_units missing). It was written before the system's "
                    f"retrieve_trace recorded them — re-run retrieve --retriever native."
                )
            # Both axes cut the SAME list, and it is a list of memories — see
            # memory_atoms for why that is not the same as a list of units.
            texts, ids = memory_atoms(trace, corpus or None)
            # SELECT in rank order, RENDER in reading order. Both cuts below
            # keep a prefix of what they are given (fit_to_budget fills greedily
            # and can only truncate its head fact), so ordering by rank first is
            # what makes "the k best" and "the best that fit T" mean that. The
            # order is then put back, because for a grouping system reading
            # order IS the system's own contribution — MemoryBank's bucket
            # concatenation — and reordering the prompt would silently swap that
            # condition for a different one. A no-op wherever the two orders
            # already agree, which is every system except MemoryBank.
            order = rank_order(trace, ids)
            if order is not None:
                texts = [texts[p] for p in order]
                ids = [ids[p] for p in order]
            sizes = [count(t) for t in texts]
            if budget is not None:
                kept, stats = fit_to_budget(texts, sizes, budget, truncate=cut)
            else:
                # Depth, measured the same way a budget is, so a top<k> cell and
                # a budget<T> cell report context_items/context_tokens on one
                # scale. budget=None means fit_to_budget only measures.
                kept = texts[:top_k]
                _, stats = fit_to_budget(kept, sizes[:top_k], None)
                if len(kept) < top_k:
                    # The ranking ran out before k — the same condition
                    # fit_to_budget reports for a budget it could not fill, so
                    # it carries the same field and _context_summary counts it
                    # for free. Without it a `top10` cell built from a 4-memory
                    # ranking would read as a top-10 result.
                    stats["context_underfilled"] = True
                    stats["context_depth_requested"] = top_k
            kept_ids = ids[:len(kept)] if ids is not None else None
            if order is not None:
                # Back to reading order. Keyed on the ORIGINAL positions, so a
                # head fact that fit_to_budget truncated travels with its own
                # text rather than being re-fetched whole.
                back = sorted(range(len(kept)), key=lambda p: order[p])
                kept = [kept[p] for p in back]
                if kept_ids is not None:
                    kept_ids = [kept_ids[p] for p in back]
            trace["retrieved_units"] = kept
            # The ids are cut alongside the texts. Without this the trace still
            # reports the full pre-cut ranking, and runner._add_trace_fields
            # derives context_items from the IDS — so every truncated cell would
            # over-report what actually reached the prompt. ``ids`` is None only
            # when the two could not be aligned (HyperMem's per-layer dict), and
            # there the stale field is left alone rather than half-cut.
            if kept_ids is not None:
                trace["retrieved_unit_ids"] = kept_ids
            # A block that was split into memories no longer matches the saved
            # rendering even when nothing was dropped, so this is keyed on the
            # texts changing rather than on the count.
            trace["budget_applied"] = kept != units
            trace["_budget_stats"] = stats
            # unit_id_groups described the PRE-cut blocks. After the cut the
            # units are memories, one id each, so a stale grouping would make a
            # later reader recount the same memories. Popped from this process's
            # COPY of the trace — the saved retrieval.jsonl is never rewritten.
            trace.pop("unit_id_groups", None)
            trace.pop("unit_texts", None)
        # The system appends its own trace during a live run; on replay nothing
        # does, so the run.tsv writer downstream still sees one trace per query.
        if hasattr(system, "traces"):
            system.traces.append(trace)
        return system.answer_from_trace(query, trace)

    system.load_memory = load_memory
    system.answer = answer
    # Read by benchmarks/runner.py: a replay never retrieves, so building the
    # system's query-time index (prepare_retrieval) would be pure waste.
    system._retrieval_replay = True


def _saved_retriever(retrieval_dir: str) -> Optional[str]:
    """The retriever segment the saved retrieval was produced under, or None.

    ``pipeline/retrieve.py`` writes it to the run dir's config.json. Reading it
    back is what keys a replayed answer to the retrieval it actually came from,
    rather than to whatever this process's --config happens to resolve to.
    """
    try:
        with open(os.path.join(retrieval_dir, "config.json")) as f:
            return json.load(f).get("retriever") or None
    except Exception:
        return None


def _run_native(parser: argparse.ArgumentParser, args) -> None:
    """``--retriever native --prompt native``: the system, as published.

    Imported lazily and kept out of ``main`` for the same reason
    pipeline/retrieve.py does it — a shared-prompt run has no business loading
    the plugin registry, the dataset adapters or a memory system.
    """
    from evalgrid.benchmarks.adapters import DATASETS, build_adapter
    from evalgrid.benchmarks.retrieval import write_run
    from evalgrid.benchmarks.runner import _sample_mem_path, run
    from evalgrid.pipeline.ingest import LimitedAdapter, ShardedAdapter, load_config
    from evalgrid.systems import native_retrieval
    from evalgrid.systems.registry import get_plugin

    if args.dataset not in DATASETS:
        parser.error(f"--retriever native --prompt native needs a dataset adapter to supply "
                     f"the queries; {args.dataset!r} is not one of {sorted(DATASETS)}")
    try:
        plugin = get_plugin(args.system)
    except KeyError:
        parser.error(f"--retriever native --prompt native is not available for --system "
                     f"{args.system!r}: it is a raw-corpus baseline with no memory system, "
                     f"so it has no retrieval loop and no prompt of its own.")

    paths = ExperimentPaths(system=args.system, dataset=args.dataset, mem_model=args.mem_model,
                            mem_variant=args.mem_variant)
    mem_dir = args.mem_dir or paths.memory()
    if not os.path.isdir(mem_dir):
        sys.exit(f"No saved memory at {mem_dir} — run evalgrid.pipeline.ingest first.")

    # Config first: every path below is derived from it (the encoder the system
    # ranks with is a config value, not a flag), so it has to be fully resolved
    # — mem model, budget and all — before any directory name is built.
    system_config, config_source = load_config(plugin, args)
    # Before apply_mem_model and before any directory name is built, exactly as
    # in pipeline/retrieve.py: the swap decides the run directory. A split
    # retrieve/answer pair must pass the same --native-retriever, or the replay
    # renders against a different ranking than the one that was saved.
    native_retrieval.apply_spec(system_config, args.native_retriever)
    plugin.apply_mem_model(system_config, args.mem_model, args.mem_base_url)
    # Budget, before build: both are constructor-time for every system.
    #
    # SAY SO WHEN THIS DOWNGRADES THE CONFIG. apply_answer_max_tokens overwrites
    # `answer_max_tokens` unconditionally, so a config asking for more than
    # --max-tokens (default 2000) is silently cut to it — and the run then
    # records the cut value as though the config had asked for it, which is the
    # one place its own config.json cannot be used to detect the problem.
    #
    # It cost a whole cell before this warning existed: hypermem's BEAM-100K
    # `native-rrf--naive` (2026-09-08) asked for 4096, sized for upstream's
    # 7-step CoT prompt, got 2000, and 299 of its 400 generations stopped on the
    # cap without ever emitting the `FINAL ANSWER:` marker the answer is parsed
    # out of. Nothing in the run announced it: the judge pass came back complete
    # with no failures, grading truncated reasoning transcripts.
    #
    # A warning rather than an error: cutting the budget deliberately is a legal
    # thing to want (it is how the answer-length axis is held equal across
    # systems), and only hypermem's config asks for more than the default at all.
    _declared = system_config.get("answer_max_tokens")
    if isinstance(_declared, int) and _declared > args.max_tokens:
        print(f"!!! --max-tokens {args.max_tokens} OVERRIDES answer_max_tokens "
              f"{_declared} from {config_source}.", file=sys.stderr)
        print(f"!!!   Generation is capped at {args.max_tokens} tokens, not "
              f"{_declared}. If this system's prompt needs the room (hypermem's "
              f"CoT does), pass --max-tokens {_declared}.", file=sys.stderr)
    plugin.apply_answer_max_tokens(system_config, args.max_tokens)
    replaying = bool(args.from_retrieval)
    # The spec's native depth, which only this path reads — same contract as
    # pipeline/retrieve._run_native, so a split retrieve/answer pair driven by
    # one spec retrieves at one depth. --top-k on the CLI still wins, and
    # --published-depth opts out to the config's own depth (the E1 control).
    #
    # LIVE RUNS ONLY, and that is the whole point of the guard: on this path
    # --top-k means retrieval depth, but on a REPLAY it means a cut applied to
    # units already retrieved (install_replay), which is a swept answer-side
    # axis. Letting a spec fill it in there would silently cut every replay to
    # the retrieval depth and key it into `top<k>/` instead of the uncut
    # segment. A replay takes its k from the CLI or not at all.
    if args.published_depth and args.top_k is not None:
        parser.error("--published-depth and --top-k contradict each other: one asks for "
                     "the config's own depth, the other imposes one.")
    if args.top_k is None and not replaying and not args.published_depth:
        args.top_k = getattr(args, "native_top_k", None)
    # k goes into the system's own retriever ONLY on a live run, where it
    # changes what retrieval does. On a replay retrieval already happened, so
    # mutating a config that no longer drives anything would do nothing but
    # mislabel the run — the cut is applied to the saved units instead, in
    # install_replay.
    if args.top_k is not None and not replaying:
        plugin.apply_retrieval_budget(system_config, args.top_k)

    # Before any path is built: a system that cannot take the configured swap
    # must fail here, not after the answers dir and manifest exist. declare, not
    # check_supported: it must also reach `plugin` before build() below — see
    # the retrieve half.
    swap = native_retrieval.declare(plugin, system_config)
    if replaying:
        # The retriever is whatever produced the saved run, read from the dir's
        # own config.json rather than re-derived: this process may have been
        # given a different --config, and the answers must be keyed by the
        # retrieval they actually came from.
        native = _saved_retriever(args.from_retrieval) or native_segment(
            plugin.native_retriever_name(system_config), args.native_variant, None)
        # Both knobs are answer-time cuts here, so the depth belongs in
        # <topk-seg> exactly as on the shared path: one deep retrieval under one
        # retriever segment, fanning out into top<k>/ and budget<T>/ answer dirs.
        topk_seg = answer_depth_segment(args.top_k, args.context_budget,
                                        no_retrieval=args.top_k is None
                                        and args.context_budget is None)
    else:
        # The encoder the system's own loop ranks with, derived from the resolved
        # config so a YAML swap cannot land in the previous encoder's directory —
        # the same guarantee --emb-model gives a flat run.
        native = native_segment(plugin.native_retriever_name(system_config),
                                args.native_variant, args.top_k)
        # <topk-seg> is "all" on a LIVE run, and k lives in the retriever segment
        # instead: the loop retrieved k units and answered from all of them, so
        # nothing was truncated afterwards and "top<k>" would name an operation
        # that never ran. Carrying it in both places was redundant
        # (…-top10/<gen>/top10) and would have invited a k level under the
        # retrieval dir, where tools/check_retrieval.py globs "*/run.tsv" and
        # would then silently report every native sample as missing.
        topk_seg = "all"
    out_dir = args.out or paths.answers(native, args.gen_model, topk_seg, prompt="native")
    out_path = os.path.join(out_dir, "predictions.jsonl")

    seg = labeler_segment(args.labeler_model, args.candidate_source)
    if args.retrieval_out is None:
        retrieval_out = paths.retrieval(native)
    else:
        retrieval_out = args.retrieval_out or None   # "" disables

    print(f"System    : {args.system}")
    print(f"Dataset   : {args.dataset}")
    print(f"Mem model : {args.mem_model}")
    print(f"Config    : {config_source}")
    print(f"Memory dir: {mem_dir}")
    print(f"Gen model : {args.gen_model}")
    print(f"Retriever : {native} (the system's own query-time loop)")
    if swap:
        print(f"  swapped : {swap} (substituted into the system's own loop)")
    print(f"Top-k     : {args.top_k if args.top_k is not None else '(each config own)'}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Prompt    : native (the system's own answer prompt)")
    print(f"Out       : {out_path}")
    print(f"Runs      : {retrieval_out or '(not written)'}")
    if args.num_shards > 1:
        print(f"Shard     : {args.shard_id}/{args.num_shards}")

    # Generator first, then the directory: a missing model/key must not leave an
    # empty run dir behind that a later `find ... -name predictions.jsonl` sweep
    # or a resumability check has to reason about.
    generator = build_generator(
        args.gen_model, base_url=args.gen_base_url,
        reasoning_effort=args.gen_reasoning_effort, verbosity=args.gen_verbosity,
        seed=args.gen_seed, temperature=args.gen_temperature,
    )
    os.makedirs(out_dir, exist_ok=True)
    write_manifest(out_dir, args, stage="answer")

    adapter = build_adapter(args.dataset)
    if args.num_shards > 1:
        adapter = ShardedAdapter(adapter, args.shard_id, args.num_shards)
    if args.limit:
        adapter = LimitedAdapter(adapter, args.limit)

    # config_for_build: same reason as the retrieve half — the swap block is
    # harness state, and install_per_sample below reads it off system_config.
    system = plugin.build(native_retrieval.config_for_build(system_config),
                          dataset=args.dataset,
                          generator=generator, phase="answer")
    # Before the first reset(), for the same reason as in pipeline/ingest.py.
    recorder = build_recorder_from_args(args)
    system.install_recorder(recorder)

    if args.from_retrieval:
        # The corpus behind the ranking, for cutting a grouped unit on memory
        # boundaries (install_replay / memory_atoms). Only MemoryBank needs it,
        # and only when a cut is asked for, so a missing dir is not an error
        # here — memory_atoms raises with the command that builds it, at the
        # one moment it is actually needed.
        install_replay(system, args.from_retrieval,
                       budget=args.context_budget, top_k=args.top_k,
                       gen_model=args.gen_model,
                       qrels_dir=args.qrels_dir or paths.qrels(seg))
        print(f"Retrieval : replayed from {args.from_retrieval} (no query-time retrieval)")
        # No swap installed on this branch, deliberately: nothing retrieves here,
        # and the saved retrieval was already produced by whatever retriever the
        # retrieve run used. `native` above names it, so the answers dir still
        # records which one that was.
    else:
        if args.context_budget is not None:
            parser.error("--context-budget requires --from-retrieval on the native path: "
                         "budgeting a live native run would pay for retrieval and then throw "
                         "part of it away, and SimpleMem's reflection has already read the "
                         "untruncated set. Run evalgrid.pipeline.retrieve --retriever native "
                         "once, then sweep budgets over it.")
        # Live retrieval — so the configured swap has to actually be installed,
        # once per sample. Without this the run would rank with the published
        # encoder while `native` (which prefers the swap slug) named the output
        # dir after the substitute: a mislabelled run, worse than an unsupported
        # one. Wraps load_memory because runner.run owns the sample loop.
        if native_retrieval.install_per_sample(system, plugin, system_config):
            print(f"Retrieval : {swap} substituted into the system's own loop")

    # Turn each sample's traces into that sample's run.tsv. Runs are written per
    # sample under the same slug the qrels builder used, so eval_retrieval pairs
    # them up with no extra bookkeeping.
    missing_ids: list[str] = []

    def write_sample_run(sample_id: str, traces: list[dict]) -> None:
        if not retrieval_out or not traces:
            return
        ranked_by_query = []
        for trace in traces:
            qid = trace.get("query_id")
            if qid is None:
                continue
            ranked = ranked_from_trace(plugin, trace)
            if not ranked:
                missing_ids.append(str(qid))
            ranked_by_query.append((qid, ranked))
        if ranked_by_query:
            write_run(_sample_mem_path(retrieval_out, sample_id), ranked_by_query)

    try:
        run(system, adapter, out_path=out_path,
            trace_path=os.path.join(out_dir, "traces.jsonl"),
            stats_path=os.path.join(out_dir, "token_usage.json"),
            phase="answer", mem_dir=mem_dir, overwrite=args.overwrite,
            sample_usage_path=os.path.join(out_dir, "sample_usage.jsonl"),
            on_sample_traces=write_sample_run if retrieval_out else None,
            trajectory_dir=out_dir if recorder.enabled else None)
    finally:
        recorder.close()

    # Self-describing run, same as the shared path writes, so judge.py --run
    # can read the dataset back off it.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "system": args.system,
            "dataset": args.dataset,
            "mem_model": args.mem_model,
            "mem_variant": args.mem_variant,
            "retriever": native,
            "native_variant": args.native_variant,
            "max_tokens": args.max_tokens,
            "prompt": "native",
            "gen_model": args.gen_model,
            "gen_seed": args.gen_seed,
            "gen_temperature": args.gen_temperature,
            "top_k": args.top_k,
            "context_budget": args.context_budget,
            # The serving cap this run was cut to fit, if any. Recorded because
            # a full-context row answered under a 131072 window is a DIFFERENT
            # result from one answered whole, and the path segment ("all") does
            # not say so — read it with context_window_truncated in
            # answer_summary.json before putting the row in a table.
            "context_window": args.context_window,
            "config_source": config_source,
            "system_config": system_config,
        }, f, indent=2, default=str)

    print(f"\nDone — predictions written to {out_path}")
    if recorder.enabled:
        print(f"       trajectory written to {os.path.join(out_dir, recorder.filename)}")
        if recorder.failures:
            print(f"       ! {recorder.failures} trajectory record(s) failed to write "
                  f"(the run itself is unaffected)")
    if retrieval_out:
        print(f"       native runs written to {retrieval_out}")
        # Always --run-name: the segment now carries the encoder (and possibly a
        # k and a variant), none of which eval_retrieval can re-derive from
        # --retriever/--emb-model. --run-name names the dir outright.
        print(f"       score them with: python -m evalgrid.pipeline.eval_retrieval "
              f"--system {args.system} --dataset {args.dataset} "
              f"--mem-model {args.mem_model} --run-name {native}")
    if missing_ids:
        print(f"  ! {len(missing_ids)} question(s) retrieved nothing "
              f"(empty run rows): {missing_ids[:5]}{' ...' if len(missing_ids) > 5 else ''}")


#: Flags that only mean something to one half of this stage. Checked against
#: what the user actually typed, so a shared --exp carrying the other half's
#: values (the usual case: one spec drives a whole sweep) stays legal.
_SHARED_ONLY = ("retrieval_dir", "emb_model", "concurrency",
                "exclude_question_type", "context_window", "max_history_tokens",
                "batch", "batch_phase", "batch_poll_interval")
_NATIVE_ONLY = ("mem_dir", "mem_base_url", "config", "native_variant",
                "published_depth", "from_retrieval", "retrieval_out",
                "num_shards", "shard_id",
                "trace_level", "trace_samples", "trace_max_chars")


def _reject_foreign_flags(parser, provided: set[str], dests, reason: str) -> None:
    bad = sorted(d for d in dests if d in provided)
    if bad:
        flags = ", ".join("--" + d.replace("_", "-") for d in bad)
        parser.error(f"{flags} {'has' if len(bad) == 1 else 'have'} no meaning {reason}")


def main():
    parser = argparse.ArgumentParser(
        description="Answer each question — from a retrieval run under the shared prompt, "
                    "or with the system's own loop and prompt (--retriever native "
                    "--prompt native)")
    add_exp_arg(parser)
    add_system_arg(parser, help="Memory system the qrels/run came from (e.g. lightmem)")
    add_dataset_arg(parser,
                    help="Benchmark dataset key (e.g. longmemeval_s) — used only to "
                         "derive default paths, no adapter is loaded")
    add_mem_model_arg(parser, help="Memory model used to build the qrels dataset")
    add_mem_variant_arg(parser)
    add_labeler_model_arg(parser,
                          help="Labeler model the qrels builder was run with (default: "
                               "gpt-4o) — used only to locate the default qrels/retrieval dirs")
    add_candidate_source_arg(parser)
    add_retriever_arg(parser, include_oracle=True, include_full=True, include_native=True,
                      help="Retriever whose run to use for top-k facts, 'oracle' to answer "
                           "from the ground-truth qrels (perfect retrieval), or 'full' to answer "
                           "from the entire corpus (full-context baseline, no retrieval). "
                           "'native' is the system's own ranking: with --prompt shared it reads "
                           "the saved native run.tsv (the cell that separates a system's "
                           "retrieval from its prompt), and with --prompt native it runs or "
                           "replays the system's own loop and answers with the system's own "
                           "prompt — the system as published.")
    parser.add_argument("--prompt", choices=list(PROMPTS), default="shared",
                        help="Which answer prompt to use: 'shared' (default) is the one "
                             "prompt every system is compared under; 'native' is the "
                             "system's own (SystemPlugin.answer_prompt, or the system's own "
                             "answer_from_trace when paired with --retriever native). "
                             "Non-shared prompts suffix the generator path segment, so the "
                             "two never collide.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Answer depth — how many ranked units reach the prompt. Wherever "
                             "the ranking already exists this is a post-hoc cut applied when "
                             "the prompt is built, so one retrieval serves every k -> top<k>/: "
                             "that is the shared path (default 10), and also "
                             "--retriever native --prompt native WITH --from-retrieval, where "
                             "it cuts the saved units. On a LIVE native run there is no ranking "
                             "yet, so k is instead pushed into the system's own retriever "
                             "(plugin.apply_retrieval_budget) and becomes retriever identity — "
                             "it goes in the retriever segment, <topk-seg> stays 'all', and "
                             "omitting it means the spec's native_top_k, or each config's own "
                             "published depth if the spec has none or --published-depth is "
                             "given. It cannot "
                             "cap A-Mem's link expansion or SimpleMem's uncapped union; see "
                             "context_items for what landed. Ignored when --context-budget is "
                             "given.")
    parser.add_argument("--context-budget", type=int, default=None, metavar="T",
                        help="Match on EVIDENCE VOLUME instead of depth: take facts in rank "
                             "order until the next would push the context past T tokens of the "
                             "generator's own tokenizer. Replaces --top-k, and writes to "
                             "budget<T>/ instead of top<k>/. This is the axis that generalises "
                             "across systems — equal k is not equal evidence when one system's "
                             "unit is twice another's (a pair doc vs a turn doc vs a LightMem "
                             "memory). The top-1 fact is always kept even if it alone exceeds T; "
                             "see context_overflow / context_underfilled in answer_stats.jsonl "
                             "for whether the cell is comparable. Not available for "
                             "--retriever oracle/full, which have no ranking to truncate.")
    parser.add_argument("--context-window", type=int, default=None, metavar="W",
                        help="The generator's SERVING window in tokens — the --max-model-len it "
                             "is actually up at (e.g. 131072). A hard cap, not an experimental "
                             "condition: it never renames the output directory and normally "
                             "never binds, but when a prompt would exceed W it is CUT to fit "
                             "instead of being sent and rejected with a 400. Which end survives "
                             "follows the order of the facts: the full-context baseline is "
                             "chronological so the OLDEST turns are dropped first (what an agent "
                             "with a finite window does, and it preserves the latest mention "
                             "that knowledge-update questions need); every ranked path drops "
                             "from the BOTTOM of the ranking. The reservation for the prompt "
                             "template, the question, --max-tokens and the chat template is "
                             "subtracted per question, so W is the SERVER's number and needs no "
                             "hand-adjustment. Off by default, because a cut that nobody asked "
                             "for is worse than a loud failure; see context_window_truncated / "
                             "context_dropped_tokens in answer_stats.jsonl for what it cost. "
                             "The row it exists for is --retriever full on all_turns, whose "
                             "rendered LongMemEval-S haystack is ~110-114k tokens: that fits "
                             "131072 with ~15k to spare and overflows 65536 by ~50k.")
    parser.add_argument("--max-history-tokens", type=int, default=None, metavar="N",
                        help="--retriever full only: cap the supplied conversation history at N "
                             "tokens of the generator's tokenizer, keeping the most recent turns "
                             "that fit (the paper's full-context baseline uses 100000). Applied "
                             "together with --context-window; the tighter of the two binds.")
    parser.add_argument("--gen-model", default="gpt-4o-mini",
                        help="Generator model that answers the question (default: gpt-4o-mini)")
    parser.add_argument("--gen-base-url", default=None,
                        help="Override base URL for the generator (e.g. http://localhost:8000/v1)")
    parser.add_argument("--gen-reasoning-effort", default=None,
                        help="GPT-5/o-series only: none|low|medium|high|xhigh on gpt-5.1+; "
                             "minimal|low|medium|high on gpt-5/-mini/-nano, which reject "
                             "'none' with a 400. "
                             "Ignored by models that don't support it. Lower is cheaper/faster.")
    parser.add_argument("--gen-verbosity", default=None, choices=["low", "medium", "high"],
                        help="GPT-5 only: constrains output length. Omitted = model default.")
    parser.add_argument("--gen-seed", type=int, default=DEFAULT_SEED,
                        help=f"Best-effort reproducibility seed for the generator (OpenAI and "
                             f"vLLM chat models). Pinned to {DEFAULT_SEED} by default.")
    parser.add_argument("--gen-temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature for the generator (OpenAI and vLLM chat "
                             f"models; gpt-5 family ignores it). Pinned to {DEFAULT_TEMPERATURE} "
                             f"(greedy) by default.")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Max tokens for the generated answer (default: 2000). On the "
                             "native path this OVERRIDES the config's answer_max_tokens, so "
                             "the two paths are not compared at different answer lengths; "
                             "pass a system's published value explicitly to run it at its own "
                             "budget (e.g. --max-tokens 4096 for hypermem).")
    parser.add_argument("--batch", action="store_true",
                        help="Answer through OpenAI's Batch API instead of live calls — half "
                             "price, no per-minute rate limit, no per-call latency (latency_sec "
                             "is recorded as None). OpenAI --gen-model only, shared path only.")
    parser.add_argument("--batch-phase", choices=["both", "submit", "collect"], default="both",
                        help="Under --batch: 'both' (default) submit then wait; 'submit' queue "
                             "the cell's batch and exit, writing no answers (fire every cell's "
                             "batch up front); 'collect' wait on the submitted batch and write "
                             "the answers.")
    parser.add_argument("--batch-poll-interval", type=int, default=30,
                        help="Seconds between status checks while waiting on a batch (default: 30)")
    add_limit_arg(parser, help="Only answer the first N samples")
    parser.add_argument("--out", default=None,
                        help="Override the answers dir / predictions.jsonl path "
                             "(default: outputs/_answers/<system>/<dataset>/<mem-model>/<retriever>/<gen-model>/top<k>/predictions.jsonl; "
                             "oracle adds an <labeler-segment> level under oracle/)")
    add_overwrite_arg(parser, help="Regenerate answers already present in predictions.jsonl")

    shared = parser.add_argument_group(
        "shared-prompt path", "Only when NOT --retriever native --prompt native")
    add_emb_model_arg(shared,
                      help="Model used for --retriever dense/sparse/multi_vector (to locate its run dir)")
    # Default is a SENTINEL, not a level: the oracle's default and every other
    # retriever's forced value differ, and a concrete default here cannot express
    # that. It also keeps "did the user ask for a level?" answerable below, which
    # is what the non-oracle guard needs.
    shared.add_argument("--gold", default=None, choices=list(ALL_GOLD_LEVELS),
                        help=f"--retriever oracle only: WHICH evidence set the oracle answers "
                             f"from (evalgrid/benchmarks/gold.py). Default {EVAL_GOLD_LEVEL!r}. "
                             f"'strict' = only the turns/facts the dataset flags has_answer "
                             f"(the ingester-stamped gold) — note this leaves 21 abstention "
                             f"questions with an EMPTY prompt and shortens 32 multi-hop chains, "
                             f"which is why it is no longer the oracle default; "
                             f"'turn' also takes whole evidence sessions that flag nothing; "
                             f"'session' takes every turn of every evidence session. Each level "
                             f"writes under its own '-gold-<level>' path (strict has none), so "
                             f"the oracles never overwrite each other. 'relabel_union' / "
                             f"'relabel_union_insession' / 'relabel' answer from the LLM-relabelled "
                             f"evidence ('turn' + judged / 'turn' + judged inside the evidence "
                             f"sessions / judged only); their qrels exist only where a builder ran with "
                             f"--relabel-set.")
    shared.add_argument("--qrels-dir", default=None,
                        help="Override the qrels dataset dir "
                             "(default: outputs/_qrels/<system>/<dataset>/<mem-model>/<labeler-segment>/). "
                             "Read by both halves: the shared path answers from its corpus, and a "
                             "native replay reads the same corpus to cut a grouped unit "
                             "(MemoryBank's date blocks) on memory boundaries.")
    shared.add_argument("--retrieval-dir", default=None,
                        help="Override the retrieval run dir "
                             "(default: outputs/_retrieval/<system>/<dataset>/<mem-model>/<labeler-segment>/<retriever>/)")
    add_concurrency_arg(shared, help="Concurrent generator calls (default: 8)")
    shared.add_argument("--exclude-question-type", action="append", default=None,
                        metavar="TYPE",
                        help="Skip queries whose metadata.question_type is this value "
                             "(e.g. single-session-assistant). Repeatable.")

    nat = parser.add_argument_group(
        "--retriever native --prompt native", "The system's own loop and its own prompt")
    nat.add_argument("--published-depth", action="store_true",
                     help="Retrieve at the depth in the system's OWN config, ignoring the "
                          "spec's native_top_k. This is the published-depth control (E1); "
                          "without it a spec carrying native_top_k imposes that depth on a "
                          "LIVE native run. Mutually exclusive with --top-k; inert on a "
                          "replay, which does no retrieving.")
    nat.add_argument("--mem-dir", default=None, help="Override the saved-memory dir")
    nat.add_argument("--mem-base-url", default=None,
                     help="Base URL for the memory LLM (e.g. http://localhost:8000/v1). Needed "
                          "by systems that call the memory model at query time (A-Mem's keyword "
                          "generation, SimpleMem's query planning) — so it matters here and on "
                          "no other answer path.")
    nat.add_argument("--native-retriever", default=None, metavar="SPEC",
                     help="Swap the retriever inside the system's own loop; same "
                          "SPEC as evalgrid.pipeline.retrieve. A split "
                          "retrieve/answer pair MUST pass the same one.")
    nat.add_argument("--config", default=None,
                     help="Explicit config path, bypassing variant resolution. This is how you "
                          "change the NATIVE retriever's settings: they live in the system's "
                          "YAML (a_mem retrieve_k/link_k, hypermem retrieval.top_k, simplemem "
                          "retrieval.semantic_top_k, ...), not behind --retriever. Pair it with "
                          "--native-variant so the runs don't collide.")
    nat.add_argument("--native-variant", default=None, metavar="NAME",
                     help="Names this native-retrieval setting and suffixes both output dirs "
                          "(native--NAME). Unlike --mem-variant it does NOT touch "
                          "outputs/_memory/, because these knobs don't change the store.")
    nat.add_argument("--from-retrieval", default=None, metavar="DIR",
                     help="Answer from retrieval already saved by 'retrieve --retriever native' "
                          "instead of running the system's loop again. This is what makes a "
                          "budget sweep affordable: SimpleMem's 5-8 planning calls and A-Mem's "
                          "keyword call are paid once, not once per budget. Required by "
                          "--context-budget on this path.")
    nat.add_argument("--retrieval-out", default=None,
                     help="Override the dir the native run.tsv files are written to. Pass '' "
                          "to skip writing them (answers only).")
    add_shard_args(nat)
    add_trace_args(nat)

    args = parser.parse_args()
    apply_spec(parser, args)
    require_resolved(parser, args, "system", "dataset", "mem_model")

    # The native body is needed only where the answers come from the system's
    # own trace rendered by its own prompt. The other three cells of the
    # (retriever x prompt) grid answer from a corpus and a run.tsv — including
    # `--retriever native --prompt shared`, which reads the saved native
    # ranking, and `--prompt native` on a flat retriever, which is just
    # SystemPlugin.answer_prompt, a template needing no loaded system.
    provided = _cli_provided_dests(parser, sys.argv[1:])
    if args.retriever == NATIVE and args.prompt == "native":
        _reject_foreign_flags(parser, provided, _SHARED_ONLY,
                              "with --retriever native --prompt native: it answers from the "
                              "system's own store and trace, not from a qrels corpus and a "
                              "retrieval run")
        return _run_native(parser, args)
    _reject_foreign_flags(parser, provided, _NATIVE_ONLY,
                          f"for --retriever {args.retriever} --prompt {args.prompt}: no memory "
                          f"system is loaded, so there is no native loop to configure, replay "
                          f"or record")
    # Applied here rather than as an argparse default because the two paths
    # disagree about it: 10 facts is the shared path's depth, while the native
    # path means "each config's own published depth" by omission.
    if args.top_k is None:
        args.top_k = 10
    if args.batch:
        if args.gen_base_url:
            parser.error("--batch is OpenAI-hosted only — incompatible with --gen-base-url")
        if not args.gen_model.startswith(_OPENAI_GEN_PREFIXES):
            parser.error(f"--batch requires an OpenAI --gen-model (got {args.gen_model!r})")

    build_prompt = resolve_prompt(args.system, args.prompt)
    oracle = args.retriever == "oracle"
    full = args.retriever == "full"
    # Neither oracle nor full read a retrieval run.tsv or honour top-k.
    no_retrieval = oracle or full
    if args.context_budget is not None:
        if no_retrieval:
            # Oracle facts are a SET (everything the labeler judged relevant, in
            # no meaningful order) and full is the whole corpus in chronological
            # order. Truncating either to a budget would silently drop evidence
            # by an order that means nothing — an upper bound that isn't one.
            sys.exit(f"--context-budget is not defined for --retriever {args.retriever}: "
                     f"it has no ranking to fill a budget from. Both are deliberately "
                     f"uncapped conditions (path segment 'all'). If what you need is for "
                     f"the prompt to FIT THE SERVER, that is --context-window: a cap at "
                     f"the generator's --max-model-len that drops the oldest turns rather "
                     f"than a budget that redefines the condition.")
        if args.context_budget < 1:
            sys.exit("--context-budget must be a positive number of tokens.")
    if args.context_window is not None and args.context_window < 1:
        sys.exit("--context-window must be a positive number of tokens — the generator's "
                 "--max-model-len.")
    rname = retriever_name(args.retriever, args.emb_model)
    paths = ExperimentPaths(system=args.system, dataset=args.dataset, mem_model=args.mem_model,
                            mem_variant=args.mem_variant)
    # Same <labeler-segment> the qrels/retrieval dirs are keyed by in
    # retrieve / eval_retrieval, so the defaults point at the trees those
    # stages actually wrote.
    seg = labeler_segment(args.labeler_model, args.candidate_source)
    qrels_dir = args.qrels_dir or paths.qrels(seg)
    retrieval_dir = args.retrieval_dir or paths.retrieval(rname)
    # The gold level keys the OUTPUT only. It selects a different qrels FILE in
    # the same qrels dir, over the same corpus — so it must never move the input
    # path, and it must always move the output path: two oracles that answered
    # different evidence sets are two conditions, not one run rewritten.
    # An oracle defaults to EVAL_GOLD_LEVEL because strict is not an upper bound
    # on this benchmark: it leaves the 21 abstention questions with an empty
    # prompt and drops a hop from 32 multi-hop chains (see the constant in
    # evalgrid/benchmarks/gold.py).
    #
    # None for every other retriever, not a level: they answer from a ranking and
    # have NO gold set, so naming one would be a lie the path could pick up. Every
    # use of gold_level below is behind `if oracle`, which is what makes None the
    # honest value rather than a landmine.
    gold_level = (args.gold or EVAL_GOLD_LEVEL) if oracle else None
    if args.gold is not None and not oracle:
        sys.exit(f"--gold {args.gold} is only defined for --retriever oracle: every other "
                 f"retriever answers from a ranking, which has no gold set. Score a RANKING "
                 f"against a wider gold with evalgrid.pipeline.eval_retrieval --gold instead.")
    # Oracle/full use a fixed fact set (relevant docs / whole corpus), so top-k
    # is irrelevant — keep the path honest. A budget run writes budget<T>/.
    topk_seg = answer_depth_segment(args.top_k, args.context_budget, no_retrieval)
    # Retriever answers don't depend on the labeler, so they carry no labeler
    # segment and are shared across labeling experiments; only oracle answers
    # (read from the labeler's qrels) are keyed by it.
    out_path = args.out or os.path.join(
        paths.answers(rname, args.gen_model, topk_seg,
                      labeler_seg=(seg + gold_suffix(gold_level)) if oracle else None,
                      prompt=args.prompt),
        "predictions.jsonl")

    if not os.path.isdir(qrels_dir):
        sys.exit(f"No qrels dataset at {qrels_dir} — run evalgrid.pipeline.qrels_has_answer first.")
    if not no_retrieval and not os.path.isdir(retrieval_dir):
        sys.exit(f"No retrieval run at {retrieval_dir} — run evalgrid.pipeline.retrieve first.")

    print(f"System    : {args.system}")
    print(f"Dataset   : {args.dataset}")
    print(f"Mem model : {args.mem_model}")
    print(f"Retriever : {rname}" + (f"  (gold: {gold_level})" if oracle else ""))
    if args.context_budget is not None:
        print(f"Budget    : {args.context_budget} tokens ({args.gen_model} tokenizer; "
              f"--top-k ignored)")
    else:
        print(f"Top-k     : {args.top_k}")
    print(f"Prompt    : {args.prompt}")
    print(f"Gen model : {args.gen_model}")
    print(f"Out       : {out_path}")

    # base_url routes to the vllm backend; reasoning_effort/verbosity are
    # OpenAI-only while seed is honoured by both OpenAI and vLLM —
    # build_generator forwards them through the tolerant factory, which drops
    # the knobs the chosen backend doesn't accept. No per-backend guard.
    generator = build_generator(
        args.gen_model, base_url=args.gen_base_url,
        reasoning_effort=args.gen_reasoning_effort, verbosity=args.gen_verbosity,
        seed=args.gen_seed, temperature=args.gen_temperature,
    )

    already: set[str] = set()
    if not args.overwrite and os.path.isfile(out_path):
        already = {row.get("question_id") for row in read_jsonl(out_path)}

    sample_ids = sorted(
        d for d in os.listdir(qrels_dir)
        if os.path.isfile(os.path.join(qrels_dir, d, "corpus.jsonl"))
    )
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    # The answering model's own tokenizer — a budget counted with any other one
    # is a different budget. Built once; loading is what fails on an offline
    # node, so do it before the sample loop rather than 400 samples in.
    count_tokens = token_counter(args.gen_model)
    # The cutter for both truncating paths — fit_to_budget's rule 2 and
    # fit_to_window's oversized-unit case. Built beside the counter and for the
    # same reason: it is the answering model's own tokenizer or it is not this
    # budget. None when neither cut can fire, so a plain top-k run does not load
    # a second handle.
    cut_tokens = (token_truncator(args.gen_model)
                  if (args.context_budget is not None or args.context_window is not None)
                  else None)

    excluded_qt = set(args.exclude_question_type or [])
    items: list[tuple[str, str, Optional[str], list[str], dict]] = []
    skipped = missing_run = excluded = 0
    for sample_id in sample_ids:
        if oracle:
            # Ground-truth facts: every doc the labeler judged relevant (score > 0).
            # No retrieval, no run.tsv, no top-k cap — this is the upper bound.
            run = read_qrels(os.path.join(qrels_dir, sample_id, "qrels",
                                          gold_filename(gold_level)))
        elif full:
            run = None  # facts come straight from the corpus, not a run
        else:
            run_path = os.path.join(retrieval_dir, sample_id, "run.tsv")
            if not os.path.isfile(run_path):
                missing_run += 1
                continue
            run = read_run(run_path)
        corpus = read_jsonl(os.path.join(qrels_dir, sample_id, "corpus.jsonl"))
        # Facts carry their timestamp prefix (LightMem retrieve() shape) so the
        # generator can reason about when each memory was recorded.
        corpus_facts = {c["_id"]: format_fact(c) for c in corpus}
        # Token sizes are a property of the doc, not of the query, so they are
        # memoized per sample: full/oracle would otherwise re-tokenise the whole
        # corpus once per question, and a plain top-k run tokenises only the ~10
        # docs it actually shows.
        doc_tokens: dict[str, int] = {}

        def size_of(doc_id: str) -> int:
            if doc_id not in doc_tokens:
                doc_tokens[doc_id] = count_tokens(corpus_facts[doc_id])
            return doc_tokens[doc_id]

        for q in read_jsonl(os.path.join(qrels_dir, sample_id, "queries.jsonl")):
            qid = q["_id"]
            if excluded_qt and (q.get("metadata") or {}).get("question_type") in excluded_qt:
                excluded += 1
                continue
            if qid in already:
                skipped += 1
                continue
            if full:
                # Full-context baseline: feed every corpus doc, in corpus
                # (chronological) order. No retrieval, no top-k cap.
                chosen = [c["_id"] for c in corpus]
            elif oracle:
                chosen = [doc_id for doc_id, score in run.get(qid, {}).items()
                          if score > 0 and doc_id in corpus_facts]
            else:
                ranked = sorted(run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
                # Under a budget the ranking is cut by tokens, not by depth, so
                # the whole ranking is a candidate — --top-k would cap it at an
                # arbitrary point before the budget ever bound.
                depth = len(ranked) if args.context_budget is not None else args.top_k
                chosen = [doc_id for doc_id, _ in ranked[:depth] if doc_id in corpus_facts]
            question_date = (q.get("metadata") or {}).get("question_date")
            sizes = [size_of(i) for i in chosen]
            facts, ctx = fit_to_budget([corpus_facts[i] for i in chosen], sizes,
                                       args.context_budget, truncate=cut_tokens)
            history_cap = args.max_history_tokens if full else None
            if args.context_window is not None or history_cap is not None:
                # Second, independent cut: the budget above (if any) chose the
                # EVIDENCE, this one only guarantees the request fits the server.
                # Sizes are those of what survived the budget cut — unbudgeted
                # that is the memoized per-doc sizes untouched, which is what
                # keeps the full-context path from re-tokenising a 110k-token
                # corpus once per question; budgeted the list is at most T tokens
                # and may end in a fragment, so recounting it is both cheap and
                # necessary.
                kept_sizes = (sizes if args.context_budget is None
                              else [count_tokens(f) for f in facts])
                room = (_facts_window(build_prompt, q["text"], question_date,
                                      count_tokens, args.context_window, args.max_tokens)
                        if args.context_window is not None else history_cap)
                if history_cap is not None:
                    room = min(room, history_cap)
                if room < 1:
                    sys.exit(
                        f"--context-window {args.context_window} leaves no room for facts on "
                        f"question {qid}: the prompt template, the question and --max-tokens "
                        f"{args.max_tokens} already exceed it. Raise the window or lower "
                        f"--max-tokens.")
                facts, wstats = fit_to_window(
                    facts, kept_sizes, room,
                    # Chronological corpus vs ranking — see fit_to_window.
                    keep="newest" if full else "top", truncate=cut_tokens)
                ctx.update(wstats)
            items.append((qid, q["text"], question_date, facts, ctx))

    if missing_run:
        print(f"  ! {missing_run} sample(s) have qrels but no run.tsv — "
              f"run evalgrid/pipeline/retrieve.py for them first.")

    results: list[tuple] = []
    run_wall_sec = 0.0
    batch_work_dir = None
    if items and args.batch:
        import openai

        from evalgrid.benchmarks.answer_batch import (collect_answer_batch,
                                                      submit_answer_batch, work_dir_for)
        client = openai.OpenAI()
        batch_work_dir = work_dir_for(os.path.dirname(out_path), model_slug(args.gen_model))
        if args.batch_phase in ("both", "submit"):
            submit_answer_batch(
                client, items, build_prompt, model=args.gen_model,
                max_tokens=args.max_tokens, work_dir=batch_work_dir,
                reasoning_effort=args.gen_reasoning_effort, verbosity=args.gen_verbosity,
                seed=args.gen_seed, temperature=args.gen_temperature,
            )
        if args.batch_phase == "submit":
            # Nothing is written beside predictions.jsonl: the cell is not
            # answered until collect, and a config.json here would say it was.
            print(f"\n[submit] batch queued for {len(items)} question(s) of {out_path} "
                  f"— collect later with --batch-phase collect")
            return
        print(f"\nCollecting batch answers for {len(items)} question(s)...")
        run_started = perf_counter()
        results = collect_answer_batch(
            client, items, work_dir=batch_work_dir, max_tokens=args.max_tokens,
            usage_sink=generator.token_usage, poll_interval=args.batch_poll_interval)
        run_wall_sec = perf_counter() - run_started
    elif items:
        print(f"\nAnswering {len(items)} question(s)...")
        run_started = perf_counter()
        results = asyncio.run(_run(generator, args.concurrency, args.max_tokens,
                                  build_prompt, items))
        run_wall_sec = perf_counter() - run_started

    out_run_dir = os.path.dirname(out_path)
    stats_path = os.path.join(out_run_dir, "answer_stats.jsonl")

    # A RUN THAT GENERATED NOTHING WRITES NOTHING. Every question of this cell
    # was already in predictions.jsonl (or excluded, or had no run.tsv), so the
    # artefacts beside it belong to the invocation that actually produced those
    # answers and must keep describing IT, not this no-op pass over them:
    #
    #   config.json / manifest.json are rewritten from the CURRENT args, so a
    #   re-sweep under a different --context-window, --gen-temperature or
    #   --max-tokens would relabel every already-complete cell with settings its
    #   answers were never generated under, and nothing on disk would say so.
    #
    #   answer_summary.json survives its own rewrite (it re-reads the cumulative
    #   answer_stats.jsonl, so `answered` and the totals stay), but
    #   latency.wall_sec_this_invocation and
    #   generator_cumulative_tokens_this_invocation are per-invocation: they
    #   would drop to ~0 for a cell that cost a GPU-hour.
    #
    # This is the resume path's ORDINARY end state, not an error — the stage
    # skips by question_id, so re-submitting a finished sweep to fill three
    # missing cells now leaves the other twenty-nine untouched, mtimes included.
    # Guarded on an existing predictions.jsonl so a genuinely new cell still
    # writes its config even when it had nothing to answer.
    if not results and os.path.isfile(out_path):
        print(f"\nNothing to answer — every question of this cell is already in "
              f"{out_path}.")
        print("  untouched: predictions.jsonl, answer_stats.jsonl, config.json, "
              "manifest.json, answer_summary.json")
        if skipped:
            print(f"  skipped  : {skipped} (already answered — use --overwrite to regenerate)")
        if excluded:
            print(f"  excluded : {excluded} (question_type in {sorted(excluded_qt)})")
        return

    os.makedirs(out_run_dir, exist_ok=True)

    # Drop a config.json next to predictions.jsonl so the run is self-describing
    # (and evalgrid/pipeline/judge.py --run can read the dataset from it). Written every
    # run so it stays in sync with the latest invocation.
    with open(os.path.join(out_run_dir, "config.json"), "w") as f:
        json.dump({
            "system": args.system,
            "dataset": args.dataset,
            "mem_model": args.mem_model,
            "retriever": rname,
            "prompt": args.prompt,
            "gen_model": args.gen_model,
            "gen_reasoning_effort": args.gen_reasoning_effort,
            "gen_verbosity": args.gen_verbosity,
            "gen_seed": args.gen_seed,
            "gen_temperature": args.gen_temperature,
            # Batch answers carry no per-call latency (latency_sec is None).
            "gen_batch": bool(args.batch),
            "top_k": None if (no_retrieval or args.context_budget is not None) else args.top_k,
            "context_budget": args.context_budget,
            "answer_stats": os.path.basename(stats_path),
            "answer_summary": "answer_summary.json",
        }, f, indent=2)
    write_manifest(out_run_dir, args, stage="answer")

    mode = "a" if (not args.overwrite and os.path.isfile(out_path)) else "w"
    errors = [err for _, _, err in results if err is not None]
    with open(out_path, mode) as f:
        for rec, _, _ in results:
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(stats_path, mode) as f:
        for _, stats, _ in results:
            if stats is not None:
                f.write(json.dumps(stats, ensure_ascii=False) + "\n")
    # Errors before the summary, which charges their tokens to failed_calls.
    if errors:
        with open(os.path.join(os.path.dirname(out_path), "errors.jsonl"), "a") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    _write_answer_summary(out_run_dir, stats_path, generator, run_wall_sec)
    if batch_work_dir:
        # Only now that its answers are on disk: a crash before this line
        # leaves the batch resumable instead of paid for and lost.
        from evalgrid.benchmarks.answer_batch import retire_answer_batch
        retire_answer_batch(batch_work_dir)

    print(f"\nDone — predictions written to {out_path}")
    print(f"Stats written to {stats_path}")
    print(f"  answered : {len(results) - len(errors)}")
    ctx_rows = [s for _, s, _ in results if s is not None]
    if ctx_rows:
        c = _context_summary(ctx_rows)
        print(f"  context  : {c['avg_tokens']:,.0f} tok avg over {c['avg_items']:.1f} items "
              f"(max {c['max_tokens']:,})")
        if args.context_window is not None and c["window_truncated"]:
            # Loud, at the end of the run, for the same reason the budget line
            # is: a row cut to fit the server is not the row its directory name
            # claims, and nobody will read answer_summary.json first.
            print(f"  ! window  : {args.context_window} tok — {c['window_truncated']} of "
                  f"{len(ctx_rows)} question(s) CUT to fit the generator "
                  f"(avg {c['avg_dropped_tokens']:,.0f} tok of evidence dropped, "
                  f"max {c['max_dropped_tokens']:,}). Report this row as truncated.")
        elif args.context_window is not None:
            print(f"  window   : {args.context_window} tok — nothing cut, every prompt fit")
        if args.context_budget is not None:
            # Printed, not buried in the summary file: an under-filled or
            # overflowing budget cell is not comparable, and that has to be
            # visible at the moment the run finishes.
            print(f"  budget   : {args.context_budget} tok — "
                  f"{c['truncated']} truncated, {c['underfilled']} under-filled, "
                  f"{c['unit_truncated']} answered from a CUT top unit"
                  + (f", {c['overflow']} over budget (no truncator)" if c['overflow'] else ""))
    if excluded:
        print(f"  excluded : {excluded} (question_type in {sorted(excluded_qt)})")
    if skipped:
        print(f"  skipped  : {skipped} (already answered — use --overwrite to regenerate)")
    if errors:
        print(f"  errors   : {len(errors)} (see {os.path.join(os.path.dirname(out_path), 'errors.jsonl')})")


if __name__ == "__main__":
    main()
