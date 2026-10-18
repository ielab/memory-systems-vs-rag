"""
evalgrid/systems/lightmem/system.py
------------------------------
Base class wrapping LightMemory behind the MemorySystem interface.

Holds the shared LightMem plumbing — model lifecycle, token accounting, and
timestamp normalisation. Dataset-specific ingest/answer behaviour lives in
per-dataset subclasses:
  longmemeval.py — LightMemLongMemEval: flat extraction (user-only),
                   combined retrieval, simple prompt
  locomo.py      — LightMemLoCoMo: event extraction (factual+relational,
                   both speakers), per-speaker balanced retrieval, rich
                   timestamped answer prompt
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from evalgrid.benchmarks.types import MemorySystem
from evalgrid.models.base import GeneratorModel, TokenUsage
from evalgrid.lib.trace import record_calls, record_steps

_FALLBACK_DT = datetime(2020, 1, 1, 0, 0)


# --- meta_text_extract return-value readers (see _install_trace) -------------
#
# The method returns one dict per API call it made, carrying that call's
# prompt, raw completion and usage — the same list process_extraction_results
# later folds into token_stats. Reading it here is what gives the extract step
# a cost and a payload at all; both are otherwise invisible to a wrapper around
# the method. Every field is optional on purpose: a call that failed is a dict
# without "usage", which must contribute nothing rather than raise.

#: Marks meta_text_extract as already wrapped by _install_extract_counters, so
#: a re-install (reset() rebuilds LightMemory, but install_recorder does not)
#: cannot stack wrappers and double-count. Mirrors trace._WRAPPED.
_COUNTER_INSTALLED = "_evalgrid_truncation_counted"

#: Same guard for the pre-update text recorder (_install_update_provenance),
#: which rebinds a second vendored method on the same instance.
_UPDATE_PROVENANCE_INSTALLED = "_evalgrid_update_provenance_recorded"

#: Which roles' content ``messages_use`` actually sends to the extractor
#: (vendor/lightmem/factory/memory_manager/openai.py: role_filter). A turn whose
#: role is not here is handed to ``add_memory`` as part of its pair and then
#: dropped by the extractor's own concatenation, so it can NEVER carry a memory
#: unit — which makes it meaningless in ``turns_without_memory``. An unknown
#: value falls back to both roles, i.e. to the old behaviour: this must never
#: under-report by guessing.
_MESSAGES_USE_ROLES = {
    "user_only": frozenset({"user"}),
    "assistant_only": frozenset({"assistant"}),
    "hybrid": frozenset({"user", "assistant"}),
}


def _extract_items(results) -> list[dict]:
    return [item for item in (results or []) if isinstance(item, dict)]


def _extract_cost(results) -> tuple[int, int, int]:
    calls = ins = outs = 0
    for item in _extract_items(results):
        usage = item.get("usage")
        if not usage:
            continue
        calls += 1
        ins += int(usage.get("prompt_tokens", 0) or 0)
        outs += int(usage.get("completion_tokens", 0) or 0)
    return (ins, outs, calls)


def _extract_output(results) -> str:
    return "\n\n".join(str(item.get("output_prompt", ""))
                       for item in _extract_items(results))


# --- per-fact provenance, captured at extraction time (see _capture_provenance)
#
# WHY THIS IS RECORDED HERE RATHER THAN READ BACK OFF THE POINT.
# ``payload["turn_id"]`` and ``payload["has_answer"]`` are wrong for ~28% of
# units and the error is systematic. Upstream's extraction prompt labels each
# user message ``sequence_number // 2``
# (vendor/lightmem/factory/memory_manager/openai.py:309) and inverts the
# extractor's ``source_id`` as ``source_id * 2``
# (vendor/lightmem/memory/utils.py:338-339). That inverse holds only while every
# user message sits at an EVEN index within one extraction call, and upstream
# itself breaks that: it fabricates empty-content placeholder messages (145 over
# 60 samples, all ``role: user``, no ``turn_id``) and its topic segmenter cuts
# the buffer mid-pair, so segments begin with an assistant turn. Each such
# message flips the parity of everything after it and ``source_id * 2`` then
# addresses the neighbouring ASSISTANT turn. ``has_answer`` reads the same
# broken index, so the two fields fail together and neither can check the other.
#
#
# The arithmetic is upstream's and stays (vendor/ is not modified). What is ours
# is trusting its inverse. So the label the prompt ACTUALLY printed is recomputed
# from the same ``extract_list`` upstream was handed, and joined to the
# ``source_id`` the model actually returned — the correct inverse, recorded while
# both halves are still in hand.

def _label_map(extract_list) -> dict:
    """``sequence_number // 2`` -> the USER messages the prompt printed under it.

    A list, not one message: the parity break can make two user messages collide
    on one label (0.8% of messages measured). Both are then credited, which
    matches the group-level provenance every other system in the grid writes and
    is honest about the ambiguity rather than picking one arbitrarily.

    ``extract_list`` nests batch -> segment -> message; a malformed level is
    skipped rather than raised on, because this runs inside ingest and must never
    be the thing that takes a sample down.
    """
    labels: dict = {}
    for batch in extract_list or ():
        if not isinstance(batch, (list, tuple)):
            continue
        for segment in batch:
            if not isinstance(segment, (list, tuple)):
                continue
            for message in segment:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                # A placeholder upstream fabricated: no turn_id, so it can carry
                # no provenance. It still occupies a sequence number — which is
                # the parity break itself — so it is skipped here, not renumbered.
                if not message.get("turn_id"):
                    continue
                try:
                    label = int(message["sequence_number"]) // 2
                except (KeyError, TypeError, ValueError):
                    continue
                labels.setdefault(label, []).append(message)
    return labels


def _extract_facts(item: dict):
    """Yield ``(source_id, fact text)`` from one extraction call's raw completion.

    Parsed per returned item rather than over the concatenation of all of them:
    one ``meta_text_extract`` call returns one dict per API call it made, so a
    call covering several segments returns several JSON objects and a single
    first-brace-to-last-brace parse over the joined string spans all of them.
    """
    text = item.get("output_prompt")
    if not text:
        return
    text = str(text)
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return
    for entry in (obj.get("data") or ()):
        if not isinstance(entry, dict):
            continue
        fact = entry.get("fact") or entry.get("relation")
        if not fact:
            continue
        try:
            source_id = int(entry.get("source_id"))
        except (TypeError, ValueError):
            # Upstream silently charges a source_id-less fact to source_id 0.
            # Dropping it is the more honest answer: the unit then falls back to
            # the payload stamp and is labelled as having done so.
            continue
        yield source_id, str(fact)


# --- _call_update_llm readers (see _install_trace) ---------------------------
#
# Signature is (system_prompt, target_entry, candidate_sources); the return is
# the parsed decision dict plus the call's own `usage`
# (vendor/lightmem/factory/memory_manager/openai.py:386-412). Every reader is
# total: a recorder that raises on an odd payload would take down an ingest.

def _update_prompt(*args, **kwargs) -> str:
    """Reconstruct what the update call was actually asked.

    Mirrors upstream's own user-prompt construction rather than dumping the raw
    arguments, so the log shows the memories being compared instead of two
    Qdrant point dicts.
    """
    try:
        system_prompt = kwargs.get("system_prompt") or (args[0] if args else "")
        target = kwargs.get("target_entry") or (args[1] if len(args) > 1 else {})
        candidates = kwargs.get("candidate_sources") or (args[2] if len(args) > 2 else [])
        target_memory = (target.get("payload") or {}).get("memory", "")
        lines = "\n".join(f"- {(c.get('payload') or {}).get('memory', '')}" for c in candidates)
        return (f"system: {system_prompt}\n"
                f"user: Target memory:{target_memory}\nCandidate memories:\n{lines}")
    except Exception:
        return "<update prompt unavailable>"


def _update_output(result) -> str:
    """The decision, without the usage block (which is logged as cost)."""
    try:
        return json.dumps({k: v for k, v in (result or {}).items() if k != "usage"},
                          ensure_ascii=False)
    except Exception:
        return repr(result)


def _update_cost(result) -> tuple[int, int, int]:
    usage = (result or {}).get("usage") or {}
    return (int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
            1 if usage else 0)

# Default output format (LongMemEval). LoCoMo overrides to match the reference
# implementation (zjunlp/LightMem/experiments/locomo), which emits
# '%Y-%m-%d %H:%M:%S'.
_LME_OUT_FMT = "%Y/%m/%d (%a) %H:%M"
_LOCOMO_OUT_FMT = "%Y-%m-%d %H:%M:%S"

_INPUT_FMTS = [
    "%Y/%m/%d (%a) %H:%M",        # LongMemEval: '2023/05/20 (Sat) 02:21'
    "%Y/%m/%d (%a) %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%I:%M %p on %d %B, %Y",      # LoCoMo:      '1:56 pm on 8 May, 2023'
]


def _to_lightmem_ts(ts: Optional[str], out_fmt: str = _LME_OUT_FMT) -> str:
    if not ts:
        return _FALLBACK_DT.strftime(out_fmt)
    for fmt in _INPUT_FMTS:
        try:
            return datetime.strptime(ts.strip(), fmt).strftime(out_fmt)
        except ValueError:
            continue
    return _FALLBACK_DT.strftime(out_fmt)


class LightMemSystem(MemorySystem):
    """Shared LightMem plumbing. Subclasses implement ``ingest`` and ``answer``.

    Args:
        config: a configured BaseMemoryConfigs instance
        generator: GeneratorModel used for answer generation (LightMem only
                   handles memory storage/retrieval; answer generation is ours)
        top_k: number of memories to retrieve per query
        max_tokens: generation budget for the answer
    """

    supports_persistence = True

    def __init__(
        self,
        config,
        generator: GeneratorModel,
        top_k: int = 20,
        max_tokens: int = 2000,
    ):
        self._config = config
        self.generator = generator
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.traces: list[dict] = []
        self._pending: Optional[tuple] = None
        self.dropped_turns = 0
        # Turn ids that DID reach an add_memory call. dropped_turns is the
        # complement of this at the pairing step; the difference between it and
        # the turn ids on the saved points is the third case, which had no name
        # until 2026-08-19: a turn that was paired, sent, and came back with no
        # memory. See save_memory's turns_without_memory.
        self.ingested_turn_ids: set[str] = set()
        # The subset of the above whose role ``messages_use`` actually sends.
        # turns_without_memory is measured against THIS, not against
        # ingested_turn_ids — see save_memory.
        self.addressable_turn_ids: set[str] = set()
        # fact text -> {"turn_ids": set, "has_answer": bool}, filled at
        # extraction time. See _capture_provenance and save_memory.
        self._fact_provenance: dict = {}
        # unit id -> the text it carried the FIRST time an offline update touched
        # it. Offline update rewrites an entry's text in place, so a rewritten
        # unit no longer matches any extracted fact by text; this is what still
        # joins it. Earliest wins, because that is the one extraction produced.
        self._pre_update_text: dict = {}
        self._lightmem = self._new_lightmem()

    def _new_lightmem(self):
        from lightmem.memory.lightmem import LightMemory
        lm = LightMemory(self._config)
        self._truncated_calls = 0
        self._failed_calls = 0
        # Cleared HERE rather than in reset(), which is the per-sample boundary:
        # both __init__ and reset() route through this method, so a future entry
        # point cannot acquire a fresh LightMemory while keeping the previous
        # sample's provenance and silently cross-attribute its units.
        self._fact_provenance = {}
        self._pre_update_text = {}
        self._install_extract_counters(lm)
        self._install_update_provenance(lm)
        return lm

    def reset(self) -> None:
        self._lightmem = self._new_lightmem()
        self.traces: list[dict] = []
        self._pending = None
        self.dropped_turns = 0
        self.ingested_turn_ids = set()
        self.addressable_turn_ids = set()
        self._install_trace()

    # ------------------------------------------------------------------
    # The end-of-sample flush
    # ------------------------------------------------------------------
    #
    # LightMem buffers: ``add_memory`` accumulates messages into the sentence
    # buffer and only segments/extracts when a threshold trips — or when
    # ``force_segment``/``force_extract`` say so. So the LAST call of a sample
    # must carry both flags, or whatever is still buffered is never emitted and
    # is lost with no error (vendor/lightmem/memory/lightmem.py:313-326 returns
    # early on an empty segment list).
    #
    # Tying that flag to "the last pair of the last session" is what broke:
    # a session can produce no ``add_memory`` call at all — LongMemEval-S has 2
    # samples (3d86fd0a, gpt4_e061b84g) ending in a single assistant turn, which
    # yields no (user, assistant) pair — and then the flush never fires.
    #
    # So the call is held back by one instead. ``queue_add_memory`` sends the
    # PREVIOUS call and holds the new one; ``flush_add_memory`` sends whatever is
    # held, forced. The sequence of calls and their contents are unchanged — only
    # which one carries the flags, and now it is always the sample's last one,
    # whichever session it came from. A sample that made no call at all holds
    # nothing and has nothing buffered to lose.

    def queue_add_memory(self, messages: list[dict], **kwargs) -> None:
        """Hold one ``add_memory`` call, sending the previously held one unforced."""
        if self._pending is not None:
            held, held_kwargs = self._pending
            self._lightmem.add_memory(held, force_extract=False, force_segment=False,
                                      **held_kwargs)
        self._pending = (messages, kwargs)

    def flush_add_memory(self) -> None:
        """Send the held call with segmentation and extraction forced.

        Idempotent and safe on an empty queue: a sample whose sessions produced
        no call has nothing buffered inside LightMem either.
        """
        if self._pending is None:
            return
        held, held_kwargs = self._pending
        self._pending = None
        self._lightmem.add_memory(held, force_extract=True, force_segment=True,
                                  **held_kwargs)

    # ------------------------------------------------------------------
    # Truncation counting
    # ------------------------------------------------------------------

    def _extract_max_tokens(self) -> Optional[int]:
        """The output budget every extraction call is sent with, if configured."""
        manager = getattr(self._config, "memory_manager", None)
        configs = getattr(manager, "configs", None)
        value = getattr(configs, "max_tokens", None)
        try:
            return int(value) if value else None
        except (TypeError, ValueError):
            return None

    def _install_extract_counters(self, lm) -> None:
        """Count extraction calls that ran to the output cap, and ones that raised.

        Two counters, one wrapper, because both are read off the same returned
        list and neither is visible anywhere else: a clipped call comes back with
        a usage whose completion_tokens hit the cap, a failed one comes back
        with no usage at all.

        Every other system in this repo reports ``truncated_calls`` in its
        meta.json, because a completion stopped at ``max_tokens`` cannot be valid
        JSON and upstream parses it inside a try/except — so a clipped extraction
        is silently *no memories* for that segment rather than an error.

        LightMem is the one system where that number cannot come from a
        ``finish_reason``: the vendored managers never read one (grep
        vendor/lightmem for it — no hits), so the per-call dicts they return
        carry usage and nothing else. The proxy is a completion whose
        ``completion_tokens`` landed exactly on the configured budget, which is
        what a truncation looks like from the outside.

        Instance rebind rather than a wrapper installed by ``record_calls``: this
        must hold whether or not tracing is on, since meta.json is written on
        every run and the trajectory is optional.
        """
        budget = self._extract_max_tokens()
        manager = getattr(lm, "manager", None)
        target = getattr(manager, "meta_text_extract", None)
        # Installed whether or not a budget is configured: truncation needs one,
        # a failed call does not, and gating both on max_tokens is what would
        # leave the second invisible on any config that sets no cap.
        if target is None or getattr(target, _COUNTER_INSTALLED, False):
            return

        def counted(*args, **kwargs):
            results = target(*args, **kwargs)
            # Same seam, second job: the correct source_id -> turn inverse, taken
            # while the extract_list and the extractor's reply are both in hand.
            # Deliberately inside the counter wrapper rather than in the trace
            # layer — provenance must hold whether or not --trace-level is on,
            # for exactly the reason the counters do.
            self._capture_provenance(
                kwargs.get("extract_list") or (args[0] if args else None), results)
            for item in _extract_items(results):
                usage = item.get("usage")
                if not usage or not any(int(v or 0) for v in usage.values()):
                    # Upstream's failed-call sentinel: process_segment_wrapper
                    # caught the exception, printed it, and returned an item
                    # with no usable usage. The segment produced no memory, and
                    # nothing else records it — the trajectory cannot either,
                    # because the exception never escapes the method
                    # record_calls wraps. This is the only counter that sees it.
                    #
                    # Both sentinels, because the two backends disagree:
                    # openai.py:373 returns "usage": None, vllm.py:317 returns a
                    # ZEROED dict, which is truthy. Testing only for falsiness
                    # counted the first and missed the second.
                    #
                    # The zero test costs one thing, on the vllm backend only: a
                    # call that genuinely succeeded against a server that sent no
                    # usage block builds the same zeroed dict, so it counts as
                    # failed. Upstream made those two states byte-identical and
                    # nothing here can separate them; the config this repo runs
                    # uses the openai backend, where the sentinel is None and the
                    # test is exact.
                    self._failed_calls += 1
                    continue
                if budget is not None and int(usage.get("completion_tokens", 0) or 0) >= budget:
                    self._truncated_calls += 1
            return results

        setattr(counted, _COUNTER_INSTALLED, True)
        manager.meta_text_extract = counted

    def _capture_provenance(self, extract_list, results) -> None:
        """Record which turns each extracted fact came from, for this call.

        Total by construction: provenance is a measurement of the ingest, and a
        measurement must never be able to fail the thing it measures. Anything
        malformed contributes nothing and the unit falls back to the payload
        stamp, which is labelled as such in the sidecar.
        """
        try:
            labels = _label_map(extract_list)
            if not labels:
                return
            for item in _extract_items(results):
                for source_id, fact in _extract_facts(item):
                    for message in labels.get(source_id, ()):
                        record = self._fact_provenance.setdefault(
                            fact, {"turn_ids": set(), "has_answer": False})
                        record["turn_ids"].add(message["turn_id"])
                        if message.get("has_answer"):
                            record["has_answer"] = True
        except Exception:
            return

    def _install_update_provenance(self, lm) -> None:
        """Remember each entry's pre-update text, so offline update cannot orphan it.

        ``offline_update_all_entries`` rewrites a target entry's ``memory`` in
        place. The fact map above is keyed by the text the extractor produced, so
        a rewritten unit matches nothing and would fall back to the payload stamp
        — the very field this exists to stop trusting. That was 18.6% of units on
        the shipped store.

        ``_call_update_llm`` is the per-call seam and receives the target entry
        with its id and its CURRENT text, before the rewrite. Recording the first
        text seen per id is enough to join the saved point back to the fact that
        produced it, however many times it is later rewritten.

        Instance rebind, like the counters, and for the same reason: it must hold
        whether or not tracing is on. It runs BEFORE record_calls wraps the same
        method in _install_trace, so both wrappers apply and neither hides the
        other.
        """
        manager = getattr(lm, "manager", None)
        target = getattr(manager, "_call_update_llm", None)
        if target is None or getattr(target, _UPDATE_PROVENANCE_INSTALLED, False):
            return

        def remembering(*args, **kwargs):
            try:
                entry = kwargs.get("target_entry") or (args[1] if len(args) > 1 else None)
                if isinstance(entry, dict):
                    unit_id = entry.get("id")
                    text = (entry.get("payload") or {}).get("memory")
                    if unit_id is not None and text:
                        # setdefault, never overwrite: a unit updated twice must
                        # keep the text extraction actually produced, not the
                        # first rewrite of it.
                        self._pre_update_text.setdefault(str(unit_id), str(text))
            except Exception:
                pass
            return target(*args, **kwargs)

        setattr(remembering, _UPDATE_PROVENANCE_INSTALLED, True)
        manager._call_update_llm = remembering

    def _addressable_roles(self) -> frozenset:
        """Roles whose content ``messages_use`` actually sends to the extractor."""
        return _MESSAGES_USE_ROLES.get(
            getattr(self._config, "messages_use", None),
            frozenset({"user", "assistant"}))

    def _failed_this_sample(self) -> int:
        """Extraction calls that raised inside ``meta_text_extract``, per sample.

        Per-sample without a baseline for the same reason as the truncation
        counter: ``reset()`` rebuilds LightMemory, so the wrapper and its count
        are new each time.
        """
        return getattr(self, "_failed_calls", 0)

    def _truncated_this_sample(self) -> int:
        """Extraction calls that ran to the output cap since ``reset()``.

        No baseline subtraction needed, unlike a_mem/hypermem: ``reset()``
        rebuilds LightMemory, so the counter is already per-sample.
        """
        return getattr(self, "_truncated_calls", 0)

    # ------------------------------------------------------------------
    # Trajectory recording (evalgrid/lib/trace.py) — L1 + L2
    # ------------------------------------------------------------------

    def install_recorder(self, recorder) -> None:
        super().install_recorder(recorder)
        self._install_trace()

    def _token_stats_triple(self):
        """(input, output, calls) from LightMemory's own counters.

        LightMem has no per-call usage object to diff — the vendored code
        accumulates into a ``token_stats`` dict — so cost deltas are read from
        the same three families ``layer_usage`` sums, which keeps a step's cost
        and the run's total from disagreeing.
        """
        ts = getattr(self._lightmem, "token_stats", {}) or {}
        return (
            ts.get("add_memory_prompt_tokens", 0) + ts.get("update_prompt_tokens", 0)
            + ts.get("summarize_prompt_tokens", 0),
            ts.get("add_memory_completion_tokens", 0) + ts.get("update_completion_tokens", 0)
            + ts.get("summarize_completion_tokens", 0),
            ts.get("add_memory_calls", 0) + ts.get("update_calls", 0)
            + ts.get("summarize_calls", 0),
        )

    def _install_trace(self) -> None:
        """Wrap LightMem's pipeline stages on the vendored instance.

        LightMem is the one system with no LLM adapter of ours to hook (the
        vendored manager talks to litellm directly and only keeps aggregate
        counters), and also the one whose pipeline is mostly NOT LLM calls —
        pre-compression, segmentation and slicing are local compute. So this is
        L1 and L2 in the same place, wrapping the four vendored methods that
        are the real stage boundaries:

        =================================  ========  =================================
        method                             stage     what it is
        =================================  ========  =================================
        ``compressor.compress``            compress  LLMLingua-2, no LLM call
        ``senmem_buffer_manager.
        cut_with_segmenter``               segment   embedding + cosine cut, no LLM call
        ``manager.meta_text_extract``      extract   THE memory-LLM call
        ``offline_update_all_entries``     update    the offline update pass
        =================================  ========  =================================

        ``compressor``/``segmenter`` exist only when ``pre_compress`` /
        ``topic_segment`` are on, so each is guarded — a config with them off
        must record no such step rather than fail.

        ``meta_text_extract`` is wrapped instead of the manager's
        ``generate_response`` because it is the stage boundary; its cost,
        however, cannot come from a delta. ``LightMemory.add_memory`` posts the
        extraction's tokens to ``token_stats`` in ``process_extraction_results``
        — which runs AFTER ``meta_text_extract`` returns (vendor/lightmem/
        memory/lightmem.py:348-363) — so the delta across the method is always
        exactly zero, and the calls it made show up nowhere. The usage is
        already on the return value, one entry per API call, so ``cost_of``
        reads it from there; that is also where the real prompts and responses
        are, hence ``prompt_of``/``response_of``.

        The other three keep the delta, which is correct for them:
        ``offline_update_all_entries`` posts its own tokens before returning,
        and the two local-compute stages spend none.

        ``cut_with_segmenter``'s first positional argument is the *segmenter*,
        not the text, so the default ``prompt_arg=0`` would log
        ``<LlmLingua2Segmenter object at 0x...>``. Its real input is the buffer
        the method then clears, read off the instance before the call.
        """
        lm = self._lightmem
        usage = self._token_stats_triple
        buffers = getattr(lm, "senmem_buffer_manager", None)
        for obj, method, stage, step, extra in (
            (getattr(lm, "compressor", None), "compress", "compress",
             "lightmem.compress", {}),
            (buffers, "cut_with_segmenter", "segment", "lightmem.cut_with_segmenter",
             {"prompt_of": lambda *a, **k: list(getattr(buffers, "buffer", []) or [])}),
            (getattr(lm, "manager", None), "meta_text_extract", "extract",
             "lightmem.meta_text_extract",
             {"prompt_of": lambda *a, **k: k.get("extract_list") or (a[0] if a else None),
              "response_of": _extract_output, "cost_of": _extract_cost}),
            # The individual update calls. Previously the update stage was one
            # record wrapping the whole pass — 149 LLM calls in a single line,
            # with their prompts and responses unrecoverable, while every other
            # system's update calls were logged one by one. `_call_update_llm`
            # is the per-call seam, and it returns its own `usage` (the delta
            # cannot see it: offline_update_all_entries posts to token_stats
            # only after every thread has finished).
            (getattr(lm, "manager", None), "_call_update_llm", "update",
             "lightmem.update_entry",
             {"prompt_of": _update_prompt, "response_of": _update_output,
              "cost_of": _update_cost}),
        ):
            if obj is not None:
                record_calls(obj, method, recorder=self.recorder,
                             stage=stage, step=step, usage=usage, **extra)

        # L2 over the whole offline pass, so the per-call records above have a
        # parent and the update stage still has one line reporting its total.
        # It MUST be a step, not a call: as a call it would carry the same
        # tokens as the 149 records nested inside it, and anything summing
        # `kind == "call"` would double-count lightmem's update cost — the one
        # invariant the trajectory has (sum of calls == usage.json).
        record_steps(lm, "offline_update_all_entries", recorder=self.recorder,
                     stage="update", step="lightmem.offline_update_all_entries",
                     usage=usage)

    def _all_points(self) -> list[dict]:
        """Every stored point, payload included — the same rows save_memory dumps.

        Only used when a substituted retriever ranks by unit id and the payload
        has to be looked back up (native_retrieval.py). The live store is the
        source, not points.jsonl, so this works mid-run before anything is saved.
        """
        return self._lightmem.embedding_retriever.get_all(
            with_vectors=False, with_payload=True) or []

    def retrieve_with_ids(self, question: str, limit: int | None = None) -> tuple[list[str], list[str]]:
        """LightMem's own retrieval, plus the point ids it ranked.

        ``LightMemory.retrieve`` returns only the formatted "<time_stamp>
        <weekday> <memory>" strings, dropping the ids the vector store came back
        with, so the native run could not be scored as retrieval. Re-querying to
        recover them would re-embed the question and could diverge from what was
        actually ranked; instead the one ``embedding_retriever.search`` call
        ``retrieve`` makes is captured, which is exact by construction.

        The ids are Qdrant point ids — the same ids
        LightMemPlugin.iter_units yields as unit_ids, since both come from the
        same points.jsonl.
        """
        captured: list[list] = []
        retriever = self._lightmem.embedding_retriever
        original_search = retriever.search
        ranker = getattr(self, "_native_ranker", None)

        def _recording_search(*args, **kwargs):
            if ranker is not None:
                # A substituted retriever (evalgrid/systems/native_retrieval.py).
                # Returned in the store's own row shape so upstream's formatter
                # renders it unchanged — LightMem has no post-retrieval step, so
                # the ranking is the whole of its read path.
                k = kwargs.get("limit") or (args[1] if len(args) > 1 else None) or self.top_k
                by_id = {str(p["id"]): p for p in self._all_points()}
                out = []
                # `question`, NOT anything off this call. By the time LightMem
                # reaches embedding_retriever.search it has already embedded the
                # question and passes the VECTOR, by keyword:
                #     search(query_vector=..., limit=..., filters=..., return_full=True)
                # (vendor/lightmem/memory/lightmem.py:684). So there is no query
                # text here to recover: `args` is empty and there is no `query`
                # key. This line used to read
                #     query=args[0] if args else kwargs.get("query", "")
                # which therefore passed "" to the ranker on EVERY call, and the
                # grid silently scored a swap that had never seen a question —
                # bm25 tied every document and returned the store's own order,
                # dense/sparse ranked against the empty string. All eight
                # lightmem cells came out at nDCG@10 0.000 with the gold fact at
                # rank 173-270 while every other system scored normally.
                #
                # The closure is the fix: `question` is this method's own
                # argument, so it is exact by construction — the same reasoning
                # the docstring gives for capturing the search call rather than
                # re-querying. `k` above is left reading the call because
                # `limit` IS passed as a keyword and does arrive.
                for unit_id, score in ranker.rank(query=question, k=k):
                    point = by_id.get(str(unit_id))
                    if point is not None:
                        # The payload stays NESTED, because that is the row shape
                        # upstream returns and reads back. `search(return_full=True)`
                        # builds {"id", "score", "payload"}
                        # (vendor/.../embeddingretriever/qdrant.py:180) and
                        # `retrieve` formats each hit as
                        #     payload = r.get("payload", {})
                        #     f"{payload['time_stamp']} {payload['weekday']} {payload['memory']}"
                        # (vendor/lightmem/memory/lightmem.py:704). This used to
                        # hoist the payload's keys to the top level, so that
                        # `.get("payload", {})` found nothing and EVERY unit
                        # rendered as the empty "  " — all three fields defaulting
                        # to "". The ids were still right, so run.tsv and every
                        # retrieval metric were unaffected and nothing failed:
                        # only the saved `retrieved_units` / `step2_retrieved_context`
                        # were blank, which a `--prompt native` answer (live or
                        # replayed from retrieval.jsonl) would have generated from.
                        out.append({"id": unit_id, "score": score,
                                    "payload": dict(point.get("payload") or {})})
            else:
                out = original_search(*args, **kwargs)
            captured.append(list(out or []))
            return out

        retriever.search = _recording_search
        try:
            memories = self._lightmem.retrieve(question, limit=limit or self.top_k)
        finally:
            retriever.search = original_search

        results = captured[-1] if captured else []
        return memories, [str(r.get("id")) for r in results if r.get("id") is not None]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    #
    # LightMem's retrievable state lives entirely in its embedding store
    # (Qdrant): one point per extracted memory, carrying the embedding vector
    # plus a payload (memory text, timestamps, tags). retrieve() only embeds
    # the query and searches that store, so dumping every point and
    # re-inserting it into a fresh store fully reconstructs memory — without
    # re-running the (LLM-based) compression/extraction pass.

    def _unit_provenance(self, points: list) -> list:
        """Join this sample's captured provenance onto its saved points.

        One record per point, in points.jsonl order::

            {"unit_id", "source_turn_ids", "has_answer", "provenance"}

        ``provenance`` names which join produced it, so a reader can tell a
        measured mapping from a fallback instead of having them blended:

          "extract"  matched the text the extractor returned — the correct
                     source_id inverse, recomputed from the prompt's own labels.
          "update"   matched via the text the entry held before offline update
                     rewrote it (_install_update_provenance). Equally correct;
                     it just took the extra hop.
          "payload"  no match. Keeps upstream's stamp, which is the field that is
                     wrong for ~28% of units — so a unit labelled "payload" is
                     the one whose provenance is NOT known to be right.
        """
        records = []
        for point in points:
            payload = point.get("payload") or {}
            unit_id = str(point.get("id"))
            text = payload.get("memory")
            found = self._fact_provenance.get(text) if text else None
            source = "extract"
            if found is None:
                original = self._pre_update_text.get(unit_id)
                if original:
                    found = self._fact_provenance.get(original)
                    source = "update"
            if found is None:
                stored = payload.get("turn_id")
                records.append({
                    "unit_id": unit_id,
                    "source_turn_ids": [stored] if stored else [],
                    "has_answer": bool(payload.get("has_answer")),
                    "provenance": "payload",
                })
                continue
            records.append({
                "unit_id": unit_id,
                "source_turn_ids": sorted(found["turn_ids"]),
                "has_answer": bool(found["has_answer"]),
                "provenance": source,
            })
        return records

    def save_memory(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        points = self._lightmem.embedding_retriever.get_all(
            with_vectors=False, with_payload=True
        )
        points.sort(key=lambda p: p.get("payload", {}).get("float_time_stamp") or 0)
        with open(os.path.join(path, "points.jsonl"), "w") as f:
            for p in points:
                f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")

        # provenance.jsonl, written on EVERY ingest. points.jsonl keeps upstream's
        # own turn_id/has_answer untouched — this is additive, and the payload
        # stays readable as the mapping it is — but everything downstream that
        # asks "which turns is this unit evidence for" reads the sidecar, via
        # LightMemPlugin.iter_units.
        provenance = self._unit_provenance(points)
        with open(os.path.join(path, "provenance.jsonl"), "w") as f:
            for record in provenance:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        by_id = {r["unit_id"]: r for r in provenance}

        # meta.json, like every other system's. LightMem was the one store you
        # could not audit without re-deriving everything from points.jsonl:
        # no unit count, no session coverage, and no signal at all that a
        # summarisation ran to the max_tokens cap and came back clipped. A
        # trajectory answers those questions, but a trajectory is optional,
        # samplable and explicitly not authoritative (lib/trace.py) — this is
        # not.
        sessions = {(p.get("payload") or {}).get("session_id") for p in points}
        sessions.discard(None)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "units": len(points),
                "sessions_with_units": len(sessions),
                # Extraction calls the server stopped at the memory_llm's
                # max_tokens. Upstream json-parses that reply and falls back to
                # an empty result, so a clipped extraction is silently no
                # memories for that segment — the same failure mode a_mem and
                # memorybank already report.
                "truncated_calls": self._truncated_this_sample(),
                # Turns that never reached an add_memory call at all — the
                # leading non-user run, the odd tail, the mis-aligned pairs the
                # role check rejects. Upstream's behaviour, kept deliberately,
                # but invisible until now: this is the
                # one system whose ingest can drop input, and a store built from
                # 8 of 10 turns looked identical to one built from all 10.
                "dropped_turns": self.dropped_turns,
                # Extraction calls that raised. Upstream catches them inside
                # meta_text_extract and returns a result with no usage, so the
                # exception never escapes the seam record_calls wraps and the
                # trajectory cannot see it either — this is the only place a
                # failed lightmem extraction is recorded.
                "failed_calls": self._failed_this_sample(),
                # Turns that DID reach an add_memory call and still produced no
                # unit — the extractor read them and returned nothing. Distinct
                # from dropped_turns, which never reached the model at all, and
                # invisible before this: on 58bf7951 dropped_turns reported 6
                # while 11 user turns were actually absent from the store, so
                # the counter looked like the whole story and was not.
                # Zero is the expected value
                # only for a sample whose every turn held something.
                #
                # Measured over addressable_turn_ids, NOT ingested_turn_ids. The
                # latter holds both turns of every pair while messages_use sends
                # one of them, so an assistant turn under `user_only` could
                # never carry a unit and counting it made this number ~99%
                # structural: 125,485 over LongMemEval-S against 124,334
                # assistant turns in the dataset.
                #
                # Read off the SIDECAR, not payload["turn_id"]. Under the broken
                # stamp a unit charged one turn to the side left its real source
                # turn looking like it produced nothing, so this counter carried
                # the mapping error rather than measuring the extractor.
                "turns_without_memory": len(self.addressable_turn_ids - {
                    t for r in by_id.values() for t in r["source_turn_ids"]}),
                "units_with_has_answer": sum(
                    1 for r in by_id.values() if r["has_answer"]),
                # How each unit's provenance was resolved. See _unit_provenance.
                "provenance_source": {
                    source: sum(1 for r in by_id.values() if r["provenance"] == source)
                    for source in ("extract", "update", "payload")
                },
                # The one number from that breakdown that is DEGRADATION: units
                # whose provenance could not be measured and fell back to
                # upstream's stamp — the field that is wrong for ~28% of units.
                # A scalar, not just the dict above, because check_ingest reads
                # scalars: a count that only ever appears nested is a count
                # nothing is watching.
                "units_without_provenance": sum(
                    1 for r in by_id.values() if r["provenance"] == "payload"),
            }, f, indent=2)

    def load_memory(self, path: str) -> None:
        jsonl_path = os.path.join(path, "points.jsonl")
        with open(jsonl_path) as f:
            points = [json.loads(line) for line in f if line.strip()]
        if not points:
            return
        retriever = self._lightmem.embedding_retriever
        embedder = self._lightmem.text_embedder
        vectors = [embedder.embed(p["payload"]["memory"]) for p in points]
        retriever.insert(
            vectors=vectors,
            payloads=[p.get("payload", {}) for p in points],
            ids=[p["id"] for p in points],
        )

    def layer_usage(self) -> dict:
        """Split tokens by layer.

        Memory-management layer: LightMemory builds/updates memory with the
        (small) memory LLM; exact prompt/completion counts come from litellm's
        ``response.usage`` accumulated in ``self._lightmem.token_stats``. Retrieval is
        embedding-based and consumes no LLM tokens.

        Generation layer: the final answer is produced by our own generator.
        """
        ts = getattr(self._lightmem, "token_stats", {}) or {}
        mem = TokenUsage(
            input_tokens=(
                ts.get("add_memory_prompt_tokens", 0)
                + ts.get("update_prompt_tokens", 0)
                + ts.get("summarize_prompt_tokens", 0)
            ),
            output_tokens=(
                ts.get("add_memory_completion_tokens", 0)
                + ts.get("update_completion_tokens", 0)
                + ts.get("summarize_completion_tokens", 0)
            ),
            calls=(
                ts.get("add_memory_calls", 0)
                + ts.get("update_calls", 0)
                + ts.get("summarize_calls", 0)
            ),
        )
        return {"memory": mem, "generation": self.generator.token_usage}
