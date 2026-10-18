"""
evalgrid/benchmarks/runner.py
---------------------
Evaluation loop: reset → ingest → answer for every sample in an adapter.

Output format: JSON Lines, one record per query.
  {"question_id": ..., "hypothesis": ..., "source": ...}

Retrieval fields are added when the system exposes self.traces:
  naive_rag: + "retrieved_turn_ids", "retrieved_session_ids", "retrieval_scores"
  a_mem:     + "retrieved_context", "retrieval_keywords"  (internal memory text)

This format is directly accepted by LongMemEval's official evaluate_qa.py;
for other benchmarks convert as needed.

``run`` returns a :class:`RunReport`; its ``to_dict()`` is what gets written to
``stats_path`` and ``render()`` is the console summary.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm

from evalgrid.benchmarks.types import MemorySystem
from evalgrid.lib.cost import query_cost, split_answer_layers
from evalgrid.lib.errors import EvalgridError, LLMCallFailed
from evalgrid.lib.io import write_json_atomic
from evalgrid.lib.trace import TRAJECTORY_FILENAMES


# ---------------------------------------------------------------------------
# Memory persistence helpers
# ---------------------------------------------------------------------------

def _sample_mem_path(mem_dir: str, sample_id: str) -> str:
    """Per-sample memory directory (sample_id slugified for filesystem safety)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("_") or "sample"
    return os.path.join(mem_dir, slug)


#: Files that live in a sample's memory dir but are NOT memory — sidecars this
#: harness writes about the ingest rather than artifacts of it. They must not
#: count as evidence that the sample was ingested (see _memory_saved). Taken
#: from trace.py rather than spelled out, so adding a level cannot leave a new
#: filename out of this set.
_MEMORY_DIR_SIDECARS = frozenset(TRAJECTORY_FILENAMES)


def _memory_saved(path: str) -> bool:
    """True if ``path`` already holds saved memory (a prior ingest of this sample).

    Sidecars are excluded deliberately. A trajectory is opened BEFORE ingest
    starts (it records ingest), so a sample that crashed mid-ingest under
    --trace-level would otherwise leave a non-empty file behind and be skipped on
    resume as if it had completed — recording would have changed what the
    pipeline does, which is the one thing it must never do. usage.json needs no
    such guard: it is written only after save_memory returns.
    """
    if not os.path.isdir(path):
        return False
    # Require a non-empty file so a crash mid-save isn't mistaken for complete.
    return any(e.is_file() and e.name not in _MEMORY_DIR_SIDECARS and e.stat().st_size > 0
               for e in os.scandir(path))


def _record_failure(mem_dir: str, sample_id: str, phase: str, exc: BaseException) -> None:
    """Append one failed sample to ``mem_dir/failures.jsonl``.

    A failure used to exist in exactly one place: a traceback on stdout, in a
    job log. Nothing downstream could see it, so a run
    that lost samples and a run that had none were the same run as far as every
    aggregate was concerned — ``ingest_usage.json`` sums the samples that
    *worked*, and a failed sample writes no ``usage.json`` at all, so it does not
    even appear as a zero.

    This is the machine-readable other half: one line per failure, beside the
    memory it failed to build, appended so a resumed run adds to it rather than
    erasing what the last one found. ``deterministic`` marks a failure that
    resending cannot fix (see LLMCallFailed) — the flag that separates "this
    configuration cannot work" from "the server had a bad minute".

    Never raises. A harness that crashes while recording a crash is worse than
    one that loses the record.
    """
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sample_id": sample_id,
            "phase": phase,
            "error": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
        }
        if isinstance(exc, LLMCallFailed):
            record.update(exc.as_record())
        os.makedirs(mem_dir, exist_ok=True)
        with open(os.path.join(mem_dir, "failures.jsonl"), "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _ingest_complete(path: str) -> bool:
    """True if ``path`` holds a COMPLETE ingest, not just some output.

    Stricter than ``_memory_saved``, and only used to decide a resume skip.
    ``save_memory`` writes several files (a_mem: notes.jsonl, then meta.json;
    hypermem: hypergraph.json, provenance.json, meta.json), and the ingest loop
    catches a per-sample exception and moves on — so a save that raised halfway
    leaves a non-empty store directory that ``_memory_saved`` reads as finished
    and every later run skips, permanently, with a truncated store.

    ``usage.json`` is the one file written strictly AFTER ``save_memory``
    returns, which makes its presence the only honest completion marker. A
    sample missing it is re-ingested; re-ingesting a sample that was in fact
    complete costs one sample's work, whereas keeping a half-written one is
    silent data loss.
    """
    return os.path.isfile(os.path.join(path, "usage.json"))


def _sum_by_layer(deltas: list[dict]) -> dict:
    """Sum a list of per-layer {input_tokens, output_tokens, calls} dicts."""
    out: dict = {}
    for d in deltas:
        for name, v in d.items():
            o = out.setdefault(name, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            o["input_tokens"] += v["input_tokens"]
            o["output_tokens"] += v["output_tokens"]
            o["calls"] += v["calls"]
    return out


def _usage_block(by_layer: dict, *, seconds: float | None = None, queries: int | None = None) -> dict:
    """A uniform per-phase usage record: totals + optional timing + by-layer split."""
    tot_in = sum(l["input_tokens"] for l in by_layer.values())
    tot_out = sum(l["output_tokens"] for l in by_layer.values())
    tot_calls = sum(l["calls"] for l in by_layer.values())
    block = {
        "input_tokens": tot_in,
        "output_tokens": tot_out,
        "tokens": tot_in + tot_out,
        "llm_calls": tot_calls,
    }
    if seconds is not None:
        block["seconds"] = round(seconds, 4)
    if queries is not None:
        block["queries"] = queries
    block["by_layer"] = by_layer
    return block


def _write_ingest_usage(path: str, by_layer: dict, seconds: float) -> None:
    """Write per-sample ingest cost into the sample's memory dir (usage.json).

    Lives beside the saved memory so the cost survives resumable skips and
    travels with the memory regardless of which run dir later answers it.

    Written atomically because this file is also the completion marker
    (_ingest_complete): a torn write would be read by every later run as "this
    sample finished", which is the one thing it must never say by accident.
    """
    block = _usage_block(by_layer, seconds=seconds)
    record = {
        "ingest_seconds": block["seconds"],
        "ingest_input_tokens": block["input_tokens"],
        "ingest_output_tokens": block["output_tokens"],
        "ingest_tokens": block["tokens"],
        "ingest_llm_calls": block["llm_calls"],
        "by_layer": by_layer,
    }
    write_json_atomic(os.path.join(path, "usage.json"), record, indent=2)


def _write_aggregate_ingest_usage(mem_dir: str) -> None:
    """Sum every sample's ``usage.json`` into ``mem_dir/ingest_usage.json``.

    Ingest cost is a property of the memory (built once, reused by every
    retrieval/answer run), so its aggregate lives with the memory rather than in
    a retrieval-specific run dir. Summing the per-sample files (not the live
    RunReport) keeps the total correct across resumable ingests, where some
    samples were skipped this run but still have a saved usage.json.
    """
    deltas: list[dict] = []
    tot_seconds = 0.0
    n = 0
    for entry in os.scandir(mem_dir):
        fp = os.path.join(entry.path, "usage.json")
        if entry.is_dir() and os.path.isfile(fp):
            with open(fp) as f:
                u = json.load(f)
            deltas.append(u.get("by_layer", {}))
            tot_seconds += u.get("ingest_seconds", 0.0)
            n += 1
    block = _usage_block(_sum_by_layer(deltas), seconds=tot_seconds)
    block["samples"] = n
    # Atomic: every shard writes this file when it finishes, so the plain
    # truncate-then-write could be read (by the cost table scripts/ingest.sh prints) or
    # written half-finished. Since it is recomputed from the per-sample files
    # rather than from this process's state, the last writer to win is correct.
    write_json_atomic(os.path.join(mem_dir, "ingest_usage.json"), block, indent=2)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

def _layer_usage(system: MemorySystem) -> dict:
    """Return {layer_name: TokenUsage} for a system.

    Systems with the two-layer split expose ``layer_usage()`` returning a dict
    keyed by layer ("memory", "generation"). Older single-model systems fall
    back to their generator's cumulative ``token_usage`` under "generation".
    """
    if hasattr(system, "layer_usage"):
        return system.layer_usage() or {}
    gen = getattr(system, "generator", None)
    tu = getattr(gen, "token_usage", None) if gen is not None else None
    return {"generation": tu} if tu is not None else {}


def _snapshot(system: MemorySystem) -> dict:
    """Read each layer's current cumulative (input, output, calls) counters."""
    return {
        name: (u.input_tokens, u.output_tokens, u.calls)
        for name, u in _layer_usage(system).items()
    }


def _by_layer_delta(before: dict, after: dict) -> dict:
    """Per-layer {input_tokens, output_tokens, calls} between two snapshots."""
    return {
        name: {
            "input_tokens": a_in - before.get(name, (0, 0, 0))[0],
            "output_tokens": a_out - before.get(name, (0, 0, 0))[1],
            "calls": a_calls - before.get(name, (0, 0, 0))[2],
        }
        for name, (a_in, a_out, a_calls) in after.items()
    }


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------

@dataclass
class PhaseStats:
    """Token counters for one (layer, phase) cell."""
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    def add(self, d_in: int, d_out: int, d_calls: int) -> None:
        self.input_tokens += d_in
        self.output_tokens += d_out
        self.llm_calls += d_calls

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self, with_tokens: bool = True) -> dict:
        d = {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
        if with_tokens:
            d["tokens"] = self.tokens
        d["llm_calls"] = self.llm_calls
        return d


_PHASES = ("ingest", "answer")


@dataclass
class LayerStats:
    """One layer's ingest + answer counters."""
    ingest: PhaseStats = field(default_factory=PhaseStats)
    answer: PhaseStats = field(default_factory=PhaseStats)

    def total_of(self, phases=_PHASES) -> PhaseStats:
        p = PhaseStats()
        for ph in phases:
            s = getattr(self, ph)
            p.add(s.input_tokens, s.output_tokens, s.llm_calls)
        return p

    @property
    def total(self) -> PhaseStats:
        return self.total_of()


@dataclass
class Latency:
    """Wall-clock latency, accumulated across samples/queries."""
    ingest_seconds: float = 0.0
    answer_seconds: float = 0.0
    samples: int = 0
    queries: int = 0

    def as_dict(self, phases=_PHASES) -> dict:
        per_sample = self.ingest_seconds / self.samples if self.samples else 0.0
        per_query = self.answer_seconds / self.queries if self.queries else 0.0
        blocks = {
            "ingest": {
                "seconds": round(self.ingest_seconds, 4),
                "samples": self.samples,
                "seconds_per_sample": round(per_sample, 4),
            },
            "answer": {
                "seconds": round(self.answer_seconds, 4),
                "queries": self.queries,
                "seconds_per_query": round(per_query, 4),
            },
        }
        out = {ph: blocks[ph] for ph in phases}
        total = sum(getattr(self, f"{ph}_seconds") for ph in phases)
        out["total_seconds"] = round(total, 4)
        return out


@dataclass
class RunReport:
    """Token (layer x phase) cross-tab plus latency for one ``run``.

    ``to_dict`` is the on-disk ``token_usage.json`` shape; ``render`` is the
    console summary.
    """
    layers: dict[str, LayerStats] = field(default_factory=dict)
    latency: Latency = field(default_factory=Latency)

    def layer(self, name: str) -> LayerStats:
        return self.layers.setdefault(name, LayerStats())

    def accumulate(self, which: str, before: dict, after: dict) -> tuple[dict, tuple[int, int, int]]:
        """Fold per-layer deltas for one phase into the report.

        Returns the per-layer delta dict and the (in, out, calls) totals so the
        caller can also record them per-sample / per-query.
        """
        by_layer = _by_layer_delta(before, after)
        tot_in = tot_out = tot_calls = 0
        for name, d in by_layer.items():
            getattr(self.layer(name), which).add(d["input_tokens"], d["output_tokens"], d["calls"])
            tot_in += d["input_tokens"]
            tot_out += d["output_tokens"]
            tot_calls += d["calls"]
        return by_layer, (tot_in, tot_out, tot_calls)

    @property
    def combined(self) -> LayerStats:
        """Cross-layer totals (one row summing every layer)."""
        c = LayerStats()
        for ls in self.layers.values():
            c.ingest.add(ls.ingest.input_tokens, ls.ingest.output_tokens, ls.ingest.llm_calls)
            c.answer.add(ls.answer.input_tokens, ls.answer.output_tokens, ls.answer.llm_calls)
        return c

    def to_dict(self, phases=_PHASES) -> dict:
        """Layer x phase cross-tab. ``phases`` restricts which phase columns are
        emitted (e.g. ("answer",) for a run dir that only answers)."""
        report: dict = {}
        if self.layers:
            for name, ls in self.layers.items():
                block = {ph: getattr(ls, ph).as_dict() for ph in phases}
                block["total"] = ls.total_of(phases).as_dict()
                report[name] = block
            c = self.combined
            # The cross-layer total row omits the per-cell "tokens" key, matching
            # the historical token_usage.json shape.
            total = {ph: getattr(c, ph).as_dict(with_tokens=False) for ph in phases}
            total["total"] = c.total_of(phases).as_dict()
            report["total"] = total
        report["latency"] = self.latency.as_dict(phases)
        return report

    def render(self) -> str:
        out: list[str] = []
        if self.layers:
            out.append("\nToken usage (layer x phase)")
            for name in sorted(self.layers):
                ls = self.layers[name]
                out.append(
                    f"  {name:<10} ingest: in={ls.ingest.input_tokens:,} out={ls.ingest.output_tokens:,} calls={ls.ingest.llm_calls:,}"
                    f"  | answer: in={ls.answer.input_tokens:,} out={ls.answer.output_tokens:,} calls={ls.answer.llm_calls:,}"
                    f"  | total: {ls.total.tokens:,} tokens / {ls.total.llm_calls:,} calls"
                )
            c = self.combined.total
            out.append(f"  {'TOTAL':<10} input={c.input_tokens:,}  output={c.output_tokens:,}  tokens={c.tokens:,}  calls={c.llm_calls:,}")
        lat = self.latency
        per_sample = lat.ingest_seconds / lat.samples if lat.samples else 0.0
        per_query = lat.answer_seconds / lat.queries if lat.queries else 0.0
        out.append(
            "\nLatency\n"
            f"  ingest: {lat.ingest_seconds:,.1f}s over {lat.samples} samples ({per_sample:,.2f}s/sample)\n"
            f"  answer: {lat.answer_seconds:,.1f}s over {lat.queries} queries ({per_query:,.2f}s/query)\n"
            f"  TOTAL : {lat.ingest_seconds + lat.answer_seconds:,.1f}s"
        )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Per-sample / per-query work
# ---------------------------------------------------------------------------

def _ingest_sample(system: MemorySystem, sample, report: RunReport) -> tuple[dict, float]:
    """Ingest one sample's sessions; fold token + latency cost into ``report``.

    Returns the per-layer token delta and the wall-clock seconds for the sample.
    """
    sessions = sample.sessions
    # Snapshot after reset(): a_mem/lightmem rebuild their models in reset(), so
    # the accumulators are fresh per sample.
    before = _snapshot(system)
    t0 = time.perf_counter()
    for i, session in enumerate(tqdm(sessions, desc="  ingest", unit="session", leave=False)):
        system.ingest(session, is_last=(i == len(sessions) - 1))
    seconds = time.perf_counter() - t0
    by_layer, _ = report.accumulate("ingest", before, _snapshot(system))
    report.latency.ingest_seconds += seconds
    report.latency.samples += 1
    return by_layer, seconds


def _answer_record(query, sample, hyp: str, seconds: float, by_layer: dict, totals: tuple) -> dict:
    """Build the JSONL prediction record for one answered query."""
    record = {
        "question_id": query.query_id,
        "hypothesis": hyp,
        "source": sample.source,
        "answer_seconds": round(seconds, 4),
    }
    if by_layer:
        q_in, q_out, q_calls = totals
        record["answer_input_tokens"] = q_in
        record["answer_output_tokens"] = q_out
        record["answer_llm_calls"] = q_calls
        # Per-layer answer-phase deltas (memory-layer retrieval prep vs.
        # generation-layer answer).
        record["answer_tokens_by_layer"] = by_layer
        # The same flat cost fields pipeline/answer.py writes, so a native row
        # and a flat row can be put on one cost axis without either reader
        # knowing about layer names. This is where a system whose RETRIEVAL
        # spends LLM calls stops being free: A-Mem's query keywords, SimpleMem's
        # plan/reflect and HyperMem's funnel land in query_retrieval_tokens,
        # which is 0 by construction on the flat path.
        record.update(query_cost(*split_answer_layers(by_layer)))
    return record


def _add_trace_fields(record: dict, system: MemorySystem) -> None:
    """Attach retrieval-trace fields from the system's last trace, if any."""
    if not (hasattr(system, "traces") and system.traces):
        return
    trace = system.traces[-1]
    # How many memory units the system's own retrieval actually put in front of
    # the generator. The same field pipeline/answer.py records for a flat-corpus
    # run, so the two are directly comparable — and it is the only way to see
    # what --top-k could NOT equalise: A-Mem expands each hit's links and
    # SimpleMem's union is uncapped, so both can exceed k by a lot.
    ids = trace.get("retrieved_unit_ids")
    if ids is not None:
        flat = ([i for level in ids.values() for i in level]
                if isinstance(ids, dict) else list(ids))
        record["context_items"] = len(set(map(str, flat)))
    # A REPLAYED answer that was cut by --top-k / --context-budget carries the
    # cut's own accounting, written by pipeline/answer.install_replay. This is
    # its only reader, and it must stay one: the fields it carries
    # (context_tokens, context_underfilled, context_overflow,
    # context_depth_requested, context_truncated) have no other route onto a
    # native record, and answer.py's _context_summary counts them off the record.
    # The shared path records the identical fields inline, so a native cell and a
    # flat cell report the same shape.
    #
    # It also OVERRIDES the id-derived context_items above, and should: it is
    # measured from the units that actually reached the prompt, while the ids can
    # legitimately describe something else — HyperMem's are level-keyed and
    # include the topics its answer context never receives.
    stats = trace.get("_budget_stats")
    if isinstance(stats, dict):
        record.update(stats)
    if "retrieved_turn_ids" in trace:
        # naive_rag: turn-level IDs + parent session IDs + scores
        record["retrieved_turn_ids"] = trace["retrieved_turn_ids"]
        record["retrieved_session_ids"] = trace["retrieved_session_ids"]
        record["retrieval_scores"] = trace["retrieval_scores"]
    elif "step2_retrieved_context" in trace:
        # a_mem: internal memory text (no original chunk IDs)
        record["retrieved_context"] = trace["step2_retrieved_context"]
        record["retrieval_keywords"] = trace.get("step1_keywords")
    _join_replayed_retrieval_cost(record, trace)


def _join_replayed_retrieval_cost(record: dict, trace: dict) -> None:
    """On a replayed answer, restore the retrieval cost the retrieve stage paid.

    A replay makes no query-time memory call, so the measured
    ``query_retrieval_*`` is legitimately 0 — the cost was paid once, earlier, by
    ``pipeline/retrieve.py --retriever native``, which records it per query in
    ``retrieval.jsonl``. Leaving the 0 in place would make SimpleMem's 5-8
    planning calls disappear from exactly the runs a budget sweep produces,
    reporting the most expensive read path here as the cheapest.

    So it is joined back and MARKED: ``query_retrieval_source`` says the number
    was measured by another run, not this one. Two things follow from that mark,
    and both matter for reading a sweep:

      * the same figure appears in every budget's predictions.jsonl, because one
        retrieval served them all. Summing a sweep's ``query_llm_tokens``
        across budgets counts it once per budget — add it once.
      * ``answer_tokens_by_layer`` is left untouched. It reports what THIS
        process spent, and must keep doing so; only the derived cost fields,
        which exist to be read as "what this question cost", are completed.
    """
    cost = trace.get("retrieval_cost")
    if not isinstance(cost, dict):
        return
    retrieval = (int(cost.get("input_tokens") or 0),
                 int(cost.get("output_tokens") or 0),
                 int(cost.get("calls") or 0))
    if retrieval == (0, 0, 0):
        # Nothing was spent retrieving (lightmem, mem0, memorybank, hypermem).
        # The 0 already in the record is correct and needs no provenance.
        return
    answer = (record.get("query_answer_input_tokens", 0),
              record.get("query_answer_tokens", 0) - record.get("query_answer_input_tokens", 0),
              record.get("query_answer_calls", 0))
    record.update(query_cost(retrieval=retrieval, answer=answer))
    record["query_retrieval_source"] = "retrieval.jsonl"
    if trace.get("retrieval_seconds") is not None:
        record["retrieval_seconds"] = trace["retrieval_seconds"]


def _answer_query(system: MemorySystem, query, sample, report: RunReport) -> dict:
    """Answer one query; fold cost into ``report`` and return its record."""
    before = _snapshot(system)
    t0 = time.perf_counter()
    hyp = system.answer(query)
    seconds = time.perf_counter() - t0
    by_layer, totals = report.accumulate("answer", before, _snapshot(system))
    report.latency.answer_seconds += seconds
    report.latency.queries += 1
    record = _answer_record(query, sample, hyp, seconds, by_layer, totals)
    _add_trace_fields(record, system)
    return record


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _answered_question_ids(out_path: str | None) -> set[str]:
    """question_ids already present in a previous answer run's JSONL."""
    if not out_path or not os.path.isfile(out_path):
        return set()
    done: set[str] = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                qid = json.loads(line).get("question_id")
                if qid is not None:
                    done.add(qid)
    return done


def run(
    system: MemorySystem,
    adapter,
    out_path: str | None = None,
    trace_path: str | None = None,
    stats_path: str | None = None,
    phase: str = "ingest",
    mem_dir: str | None = None,
    overwrite: bool = False,
    sample_usage_path: str | None = None,
    on_sample_traces=None,
    trajectory_dir: str | None = None,
) -> RunReport:
    """Evaluate ``system`` over ``adapter``; return a :class:`RunReport`.

    phase:
      "ingest" — reset → ingest → save_memory per sample. Resumable: samples
                 with existing memory are skipped unless overwrite=True.
      "answer" — reset → load_memory → answer per sample. Requires a prior
                 "ingest" run in the same mem_dir. Also resumable: a sample
                 whose every question already appears in ``out_path`` is skipped
                 and new answers are appended, unless overwrite=True (which
                 truncates and re-answers everything). Skipping is per SAMPLE,
                 not per question, because loading one sample's memory is the
                 expensive part and a half-answered sample is cheap to finish.

    Both phases require mem_dir and a system that supports persistence.

    sample_usage_path: if set (answer phase only), write one JSON line per
      sample with both answer cost and the ingest cost loaded from the
      per-sample usage.json: {"sample_id", "ingest": {...}, "answer": {...}}.

    on_sample_traces: if set (answer phase only), called as
      ``(sample_id, traces)`` after each sample with the trace dicts that
      sample's queries produced. This is how evalgrid.pipeline.answer --retriever native --prompt native
      turns a system's own retrieval into a run.tsv without the runner needing
      to know anything about SystemPlugin or the retrieval output tree.

    trajectory_dir: answer phase only — the DIRECTORY the step-level trajectory
      goes in (evalgrid/lib/trace.py names the file from its level, so a caller
      cannot name one that misdescribes its contents). The INGEST trajectory
      takes no argument: it is written into each sample's memory dir beside
      usage.json, because it describes the store rather than the run. Both are no-ops unless the
      system carries a live recorder, which only ``--trace-level`` installs.
    """
    if phase not in ("ingest", "answer"):
        raise ValueError(f"phase must be 'ingest' or 'answer', got {phase!r}")
    if not mem_dir:
        raise ValueError(f"phase={phase!r} requires mem_dir")
    if not getattr(system, "supports_persistence", False):
        raise ValueError(f"{type(system).__name__} does not support memory persistence")

    total = getattr(adapter, "limit", None) or getattr(adapter, "__len__", lambda: None)()
    report = RunReport()

    if phase == "ingest":
        skipped = 0
        failed = 0
        # The aggregate is rewritten however the loop ends, including
        # KeyboardInterrupt (a BaseException, so the per-sample `except
        # Exception` below does not catch it and the loop unwinds). Writing it
        # only on the happy path left ingest_usage.json contradicting the
        # per-sample usage.json files it is supposed to sum: a run killed for
        # time wrote none at all, and a rerun after a run that had errored on
        # every sample left the earlier all-zero file in place. Since
        # _write_aggregate_ingest_usage re-reads every sample's usage.json from
        # disk rather than the live report, the interrupted write is as correct
        # as the completed one — it just covers fewer samples.
        try:
            for sample in tqdm(adapter.samples(), total=total, desc="samples", unit="sample"):
                sample_mem = _sample_mem_path(mem_dir, sample.sample_id)
                if not overwrite and _ingest_complete(sample_mem):
                    skipped += 1
                    continue
                if _memory_saved(sample_mem):
                    # Output but no completion marker: a previous run crashed
                    # between save_memory's first file and its last. Rebuilding
                    # is the only safe move — see _ingest_complete.
                    tqdm.write(f"\n[resume] sample {sample.sample_id!r} has memory but no "
                               f"usage.json (incomplete save) — re-ingesting it")
                system.reset()
                # The ingest trajectory lives WITH the memory, not in a run dir —
                # same reasoning as usage.json above: it survives a resumable skip
                # and travels with the store. A skipped sample writes nothing, so a
                # reader must read "no file" as "not recorded", never "no steps".
                recorder = getattr(system, "recorder", None)
                if recorder is not None:
                    os.makedirs(sample_mem, exist_ok=True)
                    recorder.begin_sample(sample.sample_id, "ingest", sample_mem)
                try:
                    by_layer, seconds = _ingest_sample(system, sample, report)
                    system.save_memory(sample_mem)
                    _write_ingest_usage(sample_mem, by_layer, seconds)
                except (Exception, EvalgridError) as e:
                    # EvalgridError is named because it is a BaseException — the
                    # thing that lets it escape vendored retry loops (see
                    # lib/errors.py). Without naming it here, a no-retry failure
                    # would also escape THIS handler and take down a run that
                    # should have lost one sample.
                    failed += 1
                    _record_failure(mem_dir, sample.sample_id, "ingest", e)
                    tqdm.write(f"\n[ERROR] sample {sample.sample_id!r}: {e}")
                    tqdm.write(traceback.format_exc())
                finally:
                    if recorder is not None:
                        recorder.end_sample()
            if skipped:
                tqdm.write(f"\nSkipped {skipped} sample(s) with already-saved memory "
                           f"(use --overwrite to rebuild).")
            if failed:
                # Loud and last, because everything printed after this — the
                # usage table, the "Done" line — is computed over the samples
                # that WORKED and will look perfectly healthy.
                tqdm.write(
                    f"\n!!! {failed} sample(s) FAILED and built no memory. "
                    f"They are listed in {os.path.join(mem_dir, 'failures.jsonl')}.\n"
                    f"!!! Every total below covers only the samples that succeeded. "
                    f"Re-running resumes them (a failed sample writes no usage.json)."
                )
        finally:
            _write_aggregate_ingest_usage(mem_dir)

    else:  # phase == "answer"
        # Resume: skip samples every question of which is already answered, and
        # append rather than truncate. --overwrite starts the file fresh.
        already = set() if overwrite else _answered_question_ids(out_path)
        file_mode = "w" if (overwrite or not already) else "a"
        skipped = 0
        recorder = getattr(system, "recorder", None)
        first_traj = True
        trace_file = open(trace_path, file_mode) if trace_path else None
        usage_file = open(sample_usage_path, file_mode) if sample_usage_path else None
        try:
            with open(out_path, file_mode) as f:
                for sample in tqdm(adapter.samples(), total=total, desc="samples", unit="sample"):
                    pending = [q for q in sample.queries if q.query_id not in already]
                    if not pending:
                        skipped += 1
                        continue
                    sample_mem = _sample_mem_path(mem_dir, sample.sample_id)
                    if not _memory_saved(sample_mem):
                        raise RuntimeError(
                            f"No saved memory for sample {sample.sample_id!r} at {sample_mem!r}. "
                            f"Run evalgrid/pipeline/ingest.py first."
                        )
                    system.reset()
                    system.load_memory(sample_mem)
                    # Build any lazily-built query-time index now, so the first
                    # question's answer_seconds is not also an index build
                    # (MemorySystem.prepare_retrieval). Skipped under a replay,
                    # which never retrieves.
                    prepare = getattr(system, "prepare_retrieval", None)
                    if prepare is not None and not getattr(system, "_retrieval_replay", False):
                        prepare()
                    # Answer-phase trajectory is a property of the RUN (which
                    # prompt, which retriever settings), not of the store, so it
                    # sits in the answers dir. Reopened per sample in append
                    # mode so seq numbering restarts per sample, as it does on
                    # the ingest side.
                    if recorder is not None and trajectory_dir:
                        recorder.begin_sample(sample.sample_id, "answer", trajectory_dir,
                                              mode=file_mode if first_traj else "a")
                        first_traj = False

                    ans_deltas: list[dict] = []
                    ans_seconds = 0.0
                    sample_traces: list[dict] = []
                    for query in tqdm(pending, desc="  answer", unit="query", leave=False):
                        record = _answer_query(system, query, sample, report)
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        if hasattr(system, "traces") and system.traces:
                            sample_traces.append(system.traces[-1])
                            if trace_file:
                                trace_file.write(json.dumps(system.traces[-1], ensure_ascii=False) + "\n")
                                trace_file.flush()
                        if "answer_tokens_by_layer" in record:
                            ans_deltas.append(record["answer_tokens_by_layer"])
                        ans_seconds += record["answer_seconds"]

                    if on_sample_traces is not None:
                        on_sample_traces(sample.sample_id, sample_traces)

                    if usage_file is not None:
                        answer_part = _usage_block(
                            _sum_by_layer(ans_deltas), seconds=ans_seconds, queries=len(pending)
                        )
                        entry: dict = {"sample_id": sample.sample_id, "answer": answer_part}
                        usage_p = os.path.join(sample_mem, "usage.json")
                        if os.path.isfile(usage_p):
                            with open(usage_p) as _uf:
                                _u = json.load(_uf)
                            entry["ingest"] = {
                                "seconds": _u.get("ingest_seconds", 0.0),
                                "input_tokens": _u.get("ingest_input_tokens", 0),
                                "output_tokens": _u.get("ingest_output_tokens", 0),
                                "tokens": _u.get("ingest_tokens", 0),
                                "llm_calls": _u.get("ingest_llm_calls", 0),
                                "by_layer": _u.get("by_layer", {}),
                            }
                        usage_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        usage_file.flush()
        finally:
            if recorder is not None:
                recorder.end_sample()
            if trace_file:
                trace_file.close()
            if usage_file:
                usage_file.close()

        if skipped:
            tqdm.write(f"\nSkipped {skipped} fully-answered sample(s) "
                       f"(use --overwrite to re-answer).")

        if stats_path:
            with open(stats_path, "w") as sf:
                json.dump(report.to_dict(phases=("answer",)), sf, indent=2)

    tqdm.write(report.render())
    return report
