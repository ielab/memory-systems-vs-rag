#!/usr/bin/env python3
"""
check_ingest.py
===============
Report whether an ingest actually built every sample's memory — without
re-running anything.

Exit 0 from ``evalgrid.pipeline.ingest`` is not evidence the store is sound, and
that is by design rather than by accident: a per-sample failure is caught,
recorded and stepped over (see ``runner.run``), so a run that lost 40 samples
prints the same cheerful summary as one that lost none — its totals just cover
fewer samples. Three separate things have to be read back off disk to know, and
each is silent in a different way:

  * **completeness** — ``<sample>/usage.json`` is the only honest completion
    marker, because it is written strictly AFTER ``save_memory`` returns. A dir
    holding notes.jsonl but no usage.json is a save that crashed halfway; the
    next ingest re-does it (``runner._ingest_complete``).
  * **failures** — ``failures.jsonl``, one line per failed sample. It is
    APPENDED to across resumed runs, so a raw line count overstates the damage:
    a sample that failed on Monday and succeeded on Tuesday still has its
    Monday line. This tool resolves each failure against the store as it stands
    now, which is the whole reason it is not a ``wc -l``.
  * **degradation** — ``<sample>/meta.json`` counters (``truncated_calls``,
    ``failed_calls``, and the per-system ones, which do NOT share one name:
    memorybank's truncation is ``truncated_summaries`` and it writes no
    ``truncated_calls`` at all). These are the quiet ones: a store built
    entirely from clipped or unparseable replies raises nothing, writes its
    usage.json, and looks exactly like a good one. our development notes say what each
    counter means and which are expected. A numeric counter this tool has no
    entry for is reported as UNCLASSIFIED rather than skipped — see
    ``DEGRADATION_KEYS``. The mirror of that is ``NOTE_KEYS``: a counter that
    records the SHAPE of one system's ingest rather than damage to its store —
    lightmem's ``dropped_turns``, which counts turns a system that extracts from
    a single role was never going to read — is printed with its caveat and kept
    out of the verdict, so it cannot fail every store in a grid nor bury the
    counters that do mean damage.
  * **cause** — ``<sample>/trajectory-<level>.jsonl``. The counters above say
    how many calls failed; only the trajectory says WHY, because
    ``record_calls`` writes the raised exception into the record it was already
    writing (``lib/trace.py``: ``response=f"<error> {error}"``). That matters
    twice over. It separates "the endpoint was down" (re-running fixes it) from
    "the prompt cannot fit the window" (re-running burns the same GPU-hours to
    rebuild the same store) — the split ``failures.jsonl`` spells
    ``deterministic``. And for two systems it is the ONLY record: lightmem's
    offline update throws its exceptions into a ``ThreadPoolExecutor`` whose
    map iterator is never consumed, and mem0 turns a failed resolver call into
    ``response = ""`` — both lose content while every counter stays at zero.

Everything the checker calls "complete" comes from ``benchmarks.runner`` itself
rather than being re-spelled here, so if the completion marker ever moves, this
tool moves with it instead of quietly passing a store the pipeline would rebuild.

Usage
-----
    # from an experiment spec (the usual way — resolves the keyed mem dir)
    python -m evalgrid.tools.check_ingest --exp experiments/a_mem/longmemeval.yaml

    # a throwaway or explicitly-placed store
    python -m evalgrid.tools.check_ingest --mem-dir outputs/_smoke_ingest/a_mem \\
        --dataset longmemeval_s --limit 3

    python -m evalgrid.tools.check_ingest --exp experiments/mem0/longmemeval.yaml -v

    # keep the report; a directory gets <system>_check.txt, so the whole grid
    # can be checked into one place and the files diffed run against run
    for s in a_mem hypermem lightmem mem0 memorybank simplemem; do
        python -m evalgrid.tools.check_ingest --exp experiments/$s/longmemeval.yaml \\
            -v --out outputs/_checks/
    done

Exit code is 0 only on PASS — every expected sample complete, no unresolved
failure, no degradation counter set, no store missing its meta.json — so it can
gate the next stage in a driver script.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import sys
from pathlib import Path

# evalgrid/tools/ -> repo root; make the package importable when run as a file.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.benchmarks.runner import (  # noqa: E402
    _ingest_complete,
    _memory_saved,
    _sample_mem_path,
)
from evalgrid.lib.cli import (  # noqa: E402
    add_dataset_arg,
    add_exp_arg,
    add_limit_arg,
    add_mem_model_arg,
    add_mem_variant_arg,
    add_system_arg,
    apply_spec,
)
from evalgrid.lib.paths import ExperimentPaths  # noqa: E402
from evalgrid.lib.trace import TRAJECTORY_FILENAMES  # noqa: E402

#: Per-store degradation counters, from our development notes table. Any of these
#: being nonzero means the store is incomplete even though the run exited 0:
#: truncated_calls / failed_calls are shared, the rest are per system (a_mem,
#: simplemem, mem0, hypermem, lightmem, memorybank in that order).
#:
#: `truncated_summaries` is memorybank's. It matters more than its position in
#: the list suggests: memorybank writes no `truncated_calls` at all, so a tuple
#: covering only the shared names leaves exactly one system whose truncation is
#: structurally invisible here. It did — `outputs/_smoke4` clipped a summary in
#: 2 of 4 samples and this tool called the store clean and exited PASS. Every
#: system's truncation counter must appear, under whatever name that system gave
#: it. Since a list kept by eye will drift again, it is not left to the eye: a
#: numeric meta.json key that none of the four tuples below classify is reported
#: as UNCLASSIFIED rather than ignored (see ``unclassified_counters``), which
#: turns the next system's new counter into a line of output instead of a silent
#: pass.
DEGRADATION_KEYS = (
    "truncated_calls", "failed_calls", "fatal_calls", "degraded_notes",
    "empty_windows", "json_failures", "empty_extractions", "failed_topics",
    "failed_facts", "dropped_turns", "turns_without_memory",
    "truncated_summaries", "forgotten",
    # lightmem: units whose source turns could not be measured at ingest and
    # kept upstream's stamp instead. That stamp is wrong for ~28% of units
    #, so this is the count of units whose qrels labels are
    # not known to be right — degradation of the gold set, not of the memory.
    "units_without_provenance",
)

#: Counters a store may carry that are NOT degradation. mem0's `no_op_adds` is
#: the one that matters: it is `empty_extractions` PLUS the add()s where every
#: extracted fact resolved to NONE, and the second half is the resolver working
#: rather than content being lost. Listing it keeps the distinction visible to
#: anyone extending DEGRADATION_KEYS by pattern-matching on meta.json keys —
#: which is exactly how the two came to share one counter.
NON_DEGRADATION_KEYS = ("no_op_adds",)

#: Degradation counters that are not damage FOR THE SYSTEM THAT WRITES THEM:
#: still read, still printed, but on their own line and out of the verdict.
#: Keyed by counter, valued (systems that get the exemption, why, the caveat
#: printed under it).
#:
#: ``dropped_turns`` is lightmem's and nobody else's. It counts turns no
#: ``add_memory`` call ever saw — the leading non-user run, the odd tail, and
#: the mis-aligned pairs the role check rejects (``systems/lightmem/
#: longmemeval.py``). But lightmem runs ``messages_use: user_only``
#: (``configs/lightmem/*.yaml``), and that drops the other role's text inside
#: upstream's own concatenation whether or not the pairing step reached the
#: turn: an assistant turn costs the store nothing by being dropped, because a
#: kept one would have carried no unit either. Gating on it failed all 500
#: lightmem stores of the LongMemEval-S grid for a shape that is upstream's
#: design (our development notes") and put `truncated_calls` — 16 stores, real
#: lost content — behind a counter set on 485.
#:
#: Demoted, not deleted, and the caveat is why: the odd tail and a mis-aligned
#: pair take USER turns with them too, meta.json carries one number for both
#: cases, and 3 of the 2,020 turns LongMemEval-S drops carry ``has_answer``.
#: The number stays on the page as something to read, not as something to act
#: on.
NOTE_KEYS = {
    "dropped_turns": (
        ("lightmem",),
        "lightmem extracts from one role only (messages_use: user_only), so a "
        "dropped assistant turn was never going to carry a unit",
        "the odd tail and mis-aligned pairs drop USER turns too and meta.json "
        "does not split the two — read qrels coverage before calling a sample sound",
    ),
}

#: Keys a store's meta.json may use for "how much memory is in here". Systems
#: name their unit differently and none of them is required.
SIZE_KEYS = ("units", "notes", "entries", "facts")

#: Numeric counters that describe the SHAPE of a run rather than damage to it —
#: how many sessions were bucketed, how many links a note drew, what window the
#: config asked for. They are listed only so ``unclassified_counters`` can tell
#: "known and harmless" from "nobody has looked at this key yet"; nothing reads
#: their values. Per system: memorybank, simplemem, lightmem, mem0, a_mem,
#: hypermem.
DESCRIPTIVE_KEYS = (
    "buckets", "turn_pairs", "summaries",
    "window_size", "overlap_size",
    "sessions_with_units", "units_with_has_answer",
    "add_calls",
    "links", "evo_threshold",
    "fact_hyperedges", "episodes", "episode_hyperedges", "topics", "sessions",
)

#: How a raised LLM call is classified from the exception text the tracer
#: recorded, and whether re-sending it byte-for-byte could ever succeed.
#:
#: The vocabulary is ``LLMCallFailed``'s, deliberately: ``deterministic`` here
#: means exactly what it means in ``failures.jsonl`` — "resending is guaranteed
#: to fail the same way" — so the two halves of a run's failure record can be
#: read with one rule instead of two.
#:
#: Why the text and not ``meta.error``. The tracer records the type of whatever
#: escaped the wrapped method, and for four of the six systems that is the
#: adapter's own wrapper, not the cause: simplemem raises
#: ``RuntimeError("... call failed after 1 attempt(s) ...: {last_exc}")``, so
#: every connection drop, every 400 and every timeout arrives here typed
#: ``RuntimeError``. The cause survives only inside the message, which is why
#: this matches on text. ``meta.error`` is still printed with the sample line —
#: it is free, and it disambiguates the unclassified ones.
#:
#: ORDER MATTERS. A context overflow IS an HTTP 400, so it has to be tested
#: before the generic 400; the same for 401/403.
CALL_ERROR_CLASSES = (
    # (bucket, deterministic, needles — lowercased, substring match)
    ("context_length", True,
     ("maximum context length", "context_length_exceeded", "reduce the length",
      "longer than the maximum", "maximum number of tokens")),
    ("auth", True,
     ("error code: 401", "error code: 403", "authenticationerror",
      "permissiondeniederror", "invalid api key", "incorrect api key")),
    ("bad_request", True,
     ("error code: 400", "error code: 422", "badrequesterror",
      "unprocessableentity")),
    ("not_found", True,
     ("error code: 404", "notfounderror", "model not found",
      "does not exist")),
    ("rate_limit", False,
     ("error code: 429", "ratelimiterror", "rate limit")),
    ("server_error", False,
     ("error code: 500", "error code: 502", "error code: 503",
      "internalservererror", "bad gateway", "service unavailable")),
    ("timeout", False,
     ("timeout", "timed out")),
    ("connection", False,
     ("connection error", "connection refused", "connecterror",
      "apiconnectionerror", "errno 111", "connection reset",
      "server disconnected", "remoteprotocolerror", "all connection attempts")),
    ("cuda_oom", False,
     ("cuda out of memory", "outofmemoryerror")),
)

#: Fallback classification by EXCEPTION TYPE, for the failures whose message is
#: empty. ``CancelledError`` is the one that forced this: ``asyncio.wait_for``
#: cancels with no message at all, so the text-matching above has nothing to
#: read and every one of them landed in ``unclassified``. Consulted only when
#: the text matched nothing, so a typed guess can never override a message that
#: said something specific.
CALL_ERROR_TYPES = {
    # hypermem wraps every extractor call in asyncio.wait_for against upstream's
    # five unbounded `while True:` retry loops (our development notes). A hit
    # here is that guard firing, NOT the endpoint failing: the call was killed
    # from outside because the model would not satisfy a validator. It is also
    # why no counter moves — CancelledError is a BaseException, which is the
    # whole point of the guard and the reason the system's own `except
    # Exception` bookkeeping never sees it.
    "CancelledError": "cancelled",
    "TimeoutError": "timeout",
    "APITimeoutError": "timeout",
    "APIConnectionError": "connection",
    "ConnectError": "connection",
    "ConnectTimeout": "connection",
    "RateLimitError": "rate_limit",
    "BadRequestError": "bad_request",
    "AuthenticationError": "auth",
    "InternalServerError": "server_error",
    "OutOfMemoryError": "cuda_oom",
}

#: A raised call whose text matches nothing above. Reported, never folded into a
#: neighbouring bucket — the same rule ``unclassified_counters`` follows, and
#: for the same reason: a guess here would be indistinguishable from knowledge.
UNCLASSIFIED_ERROR = "unclassified"

#: Buckets where re-running cannot help. Kept as a set so the report and the
#: verdict cannot disagree about which those are.
DETERMINISTIC_BUCKETS = frozenset(
    bucket for bucket, deterministic, _ in CALL_ERROR_CLASSES if deterministic)

#: One line of advice per bucket, printed with the count. Sorted into the two
#: groups a reader actually has to act on differently.
BUCKET_ADVICE = {
    "context_length": "DETERMINISTIC — the prompt does not fit; re-running rebuilds the same store",
    "auth":           "DETERMINISTIC — credentials/endpoint, fix the config first",
    "bad_request":    "DETERMINISTIC — the request itself was rejected; read one before re-running",
    "not_found":      "DETERMINISTIC — wrong model or endpoint for this store",
    "rate_limit":     "transient — re-running should help (throttle first)",
    "server_error":   "transient — the server errored; re-running should help",
    "timeout":        "transient — check why the call was slow before raising any timeout",
    "connection":     "transient — the endpoint was unreachable; re-running should help",
    "cuda_oom":       "environmental — fix the GPU budget, then re-run",
    "cancelled":      "the harness's own timeout guard fired, not an endpoint failure",
    UNCLASSIFIED_ERROR: "unknown — sort it into CALL_ERROR_CLASSES",
}


COMPLETE, PARTIAL, EMPTY = "complete", "incomplete", "empty"


# ---------------------------------------------------------------------------
# Reading the store
# ---------------------------------------------------------------------------

def scan_stores(mem_dir: str) -> dict[str, str]:
    """Map each sample subdir of ``mem_dir`` to its state.

    complete   — has usage.json, written only after save_memory returned
    incomplete — has memory but no usage.json (a save that crashed part-way;
                 the next ingest rebuilds it)
    empty      — a dir with nothing in it but sidecars, or nothing at all. Not
                 the same as "never started": the dir exists because a run
                 reached this sample and produced nothing that survived.
    """
    states: dict[str, str] = {}
    for entry in sorted(os.scandir(mem_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if _ingest_complete(entry.path):
            states[entry.name] = COMPLETE
        elif _memory_saved(entry.path):
            states[entry.name] = PARTIAL
        else:
            states[entry.name] = EMPTY
    return states


def expected_slugs(dataset: str) -> dict[str, str]:
    """{sample-dir slug: sample_id} for every sample in the dataset, in order.

    Slugged the same way the runner names the dirs, so the two sets are directly
    comparable. The caller trims it to ingest's ``--limit`` (first N samples) —
    the full length is kept because it is what a smoke run's cost extrapolates
    to.
    """
    from evalgrid.benchmarks.adapters import build_adapter

    return {os.path.basename(_sample_mem_path("", s.sample_id)): s.sample_id
            for s in build_adapter(dataset).samples()}


def read_failures(mem_dir: str) -> list[dict]:
    """Parse failures.jsonl (append-only across runs; may not exist)."""
    path = os.path.join(mem_dir, "failures.jsonl")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def note_keys_for(system: str | None) -> frozenset[str]:
    """Which DEGRADATION_KEYS this store's system is exempt from.

    A store whose system is not known gets no exemption. The exemption is a
    statement about how one named system ingests, so handing it to an
    unidentified store would be a guess — the same move that let
    ``truncated_summaries`` pass as clean, run the other way round.
    """
    if not system:
        return frozenset()
    return frozenset(k for k, (systems, *_why) in NOTE_KEYS.items()
                     if system in systems)


def store_degradation(mem_dir: str, slug: str,
                      note_keys: frozenset[str] = frozenset(),
                      ) -> tuple[dict | None, dict, dict, int]:
    """(meta, flagged counters, noted counters, unit count) for one store.

    ``meta`` is None when the store has no meta.json — reported as UNKNOWN
    rather than clean, since a store that predates the counters cannot say
    whether its replies were clipped, and printing nothing would read as a pass.

    ``note_keys`` is what splits the second return from the third; both are read
    off the same meta.json and only their reporting differs. See NOTE_KEYS.
    """
    meta = read_json(os.path.join(mem_dir, slug, "meta.json"))
    if meta is None:
        return None, {}, {}, 0
    hit = {k: meta[k] for k in DEGRADATION_KEYS if meta.get(k)}
    flags = {k: v for k, v in hit.items() if k not in note_keys}
    notes = {k: v for k, v in hit.items() if k in note_keys}
    size = next((meta[k] for k in SIZE_KEYS if meta.get(k)), 0)
    return meta, flags, notes, size


#: Every key the four tuples above account for.
_CLASSIFIED = frozenset(
    DEGRADATION_KEYS + NON_DEGRADATION_KEYS + SIZE_KEYS + DESCRIPTIVE_KEYS)


def unclassified_counters(meta: dict) -> dict:
    """Numeric meta.json keys no tuple above accounts for.

    The guard against this tool's own last bug. A system that adds a counter
    under a name DEGRADATION_KEYS does not list is not caught by anything —
    memorybank's `truncated_summaries` sat unread long enough to pass a store
    with clipped summaries as clean. Rather than trust the next reader to
    notice, an unknown numeric key is surfaced as UNCLASSIFIED so it gets sorted
    into one of the tuples deliberately.

    Bools are excluded (``json_mode_fallback``, ``infer``, ``original_data_dropped``
    are settings, not counts) and so is anything non-scalar; those are config
    echoes, not counters.
    """
    return {k: v for k, v in meta.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in _CLASSIFIED}


# ---------------------------------------------------------------------------
# Reading the trajectory
# ---------------------------------------------------------------------------

def classify_call_error(text: str, error_type: str = "") -> str:
    """Which bucket a failed call falls into.

    Message first, type second. The message is the specific evidence — a 400 is
    only a context overflow if it says so — and the type is what is left when
    the exception carried no message at all, which is the normal case for a
    cancellation.
    """
    low = (text or "").lower()
    for bucket, _deterministic, needles in CALL_ERROR_CLASSES:
        if any(n in low for n in needles):
            return bucket
    return CALL_ERROR_TYPES.get((error_type or "").strip(), UNCLASSIFIED_ERROR)


class CallErrors:
    """Every raised LLM call one sample's trajectory recorded.

    ``levels`` is which trajectory files were found, not which were asked for:
    a sample carries one per level it was ever recorded at, and a store built
    across two runs at two levels has both. Empty means the sample has no
    trajectory at all — recorded as UNKNOWN by the caller rather than as zero,
    since ``trace_level: off`` and "nothing failed" are the same file on disk
    and must not be the same line in a report.
    """

    __slots__ = ("levels", "buckets", "stages", "examples")

    def __init__(self):
        self.levels: list[str] = []
        self.buckets: collections.Counter = collections.Counter()
        self.stages: collections.Counter = collections.Counter()
        self.examples: dict[str, str] = {}

    @property
    def total(self) -> int:
        return sum(self.buckets.values())

    @property
    def deterministic(self) -> int:
        return sum(n for b, n in self.buckets.items() if b in DETERMINISTIC_BUCKETS)


def read_call_errors(mem_dir: str, slug: str) -> CallErrors:
    """Scan one sample's trajectories for calls the tracer logged as raised.

    Only ``phase == "ingest"`` records count. The ingest trajectory is written
    into the sample's memory dir and a native-retrieval one is not, so today
    every record here is already ingest — the filter is against the day that
    changes, because an answer-phase failure is not damage to this store.

    The ``"<error>"`` substring test on the raw line is a prefilter, not the
    match: a completion that merely contains the word is rejected by the
    ``startswith`` below, having cost one string scan. It is what keeps this
    from being a full JSON parse of every call in the run — a_mem's smoke store
    alone is 4,360 records.

    Never raises. A trajectory that is truncated, half-written by a killed job,
    or not JSON at all costs its own line and nothing else; the alternative is a
    checker that dies on the stores it exists to describe.
    """
    out = CallErrors()
    for name in sorted(TRAJECTORY_FILENAMES):
        path = os.path.join(mem_dir, slug, name)
        if not os.path.isfile(path):
            continue
        out.levels.append(name)
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    if "<error>" not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if rec.get("phase") not in (None, "ingest"):
                        continue
                    text = ((rec.get("out") or {}).get("text") or "")
                    if not isinstance(text, str) or not text.startswith("<error>"):
                        continue
                    kind = (rec.get("meta") or {}).get("error") or "?"
                    bucket = classify_call_error(text, kind)
                    out.buckets[bucket] += 1
                    out.stages[rec.get("stage") or "?"] += 1
                    if bucket not in out.examples:
                        body = text[len("<error>"):].strip().replace("\n", " ")
                        out.examples[bucket] = f"{kind}: {body[:150]}"
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_some(rows: list[str], limit: int, verbose: bool) -> None:
    """Print rows, capped unless -v (a 500-sample store can fail wholesale)."""
    shown = rows if verbose else rows[:limit]
    for r in shown:
        print(r)
    if len(rows) > len(shown):
        print(f"    ... and {len(rows) - len(shown)} more (-v to list)")


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """The parser and its result. Split from ``run_check`` so ``main`` can read
    ``--out`` before any report has been printed into it."""
    ap = argparse.ArgumentParser(
        description="Check that an ingest built every sample's memory",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_exp_arg(ap)
    # The pipeline stages spell it --exp; the sibling checkers (check_retrieval,
    # check_answers) spell it --spec. Accept both rather than making a reader
    # remember which kind of tool this is.
    ap.add_argument("--spec", default=None, help=argparse.SUPPRESS)
    add_system_arg(ap)
    add_dataset_arg(ap)
    add_mem_model_arg(ap, help="Memory model that keys the saved-memory dir")
    add_mem_variant_arg(ap)
    ap.add_argument("--mem-dir", default=None,
                    help="Check this directory instead of the one keyed by "
                         "system/dataset/mem-model (use for a throwaway or a run that "
                         "passed its own --mem-dir)")
    add_limit_arg(ap, help="Only the first N samples were requested (mirrors ingest's "
                           "--limit), so only those count as expected")
    ap.add_argument("--no-dataset", action="store_true",
                    help="Don't read the dataset to learn which samples were expected. "
                         "Skips the missing-sample list; reports counts only. Useful when "
                         "the dataset isn't staged here (longmemeval_s is 277MB to load)")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="Also write the report to PATH. A directory (or a path ending "
                         "in a separator) gets <system>_check.txt inside it. The report "
                         "still goes to the terminal, and the exit code is unchanged")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="Skip the per-sample trajectories. They are what says WHY a call "
                         "failed (and are the only record for the systems that swallow it), "
                         "so this trades the cause for a faster check on a big store")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="List every problem sample and every store's unit count, "
                         "not just the first few")
    args = ap.parse_args()
    if args.spec and not args.exp:
        args.exp = args.spec
    apply_spec(ap, args)
    return ap, args


def run_check(ap: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Print the report; return the exit code. Everything here writes to stdout,
    which is what lets ``main`` tee it to a file without this function knowing."""
    # The mem dir is either given outright or derived from the same keys ingest
    # derived it from — never re-spelled, so a checker and a run can't disagree
    # about where the store is.
    if args.mem_dir:
        mem_dir = args.mem_dir
    else:
        missing = [d for d in ("system", "dataset", "mem_model") if not getattr(args, d, None)]
        if missing:
            ap.error("need --mem-dir, or enough to derive it: missing "
                     + ", ".join("--" + m.replace("_", "-") for m in missing)
                     + " (pass on the CLI or via --exp)")
        mem_dir = ExperimentPaths(system=args.system, dataset=args.dataset,
                                  mem_model=args.mem_model,
                                  mem_variant=args.mem_variant).memory()
    if not os.path.isdir(mem_dir):
        print(f"ERROR: no such memory dir: {mem_dir}", file=sys.stderr)
        return 1

    manifest = read_json(os.path.join(mem_dir, "manifest.json")) or {}
    man_args = manifest.get("args", {})
    dataset = args.dataset or man_args.get("dataset")
    # Only NOTE_KEYS reads this, and only to decide which counters this system
    # is allowed to set without it meaning damage. --exp fills args.system; a
    # bare --mem-dir falls back to what the run recorded for itself.
    system = args.system or man_args.get("system")

    print("=" * 71)
    print(f"Ingest check: {os.path.relpath(mem_dir, REPO)}")
    if man_args:
        print(f"  built by   : {man_args.get('system')} / {man_args.get('dataset')} "
              f"/ {man_args.get('mem_model')}")
    print("=" * 71)

    states = scan_stores(mem_dir)
    complete = sorted(s for s, st in states.items() if st == COMPLETE)
    partial = sorted(s for s, st in states.items() if st == PARTIAL)
    empty = sorted(s for s, st in states.items() if st == EMPTY)

    # ---- expected vs. present ------------------------------------------
    expected: dict[str, str] | None = None
    dataset_size: int | None = None
    if not args.no_dataset:
        if not dataset:
            print("!! no --dataset and none in manifest.json — cannot say which samples "
                  "were expected; reporting counts only (or pass --no-dataset)")
        else:
            print(f"  (reading {dataset} for the expected sample list...)")
            all_slugs = expected_slugs(dataset)
            dataset_size = len(all_slugs)
            expected = (dict(list(all_slugs.items())[:args.limit]) if args.limit
                        else all_slugs)

    print("\nstores")
    if expected is not None:
        never = sorted(set(expected) - set(states))
        unexpected = sorted(set(states) - set(expected))
        print(f"  expected           : {len(expected)}"
              + (f"  (--limit {args.limit})" if args.limit else ""))
    else:
        never, unexpected = [], []
    print(f"  complete           : {len(complete)}")
    if partial:
        print(f"  incomplete save    : {len(partial)}  (memory but no usage.json — "
              f"re-running re-ingests them)")
    if empty:
        print(f"  empty dir          : {len(empty)}  (started, produced nothing that "
              f"survived — re-running re-ingests them)")
    if never:
        print(f"  never started      : {len(never)}")
    if unexpected:
        print(f"  not in the dataset : {len(unexpected)}  (built under a different "
              f"--dataset/--limit?)")

    problems = ([f"    {s:<24} incomplete save" for s in partial]
                + [f"    {s:<24} empty dir" for s in empty]
                + [f"    {s:<24} never started" for s in never]
                + [f"    {s:<24} not in the dataset" for s in unexpected])
    _print_some(problems, 10, args.verbose)

    # ---- failures -------------------------------------------------------
    print("\nfailures.jsonl")
    rows = read_failures(mem_dir)
    if not rows:
        print("  none")
        unresolved: list[dict] = []
    else:
        # Append-only across resumed runs: a line here is a failure that
        # HAPPENED, not necessarily a sample that is still missing. Only the
        # ones whose store is still not complete are outstanding.
        latest: dict[str, dict] = {}
        for r in rows:
            latest[str(r.get("sample_id"))] = r
        unresolved = [r for sid, r in sorted(latest.items())
                      if states.get(os.path.basename(_sample_mem_path("", sid))) != COMPLETE]
        resolved = len(latest) - len(unresolved)
        det = [r for r in unresolved if r.get("deterministic")]
        print(f"  lines              : {len(rows)}  over {len(latest)} distinct sample(s)")
        print(f"  since succeeded    : {resolved}")
        print(f"  still failing      : {len(unresolved)}"
              + (f"  ({len(det)} deterministic — re-running will NOT help)" if det else ""))
        _print_some(
            [f"    {r.get('sample_id')!s:<24} {r.get('error')}: {str(r.get('message'))[:70]}"
             f"{' [deterministic]' if r.get('deterministic') else ''}"
             for r in unresolved],
            10, args.verbose)

    # ---- degradation ----------------------------------------------------
    print("\ndegradation (meta.json, over complete stores)")
    note_keys = note_keys_for(system)
    flagged: list[str] = []
    noted: list[str] = []
    nometa: list[str] = []
    counters: dict[str, int] = {}
    note_stores: collections.Counter = collections.Counter()
    note_totals: collections.Counter = collections.Counter()
    unknown: dict[str, int] = {}
    # Kept for the trajectory section below, which cross-checks it: meta.json
    # counts the calls that failed, the trajectory holds one record per failed
    # call, and the two disagreeing means one of them is not seeing the run.
    meta_failed: dict[str, int] = {}
    for slug in complete:
        meta, flags, notes, size = store_degradation(mem_dir, slug, note_keys)
        if meta is None:
            nometa.append(slug)
            continue
        meta_failed[slug] = int(meta.get("failed_calls") or 0)
        for k in unclassified_counters(meta):
            unknown[k] = unknown.get(k, 0) + 1
        if notes:
            noted.append(f"    {slug:<24} units={size:<7} {notes}")
            for k, v in notes.items():
                note_stores[k] += 1
                note_totals[k] += int(v)
        if flags:
            flagged.append(f"    {slug:<24} units={size:<7} {flags}")
            for k in flags:
                counters[k] = counters.get(k, 0) + 1
        elif args.verbose:
            print(f"    {slug:<24} units={size:<7} "
                  + ("clean (see the noted counter below)" if notes else "clean"))
    print(f"  clean              : {len(complete) - len(flagged) - len(nometa)} / {len(complete)}")
    if counters:
        print(f"  carrying a counter : {len(flagged)}  "
              + ", ".join(f"{k}={n}" for k, n in sorted(counters.items())))
        _print_some(flagged, 10, args.verbose)
    if note_stores:
        # Printed straight under "clean", because it is the qualifier on that
        # word: these stores are counted clean and are carrying a number.
        print(f"  noted, not damage  : {len(noted)}  "
              + ", ".join(f"{k}={n} store(s), {note_totals[k]:,} total"
                          for k, n in sorted(note_stores.items())))
        for k in sorted(note_stores):
            _systems, why, caveat = NOTE_KEYS[k]
            print(f"    {k}: {why}")
            print(f"      caveat: {caveat}")
        _print_some(noted, 5, args.verbose)
    if nometa:
        print(f"  no meta.json       : {len(nometa)}  (predates the counters — "
              f"degradation UNKNOWN, not clean)")
        _print_some([f"    {s}" for s in nometa], 5, args.verbose)
    if unknown:
        # Not a verdict — this tool cannot know whether a key it has never seen
        # counts damage or work done. It says so and asks, rather than doing
        # what it did with truncated_summaries and saying nothing.
        print(f"  UNCLASSIFIED       : {len(unknown)}  "
              + ", ".join(f"{k} ({n} store(s))" for k, n in sorted(unknown.items())))
        print(f"    this tool does not know if these mean damage. Sort each into "
              f"DEGRADATION_KEYS or DESCRIPTIVE_KEYS in {os.path.basename(__file__)}.")

    # ---- traced call errors ---------------------------------------------
    # The counters above say a call failed. This says what the endpoint (or the
    # client) actually reported, which is the difference between a store worth
    # rebuilding and a configuration worth fixing first.
    traced_total = 0
    traced_det = 0
    traced_unclassified = 0
    traced_stores: list[str] = []
    no_traj_count = 0
    if not args.no_trajectory:
        print("\ncall errors (trajectory, over complete stores)")
        buckets: collections.Counter = collections.Counter()
        stages: collections.Counter = collections.Counter()
        examples: dict[str, str] = {}
        levels: collections.Counter = collections.Counter()
        no_traj: list[str] = []
        # Stores whose calls raised while no counter moved. This is the case the
        # section exists for: lightmem's update pool and mem0's resolver both
        # swallow the exception, so the store saves clean and only the tracer
        # remembers (see the module docstring).
        uncounted: list[str] = []
        for slug in complete:
            errs = read_call_errors(mem_dir, slug)
            if not errs.levels:
                no_traj.append(slug)
                continue
            for lv in errs.levels:
                levels[lv] += 1
            if not errs.total:
                continue
            traced_stores.append(
                f"    {slug:<24} {errs.total:>5} raised  "
                + ", ".join(f"{b}={n}" for b, n in sorted(errs.buckets.items()))
                + (f"  [stages: {', '.join(sorted(errs.stages))}]" if errs.stages else ""))
            buckets.update(errs.buckets)
            stages.update(errs.stages)
            for b, ex in errs.examples.items():
                examples.setdefault(b, ex)
            if slug in meta_failed and meta_failed[slug] == 0:
                uncounted.append(slug)
        traced_total = sum(buckets.values())
        traced_det = sum(n for b, n in buckets.items() if b in DETERMINISTIC_BUCKETS)
        traced_unclassified = buckets.get(UNCLASSIFIED_ERROR, 0)

        no_traj_count = len(no_traj)
        read = len(complete) - no_traj_count
        print(f"  trajectories read  : {read} / {len(complete)}"
              + (f"  ({', '.join(f'{k}: {v}' for k, v in sorted(levels.items()))})"
                 if levels else ""))
        if no_traj:
            # Not "clean": trace_level 'off', --trace-samples sampling and a
            # store that predates tracing all look like this, and none of them
            # is evidence that no call failed.
            print(f"  no trajectory      : {len(no_traj)}  (cause UNKNOWN for these — "
                  f"trace_level off, sampled, or predates tracing)")
            _print_some([f"    {s}" for s in no_traj], 5, args.verbose)
        if not traced_total:
            if read:
                print(f"  raised calls       : 0  over {read} trajectory file(s)")
            # else: nothing was read, and a zero here would read as a result
            # rather than as the absence of one. The line above already said it.
        else:
            print(f"  raised calls       : {traced_total:,}  over "
                  f"{len(traced_stores)} store(s)")
            for bucket, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
                print(f"    {bucket:<16} {n:>6}  {BUCKET_ADVICE.get(bucket, '')}")
                print(f"        e.g. {examples.get(bucket, '')}")
            if stages:
                print("  by stage           : "
                      + ", ".join(f"{s}={n}" for s, n in sorted(stages.items())))
            _print_some(traced_stores, 10, args.verbose)
            if uncounted:
                print(f"  !! not counted     : {len(uncounted)} store(s) recorded a raised "
                      f"call but carry failed_calls=0")
                print("     the system swallowed it — the store looks clean and is not.")
                _print_some([f"    {s}" for s in uncounted], 5, args.verbose)
            # Both sides count the same events, so a gap is a fact about the
            # harness rather than about the run: a system whose adapter has no
            # failed_calls counter at all, or one whose retries fail inside a
            # seam record_calls does not wrap.
            counted = sum(meta_failed.values())
            if counted != traced_total:
                print(f"  vs meta.json       : failed_calls totals {counted:,} against "
                      f"{traced_total:,} traced — they count the same calls, so the gap "
                      f"is unrecorded either way")

    # ---- provenance & cost ----------------------------------------------
    print("\nprovenance")
    index_path = os.path.join(mem_dir, "index.jsonl")
    fingerprints: list[str] = []
    if os.path.isfile(index_path):
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sha = json.loads(line).get("config_sha256")
                except ValueError:
                    continue
                if sha and sha not in fingerprints:
                    fingerprints.append(sha)
    if len(fingerprints) > 1:
        # Ingest is resumable, so a run started with an edited YAML ADDS to the
        # store rather than rebuilding it — leaving one dir holding samples
        # built under two configurations, with nothing on disk marking which is
        # which. (A moved base_url alone also trips this.)
        print(f"  !! {len(fingerprints)} distinct config fingerprints in index.jsonl: "
              f"{', '.join('sha256:' + s for s in fingerprints)}")
        print("     samples in this dir were built under more than one config.")
    elif fingerprints:
        print(f"  config             : sha256:{fingerprints[0]}  (one, consistent)")

    agg = read_json(os.path.join(mem_dir, "ingest_usage.json"))
    stale_agg = False
    if agg:
        n = agg.get("samples") or 0
        note = ""
        if n != len(complete):
            # It is recomputed from the per-sample files on every run, so a
            # disagreement means it was last written when the store looked
            # different — an interrupted run, or a partial copy of the dir.
            stale_agg = True
            note = f"  <-- disagrees with {len(complete)} complete store(s); stale aggregate"
        print(f"  ingest_usage.json  : samples={n}{note}")
        if n:
            calls, secs = agg.get("llm_calls", 0), agg.get("seconds", 0.0)
            print(f"  cost               : {calls:,} calls, {agg.get('tokens', 0):,} tokens, "
                  f"{secs / 60:,.1f} min")
            line = f"  per sample         : {calls / n:,.0f} calls, {secs / n:,.0f}s"
            # What a partial store extrapolates to is the cost of the WHOLE
            # dataset, not of the --limit that produced it — the number a smoke
            # test is run to get.
            if dataset_size and dataset_size > n:
                line += (f"   -> x{dataset_size} = "
                         f"{secs / n * dataset_size / 3600:,.0f} GPU-h")
            print(line)

    # ---- verdict --------------------------------------------------------
    print("=" * 71)
    # traced_total joins the verdict rather than only the report: a raised call
    # is content that is not in the store, and for lightmem and mem0 it is the
    # ONLY signal that says so — gating on `flagged` alone is what let those two
    # save a damaged store and exit 0.
    ok = (not partial and not empty and not never and not unresolved
          and not flagged and not nometa and not traced_total
          and (expected is None or len(complete) >= len(expected)))
    if ok:
        scope = f"all {len(expected)} expected" if expected is not None else f"all {len(complete)}"
        print(f"PASS — {scope} sample(s) have a complete, clean store")
        if no_traj_count:
            # Same shape as the UNCLASSIFIED qualifier below, and same reason:
            # running without tracing is a normal, deliberate choice (a full
            # trajectory is larger than the memory it describes), so it must not
            # gate a driver script — but "clean" one line up would otherwise
            # claim a check that had nothing to read.
            print(f"       ({no_traj_count} sample(s) had no trajectory — no call "
                  f"raised in the ones that did, UNKNOWN for the rest)")
        if note_stores:
            # Same shape as the two qualifiers around it: "clean" above is a
            # claim about damage, and these stores did carry a counter — one
            # this system sets by design. The caveat printed with it upstream is
            # the reason the number is not simply dropped.
            print("       ("
                  + ", ".join(f"{k} set in {n} store(s)"
                              for k, n in sorted(note_stores.items()))
                  + " — noted above, not damage for this system)")
        if unknown:
            # Deliberately does NOT flip the exit code: a new counter is far
            # more often a new descriptive one than new damage, and gating a
            # driver script on "a system grew a field" would be wrong. But the
            # word "clean" one line up is a claim about keys this tool knows,
            # so the qualifier is printed with it rather than 30 lines above.
            print("       (\"clean\" covers the counters listed above only — "
                  f"{len(unknown)} UNCLASSIFIED key(s) were not judged)")
    else:
        print("CHECK — see above; the run's own summary covers only what succeeded")
        if partial or empty or never:
            print("        re-run the same ingest command to resume the missing samples")
        if traced_det:
            print(f"        {traced_det:,} raised call(s) are DETERMINISTIC — re-running "
                  f"reproduces them; fix the config first")
        if traced_total and not traced_det:
            print(f"        {traced_total:,} raised call(s), none deterministic — "
                  f"re-ingesting the affected samples should recover them")
        if traced_unclassified:
            print(f"        {traced_unclassified:,} raised call(s) matched no bucket — "
                  f"sort them into CALL_ERROR_CLASSES in "
                  f"{os.path.basename(__file__)}")
        if nometa:
            print("        re-ingest with --overwrite to get counters for the UNKNOWN stores")
    if stale_agg:
        print("        ingest_usage.json is stale — any cost read off it is wrong; "
              "it is rewritten by the next ingest")
    return 0 if ok else 1


class _Tee:
    """Writes to both streams. Used to capture the report without moving it.

    A tee rather than a redirect: the exit code is only half of what a driver
    script reads — someone watching the run still needs the report on screen,
    and a checker that goes silent the moment you ask it to save its output is
    the kind of thing you find out about after the run.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for s in self._streams:
            s.write(text)
        return len(text)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _out_label(args: argparse.Namespace) -> str:
    """What to call this check when ``--out`` names a directory.

    ``--system`` when there is one (set directly or by ``--exp``), else the
    store's own directory name, which is what ``--mem-dir`` runs have. Never the
    dataset or the model: a reader comparing checks is comparing systems, and
    the keyed path already carries the rest.
    """
    if getattr(args, "system", None):
        return str(args.system)
    if getattr(args, "mem_dir", None):
        return os.path.basename(str(args.mem_dir).rstrip(os.sep)) or "ingest"
    return "ingest"


def resolve_out_path(args: argparse.Namespace) -> str:
    """Where ``--out`` writes.

    A directory — existing, or written with a trailing separator — takes
    ``<label>_check.txt`` inside it, so checking six systems into one directory
    does not need six paths spelled out. Anything else is used verbatim.
    """
    out = str(args.out)
    if out.endswith((os.sep, "/")) or os.path.isdir(out):
        return os.path.join(out, f"{_out_label(args)}_check.txt")
    return out


def main() -> int:
    ap, args = _parse_args()
    if not args.out:
        return run_check(ap, args)

    path = resolve_out_path(args)
    buf = io.StringIO()
    # The report is captured whatever happens — a checker that dies half-way
    # through leaves the half it managed, which is the part that says why.
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            code = run_check(ap, args)
    finally:
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                f.write(buf.getvalue())
            print(f"\nwritten: {path}")
        except OSError as e:
            # Reported, never fatal: the check itself already ran, and its exit
            # code is the answer. Losing the copy must not lose the verdict.
            print(f"\n!! could not write {path}: {e}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
