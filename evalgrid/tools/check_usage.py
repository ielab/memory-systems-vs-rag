#!/usr/bin/env python3
"""
check_usage.py
==============
Check — and repair — each cell's usage aggregate against the per-sample
evidence it is supposed to summarise.

Every cell records its cost twice: once per sample, written as that sample
finishes, and once per cell, which is the file a cost table actually reads. The
two stages build that aggregate in opposite ways, and only one of them survives
a resumed run.

  INGEST     ``<mem_dir>/ingest_usage.json`` is RECOMPUTED from every
             ``<sample>/usage.json`` at the end of each run
             (``runner._write_aggregate_ingest_usage``). A resume is therefore
             free: the samples this run skipped still have their per-sample
             files and are still summed. It goes wrong in one way only — the
             process died before the loop returned, so the file was never
             rewritten and now describes a smaller, older store.

  RETRIEVAL  ``<out_dir>/retrieval_usage.json`` is written from the RUN's own
             counter (``retrieve.py``'s ``spent``) and TRUNCATES the file each
             time. A resumed run skips every sample that already has a
             ``retrieval.jsonl``, spends nothing, and overwrites the cell's
             totals with its own. The first pass's cost is not subtracted, it is
             forgotten — leaving a file that says a cell which made 22 LLM calls
             made none, with `samples: 0` beside it.

The second failure is not hypothetical (10 of the 48 cells under ``outputs/``
were in exactly this state when this tool was written — every one of them a
simplemem or a_mem cell, i.e. precisely the two systems whose retrieval cost is
not zero and is the whole point of the C2 column), and it CANNOT be repaired by
re-running: a re-run skips the samples again and writes the same zeros.

It is repairable from disk, because ``retrieve.py`` also records each query's
cost in the trace it writes — every line of ``<sample>/retrieval.jsonl`` carries
a ``retrieval_cost`` block — so summing those over a cell reconstructs what the
aggregate should have said.

So: this tool re-derives both aggregates from the per-sample files and prints
every disagreement; ``--fix`` writes the re-derived numbers back.

What --fix cannot recover
-------------------------
Named here rather than in a footnote, because a repaired file that quietly
understates is worse than an obviously broken one. A repaired retrieval
aggregate is stamped with a ``recomputed`` block recording all of this, and the
original file is preserved inside it.

  * **Tokens burned by FAILED queries.** ``retrieve.py`` charges them to
    ``spent`` (deliberately — a failure that costs GPU time and reports zero is
    how a cost table stops adding up) but writes no trace line for them, so they
    exist nowhere on disk. If the cell reports ``queries_failed`` or carries an
    ``errors.jsonl``, the recomputed total is a LOWER BOUND and is marked as one.
  * **Traces written before ``retrieval_cost`` existed.** A cell whose lines
    lack the block would recompute to a false zero, so ``--fix`` refuses it
    (``--force`` overrides, and stamps the count of unpriced lines).
  * **Wall-clock ``seconds``.** The original is one pass's wall time — already
    wrong for a resumed cell, and not reconstructible. The repair writes the sum
    of the per-query ``retrieval_seconds`` instead, which is in-loop retrieval
    time only (it excludes ``load_memory`` and indexing) and is a different,
    smaller quantity. ``recomputed.seconds_basis`` says so.

  * **Flat-retriever cells.** A cell whose file carries an ``encode`` block was
    written by the flat half of the stage, which keeps no per-sample traces —
    there is nothing on disk to recompute from, so the file is reported as
    correct-but-unverified and ``--fix`` leaves it alone.

A MISSING ``retrieval_usage.json`` is reported but NOT written by ``--fix``.
Its absence is load-bearing: ``retrieve.py`` writes it after the sample loop
returns, so ``check_grid`` and ``eval_retrieval`` both read "no file" as "the
loop crashed partway, do not score this cell". Synthesising one would erase that
signal and hand a crashed cell a clean bill of health. ``--write-missing`` does
it anyway, for the case where you know why the file is gone. A missing
``ingest_usage.json`` carries no such meaning and is simply written.

Usage
-----
    # every cell under a root, ingest and retrieval alike
    python -m evalgrid.tools.check_usage outputs/_memory outputs/_smoke4_retrieval

    # the whole tree, then repair what disagrees
    python -m evalgrid.tools.check_usage outputs
    python -m evalgrid.tools.check_usage outputs --fix

    python -m evalgrid.tools.check_usage outputs -v        # per-cell detail
    python -m evalgrid.tools.check_usage outputs --json     # machine-readable

Exit code is 0 only when every cell's aggregate agrees with its per-sample
files (after any ``--fix``), so it can gate a cost table in a driver script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# evalgrid/tools/ -> repo root; make the package importable when run as a file.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.lib.cost import query_cost  # noqa: E402
from evalgrid.lib.io import write_json_atomic  # noqa: E402

try:
    # The constant's home. pipeline.retrieve pulls torch in through the model
    # factory, so it is never a hard dependency here: the retrieval half of this
    # tool only reads JSON off disk and runs on any machine. (The ingest
    # half does need benchmarks.runner — see recompute_ingest.)
    from evalgrid.pipeline.retrieve import RETRIEVAL_FILE
except Exception:  # pragma: no cover - import weight, not logic
    RETRIEVAL_FILE = "retrieval.jsonl"

SAMPLE_USAGE = "usage.json"
INGEST_AGG = "ingest_usage.json"
RETRIEVAL_AGG = "retrieval_usage.json"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_json(path: str):
    """Parse ``path``, or None if it is absent/empty/corrupt.

    A torn aggregate is reported as missing rather than raising: this tool's job
    is to survey a whole grid, and one unreadable file must not end the survey.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_cells(root: str) -> list[tuple[str, str]]:
    """Every ('ingest'|'retrieval', cell_dir) under ``root``, deduplicated.

    Found from the SAMPLE dirs rather than from the aggregates, so a cell whose
    aggregate was never written is discovered too — that is one of the states
    worth reporting. The aggregates are then swept for the opposite case: a cell
    holding a total with no samples left under it (a moved or half-deleted dir).
    """
    cells: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if SAMPLE_USAGE in filenames:
            cells.setdefault(os.path.dirname(dirpath), "ingest")
            dirnames[:] = []          # a sample store; nothing below it is a cell
            continue
        if RETRIEVAL_FILE in filenames:
            cells.setdefault(os.path.dirname(dirpath), "retrieval")
            dirnames[:] = []
            continue
        if INGEST_AGG in filenames:
            cells.setdefault(dirpath, "ingest")
        elif RETRIEVAL_AGG in filenames:
            cells.setdefault(dirpath, "retrieval")
    return sorted((kind, path) for path, kind in cells.items())


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def recompute_ingest(mem_dir: str) -> dict:
    """What ``ingest_usage.json`` would say if it were rewritten right now.

    Built with the pipeline's OWN summing functions, imported rather than
    restated: the point of the comparison is that the repaired file is byte-for
    -byte what the next ingest would produce, which a reimplementation here
    could not promise.
    """
    from evalgrid.benchmarks.runner import _sum_by_layer, _usage_block

    deltas: list[dict] = []
    tot_seconds = 0.0
    n = 0
    for entry in os.scandir(mem_dir):
        fp = os.path.join(entry.path, SAMPLE_USAGE)
        if entry.is_dir() and os.path.isfile(fp):
            u = read_json(fp)
            if u is None:
                continue
            deltas.append(u.get("by_layer", {}))
            tot_seconds += u.get("ingest_seconds", 0.0)
            n += 1
    block = _usage_block(_sum_by_layer(deltas), seconds=tot_seconds)
    block["samples"] = n
    return block


def check_ingest_cell(path: str) -> dict:
    """Compare ``ingest_usage.json`` with the per-sample usage.json files."""
    fresh = recompute_ingest(path)
    prev = read_json(os.path.join(path, INGEST_AGG))
    rep = {"kind": "ingest", "path": path, "samples": fresh["samples"],
           "fresh": fresh, "prev": prev, "diffs": [], "fixable": True,
           "blockers": [], "notes": []}

    if prev is None:
        rep["status"] = "MISSING" if fresh["samples"] else "EMPTY"
        if fresh["samples"]:
            rep["diffs"].append(f"no {INGEST_AGG}; {fresh['samples']} sample(s) hold one")
        return rep

    for key in ("samples", "tokens", "llm_calls", "input_tokens", "output_tokens"):
        if int(prev.get(key, 0)) != int(fresh.get(key, 0)):
            rep["diffs"].append(f"{key}: file={prev.get(key, 0):,} disk={fresh.get(key, 0):,}")
    # Seconds is a float sum in both, so compare with a tolerance rather than
    # flagging every cell over a rounding difference.
    if abs(float(prev.get("seconds", 0.0)) - float(fresh.get("seconds", 0.0))) > 1.0:
        rep["diffs"].append(f"seconds: file={prev.get('seconds', 0.0):,.0f} "
                            f"disk={fresh.get('seconds', 0.0):,.0f}")
    rep["status"] = "STALE" if rep["diffs"] else "OK"
    return rep


def fix_ingest_cell(path: str) -> None:
    """Rewrite the aggregate with the pipeline's own writer.

    Called rather than reimplemented so that a repaired file and an
    ingest-written one cannot drift apart.
    """
    from evalgrid.benchmarks.runner import _write_aggregate_ingest_usage
    _write_aggregate_ingest_usage(path)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def recompute_retrieval(out_dir: str) -> dict:
    """Sum ``retrieval_cost`` over every ``<sample>/retrieval.jsonl`` in a cell.

    This is the only surviving per-sample record of query-time cost, which is
    what makes a resumed cell repairable at all. ``unpriced`` counts trace lines
    with no cost block — they are not zero-cost queries, they are queries whose
    cost was never written, and a sum over them is a floor, not a total.
    """
    samples = queries = unpriced = 0
    t_in = t_out = t_calls = 0
    seconds = 0.0
    for entry in sorted(os.scandir(out_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        fp = os.path.join(entry.path, RETRIEVAL_FILE)
        if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
            continue
        samples += 1
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                queries += 1
                cost = rec.get("retrieval_cost")
                if not isinstance(cost, dict):
                    unpriced += 1
                    continue
                t_in += int(cost.get("input_tokens", 0) or 0)
                t_out += int(cost.get("output_tokens", 0) or 0)
                t_calls += int(cost.get("calls", 0) or 0)
                seconds += float(rec.get("retrieval_seconds") or 0.0)
    return {"samples": samples, "queries": queries, "unpriced": unpriced,
            "input_tokens": t_in, "output_tokens": t_out, "tokens": t_in + t_out,
            "llm_calls": t_calls, "seconds": round(seconds, 3)}


def check_retrieval_cell(path: str) -> dict:
    """Compare ``retrieval_usage.json`` with the traces the cell actually holds."""
    fresh = recompute_retrieval(path)
    prev = read_json(os.path.join(path, RETRIEVAL_AGG))
    rep = {"kind": "retrieval", "path": path, "samples": fresh["samples"],
           "fresh": fresh, "prev": prev, "diffs": [], "fixable": True,
           "blockers": [], "notes": []}

    # A FLAT cell has nothing to recompute from, and must not be "repaired".
    #
    # Only the flat half of pipeline/retrieve writes an `encode` block, and that
    # half writes no per-sample retrieval.jsonl — it has no per-query trace to
    # write, because it runs no system and produces no context to replay. So
    # `fresh` is all zeros for structural reasons, and without this the cell
    # would read as STALE against its own correct file and --fix would overwrite
    # a real encoder total with that zero. The file IS the only record here;
    # treat it as authoritative and say why it was not cross-checked.
    if isinstance(prev, dict) and "encode" in prev:
        rep["status"] = "OK"
        rep["fixable"] = False
        rep["blockers"].append(
            "flat-retriever cell: no per-sample traces exist to verify it against, "
            "so these totals are taken on trust and --fix will not touch them")
        return rep

    if fresh["unpriced"]:
        # Not a diff — a reason the comparison itself cannot be trusted.
        rep["blockers"].append(f"{fresh['unpriced']} trace line(s) carry no "
                               f"retrieval_cost; the recomputed total is a floor")
        rep["fixable"] = False

    if prev is None:
        # Load-bearing absence — see the module docstring.
        rep["status"] = "MISSING" if fresh["samples"] else "EMPTY"
        if fresh["samples"]:
            rep["diffs"].append(f"no {RETRIEVAL_AGG}; {fresh['samples']} sample(s) "
                                f"retrieved (the sample loop did not return)")
        return rep

    # `samples` in the file counts what THIS run newly retrieved; the cell's
    # coverage is that plus what it skipped as already-done.
    covered = int(prev.get("samples", 0)) + int(prev.get("samples_resumed", 0))
    if covered != fresh["samples"]:
        rep["diffs"].append(f"samples: file={covered:,} "
                            f"(new={prev.get('samples', 0)}, "
                            f"resumed={prev.get('samples_resumed', 0)}) "
                            f"disk={fresh['samples']:,}")
    for key in ("queries", "tokens", "llm_calls", "input_tokens", "output_tokens"):
        if int(prev.get(key, 0)) != int(fresh.get(key, 0)):
            rep["diffs"].append(f"{key}: file={prev.get(key, 0):,} disk={fresh.get(key, 0):,}")

    # Failures matter here for one reason: retrieve.py charges a failed call's
    # tokens to its own counter but writes no trace line for them, so they are in
    # THIS file and in nothing else. Reported with the counts rather than as a
    # standing sentence, because the two cases below need opposite responses and
    # the generic wording could not tell them apart.
    n_qf = int(prev.get("queries_failed", 0))
    n_sf = int(prev.get("samples_failed", 0))
    if n_qf or n_sf:
        rep["blockers"].append(
            f"the run that wrote this file lost {n_sf} sample(s) / {n_qf} query/queries; "
            f"their tokens are in these totals and in no per-sample file, so a "
            f"recomputation would understate. They left no run.tsv, so eval_retrieval "
            f"refuses this cell and re-running retrieve retries exactly them")
    elif os.path.isfile(os.path.join(path, "errors.jsonl")):
        # errors.jsonl is appended and never truncated; the counters above are
        # rewritten every run. Counters at zero beside a non-empty error log is
        # therefore the signature of failures a LATER pass already fixed —
        # nothing to act on, and not a caveat on the totals either.
        rep["notes"].append("errors.jsonl present but this file reports no failures — "
                            "the log is append-only, so those were fixed by a later pass")
    rep["status"] = "STALE" if rep["diffs"] else "OK"
    return rep


def fix_retrieval_cell(path: str, rep: dict) -> None:
    """Write the disk-derived totals back, preserving what cannot be re-derived.

    Coverage counters that describe the LAST RUN (``samples_no_memory``,
    ``samples_failed``, ``queries_failed``) are carried through untouched:
    ``eval_retrieval`` refuses to score a cell on them, and a repair of the cost
    fields must not quietly clear a gate that is doing its job.

    ``samples_resumed`` goes to 0 and ``samples`` becomes the whole cell,
    because after this rewrite the totals cover every sample on disk — the
    new/resumed split describes a single pass and no longer means anything here.
    The untouched original is kept in ``recomputed.previous``.
    """
    fresh = rep["fresh"]
    prev = rep["prev"] or {}
    spent = (fresh["input_tokens"], fresh["output_tokens"], fresh["llm_calls"])
    queries = fresh["queries"]
    lower_bound = bool(rep["blockers"])

    out = {
        "stage": "retrieve",
        "retriever": prev.get("retriever") or os.path.basename(path.rstrip("/")),
        "samples": fresh["samples"],
        "samples_resumed": 0,
        "samples_no_memory": int(prev.get("samples_no_memory", 0)),
        "samples_failed": int(prev.get("samples_failed", 0)),
        "queries_failed": int(prev.get("queries_failed", 0)),
        "queries": queries,
        "seconds": fresh["seconds"],
        "input_tokens": fresh["input_tokens"],
        "output_tokens": fresh["output_tokens"],
        "tokens": fresh["tokens"],
        "llm_calls": fresh["llm_calls"],
        **{k: v for k, v in query_cost(retrieval=spent).items()
           if k.startswith("query_retrieval_")},
        "per_query_tokens": fresh["tokens"] / queries if queries else 0.0,
        "per_query_calls": fresh["llm_calls"] / queries if queries else 0.0,
        "note": prev.get("note",
                         "paid once; shared by every --from-retrieval answer run over this dir"),
        "recomputed": {
            "tool": "evalgrid.tools.check_usage",
            "at": datetime.now().isoformat(timespec="seconds"),
            "basis": f"sum of retrieval_cost over <sample>/{RETRIEVAL_FILE}",
            "reason": "the stage writes this file from one run's counters and "
                      "truncates it, so a resumed cell reports only the last pass",
            "seconds_basis": "sum of per-query retrieval_seconds — in-loop retrieval "
                             "only, NOT the run's wall clock",
            "samples_field": "every sample on disk; the new/resumed split described "
                             "one pass and does not apply to these totals",
            "cost_is_lower_bound": lower_bound,
            "unpriced_trace_lines": fresh["unpriced"],
            "previous": prev,
        },
    }
    if lower_bound:
        out["recomputed"]["caveats"] = list(rep["blockers"])
    # Atomic for the same reason the stage's own writes are: a cost table may be
    # reading this file while a repair sweep rewrites it.
    write_json_atomic(os.path.join(path, RETRIEVAL_AGG), out, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check (and repair) per-cell usage aggregates against per-sample files.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+",
                    help="directories to search for cells (outputs/_memory, "
                         "outputs/_smoke4_retrieval, or just outputs)")
    ap.add_argument("--fix", action="store_true",
                    help="rewrite aggregates that disagree with their per-sample files")
    ap.add_argument("--write-missing", action="store_true",
                    help="with --fix, also create an absent retrieval_usage.json "
                         "(its absence normally marks a crashed cell — see the docstring)")
    ap.add_argument("--force", action="store_true",
                    help="with --fix, repair even cells whose recomputed total is "
                         "only a lower bound (unpriced traces / failed queries)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show every cell, not only the ones that disagree")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args()

    cells: list[tuple[str, str]] = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[warn] not a directory: {root}", file=sys.stderr)
            continue
        cells.extend(find_cells(root))
    cells = sorted(set(cells))

    # benchmarks.runner is a hard dependency of the ingest half only (its
    # summing functions are imported rather than restated so a repaired file is
    # exactly what the next ingest would write). Checked once, up front, so an
    # environment without it produces one explanation instead of the same
    # ImportError repeated per cell — and never a silent pass.
    ingest_blocked = None
    if any(k == "ingest" for k, _ in cells):
        try:
            from evalgrid.benchmarks.runner import _usage_block  # noqa: F401
        except Exception as e:
            ingest_blocked = f"{type(e).__name__}: {e}"

    reports = []
    for kind, path in cells:
        if kind == "ingest" and ingest_blocked:
            reports.append({"kind": kind, "path": path, "status": "SKIPPED",
                            "diffs": [], "blockers": [], "notes": [], "fixable": False,
                            "samples": 0, "fresh": {}, "prev": None})
            continue
        try:
            rep = check_ingest_cell(path) if kind == "ingest" else check_retrieval_cell(path)
        except Exception as e:  # one unreadable cell must not end the sweep
            rep = {"kind": kind, "path": path, "status": "ERROR",
                   "diffs": [f"{type(e).__name__}: {e}"], "blockers": [], "notes": [],
                   "fixable": False, "samples": 0, "fresh": {}, "prev": None}
        reports.append(rep)

    # ---- repair ---------------------------------------------------------
    fixed, refused = [], []
    if args.fix:
        for rep in reports:
            if rep["status"] not in ("STALE", "MISSING"):
                continue
            if rep["status"] == "MISSING" and rep["kind"] == "retrieval" and not args.write_missing:
                refused.append((rep, "absent file marks a crashed cell; --write-missing to override"))
                continue
            if not rep["fixable"] and not args.force:
                refused.append((rep, "; ".join(rep["blockers"]) + " — --force to write anyway"))
                continue
            try:
                if rep["kind"] == "ingest":
                    fix_ingest_cell(rep["path"])
                else:
                    fix_retrieval_cell(rep["path"], rep)
                rep["status"] = "FIXED"
                fixed.append(rep)
            except Exception as e:
                refused.append((rep, f"{type(e).__name__}: {e}"))

    if args.json:
        print(json.dumps([{k: v for k, v in r.items() if k != "prev"} for r in reports],
                         indent=2, default=str))
        return 0 if all(r["status"] in ("OK", "EMPTY", "FIXED") for r in reports) else 1

    # ---- report ---------------------------------------------------------
    order = {"ERROR": 0, "MISSING": 1, "SKIPPED": 2, "STALE": 3, "FIXED": 4,
             "EMPTY": 5, "OK": 6}
    width = max([len(r["path"]) for r in reports] + [20])
    width = min(width, 78)
    print(f"{len(reports)} cell(s) — aggregate vs per-sample files\n")
    shown = 0
    for rep in sorted(reports, key=lambda r: (order[r["status"]], r["path"])):
        interesting = rep["status"] not in ("OK", "EMPTY") or rep["blockers"]
        if rep["status"] == "SKIPPED" and not args.verbose:
            continue
        if not interesting and not args.verbose:
            continue
        shown += 1
        path = rep["path"] if len(rep["path"]) <= width else "..." + rep["path"][-(width - 3):]
        print(f"  {rep['status']:<8} {rep['kind']:<9} {path}")
        for d in rep["diffs"]:
            print(f"      {d}")
        for b in rep["blockers"]:
            print(f"      ! {b}")
        if args.verbose:
            for n in rep["notes"]:
                print(f"      - {n}")
    if not shown:
        print("  (nothing to report — pass -v to list every cell)")

    counts: dict[str, int] = {}
    for rep in reports:
        counts[rep["status"]] = counts.get(rep["status"], 0) + 1
    print("\n" + "=" * 71)
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: order[kv[0]])))
    for rep, why in refused:
        print(f"  NOT FIXED  {rep['path']}\n             {why}")

    unverified = [r for r in reports if r["status"] == "SKIPPED"]
    if unverified:
        print(f"  {len(unverified)} ingest cell(s) NOT CHECKED — evalgrid.benchmarks.runner "
              f"will not import here ({ingest_blocked}).")
        print(f"  Run this from the environment the pipeline runs in; the retrieval "
              f"half above needs nothing.")

    bad = [r for r in reports if r["status"] in ("STALE", "MISSING", "ERROR")]
    if not bad and not unverified:
        print("PASS — every aggregate agrees with the per-sample files under it")
        caveated = [r for r in reports if r["status"] == "OK" and r["blockers"]]
        if caveated:
            # An OK cell with a "!" line has correct totals; the caveat is about
            # what a REPAIR of it could not recover, which is a different claim
            # and was being read as a failure.
            print(f"       ({len(caveated)} cell(s) carry a '!' caveat above: their totals "
                  f"are correct as written — it says what a repair could not recover)")
        return 0
    if bad:
        print(f"CHECK — {len(bad)} aggregate(s) disagree with the samples they "
              f"summarise; any cost read off them is wrong")
    else:
        print("CHECK — nothing disagreed, but not every cell could be checked")
    if bad and not args.fix:
        print("        re-run with --fix to rewrite them from the per-sample files")
    return 1


if __name__ == "__main__":
    sys.exit(main())
