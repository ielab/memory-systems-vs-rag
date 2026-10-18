#!/usr/bin/env python3
"""
rescore_eval.py
===============
Find the eval cells whose metrics no longer describe the retrieval under them,
and re-score exactly those — by replaying each cell's OWN last eval invocation.

Why a tool rather than a re-run of the driver
---------------------------------------------
``eval_retrieval`` needs no ``--overwrite``: it recomputes every metric from
qrels + run.tsv and truncates its ``--out`` (``open(out_path, "w")``). Re-running
it IS the overwrite. What is awkward is everything around that:

  * The driver never runs it alone. ``scripts/query.sh`` runs eval as
    step 3 of qrels -> retrieve -> eval, so re-submitting a cell to re-score it
    also re-runs the retrieve stage and needs vLLM and a GPU for a job whose
    actual work is a few seconds of CPU arithmetic over TSVs.
  * The arguments are not guessable. ``--k-values`` alone decides which columns
    exist, and the smoke grid ran ``3,5,10,20`` against the stage default of
    ``5,10,20`` — re-scoring "by hand" with the default silently produces a
    metrics.json with different columns under the same name.
  * ``--out`` may not be inside the run dir — in fact it usually is not. The
    stage now defaults to the keyed ``outputs/_eval`` cell, a SIBLING tree, and
    the shell drivers pass that same path explicitly; only a run named with
    ``--retrieval-dir`` (a fused run) keeps its score beside
    itself. Which file a given cell publishes to therefore has to come from
    somewhere, and a manifest written before the default moved records a null
    that means the OLD default.

All three are answered by one file that is already on disk: ``manifest.json`` in
each retrieval dir, which ``eval_retrieval`` writes with the exact resolved
arguments of its last run — retriever, emb model, qrels dir, k-values, out path.
This tool replays that, so a re-score is the same measurement, not a new one.

When a cell actually needs re-scoring
-------------------------------------
Only when the RANKINGS changed. Adding samples to a retrieval cell (the usual
case: a resumed retrieve pass) leaves the old metrics averaging over fewer
queries than the cell now holds, which is the state this looks for:

  ``metrics.json``'s ``overall.queries`` is exactly the number of judged queries
  that had a run entry when it was written (``benchmarks.retrieval.evaluate``
  counts ``qrels n run``). Recomputing that intersection from disk and comparing
  is a CONTENT check — it does not care about mtimes, copies or rsync.

An mtime check runs beside it, because re-ranking the SAME queries (a retrieve
pass with ``--overwrite``) changes every score while leaving the count identical.
It is reported as a weaker signal and says so: an rsync that did not preserve
times can raise it on a cell that is fine.

Repairing a cell's ``retrieval_usage.json`` with ``evalgrid.tools.check_usage``
is NOT a reason to re-score. Metrics never read it for anything but the
completeness gate, and the repair preserves that gate's counters, so nDCG does
not move.

Usage
-----
    # what would be re-scored, and why (default: nothing is executed)
    python -m evalgrid.tools.rescore_eval outputs/_smoke4_retrieval

    # do it
    python -m evalgrid.tools.rescore_eval outputs/_smoke4_retrieval --run

    python -m evalgrid.tools.rescore_eval outputs/_retrieval --run --all
    python -m evalgrid.tools.rescore_eval outputs/_retrieval --print-commands

Point it at a RETRIEVAL root (that is where the manifests live). Run it from the
repo root: the manifests record paths as they were typed, which was relative to
it.

Exit code is 0 when nothing is stale, or when every re-score succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.benchmarks.gold import (STAMPED_GOLD_LEVEL,  # noqa: E402
                                      gold_filename, gold_suffix)
from evalgrid.lib.paths import ExperimentPaths, labeler_segment  # noqa: E402

#: manifest key -> CLI flag. Order is the flag order of a replayed command.
#: ``exp`` is deliberately absent: apply_spec resolves a spec INTO these values
#: before the manifest is written, so a replay needs no spec file and cannot be
#: changed by one that was edited since.
_FLAGS = [
    ("system", "--system"),
    ("dataset", "--dataset"),
    ("mem_model", "--mem-model"),
    ("mem_variant", "--mem-variant"),
    ("labeler_model", "--labeler-model"),
    ("retriever", "--retriever"),
    ("emb_model", "--emb-model"),
    ("run_name", "--run-name"),
    ("candidate_source", "--candidate-source"),
    ("qrels_dir", "--qrels-dir"),
    ("retrieval_dir", "--retrieval-dir"),
    ("k_values", "--k-values"),
    # Without this a replay re-runs at eval_retrieval's CURRENT default rather
    # than the level the cell was scored at, quietly moving a strict cell to
    # turn (and writing it to a different file, leaving the stale one in place).
    # Absent from an old manifest -> no flag -> see _gold_of.
    ("gold", "--gold"),
    ("out", "--out"),
]


def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _qids(path: str, col: int = 0) -> set[str]:
    """Query ids in column ``col`` of a TSV, ignoring a header row."""
    out: set[str] = set()
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= col or not parts[col]:
                    continue
                if i == 0 and parts[col].lower() in ("query-id", "query_id", "qid"):
                    continue
                out.add(parts[col])
    except OSError:
        pass
    return out


def _gold_of(args: dict) -> str:
    """The gold level a recorded eval ran at.

    A manifest written before gold levels existed has no key, and those runs were
    all scored against the stamped flag — i.e. STAMPED_GOLD_LEVEL. Defaulting to
    eval_retrieval's CURRENT default instead would silently reinterpret every
    historical cell as something it was not.
    """
    return args.get("gold") or STAMPED_GOLD_LEVEL


def scoreable_queries(qrels_dir: str, retrieval_dir: str,
                      gold: str = STAMPED_GOLD_LEVEL) -> tuple[int, int]:
    """(judged queries with a run entry, samples holding a run.tsv), from disk.

    The first number is what ``evaluate`` puts in ``overall.queries``: it scores
    ``{qid: docs for qid in qrels if qid in run}``, so a query with judgements
    and no ranking is not counted — which is precisely why a cell that gained
    samples has a metrics file whose count is too low.
    """
    scored = samples = 0
    try:
        names = sorted(os.listdir(qrels_dir))
    except OSError:
        return (0, 0)
    for sample_id in names:
        qrels_path = os.path.join(qrels_dir, sample_id, "qrels", gold_filename(gold))
        if not os.path.isfile(qrels_path):
            continue
        run_path = os.path.join(retrieval_dir, sample_id, "run.tsv")
        if not os.path.isfile(run_path):
            continue
        samples += 1
        scored += len(_qids(qrels_path) & _qids(run_path))
    return (scored, samples)


def newest_run_mtime(retrieval_dir: str) -> float:
    newest = 0.0
    for entry in os.scandir(retrieval_dir):
        if not entry.is_dir():
            continue
        p = os.path.join(entry.path, "run.tsv")
        try:
            newest = max(newest, os.stat(p).st_mtime)
        except OSError:
            continue
    return newest


def find_manifests(root: str) -> list[str]:
    """Every retrieval cell under ``root``, found by its manifest.json."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "manifest.json" in filenames:
            out.append(dirpath)
            dirnames[:] = []      # a cell; its children are samples
    return sorted(out)


def _default_qrels_dir(a: dict) -> str | None:
    """The qrels tree eval_retrieval would have resolved for these args.

    Mirrors eval_retrieval.py:156-157 exactly — same ExperimentPaths, same
    labeler_segment — so a replay scores against the tree the original run used.
    Returns None only when the manifest lacks the keys the path is built from.
    """
    if not a.get("system") or not a.get("dataset"):
        return None
    paths = ExperimentPaths(system=a["system"], dataset=a["dataset"],
                            mem_model=a.get("mem_model"),
                            mem_variant=a.get("mem_variant"))
    return paths.qrels(labeler_segment(a.get("labeler_model") or "gpt-4o",
                                       a.get("candidate_source") or "sweep"))


def inspect(cell: str) -> dict:
    """Decide whether ``cell``'s metrics still describe the rankings under it."""
    man = read_json(os.path.join(cell, "manifest.json")) or {}
    rep = {"cell": cell, "args": man.get("args") or {}, "stage": man.get("stage"),
           "reasons": [], "status": "OK", "out": None}

    if man.get("stage") != "eval_retrieval":
        # The retrieve stage writes a manifest too; the eval stage overwrites it
        # when it runs. A retrieve-stage manifest therefore means this cell was
        # never scored, and there is no invocation here to replay.
        rep["status"] = "NEVER-SCORED"
        rep["reasons"].append(f"manifest is from stage {man.get('stage')!r}, not "
                              f"eval_retrieval — this cell has no eval run to replay")
        return rep

    a = rep["args"]
    qrels_dir = a.get("qrels_dir")
    # THE CELL IS WHERE THE RANKINGS ARE. A manifest records paths as they were
    # typed when eval last ran, and renaming a cell directory moves it
    # in all three trees WITHOUT rewriting them — so a renamed cell's
    # manifest names a directory that no longer exists. Trusting it made this
    # tool score `native/` while standing inside
    # `native-emb-<M>--naive/`: 0 queries scoreable, MISSING metrics (they were
    # moved to the new cell's _eval path, where check_eval reads them), and a
    # replay that would have written the fresh score back to the OLD _eval path
    # — a second metrics file nothing reads, beside the stale one it was called
    # to fix. It only failed loudly because eval_retrieval refused the vanished
    # run dir.
    #
    # The directory the manifest SITS IN cannot be wrong: eval_retrieval writes
    # it into the run dir it scored. So the cell wins, the recorded path is kept
    # only to rebase --out by, and the replay rewrites the manifest correctly.
    recorded_dir = a.get("retrieval_dir")
    retrieval_dir = cell
    renamed_from = None
    if recorded_dir and os.path.normpath(recorded_dir) != os.path.normpath(cell):
        renamed_from = os.path.basename(recorded_dir.rstrip("/"))
        rep["renamed_from"] = recorded_dir
        rep["reasons"].append(f"manifest records --retrieval-dir {recorded_dir}, which is "
                              f"not this cell — renamed since (a rename "
                              f"moves the directory, not the record). Replaying against "
                              f"this cell, and rebasing --out onto its name.")
    # A null "out" is not a broken manifest — it is the record of an eval run
    # BEFORE the stage stamped its resolved --out (and before the default moved
    # to the keyed _eval tree). For those runs the default was
    # <retrieval-dir>/metrics<suffix>.json, so that is what a null resolves to
    # here: the era is the fact the null carries. Resolving it at TODAY's
    # default instead would report every historical cell as MISSING while its
    # metrics sat in the run dir, unread.
    #
    # Newer manifests always carry the resolved path, so this branch is a
    # compatibility rule with a fixed set of members, not a convention still
    # being written to. command_for passes the path resolved here back as an
    # explicit --out, so a replayed cell rewrites the file it already publishes
    # rather than starting a second one at today's default.
    # Same filename eval_retrieval would compute: metrics.json at strict, and
    # metrics-gold-<level>.json otherwise. Hardcoding "metrics.json" reported
    # every non-strict cell as MISSING.
    out = a.get("out")
    # An --out under the OLD condition name is the same rename, one tree over:
    # the _eval cell was moved with it, so the file to rewrite is the one under
    # this cell's name. Rebased only when the recorded path's condition segment
    # IS the old name — an --out pointing anywhere else was aimed there on
    # purpose and is left alone.
    if out and renamed_from and os.path.basename(os.path.dirname(out)) == renamed_from:
        out = os.path.join(os.path.dirname(os.path.dirname(out)),
                           os.path.basename(cell.rstrip("/")), os.path.basename(out))
    out = out or os.path.join(retrieval_dir, f"metrics{gold_suffix(_gold_of(a))}.json")
    rep["out"] = out
    rep["retrieval_dir"] = retrieval_dir
    rep["default_out"] = not a.get("out")
    if not qrels_dir:
        # A null qrels_dir is not a missing argument — it is the record of a run
        # that did not PASS --qrels-dir, so eval_retrieval resolved the default
        # tree (eval_retrieval.py:157, `args.qrels_dir or paths.qrels(seg)`).
        # Resolve it the same way here. Treating null as unscoreable made this
        # tool refuse the entire sweep grid, which is exactly the set of cells it
        # exists to re-score: the drivers never pass --qrels-dir, so every cell
        # they write records null. Found when the lightmem provenance fix
        # changed the labels under all 9 of its cells and this
        # tool reported NO-ARGS for 8 of them.
        qrels_dir = _default_qrels_dir(a)
        rep["qrels_dir_default"] = qrels_dir
        if not qrels_dir:
            rep["status"] = "NO-ARGS"
            rep["reasons"].append("manifest records no --qrels-dir and the default "
                                  "cannot be rebuilt from it (no system/dataset)")
            return rep
    if not os.path.isdir(qrels_dir):
        # The manifest records paths as typed; a moved tree breaks the replay
        # before it breaks anything else, and saying so beats a stack trace.
        rep["status"] = "PATH-GONE"
        rep["reasons"].append(f"qrels dir in the manifest does not exist: {qrels_dir}")
        return rep

    scored, samples = scoreable_queries(qrels_dir, retrieval_dir, _gold_of(a))
    rep["scoreable"], rep["samples"] = scored, samples
    metrics = read_json(out)
    if metrics is None:
        rep["status"] = "MISSING"
        rep["reasons"].append(f"no metrics file at {out} ({scored} query/queries scoreable)")
        return rep

    had = int(((metrics.get("overall") or {}).get("queries")) or 0)
    rep["had"] = had
    if had != scored:
        rep["status"] = "STALE"
        rep["reasons"].append(f"metrics cover {had} query/queries; {scored} are now "
                              f"scoreable across {samples} sample(s)")
    else:
        newest = newest_run_mtime(retrieval_dir)
        try:
            m_time = os.stat(out).st_mtime
        except OSError:
            m_time = 0.0
        if newest > m_time + 1:
            # Weaker on purpose — see the module docstring.
            rep["status"] = "MAYBE"
            rep["reasons"].append("same query count, but a run.tsv is newer than "
                                  "metrics.json — re-ranked rather than extended? "
                                  "(mtimes do not survive every copy; treat as a hint)")
    return rep


def command_for(rep: dict) -> list[str]:
    """The eval_retrieval invocation this cell last used, as argv."""
    a = rep["args"]
    cmd = [sys.executable, "-m", "evalgrid.pipeline.eval_retrieval"]
    for key, flag in _FLAGS:
        val = a.get(key)
        if key == "retrieval_dir":
            # The cell inspect() actually read, which is the manifest's path
            # unless this cell was renamed under it.
            val = rep.get("retrieval_dir") or val
        if key == "out":
            # A manifest with no --out is one written before the stage stamped
            # its resolved path, and back then the default was the RUN DIR.
            # Omitting the flag would replay at today's default — the keyed
            # _eval cell — writing the fresh score somewhere else and leaving
            # the stale file this tool was called to fix exactly where it was.
            # inspect() already resolved which file that is — and rebased it if
            # the cell was renamed; pass that.
            val = rep.get("out") or val
        if key == "qrels_dir" and not val:
            # Same reasoning: the run did not pass --qrels-dir, so it scored
            # against the default tree. Pass the path inspect() resolved rather
            # than relying on the default still being the same one — the replay
            # then names the tree it scored against, which is what the manifest
            # it rewrites will record.
            val = rep.get("qrels_dir_default")
        if val in (None, ""):
            continue
        cmd += [flag, str(val)]
    if a.get("allow_partial"):
        cmd.append("--allow-partial")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-score eval cells whose metrics no longer match their rankings.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="retrieval root(s) to scan, e.g. outputs/_retrieval")
    ap.add_argument("--run", action="store_true",
                    help="actually re-score (default: report only)")
    ap.add_argument("--all", action="store_true",
                    help="re-score every cell, not only the stale ones")
    ap.add_argument("--include-maybe", action="store_true",
                    help="also re-score cells flagged only by the mtime hint")
    ap.add_argument("--print-commands", action="store_true",
                    help="print the replayed command for each selected cell")
    ap.add_argument("-v", "--verbose", action="store_true", help="show every cell")
    args = ap.parse_args()

    cells: list[str] = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[warn] not a directory: {root}", file=sys.stderr)
            continue
        cells.extend(find_manifests(root))

    reports = [inspect(c) for c in sorted(set(cells))]
    order = {"NO-ARGS": 0, "PATH-GONE": 1, "NEVER-SCORED": 2, "MISSING": 3,
             "STALE": 4, "MAYBE": 5, "OK": 6}
    print(f"{len(reports)} eval cell(s)\n")
    for rep in sorted(reports, key=lambda r: (order[r["status"]], r["cell"])):
        if rep["status"] == "OK" and not args.verbose:
            continue
        print(f"  {rep['status']:<13} {rep['cell']}")
        for r in rep["reasons"]:
            print(f"      {r}")

    selectable = {"STALE", "MISSING"} | ({"MAYBE"} if args.include_maybe else set())
    if args.all:
        selectable |= {"OK", "MAYBE"}
    # NEVER-SCORED / NO-ARGS / PATH-GONE are never selected: there is no
    # invocation to replay, and inventing one would be inventing a measurement.
    todo = [r for r in reports if r["status"] in selectable]

    counts: dict[str, int] = {}
    for rep in reports:
        counts[rep["status"]] = counts.get(rep["status"], 0) + 1
    print("\n" + "=" * 71)
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: order[kv[0]])))

    if args.print_commands:
        for rep in todo:
            print("  " + " ".join(command_for(rep)))

    if not todo:
        print("Nothing to re-score.")
        return 0
    if not args.run:
        print(f"{len(todo)} cell(s) would be re-scored — pass --run to do it "
              f"(eval_retrieval truncates its --out, so this overwrites their metrics.json)")
        return 0

    failed = 0
    for i, rep in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {rep['cell']}")
        proc = subprocess.run(command_for(rep), cwd=str(REPO))
        if proc.returncode != 0:
            failed += 1
            # The completeness gate lives in eval_retrieval and is the likeliest
            # non-zero exit here, so name it rather than leaving a bare code.
            print(f"    !! exit {proc.returncode} — if it refused the run dir as "
                  f"incomplete, fix the retrieval cell rather than passing "
                  f"--allow-partial blindly")
    print("\n" + "=" * 71)
    print(f"re-scored {len(todo) - failed}/{len(todo)} cell(s)"
          + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
