#!/usr/bin/env python3
"""
check_eval.py
=============
Report which retrieval conditions have RETRIEVAL METRICS at the gold level the
analysis reads, where those metrics were written, and whether they still
describe the rankings sitting under them — without re-scoring anything.

Where it sits between the tools that already exist
--------------------------------------------------
``check_grid.py`` answers "is the ranking there" (per-sample run.tsv) and stops.
``rescore_eval.py`` answers "does this cell's metrics.json still match its
rankings, and what command would refresh it" — but it walks the RETRIEVAL tree
looking for ``manifest.json``, so it can only see cells an eval run already
touched, it reports the level each cell HAPPENS to be scored at, and it never
looks at ``outputs/_eval/`` at all. Three holes follow from that, and this tool
is those three:

  * a condition that was retrieved and never scored is a ``NEVER-SCORED``
    footnote there, not a missing cell in a grid;
  * a cell scored only at ``strict`` reads as OK even though every table is
    built at ``turn`` (``benchmarks/gold.EVAL_GOLD_LEVEL``) — the metrics file
    that the analysis wants does not exist, and nothing said so;
  * a score can be in either of two trees. ``eval_retrieval`` now defaults to
    the keyed ``_eval`` cell, but it still writes beside the run when
    ``--retrieval-dir`` names one, and every cell scored before that default
    moved is sitting in a retrieval dir today — invisible to the analysis scripts,
    which reads the ``_eval`` tree.

The staleness rule is NOT restated here: ``rescore_eval.scoreable_queries`` is
imported and called, so "how many queries this cell can be scored over" has one
implementation, and a cell this tool calls STALE is exactly one that tool will
select. Repairing them is still its job — this one never writes.

What each state means, per (condition x gold level)
---------------------------------------------------
  SCORED    a metrics file exists and covers every query scoreable today
  STALE     it exists and covers FEWER — the usual cause is a resumed retrieve
            pass that added samples after the scoring run, leaving an nDCG
            averaged over a subset with nothing in the filename saying so
  PRESENT   it exists, but completeness cannot be checked from this machine —
            the qrels tree is not here, or that gold level was never built into
            it. Reported, never counted as finished (same vocabulary as
            check_retrieval / check_answers)
  UNSCORED  no metrics file at either convention's path

Conditions are taken from the retrieval tree AS FOUND, not from a preset: a
condition that was never retrieved cannot be scored, and reporting it here would
report one hole twice (check_grid owns that half). The discovery is check_grid's
``Tree``, so the NATIVE grid — ``native-<retriever>-top<k>--grid/`` — and an
``OUT_ROOT`` smoke tree are covered without either being named here.

Usage
-----
    python -m evalgrid.tools.check_eval                    # every system, gold=turn
    python -m evalgrid.tools.check_eval --gold strict,turn -v
    python -m evalgrid.tools.check_eval --systems simplemem --out-root outputs/_smoke4
    python -m evalgrid.tools.check_eval --spec-name beam_100k.yaml

Exit code is 0 only when every condition is SCORED at every requested level, so
it can gate a results-table build in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# evalgrid/tools/ -> repo root; make `evalgrid` importable when run as a file.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.benchmarks.gold import (EVAL_GOLD_LEVEL, GOLD_LEVELS,  # noqa: E402
                                      gold_filename, gold_suffix)
from evalgrid.lib.paths import mem_segment  # noqa: E402
# The tree layout is check_grid's and the staleness rule is rescore_eval's;
# both imported rather than restated.
from evalgrid.tools.check_grid import Tree, load_spec  # noqa: E402
from evalgrid.tools.rescore_eval import scoreable_queries  # noqa: E402

SCORED, STALE, PRESENT, UNSCORED = "SCORED", "STALE", "PRESENT", "UNSCORED"
ORDER = {UNSCORED: 0, STALE: 1, PRESENT: 2, SCORED: 3}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def eval_tree_index(base: Path, level: str,
                    dataset: str | None = None) -> dict[str, list[Path]]:
    """{condition-dirname: [metrics files]} under an ``_eval`` root, any depth.

    Indexed by the file's own parent directory, which is the condition in BOTH
    layouts this tree is written in: the keyed one (``paths.eval_dir``, four
    levels above the condition) and the flat one ``scripts/query.sh``
    writes (``<OUT_ROOT>_eval/<system>/<cond>/``). Globbing at a fixed depth
    would find one and silently miss the other.

    ``dataset`` scopes the hits to one benchmark, and the keyed layout MUST pass
    it — the same rule ``Tree._pick`` states one level up: the condition name is
    the only key this index has, and it is identical under ``beam_100k/`` and
    ``longmemeval_s/``. Unscoped, both files land in one list, the caller's
    mem-segment tie-break cannot separate them (that segment is the same string
    under both benchmarks), and ``sorted()`` order decides — which means
    ``beam_100k`` won every tie, and every LongMemEval cell was read against
    BEAM's query count and reported STALE. An OUT_ROOT tree has no dataset
    level; it passes None and is left alone.
    """
    out: dict[str, list[Path]] = {}
    if not base.is_dir():
        return out
    for hit in sorted(base.rglob(f"metrics{gold_suffix(level)}.json")):
        if dataset and dataset not in hit.parts:
            continue
        out.setdefault(hit.parent.name, []).append(hit)
    return out


def gold_built(qrels: Path | None, level: str) -> int:
    """How many qrels samples carry a gold file at ``level``.

    Zero means the level was never built here, which is a different fact from
    "this cell scores zero queries" — and the reason a missing level is reported
    as PRESENT rather than as a stale cell whose count fell to nothing.
    """
    if qrels is None or not qrels.is_dir():
        return 0
    return sum(1 for d in qrels.iterdir()
               if (d / "qrels" / gold_filename(level)).is_file())


def check_cell(cond_dir: Path, qrels: Path | None, level: str,
               in_tree: list[Path], level_built: bool) -> dict:
    """One condition at one gold level, against every metrics file that could be it."""
    default_out = cond_dir / f"metrics{gold_suffix(level)}.json"
    rep = {"cond": cond_dir.name, "state": UNSCORED, "markers": [], "notes": [],
           "path": None, "had": None, "scoreable": None, "samples": None}

    runs = sum(1 for _ in cond_dir.glob("*/run.tsv"))
    rep["runs"] = runs
    if not runs:
        # Nothing to score. Not this tool's finding — check_grid reports the
        # missing rankings, and calling it an unscored cell here would count one
        # hole twice.
        rep["notes"].append("no run.tsv under this condition — nothing to score "
                            "(check_grid owns the missing rankings)")
        return rep

    # The _eval tree first: it is what the analysis reads, so it is the copy whose
    # absence actually costs a row in a table.
    found = list(in_tree)
    if default_out.is_file():
        found.append(default_out)
    if not found:
        return rep
    rep["path"] = found[0]
    if not in_tree and default_out.is_file():
        rep["markers"].append(
            f"scored only into the retrieval dir ({default_out.name}); "
            f"the analysis reads the outputs/_eval tree, so this cell is invisible there")
    if len(in_tree) > 1:
        rep["markers"].append(f"{len(in_tree)} metrics files in the _eval tree name this "
                              f"condition ({', '.join(str(p) for p in in_tree)}) — the "
                              f"first is used; they may differ in mem-variant or "
                              f"labeler segment (the dataset is already scoped out)")

    metrics = read_json(rep["path"])
    if metrics is None:
        rep["markers"].append(f"{rep['path']} is unreadable (half-written?)")
        return rep

    had = int(((metrics.get("overall") or {}).get("queries")) or 0)
    rep["had"] = had
    # A cell scored twice into two conventions that DISAGREE is a real trap: the
    # table and the run dir then quote different numbers for one condition.
    counts = {p: int(((read_json(p) or {}).get("overall") or {}).get("queries") or 0)
              for p in found}
    if len(set(counts.values())) > 1:
        where = ", ".join(
            f"{'retrieval dir' if p == default_out else str(p)} {n} q"
            for p, n in counts.items())
        rep["markers"].append(f"copies of this cell's metrics disagree ({where}) — "
                              f"whatever reads the other copy quotes a different "
                              f"number; one of them is stale")
    # eval_retrieval stamps the level it ran at; files written before that key
    # existed have none, and their filename is then the only record (the same
    # assumption rescore_eval._gold_of makes).
    if metrics.get("gold") and metrics["gold"] != level:
        rep["markers"].append(f"filename says gold={level} but the file records "
                              f"gold={metrics['gold']!r}")

    if qrels is None or not qrels.is_dir():
        rep["state"] = PRESENT
        rep["notes"].append(f"{had} query/queries scored; no qrels tree here to check "
                            f"coverage against (run where the full outputs are)")
        return rep
    if not level_built:
        rep["state"] = PRESENT
        rep["notes"].append(f"{had} query/queries scored; no {gold_filename(level)} in the "
                            f"qrels here, so coverage cannot be checked (that level was "
                            f"never built on this machine)")
        return rep

    scored, samples = scoreable_queries(str(qrels), str(cond_dir), level)
    rep["scoreable"], rep["samples"] = scored, samples
    if had != scored:
        rep["state"] = STALE
        rep["markers"].append(f"metrics cover {had} query/queries; {scored} are scoreable "
                              f"today across {samples} sample(s) — an average over a "
                              f"subset, with nothing in the filename saying so")
    else:
        rep["state"] = SCORED
        if not had:
            rep["markers"].append("0 queries scored — every metric in this file is a "
                                  "default, not a measurement")
    return rep


def check_system(system: str, spec: dict, args) -> dict:
    tree = Tree(args.out_root, spec, system, args.mem_model, args.mem_variant)
    qrels, qnote = tree.qrels()
    rroot, rnote = tree.retrieval_root()
    notes = sorted({n for n in (qnote, rnote) if n})

    if rroot is None:
        return {"system": system, "rows": [], "notes": notes,
                "why": f"no retrieval runs under {tree.stage_root('retrieval')}"}

    want_key = mem_segment(tree.mem_model, tree.mem_variant)
    conds = sorted((d for d in rroot.iterdir() if d.is_dir()), key=lambda d: d.name)
    names = {d.name for d in conds}
    rows, orphans = [], set()
    for level in args.gold:
        index = eval_tree_index(tree.stage_root("eval"), level,
                                None if args.out_root else tree.dataset)
        built = gold_built(qrels, level) > 0
        # Metrics naming a condition that is not in the retrieval tree. Reported
        # rather than ignored: it is either a ranking that was never synced back
        # (the metrics are the only trace of a run that happened) or a condition
        # that has been renamed since — and a scored cell nothing points at is
        # exactly the kind of thing a checker must not drop silently.
        orphans |= {f"{name} ({level})" for name in index if name not in names}
        for cond in conds:
            hits = index.get(cond.name, [])
            # Prefer a hit under this experiment's mem key when the tree holds
            # several — the same preference Tree._pick applies one level up.
            hits = sorted(hits, key=lambda p: 0 if want_key in p.parts else 1)
            rep = check_cell(cond, qrels, level, hits, built)
            rep["level"] = level
            rows.append(rep)
    return {"system": system, "rows": rows, "notes": notes, "why": None,
            "orphans": sorted(orphans), "eval_root": tree.stage_root("eval")}


def print_system(rep: dict, args) -> None:
    print(f"\n=== {rep['system']}")
    for n in rep["notes"]:
        print(f"  !! {n}")
    if rep["why"]:
        print(f"  {rep['why']}")
        return
    rows = rep["rows"]
    if not rows:
        print("  (no retrieval conditions)")
        return

    width = max(len(r["cond"]) for r in rows) + 2
    for r in sorted(rows, key=lambda r: (ORDER[r["state"]], r["level"], r["cond"])):
        quiet = r["state"] == SCORED and not r["markers"]
        skipped = r["state"] == UNSCORED and not r["runs"]
        if (quiet or skipped) and not args.verbose:
            continue
        cov = (f"{r['had']}/{r['scoreable']} q" if r["scoreable"] is not None
               else (f"{r['had']} q" if r["had"] is not None else "-"))
        print(f"  {r['state']:<9} {r['level']:<7} {r['cond']:<{width}} {cov}")
        for m in r["markers"]:
            print(f"      !! {m}")
        for n in r["notes"]:
            print(f"      -- {n}")
        if args.verbose and r["path"]:
            print(f"         {r['path']}")

    counted = [r for r in rows if r["runs"]]
    n = {s: sum(1 for r in counted if r["state"] == s)
         for s in (SCORED, STALE, PRESENT, UNSCORED)}
    skipped = len(rows) - len(counted)
    print(f"  {len(counted)} cell(s): {n[SCORED]} scored, {n[STALE]} stale, "
          f"{n[PRESENT]} present-unverified, {n[UNSCORED]} unscored"
          + (f"   (+{skipped} with no rankings to score)" if skipped else ""))
    if rep["orphans"]:
        print(f"  -- {len(rep['orphans'])} metrics file(s) under {rep['eval_root']} name a "
              f"condition with no retrieval dir here: {', '.join(rep['orphans'])}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Is every retrieval condition scored, at the level the tables read?",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="",
                   help="OUT_ROOT the stages were run with, e.g. outputs/_smoke4 "
                        "(reads <root>_retrieval / _eval / _qrels). "
                        "Empty = the keyed default tree.")
    p.add_argument("--systems", default="",
                   help="comma-separated; default is every experiments/<sys>/<spec-name>")
    # Same knob, same spelling, same default as check_grid's: the DATASET comes
    # out of the spec, so a tool cannot be pointed at one benchmark's specs
    # while reading another's tree.
    p.add_argument("--spec-name", default="longmemeval.yaml",
                   help="which per-system spec to read, e.g. beam_100k.yaml "
                        "(default: longmemeval.yaml). The dataset comes from the spec.")
    p.add_argument("--gold", default=EVAL_GOLD_LEVEL,
                   help=f"comma-separated gold level(s) to expect a metrics file for "
                        f"(default: {EVAL_GOLD_LEVEL}, the level evaluation defaults to; "
                        f"choices: {', '.join(GOLD_LEVELS)})")
    p.add_argument("--mem-model", default=None,
                   help="override the spec's mem_model when locating the keyed tree")
    p.add_argument("--mem-variant", default=None,
                   help="the --mem-variant the store was ingested under, if any")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show every cell (default: only those needing attention) and "
                        "the metrics file each one was read from")
    args = p.parse_args()

    args.gold = [g.strip() for g in args.gold.split(",") if g.strip()]
    bad = [g for g in args.gold if g not in GOLD_LEVELS]
    if bad:
        print(f"ERROR: unknown gold level(s) {bad} (choices: {', '.join(GOLD_LEVELS)})",
              file=sys.stderr)
        return 2

    if args.systems:
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    else:
        systems = sorted(d.name for d in (REPO / "experiments").iterdir()
                         if (d / args.spec_name).is_file())

    reports = []
    for system in systems:
        spec_path = REPO / "experiments" / system / args.spec_name
        if not spec_path.is_file():
            print(f"!! no spec for {system!r}", file=sys.stderr)
            return 2
        rep = check_system(system, load_spec(spec_path), args)
        reports.append(rep)
        print_system(rep, args)

    counted = [r for rep in reports for r in rep["rows"] if r["runs"]]
    unfinished = [r for r in counted if r["state"] in (STALE, UNSCORED)]
    unverified = [r for r in counted if r["state"] == PRESENT]
    marked = [r for r in counted if r["state"] == SCORED and r["markers"]]

    print("\n" + "=" * 72)
    for rep in reports:
        mine = [r for r in rep["rows"] if r["runs"]]
        n_bad = sum(1 for r in mine if r["state"] in (STALE, UNSCORED))
        if rep["why"]:
            verdict = rep["why"]
        elif not mine:
            verdict = "no rankings to score"
        elif n_bad:
            verdict = f"{n_bad} of {len(mine)} cell(s) not scored at {','.join(args.gold)}"
        else:
            verdict = f"all {len(mine)} cell(s) scored"
        print(f"  {rep['system']:<12} {verdict}")
    print("=" * 72)
    if not counted:
        print("No retrieval condition anywhere under "
              f"{args.out_root or 'the keyed tree'} — run the retrieve stage first.")
        return 2
    if unfinished:
        n_stale = sum(1 for r in unfinished if r["state"] == STALE)
        print(f"{len(unfinished)} of {len(counted)} cell(s) not scored at "
              f"{','.join(args.gold)}.")
        if n_stale:
            print(f"  {n_stale} are STALE — `python -m evalgrid.tools.rescore_eval "
                  f"outputs/_retrieval --run` replays each cell's own last eval "
                  f"invocation, so a re-score is the same measurement, not a new one.")
        print("  The rest were never scored: run pipeline/eval_retrieval for them "
              "(stage 3 of scripts/query.sh).")
        return 1
    if unverified:
        print(f"ALL PRESENT ({len(unverified)} cell(s) unverified — run where the full outputs are "
              f"with the qrels tree to confirm coverage).")
        return 0
    print(f"Every one of the {len(counted)} cell(s) is scored and current."
          + (f" ({len(marked)} cell(s) carry a marker — read them.)" if marked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
