#!/usr/bin/env python3
"""
check_judge.py
==============
Report which answer cells the project judge has actually SCORED, and whether the
score sitting beside each one still covers the predictions under it — without
re-judging anything.

Why a separate tool
-------------------
``check_grid.py`` ends at answering: it says whether every ``predictions.jsonl``
holds every question it owes, and its own docstring ends "so it can gate a judge
run in a script". Nothing then checked the judge run itself, which is the last
stage before a number reaches a table — and it is the stage with the quietest
failure mode of them all:

  A judge call can fail (a 429, a 500, a content filter, a reply truncated at
  ``--judge-max-tokens``). Those items get NO VERDICT and are excluded from both
  the numerator and the denominator rather than counted wrong — see
  ``benchmarks/judges/base.py``, and the reasoning there is right. But it means
  an accuracy computed over 487 of 500 items is a normal-looking float in a
  normal-looking file, and NOTHING in the directory name says so. The one
  machine-readable tell is ``"complete": false`` inside the metrics file.

So this reads back the three facts that decide whether a cell's accuracy may be
quoted: the metrics file exists for THIS judge, every prediction under it has a
verdict, and no judged item was silently dropped.

What "scored" means, per cell
-----------------------------
The unit is the answer cell — a directory holding a ``predictions.jsonl`` — and
the expected question set is that file's OWN question-ids, which is exactly what
``pipeline/judge.py`` is handed. (Whether the predictions themselves cover the
grid is check_grid's question, asked one stage earlier; asking it again here
would report every answering gap twice.)

  SCORED    ``metrics.<judge-slug>.json`` exists and every prediction has a verdict
  PARTIAL   it exists, and some predictions do not have one. Two different
            causes, reported separately because they need different repairs:
              * NO VERDICT — the judge was asked and did not answer
                (``per_question`` rows with ``judged: false``). Fixed by
                ``--retry-failed``, which re-judges only those items; the driver
                does this by default (``RETRY_FAILED=1``).
              * NOT JUDGED — the id is not in the metrics file at all, i.e. the
                predictions grew after the scoring pass. Fixed by re-judging the
                cell (``FORCE=1``).
  UNSCORED  no ``metrics.<judge-slug>.json`` — the judge never ran here

Markers reported beside the state:

  ``_batch_judge_<slug>/`` with no metrics
        a Batch-API job was SUBMITTED and never collected. The queue holds it
        for up to 24h and nothing on disk expires; the fix is
        ``--batch-phase collect``, not a re-submit.
  ``judge_failures.<slug>.jsonl``
        the machine-readable half of the failures above. Written truncated (not
        appended) by each pass, so unlike retrieval's errors.jsonl its presence
        is current rather than historical.
  ``unmatched_predictions > 0``
        predictions whose question_id is not in the dataset at all — those rows
        were dropped before judging, and no re-run recovers them.
  a metrics file from ANOTHER judge
        reported as a note, never as a pass. The project judge is held fixed
        across every system, retriever and budget cell (a judge that varies
        between rows is a second independent variable in a one-variable table),
        so a cell scored only by a sensitivity judge is an UNSCORED cell that
        happens to have a number in it.

Which cells are expected
------------------------
Every ``predictions.jsonl`` under the system's answers root, at any depth —
the same ``find ... -name predictions.jsonl`` ``run_all_judge.sh`` dispatches
over, and the same reason it is depth-agnostic: the keyed tree inserts
``<dataset>/<mem-seg>/<retriever>/<gen-seg>`` above the cell and an ``OUT_ROOT``
smoke tree inserts nothing. It therefore covers the NATIVE grid
(``native-<retriever>-top<k>--grid/``) with no extra knowledge, exactly as the
driver does.

The driver's one filter is mirrored: oracle answers are keyed by labeler
segment, so only THIS experiment's oracle sets are expected. Other labelers'
(and other gold levels') oracle cells are counted and named, never silently
dropped.

Usage
-----
    python -m evalgrid.tools.check_judge                     # every system, keyed tree
    python -m evalgrid.tools.check_judge --systems lightmem -v
    python -m evalgrid.tools.check_judge --out-root outputs/_smoke4
    python -m evalgrid.tools.check_judge --judge-model gpt-4o   # a sensitivity run
    python -m evalgrid.tools.check_eval --spec-name beam_100k.yaml

Exit code is 0 only when every cell is SCORED by the named judge, so it can gate
a results-table build in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# evalgrid/tools/ -> repo root; make `evalgrid` importable when run as a file.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.lib.paths import model_slug  # noqa: E402
# The tree layout, the spec reader and the state vocabulary are check_grid's;
# imported rather than restated so a layout change lands in one file. See
# our development notes for what a second copy of one of these costs.
from evalgrid.tools.check_grid import Tree, load_spec  # noqa: E402

#: run_all_judge.sh's fallback when a spec names no judge_model. The project
#: judge is gpt-4o-mini and is held FIXED across systems; the flag exists for a
#: labelled sensitivity check only (evalgrid/pipeline/judge.py).
DEFAULT_JUDGE = "gpt-4o-mini"

SCORED, PARTIAL, UNSCORED = "SCORED", "PARTIAL", "UNSCORED"
ORDER = {UNSCORED: 0, PARTIAL: 1, SCORED: 2}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def prediction_ids(path: Path) -> tuple[set[str], int]:
    """(question-ids, unreadable line count) from a predictions.jsonl.

    Tolerant on purpose, unlike check_grid's strict reader: a cell killed
    mid-write leaves a truncated last line, and a checker that dies on it
    reports nothing about the other 500 cells. The count is surfaced instead.
    """
    ids: set[str] = set()
    bad = 0
    try:
        with path.open(errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                qid = row.get("question_id")
                if qid:
                    ids.add(qid)
    except OSError:
        pass
    return ids, bad


def oracle_labeler(parts: tuple[str, ...]) -> str | None:
    """The labeler segment an oracle cell is keyed by, or None if not oracle.

    Found by position after the "oracle" component rather than at a fixed depth,
    for the same reason the search itself is depth-agnostic — the keyed tree and
    a smoke tree put "oracle" at different levels.
    """
    for i, part in enumerate(parts):
        if part == "oracle" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def expected_cells(tree: Tree) -> tuple[list[Path], list[str], list[str]]:
    """(predictions this experiment owes a score for, other-labeler cells, other-dataset cells).

    Mirrors run_all_judge.sh's gather loop, including its one filter: an oracle
    set under a DIFFERENT labeler segment belongs to another experiment that
    shares this mem-model root. The gold-level suffix
    (``<labeler-seg>-gold-<level>``) is admitted, because strict and turn are
    two rows of one sensitivity check rather than two experiments — the shell
    grew that second pattern after the first one silently dropped every oracle
    row from the table.

    The DATASET filter is the same one ``Tree.predictions`` applies, and for the
    same reason: ``_answers/<system>/`` holds every benchmark this system was
    ever run on, so an unscoped rglob judged BEAM's cells under a LongMemEval
    spec and reported one total over both. They are not skipped silently —
    another benchmark's cells are counted and named under -v, exactly like the
    other-labeler ones.
    """
    base = tree.stage_root("answers")
    if not base.is_dir():
        return [], [], []
    want = tree.labeler_seg
    cells, skipped, off_dataset = [], [], []
    for pred in sorted(base.rglob("predictions.jsonl")):
        parts = pred.relative_to(base).parts
        if not tree.out_root and tree.dataset not in parts:
            off_dataset.append(str(pred.parent.relative_to(base)))
            continue
        seg = oracle_labeler(parts)
        if seg is not None and seg != want and not seg.startswith(f"{want}-gold-"):
            skipped.append(str(pred.parent.relative_to(base)))
            continue
        cells.append(pred)
    return cells, skipped, off_dataset


def check_cell(pred: Path, slug: str) -> dict:
    """One answer cell against the verdict file the named judge should have left."""
    cell = pred.parent
    metrics_path = cell / f"metrics.{slug}.json"
    pred_ids, bad_lines = prediction_ids(pred)
    rep = {"cell": cell, "state": UNSCORED, "markers": [], "notes": [],
           "no_verdict": [], "not_judged": [], "accuracy": None, "n_pred": len(pred_ids)}

    if bad_lines:
        rep["markers"].append(f"{bad_lines} unparsable line(s) in predictions.jsonl "
                              f"(a write that was cut short)")

    # Other judges' files are a note wherever they appear: they never satisfy
    # this check, and their absence is not a problem either.
    others = sorted(p.name.split(".", 2)[1] for p in cell.glob("metrics.*.json")
                    if p.name != f"metrics.{slug}.json" and p.name.count(".") >= 2)
    if others:
        rep["notes"].append(f"also scored by {', '.join(others)} — a sensitivity "
                            f"judge's numbers are not this table's")

    metrics = read_json(metrics_path)
    if metrics is None:
        if metrics_path.exists():
            rep["markers"].append(f"{metrics_path.name} is unreadable (half-written?)")
        state_dir = cell / f"_batch_judge_{slug}"
        if (state_dir / "judge_state.json").is_file():
            rep["markers"].append("a Batch job was submitted and never collected "
                                  "(--batch-phase collect finishes it; do NOT re-submit)")
        rep["not_judged"] = sorted(pred_ids)
        return rep

    per_q = metrics.get("per_question") or []
    judged = {r.get("question_id") for r in per_q if r.get("judged")}
    # judged is false for an item the judge was asked about and did not answer —
    # the same predicate --retry-failed selects on, so a cell this reports as
    # recoverable is exactly one that pass will finish.
    no_verdict = {r.get("question_id") for r in per_q if not r.get("judged")}
    rep["no_verdict"] = sorted(i for i in no_verdict if i in pred_ids)
    rep["not_judged"] = sorted(pred_ids - judged - no_verdict)

    ov = metrics.get("overall") or {}
    if ov.get("total"):
        rep["accuracy"] = ov.get("accuracy")
    rep["total"], rep["attempted"] = ov.get("total"), ov.get("attempted")

    if rep["no_verdict"]:
        rep["markers"].append(
            f"{len(rep['no_verdict'])} item(s) got NO VERDICT and are excluded from the "
            f"accuracy (denominator {ov.get('total')}, not {ov.get('attempted')}) — "
            f"finish them with RETRY_FAILED=1 / --retry-failed")
    if rep["not_judged"]:
        # Excluded types and unmatched ids are the two legitimate ways a
        # prediction can be absent from the metrics file. Both are recorded IN
        # it, so say which of the gap they could account for rather than
        # reporting a number that is partly explained.
        why = []
        if metrics.get("excluded_question_types"):
            why.append("--exclude-question-type "
                       f"{','.join(metrics['excluded_question_types'])} was in force")
        n_unmatched = int(metrics.get("unmatched_predictions") or 0)
        if n_unmatched:
            why.append(f"{n_unmatched} prediction(s) matched no dataset question")
        tail = f" ({'; '.join(why)} — some of these are that)" if why else ""
        rep["markers"].append(
            f"{len(rep['not_judged'])} prediction(s) are in no verdict at all: the "
            f"predictions grew after this file was written{tail} — re-judge with FORCE=1")

    if (cell / f"judge_failures.{slug}.jsonl").is_file():
        rep["notes"].append(f"judge_failures.{slug}.jsonl — the failures above, "
                            f"per item (rewritten by each pass, so it is current)")
    if metrics.get("complete") is False and not rep["no_verdict"]:
        # The file says incomplete and its per_question rows do not — trust the
        # flag (it is what the driver's own skip-check reads) and say both.
        rep["markers"].append('metrics say "complete": false but every prediction has a '
                              'verdict — the file was written against a different set')
    if metrics.get("judge_model") and model_slug(metrics["judge_model"]) != slug:
        rep["markers"].append(f"file is named for {slug} but records "
                              f"judge_model={metrics['judge_model']!r}")

    rep["state"] = SCORED if not (rep["no_verdict"] or rep["not_judged"]) else PARTIAL
    return rep


def check_system(system: str, spec: dict, args) -> dict:
    tree = Tree(args.out_root, spec, system, args.mem_model, args.mem_variant)
    judge = args.judge_model or spec.get("judge_model") or DEFAULT_JUDGE
    slug = model_slug(judge)
    base = tree.stage_root("answers")
    cells, skipped, off_dataset = expected_cells(tree)
    reps = [check_cell(p, slug) for p in cells]
    return {"system": system, "judge": judge, "base": base, "reps": reps,
            "skipped": skipped, "off_dataset": off_dataset, "dataset": tree.dataset}


def print_system(rep: dict, args) -> None:
    base, reps = rep["base"], rep["reps"]
    print(f"\n=== {rep['system']}   judge: {rep['judge']}")
    if not reps:
        extra = []
        if rep["skipped"]:
            extra.append(f"{len(rep['skipped'])} oracle cell(s) belong to another labeler")
        if rep["off_dataset"]:
            extra.append(f"{len(rep['off_dataset'])} cell(s) belong to another dataset")
        print(f"  no predictions.jsonl for {rep['dataset']!r} under {base}"
              + (f" ({'; '.join(extra)})" if extra else ""))
        return

    for r in sorted(reps, key=lambda r: (ORDER[r["state"]], str(r["cell"]))):
        if r["state"] == SCORED and not (r["markers"] or (args.verbose and r["notes"])) \
                and not args.verbose:
            continue
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "  -   "
        rel = r["cell"].relative_to(base) if r["cell"].is_relative_to(base) else r["cell"]
        print(f"  {r['state']:<9} {acc}  {rel}")
        for m in r["markers"]:
            print(f"      !! {m}")
        for n in r["notes"]:
            print(f"      -- {n}")
        if args.verbose:
            for label, key in (("no verdict", "no_verdict"), ("not judged", "not_judged")):
                if r[key]:
                    print(f"         {label}: {', '.join(r[key])}")

    n = {s: sum(1 for r in reps if r["state"] == s) for s in (SCORED, PARTIAL, UNSCORED)}
    line = (f"  {len(reps)} cell(s): {n[SCORED]} scored, {n[PARTIAL]} partial, "
            f"{n[UNSCORED]} unscored")
    if rep["skipped"]:
        line += f"   (+{len(rep['skipped'])} oracle cell(s) under another labeler, skipped)"
    if rep["off_dataset"]:
        line += (f"   (+{len(rep['off_dataset'])} cell(s) under another dataset, "
                 f"not {rep['dataset']!r}, skipped)")
    print(line)
    if args.verbose:
        for s in rep["skipped"]:
            print(f"      not this experiment's: {s}")
        for s in rep["off_dataset"]:
            print(f"      not this dataset's: {s}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Has the project judge scored every answer cell, completely?",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="",
                   help="OUT_ROOT the answer stage was run with, e.g. outputs/_smoke4 "
                        "(reads <root>_answers). Empty = the keyed default tree.")
    p.add_argument("--systems", default="",
                   help="comma-separated; default is every experiments/<sys>/<spec-name>")
    # check_grid's knob, same spelling and default: the dataset comes from the
    # spec, so this tool cannot read one benchmark's tree under another's spec.
    p.add_argument("--spec-name", default="longmemeval.yaml",
                   help="which per-system spec to read, e.g. beam_100k.yaml "
                        "(default: longmemeval.yaml). The dataset comes from the spec.")
    p.add_argument("--judge-model", default=None,
                   help=f"judge whose verdicts to check (default: each spec's judge_model, "
                        f"else {DEFAULT_JUDGE}). Naming a sensitivity judge checks THAT "
                        f"judge's coverage, not the project one's.")
    p.add_argument("--mem-model", default=None,
                   help="override the spec's mem_model when locating the keyed tree")
    p.add_argument("--mem-variant", default=None,
                   help="the --mem-variant the store was ingested under, if any")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show every cell (default: only those needing attention) and "
                        "name every unjudged question-id")
    args = p.parse_args()

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

    total = sum(len(r["reps"]) for r in reports)
    bad = sum(1 for r in reports for c in r["reps"] if c["state"] != SCORED)
    marked = sum(1 for r in reports for c in r["reps"]
                 if c["state"] == SCORED and c["markers"])
    print("\n" + "=" * 72)
    for r in reports:
        n_bad = sum(1 for c in r["reps"] if c["state"] != SCORED)
        if not r["reps"]:
            verdict = "no answers to judge"
        elif n_bad:
            verdict = f"{n_bad} of {len(r['reps'])} cell(s) not fully scored"
        else:
            verdict = f"all {len(r['reps'])} cell(s) scored"
        print(f"  {r['system']:<12} {verdict}")
    print("=" * 72)
    if not total:
        print("No predictions.jsonl anywhere under "
              f"{args.out_root or 'the keyed tree'} — run the answer stage first.")
        return 2
    if bad:
        print(f"{bad} of {total} cell(s) not fully scored. Re-run "
              f"experiments/run_all_judge.sh — it skips cells already complete, "
              f"and RETRY_FAILED=1 (its default) finishes the unjudged items.")
        return 1
    print(f"Every one of the {total} cell(s) is fully scored."
          + (f" ({marked} cell(s) carry a marker — read them before quoting.)" if marked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
