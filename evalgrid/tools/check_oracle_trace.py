#!/usr/bin/env python3
"""
check_oracle_trace.py
=====================
Assert that every oracle cell on disk still traces back to its store — that the
gold set the oracle answered from is the gold set the store, the dataset and
``evalgrid/benchmarks/gold.py`` produce TODAY.

WHY THIS EXISTS
---------------
An oracle number is a claim about evidence: "given the units this store built
from the evidence turns, the generator answers X". Three files have to agree for
that claim to hold — the store's provenance, the qrels built from it, and the
answers built from those qrels — and NOTHING checked that they still do. Each
stage is resumable and skips work already on disk, so a qrels tree outlives the
store state that produced it, and an answers tree outlives the qrels.

That is not hypothetical. Two ways it has already happened:

  * **a store lost a file.** lightmem's BEAM-100K qrels were built from a
    ``provenance.jsonl`` sidecar that was later deleted from all 20 sample dirs.
    The corpus was untouched, so ``check_corpus_matches`` passed; only the
    per-unit turn attribution moved, on 33 of 400 questions. The store could no
    longer regenerate its own gold, and nothing said so for three weeks.
  * **a gold level was redefined.** ``relabel_union`` moved from a ``strict``
    base to a ``turn`` base on 2026-09-19; the qrels were rebuilt two hours
    after the answers were generated, so every ``relabel_union`` oracle cell
    answers from a gold set narrower than the one beside it on disk.

Both are invisible to the existing checks. ``check_qrels`` counts gold rows but
never recomputes them; ``check_answers`` checks that questions were ANSWERED,
not what they were answered FROM. This closes a gap that was
still open: *nothing in the pipeline asserts that a store's provenance is the
provenance its qrels were built from.*

WHAT IT CHECKS, per oracle cell
-------------------------------
  CORPUS    corpus.jsonl is the unit stream the store yields now, same ids in
            the same order (the full-context baseline reads that order).
  GOLD      the gold tsv equals gold recomputed now from store + dataset.
            THE CENTRAL CHECK — the other four are how it fails.
  INCORPUS  every gold id is in corpus.jsonl. answer.py's oracle silently drops
            ids it cannot find, so a gold set can shrink with nothing logged.
  NAMESPACE units cite turns of their OWN sample. Foreign ids (a store ingested
            under a different adapter) intersect nothing and read as 0.0.
  ANSWERS   predictions cover every query, and answer_stats.context_items equals
            |gold n corpus| question by question — which is what proves the
            PROMPT held the gold set, not merely that a gold file exists.

An EMPTY gold set is reported, not failed: it is a real property of a store that
kept no unit from the evidence (lightmem 51 of 500 on LongMemEval-S), and of
BEAM's abstention questions, which name no evidence on purpose.

Usage
-----
    # every oracle cell under outputs/_answers/ — the whole grid, ~4 min
    python -m evalgrid.tools.check_oracle_trace

    # narrow to one store, or one gold level
    python -m evalgrid.tools.check_oracle_trace --system lightmem --dataset beam_100k
    python -m evalgrid.tools.check_oracle_trace --gold turn

    # qrels only, on a machine without the answers tree
    python -m evalgrid.tools.check_oracle_trace --skip-answers
    python -m evalgrid.tools.check_oracle_trace -v      # name the failing queries

Cells are discovered from the ANSWERS tree, not from a spec: the point is to
check what was actually reported. Exit code is 0 only when every cell traces,
so it can gate a results refresh.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from evalgrid.benchmarks.adapters import build_adapter  # noqa: E402
from evalgrid.benchmarks.gold import (  # noqa: E402
    ALL_GOLD_LEVELS, RELABEL_LEVELS, STAMPED_GOLD_LEVEL, RelabelLabels,
    evidence_turn_ids, gold_filename, grade_units, relabel_evidence,
    turn_id as _turn_id,
)
from evalgrid.systems.registry import available_systems, get_plugin  # noqa: E402

#: The raw-turn baselines have no plugin — their corpus is built by
#: qrels_turns._build_sample under one of these --roles. Mapped here so one
#: walker serves both corpus shapes, exactly as qrels_turns and
#: qrels_has_answer converge on write_beir_sample.
TURN_SYSTEMS = {"all_turns": "all", "user_turns": "user", "pair_turns": "pair"}

#: The labeler segment a store's own provenance is written under. A sidecar run
#: appends "-<file stem>" to it (qrels_has_answer --provenance-sidecar), which
#: is how the sidecar's filename is recovered below.
BASE_LABELER = "has_answer"


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_gold_tsv(path):
    """``query-id -> {corpus-id: score}`` from a BEIR/TREC gold file."""
    out = defaultdict(dict)
    with open(path) as fh:
        next(fh, None)                      # header: query-id corpus-id score
        for line in fh:
            if not line.strip():
                continue
            qid, cid, score = line.rstrip("\n").split("\t")
            out[qid][cid] = int(score)
    return out


def split_gold_suffix(labeler_seg_with_gold: str):
    """``has_answer-gold-turn`` -> ``("has_answer", "turn")``.

    The answers tree keys an oracle by labeler AND gold level in one segment
    (gold_suffix is never empty, so two levels cannot collide). Splitting on the
    LEVEL rather than on the last dash is what keeps a sidecar labeler —
    ``has_answer-provenance_turn-gold-turn`` — from losing half its name.
    """
    for level in sorted(ALL_GOLD_LEVELS, key=len, reverse=True):
        suffix = f"-gold-{level}"
        if labeler_seg_with_gold.endswith(suffix):
            return labeler_seg_with_gold[: -len(suffix)], level
    return None, None


def sidecar_for(labeler_seg: str):
    """The provenance sidecar filename a labeler segment implies, or None.

    Inverse of qrels_has_answer's ``labeler += "-" + splitext(basename(f))[0]``.
    """
    if labeler_seg == BASE_LABELER:
        return None
    if labeler_seg.startswith(BASE_LABELER + "-"):
        return labeler_seg[len(BASE_LABELER) + 1:] + ".jsonl"
    return None


def load_relabel_labels(qrels_dir: str, dataset: str):
    """The relabel label set a qrels tree records, verified against its sha256.

    Read from the tree's own ``manifest_relabel.json`` rather than from
    ``EVALGRID_RELABEL_ROOT``, for two reasons: the env var's default points at
    a root the labels may not live under, and the manifest names the exact file
    AND its digest — so a label set edited since the qrels were built is caught
    here instead of silently re-grading every unit.
    """
    manifest = os.path.join(qrels_dir, "manifest_relabel.json")
    if not os.path.isfile(manifest):
        return None, "no manifest_relabel.json beside the qrels"
    meta = json.load(open(manifest))
    path = meta.get("relabel_labels_file")
    if path and not os.path.isabs(path):
        path = os.path.join(REPO, path)
    if not (path and os.path.isfile(path)):
        return None, f"labels file missing: {meta.get('relabel_labels_file')!r}"

    digest = hashlib.sha256()
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, "rb") as fh:
        for raw in fh:
            digest.update(raw)
            if raw.strip():
                record = json.loads(raw)
                grades[record["query_id"]][record["turn_id"]] = int(record["grade"])
    if digest.hexdigest() != meta.get("relabel_labels_sha256"):
        return None, (f"labels file CHANGED since the qrels were built "
                      f"(sha256 {digest.hexdigest()[:12]} != "
                      f"{str(meta.get('relabel_labels_sha256'))[:12]})")
    return RelabelLabels(dataset, meta.get("relabel_label_set"), path,
                         digest.hexdigest(), dict(grades)), None


def build_units(system: str, sample, sample_mem: str, sidecar: str | None):
    """The corpus units for one sample, from whichever source built it."""
    if system in TURN_SYSTEMS:
        from evalgrid.pipeline.qrels_turns import _build_sample
        units = _build_sample(sample, TURN_SYSTEMS[system])
    else:
        units = list(get_plugin(system).iter_units(sample_mem))
    if sidecar:
        path = os.path.join(sample_mem, sidecar)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        recorded = {r["unit_id"]: list(r.get("source_turn_ids") or [])
                    for r in read_jsonl(path)}
        for unit in units:
            # KeyError here is the finding, not a crash to paper over: a sidecar
            # that does not cover the store was built from a different store.
            unit.source_turn_ids = recorded[unit.unit_id]
    return units


def expected_gold(units, sample, query, level, labels, per_query: bool):
    """``{unit_id: grade}`` — the gold set the builders would write now.

    Mirrors qrels_has_answer/qrels_turns exactly, including the one case where
    a level is NOT recomputed: "strict" on a benchmark whose evidence is not
    per-question is the flag the ingester stamped on the unit.
    """
    if level in RELABEL_LEVELS:
        return grade_units(units, relabel_evidence(sample, query, level, labels))
    if level == STAMPED_GOLD_LEVEL and not per_query:
        return {u.unit_id: 1 for u in units if u.has_answer}
    evidence = evidence_turn_ids(sample, query, level)
    return {u.unit_id: 1 for u in units if evidence.intersection(u.source_turn_ids)}


def discover(args):
    """Every oracle cell under outputs/_answers/, newest path layout.

    outputs/_answers/<system>/<dataset>/<mem-seg>/oracle/<labeler-seg>-gold-<level>/<gen-seg>/all/
    Several generators share one gold set, so cells are grouped by the qrels
    coordinates and the store is read once per group.
    """
    pattern = os.path.join(REPO, "outputs", "_answers", "*", "*", "*", "oracle",
                           "*", "*", "*", "predictions.jsonl")
    cells = defaultdict(list)
    for pred in sorted(glob.glob(pattern)):
        parts = pred.split(os.sep)
        system, dataset, mem_seg = parts[-8], parts[-7], parts[-6]
        labeler_seg, level = split_gold_suffix(parts[-4])
        if level is None:
            continue
        if args.system and system != args.system:
            continue
        if args.dataset and dataset != args.dataset:
            continue
        if args.gold and level != args.gold:
            continue
        cells[(system, dataset, mem_seg, labeler_seg, level)].append(os.path.dirname(pred))
    return cells


def check_cell(key, answer_dirs, args):
    """Run every check for one (store, labeler, gold level). Returns a report."""
    system, dataset, mem_seg, labeler_seg, level = key
    qrels_dir = os.path.join(REPO, "outputs", "_qrels", system, dataset, mem_seg, labeler_seg)
    mem_dir = os.path.join(REPO, "outputs", "_memory", system, dataset, mem_seg)
    rep = {"fatal": None, "corpus": [], "gold": [], "incorpus": [], "namespace": [],
           "answers": [], "queries": 0, "empty": 0, "gold_sizes": [], "samples": 0}

    if not os.path.isdir(qrels_dir):
        rep["fatal"] = f"no qrels tree at {os.path.relpath(qrels_dir, REPO)}"
        return rep

    labels = None
    if level in RELABEL_LEVELS:
        labels, why = load_relabel_labels(qrels_dir, dataset)
        if labels is None:
            rep["fatal"] = why
            return rep

    sidecar = sidecar_for(labeler_seg)
    samples = list(build_adapter(dataset).samples())
    per_query = bool(samples and samples[0].evidence_is_per_query)
    gold_by_query: dict[str, int] = {}

    for sample in samples:
        s_qrels = os.path.join(qrels_dir, sample.sample_id)
        gold_path = os.path.join(s_qrels, "qrels", gold_filename(level))
        if not os.path.isfile(gold_path):
            continue                        # not every sample is built in every tree
        rep["samples"] += 1
        s_mem = os.path.join(mem_dir, sample.sample_id)
        try:
            units = build_units(system, sample, s_mem, sidecar)
        except FileNotFoundError as exc:
            rep["corpus"].append(f"{sample.sample_id}: store file missing — "
                                 f"{os.path.relpath(str(exc), REPO)}")
            continue
        except KeyError as exc:
            rep["corpus"].append(f"{sample.sample_id}: sidecar does not cover unit {exc}")
            continue

        on_disk_corpus = [c["_id"] for c in read_jsonl(os.path.join(s_qrels, "corpus.jsonl"))]
        if on_disk_corpus != [u.unit_id for u in units]:
            rep["corpus"].append(
                f"{sample.sample_id}: {len(on_disk_corpus)} unit(s) on disk vs "
                f"{len(units)} rebuilt now")
        corpus_ids = set(on_disk_corpus)

        sample_turns = {_turn_id(s.session_id, i)
                        for s in sample.sessions for i in range(len(s.turns))}
        if units and not ({t for u in units for t in u.source_turn_ids} & sample_turns):
            rep["namespace"].append(sample.sample_id)

        on_disk_gold = read_gold_tsv(gold_path)
        written = {q["_id"] for q in read_jsonl(os.path.join(s_qrels, "queries.jsonl"))}
        for query in sample.queries:
            if query.query_id not in written:
                continue
            rep["queries"] += 1
            got = on_disk_gold.get(query.query_id, {})
            want = expected_gold(units, sample, query, level, labels, per_query)
            if got != want:
                only_disk = sorted(set(got) - set(want))
                only_now = sorted(set(want) - set(got))
                rep["gold"].append(
                    f"{query.query_id}: {len(got)} on disk vs {len(want)} recomputed "
                    f"(+{len(only_disk)} stale / -{len(only_now)} missed)")
            outside = set(got) - corpus_ids
            if outside:
                rep["incorpus"].append(f"{query.query_id}: {len(outside)} gold id(s) not in corpus")
            n_gold = len(set(got) & corpus_ids)
            rep["gold_sizes"].append(n_gold)
            gold_by_query[query.query_id] = n_gold
            rep["empty"] += (n_gold == 0)

    if args.skip_answers:
        return rep

    for adir in answer_dirs:
        label = os.path.relpath(adir, os.path.join(REPO, "outputs", "_answers"))
        answered = {r["question_id"] for r in read_jsonl(os.path.join(adir, "predictions.jsonl"))}
        missing = set(gold_by_query) - answered
        if missing:
            rep["answers"].append(f"{label}: {len(missing)} question(s) unanswered")
        errs = os.path.join(adir, "errors.jsonl")
        if os.path.isfile(errs) and read_jsonl(errs):
            rep["answers"].append(f"{label}: errors.jsonl has {len(read_jsonl(errs))} row(s)")
        stats_path = os.path.join(adir, "answer_stats.jsonl")
        if not os.path.isfile(stats_path):
            continue
        # The check that the PROMPT held the gold set. A gold file that exists
        # and an oracle that read it are different claims.
        wrong = [r["question_id"] for r in read_jsonl(stats_path)
                 if r.get("question_id") in gold_by_query
                 and r.get("context_items") is not None
                 and r["context_items"] != gold_by_query[r["question_id"]]]
        if wrong:
            rep["answers"].append(
                f"{label}: context_items != |gold| on {len(wrong)} question(s), "
                f"e.g. {wrong[:3]}")
    return rep


def main():
    ap = argparse.ArgumentParser(
        description="Check that every oracle cell still traces back to its store")
    ap.add_argument("--system", choices=sorted(set(available_systems()) | set(TURN_SYSTEMS)),
                    help="Only this system (default: every one with oracle answers)")
    ap.add_argument("--dataset", help="Only this dataset key")
    ap.add_argument("--gold", choices=list(ALL_GOLD_LEVELS),
                    help="Only this gold level")
    ap.add_argument("--skip-answers", action="store_true",
                    help="Check the qrels against the store only — for a machine "
                         "that has the stores but not the answers tree")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="List every offending sample/query, not just the count")
    args = ap.parse_args()

    cells = discover(args)
    if not cells:
        print("No oracle cells matched.", file=sys.stderr)
        return 1

    print(f"Checking {len(cells)} oracle cell(s) "
          f"({sum(len(v) for v in cells.values())} answer dir(s))\n")
    failed = 0
    totals = Counter()
    for key in sorted(cells):
        system, dataset, mem_seg, labeler_seg, level = key
        label = f"{system}/{dataset} {labeler_seg} gold={level}"
        rep = check_cell(key, cells[key], args)

        problems = {k: rep[k] for k in ("corpus", "gold", "incorpus", "namespace", "answers")
                    if rep[k]}
        if rep["fatal"]:
            print(f"ERROR     {label}\n            {rep['fatal']}")
            failed += 1
            continue
        sizes = rep["gold_sizes"]
        mean = f"{sum(sizes) / len(sizes):.2f}" if sizes else "-"
        note = (f"{rep['queries']} q, {mean} gold units/q"
                + (f", {rep['empty']} empty" if rep["empty"] else ""))
        if not problems:
            print(f"OK        {label}\n            {note}")
        else:
            failed += 1
            print(f"FAIL      {label}\n            {note}")
            for kind, items in problems.items():
                totals[kind] += len(items)
                print(f"            {kind.upper():9} {len(items)}")
                for line in (items if args.verbose else items[:3]):
                    print(f"                - {line}")
                if not args.verbose and len(items) > 3:
                    print(f"                  ... {len(items) - 3} more (-v for all)")

    print("\n" + "=" * 70)
    if failed:
        print(f"{failed} of {len(cells)} cell(s) DO NOT TRACE: "
              + ", ".join(f"{n} {k}" for k, n in sorted(totals.items())))
        print("A GOLD failure means the qrels are not what the store produces now — "
              "the store changed, a provenance sidecar went missing, or the level "
              "was redefined. Find out which BEFORE rebuilding: a rebuild makes the "
              "tree self-consistent and the reported numbers wrong.")
        return 1
    print(f"ALL {len(cells)} CELL(S) TRACE — every oracle answered from the gold set "
          f"its store produces today.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
