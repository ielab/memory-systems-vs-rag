#!/usr/bin/env python3
"""
check_native_ids.py
===================
Verify that a native run.tsv is scoreable against the qrels it will be scored
against — that the unit ids the system's own retrieval reported actually EXIST
in the corpus built from the same store.

Why this needs its own check. The two id vocabularies are produced by one class
but by different methods: ``SystemPlugin.iter_units`` builds the qrels corpus,
``SystemPlugin.native_unit_ids`` builds the run.tsv. Nothing forces them to
agree, and when they disagree **every metric is exactly 0** — not an error, not
a warning, just a run that reads as "this system retrieves nothing relevant".
HyperMem shipped that way: iter_units namespaced ids ``fact::``/``ep::`` while
native_unit_ids emitted upstream's raw ``fact_<uuid>``, so 98/98 ids were
unmatchable and its native row was a flat zero that looked like a finding.

Also checks two things that decide whether a saved retrieval is REPLAYABLE under
a depth or token budget:

  retrieved_units present   Both --top-k and --context-budget cut this list.
                            A system that saves only its rendered context block
                            has no unit boundary to cut at, so both knobs raise.
  units align with ids      They are cut together, and context_items is derived
                            from the ids — if the two lists disagree in length,
                            a truncated cell misreports what reached the prompt.

And the gold count per sample, because a sample with **no** gold unit is
silently dropped by eval_retrieval — a system can look like it scored 2
questions when it scored 1.

Reads finished artifacts only; runs nothing and loads no model.

Usage:
  python -m evalgrid.tools.check_native_ids \\
      --qrels-root outputs/_smoke2_qrels --retrieval-root outputs/_smoke2_retrieval
  python -m evalgrid.tools.check_native_ids ... --run-name native
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from evalgrid.benchmarks.gold import (EVAL_GOLD_LEVEL,
                                      GOLD_LEVELS, gold_filename)


def _corpus_ids(path: str) -> set[str]:
    with open(path) as f:
        return {json.loads(line)["_id"] for line in f if line.strip()}


def _gold_ids(path: str) -> set[str]:
    """Relevant docs (score > 0) from a TREC qrels file, header skipped."""
    if not os.path.isfile(path):
        return set()
    gold = set()
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    if float(parts[2]) > 0:
                        gold.add(parts[1])
                except ValueError:
                    continue
    return gold


def _budgetable(retrieval_path: str) -> str:
    """Can this saved retrieval be cut by --top-k / --context-budget?

    Both knobs slice ``retrieved_units``. A system whose trace carries only the
    rendered context block has no unit boundary to cut at. The id list must also
    match it in length, since the two are cut together and ``context_items``
    comes from the ids.
    """
    if not os.path.isfile(retrieval_path):
        return "no trace"
    try:
        with open(retrieval_path) as f:
            line = f.readline()
        if not line.strip():
            return "empty"
        trace = json.loads(line)
    except Exception:
        return "unreadable"
    units = trace.get("retrieved_units")
    if units is None:
        return "NO UNITS"
    ids = trace.get("retrieved_unit_ids")
    # A dict-shaped id list is level-keyed (hypermem), where a length mismatch
    # against the units is expected — it drops levels the corpus does not hold.
    #
    # ``unit_id_groups`` is the other legitimate mismatch: MemoryBank packs
    # several memories into one date-bucket block, so its id list is LONGER than
    # its unit list by construction and can never satisfy the length test below.
    # It is still fully replayable — install_replay cuts on the memory count and
    # rebuilds the ids from the groups under both knobs — so the groups are
    # checked first, and only a groups list that does not describe the units is
    # a real problem.
    groups = trace.get("unit_id_groups")
    if isinstance(groups, list):
        if len(groups) != len(units):
            return f"{len(groups)}groups/{len(units)}u"
        if isinstance(ids, list) and sum(len(g) for g in groups) != len(ids):
            return f"groups cover {sum(len(g) for g in groups)}/{len(ids)} ids"
        return "yes (grouped)"
    if isinstance(ids, list) and len(ids) != len(units):
        return f"{len(ids)}ids/{len(units)}u"
    return "yes"


def _run_ids(path: str) -> list[str]:
    ids = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ids.append(parts[1])
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qrels-root", default="outputs/_smoke2_qrels",
                    help="Root holding <system>/<sample>/corpus.jsonl")
    ap.add_argument("--retrieval-root", default="outputs/_smoke2_retrieval",
                    help="Root holding <system>/<run-name>/<sample>/run.tsv")
    ap.add_argument("--gold", default=EVAL_GOLD_LEVEL, choices=list(GOLD_LEVELS),
                    help=f"Gold level whose qrels file names the relevant ids "
                         f"(default: {EVAL_GOLD_LEVEL}, what evaluation uses)")
    ap.add_argument("--run-name", default="native",
                    help="Which run dir under each system to check (default: native)")
    args = ap.parse_args()

    systems = sorted(d for d in os.listdir(args.retrieval_root)
                     if os.path.isdir(os.path.join(args.retrieval_root, d))) \
        if os.path.isdir(args.retrieval_root) else []
    if not systems:
        print(f"No systems under {args.retrieval_root}", file=sys.stderr)
        return 1

    print(f"{'system':<12} {'sample':<10} {'corpus':>7} {'gold':>5} {'run':>5} "
          f"{'matched':>8} {'UNMATCHED':>10}  {'budgetable':>10}")
    print("-" * 82)
    broken, no_gold, missing, unbudgetable = [], [], [], []

    for s in systems:
        pattern = os.path.join(args.retrieval_root, s, args.run_name, "*", "run.tsv")
        for run_path in sorted(glob.glob(pattern)):
            sid = os.path.basename(os.path.dirname(run_path))
            corpus_path = os.path.join(args.qrels_root, s, sid, "corpus.jsonl")
            if not os.path.isfile(corpus_path):
                missing.append((s, sid, corpus_path))
                print(f"{s:<12} {sid:<10}  no qrels corpus — build it first")
                continue
            corpus = _corpus_ids(corpus_path)
            gold = _gold_ids(os.path.join(args.qrels_root, s, sid, "qrels",
                                          gold_filename(args.gold)))
            run = _run_ids(run_path)
            unmatched = [i for i in run if i not in corpus]
            budgetable = _budgetable(os.path.join(os.path.dirname(run_path), "retrieval.jsonl"))
            flag = ""
            if unmatched:
                flag = "  <-- BUG"
                broken.append((s, sid, len(unmatched), len(run), sorted(set(unmatched))[:3]))
            if not gold:
                flag += "  <-- no gold, query dropped by eval"
                no_gold.append((s, sid))
            # "yes (grouped)" is a pass — see _budgetable.
            if not budgetable.startswith("yes"):
                unbudgetable.append((s, sid, budgetable))
            print(f"{s:<12} {sid:<10} {len(corpus):>7} {len(gold):>5} {len(run):>5} "
                  f"{len(run) - len(unmatched):>8} {len(unmatched):>10}  {budgetable:>10}{flag}")

    print()
    if unbudgetable:
        print("NOT REPLAYABLE under --top-k / --context-budget — both cut")
        print("retrieved_units, and these saved runs cannot supply it:")
        for s, sid, why in unbudgetable:
            print(f"  {s}/{sid}: {why}")
        print()
    if broken:
        print("ID NAMESPACE MISMATCH — these runs score 0 on every metric regardless")
        print("of retrieval quality. native_unit_ids and iter_units disagree:")
        for s, sid, n, total, ex in broken:
            print(f"  {s}/{sid}: {n}/{total} ids absent from the corpus, e.g. {ex}")
    if no_gold:
        print("\nSamples with no gold unit (eval_retrieval drops the query, so the")
        print("system's score comes from fewer questions than it answered):")
        for s, sid in no_gold:
            print(f"  {s}/{sid}")
    if missing:
        print(f"\n{len(missing)} sample(s) had no qrels corpus to check against.")
    if not broken:
        print("No id mismatches — every native run.tsv is scoreable against its qrels.")
    return 1 if (broken or unbudgetable) else 0


if __name__ == "__main__":
    sys.exit(main())
