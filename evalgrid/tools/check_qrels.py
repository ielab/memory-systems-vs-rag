#!/usr/bin/env python3
"""
check_qrels.py
==============
Scan BEIR-format qrels (``test_oracle_<level>.tsv`` with
``query-id  corpus-id  score``) and report how many relevant texts (rows with
``score > 0``) each query has.

The script recursively finds every gold file for ``--gold`` under the given
root, aggregates
relevant ``corpus-id``s per ``query-id`` (deduped across files), and prints a
per-query count plus a summary distribution.

Usage
-----
    python evalgrid/tools/check_qrels.py outputs/_qrels/user_turns/longmemeval_s

If no path is given it defaults to ``outputs/_qrels`` (relative to repo root).
A path pointing directly at a single ``test.tsv`` is also accepted.

Output
------
With ``--verbose`` every query is listed; otherwise only the summary
(query count, total relevant rows, and min/max/mean/median per query).
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evalgrid.benchmarks.gold import EVAL_GOLD_LEVEL, GOLD_LEVELS, gold_filename

REPO = Path(__file__).resolve().parents[2]  # evalgrid/tools/ -> repo root
DEFAULT_DIR = REPO / "outputs/_qrels"


def find_qrels_files(root: Path, gold: str = EVAL_GOLD_LEVEL) -> list[Path]:
    """Return the list of gold files under `root` (or `root` itself).

    Which filename that is depends on the gold LEVEL (evalgrid/benchmarks/gold.py):
    a sample's qrels dir holds one file per level, and counting them together
    would report a query as having the union of its evidence sets.
    """
    if root.is_file():
        return [root]
    return sorted(root.rglob(gold_filename(gold)))


def parse_qrels(path: Path, rel_per_query: dict[str, set]) -> int:
    """Add relevant (score>0) corpus-ids into `rel_per_query`; return bad lines."""
    bad = 0
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                bad += 1
                continue
            qid, cid, score = parts[0], parts[1], parts[2]
            if i == 0 and score.strip().lower() == "score":
                continue  # header
            try:
                rel = float(score) > 0
            except ValueError:
                bad += 1
                continue
            if rel:
                rel_per_query[qid].add(cid)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "qrels_path",
        nargs="?",
        default=str(DEFAULT_DIR),
        help="Dir to scan for gold files (recursively), or a single .tsv",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print the relevant-count for every query, not just the summary",
    )
    ap.add_argument(
        "--gold", default=EVAL_GOLD_LEVEL, choices=list(GOLD_LEVELS),
        help="Which gold level's file to scan. Default matches what evaluation "
             "uses ('turn' = test_oracle_turn.tsv).",
    )
    ap.add_argument(
        "--sort", choices=("qid", "count"), default="qid",
        help="Sort the per-query listing by query-id (default) or count",
    )
    args = ap.parse_args()

    root = Path(args.qrels_path)
    if not root.exists():
        print(f"ERROR: path does not exist: {root}", file=sys.stderr)
        return 1

    files = find_qrels_files(root, args.gold)
    if not files:
        print(f"ERROR: no {gold_filename(args.gold)} found under {root}", file=sys.stderr)
        return 1

    rel_per_query: dict[str, set] = defaultdict(set)
    bad_lines = 0
    for path in files:
        bad_lines += parse_qrels(path, rel_per_query)

    if not rel_per_query:
        print(f"ERROR: no relevant rows parsed from {len(files)} file(s)",
              file=sys.stderr)
        return 1

    counts = {qid: len(cids) for qid, cids in rel_per_query.items()}

    if args.verbose:
        if args.sort == "count":
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        else:
            items = sorted(counts.items())
        for qid, n in items:
            print(f"{qid}\t{n}")
        print()

    values = list(counts.values())
    print("===== SUMMARY =====")
    print(f"qrels root          : {root}")
    print(f"gold level          : {args.gold}")
    print(f"gold files          : {len(files)} x {gold_filename(args.gold)}")
    print(f"queries             : {len(counts)}")
    print(f"total relevant rows : {sum(values)}")
    print(f"relevant / query    : min={min(values)} max={max(values)} "
          f"mean={statistics.mean(values):.2f} "
          f"median={statistics.median(values)}")
    if bad_lines:
        print(f"unparseable lines   : {bad_lines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
