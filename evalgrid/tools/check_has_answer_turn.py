#!/usr/bin/env python3
"""
check_has_answer_turn.py
========================
Scan saved LightMem per-sample results and report whether each sample's
``points.jsonl`` records carry the ``has_answer`` and ``turn`` keys (either at
the top level of a record or inside its ``payload``).

Usage
-----
    python -m evalgrid.tools.check_has_answer_turn \
        outputs/_memory/lightmem/longmemeval_s/Qwen_Qwen3-30B-A3B-Instruct-2507

If no directory is given it defaults to the path above (relative to repo root).

Output
------
Per-sample status for any sample that is *missing* a key (use --verbose to list
all samples), followed by a summary. Exit code 0 = every sample has both keys,
1 = at least one sample is missing a key (or has no records / no file).
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # evalgrid/tools/ -> repo root
DEFAULT_DIR = (
    REPO
    / "outputs/_memory/lightmem/longmemeval_s/Qwen_Qwen3-30B-A3B-Instruct-2507"
)

KEYS = ("has_answer", "turn")


def record_has_key(record: dict, key: str) -> bool:
    """True if `key` is present at the record top level or inside payload."""
    if key in record:
        return True
    payload = record.get("payload")
    return isinstance(payload, dict) and key in payload


def check_sample(points_path: Path) -> dict:
    """Return per-key counts for one points.jsonl file."""
    n_records = 0
    counts = {k: 0 for k in KEYS}
    bad_lines = 0
    with points_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            n_records += 1
            for k in KEYS:
                if record_has_key(rec, k):
                    counts[k] += 1
    return {"n_records": n_records, "counts": counts, "bad_lines": bad_lines}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "results_dir",
        nargs="?",
        default=str(DEFAULT_DIR),
        help="Directory containing per-sample subdirs with points.jsonl",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print a line for every sample, not just problematic ones",
    )
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not sample_dirs:
        print(f"ERROR: no sample subdirs under {root}", file=sys.stderr)
        return 1

    n_samples = 0
    n_missing_file = 0
    n_empty = 0
    n_complete = 0  # every record has both keys
    n_partial = 0   # some records have keys, not all
    n_none = 0      # no record has the keys
    missing_per_key = {k: 0 for k in KEYS}  # samples lacking key entirely

    for d in sample_dirs:
        points = d / "points.jsonl"
        if not points.is_file():
            n_missing_file += 1
            print(f"[no points.jsonl] {d.name}")
            continue
        n_samples += 1
        res = check_sample(points)
        nrec = res["n_records"]
        counts = res["counts"]

        if nrec == 0:
            n_empty += 1
            print(f"[empty]         {d.name}")
            continue

        has_all_full = all(counts[k] == nrec for k in KEYS)
        has_any = any(counts[k] > 0 for k in KEYS)

        for k in KEYS:
            if counts[k] == 0:
                missing_per_key[k] += 1

        if has_all_full:
            n_complete += 1
            status = "OK"
        elif has_any:
            n_partial += 1
            status = "PARTIAL"
        else:
            n_none += 1
            status = "MISSING"

        if args.verbose or status != "OK":
            detail = ", ".join(f"{k}={counts[k]}/{nrec}" for k in KEYS)
            print(f"[{status:7}] {d.name}  ({detail})")

    print("\n===== SUMMARY =====")
    print(f"results dir         : {root}")
    print(f"sample subdirs      : {len(sample_dirs)}")
    print(f"with points.jsonl   : {n_samples}")
    print(f"missing points.jsonl: {n_missing_file}")
    print(f"empty (0 records)   : {n_empty}")
    print(f"OK (both keys, all) : {n_complete}")
    print(f"PARTIAL             : {n_partial}")
    print(f"MISSING both keys   : {n_none}")
    for k in KEYS:
        print(f"samples lacking '{k}' entirely: {missing_per_key[k]}")

    all_good = (
        n_complete == n_samples
        and n_missing_file == 0
        and n_empty == 0
    )
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
