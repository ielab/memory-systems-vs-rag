#!/usr/bin/env python3
"""
inspect_trajectory.py
=====================
Render one trajectory-<level>.jsonl as an indented tree with per-stage cost totals.

A trajectory that nothing can read is a write-only file, so this ships with the
recorder rather than after it. It reads what evalgrid/lib/trace.py writes:

  outputs/_memory/<system>/<dataset>/<mem-seg>/<sample>/trajectory-<level>.jsonl  (ingest)
  outputs/_answers/<...>/trajectory-<level>.jsonl                                 (answer)

One file per --trace-level, never shared: a "calls" file and a "steps" file are
different runs, so they must not overwrite or append into each other. Pass a
directory and this picks the single trajectory in it, or names them if there are
several.

Nesting is rebuilt here, not stored: records are flat and append-only so a crash
mid-sample loses nothing, and ``parent_seq`` is enough to indent them at read
time.

Usage
-----
    python -m evalgrid.tools.inspect_trajectory DIR             # the tree
    python -m evalgrid.tools.inspect_trajectory PATH            # ... or one file
    python -m evalgrid.tools.inspect_trajectory PATH --summary  # stages only
    python -m evalgrid.tools.inspect_trajectory PATH -v         # + payloads
    python -m evalgrid.tools.inspect_trajectory PATH --stage extract
    python -m evalgrid.tools.inspect_trajectory PATH --sample s1 --phase ingest

Totals here are for reading, not for reporting: a trajectory can be sampled or
truncated, so ingest_usage.json / answer_stats.jsonl stay the authority on cost
.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.lib.trace import TRAJECTORY_FILENAMES  # noqa: E402


def read_records(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ! skipped malformed line {n}", file=sys.stderr)
    return rows


def _cost(rec: dict) -> tuple[int, int, int]:
    c = rec.get("cost") or {}
    return (int(c.get("input_tokens") or 0), int(c.get("output_tokens") or 0),
            int(c.get("calls") or 0))


def summarize(rows: list[dict]) -> None:
    """Per-stage totals. Steps and the calls beneath them are counted
    separately: a step's cost already contains its children's, so summing both
    into one column would double-count every LLM call inside a named stage."""
    by_stage: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        key = (r.get("stage", "?"), r.get("kind", "?"))
        acc = by_stage.setdefault(key, [0, 0, 0, 0, 0.0])
        c_in, c_out, c_calls = _cost(r)
        acc[0] += 1
        acc[1] += c_in
        acc[2] += c_out
        acc[3] += c_calls
        acc[4] += float(r.get("seconds") or 0.0)

    print(f"{'stage':<12} {'kind':<6} {'n':>5} {'in':>10} {'out':>9} {'calls':>6} {'sec':>8}")
    print("-" * 62)
    for (stage, kind), (n, c_in, c_out, calls, secs) in sorted(by_stage.items()):
        print(f"{stage:<12} {kind:<6} {n:>5} {c_in:>10,} {c_out:>9,} {calls:>6,} {secs:>8.2f}")


def _ordered(rows: list[dict]) -> list[tuple[int, dict]]:
    """(depth, record) with every parent placed before its children.

    A step's record is written when its block EXITS — that is what lets its cost
    be the delta across the block — so on disk the children come first. Reading
    order is reconstructed here rather than paid for at write time.
    """
    by_seq = {r.get("seq"): r for r in rows}
    children: dict = {}
    roots: list[dict] = []
    for r in rows:
        parent = r.get("parent_seq")
        (children.setdefault(parent, []) if parent in by_seq else roots).append(r)

    out: list[tuple[int, dict]] = []

    def walk(rec: dict, depth: int) -> None:
        out.append((depth, rec))
        for child in sorted(children.get(rec.get("seq"), []), key=lambda c: c.get("seq", 0)):
            walk(child, depth + 1)

    for root in sorted(roots, key=lambda r: r.get("seq", 0)):
        walk(root, 0)
    return out


def render_tree(rows: list[dict], verbose: bool, max_chars: int) -> None:
    for d, r in _ordered(rows):
        c_in, c_out, calls = _cost(r)
        cost = f"{c_in:,}->{c_out:,} tok" + (f" / {calls} call" if calls else "")
        marker = "*" if r.get("kind") == "step" else "-"
        print(f"{'  ' * d}{marker} [{r.get('stage','?')}] {r.get('step','?')} "
              f"({r.get('seconds', 0):.2f}s, {cost})")
        if (r.get("meta") or {}).get("error"):
            print(f"{'  ' * d}    ! {r['meta']['error']}")
        if r.get("unit_ids"):
            print(f"{'  ' * d}    units: {r['unit_ids'][:6]}"
                  f"{' …' if len(r['unit_ids']) > 6 else ''}")
        if verbose:
            for side in ("in", "out"):
                p = r.get(side) or {}
                text = (p.get("text") or "").replace("\n", " ")
                if not text:
                    continue
                clipped = text[:max_chars] + ("…" if len(text) > max_chars else "")
                note = f" [{p['chars']:,} chars, truncated at record time]" if p.get("truncated") else ""
                print(f"{'  ' * d}    {side}:{note} {clipped}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="A trajectory-<level>.jsonl, or a directory holding one")
    ap.add_argument("--sample", default=None, help="Only this sample_id")
    ap.add_argument("--phase", default=None, choices=["ingest", "answer"])
    ap.add_argument("--stage", default=None, help="Only this stage (e.g. extract)")
    ap.add_argument("--summary", action="store_true", help="Per-stage totals only")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show prompts/responses")
    ap.add_argument("--max-chars", type=int, default=400,
                    help="Clip payloads to this many chars when printing (default: 400)")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N records")
    args = ap.parse_args()

    path = Path(args.path)
    if path.is_dir():
        # Each level writes its own file, so a directory may hold several. Point
        # at one rather than silently merging them: a "calls" file and a "steps"
        # file are different runs, and concatenating them would invent a
        # trajectory that never happened.
        found = sorted(p for p in path.iterdir() if p.name in TRAJECTORY_FILENAMES)
        if len(found) == 1:
            path = found[0]
        elif not found:
            print(f"ERROR: no trajectory in {path}\n"
                  f"  Expected one of: {', '.join(sorted(TRAJECTORY_FILENAMES))}",
                  file=sys.stderr)
            return 2
        else:
            print(f"ERROR: {path} holds several levels — name the one you want:\n"
                  + "\n".join(f"    {p}" for p in found), file=sys.stderr)
            return 2
    if not path.is_file():
        print(f"ERROR: no trajectory at {path}\n"
              f"  A run records one only with --trace-level calls|steps|full, and a sample "
              f"skipped by resume writes none — a missing file means 'not recorded', not "
              f"'no steps'.", file=sys.stderr)
        return 2

    rows = read_records(path)
    for key, want in (("sample_id", args.sample), ("phase", args.phase), ("stage", args.stage)):
        if want:
            rows = [r for r in rows if r.get(key) == want]
    if args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("(no records match)")
        return 0

    samples = sorted({r.get("sample_id") for r in rows})
    phases = sorted({r.get("phase") for r in rows})
    # Each record carries the level and clip that produced it, so a file stays
    # readable after it is copied out of the tree that named it. More than one
    # level here would mean two runs interleaved — the thing per-level filenames
    # exist to prevent — so say so loudly rather than averaging over it.
    levels = sorted({str(r.get("level", "?")) for r in rows})
    clips = sorted({str(r.get("max_chars", "?")) for r in rows})
    print(f"{path}")
    print(f"records : {len(rows)}   samples: {', '.join(map(str, samples[:5]))}"
          f"{' …' if len(samples) > 5 else ''}   phases: {', '.join(map(str, phases))}")
    print(f"level   : {', '.join(levels)}   clip: {', '.join(clips)} chars")
    if len(levels) > 1:
        print("  ! this file mixes trace levels — it holds two runs, not one")
    print("=" * 62)
    if args.summary:
        summarize(rows)
    else:
        render_tree(rows, args.verbose, args.max_chars)
        print("=" * 62)
        summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
