#!/usr/bin/env python3
"""
ingest_stats.py
===============
Summarise what an ingest actually built, in the terms stage 2-3 are chosen in.

``check_ingest`` answers "did the run finish" — every sample complete, no
unresolved failure, no degradation counter. This answers the next question,
which is what the store is *made of*,
measured over the real 500-sample ingest
instead of the two smoke samples it was first measured on.

What it measures, and why each column is there
----------------------------------------------
A **unit** is one thing ``SystemPlugin.iter_units`` yields — one line of
corpus.jsonl, one document the retriever ranks, the thing ``--top-k`` counts.
It is the atom of retrieval: the smallest evidence that can reach the prompt,
since a unit cannot be split. Two numbers about it decide how systems are
compared:

  * **tok/unit** — the exchange rate between the two matching axes. ``--top-k``
    buys units, ``--context-budget`` buys tokens, and this converts one to the
    other. We measured a 43x spread across six systems, so equal k is very
    much not equal evidence, which is the whole argument for matching on
    tokens.
  * **units/sample** — what depth is even reachable. A system whose store holds
    fewer units than k cannot be asked for k, and that is a property of the
    representation, not of the run.

Tokens are counted on the string that actually reaches the generator —
``answer.format_fact``'s ``[<time>] <text>``, the same rendering the answer
stage does — not on the system's internal object. Pass ``--raw-text`` to count
``MemoryUnit.text`` alone; the gap between the two is the timestamp's cost,
which is per-unit and so is paid most by the finest-grained systems (for Mem0
it more than doubles the unit: median 10 raw, 25 rendered).

**``--raw-text`` is what reproduces the unit-size table**, because that was measured
over corpus.jsonl, and ``benchmarks.qrels`` writes clean text there with the
stamp in metadata — the rendering happens later, in answer.py. Checked against
the published numbers on the smoke stores: SimpleMem and LightMem match on
median/mean/p90 exactly, Mem0 on median/p90, MemoryBank on max. The default is
rendered anyway, since that is the number a context budget is actually spent
against.

Only **complete** stores are read (``runner._ingest_complete`` — usage.json,
written strictly after save_memory returns), so a half-written store cannot
drag a mean down and be mistaken for a system that stores less.

Usage
-----
    # one system
    python -m evalgrid.tools.ingest_stats --exp experiments/mem0/longmemeval.yaml

    # the comparison table — repeat --exp, one row per system
    python -m evalgrid.tools.ingest_stats \\
        --exp experiments/mem0/longmemeval.yaml \\
        --exp experiments/lightmem/longmemeval.yaml

    # a quick read on a big store, and the machine-readable form
    python -m evalgrid.tools.ingest_stats --exp experiments/mem0/longmemeval.yaml \\
        --limit 25 --json outputs/_stats/mem0.json

Exit code is 0 whenever every requested store could be read; this is a
measuring tool, not a gate. ``check_ingest`` is the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# evalgrid/tools/ -> repo root; make the package importable when run as a file.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.benchmarks.runner import _ingest_complete  # noqa: E402
from evalgrid.lib.cli import (  # noqa: E402
    add_dataset_arg,
    add_limit_arg,
    add_mem_model_arg,
    add_mem_variant_arg,
    add_system_arg,
)
from evalgrid.lib.paths import ExperimentPaths  # noqa: E402
from evalgrid.pipeline.answer import format_fact  # noqa: E402
from evalgrid.systems.registry import get_plugin  # noqa: E402

#: Depths to price in the exchange table. Spans the useful range: 5 is below
#: every system's median store, 50 is above some systems' entire store.
DEFAULT_TOPK = (5, 10, 20, 50)

#: Context budgets to price in the reverse direction, in tokens.
DEFAULT_BUDGETS = (2_000, 8_000)


# ---------------------------------------------------------------------------
# Resolving what to measure
# ---------------------------------------------------------------------------

def load_spec(path: str) -> dict:
    """The identity keys out of an experiment spec.

    Read directly rather than through ``lib.cli.apply_spec``, because that
    resolves one target against argv and this tool takes several. Only the four
    keys that name a memory dir are taken; the rest of the spec belongs to
    stages this tool does not run.
    """
    import yaml

    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    return {k: spec.get(k) for k in ("system", "dataset", "mem_model", "mem_variant")}


def resolve_targets(ap: argparse.ArgumentParser, args) -> list[dict]:
    """[{label, system, mem_dir}] for every store the invocation names.

    A CLI flag overrides the same key in every spec, so
    ``--exp a.yaml --exp b.yaml --mem-model X`` compares two systems at one
    model rather than needing two edited specs.
    """
    overrides = {k: getattr(args, k, None)
                 for k in ("system", "dataset", "mem_model", "mem_variant")
                 if getattr(args, k, None)}

    if args.mem_dir:
        if not overrides.get("system"):
            ap.error("--mem-dir needs --system: the plugin that reads the store "
                     "cannot be derived from a bare path")
        return [{"label": overrides["system"], "system": overrides["system"],
                 "mem_dir": args.mem_dir}]

    specs = [load_spec(p) for p in (args.exp or [])] or [{}]
    targets = []
    for spec in specs:
        keys = {**spec, **overrides}
        missing = [k for k in ("system", "dataset", "mem_model") if not keys.get(k)]
        if missing:
            ap.error("need --exp, or --mem-dir, or enough to derive the store: missing "
                     + ", ".join("--" + m.replace("_", "-") for m in missing))
        targets.append({
            "label": keys["system"] + (f"/{keys['mem_variant']}" if keys.get("mem_variant") else ""),
            "system": keys["system"],
            "mem_dir": ExperimentPaths(system=keys["system"], dataset=keys["dataset"],
                                       mem_model=keys["mem_model"],
                                       mem_variant=keys.get("mem_variant")).memory(),
        })
    return targets


# ---------------------------------------------------------------------------
# Measuring one store
# ---------------------------------------------------------------------------

def _dist(vals: list[int]) -> dict | None:
    """The columns the unit-size table reports: median / mean / p90 / max, plus n.

    p90 rather than dataset_stats' p95: the interesting tail here is the
    occasional long unit that alone overflows a small budget, and p90 is where
    that becomes visible without being one outlier wide.
    """
    v = sorted(vals)
    n = len(v)
    if not n:
        return None
    at = lambda f: v[min(n - 1, int(f * n))]
    return {"n": n, "median": at(.5), "mean": round(sum(v) / n, 1),
            "p90": at(.9), "max": v[-1], "total": sum(v)}


def complete_samples(mem_dir: str) -> list[str]:
    """Sample slugs with a usage.json, in the order the runner named them.

    Incomplete stores are excluded rather than counted as small: a save that
    crashed part-way holds real units and would otherwise read as a system that
    stores less than it does.
    """
    return [e.name for e in sorted(os.scandir(mem_dir), key=lambda e: e.name)
            if e.is_dir() and _ingest_complete(e.path)]


def measure(target: dict, enc, limit: int | None, raw_text: bool,
            threads: int) -> dict:
    """Unit statistics for one store, or an ``error`` key if it cannot be read."""
    plugin = get_plugin(target["system"])
    slugs = complete_samples(target["mem_dir"])
    if limit:
        slugs = slugs[:limit]

    tok_per_unit: list[int] = []
    units_per_sample: list[int] = []
    tokens_per_sample: list[int] = []
    has_answer_units = 0
    samples_without_evidence = 0
    samples_without_provenance = 0
    empty_stores: list[str] = []
    examples: list[str] = []

    for slug in slugs:
        path = os.path.join(target["mem_dir"], slug)
        units = list(plugin.iter_units(path))
        if not units:
            # Legitimate for a store the qrels builder will report as missing;
            # counted, not skipped, because an empty store is a real outcome of
            # a run that exited 0 (see mem0's empty_extractions).
            empty_stores.append(slug)
            units_per_sample.append(0)
            tokens_per_sample.append(0)
            continue

        # The string the generator sees. iter_units keeps text clean and the
        # stamp in `timestamp`; format_fact is what re-joins them at answer
        # time, so counting anything else prices a prompt nobody sends.
        texts = [u.text if raw_text else
                 format_fact({"text": u.text, "metadata": {"time_stamp": u.timestamp}})
                 for u in units]
        # Batched: one call over a sample's units rather than one per unit,
        # which is the difference between seconds and minutes over 500 samples.
        lengths = [len(t) for t in enc.encode_ordinary_batch(texts, num_threads=threads)]

        tok_per_unit.extend(lengths)
        units_per_sample.append(len(units))
        tokens_per_sample.append(sum(lengths))

        evidence = sum(1 for u in units if u.has_answer)
        has_answer_units += evidence
        if not evidence:
            # No unit derives from an evidence turn. Either the extraction lost
            # the provenance (a re-ingest is needed) or the store genuinely has
            # none — records_has_answer is the only thing that can tell them
            # apart, since MemoryUnit.has_answer defaults to False.
            if plugin.records_has_answer(path):
                samples_without_evidence += 1
            else:
                samples_without_provenance += 1
        if not examples:
            examples = texts[:2]

    return {
        "label": target["label"],
        "system": target["system"],
        "mem_dir": target["mem_dir"],
        "samples": len(slugs),
        "empty_stores": empty_stores,
        "tok_per_unit": _dist(tok_per_unit),
        "units_per_sample": _dist(units_per_sample),
        "tokens_per_sample": _dist(tokens_per_sample),
        # Raw per-sample counts, for the reachability check. Underscored so the
        # JSON payload stays a summary rather than 500 ints per system.
        "_units_list": units_per_sample,
        "has_answer_units": has_answer_units,
        "samples_without_evidence": samples_without_evidence,
        "samples_without_provenance": samples_without_provenance,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                     for i, (c, w) in enumerate(zip(cells, widths)))


def print_units_table(results: list[dict]) -> None:
    """The unit-size table: what one unit is, and how big it is."""
    header = ["system", "samples", "units/smp", "tok/unit", "mean", "p90", "max", "tok/smp"]
    rows = [header]
    for r in results:
        tu, us, ts = r["tok_per_unit"], r["units_per_sample"], r["tokens_per_sample"]
        if not tu:
            rows.append([r["label"], str(r["samples"]), "-", "-", "-", "-", "-", "-"])
            continue
        rows.append([
            r["label"], str(r["samples"]),
            f"{us['mean']:,.0f}", f"{tu['median']:,}", f"{tu['mean']:,.0f}",
            f"{tu['p90']:,}", f"{tu['max']:,}", f"{ts['mean']:,.0f}",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    print("\nunit size  (tok/unit is the median; mean/p90/max are its tail)")
    print("  " + _row(rows[0], widths))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows[1:]:
        print("  " + _row(row, widths))


def print_exchange_table(results: list[dict], topk: tuple[int, ...],
                         budgets: tuple[int, ...]) -> None:
    """What --top-k buys, and what a token budget buys back.

    The point this makes concrete: at one k, two systems put very
    different amounts of evidence in the prompt, so a table indexed by k is not
    a matched comparison.
    """
    measured = [r for r in results if r["tok_per_unit"]]
    if not measured:
        return
    header = ["system"] + [f"k={k}" for k in topk] + [f"{b // 1000}k tok" for b in budgets]
    rows = [header]
    for r in measured:
        mean = r["tok_per_unit"]["mean"]
        per_sample = r["units_per_sample"]
        # Share of samples that cannot even reach the deepest k asked for —
        # MemoryBank's coarse blocks make exact k unreachable, and that is the
        # representation, not a shortfall of the run.
        cells = [r["label"]]
        cells += [f"{round(mean * k):,}" for k in topk]
        cells += [f"{b / mean:,.0f}" for b in budgets]
        rows.append(cells)
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    print(f"\nexchange rate  (tokens bought at each k | units affordable at each budget)")
    print("  " + _row(rows[0], widths))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows[1:]:
        print("  " + _row(row, widths))

    # Reachability: a k the store cannot serve is a silently shorter prompt.
    deepest = max(topk)
    for r in measured:
        counts = r.get("_units_list") or []
        short = sum(1 for c in counts if c < deepest)
        if short:
            print(f"    !! {r['label']}: {short}/{len(counts)} samples hold fewer than "
                  f"{deepest} units — k={deepest} silently returns everything they have")


def print_provenance(results: list[dict]) -> None:
    """Whether the store can produce turn-level qrels at all."""
    print("\nevidence provenance  (has_answer, what qrels_has_answer labels from)")
    for r in results:
        if not r["tok_per_unit"]:
            continue
        total = r["tok_per_unit"]["n"]
        pct = 100.0 * r["has_answer_units"] / total if total else 0.0
        line = (f"  {r['label']:<16} {r['has_answer_units']:,} / {total:,} units "
                f"({pct:.1f}%) marked evidence")
        if r["samples_without_evidence"]:
            line += f"; {r['samples_without_evidence']} samples with none"
        print(line)
        if r["samples_without_provenance"]:
            print(f"    !! {r['samples_without_provenance']} samples predate provenance "
                  f"tracking (records_has_answer is False) — re-ingest, don't relabel")
        if r["empty_stores"]:
            shown = ", ".join(r["empty_stores"][:5])
            more = f" (+{len(r['empty_stores']) - 5} more)" if len(r["empty_stores"]) > 5 else ""
            print(f"    !! {len(r['empty_stores'])} store(s) hold zero units: {shown}{more}")


def print_examples(results: list[dict]) -> None:
    """The 'Indexed/rendered as' column — the actual string, not a description."""
    print("\nrendered form  (what the retriever ranks and the prompt shows)")
    for r in results:
        for text in r["examples"][:1]:
            snippet = text if len(text) <= 100 else text[:97] + "..."
            print(f"  {r['label']:<16} {snippet}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarise an ingest's units: size, depth, and evidence coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", action="append", metavar="SPEC.yaml",
                    help="Experiment YAML spec (repeatable — one table row per spec). "
                         "An explicit flag overrides the same key in every spec.")
    add_system_arg(ap)
    add_dataset_arg(ap)
    add_mem_model_arg(ap, help="Memory model that keys the saved-memory dir")
    add_mem_variant_arg(ap)
    ap.add_argument("--mem-dir", default=None,
                    help="Measure this directory instead of the one keyed by "
                         "system/dataset/mem-model (needs --system for the plugin)")
    add_limit_arg(ap, help="Only measure the first N complete samples — a fast read on a "
                           "500-sample store")
    ap.add_argument("--raw-text", action="store_true",
                    help="Count MemoryUnit.text alone instead of the rendered "
                         "'[<time>] <text>' the generator sees")
    ap.add_argument("--top-k", default=",".join(str(k) for k in DEFAULT_TOPK),
                    help=f"Depths to price in the exchange table (default: %(default)s)")
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS),
                    help="Token budgets to price in reverse (default: %(default)s)")
    ap.add_argument("--encoding", default="cl100k_base",
                    help="tiktoken encoding — cl100k_base matches dataset_stats (default: %(default)s)")
    ap.add_argument("--threads", type=int, default=8, help="tiktoken batch threads")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Also write the full numbers here as JSON")
    args = ap.parse_args()

    targets = resolve_targets(ap, args)
    topk = tuple(int(k) for k in args.top_k.split(",") if k.strip())
    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())

    import tiktoken
    enc = tiktoken.get_encoding(args.encoding)

    results = []
    failed = 0
    for target in targets:
        if not os.path.isdir(target["mem_dir"]):
            print(f"ERROR: no such memory dir: {target['mem_dir']}", file=sys.stderr)
            failed += 1
            continue
        print(f"reading {target['label']:<16} {target['mem_dir']}", file=sys.stderr)
        try:
            r = measure(target, enc, args.limit, args.raw_text, args.threads)
        except Exception as exc:  # a plugin whose deps are absent here
            print(f"ERROR: {target['label']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue
        results.append(r)

    if not results:
        return 1

    print("\n" + "=" * 71)
    print(f"Ingest summary: {len(results)} store(s), "
          f"{'raw text' if args.raw_text else 'rendered'}, {args.encoding}")
    print("=" * 71)
    print_units_table(results)
    print_exchange_table(results, topk, budgets)
    print_provenance(results)
    print_examples(results)
    print("=" * 71)
    print("units/smp and tok/smp are means. --top-k buys units, --context-budget buys")
    print("tokens; tok/unit converts between them.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
