#!/usr/bin/env python3
"""
evalgrid/tools/rebuild_qrels.py
===============================
Rebuild the whole ``outputs/_qrels`` tree — every naive-RAG baseline and every
memory system — in one CPU-only pass, and verify what was written.

WHY THIS EXISTS. The two qrels builders are per-system CLIs with *different*
interfaces: evalgrid.pipeline.qrels_turns takes ``--roles all|user|pair`` and has
no ``--exp``, while evalgrid.pipeline.qrels_has_answer takes ``--system`` plus a
``--mem-model`` and does. Rebuilding the grid by hand therefore means nine
invocations in two shapes with the mem-model spelled out each time, which is
where a typo silently writes a second tree under a slightly different path
segment. This driver reads the coordinates from ``experiments/<system>/<dataset>.yaml``
— the same specs every downstream stage resolves — so the tree it writes is the
tree ``retrieve``/``answer``/``eval_retrieval`` go looking for, by construction.

WHAT IT COSTS. Nothing on a GPU. No LLM, no embedder, no
vLLM: the corpus is read from saved memory that already exists, and the labels
are LongMemEval's own ``has_answer`` flags. The cost is file I/O and inodes, so
this runs fine straight from a shell on any machine. There is deliberately no
script wrapper: a CPU-only pass over files that already exist needs no server,
and a wrapper for it would be four lines
of directives around one command.

WHAT IT WRITES, per sample, under
``outputs/_qrels/<system>/<dataset>/<mem-seg>/has_answer/<sample>/``::

    corpus.jsonl                   the retrievable units
    queries.jsonl                  the sample's question(s)
    qrels/test_oracle_strict.tsv   gold "strict"  — the has_answer units
    qrels/test_oracle_turn.tsv     gold "turn"    — + evidence sessions flagging nothing
    qrels/test_oracle_session.tsv  gold "session" — every turn of every evidence session

The three gold levels are evalgrid/benchmarks/gold.py's, unchanged; this script
adds no labelling logic of its own. It is a driver and a verifier, nothing else.

PREFLIGHT, and why it is not optional. Both builders write a sample at a time
with no space check, so a full disk produces a *half-written tree*: some samples
with all three gold files, some with none, and a traceback in the middle. That
failure mode is silent downstream — eval_retrieval simply scores the samples it
finds. On a filesystem with an inode quota the same thing arrives as
``Errno 122 (EDQUOT)``, and inodes can bind before space does: this tree is ~7 inodes per sample per system, so nine systems over 500 samples is
~31.5k inodes for ~3 GB. ``--min-free-gb`` refuses to start rather than stopping
halfway.

FAILURE POLICY. A system whose builder exits non-zero is recorded and SKIPPED;
the remaining systems still run, and the exit code is non-zero at the end. The
alternative — aborting the batch — means one system's missing store costs the
other eight a rebuild they had already paid for.

Usage
-----
    # everything, straight from a shell (no GPU needed)
    python -m evalgrid.tools.rebuild_qrels --overwrite

    # one system, three samples — the smoke test
    python -m evalgrid.tools.rebuild_qrels --systems lightmem --limit 3 --overwrite

    # see the nine commands without running them
    python -m evalgrid.tools.rebuild_qrels --dry-run

    # verify a tree built earlier, run nothing
    python -m evalgrid.tools.rebuild_qrels --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # evalgrid/tools/ -> repo root
sys.path.insert(0, str(REPO))

from evalgrid.benchmarks.gold import GOLD_LEVELS, gold_filename  # noqa: E402
from evalgrid.lib.paths import labeler_segment, qrels_dir as build_qrels_dir  # noqa: E402

#: The raw-turn baselines. These are NOT plugins — they have no saved memory and
#: no --system flag on their builder; --roles picks both the corpus shape and the
#: output tree (evalgrid/pipeline/qrels_turns.py). Mapping them here is what lets
#: one --systems list name all nine trees uniformly.
ROLES_BY_SYSTEM = {"all_turns": "all", "user_turns": "user", "pair_turns": "pair"}

#: Memory systems, in ascending order of how long they take to flatten — the
#: cheap ones fail fast on a bad env before the expensive ones have run.
MEMORY_SYSTEMS = ["memorybank", "mem0", "a_mem", "lightmem", "simplemem", "hypermem"]

DEFAULT_SYSTEMS = list(ROLES_BY_SYSTEM) + MEMORY_SYSTEMS

LABELER = "has_answer"  # both builders' fixed labeler segment

#: Inodes one sample costs: the sample dir, corpus.jsonl, queries.jsonl, the
#: qrels/ dir, and the three gold files.
INODES_PER_SAMPLE = 7


# --- spec ------------------------------------------------------------------

def find_spec(system: str, dataset: str, spec_name: str | None = None) -> tuple[Path, dict]:
    """Locate and load one system's experiment spec.

    Specs are named for the BENCHMARK, not for the dataset key — the file is
    ``longmemeval.yaml`` while the key is ``longmemeval_s`` — so this discovers
    by CONTENT rather than by filename: every ``experiments/<system>/*.yaml``
    whose own ``system`` and ``dataset`` match what was asked for.

    Where a system has several (lightmem ships the default plus two ingest
    variants), the base spec is the one declaring NO ``mem_variant``. That
    matters because a variant keys a different saved store and therefore a
    different qrels tree: silently picking ``rate0.4-stm768`` would build a tree
    the default retrieve/answer runs never read. Name one explicitly with
    --spec to rebuild a variant's tree on purpose.
    """
    import yaml
    d = REPO / "experiments" / system
    if not d.is_dir():
        raise FileNotFoundError(f"no experiments dir for system {system!r} at {d}")

    if spec_name:
        cands = [d / (spec_name if spec_name.endswith(".yaml") else f"{spec_name}.yaml")]
        if not cands[0].exists():
            raise FileNotFoundError(f"no spec at {cands[0]}")
    else:
        cands = sorted(d.glob("*.yaml"))

    matches: list[tuple[Path, dict]] = []
    for path in cands:
        spec = yaml.safe_load(path.read_text()) or {}
        if spec.get("system") == system and spec.get("dataset") == dataset:
            matches.append((path, spec))
    if not matches:
        raise FileNotFoundError(
            f"no spec under {d} declares system={system!r} dataset={dataset!r} "
            f"(looked at: {', '.join(c.name for c in cands) or 'nothing'})")

    if not spec_name:
        base = [m for m in matches if not m[1].get("mem_variant")]
        if len(base) == 1:
            matches = base
        elif len(base) > 1:
            raise ValueError(
                f"{d}: {len(base)} specs declare system={system!r} dataset={dataset!r} "
                f"with no mem_variant ({', '.join(p.name for p, _ in base)}) — "
                f"pass --spec to choose one")
        elif len(matches) > 1:
            raise ValueError(
                f"{d}: every matching spec sets a mem_variant "
                f"({', '.join(p.name for p, _ in matches)}) — pass --spec to choose one")

    path, spec = matches[0]
    if "mem_model" not in spec:
        raise KeyError(f"{path}: spec is missing required key 'mem_model'")
    return path, spec


def out_dir_for(system: str, dataset: str, spec: dict) -> Path:
    """Where a system's qrels land — computed with the SAME helper the builders use.

    Not reimplemented as a path join: mem_segment() maps "none" and a slashed
    model id onto their segments and appends any mem_variant, and a second copy
    of that rule is a second thing to keep in sync with lib/paths.py.
    """
    return Path(build_qrels_dir(system, dataset, spec["mem_model"],
                                labeler_segment(LABELER),
                                mem_variant=spec.get("mem_variant")))


# --- preflight -------------------------------------------------------------

def preflight(systems: list[str], dataset: str, specs: dict[str, dict],
              n_samples: int, min_free_gb: float) -> list[str]:
    """Check space, inodes and saved memory BEFORE writing anything.

    Returns the list of blocking problems (empty means go). Saved memory that is
    merely incomplete is NOT blocking — simplemem's store legitimately covers
    353 of 500 samples because its ingest failed on the rest and recorded them
    (see outputs/ingest_failures/) — so it is reported and built.
    """
    problems: list[str] = []

    usage = shutil.disk_usage(REPO)
    free_gb = usage.free / 2**30
    print(f"  disk free      : {free_gb:.1f} GB")
    if free_gb < min_free_gb:
        problems.append(
            f"only {free_gb:.1f} GB free under {REPO}, need >= {min_free_gb:.1f} GB "
            f"(raise with --min-free-gb once you have checked; a full disk writes a "
            f"HALF-BUILT tree, which scores silently)")

    est_inodes = len(systems) * n_samples * INODES_PER_SAMPLE
    print(f"  inodes needed  : ~{est_inodes:,} ({len(systems)} systems x {n_samples} samples x {INODES_PER_SAMPLE})")
    # A cluster quota (e.g. GPFS) is not visible through statvfs, so this is
    # advisory there.
    if shutil.which("rquota"):
        print("  (run `rquota` — a GPFS quota is not visible to statvfs, and "
              "inodes block writes independently of space)")

    for system in systems:
        if system in ROLES_BY_SYSTEM:
            continue  # raw turns: corpus comes from the dataset, no store needed
        spec = specs[system]
        from evalgrid.lib.paths import memory_dir
        mem = Path(memory_dir(system, dataset, spec["mem_model"],
                              mem_variant=spec.get("mem_variant")))
        if not mem.is_dir():
            problems.append(f"{system}: no saved memory at {mem} — run evalgrid.pipeline.ingest first")
            continue
        n = sum(1 for d in mem.iterdir() if d.is_dir())
        note = "" if n >= n_samples else f"  <-- {n_samples - n} sample(s) never ingested; they will be skipped"
        print(f"  store {system:11}: {n} sample dir(s){note}")

    ds = REPO / "data" / "LongMemEval" / f"{dataset.replace('longmemeval_s', 'longmemeval_s_cleaned')}.json"
    if dataset.startswith("longmemeval") and not ds.exists():
        problems.append(f"dataset file not found at {ds}")
    return problems


# --- run -------------------------------------------------------------------

def command_for(system: str, dataset: str, spec: dict, limit: int | None,
                overwrite: bool) -> list[str]:
    """The exact argv for one system's builder.

    Two shapes, because the two builders genuinely differ: the turn baselines
    are selected by --roles (which also names their tree) and take no
    --mem-model, while the memory systems take --system and --mem-model. Kept
    explicit rather than smoothed into one interface, so this reads as what you
    would have typed.
    """
    if system in ROLES_BY_SYSTEM:
        cmd = [sys.executable, "-m", "evalgrid.pipeline.qrels_turns",
               "--dataset", dataset, "--roles", ROLES_BY_SYSTEM[system]]
    else:
        cmd = [sys.executable, "-m", "evalgrid.pipeline.qrels_has_answer",
               "--system", system, "--dataset", dataset,
               "--mem-model", spec["mem_model"]]
        # A variant keys its own store AND its own qrels tree; omitting it here
        # would flatten a variant rebuild onto the default tree.
        if spec.get("mem_variant"):
            cmd += ["--mem-variant", spec["mem_variant"]]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def run_system(system: str, cmd: list[str], log_dir: Path | None) -> tuple[bool, float]:
    """Run one builder, streaming its output; return (ok, seconds).

    Output is streamed rather than captured so a job log shows progress while it
    runs — these are 500-sample passes and a silent hour is indistinguishable
    from a hang.
    """
    print(f"\n{'=' * 78}\n== {system}\n== {' '.join(cmd)}\n{'=' * 78}", flush=True)
    start = time.time()
    log_path = log_dir / f"qrels_{system}.log" if log_dir else None
    try:
        if log_path:
            with open(log_path, "w") as log:
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                rc = proc.wait()
        else:
            rc = subprocess.call(cmd, cwd=REPO)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - report and keep the batch going
        print(f"  [FAIL] {system}: {exc}", file=sys.stderr)
        return False, time.time() - start
    elapsed = time.time() - start
    if rc != 0:
        print(f"  [FAIL] {system}: builder exited {rc} — skipping, batch continues",
              file=sys.stderr)
    return rc == 0, elapsed


# --- verify ----------------------------------------------------------------

def verify(system: str, out_dir: Path) -> dict | None:
    """Read the tree back and count what is actually on disk.

    Deliberately independent of what the builder PRINTED: a half-written tree
    from a full disk still prints a cheerful summary for the samples it managed,
    and the point of this pass is to catch exactly that. Counts rows, not bytes,
    so a truncated final line shows up as a missing label rather than passing.
    """
    if not out_dir.is_dir():
        return None
    samples = sorted(d for d in out_dir.iterdir() if d.is_dir())
    if not samples:
        return None
    totals = {lvl: 0 for lvl in GOLD_LEVELS}
    empty = {lvl: 0 for lvl in GOLD_LEVELS}
    docs = missing = 0
    for s in samples:
        corpus = s / "corpus.jsonl"
        if corpus.exists():
            with open(corpus) as f:
                docs += sum(1 for _ in f)
        else:
            missing += 1
        for lvl in GOLD_LEVELS:
            p = s / "qrels" / gold_filename(lvl)
            if not p.exists():
                missing += 1
                continue
            with open(p) as f:
                n = sum(1 for i, line in enumerate(f) if i > 0 and line.strip())
            totals[lvl] += n
            if n == 0:
                empty[lvl] += 1
    return {"system": system, "samples": len(samples), "docs": docs,
            "totals": totals, "empty": empty, "missing": missing}


def print_report(rows: list[dict], failures: list[str], timings: dict[str, float]) -> None:
    print(f"\n\n{'=' * 100}\nREBUILT QRELS — what is on disk\n{'=' * 100}")
    head = (f"{'system':12} {'samples':>7} {'corpus':>9} | "
            + " ".join(f"{lvl:>8}" for lvl in GOLD_LEVELS)
            + f" | {'empty@strict':>12} {'empty@turn':>10} {'missing':>7} {'mins':>6}")
    print(head)
    print("-" * len(head))
    for r in rows:
        mins = timings.get(r["system"], 0) / 60
        flag = "  <-- MISSING FILES" if r["missing"] else ""
        print(f"{r['system']:12} {r['samples']:7} {r['docs']:9} | "
              + " ".join(f"{r['totals'][lvl]:8}" for lvl in GOLD_LEVELS)
              + f" | {r['empty']['strict']:12} {r['empty']['turn']:10} "
                f"{r['missing']:7} {mins:6.1f}{flag}")
    if failures:
        print(f"\nFAILED (skipped, batch continued): {', '.join(failures)}")


# --- main ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Rebuild every qrels tree (naive-RAG baselines + memory systems), CPU-only.")
    p.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS),
                   help="Comma-separated systems to rebuild (default: all nine). "
                        f"Turn baselines: {', '.join(ROLES_BY_SYSTEM)}. "
                        f"Memory: {', '.join(MEMORY_SYSTEMS)}.")
    p.add_argument("--dataset", default="longmemeval_s",
                   help="Benchmark dataset key. Specs are matched on their own "
                        "'dataset:' value, not on filename (the file is "
                        "longmemeval.yaml while the key is longmemeval_s). "
                        "Default: longmemeval_s")
    p.add_argument("--spec", default=None, metavar="NAME.yaml",
                   help="Spec filename inside experiments/<system>/ to use instead of "
                        "the auto-discovered base spec — e.g. "
                        "longmemeval_rate0.4_stm768.yaml to rebuild a variant's tree. "
                        "Applied to every system in --systems, so pair it with a "
                        "single --systems when the variant exists for only one.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only build the first N samples per system (smoke test)")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild samples that already have qrels. WITHOUT this, an "
                        "existing sample is skipped, so a tree built before the oracle "
                        "levels existed keeps its strict-only qrels and is NOT upgraded.")
    p.add_argument("--min-free-gb", type=float, default=5.0,
                   help="Refuse to start below this much free space (default: 5)")
    p.add_argument("--log-dir", default=None,
                   help="Also tee each system's output to <dir>/qrels_<system>.log")
    p.add_argument("--dry-run", action="store_true", help="Print the commands, run nothing")
    p.add_argument("--verify-only", action="store_true",
                   help="Skip building; just report what is already on disk")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Bypass the space/inode/store checks (not recommended)")
    args = p.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in DEFAULT_SYSTEMS]
    if unknown:
        p.error(f"unknown system(s): {unknown}. Choose from {DEFAULT_SYSTEMS}")

    print(f"Repo    : {REPO}")
    print(f"Python  : {sys.executable}")
    print(f"Dataset : {args.dataset}")
    print(f"Systems : {', '.join(systems)}")
    if args.limit:
        print(f"Limit   : {args.limit} sample(s) per system")

    try:
        found = {s: find_spec(s, args.dataset, args.spec) for s in systems}
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\nspec error: {exc}", file=sys.stderr)
        return 2
    specs = {s: spec for s, (_, spec) in found.items()}
    for s, (path, spec) in found.items():
        variant = f"  mem_variant={spec['mem_variant']}" if spec.get("mem_variant") else ""
        print(f"  spec {s:11}: {path.relative_to(REPO)}{variant}")

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Before preflight: --dry-run writes nothing, so a space check cannot be a
    # reason to refuse to SHOW the commands — that is precisely what you want to
    # look at while deciding what to clear.
    if args.dry_run:
        print("\nWould run:")
        for s in systems:
            print("  " + " ".join(command_for(s, args.dataset, specs[s], args.limit, args.overwrite)))
        for s in systems:
            print(f"  -> {out_dir_for(s, args.dataset, specs[s]).relative_to(REPO)}")
        return 0

    if not args.verify_only and not args.skip_preflight:
        print("\nPreflight")
        problems = preflight(systems, args.dataset, specs, args.limit or 500,
                             args.min_free_gb)
        if problems:
            sys.stdout.flush()  # keep stdout/stderr ordered in a job log
            print("\nBLOCKED — nothing was written:", file=sys.stderr)
            for prob in problems:
                print(f"  - {prob}", file=sys.stderr)
            sys.stderr.flush()
            return 2
        print("  ok")

    failures: list[str] = []
    timings: dict[str, float] = {}
    if not args.verify_only:
        if not args.overwrite:
            print("\nNOTE: --overwrite not given; samples that already have qrels are "
                  "SKIPPED. To add the oracle gold levels to a strict-only tree, "
                  "rerun with --overwrite.")
        for s in systems:
            ok, secs = run_system(s, command_for(s, args.dataset, specs[s], args.limit,
                                                 args.overwrite), log_dir)
            timings[s] = secs
            if not ok:
                failures.append(s)

    rows = []
    for s in systems:
        r = verify(s, out_dir_for(s, args.dataset, specs[s]))
        if r is None:
            print(f"  [warn] {s}: nothing on disk to verify", file=sys.stderr)
            if s not in failures:
                failures.append(s)
            continue
        rows.append(r)
    print_report(rows, failures, timings)

    broken = [r["system"] for r in rows if r["missing"]]
    if broken:
        print(f"\nIncomplete tree(s): {', '.join(broken)} — rerun those with --overwrite",
              file=sys.stderr)
    return 1 if (failures or broken) else 0


if __name__ == "__main__":
    sys.exit(main())
