#!/usr/bin/env python3
"""
check_retrieval.py
==================
Report whether every retrieval run a ``run_all_retrieval.sh`` sweep is meant to
produce has actually finished — without re-running anything.

It re-derives the exact same set of retrievers the driver dispatches over (the
spec's RETRIEVERS list) from the *same* experiment spec, computes each one's
retrieval output dir via ``evalgrid.lib.paths`` (the single source of truth the
retrieve stage itself uses), and compares the sample-ids that already have a
``run.tsv`` against the set the stage would attempt.

Unlike the answer stage (keyed per question-id), retrieval's unit of work is the
*sample*: retrieve.py ranks every query in a sample against that sample's corpus
and writes one ``<sample>/run.tsv``. Top-k plays no part here — a run.tsv is a
full ranking; the cutoff is applied later at answer/eval time. So completeness is
measured in samples, not question-ids.

A retriever config is:
  * COMPLETE  — every expected sample has a run.tsv
  * PARTIAL   — some samples have a run.tsv but others are missing (interrupted /
                still resumable) — how many are shown
  * MISSING   — no run.tsv at all (never started)
  * ERRORS    — an errors.jsonl sits beside the runs (some samples failed); shown
                as a suffix on whatever state above applies

"Expected" mirrors evalgrid/pipeline/retrieve.py exactly: every qrels sample dir
(one with a corpus.jsonl), minus samples that have *any* query whose
metadata.question_type is in --exclude-question-type. Note this is a whole-sample
drop — retrieve.py excludes the sample, not just the offending query (which is why
this differs from check_answers.py's per-question exclusion).

Usage
-----
    # defaults mirror the driver: the has_answer spec, no excluded question types.
    python evalgrid/tools/check_retrieval.py \
        --spec experiments/lightmem/longmemeval.yaml

    python evalgrid/tools/check_retrieval.py -v      # list every missing sample-id

Exit code is 0 only when every retriever is COMPLETE with no errors, so it can
gate a follow-up stage (e.g. run_all_answers.sh) in a script.
"""

from __future__ import annotations  # PEP 604 (str | None) at def time on Py<3.10

import argparse
import os
import sys
from pathlib import Path

# evalgrid/tools/ -> repo root; make `scripts` importable when run as a file.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.lib.paths import (  # noqa: E402
    MODEL_BASED_RETRIEVERS,
    ExperimentPaths,
    labeler_segment,
    retriever_name,
)

# Mirrors run_all_retrieval.sh's EXCLUDE_QUESTION_TYPES, which is now empty: no
# sample is dropped, so every sample is "expected". Pass
# --exclude-question-type explicitly to check a subset run.
DEFAULT_EXCLUDE_QUESTION_TYPES = ()


def load_spec(spec_path: Path) -> dict:
    """Read the experiment YAML into the same keys load_spec.sh exports."""
    import yaml

    return yaml.safe_load(spec_path.read_text()) or {}


def parse_retriever_spec(spec: str) -> tuple[str, str | None]:
    """Turn a RETRIEVERS entry into (retriever, emb_model), matching the .sh.

    Accepts ``bm25``/``oracle`` and ``<kind>:MODEL[@url]`` for every model-backed
    kind ``paths.MODEL_BASED_RETRIEVERS`` names. The optional ``@base-url`` is
    irrelevant to the output path and is stripped.

    The accepted kinds are READ from that set rather than listed again here.
    They used to be listed — ``("dense", "fusion")`` — which was a subset of the
    six paths.py already knew, so this tool raised ``unknown retriever spec`` on
    a spec the pipeline runs perfectly well: ``sparse:naver/splade-v3`` and all
    three ``rrf:`` entries, i.e. half of simplemem's grid was uncheckable. Same
    failure as our development notes — a parallel list that fell behind the real one —
    so the list is gone rather than corrected.
    """
    if spec in ("bm25", "oracle"):
        return spec, None
    kind, sep, rest = spec.partition(":")
    if sep and kind in MODEL_BASED_RETRIEVERS:
        return kind, rest.split("@", 1)[0]
    raise ValueError(
        f"unknown retriever spec: {spec!r} (expected 'bm25', 'oracle', or "
        f"'<kind>:MODEL[@url]' with kind in {sorted(MODEL_BASED_RETRIEVERS)})")


def read_jsonl(path: Path):
    """Yield parsed rows from a .jsonl file (skipping blank lines)."""
    import json

    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def expected_sample_ids(qrels_dir: Path, excluded: set[str]) -> set[str] | None:
    """Sample-ids the retrieve stage would attempt for one config.

    Mirrors retrieve.py's sample loop: every qrels sample dir (a dir with a
    corpus.jsonl), minus samples where *any* query's metadata.question_type is
    excluded — retrieve.py drops the whole sample, not just that query.

    Returns None when the expected set can't be determined here — the qrels tree
    has no sample dirs (dataset not built on this machine). Callers must not
    treat None as "0 expected".
    """
    if not qrels_dir.is_dir():
        return None
    sample_dirs = [d for d in sorted(qrels_dir.iterdir()) if (d / "corpus.jsonl").is_file()]
    if not sample_dirs:
        return None  # qrels dataset not built here — can't say what's expected

    ids: set[str] = set()
    for sample_dir in sample_dirs:
        queries = sample_dir / "queries.jsonl"
        if excluded and queries.is_file():
            qtypes = {(q.get("metadata") or {}).get("question_type") for q in read_jsonl(queries)}
            if qtypes & excluded:
                continue  # sample has an excluded question type — dropped whole
        ids.add(sample_dir.name)
    return ids


def done_sample_ids(retrieval_dir: Path) -> set[str]:
    """Sample-ids that already have a run.tsv under this retriever's output dir."""
    if not retrieval_dir.is_dir():
        return set()
    return {p.parent.name for p in retrieval_dir.glob("*/run.tsv")}


def build_configs(spec: dict) -> list[dict]:
    """Every (label, retrieval_dir, qrels_dir) the sweep produces — one per retriever."""
    system = spec["system"]
    dataset = spec["dataset"]
    mem_model = spec.get("mem_model")
    labeler_model = spec.get("labeler_model")
    candidate_source = spec.get("candidate_source", "sweep")
    retrievers = spec.get("retrievers") or []
    seg = labeler_segment(labeler_model, candidate_source)

    configs: list[dict] = []
    for entry in retrievers:
        retriever, emb_model = parse_retriever_spec(entry)
        rname = retriever_name(retriever, emb_model)
        paths = ExperimentPaths(system=system, dataset=dataset, mem_model=mem_model, emb_model=emb_model)
        configs.append({
            "label": entry,
            "retrieval_dir": Path(paths.retrieval(rname)),
            "qrels_dir": Path(paths.qrels(seg)),
        })
    return configs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="experiments/lightmem/longmemeval.yaml",
                    help="Experiment spec YAML (same file run_all_retrieval.sh reads)")
    ap.add_argument("--exclude-question-type", action="append", default=None, metavar="TYPE",
                    help="Question type the sweep excluded (repeatable). "
                         "Default matches run_all_retrieval.sh.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="List the missing sample-ids for each PARTIAL config")
    ap.add_argument("--debug", action="store_true",
                    help="Print the computed qrels/retrieval paths and sample counts "
                         "for each config (diagnoses path mismatches)")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = REPO / spec_path
    if not spec_path.is_file():
        print(f"ERROR: spec not found: {spec_path}", file=sys.stderr)
        return 2

    excluded = set(args.exclude_question_type or DEFAULT_EXCLUDE_QUESTION_TYPES)

    spec = load_spec(spec_path)
    configs = build_configs(spec)

    print(f"spec        : {spec_path.relative_to(REPO) if spec_path.is_relative_to(REPO) else spec_path}")
    print(f"system      : {spec.get('system')}  dataset: {spec.get('dataset')}  mem_model: {spec.get('mem_model')}")
    print(f"excluded qt : {', '.join(sorted(excluded)) or '(none)'}")
    print("=" * 70)

    n_complete = n_partial = n_missing = n_present = n_errors = 0
    width = max((len(c["label"]) for c in configs), default=0)
    for c in configs:
        errors = c["retrieval_dir"] / "errors.jsonl"
        exp = expected_sample_ids(c["qrels_dir"], excluded)
        got = done_sample_ids(c["retrieval_dir"])
        missing = (exp - got) if exp is not None else set()
        n_err = sum(1 for _ in read_jsonl(errors)) if errors.is_file() else 0

        if not got:
            state, n_missing = "MISSING", n_missing + 1
            suffix = "  (no run.tsv)"
        elif exp is None:
            # Can't verify completeness here — report what we can see instead.
            state, n_present = "PRESENT", n_present + 1
            suffix = f"  ({len(got)} retrieved; expected set unavailable locally)"
        elif missing:
            state, n_partial = "PARTIAL", n_partial + 1
            suffix = f"  ({len(got)}/{len(exp)} retrieved, {len(missing)} left)"
        else:
            state, n_complete = "COMPLETE", n_complete + 1
            suffix = f"  ({len(got)}/{len(exp)} retrieved)" if exp else "  (0 expected)"

        if n_err:
            suffix += f"  [!{n_err} errors.jsonl]"
            n_errors += 1

        print(f"{state:8}  {c['label']:<{width}}{suffix}")
        if args.debug:
            qd, rd = c["qrels_dir"], c["retrieval_dir"]
            n_samples = sum(1 for d in qd.iterdir() if (d / "corpus.jsonl").is_file()) if qd.is_dir() else 0
            print(f"            qrels_dir     : {qd}  (is_dir={qd.is_dir()}, samples={n_samples})")
            print(f"            retrieval_dir : {rd}  (is_dir={rd.is_dir()}, run.tsv={len(got)})")
            print(f"            expected      : {'None (unavailable)' if exp is None else len(exp)}")
        if args.verbose and missing:
            for sid in sorted(missing):
                print(f"            - {sid}")

    print("=" * 70)
    total = len(configs)
    parts = [f"{n_complete} complete", f"{n_partial} partial", f"{n_missing} missing"]
    if n_present:
        parts.append(f"{n_present} present-unverified")
    if n_errors:
        parts.append(f"{n_errors} with errors")
    print(f"{total} retriever(s): " + ", ".join(parts))

    # Hard failure only for states we're sure are unfinished. PRESENT means the
    # runs are there but the qrels tree needed to confirm coverage isn't on this
    # machine (run this where the full outputs are for a full check).
    unfinished = n_partial + n_missing + n_errors
    if unfinished:
        print("NOT ALL FINISHED")
        return 1
    if n_present:
        print("ALL PRESENT (completeness unverified — run where the full outputs are with the "
              "qrels outputs to confirm coverage)")
        return 0
    print("ALL FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
