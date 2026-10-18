#!/usr/bin/env python3
"""
check_answers.py
================
Report whether every answer run a ``run_all_answers.sh`` sweep is meant to
produce has actually finished — without re-running anything.

It re-derives the exact same set of configs the driver dispatches over
(RETRIEVERS x TOP_K_VALUES, plus oracle/full) from the *same* experiment spec,
computes each one's ``predictions.jsonl`` path via ``evalgrid.lib.paths`` (the
single source of truth the answer stage itself uses), and compares the
question-ids already answered against the set the stage would attempt.

A config is:
  * COMPLETE  — predictions.jsonl covers every expected question-id
  * PARTIAL   — predictions.jsonl exists but is missing some (interrupted /
                still resumable) — how many are shown
  * MISSING   — no predictions.jsonl at all (never started)
  * ERRORS    — an errors.jsonl sits beside it (some questions failed); shown
                as a suffix on whatever state above applies

"Expected" mirrors evalgrid/pipeline/answer.py exactly: queries under the qrels
tree, minus --exclude-question-type, minus (for retriever configs) samples that
have no run.tsv yet — those aren't the answer stage's to produce.

Usage
-----
    # defaults mirror the answer sweep: the has_answer spec,
    # TOP_K_VALUES=3,5,6,10,20, INCLUDE_ORACLE at the driver default (1).
    python evalgrid/tools/check_answers.py \
        --spec experiments/lightmem/longmemeval.yaml \
        --top-k-values 3,5,6,10,20 --include-oracle

    # the token grid — a SECOND invocation, not a wider first one: a budget run
    # writes to budget<T>/ instead of top<k>/, so the two cell sets never share
    # a call. The four budgets are the paper's grid (check_grid.py's
    # DEFAULT_CELLS carries the same list for the native tree).
    python evalgrid/tools/check_answers.py \
        --spec experiments/lightmem/longmemeval.yaml \
        --context-budgets 200,500,1000,2000 --no-oracle

    python evalgrid/tools/check_answers.py -v          # list every question-id gap

Exit code is 0 only when every config is COMPLETE with no errors, so it can gate
a follow-up stage (e.g. run_all_judge.sh) in a script.
"""

from __future__ import annotations  # PEP 604 (str | None) at def time on Py<3.10

import argparse
import json
import os
import sys
from pathlib import Path

# evalgrid/tools/ -> repo root; make `scripts` importable when run as a file.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.lib.paths import (  # noqa: E402
    MODEL_BASED_RETRIEVERS,
    ExperimentPaths,
    answer_depth_segment,
    labeler_segment,
    retriever_name,
)
from evalgrid.benchmarks.gold import GOLD_LEVELS, EVAL_GOLD_LEVEL, gold_suffix  # noqa: E402

# Mirrors run_all_answers.sh's EXCLUDE_QUESTION_TYPES, which is now empty: all
# 500 questions are answered, so all 500 are "expected". Pass
# --exclude-question-type explicitly to check a subset run.
DEFAULT_EXCLUDE_QUESTION_TYPES = ()


def load_spec(spec_path: Path) -> dict:
    """Read the experiment YAML into the same keys load_spec.sh exports."""
    import yaml

    d = yaml.safe_load(spec_path.read_text()) or {}
    return d


def parse_retriever_spec(spec: str) -> tuple[str, str | None]:
    """Turn a RETRIEVERS entry into (retriever, emb_model), matching the .sh.

    Accepts ``bm25``/``oracle`` and ``<kind>:MODEL[@url]`` for every model-backed
    kind ``paths.MODEL_BASED_RETRIEVERS`` names. The optional ``@base-url`` is
    irrelevant to the output path and is stripped.

    The accepted kinds are READ from that set rather than listed again here.
    They used to be listed — ``("dense", "fusion", "sparse", "hybrid")`` — which
    omitted ``rrf``, so this tool raised ``unknown retriever spec`` on the three
    ``rrf:`` rows every system's grid runs. Its sibling in check_retrieval.py
    carried a DIFFERENT subset of the same set, which is the tell: three copies
    of one list, none of them right. A parallel list falls behind the real one, so the
    list is gone rather than corrected.
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
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def answered_ids(predictions: Path) -> set[str]:
    if not predictions.is_file():
        return set()
    return {r.get("question_id") for r in read_jsonl(predictions)}


def expected_ids(qrels_dir: Path, retrieval_dir: Path | None, excluded: set[str]) -> set[str] | None:
    """Question-ids the answer stage would attempt for one config.

    Mirrors answer.py's item loop: every sample's queries minus excluded
    question types, and — for real retriever runs — only samples that already
    have a run.tsv (oracle/full pass retrieval_dir=None to skip that gate).

    Returns None when the expected set can't be determined from local outputs —
    the qrels tree has no sample dirs, or a retriever run's retrieval tree is
    absent entirely (e.g. a partial mirror where only answers were synced back).
    Callers must not treat None as "0 expected".
    """
    if not qrels_dir.is_dir():
        return None
    sample_dirs = [d for d in sorted(qrels_dir.iterdir()) if (d / "corpus.jsonl").is_file()]
    if not sample_dirs:
        return None  # qrels dataset not built here — can't say what's expected
    if retrieval_dir is not None and not retrieval_dir.is_dir():
        return None  # retrieval stage output absent — can't apply the run.tsv gate

    ids: set[str] = set()
    for sample_dir in sample_dirs:
        if retrieval_dir is not None and not (retrieval_dir / sample_dir.name / "run.tsv").is_file():
            continue  # qrels but no run.tsv — not the answer stage's to fill
        queries = sample_dir / "queries.jsonl"
        if not queries.is_file():
            continue
        for q in read_jsonl(queries):
            qtype = (q.get("metadata") or {}).get("question_type")
            if qtype in excluded:
                continue
            ids.add(q["_id"])
    return ids


def build_configs(spec: dict, args) -> list[dict]:
    """Every (label, predictions_path, qrels_dir, retrieval_dir) the sweep produces."""
    system = spec["system"]
    dataset = spec["dataset"]
    mem_model = spec.get("mem_model")
    gen_model = os.environ.get("GEN_MODEL") or spec.get("gen_model")
    labeler_model = spec.get("labeler_model")
    candidate_source = spec.get("candidate_source", "sweep")
    retrievers = spec.get("retrievers") or []
    seg = labeler_segment(labeler_model, candidate_source)

    configs: list[dict] = []
    for entry in retrievers:
        retriever, emb_model = parse_retriever_spec(entry)
        rname = retriever_name(retriever, emb_model)
        paths = ExperimentPaths(system=system, dataset=dataset, mem_model=mem_model, emb_model=emb_model)
        retrieval_dir = Path(paths.retrieval(rname))
        # Mirrors the driver: CONTEXT_BUDGETS replaces the depth grid when set.
        if args.context_budgets:
            depths = [(answer_depth_segment(None, int(t)), f"budget{t}") for t in args.context_budgets]
        else:
            depths = [(answer_depth_segment(int(k)), f"top{k}") for k in args.top_k_values]
        for seg_name, label in depths:
            configs.append({
                "label": f"{entry}@{label}",
                "answers_dir": Path(paths.answers(rname, gen_model, seg_name)),
                "qrels_dir": Path(paths.qrels(seg)),
                "retrieval_dir": retrieval_dir,
            })

    paths = ExperimentPaths(system=system, dataset=dataset, mem_model=mem_model)
    if args.include_oracle:
        # The gold level is part of the oracle's path, so it has to be part of
        # the path checked FOR. answer.py appends gold_suffix() and the default
        # is EVAL_GOLD_LEVEL, so looking under the bare labeler segment would
        # report a present oracle as missing.
        adir = Path(paths.answers("oracle", gen_model, "all",
                                  labeler_seg=seg + gold_suffix(args.gold)))
        configs.append({
            "label": "oracle",
            "answers_dir": adir,
            "qrels_dir": Path(paths.qrels(seg)),
            "retrieval_dir": None,  # oracle answers from qrels, no run.tsv gate
        })
    if args.include_full:
        adir = Path(paths.answers("full", gen_model, "all"))
        configs.append({
            "label": "full",
            "answers_dir": adir,
            "qrels_dir": Path(paths.qrels(seg)),
            "retrieval_dir": None,  # full answers from the whole corpus
        })
    return configs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="experiments/lightmem/longmemeval.yaml",
                    help="Experiment spec YAML (same file run_all_answers.sh reads)")
    ap.add_argument("--top-k-values", default="3,5,6,10,20",
                    help="Comma-separated top-k values the sweep answered (default: 3,5,6,10,20)")
    ap.add_argument("--context-budgets", default="",
                    help="Comma-separated token budgets the sweep answered "
                         "(CONTEXT_BUDGETS in run_all_answers.sh). When set it replaces the "
                         "top-k grid, exactly as the driver does — a budget run writes to "
                         "budget<T>/, not top<k>/. The grid is 200,500,1000,2000; no default "
                         "here, because the driver has none either.")
    ap.add_argument("--gold", default=EVAL_GOLD_LEVEL, choices=list(GOLD_LEVELS),
                    help=f"Gold level the oracle run was answered at, which keys its "
                         f"path (default: {EVAL_GOLD_LEVEL}, matching the driver)")
    ap.add_argument("--include-oracle", dest="include_oracle", action="store_true", default=True,
                    help="Also check the oracle upper-bound run (default: on, matches the driver)")
    ap.add_argument("--no-oracle", dest="include_oracle", action="store_false",
                    help="Skip the oracle run (use if the sweep ran INCLUDE_ORACLE=0)")
    ap.add_argument("--include-full", action="store_true",
                    help="Also check the full-context baseline (only if the sweep ran INCLUDE_FULL=1)")
    ap.add_argument("--exclude-question-type", action="append", default=None, metavar="TYPE",
                    help="Question type the sweep excluded (repeatable). "
                         "Default matches run_all_answers.sh.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="List the missing question-ids for each PARTIAL config")
    ap.add_argument("--debug", action="store_true",
                    help="Print the computed qrels/retrieval/answers paths and gate "
                         "results for each config (diagnoses PRESENT/path mismatches)")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = REPO / spec_path
    if not spec_path.is_file():
        print(f"ERROR: spec not found: {spec_path}", file=sys.stderr)
        return 2

    args.top_k_values = [k.strip() for k in args.top_k_values.split(",") if k.strip()]
    args.context_budgets = [t.strip() for t in args.context_budgets.split(",") if t.strip()]
    excluded = set(args.exclude_question_type or DEFAULT_EXCLUDE_QUESTION_TYPES)

    spec = load_spec(spec_path)
    configs = build_configs(spec, args)

    print(f"spec        : {spec_path.relative_to(REPO) if spec_path.is_relative_to(REPO) else spec_path}")
    print(f"gen_model   : {os.environ.get('GEN_MODEL') or spec.get('gen_model')}")
    depth_desc = (f"budgets     : {','.join(args.context_budgets)}"
                  if args.context_budgets else f"top-k       : {','.join(args.top_k_values)}")
    print(f"{depth_desc}"
          f"  oracle={'on' if args.include_oracle else 'off'}"
          f"  full={'on' if args.include_full else 'off'}")
    print(f"excluded qt : {', '.join(sorted(excluded)) or '(none)'}")
    print("=" * 70)

    n_complete = n_partial = n_missing = n_present = n_errors = 0
    width = max((len(c["label"]) for c in configs), default=0)
    for c in configs:
        predictions = c["answers_dir"] / "predictions.jsonl"
        errors = c["answers_dir"] / "errors.jsonl"
        exp = expected_ids(c["qrels_dir"], c["retrieval_dir"], excluded)
        got = answered_ids(predictions)
        missing = (exp - got) if exp is not None else set()
        n_err = sum(1 for _ in read_jsonl(errors)) if errors.is_file() else 0

        if not predictions.is_file():
            state, n_missing = "MISSING", n_missing + 1
            suffix = "  (no predictions.jsonl)"
        elif exp is None:
            # Can't verify completeness here — report what we can see instead.
            state, n_present = "PRESENT", n_present + 1
            suffix = f"  ({len(got)} answered; expected set unavailable locally)"
        elif missing:
            state, n_partial = "PARTIAL", n_partial + 1
            suffix = f"  ({len(got)}/{len(exp)} answered, {len(missing)} left)"
        else:
            state, n_complete = "COMPLETE", n_complete + 1
            suffix = f"  ({len(got)}/{len(exp)} answered)" if exp else "  (0 expected)"

        if n_err:
            suffix += f"  [!{n_err} errors.jsonl]"
            n_errors += 1

        print(f"{state:8}  {c['label']:<{width}}{suffix}")
        if args.debug:
            rd = c["retrieval_dir"]
            n_runs = (sum(1 for _ in rd.glob("*/run.tsv")) if rd is not None and rd.is_dir() else 0)
            n_samples = sum(1 for d in c["qrels_dir"].iterdir()
                            if (d / "corpus.jsonl").is_file()) if c["qrels_dir"].is_dir() else 0
            print(f"            qrels_dir     : {c['qrels_dir']}  (is_dir={c['qrels_dir'].is_dir()}, samples={n_samples})")
            print(f"            retrieval_dir : {rd}"
                  + ("" if rd is None else f"  (is_dir={rd.is_dir()}, run.tsv={n_runs})"))
            print(f"            answers_dir   : {c['answers_dir']}  (predictions={predictions.is_file()})")
            print(f"            expected      : {'None (unavailable)' if exp is None else len(exp)}")
        if args.verbose and missing:
            for qid in sorted(missing):
                print(f"            - {qid}")

    print("=" * 70)
    total = len(configs)
    parts = [f"{n_complete} complete", f"{n_partial} partial", f"{n_missing} missing"]
    if n_present:
        parts.append(f"{n_present} present-unverified")
    if n_errors:
        parts.append(f"{n_errors} with errors")
    print(f"{total} config(s): " + ", ".join(parts))

    # Hard failure only for states we're sure are unfinished. PRESENT means the
    # file is there but the qrels/retrieval trees needed to confirm coverage
    # aren't on this machine (run this where the full outputs are for a full check).
    unfinished = n_partial + n_missing + n_errors
    if unfinished:
        print("NOT ALL FINISHED")
        return 1
    if n_present:
        print("ALL PRESENT (completeness unverified — run where the full outputs are with the "
              "qrels/retrieval outputs to confirm coverage)")
        return 0
    print("ALL FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
