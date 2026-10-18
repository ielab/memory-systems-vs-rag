"""
evalgrid/pipeline/eval_retrieval.py
----------------------------
Scores a retriever's run (evalgrid/pipeline/retrieve.py output) against its qrels
ground truth (the qrels builder's output) with pytrec_eval.

Reads:
  outputs/_qrels/<system>/<dataset>/<mem-model>/<labeler-segment>/<sample>/qrels/test_oracle_<level>.tsv
  (which level: --gold, default "turn")
  outputs/_retrieval/<system>/<dataset>/<mem-model>/<labeler-segment>/<retriever>/<sample>/run.tsv
Writes:
  outputs/_eval/<system>/<dataset>/<mem-model>/<labeler-segment>/<retriever>/metrics-gold-turn.json
  (the default --gold is "turn"; --gold strict writes the plain metrics.json)

  A SIBLING tree of the run it scores, not a file inside it — so scores can be
  synced back from the cluster without the run.tsv files (see paths.eval_dir).
  Passing --retrieval-dir keeps the score beside that run instead, which is what
  a throwaway or fused run wants; --out overrides both.

--labeler-model and --candidate-source must match whatever the qrels builder
and evalgrid/pipeline/retrieve.py were run with — together they form <labeler-segment>
(<labeler-model>, plus a "-retrieval" suffix when --candidate-source retrieval),
which keys both the qrels dir and the retrieval run dir.

Queries with an empty qrels (e.g. LongMemEval abstention questions — nothing
in the haystack answers them, so nothing is "relevant" by design) never get a
qrels row at the chosen level and are excluded from scoring automatically.

Usage:
  python -m evalgrid.pipeline.eval_retrieval --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --retriever bm25
"""

import argparse
import json
import os
import sys

from evalgrid.lib.cli import (
    add_candidate_source_arg,
    add_dataset_arg,
    add_emb_model_arg,
    add_exp_arg,
    add_labeler_model_arg,
    add_mem_model_arg,
    add_mem_variant_arg,
    add_retriever_arg,
    add_system_arg,
    apply_spec,
    require_resolved,
)
from evalgrid.lib.manifest import write_manifest
from evalgrid.lib.paths import ExperimentPaths, labeler_segment, retriever_name

from evalgrid.benchmarks.adapters import DATASETS
from evalgrid.benchmarks.gold import (ALL_GOLD_LEVELS, EVAL_GOLD_LEVEL,
                                      gold_filename, gold_suffix)
from evalgrid.benchmarks.retrieval import evaluate, read_jsonl, read_qrels, read_run


def _usage_count(usage_path: str, field: str) -> int:
    """One coverage counter out of retrieval_usage.json.

    Reads the counter rather than inferring one, and treats an unreadable or
    older usage file as "nothing reported" — a run written before the field
    existed is not evidence of a skip, and must not start failing eval
    retroactively.
    """
    try:
        with open(usage_path) as f:
            return int(json.load(f).get(field) or 0)
    except (OSError, ValueError, TypeError):
        return 0


def _no_memory_samples(usage_path: str) -> int:
    """Samples the retrieve loop stepped over for having no saved store."""
    return _usage_count(usage_path, "samples_no_memory")


def _failed_samples(usage_path: str) -> int:
    """Samples the loop reached, tried, and could not rank.

    The third way a native run can finish short, and the newest: since the loop
    gained per-sample isolation it no longer dies on a failure, it records the
    sample and carries on. That is the right behaviour and it opens exactly the
    hole the two checks below already close — a run that finishes, writes its
    usage file, and leaves some samples with no run.tsv. Without this the cell
    would be scored over the samples it managed, under a run name that says
    otherwise.
    """
    return _usage_count(usage_path, "samples_failed")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a retrieval run against its qrels")
    add_exp_arg(parser)
    add_system_arg(parser, help="Memory system the qrels/run came from (e.g. lightmem)")
    add_dataset_arg(parser, choices=list(DATASETS), help="Benchmark dataset key")
    add_mem_model_arg(parser, help="Memory model used to build the qrels dataset")
    add_mem_variant_arg(parser)
    add_labeler_model_arg(parser)
    add_retriever_arg(parser, required=False, include_native=True,
                      help="Retriever whose run to evaluate ('native' = the run "
                           "evalgrid.pipeline.answer --retriever native --prompt native wrote from the system's own loop)")
    add_emb_model_arg(parser,
                      help="Model used for --retriever dense/sparse/multi_vector (to locate its run dir)")
    parser.add_argument("--run-name", default=None,
                        help="Evaluate this run dir name directly (e.g. a fused run from "
                             "a run dir written outside the normal sweep) instead of deriving it from "
                             "--retriever/--emb-model")
    add_candidate_source_arg(parser)
    parser.add_argument("--gold", default=EVAL_GOLD_LEVEL, choices=list(ALL_GOLD_LEVELS),
                        help=f"Which evidence set to score the SAME ranking against "
                             f"(evalgrid/benchmarks/gold.py). Default {EVAL_GOLD_LEVEL!r}: it "
                             f"credits the 94 evidence sessions LongMemEval names but never "
                             f"flags, which 'strict' scores as distractors and, for 21 "
                             f"questions, as an EMPTY gold set (recall undefined, not zero). "
                             f"'strict' is the ingester-stamped gold, what every "
                             f"pre-levels metrics file holds; each level reads its own file "
                             f"beside the corpus and writes metrics-gold-<level>.json, so "
                             f"no level overwrites another. Rankings do not depend on "
                             f"labels, so changing this "
                             f"never needs a retrieval re-run — but recall denominators differ "
                             f"per level, so never put two levels in one column without "
                             f"saying so.")
    parser.add_argument("--qrels-dir", default=None,
                        help="Override the qrels dataset dir "
                             "(default: outputs/_qrels/<system>/<dataset>/<mem-model>/<labeler-segment>/)")
    parser.add_argument("--retrieval-dir", default=None,
                        help="Override the retrieval run dir "
                             "(default: outputs/_retrieval/<system>/<dataset>/<mem-model>/<labeler-segment>/<retriever>/)")
    parser.add_argument("--k-values", default="5,10,20",
                        help="Comma-separated cutoffs for nDCG/Recall (default: 5,10,20)")
    parser.add_argument("--out", default=None,
                        help="Output metrics path. Default: the keyed eval cell, "
                             "outputs/_eval/<system>/<dataset>/<mem-seg>/<labeler-seg>/"
                             "<retriever>/metrics<-gold-level>.json — except when "
                             "--retrieval-dir names a run outside the keyed tree, where "
                             "the score stays beside that run.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Score a run dir whose retrieve stage did not finish "
                             "(no retrieval_usage.json). Off by default: a crashed cell's "
                             "metrics are computed over however many samples it got to, "
                             "which is a different measurement under the same name")
    args = parser.parse_args()
    apply_spec(parser, args)
    require_resolved(parser, args, "system", "dataset", "mem_model")
    if not args.run_name and not args.retriever:
        parser.error("pass --retriever (with --emb-model for model-backed ones) or --run-name")

    k_values = tuple(int(k) for k in args.k_values.split(","))

    paths = ExperimentPaths(system=args.system, dataset=args.dataset, mem_model=args.mem_model,
                            mem_variant=args.mem_variant)
    seg = labeler_segment(args.labeler_model, args.candidate_source)
    qrels_dir = args.qrels_dir or paths.qrels(seg)
    rname = args.run_name or retriever_name(args.retriever, args.emb_model)
    retrieval_dir = args.retrieval_dir or paths.retrieval(rname)
    # WHERE THE SCORE LANDS, in three cases:
    #
    #   --out given            exactly there. The drivers pass it (via
    #                          evalgrid.tools.stage_dir --stage eval) and so pin
    #                          the filename, gold suffix included.
    #   --retrieval-dir given  beside the run named, keeping the historical
    #                          default. A run dir named explicitly is one OUTSIDE
    #                          the keyed tree — a throwaway (test_retrieval.sh), a
    #                          fused run, an analysis scratch tree — and its
    #                          --system/--retriever say what it resembles, not
    #                          where it lives. Publishing it into the keyed cell
    #                          for that retriever would overwrite the real cell's
    #                          metrics with a score of a different run.
    #   neither                the keyed _eval cell (paths.eval_dir). This USED to
    #                          be <retrieval-dir>/metrics<suffix>.json, which meant
    #                          a hand-run eval and a driver-run one put the same
    #                          measurement in two different trees — and only one of
    #                          them is the tree the analysis reads.
    #
    # The tree is a sibling rather than a subdirectory so scores can be synced
    # back from the cluster without the run.tsv files, which are orders of
    # magnitude larger; see paths.eval_dir.
    if args.out:
        out_path = args.out
    elif args.retrieval_dir:
        out_path = os.path.join(retrieval_dir, f"metrics{gold_suffix(args.gold)}.json")
    else:
        out_path = os.path.join(paths.eval(seg, rname),
                                f"metrics{gold_suffix(args.gold)}.json")

    if not os.path.isdir(qrels_dir):
        sys.exit(f"No qrels dataset at {qrels_dir} — run evalgrid.pipeline.qrels_has_answer first.")
    if not os.path.isdir(retrieval_dir):
        sys.exit(f"No retrieval run at {retrieval_dir} — run evalgrid/pipeline/retrieve.py first.")

    # A run dir that its own stage never finished writing holds however many
    # samples it reached, and scoring it publishes a number under the failed
    # cell's name that reads exactly like a finished one. That is how the
    # 2026-08-21 smoke grid published nDCG for two cells that had died of CUDA
    # OOM: they ranked 2 of 4 samples, and the metrics printed under the same run
    # name as their 4-sample neighbours with nothing but a "! 2 sample(s) have
    # qrels but no run.tsv" line between them.
    #
    # NOT a count check against the qrels: retrieving a --limit N slice of a
    # larger qrels set is a legitimate partial, and one that would false-positive
    # on every smoke run. What separates the two is whether the writing stage
    # RAN OUT, and each half of retrieve.py leaves a different trace of that.
    #   native  retrieval_usage.json, written after the sample loop returns —
    #           so its absence means the loop did not return.
    #           Since the loop also survives a failed sample, the same file's
    #           samples_failed is the second native signal: the loop returned,
    #           but not with everything it was asked for.
    #   flat    errors.jsonl, appended per failed sample. The flat half collects
    #           failures and carries on rather than dying, so here the evidence
    #           is a file that should NOT be there.
    #
    # The native half deliberately does NOT gate on errors.jsonl even though it
    # writes one now: that file is appended and never truncated, so a cell whose
    # failure was fixed by a later run would stay unscorable forever. The
    # counter is rewritten each run and therefore describes the run that last
    # touched this directory, which is the question being asked here.
    # Both are what those paths already write; neither needs a run re-done to
    # acquire a new marker.
    native_run = args.retriever == "native" or os.path.basename(
        retrieval_dir.rstrip("/")).startswith("native-")
    incomplete = None
    usage_path = os.path.join(retrieval_dir, "retrieval_usage.json")
    if native_run and not os.path.isfile(usage_path):
        incomplete = ("no retrieval_usage.json, which evalgrid/pipeline/retrieve.py writes "
                      "only once its sample loop finishes — the run crashed partway")
    elif native_run and _no_memory_samples(usage_path):
        # The loop can also RETURN having skipped samples: one whose store is
        # missing is reported and stepped over, so the run finishes, writes its
        # usage file, and still leaves those samples with no run.tsv. That is
        # the same partial as a crash and it is invisible in every other
        # artefact — hence a counter in the file rather than a count against
        # the qrels, which would false-positive on every --limit run.
        incomplete = (f"retrieval_usage.json reports {_no_memory_samples(usage_path)} "
                      f"sample(s) skipped for having no saved memory, so the run finished "
                      f"without ranking them — ingest did not cover this qrels set")
    elif native_run and _failed_samples(usage_path):
        incomplete = (f"retrieval_usage.json reports {_failed_samples(usage_path)} "
                      f"sample(s) that FAILED to retrieve, so the run finished without "
                      f"ranking them — see errors.jsonl in this directory for which and "
                      f"why. Re-running the cell retries exactly those")
    elif not native_run and os.path.isfile(os.path.join(retrieval_dir, "errors.jsonl")):
        incomplete = ("errors.jsonl is present, so some samples failed to retrieve and "
                      "have no ranking here")
    if incomplete and not args.allow_partial:
        sys.exit(
            f"Incomplete retrieval run at {retrieval_dir} — {incomplete}. Anything scored "
            f"from it would cover only the samples it reached, under a run name that says "
            f"otherwise. Check the stage log, re-run the retrieve stage (it skips what "
            f"already succeeded), or pass --allow-partial to score it as-is."
        )

    print(f"System    : {args.system}")
    print(f"Dataset   : {args.dataset}")
    print(f"Mem model : {args.mem_model}")
    print(f"Retriever : {rname}")
    print(f"Qrels dir : {qrels_dir}")
    print(f"Gold      : {args.gold} ({gold_filename(args.gold)})")
    print(f"Run dir   : {retrieval_dir}")

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    categories: dict[str, str] = {}
    missing_run = 0
    for sample_id in sorted(os.listdir(qrels_dir)):
        qrels_path = os.path.join(qrels_dir, sample_id, "qrels", gold_filename(args.gold))
        queries_path = os.path.join(qrels_dir, sample_id, "queries.jsonl")
        run_path = os.path.join(retrieval_dir, sample_id, "run.tsv")
        if not os.path.isfile(qrels_path):
            continue
        qrels.update(read_qrels(qrels_path))
        for q in read_jsonl(queries_path):
            categories[q["_id"]] = (q.get("metadata") or {}).get("question_type") or "unknown"
        if os.path.isfile(run_path):
            run.update(read_run(run_path))
        else:
            missing_run += 1

    if missing_run:
        print(f"  ! {missing_run} sample(s) have qrels but no run.tsv — "
              f"run evalgrid/pipeline/retrieve.py for them first (excluded below).")

    metrics = evaluate(qrels, run, k_values=k_values, categories=categories)
    metrics["system"] = args.system
    metrics["dataset"] = args.dataset
    metrics["mem_model"] = args.mem_model
    metrics["retriever"] = args.retriever or rname
    metrics["gold"] = args.gold

    os.makedirs(retrieval_dir, exist_ok=True)
    # out_path's own parent, which is NOT retrieval_dir whenever --out points
    # somewhere else — an OUT_ROOT eval tree (outputs/_smoke4_eval/<system>/...)
    # is a sibling of the run dir, not inside it. Without this the scoring ran,
    # the metrics were computed, and the write died with FileNotFoundError,
    # which reads like a missing INPUT rather than an uncreated output dir.
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    # Record the RESOLVED paths, not the flags as typed. The manifest is the
    # "what actually ran" snapshot, and a null flag records only "the default",
    # which is a moving target: cells scored before the default moved to the
    # _eval tree carry a null --out that means the retrieval dir. Stamping means
    # no later reader (evalgrid.tools.rescore_eval replays this manifest) has to
    # know which era a cell is from, or to re-derive three paths from six keys to
    # replay one command. A run that took every default used to record none of
    # them, and rescore_eval called it NO-ARGS — unreplayable, for a cell that
    # had run perfectly well.
    args.out = out_path
    args.qrels_dir = qrels_dir
    args.retrieval_dir = retrieval_dir
    write_manifest(retrieval_dir, args, stage="eval_retrieval")

    ov = metrics["overall"]
    lines = [f"\nRetrieval metrics ({ov['queries']} queries)"]
    for k, v in ov.items():
        if k != "queries":
            lines.append(f"  {k:<12} {v:.4f}")
    for cat, b in metrics["by_category"].items():
        lines.append(f"\n  {cat} ({b['queries']} queries)")
        for k, v in b.items():
            if k != "queries":
                lines.append(f"    {k:<12} {v:.4f}")
    print("\n".join(lines))
    print(f"\nMetrics written to {out_path}")


if __name__ == "__main__":
    main()
