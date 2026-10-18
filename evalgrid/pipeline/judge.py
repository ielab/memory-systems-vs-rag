"""
evalgrid/pipeline/judge.py
--------------------
LLM-as-judge scoring of a run's predictions.

Each benchmark uses its own canonical judge (evalgrid/benchmarks/judges/): LoCoMo's
generous CORRECT/WRONG prompt, LongMemEval's official per-question-type prompts.
THE JUDGE IS gpt-4o-mini, and it is held fixed across every system, every
retriever and every budget cell — a judge that varies between rows is a second
independent variable in a table that is supposed to have one. It is also
deliberately != the generator (Qwen3-30B-A3B) to avoid self-grading bias. The
flag stays configurable for a labelled sensitivity check only, and the metrics
file is keyed by the judge slug, so a second judge's scores sit beside the
first's instead of overwriting them.

Usage:
  # Point at a evalgrid.pipeline.answer predictions file and name the dataset:
  python -m evalgrid.pipeline.judge \
      --predictions outputs/_answers/lightmem/longmemeval_s/<mem>/bm25/<gen>/top10/predictions.jsonl \
      --dataset longmemeval_s --judge-model gpt-4o-mini

  # Or pass a run directory that also holds a config.json (dataset is read from it):
  python -m evalgrid.pipeline.judge --run <run-dir>/ --judge-model gpt-4o-mini

  # --batch grades via OpenAI's Batch API instead of live calls — cheaper and
  # free of the per-minute rate limit, a good fit when there are many
  # predictions to judge. Requires an OpenAI --judge-model and no --judge-base-url:
  python -m evalgrid.pipeline.judge --predictions <preds>.jsonl --dataset longmemeval_s \
      --judge-model gpt-4o-mini --batch

Writes metrics.<judge-slug>.json next to predictions.jsonl (so two judges'
scores of the same run coexist instead of clobbering) and prints a summary.
"""

import argparse
import json
import os
from types import SimpleNamespace

from evalgrid.lib.cli import add_exp_arg, apply_spec
from evalgrid.lib.manifest import write_manifest
from evalgrid.lib.paths import model_slug

from evalgrid.benchmarks.adapters import build_adapter
from evalgrid.benchmarks.judges import build_judge
from evalgrid.models.factory import _OPENAI_GEN_PREFIXES
from evalgrid.lib.llm import DEFAULT_SEED, DEFAULT_TEMPERATURE, build_generator


def _gold_map(dataset: str) -> dict:
    """Map query_id -> Query for the dataset (gold answers + categories)."""
    adapter = build_adapter(dataset)
    out = {}
    for sample in adapter.samples():
        for query in sample.queries:
            out[query.query_id] = query
    return out


def _load_predictions(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge scoring of predictions")
    add_exp_arg(parser)
    parser.add_argument("--run", default=None,
                        help="Run directory containing predictions.jsonl + config.json")
    parser.add_argument("--predictions", default=None,
                        help="Path to predictions.jsonl (if not using --run)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset key (required with --predictions; inferred from --run's config.json)")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Judge LLM (default: gpt-4o-mini — THE project judge, held fixed "
                             "across every row). Keep it different from the generator. The "
                             "default used to be gpt-4o, so a hand-run judge command that "
                             "omitted this flag scored one cell with a different judge than "
                             "the driver used for every other cell.")
    parser.add_argument("--judge-base-url", default=None,
                        help="Override base URL for the judge (e.g. http://localhost:8000/v1)")
    parser.add_argument("--judge-reasoning-effort", default=None,
                        help="GPT-5/o-series only: none|low|medium|high|xhigh. "
                             "Ignored by models that don't support it. Lower is cheaper/faster.")
    parser.add_argument("--judge-verbosity", default=None,
                        help="GPT-5 only: low|medium|high (constrains output length)")
    parser.add_argument("--judge-seed", type=int, default=DEFAULT_SEED,
                        help=f"Best-effort reproducibility seed for the judge (OpenAI chat "
                             f"models). Pinned to {DEFAULT_SEED} by default.")
    parser.add_argument("--judge-temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature for the judge (OpenAI and vLLM chat models; "
                             f"gpt-5 family ignores it). Pinned to {DEFAULT_TEMPERATURE} "
                             f"(greedy) by default.")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent judge calls (default: 8). Ignored under --batch.")
    parser.add_argument("--batch", action="store_true",
                        help="Grade via OpenAI's Batch API (submit a job file, poll, then parse "
                             "results) instead of live synchronous calls. Sidesteps the per-minute "
                             "rate limit and costs half as much — a good fit when judging many "
                             "predictions and latency doesn't matter. Requires an OpenAI "
                             "--judge-model and no --judge-base-url; ignores --concurrency.")
    parser.add_argument("--batch-poll-interval", type=int, default=30,
                        help="Seconds between status checks while waiting on a batch job (default: 30)")
    parser.add_argument("--batch-phase", choices=["both", "submit", "collect"], default="both",
                        help="Under --batch: 'both' (default) submit then wait; 'submit' submit the "
                             "job and exit without waiting (fire many files' batches concurrently); "
                             "'collect' wait on an already-submitted batch and write its metrics. "
                             "run_all_judge.sh uses submit-then-collect across all files so the jobs "
                             "run in parallel on OpenAI's side.")
    parser.add_argument("--judge-max-tokens", type=int, default=1024,
                        help="Per-call output budget for the judge (default: 1024). "
                             "Raise it for reasoning judges that 'think' before answering.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge the first N predictions (quick test)")
    parser.add_argument("--exclude-question-type", action="append", default=None,
                        metavar="TYPE",
                        help="Skip predictions whose gold question_type is this value "
                             "(e.g. single-session-assistant). Repeatable.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Judge ONLY the items a previous pass could not get a verdict for "
                             "(read from metrics.<judge-slug>.json's judge_failures), reusing "
                             "every existing verdict, then rewrite the metrics file complete. "
                             "Runs LIVE regardless of --batch: the stragglers are a handful of "
                             "calls, and a second Batch job would mean another queue wait for "
                             "them. This is the intended way to finish a batch that came back "
                             "with failures — never re-judge the whole file for a few items.")
    parser.add_argument("--out", default=None,
                        help="Output metrics path (default: metrics.<judge-slug>.json next to predictions)")
    parser.add_argument("--no-verdict-log", dest="verdict_log", action="store_false",
                        help="Do not write judge_verdicts.<judge-slug>.jsonl. The log is the "
                             "only record of WHY each score is what it is (the judge's own "
                             "`reason` field, and on BEAM the event-ordering alignment's "
                             "YES/NO calls), and it costs nothing to produce — the text is "
                             "already generated and paid for. Turn it off only when the disk "
                             "matters more than being able to check a number.")
    args = parser.parse_args()
    apply_spec(parser, args)

    if args.batch:
        if args.judge_base_url:
            parser.error("--batch is OpenAI-hosted only — incompatible with --judge-base-url")
        if not args.judge_model.startswith(_OPENAI_GEN_PREFIXES):
            parser.error(f"--batch requires an OpenAI --judge-model (got {args.judge_model!r})")

    # Resolve predictions path + dataset.
    if args.run:
        run_dir = args.run
        predictions_path = os.path.join(run_dir, "predictions.jsonl")
        dataset = args.dataset
        cfg_path = os.path.join(run_dir, "config.json")
        if dataset is None and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                dataset = json.load(f).get("dataset")
    elif args.predictions:
        predictions_path = args.predictions
        run_dir = os.path.dirname(predictions_path)
        dataset = args.dataset
    else:
        parser.error("Provide either --run or --predictions")

    if not dataset:
        parser.error("Could not determine --dataset (not in config.json; pass it explicitly)")
    if not os.path.exists(predictions_path):
        parser.error(f"Predictions not found: {predictions_path}")

    out_path = args.out or os.path.join(run_dir, f"metrics.{model_slug(args.judge_model)}.json")

    # Join predictions with gold answers by question_id.
    gold = _gold_map(dataset)
    predictions = _load_predictions(predictions_path)
    if args.limit:
        predictions = predictions[:args.limit]

    excluded_qt = set(args.exclude_question_type or [])
    items = []
    missing = excluded = 0
    for row in predictions:
        qid = row.get("question_id")
        query = gold.get(qid)
        if query is None:
            missing += 1
            continue
        if excluded_qt and query.question_type in excluded_qt:
            excluded += 1
            continue
        items.append((query, row.get("hypothesis", "")))

    print(f"Dataset     : {dataset}")
    print(f"Predictions : {predictions_path}  ({len(items)} judged"
          + (f", {excluded} excluded" if excluded else "")
          + (f", {missing} unmatched" if missing else "") + ")")
    print(f"Judge model : {args.judge_model}")

    # base_url routes to the vllm backend; reasoning_effort/verbosity/seed are
    # OpenAI-only — build_generator forwards them through the tolerant factory,
    # which drops them for a vLLM/Claude judge. No per-backend guard.
    judge_llm = build_generator(
        args.judge_model, base_url=args.judge_base_url,
        reasoning_effort=args.judge_reasoning_effort, verbosity=args.judge_verbosity,
        seed=args.judge_seed, temperature=args.judge_temperature,
    )
    judge = build_judge(dataset, judge_llm, concurrency=args.concurrency,
                        max_tokens=args.judge_max_tokens)

    if args.retry_failed:
        # Reuse every verdict already on disk and judge only what has none, then
        # rebuild the whole metrics file from the merged labels — so the output
        # is a normal complete metrics file, not a patch a reader has to know
        # about. Nothing here re-sends a call that already succeeded.
        if not os.path.exists(out_path):
            parser.error(f"--retry-failed needs an existing {out_path} to read its "
                         f"failures from — judge the file first.")
        with open(out_path) as f:
            prior = json.load(f)
        by_qid = {r["question_id"]: r for r in prior.get("per_question", [])}
        todo = [i for i, (query, _) in enumerate(items)
                if not by_qid.get(query.query_id, {}).get("judged", False)]
        if not todo:
            print(f"\nNothing to retry — every item in {out_path} already has a verdict.")
            return
        print(f"Retrying    : {len(todo)} unjudged item(s) of {len(items)} (live, "
              f"concurrency {args.concurrency})")
        new_labels, new_errors = judge.label_items([items[i] for i in todo])

        # Prior verdicts first, then overwrite the retried slots. An item absent
        # from the prior file (predictions grew since) counts as unjudged and is
        # picked up by `todo` above, so every slot ends up filled.
        # "score" is the float every judge now writes; "correct" is the bool a
        # binary judge writes beside it, and the only thing metrics files
        # predating the graded judges carry. Reading score first keeps a resume
        # exact on a graded run and byte-identical on every older file.
        labels: list[float | None] = [
            by_qid.get(q.query_id, {}).get("score",
                                           by_qid.get(q.query_id, {}).get("correct"))
            for q, _ in items]
        errors: list[str | None] = [None if by_qid.get(q.query_id, {}).get("judged")
                                    else by_qid.get(q.query_id, {}).get("error")
                                    for q, _ in items]
        for slot, label, error in zip(todo, new_labels, new_errors):
            labels[slot] = label
            errors[slot] = error

        # Tokens are cumulative across passes: the prior file's plus this pass's,
        # or the cost of a retried file would read as the cost of the retry alone.
        prior_tok = prior.get("judge_tokens", {})
        now = judge.judge.token_usage
        usage = SimpleNamespace(
            input_tokens=prior_tok.get("input_tokens", 0) + now.input_tokens,
            output_tokens=prior_tok.get("output_tokens", 0) + now.output_tokens,
            calls=prior_tok.get("calls", 0) + now.calls,
        )
        metrics = judge.build_metrics(items, labels, usage, errors)
        recovered = len(todo) - metrics["overall"]["failed"]
        print(f"Recovered   : {recovered} of {len(todo)}")
    elif args.batch:
        import openai

        from evalgrid.benchmarks.judge_batch import (
            collect_batch_judging,
            run_batch_judging,
            submit_batch_judging,
        )
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        work_dir = os.path.join(run_dir, f"_batch_judge_{model_slug(args.judge_model)}")
        if args.batch_phase == "submit":
            # Fire-and-exit: queue the job, write no metrics. A later
            # --batch-phase collect run waits on it and writes the metrics.
            submit_batch_judging(
                client, judge, args.judge_model, items, work_dir=work_dir,
                max_tokens=args.judge_max_tokens, reasoning_effort=args.judge_reasoning_effort,
                verbosity=args.judge_verbosity, seed=args.judge_seed,
            )
            print(f"\n[submit] batch queued for {predictions_path} "
                  f"— collect later with --batch-phase collect")
            return
        if args.batch_phase == "collect":
            metrics = collect_batch_judging(
                client, judge, items, work_dir=work_dir, poll_interval=args.batch_poll_interval)
        else:
            metrics = run_batch_judging(
                client, judge, args.judge_model, items, work_dir=work_dir,
                poll_interval=args.batch_poll_interval, max_tokens=args.judge_max_tokens,
                reasoning_effort=args.judge_reasoning_effort, verbosity=args.judge_verbosity,
                seed=args.judge_seed,
            )
    else:
        metrics = judge.score(items)
    metrics["dataset"] = dataset
    metrics["judge_model"] = args.judge_model
    if args.judge_reasoning_effort:
        metrics["judge_reasoning_effort"] = args.judge_reasoning_effort
    if args.judge_verbosity:
        metrics["judge_verbosity"] = args.judge_verbosity
    if args.judge_seed is not None:
        metrics["judge_seed"] = args.judge_seed
    metrics["judge_temperature"] = args.judge_temperature
    metrics["unmatched_predictions"] = missing
    if excluded_qt:
        metrics["excluded_question_types"] = sorted(excluded_qt)

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # The machine-readable half of a failure, beside the metrics that excluded
    # it — the same role failures.jsonl plays for ingest (benchmarks/runner.py).
    # OVERWRITTEN, not appended: unlike an ingest failure, this one is a property
    # of the LAST judging pass, and appending would leave a --retry-failed run
    # looking as though the items it recovered were still broken.
    failures_path = os.path.join(
        run_dir, f"judge_failures.{model_slug(args.judge_model)}.jsonl")
    if metrics["judge_failures"]:
        with open(failures_path, "w") as f:
            for rec in metrics["judge_failures"]:
                f.write(json.dumps({"predictions": predictions_path,
                                    "judge_model": args.judge_model, **rec}) + "\n")
    elif os.path.exists(failures_path):
        os.remove(failures_path)

    # The judge's own words, beside the score they produced. Written for the
    # same reason judge_failures.jsonl is: the metrics file records WHAT was
    # decided and nothing about WHY, so a surprising cell used to be
    # un-investigable except by paying for the calls a second time and hoping
    # the judge repeated itself. One line per question, in the order the
    # protocol asked. Not folded into metrics.json — every table builder reads
    # that file whole, and a BEAM cell's ~1,051 reasons would multiply it by ~50.
    #
    # A --batch-phase submit run writes nothing here (it makes no calls), and a
    # collect run writes the full log, because resolve_batch records as it
    # parses. Live and batch therefore produce the same records from the same
    # replies, which is the property the batch path exists to keep.
    verdicts_path = os.path.join(
        run_dir, f"judge_verdicts.{model_slug(args.judge_model)}.jsonl")
    if args.verdict_log:
        # --retry-failed judges only the items that had no verdict, so its
        # judge.verdicts covers those alone. Merge onto the prior log rather
        # than replacing it, or a recovered file would lose the explanation of
        # every item the first pass got right.
        merged: dict[str, list[dict]] = {}
        if args.retry_failed and os.path.exists(verdicts_path):
            with open(verdicts_path) as f:
                for line in f:
                    rec = json.loads(line)
                    merged[rec["question_id"]] = rec["calls"]
        merged.update(judge.verdicts)
        order = [q.query_id for q, _ in items]
        seen = set()
        with open(verdicts_path, "w") as f:
            for qid in order:
                if qid in seen or qid not in merged:
                    continue
                seen.add(qid)
                f.write(json.dumps({"question_id": qid,
                                    "judge_model": args.judge_model,
                                    "calls": merged[qid]}) + "\n")

    write_manifest(run_dir, args, stage="judge",
                   filename=f"manifest.judge.{model_slug(args.judge_model)}.json")

    ov = metrics["overall"]
    lines = [
        "\nAccuracy",
        f"  overall: {ov['accuracy']:.4f} ({ov['correct']}/{ov['total']})",
    ]
    for cat, b in metrics["by_category"].items():
        lines.append(f"    {cat:<28} {b['accuracy']:.4f} ({b['correct']}/{b['total']})")
    print("\n".join(lines))
    print(f"\nMetrics written to {out_path}")
    if args.verdict_log:
        print(f"Judge verdicts (score + reason per call) in {verdicts_path}")
        print(f"  read them: python -m evalgrid.tools.show_verdicts {verdicts_path} "
              f"--worst 5")

    # Said last and said plainly, because this is the number that decides whether
    # the file above may be quoted. An accuracy over 487 of 500 items is not the
    # accuracy the row claims, and nothing in the directory name says so.
    if ov["failed"]:
        print(f"\n!!! INCOMPLETE: {ov['failed']} of {ov['attempted']} item(s) got NO "
              f"VERDICT and are excluded from the accuracy above.")
        print(f"!!!   They are NOT counted wrong — the denominator is {ov['total']}, "
              f"not {ov['attempted']}.")
        print(f"!!!   Listed in {failures_path}")
        print(f"!!!   Finish them (live, no second batch wait) with:")
        print(f"!!!     python -m evalgrid.pipeline.judge --predictions {predictions_path} \\")
        print(f"!!!         --dataset {dataset} --judge-model {args.judge_model} --retry-failed")


if __name__ == "__main__":
    main()
