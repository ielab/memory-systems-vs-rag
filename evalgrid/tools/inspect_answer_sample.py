"""
evalgrid/tools/inspect_answer_sample.py
---------------------------------------
Run the answer stage for ONE sample and print the exact input prompt (and,
unless --no-generate, the generated output). A debugging companion to
evalgrid.pipeline.answer: it locates the qrels/run dirs the same way (same flags,
same labeler-segment), assembles the top-k facts with the same format_fact /
the same prompt builder, but processes a single sample so you can eyeball what the
generator actually sees instead of grepping predictions.jsonl.

Nothing is written — this only reads the qrels/retrieval trees and (optionally)
calls the generator once per query.

Usage:
  # lightmem + the has_answer qrels, dense top-5, just show the prompt (no LLM call)
  python -m evalgrid.tools.inspect_answer_sample \\
      --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --labeler-model has_answer --candidate-source sweep \\
      --retriever dense --emb-model Qwen/Qwen3-Embedding-0.6B \\
      --top-k 5 --no-generate

  # oracle facts, and actually generate the answer against a served model
  python -m evalgrid.tools.inspect_answer_sample \\
      --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --labeler-model has_answer --retriever oracle \\
      --gen-model Qwen/Qwen3-30B-A3B-Instruct-2507 --gen-base-url http://localhost:8000/v1

A specific sample is chosen with --sample-id; otherwise the first sample (sorted)
in the qrels dir is used.
"""

import argparse
import asyncio
import os
import sys

from evalgrid.lib.cli import (
    add_candidate_source_arg,
    add_dataset_arg,
    add_emb_model_arg,
    add_exp_arg,
    add_labeler_model_arg,
    add_mem_model_arg,
    add_retriever_arg,
    add_system_arg,
    apply_spec,
    require_resolved,
)
from evalgrid.lib.paths import ExperimentPaths, labeler_segment, retriever_name

from evalgrid.benchmarks.retrieval import read_jsonl, read_qrels, read_run
from evalgrid.benchmarks.gold import EVAL_GOLD_LEVEL, GOLD_LEVELS, gold_filename
from evalgrid.lib.llm import DEFAULT_SEED, DEFAULT_TEMPERATURE, build_generator

# Reuse the pipeline's exact fact/prompt assembly so what we print is byte-for-byte
# what evalgrid.pipeline.answer would feed the generator.
from evalgrid.pipeline.answer import format_fact, resolve_prompt


def _facts_for_query(qid, mode, run, corpus_facts, corpus, top_k):
    """Replicate evalgrid.pipeline.answer's per-query fact selection."""
    if mode == "full":
        return [corpus_facts[c["_id"]] for c in corpus]
    if mode == "oracle":
        return [corpus_facts[doc_id] for doc_id, score in run.get(qid, {}).items()
                if score > 0 and doc_id in corpus_facts]
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
    return [corpus_facts[doc_id] for doc_id, _ in ranked[:top_k] if doc_id in corpus_facts]


def main():
    parser = argparse.ArgumentParser(
        description="Run the answer stage for ONE sample and print the prompt + output")
    add_exp_arg(parser)
    add_system_arg(parser, help="Memory system the qrels/run came from (e.g. lightmem)")
    add_dataset_arg(parser, help="Benchmark dataset key (used only to derive default paths)")
    add_mem_model_arg(parser, help="Memory model used to build the qrels dataset")
    add_labeler_model_arg(parser,
                          help="Labeler segment to locate qrels/retrieval dirs (e.g. has_answer)")
    add_candidate_source_arg(parser)
    add_retriever_arg(parser, include_oracle=True, include_full=True,
                      help="Retriever whose run.tsv supplies the facts, 'oracle' (gold qrels) "
                           "or 'full' (whole corpus)")
    add_emb_model_arg(parser, help="Model for --retriever dense/sparse/multi_vector (to locate its run dir)")
    parser.add_argument("--sample-id", default=None,
                        help="Sample to inspect (default: the first sample in the qrels dir)")
    parser.add_argument("--gold", default=EVAL_GOLD_LEVEL, choices=list(GOLD_LEVELS),
                        help=f"--retriever oracle only: which evidence set to show "
                             f"(default: {EVAL_GOLD_LEVEL}, matching pipeline/answer.py)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k facts to use as context (default: 10; ignored for oracle/full)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Only print the prompt — don't call the generator")
    parser.add_argument("--gen-model", default="gpt-4o-mini",
                        help="Generator model (default: gpt-4o-mini). Ignored with --no-generate.")
    parser.add_argument("--gen-base-url", default=None,
                        help="Override base URL for the generator (e.g. http://localhost:8000/v1)")
    parser.add_argument("--gen-reasoning-effort", default=None)
    parser.add_argument("--gen-verbosity", default=None, choices=["low", "medium", "high"])
    parser.add_argument("--gen-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gen-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--qrels-dir", default=None, help="Override the qrels dataset dir")
    parser.add_argument("--retrieval-dir", default=None, help="Override the retrieval run dir")
    args = parser.parse_args()
    apply_spec(parser, args)
    require_resolved(parser, args, "system", "dataset", "mem_model")

    oracle = args.retriever == "oracle"
    full = args.retriever == "full"
    mode = "oracle" if oracle else "full" if full else "retriever"
    no_retrieval = oracle or full
    rname = retriever_name(args.retriever, args.emb_model)
    paths = ExperimentPaths(system=args.system, dataset=args.dataset, mem_model=args.mem_model)
    seg = labeler_segment(args.labeler_model, args.candidate_source)
    qrels_dir = args.qrels_dir or paths.qrels(seg)
    retrieval_dir = args.retrieval_dir or paths.retrieval(rname)

    if not os.path.isdir(qrels_dir):
        sys.exit(f"No qrels dataset at {qrels_dir} — run the qrels stage first.")
    if not no_retrieval and not os.path.isdir(retrieval_dir):
        sys.exit(f"No retrieval run at {retrieval_dir} — run evalgrid.pipeline.retrieve first.")

    # Pick the sample: explicit --sample-id, else the first one with a corpus.
    if args.sample_id:
        sample_id = args.sample_id
        if not os.path.isfile(os.path.join(qrels_dir, sample_id, "corpus.jsonl")):
            sys.exit(f"Sample {sample_id!r} has no corpus.jsonl under {qrels_dir}")
    else:
        candidates = sorted(
            d for d in os.listdir(qrels_dir)
            if os.path.isfile(os.path.join(qrels_dir, d, "corpus.jsonl"))
        )
        if not candidates:
            sys.exit(f"No samples with a corpus.jsonl under {qrels_dir}")
        sample_id = candidates[0]

    print(f"System    : {args.system}")
    print(f"Dataset   : {args.dataset}")
    print(f"Mem model : {args.mem_model}")
    print(f"Labeler   : {seg}")
    print(f"Retriever : {rname}" + ("" if no_retrieval else f" (top-{args.top_k})"))
    print(f"Sample    : {sample_id}")
    print(f"Qrels dir : {qrels_dir}")
    if not no_retrieval:
        print(f"Run dir   : {retrieval_dir}")

    sample_dir = os.path.join(qrels_dir, sample_id)
    corpus = read_jsonl(os.path.join(sample_dir, "corpus.jsonl"))
    corpus_facts = {c["_id"]: format_fact(c) for c in corpus}

    if oracle:
        # Same evidence set pipeline/answer.py's oracle would read, so what this
        # shows is what that run would actually be given.
        run = read_qrels(os.path.join(sample_dir, "qrels", gold_filename(args.gold)))
    elif full:
        run = None
    else:
        run_path = os.path.join(retrieval_dir, sample_id, "run.tsv")
        if not os.path.isfile(run_path):
            sys.exit(f"No run.tsv for sample {sample_id!r} at {run_path} — run retrieve first.")
        run = read_run(run_path)

    queries = read_jsonl(os.path.join(sample_dir, "queries.jsonl"))
    print(f"Corpus    : {len(corpus)} facts; {len(queries)} query(ies) in this sample\n")

    generator = None
    if not args.no_generate:
        generator = build_generator(
            args.gen_model, base_url=args.gen_base_url,
            reasoning_effort=args.gen_reasoning_effort, verbosity=args.gen_verbosity,
            seed=args.gen_seed, temperature=args.gen_temperature,
        )

    for q in queries:
        qid = q["_id"]
        meta = q.get("metadata") or {}
        question_date = meta.get("question_date")
        facts = _facts_for_query(qid, mode, run, corpus_facts, corpus, args.top_k)
        messages = resolve_prompt(args.system)(q["text"], question_date, facts)

        print("=" * 80)
        print(f"QUERY {qid}  (question_type={meta.get('question_type')})")
        print(f"  facts in context: {len(facts)}")
        print("-" * 80)
        print("INPUT PROMPT (messages — verbatim, exactly what the generator receives):")
        for m in messages:
            print(f"\n[{m['role']}]\n{m['content']}")
        print("-" * 80)
        # Readability-only breakdown: the verbatim prompt above joins facts with
        # '\n' (and inlines the first after 'memories:'), which is hard to scan.
        # This numbered list is a debug view, NOT what's sent to the model.
        print("FACTS BREAKDOWN (numbered, for readability — not sent to the model):")
        if facts:
            for i, fact in enumerate(facts, 1):
                print(f"  {i}. {fact}")
        else:
            print("  (no relevant memories found)")
        print("-" * 80)

        if generator is not None:
            result = asyncio.run(generator.generate_with_usage(messages, max_tokens=args.max_tokens))
            print("OUTPUT:")
            print(result.text)
            print(f"\n(tokens: in={result.input_tokens} out={result.output_tokens} "
                  f"total={result.total_tokens})")
        else:
            print("OUTPUT: (skipped — --no-generate)")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
