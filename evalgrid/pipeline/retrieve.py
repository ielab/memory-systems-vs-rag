"""
evalgrid/pipeline/retrieve.py
----------------------------
Produces a TREC run (ranked corpus ids + scores) per sample, by either of the
two ways this repo can rank a system's memory:

  --retriever bm25|dense|sparse|multi_vector|fusion|rrf|hybrid
      A retriever from evalgrid/benchmarks/retrievers/ over the FLAT corpus the
      qrels builder wrote. No memory system is loaded; the units are just docs.

  --retriever native
      The memory system's OWN query-time loop, over its own saved store —
      A-Mem's keyword generation plus link expansion, SimpleMem's plan/union/
      reflect, HyperMem's three-layer funnel. The system is loaded and driven.

Both write the same artifact into the same tree, which is the point: one
`run.tsv` per sample under one <retriever> segment, so
evalgrid/pipeline/eval_retrieval.py scores "the system's own retriever" and
"BM25 over the system's own memory" against the same qrels, as one table.

Reads
  flat    outputs/_qrels/<system>/<dataset>/<mem-seg>/<labeler-seg>/<sample>/{corpus.jsonl,queries.jsonl}
  native  outputs/_memory/<system>/<dataset>/<mem-seg>/<sample>/   (+ the dataset adapter, for the queries)
Writes
  outputs/_retrieval/<system>/<dataset>/<mem-seg>/<retriever>/<sample>/run.tsv
  native additionally writes retrieval.jsonl beside it (see below)
  both write retrieval_usage.json at the run root, through one writer
  (write_retrieval_usage) so a RAG row and a memory-system row carry the same
  field names. `tokens`/`llm_calls` are LLM-only on BOTH halves: a flat
  retriever calls no generator, so its zero there is structural, and what it
  actually spent is the `encode` block — texts, tokens and batches through the
  encoder, which nothing counted before. See our development notes for the two
  fields that do not mean the same thing across the halves.
(labeler-segment is <labeler-model>, plus a "-retrieval" suffix when --candidate-source retrieval)

The same qrels dataset can be retrieved by any number of retrievers — each
gets its own <retriever> subfolder, so they're directly comparable with
evalgrid/pipeline/eval_retrieval.py without rebuilding the qrels.

Resumable / sharded / error-isolated the same way as evalgrid.pipeline.ingest: a sample
with an existing run.tsv is skipped (use --overwrite to re-run), a failing
sample is recorded to errors.jsonl rather than aborting the batch, and
--shard-id/--num-shards split samples across independent job processes.

For --retriever dense/sparse/multi_vector (and the dense half of fusion/rrf/
hybrid), each sample's CORPUS representation is cached to
<out>/<sample>/(embeddings.npz|repr_cache.pkl) and reused on a later run. The
cache is content-keyed (exact corpus ids), so a corpus rebuilt by the qrels
builder with --overwrite is detected as stale and re-encoded automatically, no
manual invalidation needed. The QUERY is never cached: it is encoded live on
every run and that encode is inside the per-query clock, so `rank_seconds` is
query-encode + scoring — the latency a deployed system pays per question — and
the corpus build is reported apart as `index_seconds` (retrievers/base.py).

**The two halves take disjoint flags, and that is enforced rather than ignored.**
A flat run is configured by `--emb-model`/`--qrels-dir`/`--concurrency`; a native
run by `--mem-dir`/`--mem-base-url`/`--config`/`--native-variant`/`--top-k`. One
parser accepting both would produce the silent failure where `--emb-model` is
accepted, has no effect, and the run is named after an encoder that never ranked
anything — so passing a flag belonging to the other half is an error naming the
flag. The check is against flags given on the COMMAND LINE only: a shared --exp
spec legitimately carries `emb_model` for the flat rows of the same sweep, and
must not make the native row fail.

**Why native saves retrieval.jsonl too.** The two halves of the native path cost
wildly different things — SimpleMem spends 5-8 LLM calls planning and reflecting
before it answers, A-Mem one generating query keywords, the other four none, all
billed to the memory layer (``query_retrieval_*`` in evalgrid/lib/cost.py). So a
native run also writes, per query, the trace its retrieval produced
(``retrieved_unit_ids``, the rendered context, and ``retrieved_units`` — the
per-unit texts, rank-ordered). ``evalgrid.pipeline.answer --retriever native --prompt native --from-retrieval``
replays it, which is what makes a budget sweep affordable: retrieval is paid once,
not once per budget.

``retrieved_units`` is the field that makes a budget possible. Once retrieval has
been flattened into one context string there is no unit boundary left to cut at,
so each system's ``retrieve_trace`` renders its units individually as well. Two
systems cannot do this faithfully — A-Mem's context comes out of upstream's
``find_related_memories_raw`` as one block, and HyperMem's out of
``format_hierarchical_results`` — so for those a budgeted replay re-joins the
per-unit texts and loses upstream's own framing. That is a real difference from
an unbudgeted run and is recorded as ``budget_applied`` in the trace rather than
hidden.

Usage:
  # BM25 (lexical, no extra model needed)
  python -m evalgrid.pipeline.retrieve --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --retriever bm25

  # Dense (Qwen embedding served via vllm)
  python -m evalgrid.pipeline.retrieve --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --retriever dense \\
      --emb-model Qwen/Qwen3-Embedding-0.6B --emb-base-url http://localhost:8001/v1

  # Sparse / lexical (SPLADE, local — no server needed)
  python -m evalgrid.pipeline.retrieve --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --retriever sparse \\
      --emb-model naver/splade-v3

  # Multi-vector / late-interaction (a ColBERT checkpoint, local)
  python -m evalgrid.pipeline.retrieve --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --retriever multi_vector \\
      --emb-model colbert-ir/colbertv2.0

  # Hybrid dense+sparse fusion (PromptReps, local). Encodes each text
  # ONCE and also writes the dense and sparse component runs to their own
  # <retriever> dirs — one pass yields all three evaluable runs.
  python -m evalgrid.pipeline.retrieve --system lightmem --dataset longmemeval_s \\
      --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --retriever hybrid \\
      --emb-model promptreps:meta-llama/Meta-Llama-3-8B-Instruct

  # The system's own retrieval loop, saved for replay
  python -m evalgrid.pipeline.retrieve --exp experiments/simplemem/longmemeval.yaml \\
      --retriever native --mem-base-url http://localhost:8000/v1
  # then answer from it, as many times as you like:
  for B in 500 1000 2000 4000; do
    python -m evalgrid.pipeline.answer --retriever native --prompt native --exp experiments/simplemem/longmemeval.yaml \\
        --gen-model ... --from-retrieval <the dir printed above> --context-budget $B
  done
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from time import perf_counter
from typing import Optional

from evalgrid.lib.cli import (
    add_candidate_source_arg,
    add_concurrency_arg,
    add_dataset_arg,
    add_emb_base_url_arg,
    add_emb_model_arg,
    add_exp_arg,
    add_labeler_model_arg,
    add_limit_arg,
    add_mem_model_arg,
    add_mem_variant_arg,
    add_overwrite_arg,
    add_retriever_arg,
    add_shard_args,
    add_system_arg,
    add_trace_args,
    apply_spec,
    build_recorder_from_args,
    require_resolved,
)
# Private, deliberately: this is the one stage whose flags are only valid for
# half its invocations, so it needs to distinguish "the user typed this" from
# "a shared --exp supplied it". Nothing else needs that distinction.
from evalgrid.lib.cli import _cli_provided_dests
from evalgrid.lib.cost import query_cost
from evalgrid.lib.errors import EvalgridError, LLMCallFailed
from evalgrid.lib.manifest import write_manifest
from evalgrid.lib.paths import (
    MODEL_BASED_RETRIEVERS,
    ExperimentPaths,
    labeler_segment,
    native_segment,
    retriever_name,
)

from tqdm import tqdm

from evalgrid.benchmarks.retrievers import Retriever, build_retriever
from evalgrid.benchmarks.retrieval import read_jsonl, write_run
from evalgrid.models.base import EncodeUsage, encode_scope
from evalgrid.models.factory import load_embedder, load_multi_vector_model, load_sparse_encoder

#: What a sample's saved native retrieval is called. Read back by
#: answer --retriever native --prompt native --from-retrieval.
RETRIEVAL_FILE = "retrieval.jsonl"

# Per-retriever on-disk cache filename for the model-backed retrievers whose
# query/corpus representations are worth persisting between re-runs (see _run);
# bm25 has no model-side encode step to cache.
_CACHE_FILENAME = {
    "dense": "embeddings.npz",
    "fusion": "embeddings.npz",  # fusion caches only its dense half (DenseRetriever)
    "rrf": "embeddings.npz",     # same: its BM25 half has nothing to cache
    "hybrid": "embeddings.npz",  # hybrid derives its sparse half's repr_cache.pkl alongside
    "sparse": "repr_cache.pkl",
    "multi_vector": "repr_cache.pkl",
}


def _has_run(sample_out: str) -> bool:
    return os.path.isfile(os.path.join(sample_out, "run.tsv"))


def _record_error(out_dir: str, exc: BaseException, **fields) -> None:
    """Append one failed sample/query to ``out_dir/errors.jsonl``.

    The native half of this stage used to have no such file: the flat half
    isolated its samples (see _process_sample) while a native run let the first
    exception out of the loop and killed the process, losing every sample after
    it. Our failure policy says the opposite — "nothing aborts the
    whole run" — and the cost of the gap was a simplemem cell that died at
    162/500 on one malformed planner reply.

    Same shape and same name as the flat half's record, plus the query id when
    the failure has one, so `deterministic` (LLMCallFailed) still separates
    "this configuration cannot work" from "the server had a bad minute".

    Never raises: a harness that crashes while recording a crash is worse than
    one that loses the record.
    """
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "phase": "retrieve",
            "error": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            **fields,
        }
        if isinstance(exc, LLMCallFailed):
            record.update(exc.as_record())
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "errors.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass



def write_retrieval_usage(
    out_dir: str,
    *,
    retriever: str,
    samples: int,
    samples_resumed: int,
    samples_no_memory: int,
    samples_failed: int,
    queries_failed: int,
    queries: int,
    seconds: float,
    spent: tuple[int, int, int],
    extra: Optional[dict] = None,
) -> tuple[int, int, int]:
    """Write ``out_dir/retrieval_usage.json``. Both halves of this stage use it.

    The read-path counterpart to outputs/_memory/**/ingest_usage.json, and the
    authority for what retrieval cost (a trajectory can be sampled or off, so no
    reported number may come from one). Written even when the total is zero —
    for lightmem/mem0/memorybank/hypermem that zero IS the finding, and it is
    what a_mem's and simplemem's non-zero is read against.

    One writer for both halves so that a RAG row and a memory-system row carry
    the same field names and can be read as one table. ``spent`` is memory-layer
    (input, output, calls) and is **LLM only**: a flat retriever calls no
    generator, so its zero here is structural and true, and what its retrieval
    actually cost lives in ``extra["encode"]`` instead.

    Returns ``spent`` so a caller can print from the same numbers it wrote.
    """
    r_in, r_out, r_calls = spent
    payload = {
        "stage": "retrieve",
        "retriever": retriever,
        "samples": samples,
        # Coverage, so this file answers "did the run cover everything it was
        # asked for?" and not only "what did it cost?". Without them `samples`
        # is the count of NEWLY retrieved samples, which a resumed run reports
        # as 0 — indistinguishable from a run that did nothing.
        "samples_resumed": samples_resumed,    # already saved; a run.tsv exists
        "samples_no_memory": samples_no_memory,  # asked for, no store, NO run.tsv
        # Failed on an input that WAS there — the difference from the line
        # above, and the pair check_grid reads to mark a cell that finished with
        # holes. Both leave no run.tsv, so both are re-runnable.
        "samples_failed": samples_failed,      # every query of it failed
        "queries_failed": queries_failed,      # this many single queries did
        "queries": queries,
        "seconds": round(seconds, 3),
        "input_tokens": r_in,
        "output_tokens": r_out,
        "tokens": r_in + r_out,
        "llm_calls": r_calls,
        # The two lib/cost.py field names that apply to a stage which never
        # answers. The query_answer_* half is deliberately NOT emitted: it would
        # read as "the answer cost nothing" in a file written by a stage that
        # does not answer at all.
        **{k: v for k, v in query_cost(retrieval=spent).items()
           if k.startswith("query_retrieval_")},
        "per_query_tokens": (r_in + r_out) / queries if queries else 0.0,
        "per_query_calls": r_calls / queries if queries else 0.0,
        **(extra or {}),
        # Retrieval is paid ONCE and reused by every answer run replaying this
        # directory, so a sweep's total is not the sum of its answer runs'
        # query_retrieval_* — that would count this figure once per budget.
        # Add it once.
        "note": "paid once; shared by every --from-retrieval answer run over this dir",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retrieval_usage.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return spent


async def _process_sample(
    retriever: Retriever,
    sem: asyncio.Semaphore,
    sample_id: str,
    sample_dir: str,
    sample_out: str,
    rank_kwargs: dict,
    component_outs: Optional[dict[str, str]] = None,
) -> tuple[Optional[dict], Optional[dict]]:
    """Rank every query in one qrels sample against its own corpus, write run.tsv.

    component_outs ({mode: sample_out_dir}, hybrid only) additionally persists
    the retriever's per-mode component rankings (rank_all) to those dirs — the
    corpus is encoded once but yields the dense/sparse/hybrid runs together.

    Returns ``(stats, error)``: exactly one is non-None. The error is caught
    here so one sample's failure doesn't cancel the rest of the batch via
    asyncio.gather.

    **Both halves of the accounting are collected here rather than around the
    gather**, and for the same reason: this path runs `--concurrency` samples at
    once against ONE shared retriever. A before/after delta taken outside would
    collect whatever the other samples in flight encoded, and a clock started
    outside would be counting the wait for a semaphore slot. So the encode scopes
    (models/base.encode_scope — context-local, therefore per task) and the clocks
    all open INSIDE the semaphore.

    **Index, then search — and only search is latency.** ``retriever.index``
    encodes (or loads from the corpus cache) this sample's corpus; that is what a
    deployed memory system has already paid before a question arrives, so it is
    timed and counted on its own (``index_seconds`` / ``index_encode``) and kept
    OUT of ``rank_seconds``. ``retriever.search`` encodes the query live and
    scores it; that is the per-question retrieval latency, one clock per query.
    Before this split a cache hit skipped the query encode too (reporting a
    latency with no encoder in it) and a miss charged the whole corpus encode to
    the question (reporting an index build as a latency).
    """
    try:
        corpus = read_jsonl(os.path.join(sample_dir, "corpus.jsonl"))
        queries = read_jsonl(os.path.join(sample_dir, "queries.jsonl"))
        # Read BEFORE indexing, because indexing writes it. It is what separates
        # "encoded no corpus because the cache served it" from "encoded nothing
        # because this backend does not count" — two zeros that mean opposite
        # things in a cost file.
        cache_path = rank_kwargs.get("cache_path")
        cache_present = bool(cache_path and os.path.isfile(cache_path))
        query_seconds: list[float] = []
        async with sem:
            started = perf_counter()
            with encode_scope() as index_encoded:
                index = await retriever.index(corpus, **rank_kwargs)
            index_seconds = perf_counter() - started
            with encode_scope() as query_encoded:
                ranked_by_query = []
                by_mode: dict[str, list] = {}
                for q in queries:
                    started = perf_counter()
                    if component_outs is None:
                        ranked_by_query.append((q["_id"], await retriever.search(q["text"], index)))
                    else:
                        ranked_all = await retriever.search_all(q["text"], index)
                        for mode, ranked in ranked_all.items():
                            by_mode.setdefault(mode, []).append((q["_id"], ranked))
                    query_seconds.append(perf_counter() - started)
        if component_outs is None:
            write_run(sample_out, ranked_by_query)
        else:
            write_run(sample_out, by_mode["hybrid"])
            for mode, comp_out in component_outs.items():
                write_run(comp_out, by_mode[mode])
        return ({"queries": len(queries), "rank_seconds": sum(query_seconds),
                 "query_seconds": query_seconds, "index_seconds": index_seconds,
                 "index_encode": index_encoded, "query_encode": query_encoded,
                 "corpus_docs": len(corpus), "cache_present": cache_present}, None)
    except Exception as e:
        tqdm.write(f"\n[ERROR] sample {sample_id!r}: {e}")
        tqdm.write(traceback.format_exc())
        return (None, {"sample_id": sample_id, "error": str(e),
                       "traceback": traceback.format_exc()})


async def _run(
    retriever: Retriever,
    concurrency: int,
    to_process: list[tuple[str, str, str]],
    cache_kind: Optional[str] = None,
    component_roots: Optional[dict[str, str]] = None,
    overwrite: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Process every sample concurrently (bounded by ``concurrency``).

    cache_kind selects the on-disk cache filename for retrievers whose
    query/corpus representations are worth persisting between re-runs (one
    of _CACHE_FILENAME's keys), or None to disable caching (e.g. bm25, which
    has no model-side encode step to cache).

    component_roots ({mode: out_dir}, hybrid only) routes the per-mode
    component runs; a component whose run.tsv already exists is left alone
    unless ``overwrite`` (it's identical to what a standalone run wrote).

    Returns ``(per-sample stats, error records)`` — one entry per sample in
    exactly one of the two lists.
    """
    sem = asyncio.Semaphore(concurrency)
    with tqdm(total=len(to_process), desc="samples", unit="sample") as bar:
        async def _wrapped(sample_id, sample_dir, sample_out):
            rank_kwargs = (
                {"cache_path": os.path.join(sample_out, _CACHE_FILENAME[cache_kind])}
                if cache_kind else {}
            )
            component_outs = None
            if component_roots is not None:
                component_outs = {}
                for mode, root in component_roots.items():
                    sout = os.path.join(root, sample_id)
                    if overwrite or not _has_run(sout):
                        component_outs[mode] = sout
            stats, err = await _process_sample(retriever, sem, sample_id, sample_dir,
                                               sample_out, rank_kwargs, component_outs)
            bar.update(1)
            return stats, err

        results = await asyncio.gather(*(
            _wrapped(sid, sdir, sout) for sid, sdir, sout in to_process
        ))
    return ([s for s, _ in results if s is not None],
            [e for _, e in results if e is not None])


class NoGenerator:
    """Stands in for the generator on the native path, and raises if used.

    A retrieval run generates nothing, so a system that reaches for the
    generator is a bug — a ``retrieve_trace`` that fell back to the base
    implementation, which calls ``answer()``. Failing loudly beats silently
    spending answer tokens in a stage whose output has no answers in it.
    """

    def __init__(self):
        from evalgrid.models.base import TokenUsage
        self.token_usage = TokenUsage()

    async def generate(self, *args, **kwargs):
        raise RuntimeError(
            "the generator was called during --retriever native. This system's "
            "retrieve_trace() is not overridden, so it fell back to the base "
            "implementation, which answers. Split its answer() at the seam — see "
            "MemorySystem.retrieve_trace in evalgrid/benchmarks/types.py."
        )


def _run_native(parser: argparse.ArgumentParser, args) -> None:
    """``--retriever native``: drive each system's own query-time loop.

    Imported lazily, and the whole body kept out of ``main`` — a plain bm25 run
    has no business loading the plugin registry, the dataset adapters or the
    model stack, and this keeps that true after the merge.
    """
    from evalgrid.benchmarks.adapters import DATASETS, build_adapter
    from evalgrid.benchmarks.runner import _sample_mem_path
    from evalgrid.pipeline.answer import ranked_from_trace
    from evalgrid.pipeline.ingest import LimitedAdapter, ShardedAdapter, load_config
    from evalgrid.systems import native_retrieval
    from evalgrid.systems.registry import get_plugin

    # The spec's native depth, which only this path reads. --top-k on the CLI
    # still wins; `native_top_k:` supplies it when the flag is absent. It is a
    # separate key from `top_k:` because the two paths disagree about what
    # --top-k means (retrieval depth here, prompt depth on the shared path), and
    # one spec drives both halves of a sweep — see lib/cli.PASSTHROUGH_KEYS.
    #
    # --published-depth opts out, and it exists because omitting --top-k used to
    # BE the way to say "the config's own depth": once a spec carries the imposed
    # depth, silence means the spec, so the control needs a way to say itself.
    if args.published_depth:
        if args.top_k is not None:
            parser.error("--published-depth and --top-k contradict each other: one asks for "
                         "the config's own depth, the other imposes one.")
    elif args.top_k is None:
        args.top_k = getattr(args, "native_top_k", None)

    if args.dataset not in DATASETS:
        parser.error(f"--retriever native needs a dataset adapter to supply the queries; "
                     f"{args.dataset!r} is not one of {sorted(DATASETS)}")
    try:
        plugin = get_plugin(args.system)
    except KeyError:
        # The raw-turn baselines (all_turns / user_turns / pair_turns) are a
        # corpus shape, not a memory system — there is no query-time loop to run.
        parser.error(f"--retriever native is not available for --system {args.system!r}: "
                     f"it is a raw-corpus baseline with no memory system, so it has no "
                     f"query-time retrieval of its own. Use a flat retriever (bm25, dense, ...).")

    system_config, config_source = load_config(plugin, args)
    # Before apply_mem_model and before ANY path is derived: the swap decides the
    # run directory (native_retriever_name reads it), so injecting it later would
    # write a swapped run into the published run's folder.
    native_retrieval.apply_spec(system_config, args.native_retriever)
    plugin.apply_mem_model(system_config, args.mem_model, args.mem_base_url)
    if args.top_k is not None:
        system_config = plugin.apply_retrieval_budget(system_config, args.top_k)

    paths = ExperimentPaths(system=args.system, dataset=args.dataset,
                            mem_model=args.mem_model, mem_variant=args.mem_variant)
    mem_dir = args.mem_dir or paths.memory()
    # Before any path is built: a system that cannot take the configured swap
    # (hypermem, simplemem) must fail here rather than after the output dir and
    # manifest exist and the first sample is already loaded. declare, not
    # check_supported: this also has to reach `plugin` before build() below, so
    # a system whose constructor builds an index only its own ranking would read
    # can skip building it.
    swap = native_retrieval.declare(plugin, system_config)
    # Same derivation the native answer path uses, so a retrieve/answer pair lands in
    # matching directories and a swapped retriever cannot overwrite the
    # published one (SystemPlugin.native_retriever_name).
    native = native_segment(plugin.native_retriever_name(system_config),
                            args.native_variant, args.top_k)
    seg = labeler_segment(args.labeler_model, args.candidate_source)
    out_dir = args.out or paths.retrieval(native)
    os.makedirs(out_dir, exist_ok=True)
    write_manifest(out_dir, args, stage="retrieve")

    print(f"System    : {args.system}")
    print(f"Dataset   : {args.dataset}")
    print(f"Mem model : {args.mem_model}")
    print(f"Config    : {config_source}")
    print(f"Memory dir: {mem_dir}")
    print(f"Out dir   : {out_dir}")
    print(f"Retriever : {native} (the system's own query-time loop)")
    if swap:
        print(f"  swapped : {swap} (substituted into the system's own loop)")
    if args.top_k is not None:
        print(f"Top-k     : {args.top_k} (imposed on the system's own retriever)")
    if args.num_shards > 1:
        print(f"Shard     : {args.shard_id}/{args.num_shards}")

    adapter = build_adapter(args.dataset)
    if args.num_shards > 1:
        adapter = ShardedAdapter(adapter, args.shard_id, args.num_shards)
    if args.limit:
        adapter = LimitedAdapter(adapter, args.limit)

    # config_for_build, not system_config: the swap block is harness state and
    # no system's constructor takes it (native_retrieval.install below reads it
    # off the unfiltered config, which is where it belongs).
    system = plugin.build(native_retrieval.config_for_build(system_config),
                          dataset=args.dataset,
                          generator=NoGenerator(), phase="answer")
    recorder = build_recorder_from_args(args)
    system.install_recorder(recorder)

    # `skipped` is the RESUME skip (already saved, so a run.tsv exists);
    # `no_memory` is a sample this run was asked for and could not do, which
    # leaves no run.tsv behind and no other trace on disk. They are counted
    # apart because only the second one makes the finished run incomplete —
    # eval_retrieval refuses to score on it. Merged, a resumed run would look
    # like a broken one.
    # `failed_samples` / `failed_queries` are the third and fourth ways a sample
    # can be absent, and the two this loop used to have no way to survive: one
    # exception out of a system's own query loop ended the PROCESS, so a cell
    # died wherever the first bad LLM reply landed and every sample after it was
    # never attempted. They are counted apart from `no_memory` because that one
    # is a missing input and these are a failure on an input that was there.
    done = queries = skipped = no_memory = failed_samples = failed_queries = 0
    #: Cumulative memory-layer (input, output, calls) this run spent retrieving.
    spent = (0, 0, 0)
    wall_started = perf_counter()
    try:
        for sample in adapter.samples():
            sample_out = _sample_mem_path(out_dir, sample.sample_id)
            saved = os.path.join(sample_out, RETRIEVAL_FILE)
            if not args.overwrite and os.path.isfile(saved) and os.path.getsize(saved) > 0:
                skipped += 1
                continue
            sample_mem = _sample_mem_path(mem_dir, sample.sample_id)
            if not os.path.isdir(sample_mem):
                print(f"  [skip] {sample.sample_id!r} — no saved memory at {sample_mem}")
                no_memory += 1
                continue

            try:
                system.reset()
                system.load_memory(sample_mem)
                # After load_memory: the units have to exist before they can be
                # indexed. Returns None (and changes nothing) unless the config
                # has a native_retriever block.
                native_retrieval.install(system, plugin, system_config, sample_mem)
                # After install: build the corpus index now, OUTSIDE the
                # per-query clock below. MemoryBank and HyperMem otherwise build
                # it lazily on the first question — on LongMemEval the only
                # question — which put a whole corpus encode into every
                # reported retrieval_seconds. The query is still encoded live.
                system.prepare_retrieval()
            except (Exception, EvalgridError) as e:
                # Nothing was written, so this sample has no run.tsv and a
                # re-run picks it up without --overwrite.
                _record_error(out_dir, e, sample_id=sample.sample_id, stage="load")
                print(f"\n[ERROR] sample {sample.sample_id!r} (load): {e}")
                print(traceback.format_exc())
                failed_samples += 1
                continue

            if recorder is not None:
                os.makedirs(sample_out, exist_ok=True)
                # phase="retrieve", not "answer": this stage generates nothing,
                # and phase is the only field separating a retrieve-stage record
                # from an answer-stage one when both land in the same tree.
                recorder.begin_sample(sample.sample_id, "retrieve", sample_out)

            traces = []
            failed_here = 0
            try:
                for query in sample.queries:
                    # THIS is where query-time retrieval cost is measured, and
                    # the only place it can be: this stage is what pays it.
                    # A budget sweep runs retrieval once and answers N times, so
                    # recording it on the answer side would either miss it (N
                    # answer runs each reporting 0) or multiply it by N. See the
                    # RETRIEVAL COST note in the module docstring.
                    before = system._memory_layer_usage()
                    started = perf_counter()
                    try:
                        trace = system.retrieve_trace(query)
                    # Both, and named: EvalgridError derives from BaseException
                    # so that upstream's `while True: except Exception` retry
                    # loops cannot swallow it (lib/errors.py), and the same
                    # choice makes it invisible to a bare `except Exception`
                    # here. A no-retry LLM failure is the likeliest thing this
                    # block will ever catch, so missing it would leave the fix
                    # not working for the case it exists for.
                    except (Exception, EvalgridError) as e:
                        # A query is the smallest unit with a boundary here: the
                        # sample's OTHER queries are unaffected and still saved.
                        # The cost of that: a PARTLY failed sample does write its
                        # retrieval.jsonl, so the resume check above skips it and
                        # only --overwrite retries the missing query. errors.jsonl
                        # is what says so. (Moot on LongMemEval — one query per
                        # sample — so the two are the same event there.)
                        # The tokens this failed call already burned are charged
                        # anyway — a failure that costs GPU time and reports
                        # zero is how a cost table quietly stops adding up.
                        after = system._memory_layer_usage()
                        spent = tuple(s + (a - b) for s, a, b
                                      in zip(spent, after, before))
                        _record_error(out_dir, e, sample_id=sample.sample_id,
                                      query_id=getattr(query, "query_id", None),
                                      stage="retrieve_trace")
                        print(f"\n[ERROR] sample {sample.sample_id!r} "
                              f"query {getattr(query, 'query_id', None)!r}: {e}")
                        print(traceback.format_exc())
                        failed_queries += 1
                        failed_here += 1
                        continue
                    after = system._memory_layer_usage()
                    cost = tuple(a - b for a, b in zip(after, before))
                    # Copied, not mutated in place: the system holds this dict in
                    # self.traces and nothing of ours may edit a system's state.
                    traces.append({**trace,
                                   "retrieval_cost": {"input_tokens": cost[0],
                                                      "output_tokens": cost[1],
                                                      "calls": cost[2]},
                                   "retrieval_seconds": round(perf_counter() - started, 4)})
                    spent = tuple(s + c for s, c in zip(spent, cost))
                    queries += 1
            finally:
                if recorder is not None:
                    recorder.end_sample()

            if failed_here and not traces:
                # Every query failed. Write NOTHING: an empty retrieval.jsonl
                # would satisfy the resume check at the top of this loop and the
                # sample would be skipped forever, and a header-only run.tsv
                # reads as "retrieved, ranked nothing" to eval_retrieval. With
                # no files the hole is both visible and re-runnable.
                failed_samples += 1
                continue

            os.makedirs(sample_out, exist_ok=True)
            # Written before run.tsv: retrieval.jsonl is the artifact the answer
            # stage replays from, so a crash between the two must not leave a
            # ranking with no evidence behind it.
            with open(saved, "w") as f:
                for trace in traces:
                    f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")

            ranked_by_query = []
            for trace in traces:
                qid = trace.get("query_id")
                if qid is not None:
                    ranked_by_query.append((qid, ranked_from_trace(plugin, trace)))
            if ranked_by_query:
                write_run(sample_out, ranked_by_query)
            done += 1
    finally:
        recorder.close()

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"system": args.system, "dataset": args.dataset,
                   "mem_model": args.mem_model, "native_variant": args.native_variant,
                   "top_k": args.top_k, "stage": "retrieve", "retriever": native}, f, indent=2)

    r_in, r_out, r_calls = write_retrieval_usage(
        out_dir,
        retriever=native,
        samples=done,
        samples_resumed=skipped,
        samples_no_memory=no_memory,
        samples_failed=failed_samples,
        queries_failed=failed_queries,
        queries=queries,
        seconds=perf_counter() - wall_started,
        spent=spent,
    )

    print(f"\nDone — {done} sample(s), {queries} query/queries retrieved into {out_dir}")
    if queries:
        print(f"       retrieval cost: {r_in + r_out:,} tokens / {r_calls} LLM call(s) "
              f"({(r_in + r_out) / queries:,.0f} tok per question) -> retrieval_usage.json")
    if skipped:
        print(f"       skipped {skipped} already saved (use --overwrite to redo)")
    if failed_samples or failed_queries:
        # Loud, and last: the run "succeeded" and is still short of what it was
        # asked for. Neither number is in retrieval_usage.json's cost totals.
        print(f"\n!!! {failed_samples} sample(s) and {failed_queries} query/queries FAILED "
              f"-> {os.path.join(out_dir, 'errors.jsonl')}")
        print(f"!!!   they wrote no run.tsv, so re-running this cell retries exactly "
              f"those (no --overwrite needed).")
    if recorder.enabled and recorder.failures:
        print(f"       ! {recorder.failures} trajectory record(s) failed to write "
              f"(the run itself is unaffected)")
    print(f"\nNext — answer from it, once per budget:")
    print(f"  python -m evalgrid.pipeline.answer --retriever native --prompt native --system {args.system} "
          f"--dataset {args.dataset} \\")
    print(f"      --mem-model {args.mem_model} --gen-model <gen> \\")
    print(f"      --from-retrieval {out_dir} --context-budget <T>")


#: Flags that only mean something to one half of this stage. Checked against
#: what the user actually typed, so a shared --exp carrying the other half's
#: values (the usual case: one spec drives a whole sweep) stays legal.
_FLAT_ONLY = ("emb_model", "emb_base_url", "qrels_dir", "concurrency",
              "exclude_question_type")
_NATIVE_ONLY = ("mem_dir", "mem_base_url", "config", "native_variant", "top_k",
                "published_depth", "trace_level", "trace_samples", "trace_max_chars")


def _reject_foreign_flags(parser, provided: set[str], dests, reason: str) -> None:
    bad = sorted(d for d in dests if d in provided)
    if bad:
        flags = ", ".join("--" + d.replace("_", "-") for d in bad)
        parser.error(f"{flags} {'has' if len(bad) == 1 else 'have'} no meaning {reason}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank each sample's memory — with a flat retriever, or with the "
                    "system's own query-time loop (--retriever native)")
    add_exp_arg(parser)
    add_system_arg(parser, help="Memory system the qrels dataset came from (e.g. lightmem)")
    add_dataset_arg(parser,
                    help="Benchmark dataset key (e.g. longmemeval_s) — used only to "
                         "derive default paths, no adapter is loaded")
    add_mem_model_arg(parser)
    add_mem_variant_arg(parser)
    add_labeler_model_arg(parser,
                          help="Labeler model the qrels builder was run with (default: "
                               "gpt-4o) — locates the qrels dir this run RANKS. It no longer keys "
                               "the output dir: a ranking does not depend on the labels, so two "
                               "labelers share one run.tsv rather than each paying for a full "
                               "re-retrieval. eval_retrieval keys its metrics on the labeler")
    add_candidate_source_arg(parser)
    parser.add_argument("--out", default=None,
                        help="Override the retrieval output dir "
                             "(default: outputs/_retrieval/<system>/<dataset>/<mem-model>/<retriever>/)")
    add_retriever_arg(parser, include_native=True,
                      help="Retriever to run. 'native' runs the system's OWN query-time "
                           "loop over its saved store instead of ranking the flat corpus; "
                           "it takes the --retriever native flag group below and none of "
                           "the flat-corpus one.")
    add_limit_arg(parser, help="Only process first N samples (applied after sharding)")
    add_shard_args(parser)
    add_overwrite_arg(parser, help="Re-run samples that already have a run.tsv written")

    flat = parser.add_argument_group(
        "flat-corpus retrievers", "Only for --retriever bm25/dense/sparse/multi_vector/fusion/rrf/hybrid")
    flat.add_argument("--qrels-dir", default=None,
                      help="Override the qrels dataset dir "
                           "(default: outputs/_qrels/<system>/<dataset>/<mem-model>/<labeler-segment>/)")
    add_emb_model_arg(flat,
                      help="Model for --retriever dense/sparse/multi_vector, e.g. "
                           "Qwen/Qwen3-Embedding-0.6B (dense), naver/splade-v3 (sparse), "
                           "promptreps:meta-llama/Meta-Llama-3-8B-Instruct (dense/sparse/hybrid), "
                           "colbert-ir/colbertv2.0 (multi_vector)")
    add_emb_base_url_arg(flat)
    add_concurrency_arg(flat, help="Concurrent sample rankings in flight (default: 8)")
    flat.add_argument("--exclude-question-type", action="append", default=None,
                      metavar="TYPE",
                      help="Skip samples whose query has this metadata.question_type "
                           "(e.g. single-session-assistant). Repeatable.")

    nat = parser.add_argument_group(
        "--retriever native", "Only for --retriever native (the system's own loop)")
    nat.add_argument("--mem-dir", default=None, help="Override the saved-memory dir")
    nat.add_argument("--mem-base-url", default=None,
                     help="Base URL for the memory LLM. Needed by systems that call it at "
                          "query time (A-Mem's keywords, SimpleMem's planning).")
    nat.add_argument("--native-retriever", default=None, metavar="SPEC",
                     help="Swap the retriever INSIDE the system's own loop: bm25 | "
                          "dense:MODEL[@url] | sparse:MODEL | rrf:MODEL[@url] | "
                          "fusion:MODEL[@url] | hybrid:MODEL | multi_vector:MODEL. "
                          "Same vocabulary as a spec's retrievers: list, so one string "
                          "names the same retriever flat and native. Overrides any "
                          "native_retriever: block in the config, and names the run dir.")
    nat.add_argument("--config", default=None,
                     help="Explicit config path, bypassing variant resolution — how you "
                          "change the NATIVE retriever's settings.")
    nat.add_argument("--native-variant", default=None, metavar="NAME",
                     help="Names this native-retrieval setting and suffixes the output dir.")
    nat.add_argument("--top-k", type=int, default=None,
                     help="Impose one retrieval depth on the system's OWN retriever "
                          "(plugin.apply_retrieval_budget). Omitted = the spec's "
                          "native_top_k, or the config's own depth if the spec has none. "
                          "A flat run needs no equivalent: it ranks the whole corpus and "
                          "pipeline/answer.py truncates to k when it builds the prompt.")
    nat.add_argument("--published-depth", action="store_true",
                     help="Retrieve at the depth in the system's OWN config, ignoring the "
                          "spec's native_top_k. This is the published-depth control (E1); "
                          "without it a spec carrying native_top_k imposes that depth. "
                          "Mutually exclusive with --top-k.")
    add_trace_args(nat)

    args = parser.parse_args()
    apply_spec(parser, args)
    require_resolved(parser, args, "system", "dataset", "mem_model")

    provided = _cli_provided_dests(parser, sys.argv[1:])
    if args.retriever == "native":
        _reject_foreign_flags(parser, provided, _FLAT_ONLY,
                              "with --retriever native: it ranks the system's own store, "
                              "not the flat qrels corpus")
        return _run_native(parser, args)
    _reject_foreign_flags(parser, provided, _NATIVE_ONLY,
                          f"for --retriever {args.retriever}: no memory system is loaded, "
                          f"so there is no native loop to configure or record")
    # top_k is a --exp spec key (pipeline/answer.py reads it as PROMPT depth), and
    # a shared spec drives both halves of a sweep. Dropping it here keeps it out
    # of the flat manifest, which would otherwise claim a depth this run never
    # applied — a flat retriever ranks the WHOLE corpus and answer.py truncates
    # to k. The native depth is a different key (`native_top_k:`) and is read
    # only by _run_native, so it cannot reach this path at all.
    args.top_k = None

    if args.retriever in MODEL_BASED_RETRIEVERS and not args.emb_model:
        parser.error(f"--retriever {args.retriever} requires --emb-model")

    paths = ExperimentPaths(system=args.system, dataset=args.dataset, mem_model=args.mem_model,
                            mem_variant=args.mem_variant)
    seg = labeler_segment(args.labeler_model, args.candidate_source)
    qrels_dir = args.qrels_dir or paths.qrels(seg)
    if not os.path.isdir(qrels_dir):
        sys.exit(f"No qrels dataset at {qrels_dir} — run evalgrid.pipeline.qrels_has_answer first.")

    rname = retriever_name(args.retriever, args.emb_model)
    out_dir = args.out or paths.retrieval(rname)

    # hybrid encodes each text once but its rank_all also yields the dense and
    # sparse rankings — persist them to those retrievers' standard run dirs so
    # one hybrid pass produces all three directly evaluable runs.
    component_roots = None
    if args.retriever == "hybrid":
        component_roots = {
            mode: paths.retrieval(retriever_name(mode, args.emb_model))
            for mode in ("dense", "sparse")
        }

    print(f"System    : {args.system}")
    print(f"Dataset   : {args.dataset}")
    print(f"Mem model : {args.mem_model}")
    print(f"Qrels dir : {qrels_dir}")
    print(f"Out dir   : {out_dir}")
    print(f"Retriever : {rname}")
    if component_roots:
        for mode, root in component_roots.items():
            print(f"  +{mode:<7}: {root}")
    if args.num_shards > 1:
        print(f"Shard     : {args.shard_id}/{args.num_shards}")
    write_manifest(out_dir, args, stage="retrieve")

    emb_kwargs = {"base_url": args.emb_base_url} if args.emb_base_url else {}
    if args.retriever == "dense":
        retriever = build_retriever("dense", embedder=load_embedder(args.emb_model, **emb_kwargs))
    elif args.retriever == "fusion":
        retriever = build_retriever("fusion", embedder=load_embedder(args.emb_model, **emb_kwargs))
    elif args.retriever == "rrf":
        retriever = build_retriever("rrf", embedder=load_embedder(args.emb_model, **emb_kwargs))
    elif args.retriever == "sparse":
        retriever = build_retriever("sparse", encoder=load_sparse_encoder(args.emb_model, **emb_kwargs))
    elif args.retriever == "hybrid":
        # One dual-representation model serves both halves (PromptReps encodes
        # dense+sparse in a single forward pass; loading twice would double
        # GPU memory and defeat its encode memo).
        from evalgrid.models.base import EmbedderModel
        encoder = load_sparse_encoder(args.emb_model, **emb_kwargs)
        if not isinstance(encoder, EmbedderModel):
            parser.error(f"--retriever hybrid needs a model providing both dense and "
                         f"sparse representations (promptreps:<llm>); "
                         f"{args.emb_model!r} has no dense side")
        retriever = build_retriever("hybrid", embedder=encoder, encoder=encoder)
    elif args.retriever == "multi_vector":
        retriever = build_retriever("multi_vector", model=load_multi_vector_model(args.emb_model, **emb_kwargs))
    else:
        retriever = build_retriever("bm25")

    sample_ids = sorted(
        d for d in os.listdir(qrels_dir)
        if os.path.isfile(os.path.join(qrels_dir, d, "corpus.jsonl"))
    )
    if args.exclude_question_type:
        excluded = set(args.exclude_question_type)
        kept = []
        dropped = 0
        for sample_id in sample_ids:
            queries = read_jsonl(os.path.join(qrels_dir, sample_id, "queries.jsonl"))
            if any((q.get("metadata") or {}).get("question_type") in excluded for q in queries):
                dropped += 1
            else:
                kept.append(sample_id)
        sample_ids = kept
        print(f"Excluded  : {dropped} sample(s) with question_type in {sorted(excluded)}")
    if args.num_shards > 1:
        sample_ids = [s for i, s in enumerate(sample_ids) if i % args.num_shards == args.shard_id]
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    to_process: list[tuple[str, str, str]] = []
    skipped = 0
    for sample_id in sample_ids:
        sample_dir = os.path.join(qrels_dir, sample_id)
        sample_out = os.path.join(out_dir, sample_id)
        done = _has_run(sample_out) and all(
            _has_run(os.path.join(root, sample_id))
            for root in (component_roots or {}).values()
        )
        if not args.overwrite and done:
            skipped += 1
            continue
        to_process.append((sample_id, sample_dir, sample_out))

    stats: list[dict] = []
    errors: list[dict] = []
    model_based = args.retriever in MODEL_BASED_RETRIEVERS
    wall_started = perf_counter()
    if to_process:
        print(f"\nRetrieving {len(to_process)} sample(s)...")
        stats, errors = asyncio.run(_run(
            retriever, args.concurrency, to_process,
            cache_kind=(args.retriever if model_based else None),
            component_roots=component_roots,
            overwrite=args.overwrite,
        ))
        if errors:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "errors.jsonl"), "a") as f:
                for e in errors:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

    queries = sum(s["queries"] for s in stats)
    rank_seconds = sum(s["rank_seconds"] for s in stats)
    index_seconds = sum(s["index_seconds"] for s in stats)
    query_seconds = sorted(t for s in stats for t in s["query_seconds"])
    index_encoded, query_encoded, encoded = EncodeUsage(), EncodeUsage(), EncodeUsage()
    for s in stats:
        index_encoded.merge(s["index_encode"])
        query_encoded.merge(s["query_encode"])
        encoded.merge(s["index_encode"])
        encoded.merge(s["query_encode"])
    # A model-based sample that encoded no CORPUS is either cache-served or
    # running an un-instrumented backend, and those are opposite meanings for
    # the same zero. The cache is the ordinary case (a re-run reuses
    # embeddings.npz / repr_cache.pkl) and its zero is true — this pass really
    # did no corpus encoding. A sample that encoded NOTHING at all, query
    # included, is the other case — the query is encoded live every run, so a
    # counted backend can never report zero there — and the file says so rather
    # than reporting a free retriever. (promptreps is the one backend with no
    # counter today.)
    cache_served = sum(1 for s in stats
                       if s["index_encode"].batches == 0 and s["cache_present"])
    uninstrumented = model_based and sum(
        1 for s in stats
        if s["index_encode"].batches == 0 and s["query_encode"].batches == 0)

    def _pct(p: float) -> float:
        if not query_seconds:
            return 0.0
        return query_seconds[min(len(query_seconds) - 1, int(p * len(query_seconds)))]

    # A re-run that retrieved NOTHING must not replace a real total with zeros.
    #
    # This file is written from one pass's counters, and the flat half keeps no
    # per-sample record to rebuild an earlier pass from (check_usage can repair
    # a native cell from its traces; there is nothing here to repair from). So
    # the one case that is both common and fully recoverable — re-running a
    # finished cell, which skips every sample and encodes nothing — is handled
    # by not writing at all. A PARTIAL resume still reports only its own pass,
    # which `coverage_basis` below says out loud.
    if not to_process and os.path.isfile(os.path.join(out_dir, "retrieval_usage.json")):
        print(f"\nNothing to retrieve — {skipped} sample(s) already done; "
              f"retrieval_usage.json left as the earlier pass wrote it")
        return
    write_retrieval_usage(
        out_dir,
        retriever=rname,
        samples=len(to_process) - len(errors),
        samples_resumed=skipped,
        # No such thing on this path: the corpus is the qrels dataset, and a
        # sample with no corpus.jsonl was never listed in the first place.
        samples_no_memory=0,
        samples_failed=len(errors),
        # A flat sample ranks all its queries or raises, so a failure is always
        # whole-sample; there is no partial-query state to report like the
        # native loop has.
        queries_failed=0,
        queries=queries,
        seconds=perf_counter() - wall_started,
        # Structurally zero, and that is the finding rather than a gap: a flat
        # retriever loads no memory system and calls no generator. What this
        # run actually spent is under "encode" — see write_retrieval_usage.
        spent=(0, 0, 0),
        extra={
            # encode = index_encode + query_encode, the total this pass spent.
            # Split because only the query half recurs per question: the index
            # half is paid once per sample (and zero when the cache served it).
            "encode": encoded.as_dict(),
            "index_encode": index_encoded.as_dict(),
            "query_encode": query_encoded.as_dict(),
            "per_query_encode_tokens": query_encoded.tokens / queries if queries else 0.0,
            "encoder": args.emb_model if model_based else None,
            "samples_cache_served": cache_served,
            "samples_uncounted_encoder": uninstrumented or 0,
            "rank_seconds": round(rank_seconds, 3),
            "per_query_rank_seconds": rank_seconds / queries if queries else 0.0,
            "p50_query_rank_seconds": _pct(0.50),
            "p95_query_rank_seconds": _pct(0.95),
            "index_seconds": round(index_seconds, 3),
            "concurrency": args.concurrency,
            # Only when it applies, so its absence means "one pass covered the
            # whole cell" rather than "nobody thought about it".
            **({"coverage_basis":
                f"this pass newly retrieved {len(to_process) - len(errors)} sample(s) and "
                f"skipped {skipped} already done; the cost and seconds below cover the new "
                f"ones ONLY. The flat half writes no per-sample record, so an earlier "
                f"pass's encode total cannot be recovered — re-run with --overwrite for a "
                f"figure covering the whole cell."} if skipped and to_process else {}),
            # Named, because the same field on the native path means something
            # else. That loop is sequential, so its per-query seconds ARE
            # latency; this one runs `concurrency` samples at once against one
            # shared retriever, so a rank that waited on another sample's GPU
            # batch has that wait inside it. At --concurrency 1 the two are the
            # same quantity and comparable; above it, rank_seconds is throughput
            # and must not be reported as per-question latency.
            "seconds_basis": (
                "rank_seconds is wall time inside the semaphore, per query, around "
                "search only: the live query encode plus scoring against a corpus "
                "index built beforehand. It excludes the index build (index_seconds: "
                "corpus encode, or its cache load), queueing, corpus load and run.tsv "
                "writes. At "
                f"concurrency={args.concurrency} it is "
                + ("comparable with the native path's per-query retrieval_seconds."
                   if args.concurrency == 1 else
                   "INFLATED by the samples running alongside it — throughput, not "
                   "latency. Re-run with --concurrency 1 for a latency-grade number.")
            ),
        },
    )

    print(f"\nDone — runs written to {out_dir}")
    print(f"  succeeded : {len(to_process) - len(errors)}")
    if queries:
        print(f"  retrieval cost: 0 LLM tokens (no generator on this path); "
              f"{encoded.tokens:,} encode tokens over {encoded.texts:,} text(s) "
              f"(index {index_encoded.tokens:,}, cache-served {cache_served} sample(s); "
              f"query {query_encoded.tokens / queries:,.0f} per question) -> retrieval_usage.json")
        print(f"  latency   : {rank_seconds / queries * 1000:,.1f} ms/query mean, "
              f"p50 {_pct(0.50) * 1000:,.1f} ms, p95 {_pct(0.95) * 1000:,.1f} ms "
              f"(query encode + search; index build {index_seconds:,.1f}s kept apart)"
              + ("" if args.concurrency == 1 else
                 f" — concurrency={args.concurrency}, throughput not latency"))
        if encoded.uncounted_texts:
            print(f"       ! {encoded.uncounted_texts:,} text(s) encoded with no token "
                  f"count available; the encode total is a lower bound")
        if uninstrumented:
            print(f"       ! {uninstrumented} sample(s) encoded with no counter at all "
                  f"(not even the live query encode); the encode total is a lower bound")
    if skipped:
        print(f"  skipped   : {skipped} (already retrieved — use --overwrite to re-run)")
    if errors:
        print(f"  errors    : {len(errors)} (see {os.path.join(out_dir, 'errors.jsonl')}; "
              f"re-run to retry — succeeded samples are skipped automatically)")


if __name__ == "__main__":
    main()
