"""
evalgrid/pipeline/ingest.py
--------------------------
Build a memory system's memory from a benchmark's conversations and SAVE it to
outputs/_memory/ — ingestion only, no answering.

This is the first stage of the pipeline. Its saved memory is then consumed by:
  * evalgrid.pipeline.qrels_has_answer — marks relevant facts to make a retrieval dataset
  * evalgrid.pipeline.retrieve — retrieves over that dataset
  * evalgrid.pipeline.answer   — answers from the retrieved (or oracle/full) facts

Because the memory is keyed only by what affects ingestion
(system/dataset/mem-model/mem-variant — see evalgrid.lib.paths.memory_dir), one
ingest run is reused across every downstream retrieval / answer / judge sweep;
you never re-ingest to change a generator, top-k, or labeler.

Note what the key does NOT include: the embedder. It is not a retrieval-only
knob for every system — a_mem's link and evolution neighbours, mem0's per-fact
k=5 ADD/UPDATE/DELETE search and lightmem's topic segmentation all run through
it at INGEST time, so swapping ``embedder.model`` changes the store. Nothing in
the path says so, and a run that swaps it resumes over the previous encoder's
samples. Pass --mem-variant when you change it; the config fingerprint below
warns if you forget.

Usage examples (--exp supplies system/dataset/mem-model from a spec; explicit
flags override it):
    # From an experiment spec
    python -m evalgrid.pipeline.ingest --exp experiments/lightmem/longmemeval.yaml

    # LightMem on LongMemEval-S, memory built by a vllm-served model
    python -m evalgrid.pipeline.ingest --system lightmem --dataset longmemeval_s \\
        --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 --mem-base-url http://localhost:8000/v1

LightMem is the only system that ingests. The naive-RAG and full-context
baselines don't build memory at all: they run over the raw user-turn corpus that
evalgrid.pipeline.qrels_turns writes (system "user_turns"), answered with
`--retriever dense/bm25/...` and `--retriever full` respectively.

Resumable & shardable: a sample whose memory is already saved is skipped (use
--overwrite to rebuild it), and --shard-id/--num-shards let N parallel jobs
cover disjoint subsets of the dataset while all writing to the same memory dir
— no coordination needed beyond the filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime

import yaml

from evalgrid.lib.cli import (
    add_dataset_arg,
    add_exp_arg,
    add_limit_arg,
    add_mem_model_arg,
    add_mem_variant_arg,
    add_overwrite_arg,
    add_shard_args,
    add_system_arg,
    add_trace_args,
    apply_spec,
    build_recorder_from_args,
    require_resolved,
)
from evalgrid.lib.io import write_json_atomic
from evalgrid.lib.manifest import write_manifest
from evalgrid.lib.paths import ROOT, ExperimentPaths

from evalgrid.benchmarks.runner import run
from evalgrid.benchmarks.adapters import DATASETS, build_adapter
from evalgrid.systems.registry import available_systems, get_plugin


# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------

class NullGenerator:
    """Stand-in generator for ingestion, which never answers.

    It satisfies the runner's token-usage snapshot (zero generation cost) but
    raises if a system actually tries to generate — which would mean ingestion
    wandered into the answer path. Lets ingest run without a generator model or
    its API key.
    """

    def __init__(self):
        from evalgrid.models.base import TokenUsage
        self.token_usage = TokenUsage()

    async def generate(self, *args, **kwargs):
        raise RuntimeError("NullGenerator.generate() called during ingestion")


def load_config(plugin, args) -> tuple[dict, str]:
    """Resolve and read the config this ingest run should build memory with.

    Returns (config, source) where ``source`` describes provenance for the
    console and the run record.

    Resolution order:
      1. --config PATH            explicit override, caller owns the location
      2. --mem-variant NAME       evalgrid/configs/<system>/<NAME>.yaml
      3. (neither)                evalgrid/configs/<system>/default.yaml

    Because --mem-variant also suffixes every output dir, (2) makes the path a
    run writes to always name the config that produced it — the pairing that
    used to be enforced by rejecting --config without --mem-variant.
    """
    path = args.config or plugin.config_path(args.mem_variant)
    if not os.path.isfile(path):
        available = []
        cfg_dir = os.path.dirname(plugin.config_path())
        if os.path.isdir(cfg_dir):
            available = sorted(f[:-5] for f in os.listdir(cfg_dir) if f.endswith(".yaml"))
        raise SystemExit(
            f"No config at {path}.\n"
            f"  --mem-variant names the config file: variant 'X' means "
            f"evalgrid/configs/{plugin.name}/X.yaml.\n"
            f"  Available variants for {plugin.name}: {', '.join(available) or '(none)'}"
        )
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Spec-level per-system overrides, applied on top of the file. Shallow by
    # design: a deep merge would let a spec silently half-override a nested
    # block, and these are meant for a knob or two, not a second config.
    overrides = getattr(args, "system_config", None) or {}
    if overrides:
        config.update(overrides)

    return config, os.path.relpath(path, ROOT)


def redact_secrets(value):
    """Copy of a resolved config with real API keys masked.

    route_mem_llm_to_openai puts OPENAI_API_KEY into the memory-LLM block, and
    config.json is uploaded alongside the store. vLLM's placeholder ("token")
    is left as-is, so fingerprints of existing vLLM stores do not change.
    """
    if isinstance(value, dict):
        return {k: ("<redacted>" if k == "api_key" and v not in (None, "", "token")
                    else redact_secrets(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value


def config_fingerprint(config: dict) -> str:
    """Hash of the resolved config, recorded beside the memory it built.

    A YAML edited after a run would otherwise be indistinguishable from the one
    that produced the store — the variant names the file, not its contents.
    """
    blob = json.dumps(redact_secrets(config), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def warn_config_changed(mem_dir: str, fingerprint: str, overwrite: bool) -> None:
    """Say so when this run's config differs from the one that built the store.

    Ingestion is resumable, which means a run started with an edited YAML does
    not rebuild the store — it ADDS to it. The result is one memory dir holding
    samples built under two configurations, with nothing on disk distinguishing
    them: config.json is rewritten by whichever run went last, so it names a
    config that produced only some of what sits beside it.

    The fingerprint was already being computed and written for exactly this
    case; it was just never read back. It is a warning and not an error because
    it has a legitimate false positive: ``memory_llm.base_url`` is part of the
    hashed config and moves with the vLLM endpoint, so the same experiment run
    against a different port would be blocked from resuming. index.jsonl keeps
    every run's fingerprint, so the full history stays recoverable either way.
    """
    path = os.path.join(mem_dir, "config.json")
    if overwrite or not os.path.isfile(path):
        return
    try:
        with open(path) as f:
            previous = json.load(f).get("config_sha256")
    except (OSError, ValueError):
        return
    if not previous or previous == fingerprint:
        return
    print(
        f"\n!!! WARNING: this memory dir was last written with config sha256:{previous}, "
        f"this run resolves to sha256:{fingerprint}.\n"
        f"!!! Resuming ADDS to the existing store — already-saved samples keep the OLD "
        f"config's memory and are skipped, so the dir would hold two configurations "
        f"with nothing marking which sample came from which.\n"
        f"!!! Pass --overwrite to rebuild, or --mem-variant <name> to give this config "
        f"its own dir. (A changed base_url alone also trips this; see index.jsonl for "
        f"every run's fingerprint.)\n")


# ---------------------------------------------------------------------------
# Adapter wrappers for sharding / dry-runs
# ---------------------------------------------------------------------------

class LimitedAdapter:
    def __init__(self, adapter, limit: int):
        self.adapter = adapter
        self.limit = limit

    def samples(self):
        for i, sample in enumerate(self.adapter.samples()):
            if i >= self.limit:
                break
            yield sample


class ShardedAdapter:
    """Yields only samples whose global index satisfies index % num_shards == shard_id.

    Lets N parallel ingest jobs cover disjoint subsets of the dataset while all
    writing to the same mem_dir — no coordination needed beyond the filesystem.
    """

    def __init__(self, adapter, shard_id: int, num_shards: int):
        self.adapter = adapter
        self.shard_id = shard_id
        self.num_shards = num_shards

    def samples(self):
        for i, sample in enumerate(self.adapter.samples()):
            if i % self.num_shards == self.shard_id:
                yield sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build & save a memory system's memory (ingest only)")
    add_exp_arg(parser)
    add_system_arg(parser, choices=available_systems(),
                   help="Memory system to ingest with")
    add_dataset_arg(parser, choices=list(DATASETS))

    # ---- Memory construction inputs -------------------------------------
    add_mem_model_arg(parser,
                      help="LLM that builds the memory. Keys the saved-memory dir; required.")
    add_mem_variant_arg(parser)
    parser.add_argument("--mem-base-url", default=None,
                        help="Override base URL for the memory LLM vllm server "
                             "(e.g. http://localhost:8000/v1)")
    parser.add_argument("--config", default=None,
                        help="Explicit config path, bypassing variant resolution. Normally "
                             "unnecessary: --mem-variant NAME already resolves to "
                             "evalgrid/configs/<system>/NAME.yaml (default.yaml with no variant).")

    # ---- Memory location ------------------------------------------------
    parser.add_argument("--mem-dir", default=None,
                        help="Directory to save memory into (per-sample subdirs). "
                             "Defaults to a key derived from system/dataset/mem-model so "
                             "downstream scripts find it automatically. Override only if you "
                             "also pass a matching --mem-dir to the downstream scripts.")

    # ---- Run control ----------------------------------------------------
    add_limit_arg(parser, help="Only ingest the first N samples (applied after sharding)")
    add_shard_args(parser)
    add_overwrite_arg(parser, help="Re-ingest samples even if their memory is already saved "
                                   "(default: skip them, so ingestion resumes)")
    add_trace_args(parser)
    args = parser.parse_args()
    apply_spec(parser, args)
    require_resolved(parser, args, "system", "dataset")

    # --mem-model keys the saved-memory dir, so it must be resolved before any
    # path is derived from it.
    if not args.mem_model:
        parser.error(f"--system {args.system} requires --mem-model")

    # A shard id outside its own shard count matches no sample: ShardedAdapter
    # yields where index % num_shards == shard_id, which is unreachable for
    # shard_id >= num_shards. The job then ingests nothing, writes an
    # ingest_usage.json covering 0 samples, prints "Done" and exits 0, silently.
    # It happens the moment SHARD_ID and NUM_SHARDS disagree across processes.
    if args.num_shards < 1:
        parser.error(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error(
            f"--shard-id {args.shard_id} is outside --num-shards {args.num_shards} "
            f"(valid: 0..{args.num_shards - 1}). This shard would match no sample and "
            f"the job would report success having ingested nothing. Give each "
            f"process a SHARD_ID in 0..NUM_SHARDS-1.")

    plugin = get_plugin(args.system)

    # An explicit --config no longer needs pairing with --mem-variant (the
    # variant now names the config file), but it still bypasses that link — so
    # without a variant it would land in the default config's dir and resume
    # over its already-saved samples. An explicit --mem-dir opts out; the caller
    # then owns the location.
    if args.config and not args.mem_variant and not args.mem_dir:
        parser.error("--config without --mem-variant: output would land in the default "
                     "config's memory dir and skip its already-saved samples. Pass "
                     "--mem-variant <name> (which resolves the config on its own, so "
                     "--config is usually unnecessary).")

    # Memory location, keyed only by what affects ingestion (see evalgrid.lib.paths).
    mem_dir = args.mem_dir or ExperimentPaths(
        system=args.system, dataset=args.dataset,
        mem_model=args.mem_model, mem_variant=args.mem_variant).memory()
    os.makedirs(mem_dir, exist_ok=True)
    write_manifest(mem_dir, args, stage="ingest")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    system_config, config_source = load_config(plugin, args)
    plugin.apply_mem_model(system_config, args.mem_model, args.mem_base_url)
    fingerprint = config_fingerprint(system_config)

    print(f"System    : {args.system}")
    print(f"Dataset   : {args.dataset}")
    print(f"Mem model : {args.mem_model}")
    print(f"Config    : {config_source} (sha256:{fingerprint})")
    print(f"Memory dir: {mem_dir}")
    if args.num_shards > 1:
        print(f"Shard     : {args.shard_id}/{args.num_shards}")
    if args.limit:
        print(f"Limit     : {args.limit} samples")

    # Before the config.json below overwrites the previous run's fingerprint.
    warn_config_changed(mem_dir, fingerprint, args.overwrite)

    # Record the resolved ingestion config + a one-line run-index entry beside
    # the memory (it's a property of the memory, shared across answer runs).
    # config_sha256 fingerprints the config as actually resolved, so a YAML
    # edited after the fact is detectable — the variant names the file, not its
    # contents.
    config = {
        "timestamp": ts,
        "system": args.system,
        "dataset": args.dataset,
        "mem_model": args.mem_model,
        "mem_base_url": args.mem_base_url,
        "mem_variant": args.mem_variant,
        "config_source": config_source,
        "config_sha256": fingerprint,
        "limit": args.limit,
        "num_shards": args.num_shards,
    }
    # Atomic: every shard of a sharded ingest writes this same file. index.jsonl
    # stays a plain append — one short line, which O_APPEND keeps intact, so the
    # resolved body goes only into config.json.
    #
    # The body, not just its hash. A fingerprint proves two stores differ; it
    # cannot say HOW, and a hash cannot be run backwards — when three smoke4
    # stores turned out to have been built at temperature 0.7, recovering that
    # meant bisecting git history for a config whose hash matched (our development notes,
    # "three stores ingested at the wrong temperature"). It is a few KB beside a
    # store measured in MB, and it makes a store self-describing: post-hoc,
    # `jq .config_resolved.memory_llm` answers what the run actually decoded
    # under. mem0's meta.json["sampling"] was the per-system version of this;
    # this generalises it to every system and every key.
    write_json_atomic(os.path.join(mem_dir, "config.json"),
                      {**config, "config_resolved": redact_secrets(system_config)},
                      indent=2)
    with open(os.path.join(mem_dir, "index.jsonl"), "a") as f:
        f.write(json.dumps(config) + "\n")

    adapter = build_adapter(args.dataset)
    if args.num_shards > 1:
        adapter = ShardedAdapter(adapter, args.shard_id, args.num_shards)
    if args.limit:
        adapter = LimitedAdapter(adapter, args.limit)

    # NullGenerator: ingestion never answers, so no --gen-model or API key is
    # needed, and a system that generates here fails loudly instead of silently
    # spending tokens.
    system = plugin.build(system_config, dataset=args.dataset,
                          generator=NullGenerator(), phase="ingest")

    # After build, before the first reset(): systems that construct their LLM
    # adapter in reset() (a_mem, simplemem, mem0) wrap it there by reading
    # self.recorder, so installing later would miss the first sample.
    recorder = build_recorder_from_args(args)
    system.install_recorder(recorder)

    try:
        run(system, adapter, phase="ingest", mem_dir=mem_dir, overwrite=args.overwrite)
    finally:
        recorder.close()

    print(f"\nDone — memory built and saved to {mem_dir}")
    print(f"       ingest usage: {os.path.join(mem_dir, 'ingest_usage.json')}")
    if recorder.enabled:
        print(f"       trajectories: {mem_dir}/<sample>/{recorder.filename}")
        if recorder.failures:
            print(f"       ! {recorder.failures} trajectory record(s) failed to write "
                  f"(the run itself is unaffected)")


if __name__ == "__main__":
    main()
