"""
evalgrid/lib/cli.py
------------------
Reusable argparse flags shared by the pipeline stages, plus the experiment-spec
loader. Keeping the flag defaults/help here is what stops them drifting between
stages.

The experiment spec (--exp SPEC.yaml) is the single source of truth for the
coordinates that several stages share and that key the output tree
(system / dataset / mem_model / mem_variant / labeler_model / candidate_source /
emb_model / retrievers / gen_model / judge_model). Pass it once instead of repeating those
flags on every stage; an explicit CLI flag always overrides the spec.

Because --exp can supply system/dataset/mem-model, those flags are NOT argparse
``required`` (argparse would reject a spec-only invocation before the spec is
read). Each stage instead calls ``require_resolved`` AFTER ``apply_spec`` to
enforce the values it actually needs — works the same with or without --exp.
"""

from __future__ import annotations

import argparse
import sys

import yaml

# Keys the YAML spec may carry. Each maps 1:1 to an argparse dest (snake_case),
# so a stage that defines the matching flag picks it up automatically.
SPEC_KEYS = {
    "system", "dataset", "mem_model", "mem_variant", "labeler_model", "candidate_source",
    "emb_model", "emb_base_url", "retrievers", "native_retrievers", "gen_model",
    "judge_model", "top_k", "native_top_k", "config", "system_config",
}
# `fusions` used to sit in that set and was read by nothing — no argparse dest, no
# driver, no stage. A spec key that resolves to nowhere is worse than a missing
# one: it passes the unknown-key check, so a value written there looks accepted
# and silently does nothing. Min-max fusion is spelled `fusion:MODEL` inside
# `retrievers:` like every other retriever.

# Spec keys whose value is a nested block rather than a flag value. They are
# passed through to the stage untouched instead of being matched against an
# argparse dest — this is what lets a system carry its own hyperparameters in a
# spec without widening SPEC_KEYS once per system.
#
# ``native_retrievers`` is the native counterpart of ``retrievers:`` — the same
# retriever strings, installed inside the system's own loop rather than ranking
# the flat corpus. It is a list for a DRIVER to loop over, not a flag value (the
# stages take one ``--native-retriever`` per run), so it passes through untouched.
#
# ``native_top_k`` is here for a different reason: it is a flag VALUE, but for a
# flag (--top-k) whose meaning depends on the path. On the native path --top-k is
# retrieval depth, pushed into the system's own retriever and keyed into the run
# directory; on the shared/flat path it is PROMPT depth, the axis every system is
# compared on (default 10). One spec drives both halves of a sweep, so a plain
# `top_k:` key would set both from one number and silently move the shared
# comparison. The native stages read this key instead — see _run_native in
# pipeline/retrieve.py and pipeline/answer.py — and the flat paths ignore it.
PASSTHROUGH_KEYS = {"system_config", "native_top_k", "native_retrievers"}


# --- experiment spec --------------------------------------------------------
def add_exp_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exp", default=None, metavar="SPEC.yaml",
        help="Experiment YAML spec; fills any shared flag not given on the CLI "
             "(system/dataset/mem-model/labeler-model/candidate-source/emb-model/"
             "retrievers/...). An explicit flag always overrides the spec.")


def _cli_provided_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Dests whose flag actually appears in argv (so the user set it explicitly).

    Compares option strings, not values — so `--labeler-model gpt-4o` counts as
    provided even when gpt-4o is also the default.
    """
    provided = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in argv or any(tok.startswith(opt + "=") for tok in argv):
                provided.add(action.dest)
                break
    return provided


def apply_spec(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fill args from --exp for any flag the user didn't pass on the CLI (CLI wins)."""
    if not getattr(args, "exp", None):
        return
    with open(args.exp) as f:
        spec = yaml.safe_load(f) or {}
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        parser.error(f"--exp {args.exp}: unknown key(s) {sorted(unknown)} "
                     f"(allowed: {sorted(SPEC_KEYS)})")
    provided = _cli_provided_dests(parser, sys.argv[1:])
    for dest, val in spec.items():
        if dest in PASSTHROUGH_KEYS:
            # Not a flag — attach the block itself, so a stage can read it
            # without every system's knobs needing their own argparse dest.
            setattr(args, dest, val)
            continue
        if not hasattr(args, dest):
            continue  # this stage doesn't take that flag — ignore it
        if dest in provided:
            continue  # user set it explicitly on the CLI — CLI wins
        if dest == "retrievers" and isinstance(val, list):
            val = ",".join(str(v) for v in val)  # YAML list -> the comma form the flags expect
        setattr(args, dest, val)


def require_resolved(parser: argparse.ArgumentParser, args: argparse.Namespace, *dests: str) -> None:
    """parser.error if any dest is still unset after the spec was applied."""
    missing = [d for d in dests if getattr(args, d, None) in (None, "")]
    if missing:
        flags = ", ".join("--" + d.replace("_", "-") for d in missing)
        parser.error(f"missing required value(s): {flags} (pass on the CLI or via --exp)")


# --- shared flags -----------------------------------------------------------
# Identity flags are required=False so --exp can supply them; stages enforce
# presence with require_resolved() after apply_spec().
def add_system_arg(parser: argparse.ArgumentParser, choices=None,
                   help: str | None = None) -> None:
    """--system, optionally restricted to the registered plugins.

    Stages that construct or read a system's memory pass
    ``choices=available_systems()``. Stages downstream of the corpus (retrieve,
    answer, judge) must NOT: they also serve the raw-turn baselines, whose
    --system is one of the pseudo-systems (``all_turns`` / ``user_turns`` /
    ``pair_turns``) and has no plugin.
    """
    parser.add_argument("--system", default=None, choices=choices,
                        help=help or "Memory system the data came from (e.g. lightmem)")


def add_dataset_arg(parser: argparse.ArgumentParser, choices=None, help: str | None = None) -> None:
    parser.add_argument("--dataset", default=None, choices=choices,
                        help=help or "Benchmark dataset key (e.g. longmemeval_s)")


def add_mem_model_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--mem-model", default=None,
                        help=help or "Memory model used to build the qrels dataset (keys outputs/_qrels/)")


def add_mem_variant_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--mem-variant", default=None,
                        help=help or "Ingest-parameter variant slug (e.g. rate0.4-stm768); suffixes "
                                     "the mem-model segment of every output dir so different ingest "
                                     "configs never share a tree. Omit for the default config.")


def add_labeler_model_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--labeler-model", default="gpt-4o",
                        help=help or "Labeler model the qrels builder was run with (default: gpt-4o) "
                                     "— keys the default qrels and retrieval run dirs")


def add_retriever_arg(parser: argparse.ArgumentParser, include_oracle: bool = False,
                      include_full: bool = False, include_native: bool = False,
                      required: bool = True, help: str | None = None) -> None:
    """--retriever, with the pseudo-retrievers each stage can actually run.

    "native" is a system's own query-time loop, which two stages produce:
    evalgrid.pipeline.retrieve --retriever native (loop only, saved for replay)
    and evalgrid.pipeline.answer --retriever native --prompt native (loop + answers, one pass). On the
    retrieve side it is a retriever like any other — same output tree, same
    run.tsv — differing only in that it ranks the system's own store rather
    than the flat qrels corpus, which is why it takes its own flag group there.

    So it is offered wherever a run is read (eval_retrieval) or produced
    (retrieve), and withheld from stages that can do neither, which is why
    include_native exists rather than the choice being unconditional.
    """
    choices = (["bm25", "dense", "sparse", "multi_vector", "fusion", "rrf", "hybrid"]
               + (["native"] if include_native else [])
               + (["oracle"] if include_oracle else [])
               + (["full"] if include_full else []))
    parser.add_argument("--retriever", required=required, choices=choices,
                        help=help or "Retriever to run/evaluate")


def add_emb_model_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--emb-model", default=None,
                        help=help or "Model for --retriever dense/sparse/multi_vector "
                                     "(also locates its run dir)")


def add_emb_base_url_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--emb-base-url", default=None,
                        help=help or "Override base URL for vllm/OpenAI-backed models "
                                     "(e.g. http://localhost:8001/v1); ignored for local models")


def add_candidate_source_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--candidate-source", choices=["sweep", "retrieval"], default="sweep",
                        help=help or "Candidate source the qrels builder was run with — 'retrieval' "
                                     "appends a '-retrieval' suffix to the labeler segment of the default "
                                     "paths (default: sweep)")


def add_concurrency_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--concurrency", type=int, default=8,
                        help=help or "Concurrent calls in flight (default: 8)")


def add_limit_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--limit", type=int, default=None,
                        help=help or "Only process the first N samples (applied after sharding)")


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Zero-based shard index for parallel processing (default: 0)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (default: 1, i.e. no sharding)")


def add_overwrite_arg(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    parser.add_argument("--overwrite", action="store_true",
                        help=help or "Re-process items that already have output written")


def add_trace_args(parser: argparse.ArgumentParser) -> None:
    """Trajectory recording (evalgrid/lib/trace.py). ``full``, every sample, by default.

    A run you cannot read back is a run you have to repeat, and every bug in
    our development notes was found in a ``full`` trajectory. §29 is the sharpest case:
    LightMem's turn provenance was rebuilt from the trajectory alone
    (evalgrid.tools.lightmem_rebuild_provenance), so a store ingested without one
    could only have been repaired by re-ingesting it. The cost is disk, not GPU —
    roughly 4 bytes per recorded token, ~10 GB for a full LongMemEval-S ingest of
    all six systems.

    ``full`` and not ``steps``: the levels below it truncate payloads to
    ``--trace-max-chars``, and the §29 rebuild joins facts on VERBATIM text, so a
    truncated trajectory would read as present and repair nothing.

    ``--trace-level off`` restores the NULL_RECORDER, which is not "a recorder
    that writes nothing" — ``record_calls`` declines to wrap any method at all,
    so the run carries exactly the code it carried before this feature existed.
    Pass it explicitly; scripts/lib/vllm.sh does, because with this default an
    OMITTED flag can no longer mean "off".
    """
    from evalgrid.lib.trace import DEFAULT_MAX_CHARS, LEVELS

    parser.add_argument("--trace-level", choices=list(LEVELS), default="full",
                        help="Record what the system's pipeline did, step by step: "
                             "'off' (nothing), 'calls' (one record per LLM call), "
                             "'steps' (+ named pipeline stages), 'full' (default; steps with "
                             "untruncated payloads — large, and the only level the §29 provenance "
                             "rebuild can read). Each level writes its OWN trajectory-<level>.jsonl, so "
                             "two levels never overwrite or interleave; read one with "
                             "python -m evalgrid.tools.inspect_trajectory.")
    parser.add_argument("--trace-samples", type=int, default=None, metavar="N",
                        help="Record only the first N samples. Omitted (the default) = every "
                             "sample, which is what a repairable store needs; cap it for a "
                             "throwaway run where the trajectory is the product, not the store.")
    parser.add_argument("--trace-max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help=f"Truncate each recorded prompt/response to this many characters "
                             f"(default: {DEFAULT_MAX_CHARS}; the original length is kept in the "
                             f"record). Ignored at --trace-level full.")


def build_recorder_from_args(args):
    """The one line every stage needs to turn --trace-* into a recorder.

    The getattr fallback stays "off" even though --trace-level now defaults to
    "full", and the asymmetry is deliberate: it fires only for a parser that
    never called add_trace_args, so that caller has no flag to turn recording
    back OFF with. retrieve.py and answer.py attach the group to their NATIVE
    subparser only, so their flat paths land here — tracing them by default
    would be unfixable from the command line. A stage that wants the default
    gets it by registering the flags.
    """
    from evalgrid.lib.trace import build_recorder

    return build_recorder(getattr(args, "trace_level", "off"),
                          max_chars=getattr(args, "trace_max_chars", None) or 2000,
                          max_samples=getattr(args, "trace_samples", None))
