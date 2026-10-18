"""
evalgrid/tools/native_dir.py
----------------------------
Print the ``<retriever>`` directory segment a native run WOULD write under.

    python -m evalgrid.tools.native_dir --exp experiments/a_mem/longmemeval.yaml \
        --native-retriever rrf:Qwen/Qwen3-Embedding-4B --native-variant grid
    -> native-rrf-Qwen_Qwen3-Embedding-4B-top25--grid

The depth comes from the spec's ``native_top_k`` (25 here) exactly as the two
native stages take it, so a derived directory and the run that fills it agree.
``--top-k`` overrides it; ``--published-depth`` drops it, matching a run that
passes the same flag.

WHY THIS EXISTS. ``pipeline/retrieve.py`` derives that segment itself and needs
no help — until you pass ``--out``, which it takes LITERALLY:

    out_dir = args.out or paths.retrieval(native)

So a run redirected to a smoke tree (``--out outputs/_smoke2_retrieval/...``)
loses the derivation, and a sweep that passes the same ``--out`` for every
retriever writes all of them to one directory. That failure is quiet in the worst
way: the first setting's ``run.tsv`` makes every later setting look already-done
to the resumability check, so the sweep "succeeds" in minutes and every cell of
the table holds the first retriever's ranking.

This prints exactly what ``paths.py`` would have derived, from the same
functions, so a redirected run can put it back:

    --out ${ROOT}_retrieval/$SYS/$(python -m evalgrid.tools.native_dir ...)

It is deliberately the same three calls the pipeline makes — apply_spec, then
``SystemPlugin.native_retriever_name``, then ``native_segment`` — rather than a
reimplementation, so the two cannot drift.
"""

from __future__ import annotations

import argparse
import sys

from evalgrid.lib.cli import add_exp_arg, apply_spec as apply_exp_spec
from evalgrid.lib.paths import native_segment
from evalgrid.pipeline.ingest import load_config
from evalgrid.systems import native_retrieval
from evalgrid.systems.registry import get_plugin


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Print the <retriever> segment a native run would write under.")
    add_exp_arg(p)
    p.add_argument("--system", default=None, help="System name (or via --exp)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Native retrieval depth; part of the segment. Omitted = the spec's "
                        "native_top_k, the same fallback pipeline/retrieve applies.")
    p.add_argument("--published-depth", action="store_true",
                   help="Ignore the spec's native_top_k, as the run flag of the same name "
                        "does — the segment then carries no -top<k>.")
    p.add_argument("--native-retriever", default=None, metavar="SPEC",
                   help="Swap spec, exactly as passed to pipeline/retrieve")
    p.add_argument("--native-variant", default=None,
                   help="Variant name, exactly as passed to pipeline/retrieve")
    p.add_argument("--config", default=None, help="Config path, if not default.yaml")
    p.add_argument("--mem-variant", default=None, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_exp_spec(parser, args)
    if not args.system:
        parser.error("missing --system (pass it, or a --exp that carries it)")
    # Same resolution order as pipeline/retrieve._run_native, so this tool cannot
    # print a directory the run would not write.
    if args.published_depth:
        if args.top_k is not None:
            parser.error("--published-depth and --top-k contradict each other.")
    elif args.top_k is None:
        args.top_k = getattr(args, "native_top_k", None)

    plugin = get_plugin(args.system)
    config, _ = load_config(plugin, args)
    # Same order as the pipeline: inject the swap, THEN read the name off the
    # resolved config. Reversed, the segment would name the published encoder.
    native_retrieval.apply_spec(config, args.native_retriever)
    # Raises on an unsupported combination (e.g. hypermem under retrieval_type
    # rrf) rather than printing a directory for a run that cannot happen.
    native_retrieval.check_supported(plugin, config)
    print(native_segment(plugin.native_retriever_name(config),
                         args.native_variant, args.top_k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
