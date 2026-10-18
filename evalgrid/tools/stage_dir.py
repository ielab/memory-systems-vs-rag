"""
evalgrid/tools/stage_dir.py
---------------------------
Print the directory a query-path stage WOULD write under, from ``lib/paths``.

    python -m evalgrid.tools.stage_dir --exp experiments/simplemem/longmemeval.yaml \
        --stage retrieval --retriever-seg native-bm25-top45--grid
    -> outputs/_retrieval/simplemem/longmemeval_s/Qwen_.../has_answer/native-bm25-top45--grid

WHY THIS EXISTS. The same reason ``native_dir.py`` does, one level up. Every
stage of ``pipeline/`` already derives its own keyed path and needs no help —
until ``--out`` is passed, which each takes LITERALLY. ``scripts/engine/
query.sh`` passes it for all four stages, and built the values with four
hand-written bash functions:

    qrels_dir()     { echo "${OUT_ROOT}_qrels/$1"; }
    retrieval_dir() { echo "${OUT_ROOT}_retrieval/$1/$MODE-top$k"; }
    answers_dir()   { echo "${OUT_ROOT}_answers/$1/<retr>/<depth>"; }
    eval_dir()      { echo "${OUT_ROOT}_eval/$1/<retr>"; }

That is a second implementation of ``lib/paths`` in another language, and it had
drifted: it carries the system and the retriever, and drops ``<dataset>``,
``<mem-seg>``, ``<labeler-seg>`` and (for answers) ``<gen-seg>`` — four of the
five keys. The whole 2026-08 grid landed in
``outputs/_retrieval/simplemem/native-bm25-top45--grid/`` and the checkers, which
build the keyed path, reported every cell MISSING.

The cost is not tidiness. Those segments exist to stop silent overwrite, and
both ``gen_segment`` and ``native_segment`` say so in their own docstrings: the
retrieve stage resumes per sample on ``run.tsv`` and the answer stage resumes per
``question_id``, so a second memory variant, dataset, labeler or generator
written to the same flat cell is SKIPPED as already-done and the first run's
output is served under the second's name. lightmem ships three configs and
hypermem two, so that is a live hazard, not a hypothetical one.

WHAT IT DOES NOT DO. It does not rebuild the ``<retriever>`` segment — bash
already derives that correctly (``native-bm25-top45--grid`` is what
``native_segment`` produces, byte for byte), via ``native_dir`` for a swapped
run. Pass it in with ``--retriever-seg``. Re-deriving it here would mean
re-encoding the driver's MODE/NATIVE_RETRIEVER/TOP_K logic, i.e. making a THIRD
copy of something that is already right.

``--root`` overrides the tree so a throwaway run (MEM_ROOT=outputs/_smoke2) keeps
its scoring beside its store, which is what that knob is for; the keyed segments
below the root are the same either way.
"""

from __future__ import annotations

import argparse
import os
import sys

from evalgrid.lib.cli import add_exp_arg, apply_spec as apply_exp_spec
from evalgrid.lib.paths import (
    ExperimentPaths,
    answer_depth_segment,
    labeler_segment,
)

STAGES = ("qrels", "retrieval", "answers", "eval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Print the keyed output dir for one query-path stage.")
    add_exp_arg(p)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--system", default=None, help="System name (or via --exp)")
    p.add_argument("--dataset", default=None, help="Dataset name (or via --exp)")
    p.add_argument("--mem-model", default=None, help="Memory model (or via --exp)")
    p.add_argument("--mem-variant", default=None)
    p.add_argument("--labeler-model", default=None,
                   help="Labeler that keys the qrels/retrieval/eval level (or via --exp)")
    p.add_argument("--candidate-source", default=None,
                   help="'sweep' (default) or 'retrieval'; suffixes the labeler segment")
    p.add_argument("--gen-model", default=None,
                   help="Generator that keys the answers level (or via --exp)")
    p.add_argument("--prompt", default="shared", choices=("shared", "native"))
    p.add_argument("--retriever-seg", default=None, metavar="SEG",
                   help="The <retriever> segment, already derived by the caller "
                        "(e.g. 'bm25', 'native-bm25-top45--grid'). Required for "
                        "retrieval/answers/eval.")
    p.add_argument("--depth-seg", default=None, metavar="SEG",
                   help="The answers <topk-seg> ('top10', 'budget500', 'all'). "
                        "Required for --stage answers unless --top-k/--context-budget "
                        "is given.")
    p.add_argument("--top-k", type=int, default=None,
                   help="Build the answers depth segment from k instead of --depth-seg")
    p.add_argument("--context-budget", type=int, default=None,
                   help="Build the answers depth segment from a token budget")
    p.add_argument("--labeler-seg-for-answers", action="store_true",
                   help="Include the labeler level under answers. Oracle answers key on "
                        "it; retriever answers do not (answers_dir's default).")
    p.add_argument("--root", default=None, metavar="DIR",
                   help="Write under DIR instead of the repo's outputs/ tree "
                        "(throwaway runs; MEM_ROOT)")
    p.add_argument("--relative", action="store_true",
                   help="Print a path relative to the repo root rather than absolute")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_exp_spec(parser, args)

    missing = [n for n in ("system", "dataset") if not getattr(args, n, None)]
    if missing:
        parser.error("missing " + ", ".join("--" + m.replace("_", "-") for m in missing)
                     + " (pass it, or a --exp that carries it)")
    if args.stage != "qrels" and not args.retriever_seg:
        parser.error(f"--stage {args.stage} needs --retriever-seg")

    # Same default the pipeline stages apply, so a derived dir and the run that
    # fills it agree even when the spec omits the key.
    labeler_seg = labeler_segment(args.labeler_model or "has_answer",
                                  args.candidate_source or "sweep")

    P = ExperimentPaths(system=args.system, dataset=args.dataset,
                        mem_model=args.mem_model, mem_variant=args.mem_variant)

    if args.stage == "qrels":
        out = P.qrels(labeler_seg)
    elif args.stage == "retrieval":
        out = P.retrieval(args.retriever_seg)
    elif args.stage == "eval":
        out = P.eval(labeler_seg, args.retriever_seg)
    else:
        depth = args.depth_seg
        if depth is None:
            if args.top_k is None and args.context_budget is None:
                parser.error("--stage answers needs --depth-seg, --top-k or --context-budget")
            depth = answer_depth_segment(args.top_k, args.context_budget)
        if not args.gen_model:
            parser.error("--stage answers needs --gen-model (pass it, or a --exp "
                         "that carries it)")
        out = P.answers(args.retriever_seg, args.gen_model, depth,
                        labeler_seg=labeler_seg if args.labeler_seg_for_answers else None,
                        prompt=args.prompt)

    if args.root:
        # Re-root: keep every keyed segment, swap the tree. Split on the
        # "outputs/" the builders produce rather than recomputing the tail, so
        # the two cannot describe different layouts.
        marker = os.sep + "outputs" + os.sep
        if marker in out:
            out = os.path.join(args.root.rstrip(os.sep), out.split(marker, 1)[1])
    elif args.relative:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = os.path.relpath(out, repo)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
