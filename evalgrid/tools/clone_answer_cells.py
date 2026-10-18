#!/usr/bin/env python3
"""
clone_answer_cells.py
=====================
Answer an existing grid of cells with a NEW generator, through the Batch API.

The grid a generator sweep answers — six native systems x four budgets plus the
naive all/user/pair_turns rows, on two datasets — was built by several drivers
(native_grid.sh through query.sh, and the run_all_*.sh drivers), which
answer through a vLLM generator. An OpenAI generator needs none of that: its
cells differ from a template generator's cells ONLY in --gen-model. So instead
of re-deriving each cell's arguments, this reads them back from the template
cell's manifest.json (the exact args it was answered with) and swaps the
generator, which guarantees the new row sees the same rankings, qrels, budgets,
seed and max-tokens as the rows it will be compared with.

  python -m evalgrid.tools.clone_answer_cells --gen-model gpt-4o-mini --phase list
  python -m evalgrid.tools.clone_answer_cells --gen-model gpt-4o-mini --phase submit
  python -m evalgrid.tools.clone_answer_cells --gen-model gpt-4o-mini --phase collect
  python -m evalgrid.tools.clone_answer_cells --gen-model gpt-5-nano \\
      --gen-reasoning-effort minimal --phase submit

  # smoke: one cell, first sample only, into a scratch tree
  python -m evalgrid.tools.clone_answer_cells --gen-model gpt-5-nano \\
      --gen-reasoning-effort minimal --include 'pair_turns/longmemeval_s.*budget500' \\
      --limit 1 --out-root outputs/_smoke_openai --phase both

A cell whose predictions.jsonl already has the template's question count is
skipped, so every phase is safe to re-run. Score the result with
experiments/run_all_judge.sh.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

from evalgrid.lib.paths import model_slug

#: Manifest args carried over verbatim. Everything else is either the generator
#: (replaced), transport (base URL, concurrency), or native-path only.
_CARRIED = ("system", "dataset", "mem_model", "mem_variant", "labeler_model",
            "candidate_source", "retriever", "prompt", "context_budget",
            "gen_seed", "gen_temperature", "max_tokens", "gold", "emb_model",
            "qrels_dir", "retrieval_dir")


def _count_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _load_dotenv(path: str = ".env") -> None:
    """OPENAI_API_KEY for the child processes, as the judge drivers source .env."""
    if os.environ.get("OPENAI_API_KEY") or not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip().removeprefix("export ").strip(),
                                      value.strip().strip("'\""))


def template_cells(template_slug: str, budgets: list[str], emb: str,
                   include: str | None) -> list[str]:
    """Template cell dirs: answered by ``template_slug`` with a complete manifest."""
    cells = []
    for manifest in sorted(glob.glob(
            f"outputs/_answers/**/{template_slug}/budget*/manifest.json", recursive=True)):
        cell = os.path.dirname(manifest)
        if os.path.basename(cell) not in {f"budget{b}" for b in budgets}:
            continue
        if emb not in cell:
            continue
        if include and not re.search(include, cell):
            continue
        cells.append(cell)
    return cells


def build_command(cell: str, args) -> tuple[list[str], str, int]:
    """(answer.py argv, new predictions path, expected question count) for one cell."""
    with open(os.path.join(cell, "manifest.json")) as f:
        margs = json.load(f)["args"]
    parts = cell.split(os.sep)
    if parts[-2] != args.template_gen:
        raise ValueError(f"{cell}: generator segment is {parts[-2]!r}, "
                         f"expected {args.template_gen!r}")
    parts[-2] = model_slug(args.gen_model)
    new_cell = os.sep.join(parts)
    if args.out_root:
        new_cell = os.path.join(args.out_root, os.path.relpath(new_cell, "outputs/_answers"))
    out = os.path.join(new_cell, "predictions.jsonl")

    cmd = [sys.executable, "-m", "evalgrid.pipeline.answer"]
    for key in _CARRIED:
        value = margs.get(key)
        if value is None:
            continue
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    if margs.get("context_budget") is None:
        cmd += ["--top-k", str(margs["top_k"])]
    for qt in margs.get("exclude_question_type") or []:
        cmd += ["--exclude-question-type", qt]
    cmd += ["--gen-model", args.gen_model, "--context-window", str(args.context_window),
            "--out", out, "--batch", "--batch-phase", args.phase]
    if args.gen_reasoning_effort:
        cmd += ["--gen-reasoning-effort", args.gen_reasoning_effort]
    if args.gen_verbosity:
        cmd += ["--gen-verbosity", args.gen_verbosity]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return cmd, out, _count_lines(os.path.join(cell, "predictions.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gen-model", required=True, help="OpenAI generator, e.g. gpt-4o-mini")
    parser.add_argument("--gen-reasoning-effort", default=None,
                        help="gpt-5 family: minimal|low|medium|high (gpt-5-nano rejects none)")
    parser.add_argument("--gen-verbosity", default=None, choices=["low", "medium", "high"])
    parser.add_argument("--phase", choices=["list", "submit", "collect", "both"], default="list")
    parser.add_argument("--template-gen", default="Qwen_Qwen3-8B",
                        help="Generator slug whose cells are cloned (default: Qwen_Qwen3-8B)")
    parser.add_argument("--budgets", default="200,500,1000,2000")
    parser.add_argument("--emb", default="Qwen3-Embedding-0.6B",
                        help="Only cells whose path contains this (the fixed retrieval encoder)")
    parser.add_argument("--include", default=None, help="Regex a cell path must match")
    parser.add_argument("--context-window", type=int, default=128000,
                        help="The generator's context window (default 128000, gpt-4o-mini's)")
    parser.add_argument("--limit", type=int, default=None, help="Passed through: first N samples")
    parser.add_argument("--out-root", default=None,
                        help="Write under this root instead of outputs/_answers (smoke tests)")
    args = parser.parse_args()

    _load_dotenv()
    cells = template_cells(args.template_gen, args.budgets.split(","), args.emb, args.include)
    print(f">>> {args.phase}: {len(cells)} template cell(s) of {args.template_gen} "
          f"-> {args.gen_model}")
    done = failed = 0
    for cell in cells:
        cmd, out, expected = build_command(cell, args)
        have = _count_lines(out)
        if not args.limit and expected and have >= expected:
            print(f"[done] {out} ({have}/{expected})")
            done += 1
            continue
        if args.phase == "list":
            print(f"[todo] {out} ({have}/{expected})")
            continue
        print(f"\n[{args.phase}] {out}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"!!! FAILED: {out}", flush=True)
            failed += 1
    print(f"\n>>> {len(cells)} cell(s): {done} already complete, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
