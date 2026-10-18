"""
evalgrid/lib/manifest.py
-----------------------
Writes a manifest.json next to a stage's output, recording the resolved CLI
args (after --exp was applied and CLI overrides won) plus a UTC timestamp.

This is the reproducibility record: the --exp spec is the input you author,
but it doesn't capture CLI overrides or defaults (e.g. --labeler-model gpt-4o,
--k-values 5,10,20). The manifest is the immutable "what actually ran" snapshot,
written into the same dir as the stage's output so the two never drift apart.
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Optional

from evalgrid.lib.io import write_json_atomic


def write_manifest(out_dir: str, args: argparse.Namespace, *, stage: str,
                   filename: str = "manifest.json",
                   extra: Optional[dict] = None) -> None:
    """Write out_dir/<filename> from the resolved args namespace.

    `stage` names the pipeline step (e.g. "retrieve"). The --exp path itself is
    dropped — its values are already folded into the recorded args, and keeping
    the path would falsely imply the spec on disk still matches this run.

    `filename` defaults to manifest.json; pass a disambiguated name when several
    runs legitimately share one dir (e.g. judge writes one per judge model, like
    its per-judge metrics file).

    `extra` adds top-level keys beside "args" — for facts about the run that are
    not arguments to it, so they cannot be recovered from the namespace. It sits
    outside "args" deliberately: a reader reconstructing the command line from
    the manifest must not pick these up as flags. It cannot overwrite the three
    keys below.

    Written atomically: every shard of a sharded ingest writes this file into the
    same mem_dir, so a plain truncate-then-write can be read (or written) half
    finished. See evalgrid/lib/io.py.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        **(extra or {}),
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {k: v for k, v in vars(args).items() if k != "exp"},
    }
    write_json_atomic(os.path.join(out_dir, filename), manifest, indent=2, default=str)
