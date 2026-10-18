#!/usr/bin/env python3
"""
lightmem_rebuild_provenance.py
==============================
Rebuild LightMem's per-unit turn provenance from the ingest trajectory, and
write it beside the store as ``provenance.jsonl``.

    python -m evalgrid.tools.lightmem_rebuild_provenance \
        --dataset longmemeval_s --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507

WHY THIS EXISTS
---------------
``payload["turn_id"]`` / ``payload["has_answer"]`` on a LightMem point are wrong
for ~28% of units, and the error is systematic. The chain:

  * ``assign_sequence_numbers_with_timestamps`` (vendor/lightmem/memory/utils.py)
    numbers EVERY message — both roles, all sessions — with one global counter,
    and builds ``turn_id_list`` / ``has_answer_list`` indexed by that counter.
  * the extraction prompt shows each user message as ``sequence_number // 2``
    (vendor/lightmem/factory/memory_manager/openai.py:309).
  * the extractor returns a ``source_id``, and
    ``utils.py:338-339`` inverts it as ``sequence_n = source_id * 2``.

``source_id * 2`` is the correct inverse ONLY while every user message sits at an
even global index, i.e. while the flattened stream strictly alternates
user/assistant from index 0 across every session. It does not. Our own pairing
step (``LightMemLongMemEval.ingest``) skips leading non-user turns, drops the odd
tail, and rejects role-misaligned pairs — 2,020 turns over LongMemEval-S. Every
dropped message flips the parity of everything after it, so from that point on
``source_id * 2`` addresses the neighbouring ASSISTANT turn instead of the user
turn the fact came from. Measured on the shipped stores: **28.15% of user
messages sit at an odd sequence_number**, ~26% of saved units carry an assistant
turn id despite ``messages_use: user_only``, and only ~70% of units agree with
the provenance rebuilt here.

The damage lands on the qrels: a unit whose ``turn_id`` points one turn to the
side is not gold, so the oracle context comes back EMPTY for 97 of 500 questions
and LightMem's oracle scores below its own retrieved cells. Under the rebuilt
mapping 93% of the recoverable empties get their gold units back.

The fix is the correct inverse, and it needs no re-ingest: the trajectory records
the whole ``extract_list`` (turn_id, has_answer and sequence_number per message)
alongside the extractor's raw output, so the label the prompt actually showed can
be recomputed and joined to the ``source_id`` the model actually returned.

WHAT IT WRITES
--------------
``<sample_mem_dir>/provenance.jsonl``, one record per saved point:

    {"unit_id": ..., "source_turn_ids": [...], "has_answer": bool,
     "provenance": "trajectory" | "payload"}

``provenance: "payload"`` marks a unit the rebuild could not reach — offline
OP-update rewrites a target entry's text in place, so an updated unit no longer
matches any trajectory fact verbatim. Those keep the stored ``turn_id``, which
is the pre-update target's own stamp; they are ~15% of units and are labelled so
a reader can tell them apart rather than having them silently blended in.

Two source turns can share a label when the parity break makes two user messages
collide on the same ``seq // 2`` (78 of 9,784 messages sampled, 0.8%). Those
attribute to BOTH turns — a set, matching the group-level provenance every other
system in the grid writes, rather than an arbitrary pick.

This is additive. It writes no store field, changes no unit text, and leaves
``points.jsonl`` untouched; ``LightMemPlugin.iter_units`` prefers the sidecar
when it is present and falls back to the payload when it is not, so a store
built before this still reads exactly as it did. Rebuild the qrels afterwards
(``evalgrid.tools.rebuild_qrels``) for the new mapping to reach an oracle.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import sys
from typing import Optional

from evalgrid.lib.paths import memory_dir as build_mem_dir

TRAJECTORY = "trajectory-full.jsonl"
POINTS = "points.jsonl"
OUTPUT = "provenance.jsonl"
EXTRACT_STEP = "lightmem.meta_text_extract"
UPDATE_STEP = "lightmem.update_entry"

#: The target memory an offline-update call was asked about. Non-greedy up to
#: the candidate list, and the LAST match of a record is the real one — see
#: _rewrite_origin.
TARGET_MEMORY = re.compile(r"Target memory:(.*?)\nCandidate memories:", re.S)

#: Cap on how far a rewritten unit is chased back through successive rewrites.
#: A cycle is impossible in principle (each rewrite produces new text) but the
#: map is built from model output, so the walk is bounded rather than trusted.
MAX_REWRITE_HOPS = 8


def _messages(record: dict) -> list[dict]:
    """Flatten one meta_text_extract call's extract_list back to messages."""
    payload = record.get("in") or {}
    raw = payload.get("text") if isinstance(payload, dict) else payload
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    out: list[dict] = []
    # extract_list nests batch -> segment -> message.
    for batch in raw:
        for segment in batch:
            for message in segment:
                if isinstance(message, dict):
                    out.append(message)
    return out


def _json_objects(text: str):
    """Every top-level JSON object in ``text``, in order.

    One ``meta_text_extract`` call returns one dict per API call it made, and the
    trajectory stores their replies JOINED. The previous single
    ``text[index("{"):rindex("}")+1]`` parse therefore spanned from the FIRST
    call's opening brace to the LAST call's closing one, which is not valid JSON
    whenever a call covered more than one segment — so every fact in those
    replies was dropped and its unit fell back to the payload stamp.

    Scanned with ``raw_decode`` from each candidate brace rather than split on a
    separator, because a reply's own strings may contain one.
    """
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1
            continue
        index = end
        if isinstance(obj, dict):
            yield obj


def _facts(record: dict):
    """Yield (source_id, fact_text) from one call's extractor output."""
    payload = record.get("out") or {}
    text = payload.get("text") if isinstance(payload, dict) else payload
    if not text:
        return
    for obj in _json_objects(str(text)):
        for entry in obj.get("data") or []:
            if not isinstance(entry, dict):
                continue
            fact = entry.get("fact") or entry.get("relation")
            if not fact:
                continue
            try:
                source_id = int(entry.get("source_id"))
            except (TypeError, ValueError):
                # A fact with no usable source_id is exactly what upstream
                # silently charges to source_id 0. Drop it instead: the payload
                # fallback is a more honest answer than a confident wrong turn.
                continue
            yield source_id, fact


def _rewrite_origin(record: dict, back: dict, stats: dict) -> None:
    """Record ``new text -> the text it replaced`` for one offline-update call.

    Offline update rewrites an entry's ``memory`` in place, so a rewritten unit
    matches no extracted fact by text and used to fall back to the payload stamp
    — 18.6% of units, the largest remaining hole in the mapping. The trajectory
    holds both halves: the update prompt prints ``Target memory:<before>`` and
    the reply carries ``new_memory``. Chaining them walks a rewritten unit back
    to the fact extraction produced.

    ``findall(...)[-1]``, never the first match: UPDATE_PROMPT's own instructions
    contain a worked example that also says "Target memory:", so the first match
    is the template's ``"The user likes coffee."`` on every single record.
    """
    matches = TARGET_MEMORY.findall(str((record.get("in") or {}).get("text") or ""))
    if not matches:
        return
    before = matches[-1].strip()
    try:
        reply = json.loads(str((record.get("out") or {}).get("text") or ""))
    except (ValueError, json.JSONDecodeError):
        return
    if not isinstance(reply, dict) or reply.get("action") != "update":
        return
    after = str(reply.get("new_memory") or "").strip()
    if not after or not before or after == before:
        return
    if after in back and back[after] != before:
        # Two entries rewritten to the same string. Keeping the first is
        # arbitrary, so it is counted rather than hidden: a non-zero value here
        # means some units' provenance was picked between two candidates.
        stats["rewrite_collisions"] = stats.get("rewrite_collisions", 0) + 1
        return
    back.setdefault(after, before)


def rebuild_sample(sample_mem_dir: str, stats: Optional[dict] = None):
    """fact text -> source turn ids, fact text -> has_answer, and the rewrite map."""
    path = os.path.join(sample_mem_dir, TRAJECTORY)
    turns: dict[str, set[str]] = collections.defaultdict(set)
    flags: dict[str, bool] = {}
    back: dict[str, str] = {}
    if stats is None:
        stats = {}
    if not os.path.isfile(path):
        return turns, flags, back
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("step") == UPDATE_STEP:
                _rewrite_origin(record, back, stats)
                continue
            if record.get("step") != EXTRACT_STEP:
                continue
            # The label the prompt printed for each USER message is seq // 2 —
            # recompute it rather than assuming source_id * 2 inverts it.
            by_label: dict[int, list[dict]] = collections.defaultdict(list)
            for message in _messages(record):
                if message.get("role") != "user":
                    continue
                turn_id = message.get("turn_id")
                if not turn_id:
                    continue
                by_label[message["sequence_number"] // 2].append(message)
            for source_id, fact in _facts(record):
                for message in by_label.get(source_id, ()):
                    turns[fact].add(message["turn_id"])
                    if message.get("has_answer"):
                        flags[fact] = True
                    flags.setdefault(fact, False)
    return turns, flags, back


def write_sample(sample_mem_dir: str) -> dict:
    """Join the rebuilt provenance onto the saved points. Returns counters."""
    points_path = os.path.join(sample_mem_dir, POINTS)
    empty = {"units": 0, "trajectory": 0, "update": 0, "payload": 0, "moved": 0,
             "rewrite_collisions": 0}
    if not os.path.isfile(points_path):
        return dict(empty)
    stats = dict(empty)
    turns, flags, back = rebuild_sample(sample_mem_dir, stats)
    records = []
    with open(points_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            point = json.loads(line)
            payload = point.get("payload") or {}
            text = payload.get("memory")
            stored = payload.get("turn_id")
            stats["units"] += 1
            rebuilt = turns.get(text)
            source = "trajectory"
            if not rebuilt:
                # Offline update rewrote this entry's text. Walk the rewrite
                # chain back to the text extraction actually produced.
                seen: set[str] = set()
                current = (text or "").strip()
                for _ in range(MAX_REWRITE_HOPS):
                    if current not in back or current in seen:
                        break
                    seen.add(current)
                    current = back[current]
                    rebuilt = turns.get(current)
                    if rebuilt:
                        text = current
                        source = "update"
                        break
            if rebuilt:
                ids = sorted(rebuilt)
                has_answer = bool(flags.get(text))
                stats["trajectory" if source == "trajectory" else "update"] += 1
                if stored not in rebuilt:
                    stats["moved"] += 1
            else:
                # Neither the extractor's own text nor any rewrite chain reaches
                # this unit. Keep what the store says and say so — this is the
                # count whose provenance is NOT known to be right.
                source = "payload"
                ids = [stored] if stored else []
                has_answer = bool(payload.get("has_answer"))
                stats["payload"] += 1
            records.append({
                "unit_id": point["id"],
                "source_turn_ids": ids,
                "has_answer": has_answer,
                "provenance": source,
            })
    with open(os.path.join(sample_mem_dir, OUTPUT), "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mem-model", required=True)
    parser.add_argument("--system", default="lightmem")
    parser.add_argument("--overwrite", action="store_true",
                        help="rewrite provenance.jsonl for samples that already have one")
    args = parser.parse_args()

    root = build_mem_dir(args.system, args.dataset, args.mem_model)
    if not os.path.isdir(root):
        print(f"no store at {root}", file=sys.stderr)
        return 1

    total = collections.Counter()
    samples = skipped = 0
    for name in sorted(os.listdir(root)):
        sample_dir = os.path.join(root, name)
        if not os.path.isdir(sample_dir):
            continue
        if not args.overwrite and os.path.isfile(os.path.join(sample_dir, OUTPUT)):
            skipped += 1
            continue
        stats = write_sample(sample_dir)
        if not stats["units"]:
            continue
        samples += 1
        total.update(stats)

    units = max(1, total["units"])
    resolved = total["trajectory"] + total["update"]
    print(f"Rebuilt provenance for {samples} sample(s) ({skipped} already had one)")
    print(f"  units               : {total['units']}")
    print(f"  from extractor text : {total['trajectory']}  "
          f"({100 * total['trajectory'] / units:.1f}%)")
    print(f"  via rewrite chain   : {total['update']}  "
          f"({100 * total['update'] / units:.1f}%)  (offline update rewrote the text)")
    print(f"  KEPT PAYLOAD STAMP  : {total['payload']}  "
          f"({100 * total['payload'] / units:.1f}%)  <- provenance NOT known correct")
    print(f"  turn id CHANGED     : {total['moved']}"
          f"  ({100 * total['moved'] / max(1, resolved):.1f}% of resolved)")
    if total.get("rewrite_collisions"):
        print(f"  rewrite collisions  : {total['rewrite_collisions']}"
              f"  (two entries rewritten to one string; first kept)")
    # --systems (plural) and no --mem-model: rebuild_qrels reads the coordinates
    # from experiments/<system>/<dataset>.yaml rather than taking them on the
    # command line. --overwrite is NOT optional here — without it every sample
    # that already has qrels is skipped, so the rebuild silently does nothing on
    # exactly the tree this tool was just run to correct.
    print("\nNext: rebuild the qrels so the new mapping reaches an oracle,")
    print(f"  python -m evalgrid.tools.rebuild_qrels --systems {args.system} --overwrite")
    print("\nThen drop the payload-mapping qrels, which were built from the OLD")
    print("sidecars and would otherwise be reused as-is by the oracle sweep:")
    print(f"  rm -rf outputs/_qrels_payloadmap/{args.system}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
