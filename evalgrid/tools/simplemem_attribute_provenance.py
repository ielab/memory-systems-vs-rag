#!/usr/bin/env python3
"""
simplemem_attribute_provenance.py
=================================
Attribute every SimpleMem entry to the turn(s) of its window it restates, and
write the result beside the store as ``provenance_turn.jsonl``.

    python -m evalgrid.tools.simplemem_attribute_provenance \
        --dataset longmemeval_s --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507

then build qrels from it (a separate labeler segment, nothing overwritten):

    python -m evalgrid.pipeline.qrels_has_answer --system simplemem \
        --dataset longmemeval_s --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --provenance-sidecar provenance_turn.jsonl --relabel-set graded_v4
    # -> outputs/_qrels/simplemem/<dataset>/<mem-model>/has_answer-provenance_turn/

WHY THIS EXISTS
---------------
SimpleMem extracts a whole window in one LLM call and keeps no per-entry link
back to a turn, so the ingester records the WINDOW's turn ids on every entry it
returns (plugin.records_has_answer). An entry is then gold whenever any turn of
its window is, and one evidence turn makes every fact of its ~20-turn window
gold. On LongMemEval-S that puts ~60 entries (3,650 tokens) in the relabel_union
oracle prompt, of which ~5 restate an evidence turn; the rest are the window's
other facts. The oracle then scores BELOW SimpleMem's own best retrieved cells
(54-56 vs 58.0), because a top-k ranking drops that by-catch and the oracle
cannot. It measures "perfect selection at window granularity", not a ceiling.

The fix needs no re-ingest and no change to the store or to vendor/: every
entry's ``lossless_restatement`` is compared with the text of each turn in its
own window, and the entry is attributed to the turn(s) it restates.

HOW AN ENTRY IS ATTRIBUTED
--------------------------
Lexical, deterministic, and scoped to the entry's own window, so a fact can
never be attributed outside the turns the ingester says it came from:

  * content tokens of the restatement and of each window turn (lowercase,
    stopwords dropped, a plural ``s`` folded);
  * each token weighted by its IDF over the window's turns, so a word every
    turn shares ("user", "recommend") decides nothing;
  * score(turn) = the share of the restatement's weight that turn contains,
    over the tokens found in the window at all (the date the prompt's
    absolute-time rule writes into a restatement appears in no turn, and must
    not count against every turn equally);
  * a restatement that names its actor first ("A user ...", "The assistant
    ...") is scored among that role's turns (see attribute());
  * keep the best turn and any turn within RATIO of it, at most MAX_TURNS — a
    restatement often merges a user question with the assistant's answer.

An entry whose best score is below MIN_SCORE keeps its whole window
(``method: "window"``): the restatement shares too little wording with any one
turn to say where it came from, and guessing would drop real evidence. The
summary prints how many entries fell back.

WHAT IT WRITES
--------------
``<sample_mem_dir>/provenance_turn.jsonl``, one record per entry:

    {"unit_id": ..., "source_turn_ids": [...],
     "method": "lexical" | "lexical_unresolved" | "window" | "none",
     "score": float, "window_turns": int}

``lexical_unresolved`` is an entry attributed among the turns that could be
scored while its window ALSO names turns the local dataset does not have (13
LongMemEval-S stores were ingested from a copy with one extra session) — the turn it restates may be one of those.

``method: "none"`` is an entry whose store record carries no provenance; it
stays empty. Additive: no store field is written, the store's own
``source_turn_ids`` are untouched, and the window-level qrels keep being built
from them unless ``--provenance-sidecar`` asks otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

from evalgrid.benchmarks.adapters import DATASETS, build_adapter
from evalgrid.benchmarks.gold import turn_id
from evalgrid.lib.paths import ExperimentPaths

OUTPUT = "provenance_turn.jsonl"

#: A turn within this fraction of the best turn's score is kept too.
RATIO = 0.8
#: At most this many turns per entry; past it the attribution is not saying much.
MAX_TURNS = 3
#: Below this best score the entry keeps its whole window.
MIN_SCORE = 0.25

_STOP = set("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing done down during
each few for from further get got had has have having he her here hers herself him
himself his how i if in into is it its itself just like may me might more most much
must my myself no nor not now of off on once only or other our ours ourselves out over
own really same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up upon us very was we were what
when where which while who whom why will with would you your yours yourself yourselves
user assistant users assistants mentioned mention mentions asked ask asks said says
""".split())


def tokens(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if w in _STOP or (len(w) < 3 and not w.isdigit()):
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


_ACTOR = re.compile(r"\s*(?:the |a |an )?(user|assistant)\b", re.IGNORECASE)


def attribute(text: str, window: list[tuple[str, str, set[str]]]
              ) -> tuple[list[str], str, float]:
    """(attributed turn ids, method, best score) for one entry over its window.

    ``window`` is (turn id, role, content tokens) per turn. A restatement that
    opens with its actor ("A user ...", "The assistant ...", ~56% of them) is
    attributed among that role's turns first: an assistant reply routinely
    echoes the user's wording back, and without the prior it wins on overlap —
    "A user considered displaying a signed poster ... 500 copies" went to the
    assistant's "displaying your signed poster alongside your antique clock"
    instead of the user turn that holds the 500. The prior is dropped when no
    turn of that role reaches MIN_SCORE.
    """
    entry = tokens(text)
    df = Counter(w for _, _, toks in window for w in toks & entry)
    if not window or not df:
        return [tid for tid, _, _ in window], "window", 0.0
    n = len(window)
    idf = {w: math.log(1 + n / c) for w, c in df.items()}
    total = sum(idf.values())
    scored = [(sum(idf[w] for w in toks & entry if w in idf) / total, i, tid, role)
              for i, (tid, role, toks) in enumerate(window)]
    actor = _ACTOR.match(text or "")
    if actor:
        same_role = [x for x in scored if x[3] == actor.group(1).lower()]
        if same_role and max(x[0] for x in same_role) >= MIN_SCORE:
            scored = same_role
    best = max(x[0] for x in scored)
    if best < MIN_SCORE:
        return [tid for tid, _, _ in window], "window", best
    kept = sorted((x for x in scored if x[0] >= RATIO * best), key=lambda x: -x[0])[:MAX_TURNS]
    return [x[2] for x in sorted(kept, key=lambda x: x[1])], "lexical", best


def write_sample(sample_mem_dir: str, turn_text: dict[str, tuple[str, str]],
                 own: set[str]) -> Counter:
    stats = Counter()
    token_cache: dict[str, set[str]] = {}
    records = []
    with open(os.path.join(sample_mem_dir, "entries.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            window_ids = list(rec.get("source_turn_ids") or [])
            if not window_ids:
                records.append({"unit_id": rec["entry_id"], "source_turn_ids": [],
                                "method": "none", "score": 0.0, "window_turns": 0})
                stats["none"] += 1
                continue
            stats["foreign_ids"] += sum(1 for tid in window_ids if tid not in own)
            window = []
            for tid in window_ids:
                if tid not in turn_text:
                    stats["unknown_turn"] += 1
                    continue
                if tid not in token_cache:
                    token_cache[tid] = tokens(turn_text[tid][1])
                window.append((tid, turn_text[tid][0], token_cache[tid]))
            ids, method, score = attribute(rec.get("lossless_restatement", ""), window)
            if not ids:          # every window id unknown: keep what the store says
                ids, method = window_ids, "window"
            elif len(window) < len(window_ids) and method == "lexical":
                # Part of the window names turns the dataset does not have, so the
                # turn this entry restates may be one that could not be scored.
                # Kept, but marked, so its effect on the oracle can be counted.
                method = "lexical_unresolved"
            records.append({"unit_id": rec["entry_id"], "source_turn_ids": ids,
                            "method": method, "score": round(score, 4),
                            "window_turns": len(window_ids)})
            stats[method] += 1
            stats["window_ids"] += len(window_ids)
            stats["kept_ids"] += len(ids)
    tmp = os.path.join(sample_mem_dir, OUTPUT + ".tmp")
    with open(tmp, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp, os.path.join(sample_mem_dir, OUTPUT))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--mem-model", required=True)
    parser.add_argument("--mem-variant", default=None)
    parser.add_argument("--mem-dir", default=None,
                        help="Override the saved-memory dir "
                             "(default: outputs/_memory/simplemem/<dataset>/<mem-model>/)")
    parser.add_argument("--overwrite", action="store_true",
                        help=f"rewrite {OUTPUT} for samples that already have one")
    args = parser.parse_args()

    mem_dir = args.mem_dir or ExperimentPaths(system="simplemem", dataset=args.dataset,
                                              mem_model=args.mem_model,
                                              mem_variant=args.mem_variant).memory()
    if not os.path.isdir(mem_dir):
        sys.exit(f"No saved memory at {mem_dir}")

    samples = list(build_adapter(args.dataset).samples())
    # Every turn of the dataset, not only the sample's own: 13 LongMemEval-S
    # stores were ingested from a dataset copy with one extra session in the
    # haystack (every memory system, not only this one).
    # Where that session exists in another sample, resolving its ids lets an
    # entry be attributed to the turn it restates — where it can never be gold —
    # instead of being forced onto the nearest turn of this sample. Same id,
    # same text: LongMemEval shares sessions.
    all_text: dict[str, tuple[str, str]] = {}   # turn id -> (role, content)
    for sample in samples:
        for s in sample.sessions:
            for i, t in enumerate(s.turns):
                all_text.setdefault(turn_id(s.session_id, i), (t.role, t.content))

    total = Counter()
    leaked: list[tuple[str, int]] = []
    done = skipped = missing = 0
    for sample in samples:
        sample_dir = os.path.join(mem_dir, sample.sample_id)
        if not os.path.isfile(os.path.join(sample_dir, "entries.jsonl")):
            missing += 1
            continue
        if not args.overwrite and os.path.isfile(os.path.join(sample_dir, OUTPUT)):
            skipped += 1
            continue
        own = {turn_id(s.session_id, i) for s in sample.sessions for i in range(len(s.turns))}
        stats = write_sample(sample_dir, all_text, own)
        total += stats
        if stats["foreign_ids"]:
            leaked.append((sample.sample_id, stats["foreign_ids"]))
        done += 1

    entries = total["lexical"] + total["lexical_unresolved"] + total["window"] + total["none"]
    print(f"Wrote {OUTPUT} for {done} sample(s) under {mem_dir}"
          f"  ({skipped} already had one, {missing} with no store)")
    if entries:
        print(f"  entries             : {entries:,}")
        attributed = total["lexical"] + total["lexical_unresolved"]
        print(f"  attributed (lexical): {attributed:,}  ({attributed / entries:.1%})")
        if total["lexical_unresolved"]:
            print(f"    of which unresolved: {total['lexical_unresolved']:,}  window also names "
                  f"turns the dataset does not have (method: lexical_unresolved)")
        print(f"  kept whole window   : {total['window']:,}  ({total['window'] / entries:.1%})"
              f"  best score < {MIN_SCORE}")
        print(f"  no provenance       : {total['none']:,}")
        scored = entries - total["none"]
        if scored:
            print(f"  turns per entry     : {total['window_ids'] / scored:.1f} window -> "
                  f"{total['kept_ids'] / scored:.1f} attributed")
    if leaked:
        print(f"  [warn] {len(leaked)} sample(s) hold entries citing ANOTHER sample's turns "
              f"({sum(n for _, n in leaked):,} ids), e.g. {leaked[0][0]} — the store was "
              f"ingested from a different dataset copy, not an attribution "
              f"error; resolved against the whole dataset")
    if total["unknown_turn"]:
        print(f"  [warn] {total['unknown_turn']:,} window id(s) name no turn of the dataset")
    print("\nNext: build qrels from it under their own labeler segment —")
    print(f"  python -m evalgrid.pipeline.qrels_has_answer --system simplemem "
          f"--dataset {args.dataset} --mem-model {args.mem_model} "
          f"--provenance-sidecar {OUTPUT} [--relabel-set <label-set>]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
