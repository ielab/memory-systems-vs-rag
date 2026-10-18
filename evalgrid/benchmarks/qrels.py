"""
evalgrid/benchmarks/qrels.py
---------------------
Writes a memory system's extracted facts out as retrieval ground truth (qrels).

Each sample becomes a standalone BEIR/TREC triple — corpus.jsonl (the facts),
queries.jsonl (the sample's question(s)), qrels/test.tsv (labeled relevance) —
scorable directly with pytrec_eval / beir.

The relevance labels themselves come from the caller. The two builders in use
derive them from LongMemEval-S's per-turn ``has_answer`` flag:
evalgrid/pipeline/qrels_has_answer.py (over constructed memories) and
evalgrid/pipeline/qrels_turns.py (over raw user turns).
"""

from __future__ import annotations

import json
import os

from evalgrid.benchmarks.gold import STAMPED_GOLD_LEVEL, gold_filename
from evalgrid.benchmarks.types import Query
from evalgrid.systems.base import MemoryUnit


def write_beir_sample(
    out_dir: str,
    units: list[MemoryUnit],
    labeled: list[tuple[Query, list[str]]],
    extra_gold: dict[str, list[tuple[Query, list[str]]]] | None = None,
) -> None:
    """Write one sample's standalone BEIR triple.

    ``units`` is the sample's corpus, in the order it should be read — the
    full-context baseline feeds corpus.jsonl to the generator as-is. They come
    from a SystemPlugin.iter_units for constructed memory, or straight from the
    raw turns for the naive-RAG baselines; the two paths converge here so both
    corpus shapes are byte-comparable downstream.

    ``labeled`` is (query, relevant_unit_ids) for every query in the sample —
    LongMemEval has exactly one; multi-query benchmarks share one corpus. It is
    the ingester-stamped gold level (STAMPED_GOLD_LEVEL, "strict"), and lands in
    that level's own explicitly-named file like every other level.

    ``extra_gold`` maps a wider gold level (evalgrid/benchmarks/gold.py) to its
    own labeling of the same corpus, each written beside test.tsv under
    gold_filename(level). Corpus and queries are shared: a level is a different
    answer to "what counts as evidence", not a different dataset, so adding one
    costs a few KB and invalidates no retrieval run.

    MemoryUnit.source_turn_ids is deliberately not emitted: adding a field to
    corpus.jsonl invalidates every existing retrieval cache keyed on it.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(
                {
                    "_id": u.unit_id,
                    "title": "",
                    "text": u.text,
                    "metadata": {"time_stamp": u.timestamp},
                },
                ensure_ascii=False,
            ) + "\n")

    with open(os.path.join(out_dir, "queries.jsonl"), "w") as f:
        for query, _ in labeled:
            f.write(json.dumps(
                {
                    "_id": query.query_id,
                    "text": query.question,
                    "metadata": {
                        "question_date": query.question_date,
                        "question_type": query.question_type,
                    },
                },
                ensure_ascii=False,
            ) + "\n")

    write_qrels_only(out_dir, {STAMPED_GOLD_LEVEL: labeled, **(extra_gold or {})})


def write_qrels_only(out_dir: str, gold: dict[str, list[tuple[Query, object]]]) -> None:
    """Write ``qrels/<gold_filename(level)>`` for each level in ``gold`` — nothing else.

    What adds a level to a sample that is already built: corpus.jsonl and
    queries.jsonl are left untouched, so no retrieval run or cache keyed on them
    moves, and the levels already on disk are not rewritten.

    Each entry is ``(query, relevant)``. ``relevant`` is a list of unit ids,
    written at score 1 — every published level, byte-identical to what this
    wrote before grades existed — or a ``{unit_id: grade}`` dict, written at its
    grade (the relabel levels, gold.RELABEL_LEVELS).
    """
    qrels_dir = os.path.join(out_dir, "qrels")
    os.makedirs(qrels_dir, exist_ok=True)
    for level, level_labeled in sorted(gold.items()):
        with open(os.path.join(qrels_dir, gold_filename(level)), "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for query, relevant in level_labeled:
                items = relevant.items() if isinstance(relevant, dict) else ((fid, 1) for fid in relevant)
                for fid, grade in items:
                    f.write(f"{query.query_id}\t{fid}\t{grade}\n")


def check_corpus_matches(out_dir: str, units: list[MemoryUnit]) -> None:
    """Refuse to label a built sample whose corpus is not the one ``units`` describe.

    Adding a level to an existing sample labels units rebuilt NOW against a
    corpus written THEN. If the two differ — the sample was built with another
    --roles, or the store was re-ingested since — the new qrels would name ids
    that are not in the corpus, or miss ones that are, and nothing downstream
    would notice: answer.py's oracle silently drops ids absent from the corpus.
    """
    path = os.path.join(out_dir, "corpus.jsonl")
    with open(path, encoding="utf-8") as fh:
        on_disk = [json.loads(line)["_id"] for line in fh if line.strip()]
    now = [u.unit_id for u in units]
    if on_disk != now:
        missing = sorted(set(now) - set(on_disk))[:3]
        extra = sorted(set(on_disk) - set(now))[:3]
        raise SystemExit(
            f"{path}: the corpus on disk ({len(on_disk)} units) is not the corpus rebuilt "
            f"now ({len(now)} units; e.g. not on disk {missing}, only on disk {extra}). "
            f"Rebuild this tree with --overwrite rather than adding a level to it.")
