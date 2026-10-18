"""
evalgrid/systems/memorybank/plugin.py
-------------------------------------
MemoryBank's SystemPlugin: how to build it, and how to read its saved memory
back as a flat retrieval corpus.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from evalgrid.systems.base import MemoryUnit, SystemPlugin, route_mem_llm_to_openai


class MemoryBankPlugin(SystemPlugin):

    name = "memorybank"

    # Turn-pair units carry their own two turn ids; a bucket summary
    # inherits the flag from its bucket.
    has_answer_granularity = "pair"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, config: dict, *, dataset: str, generator, phase: str):
        config = json.loads(json.dumps(config))  # don't mutate the caller's dict

        if dataset == "locomo":
            from evalgrid.systems.memorybank.locomo import MemoryBankLoCoMo
            return MemoryBankLoCoMo(config, generator)
        if dataset.startswith("beam"):
            from evalgrid.systems.memorybank.beam import MemoryBankBEAM
            return MemoryBankBEAM(config, generator)
        from evalgrid.systems.memorybank.longmemeval import MemoryBankLongMemEval
        return MemoryBankLongMemEval(config, generator)

    def apply_mem_model(self, config: dict, mem_model: str,
                        mem_base_url: Optional[str] = None) -> dict:
        """MemoryBank's *write* path uses no LLM at all — turn-pairs are appended
        verbatim. The memory model appears only in the summarisation pass, which
        is what produces the second granularity in the corpus, so it still keys
        outputs/_memory/.

        With every ``summarize`` flag off, the store is raw turn-pairs and no
        model runs: ``--mem-model`` then names the output directory and nothing
        else. That is a legitimate configuration (it is the naive-RAG corpus with
        MemoryBank's unit rendering), not a mistake — but it does mean two
        different ``--mem-model`` values would produce byte-identical stores in
        two directories.
        """
        llm = config.setdefault("memory_llm", {})
        llm["model"] = mem_model
        if mem_base_url:
            llm["base_url"] = mem_base_url
        route_mem_llm_to_openai(llm, llm.get("base_url"), mem_model)
        return config

    # ------------------------------------------------------------------
    # Flatten saved memory into a retrieval corpus
    # ------------------------------------------------------------------

    def iter_units(self, sample_mem_dir: str) -> Iterable[MemoryUnit]:
        """Turn-pairs and bucket summaries, in one flat stream.

        Both granularities in a single index competing for the same top-k is
        MemoryBank's actual design — a hierarchy with no traversal cost — so the
        corpus keeps them together and tags each unit with ``granularity``,
        which lets the retrieval metrics be broken down by level afterwards.

        The text is what upstream indexes, byte for byte
        (``store.render_turn_unit`` / ``render_summary_unit``), including the
        ``Conversation content on {bucket}:`` prefix. That prefix is inside the
        indexed string upstream rather than in metadata, which is a deliberate
        choice worth preserving — it is what lets a lexical retriever match on
        dates. Note the consequence for this repo:
        ``evalgrid.pipeline.answer.format_fact`` prepends the unit's timestamp
        too, so an answer prompt shows the date twice, once in MemoryBank's own
        wording. Stripping it would change what is retrieved.

        Order is ingest order: each bucket's turn-pairs, then that bucket's
        summary — the order upstream's loader emits documents in.

        Units marked forgotten are skipped (see store.apply_forgetting); the
        store keeps them, so the same memory can be re-flattened with forgetting
        off and the corpora diffed.
        """
        path = os.path.join(sample_mem_dir, "memory.json")
        if not os.path.isfile(path):
            return
        # store.py is pure stdlib, so flattening works without MemoryBank's
        # (unimportable) dependency tree.
        from evalgrid.systems.memorybank.store import iter_store_units

        with open(path) as f:
            data = json.load(f)

        for unit_id, text, granularity, record in iter_store_units(data):
            yield MemoryUnit(
                unit_id=unit_id,
                text=text,
                timestamp=record.get("timestamp"),
                has_answer=bool(record.get("has_answer")),
                source_turn_ids=list(record.get("turn_ids") or []),
                metadata={
                    "granularity": granularity,
                    "bucket": record.get("source"),
                    "memory_strength": record.get("memory_strength"),
                    "last_recall_date": record.get("last_recall_date"),
                    "session_id": record.get("session_id"),
                    "session_ids": record.get("session_ids") or [],
                },
            )

    def records_has_answer(self, sample_mem_dir: str) -> bool:
        """True if this sample's store carries has_answer on its units.

        Granularity note, because it differs from every other system here:
        MemoryBank's turn-pair unit is *one exchange*, so its has_answer is the
        OR over at most two turns — the finest provenance of any constructed
        memory in this repo, and directly comparable with LightMem's per-fact
        flag and A-Mem's one-note-per-turn. The summary units are coarser (OR
        over a bucket, i.e. a session), so a corpus containing both mixes two
        granularities of gold. Break the retrieval metrics down by
        ``metadata.granularity`` before comparing across systems.
        """
        path = os.path.join(sample_mem_dir, "memory.json")
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            data = json.load(f)
        for user_memory in data.values():
            for pairs in (user_memory.get("history") or {}).values():
                for record in pairs:
                    return "has_answer" in record
        return False
