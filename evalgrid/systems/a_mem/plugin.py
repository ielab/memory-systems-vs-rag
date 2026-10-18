"""
evalgrid/systems/a_mem/plugin.py
--------------------------------
A-Mem's SystemPlugin: how to build it, and how to read its saved memory back as
a flat retrieval corpus.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from evalgrid.systems.base import MemoryUnit, SystemPlugin, route_mem_llm_to_openai


class AMemPlugin(SystemPlugin):

    name = "a_mem"

    # One note per turn, so the flag is the evidence turn itself.
    has_answer_granularity = "turn"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, config: dict, *, dataset: str, generator, phase: str):
        config = json.loads(json.dumps(config))  # don't mutate the caller's dict

        # Subclasses imported inside the branches, not at module scope: they
        # reach vendor/a_mem, which pulls in torch, sentence-transformers,
        # rank_bm25 and nltk. None of that may be a prerequisite for listing the
        # registry or for running some other system.
        if dataset == "locomo":
            from evalgrid.systems.a_mem.locomo import AMemLoCoMo
            return AMemLoCoMo(config, generator)
        if dataset.startswith("beam"):
            from evalgrid.systems.a_mem.beam import AMemBEAM
            return AMemBEAM(config, generator)
        from evalgrid.systems.a_mem.longmemeval import AMemLongMemEval
        return AMemLongMemEval(config, generator)

    def apply_mem_model(self, config: dict, mem_model: str,
                        mem_base_url: Optional[str] = None) -> dict:
        """A-Mem's memory LLM writes every note's keywords/tags/context and runs
        link generation and evolution, so it is the model that keys
        outputs/_memory/.

        It is also called at *answer* time, for query-keyword generation — the
        saved store does not capture that call, so an answer run needs this set
        too, not only ingest.
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
        """One unit per A-Mem note, in ingest order.

        ``text`` is the note as A-Mem itself indexes it — paper Eq. 3,
        ``concat(content, keywords, tags, context)`` — so swapping in BM25 or a
        dense retriever ranks the same strings A-Mem's own retriever would,
        and the comparison is about the representation rather than about which
        fields we happened to expose.

        The note's own ``links`` are deliberately not followed here. A-Mem's
        native retrieval expands each hit into its linked neighbours, but the
        flat-corpus protocol ranks independent units; folding a note's
        neighbours into its text would make one unit un-independent and inflate
        every token count. Link expansion stays in the native path
        (AMemSystem.retrieve_context).
        """
        path = os.path.join(sample_mem_dir, "notes.jsonl")
        if not os.path.isfile(path):
            return
        # Imported lazily: keeps the registry importable without A-Mem's deps.
        from evalgrid.systems.a_mem.system import index_document

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield MemoryUnit(
                    unit_id=rec["id"],
                    text=index_document(
                        rec["content"],
                        rec.get("context") or "General",
                        rec.get("keywords") or [],
                        rec.get("tags") or [],
                    ),
                    timestamp=rec.get("timestamp"),
                    has_answer=bool(rec.get("has_answer")),
                    source_turn_ids=list(rec.get("source_turn_ids") or []),
                    metadata={
                        "content": rec["content"],
                        "context": rec.get("context"),
                        "keywords": rec.get("keywords") or [],
                        "tags": rec.get("tags") or [],
                        "links": rec.get("links") or [],
                        "session_id": rec.get("session_id"),
                    },
                )

    def records_has_answer(self, sample_mem_dir: str) -> bool:
        """True if this sample's saved notes carry the has_answer key.

        A store written before provenance was threaded through ingest has no
        such key, and would otherwise look like a sample where no note is
        evidence — producing empty qrels instead of an error.
        """
        path = os.path.join(sample_mem_dir, "notes.jsonl")
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    return "has_answer" in json.loads(line)
        return False
