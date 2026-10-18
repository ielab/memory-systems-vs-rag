"""
evalgrid/systems/mem0/plugin.py
-------------------------------
Mem0's SystemPlugin: how to build it, and how to read its saved store back as a
flat retrieval corpus.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from evalgrid.systems.base import MemoryUnit, SystemPlugin, route_mem_llm_to_openai


class Mem0Plugin(SystemPlugin):

    name = "mem0"

    # One add() per (user, assistant) exchange; a fact carries that pair's
    # turn_ids.
    has_answer_granularity = "pair"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, config: dict, *, dataset: str, generator, phase: str):
        config = json.loads(json.dumps(config))  # don't mutate the caller's dict

        # Imported inside the branches: these reach vendor/mem0, which pulls in
        # qdrant_client, openai, posthog, httpx and requests. None of that may
        # be a prerequisite for listing the registry or running another system.
        if dataset == "locomo":
            from evalgrid.systems.mem0.locomo import Mem0LoCoMo
            return Mem0LoCoMo(config, generator)
        if dataset.startswith("beam"):
            from evalgrid.systems.mem0.beam import Mem0BEAM
            return Mem0BEAM(config, generator)
        from evalgrid.systems.mem0.longmemeval import Mem0LongMemEval
        return Mem0LongMemEval(config, generator)

    def apply_mem_model(self, config: dict, mem_model: str,
                        mem_base_url: Optional[str] = None) -> dict:
        """Mem0's memory LLM both extracts the facts and decides ADD/UPDATE/
        DELETE/NONE against the existing store, so it determines the entire
        contents of the memory and is the model that keys outputs/_memory/.

        It is NOT needed at answer time: 0.1.x retrieval is one embedding and one
        dense query.
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
        """One unit per stored fact — the easiest flattening in this repo.

        Mem0's memory is already a flat list of self-contained sentences with
        UUID ids: the extraction prompt demands standalone declarative facts, and
        the vector-store point IS the unit. No traversal, no verbalisation, no
        granularity choice. ``points.jsonl`` is written in created_at order, so
        reading it in order is chronological — which is what the full-context
        baseline feeds the generator.

        (Mem0^g would not flatten this way: its unit is a graph edge, which is
        not an independently retrievable document without verbalisation. That is
        one of the reasons the graph variant is out of scope — see
        our development notes)
        """
        path = os.path.join(sample_mem_dir, "points.jsonl")
        if not os.path.isfile(path):
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                point = json.loads(line)
                payload = point.get("payload") or {}
                yield MemoryUnit(
                    unit_id=str(point["id"]),
                    # "data" is where mem0 keeps the fact text; "memory" is the
                    # name it takes on only after get_all()/search() reshape it.
                    text=payload.get("data") or "",
                    # The benchmark's own session stamp, not mem0's created_at:
                    # created_at is an ISO rendering of the same instant, and
                    # every other system's corpus carries the raw form for
                    # pipeline.answer.format_fact to normalise.
                    timestamp=payload.get("session_ts"),
                    has_answer=bool(payload.get("has_answer")),
                    source_turn_ids=list(payload.get("turn_ids") or []),
                    metadata={
                        "session_id": payload.get("session_id"),
                        "session_ids": payload.get("session_ids") or [],
                        "created_at": payload.get("created_at"),
                        "updated_at": payload.get("updated_at"),
                        # Present only on a fact that was UPDATE'd after
                        # creation, which is the interesting subset: it is where
                        # Mem0's consolidation actually fired.
                        "updated": bool(payload.get("updated_at")),
                    },
                )

    def records_has_answer(self, sample_mem_dir: str) -> bool:
        """True if this sample's points carry the has_answer key.

        A store built before provenance metadata was threaded through `add()`
        has no such key, and would otherwise yield empty qrels for every sample
        rather than an error.
        """
        path = os.path.join(sample_mem_dir, "points.jsonl")
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and "has_answer" in (json.loads(line).get("payload") or {}):
                    return True
        return False
