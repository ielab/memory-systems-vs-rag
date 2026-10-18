"""
evalgrid/benchmarks/adapters/locomo.py
------------------------------
Adapter for LoCoMo (local JSON).

File: data/LoCoMo/locomo10.json

Conversation keys follow the pattern session_<n> / session_<n>_date_time.
Each turn has: speaker, dia_id, text. Some turns also have img_url /
blip_caption / query for the multimodal subset.
"""

from __future__ import annotations

import json
from typing import Iterator

from evalgrid.benchmarks.types import Session, Sample, Query, Turn


class LoCoMoAdapter:

    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)

    def samples(self) -> Iterator[Sample]:
        for item in self.data:
            conv = item["conversation"]
            sessions: list[Session] = []
            n = 1
            while f"session_{n}" in conv:
                turns = [
                    Turn(
                        role=t["speaker"],
                        content=t["text"],
                        turn_id=t.get("dia_id"),
                        extra={
                            k: t[k]
                            for k in ("img_url", "blip_caption", "query")
                            if k in t
                        },
                    )
                    for t in conv[f"session_{n}"]
                ]
                sessions.append(
                    Session(
                        session_id=f"session_{n}",
                        turns=turns,
                        timestamp=conv.get(f"session_{n}_date_time"),
                    )
                )
                n += 1

            queries = [
                Query(
                    query_id=f'{item["sample_id"]}_q{i}',
                    question=qa["question"],
                    answer=qa.get("answer"),
                    question_type=str(qa.get("category")),
                    evidence_turn_ids=qa.get("evidence", []),
                )
                for i, qa in enumerate(item["qa"])
                if qa.get("category") != 5  # cat 5 = adversarial/multimodal, no ground-truth answer
            ]
            yield Sample(
                sample_id=item["sample_id"],
                source="locomo",
                sessions=sessions,
                queries=queries,
            )
