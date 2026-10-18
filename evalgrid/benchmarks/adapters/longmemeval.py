"""
evalgrid/benchmarks/adapters/longmemeval.py
-----------------------------------
Adapter for LongMemEval (local JSON).

Files:
  data/LongMemEval/longmemeval_s_cleaned.json
  data/LongMemEval/longmemeval_m_cleaned.json
  data/LongMemEval/longmemeval_oracle.json
"""

from __future__ import annotations

import json
from typing import Iterator

from evalgrid.benchmarks.types import Session, Sample, Query, Turn


class LongMemEvalAdapter:
    """Each item in the JSON is one question over a haystack of sessions."""

    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)

    @staticmethod
    def _unique_session_ids(session_ids: list[str]) -> list[str]:
        """Disambiguate a haystack session id that appears more than once.

        The dataset repeats one: 13 of the 500 longmemeval_s samples list the
        same ``haystack_session_id`` twice, with identical turns but DIFFERENT
        dates (e.g. 58bf7951 carries 07b7a667_1 at both 2023/05/23 and
        2023/05/29). Every stage downstream treats a session id as unique within
        a sample, and each breaks differently when it is not:

          * hypermem sets ``Episode.event_id`` from it, and upstream's
            ``hypergraph.episodes[event_id] = ...`` keeps only the last — so one
            episode vanishes from the store AFTER its LLM call was paid for, and
            the provenance left behind carries only the later date.
          * memorybank's ``bucket_by: session`` merges the two into one bucket,
            summarised as if they were one conversation.
          * every system synthesises turn ids as ``<session_id>:t<i>``, so the
            twins' turns collide — including in qrels_turns, which then writes
            duplicate ``_id`` lines into corpus.jsonl.

        None of the repeated sessions carries has_answer today, so no gold is
        affected; the fix is here rather than in each system because the
        invariant is the adapter's to keep, and the next dataset need not be so
        forgiving.

        The FIRST occurrence keeps the dataset's own id, so ids already written
        into an existing store stay valid; only the repeat is renamed.
        """
        seen: dict[str, int] = {}
        out: list[str] = []
        for sid in session_ids:
            seen[sid] = seen.get(sid, 0) + 1
            out.append(sid if seen[sid] == 1 else f"{sid}#{seen[sid]}")
        return out

    def samples(self) -> Iterator[Sample]:
        for item in self.data:
            sessions = [
                Session(
                    session_id=sid,
                    turns=[
                        Turn(
                            role=t["role"],
                            content=t["content"],
                            has_answer=t.get("has_answer", False),
                        )
                        for t in session
                    ],
                    timestamp=date,
                )
                for sid, date, session in zip(
                    self._unique_session_ids(item["haystack_session_ids"]),
                    item["haystack_dates"],
                    item["haystack_sessions"],
                )
            ]
            qid = item["question_id"]
            question_type = item["question_type"]
            if qid.endswith("_abs"):
                question_type += "/abstention"
            yield Sample(
                sample_id=qid,
                source="longmemeval",
                sessions=sessions,
                queries=[
                    Query(
                        query_id=qid,
                        question=item["question"],
                        answer=item["answer"],
                        question_type=question_type,
                        question_date=item.get("question_date"),
                        evidence_session_ids=item.get("answer_session_ids", []),
                    )
                ],
            )
