"""
evalgrid/systems/lightmem/longmemeval.py
----------------------------------
LightMem wrapper for LongMemEval: flat extraction over user-only messages,
combined embedding retrieval, simple answer prompt.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from evalgrid.benchmarks.types import Session, Query

from evalgrid.systems.lightmem.system import LightMemSystem, _to_lightmem_ts


def native_answer_prompt(question: str, question_date: Optional[str],
                         facts: list[str]) -> list[dict]:
    """LightMem's own LongMemEval prompt, byte-for-byte.

    This is the prompt evalgrid.systems.base.shared_answer_prompt was derived
    from, and the difference between them is exactly the whitespace the shared
    version re-flowed: here the question time and question share one line joined
    by " and question:", and the memories start on the SAME line as the
    "Please answer..." lead-in. Kept verbatim so the native path reproduces
    LightMem's published setting rather than approximating it, and so the
    shared-vs-native prompt comparison has a real difference to measure.

    Facts arrive as LightMemory.retrieve() formats them
    ("<time_stamp> <weekday> <memory>"), not as pipeline.answer.format_fact
    renders them — that difference is part of what the native path is.
    """
    if question_date:
        question_line = f"Question time:{question_date} and question:{question}"
    else:
        question_line = f"Question: {question}"
    memories_text = "\n".join(facts) if facts else "(no relevant memories found)"
    user_content = (f"{question_line}\nPlease answer the question based on the "
                    f"following memories: {memories_text}")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]


class LightMemLongMemEval(LightMemSystem):

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        ts = _to_lightmem_ts(session.timestamp)
        if session.turns:
            turns = session.turns
            # Skip any leading non-user turns (mirrors reference behaviour).
            start = 0
            while start < len(turns) and turns[start].role != "user":
                start += 1
            # Strides from `start` over the ORIGINAL list rather than over a
            # slice of it: the pairs selected are identical either way (the old
            # slice-then-stride reached the same turns), but `i` is now the
            # index qrels_turns._turn_id uses, which is what makes the
            # synthesised turn_id below joinable to the raw-turn corpora.
            pairs = []
            covered: set[int] = set()
            for i in range(start, len(turns) - 1, 2):
                if turns[i].role == "user" and turns[i + 1].role == "assistant":
                    covered.update((i, i + 1))
                    pairs.append([
                        {"role": turns[i].role, "content": turns[i].content, "time_stamp": ts,
                         "session_id": session.session_id,
                         "turn_id": turns[i].turn_id or f"{session.session_id}:t{i}",
                         "has_answer": turns[i].has_answer},
                        {"role": turns[i + 1].role, "content": turns[i + 1].content, "time_stamp": ts,
                         "session_id": session.session_id,
                         "turn_id": turns[i + 1].turn_id or f"{session.session_id}:t{i + 1}",
                         "has_answer": turns[i + 1].has_answer},
                    ])
            # Turns no pair reached: the leading non-user run skipped above, the
            # odd tail, and the mis-aligned pairs the role check rejects. All
            # three are upstream's behaviour, so they are
            # counted rather than fixed — but counted they must be, because a
            # dropped turn is otherwise indistinguishable from a turn that
            # produced no memory. Over LongMemEval-S it is 2,020 turns, 3 of
            # which carry has_answer, and those 3 samples cannot answer their own
            # question from this store. save_memory reports it as dropped_turns.
            self.dropped_turns += len(turns) - len(covered)
            # The other side of the same ledger: what DID go to the model, so
            # save_memory can tell a turn that was never sent from one that was
            # sent and produced nothing.
            self.ingested_turn_ids.update(
                t["turn_id"] for pair in pairs for t in pair)
            # ...and the half of it that can actually come back with a unit.
            # `messages_use` drops the other role inside the extractor's own
            # concatenation, so counting those in turns_without_memory measured
            # the config rather than the model.
            roles = self._addressable_roles()
            self.addressable_turn_ids.update(
                t["turn_id"] for pair in pairs for t in pair if t["role"] in roles)
            # Held back by one so the sample's LAST call carries the flush; see
            # LightMemSystem.queue_add_memory. A session with no pairs correctly
            # contributes nothing here and no longer swallows the flush.
            for pair in pairs:
                self.queue_add_memory(pair)
        else:
            messages = [{"role": "user", "content": session.text or "", "time_stamp": ts,
                         "session_id": session.session_id}]
            self.queue_add_memory(messages)

        if is_last:
            self.flush_add_memory()
            cfg = self._lightmem.config
            self._lightmem.construct_update_queue_all_entries(
                top_k=cfg.update_queue_top_k,
                keep_top_n=cfg.update_queue_keep_top_n,
            )
            self._lightmem.offline_update_all_entries(
                score_threshold=cfg.update_sim_threshold
            )

    def retrieve_trace(self, query: Query) -> dict:
        """One Qdrant search. No generator call."""
        memories, unit_ids = self.retrieve_with_ids(query.question, limit=self.top_k)
        # Trace keys are the ones benchmarks/runner.py::_add_trace_fields and
        # SystemPlugin.native_unit_ids read.
        trace = {
            "query_id": query.query_id,
            "step2_retrieved_context": "\n".join(memories),
            "retrieved_unit_ids": unit_ids,
            # 1:1 with unit_ids, in rank order — what a token budget truncates.
            "retrieved_units": list(memories),
        }
        self.traces.append(trace)
        return trace

    def answer(self, query: Query) -> str:
        return self.answer_from_trace(query, self.retrieve_trace(query))

    def answer_from_trace(self, query: Query, trace: dict) -> str:
        memories = trace.get("retrieved_units")
        if memories is None:
            memories = [m for m in (trace.get("step2_retrieved_context") or "").split("\n") if m]
        messages = native_answer_prompt(query.question, query.question_date, memories)
        return asyncio.run(self.generator.generate(messages, max_tokens=self.max_tokens))
