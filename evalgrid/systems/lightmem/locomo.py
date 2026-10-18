"""
evalgrid/systems/lightmem/locomo.py
-----------------------------
LightMem wrapper for LoCoMo, aligned with the reference baseline
(zjunlp/LightMem/experiments/locomo):
  - flat extraction (METADATA_GENERATE_PROMPT_locomo) over both speakers
  - turn-by-turn ingest; segmentation/extraction forced on the final turn of
    the final session
  - combined retrieval across both speakers (reference reported setting:
    --retrieval-mode combined --total-limit 60), then split into per-speaker
    blocks for the answer prompt
  - rich timestamped answer prompt (verbatim ANSWER_PROMPT)
"""

from __future__ import annotations

import asyncio

from evalgrid.benchmarks.types import Session, Query

from .system import LightMemSystem, _to_lightmem_ts, _LOCOMO_OUT_FMT

# LoCoMo answer prompt (from zjunlp/LightMem/experiments/locomo/prompts.py)
_LOCOMO_ANSWER_PROMPT = """\
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp. For example, if a memory from
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example,
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {speaker_1_name}:

{speaker_1_memories}

Memories for user {speaker_2_name}:

{speaker_2_memories}

Question: {question}

Answer:"""


class LightMemLoCoMo(LightMemSystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._speaker_names: list[str] = []  # tracked during ingest

    def reset(self) -> None:
        self._speaker_names = []
        super().reset()

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        from lightmem.memory.prompts import METADATA_GENERATE_PROMPT_locomo
        if not session.turns:
            # Still flush: a turnless LAST session must not swallow the sample's
            # end-of-ingest extraction (see LightMemSystem.queue_add_memory).
            if is_last:
                self.flush_add_memory()
            return

        ts = _to_lightmem_ts(session.timestamp, _LOCOMO_OUT_FMT)

        # Track the two speaker names in order of first appearance.
        for turn in session.turns:
            if turn.role not in self._speaker_names:
                self._speaker_names.append(turn.role)

        # Reference message layout: each real utterance becomes a "user" message
        # (carrying its own speaker), padded with an empty "assistant" message so
        # LightMem's (user, assistant) pairing and source_id*2 math line up.
        # messages_use="user_only" means only the real user messages reach
        # extraction; the empty assistant is pure padding. Image captions are
        # appended inline. add_memory is called once per turn (one pair), with
        # segmentation/extraction forced on the sample's final call.
        for i, turn in enumerate(session.turns):
            content = turn.content
            if turn.extra.get("blip_caption"):
                content += f" (image description: {turn.extra['blip_caption']})"
            pair = [
                # LoCoMo ships real dia_ids, so the fallback never fires here —
                # it is the same expression every other adapter uses so the two
                # datasets cannot drift into different id namespaces.
                {"role": "user", "content": content, "time_stamp": ts,
                 "speaker_id": turn.role, "speaker_name": turn.role,
                 "session_id": session.session_id,
                 "turn_id": turn.turn_id or f"{session.session_id}:t{i}",
                 "has_answer": turn.has_answer},
                # The padding side carries the SAME turn_id: it is structural
                # filler for the source_id*2 arithmetic, not a second turn, so
                # naming it :t{i+1} would invent a turn that does not exist.
                {"role": "assistant", "content": "", "time_stamp": ts,
                 "speaker_id": turn.role, "speaker_name": turn.role,
                 "session_id": session.session_id,
                 "turn_id": turn.turn_id or f"{session.session_id}:t{i}",
                 "has_answer": turn.has_answer},
            ]
            # Held back by one so the sample's LAST call carries the flush,
            # rather than the last turn of the last session — which is the same
            # call here (LoCoMo has no empty sessions), but stops being it the
            # moment one appears. See LightMemSystem.queue_add_memory.
            self.queue_add_memory(
                pair, METADATA_GENERATE_PROMPT=METADATA_GENERATE_PROMPT_locomo)

        if is_last:
            self.flush_add_memory()

    def _retrieve_with_speaker(self, query: str, limit: int) -> list[tuple[str, str]]:
        """Combined retrieval returning (formatted_memory, speaker_name) pairs.

        Mirrors ``LightMemory.retrieve``'s formatting but preserves the speaker
        metadata (dropped by the public method) so combined results can be split
        into per-speaker blocks for the answer prompt.
        """
        lm = self._lightmem
        query_vector = lm.text_embedder.embed(query)
        results = lm.embedding_retriever.search(
            query_vector=query_vector, limit=limit, filters=None, return_full=True,
        )
        out: list[tuple[str, str]] = []
        for r in results:
            payload = r.get("payload", {})
            ts = payload.get("time_stamp", "")
            weekday = payload.get("weekday", "")
            memory = payload.get("memory", "")
            speaker = payload.get("speaker_name") or payload.get("speaker_id") or ""
            out.append((f"{ts} {weekday} {memory}", speaker))
        return out

    def answer(self, query: Query) -> str:
        speakers = (self._speaker_names + ["Unknown", "Unknown"])[:2]

        # Combined retrieval (top_k across both speakers by similarity), then
        # group the retrieved memories into per-speaker blocks for the prompt.
        results = self._retrieve_with_speaker(query.question, limit=self.top_k)
        s1_mems: list[str] = []
        s2_mems: list[str] = []
        for text, speaker in results:
            # Second speaker gets its own block; everything else (incl. unknown
            # speakers) falls back to the first block.
            if speaker == speakers[1]:
                s2_mems.append(text)
            else:
                s1_mems.append(text)

        s1_text = "\n".join(s1_mems) if s1_mems else "(none)"
        s2_text = "\n".join(s2_mems) if s2_mems else "(none)"

        prompt = _LOCOMO_ANSWER_PROMPT.format(
            speaker_1_name=speakers[0],
            speaker_2_name=speakers[1],
            speaker_1_memories=s1_text,
            speaker_2_memories=s2_text,
            question=query.question,
        )
        messages = [{"role": "user", "content": prompt}]
        return asyncio.run(self.generator.generate(messages, max_tokens=self.max_tokens))
