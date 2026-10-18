"""
evalgrid/systems/a_mem/locomo.py
--------------------------------
A-Mem wrapper for LoCoMo — the benchmark it was published on.

Aligned with upstream's own driver (``test_advanced.py::evaluate_dataset``):
  - one note per turn, both speakers, ingested in session order
  - note content is the literal string upstream builds,
    ``"Speaker " + speaker + "says : " + text``
  - retrieval is keyword-generation then ``find_related_memories_raw``
  - the answer prompt is upstream's, and varies by LoCoMo question category

Category 5 (adversarial) has no branch here. Upstream's category-5 prompt puts
the gold answer in front of the model as one of two options, which measures a
coin flip rather than memory; this repo's LoCoMo adapter drops those questions
before they reach any system (benchmarks/adapters/locomo.py), so the case
cannot arise. Categories 1-4 are reproduced verbatim.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Query, Session

from evalgrid.systems.a_mem.system import AMemSystem


class AMemLoCoMo(AMemSystem):

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        """One note per turn.

        A-Mem has no session concept and no end-of-ingest flush: every note is
        constructed, linked and evolved at the moment it is added, so ``is_last``
        has nothing to trigger. Contrast LightMem, which buffers into short-term
        memory and needs the final-turn flush.
        """
        for turn in session.turns:
            content = f"Speaker {turn.role}says : {turn.content}"
            self.add_note(
                content,
                timestamp=session.timestamp,
                turn_id=turn.turn_id,
                session_id=session.session_id,
                has_answer=turn.has_answer,
            )

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        """Upstream's per-category answer prompts, verbatim.

        ``question_type`` carries the LoCoMo category as a string (set by
        benchmarks/adapters/locomo.py from the dataset's ``category``).
        """
        category = str(query.question_type or "")
        question = query.question

        if category == "2":  # temporal reasoning
            prompt = f"""
                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

                            Question: {question} Short answer:
                            """
        elif category == "3":  # open-domain
            # Upstream's category-3 branch and its default branch hold the same
            # sentences but not the same string — this one opens with a newline
            # and indentation, the default opens flush at "Based". Kept apart so
            # the bytes we send match the bytes upstream sent.
            prompt = f"""
                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        else:  # 1 multi-hop, 4 single-hop
            prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        return [{"role": "user", "content": prompt}]
