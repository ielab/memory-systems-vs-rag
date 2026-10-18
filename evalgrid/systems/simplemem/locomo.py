"""
evalgrid/systems/simplemem/locomo.py
------------------------------------
SimpleMem wrapper for LoCoMo — the benchmark it ships a driver for.

Aligned with upstream's own harness (``test_locomo10.py``, which is not part of
the installable package and so is not vendored here — see
our development notes):
  - both speakers, one ``Dialogue`` per turn, in session order
  - ``speaker`` is the LoCoMo speaker name; upstream renders each dialogue as
    ``"[<timestamp>] <speaker>: <text>"`` (``Dialogue.__str__``), which is what
    the extraction prompt reads
  - the session's ``date_time`` as every turn's timestamp
  - upstream's answer prompt, whose "answer in a short phrase" and "dates as
    'DD Month YYYY'" instructions are LoCoMo output conventions

Windows are driven by the config's ``ingest_policy`` like everywhere else, and
``evalgrid/configs/simplemem/default.yaml`` — the only SimpleMem config — sets
``window``, which is the LoCoMo-faithful policy: LoCoMo is one long conversation
across ~20 sessions and upstream never flushes at a session boundary. Its
``window_size``/``overlap_size`` are the paper's 20/5 rather than the released
code's 40/2 ("for locomo"); if a LoCoMo run should reproduce the code's own
windowing instead, that is a config edit, not a code path.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Query, Session

from evalgrid.systems.simplemem.system import SimpleMemBase


class SimpleMemLoCoMo(SimpleMemBase):

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        dialogues = [
            self.make_dialogue(
                turn.role,                       # LoCoMo speaker name
                turn.content,
                timestamp=session.timestamp,
                turn_id=turn.turn_id,            # LoCoMo dia_id, e.g. "D1:3"
                session_id=session.session_id,
                has_answer=turn.has_answer,
            )
            for turn in session.turns
        ]
        self.ingest_dialogues(dialogues, is_last=is_last)

    # Upstream returns this string without calling the LLM when retrieval found
    # nothing (``AnswerGenerator.generate_answer``, first branch).
    empty_context_answer = "No relevant information found"

    def postprocess_answer(self, answer: str) -> str:
        """Unwrap upstream's ``{"reasoning": ..., "answer": ...}`` reply.

        ``AnswerGenerator.generate_answer`` returns only the ``answer`` field,
        so it has to be unwrapped here too or every hypothesis would reach the
        judge as a JSON blob. Parsing goes through the same scraper upstream
        uses, and falls back to the raw text exactly as upstream does.
        """
        try:
            parsed = self._llm.extract_json(answer)
        except Exception:
            return answer.strip()
        if isinstance(parsed, dict):
            return parsed.get("answer", answer.strip())
        return answer.strip()

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        """Upstream's answer prompt, verbatim.

        Built by delegating to the vendored ``AnswerGenerator`` rather than
        copied, so it cannot drift from the string upstream sends.
        """
        generator = self._sys.answer_generator
        prompt = generator._build_answer_prompt(query.question, context)
        return [
            {
                "role": "system",
                "content": (
                    "You are a professional Q&A assistant. Extract concise answers "
                    "from context. You must output valid JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]
