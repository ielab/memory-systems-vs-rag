"""
evalgrid/systems/simplemem/longmemeval.py
-----------------------------------------
SimpleMem wrapper for LongMemEval-S — the benchmark its Table 2 reports.

SimpleMem ships no LongMemEval runner (its only benchmark driver is
``test_locomo10.py``; the LongMemEval loader under ``simplemem/evolver/`` belongs
to the separate EvolveMem subsystem and prompts on the gold ``question_type``
field). So nothing here is "the reference implementation", and the deviations
are the interesting part of any number this produces:

  - both speakers (``roles: all``, the default). The paper's Table 2 reports
    SS-Assistant 75.00 (gpt-4.1-mini) / 76.79 (gpt-4.1), and 51 of those 56
    questions have evidence ONLY in assistant turns, so that column is
    unattainable under ``user_only`` — which therefore cannot be the setting the
    paper used. ``roles: user_only`` remains as a labelled ablation.
  - session-aligned windows (config default), so an extraction call never spans
    two sessions. Upstream's 40-turn window would straddle ~5 LongMemEval
    sessions, mixing unrelated conversations into one extraction call and one
    provenance record.
  - the shared answer prompt BY DEFAULT, not upstream's ``AnswerGenerator``
    one. Holding the prompt fixed is what keeps the comparison about memory
    representation; upstream's asks for a short phrase with dates as
    "DD Month YYYY", a LoCoMo output convention.

``answer_prompt: native`` opts into ``AnswerGenerator``'s prompt instead —
opt-in rather than default because every run already in ``outputs/`` was
answered under the shared prompt, and flipping the default would change what
those directories mean without changing their names.

NOTE which upstream prompt this is, because there are two and only one is
usable. ``core/answer_generator.py`` holds the dataset-agnostic one the main
pipeline sends, and that is what is wired here. The other,
``evolver/benchmarks/longmemeval.py:build_answer_prompt``, is the one that
looks LongMemEval-specific — and it is unusable: it switches its instructions
on ``meta["qtype"]``, the benchmark's GOLD ``question_type`` field, so sending
it would leak the question's category into the prompt. It also belongs to the
separate EvolveMem self-evolution subsystem, which is not the path behind the
paper's Table 2.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Query, Session
from evalgrid.systems.base import shared_answer_prompt

from evalgrid.systems.simplemem.system import SimpleMemBase


class SimpleMemLongMemEval(SimpleMemBase):

    def __init__(self, config: dict, generator):
        self.roles = str(config.get("roles", "all"))
        if self.roles not in ("user_only", "all"):
            raise ValueError(f"roles must be 'user_only' or 'all', got {self.roles!r}")
        #: "shared" (default) or "native" — see the module docstring. Read here
        #: rather than at call time so a typo fails at construction, before a
        #: 500-sample run has answered anything.
        self.answer_prompt_mode = str(config.get("answer_prompt", "shared"))
        if self.answer_prompt_mode not in ("shared", "native"):
            raise ValueError(
                f"answer_prompt must be 'shared' or 'native', got "
                f"{self.answer_prompt_mode!r}")
        super().__init__(config, generator)

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        dialogues = []
        if session.turns:
            for i, turn in enumerate(session.turns):
                if self.roles == "user_only" and turn.role != "user":
                    continue
                dialogues.append(self.make_dialogue(
                    turn.role,
                    turn.content,
                    timestamp=session.timestamp,
                    # LongMemEval ships no turn ids, so synthesise the same ones
                    # evalgrid.pipeline.qrels_turns does — indexed into the
                    # ORIGINAL turn list, so an id means the same turn whether
                    # it reaches the corpus through this system or through the
                    # raw-turn baseline.
                    turn_id=turn.turn_id or f"{session.session_id}:t{i}",
                    session_id=session.session_id,
                    has_answer=turn.has_answer,
                ))
        elif session.text:
            # Document-shaped session (no turns). One dialogue for the whole
            # text: SimpleMem's window is a list of utterances and it has no
            # chunker of its own.
            dialogues.append(self.make_dialogue(
                "user",
                session.text,
                timestamp=session.timestamp,
                session_id=session.session_id,
            ))

        self.ingest_dialogues(dialogues, is_last=is_last)

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        if self.answer_prompt_mode == "native":
            return self._native_answer_messages(context, query.question)
        # context is one block of SimpleMem's own formatting, not a list of
        # facts, so it goes in as a single "fact" rather than being re-split.
        return shared_answer_prompt(
            query.question, query.question_date, [context] if context else []
        )

    def _native_answer_messages(self, context: str, question: str) -> list[dict]:
        """Upstream's ``AnswerGenerator`` prompt, verbatim.

        Built by delegating to the vendored generator rather than copied, so it
        cannot drift from the string upstream sends — the same construction
        ``locomo.py`` uses, and the system message is upstream's too.

        The question DATE is dropped: upstream's template has no slot for one,
        and adding it would make this our prompt rather than upstream's.

        Not reproduced: upstream sends this call at ``temperature=0.1``
        (``core/answer_generator.py:65``), not at the harness generator's 0.0.
        Table 7's "temperature = 0.0" describes the model backend, not this call
        site — the two disagree, and the code is what ran. The generator config
        is deliberately the harness's for every system, so this is a known and
        shared departure rather than a simplemem one; see a_mem's longmemeval.py
        for the same note against its 0.7.
        """
        generator = self._sys.answer_generator
        prompt = generator._build_answer_prompt(question, context)
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

    def postprocess_answer(self, answer: str) -> str:
        """Unwrap upstream's ``{"reasoning": ..., "answer": ...}`` reply.

        A no-op under the shared prompt, which asks for the answer directly.
        Under the native one the reply is a JSON object and
        ``AnswerGenerator.generate_answer`` returns only its ``answer`` field,
        so it has to be unwrapped here too or every hypothesis would reach the
        judge as a JSON blob. Parsing goes through the same scraper upstream
        uses and falls back to the raw text exactly as upstream does.
        """
        if self.answer_prompt_mode != "native":
            return answer
        try:
            parsed = self._llm.extract_json(answer)
        except Exception:
            return answer.strip()
        if isinstance(parsed, dict):
            return parsed.get("answer", answer.strip())
        return answer.strip()
