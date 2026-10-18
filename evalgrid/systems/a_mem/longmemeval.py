"""
evalgrid/systems/a_mem/longmemeval.py
-------------------------------------
A-Mem wrapper for LongMemEval-S.

A-Mem was never published on this benchmark, so nothing here is "the reference
implementation" — the choices below port its LoCoMo ingest policy across
unchanged, which is the most defensible reading of "A-Mem's own setting":

  - **both speakers** (``roles: all``), as on LoCoMo, where upstream's
    ``evaluate_dataset`` loops every turn of every session regardless of
    speaker. It is also what A-Mem's own results need: LightMem's paper
    (arXiv 2510.18866, Table 7) scores A-MEM at **96.43 / 96.43** on the 56
    single-session-assistant questions, which is unreachable if assistant turns
    never enter the store.
  - one note per turn, as on LoCoMo — the source of A-Mem's cost profile.
  - note content templated as upstream builds it, ``"Speaker {role}says : ..."``
    (the missing space is upstream's; see locomo.py).
  - the shared answer prompt BY DEFAULT, not a ported LoCoMo one. Holding the
    prompt fixed is what keeps the comparison about memory representation.

``answer_prompt: native`` ports A-Mem's own prompt across instead — opt-in
rather than default because every run already in ``outputs/`` was answered
under the shared prompt, and flipping the default would change what those
directories mean without changing their names.

READ THIS BEFORE QUOTING A NUMBER FROM IT. Unlike hypermem's and simplemem's,
this is NOT a verbatim upstream prompt for this benchmark, because no such
thing exists. ``test_advanced.answer_question`` is a five-way switch on the
LoCoMo category and opens with ``assert category in [1,2,3,4,5]``; LongMemEval
questions have no such category, so there is no branch to take. What is sent is
upstream's DEFAULT branch — the one categories 1 and 4 fall through to,
byte-identical to ``locomo.py``'s ``else``. That is a port of A-Mem's general
answer instruction, not a reproduction of a published setting, and it should be
labelled that way.

One thing deliberately not ported: upstream pins ``temperature=0.7`` for this
call and forces a ``json_schema`` response of ``{"answer": string}``. Neither
is reproduced here, exactly as ``locomo.py`` does not reproduce them — the
generator config is the harness's, and the schema would need a response_format
this path does not send.

``roles: user_only`` remains available as a labelled ablation.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Query, Session
from evalgrid.systems.base import shared_answer_prompt

from evalgrid.systems.a_mem.system import AMemSystem


class AMemLongMemEval(AMemSystem):

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
        if session.turns:
            for i, turn in enumerate(session.turns):
                if self.roles == "user_only" and turn.role != "user":
                    continue
                self.add_note(
                    # Upstream's template, verbatim down to the missing space
                    # before "says" (test_advanced.py:304). Under roles: all it
                    # is the only thing telling the extractor who spoke.
                    f"Speaker {turn.role}says : {turn.content}",
                    timestamp=session.timestamp,
                    # LongMemEval ships no turn ids, so synthesise the same ones
                    # evalgrid.pipeline.qrels_turns does — indexed into the
                    # ORIGINAL turn list, so an id means the same turn whether it
                    # reaches the corpus through this system or the raw baseline.
                    turn_id=turn.turn_id or f"{session.session_id}:t{i}",
                    session_id=session.session_id,
                    has_answer=turn.has_answer,
                )
        elif session.text:
            # Document-shaped session (no turns). One note for the whole text:
            # A-Mem's unit is the note, and it has no chunker of its own.
            self.add_note(
                session.text,
                timestamp=session.timestamp,
                session_id=session.session_id,
            )

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        if self.answer_prompt_mode == "native":
            return self._native_answer_messages(context, query.question)
        # context is one block of A-Mem's own formatting, not a list of facts,
        # so it goes in as a single "fact" rather than being re-split.
        return shared_answer_prompt(
            query.question, query.question_date, [context] if context else []
        )

    def _native_answer_messages(self, context: str, question: str) -> list[dict]:
        """A-Mem's DEFAULT LoCoMo branch, ported. See the module docstring.

        Byte-identical to ``locomo.py``'s ``else`` branch, indentation included
        — upstream's category-3 branch holds the same sentences but a different
        string, and the default is the one a category-less question falls to.
        Sent as a single user message with no system message, as upstream does.

        The question DATE is dropped: upstream's template has no slot for one,
        and adding it would make this our prompt rather than A-Mem's.
        """
        prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        return [{"role": "user", "content": prompt}]
