"""
evalgrid/systems/hypermem/longmemeval.py
----------------------------------------
HyperMem wrapper for LongMemEval-S.

HyperMem has no LongMemEval numbers — the paper (ACL 2026) reports LoCoMo only,
and the repo's only driver is ``locomo10.json``. So nothing here is "the
reference implementation", and the deviations are the interesting part of any
number this produces:

  - both speakers (``roles: all``, the default), as upstream's LoCoMo pipeline
    does — it feeds the whole message stream and detects episode boundaries over
    it. ``roles: user_only`` is available as a labelled ablation.
  - one episode per session, which is what LongMemEval's structure gives us and
    what makes the ~600-call-per-conversation boundary detector unnecessary
    (see system.py's first ADAPTER note).
  - the shared answer prompt BY DEFAULT, not upstream's 7-step CoT one. Holding
    the prompt fixed is what keeps the comparison about memory representation;
    upstream's prompt is written around LoCoMo's "answer in 5-6 words"
    convention and its two named speakers.

``answer_prompt: native`` opts into upstream's prompt instead. It is opt-in
rather than the default because every run already in ``outputs/`` was answered
under the shared prompt, and flipping the default would change what those
directories mean without changing their names. ``ANSWER_PROMPT_NEMORI_COT`` is
dataset-agnostic — it interpolates only ``{context}`` and ``{question}`` — so
porting it here is a real port and not an approximation. What it is NOT is a
published LongMemEval setting: HyperMem reports LoCoMo only, so this is
upstream's prompt on a benchmark upstream never ran.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Query, Session
from evalgrid.systems.base import shared_answer_prompt

from evalgrid.systems.hypermem.system import HyperMemBase


class HyperMemLongMemEval(HyperMemBase):

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
        self.add_session(session, roles="all" if self.roles == "all" else "user_only")
        if is_last:
            # Facts are extracted per topic over all of a topic's episodes, and
            # topics keep absorbing episodes, so the whole L3/L1 pass waits for
            # the last session. See HyperMemBase.build_hypergraph.
            self.build_hypergraph()

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        if self.answer_prompt_mode == "native":
            return self._native_answer_messages(context, query.question)
        # context is one block of HyperMem's own hierarchical formatting, not a
        # list of facts, so it goes in as a single "fact" rather than re-split.
        return shared_answer_prompt(
            query.question, query.question_date, [context] if context else []
        )

    def _native_answer_messages(self, context: str, question: str) -> list[dict]:
        """Upstream's 7-step CoT answer prompt, verbatim.

        Imported from the vendored template rather than copied, so it cannot
        drift — the same construction ``locomo.py`` uses. Stage 5 sends it as a
        single *system* message; kept.

        The question DATE is dropped, because upstream's template has no slot
        for it: it anchors time on the timestamps inside the retrieved
        memories (instructions 5-6) rather than on a question date. Adding a
        slot would make this our prompt rather than upstream's, so the date is
        left out and the difference recorded here.
        """
        from hypermem.prompts.answer_prompts import ANSWER_PROMPT_NEMORI_COT

        prompt = ANSWER_PROMPT_NEMORI_COT.format(context=context, question=question)
        return [{"role": "system", "content": prompt}]

    def postprocess_answer(self, answer: str) -> str:
        """Take the text after ``FINAL ANSWER:`` when the CoT prompt was sent.

        A no-op under the shared prompt, which asks for the answer directly.
        Under the native one the reply carries seven labelled sections before
        the answer, and handing all of that to the judge would score the
        reasoning rather than the answer. Falls back to the whole response when
        the marker is missing, exactly as ``locomo.py`` does and for the same
        reason (upstream instead retries the generation up to 10 times and then
        returns "").
        """
        if self.answer_prompt_mode != "native":
            return answer
        _, marker, tail = answer.partition("FINAL ANSWER:")
        return tail.strip() if marker else answer.strip()
