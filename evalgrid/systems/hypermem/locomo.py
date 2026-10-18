"""
evalgrid/systems/hypermem/locomo.py
-----------------------------------
HyperMem wrapper for LoCoMo — the benchmark it was published on.

Aligned with upstream's pipeline (stages 1-5) except for episode segmentation:

  - both speakers, one message per turn, in session order
  - the session's ``date_time`` as the episode timestamp, with per-message
    timestamps synthesised as ``+30s`` per message, which is what upstream's
    ``load_locomo_raw_data`` does
  - upstream's 7-step CoT answer prompt (``ANSWER_PROMPT_NEMORI_COT``), with the
    answer parsed out of ``FINAL ANSWER:`` as stage 5 does
  - the two speaker names interpolated into the retrieval context template

**One episode per session, not per detected boundary.** Upstream flattens a
LoCoMo conversation into one message stream and pays one LLM call per message to
find episode boundaries; a session boundary is a coarser, free segmentation.
That changes what an episode *is* on this benchmark, so a LoCoMo number from
this adapter is not a reproduction of the paper's 92.73 — it is HyperMem with
session-level episodes. Restoring the published behaviour means porting
``episode_extraction_from_conversation`` from stage 1, which is the one part of
upstream's ingest this package deliberately does not carry.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Query, Session

from evalgrid.systems.hypermem.system import HyperMemBase


class HyperMemLoCoMo(HyperMemBase):

    def reset(self) -> None:
        super().reset()
        self._speakers: list[str] = []
        self.speaker_a, self.speaker_b = "user", "assistant"

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        for turn in session.turns:
            if turn.role and turn.role not in self._speakers:
                self._speakers.append(turn.role)
        self._apply_speakers(self._speakers)

        self.add_session(session, roles="all")
        if is_last:
            self.build_hypergraph()

    def load_memory(self, path: str) -> None:
        """Restore the store, and recover the speaker names from it.

        The answer phase never runs ``ingest``, so the names that ingest
        collected are gone — but every episode node carries ``participants``,
        so upstream's context header can be rebuilt rather than falling back to
        "user"/"assistant".
        """
        super().load_memory(path)
        names: list[str] = []
        for episode in self.hypergraph.episodes.values():
            for participant in episode.participants or []:
                if participant not in names:
                    names.append(participant)
        self._apply_speakers(names)

    def _apply_speakers(self, names: list[str]) -> None:
        if names:
            self.speaker_a = names[0]
            self.speaker_b = names[1] if len(names) > 1 else names[0]

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        """Upstream's CoT answer prompt, verbatim.

        Built by importing the vendored template rather than copying it, so it
        cannot drift. Stage 5 sends it as a single *system* message with
        temperature 0; kept.
        """
        from hypermem.prompts.answer_prompts import ANSWER_PROMPT_NEMORI_COT

        prompt = ANSWER_PROMPT_NEMORI_COT.format(context=context, question=query.question)
        return [{"role": "system", "content": prompt}]

    def postprocess_answer(self, answer: str) -> str:
        """Take the text after ``FINAL ANSWER:``, as stage 5 does.

        Upstream *retries the whole generation* up to ``llm_max_retries`` (10)
        times when the marker is missing and returns "" if it never appears — a
        weak generator can burn ten answer calls per question that way. Here a
        missing marker falls back to the full response: the judge sees the
        model's reasoning instead of an empty string, which is the more honest
        failure, and the CoT template's own text makes it obvious in the
        predictions file when this happens.
        """
        _, marker, tail = answer.partition("FINAL ANSWER:")
        return tail.strip() if marker else answer.strip()
