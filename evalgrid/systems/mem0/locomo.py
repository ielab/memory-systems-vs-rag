"""
evalgrid/systems/mem0/locomo.py
-------------------------------
Mem0 wrapper for LoCoMo — the benchmark the paper reports.

**Do not read a number from this file as a reproduction of the paper's LoCoMo
table.** That table is disputed from both directions: upstream's own harness scores 92.5 % on the platform against the paper's
~61.7 % on the same benchmark; two open issues report third parties unable to
reproduce it; the judge prompt instructs the grader to "be generous"; and
full-context beats Mem0 inside Mem0's own table. What this adapter gives is Mem0
under this repo's protocol, judged the way every other system here is judged.

Role mapping is the one thing this file exists for. LoCoMo turns carry *speaker
names* as roles, and mem0's ``parse_messages`` (vendor/mem0/memory/utils.py:11)
only emits lines for the literal roles "system", "user" and "assistant" — every
other role is silently skipped, and the extraction prompt would receive an empty
transcript. Speakers are therefore mapped in first-seen order to user/assistant,
with the real name kept as a prefix inside the content so the extractor can
still attribute a fact to the right person.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Session

from evalgrid.systems.mem0.system import Mem0Base


class Mem0LoCoMo(Mem0Base):

    def reset(self) -> None:
        super().reset()
        self._speakers: list[str] = []

    def _role_for(self, speaker: str) -> str:
        """First speaker seen becomes "user", the second "assistant".

        Order is per-sample and stable, because ``reset()`` clears it and
        sessions arrive chronologically. A third speaker (LoCoMo has none) would
        fall back to "user" rather than be dropped by parse_messages.
        """
        if speaker not in self._speakers:
            self._speakers.append(speaker)
        return "assistant" if self._speakers.index(speaker) == 1 else "user"

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        turns = list(session.turns or [])
        if not turns:
            if session.text:
                self.add([{"role": "user", "content": session.text}],
                         session=session, turn_ids=[], has_answer=False)
            return

        messages, turn_ids = [], []
        for i, turn in enumerate(turns):
            speaker = turn.role or "user"
            messages.append({
                "role": self._role_for(speaker),
                "content": f"{speaker}: {turn.content}",
            })
            turn_ids.append(turn.turn_id or f"{session.session_id}:t{i}")

        # Both speakers, one add() per session — LoCoMo's "sessions" are already
        # the unit its own harness ingests.
        self.add(messages, session=session, turn_ids=turn_ids,
                 has_answer=any(t.has_answer for t in turns))
