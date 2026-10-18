"""
evalgrid/systems/mem0/longmemeval.py
------------------------------------
Mem0 wrapper for LongMemEval-S.

Upstream *does* publish LongMemEval numbers, and none of them characterise the
library this package wraps:

  * 94.4 % / 94.8 % are the **hosted platform** (`mode="cloud"`);
  * the best OSS-model row, 91.0 %, still runs through the `server/` REST app
    and passes a per-session `timestamp=` that OSS `Memory.add()` does not
    accept in 0.1.x at all;
  * every row uses **top_k = 200**. A LongMemEval-S sample's whole store is a
    few hundred facts, so that is "compress the history, then feed nearly all
    of it back" — top-50 and top-200 differ by 0.4 points, which is what a
    retriever doing almost no work looks like.

So this row does not reproduce a published number, and should not be presented
as one. What it produces is Mem0 under this repo's protocol: same corpus
universe, same qrels, same retrievers, same prompt, same judge as every other
system — which is the only internally consistent comparison available.

Ingest follows upstream's harness on both axes that matter:

  - **both speakers** (`roles: all`). `run.py` cleans each session to
    `[{"role", "content"}]` and sends it whole; nothing is filtered by role.
  - **one `add()` per (user, assistant) turn-pair** (`ingest_policy: pair`),
    upstream's `CHUNK_SIZE = 2`. `add()`'s metadata is per-call, so this also
    sets provenance granularity: turn-pair level, comparable with lightmem's
    per-fact source ids and finer than simplemem's or hypermem's session-level.

`ingest_policy: session` (~25x cheaper) and `roles: user_only` remain available
as labelled ablations — but they are this repo's choices, not mem0's, and a run
using them must not be presented as Mem0's own setting.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Session

from evalgrid.systems.mem0.system import Mem0Base


class Mem0LongMemEval(Mem0Base):

    def __init__(self, config: dict, generator):
        self.roles = str(config.get("roles", "all"))
        if self.roles not in ("user_only", "all"):
            raise ValueError(f"roles must be 'user_only' or 'all', got {self.roles!r}")
        self.ingest_policy = str(config.get("ingest_policy", "pair"))
        if self.ingest_policy not in ("pair", "session", "turn"):
            raise ValueError(
                f"ingest_policy must be 'pair', 'session' or 'turn', "
                f"got {self.ingest_policy!r}")
        super().__init__(config, generator)

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        """Store one session. ``is_last`` is a no-op — nothing is buffered.

        Unlike lightmem (which defers consolidation to an offline sweep) or
        memorybank (which defers summarisation), Mem0 consolidates inside every
        `add()`: extraction, neighbour search and the ADD/UPDATE/DELETE decision
        all happen before the call returns. There is nothing left to flush.
        """
        turns = list(session.turns or [])
        if not turns:
            if session.text:
                self.add([{"role": "user", "content": session.text}],
                         session=session, turn_ids=[], has_answer=False)
            return

        selected = [(i, t) for i, t in enumerate(turns)
                    if self.roles == "all" or t.role == "user"]
        if not selected:
            return

        if self.ingest_policy == "session":
            groups = [selected]
        elif self.ingest_policy == "turn":
            groups = [[item] for item in selected]
        else:
            # pair — upstream's `CHUNK_SIZE = 2`, applied to the cleaned message
            # list. With roles: all that is a (user, assistant) exchange; with
            # user_only a "pair" would be two unrelated user turns, so the chunk
            # degrades to 1 rather than fusing them.
            size = 2 if self.roles == "all" else 1
            groups = [selected[i:i + size] for i in range(0, len(selected), size)]

        for group in groups:
            messages = [{"role": turn.role, "content": turn.content} for _, turn in group]
            turn_ids = [turn.turn_id or f"{session.session_id}:t{i}" for i, turn in group]
            has_answer = any(turn.has_answer for _, turn in group)
            self.add(messages, session=session, turn_ids=turn_ids, has_answer=has_answer)
