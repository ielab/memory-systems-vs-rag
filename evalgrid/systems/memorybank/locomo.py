"""
evalgrid/systems/memorybank/locomo.py
-------------------------------------
MemoryBank wrapper for LoCoMo.

Both speakers, so a ``{query, response}`` pair is a genuine exchange — which is
what MemoryBank's unit is meant to be, and closer to its own synthetic data than
the user-only LongMemEval configuration.

The two speakers map onto ``query``/``response`` by position, not by name:
upstream's unit renders them as ``[|User|]`` and ``[|AI|]`` regardless, since
its data always has a human and the companion. On LoCoMo both sides are human,
so those markers are cosmetic — but they are kept, because they are what the
retriever's indexed strings contain upstream and changing them would change what
is embedded.

Upstream has no LoCoMo path and no per-category prompts; answering uses the
shared prompt, as on LongMemEval.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Session

from evalgrid.systems.memorybank.system import MemoryBankBase


class MemoryBankLoCoMo(MemoryBankBase):

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        self.add_session(session, roles="all")
        if is_last:
            self.finalize()
