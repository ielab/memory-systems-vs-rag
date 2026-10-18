"""
evalgrid/systems/memorybank/longmemeval.py
------------------------------------------
MemoryBank wrapper for LongMemEval-S.

MemoryBank predates both LongMemEval and LoCoMo and evaluates only on its own
synthetic set (15 virtual users × 10 dates × 4-6 turn-pairs, 194 probing
questions, human/LLM-graded) — and its evaluation harness was never released:
``build_prompt_with_search_memory_*_eval`` are defined in
``utils/prompt_utils.py`` and called from nowhere. So there is no reference
number here to reproduce, and the deviations are the interesting part:

  - both speakers (``roles: all``, the default), so a unit is a genuine
    ``{query, response}`` exchange — which is what MemoryBank's unit is meant to
    be and what its own synthetic dataset always contains. ``roles: user_only``
    empties the response side, leaving a single user turn with a date prefix; it
    costs this system more than any other here, so run it only as a labelled
    ablation.
  - session buckets rather than date buckets (see store.py).
  - the shared answer prompt, not upstream's companion persona.

What this row is for: MemoryBank is the cheap floor. With the default config it
spends one LLM call per session and stores raw turn-pairs plus one summary each.
If a system with an extraction pipeline cannot beat it, that is the most
informative negative result the table can produce.
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Session

from evalgrid.systems.memorybank.system import MemoryBankBase


class MemoryBankLongMemEval(MemoryBankBase):

    def __init__(self, config: dict, generator):
        self.roles = str(config.get("roles", "all"))
        if self.roles not in ("user_only", "all"):
            raise ValueError(f"roles must be 'user_only' or 'all', got {self.roles!r}")
        super().__init__(config, generator)

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        self.add_session(session, roles=self.roles)
        if is_last:
            # The write path made no LLM call; this is where the summarisation
            # pass (and, if enabled, forgetting) happens — upstream's explicit
            # offline batch job.
            self.finalize()
