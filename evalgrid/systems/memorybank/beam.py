"""
evalgrid/systems/memorybank/beam.py
-----------------------------------
MemoryBankBEAM: this system's driver for the BEAM tiers.

MemoryBank stores one {query, response} record per pair and buckets by session.
Both are clean on BEAM: pairing is exact (even counts, user-led), and the
session boundary is the dataset's own ``time_anchor`` block.

BEAM reuses this system's LongMemEval ingest policy, and that is a MEASURED
choice rather than a fallthrough. On the 100K tier (measured by
evalgrid.tools.beam_stats), across all 90 sessions:

  * every session has an EVEN message count -> a stride-2 pairing loop drops
    nothing, where on LongMemEval-S it discards 2,020 turns;
  * every session starts with a ``user`` turn -> no leading-turn skip, so no
    sequence-number parity flip;
  * user/assistant alternation never breaks, and those are the only two roles;
  * ``question_date`` is None on all 400 questions, so the answer prompts take
    their undated branch, as they do on LoCoMo.

So BEAM is structurally cleaner for this driver than the benchmark it was
written for. The subclass exists to make that an explicit, named decision — a
run's trajectory says BEAM, an error names this class, and a divergence has
somewhere to go — rather than a dataset silently landing on the last ``return``
of ``plugin.build``.

WHAT IS NOT SOLVED BY BEING A SUBCLASS: session SIZE. A BEAM-100K session is a
median ~25 K tokens against LongMemEval's ~1 K, and 25 of the 90 exceed the
32,768-token window ``lightmem`` and ``simplemem`` are configured at. That is a
config question (evalgrid/configs/), not an ingest-policy one, and it is why
500K/1M are treated as a length-stress study rather than a comparison row.
"""

from __future__ import annotations

from evalgrid.systems.memorybank.longmemeval import MemoryBankLongMemEval


class MemoryBankBEAM(MemoryBankLongMemEval):
    """BEAM driver. Ingest policy is inherited unchanged — see the module docstring."""
