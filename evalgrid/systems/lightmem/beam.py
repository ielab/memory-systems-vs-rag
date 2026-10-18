"""
evalgrid/systems/lightmem/beam.py
---------------------------------
LightMemBEAM: this system's driver for the BEAM tiers.

The parity note above is load-bearing for LightMem specifically. Its published
LongMemEval driver prices each fact to ``turn_id_list[source_id * 2]``, an
inverse that holds only while every user message sits at an even sequence
number — and on LongMemEval-S the dropped turns break it for 28.15% of user
messages. BEAM drops nothing and skips nothing, so that
mapping should hold here. "Should": verify with
``evalgrid.tools.lightmem_rebuild_provenance`` before trusting a BEAM
provenance number, exactly as on LongMemEval.

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

from evalgrid.systems.lightmem.longmemeval import LightMemLongMemEval


class LightMemBEAM(LightMemLongMemEval):
    """BEAM driver. Ingest policy is inherited unchanged — see the module docstring."""
