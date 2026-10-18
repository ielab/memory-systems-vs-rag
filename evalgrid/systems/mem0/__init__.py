"""
evalgrid/systems/mem0/
----------------------
This repo's adapters for Mem0 (Chhikara et al., arXiv 2504.19413), wrapping the
vendored **mem0ai 0.1.118** at vendor/mem0/.

**Which Mem0 this is.** "Mem0" names two different systems that share a package.
The paper's design — extract facts, then a second LLM call resolving each
against the store as ADD / UPDATE / DELETE / NONE — is the 0.1 line, which ends
at 0.1.118 (2025-09-25). Version 2.x (2026-04 onward) replaced it with a single
"additive extraction" call, md5-hash dedup instead of an LLM decision, hybrid
BM25 + entity-boost retrieval, and no graph memory at all.

0.1.118 is pinned here for two reasons. First, this repo's protocol flattens
each system's memory into a shared corpus and swaps in a shared retriever, which
is only fair to systems whose contribution is construction-side; 0.1.x is such a
system, while 2.x's contribution IS its retriever and a flat-corpus run would
delete it. Second, ADD/UPDATE/DELETE is the mechanism worth studying here — it
is directly contrastable with lightmem's offline merge/revise/delete under an
identical corpus and retriever, and 2.x has no consolidation to compare.

Within the 0.1 line, 0.1.118 rather than the release contemporaneous with the
paper (0.1.94, 2025-04-26): the two are algorithmically identical — the
extraction and update prompts are byte-identical, and `_add_to_vector_store`
differs only in logging, an `infer=False` branch we do not use, and a guard that
skips the second LLM call when extraction returned nothing — while 0.1.118 also
carries upstream's own fixes, including the `deepcopy(metadata)` in
`_update_memory` that the paper's version lacks. Pinning 0.1.94 would reproduce
the same algorithm plus bugs its authors later fixed.

**Say "v0.1.118", never "the paper's version".** No build is identified in the
paper, so which patch produced the arXiv numbers is not knowable.

Out of scope: **Mem0^g**, the graph variant. It costs ~3 further LLM calls per
`add()` and needs a live Neo4j (or Memgraph / Neptune / Kuzu), its memory unit
is a graph edge rather than a retrievable document, and it has no file-based
persistence to snapshot — for a reported +0 to +2 points. Also out of scope: the
hosted platform (`mem0.client`), which is where every published 94 % LongMemEval
number comes from.
"""
