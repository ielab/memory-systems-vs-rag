"""
evalgrid/benchmarks/retrievers/rrf.py
---------------------------------
Hybrid retriever: BM25 (lexical) + a dense embedder, fused by **reciprocal rank
fusion**.

    score(d) = SUM over legs of  1 / (k + rank_leg(d) + 1)      k = 60, rank 0-based

This is fusion.py's pair of legs under a different combiner, and it exists as its
own class rather than a `method:` flag on FusionRetriever because the run
directory is named after the retriever's NAME (lib/paths.retriever_name). Two
combiners sharing one name would share one directory and the second run would
silently overwrite the first — the collision that naming exists to prevent.

WHY RRF RATHER THAN fusion.py's MIN-MAX. Min-max normalizes each leg over the
scores it produced for *this* sample's corpus, so the fused score depends on the
spread of that particular corpus: a sample where BM25 happens to separate its
top hit sharply contributes a different effective weight than one where it does
not. Sample corpora here differ in size by an order of magnitude across systems
(a_mem's tens of notes vs. lightmem's hundreds of memories), so that spread is
not a constant of the experiment. RRF reads only ranks, so nothing about a leg's
score distribution reaches the fused ordering, and there is no alpha to tune.
Both remain available; this is the one the grid runs.

THE ARITHMETIC IS UPSTREAM'S, deliberately. It is transcribed from
``vendor/hypermem/main/stage4_hypergraph_retrieval.py:47``
(``reciprocal_rank_fusion``), including RRF_K = 60 and the ``+ 1`` that makes the
first-ranked document score 1/61 rather than 1/60. HyperMem fuses its two legs
this way at every layer of its funnel, so a run that swaps HyperMem's RRF for
ours differs in the dense REPRESENTATION (flat sentence embeddings vs. vectors
carrying stage 3's hyperedge propagation) and in nothing else. That is only true
if the combiner matches to the constant.

ONE DELIBERATE DEVIATION: ``depth``. Upstream truncates each leg to ``top_n``
before fusing. The default here is to fuse the FULL ranked lists, because
``Retriever.rank``'s contract is a complete ranking — pipeline/retrieve.py writes
every returned row to run.tsv and eval scores it at K up to 20, and a truncated
fusion leaves everything below the cut tied at zero, which makes the tail's
order arbitrary rather than merely uninteresting. Set ``depth`` to reproduce
upstream's behaviour exactly for the one comparison that needs it.

The difference is NOT confined to the tail, which is worth stating because it is
the intuitive and wrong assumption. Under ``depth`` a document scores only from
the legs whose window it entered, so one ranked 1st by BM25 and 200th by dense
keeps 1/61 and loses its 1/261 — while a document ranked 6th and 2nd keeps both.
That reorders the HEAD. Measured on a 50-doc corpus with two random
permutations: full-list fusion leads with a doc ranked (6, 1), whereas the same
legs truncated at 5 lead with a doc ranked (0, deep) that the full fusion places
outside its top 6. Truncation is a different ranking, not a cheaper
approximation of the same one — which is the argument for the full-list default:
it uses every rank both legs produced and has no window artifact.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from evalgrid.benchmarks.retrievers.base import Retriever
from evalgrid.benchmarks.retrievers.bm25 import BM25Retriever
from evalgrid.benchmarks.retrievers.dense import DenseRetriever
from evalgrid.models.base import EmbedderModel

#: Upstream's RRF_K, and the value the original RRF paper recommends. Held fixed
#: across the grid — it is a constant of the
#: combiner, not a knob this experiment sweeps.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    corpus_ids: list[str],
    *,
    k: int = RRF_K,
    depth: Optional[int] = None,
) -> list[tuple[str, float]]:
    """Fuse ranked (id, score) lists by RRF. Returns every id in ``corpus_ids``.

    Args:
        rankings: one ranked list per leg, best first. Their scores are read
            for nothing — only position matters, which is the point.
        corpus_ids: the full corpus, in corpus order. Ids absent from every
            leg's fused window score 0.0 and land at the end; without this the
            result would not satisfy ``Retriever.rank``'s "every doc" contract
            whenever ``depth`` cuts a leg short.
        k: the RRF constant.
        depth: truncate each leg to this many before fusing (upstream's
            behaviour). None fuses the full lists — the default, see the module
            docstring.

    Ties are broken by corpus order, via a stable sort over a list built in
    corpus order. That matters more here than for a score-based retriever:
    under ``depth`` the entire tail ties at 0.0, and an unstable order there
    would make run.tsv differ between runs over identical inputs.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        window = ranking[:depth] if depth else ranking
        for rank, (doc_id, _) in enumerate(window):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    fused = [(cid, scores.get(cid, 0.0)) for cid in corpus_ids]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


class RRFRetriever(Retriever):
    """BM25 + dense, fused by reciprocal rank fusion.

    Composes BM25Retriever and DenseRetriever rather than reimplementing either,
    exactly as FusionRetriever does — so the dense half reuses DenseRetriever's
    corpus embedding cache (``cache_path``) unchanged and a `rrf` run costs the same
    encodes as the `dense` run on the same model.

    A sparse-legged variant (SPLADE + dense under RRF) would be a sibling class,
    the way hybrid.py is a sibling of fusion.py; it is not in the grid.
    """

    def __init__(self, embedder: EmbedderModel, *, rrf_k: int = RRF_K,
                 depth: Optional[int] = None):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(embedder)
        self.rrf_k = int(rrf_k)
        self.depth = int(depth) if depth else None

    async def index(self, corpus: list[dict], *, cache_path: Optional[str] = None) -> Any:
        bm25_index, dense_index = await asyncio.gather(
            self.bm25.index(corpus),
            self.dense.index(corpus, cache_path=cache_path),
        )
        return [c["_id"] for c in corpus], bm25_index, dense_index

    async def search(self, query: str, index: Any) -> list[tuple[str, float]]:
        corpus_ids, bm25_index, dense_index = index
        if not corpus_ids:
            return []
        # Gathered, unlike hybrid.py's sequential halves: these are two
        # different models with no shared encode memo to race past.
        bm25_ranked, dense_ranked = await asyncio.gather(
            self.bm25.search(query, bm25_index),
            self.dense.search(query, dense_index),
        )
        return reciprocal_rank_fusion(
            [bm25_ranked, dense_ranked],
            corpus_ids,
            k=self.rrf_k,
            depth=self.depth,
        )
