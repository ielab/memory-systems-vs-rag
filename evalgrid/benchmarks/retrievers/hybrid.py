"""
evalgrid/benchmarks/retrievers/hybrid.py
---------------------------------
Hybrid retriever: linear score fusion of a dense embedder and a learned-sparse
encoder — fusion.py with the BM25 half swapped for SparseRetriever. This is
the PromptReps paper's hybrid setting; it works with any
(EmbedderModel, SparseEncoderModel) pair, and in particular with ONE model
instance serving as both (PromptReps), which retrieve.py passes when
the --emb-model provides both representations.

    score = alpha * dense_norm + (1 - alpha) * sparse_norm   (alpha=0.5 -> equal)

Caching: index() receives the dense half's cache_path (embeddings.npz, per
retrieve.py's _CACHE_FILENAME) and derives the sparse half's repr_cache.pkl
next to it, so both corpus representations are cached per sample exactly like
the standalone dense/sparse runs. The query is encoded live by search(), as
everywhere (base.py).

The two halves run sequentially rather than asyncio.gather'd, on purpose: a
backend that computes both representations in one forward pass (PromptReps)
memoizes its recent encodes, so the second half's encode is a memo hit —
gathering the halves would race past that memo and double the LLM cost.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from evalgrid.benchmarks.retrievers.base import Retriever
from evalgrid.benchmarks.retrievers.dense import DenseRetriever
from evalgrid.benchmarks.retrievers.fusion import _minmax
from evalgrid.benchmarks.retrievers.sparse import SparseRetriever
from evalgrid.models.base import EmbedderModel, SparseEncoderModel

_SPARSE_CACHE_FILENAME = "repr_cache.pkl"


class HybridRetriever(Retriever):
    def __init__(self, embedder: EmbedderModel, encoder: SparseEncoderModel, *,
                 alpha: float = 0.5):
        self.dense = DenseRetriever(embedder)
        self.sparse = SparseRetriever(encoder)
        self.alpha = alpha  # weight on the dense score; sparse gets (1 - alpha)

    async def index(self, corpus: list[dict], *, cache_path: Optional[str] = None) -> Any:
        sparse_cache = (
            os.path.join(os.path.dirname(cache_path), _SPARSE_CACHE_FILENAME)
            if cache_path else None
        )
        # Sequential, like search_all: the second half's encode is a memo hit
        # for a one-pass backend (PromptReps).
        dense_index = await self.dense.index(corpus, cache_path=cache_path)
        sparse_index = await self.sparse.index(corpus, cache_path=sparse_cache)
        return [c["_id"] for c in corpus], dense_index, sparse_index

    async def search_all(self, query: str, index: Any) -> dict[str, list[tuple[str, float]]]:
        """Search once, return the "dense", "sparse", and fused "hybrid" rankings.

        The dense/sparse entries are exactly what the standalone retrievers
        would produce for the same model, so callers (evalgrid/pipeline/
        retrieve.py) can persist all three runs from one encode pass.
        """
        corpus_ids, dense_index, sparse_index = index
        if not corpus_ids:
            return {"dense": [], "sparse": [], "hybrid": []}
        dense_ranked = await self.dense.search(query, dense_index)
        sparse_ranked = await self.sparse.search(query, sparse_index)
        dense_norm = _minmax(dict(dense_ranked))
        sparse_norm = _minmax(dict(sparse_ranked))

        fused = [
            (
                cid,
                self.alpha * dense_norm.get(cid, 0.0)
                + (1.0 - self.alpha) * sparse_norm.get(cid, 0.0),
            )
            for cid in corpus_ids
        ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return {"dense": dense_ranked, "sparse": sparse_ranked, "hybrid": fused}

    async def search(self, query: str, index: Any) -> list[tuple[str, float]]:
        return (await self.search_all(query, index))["hybrid"]
