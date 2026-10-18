"""
evalgrid/benchmarks/retrievers/fusion.py
---------------------------------
Hybrid retriever: linear score fusion of BM25 (lexical) and a dense embedder.

For each sample it ranks the corpus twice — once with BM25, once with cosine
similarity over embeddings — min-max normalizes each retriever's raw scores to
[0, 1] independently (their raw scales are not comparable: BM25 is unbounded,
cosine is [-1, 1]), then combines them per doc:

    score = alpha * dense_norm + (1 - alpha) * bm25_norm      (alpha=0.5 -> equal)

Composes BM25Retriever + DenseRetriever rather than reimplementing either, so
the dense half reuses DenseRetriever's corpus embedding cache (cache_path)
unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from evalgrid.benchmarks.retrievers.base import Retriever
from evalgrid.benchmarks.retrievers.bm25 import BM25Retriever
from evalgrid.benchmarks.retrievers.dense import DenseRetriever
from evalgrid.models.base import EmbedderModel


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]. Constant scores (zero range) map to
    0.0 — they carry no ranking signal, so they add nothing to the fusion."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    rng = hi - lo
    if rng == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / rng for k, v in scores.items()}


class FusionRetriever(Retriever):
    def __init__(self, embedder: EmbedderModel, *, alpha: float = 0.5):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(embedder)
        self.alpha = alpha  # weight on the dense score; bm25 gets (1 - alpha)

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
        bm25_ranked, dense_ranked = await asyncio.gather(
            self.bm25.search(query, bm25_index),
            self.dense.search(query, dense_index),
        )
        bm25_norm = _minmax(dict(bm25_ranked))
        dense_norm = _minmax(dict(dense_ranked))

        fused = [
            (
                cid,
                self.alpha * dense_norm.get(cid, 0.0)
                + (1.0 - self.alpha) * bm25_norm.get(cid, 0.0),
            )
            for cid in corpus_ids
        ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused
