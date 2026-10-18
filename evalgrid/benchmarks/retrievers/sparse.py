"""
evalgrid/benchmarks/retrievers/sparse.py
----------------------------------
Sparse/lexical retriever: dot product between learned term-weight maps
(e.g. SPLADE's vocab weights) rather than embedding cosine
similarity (dense.py) or a hand-tuned BM25 formula (bm25.py). Works with any
SparseEncoderModel.

The corpus weights are built by ``index`` and cached (cache.py); the query is
encoded live by ``search`` — see base.py for why the split.
"""

from __future__ import annotations

from typing import Any, Optional

from evalgrid.benchmarks.retrievers.base import Retriever
from evalgrid.benchmarks.retrievers.cache import load_cached, save_cache
from evalgrid.models.base import SparseEncoderModel


def sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b[t] for t, w in a.items() if t in b)


class SparseRetriever(Retriever):
    def __init__(self, encoder: SparseEncoderModel):
        self.encoder = encoder

    async def index(
        self, corpus: list[dict], *, cache_path: Optional[str] = None,
    ) -> tuple[list[str], list[dict[int, float]]]:
        corpus_ids = [c["_id"] for c in corpus]
        if not corpus:
            return corpus_ids, []
        weights = load_cached(cache_path, corpus_ids) if cache_path else None
        if weights is None:
            weights = await self.encoder.encode_sparse(
                [c["text"] for c in corpus], is_query=False)
            if cache_path:
                save_cache(cache_path, corpus_ids, weights)
        return corpus_ids, weights

    async def search(self, query: str, index: Any) -> list[tuple[str, float]]:
        corpus_ids, corpus_weights = index
        if not corpus_ids:
            return []
        query_weights = (await self.encoder.encode_sparse([query], is_query=True))[0]
        scores = [sparse_dot(query_weights, w) for w in corpus_weights]
        ranked = list(zip(corpus_ids, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
