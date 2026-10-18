"""
evalgrid/benchmarks/retrievers/multi_vector.py
----------------------------------------
Late-interaction retriever (ColBERT-style): MaxSim between every query token
vector and every doc token vector, summed over query tokens. Works with any
MultiVectorModel — e.g. a ColBERT
checkpoint (models/embedders/colbert_embed.py).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evalgrid.benchmarks.retrievers.base import Retriever
from evalgrid.benchmarks.retrievers.cache import load_cached, save_cache
from evalgrid.models.base import MultiVectorModel


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Sum over query tokens of the max cosine similarity to any doc token."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    q = _l2_normalize_rows(query_vecs)
    d = _l2_normalize_rows(doc_vecs)
    sims = q @ d.T  # (num_query_tokens, num_doc_tokens)
    return float(sims.max(axis=1).sum())


class MultiVectorRetriever(Retriever):
    def __init__(self, model: MultiVectorModel):
        self.model = model

    async def index(
        self, corpus: list[dict], *, cache_path: Optional[str] = None,
    ) -> tuple[list[str], list]:
        corpus_ids = [c["_id"] for c in corpus]
        if not corpus:
            return corpus_ids, []
        vecs = load_cached(cache_path, corpus_ids) if cache_path else None
        if vecs is None:
            vecs = await self.model.encode_multi([c["text"] for c in corpus], is_query=False)
            if cache_path:
                save_cache(cache_path, corpus_ids, vecs)
        return corpus_ids, vecs

    async def search(self, query: str, index: Any) -> list[tuple[str, float]]:
        corpus_ids, corpus_vecs = index
        if not corpus_ids:
            return []
        query_vecs = (await self.model.encode_multi([query], is_query=True))[0]
        scores = [maxsim(query_vecs, dv) for dv in corpus_vecs]
        ranked = list(zip(corpus_ids, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
