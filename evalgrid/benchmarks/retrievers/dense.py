"""
evalgrid/benchmarks/retrievers/dense.py
--------------------------------
Dense retriever: cosine similarity between an embedder's query vector and
every corpus doc's vector. Works with any EmbedderModel (models/embedders) —
including Qwen3-Embedding-* served via vllm, the same way naive_rag uses it.

Corpus embeddings are deterministic given (embedder, text), so ``index`` can
persist them to ``cache_path`` (a small .npz) and skip the corpus encode on a
later call — the corpus side is what a deployed system has already embedded by
query time. The cache is self-validating: it's keyed on the exact corpus ids it
was written for, so a rebuilt corpus (e.g. qrels_has_answer --overwrite) is
detected as stale and re-embedded rather than silently scored against old
vectors.

The QUERY is never cached: ``search`` embeds it on every call, because that
encode is part of per-question latency (see base.py). Caches written before
this split also hold a ``query``/``query_vector``; they are ignored, and the
corpus half is still a hit.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from evalgrid.benchmarks.retrievers.base import Retriever
from evalgrid.models.base import EmbedderModel


def cosine_rank(
    corpus_ids: list[str], corpus_vectors: np.ndarray, query_vector: np.ndarray,
) -> list[tuple[str, float]]:
    q_norm = np.linalg.norm(query_vector)
    doc_norms = np.linalg.norm(corpus_vectors, axis=1)
    denom = doc_norms * q_norm
    denom[denom == 0] = 1.0  # zero-vector guard: score those docs 0, not NaN
    scores = (corpus_vectors @ query_vector) / denom

    ranked = list(zip(corpus_ids, scores.tolist()))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _load_cached_vectors(cache_path: str, corpus_ids: list[str]) -> Optional[np.ndarray]:
    """Return the corpus vectors if the cache matches ``corpus_ids`` exactly,
    else None (missing, unreadable, or stale)."""
    if not os.path.isfile(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if list(data["corpus_ids"]) != corpus_ids:
            return None
        return data["corpus_vectors"]
    except Exception:
        return None


def _save_cache(cache_path: str, corpus_ids: list[str], corpus_vectors: np.ndarray) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, corpus_ids=np.array(corpus_ids), corpus_vectors=corpus_vectors)


class DenseRetriever(Retriever):
    def __init__(self, embedder: EmbedderModel):
        self.embedder = embedder

    async def index(
        self, corpus: list[dict], *, cache_path: Optional[str] = None,
    ) -> tuple[list[str], Optional[np.ndarray]]:
        corpus_ids = [c["_id"] for c in corpus]
        if not corpus:
            return corpus_ids, None
        vectors = _load_cached_vectors(cache_path, corpus_ids) if cache_path else None
        if vectors is None:
            # Embedded apart from the query (never as one batch) so asymmetric /
            # instruction-tuned embedders (e.g. Qwen3-Embedding) can prefix the
            # query without also prefixing every document.
            vectors = np.asarray(await self.embedder.embed(
                [c["text"] for c in corpus], is_query=False))
            if cache_path:
                _save_cache(cache_path, corpus_ids, vectors)
        return corpus_ids, vectors

    async def search(
        self, query: str, index: tuple[list[str], Optional[np.ndarray]],
    ) -> list[tuple[str, float]]:
        corpus_ids, corpus_vectors = index
        if corpus_vectors is None:
            return []
        query_vector = np.asarray((await self.embedder.embed([query], is_query=True))[0])
        return cosine_rank(corpus_ids, corpus_vectors, query_vector)
