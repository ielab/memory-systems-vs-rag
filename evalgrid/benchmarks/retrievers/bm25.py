"""
evalgrid/benchmarks/retrievers/bm25.py
-------------------------------
Lexical retriever: BM25 over each sample's own corpus (rank_bm25 — pure
Python). No Lucene index needed: corpora here are small (hundreds of facts)
and disjoint per sample, not one large shared index, so an in-memory BM25
built once per sample (``index``) is plenty fast and simpler than a
Pyserini/Lucene index. Nothing is cached to disk: there is no model-side encode
to save, and the build is the corpus tokenization only.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from evalgrid.benchmarks.retrievers.base import Retriever

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    async def index(
        self, corpus: list[dict], *, cache_path: Optional[str] = None,
    ) -> tuple[list[str], Optional[BM25Okapi]]:
        corpus_ids = [c["_id"] for c in corpus]
        if not corpus:
            return corpus_ids, None
        return corpus_ids, BM25Okapi([_tokenize(c["text"]) for c in corpus])

    async def search(self, query: str, index: Any) -> list[tuple[str, float]]:
        corpus_ids, bm25 = index
        if bm25 is None:
            return []
        scores = bm25.get_scores(_tokenize(query))
        ranked = list(zip(corpus_ids, scores.tolist()))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
