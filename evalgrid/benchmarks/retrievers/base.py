"""
evalgrid/benchmarks/retrievers/base.py
-------------------------------
Common interface for ranking one sample's BEIR corpus against its query.

Implementations: dense.py (embedding cosine similarity), bm25.py (lexical).
Each sample's corpus is small and disjoint from every other sample's (see
evalgrid/benchmarks/qrels.py), so retrievers build a per-sample index rather
than any persistent cross-sample one.

**Two halves, because they cost different things and only one is latency.**

    index(corpus)          the corpus side: encode every doc (or build BM25's
                           term statistics). Done ONCE per sample, and for the
                           model-backed retrievers cached to disk, so a re-run
                           loads it instead of encoding. This is what a deployed
                           memory system has already paid by query time.
    search(query, index)   the query side: encode the query LIVE and score it
                           against the index. Never cached — this is the
                           per-question retrieval latency, and a cached query
                           vector would report a latency that includes no
                           encoder at all.

``rank`` is the two composed, for callers that do not time anything. A caller
that reports latency (pipeline/retrieve.py, native_retrieval's rankers) must
call ``index`` outside its clock and ``search`` inside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Retriever(ABC):
    """Ranks one query against one sample's corpus."""

    @abstractmethod
    async def index(self, corpus: list[dict], *, cache_path: Optional[str] = None) -> Any:
        """Build this sample's corpus-side representation. Opaque to callers:
        hand it back to ``search`` unchanged. ``corpus`` is parsed corpus.jsonl
        rows: ``[{"_id": ..., "title": ..., "text": ..., "metadata": {...}}, ...]``.

        ``cache_path`` (model-backed retrievers only) persists the CORPUS
        representation, keyed on the exact corpus ids, so a rebuilt corpus is
        detected as stale and re-encoded.
        """

    @abstractmethod
    async def search(self, query: str, index: Any) -> list[tuple[str, float]]:
        """Return (corpus_id, score) for every doc in the index, sorted by
        score descending. Encodes ``query`` live every call."""

    async def rank(
        self, query: str, corpus: list[dict], *, cache_path: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """``index`` then ``search``: one-shot ranking, for callers that time nothing."""
        return await self.search(query, await self.index(corpus, cache_path=cache_path))
