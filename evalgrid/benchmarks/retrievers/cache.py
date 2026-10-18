"""
evalgrid/benchmarks/retrievers/cache.py
---------------------------------
Generic content-keyed cache for a retriever's per-sample CORPUS representation
(sparse weight maps, multi-vector token matrices — anything picklable).
Mirrors the .npz cache in dense.py but isn't tied to numpy arrays, since
sparse/multi-vector representations are variable-shaped (dict, or a matrix
with a different token count per text).

Keyed on the exact corpus ids, like dense.py's cache: a sample whose corpus was
rebuilt (qrels_has_answer --overwrite) is detected as stale and re-encoded
rather than silently scored against old vectors.

Only the corpus side is cached. The query is encoded live on every search (see
base.py), so a sample with several queries shares one cache file instead of
each query overwriting the last one's. Files written before that change also
carry ``query``/``query_repr``; they are ignored, and the corpus half of such a
file is still a valid hit.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional


def load_cached(cache_path: str, corpus_ids: list[str]) -> Optional[Any]:
    """Return the cached corpus_repr if it matches ``corpus_ids`` exactly, else
    None (missing, unreadable, or stale)."""
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data["corpus_ids"] != corpus_ids:
            return None
        return data["corpus_repr"]
    except Exception:
        return None


def save_cache(cache_path: str, corpus_ids: list[str], corpus_repr: Any) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"corpus_ids": corpus_ids, "corpus_repr": corpus_repr}, f)
