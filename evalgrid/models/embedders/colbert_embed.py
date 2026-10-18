"""
models/embedders/colbert_embed.py
-------------------------------------
ColBERT-style late-interaction encoder via pylate — a lightweight,
sentence-transformers-based ColBERT implementation. Deliberately not the
original Stanford colbert-ai package, which is built around large-scale
FAISS indexing we don't need: every sample's corpus here is small and
disjoint, scored fresh per query (same reasoning as evalgrid/benchmarks/retrievers/
dense.py and bm25.py).

Supports any pylate-compatible checkpoint, including HuggingFace ColBERT
checkpoints such as "colbert-ir/colbertv2.0" or pylate-trained ones like
"lightonai/Reason-ModernColBERT".

Requires: pip install pylate
Runs locally (no vllm server) — same as HuggingFaceEmbedder.
"""

from __future__ import annotations

import asyncio
from functools import partial

import numpy as np

from evalgrid.models.base import MultiVectorModel


class ColBERTModel(MultiVectorModel):
    """
    Args:
        model: HuggingFace/pylate model id, e.g. "colbert-ir/colbertv2.0"
        batch_size: texts per encode() call
        device: e.g. "cpu", "cuda" — None lets pylate choose
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 16,
        device: str | None = None,
    ):
        from pylate import models

        self.model_name = model
        self.batch_size = batch_size
        kwargs = {"device": device} if device is not None else {}
        self._model = models.ColBERT(model_name_or_path=model, **kwargs)

    async def encode_multi(self, texts: list[str], *, is_query: bool = False) -> list[np.ndarray]:
        loop = asyncio.get_event_loop()
        # pylate applies the right marker token / prefix internally based on is_query.
        encode = partial(
            self._model.encode,
            texts,
            batch_size=self.batch_size,
            is_query=is_query,
            show_progress_bar=False,
        )
        vectors = await loop.run_in_executor(None, encode)
        out = [np.asarray(v) for v in vectors]
        # Free and exact, uniquely among the backends: late interaction emits
        # one vector PER TOKEN, so the representation's own row count is the
        # token count and no tokenizer pass is needed to learn it.
        self._record_encode(len(texts), sum(int(v.shape[0]) for v in out))
        return out
