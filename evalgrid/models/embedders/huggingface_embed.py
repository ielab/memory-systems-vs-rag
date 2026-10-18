"""
models/embedders/huggingface_embed.py
---------------------------------------
Local text embedding via sentence-transformers.

Usage:
    emb = load_embedder("sentence-transformers/all-MiniLM-L6-v2")

The HuggingFace Hub model id is passed directly to SentenceTransformer.

Unlike the vllm path (models/embedders/vllm_embed.py), which has to guess a
model's query/document prefix convention from its name (see
models/embedders/instructions.py), sentence-transformers models that need
one (BGE, E5, GTE, Nomic, ...) typically ship it in their own HF config as a
named "prompt" — so we just ask the loaded model for it via prompt_name,
rather than hardcoding a second registry here. Models with no configured
prompts (e.g. MiniLM) are unaffected: prompt_name stays None and is_query is
effectively a no-op, same as before.
"""

from __future__ import annotations

import asyncio
from functools import partial

from sentence_transformers import SentenceTransformer

from evalgrid.models.base import EmbedderModel
from evalgrid.models.embedders import count_processed_tokens

# Candidate prompt names for the document side — model authors aren't
# consistent about this key, unlike "query" which is near-universal.
_DOC_PROMPT_NAMES = ("passage", "document", "corpus")


class HuggingFaceEmbedder(EmbedderModel):
    """
    Args:
        model: HuggingFace model id, e.g. "sentence-transformers/all-MiniLM-L6-v2"
        batch_size: texts per encode() call
        device: e.g. "cpu", "cuda", "mps" — None lets sentence-transformers choose
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.model_name = model
        self.batch_size = batch_size
        kwargs = {"device": device} if device is not None else {}
        self._model = SentenceTransformer(model, **kwargs)

    async def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prompts = self._model.prompts or {}
        if is_query:
            prompt_name = "query" if "query" in prompts else None
        else:
            prompt_name = next((n for n in _DOC_PROMPT_NAMES if n in prompts), None)

        loop = asyncio.get_event_loop()
        encode = partial(
            self._model.encode,
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name=prompt_name,
        )
        vectors = await loop.run_in_executor(None, encode)
        # Counted in the executor, not on the event loop: this backend
        # re-encodes a whole corpus per query, and a tokenizer pass on the loop
        # thread would stall every other sample in flight for its duration.
        self._record_encode(len(texts), await loop.run_in_executor(
            None, partial(count_processed_tokens, self._model, texts)))
        return [v.tolist() for v in vectors]
