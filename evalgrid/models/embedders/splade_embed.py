"""
models/embedders/splade_embed.py
------------------------------------
SPLADE-family sparse encoders (e.g. naver/splade-v3) via the SparseEncoder
class that ships with sentence-transformers v5+. SPLADE expands each text into
a {vocab token_id: weight} map through its MLM head, so it plugs straight into
SparseRetriever's dot-product scoring.

encode_query/encode_document (rather than one encode call) let the library
apply any query/document-side settings the checkpoint ships with; for
splade-v3 the two sides share one encoder.

Note: naver/splade-v3 is a gated HF repo (CC BY-NC-SA) — accept its terms on
the Hub and authenticate (huggingface-cli login) before first download.

Requires: sentence-transformers>=5 (already a project dependency)
Runs locally (no vllm server) — same as HuggingFaceEmbedder.
"""

from __future__ import annotations

import asyncio
from functools import partial

from evalgrid.models.base import SparseEncoderModel
from evalgrid.models.embedders import count_processed_tokens


class SpladeModel(SparseEncoderModel):
    """
    Args:
        model: HuggingFace model id, e.g. "naver/splade-v3"
        batch_size: texts per encode() call
        device: e.g. "cpu", "cuda", "mps" — None lets sentence-transformers choose
    """

    def __init__(
        self,
        model: str = "naver/splade-v3",
        batch_size: int = 32,
        device: str | None = None,
    ):
        from sentence_transformers import SparseEncoder

        self.model_name = model
        self.batch_size = batch_size
        kwargs = {"device": device} if device is not None else {}
        self._model = SparseEncoder(model, **kwargs)

    async def encode_sparse(self, texts: list[str], *, is_query: bool = False) -> list[dict[int, float]]:
        encode = self._model.encode_query if is_query else self._model.encode_document
        loop = asyncio.get_event_loop()
        fn = partial(encode, texts, batch_size=self.batch_size, show_progress_bar=False)
        rows = await loop.run_in_executor(None, fn)
        # The INPUT length, not the size of the expanded weight maps: SPLADE's
        # output is a vocabulary-sized projection, so counting non-zero weights
        # would report term expansion rather than work done and would not be
        # comparable with a dense backend's number in the same column.
        self._record_encode(len(texts), await loop.run_in_executor(
            None, partial(count_processed_tokens, self._model, texts)))
        out: list[dict[int, float]] = []
        for row in rows:
            if not row.is_sparse:
                row = row.to_sparse()
            row = row.coalesce()
            token_ids = row.indices()[0].tolist()
            weights = row.values().tolist()
            out.append({int(t): float(w) for t, w in zip(token_ids, weights)})
        return out
