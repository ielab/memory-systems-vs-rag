"""
models/embedders/openai_embed.py
----------------------------------
Text embedding via OpenAI Embeddings API.
Requires OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import os

import openai

from evalgrid.models.base import EmbedderModel


class OpenAIEmbedder(EmbedderModel):
    """
    Args:
        model: e.g. "text-embedding-3-small", "text-embedding-3-large"
        api_key: defaults to OPENAI_API_KEY env var
        batch_size: max texts per request (OpenAI limit: 2048)
        dimensions: optional reduced output dimension (text-embedding-3 only)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 512,
        dimensions: int | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.client = openai.AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    async def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        # OpenAI embedding models are symmetric — is_query is accepted but unused.
        results: list[list[float]] = []
        extra = {"dimensions": self.dimensions} if self.dimensions else {}
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = await self.client.embeddings.create(
                model=self.model, input=batch, **extra
            )
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            self._record_encode(len(batch), int(tokens) if tokens else None)
            results.extend(item.embedding for item in response.data)
        return results
