"""
models/base.py
--------------
Abstract interfaces for generator and embedder models.
All backends (vllm, OpenAI, Claude) implement these.
"""

from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TokenUsage:
    """Cumulative token counts and LLM-call count on a model instance."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int, calls: int = 1) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += calls

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0


@dataclass
class EncodeUsage:
    """What a retrieval encoder processed — the read path's counterpart to TokenUsage.

    Kept as its own type, and reported in its own fields, because an encode has
    no completion: ``output_tokens`` would be permanently 0 and ``calls`` would
    mean "batches sent to a model" where every cost artifact in this repo uses
    it for "LLM calls". Folding the two together would silently add unlike
    quantities AND change what ``tokens`` means in every cell already on disk.

    ``texts`` and ``batches`` are always exact. ``tokens`` is exact where the
    backend is told (served embedders report ``usage.prompt_tokens``) or where
    the representation IS per-token (ColBERT), and tokenizer-derived for the
    local sentence-transformers backends. ``uncounted_texts`` is how many texts
    passed with no count available — a number, not a flag, so a partly counted
    run says how partly instead of collapsing to "approximate".
    """

    texts: int = 0
    tokens: int = 0
    batches: int = 0
    uncounted_texts: int = 0

    def add(self, texts: int, tokens: int | None, batches: int = 1) -> None:
        """Record one batch. ``tokens=None`` means this backend could not say."""
        self.texts += int(texts)
        self.batches += int(batches)
        if tokens is None:
            self.uncounted_texts += int(texts)
        else:
            self.tokens += int(tokens)

    def merge(self, other: "EncodeUsage") -> None:
        """Fold another counter in field by field.

        Not ``add``: that one takes ``tokens=None`` to mean "uncounted" and
        would throw away a partly counted total (some tokens known, some texts
        uncounted), which is exactly the shape a run mixing backends produces.
        """
        self.texts += other.texts
        self.tokens += other.tokens
        self.batches += other.batches
        self.uncounted_texts += other.uncounted_texts

    def as_dict(self) -> dict:
        return {"texts": self.texts, "tokens": self.tokens,
                "batches": self.batches, "uncounted_texts": self.uncounted_texts}


#: The accumulator the current unit of work charges encodes to, or None.
#:
#: An instance counter cannot attribute work per sample on the flat retrieval
#: path: that path runs ``--concurrency`` samples at once against ONE shared
#: embedder, so a before/after delta around one sample would collect whatever
#: the other seven encoded in between. A context variable has no such problem —
#: asyncio copies the context into every task it creates, so a scope opened in
#: one sample's task is that sample's alone, and the nested ``gather`` inside
#: rrf/fusion still reaches it because those tasks inherit the copy and only
#: ever mutate the object it points at.
_encode_scope: contextvars.ContextVar = contextvars.ContextVar(
    "evalgrid_encode_scope", default=None)


class encode_scope:
    """Charge every encode inside this block to a fresh ``EncodeUsage``::

        with encode_scope() as used:
            ...                  # anything that encodes
        used.tokens              # what THIS block spent
    """

    def __init__(self) -> None:
        self.usage = EncodeUsage()
        self._token = None

    def __enter__(self) -> EncodeUsage:
        self._token = _encode_scope.set(self.usage)
        return self.usage

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _encode_scope.reset(self._token)
            self._token = None


class _EncodeAccounted:
    """An ``encode_usage`` counter for an encoder backend, and one way to fill it.

    A lazy property rather than an ``__init__``, because no backend in
    models/embedders/ calls ``super().__init__()``; requiring that call would
    change every constructor, and one that forgot would raise at the first
    encode instead of simply counting nothing.
    """

    @property
    def encode_usage(self) -> EncodeUsage:
        usage = getattr(self, "_encode_usage", None)
        if usage is None:
            usage = EncodeUsage()
            self._encode_usage = usage
        return usage

    def _record_encode(self, texts: int, tokens: int | None, batches: int = 1) -> None:
        """Charge one batch to this instance and to the open scope, if any.

        Never raises: accounting that can take a retrieval run down is worse
        than accounting that is missing, and this sits in every encode path.
        """
        try:
            self.encode_usage.add(texts, tokens, batches)
            scope = _encode_scope.get()
            if scope is not None:
                scope.add(texts, tokens, batches)
        except Exception:
            pass


class OutputCapError(RuntimeError):
    """Raised when a generation returned no usable text because the output token
    budget was exhausted.

    Some backends surface this as an API error (e.g. vLLM 400s a reasoning model
    that spends its whole max_tokens inside <think>); OpenAI instead returns an
    empty message with finish_reason='length' and NO error. Both must be treated
    the same way — retried with a larger budget or surfaced as a failure — rather
    than silently read as an empty result. Backends normalise the silent case
    into this exception so callers (judge/labeler retry paths) can recognise it.

    The tokens the failed call spent ride along on the exception. They were
    paid for, and without them a failed question simply drops out of the
    per-question cost: Qwen3.8's 65 cap-stopped oracle answers were 130k
    output tokens with no stats row.
    """

    def __init__(self, message: str, *, input_tokens: int = 0, output_tokens: int = 0,
                 reasoning_tokens: int | None = None):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens


@dataclass
class GenerationResult:
    """One generation response plus per-call accounting."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    #: The part of output_tokens spent on a reasoning trace that ``text`` no
    #: longer contains. Already INCLUDED in output_tokens, never added to it.
    #: None = not measured (a backend that does not look, or a trace the
    #: tokenizer could not count); 0 = measured, and there was no trace.
    reasoning_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GeneratorModel(ABC):
    """Text generation interface (chat-completion style)."""

    def __init__(self) -> None:
        self.token_usage = TokenUsage()
        #: Completions the server stopped at max_tokens (finish_reason
        #: "length"). Distinct from OutputCapError, which is only for the case
        #: where NO usable text came back: a truncated completion still returns
        #: text, so nothing raises and nothing looks wrong — the summary is just
        #: cut off mid-word and stored that way. Counted rather than raised
        #: (one clipped summary is a degraded sample, not a failed one) and
        #: surfaced in the ingesting system's meta.json, so "my memory is full
        #: of half-sentences" is a number in an artifact instead of something
        #: you find by reading trajectories.
        self.truncated_calls = 0

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],   # [{"role": "user", "content": "..."}]
        *,
        max_tokens: int = 2000,
        temperature: float | None = None,   # None = backend's pinned instance default
        **kwargs,
    ) -> str:
        """Return the assistant reply as a plain string."""

    async def generate_with_usage(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2000,
        temperature: float | None = None,   # None = backend's pinned instance default
        **kwargs,
    ) -> GenerationResult:
        """Return the assistant reply with per-call token usage when available."""
        before_in = self.token_usage.input_tokens
        before_out = self.token_usage.output_tokens
        text = await self.generate(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return GenerationResult(
            text=text,
            input_tokens=max(0, self.token_usage.input_tokens - before_in),
            output_tokens=max(0, self.token_usage.output_tokens - before_out),
        )

    async def generate_sync(self, messages: list[dict], **kwargs) -> str:
        """Convenience wrapper for non-async callers."""
        import asyncio
        return asyncio.run(self.generate(messages, **kwargs))


class EmbedderModel(_EncodeAccounted, ABC):
    """Text embedding interface."""

    @abstractmethod
    async def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Return one embedding vector per input text.

        is_query marks `texts` as search queries rather than corpus documents.
        Symmetric embedders ignore it; asymmetric/instruction-tuned ones (e.g.
        Qwen3-Embedding) use it to prefix queries with their retrieval
        instruction while leaving documents unprefixed, as their training
        expects.
        """

    async def embed_one(self, text: str, *, is_query: bool = False) -> list[float]:
        return (await self.embed([text], is_query=is_query))[0]


class SparseEncoderModel(_EncodeAccounted, ABC):
    """Lexical/sparse text encoding interface (e.g. SPLADE's vocab weights —
    a learned, per-token-id weight map, scored like a learned BM25 rather
    than embedding cosine similarity)."""

    @abstractmethod
    async def encode_sparse(self, texts: list[str], *, is_query: bool = False) -> list[dict[int, float]]:
        """Return one {token_id: weight} map per input text."""


class MultiVectorModel(_EncodeAccounted, ABC):
    """Late-interaction text encoding interface (ColBERT-style): one
    embedding per token rather than one per text, scored with MaxSim."""

    @abstractmethod
    async def encode_multi(self, texts: list[str], *, is_query: bool = False) -> list[np.ndarray]:
        """Return one (num_tokens, dim) array per input text."""
