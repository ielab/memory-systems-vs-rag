"""
models/embedders/instructions.py
------------------------------------
Query/document prefix conventions for text embedding models served over
vllm's plain-text /v1/embeddings endpoint (models/embedders/vllm_embed.py).
vllm hands back raw vectors with no notion of "prompt" or "instruction", so
unlike the sentence-transformers path (huggingface_embed.py, which can ask
the model's own config for the right prompt via prompt_name), we have to
know each model family's convention ourselves. They are NOT interchangeable:

  Qwen3-Embedding / gte-Qwen2-*-instruct
      query only — "Instruct: {task}\\nQuery:{text}", documents get no prefix.
  E5 (intfloat/e5-*)
      both sides, different prefixes — "query: {text}" / "passage: {text}".

Add a new family here (rather than special-casing it inside VLLMEmbedder)
when adding support for another asymmetric model — that's the only place
this knowledge should live.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixRule:
    query_prefix: str  # prepended to every query text, "" = no prefix
    doc_prefix: str = ""  # prepended to every document text, "" = no prefix (most families)


_QWEN3_INSTRUCT_PREFIX = (
    "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery:"
)

# First matching rule wins; checked as a case-insensitive substring of the model name.
_RULES: list[tuple[str, PrefixRule]] = [
    ("qwen3-embedding", PrefixRule(query_prefix=_QWEN3_INSTRUCT_PREFIX)),
    ("gte-qwen", PrefixRule(query_prefix=_QWEN3_INSTRUCT_PREFIX)),
    ("e5-", PrefixRule(query_prefix="query: ", doc_prefix="passage: ")),
]


def default_rule(model: str) -> PrefixRule | None:
    """Return the known PrefixRule for `model`, or None if it's symmetric
    (no prefix needed) or not a recognized family."""
    name = model.lower()
    for needle, rule in _RULES:
        if needle in name:
            return rule
    return None
