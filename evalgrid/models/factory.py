"""
models/factory.py
------------------
Config-driven loader for generator and embedder backends.

Usage:
    gen = load_generator("gpt-4o")
    gen = load_generator("Qwen/Qwen3-8B")           # routed to vllm
    gen = load_generator("Qwen/Qwen3-8B", base_url="http://gpu1:8000/v1")

    emb = load_embedder("text-embedding-3-small")
    emb = load_embedder("Qwen/Qwen3-Embedding-0.6B") # routed to vllm

    sparse = load_sparse_encoder("naver/splade-v3")
    sparse = load_sparse_encoder("promptreps:meta-llama/Meta-Llama-3-8B-Instruct")
    emb = load_embedder("promptreps:meta-llama/Meta-Llama-3-8B-Instruct")  # dense half
    mv = load_multi_vector_model("colbert-ir/colbertv2.0")
"""

from __future__ import annotations

import inspect

from evalgrid.models.base import EmbedderModel, GeneratorModel, MultiVectorModel, SparseEncoderModel

# Model name prefixes that are routed to vllm
_VLLM_PREFIXES = ("Qwen/", "meta-llama/", "mistralai/", "google/", "microsoft/")

# sentence-transformers models: local by default, vllm when base_url is given
_HF_EMBED_PREFIXES = ("sentence-transformers/",)

# OpenAI model name prefixes
_OPENAI_GEN_PREFIXES = ("gpt-", "o1", "o3")
_OPENAI_EMBED_PREFIXES = ("text-embedding-",)

# PromptReps: "promptreps:<hf-chat-model>" — one prompted LLM forward pass
# yields both a dense and a sparse representation (models/embedders/promptreps_embed.py)
_PROMPTREPS_PREFIX = "promptreps:"


def _is_vllm(name: str) -> bool:
    return any(name.startswith(p) for p in _VLLM_PREFIXES)


def _is_colbert(name: str) -> bool:
    return "colbert" in name.lower()


def _is_splade(name: str) -> bool:
    return "splade" in name.lower()


def _is_promptreps(name: str) -> bool:
    return name.startswith(_PROMPTREPS_PREFIX)


def _load_promptreps(model: str, **kwargs):
    from evalgrid.models.embedders.promptreps_embed import PromptRepsModel
    return PromptRepsModel(
        model=model[len(_PROMPTREPS_PREFIX):],
        **_accepted_kwargs(PromptRepsModel, kwargs),
    )


def _accepted_kwargs(cls, kwargs: dict) -> dict:
    """Subset of kwargs that ``cls.__init__`` actually accepts.

    Generator backends take different constructor knobs (reasoning_effort/
    verbosity are OpenAI-only; seed is OpenAI+vllm; base_url is vllm-only).
    Filtering here lets a
    caller pass a uniform model config — the knobs that don't apply to the chosen
    backend are dropped instead of raising — so the scripts stay backend-agnostic
    and don't need their own per-backend guards. A backend declaring **kwargs
    opts out of filtering (everything is forwarded).
    """
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = {name for name in sig.parameters if name != "self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def load_generator(model: str, **kwargs) -> GeneratorModel:
    """
    Route `model` to the right backend and return a GeneratorModel.
    Extra kwargs are forwarded to the backend constructor (e.g. base_url, api_key,
    reasoning_effort); knobs the chosen backend doesn't accept are dropped (see
    _accepted_kwargs), so a uniform model config works across backends.
    """
    if _is_vllm(model) or "base_url" in kwargs:
        from evalgrid.models.generators.vllm_gen import VLLMGenerator
        return VLLMGenerator(model=model, **_accepted_kwargs(VLLMGenerator, kwargs))

    if any(model.startswith(p) for p in _OPENAI_GEN_PREFIXES):
        from evalgrid.models.generators.openai_gen import OpenAIGenerator
        return OpenAIGenerator(model=model, **_accepted_kwargs(OpenAIGenerator, kwargs))

    raise ValueError(
        f"Cannot infer backend for generator model '{model}'. "
        "Pass base_url=... to force vllm, or add a prefix match in factory.py."
    )


def load_embedder(model: str, **kwargs) -> EmbedderModel:
    """
    Route `model` to the right backend and return an EmbedderModel.
    Extra kwargs are forwarded to the backend constructor.

    sentence-transformers/* runs locally by default; pass base_url=... to serve via vllm instead.
    "promptreps:<hf-chat-model>" always runs locally (dense half of the same
    PromptReps model load_sparse_encoder returns).
    """
    if _is_promptreps(model):
        return _load_promptreps(model, **kwargs)

    if any(model.startswith(p) for p in _HF_EMBED_PREFIXES):
        if "base_url" in kwargs:
            from evalgrid.models.embedders.vllm_embed import VLLMEmbedder
            return VLLMEmbedder(model=model, **kwargs)
        from evalgrid.models.embedders.huggingface_embed import HuggingFaceEmbedder
        return HuggingFaceEmbedder(model=model, **kwargs)

    if _is_vllm(model) or "base_url" in kwargs:
        from evalgrid.models.embedders.vllm_embed import VLLMEmbedder
        return VLLMEmbedder(model=model, **kwargs)

    if any(model.startswith(p) for p in _OPENAI_EMBED_PREFIXES):
        from evalgrid.models.embedders.openai_embed import OpenAIEmbedder
        return OpenAIEmbedder(model=model, **kwargs)

    raise ValueError(
        f"Cannot infer backend for embedder model '{model}'. "
        "Pass base_url=... to force vllm, or add a prefix match in factory.py."
    )


def load_sparse_encoder(model: str, **kwargs) -> SparseEncoderModel:
    """
    Route `model` to a SparseEncoderModel backend (learned term-weight
    encoding, e.g. for evalgrid/benchmarks/retrievers/sparse.py). Any SPLADE-style
    checkpoint (model name containing "splade", e.g. "naver/splade-v3"), or
    "promptreps:<hf-chat-model>" (e.g.
    "promptreps:meta-llama/Meta-Llama-3-8B-Instruct").
    """
    if _is_splade(model):
        from evalgrid.models.embedders.splade_embed import SpladeModel
        return SpladeModel(model=model, **kwargs)

    if _is_promptreps(model):
        return _load_promptreps(model, **kwargs)

    raise ValueError(
        f"No sparse encoder backend for '{model}'. "
        "Pass a SPLADE or promptreps:<llm> model id, or add a new "
        "SparseEncoderModel backend in factory.py."
    )


def load_multi_vector_model(model: str, **kwargs) -> MultiVectorModel:
    """
    Route `model` to a MultiVectorModel backend (late-interaction /
    ColBERT-style encoding, e.g. for evalgrid/benchmarks/retrievers/multi_vector.py).
    Any pylate-compatible ColBERT checkpoint (model name containing
    "colbert", e.g. "colbert-ir/colbertv2.0").
    """
    if _is_colbert(model):
        from evalgrid.models.embedders.colbert_embed import ColBERTModel
        return ColBERTModel(model=model, **kwargs)

    raise ValueError(
        f"No multi-vector backend for '{model}'. "
        "Pass a ColBERT-style model id, or add a new MultiVectorModel backend in factory.py."
    )
