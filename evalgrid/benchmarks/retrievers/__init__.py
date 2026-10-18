from evalgrid.benchmarks.retrievers.base import Retriever
from evalgrid.benchmarks.retrievers.bm25 import BM25Retriever
from evalgrid.benchmarks.retrievers.dense import DenseRetriever
from evalgrid.benchmarks.retrievers.fusion import FusionRetriever
from evalgrid.benchmarks.retrievers.hybrid import HybridRetriever
from evalgrid.benchmarks.retrievers.rrf import RRFRetriever
from evalgrid.benchmarks.retrievers.sparse import SparseRetriever
from evalgrid.benchmarks.retrievers.multi_vector import MultiVectorRetriever


def build_retriever(name: str, **kwargs) -> Retriever:
    """Return a Retriever for ``name``.

    "bm25"         — lexical, no kwargs.
    "dense"        — embedding-based; pass embedder=<EmbedderModel> (e.g. from
                      models.factory.load_embedder, including Qwen3-Embedding-* via vllm).
    "fusion"       — hybrid BM25 + dense, MIN-MAX linear fusion; pass
                      embedder=<EmbedderModel> (same as dense) and optional
                      alpha (dense weight, default 0.5).
    "rrf"          — hybrid BM25 + dense, RECIPROCAL RANK FUSION; pass
                      embedder=<EmbedderModel> and optional rrf_k (default 60)
                      / depth (default None = fuse full lists). Same legs as
                      "fusion", different combiner, own run directory.
    "hybrid"       — dense + learned sparse (PromptReps-style); pass
                      embedder=<EmbedderModel>, encoder=<SparseEncoderModel>
                      (usually one dual-representation instance for both, e.g.
                      PromptReps) and optional alpha (dense weight, default 0.5).
    "sparse"       — learned term-weight based; pass encoder=<SparseEncoderModel>
                      (e.g. models.factory.load_sparse_encoder("naver/splade-v3")).
    "multi_vector" — late-interaction / ColBERT-style; pass model=<MultiVectorModel>
                      (e.g. models.factory.load_multi_vector_model("colbert-ir/colbertv2.0")).
    """
    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        return DenseRetriever(**kwargs)
    if name == "fusion":
        return FusionRetriever(**kwargs)
    if name == "rrf":
        return RRFRetriever(**kwargs)
    if name == "hybrid":
        return HybridRetriever(**kwargs)
    if name == "sparse":
        return SparseRetriever(**kwargs)
    if name == "multi_vector":
        return MultiVectorRetriever(**kwargs)
    raise ValueError(f"Unknown retriever: {name}")


__all__ = [
    "Retriever", "BM25Retriever", "DenseRetriever", "FusionRetriever",
    "RRFRetriever", "HybridRetriever", "SparseRetriever", "MultiVectorRetriever",
    "build_retriever",
]
