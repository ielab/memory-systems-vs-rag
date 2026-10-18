"""
evalgrid/systems/hypermem/embedding.py
--------------------------------------
The dense half of HyperMem's native retrieval: node embeddings carrying stage
3's hyperedge propagation, in the shape ``stage4.search_with_emb`` expects.

This is the part of the method that is NOT a candidate filter. Upstream's own
comments say the hyperedge weights are never fused into a retrieval score, so
under ``retrieval_type: keyword`` the hypergraph only ever prunes candidates.
The weights do one other thing, and it happens here: they are softmaxed into
hyperedge embeddings, and those are added back into their member nodes as

    h'_v = h_v + alpha * Agg_e(h_e)      alpha = 0.5, Agg = sum

so a fact's vector is pulled toward the episode it belongs to and an episode's
toward its topic cluster. That residual is the only path by which the structure
reaches ranking, which is why running BM25-only is an ablation of the method
rather than a cheaper spelling of it.

TWO ADAPTERS LIVE HERE, and both exist for the same reason ``_bm25_index``
exists in system.py: upstream's builder is coupled to files on disk, and its
provider is coupled to policies this harness sets centrally.

``HyperMemEmbedder`` implements upstream's ``EmbeddingProvider`` surface —
``embed(texts) -> list[list[float]]`` and ``cosine_similarity(q, docs)`` — on
top of the harness's own embedder (``evalgrid.models.factory.load_embedder``).
Upstream's provider is bypassed rather than used, for three reasons:

  1. It retries connection failures five times with exponential backoff, which
     contradicts this repo's no-retry rule.
  2. It POSTs ``base_url`` verbatim while ``config.py`` defaults that to
     ``.../v1`` rather than ``.../v1/embeddings`` — a stock clone cannot reach
     its own server.
  3. Going through the factory means the embedding server, batching and dtype
     are the same ones every other system in the grid uses.

``build_emb_index`` calls upstream's ``build_embedding_index_for_single_conv``
**verbatim** through a temporary directory rather than reimplementing it. The
propagation is ~150 lines of node-text assembly, softmax weighting and residual
update, and the node text it embeds is NOT the BM25 searchable text (no field
repetition), so a reimplementation would be a second copy free to drift from
the vectors upstream actually scores. Our ``hypergraph.json`` is already
byte-compatible with the ``hypergraph_conv_N.json`` that builder reads (see
``HyperMemBase.save_memory``), so the round trip is a rename. The pickle it
writes is plain lists and dicts — no library objects — and is written and read
by the same process, so the version hazard that keeps ``_bm25_index`` in memory
does not apply.

WHAT UPSTREAM PROPAGATES, AND WHAT IT DOES NOT: facts are updated through
``fact_hyperedges`` and episodes through ``episode_hyperedges``. There is no
topic hyperedge, so **topic vectors are never propagated** — layer 1 ranks raw
title+summary embeddings. Nodes with no hyperedge keep their original vector.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


class HyperMemEmbedder:
    """Upstream's ``EmbeddingProvider`` contract, served by the harness embedder.

    Args:
        model: embedder id. The paper's native setting is Qwen3-Embedding-4B;
            anything ``evalgrid.models.factory.load_embedder`` can route works,
            and unlike upstream there is no ``'Qwen3' in model_name`` guard —
            that check lives in the vendored provider, which is not on this path.
        base_url: the OpenAI-compatible endpoint. Pass the server root
            (``http://host:11810/v1``); the harness's vLLM embedder appends the
            route, so upstream's ``/v1`` vs ``/v1/embeddings`` trap is not
            reproduced here.
        batch_size: texts per request when embedding the corpus.
        query_instruction: whether to mark queries as queries.

            **Default False, which is upstream's behaviour and is very likely
            wrong.** Qwen3-Embedding is instruction-tuned and asymmetric: its
            training expects queries to carry a retrieval instruction prefix and
            documents to go bare. Upstream's provider has no notion of the
            distinction and embeds both identically, so the published 92.73 was
            measured with queries unprefixed. Setting this True is expected to
            *help* retrieval, which is exactly why it is not the default — it
            would silently make our HyperMem a different (better) system than
            the one being compared. Flip it deliberately, as an ablation.
    """

    def __init__(self, model: str, *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, batch_size: int = 256,
                 query_instruction: bool = False, **kwargs):
        from evalgrid.models.factory import load_embedder

        load_kwargs = dict(kwargs)
        if base_url:
            load_kwargs["base_url"] = base_url
        if api_key:
            load_kwargs["api_key"] = api_key

        self.model = model
        self.batch_size = int(batch_size)
        self.query_instruction = bool(query_instruction)
        self._embedder = load_embedder(model, **load_kwargs)
        # stage4 embeds the SAME query once per layer — three identical round
        # trips per question. Caching by text is behaviour-preserving (the
        # embedder is deterministic for a fixed model) and cuts it to one.
        self._query_cache: dict[str, list[float]] = {}
        self._in_corpus = False
        #: Embedding cost, which HyperMemBase.layer_usage cannot report because
        #: embedders carry no TokenUsage. Counted here so a run that needs the
        #: number has somewhere to read it from. Cache hits are not calls.
        self.calls = 0
        self.texts_embedded = 0

    # -- upstream's surface ------------------------------------------------

    @contextlib.contextmanager
    def corpus_mode(self):
        """Mark the enclosed ``embed`` calls as documents, not queries.

        Upstream has one method for both sides, so which one a call means is
        context, not argument. ``build_emb_index`` enters this; everything else
        that reaches ``embed`` is stage 4 embedding a query. Getting this wrong
        matters only when ``query_instruction`` is on, but then it matters at
        every layer, so the distinction is made explicit rather than inferred
        from batch size (a one-node corpus is a legitimate single-text batch).
        """
        previous = self._in_corpus
        self._in_corpus = True
        try:
            yield
        finally:
            self._in_corpus = previous

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Upstream's signature: sync, one vector per text.

        Documents are batched and uncached; queries are memoised, because
        ``hierarchical_retrieval`` embeds the same question once per layer and
        would otherwise pay three identical round trips per question.
        """
        if self._in_corpus:
            out: list[list[float]] = []
            for i in range(0, len(texts), self.batch_size):
                out.extend(self._embed_sync(texts[i:i + self.batch_size], is_query=False))
            return out

        missing = [t for t in dict.fromkeys(texts) if t not in self._query_cache]
        if missing:
            for text, vector in zip(
                    missing, self._embed_sync(missing, is_query=self.query_instruction)):
                self._query_cache[text] = vector
        return [self._query_cache[t] for t in texts]

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Upstream's implementation, kept identical including the 1e-9 floor.

        Note that ``search_with_emb`` L2-normalises both sides *before* calling
        this, so the division is a no-op there; the two fact/episode helpers do
        the same. Reimplementing it differently would be a silent scoring change
        for no gain, so it is copied rather than replaced with a one-liner.
        """
        dot_product = np.dot(doc_vecs, query_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        denominator = query_norm * doc_norms
        denominator[denominator == 0] = 1e-9
        return dot_product / denominator

    # -- plumbing ----------------------------------------------------------

    def _embed_sync(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """Run the async embedder from stage 4's synchronous call stack.

        Safe because nothing on this path holds a running loop: the answer phase
        drives generation with its own ``asyncio.run`` and has returned by the
        time retrieval starts.
        """
        self.calls += 1
        self.texts_embedded += len(texts)
        return asyncio.run(self._embedder.embed(texts, is_query=is_query))


def build_emb_index(hypergraph: dict, embedder: HyperMemEmbedder, *,
                    alpha: float = 0.5, aggregate_type: str = "sum",
                    batch_size: int = 256) -> list[dict]:
    """Embed every node and propagate hyperedges, via upstream's own builder.

    Returns upstream's flat list format — ``{"type", "id", "field",
    "embedding", "data"}`` per node — which is what ``search_with_emb`` and the
    two ``search_*_emb`` closures in stage 4 index into.

    Args:
        hypergraph: ``Hypergraph.to_dict()``, i.e. our ``hypergraph.json``.
        alpha: the residual weight, upstream's ``node_emb_update_weight``.
            Paper §4.1: 0.5.
        aggregate_type: ``"sum"`` (hub nodes move further) or ``"mean"``.
            ``config.py`` ships "sum"; the builder's own default is "mean", so
            this is passed explicitly rather than left to the signature.
    """
    from hypermem.main.stage3_hypergraph_index import (
        build_embedding_index_for_single_conv,
    )

    with tempfile.TemporaryDirectory(prefix="hypermem-emb-") as tmp:
        data_dir = Path(tmp) / "hypergraphs"
        emb_dir = Path(tmp) / "vectors"
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "hypergraph_conv_0.json", "w") as f:
            json.dump(hypergraph, f)

        # The builder swallows every exception into (False, "failed") and prints
        # the reason with rich. Capture that so a dead embedding server surfaces
        # as a message instead of an empty index.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), embedder.corpus_mode():
            ok, status = build_embedding_index_for_single_conv(
                conv_id=0,
                data_dir=data_dir,
                emb_save_dir=emb_dir,
                embedding_provider=embedder,
                alpha=alpha,
                aggregate_type=aggregate_type,
                batch_size=batch_size,
                # Each sample builds its own index into a fresh temp dir, so
                # there is never a stale one to resume from; leaving this True
                # would be harmless but would hide a genuine no-op.
                skip_existing=False,
            )
        if not ok:
            tail = buffer.getvalue().strip().splitlines()[-8:]
            raise RuntimeError(
                "HyperMem embedding index build failed"
                + (f" ({status})" if status else "")
                + ". Upstream returns a bare False here, so the cause is only in "
                "its console output:\n  " + "\n  ".join(tail or ["<no output>"])
                + "\nMost often this is the embedding server being unreachable, or "
                "a hypergraph with no embeddable node text."
            )

        with open(emb_dir / "hypergraph_embedding_index_conv_0.pkl", "rb") as f:
            return pickle.load(f)
