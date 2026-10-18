"""
evalgrid/systems/hypermem/reranker.py
-------------------------------------
The cross-encoder stage of HyperMem's native retrieval: Qwen3-Reranker scoring
(query, document) pairs, in the shape ``stage4.rerank_results`` expects.

WHAT IT ACTUALLY DOES, because the name understates it. This is not a cosmetic
reordering of a finished result list. With ``use_reranker`` on, stage 4 changes
how much each layer *retrieves* before it ranks:

    retrieve_top_n = initial_candidates if config.use_reranker else topic_top_k

(stage4_hypergraph_retrieval.py L552, and the same line for episodes at L665 and
facts at L781). So layer 1 pulls ``initial_candidates`` topics — 100, not 10 —
reranks them, and keeps ``topic_top_k``. Those kept topics become
``relevant_topic_ids``, which is what ``get_connected_episodes`` traverses. The
reranker therefore decides **which hyperedges the funnel walks**, and the same
again one level down for episodes → facts. Only the layer-3 fact rerank is a
pure reorder-and-truncate at the end of the pipeline.

That is worth being explicit about because this file's predecessor state — a
literal ``reranker_provider=None`` in system.py — was justified in a comment
saying the reranker "runs AFTER the funnel, so it cannot change what the
hyperedges admit". It runs *inside* it, at two of the three layers.

REPLACEMENT, NOT FUSION. ``rerank_results`` builds ``zip(docs, rerank_scores)``
and sorts on that alone. The first-stage score — BM25, cosine, or the RRF value
that fused them — is discarded, not blended. The retrieval type still decides
*which* candidates reach this stage, but past that point ranking is the
cross-encoder's alone. There is no fusion weight to tune because there is no
fusion.

WHY NOT UPSTREAM'S ``RerankerProvider``. Same three reasons ``HyperMemEmbedder``
bypasses upstream's ``EmbeddingProvider``, and one more:

  1. It retries twice over — ``Retry(total=10)`` mounted on the session AND a
     10-attempt exponential-backoff loop around the POST — which compounds to
     ~100 sends of a request that failed, against a repo-wide rule of one
     (our development notes §"no-retry"). The backoff alone sleeps 2^0..2^9 ≈ 17 minutes.
  2. It POSTs every candidate in one request body. At ``initial_candidates: 100``
     with full episode descriptions that is a large single POST, three times per
     question, with no way to bound it.
  3. It hard-guards ``'Qwen3' not in model_name``, so the vendored path cannot
     score with anything else even as an ablation.
  4. Going through one client means the reranker endpoint is configured, keyed
     and failure-reported like every other served model in the grid.

The **prompt template and the scoring arithmetic are copied verbatim** from
upstream, including the ``-10`` default logprob and the ``logprobs=20`` request.
Those are what make a score a Qwen3-Reranker score, and a run that changed them
would not be comparable to the published one. What changes here is transport,
batching and failure policy.

BATCHING IS SCORE-PRESERVING. Each (query, document) pair is an independent
completion — one prompt, one output token, no shared state — so splitting 100
prompts into four requests of 25 returns the same 100 scores in the same order
as one request of 100. It bounds the request body; it does not touch the model.

A FAILURE HERE FAILS THE QUESTION, deliberately. Upstream catches its own
reranker errors and falls back to ``results[:top_k]`` — the first-stage order —
which yields a run that reports "reranked" while some unknown subset of its
questions was not. ``LLMCallFailed`` is a ``BaseException``, so upstream's
``except Exception`` at L748/L885 cannot catch it and the fallback never fires.
Consistent with the generator: at answer time this repo has no per-query
handler, so a dead endpoint stops the run loudly rather than quietly degrading
its ranking.
"""

from __future__ import annotations

import math
from typing import List, Optional

from evalgrid.lib.errors import LLMCallFailed

#: Upstream's chat-template scaffold, verbatim (reranker_provider.py L45-L46).
#: Qwen3-Reranker is a causal LM scored on a single yes/no continuation, so the
#: template is load-bearing: the tokens the logprobs are read from only appear
#: in the assistant slot this suffix opens.
_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based '
    'on the Query and the Instruct provided. Note that the answer can only be '
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

#: Upstream's default instruction, verbatim (reranker_provider.py L48).
_DEFAULT_INSTRUCTION = (
    "Given a user's question and a text passage, determine if the passage "
    "contains specific information that directly answers the question. A "
    "relevant passage should provide a clear and precise answer, not just be on "
    "the same topic."
)

#: Logprob assumed when the token is absent from the top-20. Upstream's value
#: and the HuggingFace model card's; -10 rather than -inf so a pair where
#: neither token surfaces scores 0.5 instead of dividing by zero.
_MISSING_LOGPROB = -10.0

#: 400s that resending byte-for-byte cannot fix. Mirrors provider.py's list —
#: the failure mode here is a document longer than the reranker's window, which
#: is a configuration error, not a bad minute on the server.
_FATAL_MARKERS = (
    "maximum context length",
    "context length exceeded",
)


class HyperMemReranker:
    """Upstream's ``RerankerProvider`` contract over an OpenAI-compatible server.

    The surface stage 4 calls is one method::

        rerank(queries, docs, instruction=None) -> list[float]

    with ``queries`` and ``docs`` parallel lists (upstream passes the same query
    repeated), and one score per pair in input order.

    Args:
        model: reranker id the endpoint serves. The paper's native setting is
            Qwen3-Reranker-4B (§4.1 of arXiv:2604.08256). Unlike upstream there
            is no ``'Qwen3' in model_name`` guard — but the yes/no template
            below IS Qwen3's, so another family needs its own template, not just
            another id.
        base_url: the OpenAI-compatible endpoint, ending in ``/v1``. Upstream's
            config.py holds the server ROOT (``http://host:12810``) and appends
            ``/v1/completions`` itself; the client used here appends
            ``/completions``, so the config value carries the ``/v1`` — the same
            convention the embedding block already uses.
        api_key: vLLM accepts any non-empty string.
        batch_size: pairs per request. See BATCHING above — this bounds the
            request body and changes no score.
        timeout: seconds per request.
        max_retries: attempts per request. **1 is the repo policy**, not a
            default to tune away: a retried failure costs N times as much and
            reports as clean. Raising it hides endpoint flakiness rather than
            fixing it.
    """

    def __init__(self, model: str, *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, batch_size: int = 32,
                 timeout: float = 120.0, max_retries: int = 1):
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "token",
            timeout=self.timeout,
            max_retries=0,   # ours, not the SDK's — see max_retries above
        )
        #: Reranking cost, which ``HyperMemBase.layer_usage`` cannot report:
        #: this is not a generator and carries no TokenUsage. Counted here so a
        #: run that needs the number has somewhere to read it from.
        self.calls = 0
        self.pairs_scored = 0

    # -- upstream's surface ------------------------------------------------

    def rerank(self, queries: List[str], docs: List[str],
               instruction: Optional[str] = None) -> List[float]:
        """One relevance score per (query, document) pair, in input order.

        Upstream's signature and upstream's arithmetic:
        ``P(yes) / (P(yes) + P(no))`` over the first generated token's top-20
        logprobs. The result is in [0, 1] and is comparable across pairs, which
        is all stage 4 needs — it sorts and truncates.
        """
        if len(queries) != len(docs):
            raise ValueError(
                f"rerank needs parallel lists, got {len(queries)} queries and "
                f"{len(docs)} documents")
        if not docs:
            return []

        instruct = _DEFAULT_INSTRUCTION if instruction is None else instruction
        prompts = [
            f"{_PREFIX}<Instruct>: {instruct}\n\n<Query>: {q}\n\n<Document>: {d}{_SUFFIX}"
            for q, d in zip(queries, docs)
        ]

        scores: List[float] = []
        for i in range(0, len(prompts), self.batch_size):
            scores.extend(self._score_batch(prompts[i:i + self.batch_size]))
        return scores

    # -- internals ---------------------------------------------------------

    def _score_batch(self, prompts: List[str]) -> List[float]:
        """One request. Raises LLMCallFailed rather than degrading silently."""
        last: Optional[BaseException] = None
        for _ in range(self.max_retries):
            try:
                response = self._client.completions.create(
                    model=self.model,
                    prompt=prompts,
                    max_tokens=1,
                    logprobs=20,      # upstream's value, and the model card's
                    temperature=0,
                )
            except BaseException as e:      # noqa: BLE001 — re-raised below
                last = e
                continue
            self.calls += 1
            self.pairs_scored += len(prompts)
            # vLLM returns choices in prompt order, but it is a list and the
            # index is on each choice, so order it explicitly rather than
            # trusting the wire: a silently permuted batch would be a ranking
            # bug with no symptom.
            ordered = sorted(response.choices, key=lambda c: c.index)
            if len(ordered) != len(prompts):
                raise LLMCallFailed(
                    f"reranker returned {len(ordered)} scores for {len(prompts)} "
                    f"prompts (model={self.model!r}, base_url={self.base_url!r})",
                    system="hypermem", deterministic=True)
            return [self._score(c) for c in ordered]

        text = str(last).lower()
        raise LLMCallFailed(
            f"reranker call failed (model={self.model!r}, "
            f"base_url={self.base_url!r}, {len(prompts)} pairs): {last}",
            system="hypermem",
            deterministic=any(m in text for m in _FATAL_MARKERS),
            cause=last if isinstance(last, BaseException) else None,
        )

    @staticmethod
    def _score(choice) -> float:
        """P(yes) / (P(yes) + P(no)) from the first token's top-20 logprobs.

        Upstream's arithmetic, including the ``-10`` fallback and the
        ``> 0`` guard. Written against the SDK's typed objects rather than raw
        JSON, so a server that returns no logprobs at all yields 0.5 (both
        tokens missing) instead of raising — the same outcome upstream's
        ``.get('logprobs', {})`` chain produces, and the honest one: it means
        the model expressed no preference this stage can read.
        """
        logprobs = getattr(choice, "logprobs", None)
        top = getattr(logprobs, "top_logprobs", None) or [{}]
        first = top[0] or {}
        yes = float(first.get("yes", _MISSING_LOGPROB))
        no = float(first.get("no", _MISSING_LOGPROB))
        yes_p, no_p = math.exp(yes), math.exp(no)
        total = yes_p + no_p
        return yes_p / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (f"HyperMemReranker(model={self.model!r}, base_url={self.base_url!r}, "
                f"batch_size={self.batch_size})")
