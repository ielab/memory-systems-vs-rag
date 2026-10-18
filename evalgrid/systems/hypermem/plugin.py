"""
evalgrid/systems/hypermem/plugin.py
-----------------------------------
HyperMem's SystemPlugin: how to build it, and how to read its saved hypergraph
back as a flat retrieval corpus.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from evalgrid.systems.base import MemoryUnit, SystemPlugin, route_mem_llm_to_openai


def fact_text(fact: dict) -> str:
    """The text HyperMem shows for a fact — ``FactNode.to_text()``, reimplemented.

    ``content (Time: <temporal>; Location: <spatial>)``. Kept byte-identical to
    ``structure.py::FactNode.to_text`` but written out here so ``iter_units``
    stays importable without HyperMem's dependency tree: the qrels stage must be
    able to flatten a store on a machine that cannot run the system.
    """
    parts = [fact.get("content") or ""]
    if fact.get("temporal"):
        parts.append(f"Time: {fact['temporal']}")
    if fact.get("spatial"):
        parts.append(f"Location: {fact['spatial']}")
    if len(parts) > 1:
        return f"{parts[0]} ({'; '.join(parts[1:])})"
    return parts[0]


class HyperMemPlugin(SystemPlugin):

    name = "hypermem"

    # An episode IS a session, and a fact inherits the flag from the
    # episodes it was extracted from — so every fact of the evidence session is
    # labelled. See records_has_answer.
    has_answer_granularity = "session"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, config: dict, *, dataset: str, generator, phase: str):
        config = json.loads(json.dumps(config))  # don't mutate the caller's dict

        # Subclasses imported inside the branches, not at module scope: they
        # reach vendor/hypermem, which pulls in aiohttp, json_repair, nltk,
        # rank_bm25 and torch. None of that may be a prerequisite for listing
        # the registry or for running some other system.
        if dataset == "locomo":
            from evalgrid.systems.hypermem.locomo import HyperMemLoCoMo
            return HyperMemLoCoMo(config, generator)
        if dataset.startswith("beam"):
            from evalgrid.systems.hypermem.beam import HyperMemBEAM
            return HyperMemBEAM(config, generator)
        from evalgrid.systems.hypermem.longmemeval import HyperMemLongMemEval
        return HyperMemLongMemEval(config, generator)

    def apply_mem_model(self, config: dict, mem_model: str,
                        mem_base_url: Optional[str] = None) -> dict:
        """HyperMem's memory LLM writes every layer of the hypergraph — episode
        summaries, topic titles, topic membership, role and weight assignments,
        facts — so it is the model that keys outputs/_memory/.

        Unlike A-Mem and SimpleMem it is NOT needed at answer time — nothing in
        the funnel prompts the memory model. The funnel is not model-free
        though: under the shipped config it calls a served embedder and a served
        cross-encoder reranker at query time. Neither is this model, and neither
        is configured from --mem-model.
        """
        llm = config.setdefault("memory_llm", {})
        llm["model"] = mem_model
        if mem_base_url:
            llm["base_url"] = mem_base_url
        route_mem_llm_to_openai(llm, llm.get("base_url"), mem_model)
        return config

    def native_retriever_name(self, config: dict) -> str:
        """HyperMem's first stage, plus whether a cross-encoder follows it.

        ``retrieval_type`` is the thing that varies: "keyword" is BM25 over the
        hypergraph's docs, while the default "rrf" fuses that with dense vectors
        carrying stage 3's embedding propagation. Both are built (embedding.py),
        so both have run dirs, and naming the segment by retrieval_type keeps
        them apart.

        ``-norerank`` marks the cross-encoder being OFF, and it has to be in the
        path because that flag changes the ranking twice over: it removes the
        rerank passes *and* narrows layers 1 and 2 from ``initial_candidates``
        to their ``top_k`` (stage4 L552/L665). Two runs differing by it are two
        retrieval conditions, and without a segment the second would overwrite
        the first — and be skipped as already-answered on the answer side.

        Which way round the suffix goes follows the repo convention (see
        lib/paths): the DEFAULT keeps the plain name, the non-default gets the
        suffix. The reranker is on by default, so ``rrf`` means with it.
        """
        # A substituted layer scorer wins, exactly as in the base implementation:
        # once native_retrieval swaps search_with_bm25 the funnel is no longer
        # scoring with BM25, and naming the run "bm25" would drop a SPLADE run
        # on top of the published keyword one. The rerank suffix still applies
        # on top of a swap — the two axes are independent.
        from evalgrid.systems.native_retrieval import retriever_slug

        swapped = retriever_slug(config)
        if swapped:
            base = swapped
        else:
            kind = str((config.get("retrieval") or {}).get("retrieval_type", "keyword"))
            base = "bm25" if kind == "keyword" else kind

        reranker = config.get("reranker") or {}
        if not bool(reranker.get("enabled", True)):
            base += "-norerank"
        return base

    def apply_retrieval_budget(self, config: dict, top_k: int) -> dict:
        """HyperMem retrieves per hypergraph level, not with one k.

        ``top_k`` sets the **fact** level, which is the atomic one and the only
        one directly comparable to another system's top-k. The coarser levels
        are candidate filters feeding it, so they scale with it rather than
        being pinned to the same number — pinning all three to top_k would
        starve the filter and measure the starvation.

        The ratios come from the PAPER's depths (§4.1 of arXiv:2604.08256:
        10 topics / 10 episodes / 30 facts), so both coarse levels sit at k/3
        and the funnel keeps its published shape as k sweeps. They previously
        followed ``config.py``'s shipped defaults instead (15/20/30 → 1/2 and
        2/3), which are not the configuration the paper reports.

        ``initial_candidates`` (the BM25 pool the funnel draws from) is left
        alone: it is a recall ceiling, not a context budget.
        """
        retrieval = config.setdefault("retrieval", {})
        levels = retrieval.setdefault("top_k", {})
        k = int(top_k)
        levels["fact_top_k"] = k
        levels["episode_top_k"] = max(1, k // 3)
        levels["topic_top_k"] = max(1, k // 3)
        return config

    # ------------------------------------------------------------------
    # Flatten saved memory into a retrieval corpus
    # ------------------------------------------------------------------

    def iter_units(self, sample_mem_dir: str) -> Iterable[MemoryUnit]:
        """Flatten the hypergraph into independently-retrievable units.

        Which levels are emitted follows the config's ``corpus_levels``, read
        from the store's own meta.json so the corpus of an existing store cannot
        change under it when the YAML is edited. The default is facts +
        episodes, which is what upstream's default ``output_type="011"`` feeds
        its generator — so the flat-corpus run is a like-for-like swap of the
        retriever, not of the content.

        Ids are namespaced ``fact::`` / ``ep::`` / ``topic::``. Fact ids are
        uuid4 and episode ids are session ids, so they could not collide today,
        but the namespaces make a unit's level readable from a run.tsv.

        Order is chronological by ingest: each episode's facts, then the episode
        itself, in session order. A fact derived from several episodes is
        emitted once, at the first. corpus.jsonl order is what the full-context
        baseline feeds the generator.

        Topics are excluded by default. They are the one level that is not
        self-contained — a title plus a summary of a cluster — and upstream
        leaves them out of the answer context too.
        """
        hypergraph_path = os.path.join(sample_mem_dir, "hypergraph.json")
        if not os.path.isfile(hypergraph_path):
            return
        with open(hypergraph_path) as f:
            hypergraph = json.load(f)
        provenance = self._provenance(sample_mem_dir)
        levels = self._levels(sample_mem_dir)

        facts = hypergraph.get("facts", {})
        episodes = hypergraph.get("episodes", {})
        fact_prov = provenance.get("facts", {})
        episode_prov = provenance.get("episodes", {})

        # fact -> the episode it should be emitted under (its first source)
        by_episode: dict[str, list[str]] = {}
        for fact_id, fact in facts.items():
            source_ids = [e for e in (fact.get("episode_ids") or []) if e in episodes]
            key = source_ids[0] if source_ids else ""
            by_episode.setdefault(key, []).append(fact_id)

        for episode_id, episode in episodes.items():
            if "facts" in levels:
                for fact_id in by_episode.get(episode_id, []):
                    yield self._fact_unit(fact_id, facts[fact_id], fact_prov)
            if "episodes" in levels:
                yield self._episode_unit(episode_id, episode, episode_prov)

        # Facts whose source episodes are all missing from the store — possible
        # only in a store written before an episode was dropped. Emitted last so
        # they are never silently lost from the corpus.
        if "facts" in levels:
            for fact_id in by_episode.get("", []):
                yield self._fact_unit(fact_id, facts[fact_id], fact_prov)

        if "topics" in levels:
            for topic_id, topic in hypergraph.get("topics", {}).items():
                yield MemoryUnit(
                    unit_id=f"topic::{topic_id}",
                    text=f"{topic.get('title') or ''}. {topic.get('summary') or ''}".strip(),
                    timestamp=None,
                    has_answer=False,
                    metadata={"level": "topic",
                              "keywords": topic.get("keywords") or [],
                              "episode_ids": topic.get("episode_ids") or []},
                )

    @staticmethod
    def _fact_unit(fact_id: str, fact: dict, fact_prov: dict) -> MemoryUnit:
        prov = fact_prov.get(fact_id, {})
        return MemoryUnit(
            unit_id=f"fact::{fact_id}",
            text=fact_text(fact),
            # Conversation time of the first source session, NOT the fact's own
            # `temporal` field: that is an event time in "relative (absolute)"
            # form, already inside the text, and null for many facts. Stamping
            # the corpus with when it was said keeps "[when] <fact>" uniform
            # with LightMem's memories and the raw-turn corpus, which is what
            # evalgrid.pipeline.answer.format_fact renders.
            timestamp=prov.get("timestamp"),
            has_answer=bool(prov.get("has_answer")),
            source_turn_ids=list(prov.get("source_turn_ids") or []),
            metadata={
                "level": "fact",
                "episode_ids": fact.get("episode_ids") or [],
                "topic_id": fact.get("topic_id"),
                "temporal": fact.get("temporal"),
                "spatial": fact.get("spatial"),
                "keywords": fact.get("keywords") or [],
                "query_patterns": fact.get("query_patterns") or [],
                "confidence": fact.get("confidence"),
                "session_ids": prov.get("session_ids") or [],
            },
        )

    @staticmethod
    def _episode_unit(episode_id: str, episode: dict, episode_prov: dict) -> MemoryUnit:
        prov = episode_prov.get(episode_id, {})
        return MemoryUnit(
            unit_id=f"ep::{episode_id}",
            # episode_description is the level upstream's formatter renders as
            # "[Memory i]"; subject and summary are index-side fields.
            text=episode.get("episode_description") or episode.get("summary") or "",
            timestamp=prov.get("timestamp"),
            has_answer=bool(prov.get("has_answer")),
            source_turn_ids=list(prov.get("source_turn_ids") or []),
            metadata={
                "level": "episode",
                "subject": episode.get("subject"),
                "summary": episode.get("summary"),
                "keywords": episode.get("keywords") or [],
                "session_id": prov.get("session_id"),
            },
        )

    @staticmethod
    def _provenance(sample_mem_dir: str) -> dict:
        path = os.path.join(sample_mem_dir, "provenance.json")
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _levels(sample_mem_dir: str) -> set:
        """Corpus levels for this store, from its meta.json.

        Read from the store rather than the live config: the corpus a store
        yields is a property of what was built, and a YAML edited between the
        qrels stage and the retrieve stage would otherwise silently change the
        corpus under an existing run.
        """
        path = os.path.join(sample_mem_dir, "meta.json")
        levels = "facts+episodes"
        if os.path.isfile(path):
            with open(path) as f:
                levels = json.load(f).get("corpus_levels", levels)
        return {part.strip() for part in str(levels).split("+")}

    def records_has_answer(self, sample_mem_dir: str) -> bool:
        """True if this sample's store carries provenance at all.

        Note what the flag means here. An episode is exactly one session, so an
        episode unit's has_answer is exact at session granularity. A fact
        inherits has_answer from the episodes it was extracted from — the
        extractor's own ``episode_ids`` link, validated against the topic's
        episode set — so it is session-level too, not turn-level. That is
        coarser than LightMem's per-fact ``source_id`` or A-Mem's
        one-note-per-turn, and the same granularity as SimpleMem's
        session-aligned windows: comparable between those two, not against
        LightMem or A-Mem. Use an LLM labeler for cross-system retrieval numbers.
        """
        path = os.path.join(sample_mem_dir, "provenance.json")
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            provenance = json.load(f)
        for section in ("episodes", "facts"):
            for record in (provenance.get(section) or {}).values():
                return "has_answer" in record
        return False

    # ------------------------------------------------------------------
    # The native path
    # ------------------------------------------------------------------

    def native_unit_ids(self, trace: dict) -> list[str]:
        """Flatten HyperMem's level-keyed trace into one ranked id list.

        Its retrieval is coarse-to-fine, so the trace is
        ``{"topics": [...], "episodes": [...], "facts": [...]}`` rather than a
        flat ranking. Facts first, then episodes: facts are the atomic level and
        are ranked last (most precisely), and this is also corpus order priority
        in iter_units.

        Topics are dropped. They are not in the corpus under either default —
        ``corpus_levels: facts+episodes`` leaves them out because a topic node is
        a title plus a cluster summary rather than a self-contained unit, and
        upstream's own ``output_type: "011"`` leaves them out of the answer
        context. Emitting them would rank ids the qrels cannot match and depress
        nDCG for a level neither pipeline actually answers from.

        **Ids are namespaced to match ``iter_units``.** The trace carries
        upstream's raw ids (``fact_<uuid>``, and a session id for an episode)
        while the corpus this run is scored against namespaces them
        ``fact::`` / ``ep::`` — see iter_units. Emitting the raw form made every
        id in the run.tsv unmatchable, so every HyperMem native run scored
        exactly 0 on every metric and read as a retrieval-quality result rather
        than a namespace bug. The two id vocabularies are produced by this one
        class and must not drift again; ``tools/check_native_ids.py`` pins it.
        """
        ids = trace.get("retrieved_unit_ids") or {}
        if isinstance(ids, list):     # a flat trace, if the shape ever changes
            return [str(i) for i in ids]
        return [f"{prefix}{i}"
                for level, prefix in (("facts", "fact::"), ("episodes", "ep::"))
                for i in (ids.get(level) or [])]
