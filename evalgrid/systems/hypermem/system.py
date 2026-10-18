"""
evalgrid/systems/hypermem/system.py
-----------------------------------
Base class wrapping the vendored HyperMem behind the MemorySystem interface.

There is nothing to wrap in the usual sense: upstream has no ``HyperMem`` class,
no ``add``/``search``, and no library entry point — ``hypermem/__init__.py``
exports dataclasses. The system *is* six ``python stageN_*.py`` scripts passing
JSON through ``results/<experiment>/``. So this class re-expresses the
orchestration those scripts' ``main()`` functions hold, calling upstream's
extractors — and therefore upstream's prompts and validation — unchanged.

Everything under vendor/hypermem/ is upstream verbatim. Where upstream's
behaviour needs changing to run here, it is changed from this side. Each such
point is marked ADAPTER: below.

Four of them are worth reading before you trust a run:

ADAPTER: **stage 1 is replaced, not ported.** Upstream infers episode boundaries
because LoCoMo hands it a flat message stream: one ``_detect_boundary`` LLM call
*per message*, ~600 per conversation. Every benchmark here hands us sessions
already, so a Session maps 1:1 to an Episode and only
``_generate_episode_memory`` (1 call per session) survives. This is upstream's
own definition of an episode arrived at for free, and it is the single largest
cost saving available —

ADAPTER: **five uncapped retry loops.** ``_extract_new_topic``,
``_update_existing_topic``, ``_assign_episode_roles_and_weights``,
``_extract_facts_stage`` and ``_assign_fact_roles_stage`` each retry on a
validation failure inside ``while True:`` with no attempt limit, and each
catches bare ``Exception``, so nothing raised from inside can stop them. A model
that will not produce a schema-valid response for one topic therefore burns LLM
calls until the job is killed. Every extractor call here is wrapped in
``asyncio.wait_for``; ``CancelledError`` derives from BaseException, so it is
the one thing those handlers do not swallow. ``extraction_timeout_seconds``
bounds each call.

ADAPTER: **episode ids are the session ids**, not ``uuid4()``. Upstream mints a
random id per episode, which makes the corpus ids of a re-ingested sample
disagree with every run file that ranked the old ones. Session ids are stable,
already unique within a sample, and make provenance readable. They are never
shown to the model — every prompt renders episodes with simple ``episode_N``
labels and maps back afterwards.

ADAPTER: **provenance is threaded through the hypergraph.** Nothing upstream
records which turn produced what. Episodes get it exactly (1:1 with sessions);
facts inherit it from ``fact.episode_ids``, which the extractor fills in and
validates. So a fact's evidence granularity is the session, not the turn — see
plugin.records_has_answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

from evalgrid.benchmarks.types import MemorySystem, Query, Session
from evalgrid.models.base import GeneratorModel, TokenUsage
from evalgrid.lib.trace import record_calls
from evalgrid.lib.errors import LLMCallFailed

# Timestamp shapes the benchmarks ship. LongMemEval: "2023/05/20 (Sat) 01:33";
# LoCoMo: "1:56 pm on 8 May, 2023". ISO is tried first, via fromisoformat.
_TS_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%I:%M %p on %d %B, %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def parse_session_timestamp(raw: Optional[str]) -> datetime:
    """Parse a benchmark session timestamp into a datetime, or raise.

    Deliberately has no ``datetime.now()`` fallback. HyperMem's prompts lean
    hard on absolute time — the episode prompt is handed
    "March 14, 2024 (Thursday) at 3:00 PM UTC" and fact extraction gets a
    reference time for relative->absolute conversion — so a session silently
    stamped with today's date corrupts every temporal question in the sample and
    does it invisibly.
    """
    if not raw:
        raise ValueError(
            "session has no timestamp. HyperMem resolves relative time against it "
            "in both the episode and fact prompts, so an absent timestamp silently "
            "corrupts every temporal question."
        )
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"unrecognised session timestamp {raw!r}. Add its format to _TS_FORMATS in "
        f"evalgrid/systems/hypermem/system.py rather than letting it fall back."
    )


class HyperMemBase(MemorySystem):
    """Shared HyperMem plumbing. Subclasses implement ``ingest`` and ``answer_messages``.

    Args:
        config: the parsed evalgrid/configs/hypermem/<variant>.yaml.
        generator: answers queries. Upstream answers with its own LLM; this repo
            separates the two so every system is compared under one generator.
    """

    supports_persistence = True

    def __init__(self, config: dict, generator: GeneratorModel):
        self.config = config
        self.generator = generator

        self.topic_match_batch_size = int(config.get("topic_match_batch_size", 10))
        self.extraction_timeout_seconds = float(config.get("extraction_timeout_seconds", 600))
        self.max_extraction_retries = int(config.get("max_extraction_retries", 3))
        self.drop_original_data = bool(config.get("drop_original_data", True))
        # Which hypergraph levels become corpus units. Recorded into the store's
        # meta.json, because the corpus a store yields must not change when the
        # YAML is edited after the store was built (HyperMemPlugin._levels).
        self.corpus_levels = str(config.get("corpus_levels", "facts+episodes"))
        self.per_message_seconds = int(config.get("per_message_seconds", 30))
        self.max_tokens = int(config.get("answer_max_tokens", 4096))
        self.silence_upstream_stdout = bool(config.get("silence_upstream_stdout", True))

        native = config.get("retrieval") or {}
        self.retrieval_type = str(native.get("retrieval_type", "rrf"))
        self.output_type = str(native.get("output_type", "011"))
        self.rerank_type = str(native.get("rerank_type", "fix"))
        self.retrieval_config = dict(native.get("top_k") or {
            "initial_candidates": 100, "topic_top_k": 15,
            "episode_top_k": 20, "fact_top_k": 30,
        })

        # Dense leg. Only read when retrieval_type is rrf/vector; a keyword run
        # never builds an embedder, so a BM25-only config still needs no server.
        emb = config.get("embedding") or {}
        self.embedding_model = emb.get("model")
        self.embedding_base_url = emb.get("base_url")
        self.embedding_api_key = emb.get("api_key")
        self.embedding_batch_size = int(emb.get("batch_size", 256))
        self.embedding_query_instruction = bool(emb.get("query_instruction", False))
        # Upstream's propagation knobs. config.py calls these
        # node_emb_update_weight and hyperedge_emb_aggregate_type; paper §4.1
        # pins alpha (its lambda) at 0.5, and config.py ships "sum".
        self.embedding_alpha = float(emb.get("alpha", 0.5))
        self.embedding_aggregate_type = str(emb.get("aggregate_type", "sum"))

        # Cross-encoder rerank stage, ON by default — the paper's §4.1 setting.
        # It is not a post-filter: stage 4 reads
        # `initial_candidates if use_reranker else <layer>_top_k` at each layer,
        # so turning it on widens layers 1 and 2 to the full candidate pool and
        # lets the cross-encoder pick which topics and episodes the funnel then
        # traverses. See reranker.py; `enabled: false` restores the previous
        # behaviour and is a named variant, not a fallback.
        rr = config.get("reranker") or {}
        self.use_reranker = bool(rr.get("enabled", True))
        self.reranker_model = rr.get("model")
        self.reranker_base_url = rr.get("base_url")
        self.reranker_api_key = rr.get("api_key")
        self.reranker_batch_size = int(rr.get("batch_size", 32))
        self.reranker_timeout = float(rr.get("timeout_seconds", 120))
        self.reranker_max_retries = int(rr.get("max_retries", 1))

        self._provider = self._new_provider()
        self._extractors_cache = None
        self._bm25 = None          # (bm25, docs), built lazily at answer time
        self._embedder_cache = None  # HyperMemEmbedder, one per system
        self._emb = None           # emb_index list, built lazily at answer time
        self._reranker_cache = None  # HyperMemReranker, one per system

        self.episodes: list = []           # upstream Episode objects, ingest order
        self.hypergraph = None             # upstream Hypergraph, built at is_last
        self._episode_prov: dict[str, dict] = {}
        self.traces: list[dict] = []
        self.failed_topics = 0
        self.failed_facts = 0

        self.reset()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _new_provider(self):
        from evalgrid.systems.hypermem.provider import OpenAICompatProvider

        llm = self.config.get("memory_llm") or {}
        if not llm.get("model"):
            raise ValueError(
                "hypermem config has no memory_llm.model. It is injected by "
                "HyperMemPlugin.apply_mem_model from --mem-model; if you built this "
                "system directly, set it yourself."
            )
        return OpenAICompatProvider(
            model=llm["model"],
            base_url=llm.get("base_url"),
            api_key=llm.get("api_key"),
            temperature=float(llm.get("temperature", 0.0)),
            max_tokens=llm.get("max_tokens", 20000),
            json_mode=llm.get("json_mode", "auto"),
            max_retries=self.max_extraction_retries,
        )

    def _extractors(self):
        """The four upstream extractors, built once.

        Imported here, not at module scope: reaching vendor/hypermem pulls in
        aiohttp, json_repair and (through the extractors' own imports) the rest
        of the tree. None of that may be a prerequisite for listing the registry
        or running some other system.
        """
        if self._extractors_cache is None:
            from hypermem.extractors.conv_episode_extractor import ConvEpisodeExtractor
            from hypermem.extractors.fact_extractor import FactExtractor
            from hypermem.extractors.hypergraph_extractor import HypergraphExtractor
            from hypermem.extractors.topic_extractor import TopicExtractor

            self._extractors_cache = (
                # ADAPTER: our provider in place of upstream's OpenRouter client.
                # ConvEpisodeExtractor's signature defaults llm_provider to the
                # *class* LLMProvider (a typo upstream); passing ours explicitly
                # is what makes it work at all.
                ConvEpisodeExtractor(llm_provider=self._provider),
                TopicExtractor(llm_provider=self._provider,
                               topic_match_batch_size=self.topic_match_batch_size),
                FactExtractor(llm_provider=self._provider),
                HypergraphExtractor(),
            )
        return self._extractors_cache

    def reset(self) -> None:
        """Clear all memory before each sample.

        Everything HyperMem holds is in-process: a list of episodes and one
        hypergraph. There is no server, no database and no index directory —
        which is the one genuinely pleasant thing about integrating it.
        """
        # _extractors_cache and _provider deliberately survive: they hold no
        # per-sample state, and rebuilding them would re-open a client per sample.
        self.episodes = []
        self.hypergraph = None
        self._episode_prov = {}
        self._bm25 = None
        # Per-sample, like _bm25: the vectors belong to one sample's hypergraph
        # and ranking sample 2 against sample 1's index fails silently. The
        # embedder itself survives, for the same reason _provider does.
        self._emb = None
        self.traces = []
        self.failed_topics = 0
        self.failed_facts = 0
        # Baseline, not a reset: the provider outlives every sample by design
        # (see above), so zeroing its counter would lie to anything else
        # reading it. meta.json reports the delta.
        self._counters_at_reset = {
            name: getattr(self._provider, name, 0)
            for name in ("truncated_calls", "failed_calls", "fatal_calls")
        }
        self._install_trace()

    def _since_reset(self, name: str) -> int:
        """A provider counter's delta over this sample."""
        total = getattr(self._provider, name, 0)
        return max(0, total - getattr(self, "_counters_at_reset", {}).get(name, 0))

    def _truncated_this_sample(self) -> int:
        """Calls stopped at ``memory_llm.max_tokens`` since ``reset()``."""
        return self._since_reset("truncated_calls")

    # ------------------------------------------------------------------
    # Trajectory recording (evalgrid/lib/trace.py) — L1
    # ------------------------------------------------------------------

    def install_recorder(self, recorder) -> None:
        super().install_recorder(recorder)
        self._install_trace()

    def _install_trace(self) -> None:
        """Log every memory-LLM call HyperMem makes.

        ``generate`` is the whole LLMProvider protocol, and every extractor in
        the 6-stage chain (episode -> topic -> fact) is constructed with this
        one provider, so the chain shows up in call order. The provider survives
        reset by design, and ``record_calls`` is idempotent, so re-installing
        per sample cannot stack wrappers.

        One provider for six stages is also why L1 alone is not enough here: it
        makes every record identical except for its prompt, and their stage can
        only be "llm". The stage names come from ``_step`` instead, wrapped
        around this adapter's own calls into the chain — see ``_step``.
        """
        record_calls(self._provider, "generate",
                     recorder=self.recorder, step="hypermem.generate")

    @contextlib.contextmanager
    def _step(self, stage: str, step: str, *, inputs=None, meta: dict | None = None):
        """Name one stage of the extraction chain, priced off the provider.

        The chain's boundaries are visible here and nowhere else: upstream
        reaches all six of them through the same ``LLMProvider.generate``, so a
        wrapper on that method cannot tell an episode-memory call from a fact
        one, and the closed stage vocabulary in lib/trace.py would go unused.
        These are the seams this adapter already calls, mapped onto it:

        ``_generate_episode_memory``  extract  session -> episodic memory
        ``_extract_new_topic``        extract  seed the topic set
        ``_build_episode_hyperedge``  store    episode -> topic edge
        ``extract_topic``             update   match an episode into the set,
                                               creating or updating topics
        ``extract_facts``             extract  facts + their roles/weights
        ``build_hypergraph``          store    assemble the final graph

        The provider's TokenUsage is the cost source, which is the same counter
        ``layer_usage`` sums, so a step's cost and the run's total cannot
        disagree.
        """
        with self.recorder.step(stage, step, inputs=inputs,
                                usage=self._provider.token_usage, meta=meta) as handle:
            yield handle

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_session(self, session: Session, roles: str = "all") -> None:
        """Turn one Session into one Episode and generate its episodic memory.

        One LLM call. ``original_data`` is the message list upstream's prompts
        expect — ``speaker_name`` / ``content`` / ``timestamp`` — with per-message
        timestamps synthesised as ``session_time + per_message_seconds * i``,
        which is exactly what upstream's LoCoMo loader does for a session whose
        messages share one date.
        """
        # (original index, turn) — the index is into the ORIGINAL turn list, so a
        # turn id means the same turn here as in evalgrid.pipeline.qrels_turns
        # even when assistant turns are filtered out.
        kept = [(i, t) for i, t in enumerate(session.turns or [])
                if roles == "all" or t.role == "user"]
        if not kept and not session.text:
            return

        base = parse_session_timestamp(session.timestamp)
        messages = []
        provenance_turn_ids: list[str] = []
        has_answer = False

        if kept:
            for n, (i, turn) in enumerate(kept):
                stamp = base + timedelta(seconds=self.per_message_seconds * n)
                messages.append({
                    "speaker_name": turn.role,
                    "speaker_id": turn.role,
                    "content": turn.content,
                    "timestamp": stamp.isoformat(),
                })
                provenance_turn_ids.append(turn.turn_id or f"{session.session_id}:t{i}")
                has_answer = has_answer or bool(turn.has_answer)
        else:
            messages.append({
                "speaker_name": "user",
                "speaker_id": "user",
                "content": session.text,
                "timestamp": base.isoformat(),
            })

        episode = self._new_episode(session, base, messages)
        extractor = self._extractors()[0]
        with self._captured_stdout():
            with self._step("extract", "hypermem.episode_memory", inputs=messages):
                self._run(extractor._generate_episode_memory(episode, messages))

        self.episodes.append(episode)
        self._episode_prov[episode.event_id] = {
            "session_id": session.session_id,
            "source_turn_ids": provenance_turn_ids,
            "has_answer": has_answer,
            "timestamp": session.timestamp,
        }

    def _new_episode(self, session: Session, base: datetime, messages: list[dict]):
        from hypermem.types import Episode, RawDataType

        return Episode(
            # ADAPTER: session id, not uuid4 — see the module docstring.
            event_id=str(session.session_id),
            user_id_list=[str(session.session_id)],
            original_data=messages,
            timestamp=base,
            # Episode.__post_init__ rejects an empty summary, and upstream fills
            # this from the boundary detector's topic_summary — which we do not
            # run. _generate_episode_memory overwrites it on the next line.
            summary=f"Session {session.session_id}",
            participants=sorted({m["speaker_name"] for m in messages}),
            type=RawDataType.CONVERSATION,
        )

    def build_hypergraph(self) -> None:
        """Topics, then facts, then the hypergraph. Mirrors stage 2's main().

        Order matters and is not negotiable: topics absorb episodes as they
        stream in, and facts are extracted *per topic over all its episodes*, so
        fact extraction cannot run until the topic set is final. That is why
        this runs at ``is_last`` rather than per session.

        Cost, per sample, with E episodes ending in T topics:
            topic matching   sum over episodes of ceil(T_so_far / batch_size)
            topic create      1 per new topic
            topic update      1 per (episode, matched topic) pair
            episode hyperedge 1 per topic touched, per episode
            fact extraction   2 per topic
        On LongMemEval-S (E≈50, and haystack sessions are mostly unrelated
        distractors so T≈E) that is a few hundred calls per sample, which is why
        cost, not correctness, is the blocker for this system.
        """
        if not self.episodes:
            raise RuntimeError(
                "HyperMem has no episodes for this sample — every session failed "
                "episode-memory generation, or none was ingested."
            )
        _, topic_extractor, fact_extractor, hypergraph_extractor = self._extractors()

        topics, episode_hyperedges = self._extract_topics(topic_extractor)
        fact_results, fact_hyperedges = self._extract_facts(fact_extractor, topics)

        from hypermem.extractors.topic_extractor import TopicExtractResult

        with self._captured_stdout(), self._step("store", "hypermem.build_hypergraph"):
            self.hypergraph = hypergraph_extractor.build_hypergraph(
                episodes=self.episodes,
                fact_results=fact_results,
                fact_hyperedge_results=fact_hyperedges,
                # "merged" is the action stage 2 reports once every episode has
                # been folded in; the field is descriptive only.
                topic_extract_result=TopicExtractResult(topics=topics, action="merged"),
                episode_hyperedge_results=episode_hyperedges,
            )

        if not self.hypergraph.facts and not self.hypergraph.episodes:
            raise RuntimeError(
                "HyperMem built an empty hypergraph for this sample "
                f"({self.failed_topics} topic failure(s), {self.failed_facts} fact "
                f"failure(s)). Nothing would be saved, so the sample is left unsaved "
                f"for the next ingest to retry."
            )

    def _extract_topics(self, topic_extractor):
        """Stream episodes into the topic set, one at a time.

        Reproduces ``stage2.extract_topics_for_episodes``: seed a topic from the
        first episode, then for each subsequent episode call ``extract_topic``,
        which LLM-matches it against every existing topic (batched) and either
        creates a new topic or updates the matched ones. ``topic_map`` and
        ``hyperedge_map`` are keyed by topic id so a topic updated five times
        keeps one entry — its latest.
        """
        from hypermem.extractors.topic_extractor import TopicExtractRequest

        topic_map: dict = {}
        hyperedge_map: dict = {}
        # Collected inside the capture, printed after it. A print() in an except
        # block INSIDE _captured_stdout() goes to the StringIO with upstream's
        # prompt dumps and is discarded — which is how a run that lost a topic's
        # entire fact set to 331 rejected requests produced a log with no failure
        # line in it at all.
        failures: list[str] = []

        with self._captured_stdout():
            first = self.episodes[0]
            with self._step("extract", "hypermem.topic_new", inputs=first.summary):
                topic = self._run(topic_extractor._extract_new_topic([first]))
            if topic is None:
                raise RuntimeError(
                    "HyperMem could not create an initial topic for the first session; "
                    "upstream treats this as fatal for the conversation."
                )
            topic_map[topic.topic_id] = topic
            with self._step("store", "hypermem.episode_hyperedge", inputs=topic.topic_id):
                hyperedge_map[topic.topic_id] = self._run(
                    topic_extractor._build_episode_hyperedge(topic=topic, episodes=[first]))

            for idx in range(1, len(self.episodes)):
                request = TopicExtractRequest(
                    history_episode_list=self.episodes[:idx],
                    new_episode=self.episodes[idx],
                    existing_topics=list(topic_map.values()),
                )
                try:
                    with self._step("update", "hypermem.topic_match",
                                    inputs=self.episodes[idx].summary,
                                    meta={"existing_topics": len(topic_map)}):
                        result, hyperedges = self._run(topic_extractor.extract_topic(request))
                except (Exception, LLMCallFailed) as e:
                    # One session's topic assignment failing costs that session
                    # its topic membership — and so its facts — but the rest of
                    # the sample is still worth building. Counted into meta.json.
                    #
                    # LLMCallFailed is named explicitly because it is a
                    # BaseException (so upstream's retry loops cannot swallow
                    # it); a bare `except Exception` here would let it abort the
                    # whole sample instead of costing one episode.
                    self.failed_topics += 1
                    failures.append(f"[hypermem] topic extraction failed for episode "
                                    f"{self.episodes[idx].event_id!r}: {e}")
                    continue
                for topic in (result.topics if result else []):
                    topic_map[topic.topic_id] = topic
                for hyperedge in hyperedges or []:
                    if hyperedge.episode_hyperedge:
                        hyperedge_map[hyperedge.topic_id] = hyperedge

        for line in failures:
            print(line, file=sys.stderr)
        return list(topic_map.values()), list(hyperedge_map.values())

    def _extract_facts(self, fact_extractor, topics: list):
        """Two LLM calls per topic: extract facts, then assign roles/weights.

        Reproduces ``stage2.extract_facts_for_topics``. A topic whose episodes
        have all been lost (see ``_extract_topics``) is skipped, as upstream
        skips it.
        """
        episode_map = {e.event_id: e for e in self.episodes}
        fact_results, fact_hyperedges = [], []
        # See _extract_topics: printed after the capture, never inside it.
        failures: list[str] = []

        with self._captured_stdout():
            for topic in topics:
                topic_episodes = [episode_map[eid] for eid in topic.episode_ids
                                  if eid in episode_map]
                if not topic_episodes:
                    continue
                try:
                    with self._step("extract", "hypermem.fact_extract",
                                    inputs=topic.title,
                                    meta={"episodes": len(topic_episodes)}):
                        result, hyperedges = self._run(
                            fact_extractor.extract_facts(topic=topic, episodes=topic_episodes))
                except (Exception, LLMCallFailed) as e:
                    self.failed_facts += 1
                    failures.append(
                        f"[hypermem] fact extraction failed for topic "
                        f"{topic.topic_id!r} ({topic.title!r}, {len(topic_episodes)} "
                        f"episode(s) left with no facts): {e}")
                    continue
                if result:
                    fact_results.append(result)
                if hyperedges:
                    fact_hyperedges.extend(hyperedges)

        for line in failures:
            print(line, file=sys.stderr)
        return fact_results, fact_hyperedges

    def _run(self, coro):
        """Await one extractor coroutine under a hard wall-clock cap.

        The cap is not a nicety. Five of upstream's extraction stages retry
        inside ``while True:`` with no attempt limit and a bare
        ``except Exception``, so a model that will not satisfy one of their
        validators loops forever and nothing raised from inside can stop it.
        ``asyncio.wait_for`` cancels from the outside, and ``CancelledError``
        derives from BaseException — the one thing those handlers do not catch.
        See the module docstring.
        """
        async def _capped():
            return await asyncio.wait_for(coro, timeout=self.extraction_timeout_seconds)

        try:
            return asyncio.run(_capped())
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"HyperMem extraction exceeded extraction_timeout_seconds="
                f"{self.extraction_timeout_seconds}. Upstream retries schema-invalid "
                f"responses in an unbounded loop, so this usually means the memory model "
                f"cannot satisfy one of its validators — check the endpoint and "
                f"memory_llm.json_mode."
            ) from e

    @contextlib.contextmanager
    def _captured_stdout(self):
        """Swallow upstream's per-call prompt/response dumps.

        Every extraction stage prints its prompt (first 1000 chars) and its raw
        response to stdout, on a path that runs hundreds of calls per sample.
        Failures are counted structurally instead (``failed_topics`` /
        ``failed_facts``, both saved into meta.json).
        """
        if not self.silence_upstream_stdout:
            yield
            return
        with contextlib.redirect_stdout(io.StringIO()):
            yield

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    #
    # hypergraph.json is byte-compatible with what upstream's stage 2 writes as
    # hypergraph_conv_N.json, so a saved store can be fed straight to upstream's
    # own stage 3/4 scripts. Our provenance is kept in a separate file rather
    # than folded into it, precisely to preserve that property.

    def save_memory(self, path: str) -> None:
        if self.hypergraph is None:
            raise RuntimeError("nothing to save: build_hypergraph() has not run")
        os.makedirs(path, exist_ok=True)

        data = self.hypergraph.to_dict()
        if self.drop_original_data:
            # original_data is the verbatim message list of every session. It
            # duplicates the raw corpus, dominates file size, and no retrieval
            # path reads it — stage 3 indexes subject/summary/episode_description
            # and stage 4 formats episode_description.
            for episode in data.get("episodes", {}).values():
                episode["original_data"] = []

        with open(os.path.join(path, "hypergraph.json"), "w") as f:
            json.dump(data, f, ensure_ascii=False)

        with open(os.path.join(path, "provenance.json"), "w") as f:
            json.dump({
                "episodes": self._episode_prov,
                "facts": self._fact_provenance(),
            }, f, ensure_ascii=False, indent=2)

        stats = self.hypergraph.get_stats()
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                **stats,
                "sessions": len(self.episodes),
                "failed_topics": self.failed_topics,
                "failed_facts": self.failed_facts,
                # Calls the model ran to the max_tokens cap. These do NOT show
                # up in failed_topics/failed_facts: the truncated JSON is
                # repaired downstream and the stage "succeeds" on whatever
                # survived, so this is the only signal that a topic's facts
                # came from a clipped response.
                "truncated_calls": self._truncated_this_sample(),
                # LLM calls that raised. token_usage counts only calls that
                # returned, so ingest_usage.json's llm_calls cannot show these —
                # a sample that rejected 45% of its requests used to report a
                # perfectly clean cost table. failed_calls is every call that
                # exhausted its retries; fatal_calls is the subset abandoned as
                # unsatisfiable (prompt + max_tokens > the served window), which
                # is a configuration error, not a flaky server.
                "failed_calls": self._since_reset("failed_calls"),
                "fatal_calls": self._since_reset("fatal_calls"),
                # Which prompting mode actually built this store: json_mode
                # "auto" latches off on the first rejection and never returns.
                "json_mode": getattr(self._provider, "json_mode", None),
                "json_mode_fallback": bool(
                    getattr(self._provider, "json_mode_fallback", False)),
                "original_data_dropped": self.drop_original_data,
                "corpus_levels": self.corpus_levels,
            }, f, indent=2)

    def _fact_provenance(self) -> dict:
        """Derive each fact's provenance from the episodes it was extracted from.

        ``FactNode.episode_ids`` is filled by the extractor and validated against
        the topic's episode set, so this is the system's own link — not an
        inference of ours. Because an episode is exactly one session, a fact's
        evidence granularity is the session.
        """
        out = {}
        for fact_id, fact in self.hypergraph.facts.items():
            provs = [self._episode_prov[eid] for eid in fact.episode_ids
                     if eid in self._episode_prov]
            turn_ids: list[str] = []
            for prov in provs:
                turn_ids.extend(prov["source_turn_ids"])
            out[fact_id] = {
                "session_ids": [p["session_id"] for p in provs],
                "source_turn_ids": turn_ids,
                "has_answer": any(p["has_answer"] for p in provs),
                # First source episode's conversation time. Facts also carry
                # their own `temporal` string (the event time, in
                # "relative (absolute)" form); this is when it was said.
                "timestamp": provs[0]["timestamp"] if provs else None,
            }
        return out

    def load_memory(self, path: str) -> None:
        from hypermem.structure import Hypergraph

        with open(os.path.join(path, "hypergraph.json")) as f:
            self.hypergraph = Hypergraph.from_dict(json.load(f))
        prov_path = os.path.join(path, "provenance.json")
        if os.path.isfile(prov_path):
            with open(prov_path) as f:
                self._episode_prov = json.load(f).get("episodes", {})
        self._bm25 = None
        self._emb = None

    # ------------------------------------------------------------------
    # Retrieval + answering (HyperMem's native path)
    # ------------------------------------------------------------------

    def _native_config(self):
        """An ExperimentConfig with this run's knobs.

        ADAPTER: upstream's config is a class of class attributes read at import
        time from env vars, and ``hierarchical_retrieval`` takes it as an
        argument. Subclassing it overrides exactly what we set and leaves the
        rest — the adaptive top-k table, the temporal keyword lists — upstream's.
        """
        from hypermem.config import ExperimentConfig

        return type("EvalGridHyperMemConfig", (ExperimentConfig,), {
            "retrieval_type": self.retrieval_type,
            "rerank_type": self.rerank_type,
            "hypergraph_retrieval_output_type": self.output_type,
            "retrieval_config": self.retrieval_config,
            # Drives two things at once in stage 4: whether the rerank blocks
            # run, and whether each layer retrieves `initial_candidates` or
            # goes straight to its `<layer>_top_k`. The pool is only live when
            # this is True.
            "use_reranker": self.use_reranker,
            "reranker_config": {
                "model_name": self.reranker_model,
                "base_url": self.reranker_base_url,
            },
        })

    def prepare_retrieval(self) -> None:
        """Build this sample's indexes now, outside the query clock.

        Both are otherwise built lazily inside ``retrieve_context`` on the first
        question, which on LongMemEval (one question per sample) put a BM25
        build plus, under ``rrf``/``vector``, an embedding pass over every fact,
        episode and topic into EVERY reported retrieval latency. The query is
        still embedded live per question by upstream's funnel. Also primes a
        swapped ``LayerRanker``'s per-layer indexes over the same docs.
        """
        if self.hypergraph is None:
            return
        _, docs = self._bm25_index()
        if self._uses_dense():
            self._emb_index()
        if self.use_reranker:
            self._reranker()
        ranker = getattr(self, "_native_ranker", None)
        if ranker is not None and hasattr(ranker, "prepare"):
            ranker.prepare(docs)

    def _bm25_index(self):
        """Build stage 3's BM25 index in memory.

        Upstream's ``build_bm25_index_for_single_conv`` reads
        ``hypergraph_conv_N.json`` off disk and pickles the result; we have the
        hypergraph in hand and pickles of ``rank_bm25`` objects are a known
        version hazard. The corpus is built with upstream's own
        ``build_*_searchable_text`` helpers, so the indexed strings — including
        the token-repetition field weighting (fact content ×3, query_patterns
        ×2, keywords ×1) — are identical to what stage 3 would have produced.
        """
        if self._bm25 is not None:
            return self._bm25
        from rank_bm25 import BM25Okapi
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from hypermem.main.stage3_hypergraph_index import (
            build_episode_searchable_text,
            build_fact_searchable_text,
            build_topic_searchable_text,
            ensure_nltk_data,
            tokenize,
        )

        with self._captured_stdout():
            ensure_nltk_data()
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        data = self.hypergraph.to_dict()
        corpus, docs = [], []
        for kind, builder in (("fact", build_fact_searchable_text),
                              ("episode", build_episode_searchable_text),
                              ("topic", build_topic_searchable_text)):
            for node_id, node in data.get(f"{kind}s", {}).items():
                text = builder(node)
                # searchable_text is OURS, and upstream never reads it
                # (search_with_bm25 takes doc["data"] and doc["type"], nothing
                # else). It carries the untokenized string so a swapped layer
                # scorer — native_retrieval.LayerRanker — ranks the identical
                # documents this BM25 index is built over, field weighting and
                # all, instead of re-deriving them and quietly changing the
                # corpus along with the scorer.
                docs.append({"id": node_id, "type": kind, "data": node,
                             "searchable_text": text})
                corpus.append(tokenize(text, stemmer, stop_words))

        if not corpus:
            self._bm25 = (None, [])
        else:
            self._bm25 = (BM25Okapi(corpus), docs)
        return self._bm25

    def _uses_dense(self) -> bool:
        """Whether this run has a dense leg at all.

        Upstream decides the same way — ``use_emb = retrieval_type in ('vector',
        'rrf')`` — and then guards every dense branch on ``emb_index`` being
        truthy, so a config asking for rrf with no index silently degrades to
        BM25-only. That silence is the thing worth avoiding: a run that reported
        "rrf" in its output path while scoring pure BM25 would be unfalsifiable
        from the results. ``_embedder`` raises instead.
        """
        return self.retrieval_type in ("vector", "rrf")

    def _embedder(self):
        """The dense provider, built once and reused across samples."""
        if self._embedder_cache is None:
            from evalgrid.systems.hypermem.embedding import HyperMemEmbedder

            if not self.embedding_model:
                raise ValueError(
                    f"retrieval.retrieval_type={self.retrieval_type!r} needs a dense "
                    "leg, but the config has no embedding.model. Set one (the paper's "
                    "native setting is Qwen3-Embedding-4B, §4.1 of arXiv:2604.08256) "
                    "with embedding.base_url pointing at its server, or set "
                    "retrieval_type: keyword to run BM25-only — which ablates the "
                    "hyperedge propagation, so say so when reporting it."
                )
            self._embedder_cache = HyperMemEmbedder(
                self.embedding_model,
                base_url=self.embedding_base_url,
                api_key=self.embedding_api_key,
                batch_size=self.embedding_batch_size,
                query_instruction=self.embedding_query_instruction,
            )
        return self._embedder_cache

    def _reranker(self):
        """The cross-encoder, built once and reused across samples.

        Raises rather than degrading, for the reason ``_embedder`` does: a run
        whose directory says the reranker was on while its ranking came from
        the first stage alone is unfalsifiable from the results.
        """
        if self._reranker_cache is None:
            from evalgrid.systems.hypermem.reranker import HyperMemReranker

            if not self.reranker_model:
                raise ValueError(
                    "reranker.enabled is true but the config has no "
                    "reranker.model. Set one (the paper's native setting is "
                    "Qwen3-Reranker-4B, §4.1 of arXiv:2604.08256) with "
                    "reranker.base_url pointing at its server, or set "
                    "reranker.enabled: false to run the funnel without a "
                    "cross-encoder — which narrows layers 1 and 2 to their "
                    "top_k off the first-stage ranking, so say so when "
                    "reporting it.")
            self._reranker_cache = HyperMemReranker(
                self.reranker_model,
                base_url=self.reranker_base_url,
                api_key=self.reranker_api_key,
                batch_size=self.reranker_batch_size,
                timeout=self.reranker_timeout,
                max_retries=self.reranker_max_retries,
            )
        return self._reranker_cache

    def _emb_index(self):
        """Stage 3's propagated node vectors, built in memory for this sample.

        Cached exactly like ``_bm25_index``: one build per sample, reused by
        every question against it, dropped by ``reset``/``load_memory``.

        The cost is one embedding pass over every fact, episode and topic in the
        sample's hypergraph — real, but paid once per sample rather than per
        question, and small next to the hundreds of extraction calls ingest
        already spends.
        """
        if self._emb is None:
            from evalgrid.systems.hypermem.embedding import build_emb_index

            self._emb = build_emb_index(
                self.hypergraph.to_dict(),
                self._embedder(),
                alpha=self.embedding_alpha,
                aggregate_type=self.embedding_aggregate_type,
                batch_size=self.embedding_batch_size,
            )
        return self._emb

    def retrieve_context(self, question: str) -> tuple[str, dict]:
        """Coarse-to-fine hierarchical retrieval, then upstream's formatter.

        This is HyperMem's contribution and the part the staged pipeline
        replaces: layer 1 ranks topics, layer 2 ranks episodes **restricted to
        the hyperedges under the retained topics**, layer 3 ranks facts
        restricted to the retained episodes.

        The hyperedge weights are not fused into any retrieval score — upstream's
        own comments say so — so as a *filter* the structure only prunes, and a
        flat retriever over the same facts searches a strict superset. The
        weights do reach ranking, but only through the dense leg, where they are
        softmaxed into hyperedge embeddings and added back into member nodes
        (see embedding.py). Under ``retrieval_type: keyword`` that path is
        absent and the structure is filter-only.

        Returns (formatted context, raw results) so the trace can record ids.
        """
        from hypermem.main.stage4_hypergraph_retrieval import (
            format_hierarchical_results,
            hierarchical_retrieval,
        )

        bm25, docs = self._bm25_index()
        dense = self._uses_dense()
        with self._captured_stdout():
            results = hierarchical_retrieval(
                query=question,
                hypergraph=self.hypergraph.to_dict(),
                bm25=bm25,
                docs=docs,
                # Both None under "keyword": upstream guards its dense branches
                # on emb_index being truthy, so passing them would be inert, and
                # building an index a keyword run never reads would demand an
                # embedding server for nothing.
                emb_index=self._emb_index() if dense else None,
                embedding_provider=self._embedder() if dense else None,
                # Qwen3-Reranker-4B, the paper's §4.1 setting, on by default.
                # It is NOT a post-filter — layers 1 and 2 rerank the full
                # candidate pool and the survivors are what get_connected_*
                # traverses, so this decides which hyperedges the funnel walks
                # (reranker.py). Only the layer-3 fact pass is a terminal
                # reorder. None under `reranker.enabled: false`, which
                # `use_reranker` in _native_config matches so the pool width
                # and the rerank blocks stay consistent.
                reranker_provider=self._reranker() if self.use_reranker else None,
                config=self._native_config(),
            )
            context = format_hierarchical_results(
                results, self.speaker_a, self.speaker_b, self.output_type)
        return context, results

    #: Names upstream's context template interpolates ("memories for conversation
    #: between {speaker_1} and {speaker_2}"). LoCoMo overrides them per sample.
    speaker_a = "user"
    speaker_b = "assistant"

    def retrieve_trace(self, query: Query) -> dict:
        """The three-layer funnel. No generator call."""
        if self.hypergraph is None:
            raise RuntimeError("no memory loaded: call load_memory() first")
        context, results = self.retrieve_context(query.question)
        trace = {
            "query_id": query.query_id,
            "step2_retrieved_context": context,
            "retrieved_unit_ids": {
                level: [doc.get("id", "") for doc, _ in results.get(level, [])]
                for level in ("topics", "episodes", "facts")
            },
            # Facts then episodes — the order native_unit_ids flattens them in,
            # so a budget keeps the same prefix the run.tsv ranks first. Topics
            # are excluded here for the same reason they are excluded there:
            # they are not corpus units under either default.
            "retrieved_units": [
                self._unit_text(level, doc)
                for level in ("facts", "episodes")
                for doc, _ in results.get(level, [])
            ],
        }
        self.traces.append(trace)
        return trace

    @staticmethod
    def _unit_text(level: str, doc: dict) -> str:
        """The text of one retrieved node, per level.

        Must agree with the two other places this repo renders a HyperMem node,
        or a budgeted run answers from different strings than an unbudgeted one
        over the same ids:

          * ``format_hierarchical_results`` (upstream), which the unbudgeted
            answer path replays verbatim — ``content`` for a fact,
            ``episode_description`` for an episode.
          * ``HyperMemPlugin.iter_units``, which builds the flat corpus.

        The level is passed rather than sniffed from the keys, because the
        ``content or text or summary`` chain that used to build this looked
        level-agnostic and was not. It got both levels wrong:

          * an **episode** has no ``content`` and no ``text``, so it fell
            through to ``summary`` — the one-line index-side field — while the
            other two renderers used the long ``episode_description``.
          * a **fact** matched ``content`` on the first term and therefore
            dropped its ``temporal`` and ``spatial`` fields, which both other
            renderers carry. On a benchmark with a temporal question category,
            silently removing the time from every fact is the worse of the two.

        A ``--context-budget`` native run answered from those strings AND had
        its budget measured against them, over ids identical to the unbudgeted
        run's. ``fact_text`` is imported rather than reimplemented so there is
        one fact rendering in the repo, not a third copy free to drift.
        """
        from evalgrid.systems.hypermem.plugin import fact_text

        if level == "facts":
            return fact_text(doc)
        if level == "episodes":
            # Same fallback order as iter_units: an episode whose
            # _generate_episode_memory call failed has no description, and the
            # summary is better than an empty unit.
            return str(doc.get("episode_description") or doc.get("summary") or "")
        return str(doc.get("title") or doc.get("summary") or "")

    def answer(self, query: Query) -> str:
        return self.answer_from_trace(query, self.retrieve_trace(query))

    def answer_from_trace(self, query: Query, trace: dict) -> str:
        # Unbudgeted replay reuses upstream's own formatter output verbatim; a
        # budgeted one cannot, because format_hierarchical_results emits one
        # string for the whole funnel and there is no unit boundary inside it.
        context = trace.get("step2_retrieved_context") or ""
        if trace.get("budget_applied"):
            context = "\n".join(trace.get("retrieved_units") or [])
        messages = self.answer_messages(query, context)
        answer = asyncio.run(self.generator.generate(messages, max_tokens=self.max_tokens))
        return self.postprocess_answer(answer)

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        raise NotImplementedError

    def postprocess_answer(self, answer: str) -> str:
        """Hook for a prompt whose reply is not the answer itself (the CoT one)."""
        return answer

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------

    def layer_usage(self) -> dict:
        """Split tokens by layer.

        Memory layer: episode-memory generation, topic matching/creation/update,
        episode role-weight assignment, fact extraction and fact role
        assignment — all of it at ingest. Unlike A-Mem and SimpleMem, HyperMem
        calls no LLM at answer time: the funnel is BM25, vector similarity and
        hypergraph traversal, with no model in the loop.

        Generation layer: the final answer, from our generator.

        NOT COUNTED, and both are non-zero under this config's defaults:

        * the dense leg's embedding calls (retrieval_type rrf/vector). One pass
          over every node in the sample's hypergraph when ``_emb_index`` is
          first built, plus one query vector per distinct question.
        * the reranker's forward passes (``reranker.enabled``). Up to
          ``initial_candidates`` pairs per layer per question — so up to ~300
          one-token completions on the served cross-encoder for every question,
          which is the larger of the two by a wide margin.

        Neither carries TokenUsage in this harness (see evalgrid/models/base.py
        — the counter lives on GeneratorModel), so there is nothing to fold in
        here without changing what this dict means for every other system.
        ``HyperMemEmbedder.calls`` / ``.texts_embedded`` and
        ``HyperMemReranker.calls`` / ``.pairs_scored`` hold the raw counts if a
        run needs to report them. Only a ``keyword`` run with the reranker off
        constructs neither, and the figure below is then complete.
        """
        return {
            "memory": TokenUsage(
                input_tokens=self._provider.token_usage.input_tokens,
                output_tokens=self._provider.token_usage.output_tokens,
                calls=self._provider.token_usage.calls,
            ),
            "generation": self.generator.token_usage,
        }
