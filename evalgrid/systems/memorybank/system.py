"""
evalgrid/systems/memorybank/system.py
-------------------------------------
MemoryBank behind the MemorySystem interface.

The shape of the system is upstream's, and it is unusually well suited to this
repo's staged pipeline — better separated in the original than any other system
here. The three phases are already three separate functions upstream, and two of
them touch no model at all:

    write        append {query, response} to history[bucket]      0 LLM calls
    summarise    one batch pass over the buckets                  2 calls/bucket
                 (+ 2 global), explicitly offline and idempotent
    index        build a vector store over turn-pairs + summaries 0 LLM calls

So ``ingest`` is a pure append and ``is_last`` triggers the one expensive pass —
exactly the split ``is_last`` exists for.

Everything here is a rewrite rather than a wrapper; see the package docstring
for why (upstream's 2023 dependency set does not co-install on a 2026 Python).
The prompts are the exception and are imported verbatim. Each departure from
upstream's behaviour is marked DEVIATION below, and the load-bearing ones are:

DEVIATION: **the date key.** ``save_local_memory`` buckets by
``time.strftime("%Y-%m-%d")`` — the wall-clock date of the run, not any date in
the data — so every sample would be stamped with the day the job ran. Buckets
here are sessions by default (see store.py); ``bucket_by: date`` uses the
session's own date.

DEVIATION: **forgetting is off, non-destructive and seeded** when on. Upstream
runs it at index-build time, pops turns out of the dict, rewrites the source
JSON, and drives it from an unseeded ``random.random()``. On a benchmark where
the answer lives in exactly one session that destroys recall irreversibly and
irreproducibly. See store.apply_forgetting.

DEVIATION: **retrieval does not write.** ``search_memory`` bumps
``memory_strength`` and rewrites the whole memory JSON on every query, so
answering mutates the memory being evaluated — which would make two retriever
configs see different stores. The recall-feedback loop is implemented but
in-memory only, and off by default.

DEVIATION: **no persona prompt.** All four of upstream's answer templates are
AI-companion / psychological-counsellor personas, they live in
``utils/prompt_utils.py`` (not vendored here), and their ``{history_summary}``
slot is dead — ``overall_history`` is computed by a dedicated LLM call and then
interpolated into no template at all. Answering uses this repo's shared prompt,
as every other system does.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime
from typing import Optional

from evalgrid.benchmarks.types import MemorySystem, Query, Session
from evalgrid.models.base import GeneratorModel
from evalgrid.lib.trace import record_calls

from evalgrid.systems.memorybank.store import (
    MemoryBankStore,
    iter_store_units,
    pair_turns,
)

# Timestamp shapes the benchmarks ship, as in hypermem's parser. MemoryBank
# needs a *date* out of these, for the bucket key and the forgetting clock —
# upstream's ``_get_date_difference`` does strptime(x, "%Y-%m-%d") and raises on
# anything else (forget_memory.py:56-60).
_TS_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%I:%M %p on %d %B, %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def to_date(raw: Optional[str]) -> Optional[date]:
    """Parse a benchmark timestamp down to a date, or None if it has none."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        pass
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def day_difference(earlier: Optional[str], later: Optional[str]) -> float:
    """Days between two benchmark timestamps; 0 when either is unparseable.

    Upstream raises here instead (strptime on a non-ISO string). Returning 0 —
    "no time has passed" — is the conservative direction: it keeps a unit rather
    than deleting it on a parse failure.
    """
    a, b = to_date(earlier), to_date(later)
    if a is None or b is None:
        return 0.0
    return float((b - a).days)


class MemoryBankBase(MemorySystem):
    """Shared MemoryBank plumbing. Subclasses implement ``ingest``.

    Args:
        config: the parsed evalgrid/configs/memorybank/<variant>.yaml.
        generator: answers queries. The *memory* model is separate and is only
            used by the summarisation pass.
    """

    supports_persistence = True

    def __init__(self, config: dict, generator: GeneratorModel):
        self.config = config
        self.generator = generator

        self.bucket_by = str(config.get("bucket_by", "session"))
        if self.bucket_by not in ("session", "date"):
            raise ValueError(f"bucket_by must be 'session' or 'date', got {self.bucket_by!r}")

        summarise = config.get("summarize") or {}
        self.summarize_events = bool(summarise.get("events", True))
        self.summarize_personality = bool(summarise.get("personality", False))
        self.summarize_overall = bool(summarise.get("overall", False))

        forget = config.get("forget") or {}
        self.forget_enabled = bool(forget.get("enabled", False))
        self.forget_curve = str(forget.get("curve", "paper"))
        self.forget_seed = int(forget.get("seed", 0))

        self.update_on_recall = bool(config.get("update_on_recall", False))
        self.top_k = int(config.get("retrieve_k", 6))
        self.max_tokens = int(config.get("answer_max_tokens", 2000))
        self.embedder_model = (config.get("embedder") or {}).get(
            "model", "sentence-transformers/all-MiniLM-L6-v2")

        # Upstream's generation config for every summarisation call
        # (summarize_memory.py:56-63), minus "n" (always 1 here, as upstream's
        # callers pass prompt_num=1). Forwarded to the endpoint as-is.
        self.summary_params = dict(config.get("summary_params") or {
            "temperature": 0.7, "max_tokens": 400, "top_p": 1.0,
            "frequency_penalty": 0.4, "presence_penalty": 0.2,
        })

        self._memory_llm = None     # built lazily: unused when summaries are off
        self._embedder = None       # built lazily: only the native answer path
        self._index = None          # (unit_ids, texts, records, matrix)

        self.store = MemoryBankStore()
        self.traces: list[dict] = []
        self.forgotten_units = 0
        self.reset()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @property
    def memory_llm(self) -> GeneratorModel:
        """The model the summarisation pass uses.

        This repo's own generator adapter, not a bespoke client: MemoryBank's
        LLM use is plain chat completion with no structured output and no
        interface to satisfy, so there is nothing a hand-rolled controller would
        buy that ``load_generator`` does not already give (backend routing,
        token accounting).
        """
        if self._memory_llm is None:
            from evalgrid.models.factory import load_generator

            llm = self.config.get("memory_llm") or {}
            if not llm.get("model"):
                raise ValueError(
                    "memorybank config has no memory_llm.model, but summarisation is "
                    "enabled. It is injected by MemoryBankPlugin.apply_mem_model from "
                    "--mem-model; if you built this system directly, set it yourself."
                )
            kwargs = {"model": llm["model"]}
            if llm.get("base_url"):
                kwargs["base_url"] = llm["base_url"]
            if llm.get("api_key"):
                kwargs["api_key"] = llm["api_key"]
            self._memory_llm = load_generator(**kwargs)
            self._install_trace()
        return self._memory_llm

    def reset(self) -> None:
        """Clear all memory before each sample.

        A dict and (on the answer path) a numpy matrix. There is no server, no
        FAISS directory to remove, and no process to tear down — the cheapest
        reset of any system here.
        """
        self.store = MemoryBankStore()
        self._index = None
        self.traces = []
        self.forgotten_units = 0
        # Baseline, not a reset: the generator is built once and outlives every
        # sample, so meta.json must report this sample's truncations, not the
        # run's running total. Zeroing a counter on a shared object would lie
        # to anything else reading it.
        self._truncated_at_reset = getattr(self._memory_llm, "truncated_calls", 0)
        self._install_trace()

    # ------------------------------------------------------------------
    # Trajectory recording (evalgrid/lib/trace.py) — L1
    # ------------------------------------------------------------------

    def install_recorder(self, recorder) -> None:
        super().install_recorder(recorder)
        self._install_trace()

    def _install_trace(self) -> None:
        """Log MemoryBank's summarisation calls — the only LLM it makes.

        The write path and retrieval spend nothing, so with ``summarize.events``
        off this records zero calls, which is the correct trajectory for that
        configuration rather than a gap. The generator is built lazily, so this
        also runs from ``_llm()`` for the case where the first summarisation
        happens after the last reset.

        The stage is static, and that is not a shortcut: MemoryBank has exactly
        one kind of LLM call. Naming it here rather than leaving it to default
        to "llm" is the difference between a trajectory that can be grouped
        with the other systems' and one that reads as "an LLM call nobody
        claimed" — which is what "llm" means (see lib/trace.py STAGES).
        """
        if self._memory_llm is not None:
            record_calls(self._memory_llm, "generate", recorder=self.recorder,
                         stage="summarize", step="memorybank.summarize")

    # ------------------------------------------------------------------
    # Ingest — the write path, no LLM
    # ------------------------------------------------------------------

    def add_session(self, session: Session, roles: str = "all") -> None:
        """Append one session's turn-pairs. Zero LLM calls, as upstream."""
        bucket = self._bucket_key(session)
        turns = list(session.turns or [])

        if not turns and session.text:
            self.store.add_pair(
                bucket, session.text, "",
                session_id=session.session_id, turn_ids=[], has_answer=False,
                timestamp=session.timestamp)
            return

        # "user" opens a pair only where that role exists. LoCoMo's roles are
        # speaker names, so pairing falls back to alternation.
        query_role = "user" if any(t.role == "user" for t in turns) else None
        for query_turn, response_turn, indices in pair_turns(turns, query_role):
            if roles == "user_only" and query_turn is None:
                # An assistant-only pair carries no user turn; under user_only
                # it would be an empty unit.
                continue
            turn_ids, has_answer = [], False
            for i in indices:
                turn = turns[i]
                if roles == "user_only" and turn.role != "user":
                    continue
                turn_ids.append(turn.turn_id or f"{session.session_id}:t{i}")
                has_answer = has_answer or bool(turn.has_answer)
            self.store.add_pair(
                bucket,
                query_turn.content if query_turn is not None else "",
                "" if roles == "user_only" or response_turn is None
                else response_turn.content,
                session_id=session.session_id,
                turn_ids=turn_ids,
                has_answer=has_answer,
                timestamp=session.timestamp,
            )

    def _bucket_key(self, session: Session) -> str:
        if self.bucket_by == "session":
            return str(session.session_id)
        parsed = to_date(session.timestamp)
        if parsed is None:
            raise ValueError(
                f"bucket_by='date' but session {session.session_id!r} has an "
                f"unparseable timestamp {session.timestamp!r}. Upstream would bucket it "
                f"under today's date; that silently stamps the run date onto the data."
            )
        return parsed.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Summarisation — the one expensive pass
    # ------------------------------------------------------------------

    def summarize(self) -> None:
        """Upstream's ``summarize_memory``, re-expressed. Runs at ``is_last``.

        Cost, with B buckets: ``events`` is 1 call per bucket, ``personality``
        another, and ``overall`` two more for the whole sample. With the default
        config (events only) MemoryBank ingests a LongMemEval-S sample in ~50
        LLM calls — the cheapest constructed memory in this repo by a wide
        margin, which is exactly what makes it useful as a floor.

        Upstream skips a bucket that already has a summary; the same check is
        here, so re-running is idempotent.
        """
        from memory_bank.summarize_memory import (
            summarize_content_prompt,
            summarize_overall_personality,
            summarize_overall_prompt,
            summarize_person_prompt,
        )

        user = self.store.user
        for bucket, pairs in self.store.history.items():
            if not pairs:
                continue
            if self.summarize_events and not (self.store.summaries.get(bucket) or {}).get("content"):
                text = self._summarise(summarize_content_prompt(pairs, user, "AI", "en"))
                self.store.set_summary(bucket, text)
            if self.summarize_personality and not self.store.memory["personality"].get(bucket):
                self.store.memory["personality"][bucket] = self._summarise(
                    summarize_person_prompt(pairs, user, "AI", "en"))

        if not self.summarize_overall:
            return
        # DEVIATION: upstream recomputes both globals from scratch on every run
        # and feeds the *whole* history into one prompt. At 50+ buckets that
        # input grows without bound; at LongMemEval-M scale it needs hierarchical
        # folding, which is why this is off by default.
        if self.store.summaries:
            # NOTE: `overall_history` is dead code upstream — computed here and
            # interpolated into none of the four answer templates. It is kept
            # because it is part of the store's schema, not because it is used.
            self.store.memory["overall_history"] = self._summarise(
                summarize_overall_prompt(list(self.store.summaries.items()), language="en"))
        personalities = self.store.memory["personality"]
        if personalities:
            self.store.memory["overall_personality"] = self._summarise(
                summarize_overall_personality(list(personalities.items()), language="en"))

    def _summarise(self, prompt: str) -> str:
        """One summarisation call, with upstream's four-message framing.

        The system/user/system/user preamble is ``LLMClientSimple.generate_text_simple``
        (summarize_memory.py:33-37) verbatim — it is part of the prompt the
        paper's summaries were produced under.

        Upstream's overflow handler is deliberately NOT reproduced: on a context
        error it truncates the prompt to its last ``1800 - 200*retry``
        *characters* and retries, which on a benchmark-scale session throws away
        almost the entire input and does it silently. Here an overflow raises,
        the runner records the sample as failed, and the fix (fewer turns per
        bucket, or a larger server window) is a decision rather than a silent
        loss.
        """
        messages = [
            {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
            {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
            {"role": "system", "content": "Sure, I will do my best to assist you."},
            {"role": "user", "content": prompt},
        ]
        return asyncio.run(self.memory_llm.generate(messages, **self.summary_params))

    def _truncated_this_sample(self) -> int:
        """Summaries clipped at ``summary_params.max_tokens`` since ``reset()``."""
        total = getattr(self._memory_llm, "truncated_calls", 0)
        return max(0, total - getattr(self, "_truncated_at_reset", 0))

    def finalize(self) -> None:
        """Summarise, then optionally forget. Called on the last session."""
        if self.summarize_events or self.summarize_personality or self.summarize_overall:
            self.summarize()
        if self.forget_enabled:
            buckets = self.store.buckets()
            last = self.store.history[buckets[-1]][-1]["timestamp"] if buckets else None
            self.forgotten_units = self.store.apply_forgetting(
                last, variant=self.forget_curve, seed=self.forget_seed,
                day_difference=day_difference)
            if self.forgotten_units:
                print(f"[memorybank] forgetting ({self.forget_curve} curve, seed "
                      f"{self.forget_seed}) marked {self.forgotten_units} unit(s) "
                      f"unretrievable.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    #
    # memory.json is upstream's per-user dict — the complete state, and the only
    # state. Our provenance keys ride on the same turn-pair dicts upstream
    # injects memory_id/memory_strength/last_recall_date into, so the file is
    # still readable by anything that understands the original schema.

    def save_memory(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "memory.json"), "w") as f:
            json.dump(self.store.to_dict(), f, ensure_ascii=False)

        units = list(iter_store_units(self.store.to_dict()))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "buckets": len(self.store.history),
                "turn_pairs": self.store.turn_count(),
                "summaries": sum(1 for s in self.store.summaries.values() if s.get("content")),
                # Summaries the server stopped at summary_params.max_tokens, so
                # they end mid-word. Nothing raises when this happens, so
                # without the count a store of clipped summaries is
                # indistinguishable from a store of complete ones.
                "truncated_summaries": self._truncated_this_sample(),
                "units": len(units),
                "forgotten": self.forgotten_units,
                "bucket_by": self.bucket_by,
                "summarize": {"events": self.summarize_events,
                              "personality": self.summarize_personality,
                              "overall": self.summarize_overall},
                "forget": {"enabled": self.forget_enabled, "curve": self.forget_curve,
                           "seed": self.forget_seed},
            }, f, indent=2)

    def load_memory(self, path: str) -> None:
        with open(os.path.join(path, "memory.json")) as f:
            self.store = MemoryBankStore.from_dict(json.load(f))
        self._index = None

    # ------------------------------------------------------------------
    # Retrieval + answering (MemoryBank's native path)
    # ------------------------------------------------------------------

    def prepare_retrieval(self) -> None:
        """Embed the corpus now, outside the query clock (MemorySystem's hook).

        Under a native_retriever swap MemoryBank's own vectors are never read —
        the substitute ranks and ``regroup_by_bucket`` needs only the unit ids —
        so only the units are listed and MiniLM is not run at all.
        """
        self._build_index(embed=getattr(self, "_native_ranker", None) is None)

    def _build_index(self, *, embed: bool = True):
        """Embed every unit once. Upstream's FAISS store, without FAISS.

        A LongMemEval-S sample is a few hundred units, so an exact numpy cosine
        scan is both faster than building an ANN index and exactly equivalent to
        one. ``langchain.vectorstores.FAISS`` (0.0.146) cannot be imported on a
        current Python anyway.

        The embedder is MemoryBank's own EN default, all-MiniLM-L6-v2, loaded
        through this repo's factory so it is the same encoder the pipeline's
        dense retriever can be pointed at.

        ``embed=False`` lists the units without vectors, for callers that need
        only ids and records (``regroup_by_bucket`` under a swapped ranker); a
        later ``embed=True`` call fills the vectors in.
        """
        if self._index is not None and (
                not embed or not self._index[0] or self._index[3] is not None):
            return self._index
        import numpy as np

        from evalgrid.models.factory import load_embedder

        units = list(iter_store_units(self.store.to_dict()))
        if not units:
            self._index = ([], [], [], None)
            return self._index

        texts = [text for _, text, _, _ in units]
        vectors = None
        if embed:
            if self._embedder is None:
                self._embedder = load_embedder(self.embedder_model)
            vectors = np.asarray(asyncio.run(self._embedder.embed(texts)), dtype=np.float32)
            vectors /= (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        self._index = ([uid for uid, _, _, _ in units], texts,
                       [record for _, _, _, record in units], vectors)
        return self._index

    def retrieve(self, question: str,
                 k: Optional[int] = None,
                 ) -> tuple[list[str], str, list[str], list[list[str]], list[str]]:
        """Top-k cosine, then upstream's chronological regrouping.

        ``search_memory`` (local_doc_qa.py:326-357) sorts the hits by
        ``metadata["source"]`` and concatenates hits sharing a source into one
        block, so the generator reads memories grouped by bucket rather than in
        rank order — reading order is not ranking order. It returns that list
        plus a comma-joined date string.

        Not reproduced: the neighbour-merge patch
        (``similarity_search_with_score_by_vector``, forget_memory.py:187-230),
        which walks outward from each hit absorbing adjacent docs that share a
        ``source`` until 200 characters. Under the id scheme it is installed
        with, ``source`` is the unit's own unique ``memory_id``, so no neighbour
        can ever match and the expansion is dead code. Under the *other*
        loader's ``source`` (the bucket) it would absorb whole buckets — a
        different mechanism than the one described. Rather than pick one, this
        keeps the grouping (which is well defined either way) and drops the
        expansion.

        Returns (blocks, joined bucket string, hit unit ids, ids-per-block,
        ids in RANK order). The last one exists because the sort above is
        lossy and both orders are load-bearing — see :meth:`_regroup`.
        """
        import numpy as np

        unit_ids, _, records, vectors = self._build_index()
        if vectors is None:
            return [], "", [], [], []

        query_vector = np.asarray(
            asyncio.run(self._embedder.embed([question], is_query=True))[0], dtype=np.float32)
        query_vector /= (np.linalg.norm(query_vector) + 1e-12)
        scores = vectors @ query_vector
        top = np.argsort(-scores)[: (k or self.top_k)]

        return self._regroup(list(top))

    def regroup_by_bucket(self, unit_ids: list[str]):
        """Upstream's regrouping, driven by unit ids rather than index positions.

        Exists so a substituted ranker (evalgrid/systems/native_retrieval.py) can
        reuse this step: the regrouping is MemoryBank's method and is the part
        worth holding fixed while the ranking function changes.

        ``unit_ids`` must arrive in RANK order — it is what the 5th return
        value carries back out. Ids the index does not know are dropped, so a
        substituted ranker that returns foreign ids shortens the ranking rather
        than corrupting it.
        """
        known, _, _, _ = self._build_index(embed=False)
        position = {uid: i for i, uid in enumerate(known)}
        return self._regroup([position[u] for u in unit_ids if u in position])

    def _regroup(self, hits: list[int]):
        """Sort hits by bucket and concatenate those sharing one.

        ``search_memory`` (local_doc_qa.py:326-357) groups by
        ``metadata["source"]``, so the generator reads memories grouped by
        bucket rather than in rank order — reading order is not ranking order.

        BOTH orders leave here, because both are load-bearing and the sort
        below destroys the one it does not keep. Upstream never has to choose:
        ``retrieve`` cuts to k by score and only *then* regroups, so its
        grouping only ever sees memories that already won on rank. This repo
        splits those two steps across stages — retrieval saves one deep
        ranking and the answer stage sweeps k over it (pipeline/answer.py's
        depth cut) — so a saved run that carries reading order alone makes
        every later cut an alphabetical slice of the buckets. Two documented
        contracts say the same thing: ``systems/base.native_unit_ids`` returns
        ids "rank-ordered" (it becomes run.tsv, and recall@k on a bucket sort
        is meaningless), and ``lib/tokens.fit_to_budget`` takes facts "in rank
        order". So ``hit_ids``/``groups`` describe the PROMPT and ``ranked``
        describes the RANKING.
        """
        unit_ids, _, records, _ = self._index
        ranked = [unit_ids[i] for i in hits]      # captured before the sort
        hits = sorted(hits, key=lambda i: str(records[i].get("source", "")))
        blocks, buckets, hit_ids = [], [], []
        # Which hit ids went into which block. A block is several memories
        # concatenated, so without this a replay cutting to "k units" is
        # cutting BLOCKS and has no way to say how many memories that was —
        # see pipeline/answer.py's depth cut.
        groups: list[list[str]] = []
        previous = None
        for i in hits:
            record = records[i]
            source = str(record.get("source", ""))
            text = self._render(i)
            hit_ids.append(unit_ids[i])
            if source != previous:
                blocks.append(text)
                buckets.append(source)
                groups.append([unit_ids[i]])
                previous = source
            else:
                blocks[-1] += f"\n{text}"
                groups[-1].append(unit_ids[i])
            if self.update_on_recall:
                # Upstream's retrieval->storage feedback loop, in memory only:
                # it rewrites the whole JSON per query, which would make
                # answering mutate the store being evaluated.
                record["memory_strength"] = record.get("memory_strength", 1) + 1
        return blocks, ", ".join(buckets), hit_ids, groups, ranked

    def _render(self, i: int) -> str:
        _, texts, _, _ = self._index
        return texts[i]

    def retrieve_trace(self, query: Query) -> dict:
        """Dense top-k plus upstream's chronological regrouping. No generator call."""
        blocks, buckets, _reading, groups, ranked = self.retrieve(query.question)
        trace = {
            "query_id": query.query_id,
            "step1_keywords": buckets,          # the memo-date string
            "step2_retrieved_context": "\n\n".join(blocks),
            # RANK order, per systems/base.native_unit_ids — this is what
            # becomes run.tsv, and recall@k / NDCG@k / MRR are only defined on a
            # ranking. The bucket-sorted reading order is not lost: it is
            # `unit_id_groups` below, flattened.
            "retrieved_unit_ids": ranked,
            # The blocks, not the hits: upstream concatenates hits that share a
            # bucket, so a block is what the generator actually sees as one
            # item. Budgeting on hits would cut inside a block.
            "retrieved_units": blocks,
            # Which ids each block holds. MemoryBank is the one system whose
            # unit is not one memory — its regrouping concatenates hits sharing
            # a date bucket — so a depth cut needs this to answer "k what?".
            # pipeline/answer.py cuts to k MEMORIES using it; without it k would
            # silently mean k blocks, which is 1.5x more memories on average.
            "unit_id_groups": groups,
        }
        self.traces.append(trace)
        return trace

    def answer(self, query: Query) -> str:
        return self.answer_from_trace(query, self.retrieve_trace(query))

    def answer_from_trace(self, query: Query, trace: dict) -> str:
        blocks = trace.get("retrieved_units")
        if blocks is None:
            blocks = [b for b in (trace.get("step2_retrieved_context") or "").split("\n\n") if b]
        messages = self.answer_messages(query, blocks)
        return asyncio.run(self.generator.generate(messages, max_tokens=self.max_tokens))

    def answer_messages(self, query: Query, blocks: list[str]) -> list[dict]:
        """This repo's shared prompt. See the module docstring for why not upstream's."""
        from evalgrid.systems.base import shared_answer_prompt

        return shared_answer_prompt(query.question, query.question_date, blocks)

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------

    def layer_usage(self) -> dict:
        """Split tokens by layer.

        Memory layer: the summarisation pass, and nothing else — the write path
        makes no LLM call and neither does retrieval. With ``summarize.events``
        off, this layer is exactly zero, which is a legitimate configuration:
        MemoryBank without summaries is a raw turn-pair corpus.

        Generation layer: the final answer, from our generator.
        """
        from evalgrid.models.base import TokenUsage

        usage = self._memory_llm.token_usage if self._memory_llm is not None else TokenUsage()
        return {
            "memory": TokenUsage(input_tokens=usage.input_tokens,
                                 output_tokens=usage.output_tokens,
                                 calls=usage.calls),
            "generation": self.generator.token_usage,
        }
