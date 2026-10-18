"""
evalgrid/systems/simplemem/system.py
------------------------------------
Base class wrapping the vendored SimpleMem (``SimpleMemSystem``) behind the
MemorySystem interface.

Holds the shared plumbing — construction, the metered memory-LLM client, the
window loop, provenance tracking, and persistence. Dataset-specific
ingest/answer behaviour lives in per-dataset subclasses:
  locomo.py      — SimpleMemLoCoMo: both speakers, upstream's native windowing
  longmemeval.py — SimpleMemLongMemEval: both speakers, session-aligned
                   windows by default, shared prompt

Everything under vendor/simplemem/ is upstream verbatim. Where upstream's
behaviour needs changing to run here, it is changed from this side — by
assigning over an attribute, or by driving a component directly — never by
editing the vendored file. Each such point is marked ADAPTER: below.

Three of them are worth reading before you trust a run:

ADAPTER: **we drive the window loop ourselves** instead of calling
``add_dialogue``/``finalize``. Upstream threads ``dialogue_ids`` into
``_parse_llm_response(response, dialogue_ids)`` and then never uses the
argument (memory_builder.py:308), so by the time entries reach the vector store
there is nothing left saying which turns produced them. Provenance is exactly
what evalgrid.pipeline.qrels_has_answer needs, so ``_extract_window`` calls
``MemoryBuilder._generate_memory_entries`` itself — the same single LLM call
``process_window`` would have made, with the same dedup hint and the same
``previous_entries`` bookkeeping — and records the window's turn ids alongside
each entry it returned.

ADAPTER: **provenance is window-level, not turn-level.** One extraction call
covers a whole window, so an entry maps to a *set* of turns and inherits
``has_answer`` from the whole window. Under the default session-aligned policy
that set is one session, which is precisely the granularity LongMemEval itself
ships (``answer_session_ids``) — but it is coarser than LightMem's per-fact
``source_id`` or A-Mem's one-note-per-turn, so has_answer qrels are NOT
comparable across those systems. Use the LLM labeler for cross-system retrieval
numbers: batched extraction blurs provenance.

ADAPTER: **a window whose extraction fails is silently dropped upstream.**
``_generate_memory_entries`` returns ``[]`` after three JSON-parse failures,
prints, and carries on — on LongMemEval-S that is a whole session of the
haystack deleted with no error anywhere. We count them (``empty_windows``,
saved into the store's meta.json) and raise if a sample produced no memory at
all, so the runner leaves that sample unsaved and the next ingest retries it.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
from typing import Optional

from evalgrid.benchmarks.types import MemorySystem, Query, Session
from evalgrid.models.base import GeneratorModel, TokenUsage
from evalgrid.lib.trace import record_calls
from evalgrid.systems.simplemem.plugin import parse_legs


class _UnusedEncoder:
    """Stands in for SimpleMem's ``EmbeddingModel`` when its vectors are dead.

    Under a ``native_retriever`` swap the substitute ranks the units and
    ``VectorStore.semantic_search`` — the only reader of the stored vectors — is
    replaced wholesale, so every embedding SimpleMem computes is written and
    never looked at. Upstream computes them anyway: ``SimpleMemSystem.__init__``
    constructs an ``EmbeddingModel`` (which loads the encoder onto the GPU in
    ITS constructor, eagerly) and ``add_entries`` runs a forward pass over every
    restatement on every ``load_memory``. That is ~4 GB resident plus a pass per
    sample, and on a node already running the memory server and a served
    embedder it is what tips the process over — see native_retrieval.declare.

    So the swap gets this instead. Entries are still inserted, with placeholder
    vectors of a nominal width: the rows are what the keyword (Tantivy) and
    structured (SQL) legs read, and what ``_install_simplemem`` reads back
    through ``get_all_entries``, and none of those touch the vector column.

    ``encode_single`` RAISES rather than returning a placeholder. Its one caller
    is the ``semantic_search`` the swap replaces, so reaching it means the swap
    did not install — and a zero query vector would rank by nothing at all
    while every downstream file still named the substitute.
    """

    #: Nominal, not real: pyarrow needs a fixed width for the vector column and
    #: nothing ever reads it back. Small on purpose — 500 samples of placeholder
    #: vectors at the real 1024 would be pure disk.
    dimension = 8
    model_name = "<none: ranking replaced by native_retriever>"
    model_type = "unused"
    supports_query_prompt = False

    def encode(self, texts, is_query: bool = False):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), self.dimension), dtype=np.float32)

    def encode_documents(self, documents):
        return self.encode(documents, is_query=False)

    def encode_query(self, queries):
        raise RuntimeError(
            "simplemem: the query encoder ran under a native_retriever swap. "
            "VectorStore.semantic_search should have been replaced by "
            "native_retrieval.install before any query — reaching this means it "
            "was not, and the run would have ranked against placeholder vectors."
        )

    def encode_single(self, text, is_query: bool = False):
        raise RuntimeError(
            "simplemem: the query encoder ran under a native_retriever swap. "
            "VectorStore.semantic_search should have been replaced by "
            "native_retrieval.install before any query — reaching this means it "
            "was not, and the run would have ranked against placeholder vectors."
        )


#: sentence-transformers' own default batch, which upstream never overrides
#: (core/utils/embedding.py's ``_encode_standard`` passes no ``batch_size``).
#: The ceiling below, never raised past it: this cap exists to make a run
#: possible, not to retune one that already works.
UPSTREAM_ENCODE_BATCH = 32

#: Ceiling on ``batch x maxlen^2`` — the attention working set of ONE forward
#: pass of SimpleMem's own encoder, in chars^2 — above which the batch is cut.
#:
#: WHY A BUDGET AND NOT A FIXED BATCH. Upstream sends no batch size, so
#: sentence-transformers uses 32; because it sorts by length first, batch 0 is
#: the 32 LONGEST entries padded to the longest of them, and that single batch
#: sets the peak for the whole sample. Attention goes as ``batch x maxlen^2``,
#: so the peak tracks the store's TAIL, not its size or its mean. Measured over
#: all 353 longmemeval_s stores, on the field actually embedded:
#:
#:     median store   maxlen   628 chars ->  32 x   628^2 =  12.6e6
#:     77eafa52       maxlen 2,112 chars ->  32 x 2,112^2 = 142.7e6
#:     5e1b23de       maxlen 2,543 chars ->  32 x 2,543^2 = 207.0e6
#:     9ea5eabc       maxlen 3,349 chars ->  32 x 3,349^2 = 358.9e6   <- OOM
#:
#: 9ea5eabc is the worst of the 353 by 1.7x and ~28x the median. At 32 it took
#: 14.68 GiB in the retrieve process against the 14.85 GiB the memory server
#: leaves on the card, and died 326 MiB short — deterministically, every run,
#: while every other sample fitted. A FIXED batch would have cut all 353 to fix
#: one; the budget cuts only the passes that exceed it, which is what "only the
#: failure case" means when the failure is a property of the data.
#:
#: 90e6 is set from that table: it is the smallest ceiling that quarters
#: 9ea5eabc's pass (batch 32 -> 8, ~14.7 GiB -> ~5 GiB, clear of the shortfall)
#: while leaving every store with maxlen <= 1,677 chars at upstream's 32. That
#: is 347 of the 353 measured; the other six are cut to 31, 23, 21, 20, 13 and 8
#: (3f1e9474, 2311e44b_abs, 7024f17c, 77eafa52, 5e1b23de, 9ea5eabc). It is also
#: a uniform PEAK bound: no pass, on any sample, can now cost more than a
#: typical one does today.
#:
#: The GPU fraction cannot absorb this instead. At CTX 32768 the memory LLM needs
#: ~3 GiB of KV to admit one request, which is what util 0.8 leaves after its
#: 60.01 GiB of weights (scripts/native_grid.sh's budget block), so
#: there is nothing to give back — and the window cannot shrink either, because
#: naive_query.yaml's ``max_tokens: null`` makes it the output cap.
#:
#: NOT A DEVIATION IN WHAT IS EMBEDDED. Same model, same texts, same order; only
#: how many share a padded forward pass changes, and padded positions are masked
#: out of attention. Vectors are identical up to float reduction order, which
#: cannot reorder a cosine ranking except on an exact tie. Nor is it a published
#: setting: upstream names no value and the paper names none, so this is an
#: infrastructure default in exactly the sense ``enable_parallel_retrieval`` is.
#:
#: NOT A CONFIG KEY, deliberately. ``embedder`` is fingerprinted into every
#: store's ``config_sha256``, and a knob there would mark all six simplemem
#: stores stale over a change that alters no saved byte — the vectors live in a
#: per-process LanceDB that ``save_memory`` never writes. The usual
#: caveat cuts the other way for once: the fingerprint covers the config, not the code.
#:
#: Override with EVALGRID_SIMPLEMEM_ATTN_BUDGET on a card with more room; 0 or
#: less disables the cap entirely and restores upstream's unbounded default.
ENCODE_ATTENTION_BUDGET = float(
    os.environ.get("EVALGRID_SIMPLEMEM_ATTN_BUDGET", 90e6))


class _BoundedEncoder:
    """A ``SentenceTransformer`` whose forward passes are memory-bounded.

    Wraps the MODEL rather than overriding ``EmbeddingModel._encode_standard``,
    so it holds for every path upstream takes into ``encode`` — the standard one
    and the Qwen3 query-prompt one — without this file restating a vendored
    method body that would then have to be kept in step with it.

    Every call that stays under the budget is passed through with upstream's
    batch, so the wrapper is INERT for all but the long-tailed stores: same
    batching, same vectors, same ranking, byte for byte.
    """

    def __init__(self, model, budget: float):
        self._model = model
        self._budget = budget

    def batch_size_for(self, texts) -> int:
        """Largest batch whose padded pass stays under the budget.

        Public because it is the whole contract, and it is checkable without a
        GPU: run it over ``entries.jsonl`` for every store in a cell and it must
        return ``UPSTREAM_ENCODE_BATCH`` for all but the long-tailed few.
        """
        if isinstance(texts, str):
            texts = [texts]
        try:
            longest = max((len(t) for t in texts), default=0)
        except TypeError:      # not a sequence of str — let upstream deal with it
            return UPSTREAM_ENCODE_BATCH
        if longest <= 0:
            return UPSTREAM_ENCODE_BATCH
        fits = int(self._budget // (longest * longest))
        # Floor of 1: a single text over budget still has to be embedded, and
        # one is the smallest pass there is. Dropping it instead would be a
        # silent hole in the store.
        return max(1, min(UPSTREAM_ENCODE_BATCH, fits))

    def encode(self, *args, **kwargs):
        # A caller that names its own batch size has a reason to, and keeps it.
        if "batch_size" not in kwargs:
            texts = args[0] if args else kwargs.get("sentences")
            kwargs["batch_size"] = self.batch_size_for(texts)
        return self._model.encode(*args, **kwargs)

    def __getattr__(self, name):
        # Everything else — get_sentence_embedding_dimension, tokenizer, device,
        # max_seq_length — reaches the real model untouched. Only called when
        # normal lookup fails, so _model and _budget resolve on self.
        return getattr(self._model, name)


def _bounded_encoder(original):
    """``original``, with its sentence-transformers model memory-bounded."""

    def build(*args, **kwargs):
        encoder = original(*args, **kwargs)
        model = getattr(encoder, "model", None)
        if model is not None and ENCODE_ATTENTION_BUDGET > 0:
            encoder.model = _BoundedEncoder(model, ENCODE_ATTENTION_BUDGET)
        return encoder

    return build


class SimpleMemBase(MemorySystem):
    """Shared SimpleMem plumbing. Subclasses implement ``ingest`` and ``answer_messages``.

    Args:
        config: the parsed evalgrid/configs/simplemem/<variant>.yaml.
        generator: answers queries. SimpleMem's own ``AnswerGenerator`` answers
            with the memory model; this repo separates the two so every system
            is compared under one generator, as LightMem and A-Mem are.
    """

    supports_persistence = True

    def __init__(self, config: dict, generator: GeneratorModel):
        self.config = config
        self.generator = generator

        # --- ingest policy -------------------------------------------------
        self.ingest_policy = str(config.get("ingest_policy", "session"))
        if self.ingest_policy not in ("session", "window"):
            raise ValueError(
                f"ingest_policy must be 'session' or 'window', got {self.ingest_policy!r}"
            )
        self.window_size = int(config.get("window_size", 40))
        self.overlap_size = int(config.get("overlap_size", 2))
        if self.overlap_size >= self.window_size:
            raise ValueError(
                f"overlap_size ({self.overlap_size}) must be smaller than window_size "
                f"({self.window_size}); step_size would be clamped to 1 and ingest "
                f"would re-extract nearly every window."
            )

        # --- retrieval (SimpleMem's native path only) ----------------------
        retrieval = config.get("retrieval") or {}
        self.semantic_top_k = int(retrieval.get("semantic_top_k", 25))
        self.keyword_top_k = int(retrieval.get("keyword_top_k", 5))
        self.structured_top_k = int(retrieval.get("structured_top_k", 5))
        # HARNESS state, set by SimpleMemPlugin.build from the flag
        # native_retrieval.declare put on the plugin: true when this run
        # replaces semantic_search with a substitute ranker. Two things below
        # depend on it — the legs guard, and _new_system not loading an encoder
        # whose vectors nothing would read.
        self.native_swap = bool(config.get("native_swap", False))
        # OURS, not upstream's: which of the tri-path union's legs run at all.
        # Parsed (and validated) here so a bad value fails at construction rather
        # than on the first question of a 500-sample run.
        self.legs = parse_legs(config)
        if "semantic" not in self.legs and self.native_swap:
            # native_retrieval swaps the DENSE leg's ranking function, and this
            # config has turned that leg off — so the swap would be configured,
            # named in the run directory, and never once called. Loud here rather
            # than a table row reporting a retriever it never used.
            raise ValueError(
                "simplemem: native_retriever is set but retrieval.legs excludes 'semantic'. "
                "The swap replaces the semantic leg's ranking function, so with that leg "
                "disabled it would never run. Either add 'semantic' to legs, or drop the "
                "native_retriever block."
            )
        self.enable_planning = bool(retrieval.get("enable_planning", True))
        self.enable_reflection = bool(retrieval.get("enable_reflection", True))
        self.max_reflection_rounds = int(retrieval.get("max_reflection_rounds", 2))
        self.enable_parallel_retrieval = bool(retrieval.get("enable_parallel_retrieval", False))
        self.max_retrieval_workers = int(retrieval.get("max_retrieval_workers", 3))

        # --- storage & models ----------------------------------------------
        self.embedder_model = (config.get("embedder") or {}).get(
            "model", "Qwen/Qwen3-Embedding-0.6B")
        self.db_path = str(config.get("db_path", "lancedb_data"))
        self.table_name = str(config.get("table_name", "memory_entries"))
        self.use_json_format = bool(config.get("use_json_format", False))
        self.max_tokens = int(config.get("answer_max_tokens", 2000))
        self.silence_upstream_stdout = bool(config.get("silence_upstream_stdout", True))

        self._llm = self._new_llm()
        self._sys = None                      # SimpleMemSystem, built once and reused
        self._entries: list = []              # ordered MemoryEntry list (the store)
        self._provenance: dict[str, dict] = {}
        self._pending: list[tuple] = []       # (Dialogue, provenance) awaiting a window
        self._next_dialogue_id = 1
        self.traces: list[dict] = []
        self.empty_windows = 0                # windows whose extraction returned nothing

        self.reset()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _new_llm(self):
        from evalgrid.systems.simplemem.controller import OpenAICompatLLMClient

        llm = self.config.get("memory_llm") or {}
        if not llm.get("model"):
            raise ValueError(
                "simplemem config has no memory_llm.model. It is injected by "
                "SimpleMemPlugin.apply_mem_model from --mem-model; if you built this "
                "system directly, set it yourself."
            )
        return OpenAICompatLLMClient(
            model=llm["model"],
            base_url=llm.get("base_url"),
            api_key=llm.get("api_key"),
            max_tokens=llm.get("max_tokens"),
            json_mode=llm.get("json_mode", "auto"),
            # Absent -> None -> each call site keeps its own temperature, which is
            # upstream's behaviour. The config sets 0 (paper Table 7).
            temperature=llm.get("temperature"),
        )

    def _new_system(self):
        # Imported here, not at module scope: SimpleMem pulls in lancedb,
        # pyarrow, dateparser and sentence-transformers, which must not be a
        # prerequisite for merely listing the registry or running LightMem.
        from simplemem.core.settings import settings
        from simplemem.text.system import SimpleMemSystem

        # ADAPTER: settings is a singleton whose __getattr__ falls back to
        # config.py / env / defaults, and only fires when normal attribute
        # lookup fails — so assigning an instance attribute pins the value for
        # this process without touching os.environ or shipping a config.py.
        # These two are read deep inside the vendored components (EmbeddingModel
        # at construction, USE_JSON_FORMAT at every call site) and cannot be
        # passed as arguments.
        settings.EMBEDDING_MODEL = self.embedder_model
        settings.USE_JSON_FORMAT = self.use_json_format

        llm = self.config.get("memory_llm") or {}
        with self._captured_stdout(), self._encoder_for_this_run():
            system = SimpleMemSystem(
                # The LLMClient this builds is replaced below. Its endpoint is
                # passed anyway so its constructor cannot trip on a machine with
                # no OPENAI_API_KEY, and so anything that reads it while
                # debugging names the endpoint the run actually used.
                api_key=llm.get("api_key") or os.getenv("OPENAI_API_KEY") or "token",
                model=llm.get("model"),
                base_url=llm.get("base_url"),
                db_path=self._process_db_path(),
                table_name=self.table_name,
                clear_db=True,
                enable_thinking=False,
                use_streaming=False,
                enable_planning=self.enable_planning,
                enable_reflection=self.enable_reflection,
                max_reflection_rounds=self.max_reflection_rounds,
                # Parallel ingest is upstream's default (16 workers). It is off
                # here for three reasons: every worker reads the same stale
                # `previous_entries`, so the anti-duplication hint is vacuous;
                # window completion order — and therefore store order — becomes
                # nondeterministic; and one sample would open 16 concurrent
                # connections to the vLLM endpoint while N ingest shards are
                # already running. We drive windows ourselves anyway.
                enable_parallel_processing=False,
                max_parallel_workers=1,
                enable_parallel_retrieval=self.enable_parallel_retrieval,
                max_retrieval_workers=self.max_retrieval_workers,
            )

        # ADAPTER: route every LLM call through our metered client. All four
        # references must be reassigned — SimpleMemSystem hands its client to
        # each component at construction, and each keeps its own reference.
        system.llm_client = self._llm
        system.memory_builder.llm_client = self._llm
        system.hybrid_retriever.llm_client = self._llm
        system.answer_generator.llm_client = self._llm

        # ADAPTER: MemoryBuilder takes window_size as an argument but reads
        # OVERLAP_SIZE off the global settings and derives step_size from it, so
        # SimpleMemSystem cannot forward either. Set all three together — they
        # must stay consistent, and step_size is what actually advances the
        # window.
        builder = system.memory_builder
        builder.window_size = self.window_size
        builder.overlap_size = self.overlap_size
        builder.step_size = max(1, self.window_size - self.overlap_size)

        # ADAPTER: same story for the three top-k values — HybridRetriever
        # accepts them, SimpleMemSystem does not forward them.
        retriever = system.hybrid_retriever
        retriever.semantic_top_k = self.semantic_top_k
        retriever.keyword_top_k = self.keyword_top_k
        retriever.structured_top_k = self.structured_top_k

        self._apply_legs(retriever)
        return system

    def _apply_legs(self, retriever) -> None:
        """ADAPTER: disable the union's legs that ``retrieval.legs`` leaves out.

        Each leg is one method on HybridRetriever, so switching one off is one
        instance rebinding and everything around it — planning, the merge
        priority, reflection — runs exactly as published on whatever the
        remaining legs return. Rebinding rather than setting the leg's top-k to
        0, because ``limit(0)`` is LanceDB behaviour we would be trusting
        blindly; returning ``[]`` is ours and is unambiguous.

        Applied once, at construction: legs are a property of the config, not of
        the sample, unlike the per-sample ranker native_retrieval installs.

        Two interactions worth knowing, neither of which this method decides for
        you:

        * ``enable_planning: false`` makes ``retrieve()`` return
          ``_semantic_search(query)`` and nothing else, bypassing the other two
          legs entirely — so a keyword-only or structured-only run REQUIRES
          planning left on.
        * with the semantic leg off, planning still spends its 2-3 LLM calls
          generating sub-queries that now retrieve nothing. That is a real cost
          and it is left visible rather than silently switched off, because
          "the union minus its dense leg" and "the union minus its dense leg
          minus planning" are different ablations.
        """
        if "semantic" not in self.legs:
            retriever._semantic_search = lambda query: []
        if "keyword" not in self.legs:
            retriever._keyword_search = lambda query, query_analysis: []
        if "structured" not in self.legs:
            retriever._structured_search = lambda query_analysis: []
        if not (self.legs & {"keyword", "structured"}):
            # _analyze_query exists ONLY to feed those two legs (it is its sole
            # consumer in HybridRetriever._retrieve_with_planning), so with both
            # off it is a paid LLM call whose result nothing reads.
            retriever._analyze_query = lambda query: {}

    def _process_db_path(self) -> str:
        """LanceDB directory for THIS process.

        Each parallel ingest shard needs its own on-disk database: they all run
        the same table name and each ``reset()`` drops and recreates it, so two
        processes sharing a directory would delete each other's sample
        mid-ingest. PID makes it unique even for non-sharded parallel runs; all
        of them nest under the configured base dir (lancedb_data/<pid>/).
        """
        return os.path.join(self.db_path.rstrip("/"), str(os.getpid()))

    def reset(self) -> None:
        """Clear all memory before each sample.

        The SimpleMemSystem itself is built once and reused: rebuilding it per
        sample would reload the sentence-transformers encoder 500 times over
        LongMemEval-S. What is per-sample is the LanceDB table and the builder's
        buffers — and the buffers matter. ``vector_store.clear()`` does not
        touch ``MemoryBuilder``, so a harness that only clears the store carries
        ``previous_entries`` (the next window's dedup hint) and
        ``processed_count`` across sample boundaries. Upstream's own LoCoMo
        driver has that leak; this does not.
        """
        if self._sys is None:
            self._sys = self._new_system()
        else:
            with self._captured_stdout():
                self._sys.vector_store.clear()

        builder = self._sys.memory_builder
        builder.dialogue_buffer = []
        builder.previous_entries = []
        builder.processed_count = 0

        self._entries = []
        self._provenance = {}
        self._pending = []
        self._next_dialogue_id = 1
        self.traces = []
        self.empty_windows = 0
        # Baseline, not a reset: the controller outlives every sample, and
        # zeroing a counter on a shared object would lie to anything else
        # reading it. meta.json reports the delta.
        self._counters_at_reset = {
            name: getattr(self._llm, name, 0)
            for name in ("truncated_calls", "failed_calls", "json_failures")
        }
        self._install_trace()

    def _since_reset(self, name: str) -> int:
        """A controller counter's delta over this sample."""
        total = getattr(self._llm, name, 0)
        return max(0, total - getattr(self, "_counters_at_reset", {}).get(name, 0))

    def _truncated_this_sample(self) -> int:
        """Extraction calls stopped at ``memory_llm.max_tokens`` since ``reset()``."""
        return self._since_reset("truncated_calls")

    # ------------------------------------------------------------------
    # Trajectory recording (evalgrid/lib/trace.py) — L1
    # ------------------------------------------------------------------

    def install_recorder(self, recorder) -> None:
        super().install_recorder(recorder)
        self._install_trace()

    def _install_trace(self) -> None:
        """Log every memory-LLM call SimpleMem makes.

        ``chat_completion`` is the controller's whole interface, so this covers
        both window extraction at ingest and — the part that dominates for this
        system — the 5-8 planning/analysis/reflection calls per question at
        answer time. Their `phase` field separates the two.
        """
        record_calls(self._llm, "chat_completion",
                     recorder=self.recorder, step="simplemem.chat_completion")

    # ------------------------------------------------------------------
    # Ingest helpers (subclasses call these)
    # ------------------------------------------------------------------

    def make_dialogue(self, speaker: str, content: str, *,
                      timestamp: Optional[str] = None,
                      turn_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      has_answer: bool = False) -> tuple:
        """Build one upstream ``Dialogue`` plus the provenance we keep beside it.

        ``dialogue_id`` is a monotonic int over the whole sample, as upstream's
        own ``add_dialogue`` assigns it. It is never persisted — it only orders
        the window — so it does not need to survive save/load.

        Provenance rides alongside rather than on the Dialogue because
        ``Dialogue`` is a pydantic model with a fixed field set: an extra
        ``turn_id=`` would be dropped or rejected depending on its config, and
        editing the vendored model is out of bounds.
        """
        from simplemem.core.models.memory_entry import Dialogue

        dialogue = Dialogue(
            dialogue_id=self._next_dialogue_id,
            speaker=speaker,
            content=content,
            timestamp=timestamp,
        )
        self._next_dialogue_id += 1
        return dialogue, {
            "turn_id": turn_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "has_answer": bool(has_answer),
        }

    def ingest_dialogues(self, dialogues: list[tuple], *, is_last: bool = False) -> None:
        """Feed one session's dialogues through the window loop.

        ``ingest_policy`` decides where window boundaries fall:

        session — flush at every session boundary, so a window never straddles
            two sessions. One extraction call per session (LongMemEval-S
            sessions are shorter than any sane window), which keeps each entry's
            provenance inside one session and its timestamps coherent. Costs
            ~50 calls per sample instead of ~13.
        window — upstream's own behaviour: dialogues accumulate across sessions
            and a window is extracted whenever ``window_size`` are available,
            boundaries wherever they fall. Cheaper, closer to the paper's LoCoMo
            setup, and mixes unrelated sessions into one extraction call.
        """
        if self.ingest_policy == "session":
            self._extract_all(dialogues)
        else:
            self._pending.extend(dialogues)
            self._drain()
        if is_last:
            self._finalize()

    def _drain(self) -> None:
        """Extract every full window currently buffered.

        Mirrors ``MemoryBuilder.process_window``: take the first ``window_size``
        dialogues, advance by ``step_size``, so consecutive windows share
        ``overlap_size`` dialogues and a fact spanning a boundary is seen whole
        at least once.
        """
        while len(self._pending) >= self.window_size:
            self._extract_window(self._pending[:self.window_size])
            self._pending = self._pending[max(1, self.window_size - self.overlap_size):]

    def _extract_all(self, dialogues: list[tuple]) -> None:
        """Extract a self-contained group (one session) as whole windows + tail.

        The trailing partial window is extracted too — that is
        ``process_remaining``, and it is what stops the last few turns of every
        session from being dropped.
        """
        buffer = list(dialogues)
        step = max(1, self.window_size - self.overlap_size)
        while len(buffer) >= self.window_size:
            self._extract_window(buffer[:self.window_size])
            buffer = buffer[step:]
        if buffer:
            self._extract_window(buffer)

    def _extract_window(self, window: list[tuple]) -> None:
        """One extraction call over one window; record what it produced.

        Reproduces ``MemoryBuilder.process_window`` step for step — the same
        ``_generate_memory_entries`` call (which reads ``previous_entries`` for
        its dedup hint), the same ``add_entries``, the same ``previous_entries``
        / ``processed_count`` update — with the window's provenance attached to
        each entry on the way past. See the module docstring.
        """
        if not window:
            return
        builder = self._sys.memory_builder
        dialogues = [d for d, _ in window]

        # L2. ``chat_completion`` is the controller's whole interface, shared
        # with the 5-8 planning/analysis/reflection calls the answer path makes,
        # so an L1 record on its own can only be stage "llm" — "an LLM call
        # nobody claimed". This names the ingest one. It also carries the
        # window's dialogue count and the entry count it produced, which is the
        # per-window yield that empty_windows only summarises.
        with self.recorder.step("extract", "simplemem.extract_window",
                                usage=self._llm.token_usage,
                                meta={"dialogues": len(dialogues)}) as step:
            with self._captured_stdout():
                entries = builder._generate_memory_entries(dialogues) or []
            step.meta(entries=len(entries))

        if not entries:
            # Upstream returns [] after three JSON-parse failures. Counted, not
            # raised: one bad window out of fifty is a degraded sample, not a
            # failed one. _finalize raises if the whole sample came back empty.
            self.empty_windows += 1
            return

        with self._captured_stdout():
            self._sys.vector_store.add_entries(entries)

        builder.previous_entries = entries
        builder.processed_count += len(dialogues)

        provenance = self._window_provenance(window)
        for entry in entries:
            self._entries.append(entry)
            self._provenance[entry.entry_id] = dict(provenance)

    @staticmethod
    def _window_provenance(window: list[tuple]) -> dict:
        """Collapse a window's per-turn provenance into one record per entry.

        ``has_answer`` is the OR over the window: the extraction call saw every
        turn in it, so any entry it produced may encode the evidence turn, and
        there is no per-entry signal to narrow it with. This over-marks — see
        the ADAPTER note in the module docstring.

        ``timestamp`` is the FIRST turn's, i.e. the time the window opens. Under
        the session policy every turn in a window shares one session timestamp,
        so the choice only bites under the window policy, where a window that
        straddles sessions is stamped with the earlier one.
        """
        provs = [p for _, p in window]
        session_ids: list[str] = []
        for p in provs:
            sid = p.get("session_id")
            if sid and sid not in session_ids:
                session_ids.append(sid)
        return {
            "source_turn_ids": [p["turn_id"] for p in provs if p.get("turn_id")],
            "session_ids": session_ids,
            "has_answer": any(p.get("has_answer") for p in provs),
            "timestamp": provs[0].get("timestamp") if provs else None,
        }

    def _finalize(self) -> None:
        """Flush the tail of the sample and check the store is not empty."""
        if self._pending:
            self._extract_window(self._pending)
            self._pending = []

        if not self._entries:
            raise RuntimeError(
                f"SimpleMem extracted no memory at all for this sample: "
                f"{self.empty_windows} window(s) returned no entries. Upstream drops a "
                f"window after three JSON-parse failures, so this is usually the memory "
                f"model emitting prose or <think> blocks instead of the JSON array the "
                f"extraction prompt asks for — check the endpoint, or set "
                f"memory_llm.json_mode / use_json_format in the config."
            )
        if self.empty_windows:
            print(f"[simplemem] WARNING: {self.empty_windows} window(s) produced no "
                  f"entries and were dropped from this sample's memory.")

    @contextlib.contextmanager
    def _encoder_for_this_run(self):
        """Substitute the encoder this run needs for the run's construction.

        ADAPTER: ``SimpleMemSystem.__init__`` builds its own ``EmbeddingModel()``
        and hands it to the ``VectorStore``, so there is no argument to pass and
        no instance to rebind afterwards — by the time the system exists, the
        encoder is already on the GPU. Patching the name in the module that
        constructs it is therefore the only seam, and it is the same kind of
        module-attribute patch native_retrieval uses for HyperMem's free
        function. vendor/ stays verbatim either way.

        Two substitutions, and which one depends on whether this run's ranking
        is SimpleMem's own:

        swapped     ``_UnusedEncoder``. The substitute ranks and
                    ``semantic_search`` is replaced wholesale, so every vector
                    the encoder would compute is written and never read. Skip
                    the model entirely.
        unswapped   ``_BoundedEncoder``. The vectors ARE the ranking here, so
                    they have to be computed — but bounded, because upstream's
                    unbounded batch is what took 14.68 GiB and OOM'd on the one
                    store with a long enough tail. Inert on every store that
                    already fits: same batch, same vectors. See
                    ``ENCODE_ATTENTION_BUDGET``.

        Scoped to the ``with`` block and restored in a finally, so a failed
        construction cannot leave either one installed for whatever the process
        builds next.
        """
        import simplemem.text.system as upstream

        original = upstream.EmbeddingModel
        upstream.EmbeddingModel = (
            _UnusedEncoder if self.native_swap else _bounded_encoder(original))
        try:
            yield
        finally:
            upstream.EmbeddingModel = original

    @contextlib.contextmanager
    def _captured_stdout(self):
        """Swallow upstream's per-window and per-search debug prints.

        SimpleMem prints a banner per construction, a line per window, and
        roughly ten lines per question (planning, every sub-query, every search,
        every reflection round). Over a 500-sample run that buries anything
        worth reading. Dropped windows are counted structurally in
        ``_extract_window`` instead of by scraping these lines, so nothing is
        lost.
        """
        if not self.silence_upstream_stdout:
            yield
            return
        with contextlib.redirect_stdout(io.StringIO()):
            yield

    # ------------------------------------------------------------------
    # Retrieval + answering (SimpleMem's native path)
    # ------------------------------------------------------------------

    def retrieve_entries(self, question: str) -> list:
        """SimpleMem's own query loop: plan -> tri-path union -> reflect.

        This is 5-8 LLM calls per question (§3.3: information-requirements
        analysis, targeted-query generation, query analysis, then up to two
        rounds of completeness check + follow-up query generation), all billed
        to the memory layer. It is also the part of SimpleMem that the staged
        pipeline replaces with a single-shot retriever — which is the honest
        comparison of memory representations, but is not what the paper's
        Table 2 measures. Kept here so the native path stays runnable.

        Note the union is unbounded: there is no final top-k anywhere in
        upstream's merge, so a hard question can return most of the store.
        """
        with self._captured_stdout():
            return self._sys.hybrid_retriever.retrieve(question)

    #: Returned instead of calling the generator when native retrieval came back
    #: empty, for a subclass whose reference harness short-circuits that case
    #: (upstream's ``AnswerGenerator.generate_answer`` does). None = always
    #: generate, which is what the shared-prompt path wants: it renders an
    #: explicit "(no relevant memories found)" and lets the model abstain.
    empty_context_answer: Optional[str] = None

    def retrieve_trace(self, query: Query) -> dict:
        """The plan -> tri-path union -> reflect loop. 5-8 LLM calls, no generation.

        This is overwhelmingly the expensive half of SimpleMem's query time,
        which is why saving it and replaying the answer under several budgets is
        worth more here than for any other system.
        """
        entries = self.retrieve_entries(query.question)
        # ADAPTER: upstream's own context rendering ("[Context i] Content / Time
        # / Location / Persons / Related Entities / Topic"), reused so the
        # native path shows the generator what SimpleMem would have shown it.
        # _format_contexts is pure — no store access, no LLM call.
        context = self._sys.answer_generator._format_contexts(entries)
        # Trace keys are the ones benchmarks/runner.py::_add_trace_fields reads.
        trace = {
            "query_id": query.query_id,
            "step2_retrieved_context": context,
            "retrieved_unit_ids": [e.entry_id for e in entries],
            # Rendered one entry at a time, through the same pure formatter, so
            # a budget can keep a prefix of the union and still show the
            # generator exactly what SimpleMem would have.
            "retrieved_units": [self._sys.answer_generator._format_contexts([e])
                                for e in entries],
            "empty": not entries,
        }
        self.traces.append(trace)
        return trace

    def answer(self, query: Query) -> str:
        """Native retrieval -> the subclass's prompt -> our generator."""
        return self.answer_from_trace(query, self.retrieve_trace(query))

    def answer_from_trace(self, query: Query, trace: dict) -> str:
        units = trace.get("retrieved_units")
        empty = trace.get("empty", not (trace.get("retrieved_unit_ids") or []))
        if empty and self.empty_context_answer is not None:
            return self.empty_context_answer
        # Re-joined rather than reusing step2_retrieved_context, so a budgeted
        # replay renders only the units that survived the cut.
        context = "\n".join(units) if units is not None else \
            (trace.get("step2_retrieved_context") or "")
        messages = self.answer_messages(query, context)
        answer = asyncio.run(self.generator.generate(messages, max_tokens=self.max_tokens))
        return self.postprocess_answer(answer)

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        raise NotImplementedError

    def postprocess_answer(self, answer: str) -> str:
        """Hook for a prompt whose reply is not the answer itself (LoCoMo's JSON)."""
        return answer

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    #
    # SimpleMem's state is the entry set: the LanceDB row is the entry plus an
    # embedding of its lossless_restatement, so re-embedding on load rebuilds
    # the store exactly. We save JSONL rather than copying the LanceDB
    # directory so the store is readable, diffable, usable by iter_units without
    # importing SimpleMem, and outlives the schema that wrote it.
    #
    # We save our own ordered list, not vector_store.get_all_entries(): LanceDB
    # makes no ordering guarantee, and corpus.jsonl order is what the
    # full-context baseline feeds the generator. Insertion order is ingest
    # order, which is chronological.
    #
    # entry_id is a uuid4 minted at parse time, so it is stable only once
    # written. Re-ingesting a sample mints new ids and invalidates every run
    # file that ranked the old ones — persist the store, don't rebuild it.

    def save_memory(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "entries.jsonl"), "w") as f:
            for entry in self._entries:
                prov = self._provenance.get(entry.entry_id, {})
                f.write(json.dumps({
                    "entry_id": entry.entry_id,
                    "lossless_restatement": entry.lossless_restatement,
                    "keywords": list(entry.keywords),
                    # SimpleMem's own field: the EVENT time the extractor read
                    # out of the dialogue, often null and often not the time the
                    # turn was spoken. Kept, but not what the corpus is stamped
                    # with — see window_timestamp.
                    "event_timestamp": entry.timestamp,
                    "location": entry.location,
                    "persons": list(entry.persons),
                    "entities": list(entry.entities),
                    "topic": entry.topic,
                    # Ours: the conversation time of the window this entry came
                    # from, i.e. the same clock LightMem's time_stamp and the
                    # raw-turn corpus use.
                    "window_timestamp": prov.get("timestamp"),
                    "session_ids": prov.get("session_ids", []),
                    "source_turn_ids": prov.get("source_turn_ids", []),
                    "has_answer": bool(prov.get("has_answer", False)),
                }, ensure_ascii=False) + "\n")

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "entries": len(self._entries),
                "empty_windows": self.empty_windows,
                # An empty_window whose cause was the output cap rather than a
                # model that would not produce JSON. The distinction matters:
                # the first is fixed by raising max_tokens, the second is not.
                "truncated_calls": self._truncated_this_sample(),
                # The other cause of an empty_window: a completion that came
                # back whole but held no scrapeable JSON. Counted in the
                # controller's extract_json
                # and surfaced here, because empty_windows alone cannot say
                # whether a window held no facts or was simply unparseable.
                "json_failures": self._since_reset("json_failures"),
                # Calls that raised after exhausting their retries. token_usage
                # counts only calls that returned, so ingest_usage.json's
                # llm_calls cannot show these.
                "failed_calls": self._since_reset("failed_calls"),
                "ingest_policy": self.ingest_policy,
                "window_size": self.window_size,
                "overlap_size": self.overlap_size,
            }, f, indent=2)

    def load_memory(self, path: str) -> None:
        from simplemem.core.models.memory_entry import MemoryEntry

        records = []
        with open(os.path.join(path, "entries.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return

        entries = []
        for rec in records:
            entry = MemoryEntry(
                entry_id=rec["entry_id"],
                lossless_restatement=rec["lossless_restatement"],
                keywords=rec.get("keywords") or [],
                timestamp=rec.get("event_timestamp"),
                location=rec.get("location"),
                persons=rec.get("persons") or [],
                entities=rec.get("entities") or [],
                topic=rec.get("topic"),
            )
            entries.append(entry)
            self._entries.append(entry)
            self._provenance[entry.entry_id] = {
                "timestamp": rec.get("window_timestamp"),
                "session_ids": rec.get("session_ids", []),
                "source_turn_ids": rec.get("source_turn_ids", []),
                "has_answer": bool(rec.get("has_answer", False)),
            }

        # One batched insert: add_entries re-embeds every restatement in one
        # sentence-transformers pass and preserves entry_id, so the reloaded
        # store ranks identically to the live one. Under a native_retriever swap
        # that pass is the substitute's to make, not ours — _UnusedEncoder
        # returns placeholders and the rows go in for the keyword and structured
        # legs to read.
        with self._captured_stdout():
            self._sys.vector_store.add_entries(entries)

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------

    def layer_usage(self) -> dict:
        """Split tokens by layer.

        Memory layer: window extraction during ingest, and — the part that
        dominates for this system — the 5-8 planning, analysis and reflection
        calls SimpleMem makes per question at ANSWER time. The runner snapshots
        per phase, so those land under answer_tokens_by_layer rather than being
        mistaken for construction cost. SimpleMem's cost profile is inverted
        relative to LightMem and A-Mem (cheap to write, expensive to read), and
        this split is where that shows up.

        Generation layer: the final answer, from our generator.
        """
        return {
            "memory": TokenUsage(
                input_tokens=self._llm.token_usage.input_tokens,
                output_tokens=self._llm.token_usage.output_tokens,
                calls=self._llm.token_usage.calls,
            ),
            "generation": self.generator.token_usage,
        }
