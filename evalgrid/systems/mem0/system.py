"""
evalgrid/systems/mem0/system.py
-------------------------------
Mem0 (v0.1.118) behind the MemorySystem interface.

Upstream is wrapped, not rewritten: ``vendor/mem0/`` is verbatim mem0ai 0.1.118
and every LLM call, prompt and store operation is theirs. What this module adds
is the three things the pipeline needs and upstream does not provide — a
per-sample lifecycle, provenance that survives ingest, and a session clock —
plus the process isolation a sharded run needs.

The two behavioural changes are in ``_memory_class`` and each is marked
DEVIATION there. Both exist because a fact's payload is what becomes a corpus
unit, so anything upstream drops from a payload is a hole in the retrieval
dataset:

DEVIATION 1 — **the session clock.** ``_create_memory`` (vendor/mem0/memory/
main.py:851) hardcodes ``created_at = datetime.now(pytz.timezone("US/Pacific"))``,
the wall-clock time of the ingest job, and there is no ``timestamp=`` parameter
on 0.1.x's ``add()`` to override it with. Every LongMemEval temporal-reasoning
and knowledge-update question would then be asked against a store where all 300
facts share one timestamp — the afternoon the job ran. A caller-supplied
``created_at`` is honoured here instead.

DEVIATION 2 — **provenance across UPDATE.** ``_update_memory`` (main.py:908)
rebuilds the payload from the *new* call's metadata and restores only
``created_at``/``user_id``/``agent_id``/``run_id``/``actor_id``/``role`` from the
old one. Custom keys — which is where ``session_id``, ``turn_ids`` and
``has_answer`` live — are dropped, so a fact created in session 3 and UPDATE'd in
session 40 silently claims session 40 as its only source. In LongMemEval that is
exactly the ``knowledge-update`` question type, the one where UPDATE fires most,
so the qrels would be wrong precisely where the system is most interesting.
Provenance is unioned across the two writes here.

Note what is NOT changed: the extraction prompt, the update prompt, the two-call
pipeline, the per-fact ``limit=5`` neighbour search, the integer id remapping,
and the SQLite history log all run exactly as upstream wrote them.

**Ingest is lossy and order-dependent, by design of the system.** UPDATE
overwrites a fact's text and DELETE soft-deletes it, both decided by an LLM that
sees only the top-5 neighbours of each new fact, so re-ingesting the same
sessions in the same order does not reproduce a byte-identical store. The
snapshot written by ``save_memory`` is therefore the artifact — never re-derive
memory at retrieval time. The full mutation log travels with it in history.db.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import os
import re
import shutil
from datetime import datetime
from typing import Optional

from evalgrid.benchmarks.types import MemorySystem, Query, Session
from evalgrid.models.base import GeneratorModel, TokenUsage
from evalgrid.lib.trace import record_calls
from evalgrid.systems.base import shared_answer_prompt


# --- Call classification for the trajectory (see Mem0Base._install_trace) ----
#
# Both of mem0's per-add() calls go through one method, so the stage has to
# come from the arguments. The extraction call is the only one that sends a
# system prompt; the resolution call sends one user message. Anything else is a
# call this mapping does not know about, which must read as unnamed ("llm")
# rather than be forced into one of the two.

def _mem0_messages(*args, **kwargs) -> list:
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]
    return messages if isinstance(messages, list) else []


def _mem0_stage(*args, **kwargs) -> Optional[str]:
    messages = _mem0_messages(*args, **kwargs)
    if any((m or {}).get("role") == "system" for m in messages):
        return "extract"
    if len(messages) == 1:
        return "update"
    return None


def _extracted_nothing(choices: list) -> bool:
    """True if a fact-extraction completion carried no facts.

    Mirrors upstream's own reading of the reply (`Memory._add_to_vector_store`
    strips a ```json fence, `json.loads`, then `["facts"]`) rather than
    re-deriving it, so this counts exactly the replies upstream would have got
    nothing from. Anything unparseable is NOT counted: upstream also ends up
    with no facts there, but that is a JSON failure rather than an extractor
    that read the turn and found nothing worth keeping, and conflating the two
    overstated the failure count. A clipped reply is already counted as
    ``truncated_calls``.
    """
    if not choices:
        return False
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if not isinstance(content, str):
        return False
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return not (json.loads(text) or {}).get("facts")
    except Exception:
        return False


def _mem0_step(*args, **kwargs) -> Optional[str]:
    stage = _mem0_stage(*args, **kwargs)
    return {"extract": "mem0.extract_facts",
            "update": "mem0.update_memory"}.get(stage)


# Benchmark timestamp shapes, as in memorybank/hypermem's parsers. Mem0 stores
# `created_at` as an ISO string, so a benchmark stamp has to be parsed to be
# stored in the field upstream's own code reads.
_TS_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",        # LongMemEval: '2023/05/20 (Sat) 02:21'
    "%I:%M %p on %d %B, %Y",      # LoCoMo:      '1:56 pm on 8 May, 2023'
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)

#: Payload keys this adapter owns. They are what ``iter_units`` reads back, and
#: the reason ``_update_memory`` is overridden — upstream drops exactly these.
PROVENANCE_KEYS = ("session_id", "session_ids", "session_ts", "turn_ids", "has_answer")

#: The wall-clock line mem0 bakes into FACT_RETRIEVAL_PROMPT at import time.
#: Matched by pattern, not by recomputing datetime.now() — see _prompt_for_date.
_PROMPT_DATE_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.")


def to_iso(raw: Optional[str]) -> Optional[str]:
    """Parse a benchmark timestamp into ISO-8601, or None if it has none.

    Returning None rather than a fallback date is deliberate: a session with no
    timestamp should keep upstream's wall-clock behaviour and be visibly odd,
    not silently claim to have happened on some default day.
    """
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).isoformat()
    except ValueError:
        pass
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


def _merge_ids(old, new) -> list:
    """Union two id lists, order-preserving, old first."""
    out: list = []
    for seq in (old or [], new or []):
        if isinstance(seq, str):
            seq = [seq]
        for item in seq:
            if item and item not in out:
                out.append(item)
    return out


#: The payload keys upstream lifts out of the payload instead of folding them
#: under "metadata" (vendor/mem0/memory/main.py:724-732).
_PROMOTED_PAYLOAD_KEYS = ("user_id", "agent_id", "run_id", "actor_id", "role")
_CORE_PAYLOAD_KEYS = {"data", "hash", "created_at", "updated_at", "id",
                      *_PROMOTED_PAYLOAD_KEYS}


def _as_search_result(point: dict, unit_id: str, score: float) -> dict:
    """One raw store point in the shape ``Memory.search`` returns.

    A substituted retriever (evalgrid/systems/native_retrieval.py) ranks unit
    ids and then has to hand the rows back to code that was written against
    upstream's OUTPUT, not against the store: ``_render`` below reads
    ``item["memory"]`` and ``item["metadata"]["session_ts"]``, and both are
    upstream's names, not the payload's. In the store the text is
    ``payload["data"]`` and every custom key sits at the payload's top level.

    This used to return the payload itself with ``id``/``score`` added, so
    ``.get("memory")`` found nothing and ``.get("metadata")`` found nothing, and
    every swapped unit rendered as the EMPTY STRING — silently, since the ids
    beside it were right and nothing downstream of the ranking reads the text.
    The mapping below is upstream's own, from ``_search_vector_store``
    (vendor/mem0/memory/main.py:734-754): core fields by name, the promoted keys
    if present, everything else under "metadata" — and "metadata" omitted
    entirely when there is nothing left, exactly as upstream omits it.
    """
    payload = dict(point.get("payload") or {})
    item = {
        "id": str(unit_id),
        "memory": payload.get("data"),
        "hash": payload.get("hash"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "score": score,
    }
    for key in _PROMOTED_PAYLOAD_KEYS:
        if key in payload:
            item[key] = payload[key]
    extra = {k: v for k, v in payload.items() if k not in _CORE_PAYLOAD_KEYS}
    if extra:
        item["metadata"] = extra
    return item


def _memory_class():
    """Build (once) the ``Memory`` subclass carrying the two DEVIATIONs.

    Defined inside a function because importing it pulls in qdrant_client,
    openai, posthog and — through ``mem0/__init__.py`` — httpx and requests.
    None of that may be a prerequisite for listing the registry or running some
    other system.
    """
    global _MEMORY_CLASS
    if _MEMORY_CLASS is not None:
        return _MEMORY_CLASS

    import hashlib
    import uuid

    import pytz
    from mem0.memory import main as mem0_main
    from mem0.memory.main import Memory

    class ProvenanceMemory(Memory):
        """``Memory``, with the session clock and provenance union."""

        def _create_memory(self, data, existing_embeddings, metadata=None):
            """Upstream ``Memory._create_memory`` (vendor/mem0/memory/main.py:841).

            Reproduced line for line except for the ``created_at`` assignment,
            marked below. Copied rather than delegated because upstream sets
            that field unconditionally *before* the insert, so there is no
            wrapper position from which it can be corrected without a second
            write to the store.
            """
            logger.debug(f"Creating memory with {data=}")
            if data in existing_embeddings:
                embeddings = existing_embeddings[data]
            else:
                embeddings = self.embedding_model.embed(data, memory_action="add")
            memory_id = str(uuid.uuid4())
            metadata = metadata or {}
            metadata["data"] = data
            metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
            # DEVIATION 1: upstream is
            #     metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
            # which discards any caller value. A caller-supplied created_at (the
            # session's own time) wins; with none supplied the behaviour is
            # upstream's, wall clock and all.
            if not metadata.get("created_at"):
                metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

            self.vector_store.insert(
                vectors=[embeddings],
                ids=[memory_id],
                payloads=[metadata],
            )
            self.db.add_history(
                memory_id,
                None,
                data,
                "ADD",
                created_at=metadata.get("created_at"),
                actor_id=metadata.get("actor_id"),
                role=metadata.get("role"),
            )
            # Through the module attribute rather than the name imported into
            # this file, so _patch_telemetry's no-op applies to it too.
            mem0_main.capture_event(
                "mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
            return memory_id

        def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
            """Upstream's ``_update_memory``, given a pre-merged metadata dict.

            DEVIATION 2. Upstream's body is correct once its input is: it
            restores identity keys from the old payload and takes everything
            else from the caller. So rather than copy 50 lines, the *caller's*
            metadata is unioned with the old payload's provenance first and
            upstream then does the rest — including preserving ``created_at``,
            which is what keeps DEVIATION 1's session clock pointing at the
            session where the fact FIRST appeared.

            The union, not the old value and not the new one: an UPDATE'd fact
            genuinely derives from both sessions. ``has_answer`` is the OR (it
            marks a unit as derived from an evidence turn, and it now is);
            ``session_ids``/``turn_ids`` accumulate; ``session_ts`` tracks the
            LATEST write, because that is when the fact took its current value
            and is what the answer context should stamp it with. The original
            time is not lost — it is ``created_at``.
            """
            merged = dict(metadata or {})
            try:
                old = (self.vector_store.get(vector_id=memory_id).payload) or {}
            except Exception:  # upstream raises the same lookup below
                old = {}
            merged["session_ids"] = _merge_ids(
                old.get("session_ids") or old.get("session_id"),
                merged.get("session_ids") or merged.get("session_id"),
            )
            merged["turn_ids"] = _merge_ids(old.get("turn_ids"), merged.get("turn_ids"))
            merged["has_answer"] = bool(old.get("has_answer")) or bool(merged.get("has_answer"))
            return super()._update_memory(memory_id, data, existing_embeddings, merged)

    _MEMORY_CLASS = ProvenanceMemory
    return _MEMORY_CLASS


_MEMORY_CLASS = None
logger = __import__("logging").getLogger(__name__)


class Mem0Base(MemorySystem):
    """Shared Mem0 plumbing. Subclasses implement ``ingest``.

    Args:
        config: the parsed evalgrid/configs/mem0/<variant>.yaml.
        generator: answers queries. Mem0's memory LLM is separate and is
            configured under ``memory_llm``; it is used at ingest only, since
            0.1.x retrieval is a single dense search with no LLM in the loop.
    """

    supports_persistence = True

    def __init__(self, config: dict, generator: GeneratorModel):
        self.config = config
        self.generator = generator

        self.top_k = int(config.get("retrieve_k", 10))
        self.max_tokens = int(config.get("answer_max_tokens", 2000))
        self.infer = bool(config.get("infer", True))
        # Upstream's LongMemEval harness uses one user_id per question
        # (`longmemeval_{question_id}_{run_id}`). Here every sample gets its own
        # store, so the namespace is not what isolates samples — the store is.
        # mem0 requires one of user_id/agent_id/run_id on every call, so this
        # exists to satisfy that.
        self.user_id = str(config.get("user_id") or "memgrid_sample")
        # Which clock the fact-extraction prompt runs on. "session" points it at
        # the session being ingested (DEVIATION 3, see _set_extraction_clock);
        # "wall" leaves upstream's datetime.now(), i.e. the day the job ran.
        clock = str(config.get("fact_extraction_clock", "session"))
        if clock not in ("session", "wall"):
            raise ValueError(
                f"fact_extraction_clock must be 'session' or 'wall', got {clock!r}")
        self.fact_extraction_clock = clock
        self.session_extraction_clock = clock == "session"
        # One substituted prompt per distinct session date, not per add(): a
        # LongMemEval sample is ~50 sessions but 277 add() calls.
        self._prompt_cache: dict = {}
        self._extraction_dates: set = set()

        # Recorded into meta.json so a store says how it was decoded. Defaults
        # are mem0's own (vendor/mem0/configs/llms/base.py:19-22), applied here
        # rather than left null: an omitted key means the run used BaseLlmConfig's
        # value, and writing null would say only "unknown", which is what this is
        # meant to stop. mem0 0.1.x sends no seed, so this is the whole sampling
        # state — temperature 0.0 is greedy, not reproducible-to-the-token.
        _llm = self.config.get("memory_llm") or {}
        self.sampling = {
            "temperature": float(_llm.get("temperature", 0.1)),
            "top_p": float(_llm.get("top_p", 0.1)),
            "max_tokens": int(_llm.get("max_tokens", 2000)),
        }

        self._mem = None
        self._mem_usage = TokenUsage()
        self.events: dict[str, int] = {}
        self.traces: list[dict] = []
        # Two different silences, counted apart. See _record_usage / add():
        # empty_extractions is the extraction call returning {"facts": []} —
        # the turn pair contributed nothing and nothing downstream ran for it.
        # no_op_adds is an add() whose RESULT was empty, which also happens
        # when facts WERE extracted and the resolution call answered NONE to
        # every one of them. Until 2026-08-19 one counter named
        # "empty_extractions" carried both, so 167 of 58bf7951's 311 add()s
        # were reported as failed extractions when only 138 were
        # — a 21% overstatement of a degradation counter check_ingest
        # gates on.
        self.empty_extractions = 0
        self.no_op_adds = 0
        self.add_calls = 0
        # Completions the endpoint stopped at max_tokens. See _record_usage:
        # mem0 was the only system with no truncation signal at all, so a
        # clipped ADD/UPDATE/DELETE resolution — which costs real memory
        # operations, not just text — was invisible in meta.json. Split by
        # stage because the two calls fail differently: a clipped extraction
        # loses facts, a clipped resolution loses decisions ABOUT facts.
        self.truncated_calls = 0
        self.truncated_by_stage: dict[str, int] = {}

        self._store_dir = self._process_store_dir()
        self.reset()

    # ------------------------------------------------------------------
    # Process isolation
    # ------------------------------------------------------------------

    def _process_store_dir(self) -> str:
        """This process's private mem0 state directory, and the env that pins it.

        Three things mem0 keeps outside the vector store are shared-by-default
        and would collide between concurrent shards:

          * ``history_db_path`` defaults to a single ``~/.mem0/history.db``;
          * ``Memory.__init__`` opens a SECOND local Qdrant for telemetry at
            ``~/.mem0/migrations_qdrant``, and local Qdrant takes an exclusive
            lock on its folder, so the second process to start would die on a
            store it never asked for;
          * ``setup_config()`` writes ``~/.mem0/config.json`` at import.

        All three follow ``MEM0_DIR``, which mem0 reads at *import* time — hence
        setting it here, before ``_memory_class()`` imports anything. PID-keyed,
        as lightmem's qdrant path is, so a non-sharded parallel run is safe too.
        """
        base = str(self.config.get("store_dir") or "mem0_data").rstrip("/")
        path = os.path.join(base, str(os.getpid()))
        os.makedirs(path, exist_ok=True)
        os.environ["MEM0_DIR"] = os.path.abspath(os.path.join(path, "home"))
        if self.config.get("disable_telemetry", True):
            # Read at import by mem0.memory.telemetry; the no-op patch below
            # covers what this flag does not.
            os.environ["MEM0_TELEMETRY"] = "false"
        return path

    @property
    def history_db_path(self) -> str:
        return os.path.join(self._store_dir, "history.db")

    def _build_config(self):
        """Translate our YAML into a ``MemoryConfig``."""
        from mem0.configs.base import MemoryConfig

        llm = dict(self.config.get("memory_llm") or {})
        model = llm.pop("model", None)
        base_url = llm.pop("base_url", None)
        if not model:
            raise ValueError(
                "mem0 config has no memory_llm.model. It is injected by "
                "Mem0Plugin.apply_mem_model from --mem-model; if you built this "
                "system directly, set it yourself."
            )
        if os.environ.get("OPENROUTER_API_KEY"):
            # mem0/llms/openai.py:43 checks this env var BEFORE openai_base_url
            # and silently routes every memory call to OpenRouter — i.e. the
            # store would be built by a different model than --mem-model names
            # and keys the output dir, with nothing in the logs to say so.
            raise RuntimeError(
                "OPENROUTER_API_KEY is set. mem0's OpenAI client checks it before "
                "openai_base_url and would silently route memory construction to "
                "OpenRouter instead of --mem-base-url. Unset it for this run."
            )
        llm_config = {
            "model": model,
            # Upstream's own key name for "the OpenAI-compatible endpoint".
            "openai_base_url": base_url,
            # Token accounting: upstream calls this after every completion with
            # the raw response, which is the only place usage is visible.
            "response_callback": self._record_usage,
            **llm,
        }

        embedder = dict(self.config.get("embedder") or {})
        dims = embedder.get("embedding_dims")
        if not dims:
            raise ValueError(
                "mem0 config needs embedder.embedding_dims: it is also the Qdrant "
                "collection's vector size, and a mismatch is only discovered on the "
                "first insert."
            )

        store = dict(self.config.get("vector_store") or {})
        store.setdefault("collection_name", "mem0")
        store["embedding_model_dims"] = int(dims)
        # Local path mode: no server. mem0's Qdrant wrapper rmtree()s this path
        # when on_disk is false, which is what makes reset() a true wipe.
        store["path"] = os.path.join(self._store_dir, "qdrant")

        return MemoryConfig(
            llm={"provider": "openai", "config": llm_config},
            embedder={"provider": str(embedder.pop("provider", "huggingface")),
                      "config": embedder},
            vector_store={"provider": "qdrant", "config": store},
            history_db_path=self.history_db_path,
            version="v1.1",
        )

    def _set_extraction_clock(self, created_at: Optional[str]) -> None:
        """Point the fact-extraction prompt at the session being ingested.

        DEVIATION 3, and DEVIATION 1 applied one layer up.

        ``vendor/mem0/configs/prompts.py`` builds ``FACT_RETRIEVAL_PROMPT`` at
        import time with an f-string containing ``datetime.now()``, so every
        extraction prompt says "Today's date is <the afternoon the job ran>".
        Combined with the fact that 0.1.x's ``add()`` takes no ``timestamp=``,
        that made the wall clock the extractor's ONLY temporal anchor: it
        resolves "yesterday" and "last March" against the ingest date, for a
        conversation that happened years earlier.

        This is the same defect DEVIATION 1 fixes for ``created_at`` — and for
        the same stated reason — so it gets the same answer rather than a
        different one. Note what is NOT being done: no date is invented. The
        value is the session's own timestamp, which the benchmark supplies and
        which this method's caller already writes into ``metadata["created_at"]``
        two lines up. A fixed pin would be a fabrication; this is the clock the
        conversation actually had.

        Mechanically this rides upstream's own extension point.
        ``custom_fact_extraction_prompt`` is a documented ``MemoryConfig`` field
        (configs/base.py:54), and ``_add_to_vector_store`` reads it off
        ``self.config`` on every call (main.py:349) rather than the copy cached
        in ``__init__`` (main.py:135) — so assigning it per ``add()`` is
        supported, not a patch. The substituted prompt is byte-identical to
        upstream's apart from that one line, because main.py:350-353 builds the
        same ``f"Input:\\n{parsed_messages}"`` user prompt either way.

        ``fact_extraction_clock: wall`` restores upstream's behaviour. A session
        with no parseable timestamp also falls back to it — the wall clock is
        wrong there too, but inventing a date would be worse.
        """
        if not self.session_extraction_clock:
            return
        date = (created_at or "")[:10]
        if not date:
            self._mem.config.custom_fact_extraction_prompt = None
            return
        prompt = self._prompt_cache.get(date)
        if prompt is None:
            prompt = self._prompt_for_date(date)
            self._prompt_cache[date] = prompt
        self._mem.config.custom_fact_extraction_prompt = prompt
        self._extraction_dates.add(date)

    def _prompt_for_date(self, date: str) -> str:
        """Upstream's extraction prompt with ``date`` in place of the wall clock.

        The baked-in literal is matched by pattern rather than by recomputing
        ``datetime.now()``: the prompt is an f-string evaluated when mem0 was
        imported, which is not necessarily today for a process that has been
        alive across midnight — and a silently-failed substitution here would
        ship the wall clock while claiming the session clock.
        """
        from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT

        out, n = _PROMPT_DATE_RE.subn(f"Today's date is {date}.", FACT_RETRIEVAL_PROMPT)
        if n != 1:
            raise RuntimeError(
                f"mem0's FACT_RETRIEVAL_PROMPT has {n} \"Today's date is <date>.\" "
                f"lines, expected exactly 1 — upstream reworded it, so the "
                f"session-clock substitution in evalgrid/systems/mem0/system.py "
                f"needs updating. Set fact_extraction_clock: wall to run without it."
            )
        return out

    def _record_usage(self, llm, response, params) -> None:
        """``OpenAIConfig.response_callback`` — one completion's token usage,
        and whether the endpoint stopped it at ``max_tokens``.

        Mem0 has no usage accounting of its own; this is the hook upstream
        already calls, so no monkeypatch is needed. Failures here are swallowed
        by upstream's caller, so keep it total.

        Truncation is counted here rather than in ``_install_trace`` for two
        reasons. This hook is handed the RAW response, so ``finish_reason`` is
        readable — ``generate_response`` returns only the parsed content, which
        is what upstream's own JSON handling then swallows. And the counter has
        to exist whether or not tracing is on, exactly as lightmem's does
       ; ``meta.json`` is written either way.

        ``params`` carries the message list, so the same shape test the
        trajectory uses names the stage without a second classifier.

        ``empty_extractions`` is counted here for the same two reasons, plus a
        third: it is the ONLY place the extraction's own output is visible.
        ``add()`` sees the pipeline's end result, which is empty both when the
        extractor returned no facts and when the resolver answered NONE to all
        of them — see the counter's definition in ``__init__``.
        """
        usage = getattr(response, "usage", None)
        self._mem_usage = TokenUsage(
            input_tokens=self._mem_usage.input_tokens + int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=self._mem_usage.output_tokens + int(getattr(usage, "completion_tokens", 0) or 0),
            calls=self._mem_usage.calls + 1,
        )

        try:
            stage = _mem0_stage(messages=(params or {}).get("messages")) or "unknown"
            choices = getattr(response, "choices", None) or []
            if stage == "extract" and _extracted_nothing(choices):
                self.empty_extractions += 1
            if choices and (getattr(choices[0], "finish_reason", "") or "") == "length":
                self.truncated_calls += 1
                self.truncated_by_stage[stage] = self.truncated_by_stage.get(stage, 0) + 1
        except Exception:
            # Upstream logs and drops anything raised here, so a malformed
            # response must not cost the run its token accounting above.
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Fresh store per sample: close the old Memory, then build a new one.

        Closing first is not optional. Local Qdrant holds an exclusive lock on
        its folder, and ``Memory.__init__`` opens two of them (the store and the
        telemetry store), so constructing the replacement while the old object
        is still alive raises "already accessed by another instance". The
        constructor then rmtree()s the store path, which is what makes this a
        wipe rather than a reuse.
        """
        self._close()
        memory_class = _memory_class()
        self._patch_telemetry()
        self._mem = memory_class(self._build_config())
        self._reset_history()
        self._mem_usage = TokenUsage()
        self.events = {}
        self.traces = []
        self.empty_extractions = 0
        self.no_op_adds = 0
        self.add_calls = 0
        self.truncated_calls = 0
        self.truncated_by_stage = {}
        # Per-sample, so meta.json reports the dates THIS sample extracted
        # under. _prompt_cache is not cleared: a substituted prompt depends only
        # on its date, so it is safe (and cheaper) to share across samples.
        self._extraction_dates = set()
        self._install_trace()

    # ------------------------------------------------------------------
    # Trajectory recording (evalgrid/lib/trace.py) — L1
    # ------------------------------------------------------------------

    def install_recorder(self, recorder) -> None:
        super().install_recorder(recorder)
        self._install_trace()

    def _install_trace(self) -> None:
        """Log both LLM calls mem0 makes per ``add()``.

        Wrapping ``self._mem.llm.generate_response`` rather than reusing the
        ``response_callback`` cost hook: that hook is handed the response only,
        so it can price a call but cannot show the prompt — and the prompt is
        the point of a trajectory. The two coexist; the callback stays the cost
        source of record.

        Cost is read through a callable because ``_record_usage`` REASSIGNS
        ``self._mem_usage`` rather than mutating it, so a captured reference
        would go stale after the first call and every delta would be wrong.
        Mem0 is rebuilt every reset, so this must run from ``reset``.

        The two calls are told apart by the SHAPE of the message list, not by
        matching prompt text. ``_add_to_vector_store`` sends the fact
        extraction as (system, user) — a system prompt plus the parsed
        messages (vendor/mem0/memory/main.py:353-360) — and the
        ADD/UPDATE/DELETE resolution as a single user message carrying the
        whole function-calling prompt (main.py:400-407). One boundary, two
        stages, and no vendor edit; text matching would break on the first
        custom_fact_extraction_prompt, whereas the shapes are structural.
        """
        if self._mem is None or getattr(self._mem, "llm", None) is None:
            return
        record_calls(
            self._mem.llm, "generate_response",
            recorder=self.recorder, step="mem0.generate_response",
            usage=lambda: (self._mem_usage.input_tokens,
                           self._mem_usage.output_tokens,
                           self._mem_usage.calls),
            stage_of=_mem0_stage, step_of=_mem0_step,
        )

    def _reset_history(self) -> None:
        """Empty the mutation log so it covers only the sample about to start.

        ``reset()`` makes the VECTOR store per-sample — ``Memory.__init__``
        rmtree()s the Qdrant path — but the history db is not part of that. It
        lives at a fixed per-process path (``_process_store_dir``) and
        ``SQLiteManager`` merely OPENS it, so without this the log accumulates
        across every sample the process ingests, and ``save_memory`` copies the
        whole accumulation into each sample's dir.

        Observed in ``outputs/_smoke4`` before this existed: the four samples'
        copies held 696 / 1,355 / 2,039 / 2,594 rows against their own 696 /
        659 / 684 / 555 events, so the last sample's history.db was 78%
        other samples' mutations. Two costs — the log stops being an auditable
        record of THIS sample, and the copies grow as O(n^2): ~21 GB over a
        500-sample non-sharded run against ~83 MB correct, a 250x blowup.

        Retrieval never read it (``iter_units`` reads points.jsonl), so no
        published number moved; what was wrong was the audit trail and the disk.

        NOT via ``SQLiteManager.reset()``, which **deadlocks**: it takes
        ``self._lock`` and then calls ``_create_history_table()``, which takes
        the same non-reentrant ``threading.Lock``
        ([storage.py:199-206](../../../vendor/mem0/memory/storage.py#L199) vs
        [:101](../../../vendor/mem0/memory/storage.py#L101)), so the thread
        blocks on itself forever. Nothing upstream calls it — ``Memory.reset()``
        does the DROP inline on ``db.connection`` instead (main.py:988-992) —
        which is why the deadlock has gone unnoticed.

        So this empties the table through the connection directly, the same way
        ``Memory.reset()`` does, and keeps the connection and schema mem0 just
        built. Emptying rather than unlinking matters: the file is already open,
        and replacing it per sample would churn a file descriptor 500 times.

        Tolerant of a missing db — a provider without one is not a reason to
        fail an ingest — but NOT of a failing delete: silently keeping a
        cumulative log is the bug this exists to prevent.
        """
        db = getattr(self._mem, "db", None)
        connection = getattr(db, "connection", None)
        if connection is None:
            return
        with getattr(db, "_lock", contextlib.nullcontext()):
            connection.execute("DELETE FROM history")
            connection.commit()

    def _close(self) -> None:
        if self._mem is None:
            return
        for store in (getattr(self._mem, "vector_store", None),
                      getattr(self._mem, "_telemetry_vector_store", None)):
            client = getattr(store, "client", None)
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        db = getattr(self._mem, "db", None)
        connection = getattr(db, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        self._mem = None
        gc.collect()

    def _patch_telemetry(self) -> None:
        """No-op mem0's PostHog events, when ``disable_telemetry`` is set.

        ``MEM0_TELEMETRY=false`` alone only sets ``posthog.disabled`` on an
        object that ``capture_event`` rebuilds *per event* — and rebuilding it
        constructs a PostHog client and touches the telemetry vector store, per
        created fact. On an offline compute node it is also a network call that
        cannot succeed. The events carry no information we want, so they are
        replaced by a no-op at both the definition and the call sites, which
        imported the name directly.
        """
        if not self.config.get("disable_telemetry", True):
            return
        import mem0.memory.main as mem0_main
        import mem0.memory.telemetry as mem0_telemetry

        def _noop(*args, **kwargs):
            return None

        mem0_telemetry.capture_event = _noop
        mem0_main.capture_event = _noop

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add(self, messages: list[dict], *, session: Session,
            turn_ids: list[str], has_answer: bool) -> None:
        """One ``Memory.add()`` — the whole of Mem0's construction pipeline.

        Two LLM calls (fact extraction, then ADD/UPDATE/DELETE/NONE resolution),
        plus one embed and one k=5 neighbour search per extracted fact. The
        metadata here rides into every fact this call creates: upstream deepcopies
        ``base_metadata_template`` onto each one, and strips only the identity
        keys (``user_id``/``agent_id``/``run_id``) — which is why no provenance
        field may be named any of those.
        """
        if not messages:
            return
        metadata = {
            "session_id": session.session_id,
            "session_ids": [session.session_id],
            # The benchmark's own string, kept verbatim: this is what becomes
            # MemoryUnit.timestamp and what the answer stage renders. The ISO
            # form goes into created_at, which is mem0's own field.
            "session_ts": session.timestamp,
            "turn_ids": list(turn_ids),
            "has_answer": bool(has_answer),
        }
        created_at = to_iso(session.timestamp)
        if created_at:
            metadata["created_at"] = created_at

        self._set_extraction_clock(created_at)
        result = self._mem.add(messages, user_id=self.user_id,
                               metadata=metadata, infer=self.infer)
        self.add_calls += 1
        written = (result or {}).get("results") or []
        if not written:
            # Upstream swallows extraction and JSON failures and returns [] —
            # an entire sample can end up with an empty store and no exception.
            # Counted here and asserted on in save_memory. This is the WEAKER
            # of the two signals: it fires both when nothing was extracted and
            # when everything extracted resolved to NONE. empty_extractions,
            # counted off the extraction reply itself in _record_usage, is the
            # one that means facts were lost.
            self.no_op_adds += 1
        for item in written:
            event = str(item.get("event") or "UNKNOWN")
            self.events[event] = self.events.get(event, 0) + 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    #
    # points.jsonl is deliberately the same shape as lightmem's: one line per
    # vector-store point, {"id", "payload"}. The payload IS mem0's memory — the
    # text under "data", our provenance beside it — so the corpus builder needs
    # no system-specific knowledge beyond key names, and iter_units stays
    # importable on a machine that cannot run mem0.

    def save_memory(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        points = self._points()
        points.sort(key=lambda p: (p.get("payload", {}).get("created_at") or ""))
        with open(os.path.join(path, "points.jsonl"), "w") as f:
            for point in points:
                f.write(json.dumps(point, ensure_ascii=False, default=str) + "\n")

        # The complete mutation log: (memory_id, old_memory, new_memory, event).
        # It is the only place a DELETE'd or overwritten fact's prior text
        # survives, and the only auditable record of an ingest whose end state
        # is not deterministic.
        if os.path.isfile(self.history_db_path):
            shutil.copy(self.history_db_path, os.path.join(path, "history.db"))

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "units": len(points),
                "add_calls": self.add_calls,
                "events": self.events,
                # Extraction calls that returned {"facts": []}: the turn pair
                # held nothing the extractor recognised. On LongMemEval that is
                # 38-54% of pairs and 20-32% of whole SESSIONS, because mem0's
                # prompt asks only for personal facts —
                "empty_extractions": self.empty_extractions,
                # add() calls that wrote nothing: empty_extractions PLUS the
                # ones where every extracted fact resolved to NONE. Always >=
                # empty_extractions; the gap is the resolver declining to store
                # facts it judged already known.
                "no_op_adds": self.no_op_adds,
                # Nonzero means the store is quietly incomplete: check_ingest
                # reads "truncated_calls" as a shared degradation counter, and
                # the split says which half was lost. Read the completion before
                # raising memory_llm.max_tokens: in our runs
                # every truncation in a_mem/hypermem/simplemem was a repetition
                # loop and mem0's were the only genuinely long replies.
                "truncated_calls": self.truncated_calls,
                "truncated_by_stage": dict(sorted(self.truncated_by_stage.items())),
                "infer": self.infer,
                # Which clock the extractor ran on, and the dates it actually
                # saw. Under "wall" this is one date — whatever day the job ran
                # — and the store is only reproducible within that day; under
                # "session" it is the sample's own session dates. See
                # _set_extraction_clock and self.fact_extraction_clock.
                "extraction_dates": sorted(self._extraction_dates) or [
                    datetime.now().strftime("%Y-%m-%d")],
                # How the extraction calls were decoded. Nothing else on disk
                # distinguishes a store built at temperature 0.1 (mem0's library
                # default, and what this repo shipped until 2026-08-18) from one
                # built at the paper's 0 — the points are just text either way.
                "sampling": self.sampling,
                "mem0_version": "0.1.118",
            }, f, indent=2)

        if not points:
            # Loud, because it is otherwise invisible: the sample looks ingested
            # (files exist), and only the qrels stage notices, 500 samples later.
            print(f"[mem0] WARNING: sample stored 0 memories after {self.add_calls} "
                  f"add() call(s); {self.no_op_adds} wrote nothing, of which "
                  f"{self.empty_extractions} extracted nothing. "
                  f"Check the extraction model's JSON output.")

    def _points(self) -> list[dict]:
        """Raw vector-store points for this sample.

        ``get_all()`` would reshape them into MemoryItems and fold every custom
        key under "metadata"; the raw payload is closer to the store and is what
        ``load_memory`` puts back.
        """
        result = self._mem.vector_store.list(
            filters={"user_id": self.user_id}, limit=100_000)
        # Qdrant's scroll returns (records, next_offset); other backends return
        # a bare list. Same unwrap upstream does in _get_all_from_vector_store.
        records = result[0] if isinstance(result, (tuple, list)) and result else result
        return [{"id": str(record.id), "payload": record.payload} for record in records or []]

    def load_memory(self, path: str) -> None:
        """Restore a saved store into a fresh Memory.

        Vectors are recomputed rather than persisted, exactly as in lightmem's
        loader, so loading with a different embedder than the one that ingested
        silently changes what native search returns. It does NOT change the flat
        corpus, which is built from the text.
        """
        with open(os.path.join(path, "points.jsonl")) as f:
            points = [json.loads(line) for line in f if line.strip()]
        if not points:
            return
        vectors = [self._mem.embedding_model.embed(p["payload"]["data"], "add")
                   for p in points]
        self._mem.vector_store.insert(
            vectors=vectors,
            payloads=[p["payload"] for p in points],
            ids=[p["id"] for p in points],
        )

    # ------------------------------------------------------------------
    # Retrieval + answering (Mem0's native path)
    # ------------------------------------------------------------------

    def retrieve(self, question: str, k: Optional[int] = None) -> list[dict]:
        """Mem0 0.1.x native retrieval: embed the query, one dense top-k.

        That is the whole of it — no fusion, no reranking, no LLM. Which is why
        this system loses almost nothing when the pipeline swaps in BM25 or a
        different dense retriever over the flat corpus, and why it is the fairest
        row in the table for that comparison. (2.x would not be: its retriever
        is a BM25 + entity-boost fusion, and replacing it deletes the system's
        actual contribution.)
        """
        ranker = getattr(self, "_native_ranker", None)
        if ranker is not None:
            # A substituted retriever (evalgrid/systems/native_retrieval.py).
            # Mem0 has no post-retrieval step, so the ranking IS the result; the
            # payloads are looked up so downstream rendering is unchanged.
            ranked = ranker.rank(question, k or self.top_k)
            by_id = {str(p.get("id")): p for p in self._points()}
            out = []
            for unit_id, score in ranked:
                point = by_id.get(str(unit_id))
                if point is None:
                    continue
                out.append(_as_search_result(point, unit_id, score))
            return out
        out = self._mem.search(question, user_id=self.user_id, limit=k or self.top_k)
        return (out or {}).get("results") or []

    def retrieve_trace(self, query: Query) -> dict:
        """One dense top-k. No generator call."""
        results = self.retrieve(query.question)
        facts = [self._render(item) for item in results]
        trace = {
            "query_id": query.query_id,
            "retrieved_unit_ids": [str(item.get("id")) for item in results],
            "retrieval_scores": [item.get("score") for item in results],
            "step2_retrieved_context": "\n".join(facts),
            # Rank-ordered, one entry per retrieved unit. This is what a token
            # budget truncates — the joined context above cannot be cut at a
            # unit boundary once it is a single string.
            "retrieved_units": facts,
        }
        self.traces.append(trace)
        return trace

    def answer(self, query: Query) -> str:
        return self.answer_from_trace(query, self.retrieve_trace(query))

    def answer_from_trace(self, query: Query, trace: dict) -> str:
        facts = trace.get("retrieved_units")
        if facts is None:
            facts = [f for f in (trace.get("step2_retrieved_context") or "").split("\n") if f]
        messages = self.answer_messages(query, facts)
        return asyncio.run(self.generator.generate(messages, max_tokens=self.max_tokens))

    @staticmethod
    def _render(item: dict) -> str:
        """'[<session time>] <fact>', matching pipeline.answer.format_fact.

        Mem0 never puts a time in the retrieved string — ``search()`` returns
        the bare fact and leaves ``created_at`` in the payload — so a temporal
        question is unanswerable from its native output as shipped. The stamp is
        added here in the same shape every other corpus is rendered in, so the
        native row and the flat-corpus rows differ in the retriever and nothing
        else.
        """
        text = item.get("memory") or ""
        stamp = (item.get("metadata") or {}).get("session_ts")
        return f"[{stamp}] {text}" if stamp and stamp not in text else text

    def answer_messages(self, query: Query, facts: list[str]) -> list[dict]:
        """This repo's shared prompt.

        Mem0's own ``MEMORY_ANSWER_PROMPT`` is unused by upstream's benchmarks
        and ``Memory.chat()`` raises NotImplementedError; the prompt behind the
        published numbers lives in the (submoduled) benchmark harness, not in
        the library. So there is no "upstream prompt" to be faithful to here.
        """
        return shared_answer_prompt(query.question, query.question_date, facts)

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------

    def layer_usage(self) -> dict:
        """Split tokens by layer.

        Memory layer: the two calls per ``add()``. Retrieval spends none — 0.1.x
        search is one embedding and one dense query — so this layer is exactly
        the construction cost.

        Generation layer: the final answer, from our generator.
        """
        return {"memory": self._mem_usage, "generation": self.generator.token_usage}
