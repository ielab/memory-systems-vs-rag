"""
evalgrid/systems/a_mem/system.py
--------------------------------
Base class wrapping the vendored A-Mem (``AgenticMemorySystem``) behind the
MemorySystem interface.

Holds the shared plumbing — construction, the memory-LLM controller, token
accounting, provenance tracking, and persistence. Dataset-specific
ingest/answer behaviour lives in per-dataset subclasses:
  locomo.py      — AMemLoCoMo: one note per turn, per-category answer prompts
  longmemeval.py — AMemLongMemEval: one note per user turn, shared prompt

Everything under vendor/a_mem/ is upstream verbatim. Where upstream's behaviour
needs changing to run here, it is changed from this side — by assigning over an
attribute, or by injecting a name into the module — never by editing the
vendored file. Each such point is marked ADAPTER: below.

Two of them are worth reading before you trust a run:

ADAPTER: memory_layer.py:380 calls ``re.sub`` but memory_layer.py never imports
``re``. The NameError is swallowed by a bare ``except:`` whose handler is itself
broken, so ``analyze_content`` returns empty metadata for EVERY note no matter
what the LLM said — keywords ``[]``, tags ``[]``, context ``"General"``. A-Mem
then degenerates to embedding raw content, which is naive RAG wearing A-Mem's
name, with no error anywhere. The regression is upstream commit ab06c22
(2025-12-28), ten months after the paper; every paper-era commit parses fine. We
bind ``re`` into the module at construction so note construction actually runs.
Set ``patch_missing_re: false`` to reproduce the broken behaviour deliberately.

ADAPTER: ``links`` hold positional indices into ``list(memories.values())``,
straight from the LLM, and ``find_related_memories_raw`` indexes with them
unchecked — one out-of-range integer is an IndexError that kills the sample
mid-ingest. We drop invalid indices after each note is added. Valid links are
untouched.

ADAPTER: ``process_memory`` calls ``find_related_memories(note.content, k=5)``
with the 5 written inline (memory_layer.py:755), so the paper's Eq. 5 neighbour
count M_near is not a parameter of anything. We rebind the method per instance so
``link_k`` from the config reaches it. ``find_related_memories`` has exactly one
in-class caller, so the rebind changes nothing else; the separate query path
(``find_related_memories_raw``) takes its own ``retrieve_k``.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
from typing import Optional

from evalgrid.benchmarks.types import MemorySystem, Query, Session
from evalgrid.models.base import GeneratorModel, TokenUsage
from evalgrid.lib.trace import record_calls, record_steps


def index_document(content: str, context: str, keywords: list, tags: list) -> str:
    """The text A-Mem embeds for one note — paper Eq. 3, concat(c, K, G, X).

    Byte-identical to what ``AgenticMemorySystem.add_note`` builds, because
    ``load_memory`` has to rebuild the index to the same strings the live system
    would have produced, and the retrieval corpus has to be the text A-Mem
    actually ranks on.

    Note this is NOT the only such format upstream uses: ``consolidate_memories``
    and ``load_from_local_memory`` build a different string from the same
    fields. Ours follows add_note, the one that indexes notes during ingest.
    """
    return ("content:" + content
            + " context:" + context
            + " keywords: " + ", ".join(keywords)
            + " tags: " + ", ".join(tags))


class AMemSystem(MemorySystem):
    """Shared A-Mem plumbing. Subclasses implement ``ingest`` and ``answer``.

    Args:
        config: the parsed evalgrid/configs/a_mem/<variant>.yaml.
        generator: answers queries. A-Mem's own code answers with the memory
            model; this repo separates the two so every system is compared under
            one generator, as LightMem is.
    """

    supports_persistence = True

    def __init__(self, config: dict, generator: GeneratorModel):
        self.config = config
        self.generator = generator

        self.embedder_model = (config.get("embedder") or {}).get(
            "model", "sentence-transformers/all-MiniLM-L6-v2")
        self.link_k = int(config.get("link_k", 5))
        self.top_k = int(config.get("retrieve_k", 10))
        self.evo_threshold = int(config.get("evo_threshold", 100))
        self.max_tokens = int(config.get("answer_max_tokens", 2000))
        self.patch_missing_re = bool(config.get("patch_missing_re", True))
        self.silence_upstream_stdout = bool(config.get("silence_upstream_stdout", True))

        self._controller = self._new_controller()
        self._encoder = None       # SentenceTransformer, loaded once and reused
        self._amem = None
        self._provenance: dict[str, dict] = {}
        self.traces: list[dict] = []
        self.degraded_notes = 0    # notes whose metadata generation failed

        self.reset()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _new_controller(self):
        from evalgrid.systems.a_mem.controller import OpenAICompatController

        llm = self.config.get("memory_llm") or {}
        if not llm.get("model"):
            raise ValueError(
                "a_mem config has no memory_llm.model. It is injected by "
                "AMemPlugin.apply_mem_model from --mem-model; if you built this system "
                "directly, set it yourself."
            )
        return OpenAICompatController(
            model=llm["model"],
            base_url=llm.get("base_url"),
            api_key=llm.get("api_key"),
            max_tokens=int(llm.get("max_tokens", 1000)),
            temperature=float(llm.get("temperature", 0.7)),
            structured_output=llm.get("structured_output", "auto"),
        )

    def _new_amem(self):
        # Imported here, not at module scope: A-Mem pulls in torch,
        # sentence-transformers, rank_bm25 and nltk, which must not be a
        # prerequisite for merely listing the registry or running LightMem.
        from a_mem import memory_layer

        # ADAPTER: bind the name memory_layer.py:380 uses but never imports.
        # Module-global lookup, so this is exactly what `import re` would have
        # done — without touching the vendored file. See the module docstring.
        if self.patch_missing_re and not hasattr(memory_layer, "re"):
            memory_layer.re = re

        # llm_backend="sglang" is the only backend whose controller constructor
        # is inert — it stores two strings. openai would build a real client
        # against api.openai.com and raise without a key. We overwrite .llm
        # immediately, so which one upstream picked never matters.
        amem = memory_layer.AgenticMemorySystem(
            model_name=self.embedder_model,
            llm_backend="sglang",
            llm_model=self._controller.model,
            evo_threshold=self.evo_threshold,
        )
        # ADAPTER: route every memory call through our OpenAI-compatible
        # controller (token accounting + a base_url upstream cannot express).
        amem.llm_controller.llm = self._controller

        # ADAPTER: make link_k reach process_memory's inline k=5. See the module
        # docstring. The bound original is kept so the neighbour search itself is
        # upstream's, only the count is ours.
        upstream_find = amem.find_related_memories
        link_k = self.link_k

        def find_related_memories(query: str, k: int = 5, _f=upstream_find, _k=link_k):
            return _f(query, k=_k)

        amem.find_related_memories = find_related_memories

        # Reuse one SentenceTransformer across samples. reset() builds a fresh
        # AgenticMemorySystem per sample, and each would otherwise re-instantiate
        # the encoder — 500 reloads over LongMemEval.
        if self._encoder is None:
            self._encoder = amem.retriever.model
        else:
            amem.retriever.model = self._encoder
        return amem

    def reset(self) -> None:
        self._amem = self._new_amem()
        self._provenance = {}
        self.traces = []
        # Per-sample counters. degraded_notes was previously never cleared and
        # never written anywhere, so it accumulated across a run and then went
        # nowhere; both halves are fixed here and in save_memory.
        self.degraded_notes = 0
        # A baseline rather than a reset: the controller outlives every sample,
        # and zeroing a counter on a shared object would lie to anything else
        # reading it.
        self._counters_at_reset = {
            name: getattr(self._controller, name, 0)
            for name in ("truncated_calls", "failed_calls")
        }
        self._install_trace()

    def _since_reset(self, name: str) -> int:
        """A controller counter's delta over this sample."""
        total = getattr(self._controller, name, 0)
        return max(0, total - getattr(self, "_counters_at_reset", {}).get(name, 0))

    def _truncated_this_sample(self) -> int:
        """Memory calls stopped at ``memory_llm.max_tokens`` since ``reset()``."""
        return self._since_reset("truncated_calls")

    # ------------------------------------------------------------------
    # Trajectory recording (evalgrid/lib/trace.py) — L1
    # ------------------------------------------------------------------

    def install_recorder(self, recorder) -> None:
        super().install_recorder(recorder)
        self._install_trace()

    def _install_trace(self) -> None:
        """Log every memory-LLM call A-Mem makes, under the stage that made it.

        L1: one method covers the whole system — ``get_completion`` is
        upstream's entire LLM interface, so note construction, metadata
        generation and every evolution prompt come through here.

        L2 is what makes that log readable. Because every call shares one
        method, an L1-only trajectory is N indistinguishable records whose
        stage can only be "llm" — "an LLM call nobody claimed". The two
        boundaries below name them instead, and ``recorder.call`` inherits the
        stage and parent from whichever is open:

        ``add_note``        extract   MemoryNote(...) -> analyze_content, the
                                      content/keywords/tags generation
        ``process_memory``  update    the evolution prompt, which rewrites the
                                      new note and its k nearest neighbours

        ``process_memory`` runs INSIDE ``add_note`` (memory_layer.py:720), so
        it nests: the evolution call lands under "update", the analyze call
        under "extract", and both point at the enclosing add_note step. Both
        are instance rebinds on the vendored system, and ``record_steps`` is
        idempotent, so re-installing per sample cannot stack wrappers.
        """
        record_calls(self._controller, "get_completion",
                     recorder=self.recorder, step="a_mem.get_completion")
        record_steps(self._amem, "add_note", recorder=self.recorder,
                     stage="extract", step="a_mem.add_note",
                     usage=self._controller.token_usage,
                     inputs_of=lambda content, *a, **k: content,
                     out_of=str)
        record_steps(self._amem, "process_memory", recorder=self.recorder,
                     stage="update", step="a_mem.process_memory",
                     usage=self._controller.token_usage,
                     inputs_of=lambda note, *a, **k: getattr(note, "content", None),
                     out_of=lambda result: f"evolved={bool(result[0])}"
                     if isinstance(result, tuple) else str(result))

    # ------------------------------------------------------------------
    # Ingest helper (subclasses call this)
    # ------------------------------------------------------------------

    def add_note(self, content: str, *, timestamp: Optional[str] = None,
                 turn_id: Optional[str] = None, session_id: Optional[str] = None,
                 has_answer: bool = False) -> str:
        """Add one note and record where it came from.

        Provenance is tracked here rather than on the note because upstream's
        ``MemoryNote.__init__`` takes a fixed parameter list — an extra
        ``turn_id=`` would be a TypeError. Keeping it in a side map alongside
        the note id costs nothing and keeps vendor/ verbatim.
        """
        with self._captured_stdout():
            note_id = self._amem.add_note(content, time=timestamp)

        self._sanitize_links(note_id)
        self._provenance[note_id] = {
            "source_turn_ids": [turn_id] if turn_id else [],
            "session_id": session_id,
            "has_answer": bool(has_answer),
        }
        note = self._amem.memories[note_id]
        if not note.keywords and not note.tags:
            self.degraded_notes += 1
        return note_id

    def _sanitize_links(self, note_id: str) -> None:
        """Drop link indices that do not address a memory. See module docstring.

        Upstream stores whatever integers the LLM emitted for
        ``suggested_connections`` and later does ``all_memories[neighbor]`` with
        no bounds check. Anything outside the current memory list can only crash
        the sample, so it is dropped; in-range links are left exactly as the LLM
        produced them.
        """
        note = self._amem.memories.get(note_id)
        if note is None or not note.links:
            return
        n = len(self._amem.memories)
        note.links = [i for i in note.links if isinstance(i, int) and 0 <= i < n]

    @contextlib.contextmanager
    def _captured_stdout(self):
        """Swallow upstream's per-note debug prints.

        ``process_memory`` and ``analyze_content`` print the full prompt and the
        full response for every note — megabytes per sample over a run that
        already takes hours, and it buries anything worth reading. Degraded
        notes are counted structurally in ``add_note`` instead of by scraping
        these lines, so nothing is lost by dropping them.
        """
        if not self.silence_upstream_stdout:
            yield
            return
        with contextlib.redirect_stdout(io.StringIO()):
            yield

    # ------------------------------------------------------------------
    # Retrieval (shared by both datasets)
    # ------------------------------------------------------------------

    def generate_query_keywords(self, question: str) -> str:
        """Turn the question into keywords before retrieving.

        A-Mem does not embed the question — Figure 2's retrieval stage "analyzes
        queries into constituent keywords and utilizes these keywords to search
        through the memory network". Prompt and schema are upstream's
        ``advancedMemAgent.generate_query_llm``, verbatim.
        """
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text.

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
        response = self._controller.get_completion(
            prompt,
            {"type": "json_schema", "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {"keywords": {"type": "string"}},
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
                "strict": True,
            }},
        )
        try:
            return json.loads(response)["keywords"]
        except Exception:
            return response.strip()

    @staticmethod
    def _render_note(note) -> str:
        """One note, in the EXACT shape upstream's context block renders it.

        Byte-identical to the per-note string ``find_related_memories_raw``
        concatenates (vendor/a_mem/memory_layer.py:891), minus the trailing
        newline it joins on. That identity is the whole point: a budget counts
        the tokens of these strings, so they must be the tokens the generator
        actually reads. Rendering the note any other way — e.g. through
        ``index_document``, the *indexing* string — would budget against text
        that never reaches the prompt, and the cell would be labelled with a
        token count the run never had.
        """
        return ("talk start time:" + (note.timestamp or "")
                + "memory content: " + (note.content or "")
                + "memory context: " + (note.context or "")
                + "memory keywords: " + str(note.keywords)
                + "memory tags: " + str(note.tags))

    def retrieve_context(self, keywords: str,
                         k: Optional[int] = None) -> tuple[str, list[str], list[str]]:
        """A-Mem's own retrieval: top-k by cosine similarity, plus linked notes.

        Delegates to upstream's ``find_related_memories_raw``, so the returned
        block is more than k notes — each hit also drags in its ``links``. That
        is A-Mem's design, and it is why its context is larger than top-k
        suggests when you compare token counts against the paper's.

        Returns (context, note ids, per-note texts). The ids let the native run
        be scored as retrieval; the texts are what make it budgetable, because
        the block upstream returns is one string with no unit boundary inside
        it and neither ``--top-k`` nor ``--context-budget`` can cut that.
        All three are produced by ONE walk, so element *n* of the id list and of
        the text list are the same note — a budget that cut them out of step
        would report ids that are not what the prompt received.

        Upstream returns only the formatted string, and re-running the search to
        recover ids would both re-encode the query and risk diverging from what
        was actually ranked — so the indices are captured from the one call
        ``find_related_memories_raw`` makes, and walked exactly as it walks them
        (hit, then its links, ``k+1`` neighbours at most; see the loop below and
        memory_layer.py:877-898). Ids repeat when two hits share a neighbour;
        de-duplication is the caller's, so this stays a faithful record of what
        upstream put in the prompt.
        """
        k = k or self.top_k
        indices: list = []
        retriever = self._amem.retriever
        original_search = retriever.search

        def _recording_search(query, k_=k, *args, **kwargs):
            out = original_search(query, k_, *args, **kwargs)
            indices.extend(list(out))
            return out

        retriever.search = _recording_search
        try:
            with self._captured_stdout():
                context = self._amem.find_related_memories_raw(keywords, k=k)
        finally:
            retriever.search = original_search

        # Note order is load-bearing: links are positional indices into
        # list(memories.values()), which is the same order the retriever indexed
        # (see the Persistence note below), so this list is what upstream subscripts.
        notes = list(self._amem.memories.values())
        unit_ids: list[str] = []
        unit_texts: list[str] = []
        for i in indices:
            if i >= len(notes):
                continue
            unit_ids.append(notes[i].id)
            unit_texts.append(self._render_note(notes[i]))
            j = 0
            for neighbor in notes[i].links:
                if neighbor >= len(notes):
                    continue
                unit_ids.append(notes[neighbor].id)
                unit_texts.append(self._render_note(notes[neighbor]))
                if j >= k:
                    break
                j += 1
        return context, unit_ids, unit_texts

    def retrieve_trace(self, query: Query) -> dict:
        """Keyword generation -> A-Mem retrieval, de-duplicated. No generator call.

        This is the expensive half: one memory-LLM call for the keywords, then
        the cosine search and its link expansion.

        DEVIATION from upstream, applied here rather than in
        ``retrieve_context``: that method stays a faithful record of upstream's
        walk (it promises as much, and ``ranked_from_trace`` relies on the
        repeats carrying a note's *best* rank), so the filter is the caller's,
        exactly where its docstring says de-duplication belongs.

        A-Mem's expansion re-adds a note that is already in the block whenever
        two hits share a neighbour, or a neighbour is itself a hit — 47 units
        for 26 distinct notes on the 2026-08-15 smoke sample, i.e. 45% of the
        context is text the generator has already read. Upstream repeats it
        (``memory_layer.py:889-897`` appends each hit, then all of its links,
        with no seen-set), so this is a departure from A-Mem as published and a
        row built on it should say so. What it buys is the two knobs meaning
        what they say: under ``--top-k 5`` five units are five distinct notes,
        and under ``--context-budget`` the budget buys evidence rather than
        copies. Link expansion itself is untouched — the notes a hit drags in
        are still there, each of them once.

        The context block is re-joined from the kept texts rather than the
        de-duplicated list being left to disagree with upstream's string: the
        join is "\\n" and ``_render_note`` is byte-identical to the per-note
        string upstream concatenates, so an unfiltered query reproduces the
        block exactly and a filtered one differs only by the removed copies.
        Keeping the original string here would put the repeats back in every
        *unbudgeted* prompt — ``answer_from_trace`` reads the block, not the
        units, unless a cut set ``budget_applied``.
        """
        keywords = self.generate_query_keywords(query.question)
        _, unit_ids, unit_texts = self.retrieve_context(keywords)
        seen: set[str] = set()
        kept_ids: list[str] = []
        kept_texts: list[str] = []
        for uid, text in zip(unit_ids, unit_texts):
            if uid in seen:
                continue
            seen.add(uid)
            kept_ids.append(uid)
            kept_texts.append(text)
        # Trace keys are the ones benchmarks/runner.py::_add_trace_fields and
        # SystemPlugin.native_unit_ids read. ``retrieved_units`` is additionally
        # what pipeline/answer.py's --top-k and --context-budget cut on a replay:
        # A-Mem's context is one upstream-rendered block with no unit boundary
        # inside it, so without this list neither knob can be applied and both
        # raise.
        trace = {
            "query_id": query.query_id,
            "step1_keywords": keywords,
            "step2_retrieved_context": "\n".join(kept_texts),
            "retrieved_unit_ids": kept_ids,
            "retrieved_units": kept_texts,
            # How much of the block was a copy. Recorded rather than inferred:
            # once the trace is filtered, nothing downstream can tell a query
            # whose expansion overlapped heavily from one whose did not.
            "duplicate_units_removed": len(unit_ids) - len(kept_ids),
        }
        self.traces.append(trace)
        return trace

    def answer(self, query: Query) -> str:
        """Keyword generation -> A-Mem retrieval -> the subclass's prompt."""
        return self.answer_from_trace(query, self.retrieve_trace(query))

    def answer_from_trace(self, query: Query, trace: dict) -> str:
        """Answer from a trace, honouring a budget/depth cut if one was applied.

        The base implementation reads ``step2_retrieved_context`` only. That is
        correct for an UNBUDGETED replay — it reuses upstream's own formatter
        output verbatim — but wrong the moment pipeline/answer.py cuts the
        units, because it cuts ``retrieved_units`` and cannot touch the block
        (there is no unit boundary in it to cut at). Inheriting the base here
        meant a budgeted run reported the cut size in answer_stats.jsonl while
        the prompt still received the whole block: a 19x understatement on the
        smoke sample, silent, and in exactly the runs a matched-budget table is
        built from.

        So when ``budget_applied`` is set the kept units are re-joined. The
        join is "\\n", the separator upstream concatenates notes on, and
        ``_render_note`` produces the same per-note string it does — so an
        uncut re-join reproduces the block and the token count a budget was
        measured against is the token count the generator reads.

        Same shape as HyperMem's override, for the same reason.
        """
        context = trace.get("step2_retrieved_context") or ""
        if trace.get("budget_applied"):
            context = "\n".join(trace.get("retrieved_units") or [])
        messages = self.answer_messages(query, context)
        return asyncio.run(self.generator.generate(
            messages, max_tokens=getattr(self, "max_tokens", 2000)))

    def answer_messages(self, query: Query, context: str) -> list[dict]:
        raise NotImplementedError

    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    #
    # A-Mem's state is the note set: everything retrieval needs is derivable
    # from it, since the index is just index_document() over each note, embedded.
    # We save notes as JSONL rather than upstream's pickle so the store is
    # readable, diffable, and usable by iter_units without importing A-Mem —
    # and so a store outlives the class layout that wrote it.
    #
    # Note order is load-bearing twice over: links are positional indices into
    # list(memories.values()), and the retriever's document order must match
    # that list. Writing and reading in insertion order preserves both.

    def save_memory(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "notes.jsonl"), "w") as f:
            for note_id, note in self._amem.memories.items():
                prov = self._provenance.get(note_id, {})
                f.write(json.dumps({
                    "id": note_id,
                    "content": note.content,
                    "context": note.context,
                    "keywords": list(note.keywords),
                    "tags": list(note.tags),
                    "timestamp": note.timestamp,
                    "links": list(note.links),
                    "source_turn_ids": prov.get("source_turn_ids", []),
                    "session_id": prov.get("session_id"),
                    "has_answer": bool(prov.get("has_answer", False)),
                }, ensure_ascii=False) + "\n")

        # The two ways a sample degrades without raising, as numbers rather
        # than as something you find by reading a trajectory. Both are silent
        # by construction: upstream json-parses the metadata and evolution
        # replies inside try/excepts, so a note that lost its keywords/tags and
        # an evolution that never happened both look like ordinary success.
        notes = self._amem.memories if self._amem is not None else {}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "notes": len(notes),
                "degraded_notes": self.degraded_notes,
                "truncated_calls": self._truncated_this_sample(),
                # Calls that raised after exhausting their retries. token_usage
                # counts only calls that returned, so ingest_usage.json's
                # llm_calls cannot show these.
                "failed_calls": self._since_reset("failed_calls"),
                # Which prompting mode built this store: structured_output "auto"
                # latches off on the first rejection and never returns.
                "structured_output": getattr(self._controller, "structured_output", None),
                "schema_fallback": bool(getattr(self._controller, "schema_fallback", False)),
                "links": sum(len(n.links or []) for n in notes.values()),
                "evo_threshold": self.evo_threshold,
            }, f, indent=2)

    def load_memory(self, path: str) -> None:
        from a_mem import memory_layer

        records = []
        with open(os.path.join(path, "notes.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return

        documents = []
        for rec in records:
            # llm_controller=None is what suppresses metadata generation: with a
            # controller, MemoryNote.__init__ regenerates whenever any of
            # keywords/context/category/tags is None — and `category` always is.
            # Passing None here would re-run (and re-bill) the whole
            # construction stage on every answer run.
            note = memory_layer.MemoryNote(
                content=rec["content"],
                id=rec["id"],
                keywords=rec.get("keywords") or [],
                links=list(rec.get("links") or []),
                timestamp=rec.get("timestamp"),
                context=rec.get("context") or "General",
                tags=rec.get("tags") or [],
                llm_controller=None,
            )
            self._amem.memories[rec["id"]] = note
            self._provenance[rec["id"]] = {
                "source_turn_ids": rec.get("source_turn_ids", []),
                "session_id": rec.get("session_id"),
                "has_answer": bool(rec.get("has_answer", False)),
            }
            documents.append(index_document(
                note.content, note.context, note.keywords, note.tags))

        # One batched call: add_documents assigns indices sequentially, so this
        # reproduces the per-note order the live system had.
        self._amem.retriever.add_documents(documents)

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------

    def layer_usage(self) -> dict:
        """Split tokens by layer.

        Memory layer: note construction and evolution during ingest, and — the
        part that surprises people — query-keyword generation during answering,
        because A-Mem calls the memory model again at query time. The runner
        snapshots per phase, so that shows up under answer_tokens_by_layer
        rather than being mistaken for construction cost.

        Generation layer: the final answer, from our generator.
        """
        return {
            "memory": TokenUsage(
                input_tokens=self._controller.token_usage.input_tokens,
                output_tokens=self._controller.token_usage.output_tokens,
                calls=self._controller.token_usage.calls,
            ),
            "generation": self.generator.token_usage,
        }
