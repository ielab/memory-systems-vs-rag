"""
evalgrid/systems/simplemem/plugin.py
------------------------------------
SimpleMem's SystemPlugin: how to build it, and how to read its saved memory back
as a flat retrieval corpus.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from evalgrid.systems.base import MemoryUnit, SystemPlugin, route_mem_llm_to_openai

#: SimpleMem's three retrieval paths, in upstream's own merge-priority order
#: (semantic > keyword > structured). That order is NOT the one
#: HybridRetriever._merge_and_deduplicate names in its comment — that method is
#: dead code, defined at hybrid_retriever.py:326 and called from nowhere in the
#: module this adapter runs (it is live only in integrations/server/). The path
#: that runs is _merge_and_deduplicate_entries (L409), a flat first-seen-wins
#: dedup over an all_results list built semantic (L101) -> keyword (L109) ->
#: structured (L114), so arrival order IS the priority.
#: Listed here in the order a directory tag spells them, which is fixed so two
#: spellings of the same subset cannot produce two directories.
LEGS = ("semantic", "keyword", "structured")


def parse_legs(config: dict) -> frozenset:
    """Which of the tri-path union's legs are enabled. Default: all three.

    ``retrieval.legs`` is OURS — upstream has no such switch, and the union is
    what SimpleMem's §3.3 calls its method. It exists because "SimpleMem with
    only the dense path" and "SimpleMem with only the lexical path" are the two
    ablations that say what the union is worth, and there is no honest way to
    reach them by setting a leg's top-k to 0: whether LanceDB's ``limit(0)``
    empties a search or ignores the limit is backend behaviour we would be
    depending on silently.
    """
    retrieval = (config or {}).get("retrieval") or {}
    raw = retrieval.get("legs")
    if raw is None:
        return frozenset(LEGS)
    if isinstance(raw, str):
        raw = [raw]
    legs = frozenset(str(x).strip() for x in raw if str(x).strip())
    unknown = legs - set(LEGS)
    if unknown:
        raise ValueError(f"retrieval.legs: unknown leg(s) {sorted(unknown)}; "
                         f"valid legs are {list(LEGS)}")
    if not legs:
        raise ValueError("retrieval.legs is empty — SimpleMem would retrieve nothing. "
                         "Omit the key for all three legs.")
    return legs


def legs_tag(config: dict) -> Optional[str]:
    """Directory tag for a leg ablation, or None when all three legs are on.

    Without this, a ``legs: [semantic]`` run and the full-union run land in the
    same directory and the second silently overwrites the first — the same
    collision ``native_retriever_name`` exists to prevent on the encoder axis.
    """
    legs = parse_legs(config)
    if legs == frozenset(LEGS):
        return None
    return "legs-" + "+".join(leg for leg in LEGS if leg in legs)


class SimpleMemPlugin(SystemPlugin):

    name = "simplemem"

    # One extraction call per window, and every entry it returns inherits
    # the window's flag. Session-aligned under the default policy. See
    # records_has_answer.
    has_answer_granularity = "window"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, config: dict, *, dataset: str, generator, phase: str):
        config = json.loads(json.dumps(config))  # don't mutate the caller's dict

        # native_swap reaches the constructor on OUR copy of the config, never
        # on the caller's: SimpleMemBase must know at construction that its own
        # encoder is dead weight this run (it loads one in __init__ and embeds
        # on every load_memory, and native_retrieval replaces the only call that
        # reads those vectors). Set only when true, so an unswapped run's config
        # is byte-identical to one that never knew the flag existed.
        if self.native_swap:
            config["native_swap"] = True

        # Subclasses imported inside the branches, not at module scope: they
        # reach vendor/simplemem, which pulls in lancedb, pyarrow, dateparser,
        # torch and sentence-transformers. None of that may be a prerequisite
        # for listing the registry or for running some other system.
        if dataset == "locomo":
            from evalgrid.systems.simplemem.locomo import SimpleMemLoCoMo
            return SimpleMemLoCoMo(config, generator)
        if dataset.startswith("beam"):
            from evalgrid.systems.simplemem.beam import SimpleMemBEAM
            return SimpleMemBEAM(config, generator)
        from evalgrid.systems.simplemem.longmemeval import SimpleMemLongMemEval
        return SimpleMemLongMemEval(config, generator)

    def apply_mem_model(self, config: dict, mem_model: str,
                        mem_base_url: Optional[str] = None) -> dict:
        """SimpleMem's memory LLM writes every entry — the restatement, the
        keywords and the symbolic fields all come out of one extraction call per
        window — so it is the model that keys outputs/_memory/.

        It is also called at *answer* time, and much harder than at ingest:
        native retrieval spends 5-8 calls per question on planning, query
        analysis and reflection. The saved store does not capture those, so an
        answer run on the native path needs this set too, not only ingest.
        """
        llm = config.setdefault("memory_llm", {})
        llm["model"] = mem_model
        if mem_base_url:
            llm["base_url"] = mem_base_url
        route_mem_llm_to_openai(llm, llm.get("base_url"), mem_model)
        return config

    def native_retriever_name(self, config: dict) -> str:
        """The base name (published encoder, or a substitute), plus the leg tag.

        A leg ablation changes what SimpleMem retrieves just as surely as a
        swapped ranker does, so it has to key the run directory too. The two
        compose: ``legs: [semantic]`` with ``native_retriever: bm25`` gives
        ``bm25--legs-semantic`` — SimpleMem's planning and reflection driving a
        single lexical path, which is the cell that is actually comparable to
        another system's BM25 row.
        """
        base = super().native_retriever_name(config)
        tag = legs_tag(config)
        return f"{base}--{tag}" if tag else base

    def apply_retrieval_budget(self, config: dict, top_k: int) -> dict:
        """SimpleMem retrieves down three paths and unions them.

        ``top_k`` sets the semantic path, which is the dense top-k directly
        comparable to another system's. The keyword and structured paths keep
        upstream's much smaller budgets (5 each) rather than being raised to
        ``top_k``: they are precision-oriented side channels, and upstream's
        ratio is what the union was tuned around.

        Note what this does NOT do. The union has no final cap anywhere in
        upstream's merge, and planning can issue up to four sub-queries, so the
        context can still be far larger than ``top_k`` — on a hard question,
        most of the store. That is SimpleMem's actual behaviour and the native
        path exists to measure it; ``context_items`` on each answer record is
        how it stays visible.
        """
        retrieval = config.setdefault("retrieval", {})
        retrieval["semantic_top_k"] = int(top_k)
        return config

    # ------------------------------------------------------------------
    # Flatten saved memory into a retrieval corpus
    # ------------------------------------------------------------------

    def iter_units(self, sample_mem_dir: str) -> Iterable[MemoryUnit]:
        """One unit per SimpleMem entry, in ingest order.

        ``text`` is ``lossless_restatement`` alone, because that is the only
        field SimpleMem itself indexes: the dense vector is an embedding of it
        (``VectorStore.add_entries``) and the Tantivy full-text index is built
        over that one column (``vector_store_backend._init_fts_index``). Folding
        keywords or topic into the text would rank strings SimpleMem's own
        retriever never sees, and the comparison would stop being about the
        representation.

        The symbolic fields (``persons``/``entities``/``location``/the event
        timestamp) are carried in ``metadata`` rather than dropped: they are
        SimpleMem's third retrieval path, and a future hybrid retriever that
        wants metadata filters needs them. They are not written to corpus.jsonl
        today — that would be a corpus format change requiring stage 2 to be
        re-run for every existing store.

        ``timestamp`` is the WINDOW's conversation time, not SimpleMem's own
        ``timestamp`` field. The latter is an event time the extractor read out
        of the dialogue ("the meeting is on 2025-11-16"), is null for most
        entries, and is already spelled out inside the restatement text by the
        prompt's absolute-time rule. Stamping the corpus with conversation time
        instead keeps ``[<when it was said>] <fact>`` uniform with LightMem's
        memories and the raw-turn corpus, which is what
        evalgrid.pipeline.answer.format_fact renders.
        """
        path = os.path.join(sample_mem_dir, "entries.jsonl")
        if not os.path.isfile(path):
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield MemoryUnit(
                    unit_id=rec["entry_id"],
                    text=rec["lossless_restatement"],
                    timestamp=rec.get("window_timestamp"),
                    has_answer=bool(rec.get("has_answer")),
                    source_turn_ids=list(rec.get("source_turn_ids") or []),
                    metadata={
                        "keywords": rec.get("keywords") or [],
                        "topic": rec.get("topic"),
                        "persons": rec.get("persons") or [],
                        "entities": rec.get("entities") or [],
                        "location": rec.get("location"),
                        "event_timestamp": rec.get("event_timestamp"),
                        "session_ids": rec.get("session_ids") or [],
                    },
                )

    def records_has_answer(self, sample_mem_dir: str) -> bool:
        """True if this sample's saved entries carry the has_answer key.

        A store written before provenance was threaded through ingest has no
        such key, and would otherwise look like a sample where no entry is
        evidence — producing empty qrels instead of an error.

        Note what the flag means here: SimpleMem extracts a whole window in one
        LLM call and keeps no per-entry link back to a turn, so every entry from
        a window inherits ``has_answer`` from the window as a whole. Under the
        default session-aligned policy that is session-level relevance — the
        granularity LongMemEval's own ``answer_session_ids`` ships at, but
        coarser than LightMem's per-fact ``source_id`` or A-Mem's
        one-note-per-turn. has_answer qrels are therefore comparable across
        SimpleMem variants, not across systems; use an LLM labeler for that.
        """
        path = os.path.join(sample_mem_dir, "entries.jsonl")
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    return "has_answer" in json.loads(line)
        return False
