"""
evalgrid/systems/base.py
------------------------
The plugin contract every memory system implements to enter the pipeline.

``benchmarks/types.py`` defines what a memory system *is* at runtime
(``reset``/``ingest``/``answer``/``save_memory``/``load_memory``). This module
defines what the *pipeline* needs to know about one, which is a different and
smaller thing: how to construct it, how to read its saved memory back as a flat
retrieval corpus, and how it wants to be prompted.

The axes that vary per system, and where each is answered:

  construction      SystemPlugin.build           — lazily imports the system's deps
  hyperparameters   SystemPlugin.config_path     — evalgrid/configs/<system>/<variant>.yaml
  flatten-to-corpus SystemPlugin.iter_units      — saved memory -> MemoryUnit stream
  answering         SystemPlugin.answer_prompt   — --prompt native
  native retrieval  SystemPlugin.native_unit_ids — trace -> ranked unit ids

``iter_units`` is the load-bearing one. It is the executable form of the
question this whole project asks — *can this system's memory be
flattened into self-contained, independently-retrievable text units with stable
ids?* — because that is what lets us swap a system's native retriever for
BM25 / dense / SPLADE and study memory as an IR problem. A system that cannot
implement it cannot be evaluated on the flat-corpus protocol, and saying so in
code is more useful than discovering it three stages downstream.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional
from urllib.parse import urlparse

from evalgrid.lib.paths import ROOT


#: Output-token ceilings OpenAI enforces per model. A request over the ceiling
#: is a 400, which hypermem's provider treats as fatal (it cannot be retried
#: into fitting), so a vLLM-sized max_tokens has to be cut down, not passed on.
OPENAI_MAX_OUTPUT_TOKENS = {
    "gpt-4o-mini": 16384,
    "gpt-4o": 16384,
}


def route_mem_llm_to_openai(llm: dict, base_url: Optional[str],
                            mem_model: str) -> None:
    """Make a memory-LLM config block usable against the OpenAI API, in place.

    Every config ships ``api_key: token`` for vLLM, and each system passes that
    to its client verbatim — so the OPENAI_API_KEY fallback in the controllers
    never fires and an OpenAI --mem-model gets a 401 on its first call. An
    endpoint counts as OpenAI when no base URL is given (the client's default)
    or its host is api.openai.com; any other endpoint is left untouched.
    """
    if base_url and urlparse(base_url).hostname != "api.openai.com":
        return
    key = os.environ.get("OPENAI_API_KEY")
    if key and llm.get("api_key") in (None, "", "token"):
        llm["api_key"] = key
    cap = OPENAI_MAX_OUTPUT_TOKENS.get(mem_model)
    if cap and isinstance(llm.get("max_tokens"), int) and llm["max_tokens"] > cap:
        llm["max_tokens"] = cap


@dataclass
class MemoryUnit:
    """One independently-retrievable piece of a system's saved memory.

    This is the common denominator across systems: LightMem's extracted fact,
    A-Mem's note, Mem0's fact, MemoryBank's turn-pair or daily summary. The
    retrieval corpus is a stream of these, so anything a retriever or the answer
    prompt needs must be on this object.

    unit_id   stable within a sample; becomes the BEIR corpus ``_id`` and the
              id a run.tsv ranks, so it must survive save/load unchanged.
    text      the retrievable text, clean — no timestamp folded in. The stamp
              travels in ``timestamp`` and is rendered at answer time by
              evalgrid.pipeline.answer.format_fact, so every corpus shape
              renders identically.
    has_answer  the unit derives from a turn the benchmark marks as evidence.
              This is what qrels_has_answer turns into relevance labels, so a
              system that loses provenance during extraction cannot produce
              turn-level qrels —
    source_turn_ids  provenance, when the system tracks it. Carried on the unit
              but NOT written to corpus.jsonl today; emitting it is a corpus
              format change and would require re-running stage 2 for every
              existing store.
    """

    unit_id: str
    text: str
    timestamp: Optional[str] = None
    has_answer: bool = False
    source_turn_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def shared_answer_prompt(question: str, question_date: Optional[str],
                         facts: list[str]) -> list[dict]:
    """The prompt every system is compared under by default.

    Derived from LightMem's LongMemEval prompt, restructured for readability:
    same content — the question (with its time, when present), the "Please
    answer ... based on the following memories:" lead-in, and the facts each as
    a "[<time_stamp>] <text>" line joined by newlines — but laid out more
    legibly, with the question time and question on their own labelled lines and
    the memories starting on the line AFTER the lead-in. This is deliberately
    NOT byte-verbatim with upstream LightMem's prompt; it re-flows the
    whitespace. The fact format and the join are unchanged.

    Holding this fixed across systems is what makes the comparison about memory
    representation rather than prompt engineering. A system that needs its own
    prompt overrides SystemPlugin.answer_prompt.
    """
    header = (f"Question time: {question_date}\nQuestion: {question}"
              if question_date else f"Question: {question}")
    facts_text = "\n".join(facts) if facts else "(no relevant memories found)"
    user_content = (
        f"{header}\n\n"
        f"Please answer the question based on the following memories:\n{facts_text}"
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]


class SystemPlugin(ABC):
    """What the pipeline needs to know about one memory system.

    Subclass, set ``name``, implement ``build`` and ``iter_units``, and register
    it in evalgrid/systems/registry.py.
    """

    #: Registry key. Also the top-level path segment under outputs/, so changing
    #: it orphans every tree a previous run wrote.
    name: str

    #: True when this run replaces the system's own ranker — set by
    #: ``native_retrieval.declare`` before ``build``, so a system whose
    #: constructor builds an index (or loads an encoder) only its own ranking
    #: would read can skip that work. Read it in ``build``; do not set it.
    native_swap: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self, config: dict, *, dataset: str, generator, phase: str):
        """Return a configured MemorySystem.

        Import the system's own dependencies INSIDE this method. Systems have
        conflicting dependency trees (langchain, neo4j, qdrant), and a
        module-level import would make one system's missing package break every
        other system's run.

        config    the parsed YAML from ``config_path`` (already merged with any
                  spec-level system_config overrides).
        dataset   lets a system vary its ingest policy per benchmark, the way
                  LightMem uses flat extraction on LongMemEval and event
                  extraction on LoCoMo.
        generator answers queries. Ingestion passes a generator that raises if
                  called, so a system that generates during ingest fails loudly
                  rather than silently spending tokens.
        phase     "ingest" or "answer".
        """

    def apply_mem_model(self, config: dict, mem_model: str,
                        mem_base_url: Optional[str] = None) -> dict:
        """Point the system's memory-construction LLM at ``mem_model``.

        Separate from ``build`` so ingest can set the model without knowing any
        system's config shape. ``mem_model`` keys outputs/_memory/, so a system
        that quietly ignored it would write a store whose path names a model
        that never ran — hence the default raises rather than no-ops. A system
        that genuinely builds memory without an LLM (MemoryBank's write path)
        overrides this to return ``config`` unchanged, explicitly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement apply_mem_model, so --mem-model "
            f"would not reach its config while still keying the output dir. Override it "
            f"(return config unchanged if the system builds memory without an LLM)."
        )

    # ------------------------------------------------------------------
    # Retrieval / answering budget
    # ------------------------------------------------------------------
    #
    # Both of these exist because the native path would otherwise compare
    # systems at different budgets and call the difference a result. On the flat
    # corpus the pipeline imposes one --top-k and one --max-tokens on everybody;
    # natively each system brings its own (a_mem 10 + link expansion, memorybank
    # 6, hypermem 30 facts + 20 episodes at 4096 output tokens, simplemem an
    # uncapped union), so a native-vs-flat gap has retrieval depth and output
    # length folded into it unless these are set.

    def apply_retrieval_budget(self, config: dict, top_k: int) -> dict:
        """Point the system's OWN retrieval at ``top_k``.

        The default writes ``retrieve_k``, which is the key a_mem, mem0,
        memorybank and lightmem all read. A system whose native retrieval is
        shaped differently — HyperMem's per-level block, SimpleMem's tri-path
        block — overrides this.

        This sets the system's *internal* k; it does NOT cap the final context.
        A-Mem still expands each hit's links and SimpleMem's union is still
        unbounded, because that expansion IS the method the native path exists
        to measure — truncating it would delete the thing being studied. The
        resulting context size therefore still varies between systems, which is
        why every native answer record carries ``context_items``: the part of
        the budget that cannot be equalised is measured instead of hidden.
        """
        config["retrieve_k"] = int(top_k)
        return config

    def native_retriever_name(self, config: dict) -> str:
        """Name the retriever the system's OWN loop will use, from its config.

        Mirrors evalgrid.lib.paths.retriever_name for the flat corpus, and for
        the same reason: there, `--emb-model` is folded into the run dir
        automatically (`emb-<model>`, `sparse-<model>`) so two encoders can never
        share — and silently clobber — a directory, and no caller has to remember
        to name it. The native path had no such derivation, so swapping a
        system's encoder in its YAML changed the ranking while leaving the run in
        the same folder as the previous encoder's.

        Returned in the same vocabulary the flat side uses, so the segments pair
        up: a native run on all-MiniLM-L6-v2 gives `emb-sentence-transformers_
        all-MiniLM-L6-v2`, exactly the flat dense run's name. That is what makes
        the equivalence check readable — for LightMem, Mem0 and MemoryBank the
        native ranking should equal the flat dense ranking at the same k, and the
        two dirs now differ by the `native-` prefix and nothing else.

        The default reads ``embedder.model`` (a_mem, mem0, memorybank,
        simplemem). LightMem nests it under ``text_embedder.configs.model`` and
        HyperMem has no encoder at all — both override.
        """
        from evalgrid.lib.paths import retriever_name

        # A substituted retriever wins: when native_retrieval.install swaps the
        # ranking function, the encoder named below is no longer what ranked, and
        # naming the run after it would put a SPLADE run in the dense run's
        # directory — the exact collision this method exists to prevent.
        from evalgrid.systems.native_retrieval import retriever_slug

        swapped = retriever_slug(config)
        if swapped:
            return swapped

        model = (config.get("embedder") or {}).get("model")
        if not model:
            raise NotImplementedError(
                f"{type(self).__name__}.native_retriever_name: no 'embedder.model' in the "
                f"config, so the native run dir could not name the encoder it ranked with "
                f"and two encoders would share one directory. Override this to point at "
                f"wherever this system keeps it."
            )
        return retriever_name("dense", model)

    def apply_answer_max_tokens(self, config: dict, max_tokens: int) -> dict:
        """Set the generation budget for the system's own answer call.

        The default writes ``answer_max_tokens``, which every system's
        MemorySystem reads. Kept as a hook rather than a bare config key so a
        system that names it differently has somewhere to say so.
        """
        config["answer_max_tokens"] = int(max_tokens)
        return config

    def config_path(self, variant: Optional[str] = None) -> str:
        """Absolute path to this system's config for ``variant``.

        The variant name IS the config filename, so the two cannot drift:
        --mem-variant rate0.4-stm768 resolves to
        evalgrid/configs/<name>/rate0.4-stm768.yaml, and no variant resolves to
        default.yaml. Because --mem-variant also suffixes every output dir, the
        path a run writes to always names the config that produced it.
        """
        return os.path.join(ROOT, "evalgrid", "configs", self.name,
                            f"{variant or 'default'}.yaml")

    # ------------------------------------------------------------------
    # Flatten saved memory into a retrieval corpus
    # ------------------------------------------------------------------

    @abstractmethod
    def iter_units(self, sample_mem_dir: str) -> Iterable[MemoryUnit]:
        """Read one sample's saved memory back as retrievable units.

        ``sample_mem_dir`` is what MemorySystem.save_memory wrote. Yield in a
        stable order (chronological where meaningful) — corpus.jsonl order is
        what the full-context baseline feeds the generator.

        Return an empty iterable when the directory holds no memory; the qrels
        builder reports those as missing rather than failing the batch.
        """

    def records_has_answer(self, sample_mem_dir: str) -> bool:
        """Whether this saved memory records evidence provenance at all.

        Distinguishes "no unit is evidence" (legitimate — the sample yields
        empty qrels) from "this store predates provenance tracking" (a re-ingest
        is needed). MemoryUnit.has_answer defaults to False and so cannot
        express the difference; only the raw saved form can.

        Defaults to True — a system that always threads provenance through
        extraction has no second case to distinguish.
        """
        return True

    #: At what granularity ``has_answer`` is exact on this system's units, and
    #: therefore what qrels_has_answer's labels really mean here.
    #:
    #: "turn"    one unit per turn (or per-fact provenance back to one turn):
    #:           the label set is the evidence turn itself.
    #: "pair"    one unit per (user, assistant) exchange.
    #: "session" a unit inherits the flag from a whole session, so EVERY unit
    #:           extracted from the evidence session is labelled relevant.
    #: "window"  same, over a batching window, which may span sessions.
    #:
    #: This is not cosmetic. Measured over the four outputs/_smoke4 samples, one
    #: evidence turn yields 1-2 labels under "turn" and 16-25 under "session" —
    #: so recall@k over a session-level store is scored against a gold set an
    #: order of magnitude larger, and the systems are not on the same task. Each
    #: plugin's records_has_answer docstring has said "not comparable across
    #: systems; use an LLM labeler" since these were written; this makes the
    #: claim machine-readable so qrels_has_answer can say so at build time
    #: rather than leaving it to be read.
    has_answer_granularity: str = "turn"

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def answer_prompt(self, question: str, question_date: Optional[str],
                      facts: list[str]) -> list[dict]:
        """Messages for the generator, given the retrieved facts.

        Defaults to the shared prompt. Override to reproduce a system's
        published prompt — answers are keyed by <system> in the output tree, so
        per-system prompts never collide with each other. (Running one system
        under BOTH the shared and its native prompt would collide, and needs a
        path segment first; see Design.md.)
        """
        return shared_answer_prompt(question, question_date, facts)

    # ------------------------------------------------------------------
    # The native path
    # ------------------------------------------------------------------

    def native_unit_ids(self, trace: dict) -> list[str]:
        """Rank-ordered unit ids the system's OWN retrieval touched for one query.

        This is what makes the native pipeline measurable as retrieval rather
        than only as answer quality. ``evalgrid.pipeline.answer --retriever native --prompt native`` runs
        ``MemorySystem.answer`` — the system's real query-time loop, keyword
        generation, link expansion, hypergraph pruning, multi-round reflection
        and all — and turns the ids returned here into the same run.tsv a BM25
        or dense run writes. Scored against the same ``has_answer`` qrels, so
        "A-Mem's own retriever" and "BM25 over A-Mem's notes" become one table.

        ``trace`` is the system's last ``self.traces`` entry. The default reads
        ``retrieved_unit_ids``, which every system in this repo records; a
        system whose trace shape differs (HyperMem's is keyed by hypergraph
        level) overrides this.

        Ids must be the SAME ids ``iter_units`` yields — a native run whose ids
        do not exist in the corpus scores 0 recall and looks like a retrieval
        failure rather than the wiring bug it is. Duplicates are allowed and
        de-duplicated by the caller (A-Mem's link expansion can return a note
        twice); rank order is significant, score is not.

        Return an empty list when retrieval found nothing.
        """
        ids = trace.get("retrieved_unit_ids")
        if ids is None:
            raise NotImplementedError(
                f"{type(self).__name__}.native_unit_ids: the system's trace has no "
                f"'retrieved_unit_ids' key, so its native retrieval cannot be scored "
                f"against qrels. Record the ids in MemorySystem.answer's trace, or "
                f"override native_unit_ids to map this system's trace shape."
            )
        return [str(i) for i in ids]
