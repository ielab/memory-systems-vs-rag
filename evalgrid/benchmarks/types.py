"""
evalgrid/benchmarks/types.py
--------------------
Shared data models and the MemorySystem interface.

Implementing MemorySystem means a system can be evaluated on any benchmark
in evalgrid/benchmarks/adapters/. All memory systems import from here.

Hierarchy:
  Sample   — one user's full history (all sessions + all queries)
    └── Session  — one conversation sitting (one date/time block)
          └── Turn     — one speaker utterance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from evalgrid.lib.trace import NULL_RECORDER


@dataclass
class Turn:
    role: str
    content: str
    turn_id: Optional[str] = None    # e.g. LoCoMo dia_id "D1:3"
    has_answer: bool = False          # LongMemEval turn-level evidence flag
    extra: dict = field(default_factory=dict)


@dataclass
class Session:
    """One unit ingested into the memory system.

    Dialogue benchmarks: one session = one conversation sitting (turns non-empty).
    Document benchmarks (MemoryAgentBench): turns empty, text set instead.
    """
    session_id: str
    text: Optional[str] = None
    turns: list[Turn] = field(default_factory=list)
    timestamp: Optional[str] = None

    def render(self) -> str:
        if self.text is not None:
            return self.text
        header = f"[{self.timestamp}]\n" if self.timestamp else ""
        return header + "\n".join(f"{t.role}: {t.content}" for t in self.turns)


@dataclass
class Query:
    query_id: str
    question: str
    answer: Any                               # str or list[str]
    question_type: Optional[str] = None
    question_date: Optional[str] = None      # LongMemEval: date the question is asked
    evidence_session_ids: list[str] = field(default_factory=list)
    evidence_turn_ids: list[str] = field(default_factory=list)  # LoCoMo dia_ids
    extra: dict = field(default_factory=dict)


@dataclass
class Sample:
    """One context + its associated questions.

    LongMemEval: 1 sample = 1 question.
    LoCoMo / BEAM / MemoryAgentBench: 1 sample = multiple questions.
    """
    sample_id: str
    source: str        # "longmemeval" | "locomo" | "beam" | "memoryagentbench"
    sessions: list[Session]
    queries: list[Query]

    #: True when this benchmark localises evidence PER QUESTION, in
    #: ``Query.evidence_turn_ids``, rather than per sample in ``Turn.has_answer``.
    #:
    #: An adapter declaration, not something gold.py can infer. The two states it
    #: separates look identical from a single query: a BEAM abstention question
    #: names no evidence because it HAS none by design, while a LongMemEval
    #: question names none because that benchmark puts its evidence on the turns.
    #: Read the first as the second and the abstention questions inherit the
    #: union of their conversation's other 19 gold sets — measured at 44 turns
    #: where the answer is 0, which is the widest a wrong gold set gets.
    #:
    #: So it is set for the whole sample, by the adapter that knows which
    #: benchmark it is reading, and it turns the fallback OFF for every query in
    #: that sample including the ones naming nothing. Default False keeps every
    #: has_answer-based benchmark reading exactly as before.
    evidence_is_per_query: bool = False


class MemorySystem(ABC):
    """Interface every memory system must implement to be evaluated."""

    #: Trajectory recorder (evalgrid/lib/trace.py). Defaults to the no-op
    #: singleton, so ``self.recorder.call(...)`` and ``with
    #: self.recorder.step(...)`` are always safe to write with no guard and cost
    #: nothing when recording is off. Set through ``install_recorder``, never in
    #: a constructor: recording must stay off the construction path.
    recorder: Any = NULL_RECORDER

    def install_recorder(self, recorder) -> None:
        """Attach a trajectory recorder.

        Called by the pipeline immediately after ``plugin.build`` and BEFORE the
        first ``reset()``, so a system that builds its LLM adapter in ``reset``
        (a_mem, simplemem, mem0 all do) can wrap it there by reading
        ``self.recorder``. The default just stores it — a system that records
        nothing is a system with no trajectory, not an error.

        It also names the read path for every system at once. ``retrieve_trace``
        is the one method the retrieve stage drives per query, so wrapping it
        here in an L2 step gives each system's query-time LLM calls a stage —
        SimpleMem's 5-8 plan/reflect calls and A-Mem's keyword call would
        otherwise all land as ``"llm"``, trace.py's word for "an LLM call nobody
        claimed", and 8 identical-looking records per question is not a
        trajectory anyone can read. The step also carries the loop's own wall
        time and cost delta, which is the only place the *whole* read path is
        priced as one unit.

        Wrapped HERE and not in each system's ``_install_trace`` for two
        reasons. It is the base method every system already overrides, so one
        wrap covers all six. And it sits ABOVE ``retrieve()``, which
        native_retrieval rebinds when a retriever is swapped in — wrapping
        ``retrieve`` instead would mean whichever of the two installed second
        silently discarded the other.
        """
        self.recorder = recorder
        # Idempotent (record_steps checks its own marker), so a re-install
        # cannot stack wrappers, and a no-op entirely when recording is off.
        from evalgrid.lib.trace import record_steps

        record_steps(self, "retrieve_trace", recorder=recorder,
                     stage="retrieve", step=f"{type(self).__name__}.retrieve_trace",
                     usage=self._memory_layer_usage,
                     inputs_of=lambda query, *a, **k: getattr(query, "question", None),
                     out_of=lambda trace: (trace or {}).get("retrieved_unit_ids"))

    def _memory_layer_usage(self) -> tuple[int, int, int]:
        """The memory layer's cumulative (input, output, calls), for step costs.

        The memory layer and not the generator's: nothing on the read path is
        allowed to generate (pipeline/retrieve.py hands these systems a
        generator that raises), so every token a retrieval step spends is a
        memory-layer token — A-Mem's keyword call, SimpleMem's plan/reflect
        loop. Read from ``layer_usage()``, the same counter runner.py sums into
        ``query_retrieval_*``, so a step's cost and the run's total cannot
        disagree.

        Defensive throughout: this is called on entry and exit of every
        recorded step, and a cost that cannot be read must degrade to zero
        rather than take a run down — the recorder is required to be inert.
        """
        try:
            layers = self.layer_usage() if hasattr(self, "layer_usage") else None
            usage = (layers or {}).get("memory")
            if usage is None:
                return (0, 0, 0)
            return (int(getattr(usage, "input_tokens", 0) or 0),
                    int(getattr(usage, "output_tokens", 0) or 0),
                    int(getattr(usage, "calls", 0) or 0))
        except Exception:
            return (0, 0, 0)

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory before each sample."""

    @abstractmethod
    def ingest(self, session: Session, *, is_last: bool = False) -> None:
        """Incrementally store one memory session.

        is_last: True on the final session of a sample — some systems (e.g.
                 LightMem with update=offline) need to flush buffered state.
        """

    @abstractmethod
    def answer(self, query: Query) -> str:
        """Return a free-text answer given current memory state."""

    # ------------------------------------------------------------------
    # The native seam: retrieval and answering as two halves
    # ------------------------------------------------------------------
    #
    # ``answer()`` is one call because that is how every one of these systems is
    # published — the retrieval loop and the prompt are the same method. But the
    # two halves cost wildly different things (SimpleMem spends 5-8 LLM calls
    # retrieving and 1 answering) and a budget sweep only varies the second, so
    # running ``answer()`` once per budget re-pays for retrieval every time.
    #
    # These two methods expose the seam without changing what a single
    # ``answer()`` call does: the default ``answer()`` below is exactly their
    # composition, so a system that overrides them gets the split for free and a
    # system that overrides ``answer()`` directly still works.
    #
    #   retrieve_trace(query)        -> the trace dict, retrieval only, no
    #                                   generator call
    #   answer_from_trace(query, t)  -> the answer, from a trace this system (or
    #                                   a previous run of it) produced
    #
    # The trace is the interface between them, and it is the same dict
    # ``self.traces`` already holds — ``query_id``, ``retrieved_unit_ids``,
    # ``step2_retrieved_context``. That makes a saved retrieval run replayable:
    # see retrieve --retriever native and answer --prompt native --from-retrieval.

    def retrieve_trace(self, query: Query) -> dict:
        """Run ONLY this system's native retrieval; return its trace dict.

        Must append to ``self.traces`` exactly as ``answer()`` does, and must not
        call the generator. Systems that do not override this fall back to
        ``answer()``, which does generate — correct, just not cheaper.
        """
        self.answer(query)
        return self.traces[-1] if getattr(self, "traces", None) else {}

    def prepare_retrieval(self) -> None:
        """Build whatever query-time index this system derives from its loaded
        memory, BEFORE the first question is timed.

        Called once per sample, after ``load_memory`` (and after any
        native_retrieval swap is installed), outside every latency clock. A
        system that builds its corpus index lazily on the first query must do
        that work here instead: otherwise the first question of every sample —
        on LongMemEval, the only question — reports a corpus encode as its
        retrieval latency. Queries themselves are still encoded live per call.
        The default does nothing, for systems whose ``load_memory`` already
        rebuilds the store.
        """

    def answer_from_trace(self, query: Query, trace: dict) -> str:
        """Answer from an already-computed retrieval trace.

        ``trace`` may come from ``retrieve_trace`` in this process or from a
        previous run's saved retrieval, so this must not assume any state beyond
        the loaded memory. The default renders the trace's stored context through
        the system's own ``answer_messages``.
        """
        context = trace.get("step2_retrieved_context", "")
        messages = self.answer_messages(query, context)
        import asyncio as _asyncio
        return _asyncio.run(self.generator.generate(
            messages, max_tokens=getattr(self, "max_tokens", 2000)))

    # ------------------------------------------------------------------
    # Optional persistence — lets the eval decouple ingestion from
    # retrieval/answer. A system ingests once, ``save_memory`` writes the
    # built memory to ``path`` (a per-sample directory), and a later run
    # ``load_memory``s it instead of re-ingesting. Only the (expensive)
    # memory-construction work is saved; query-time retrieval may still use
    # the memory model (e.g. a_mem's query-keyword generation).
    # Systems that support it override both; the default makes the
    # capability discoverable via ``supports_persistence``.
    # ------------------------------------------------------------------

    supports_persistence: bool = False

    def save_memory(self, path: str) -> None:
        """Serialise current memory state into directory ``path``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support memory persistence"
        )

    def load_memory(self, path: str) -> None:
        """Restore memory state previously written by ``save_memory(path)``.

        Implementations should behave as if the sessions had just been
        ingested: callers invoke ``reset()`` first, then ``load_memory``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support memory persistence"
        )
