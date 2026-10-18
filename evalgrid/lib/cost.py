"""
evalgrid/lib/cost.py
--------------------
The per-question query-time cost record, in one shape for both answer stages.

An answering-token budget equalises what the generator READS. It says nothing
about what was spent to decide what it reads, and that assumption — retrieval is
free — is true for only half the systems here. lightmem, mem0, memorybank and
every raw-turn baseline retrieve with one dense top-k or BM25: no LLM call at
query time. A-Mem generates query keywords then walks links, SimpleMem runs
plan -> tri-path union -> reflect (5-8 calls), HyperMem runs a 30/20/15 funnel.
Two systems can land on identical ``context_tokens`` and differ by an order of
magnitude in test-time cost, so a matched-budget cell alone would report the
cheap one and the expensive one as equals.

The answer is not to equalise it — the loop IS the method being measured — but
to record it, so accuracy can be read against cost instead of against a number
that hides half of it. Three costs, none collapsible into the others:

  C1  construction   ingest tokens + LLM calls   outputs/_memory/**/ingest_usage.json
  C2  query-time LLM this module                 answer_stats.jsonl / predictions.jsonl
  C3  answering context  context_tokens          see evalgrid/lib/tokens.py

On LongMemEval-S each of the 500 samples is one question over its own haystack,
so C1 is not amortised across questions and ``C1 + C2`` is a well-defined
per-question total. That stops being true on any benchmark with several
questions per haystack (LoCoMo) — say which is meant before adding them up.

C2 splits in two, because "spent before the answer call" and "spent on the
answer call" are different claims:

  ``query_retrieval_*``  every layer except "generation" — the system's own
                         query-time reasoning. Zero on the flat path, where
                         retrieval is an offline embedding/BM25 pass whose cost
                         belongs to stage 3 and is not per-question LLM spend.
  ``query_answer_*``     the "generation" layer — the final answer call.
  ``query_llm_*``        the sum. This is the x-axis of the accuracy/cost curve.

The layer split comes from each system's ``layer_usage()``, snapshotted per
phase by benchmarks/runner.py, so a query-time memory-model call lands here and
not in the ingest total.
"""

from __future__ import annotations

#: Layer name holding the final answer call. Everything else a system reports is
#: query-time reasoning that happened BEFORE it, so the split needs no per-system
#: knowledge — a system that adds a third layer is counted correctly by default.
GENERATION_LAYER = "generation"


def query_cost(retrieval: tuple[int, int, int] = (0, 0, 0),
               answer: tuple[int, int, int] = (0, 0, 0)) -> dict:
    """Build the cost fields from ``(input_tokens, output_tokens, calls)`` pairs.

    ``query_answer_input_tokens`` is broken out because it is the only C3 the
    native path can report: a system builds its own prompt inside ``answer()``,
    so the evidence cannot be isolated from the template the way
    ``context_tokens`` isolates it on the flat path. It is the same quantity the
    flat path records as ``input_tokens``, so the two are directly comparable —
    just measured one template wider than ``context_tokens``.
    """
    r_in, r_out, r_calls = retrieval
    a_in, a_out, a_calls = answer
    return {
        "query_retrieval_tokens": r_in + r_out,
        "query_retrieval_calls": r_calls,
        "query_answer_tokens": a_in + a_out,
        "query_answer_input_tokens": a_in,
        "query_answer_calls": a_calls,
        "query_llm_tokens": r_in + r_out + a_in + a_out,
        "query_llm_calls": r_calls + a_calls,
    }


def split_answer_layers(by_layer: dict) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Split a runner per-layer answer-phase delta into (retrieval, answer).

    ``by_layer`` is ``{layer: {input_tokens, output_tokens, calls}}`` as
    ``RunReport.accumulate`` returns it. Systems that expose no layer split
    report everything under "generation" (runner._layer_usage's fallback), which
    attributes it all to the answer call — correct for them, since a system
    without a memory layer makes no query-time memory calls.
    """
    def _triple(d: dict) -> tuple[int, int, int]:
        return (d.get("input_tokens", 0), d.get("output_tokens", 0), d.get("calls", 0))

    r_in = r_out = r_calls = 0
    answer = (0, 0, 0)
    for name, d in (by_layer or {}).items():
        if name == GENERATION_LAYER:
            answer = _triple(d)
        else:
            i, o, c = _triple(d)
            r_in += i
            r_out += o
            r_calls += c
    return (r_in, r_out, r_calls), answer


def sum_cost(records: list[dict]) -> dict:
    """Total the cost fields over a list of per-question records."""
    keys = ("query_retrieval_tokens", "query_retrieval_calls",
            "query_answer_tokens", "query_answer_input_tokens", "query_answer_calls",
            "query_llm_tokens", "query_llm_calls")
    return {k: sum(int(r.get(k) or 0) for r in records) for k in keys}
