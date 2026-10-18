"""
evalgrid/benchmarks/retrieval.py
-------------------------
Run persistence + evaluation for evalgrid/benchmarks/retrievers/ output.

Pairs with evalgrid/benchmarks/qrels.py: qrels.py builds the ground truth (corpus,
queries, qrels) once per (system, dataset, mem-model); this module writes and
scores a retriever's run against that ground truth, any number of times
(every retriever you want to compare reuses the same qrels).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

# pytrec_eval is imported inside evaluate_run, not here: it is a C extension and
# only the scoring path needs it, while the read/write helpers in this module are
# pure IO used by the answer stage. A module-level import made
# those stages fail to start on a machine that only generates answers.

DEFAULT_K_VALUES = (5, 10, 20)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_run(out_dir: str, ranked_by_query: list[tuple[str, list[tuple[str, float]]]]) -> None:
    """Write one sample's run.tsv: every query's ranked (corpus_id, score) pairs."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.tsv"), "w") as f:
        f.write("query-id\tcorpus-id\trank\tscore\n")
        for qid, ranked in ranked_by_query:
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid}\t{doc_id}\t{rank}\t{score}\n")


def read_run(path: str) -> dict[str, dict[str, float]]:
    """Parse a run.tsv into {query_id: {corpus_id: score}}."""
    run: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        next(f, None)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, doc_id, _rank, score = line.split("\t")
            run[qid][doc_id] = float(score)
    return dict(run)


def read_qrels(path: str) -> dict[str, dict[str, int]]:
    """Parse a qrels/test.tsv into {query_id: {corpus_id: relevance}}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path) as f:
        next(f, None)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, doc_id, score = line.split("\t")
            qrels[qid][doc_id] = int(score)
    return dict(qrels)


def _measure_strings(k_values: tuple[int, ...]) -> set[str]:
    cutoffs = ",".join(str(k) for k in k_values)
    return {f"ndcg_cut.{cutoffs}", f"recall.{cutoffs}", "recip_rank", "map"}


def _metric_keys(k_values: tuple[int, ...]) -> list[str]:
    return [f"ndcg_cut_{k}" for k in k_values] + [f"recall_{k}" for k in k_values] + ["recip_rank", "map"]


def _mean(metric: str, per_query: dict[str, dict[str, float]]) -> float:
    vals = [r[metric] for r in per_query.values() if metric in r]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    categories: Optional[dict[str, str]] = None,
) -> dict:
    """Score ``run`` against ``qrels`` with pytrec_eval.

    Queries with no entry in ``run`` at all (e.g. abstention questions with an
    empty qrels and nothing to retrieve) are excluded — pytrec_eval requires
    a run entry per judged query.

    ``categories``: optional {query_id: category} for a by_category breakdown
    (mirrors Judge.score's shape), e.g. LongMemEval question_type.
    """
    import pytrec_eval   # see the module-level note on why this is not top-level

    judged_qrels = {qid: docs for qid, docs in qrels.items() if qid in run}
    evaluator = pytrec_eval.RelevanceEvaluator(judged_qrels, _measure_strings(k_values))
    per_query = evaluator.evaluate(run)

    metric_keys = _metric_keys(k_values)
    overall = {m: _mean(m, per_query) for m in metric_keys}
    overall["queries"] = len(per_query)

    by_category: dict[str, dict] = {}
    if categories:
        grouped: dict[str, dict[str, dict]] = defaultdict(dict)
        for qid, res in per_query.items():
            grouped[categories.get(qid, "unknown")][qid] = res
        by_category = {
            cat: {**{m: _mean(m, res) for m in metric_keys}, "queries": len(res)}
            for cat, res in sorted(grouped.items())
        }

    return {"overall": overall, "by_category": by_category, "per_query": per_query}
