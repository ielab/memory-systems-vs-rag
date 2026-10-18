from evalgrid.benchmarks.judges.base import Judge
from evalgrid.benchmarks.judges.beam import BEAMJudge
from evalgrid.benchmarks.judges.locomo import LoCoMoJudge
from evalgrid.benchmarks.judges.longmemeval import LongMemEvalJudge


def build_judge(dataset: str, judge, **kwargs) -> Judge:
    """Return the canonical judge for a dataset key (see adapters.DATASETS)."""
    if dataset.startswith("longmemeval"):
        return LongMemEvalJudge(judge, **kwargs)
    if dataset == "locomo":
        return LoCoMoJudge(judge, **kwargs)
    if dataset.startswith("beam"):
        # Graded, not pass/fail — its metrics file records
        # "metric": "mean_rubric_score". See judges/beam.py.
        return BEAMJudge(judge, **kwargs)
    raise ValueError(f"No judge defined for dataset: {dataset}")


__all__ = ["Judge", "BEAMJudge", "LoCoMoJudge", "LongMemEvalJudge", "build_judge"]
