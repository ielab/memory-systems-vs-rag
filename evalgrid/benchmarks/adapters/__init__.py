import os

from evalgrid.benchmarks.adapters.longmemeval import LongMemEvalAdapter
from evalgrid.benchmarks.adapters.locomo import LoCoMoAdapter
from evalgrid.benchmarks.adapters.beam import BEAMAdapter

# Shared dataset registry — single source of truth for the ingest and judge stages.
# Datasets live at <repo root>/data/, four levels up from this file
# (evalgrid/benchmarks/adapters/__init__.py).
_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    "data",
)

DATASETS = {
    "longmemeval_s": os.path.join(_DATA_ROOT, "LongMemEval", "longmemeval_s_cleaned.json"),
    "longmemeval_m": os.path.join(_DATA_ROOT, "LongMemEval", "longmemeval_m_cleaned.json"),
    "longmemeval_oracle": os.path.join(_DATA_ROOT, "LongMemEval", "longmemeval_oracle.json"),
    "locomo": os.path.join(_DATA_ROOT, "LoCoMo", "locomo10.json"),
    # BEAM ships one parquet shard per tier, and the tier name is the target
    # token length of ONE conversation rather than a split of a shared corpus —
    # so each tier is its own dataset key with its own output tree, not a
    # --limit over a common one. The folder called "100K" is the tier the paper
    # calls 128K and measures ~120 K; the folder name is kept so a path is
    # reproducible from the download.
    "beam_100k": os.path.join(_DATA_ROOT, "BEAM", "data", "100K-00000-of-00001.parquet"),
    "beam_500k": os.path.join(_DATA_ROOT, "BEAM", "data", "500K-00000-of-00001.parquet"),
    "beam_1m": os.path.join(_DATA_ROOT, "BEAM", "data", "1M-00000-of-00001.parquet"),
}

#: dataset key -> the tier name BEAM's own release uses. Kept beside DATASETS
#: rather than derived from the key, because "beam_1m" -> "1M" and "beam_100k"
#: -> "100K" do not share a rule.
BEAM_TIERS = {"beam_100k": "100K", "beam_500k": "500K", "beam_1m": "1M"}


def build_adapter(dataset: str):
    """Return the adapter instance for a dataset key in DATASETS."""
    path = DATASETS[dataset]
    # DATASETS is the registry of keys the pipeline ACCEPTS, not of files that
    # are staged — longmemeval_m is listed but not shipped. Without this the
    # failure is a bare FileNotFoundError from json.load two frames down, after
    # the stage has already printed its banner and made its output dir.
    if not os.path.isfile(path):
        raise SystemExit(
            f"Dataset {dataset!r} is registered but its file is not present:\n"
            f"  {path}\n"
            f"Stage it there, or pick one of: "
            f"{', '.join(k for k, p in DATASETS.items() if os.path.isfile(p))}")
    if dataset.startswith("longmemeval"):
        return LongMemEvalAdapter(path)
    if dataset == "locomo":
        return LoCoMoAdapter(path)
    if dataset in BEAM_TIERS:
        return BEAMAdapter(path, BEAM_TIERS[dataset])
    raise ValueError(f"Unknown dataset: {dataset}")


__all__ = [
    "LongMemEvalAdapter",
    "LoCoMoAdapter",
    "BEAMAdapter",
    "BEAM_TIERS",
    "DATASETS",
    "build_adapter",
]
