"""evalgrid — the staged memory-IR evaluation harness.

Importing the package runs the project bootstrap (loads .env, puts the repo root
and vendor/ on sys.path), so any `python -m evalgrid.pipeline.<stage>` has the
environment ready before the stage's own imports run.
"""

from evalgrid.lib import bootstrap  # noqa: F401 — imported for its side effects
