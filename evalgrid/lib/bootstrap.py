"""
evalgrid/lib/bootstrap.py
-------------------------
Project bootstrap, imported for its side effects: it puts the repo root (and
vendor/) on sys.path and loads the project's .env, so any pipeline stage can
`from evalgrid... import` / `import lightmem...` and see the same environment.
Pulled in by evalgrid/__init__.py, so it runs before any stage body when a
module is launched as `python -m evalgrid.pipeline.<stage>`.

Only third-party code needs the vendor/ path entry: our own adapters live in
evalgrid/systems/<name>/ and import as ordinary evalgrid submodules. vendor/ is
prepended, so a vendored copy wins over a same-named pip install (e.g. mem0),
which is what pins the version the results were produced with.
"""

import os
import sys

from evalgrid.lib.paths import ROOT

# Repo root on the path so `evalgrid` imports as a top-level package.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# vendor/ on the path so vendored upstream systems import under their own
# top-level names (`lightmem.*`, ...) without being installed.
_VENDOR = os.path.join(ROOT, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

# Load .env (OPENAI_API_KEY etc.). dotenv is a hard dep of the pipeline; import
# it lazily so paths.py stays importable for tooling/tests without it.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except ModuleNotFoundError:  # pragma: no cover - only on a stripped-down env
    pass
