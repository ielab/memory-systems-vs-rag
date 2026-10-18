# LOCAL CHANGE (the only patch to this vendored tree). Upstream reads its own
# version from installed-package metadata:
#
#     import importlib.metadata
#     __version__ = importlib.metadata.version("mem0ai")
#
# which raises PackageNotFoundError for a copy that lives on sys.path but was
# never pip-installed — killing `import mem0` at the first line. The literal
# below is the version this directory actually holds. Update it whenever
# vendor/mem0/ is re-pinned, and record the change in vendor/VERSIONS.md.
#
# Still read by mem0/memory/telemetry.py:39, so the attribute must exist.
__version__ = "0.1.118"

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
