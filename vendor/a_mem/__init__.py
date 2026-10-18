"""
vendor/a_mem/__init__.py
------------------------
Not upstream code. Everything else in this directory is A-Mem verbatim from
WujiangXu/A-mem @ 0c8039f (2026-03-05); this file is the only addition.

A-Mem ships as a flat repo, not a package: `memory_layer.py`, `utils.py`,
`load_dataset.py` etc. sit at its root and import each other by their bare
top-level names (`from memory_layer import SimpleEmbeddingRetriever`). Vendored
under vendor/, that directory becomes the package `a_mem`, and those bare
imports no longer resolve — so this module appends its own directory to
sys.path, which is what makes upstream's internal imports work unmodified.

The path entry is *appended*, not prepended: A-Mem contributes generic names
(`utils`, `load_dataset`) to the top level, and appending keeps them from
shadowing anything already importable.

Import A-Mem's modules through the package, so each resolves once:

    import a_mem
    from a_mem import memory_layer        # same module object as `memory_layer`

`from a_mem import X` is routed to the top-level `X` and cached under both
names. Avoid `import a_mem.X` — that goes through the import system rather than
this hook and would load a second, independent copy of the module.
"""

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)


def __getattr__(name):
    """Resolve `from a_mem import <module>` to the top-level flat module."""
    if name.startswith("_"):
        raise AttributeError(name)
    module = importlib.import_module(name)
    sys.modules[f"{__name__}.{name}"] = module
    return module
