"""
evalgrid/systems/registry.py
----------------------------
Maps a --system value to its SystemPlugin.

Adding a memory system means adding one line here and one package under
evalgrid/systems/. No pipeline stage changes.

Entries are (module path, class name) rather than imported classes so the
registry stays importable without any system's dependencies installed —
``available_systems()`` can list ten systems on a machine that can only run one.
The module is imported lazily, on first ``get_plugin`` for that name.
"""

from __future__ import annotations

import importlib

from evalgrid.systems.base import MemoryUnit, SystemPlugin, shared_answer_prompt

# name -> (module, class)
_REGISTRY: dict[str, tuple[str, str]] = {
    "a_mem": ("evalgrid.systems.a_mem.plugin", "AMemPlugin"),
    "hypermem": ("evalgrid.systems.hypermem.plugin", "HyperMemPlugin"),
    "lightmem": ("evalgrid.systems.lightmem.plugin", "LightMemPlugin"),
    "mem0": ("evalgrid.systems.mem0.plugin", "Mem0Plugin"),
    "memorybank": ("evalgrid.systems.memorybank.plugin", "MemoryBankPlugin"),
    "simplemem": ("evalgrid.systems.simplemem.plugin", "SimpleMemPlugin"),
}

# Corpus sources that are not memory systems: the raw-turn baselines have no
# construction stage and no plugin, but they are legal --system values for the
# stages downstream of qrels_turns. Listed so error messages can say so instead
# of "unknown system".
#
# all_turns  both speakers, one doc per turn — the published Naive RAG baseline.
# user_turns user turns only — the ablation.
# pair_turns both speakers, one doc per (user, assistant) exchange — the matched
#            control for the systems that build memory a pair at a time.
# Which one a run used is recoverable from the path, which is why these are
# three names and not a flag.
PSEUDO_SYSTEMS = frozenset({"all_turns", "user_turns", "pair_turns"})

_cache: dict[str, SystemPlugin] = {}


def available_systems() -> list[str]:
    """Registered system names, sorted. Cheap — imports nothing."""
    return sorted(_REGISTRY)


def get_plugin(name: str) -> SystemPlugin:
    """Return the plugin for ``name``, importing its module on first use."""
    if name in _cache:
        return _cache[name]
    if name not in _REGISTRY:
        if name in PSEUDO_SYSTEMS:
            raise KeyError(
                f"{name!r} is a raw-corpus baseline, not a memory system: it has no "
                f"construction stage and no plugin. Build its corpus with "
                f"evalgrid.pipeline.qrels_turns instead of evalgrid.pipeline.ingest."
            )
        raise KeyError(
            f"Unknown system {name!r}. Registered: {', '.join(available_systems())}. "
            f"Add it to _REGISTRY in evalgrid/systems/registry.py."
        )
    module_path, class_name = _REGISTRY[name]
    plugin = getattr(importlib.import_module(module_path), class_name)()
    if plugin.name != name:
        raise ValueError(
            f"plugin {class_name}.name is {plugin.name!r} but it is registered as "
            f"{name!r} — the registry key keys the output tree, so they must match."
        )
    _cache[name] = plugin
    return plugin


__all__ = [
    "MemoryUnit",
    "SystemPlugin",
    "shared_answer_prompt",
    "available_systems",
    "get_plugin",
    "PSEUDO_SYSTEMS",
]
