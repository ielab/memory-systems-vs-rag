#!/usr/bin/env python3
"""
check_native_swap.py
====================
Verify that ``native_retriever:`` — the swap that replaces the ranking function
inside a system's own query loop (evalgrid/systems/native_retrieval.py) — is
actually installed, for every system that claims to support it.

The failures this exists to catch are all silent. A swap can be *configured* and
not applied, or applied to the wrong sample's units, and in every case the run
completes, writes a full run.tsv, and reports nothing wrong — the ids it returns
are real ids. Only the numbers are wrong, and only against a baseline nobody
re-derives. So the checks here are:

  installed         install() points the system's ranking at the substitute
  per-sample        a second install() re-points it at the SECOND sample's units.
                    This is the correctness condition, not an optimisation: a
                    ranker is built from one sample's units and every sample's
                    units are disjoint, so a ranker that survived into sample 2
                    would rank every later question against sample 1's corpus.
  strategy kept     the system's post-retrieval step still runs on the result —
                    A-Mem's link expansion is fed positional indices, MemoryBank
                    still regroups by bucket. Swapping those away would make this
                    a different system rather than a retriever experiment.
  named             the run dir is named after the SUBSTITUTE, in the same
                    vocabulary a flat run uses, so a SPLADE run cannot land in
                    the published encoder's directory and the two are comparable
                    by their path.
  rejected          a system with no installer FAILS rather than silently
                    ignoring the block. All six have one now, so this guards the
                    next system added, not a structural exclusion.

SimpleMem and HyperMem are swapped at a finer seam than the other four — the
dense leg of the tri-path union, and the funnel's per-layer scorer — so they get
checks the others do not: that the OTHER legs and the funnel's layer separation
survive, and (HyperMem) that layer 1's topics are ranked at all, since they are
absent from the unit corpus every other system's ranker is built from.

No models and no stores are loaded: the retriever is stubbed and units come from
a stub plugin, because none of the above is a question about a model. What it
does exercise is the real ``install`` / ``install_per_sample`` / ``check_
supported`` / ``retriever_slug`` and the real plugins' ``native_retriever_name``.

Usage:
  python -m evalgrid.tools.check_native_swap          # exit 0 = every check passed
  python -m evalgrid.tools.check_native_swap -v       # show each check
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

from evalgrid.lib.paths import retriever_name
from evalgrid.systems import native_retrieval as nr
from evalgrid.systems.registry import get_plugin

#: The swap every check is run under. sparse/SPLADE because it is maximally
#: unlike every system's published dense encoder — nothing can coincidentally
#: pass by still using its own retriever.
SWAP = {"native_retriever": {"name": "sparse", "encoder": "naver/splade-v3"}}
SWAP_DIR = retriever_name("sparse", "naver/splade-v3")

#: HyperMem's swap needs one extra key. The installer replaces search_with_bm25,
#: which is only the sparse leg, so ``check_supported`` refuses to run it under
#: the shipped ``retrieval_type: rrf`` — the substitute would be RRF-fused with
#: the propagated dense vectors it is meant to replace. Setting keyword here is
#: what the guard's own error message instructs, and exercising the guard's happy
#: path is the point: the refusal itself is checked separately below.
HM_SWAP = dict(SWAP, retrieval={"retrieval_type": "keyword"})

#: Where each supported system keeps its published encoder, and what that encoder
#: is named on the flat side. Used for the "an unswapped config is unchanged"
#: half of the naming check.
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
PUBLISHED = {
    "a_mem": {"embedder": {"model": MINILM}},
    "mem0": {"embedder": {"model": MINILM}},
    "memorybank": {"embedder": {"model": MINILM}},
    "lightmem": {"text_embedder": {"configs": {"model": MINILM}}},
    "simplemem": {"embedder": {"model": MINILM}},
}

#: HyperMem names its unswapped runs after retrieval_type, not an encoder, so it
#: sits outside PUBLISHED's "an unswapped config names its embedder" check and
#: gets its own pair below.
HYPERMEM_PUBLISHED = {"retrieval": {"retrieval_type": "keyword"}}


class _Unit:
    def __init__(self, uid: str):
        self.unit_id, self.text, self.metadata = uid, f"text of {uid}", {}


class _StubRetriever:
    """Ranks in corpus order — enough to see WHICH corpus was indexed."""

    async def index(self, corpus, **kwargs):
        return corpus

    async def search(self, query, index):
        return [(doc["_id"], 1.0 / (i + 1)) for i, doc in enumerate(index)]


class _StubPlugin:
    """``sample_mem_dir`` doubles as the sample id, so each sample's units differ."""

    def __init__(self, name: str):
        self.name = name

    def iter_units(self, sample_mem_dir: str):
        return [_Unit(f"{sample_mem_dir}-u1"), _Unit(f"{sample_mem_dir}-u2")]


class _Checks:
    def __init__(self, verbose: bool):
        self.verbose, self.failed = verbose, []

    def __call__(self, name: str, got, want) -> bool:
        ok = got == want
        if not ok:
            self.failed.append(name)
            print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")
        elif self.verbose:
            print(f"  ok    {name}")
        return ok


def _ranked(system) -> list[str]:
    return [uid for uid, _ in system._native_ranker.rank("q")]


def _check_lightmem_like(check, name: str) -> None:
    """lightmem / mem0: ``retrieve`` reads ``self._native_ranker`` per call."""
    system, plugin = SimpleNamespace(top_k=2), _StubPlugin(name)
    nr.install(system, plugin, SWAP, "s1")
    check(f"{name}: installed for sample 1", _ranked(system), ["s1-u1", "s1-u2"])
    nr.install(system, plugin, SWAP, "s2")
    check(f"{name}: re-pointed for sample 2", _ranked(system), ["s2-u1", "s2-u2"])


def _check_memorybank(check) -> None:
    """The wrapper installed for sample 1 is the one that serves every sample."""
    seen: dict = {}

    class MB:
        top_k = 2

        def retrieve(self, question, k=None):
            return "UPSTREAM (not swapped)"

        def regroup_by_bucket(self, unit_ids):
            seen["ids"] = unit_ids            # the strategy that must survive
            return unit_ids

    mb, plugin = MB(), _StubPlugin("memorybank")
    nr.install(mb, plugin, SWAP, "s1")
    mb.retrieve("q")
    check("memorybank: installed for sample 1", seen["ids"], ["s1-u1", "s1-u2"])
    check("memorybank: regrouping still runs", "ids" in seen, True)
    nr.install(mb, plugin, SWAP, "s2")        # reset() does NOT rebind retrieve
    mb.retrieve("q")
    check("memorybank: re-pointed for sample 2", seen["ids"], ["s2-u1", "s2-u2"])


def _check_a_mem(check) -> None:
    """A-Mem's link expansion subscripts ``list(memories.values())``, so the
    substitute must return POSITIONS, not ids."""

    def fresh():
        return SimpleNamespace(retriever=SimpleNamespace(
            search=lambda q, k=10: "UPSTREAM (not swapped)"))

    am, plugin = SimpleNamespace(_amem=fresh()), _StubPlugin("a_mem")
    nr.install(am, plugin, SWAP, "s1")
    check("a_mem: search returns positions for link expansion",
          am._amem.retriever.search("q", 2), [0, 1])
    am._amem = fresh()                         # reset() rebuilds _amem
    nr.install(am, plugin, SWAP, "s2")
    check("a_mem: re-wrapped after reset", am._amem.retriever.search("q", 2), [0, 1])
    check("a_mem: re-pointed for sample 2", _ranked(am), ["s2-u1", "s2-u2"])


def _check_simplemem(check) -> None:
    """The DENSE LEG is replaced; planning, the keyword/SQL legs and reflection stay.

    Modelled on upstream's shape: HybridRetriever._semantic_search is the only
    caller of vector_store.semantic_search, and the other two legs go elsewhere.
    """
    class Entry:
        def __init__(self, entry_id):
            self.entry_id = entry_id

    class Store:
        def __init__(self, sample):
            self.sample = sample

        def semantic_search(self, query, top_k=5):
            return "UPSTREAM (not swapped)"

        def get_all_entries(self):
            return [Entry(f"{self.sample}-u1"), Entry(f"{self.sample}-u2")]

    store = Store("s1")
    system = SimpleNamespace(_sys=SimpleNamespace(vector_store=store))
    plugin = _StubPlugin("simplemem")

    nr.install(system, plugin, SWAP, "s1")
    check("simplemem: dense leg swapped for sample 1",
          [e.entry_id for e in store.semantic_search("q", 2)], ["s1-u1", "s1-u2"])

    # reset() clears the LanceDB table but keeps the SimpleMemSystem, so the
    # wrapper installed for sample 1 is the one serving sample 2 — and its
    # entry map must expire with the ranker, not outlive it.
    store.sample = "s2"
    nr.install(system, plugin, SWAP, "s2")
    check("simplemem: re-pointed for sample 2 (entry map not stale)",
          [e.entry_id for e in store.semantic_search("q", 2)], ["s2-u1", "s2-u2"])
    check("simplemem: top_k still governs the leg",
          [e.entry_id for e in store.semantic_search("q", 1)], ["s2-u1"])


def _check_hypermem(check) -> None:
    """The layer SCORER is replaced at all three layers; the funnel is untouched.

    Ranks the docs list upstream hands the scorer — NOT plugin.iter_units, whose
    corpus excludes topics and would starve layer 1.
    """
    import sys as _sys
    from types import ModuleType

    # Stand in for vendor/hypermem's stage 4, whose real import needs nltk. Saved
    # and restored below: leaving a stub package in sys.modules would make every
    # later check import it instead of the real one.
    names = ("hypermem", "hypermem.main", "hypermem.main.stage4_hypergraph_retrieval")
    saved = {n: _sys.modules.get(n) for n in names}
    mod = ModuleType(names[-1])
    mod.search_with_bm25 = lambda **kw: "UPSTREAM (not swapped)"
    for name in names[:-1]:
        _sys.modules[name] = ModuleType(name)
    _sys.modules[names[-1]] = mod
    try:
        _check_hypermem_body(check, mod)
    finally:
        for name, was in saved.items():
            if was is None:
                _sys.modules.pop(name, None)
            else:
                _sys.modules[name] = was


def _check_hypermem_body(check, mod) -> None:
    import sys as _sys

    def docs_for(sample):
        return [
            {"id": f"{sample}-t1", "type": "topic", "data": {"id": f"{sample}-t1"},
             "searchable_text": "topic one"},
            {"id": f"{sample}-e1", "type": "episode", "data": {"id": f"{sample}-e1"},
             "searchable_text": "episode one"},
            {"id": f"{sample}-f1", "type": "fact", "data": {"id": f"{sample}-f1"},
             "searchable_text": "fact one"},
            {"id": f"{sample}-f2", "type": "fact", "data": {"id": f"{sample}-f2"},
             "searchable_text": "fact two"},
        ]

    system, plugin = SimpleNamespace(), _StubPlugin("hypermem")
    nr.install(system, plugin, HM_SWAP, "s1")

    scorer = _sys.modules[mod.__name__].search_with_bm25
    def ranked(sample, doc_type, top_n=5):
        return [d["id"] for d, _ in scorer(query="q", bm25=None, docs=docs_for(sample),
                                          doc_type=doc_type, top_n=top_n)]

    # Layer 1 is the one a units-based ranker would silently empty.
    check("hypermem: layer 1 (topics) ranked", ranked("s1", "topic"), ["s1-t1"])
    check("hypermem: layer 2 (episodes) ranked", ranked("s1", "episode"), ["s1-e1"])
    check("hypermem: layer 3 (facts) ranked", ranked("s1", "fact"), ["s1-f1", "s1-f2"])
    check("hypermem: top_n still cuts the layer", ranked("s1", "fact", 1), ["s1-f1"])
    check("hypermem: layers do not leak into each other",
          ranked("s1", "topic") + ranked("s1", "fact"), ["s1-t1", "s1-f1", "s1-f2"])

    nr.install(system, plugin, HM_SWAP, "s2")
    check("hypermem: re-pointed for sample 2 (layer memo not stale)",
          ranked("s2", "fact"), ["s2-f1", "s2-f2"])

    # The refusal is half the contract: under the SHIPPED config (rrf) the swap
    # must not run at all, or the substitute is fused with what it replaces.
    try:
        nr.install(system, plugin, SWAP, "s3")   # no retrieval block -> rrf default
        check("hypermem: a swap under rrf is refused", "no error", "NotImplementedError")
    except NotImplementedError as e:
        check("hypermem: a swap under rrf is refused, naming the fix",
              "retrieval_type: keyword" in str(e), True)


def _check_answer_seam(check) -> None:
    """the native answer path hands its loop to runner.run, so load_memory is the seam."""
    loaded: list[str] = []

    class Sys:
        top_k = 2

        def load_memory(self, path):
            loaded.append(path)

    system, plugin = Sys(), _StubPlugin("mem0")
    nr.install_per_sample(system, plugin, SWAP)
    system.load_memory("s1")
    check("answer seam: installed on load 1", _ranked(system), ["s1-u1", "s1-u2"])
    system.load_memory("s2")
    check("answer seam: re-pointed on load 2", _ranked(system), ["s2-u1", "s2-u2"])
    check("answer seam: original load_memory still called", loaded, ["s1", "s2"])


def _check_naming(check) -> None:
    for name, published in PUBLISHED.items():
        plugin = get_plugin(name)
        check(f"{name}: run dir names the substitute",
              plugin.native_retriever_name({**published, **SWAP}), SWAP_DIR)
        check(f"{name}: unswapped run dir is unchanged",
              plugin.native_retriever_name(dict(published)),
              retriever_name("dense", MINILM))
    # Swapping a system to its OWN published encoder must reproduce its
    # unswapped directory — the no-op check on this whole mechanism.
    check("a_mem: swapping in its own encoder is a naming no-op",
          get_plugin("a_mem").native_retriever_name(
              {**PUBLISHED["a_mem"],
               "native_retriever": {"name": "dense", "embedder": MINILM}}),
          retriever_name("dense", MINILM))
    # HyperMem names by retrieval_type rather than by encoder, and overrides
    # native_retriever_name — so it needs its own check that the override still
    # lets a substitute win, or a SPLADE funnel would land in bm25/.
    hyper = get_plugin("hypermem")
    check("hypermem: run dir names the substitute",
          hyper.native_retriever_name({**HYPERMEM_PUBLISHED, **SWAP}), SWAP_DIR)
    check("hypermem: unswapped run dir is unchanged",
          hyper.native_retriever_name(dict(HYPERMEM_PUBLISHED)), "bm25")


def _check_simplemem_legs(check) -> None:
    """A leg ablation must key its own run dir, and must not silently no-op a swap.

    ``retrieval.legs`` changes what SimpleMem retrieves as surely as a swapped
    ranker does, so a legs-[semantic] run sharing a directory with the
    full-union run would overwrite it with the same silence this whole file
    exists to catch.
    """
    from evalgrid.systems.simplemem.plugin import legs_tag, parse_legs

    plugin = get_plugin("simplemem")
    published = PUBLISHED["simplemem"]
    full = retriever_name("dense", MINILM)

    check("simplemem legs: default is all three", parse_legs({}), frozenset(
        {"semantic", "keyword", "structured"}))
    check("simplemem legs: all three = no tag, unchanged dir",
          plugin.native_retriever_name(dict(published)), full)
    check("simplemem legs: dense-only keys its own dir",
          plugin.native_retriever_name({**published, "retrieval": {"legs": ["semantic"]}}),
          f"{full}--legs-semantic")
    check("simplemem legs: lexical-only keys its own dir",
          plugin.native_retriever_name({**published, "retrieval": {"legs": ["keyword"]}}),
          f"{full}--legs-keyword")
    # Order in the YAML must not produce two directories for one subset.
    check("simplemem legs: tag order is canonical, not the YAML's",
          legs_tag({"retrieval": {"legs": ["structured", "semantic"]}}),
          "legs-semantic+structured")
    check("simplemem legs: composes with a swap",
          plugin.native_retriever_name({**published, **SWAP,
                                        "retrieval": {"legs": ["semantic"]}}),
          f"{SWAP_DIR}--legs-semantic")
    for bad, why in (({"legs": ["dense"]}, "unknown leg"), ({"legs": []}, "empty")):
        try:
            parse_legs({"retrieval": bad})
            check(f"simplemem legs: {why} rejected", "no error", "ValueError")
        except ValueError:
            check(f"simplemem legs: {why} rejected", "ValueError", "ValueError")


def _check_declare(check) -> None:
    """The swap must reach the PLUGIN before build, and SimpleMem must act on it.

    ``install`` is per-sample and runs after construction, which is too late for
    a system that builds an index in its constructor. SimpleMem does: it loads a
    sentence-transformers encoder eagerly and re-embeds on every ``load_memory``,
    and under a swap the only reader of those vectors is the ``semantic_search``
    the swap replaces. Left in, that is ~4 GB resident and a forward pass per
    sample spent on nothing — it is what killed two cells of the 2026-08-21 smoke
    grid with CUDA OOM. So the flag has to arrive one stage earlier than the rest
    of this file's machinery, and the links in that chain are what is checked
    here: declare -> plugin.native_swap -> the config build hands the system.

    The last link (the system reading the key) is not exercised here for the
    reason at the top of this file — constructing one would load vendor/simplemem
    and a real encoder, which is precisely what a swapped run must not do.
    """
    import json as _json

    a_mem = get_plugin("a_mem")
    plugin = get_plugin("simplemem")
    check("declare: a plugin nobody declared to is unswapped", a_mem.native_swap, False)
    check("declare: returns the slug, and records the swap on the plugin",
          (nr.declare(plugin, dict(SWAP)), plugin.native_swap), (SWAP_DIR, True))

    # Plugins are cached per process, so a flag left set by one run's config
    # would be read by the next system built in the same process.
    check("declare: an unswapped config CLEARS the flag, not just leaves it",
          (nr.declare(plugin, {}), plugin.native_swap), (None, False))

    # build's inner import, stubbed: what is under test is the config it hands
    # the system, and constructing a real one would load the encoder this whole
    # check exists to avoid.
    seen: dict = {}

    class _StubSystem:
        def __init__(self, config, generator):
            seen["config"] = config

    stub_mod = SimpleNamespace(SimpleMemLongMemEval=_StubSystem)
    real_mod = sys.modules.get("evalgrid.systems.simplemem.longmemeval")
    sys.modules["evalgrid.systems.simplemem.longmemeval"] = stub_mod
    try:
        caller_config = _json.loads(_json.dumps(PUBLISHED["simplemem"]))
        nr.declare(plugin, dict(SWAP))
        plugin.build(caller_config, dataset="longmemeval_s", generator=None, phase="answer")
        check("declare: build hands the swapped system native_swap",
              seen["config"].get("native_swap"), True)
        check("declare: the caller's config is not mutated",
              "native_swap" in caller_config, False)

        nr.declare(plugin, {})
        plugin.build(caller_config, dataset="longmemeval_s", generator=None, phase="answer")
        check("declare: an unswapped build's config is untouched",
              "native_swap" in seen["config"], False)
    finally:
        if real_mod is None:
            sys.modules.pop("evalgrid.systems.simplemem.longmemeval", None)
        else:
            sys.modules["evalgrid.systems.simplemem.longmemeval"] = real_mod
        plugin.native_swap = False

    # The placeholder encoder itself: documents get a nominal vector, queries
    # raise. A zero query vector would rank by nothing while every file
    # downstream still named the substitute.
    from evalgrid.systems.simplemem.system import _UnusedEncoder

    enc = _UnusedEncoder()
    vecs = enc.encode_documents(["a", "b"])
    check("declare: placeholder vectors match the declared width",
          (vecs.shape, _UnusedEncoder.dimension), ((2, 8), 8))
    for call, how in ((lambda: enc.encode_single("q", is_query=True), "encode_single"),
                      (lambda: enc.encode_query(["q"]), "encode_query")):
        try:
            call()
            check(f"declare: {how} raises rather than ranking by placeholders",
                  "no error", "RuntimeError")
        except RuntimeError:
            check(f"declare: {how} raises rather than ranking by placeholders",
                  "RuntimeError", "RuntimeError")


def _check_rejected(check) -> None:
    """A system with no installer must FAIL, not quietly ignore the block.

    All six memory systems now have one, so this is the guard for the next one
    added: a config that names a swap and a plugin that cannot apply it would
    otherwise produce a run reporting a substitution it never made.
    """
    try:
        nr.check_supported(_StubPlugin("no_such_system"), SWAP)
        check("unknown system: swap rejected", "no error", "NotImplementedError")
    except NotImplementedError:
        check("unknown system: swap rejected", "NotImplementedError", "NotImplementedError")
    check("no native_retriever block = no swap",
          nr.check_supported(_StubPlugin("a_mem"), {}), None)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true", help="Show every check, not only failures")
    args = ap.parse_args()

    check = _Checks(args.verbose)
    real_build = nr.build_from_config
    nr.build_from_config = lambda config: _StubRetriever()   # no model loading
    try:
        print(f"Swap under test : {SWAP['native_retriever']} -> dir segment {SWAP_DIR!r}")
        print(f"Supported       : {', '.join(nr.SUPPORTED)}\n")
        for name in ("lightmem", "mem0"):
            _check_lightmem_like(check, name)
        _check_memorybank(check)
        _check_a_mem(check)
        _check_simplemem(check)
        _check_hypermem(check)
        _check_answer_seam(check)
        _check_naming(check)
        _check_simplemem_legs(check)
        _check_declare(check)
        _check_rejected(check)
    finally:
        nr.build_from_config = real_build

    if check.failed:
        print(f"\n{len(check.failed)} check(s) FAILED: {check.failed}")
        return 1
    print("\nAll checks passed — the swap installs, re-points per sample, keeps each "
          "system's strategy, and names its own run dir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
