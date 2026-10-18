"""
evalgrid/systems/native_retrieval.py
------------------------------------
Swap the retriever *inside* a system's own query loop, keeping everything else
about that loop intact.

**What this is for.** The staged pipeline already answers the "which retriever
wins" question — but it answers it with the shared prompt over a flat corpus,
which deletes the system's retrieval strategy along with its retriever. That is
the right control, and it cannot tell you whether A-Mem's *link expansion* helps
a sparse retriever, or whether MemoryBank's *chronological regrouping* survives
a switch to SPLADE. Those are questions about the strategy, and answering them
means changing the ranking function and nothing else.

So this replaces the one call each system makes to rank its units, and leaves
the code around that call alone:

    A-Mem       retriever.search -> indices, THEN link expansion   (expansion kept)
    LightMem    embedding_retriever.search -> hits                 (no post-step)
    MemoryBank  our own cosine, THEN regrouping by bucket          (regrouping kept)
    Mem0        Memory.search -> results                           (no post-step)
    SimpleMem   vector_store.semantic_search -> the DENSE LEG only (planning,
                the BM25 and SQL legs, the merge priority and reflection kept)
    HyperMem    search_with_bm25 -> the LAYER SCORER only          (the
                topic->episode->fact funnel and its hard filtering kept)

All six are covered, but the last two are covered at a seam one level finer
than the other four, and the difference matters when reading the result.

    **SimpleMem** does not have "a retriever" to replace — it has a tri-path
    union (dense ∪ BM25 ∪ structured SQL) wrapped in LLM planning and
    reflection. Replacing the union would leave nothing of SimpleMem but its
    prompt, so what is replaced is the *dense leg's ranking function*
    (§3.3's ``R_sem = Top-n(cos(E(q), E(m)))``) and nothing else. Every
    sub-query the planner issues and every reflection round is routed through
    the substitute, because upstream funnels them all through the same
    ``_semantic_search``. A ``native_retriever: bm25`` here therefore does NOT
    make SimpleMem lexical — it makes both its lexical legs lexical, which is
    a leg-redundancy ablation and should be labelled as one.

    **HyperMem** ranks three times, once per layer, each restricted to the
    hyperedges under the layer above. The funnel is the contribution and the
    scorer is BM25, so the scorer is what is replaced — at all three layers,
    over the same per-layer candidate sets and the same searchable strings
    stage 3 indexes (including its token-repetition field weighting: fact
    content ×3, query_patterns ×2, keywords ×1). The hyperedge weights never
    enter a score upstream either, so the structure stays a pure hard filter.
    This is the one system whose swap is NOT ranked over ``iter_units``: the
    funnel's layer-1 candidates are *topics*, which the corpus excludes by
    default, so a units-based ranker would starve layer 1 and empty the funnel.
    See ``LayerRanker``.

**Ranking is over the system's own units**, read through
``SystemPlugin.iter_units`` — the same corpus the flat pipeline indexes. So a
native run with ``native_retriever: sparse`` and a flat ``--retriever sparse``
run rank an identical document set; what differs is only what the system does
with the ranking afterwards. That is what makes the pair interpretable.
(HyperMem excepted, as above: it ranks the funnel's own per-layer doc sets.)

Config, in a system's YAML (or via ``--config`` + ``--native-variant``):

    native_retriever:
      name: sparse                     # bm25 | dense | sparse | rrf | fusion | hybrid | multi_vector
      encoder: naver/splade-v3         # for sparse/hybrid
      embedder: Qwen/Qwen3-Embedding-0.6B   # for dense/fusion/hybrid
      base_url: http://localhost:8001/v1    # if the embedder/encoder is served
      alpha: 0.5                       # dense weight, fusion/hybrid (min-max only)
      rrf_k: 60                        # rrf; depth: N to truncate each leg first

Absent (the default), nothing is installed and every system retrieves exactly as
it was published.

**Both native stages install it, once per sample.**

    pipeline/retrieve.py --retriever native   owns its sample loop -> calls install()
    pipeline/answer.py --prompt native        hands the loop to runner.run
                                              -> install_per_sample() wraps load_memory

Per sample is not an optimisation, it is the correctness condition: a ranker is
built from ONE sample's units, and every sample's units are disjoint. Installing
once per process would leave sample 1's ranker serving samples 2..N, which fails
silently — the ids it returns are real ids, from the wrong sample — and
invalidates every row after the first.

**The run directory is named after the substitute, not the published encoder.**
``retriever_slug`` produces that name and every ``SystemPlugin.native_retriever_
name`` prefers it, so a SPLADE run cannot land in the dense run's folder. A
plugin that overrides ``native_retriever_name`` must keep that check.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

#: Marks a method already wrapped, so a re-install (reset() rebuilds the store)
#: cannot stack two rankers. Mirrors trace._WRAPPED.
_INSTALLED = "_evalgrid_native_retriever"


def build_from_config(config: dict):
    """Return a ``Retriever`` from a ``native_retriever`` block, or None.

    None means "not configured", which must leave the system untouched — an
    extension experiment that silently changed the baseline would make every
    other row in the table wrong.
    """
    block = (config or {}).get("native_retriever") or {}
    name = str(block.get("name") or "").strip()
    if not name:
        return None

    from evalgrid.benchmarks.retrievers import build_retriever

    # Only when set, never base_url=None: these forward **kwargs straight to the
    # backend constructor, and the local backends do not take one. Same shape as
    # pipeline/retrieve.py's emb_kwargs on the flat side.
    load_kwargs = {"base_url": block["base_url"]} if block.get("base_url") else {}

    kwargs: dict = {}
    if name in ("dense", "fusion", "rrf", "hybrid"):
        from evalgrid.models.factory import load_embedder
        emb = block.get("embedder")
        if not emb:
            raise ValueError(f"native_retriever.name={name!r} needs an 'embedder'")
        kwargs["embedder"] = load_embedder(emb, **load_kwargs)
    if name in ("sparse", "hybrid"):
        from evalgrid.models.factory import load_sparse_encoder
        enc = block.get("encoder")
        if not enc:
            raise ValueError(f"native_retriever.name={name!r} needs an 'encoder'")
        # Forwarded here too, not only to the dense half: the flat side passes
        # --emb-base-url to every loader, and a served SPLADE would otherwise be
        # silently loaded locally — a different run under the same name.
        kwargs["encoder"] = load_sparse_encoder(enc, **load_kwargs)
    if name == "multi_vector":
        from evalgrid.models.factory import load_multi_vector_model
        model = block.get("model") or block.get("encoder")
        if not model:
            raise ValueError("native_retriever.name='multi_vector' needs a 'model'")
        kwargs["model"] = load_multi_vector_model(model, **load_kwargs)
    if name in ("fusion", "hybrid") and block.get("alpha") is not None:
        kwargs["alpha"] = float(block["alpha"])
    if name == "rrf":
        # Both optional and both held fixed by the grid (rrf_k 60, depth unset =
        # fuse the full lists). Forwarded only when set so the retriever's own
        # defaults stay the single source of truth for them.
        if block.get("rrf_k") is not None:
            kwargs["rrf_k"] = int(block["rrf_k"])
        if block.get("depth") is not None:
            kwargs["depth"] = int(block["depth"])

    return build_retriever(name, **kwargs)


def block_from_spec(spec: str) -> dict:
    """Parse a compact retriever spec into a ``native_retriever`` block.

        bm25
        sparse:naver/splade-v3
        dense:Qwen/Qwen3-Embedding-4B@http://localhost:8002/v1
        rrf:sentence-transformers/all-MiniLM-L6-v2
        fusion:MODEL[@url] | hybrid:MODEL | multi_vector:MODEL

    Deliberately the SAME vocabulary as a spec's ``retrievers:`` list and
    ``run_all_retrieval.sh``'s ``build_flags``, so one string names the same
    retriever on the flat path and inside a system's own loop, and the two runs
    it produces sit in directories differing by the ``native-`` prefix alone.

    This exists so a swap needs no YAML file. ``--config`` REPLACES a system's
    default.yaml rather than merging with it, so expressing eight retrievers as
    configs meant eight near-identical copies of a 300-line file per system —
    48 derived files whose only real content was three lines each. The block is
    injected into the resolved config instead, which is also why the run
    directory still derives correctly: ``native_retriever_name`` reads the
    config, and the injection happens before any path is built.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty --native-retriever spec")
    name, _, rest = spec.partition(":")
    name = name.strip()
    model, _, url = rest.partition("@")
    model, url = model.strip(), url.strip()

    if name == "bm25":
        if model:
            raise ValueError(f"--native-retriever bm25 takes no model, got {spec!r}")
        return {"name": "bm25"}
    if name not in ("dense", "sparse", "rrf", "fusion", "hybrid", "multi_vector"):
        raise ValueError(
            f"unknown --native-retriever {spec!r}. Expected bm25, or one of "
            f"dense/sparse/rrf/fusion/hybrid/multi_vector as NAME:MODEL[@url]")
    if not model:
        raise ValueError(f"--native-retriever {name!r} needs a model: {name}:MODEL[@url]")

    block: dict = {"name": name}
    if name in ("dense", "rrf", "fusion"):
        block["embedder"] = model
    elif name == "sparse":
        block["encoder"] = model
    elif name == "multi_vector":
        block["model"] = model
    else:  # hybrid — one dual-representation model serves both halves, as on
           # the flat side, where retrieve.py passes the same object twice.
        block["embedder"] = block["encoder"] = model
    if url:
        block["base_url"] = url
    return block


def apply_spec(config: dict, spec: Optional[str]) -> dict:
    """Inject ``--native-retriever`` into a resolved config, in place.

    A no-op when ``spec`` is None, so an unswapped run is byte-identical to one
    that never knew the flag existed. An explicit flag WINS over a
    ``native_retriever:`` block already in the YAML — the flag is the more
    specific instruction, and a silent tie would make the run directory depend
    on which one the reader happened to notice.
    """
    if not spec:
        return config
    config["native_retriever"] = block_from_spec(spec)
    return config


def config_for_build(config: dict) -> dict:
    """``config`` minus the ``native_retriever`` block, for ``SystemPlugin.build``.

    ``native_retriever`` is HARNESS state: it names the retriever this run swaps
    into the system's query loop, and it is read by ``build_from_config`` /
    ``install`` / ``retriever_slug`` here — never by a system's own constructor.
    Most plugins ignore an unknown key, so leaking it was invisible; LightMem
    validates its config against ``BaseMemoryConfigs`` and rejects extras (by
    design — pydantic would otherwise drop a misspelt key in silence and run on
    the default), so every swapped lightmem cell died with "config has keys
    LightMem does not define: ['native_retriever']" before retrieving a sample.

    Stripped here rather than whitelisted in that one plugin: the key is not a
    LightMem knob, and the next strict plugin would rediscover the same bug.
    Shallow copy — the nested blocks are still shared with the caller, which is
    what the callers below want (they keep using ``system_config`` for the swap
    itself after handing the filtered copy to ``build``).
    """
    return {k: v for k, v in (config or {}).items() if k != "native_retriever"}


def retriever_slug(config: dict) -> Optional[str]:
    """A short name for this setting, for the run directory.

    Without it, a native run on SPLADE and one on the published encoder land in
    the same folder and the second silently overwrites the first — the exact
    failure ``SystemPlugin.native_retriever_name`` exists to prevent on the
    encoder axis.

    Built with ``lib.paths.retriever_name``, the SAME function that names a flat
    run's directory, so the two vocabularies are one: a swap to
    Qwen3-Embedding gives ``emb-Qwen_Qwen3-Embedding-0.6B``, which is the flat
    dense run's name with a ``native-`` prefix added by ``native_segment``. Two
    consequences worth having: the pair of directories differs by that prefix
    and nothing else, and swapping in a system's OWN published encoder
    reproduces its unswapped directory name — which is what makes that the
    natural no-op check on this machinery.
    """
    from evalgrid.lib.paths import retriever_name

    block = (config or {}).get("native_retriever") or {}
    name = str(block.get("name") or "").strip()
    if not name:
        return None
    tag = block.get("encoder") or block.get("embedder") or block.get("model")
    return retriever_name(name, tag) if tag else name


class UnitRanker:
    """Ranks a system's own memory units with an arbitrary Retriever.

    Built once per sample from ``plugin.iter_units``, because a sample's corpus
    is small and disjoint from every other sample's — the same assumption
    benchmarks/retrievers/base.py documents.

    The corpus is indexed (encoded) HERE, at construction — which ``install``
    does after ``load_memory`` and before any question is timed — and each
    ``rank`` only encodes the query and searches. Before this split every call
    re-encoded the whole corpus, so every native-swap retrieval latency carried
    a full corpus encode.
    """

    def __init__(self, retriever, units: list):
        self.retriever = retriever
        # BEIR rows, the shape every Retriever already consumes.
        self.corpus = [{"_id": u.unit_id, "title": "", "text": u.text,
                        "metadata": dict(getattr(u, "metadata", None) or {})}
                       for u in units]
        self.unit_ids = [u.unit_id for u in units]
        self._position = {uid: i for i, uid in enumerate(self.unit_ids)}
        self._index = asyncio.run(retriever.index(self.corpus)) if self.corpus else None

    def rank(self, query: str, k: Optional[int] = None) -> list[tuple[str, float]]:
        if not self.corpus:
            return []
        ranked = asyncio.run(self.retriever.search(query, self._index))
        return ranked[:k] if k else ranked

    def rank_positions(self, query: str, k: Optional[int] = None) -> list[int]:
        """Ranking as positional indices into the unit list.

        A-Mem's retriever returns indices into ``list(memories.values())`` and
        its link expansion subscripts that list, so a substitute has to speak
        the same language or the expansion would follow the wrong notes.
        """
        return [self._position[uid] for uid, _ in self.rank(query, k)
                if uid in self._position]


class LayerRanker:
    """Ranks a LAYERED candidate set — HyperMem's funnel — with a Retriever.

    HyperMem does not rank one corpus, it ranks three: topics, then the episodes
    under the retained topics, then the facts under the retained episodes. Its
    scorer is ``search_with_bm25(query, bm25, docs, doc_type, top_n)``, called
    once per layer with the SAME ``docs`` list and a different ``doc_type``
    filter — so a substitute is per-layer by construction, and this stands in
    for that function rather than for a corpus.

    Two things it deliberately does not do:

    * It does not read ``plugin.iter_units``. The funnel's layer-1 candidates
      are topics, which ``HyperMemPlugin.iter_units`` excludes by default (a
      topic is a title plus a cluster summary, not a self-contained unit), so a
      units-based ranker would rank nothing at layer 1 and the funnel would come
      back empty. The docs upstream hands the scorer ARE the candidate set, and
      ranking those is what keeps the ablation honest.
    * It does not re-derive the searchable text. It reads ``doc["searchable_
      text"]``, which ``HyperMemSystem._bm25_index`` attaches from upstream's own
      ``build_*_searchable_text`` helpers — the identical strings stage 3's BM25
      index is built over, field weighting included. Anything else would change
      the documents as well as the scorer and stop being a retriever swap.

    Corpora and their retriever indexes are built per ``doc_type`` and memoised,
    because all three layers share one ``docs`` list and only the filter
    differs. ``prepare`` builds all three up front (HyperMem's
    ``prepare_retrieval`` calls it before the first question is timed), so
    ``rank_layer`` only encodes the query; a layer missed there is still built
    lazily, as before. The memo is safe
    only because ``install`` builds a NEW LayerRanker per sample — see its
    docstring for why that is a correctness condition rather than an
    optimisation.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        #: doc_type -> (BEIR corpus rows, the payloads to hand back, retriever index)
        self._layers: dict[str, tuple[list[dict], list[Any], Any]] = {}

    def prepare(self, docs: list[dict]) -> None:
        """Index every layer present in ``docs`` now, outside the query clock."""
        for doc_type in sorted({d.get("type") for d in docs or [] if d.get("type")}):
            self._layer(docs, doc_type)

    def _layer(self, docs: list[dict], doc_type: str) -> tuple[list[dict], list[Any], Any]:
        if doc_type not in self._layers:
            # Positional ids, not the node ids: three layers share one docs list
            # and nothing guarantees a fact id cannot equal an episode id, so a
            # collision would silently hand back the wrong layer's payload.
            corpus, payload = [], []
            for doc in docs or []:
                if doc.get("type") != doc_type:
                    continue
                corpus.append({"_id": str(len(corpus)), "title": "",
                               "text": doc.get("searchable_text") or "",
                               "metadata": {"id": doc.get("id", "")}})
                payload.append(doc.get("data", doc))
            index = asyncio.run(self.retriever.index(corpus)) if corpus else None
            self._layers[doc_type] = (corpus, payload, index)
        return self._layers[doc_type]

    def rank_layer(self, query: str, docs: list[dict], doc_type: str,
                   top_n: int = 5) -> list[tuple[Any, float]]:
        """``search_with_bm25``'s contract: ``(doc["data"], score)``, best first."""
        corpus, payload, index = self._layer(docs, doc_type)
        if not corpus:
            return []
        ranked = asyncio.run(self.retriever.search(query, index))
        return [(payload[int(cid)], float(score)) for cid, score in ranked[:top_n]]


def load_units(plugin, sample_mem_dir: str) -> list:
    """The system's own units for one sample, in corpus order."""
    return list(plugin.iter_units(sample_mem_dir))


def make_ranker(plugin, retriever, sample_mem_dir: str):
    """The ranker object this system's installer expects.

    Everything but HyperMem ranks one flat corpus of the system's own units;
    HyperMem ranks the funnel's per-layer doc sets and never needs the units
    read at all (which also spares it a hypergraph.json parse per sample).
    """
    if getattr(plugin, "name", None) == "hypermem":
        return LayerRanker(retriever)
    return UnitRanker(retriever, load_units(plugin, sample_mem_dir))


def check_supported(plugin, config: dict) -> Optional[str]:
    """Validate that this system can take the configured swap. Returns the slug.

    Split out of ``install`` so a stage can fail before it builds a run
    directory and a manifest: ``install`` runs inside the per-sample loop, so
    without this a hypermem run with a ``native_retriever`` block would write
    its output dir, announce itself, and only then raise on sample one.
    """
    # Same defensive read as build_from_config: a bare "native_retriever:" in
    # YAML parses to None, not {}.
    if not ((config or {}).get("native_retriever") or {}).get("name"):
        return None
    name = getattr(plugin, "name", type(plugin).__name__)
    if name == "hypermem":
        kind = str(((config or {}).get("retrieval") or {}).get("retrieval_type", "rrf"))
        if kind != "keyword":
            raise NotImplementedError(
                f"native_retriever cannot be applied to hypermem with "
                f"retrieval_type={kind!r}. The installer replaces search_with_bm25, "
                f"which is only the sparse leg; under {kind!r} the funnel also scores "
                f"HyperMem's own propagated Qwen3 vectors and fuses the two with RRF, "
                f"so the substitute would be blended with the retriever it is meant to "
                f"replace. Set retrieval.retrieval_type: keyword in the swap's config "
                f"(the dense leg is what the swap is ablating anyway)."
            )
        # The cross-encoder is NOT blocked: it is a legitimate second stage over
        # whatever the first one produced, and the run dir names both axes. But
        # a swap measured with it on measures less than it looks like it does —
        # a cross-encoder re-ranking 100 candidates absorbs much of the
        # difference between two first-stage scorers. For a clean scorer
        # ablation set reranker.enabled: false as well.
    if name not in _INSTALLERS:
        raise NotImplementedError(
            f"native_retriever is not supported for {name!r}: no installer names the "
            f"call that ranks, so the block would be read and silently ignored — a "
            f"run that reported a swap it never applied. Supported: "
            f"{', '.join(sorted(_INSTALLERS))}. Add one in "
            f"evalgrid/systems/native_retrieval.py (see the module docstring for how "
            f"each system's seam was chosen)."
        )
    return retriever_slug(config)


def declare(plugin, config: dict) -> Optional[str]:
    """Validate the swap and tell ``plugin`` — BEFORE ``build`` — about it.

    ``check_supported`` with one side effect, and the call both native stages
    make: it returns that function's slug, and it records on the plugin whether
    a swap is coming.

    ``install`` cannot carry this: it runs per sample, long after the system was
    constructed, and what a system needs the warning FOR is construction. The
    case that forced it is SimpleMem, whose ``SimpleMemSystem`` loads a
    sentence-transformers encoder onto the GPU in its constructor and re-embeds
    every restatement on every ``load_memory`` — work whose only reader is the
    ``semantic_search`` this module then replaces. Under a swap that encoder is
    ~4 GB of resident weights and one forward pass per sample spent on vectors
    nothing will ever read, and on a node already hosting the memory server plus
    a served embedder it is what pushes the process over: the Qwen3-Embedding-0.6B
    cells of the 2026-08-21 smoke grid died in ``add_entries`` with
    ``CUDA out of memory``, in the retriever-swap run that had made those vectors
    dead weight.

    Set on the PLUGIN, not passed through ``config``: ``config_for_build`` strips
    the swap block precisely because a system's own constructor must not see
    harness keys (LightMem validates its config and rejects extras), so
    threading a new key through the same dict would rediscover that bug. The
    plugin is harness state already.

    Returns the slug, or None when no swap is configured — and then the flag is
    CLEARED rather than left alone, so a reused plugin cannot report a swap this
    run is not doing. Idempotent.
    """
    slug = check_supported(plugin, config)
    plugin.native_swap = slug is not None
    return slug


def install(system, plugin, config: dict, sample_mem_dir: str) -> Optional[str]:
    """Point ``system``'s ranking at the configured retriever. Returns its slug.

    **Call this once per sample, after ``load_memory``** — the units have to
    exist to be indexed, and each sample has its own disjoint set of them. The
    ranker is rebuilt and re-pointed every call for that reason: a ranker built
    from sample 1 still installed for sample 2 would rank every later question
    against the first sample's corpus, which fails silently (the ids it returns
    are real ids, just from the wrong sample) and invalidates every row after
    the first.

    Returns None when nothing is configured, which must leave the system
    untouched.
    """
    slug = check_supported(plugin, config)
    if slug is None:
        return None
    retriever = build_from_config(config)
    if retriever is None:
        return None

    # The live ranker lives on the system, in ONE place, and is replaced here
    # every sample. Every installer below reads it from here rather than closing
    # over it, so re-pointing is a single assignment and a wrapper installed for
    # an earlier sample cannot keep serving that sample's units.
    system._native_ranker = make_ranker(plugin, retriever, sample_mem_dir)
    _INSTALLERS[getattr(plugin, "name")](system)
    return slug


def install_per_sample(system, plugin, config: dict) -> Optional[str]:
    """``install`` for a stage that does not own its own sample loop.

    ``pipeline/retrieve.py`` walks samples itself and calls ``install`` directly.
    ``pipeline/answer.py --retriever native --prompt native`` hands the loop to ``benchmarks/runner.run``,
    which drives ``reset`` / ``load_memory`` / ``answer`` internally — so the
    only seam where this stage can see a sample boundary is ``load_memory``,
    whose argument is that sample's memory dir. Wrapping it is the same
    instance-rebinding trick ``install_replay`` in that module already uses.

    Returns the slug, or None when nothing is configured (and then nothing is
    wrapped at all).
    """
    slug = check_supported(plugin, config)
    if slug is None:
        return None

    original = system.load_memory

    def load_memory(path: str) -> None:
        original(path)
        # After, not before: the units have to be loaded to be indexed.
        install(system, plugin, config, path)

    system.load_memory = load_memory
    return slug


# ---------------------------------------------------------------------------
# Per-system installation
#
# Each rebinds exactly one method on the instance — the call that ranks — and
# nothing else, so whatever the system does with the ranking afterwards is
# untouched. Instance rebinding rather than subclassing, for the same reason
# evalgrid/lib/trace.py gives: vendor/ stays verbatim.
#
# None of them takes the ranker as an argument: they read ``system._native_
# ranker`` at CALL time, which ``install`` refreshes per sample. A wrapper that
# closed over a ranker would be installed once and then keep serving the first
# sample's units for the rest of the run — see install's docstring.
# ---------------------------------------------------------------------------

def _install_a_mem(system) -> None:
    """Replace the note search. Link expansion downstream is preserved.

    Upstream returns positional indices into ``list(memories.values())``; so does
    this, via ``rank_positions``. That is what lets ``retrieve_context`` walk
    each hit's ``links`` exactly as before — the expansion is A-Mem's method and
    the point of swapping only the ranker is to keep it.

    ``reset()`` rebuilds ``_amem``, so this re-wraps a fresh retriever object
    every sample; the guard only stops a double-install within one sample.
    """
    inner = getattr(system, "_amem", None)
    target = getattr(inner, "retriever", None)
    if target is None or getattr(target.search, _INSTALLED, False):
        return

    def search(query, k=10, *args, **kwargs):
        return system._native_ranker.rank_positions(query, k)

    setattr(search, _INSTALLED, True)
    target.search = search


def _install_memorybank(system) -> None:
    """Replace the cosine scoring; upstream's regrouping by bucket is preserved.

    MemoryBank's retrieve() sorts hits by ``source`` and concatenates those
    sharing one, so reading order is not ranking order. That regrouping is the
    part worth holding fixed while the ranker changes.

    ``ids`` goes in ranked and comes back out as the 5th return value, which is
    what reaches ``retrieved_unit_ids`` — the substituted ranker's ordering is
    the whole measurement here, so it must survive the regrouping rather than
    be replaced by it. See ``memorybank/system._regroup``.

    ``reset()`` does not rebind ``retrieve``, so the wrapper installed for
    sample 1 is the one that serves every later sample — which is exactly why it
    reads the ranker off the system instead of closing over it.
    """
    if getattr(system.retrieve, _INSTALLED, False):
        return
    original = system.retrieve

    def retrieve(question: str, k=None):
        ranker = system._native_ranker
        ids = [uid for uid, _ in ranker.rank(question, k or system.top_k)]
        return system.regroup_by_bucket(ids)

    setattr(retrieve, _INSTALLED, True)
    system.retrieve = retrieve
    system._native_retrieve_original = original


def _install_lightmem(system) -> None:
    """Replace the Qdrant search. LightMem has no post-retrieval step.

    Nothing to rebind: ``LightMemSystem.retrieve`` already reads
    ``self._native_ranker`` per call, which ``install`` has just refreshed.
    """


def _install_mem0(system) -> None:
    """Replace Mem0's dense top-k. No post-retrieval step, and — like lightmem —
    ``Mem0System.retrieve`` reads ``self._native_ranker`` per call."""


def _install_simplemem(system) -> None:
    """Replace the DENSE LEG only. Planning, the other two legs and reflection stay.

    ``VectorStore.semantic_search`` is the single call behind
    ``HybridRetriever._semantic_search``, which every semantic read goes through
    — the initial query, each of the planner's up-to-four sub-queries (including
    the ones dispatched on a ThreadPoolExecutor), and each reflection round's
    follow-up. Rebinding it therefore routes the whole semantic path through the
    substitute in one place, while ``_keyword_search`` (Tantivy full-text),
    ``_structured_search`` (the SQL metadata filter), the union's
    semantic > keyword > structured merge priority and the reflection loop are
    all untouched.

    That priority is ``_merge_and_deduplicate_entries``' (hybrid_retriever.py
    L409), which is a flat first-seen-wins dedup over an ``all_results`` list
    built semantic (L101) -> keyword (L109) -> structured (L114). It is NOT the
    ``structured > semantic > keyword`` that ``_merge_and_deduplicate`` (L326)
    names — that method is dead code in this module, called only from
    ``integrations/server/core/retriever.py``. The distinction matters here
    because this docstring is a claim about what the swap PRESERVES.

    ``asyncio.run`` inside a pool worker is fine: those threads have no running
    loop of their own.

    The entry map is cached ON THE RANKER, not on the system — the ranker is
    rebuilt per sample, so the cache expires exactly when the store's contents
    do. Caching it on the system would serve sample 1's entries forever, since
    ``reset()`` clears the LanceDB table but keeps the SimpleMemSystem (and so
    this wrapper) alive across every sample.
    """
    store = getattr(getattr(system, "_sys", None), "vector_store", None)
    if store is None:
        # Loud, not a silent return: a missing store means install() ran before
        # reset()/load_memory built the SimpleMemSystem, and the run would then
        # complete having reported a swap it never applied — the one failure
        # mode this whole module is written against.
        raise RuntimeError(
            "native_retriever for simplemem: no SimpleMemSystem vector_store yet. "
            "install() must run after reset() and load_memory()."
        )
    if getattr(store.semantic_search, _INSTALLED, False):
        return

    def semantic_search(query: str, top_k: int = 5):
        ranker = system._native_ranker
        entries = getattr(ranker, "_simplemem_entries", None)
        if entries is None:
            # Read back through the store rather than from iter_units: the
            # tri-path union merges MemoryEntry objects by identity downstream,
            # and the other two legs return the store's own.
            entries = {e.entry_id: e for e in system._sys.vector_store.get_all_entries()}
            ranker._simplemem_entries = entries
        return [entries[uid] for uid, _ in ranker.rank(query, top_k) if uid in entries]

    setattr(semantic_search, _INSTALLED, True)
    store.semantic_search = semantic_search


def _install_hypermem(system) -> None:
    """Replace the layer scorer. The topic->episode->fact funnel is preserved.

    The one installer that patches a MODULE attribute rather than rebinding a
    method on an instance, because upstream's scorer is a free function:
    ``hierarchical_retrieval`` and its two inner ``search_*_bm25`` closures
    resolve ``search_with_bm25`` from their module's globals at call time, and
    there is no object in between to rebind. Patching it therefore covers all
    three layers at once, which is what makes this a scorer swap and not a
    layer-1-only one.

    Only the BM25 leg is replaced, so this installer requires
    ``retrieval_type: keyword`` and ``check_supported`` rejects anything else.
    That guard is load-bearing now that the dense leg is wired: under ``rrf``
    the funnel scores BM25 ∪ dense and fuses the two by RRF, so patching
    ``search_with_bm25`` alone would leave HyperMem's own Qwen3 vectors ranking
    beside the substitute and fusing with it. The run would be neither a scorer
    swap nor native HyperMem, and nothing in the output would say so.

    ``bm25`` and ``docs`` keep arriving and ``docs`` is still what gets ranked;
    the BM25Okapi index itself simply goes unread. It is still built, on purpose:
    upstream guards several branches on ``bm25 and docs`` being truthy, and
    passing None would change which branch runs.
    """
    import hypermem.main.stage4_hypergraph_retrieval as stage4

    if getattr(stage4.search_with_bm25, _INSTALLED, False):
        return

    def search_with_bm25(query, bm25, docs, doc_type, top_n=5):
        return system._native_ranker.rank_layer(query, docs, doc_type, top_n)

    setattr(search_with_bm25, _INSTALLED, True)
    stage4.search_with_bm25 = search_with_bm25


_INSTALLERS: dict[str, Callable[[Any], None]] = {
    "a_mem": _install_a_mem,
    "memorybank": _install_memorybank,
    "lightmem": _install_lightmem,
    "mem0": _install_mem0,
    "simplemem": _install_simplemem,
    "hypermem": _install_hypermem,
}

#: Systems this module can swap. Anything else raises in ``install``.
SUPPORTED = tuple(_INSTALLERS)
