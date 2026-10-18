"""
evalgrid/lib/paths.py
--------------------
The single source of truth for the outputs/ directory layout, plus model/
retriever slugging. Pure stdlib (no .env, no heavy imports) so it can be
imported and unit-tested in isolation.

The layout (<mem-seg> = <mem-model-slug>, plus a "--<mem-variant>" suffix when
ingest ran with a non-default parameter set, e.g. ...-2507--rate0.4-stm768):
  outputs/_memory/<system>/<dataset>/<mem-seg>/
  outputs/_qrels/<system>/<dataset>/<mem-seg>/<labeler-segment>/
  outputs/_retrieval/<system>/<dataset>/<mem-seg>/<labeler-segment>/<retriever>/
  outputs/_answers/<system>/<dataset>/<mem-seg>/<retriever>/<gen-seg>/<topk-seg>/
  outputs/_answers/<system>/<dataset>/<mem-seg>/oracle/<labeler-segment>/<gen-seg>/all/

<gen-seg> = <gen-model-slug>, plus a "--<prompt>-prompt" suffix when the run
answered under a non-shared prompt (see gen_segment). <retriever> is "native"
for a system's own query-time loop, which is how the native pipeline's runs and
answers stay separate from the flat-corpus ones.

<mem-variant> names an ingest-time parameter set (it suffixes every stage's
mem segment, so two ingest configs never share — and silently clobber — a
memory dir or anything derived from it). Parameters that only affect
retrieval/answering already have their own path segment and must NOT go in the
variant. No variant (None/"") keeps the historical paths, so pre-variant runs
remain valid as the implicit default.

<labeler-segment> is the labeler model slug, plus a "-retrieval" suffix when
the qrels builder was run with --candidate-source retrieval, so the two labeling
strategies never share (and silently clobber) a directory.

Retriever answers carry no <labeler-segment> — they're built from the retrieval
run + corpus + generator, none of which depend on the labeler, so one set of
answers is shared across labeling experiments. Only oracle answers (facts read
straight from the labeler's qrels) are keyed by <labeler-segment>.

Prefer the ExperimentPaths object over the bare functions: it's keyword-only and
carries every key together, so a caller can't silently drop one and point a
later stage at a directory ingestion never wrote.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# evalgrid/lib/paths.py -> evalgrid/lib -> evalgrid -> <repo root>
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Retrievers that need a model (vs. bm25, which is model-free); the prefix is
# how the model-backed run dirs are named (e.g. emb-<model>, sparse-<model>).
MODEL_BASED_RETRIEVERS = {"dense", "sparse", "multi_vector", "fusion", "rrf", "hybrid"}
# "fusion" and "rrf" are the SAME two legs (BM25 + dense) under different
# combiners, so they must not share a prefix: the model tag alone would not
# tell them apart and the second run would land on the first's run.tsv.
RETRIEVER_NAME_PREFIX = {"dense": "emb", "sparse": "sparse", "multi_vector": "mv", "fusion": "fusion",
                         "rrf": "rrf", "hybrid": "hybrid"}


def model_slug(name: str) -> str:
    """Path-safe slug for a model name (e.g. 'Qwen/Qwen3-30B' -> 'Qwen_Qwen3-30B')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def retriever_name(retriever: str, emb_model: str | None = None) -> str:
    """Directory name for a retriever's runs.

    bm25/oracle name themselves; the model-backed retrievers fold their
    embedding model into the name (so two models' runs stay separate) and
    therefore require an emb_model.
    """
    if retriever == "oracle":
        return "oracle"
    if retriever in MODEL_BASED_RETRIEVERS:
        if not emb_model:
            raise ValueError(f"retriever {retriever!r} requires emb_model (used to locate its run dir)")
        return f"{RETRIEVER_NAME_PREFIX[retriever]}-{model_slug(emb_model)}"
    return retriever


#: The <mem-seg> for a corpus no memory model built — the raw-turn baselines,
#: whose corpus IS the dataset. Spelled once here and imported by
#: pipeline/qrels_turns (MEM_MODEL), so the pseudo-systems and the memory
#: systems cannot end up in differently-shaped trees.
NO_MEM_MODEL = "none"


def mem_segment(mem_model: str | None, mem_variant: str | None = None) -> str:
    """The <mem-seg> path component shared by every stage's dir.

    "" counts as no variant so shell drivers can pass --mem-variant "$MEM_VARIANT"
    unconditionally.

    ``None`` is the raw-turn baselines' case and yields ``NO_MEM_MODEL``, the
    same segment ``--mem-model none`` produces. It used to raise TypeError, so a
    caller that omitted the flag got a traceback where the on-disk convention
    already had an answer — and a shell driver that swallowed it wrote a path
    with an empty segment.
    """
    seg = model_slug(mem_model) if mem_model else NO_MEM_MODEL
    if mem_variant:
        seg += f"--{model_slug(mem_variant)}"
    return seg


def labeler_segment(labeler_model: str, candidate_source: str = "sweep") -> str:
    """The <labeler-segment> path component shared by the qrels/retrieval dirs."""
    seg = model_slug(labeler_model)
    if candidate_source == "retrieval":
        seg += "-retrieval"
    return seg


#: The prompt a run answered under. "shared" is evalgrid.systems.base's prompt,
#: held fixed across systems so the comparison is about memory representation;
#: "native" is the system's own (SystemPlugin.answer_prompt, or the MemorySystem
#: subclass's answer_messages on the native path).
PROMPTS = ("shared", "native")


def native_segment(retriever: str | None = None, native_variant: str | None = None,
                   top_k: int | None = None) -> str:
    """The <retriever> segment a native run writes under.

    ``retriever`` is what the system's own loop ranks with, from
    SystemPlugin.native_retriever_name — in the same vocabulary the flat side
    uses, so `native-emb-<model>` sits directly beside `emb-<model>` and the two
    differ by the prefix alone. Derived rather than passed by the caller, exactly
    as `--emb-model` is folded into a flat run dir automatically: an encoder swap
    in a system's YAML changes the ranking, and must not land in the previous
    encoder's directory.

    ``top_k`` belongs to this segment rather than to a <topk-seg> level, because
    on this path k is retriever *identity*, not a slice of one ranking.
    ``pipeline/retrieve.py`` ranks the WHOLE corpus and ``pipeline/answer.py``
    truncates to k when it builds the prompt, so one flat run serves every k and
    only the answers dir needs a top<k> level. A native loop instead retrieves k
    units and answers from all of them — k changes the run.tsv, and nothing is
    truncated afterwards — so `native-…-top5` and `native-…-top20` are two
    retrievers, the way `emb-minilm` and `emb-qwen` are, and a native answers dir
    writes <topk-seg> "all" like oracle and full do.

    Keeping k here also keeps every retrieval run exactly one level above its
    samples: tools/check_retrieval.py globs "*/run.tsv" under a run dir, and a
    k level would make it report every native sample as missing.

    A native run's retriever is the system's own, so it cannot be selected with
    --retriever/--emb-model the way a flat-corpus run's can — it is configured in
    the system's YAML (A-Mem's retrieve_k/link_k/embedder, HyperMem's
    retrieval.top_k block, SimpleMem's retrieval.semantic_top_k, ...). Those
    knobs therefore have no other path segment to key on, so sweeping them would
    write every setting to this one directory — and, worse, the answer stage's
    resumability would skip the second run as already answered.

    --native-variant names the setting, exactly as --mem-variant names an ingest
    parameter set. No variant keeps the plain "native" path, so the default run
    is unchanged.

    Deliberately NOT --mem-variant: these knobs do not affect the store, so
    folding them into the memory segment would fork outputs/_memory/ and demand
    a re-ingest for a retrieval-only change.

        native-emb-<model>              each system at its own published depth
        native-emb-<model>-top10        one depth imposed on every system
        native-rrf-top10                hypermem's first stage, not an encoder
        native-rrf-norerank-top10       ... with its cross-encoder disabled
        native-emb-<model>-top10--nolinks   ... plus a named config variant

    ``retriever=None`` yields a bare "native", kept only for callers that are
    naming the family rather than a run (help text, error messages).
    """
    seg = "native"
    if retriever:
        seg += f"-{retriever}"
    if top_k is not None:
        seg += f"-top{int(top_k)}"
    if native_variant:
        seg += f"--{model_slug(native_variant)}"
    return seg


def answer_depth_segment(top_k: int | None, context_budget: int | None = None,
                         no_retrieval: bool = False) -> str:
    """The <topk-seg> path component: how much context the prompt was given.

        top10       the top 10 ranked facts (the historical default)
        budget500   facts in rank order until the next would cross 500 tokens
        all         no answer-time truncation (oracle / full / native)

    A budget run is a different retrieval condition, not the same run read to a
    different depth, so it gets its own directory rather than overwriting the
    top-k one — the same reason --top-k is a segment at all. "top<k>" is kept
    byte-identical so no existing answers dir moves.
    """
    if no_retrieval:
        return "all"
    if context_budget is not None:
        return f"budget{int(context_budget)}"
    return f"top{int(top_k)}"


def gen_segment(gen_model: str, prompt: str = "shared") -> str:
    """The <gen-seg> path component: generator model, plus the prompt it used.

    The prompt is a suffix rather than its own level, and absent for "shared",
    so every pre-existing answers dir keeps its exact path — there is roughly a
    GPU-day of judged runs under outputs/ and the default must not move. The
    suffix exists because <system> alone stops separating runs the moment one
    system answers under BOTH prompts: without it the second run lands on the
    first's predictions.jsonl and is silently skipped as already-answered by
    pipeline/answer.py's resumability check.

    Same reasoning as mem_segment/labeler_segment: non-default gets a suffix,
    default keeps the historical path.
    """
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")
    seg = model_slug(gen_model)
    if prompt != "shared":
        seg += f"--{prompt}-prompt"
    return seg


# --- bare path builders (kept as the primitives ExperimentPaths composes) ----
def memory_dir(system: str, dataset: str, mem_model: str | None = None,
               mem_variant: str | None = None) -> str:
    """Saved-memory dir, keyed only by what affects ingestion.

    Keyed by the memory model plus the ingest-parameter variant (see
    mem_segment), so two ingest runs that differ only in a retrieval/answer knob
    still share one memory.
    """
    return os.path.join(ROOT, "outputs", "_memory", system, dataset,
                        mem_segment(mem_model, mem_variant))


def qrels_dir(system: str, dataset: str, mem_model: str, labeler_seg: str,
              mem_variant: str | None = None) -> str:
    return os.path.join(ROOT, "outputs", "_qrels", system, dataset,
                        mem_segment(mem_model, mem_variant), labeler_seg)


def retrieval_dir(system: str, dataset: str, mem_model: str, retriever: str,
                  mem_variant: str | None = None) -> str:
    """Dir holding one retrieval run's per-sample run.tsv.

    NO <labeler-seg>, deliberately — and it used to have one. A ranking is a
    function of the corpus, the queries and the retriever; the qrels are read
    only by eval_retrieval. So two labelers over one store produce byte-identical
    run.tsv files, and keying this on the labeler meant a labeler change forced a
    full re-retrieval to reproduce them — GPU-DAYS for the native grid, where
    simplemem/a_mem/hypermem spend 5-8 query-time LLM calls per question. Worse,
    under `candidate_source: retrieval` the labeler judges what retrieval
    returned, so re-retrieving under the new labeler's segment is circular.

    ``answers_dir`` already drew this line — retriever answers take no labeler
    level and only oracle answers, which are read FROM the gold, do. This is the
    same rule applied to the stage before it: key each output on what determines
    it. ``qrels_dir`` and ``eval_dir`` keep the labeler because their contents
    depend on it.

    What that costs, stated: the corpus and queries physically live in the
    labeler-keyed qrels dir, so if two labeler runs ever differ in the corpus
    they emit or the samples they skip, one run.tsv is silently shared between
    them. Nothing enforces that they cannot. The record that survives is the run
    dir's own manifest, which `write_manifest` fills with the resolved args —
    including `qrels_dir` and `labeler_model` — so which qrels a ranking was
    computed over is still answerable, from the run rather than from its path.
    """
    return os.path.join(
        ROOT, "outputs", "_retrieval", system, dataset,
        mem_segment(mem_model, mem_variant), retriever)


def eval_dir(system: str, dataset: str, mem_model: str, labeler_seg: str, retriever: str,
             mem_variant: str | None = None) -> str:
    """Dir holding one retrieval run's metrics.json.

    A SEPARATE TREE, mirroring retrieval_dir level for level under ``_eval``
    instead of ``_retrieval``. It is where ``pipeline/eval_retrieval`` writes by
    default, and what ``scripts/query.sh`` passes as --out (through
    ``evalgrid.tools.stage_dir --stage eval``, so there is one spelling rather
    than a bash re-implementation of this function).

    A sibling rather than a file inside the run dir because the two have very
    different sizes and lifetimes: a cell's ``run.tsv`` files are orders of
    magnitude larger than its metrics, so scores can be synced back from the
    cluster on their own — which is the state a working copy is usually in.
    the analysis scripts reads this tree for exactly that reason.

    The one case that still scores in place is ``--retrieval-dir``: a run dir
    named explicitly is one outside this tree (a throwaway, a fused run), and its
    --system/--retriever say what it resembles, not where it lives, so filing its
    score in the keyed cell would overwrite the real cell's metrics.

    Both halves of this were once wrong. The default was the run dir, so a
    hand-run eval and a driver-run one put the same measurement in two trees; and
    the only spelling of the tree lived in query.sh as
    ``${OUT_ROOT}_eval/$sys/$(basename ...)``, which is how it came to carry
    neither the dataset, the memory model nor the labeler — three keys every
    sibling tree keys on, and without which two mem-models' scores of one
    condition name silently overwrite each other. Runs from before both fixes are
    moved into place by ``evalgrid.tools.migrate_outputs --stage eval``.
    """
    return os.path.join(
        ROOT, "outputs", "_eval", system, dataset,
        mem_segment(mem_model, mem_variant), labeler_seg, retriever)


def answers_dir(system: str, dataset: str, mem_model: str, retriever: str,
                gen_model: str, topk_seg: str, labeler_seg: str | None = None,
                mem_variant: str | None = None, prompt: str = "shared") -> str:
    """Dir holding one run's predictions.jsonl.

    <retriever> is the retriever_name() slug ("native" for the system's own
    query-time loop); <topk-seg> is "top<k>" for a real retrieval run, or "all"
    for oracle/full/native. Retriever answers don't depend on the labeler
    (labeler_seg=None); oracle answers do (pass labeler_seg). <prompt> suffixes
    the generator segment — see gen_segment.
    """
    parts = [ROOT, "outputs", "_answers", system, dataset,
             mem_segment(mem_model, mem_variant), retriever]
    if labeler_seg is not None:
        parts.append(labeler_seg)
    parts += [gen_segment(gen_model, prompt), topk_seg]
    return os.path.join(*parts)


# --- the object every stage should use ---------------------------------------
@dataclass(frozen=True)
class ExperimentPaths:
    """Keyword-only view of one experiment's output tree.

    Build it once from the resolved CLI args / spec, then ask it for each
    stage's dir, so no caller can silently drop a key.

    emb_model keys nothing here — it is the retriever's model, folded into the
    run-dir name by retriever_name(). It is accepted so callers that already
    carry one can pass it through unchanged.
    """
    system: str
    dataset: str
    mem_model: str | None = None
    emb_model: str | None = None
    mem_variant: str | None = None

    def memory(self) -> str:
        return memory_dir(self.system, self.dataset, self.mem_model,
                          mem_variant=self.mem_variant)

    def qrels(self, labeler_seg: str) -> str:
        return qrels_dir(self.system, self.dataset, self.mem_model, labeler_seg,
                         mem_variant=self.mem_variant)

    def retrieval(self, retriever: str) -> str:
        return retrieval_dir(self.system, self.dataset, self.mem_model, retriever,
                             mem_variant=self.mem_variant)

    def eval(self, labeler_seg: str, retriever: str) -> str:
        return eval_dir(self.system, self.dataset, self.mem_model, labeler_seg, retriever,
                        mem_variant=self.mem_variant)

    def answers(self, retriever: str, gen_model: str, topk_seg: str,
                labeler_seg: str | None = None, prompt: str = "shared") -> str:
        return answers_dir(self.system, self.dataset, self.mem_model, retriever,
                           gen_model, topk_seg, labeler_seg=labeler_seg,
                           mem_variant=self.mem_variant, prompt=prompt)
