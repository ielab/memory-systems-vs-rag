#!/usr/bin/env python3
"""
check_grid.py
=============
One table saying, for every system in the sweep, whether INGEST, RETRIEVAL and
ANSWERING actually finished — memory systems on their native grid and the
naive-RAG turn baselines on the flat one, in the same run.

Why this exists beside check_retrieval.py / check_answers.py
------------------------------------------------------------
Those two check ONE system against the flat ``run_all_*.sh`` sweep and derive
their paths from the keyed ``ExperimentPaths`` tree. Neither can see:

  * the NATIVE GRID layout — ``native-<retriever>[-<decor>]-top<k>--grid/``,
    where the retriever a cell used is buried in a directory name that also
    carries the system's own depth and any config variant;
  * an ``OUT_ROOT`` tree (``outputs/_smoke4_retrieval`` / ``_answers``), which
    is where every smoke run lands;
  * both halves of the table at once, which is the actual question before a
    judge run — "is the grid finished?" is a property of nine systems, not one.

What "finished" means, per stage
--------------------------------
INGEST, per system. The unit is the SAMPLE STORE, and the predicates are
``benchmarks/runner``'s own — imported, not restated, so that if the completion
marker ever moves this follows it. ``<sample>/usage.json`` is the only honest
one: it is written strictly AFTER ``save_memory`` returns, so a dir holding
notes but no usage.json is a save that crashed part-way and the next ingest
rebuilds it. ``failures.jsonl`` and a missing
``ingest_usage.json`` are reported beside the counts.

The expectation is the qrels sample set — the only one derivable without
loading the dataset, and the one the later stages actually need. With no qrels
built yet the stores are COUNTED but not judged, because there is no number to
judge them against.

RETRIEVAL, per condition. The unit of work is the SAMPLE — retrieve.py ranks a
sample's whole corpus into one ``<sample>/run.tsv``, and every answer-side k is
a later cut of it.

  COMPLETE  every expected sample has a non-empty run.tsv, and the stage left
            its end-of-loop markers behind
  PARTIAL   some samples have one, some do not
  MISSING   the condition's directory does not exist, or holds no run.tsv

Three failure markers are checked beyond the sample count, because a run can be
short WITHOUT any sample looking missing — the run.tsv files that do exist are
perfectly well formed:

  ``config.json`` / ``retrieval_usage.json`` absent (native only)
        retrieve.py writes both after its sample loop RETURNS, so their absence
        means the loop did not return — it crashed partway. This is how
        outputs/_smoke4's two SimpleMem cells published nDCG over 2 of 4
        samples.
  ``samples_no_memory > 0`` in retrieval_usage.json (native only)
        the loop DID return, having stepped over samples whose store was
        missing. The run finished; it just did not cover the qrels set.
  ``errors.jsonl`` present (flat only)
        the flat half collects per-sample failures and carries on, so here the
        evidence is a file that should not be there.

ANSWERING, per condition x cell. The unit is the QUESTION: a cell is complete
when its ``predictions.jsonl`` holds every question-id the answer stage would
attempt, which is the sample's queries gated on that sample having a run.tsv —
a question whose sample was never retrieved is not the answer stage's to fill.
The end-of-loop marker is checked too, and the two halves write DIFFERENT ones:
answer_summary.json on the shared path, token_usage.json on the native one
(``main()`` returns at ``_run_native`` and never reaches _write_answer_summary).
A predictions file with neither is a cell that died after its last write.

Expected retrievers
-------------------
NOT the spec's full ``retrievers:`` list by default. A spec lists every
retriever the system could be swept over, while the drivers dispatch a preset:

  grid8   (default) the eight of the paper's eight-retriever grid, which is what
          scripts/native_grid.sh runs
  rag3    bm25 + SPLADE-v3 + MiniLM, E2's three-retriever table
  spec    whatever the YAML lists

A condition on disk is matched to a retriever by stripping the ``native-``
prefix and the ``-top<k>`` / ``--<variant>`` / config decorations, so a system
that renames its depth or adds a variant does not read as a missing retriever.

Controls — the rows no preset names
-----------------------------------
That same peeling means a CONTROL matches nothing: ``native-rrf--naive`` peels
to ``rrf`` while grid8 lists ``rrf-<model>``, and a bare ``native`` peels to
itself. Those are the E1 rows — each system at its published depth, or at the
spec's imposed one — and they are what every swapped cell is read against
. Printing them as "ignored" meant a grid
could report COMPLETE with its own baseline unverified, so they are now checked
as rows, marked ``-- control``. ``--no-controls`` restores the old silence.

Their cells come from DISK, not from ``--cells``: a control has no planned cell
set, and a live published-depth run has exactly one cell (``all`` — nothing was
truncated after retrieval), so demanding the grid's eight would report seven
phantom gaps. Absence is therefore never a problem on a control row; an
incomplete cell that EXISTS, or a short retrieval, still is.

Finding the tree — discovered, not computed
-------------------------------------------
The keyed tree is ``<stage>/<system>/<dataset>/<mem-seg>/<labeler-seg>/...``,
and the answers tree adds ``<gen-seg>`` BETWEEN the condition and the cell
while an ``OUT_ROOT`` tree adds nothing. Computing those four keys from the
spec means being right about all four; being wrong about any one reports the
system as never run. An earlier version of this tool did exactly that to five
memory systems whose retrieval had been on disk the whole time.

So each stage is located by the artefact that DEFINES it — ``corpus.jsonl``,
``<cond>/<sample>/run.tsv``, ``predictions.jsonl`` — searched at ANY depth
(``rglob``, never a fixed number of ``*`` segments) and the keys above read back
off the path. A fixed pattern hardcodes exactly the levels the layouts disagree
about: a 4-deep glob finds a keyed tree and silently misses 500 samples sitting
directly under ``_qrels/<system>/``. Depth-agnostic, a ``--mem-variant`` nobody
remembered, a changed labeler segment, the extra generator level and a flat tree
written straight under the stage root all cost nothing.

When the key on disk is not the one the spec names: a single candidate is USED
and announced (``!! using a tree keyed ...``), because the alternative is
reporting a system as absent over a naming difference — but the run and the
spec disagree, and only the reader can say which is right. Several candidates
are never guessed between; they are listed for ``--mem-model`` /
``--mem-variant``. A system with no trace of any stage is still printed, since
a system missing from the report reads like one that finished; ``--skip-unrun``
opts out.

Usage
-----
    # a smoke tree
    python evalgrid/tools/check_grid.py --out-root outputs/_smoke4

    # the keyed tree (outputs/_qrels, _retrieval, _answers)
    python evalgrid/tools/check_grid.py

    # one system, and name every gap
    python evalgrid/tools/check_grid.py --out-root outputs/_smoke4 \
        --systems memorybank -v

    # a different answer grid
    python evalgrid/tools/check_grid.py --out-root outputs/_smoke4 \
        --cells top5,budget1000 --no-oracle

Exit code is 0 only when every expected cell is COMPLETE with no markers, so it
can gate a judge run in a script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evalgrid.lib.paths import (  # noqa: E402
    ExperimentPaths,
    labeler_segment,
    mem_segment,
    retriever_name,
)
from evalgrid.systems.registry import PSEUDO_SYSTEMS  # noqa: E402

# The eight settings of the paper's eight-retriever grid, spelled as spec entries so one
# parser serves both these and a --retrievers spec list.
GRID8 = (
    "bm25",
    "sparse:naver/splade-v3",
    "dense:sentence-transformers/all-MiniLM-L6-v2",
    "dense:Qwen/Qwen3-Embedding-0.6B",
    "dense:Qwen/Qwen3-Embedding-4B",
    "rrf:sentence-transformers/all-MiniLM-L6-v2",
    "rrf:Qwen/Qwen3-Embedding-0.6B",
    "rrf:Qwen/Qwen3-Embedding-4B",
)
RAG3 = (
    "bm25",
    "sparse:naver/splade-v3",
    "dense:sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_CELLS = ("top3", "top5", "top10", "top20",
                 "budget200", "budget500", "budget1000", "budget2000")

# The raw-turn baselines answer a `full` row; pair_turns deliberately does not —
# --retriever full feeds every doc regardless of the document boundary, so its
# prompt is byte-identical to all_turns'.
DEFAULT_FULL_SYSTEMS = ("all_turns", "user_turns")

OK, PARTIAL, MISSING = "COMPLETE", "PARTIAL", "MISSING"


# --------------------------------------------------------------------------
# spec / naming
# --------------------------------------------------------------------------

def retriever_slug(entry: str) -> str:
    """A spec entry (``bm25``, ``dense:MODEL@url``) -> its run-dir slug.

    The same ``retriever_name`` the stages use, so a slug here is the directory
    a run would actually be written to rather than a second spelling of it.
    """
    if entry == "bm25":
        return "bm25"
    kind, _, rest = entry.partition(":")
    return retriever_name(kind, rest.split("@", 1)[0])


def condition_variant(dirname: str) -> str | None:
    """The ``--<variant>`` a condition carries, or None.

    ``native_segment`` appends it LAST, so everything after the first ``--`` is
    it. None for a run that named no variant, and for every flat baseline.

    This is what separates two conditions that rank with the SAME retriever but
    are not the same run: `native-emb-<M>-top25--grid` is a swept cell,
    `native-emb-<M>--naive` is the published control, and `condition_slug` peels
    both to `emb-<M>`. For the four systems whose own encoder IS the grid's
    dense row that collision is guaranteed, not incidental — a_mem's control
    was matched into the expected set and then reported as seven missing budget
    cells it was never going to have.
    """
    return dirname.split("--", 1)[1] if "--" in dirname else None


def condition_slug(dirname: str) -> str:
    """A condition directory -> the retriever slug it ranked with.

    ``native-emb-Qwen_Qwen3-Embedding-4B-norerank-top95--grid`` -> ``emb-Qwen_Qwen3-Embedding-4B``

    Peeled rather than reconstructed: `native_segment` appends the system's own
    depth and any `--variant` AFTER the retriever, and a system's config can add
    its own token in between (HyperMem's `norerank`). Rebuilding the whole
    segment here would mean re-deriving every system's depth and variant, and
    would report a renamed depth as a missing retriever. A flat baseline's
    directory is already the bare slug and falls through untouched.
    """
    name = dirname
    name = name[len("native-"):] if name.startswith("native-") else name
    name = name.split("--", 1)[0]                    # drop --grid / --nolinks
    name = re.sub(r"-top\d+$", "", name)             # drop the imposed depth
    name = re.sub(r"-norerank$", "", name)           # drop config decorations
    return name


def is_native_cond(dirname: str) -> bool:
    """Did the system's OWN loop write this condition directory?

    Every native segment starts with ``native`` — ``lib/paths.native_segment``
    prefixes it, and query.sh's hand-built form is ``$MODE-top$k`` with
    MODE=native — so the prefix is the whole test.

    Asked PER DIRECTORY rather than per system because the two can disagree: a
    memory system swept on the flat path as well (E2-A, `--retriever bm25
    --prompt shared`) writes `bm25/` beside its `native-*` cells, in the same
    tree. Judged by the system's `native_retrievers:` key alone, those flat
    cells would be checked for `config.json` / `retrieval_usage.json`, which
    only `retrieve.py::_run_native` writes — reporting every E2-A cell as a
    loop that never returned.
    """
    return dirname == "native" or dirname.startswith("native-")


def load_spec(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def expected_retrievers(spec: dict, preset: str) -> list[str]:
    if preset == "grid8":
        return [retriever_slug(e) for e in GRID8]
    if preset == "rag3":
        return [retriever_slug(e) for e in RAG3]
    entries = spec.get("native_retrievers") or spec.get("retrievers") or []
    return [retriever_slug(e) for e in entries]


# --------------------------------------------------------------------------
# tree layout
# --------------------------------------------------------------------------

class Tree:
    """Where one system's three stage outputs live.

    Two layouts, one interface. ``out_root`` non-empty is the side tree every
    smoke and every ``OUT_ROOT=`` submission writes — flat, one level per
    system. Empty is the keyed default, whose extra levels (dataset, mem model,
    labeler, generator) are asked of ``ExperimentPaths`` rather than spelled
    again here.
    """

    def __init__(self, out_root: str | None, spec: dict, system: str,
                 mem_model: str | None = None, mem_variant: str | None = None):
        self.system = system
        self.out_root = out_root
        self.spec = spec
        self.dataset = spec.get("dataset") or "longmemeval_s"
        self.mem_model = mem_model or spec.get("mem_model")
        self.mem_variant = mem_variant or spec.get("mem_variant")
        self._seg = labeler_segment(spec.get("labeler_model") or "has_answer",
                                    spec.get("candidate_source") or "sweep")
        self._paths = ExperimentPaths(system=system, dataset=self.dataset,
                                      mem_model=self.mem_model,
                                      mem_variant=self.mem_variant)

    # ----- discovery ------------------------------------------------------
    # The keyed tree's shape is NOT assumed. Deriving it from the spec means
    # deriving four keys (dataset, mem-seg, labeler-seg, gen-seg) that the run
    # itself chose, and being wrong about ANY of them reports the system as
    # "not run" — which is what a computed-path-only version of this tool did
    # to five memory systems whose retrieval was sitting on disk the whole
    # time. Worse, the answers tree inserts <labeler-seg> and <gen-seg> BETWEEN
    # the condition and the cell while an OUT_ROOT tree inserts nothing, so a
    # single hardcoded depth cannot serve both.
    #
    # So each stage is found by the artefact that defines it — corpus.jsonl,
    # run.tsv, predictions.jsonl — and the levels above are read back off the
    # path. That works for both layouts, for a --mem-variant nobody passed, and
    # for a labeler segment that changed.

    def stage_root(self, stage: str) -> Path:
        """Where one stage's outputs for this system begin.

        Public because check_judge / check_eval need the same two layouts and
        must not respell them: "the answers root" and "the eval root" are the
        same two conventions this class already resolves, and a second spelling
        is how a tool ends up reporting a finished stage as absent.

        Any stage name works, including ones this tool does not itself read
        ("eval"), because both layouts are a naming rule rather than a list.
        """
        if self.out_root:
            root = Path(self.out_root) if stage == "memory" else Path(f"{self.out_root}_{stage}")
            return root / self.system
        return Path(REPO) / "outputs" / f"_{stage}" / self.system

    @property
    def labeler_seg(self) -> str:
        """The <labeler-segment> the spec names — the key oracle outputs carry."""
        return self._seg

    def _wanted_key(self) -> str:
        """The mem-segment the spec (plus any override) points at."""
        return mem_segment(self.mem_model, self.mem_variant)

    def _candidates(self, base: Path, marker: str, up: int) -> dict[str, Path]:
        """{key: root} for every root under ``base`` that holds ``marker``.

        ``up`` is how many levels sit between the marker file and the root:
        a qrels root is the parent of the sample dir holding ``corpus.jsonl``
        (2), a retrieval root the grandparent of the condition dir holding
        ``<cond>/<sample>/run.tsv`` (3).

        The SEARCH is depth-agnostic — ``rglob``, never a fixed number of ``*``
        segments. A fixed pattern hardcodes exactly the levels the layouts
        disagree about: the keyed tree inserts ``<dataset>/<mem-seg>/<labeler-seg>``
        and a flat one inserts nothing, so a 4-deep glob finds a keyed tree and
        silently misses 500 samples sitting directly under ``_qrels/<system>/``.
        The key is whatever path remains above the root, "." when the root IS
        the base — a label for the reader, not something to parse.
        """
        found: dict[str, Path] = {}
        if not base.is_dir():
            return found
        for hit in base.rglob(marker):
            parents = hit.parents
            if len(parents) < up:
                continue
            root = parents[up - 1]
            try:
                key = str(root.relative_to(base))
            except ValueError:
                continue          # the marker sat ABOVE base — not ours
            found.setdefault(key, root)
        return found

    def _pick(self, found: dict[str, Path], stage: str) -> tuple[Path | None, str | None]:
        """One root out of the candidates, and a note if it was not the obvious one.

        Preference order, and each step exists for a layout seen in practice:
        a key naming the spec's mem-segment; the FLAT root ("." — an OUT_ROOT
        style tree written straight under ``_qrels/<system>/``); then a single
        candidate whatever its key.

        A single candidate under an unexpected key is USED, not rejected — the
        alternative is reporting a system as absent over a naming difference —
        but it is announced, because the run and the spec disagree and only the
        reader can say which is right. Several candidates are never guessed
        between.
        """
        if not found:
            return None, None
        # THE DATASET IS NOT A NAMING DIFFERENCE. The fallbacks below exist to
        # survive a mem-variant nobody passed or a labeler segment that changed
        # — differences in how one experiment was keyed. A key naming ANOTHER
        # BENCHMARK is a different experiment, and using it reports that
        # benchmark's results under this spec: a beam_100k run showed a complete
        # LongMemEval answer grid beside its own empty retrieval. The keyed tree
        # always carries a <dataset> level (ExperimentPaths builds it), so in
        # that layout a candidate must name this dataset before it is eligible
        # for any preference below. An OUT_ROOT tree has no dataset level and is
        # left alone.
        if not self.out_root:
            scoped = {k: v for k, v in found.items()
                      if k == "." or self.dataset in Path(k).parts}
            if not scoped:
                return None, (f"{stage}: nothing on disk for {self.dataset!r} — the "
                              f"tree(s) here are keyed {', '.join(sorted(found))}. "
                              f"Check --spec-name.")
            found = scoped
        want = self._wanted_key()
        # The keyed tree is <stage>/<system>/<dataset>/<mem-seg>/<labeler-seg>, so
        # a key names BOTH. Matching on the mem-segment ALONE is not enough the
        # moment a checkout holds two benchmarks: the mem-segment is the same
        # string under longmemeval_s and beam_100k, so every key matched and the
        # winner was whichever `rglob` happened to yield first — filesystem
        # order, differing between machines, with no note printed because the
        # match was considered exact. A beam_100k spec then reported 500/500
        # LongMemEval samples as though they were its own.
        for key in found:
            parts = Path(key).parts
            if want and want in parts and self.dataset in parts:
                return found[key], None
        if "." in found:
            return found["."], None
        if len(found) == 1:
            key, root = next(iter(found.items()))
            return root, (f"{stage}: using the tree keyed {key!r}, which does not name "
                          f"the spec's {self.dataset!r}/{want!r}")
        # Named per stage, because they are found independently: qrels can match
        # the spec while retrieval sits under an older key, and a note that does
        # not say which stage sends the reader to the wrong tree.
        return None, (f"{stage}: several trees on disk and none names the spec's "
                      f"{self.dataset!r}/{want!r} ({', '.join(sorted(found))}) — check "
                      f"--spec-name, or pass --mem-model/--mem-variant")

    def memory(self) -> Path:
        # The store is keyed ONLY by what affects ingestion (paths.memory_dir),
        # so it has no labeler or generator level. An OUT_ROOT tree keeps it
        # under <root>/<system>: MEM_ROOT names the store directly, which is why
        # ingest takes MEM_ROOT and the later stages take OUT_ROOT.
        base = self.stage_root("memory")
        if self.out_root:
            return base
        # usage.json is runner._ingest_complete's marker and sits one level in.
        root, _ = self._pick(self._candidates(base, "usage.json", 2), "store")
        return root or Path(self._paths.memory())

    def qrels(self) -> tuple[Path | None, str | None]:
        """The dir whose children are sample dirs — found by corpus.jsonl."""
        return self._pick(self._candidates(self.stage_root("qrels"), "corpus.jsonl", 2), "qrels")

    def retrieval_root(self) -> tuple[Path | None, str | None]:
        """The dir whose children are conditions — found by <cond>/<sample>/run.tsv."""
        return self._pick(self._candidates(self.stage_root("retrieval"), "run.tsv", 3), "retrieval")

    def predictions(self) -> dict[tuple[str, str], Path]:
        """{(condition, cell): predictions.jsonl} — every layout, one index.

        The cell is always the file's own directory. The condition is found by
        walking up until a component is a condition name, which the caller
        supplies; nothing here has to know how many levels the tree inserts
        between them, which is the whole point.

        Scoped to this spec's dataset in the keyed layout, for the reason spelled
        out in ``_pick``: indexing EVERY component as a possible condition is
        what makes this work across layouts, and it is also what let another
        benchmark's answers answer for this one. Unlike the other stages this
        one takes no ``_pick`` note — it returns a dict, not a root — so an
        unscoped hit here was invisible rather than merely unannounced: a
        beam_100k report showed eight LongMemEval cells as `ok` above its own
        `0/20` retrieval, which is the one combination that cannot happen.
        """
        base = self.stage_root("answers")
        out: dict[tuple[str, str], Path] = {}
        if not base.is_dir():
            return out
        for pred in base.rglob("predictions.jsonl"):
            parts = pred.relative_to(base).parts
            if len(parts) < 2:
                continue
            if not self.out_root and self.dataset not in parts:
                continue
            cell = parts[-2]
            for part in parts[:-2]:
                out.setdefault((part, cell), pred)
        return out

    def keys_on_disk(self) -> dict[str, list[str]]:
        """Every mem-key present per stage — the diagnosis when nothing matched."""
        if self.out_root:
            return {}
        markers = {"memory": ("usage.json", 2), "qrels": ("corpus.jsonl", 2),
                   "retrieval": ("run.tsv", 3), "answers": ("predictions.jsonl", 2)}
        out = {}
        for stage, (marker, up) in markers.items():
            keys = sorted(self._candidates(self.stage_root(stage), marker, up))
            if keys:
                out[stage] = keys
        return out


def _dirs(path: Path) -> list[Path]:
    return sorted((d for d in path.iterdir() if d.is_dir()), key=lambda d: d.name) \
        if path.is_dir() else []


# --------------------------------------------------------------------------
# reading what is on disk
# --------------------------------------------------------------------------

def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_ingest(mem: Path, expect: set[str] | None) -> dict:
    """The store behind everything else: how many sample dirs are finished.

    The predicates are ``benchmarks/runner``'s own — imported, not restated, so
    that if the completion marker ever moves this follows it. ``usage.json`` is
    the only honest one: it is written strictly AFTER ``save_memory`` returns,
    so a dir holding notes but no usage.json is a save that crashed part-way and
    the next ingest rebuilds it.

    ``expect`` is the sample set the later stages need — the qrels samples,
    which is the only expectation derivable without loading the dataset. It is
    None when qrels have not been built, and then the counts are reported
    without a verdict rather than compared against a number we do not have.
    """
    from evalgrid.benchmarks.runner import _ingest_complete, _memory_saved

    if not mem.is_dir():
        return {"state": MISSING, "complete": [], "partial": [], "empty": [],
                "missing": sorted(expect or ()), "counted": expect is not None,
                "markers": ["no store directory"], "notes": []}

    complete, partial, empty = [], [], []
    for d in sorted(p for p in mem.iterdir() if p.is_dir()):
        if _ingest_complete(str(d)):
            complete.append(d.name)
        elif _memory_saved(str(d)):
            partial.append(d.name)
        else:
            empty.append(d.name)

    markers = []
    if partial:
        markers.append(f"{len(partial)} save(s) crashed part-way (no usage.json) — "
                       f"re-run ingest, it rebuilds only these")
    if empty:
        markers.append(f"{len(empty)} store(s) hold nothing (the run reached the sample "
                       f"and nothing survived)")
    notes = []
    if (mem / "failures.jsonl").is_file():
        # A NOTE, not a marker. Every sample here still completed — the skipped
        # unit is inside one, by the failure policy's "cost the smallest unit
        # that has a boundary" — so this is a caveat on the DATA, not a hole in
        # the grid. Counted as a marker it made two complete 500/500 stores
        # report "1 incomplete cell(s)", sending the reader hunting through a
        # table where every cell says ok.
        notes.append("failures.jsonl — some units were skipped; read it before "
                     "quoting any count from this store")
    # ingest_usage.json is the read-path counterpart every cost figure comes
    # from; its absence means no ingest ever finished over this dir.
    if not (mem / "ingest_usage.json").is_file():
        markers.append("no ingest_usage.json (no ingest run completed over this store)")

    done = set(complete)
    missing = sorted(expect - done) if expect is not None else []
    if not complete:
        state = MISSING
    elif missing or partial:
        state = PARTIAL
    else:
        state = OK
    return {"state": state, "complete": complete, "partial": partial, "empty": empty,
            "missing": missing, "counted": expect is not None,
            "markers": markers, "notes": notes}


def qrels_samples(qrels: Path) -> list[Path]:
    """Sample dirs with a corpus.jsonl — retrieve.py's own unit of work."""
    if not qrels.is_dir():
        return []
    return [d for d in sorted(qrels.iterdir()) if (d / "corpus.jsonl").is_file()]


def sample_questions(sample: Path) -> set[str]:
    q = sample / "queries.jsonl"
    if not q.is_file():
        return set()
    return {r["_id"] for r in read_jsonl(q)}


def check_retrieval(cond: Path, samples: list[Path], native: bool) -> dict:
    """One condition's retrieval state, plus the markers a sample count misses."""
    have, missing = [], []
    for s in samples:
        run = cond / s.name / "run.tsv"
        # A header-only run.tsv is a sample that produced no ranking; it is not
        # "done" in any sense a later stage can use.
        if run.is_file() and sum(1 for _ in run.open()) > 1:
            have.append(s.name)
        else:
            missing.append(s.name)

    markers: list[str] = []
    if not cond.is_dir():
        state = MISSING
    elif not have:
        state = MISSING
    elif missing:
        state = PARTIAL
    else:
        state = OK

    if cond.is_dir() and native:
        for f in ("config.json", "retrieval_usage.json"):
            if not (cond / f).is_file():
                markers.append(f"no {f} (loop never returned)")
        usage = cond / "retrieval_usage.json"
        if usage.is_file():
            try:
                u = json.loads(usage.read_text())
            except (OSError, ValueError, TypeError):
                u = {}
            # samples_failed/queries_failed did not exist before the native loop
            # could survive a failure at all, so an older cell simply has none
            # of them — absent reads as 0, which is what it meant.
            for field in ("samples_no_memory", "samples_failed", "queries_failed"):
                try:
                    n = int(u.get(field) or 0)
                except (ValueError, TypeError):
                    n = 0
                if n:
                    markers.append(f"{field}={n}")
    # Both halves write it now, and on the native side it is the only record of
    # a query the system's own loop threw on.
    #
    # It is APPENDED, never truncated (retrieve.py, and ingest's failures.jsonl
    # before it), so a resumed run adds to the record rather than erasing what
    # the last one found. That is right for the record and wrong for a verdict:
    # a cell whose failed sample was retrieved successfully on the next run
    # would carry "errors.jsonl" — and therefore count as an incomplete cell —
    # forever. So the file is read, not merely detected: a recorded failure
    # whose sample now HAS a ranking is resolved, and resolved failures are
    # reported as history rather than as a problem.
    notes: list[str] = []
    errs = cond / "errors.jsonl"
    if cond.is_dir() and errs.is_file():
        failed_ids, unparsable = set(), 0
        for line in errs.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                failed_ids.add(json.loads(line).get("sample_id"))
            except ValueError:
                unparsable += 1
        done = set(have)
        open_ids = sorted(i for i in failed_ids if i and i not in done)
        fixed = sorted(i for i in failed_ids if i and i in done)
        if open_ids or unparsable:
            markers.append(f"errors.jsonl — {len(open_ids)} unresolved: "
                           f"{', '.join(open_ids) or '(unreadable lines)'}")
        if fixed:
            notes.append(f"errors.jsonl — {len(fixed)} earlier failure(s) since "
                         f"retrieved: {', '.join(fixed)}")

    return {"state": state, "have": have, "missing": missing,
            "markers": markers, "notes": notes}


def check_answer_cell(preds: Path | None, expected: set[str]) -> dict:
    """One answer cell against the question-ids it owes.

    Takes the predictions.jsonl itself rather than its directory: the tree
    inserts a different number of levels above it depending on layout, so the
    caller resolves it from an index and this only has to read it.
    """
    if preds is None or not preds.is_file():
        return {"state": MISSING, "missing": sorted(expected), "markers": []}
    cell = preds.parent
    answered = {r.get("question_id") for r in read_jsonl(preds)}
    gap = sorted(expected - answered)
    markers = []
    # The end-of-loop marker, and the two halves write DIFFERENT ones. The
    # shared path writes answer_summary.json after its last append
    # (answer.py::_write_answer_summary); the native path returns at
    # `main()`'s `return _run_native(...)` and never reaches it — its marker is
    # token_usage.json, written by benchmarks/runner after the sample loop, the
    # same position retrieval_usage.json holds on the retrieve side. Demanding
    # answer_summary.json of both flagged every native cell ever written,
    # controls included, as a stage that died.
    if not any((cell / m).is_file() for m in ("answer_summary.json", "token_usage.json")):
        markers.append("no answer_summary.json / token_usage.json "
                       "(stage died after its last write)")
    if (cell / "errors.jsonl").is_file():
        markers.append("errors.jsonl")
    state = OK if not gap else (PARTIAL if answered else MISSING)
    return {"state": state, "missing": gap, "markers": markers}


# --------------------------------------------------------------------------
# per-system report
# --------------------------------------------------------------------------

def check_system(system: str, spec: dict, args) -> dict:
    tree = Tree(args.out_root, spec, system, args.mem_model, args.mem_variant)
    qrels, qnote = tree.qrels()
    rroot, rnote = tree.retrieval_root()
    mem = tree.memory()
    preds = tree.predictions()
    samples = qrels_samples(qrels) if qrels else []
    # De-duplicated: qrels and retrieval usually miss the spec's key for the
    # same reason, and saying it twice reads as two problems.
    notes = sorted({n for n in (qnote, rnote) if n})

    rows: list[dict] = []
    problems = 0

    # Ingest first — it is the one stage checkable with no qrels, and when the
    # qrels lookup below misses it is the evidence for whether this system was
    # ever run at all. The pseudo-systems have no ingest stage to check: their
    # "memory" is the dataset's own turns, built by pipeline/qrels_turns.py
    # straight into the qrels tree, and there is no store anywhere. Expecting
    # one would mark all three permanently incomplete.
    expect = {s.name for s in samples} if samples else None
    ing = None
    if system not in PSEUDO_SYSTEMS:
        ing = check_ingest(mem, expect)
        if ing["state"] != OK or ing["markers"]:
            problems += 1

    if not samples:
        # NOT a silent skip. A system whose qrels are not where this tool looks
        # is either (a) not part of this run, or (b) keyed differently — the
        # store was ingested under another mem-model or a --mem-variant, and
        # every later stage inherits that key. Those two are worlds apart and
        # only the paths can tell them apart, so both are printed. Dropping the
        # system instead is the exact failure this tool exists to catch: a
        # column that is absent from the report rather than marked missing.
        evidence, downstream = [], False
        for label, path in (("store", mem), ("retrieval", rroot)):
            if path is not None and path.is_dir():
                evidence.append(f"{label} at {path}")
                # A store alone is just "ingested, qrels not built yet" — the
                # ordinary state between two stages. Retrieval, though, cannot
                # exist without qrels having been there when it ran, so its
                # presence beside missing qrels IS a mismatch.
                downstream |= label != "store"
        if preds:
            evidence.append(f"{len(preds)} answer cell(s) under {tree.stage_root('answers')}")
            downstream = True
        keys = tree.keys_on_disk()
        return {"system": system, "rows": [], "ingest": ing,
                "problems": problems + 1,
                "note": (f"no qrels found under {tree.stage_root('qrels')}"
                         if qrels is None else f"no qrels samples under {qrels}"),
                "evidence": evidence, "mismatch": downstream,
                "notes": notes,
                "keys": {k: v for k, v in keys.items() if v}}

    # A memory system ranks inside its own loop; the raw-turn baselines rank a
    # flat corpus. The marker files and the directory naming differ, and both
    # follow from this one bit.
    native = bool(spec.get("native_retrievers"))
    want = expected_retrievers(spec, args.retrievers)

    # Only cells carrying the SWEEP's variant can satisfy an expected slug. A
    # cell with no variant still can — the grid is runnable without one, and
    # every pre-existing run predates the knob — so this reclassifies exactly
    # the cells that explicitly say they are something else.
    on_disk = sorted(d.name for d in _dirs(rroot)) if rroot else []
    by_slug: dict[str, list[str]] = {}
    off_variant: list[str] = []
    for d in on_disk:
        v = condition_variant(d)
        if v is not None and v != args.grid_variant:
            off_variant.append(d)
            continue
        by_slug.setdefault(condition_slug(d), []).append(d)

    for slug in want:
        dirs = by_slug.get(slug)
        if not dirs:
            # The cells are spelled out as MISSING rather than left absent: an
            # empty dict prints as "?" (unknown), and there is nothing unknown
            # here — no ranking exists, so no cell can.
            rows.append({"cond": slug, "retrieval": {"state": MISSING, "have": [],
                                                     "missing": [s.name for s in samples],
                                                     "markers": ["no directory"]},
                         "cells": {c: {"state": MISSING, "missing": [], "markers": []}
                                   for c in args.cells}})
            problems += 1
            continue
        for d in dirs:
            r = check_retrieval(rroot / d, samples, native and is_native_cond(d))
            if r["state"] != OK or r["markers"]:
                problems += 1
            # Expected questions are gated on the sample having a run.tsv: a
            # question whose sample was never retrieved is not the answer
            # stage's to fill, and counting it would report every retrieval gap
            # twice.
            expected = set()
            for s in samples:
                if s.name in r["have"]:
                    expected |= sample_questions(s)
            cells = {}
            for cell in args.cells:
                res = check_answer_cell(preds.get((d, cell)), expected)
                if res["state"] != OK or res["markers"]:
                    problems += 1
                cells[cell] = res
            rows.append({"cond": d, "retrieval": r, "cells": cells})

    # oracle and full take no ranking at all — oracle reads qrels/test.tsv and
    # full reads the corpus — so they have no retrieval half and are expected
    # even when every retrieval cell is missing.
    all_questions = set()
    for s in samples:
        all_questions |= sample_questions(s)
    for special in ("oracle", "full"):
        if special == "oracle" and not args.oracle:
            continue
        if special == "full" and system not in args.full_systems:
            continue
        # "all" is the <topk-seg> both write (answer_depth_segment(no_retrieval)),
        # and the index already collapsed whatever levels sit above it.
        res = check_answer_cell(preds.get((special, "all")), all_questions)
        if res["state"] != OK or res["markers"]:
            problems += 1
        rows.append({"cond": special, "retrieval": None, "cells": {"all": res}})

    # Conditions on disk that no preset names. Chiefly the E1 CONTROLS: the
    # published-depth run (`native-rrf--naive`), the imposed-depth one
    # (`native-top95`), anything carrying a --native-variant. `condition_slug`
    # peels those to something no preset lists — `native-rrf--naive` becomes
    # `rrf`, while grid8 lists `rrf-<model>` — so they used to be printed as
    # "ignored" and never checked. That hid the one row every swapped cell is
    # read against: a grid reported
    # COMPLETE while its own baseline was unverified.
    #
    # THE CELLS COME FROM DISK, not from `args.cells`. An unexpected condition
    # has no planned cell set, and demanding the grid's seven would report a
    # live published-depth run — whose only cell is `all`, because nothing was
    # truncated after retrieval — as six cells missing. So absence is never a
    # problem here; only a cell that EXISTS and is incomplete, or a retrieval
    # that is short. A control is still reported when it has no answers at all,
    # since "retrieved but never answered" is exactly what a reader wants to see.
    extra = sorted([d for slug, ds in by_slug.items() if slug not in want for d in ds]
                   + off_variant)
    controls = []
    if args.controls:
        for d in extra:
            r = check_retrieval(rroot / d, samples, native and is_native_cond(d))
            if r["state"] != OK or r["markers"]:
                problems += 1
            expected = set()
            for s in samples:
                if s.name in r["have"]:
                    expected |= sample_questions(s)
            cells = {}
            for cell in sorted({c for cond, c in preds if cond == d}):
                res = check_answer_cell(preds.get((d, cell)), expected)
                if res["state"] != OK or res["markers"]:
                    problems += 1
                cells[cell] = res
            rows.append({"cond": d, "retrieval": r, "cells": cells, "control": True})
            controls.append(d)

    return {"system": system, "rows": rows, "problems": problems,
            "ingest": ing, "notes": notes,
            "extra": [d for d in extra if d not in controls]}


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

MARK = {OK: "ok", PARTIAL: "PART", MISSING: "----"}


def print_ingest(ing: dict, args) -> None:
    n = len(ing["complete"])
    total = n + len(ing["missing"])
    if ing["state"] == MISSING and not ing["complete"]:
        print("  ingest      ---- no store")
    elif not ing["counted"]:
        # No qrels to compare against, so the count is a count and not a
        # verdict: 4 complete stores out of an unknown number owed.
        print(f"  ingest      ok {n} store(s) complete (no qrels — nothing to compare against)"
              + (f", {len(ing['partial'])} part-way" if ing["partial"] else ""))
    elif ing["state"] == OK:
        print(f"  ingest      ok {n}/{n} store(s) complete")
    else:
        print(f"  ingest      {MARK[ing['state']]} {n}/{total} store(s) complete"
              + (f", {len(ing['partial'])} part-way" if ing["partial"] else ""))
    for m in ing["markers"]:
        print(f"              !! {m}")
    for n in ing.get("notes") or []:
        print(f"              -- {n}")
    if args.verbose:
        for label, key in (("no store", "missing"), ("part-way", "partial"), ("empty", "empty")):
            if ing.get(key):
                print(f"                 {label}: {', '.join(ing[key])}")


def print_system(rep: dict, args) -> None:
    system = rep["system"]
    print(f"\n=== {system}")
    for n in rep.get("notes") or []:
        # A stage found under a key the spec does not name is USED, but never
        # silently: the run and the spec disagree and only the reader can say
        # which is right.
        print(f"  !! {n}")
    if rep.get("ingest"):
        print_ingest(rep["ingest"], args)
    elif system in PSEUDO_SYSTEMS:
        print("  ingest      n/a  (raw turns — no memory system, no store)")
    if rep.get("note"):
        print(f"  qrels       ---- {rep['note']}")
        for e in rep.get("evidence", []):
            print(f"              !! {e}")
        if rep.get("mismatch"):
            print(f"              !! a LATER stage has outputs but the qrels are not at the "
                  f"path above —")
            print(f"                 the tree is keyed by mem-model (+ --mem-variant) and "
                  f"every stage inherits it.")
            for stage, ks in (rep.get("keys") or {}).items():
                print(f"                 {stage:<10} keys on disk: {', '.join(ks)}")
            if rep.get("keys"):
                print("                 pass one back as --mem-model / --mem-variant, "
                      "or --out-root for a side tree.")
        elif rep.get("evidence"):
            print("                 the store is there but no qrels were built from it yet: "
                  "run pipeline/qrels_has_answer.")
        else:
            print("                 no store, no retrieval, no answers — not part of this run.")
        return
    if not rep["rows"]:
        print("  (nothing expected)")
        return

    cells = list(args.cells)
    width = max(len(r["cond"]) for r in rep["rows"]) + 2
    header = f"  {'condition':<{width}} {'retrieval':<12} " + " ".join(f"{c:>10}" for c in cells)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rep["rows"]:
        r = row["retrieval"]
        if r is None:
            rtxt = "n/a"
        elif r["state"] == OK:
            rtxt = f"ok {len(r['have'])}/{len(r['have'])}"
        else:
            rtxt = f"{MARK[r['state']]} {len(r['have'])}/{len(r['have']) + len(r['missing'])}"
        if set(row["cells"]) == {"all"}:
            c = row["cells"]["all"]
            body = f"{MARK[c['state']]} (all)"
        elif row.get("control") and not row["cells"]:
            # Retrieved, never answered. Blank columns would read as "?"
            # (unknown) seven times over for a row that has no planned cells.
            body = "no answer cells"
        else:
            body = " ".join(f"{MARK[row['cells'][c]['state']]:>10}" if c in row["cells"]
                            else f"{'?':>10}" for c in cells)
        print(f"  {row['cond']:<{width}} {rtxt:<12} {body}")
        if row.get("control"):
            # Never silently: this row was not asked for by any preset, so its
            # "ok" means "complete as far as what is on disk", not "complete
            # against a plan" — and a reader who cannot tell those apart will
            # trust a half-empty baseline. Spelled out rather than labelled,
            # and the cells are NAMED (they may sit outside the columns above,
            # where a `-- also:` note is the only place they can appear).
            got = ", ".join(row["cells"]) if row["cells"] else None
            print(f"  {'':<{width}} -- baseline row: no preset expects it, so no cells are "
                  f"required here.")
            print(f"  {'':<{width}}    " + (
                f"Checked the cell(s) that exist ({got}); a missing one is NOT reported."
                if got else "No answer cells exist yet — nothing was checked."))

        for m in (r or {}).get("markers", []):
            print(f"  {'':<{width}} !! retrieval: {m}")
        for n in (r or {}).get("notes", []):
            print(f"  {'':<{width}} -- retrieval: {n}")
        for name, c in row["cells"].items():
            for m in c["markers"]:
                print(f"  {'':<{width}} !! {name}: {m}")
        if args.verbose:
            if r and r["missing"]:
                print(f"  {'':<{width}}    no run.tsv: {', '.join(r['missing'])}")
            for name, c in row["cells"].items():
                if c["missing"]:
                    print(f"  {'':<{width}}    {name} unanswered: {', '.join(c['missing'])}")
    if rep.get("extra"):
        print(f"  (not in the expected set, ignored: {', '.join(rep['extra'])})")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Is the retrieval + answering grid finished?",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="",
                   help="OUT_ROOT the stages were run with, e.g. outputs/_smoke4 "
                        "(reads <root>_qrels / _retrieval / _answers). "
                        "Empty = the keyed default tree.")
    p.add_argument("--systems", default="",
                   help="comma-separated; default is every experiments/<sys>/<spec-name> "
                        "that has outputs under the tree")
    # Same knob, same spelling, same default as the shell drivers' SPEC_NAME
    # (scripts/lib/vllm.sh): experiments/<system>/<SPEC_NAME>. The DATASET is
    # not a separate flag because it is not a separate fact — it comes out of
    # the spec that was actually run, the way every other key here does, so this
    # tool cannot be pointed at one benchmark's specs while reading another's
    # tree.
    p.add_argument("--spec-name", default="longmemeval.yaml",
                   help="which per-system spec to read, e.g. beam_100k.yaml "
                        "(default: longmemeval.yaml). Mirrors SPEC_NAME in the "
                        "shell drivers; the dataset comes from the spec.")
    p.add_argument("--retrievers", default="grid8", choices=("grid8", "rag3", "spec"),
                   help="which retrievers to expect (default: grid8, what the native "
                        "grid runs). 'spec' uses the YAML's own list.")
    p.add_argument("--grid-variant", default="grid",
                   help="the --native-variant the SWEPT cells carry (default: grid, what "
                        "scripts/native_grid.sh passes). A cell naming a "
                        "different one is a control, not a grid cell that lost its budget "
                        "columns — for the four systems whose own encoder is the grid's "
                        "dense row, the two are otherwise indistinguishable. A cell with "
                        "no variant at all is still eligible.")
    p.add_argument("--no-controls", dest="controls", action="store_false",
                   help="do not check conditions no preset names (the E1 controls: "
                        "native-<r>--naive, native-top<k>, any --native-variant). "
                        "They are checked by default, against the cells found on disk, "
                        "because a grid reported COMPLETE while its own baseline was "
                        "unverified is the failure this tool exists to prevent.")
    p.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                   help="answer cells to expect per condition "
                        f"(default: {','.join(DEFAULT_CELLS)})")
    p.add_argument("--no-oracle", dest="oracle", action="store_false",
                   help="do not expect an oracle row")
    p.add_argument("--full-systems", default=",".join(DEFAULT_FULL_SYSTEMS),
                   help="systems that owe a full-context row (pair_turns is "
                        "deliberately excluded: its full-context prompt equals all_turns')")
    p.add_argument("--mem-model", default=None,
                   help="override the spec's mem_model when locating the keyed tree "
                        "(the store, qrels, retrieval and answers all inherit this key)")
    p.add_argument("--mem-variant", default=None,
                   help="the --mem-variant the store was ingested under, if any")
    p.add_argument("--skip-unrun", action="store_true",
                   help="omit systems with no store, no retrieval and no answers "
                        "(default: report them, so nothing is dropped silently)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="name every missing sample-id and question-id")
    args = p.parse_args()
    args.cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    args.full_systems = {s.strip() for s in args.full_systems.split(",") if s.strip()}

    if args.systems:
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    else:
        systems = sorted(d.name for d in (REPO / "experiments").iterdir()
                         if (d / args.spec_name).is_file())

    reports = []
    for system in systems:
        spec_path = REPO / "experiments" / system / args.spec_name
        if not spec_path.is_file():
            print(f"!! no spec for {system!r}", file=sys.stderr)
            return 2
        rep = check_system(system, load_spec(spec_path), args)
        # Dropped ONLY when the caller asked, and only for a system with no
        # trace of any stage. The default is to print it: a system that vanishes
        # from this report because its tree is keyed differently reads exactly
        # like a system that finished, which is the failure this tool exists to
        # make impossible.
        if args.skip_unrun and rep.get("note") and not rep.get("evidence"):
            continue
        reports.append(rep)
        print_system(rep, args)

    total = sum(r.get("problems", 0) for r in reports)
    print("\n" + "=" * 72)
    if not reports:
        print(f"No system has outputs under {args.out_root or 'the keyed tree'}.")
        return 2
    for r in reports:
        n = r.get("problems", 0)
        if r.get("note") and r.get("mismatch"):
            verdict = "KEY MISMATCH — later stages ran, qrels not at the expected path"
        elif r.get("note") and r.get("evidence"):
            verdict = "ingested, qrels not built yet"
        elif r.get("note"):
            verdict = "not run"
        else:
            verdict = "finished" if not n else f"{n} incomplete cell(s)"
        print(f"  {r['system']:<12} {verdict}")
    print("=" * 72)
    print("Grid is COMPLETE." if not total else
          f"{total} cell(s) not finished. Re-run the stage that owns them; every "
          f"stage resumes, so nothing already done is repeated.")
    return 0 if not total else 1


if __name__ == "__main__":
    sys.exit(main())
