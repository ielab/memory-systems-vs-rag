"""
evalgrid/pipeline/qrels_has_answer.py
------------------------------------
Builds per-sample BEIR/TREC qrels DIRECTLY from a memory system's saved
points.jsonl — no LLM labeler. The corpus is every saved fact; the relevance
labels are the facts whose own ``payload.has_answer`` flag is set (propagated
from the source turn during ingestion).

This is the memory-level analogue of evalgrid.pipeline.qrels_turns: that one
labels raw user TURNS by has_answer, this one labels the extracted memory FACTS
by the has_answer the ingester stamped onto each fact. It needs a memory system
that records has_answer per fact (e.g. the current lightmem) — older saved
memories without the key produce empty qrels.

It is a free, deterministic source of relevance labels, replacing an LLM
labeler when you trust the per-fact has_answer flag: same corpus, same downstream
wiring, but the gold set is the flag instead of a labeler's judgment.

Writes, per sample, into
  outputs/_qrels/<system>/<dataset>/<mem-model>/has_answer/<sample>/:
    corpus.jsonl              every saved fact (the retrievable units)
    queries.jsonl             the sample's question(s)
    qrels/test_oracle_strict.tsv   gold "strict" — the has_answer fact ids, score 1
    qrels/test_oracle_turn.tsv     gold "turn"
    qrels/test_oracle_session.tsv  gold "session"

The wider gold levels (evalgrid/benchmarks/gold.py) are recomputed from each
unit's ``source_turn_ids`` — the turn ids the ingester recorded IN the saved
store — so they cost one pass over saved memory and NO re-ingest. "strict" is
left exactly as the ingester stamped it (STAMPED_GOLD_LEVEL), so the numbers
computed from it are unchanged — only its FILENAME is now explicit.

A level cannot be finer than the units the system built: a hypermem episode's
provenance is a whole session, so its "turn" gold and its "session" gold are
nearly the same set, and a simplemem window spanning three sessions is gold at
"session" if ANY of the three is evidence. The levels still mean one thing
across systems — "which turns count as evidence" — but how precisely a system
can be held to one is a property of its representation, which is what
has_answer_granularity above reports.

The labeler-model path segment is the fixed sentinel ``has_answer`` (mirroring
qrels_turns), so it never collides with an LLM labeler's qrels under the same
system/dataset/mem-model, and evalgrid.pipeline.retrieve / evalgrid.pipeline.answer
/ evalgrid.pipeline.eval_retrieval reconstruct the same default path from
--labeler-model has_answer.

Resumable: a sample whose qrels dir already has output is skipped (use
--overwrite to rebuild).

Usage:
  python -m evalgrid.pipeline.qrels_has_answer --system lightmem \\
      --dataset longmemeval_s --mem-model Qwen/Qwen3-30B-A3B-Instruct-2507
"""

import argparse
import json
import os
import sys

from evalgrid.lib.cli import (
    add_dataset_arg,
    add_exp_arg,
    add_limit_arg,
    add_mem_model_arg,
    add_mem_variant_arg,
    add_overwrite_arg,
    add_system_arg,
    apply_spec,
    require_resolved,
)
from evalgrid.lib.manifest import write_manifest
from evalgrid.lib.paths import ExperimentPaths, labeler_segment

from evalgrid.benchmarks.adapters import DATASETS, build_adapter
from evalgrid.benchmarks.gold import (GOLD_LEVELS, RELABEL_LEVELS, STAMPED_GOLD_LEVEL,
                                      RelabelLabels, grade_units, relabel_evidence,
                                      turn_id as _turn_id,
                                      evidence_turn_ids, gold_filename)
from evalgrid.benchmarks.qrels import (check_corpus_matches, write_beir_sample,
                                       write_qrels_only)
from evalgrid.benchmarks.runner import _memory_saved, _sample_mem_path
from evalgrid.systems.registry import available_systems, get_plugin

# Fixed labeler-model path segment: the has_answer flag IS the relevance label,
# there is no LLM labeler. Downstream scripts rebuild the same default path from
# --labeler-model has_answer.
LABELER = "has_answer"


def _relabel_gold(units, sample, queries, labels: RelabelLabels):
    """Both relabel levels: {level: [(query, {unit_id: grade}), ...]}.

    Recomputed from each unit's ``source_turn_ids`` like the wider published
    levels, so a system that records coarse provenance (a whole session per
    unit) gets coarse relabel gold too — the granularity caveat printed above
    applies to these levels unchanged.
    """
    return {level: [(q, grade_units(units, relabel_evidence(sample, q, level, labels)))
                    for q in queries]
            for level in RELABEL_LEVELS}


def _read_sidecar(sample_mem: str, filename: str):
    """unit_id -> source_turn_ids from a provenance sidecar, or None when absent."""
    path = os.path.join(sample_mem, filename)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return {rec["unit_id"]: list(rec.get("source_turn_ids") or [])
                for rec in map(json.loads, filter(str.strip, f))}


def main():
    parser = argparse.ArgumentParser(
        description="Build per-sample BEIR/TREC qrels from saved memory's per-fact has_answer flag")
    add_exp_arg(parser)
    add_system_arg(parser, choices=available_systems(),
                   help="Memory system whose saved memory to flatten into a corpus")
    add_dataset_arg(parser, choices=list(DATASETS), help="Benchmark dataset key")
    add_mem_model_arg(parser, help="Memory model used to build the saved memory (keys outputs/_memory/)")
    add_mem_variant_arg(parser)
    parser.add_argument("--mem-dir", default=None,
                        help="Override the saved-memory dir "
                             "(default: outputs/_memory/<system>/<dataset>/<mem-model>/)")
    parser.add_argument("--out", default=None,
                        help="Override the qrels output dir "
                             "(default: outputs/_qrels/<system>/<dataset>/<mem-model>/has_answer/)")
    add_limit_arg(parser, help="Only process first N samples")
    add_overwrite_arg(parser, help="Rebuild samples that already have qrels written")
    parser.add_argument("--relabel-set", default=None,
                        help="Also write the relabel gold levels (relabel_union, relabel_union_insession, relabel; "
                             "evalgrid/benchmarks/gold.py) from this relabel label set, e.g. "
                             "graded_v4 = outputs/_relabel/<dataset>/labels/graded_v4/labels.jsonl")
    parser.add_argument("--relabel-only", action="store_true",
                        help="With --relabel-set: add ONLY the relabel qrels files to samples "
                             "that are already built. corpus.jsonl, queries.jsonl and the "
                             "existing levels are not touched.")
    parser.add_argument("--provenance-sidecar", default=None,
                        help="Read each unit's source_turn_ids from this file beside the "
                             "store (one {unit_id, source_turn_ids} record per unit) instead "
                             "of from the store, e.g. provenance_turn.jsonl from "
                             "evalgrid.tools.simplemem_attribute_provenance. Writes under its "
                             "own labeler segment, has_answer-<file stem>, so the store's own "
                             "qrels are not touched; 'strict' is then recomputed from the "
                             "sidecar's provenance rather than read from the stamped flag.")
    parser.add_argument("--skip-question-types", default=None,
                        help="Comma-separated question_type values to skip entirely "
                             "(no qrels written) — e.g. single-session-assistant.")
    args = parser.parse_args()
    apply_spec(parser, args)
    require_resolved(parser, args, "system", "dataset", "mem_model")
    if args.relabel_only and not args.relabel_set:
        parser.error("--relabel-only needs --relabel-set")
    skip_types = {t.strip() for t in args.skip_question_types.split(",")} if args.skip_question_types else set()

    plugin = get_plugin(args.system)
    paths = ExperimentPaths(system=args.system, dataset=args.dataset, mem_model=args.mem_model,
                            mem_variant=args.mem_variant)
    mem_dir = args.mem_dir or paths.memory()
    labeler = LABELER
    if args.provenance_sidecar:
        labeler += "-" + os.path.splitext(os.path.basename(args.provenance_sidecar))[0]
    out_dir = args.out or paths.qrels(labeler_segment(labeler))
    if not os.path.isdir(mem_dir):
        sys.exit(f"No saved memory at {mem_dir} — run evalgrid.pipeline.ingest first.")

    print(f"System  : {args.system}")
    print(f"Dataset : {args.dataset}")
    print(f"Mem model: {args.mem_model}")
    print(f"Mem dir : {mem_dir}")
    print(f"Out dir : {out_dir}")
    if args.provenance_sidecar:
        print(f"Provenance: {args.provenance_sidecar} beside each store "
              f"(labeler segment {labeler_segment(labeler)!r}, --labeler-model {labeler!r} "
              f"downstream)")

    # Say what the labels mean here, at the only moment a reader is looking.
    # These qrels are comparable between systems that label at the SAME
    # granularity and not otherwise: one evidence turn produces 1-2 labels under
    # "turn" and 16-25 under "session", so a recall@k table mixing the two ranks
    # systems by how coarsely they record provenance. Both plugins that label
    # coarsely have said so in records_has_answer since they were written; the
    # cost of leaving it there was that nothing said it where the qrels are
    # actually built. Printed, not enforced — a single-system sweep over
    # SimpleMem variants is a legitimate use of window-level labels, and only
    # the caller knows which of the two they are doing.
    granularity = getattr(plugin, "has_answer_granularity", "turn")
    if args.provenance_sidecar:
        # The sidecar replaces the store's provenance, so the store's declared
        # granularity no longer describes these labels.
        granularity = "turn"
    print(f"Label granularity: {granularity}"
          + ("" if granularity == "turn" else
             f"  <-- every unit from the evidence {granularity} is labelled "
             f"relevant, not just the evidence turn"))
    if granularity != "turn":
        print(f"    Comparable only against other {granularity}-level systems "
              f"({', '.join(sorted(n for n in available_systems() if getattr(get_plugin(n), 'has_answer_granularity', 'turn') == granularity))}). "
              f"For a cross-system retrieval table use an LLM labeler instead.",
              file=sys.stderr)

    labels = RelabelLabels.load(args.dataset, args.relabel_set) if args.relabel_set else None
    if labels:
        print(f"Relabel : {labels.label_set} ({labels.path}, sha256 {labels.sha256[:12]})")
    # --relabel-only adds files to a tree another run built, so it must not
    # overwrite that run's manifest. The relabel provenance gets its own file.
    if not args.relabel_only:
        write_manifest(out_dir, args, stage="qrels_has_answer",
                       # Into the manifest as well as onto the terminal: a qrels dir
                       # outlives the run that printed the warning, and eval_retrieval
                       # reads the dir, not the scrollback.
                       extra={"has_answer_granularity": granularity,
                              "provenance_sidecar": args.provenance_sidecar})
    if labels:
        write_manifest(out_dir, args, stage="qrels_has_answer.relabel",
                       filename="manifest_relabel.json",
                       extra={**labels.provenance(), "has_answer_granularity": granularity,
                              "provenance_sidecar": args.provenance_sidecar})

    adapter = build_adapter(args.dataset)
    samples = list(adapter.samples())
    if args.limit:
        samples = samples[:args.limit]

    written = skipped = skipped_types = missing = no_flag = no_gold = no_prov = 0
    no_prov_sample = bad_prov_sample = 0
    not_built = empty_union = no_sidecar = 0
    #: Whether this benchmark localises evidence per question — a property of the
    #: adapter, uniform across its samples, read once so the closing report can
    #: explain an empty gold set in the terms that actually apply to it.
    per_query = bool(samples and samples[0].evidence_is_per_query)
    tot_docs = 0
    levels = (() if args.relabel_only else GOLD_LEVELS) + (RELABEL_LEVELS if labels else ())
    tot_rel = {level: 0 for level in levels}
    for sample in samples:
        queries = [q for q in sample.queries if (q.question_type or "") not in skip_types]
        if not queries:
            skipped_types += 1
            continue
        sample_out = _sample_mem_path(out_dir, sample.sample_id)
        if args.relabel_only:
            if not _memory_saved(sample_out):
                not_built += 1
                continue
        elif not args.overwrite and _memory_saved(sample_out):
            skipped += 1
            continue
        sample_mem = _sample_mem_path(mem_dir, sample.sample_id)
        if not _memory_saved(sample_mem):
            print(f"  [skip] {sample.sample_id!r} — no saved memory at {sample_mem!r}")
            missing += 1
            continue
        if not plugin.records_has_answer(sample_mem):
            print(f"  [skip] {sample.sample_id!r} — saved memory records no has_answer "
                  f"provenance (re-ingest with the has_answer-aware version)")
            no_flag += 1
            continue
        units = list(plugin.iter_units(sample_mem))
        if args.provenance_sidecar:
            sidecar = _read_sidecar(sample_mem, args.provenance_sidecar)
            if sidecar is None:
                print(f"  [skip] {sample.sample_id!r} — no {args.provenance_sidecar} beside the "
                      f"store; build it first")
                no_sidecar += 1
                continue
            uncovered = [u.unit_id for u in units if u.unit_id not in sidecar]
            if uncovered:
                sys.exit(f"{sample.sample_id}: {len(uncovered)} unit(s) missing from "
                         f"{args.provenance_sidecar}, e.g. {uncovered[0]!r} — the sidecar was "
                         f"built from a different store; rebuild it with --overwrite")
            for u in units:
                u.source_turn_ids = sidecar[u.unit_id]

        # DOES THIS STORE'S PROVENANCE POINT AT THIS SAMPLE'S TURNS?
        #
        # Every gold level below is an intersection of ``source_turn_ids`` with
        # turn ids gold.py built, so the two have to be the same namespace or
        # every intersection is empty and every qrels file is silently blank.
        # The gate above does NOT establish that: it asks whether units carry
        # the ``has_answer`` key, which is the precondition for the STAMPED
        # level. On a per-question benchmark there is no stamped level, so the
        # two questions come apart exactly where it matters.
        #
        # Checking that provenance merely EXISTS is not enough either, and that
        # is the more dangerous half. Two ways to have provenance and still
        # label nothing:
        #   * empty — lightmem's iter_units documents a period when it read a
        #     payload key nothing ever wrote, so every unit came back with
        #     source_turn_ids == [] while has_answer was present;
        #   * FOREIGN — ids in some other namespace. A dataset adapter that
        #     leaves its own message ids in Turn.turn_id gets them recorded
        #     verbatim by every system (they all write
        #     ``turn.turn_id or f"{session_id}:t{i}"``), so the store fills with
        #     ids like "0"/"1"/"2" while gold.py emits "session_1:t24". Nothing
        #     raises, nothing is empty, and every recall reads 0.0.
        #
        # So the check is the one that actually matters: does ANY unit cite a
        # turn belonging to THIS sample? That subsumes both failures, and it
        # cannot false-positive on a merely BAD memory system — a system that
        # stored anything at all cites turns from the sample it just ingested,
        # however poorly it summarised them. Retrieval quality is not on trial
        # here; namespace agreement is.
        sample_turn_ids = {_turn_id(s.session_id, i)
                           for s in sample.sessions for i in range(len(s.turns))}
        cited = {tid for u in units for tid in u.source_turn_ids}
        if units and not (cited & sample_turn_ids):
            detail = ("carry no source_turn_ids at all" if not cited else
                      f"cite {len(cited)} id(s) that belong to no turn of this sample, "
                      f"e.g. {sorted(cited)[:3]} — expected the "
                      f"'<session_id>:t<i>' form, e.g. {sorted(sample_turn_ids)[:1]}")
            fix = ("Re-ingest, or rebuild provenance "
                   "(evalgrid.tools.lightmem_rebuild_provenance for lightmem).")
            if sample.evidence_is_per_query:
                # Fatal: every level is recomputed from provenance and there is
                # no stamped level to fall back on, so writing this sample would
                # produce three empty qrels files.
                print(f"  [skip] {sample.sample_id!r} — {len(units)} unit(s) {detail}. "
                      f"Every gold level is an intersection with provenance, so all of "
                      f"them would be empty. {fix}")
                no_prov_sample += 1
                continue
            # Not fatal here: "strict" comes from the ingester's own stamped
            # flag rather than from provenance, so the sample is still worth
            # writing. Only the recomputed levels are lost, and that must not be
            # discovered later as a mystery zero.
            print(f"  [warn] {sample.sample_id!r} — {len(units)} unit(s) {detail}. "
                  f"'strict' comes from the stamped flag and survives, but 'turn' and "
                  f"'session' are recomputed from provenance and will be EMPTY for this "
                  f"sample. {fix}")
            bad_prov_sample += 1
            if args.relabel_only:
                continue   # the relabel levels are provenance-only: they would be empty

        if args.relabel_only:
            check_corpus_matches(sample_out, units)
            gold = _relabel_gold(units, sample, queries, labels)
            write_qrels_only(sample_out, gold)
            written += 1
            tot_docs += len(units)
            for level, labeled in gold.items():
                tot_rel[level] += sum(len(ids) for _, ids in labeled)
            empty_union += sum(1 for _, ids in gold["relabel_union"] if not ids)
            continue

        # "strict" stays the flag the INGESTER stamped on the unit, not a
        # recomputation: it is what every existing metrics.json was scored
        # against, and it is defined even for the units whose provenance came
        # back empty (lightmem points written before source_turn_ids existed).
        # The wider levels are recomputed here from provenance — no re-ingest,
        # because the saved store already carries the turn ids each unit came
        # from. Where both are defined they agree exactly;
        #
        # EXCEPT on a benchmark that localises evidence per question. The
        # stamped flag comes from ``Turn.has_answer``, which is one flag per
        # SAMPLE, so on BEAM's 20-questions-per-conversation shape it is the
        # union over all 20 — and the line below would hand that identical union
        # to every one of them as its own gold set, a median 45 units where the
        # question's evidence is a median 2. There is no per-question stamped
        # level to write, so strict is recomputed from provenance with the rest
        # and means what the other levels mean.
        #
        # And under --provenance-sidecar: the stamped flag was set from the
        # store's own provenance, which the sidecar exists to replace — keeping
        # it would leave strict at the old granularity while every other level
        # moved.
        stamped = (STAMPED_GOLD_LEVEL
                   if not (sample.evidence_is_per_query or args.provenance_sidecar) else None)
        gold = {}
        if stamped:
            gold[stamped] = [(q, [u.unit_id for u in units if u.has_answer])
                             for q in queries]
        for level in GOLD_LEVELS:
            if level == stamped:
                continue
            gold[level] = []
            for q in queries:
                evidence = evidence_turn_ids(sample, q, level)
                gold[level].append((q, [u.unit_id for u in units
                                        if evidence.intersection(u.source_turn_ids)]))
        if labels:
            gold.update(_relabel_gold(units, sample, queries, labels))
            empty_union += sum(1 for _, ids in gold["relabel_union"] if not ids)
        # LongMemEval has one query per sample; this generalises to several
        # queries sharing the corpus.
        write_beir_sample(sample_out, units, gold[STAMPED_GOLD_LEVEL],
                          extra_gold={k: v for k, v in gold.items()
                                      if k != STAMPED_GOLD_LEVEL})
        written += 1
        tot_docs += len(units)
        for level, labeled in gold.items():
            tot_rel[level] += sum(len(ids) for _, ids in labeled)
        no_prov += sum(1 for u in units if not u.source_turn_ids)
        if not any(ids for _, ids in gold[STAMPED_GOLD_LEVEL]):
            no_gold += 1

    print(f"\nDone — qrels written to {out_dir}")
    print(f"  written : {written} sample(s); {tot_docs} corpus facts")
    for level in levels:
        print(f"  gold {level:14}: {tot_rel[level]:6} relevant  ({gold_filename(level)})")
    if empty_union:
        print(f"  empty   : {empty_union} question(s) have NO fact at relabel_union — no fact "
              f"cites a judged or published evidence turn. Their oracle prompt is empty at "
              f"that level; report them separately.")
    if no_sidecar:
        print(f"  no sidecar: {no_sidecar} sample(s) skipped — no {args.provenance_sidecar} "
              f"beside the store.")
    if not_built:
        print(f"  not built: {not_built} sample(s) have no qrels dir yet — --relabel-only adds "
              f"to built samples; build them first without it.")
    if no_gold:
        # Two different causes, and widening the level only helps the first.
        print(f"  no gold : {no_gold} sample(s) have an EMPTY gold set at the narrowest "
              f"level." + ("  Every level here is recomputed from provenance, so a wider "
                           "--gold does NOT repair it — the evidence simply did not survive "
                           "into any unit."
                           if per_query else
                           "  No fact is flagged has_answer; their oracle prompt is empty "
                           "at that level — answer them at --gold turn or wider, which is "
                           "the oracle's default."))
    if no_prov:
        print(f"  no prov : {no_prov} unit(s) carry no source_turn_ids, so they can never "
              f"be gold at a recomputed level" + ("." if per_query else
              " (they keep whatever the ingester stamped, which is what 'strict' reads)"))
    if no_prov_sample:
        print(f"  NO PROV : {no_prov_sample} sample(s) SKIPPED — not one unit cited a turn "
              f"of its own sample, so every level would have been empty. These are broken "
              f"stores, not samples without evidence.")
    if bad_prov_sample:
        print(f"  BAD PROV: {bad_prov_sample} sample(s) written whose units cite no turn of "
              f"their own sample — 'strict' is the stamped flag and survives, but their "
              f"'turn'/'session' gold is empty. Re-ingest before reading a recomputed level.")
    if no_flag:
        print(f"  no flag : {no_flag} (saved facts predate the has_answer key — re-ingest)")
    if skipped_types:
        print(f"  skipped : {skipped_types} (question_type in --skip-question-types: {sorted(skip_types)})")
    if missing:
        print(f"  missing : {missing} (no saved memory)")
    if skipped:
        print(f"  skipped : {skipped} (already built — use --overwrite to rebuild)")

    print(
        "\nNext (e.g. dense retrieval eval over the saved facts):\n"
        f"  python -m evalgrid.pipeline.retrieve --system {args.system} --dataset {args.dataset} \\\n"
        f"      --mem-model {args.mem_model} --labeler-model {LABELER} \\\n"
        f"      --retriever dense --emb-model Qwen/Qwen3-Embedding-0.6B\n"
        f"  python -m evalgrid.pipeline.eval_retrieval --system {args.system} --dataset {args.dataset} \\\n"
        f"      --mem-model {args.mem_model} --labeler-model {LABELER} --retriever dense"
    )


if __name__ == "__main__":
    main()
