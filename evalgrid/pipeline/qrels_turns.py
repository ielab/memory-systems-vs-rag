"""
evalgrid/pipeline/qrels_turns.py
----------------------------
Builds a per-sample BEIR/TREC dataset DIRECTLY from the raw conversation turns —
no memory ingestion, no LLM labeler. The corpus is the dataset's user turns; the
relevance labels are the dataset's own has_answer flags.

This one shared dataset feeds every turn-level baseline. None of them is a
memory "system" — they are all just answer modes over this corpus:
  * naive RAG      — retrieve over the turn corpus (evalgrid.pipeline.retrieve) then answer
                     the top-k (evalgrid.pipeline.answer --retriever dense ...).
  * full context   — answer from the WHOLE corpus, no retrieval
                     (evalgrid.pipeline.answer --retriever full).
  * oracle_context — answer straight from the evidence turns
                     (evalgrid.pipeline.answer --retriever oracle), i.e. the upper bound of
                     perfect retrieval. Oracle only ever reads the qrels
                     docs, so the extra (distractor) corpus turns don't change
                     its answer — they just make retrieval meaningful for
                     naive RAG. WHICH turns are evidence is the gold level
                     (--gold strict|turn|session, evalgrid/benchmarks/gold.py):
                     all three are written here, so an oracle can be run at any
                     of them with no rebuild.

--roles picks which turns become the corpus, and it also picks the output tree:

  all   (default) both speakers -> system segment "all_turns". This is the
        published Naive RAG baseline: LightMem's paper (arXiv 2510.18866,
        Table 7) scores Naive RAG at 98.21 on the 56 single-session-assistant
        questions, which is only reachable with assistant turns in the index.
        Unit text is prefixed with the speaker ("user: ..." / "assistant: ..."),
        because in a mixed corpus the speaker is real signal — an unattributed
        "I'd recommend the 35mm" is ambiguous — and because every memory system
        here sees role markers too (mem0's parse_messages emits "user: ...",
        a_mem templates "Speaker {role}says : ...").
  user  user turns only -> system segment "user_turns", text unprefixed. This
        was the sole behaviour before 2026-08, and the stores it wrote stay
        byte-valid. Keep it as the ablation that isolates what the assistant
        side contributes; do not use it as the main baseline, because no
        published LongMemEval table is computed that way.
  pair  one (user, assistant) EXCHANGE per document -> system segment
        "pair_turns". Same turns as "all", different document boundary: the
        retrievable unit is the exchange, not the utterance. This is the matched
        control for the systems that construct memory a pair at a time —
        lightmem's LongMemEval driver calls add_memory once per pair, memorybank
        stores one {query, response} record per pair, mem0's conversational
        ingest is per turn-pair. Against "all" it isolates document granularity
        with the content held constant: a question whose evidence is an
        assistant reply retrieves its own user turn for free, which no
        utterance-level corpus can do.

The three trees never collide, so the baselines can be built and compared.
Whichever is used, qrels label exactly the has_answer turns present in that
corpus (842 user + 54 assistant on longmemeval_s), and run_answer's oracle path
guards on doc-in-corpus anyway.

Two caveats when reading "pair" next to "all" — both are properties of the
representation, not bugs, but they must be stated wherever the rows share a
table:
  * a pair is relevant if EITHER side is, so the gold set collapses wherever
    both sides of one exchange are flagged. Recall@k denominators differ.
  * a pair doc is ~2x the text of a turn doc, so top-k pairs is ~2x the context
    of top-k turns. Compare at matched k or at matched context tokens (every
    answer record carries ``context_items``), and say which.

Writes, per sample, into
  outputs/_qrels/<all_turns|user_turns|pair_turns>/<dataset>/none/has_answer/<sample>/:
    corpus.jsonl              every selected turn (the retrievable units)
    queries.jsonl             the sample's question
    qrels/test_oracle_strict.tsv   gold "strict"  — the has_answer turn ids, score 1
    qrels/test_oracle_turn.tsv     gold "turn"    — strict + evidence sessions that flag nothing
    qrels/test_oracle_session.tsv  gold "session" — every turn of every evidence session

Every level is named explicitly — no level holds a bare test.tsv — and all three
are files over the SAME corpus and queries, which is why adding one invalidates
no retrieval run: a ranking does not depend on its labels.

The path coordinates (system from --roles, mem-model=none,
labeler-model=has_answer) are what evalgrid.pipeline.retrieve / evalgrid.pipeline.answer reconstruct
from those same flags, so no path overrides are needed downstream — see the
commands printed at the end.

Resumable: a sample whose qrels dir already has output is skipped (use
--overwrite to rebuild).

Usage:
  python -m evalgrid.pipeline.qrels_turns --dataset longmemeval_s              # all_turns
  python -m evalgrid.pipeline.qrels_turns --dataset longmemeval_s --roles user # user_turns
  python -m evalgrid.pipeline.qrels_turns --dataset longmemeval_s --roles pair # pair_turns
"""

import argparse

from evalgrid.lib.manifest import write_manifest
from evalgrid.lib.paths import NO_MEM_MODEL, labeler_segment, qrels_dir as build_qrels_dir

from evalgrid.benchmarks.adapters import DATASETS, build_adapter
from evalgrid.benchmarks.gold import (GOLD_LEVELS, RELABEL_LEVELS, STAMPED_GOLD_LEVEL,
                                      RelabelLabels, evidence_turn_ids, gold_filename,
                                      grade_units, relabel_evidence,
                                      turn_id as _turn_id)
from evalgrid.benchmarks.qrels import (check_corpus_matches, write_beir_sample,
                                       write_qrels_only)
from evalgrid.systems.base import MemoryUnit
from evalgrid.benchmarks.runner import _memory_saved, _sample_mem_path
# The pairing used by --roles pair. Imported from memorybank rather than
# reimplemented so the baseline's document boundary is the SAME boundary a
# pair-at-a-time system ingests on — that identity is the whole point of the
# mode, and a second copy of the loop would drift from it. store.py is pure
# stdlib (see its docstring), so this pulls in no system dependency.
from evalgrid.systems.memorybank.store import pair_turns

# Shared coordinates in the _qrels / _retrieval / _answers trees. There's no
# memory model (corpus = raw turns) and no LLM labeler (the has_answer flag IS
# the relevance label) — these sentinels just name that, and let the downstream
# scripts rebuild the same default paths from
# --mem-model none --labeler-model has_answer.
#
# The system segment encodes --roles, so the two baselines occupy separate trees
# and a retrieve/answer run always names the corpus it ranked. Both are listed in
# evalgrid.systems.registry.PSEUDO_SYSTEMS.
SYSTEM_BY_ROLES = {"all": "all_turns", "user": "user_turns", "pair": "pair_turns"}
MEM_MODEL = NO_MEM_MODEL   # lib/paths, so the sentinel has one spelling
LABELER = "has_answer"


def _pair_units(session, stamp):
    """Yield one MemoryUnit per (user, assistant) exchange in ``session``.

    Pairing is ``memorybank.store.pair_turns`` — the same call memorybank's own
    ingest makes, including its ``query_role`` heuristic
    ([memorybank/system.py:222](../systems/memorybank/system.py#L222)): pair on
    the "user" role where the dataset has one (LongMemEval), by alternation
    where roles are speaker names (LoCoMo). So a doc here has the same span as
    that system's memory record, which is what makes the row a control rather
    than a third arbitrary chunking.

    An unmatched turn becomes a one-sided document rather than being dropped.
    That is memorybank's and lightmem/locomo's behaviour, and deliberately NOT
    lightmem's LongMemEval driver, whose stride-2 loop silently discards 51 user
    + 19 assistant turns (our development notes). A baseline corpus that drops
    turns is not comparable to all_turns, and losing evidence turns to a
    formatting rule would be a measurement error, not an alignment.

    The unit id is the id of the turn that OPENS the pair, so pair ids live in
    the same namespace as all_turns/user_turns ids and ``source_turn_ids``
    carries both sides. Text keeps all_turns' speaker prefixes, one turn per
    line, so document boundary is the only variable between the two corpora.
    """
    turns = session.turns
    # Mirrors memorybank/system.py: a fixed asker role where the dataset has
    # one, alternation where roles are speaker names.
    query_role = "user" if any(t.role == "user" for t in turns) else None
    for query_turn, response_turn, indices in pair_turns(turns, query_role):
        sides = [t for t in (query_turn, response_turn) if t is not None]
        yield MemoryUnit(
            unit_id=_turn_id(session.session_id, indices[0]),
            text="\n".join(f"{t.role}: {t.content}" for t in sides),
            timestamp=stamp,
            # Either side carrying the evidence makes the exchange relevant —
            # the retrievable unit is the exchange.
            has_answer=any(t.has_answer for t in sides),
            source_turn_ids=[_turn_id(session.session_id, i) for i in indices],
        )


def _build_sample(sample, roles: str):
    """Return the corpus units for one sample.

    corpus_units: every selected turn as a MemoryUnit — the same type a
    SystemPlugin.iter_units yields, so the raw-turn corpus and a constructed
    memory corpus are built through one path and stay byte-comparable. The
    session timestamp travels in ``timestamp``, not folded into the text,
    because evalgrid.pipeline.answer's format_fact() prepends it at answer time.

    Under ``roles="user"`` the text is the bare turn content — a constant
    "user:" prefix would carry no signal, and this keeps the corpus byte-identical
    to what this script wrote before --roles existed. Under ``roles="all"`` the
    speaker IS signal, so it is prefixed into the TEXT rather than carried in
    metadata: write_beir_sample emits only ``time_stamp`` from a unit's metadata
    (adding a corpus field invalidates every retrieval cache keyed on it), and a
    retriever can only match what is in the text anyway.

    Under ``roles="pair"`` the document is a whole (user, assistant) exchange —
    see _pair_units. Same turns as ``"all"``, coarser boundary.

    The gold sets are NOT built here: _gold() labels these units at each level
    in evalgrid/benchmarks/gold.py by intersecting a unit's source_turn_ids with
    the level's evidence turns. Under "strict" that reproduces the old rule
    exactly — the has_answer turns under "user"/"all", the exchanges containing
    one under "pair" — which also means the gold set is SMALLER under "pair"
    wherever both sides of one exchange are flagged, so recall is not comparable
    across the two corpora without saying which one it was computed over.
    """
    units = []
    for session in sample.sessions:
        stamp = session.timestamp
        if roles == "pair":
            session_units = list(_pair_units(session, stamp))
        else:
            session_units = []
            for i, turn in enumerate(session.turns):
                if roles == "user" and turn.role != "user":
                    continue
                tid = _turn_id(session.session_id, i)
                session_units.append(MemoryUnit(
                    unit_id=tid,
                    text=turn.content if roles == "user" else f"{turn.role}: {turn.content}",
                    timestamp=stamp,
                    has_answer=turn.has_answer,
                    source_turn_ids=[tid],   # a raw turn is its own provenance
                ))
        units.extend(session_units)
    return units


def _gold(units, sample, level: str):
    """Label ``units`` at one gold level: [(query, relevant_unit_ids), ...].

    A unit is relevant iff its provenance meets the level's evidence turns.
    Every raw-turn unit carries provenance by construction (a turn is its own
    source, a pair is its two sides), so this is exact here; over constructed
    memory the same rule runs in qrels_has_answer against whatever the ingester
    recorded.
    """
    labeled = []
    for query in sample.queries:
        evidence = evidence_turn_ids(sample, query, level)
        labeled.append((query, [u.unit_id for u in units
                                if evidence.intersection(u.source_turn_ids)]))
    return labeled


def _relabel_gold(units, sample, labels: RelabelLabels):
    """Both relabel levels for one sample: {level: [(query, {unit_id: grade}), ...]}."""
    return {level: [(q, grade_units(units, relabel_evidence(sample, q, level, labels)))
                    for q in sample.queries]
            for level in RELABEL_LEVELS}


def main():
    parser = argparse.ArgumentParser(
        description="Build a user-turn BEIR corpus + has_answer qrels (for the naive-RAG / full-context / oracle baselines)")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="Benchmark dataset key (e.g. longmemeval_s). Only datasets with "
                             "turn-level has_answer flags are meaningful (LongMemEval).")
    parser.add_argument("--roles", default="all", choices=sorted(SYSTEM_BY_ROLES),
                        help="Which turns form the corpus, and at what document "
                             "granularity. 'all' (default) = both speakers, one doc per "
                             "turn, under system 'all_turns' — the published Naive RAG "
                             "baseline. 'user' = user turns only, under 'user_turns' — an "
                             "ablation. 'pair' = both speakers, one doc per (user, "
                             "assistant) exchange, under 'pair_turns' — the matched control "
                             "for pair-at-a-time systems. See the module docstring.")
    parser.add_argument("--out", default=None,
                        help="Override the qrels output dir (default: "
                             "outputs/_qrels/<all_turns|user_turns|pair_turns>/<dataset>/none/has_answer/)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N samples")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild samples that already have qrels written")
    parser.add_argument("--relabel-set", default=None,
                        help="Also write the relabel gold levels (relabel_union, relabel_union_insession, relabel; "
                             "evalgrid/benchmarks/gold.py) from this relabel label set, e.g. "
                             "graded_v4 = outputs/_relabel/<dataset>/labels/graded_v4/labels.jsonl")
    parser.add_argument("--relabel-only", action="store_true",
                        help="With --relabel-set: add ONLY the relabel qrels files to samples "
                             "that are already built. corpus.jsonl, queries.jsonl and the "
                             "existing levels are not touched.")
    args = parser.parse_args()
    if args.relabel_only and not args.relabel_set:
        parser.error("--relabel-only needs --relabel-set")

    system = SYSTEM_BY_ROLES[args.roles]
    out_dir = args.out or build_qrels_dir(system, args.dataset, MEM_MODEL, labeler_segment(LABELER))

    print(f"System  : {system}")
    print(f"Roles   : {args.roles}")
    print(f"Dataset : {args.dataset}")
    print(f"Out dir : {out_dir}")
    labels = RelabelLabels.load(args.dataset, args.relabel_set) if args.relabel_set else None
    if labels:
        print(f"Relabel : {labels.label_set} ({labels.path}, sha256 {labels.sha256[:12]})")
    # --relabel-only adds files to a tree another run built, so it must not
    # overwrite that run's manifest. The relabel provenance gets its own file.
    if not args.relabel_only:
        write_manifest(out_dir, args, stage="qrels_turns")
    if labels:
        write_manifest(out_dir, args, stage="qrels_turns.relabel",
                       filename="manifest_relabel.json", extra=labels.provenance())

    adapter = build_adapter(args.dataset)
    samples = list(adapter.samples())
    if args.limit:
        samples = samples[:args.limit]

    written = skipped = no_gold = not_built = empty_union = 0
    tot_docs = 0
    levels = (() if args.relabel_only else GOLD_LEVELS) + (RELABEL_LEVELS if labels else ())
    tot_rel = {level: 0 for level in levels}
    for sample in samples:
        sample_out = _sample_mem_path(out_dir, sample.sample_id)
        if args.relabel_only:
            if not _memory_saved(sample_out):
                not_built += 1
                continue
            units = _build_sample(sample, args.roles)
            check_corpus_matches(sample_out, units)
            gold = _relabel_gold(units, sample, labels)
            write_qrels_only(sample_out, gold)
            written += 1
            tot_docs += len(units)
            for level, labeled in gold.items():
                tot_rel[level] += sum(len(ids) for _, ids in labeled)
            empty_union += sum(1 for _, ids in gold["relabel_union"] if not ids)
            continue
        if not args.overwrite and _memory_saved(sample_out):
            skipped += 1
            continue
        units = _build_sample(sample, args.roles)
        # LongMemEval has one query per sample, but this generalises to several
        # queries sharing the corpus — every level labels every query.
        gold = {level: _gold(units, sample, level) for level in GOLD_LEVELS}
        if labels:
            gold.update(_relabel_gold(units, sample, labels))
            empty_union += sum(1 for _, ids in gold["relabel_union"] if not ids)
        write_beir_sample(sample_out, units, gold[STAMPED_GOLD_LEVEL],
                          extra_gold={k: v for k, v in gold.items() if k != STAMPED_GOLD_LEVEL})
        written += 1
        tot_docs += len(units)
        for level, labeled in gold.items():
            tot_rel[level] += sum(len(ids) for _, ids in labeled)
        if not any(ids for _, ids in gold[STAMPED_GOLD_LEVEL]):
            no_gold += 1

    print(f"\nDone — qrels written to {out_dir}")
    print(f"  written : {written} sample(s); {tot_docs} corpus turns")
    for level in levels:
        print(f"  gold {level:14}: {tot_rel[level]:6} relevant  ({gold_filename(level)})")
    if empty_union:
        print(f"  empty   : {empty_union} question(s) have NO unit at relabel_union — the judge "
              f"found nothing and the benchmark flags nothing (abstention questions, mostly). "
              f"Their oracle prompt is empty at that level; report them separately.")
    if not_built:
        print(f"  not built: {not_built} sample(s) have no qrels dir yet — --relabel-only adds "
              f"to built samples; build them first without it.")
    if no_gold:
        print(f"  no gold : {no_gold} sample(s) have an EMPTY strict gold set (abstention "
              f"questions, whose evidence sessions flag nothing). Their oracle prompt is "
              f"empty at that level — which is why the oracle defaults to --gold turn.")
    if skipped:
        print(f"  skipped : {skipped} (already built — use --overwrite to rebuild)")

    em = "sentence-transformers/all-MiniLM-L6-v2"
    print(
        "\nNext:\n"
        "  # naive RAG — dense retrieval over the turn corpus, then answer top-10\n"
        f"  python -m evalgrid.pipeline.retrieve --system {system} --dataset {args.dataset} \\\n"
        f"      --mem-model {MEM_MODEL} --labeler-model {LABELER} \\\n"
        f"      --retriever dense --emb-model {em}\n"
        f"  python -m evalgrid.pipeline.answer --system {system} --dataset {args.dataset} \\\n"
        f"      --mem-model {MEM_MODEL} --labeler-model {LABELER} \\\n"
        f"      --retriever dense --emb-model {em} --top-k 10 --gen-model gpt-4o-mini\n"
        "\n"
        "  # oracle_context — answer straight from the has_answer turns (no retrieval)\n"
        "  # oracle defaults to --gold turn; add --gold strict|session for the others\n"
        f"  python -m evalgrid.pipeline.answer --system {system} --dataset {args.dataset} \\\n"
        f"      --mem-model {MEM_MODEL} --labeler-model {LABELER} \\\n"
        f"      --retriever oracle --gen-model gpt-4o-mini"
    )


if __name__ == "__main__":
    main()
