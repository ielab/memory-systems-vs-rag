"""
evalgrid/benchmarks/gold.py
---------------------------
Gold-set LEVELS — the four answers to "which turns are the evidence for this
question" — and the one function that turns a level into a set of turn ids.

Both qrels builders (evalgrid/pipeline/qrels_turns.py over raw turns,
evalgrid/pipeline/qrels_has_answer.py over constructed memory) label a unit
relevant iff its ``source_turn_ids`` meet that set. Putting the expansion here
rather than in either builder is what makes an oracle over lightmem's facts and
an oracle over raw turns the SAME condition asked of two representations, which
is the only way the two rows belong in one table.

The levels, widest last:

  strict   exactly the turns LongMemEval flags ``has_answer``. The published
           retrieval gold — 896 turns over longmemeval_s — and the definition
           every pre-levels metrics file was computed under. Never redefine it;
           add a level instead. NOT the default anything: evaluation defaults to
           "turn" (EVAL_GOLD_LEVEL) because strict leaves 21 questions with no
           gold at all.
  turn     strict, PLUS every turn of an evidence session that flags nothing.
           strict is silent about 94 of longmemeval_s's 948 evidence sessions —
           listed in ``answer_session_ids``, no turn inside marked — an
           annotation gap that calls for a sensitivity check. This level
           IS that check, as a gold set. 45 of the 94 (529 turns) sit in 32
           non-abstention multi-hop questions whose other hops ARE flagged, so
           a strict oracle answers those with a hop missing; the other 49 sit in
           the abstention (``_abs``) questions, where the evidence sessions are
           deliberately distractors — and for the 21 that flag nothing at all,
           strict leaves the oracle prompt EMPTY. An abstention scored against
           no context is not a test of abstention.
  session  every turn of every evidence session. The upper bound of perfect
           SESSION retrieval, and the level a session-granularity system
           (hypermem) can actually be held to.

Two more levels come from an LLM RELABEL of the evidence rather than from the
benchmark — ``relabel_union`` ("turn" + judged), ``relabel_union_insession``
("turn" + judged inside the evidence sessions) and ``relabel`` (judged only).
They are graded, exist only for relabelled datasets, and are written only on
request; see RELABEL_LEVELS.

There is deliberately NO "pair" level — no level that widens a flagged turn to
the other side of its exchange. It was built and then removed, because the
premise is false: ``has_answer`` is already role-aware. The dataset flags the
assistant turn exactly where the assistant turn is the evidence — 51 of its 54
assistant flags are the 56 ``single-session-assistant`` questions, and four of
six categories carry zero assistant labels. A widened gold set does score better on a lexical answer-coverage probe,
but it scores no better than adding a RANDOM other turn of the same role from
the same session — measured on raw turns (57.9% vs 57.4%) and on three systems'
constructed memory (lightmem 21.5% vs 23.2%, mem0 25.5% vs 26.4%, a_mem 62.5%
vs 62.1%). The apparent gain is text volume, not evidence. Widening an upper
bound with text indistinguishable from random adjacent text does not raise the
bound, it just stops it being one — and on ``user_turns`` it would have
manufactured gold for the 51 questions that corpus provably cannot answer,
defeating a ceiling that corpus is meant to set.

An evidence session is one named in ``Query.evidence_session_ids`` OR one
holding a flagged turn. The union, not either alone: LongMemEval populates the
first and 94 of those sessions flag nothing, while benchmarks that ship only
turn-level evidence (LoCoMo) populate the second. Levels stay defined for both.

WHERE THE EVIDENCE COMES FROM depends on the benchmark, and there are two
sources because ``Turn.has_answer`` cannot express the second. It is a flag on
the SAMPLE, so it means "evidence for the question asked here" — exactly right
for LongMemEval, where a sample IS a question, and unusable for a benchmark that
hangs many questions off one conversation. BEAM hangs 20, with disjoint
evidence: their union is a median 45 turns of 271 where one question's is a
median 2, so a flag-derived gold set there is 22x too wide and every level
computed from it is meaningless. ``Query.evidence_turn_ids`` is the per-question
source, resolved by ``query_evidence_turn_ids`` below and preferred at every
level where a query populates it. The levels themselves do not change meaning —
only which set they widen from.

Every level is written to its own explicitly-named file next to the corpus —
``test_oracle_<level>.tsv``, with no level holding a bare ``test.tsv`` — so a
level costs a few KB of qrels and NOTHING else — same corpus.jsonl, same run.tsv. Changing
the gold set never invalidates a retrieval run, because a ranking does not
depend on the labels it is scored against.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

from evalgrid.benchmarks.types import Query, Sample

#: Gold levels, narrowest first. "strict" is the default everywhere: it is what
#: the trees already hold, and every other level is a superset of it.
GOLD_LEVELS = ("strict", "turn", "session")

#: The level built from the flag the INGESTER stamped on each unit, rather than
#: recomputed from provenance — see qrels_has_answer. Also what a run recorded
#: before gold levels existed was scored against, so it is the right fallback
#: when a manifest carries no ``gold`` key.
#:
#: This is NOT "the default level": nothing defaults to it, and no file or path
#: is named after it implicitly. It names one specific construction, and the
#: constant exists so the code can say WHICH construction it means instead of
#: spelling a bare "strict" at six call sites.
STAMPED_GOLD_LEVEL = "strict"

#: The level EVALUATION happens at unless told otherwise — both the evidence the
#: oracle answers from (pipeline/answer.py --retriever oracle) and the labels a
#: ranking is scored against (pipeline/eval_retrieval.py).
#:
#: "turn", not "strict", because strict is not a complete account of the
#: evidence. LongMemEval leaves 94 of its 948 evidence sessions with no turn
#: flagged — sessions it names in
#: ``answer_session_ids`` while marking nothing inside. Two consequences, and
#: both are measurement errors rather than conservative choices:
#:
#:   * for the 21 abstention questions where NO session flags anything, strict
#:     gold is EMPTY. The oracle answers from an empty prompt, and retrieval is
#:     scored against nothing — an empty qrels makes recall undefined, not zero.
#:   * another 45 unflagged sessions sit in 32 non-abstention multi-hop
#:     questions whose other hops ARE flagged. A strict oracle answers those
#:     with a hop missing, and a strict recall@k counts a system that DID
#:     retrieve that hop as having retrieved a distractor.
#:
#: "turn" repairs exactly those cases — an evidence session that flags nothing
#: contributes all of itself — and is otherwise identical to strict, unit for
#: unit. It is the annotation-gap sensitivity check, promoted to
#: the default because the thing it corrects is an annotation gap, not a
#: property of any system under test.
#:
#: Not "session": that widens EVERY evidence session to all ~12 of its turns,
#: including the 854 where the dataset did localise the evidence, which measures
#: perfect SESSION retrieval instead of perfect evidence retrieval.
#:
#: This is a DEFAULT, not a redefinition. "strict" stays available as
#: ``--gold strict``, stays what ``test.tsv`` holds, and stays what every
#: existing metrics.json was computed under. Because gold_suffix("turn") is
#: non-empty, evaluating at this default writes to its own ``-gold-turn`` path
#: and cannot overwrite anything scored at strict.
EVAL_GOLD_LEVEL = "turn"

#: Optional levels built from an external LLM-graded label set (not used in the
#: paper), as opposed to the three above, which the benchmark itself defines.
#:
#:   relabel_union  the "turn" level UNION every turn the judge graded >= 1.
#:                  The headline level: it keeps every turn the benchmark
#:                  labelled and adds what the benchmark missed — needed hops,
#:                  restatements of the answer, exclusion turns. A "turn" turn
#:                  the judge rejected stays in at grade 1: evidence on the
#:                  benchmark's word, partial credit under nDCG.
#:
#:                  The base is "turn", not "strict", so the level is a SUPERSET
#:                  of the evaluation default and never narrower than it. On a
#:                  strict base, the whole unflagged evidence sessions "turn"
#:                  takes were dropped wherever the judge picked nothing in them:
#:                  9 LongMemEval-S abstention questions came back with an EMPTY
#:                  oracle prompt (their evidence sessions are the deliberate
#:                  distractors, which the judge rightly grades 0), and all_turns'
#:                  oracle averaged 3.8 units against gold-turn's 4.0.
#:   relabel_union_insession
#:                  relabel_union with the JUDGED turns limited to the question's
#:                  evidence sessions (``evidence_session_ids`` plus any session
#:                  holding a flagged turn — the same sessions "turn" widens).
#:                  The judge sees one turn at a time and cannot tell which
#:                  session is evidence, so on LongMemEval-S it grades ~15% of
#:                  questions' turns in OTHER sessions >= 1: style instructions
#:                  from unrelated chats on 19 of 30 preference questions, and
#:                  near-miss events on temporal ones. On the oracle those turns
#:                  cost every verbatim store (net -6 to -15 questions) — the
#:                  sensitivity check for that, not a replacement: an
#:                  off-session turn can also be a hop the benchmark missed.
#:   relabel        every turn the judge graded >= 1, and nothing else. Single-
#:                  source provenance; the sensitivity check. It is NOT an upper
#:                  bound on its own: on the calibration slices the judge
#:                  rejected 16% (LongMemEval-S) and 23% (BEAM-100K) of published
#:                  gold turns, and reading them showed mostly judge misses
#:                  rather than label errors.
#:
#: Kept OUT of GOLD_LEVELS on purpose. The qrels builders compute every level in
#: GOLD_LEVELS for every dataset, and a label set exists only for the datasets
#: that were relabelled — building LoCoMo's qrels must not fail for lack of one.
#: These are written only when a builder is given ``--relabel-set``.
#:
#: They are also the only GRADED levels: the qrels score is the judge's grade
#: (2 = Relevant, 1 = Partial Relevant) rather than 1. Recall and precision count
#: score >= 1, so they are unchanged by the grades; nDCG is not.
RELABEL_LEVELS = ("relabel_union", "relabel_union_insession", "relabel")

#: Every level a qrels file can exist for — what ``--gold`` accepts.
ALL_GOLD_LEVELS = GOLD_LEVELS + RELABEL_LEVELS


def turn_id(session_id: str, index: int) -> str:
    """The canonical id of one turn: ``<session_id>:t<index>``.

    Indexes into the ORIGINAL ``session.turns``, never a filtered view, so the
    id is the same string whether it is emitted as a corpus doc (--roles user
    drops the assistant turns), carried in a memory unit's provenance, or used
    as a gold label. That single namespace is what lets a memory unit built by
    lightmem be scored against the same evidence set as a raw turn.
    """
    return f"{session_id}:t{index}"


def gold_filename(level: str) -> str:
    """The qrels file a level is written to, inside a sample's ``qrels/``.

    EVERY level is named explicitly — there is no privileged level holding a
    bare ``test.tsv``. That special case existed only because strict was written
    before levels did, and it cost exactly what an implicit name costs: a file
    whose meaning depended on a constant defined elsewhere, so flipping that
    constant silently reinterpreted 4,353 files already on disk.

    ``level`` is required for the same reason. A default here would make one
    level the one you get by not thinking, which is the privilege this removed.
    """
    if level not in ALL_GOLD_LEVELS:
        raise ValueError(f"gold level must be one of {ALL_GOLD_LEVELS}, got {level!r}")
    return f"test_oracle_{level}.tsv"


def gold_suffix(level: str) -> str:
    """Path suffix keying an output by gold level — never empty.

    An oracle answered at two levels is two different conditions over one qrels
    dir, so its predictions must not share a path; same for a metrics file. When
    one level's suffix was empty, that level's outputs were the ones a NEW
    default would land on top of — the collision this exists to prevent, aimed
    squarely at whichever results were oldest.
    """
    if level not in ALL_GOLD_LEVELS:
        raise ValueError(f"gold level must be one of {ALL_GOLD_LEVELS}, got {level!r}")
    return f"-gold-{level}"


def query_evidence_turn_ids(sample: Sample, query: Query) -> set[str] | None:
    """The canonical turn ids ``query`` names ITSELF, or None if it names none.

    The multi-query escape hatch. ``Turn.has_answer`` is a flag on the SAMPLE,
    so it can only express "this turn is evidence for the one question here" —
    true for LongMemEval, where a sample IS a question, and false for every
    benchmark that hangs many questions off one conversation. On BEAM a
    conversation carries 20 questions with disjoint evidence: the union of their
    evidence is a median 45 of 271 turns while a single question's is a median
    2, so a flag-derived gold set is 22x too wide and recall@10 against it is
    very nearly free. ``Query.evidence_turn_ids`` is the per-question answer,
    and this is what reads it.

    Ids are resolved through ``Turn.turn_id`` — the dataset's OWN name for a
    turn (LoCoMo's ``dia_id``, BEAM's ``source_chat_ids`` message id) — into the
    ``<session_id>:t<i>`` namespace everything downstream compares on. An id
    that resolves to no turn is DROPPED rather than raising: BEAM's evidence
    dangles on 0.6% of ids at 128K and ~3% at 500K/1M, which is a property of the release, not a reason to lose the sample.

    Returns None, not an empty set, when this sample carries no per-question
    evidence at all — the two are different questions. None means "fall back to
    the sample's ``has_answer`` flags"; an empty set means "this question HAS no
    evidence", which is what BEAM's abstention questions assert on purpose and
    what a fully-dangling id list leaves behind.

    Which of the two applies is ``Sample.evidence_is_per_query``, an adapter
    declaration, and NOT whether this particular query happens to name any ids.
    Inferring it per query is the trap: BEAM's 40 abstention questions name
    nothing by design, so a per-query test hands each of them the fallback and
    they inherit the union of their conversation's other 19 gold sets — 44 turns
    where the correct answer is 0.
    """
    if not sample.evidence_is_per_query:
        return None
    by_native: dict[str, str] = {}
    for session in sample.sessions:
        for i, t in enumerate(session.turns):
            if t.turn_id:
                by_native[str(t.turn_id)] = turn_id(session.session_id, i)
    return {canonical
            for tid in query.evidence_turn_ids
            if (canonical := by_native.get(str(tid))) is not None}


def evidence_turn_ids(sample: Sample, query: Query, level: str) -> set[str]:
    """The turn ids that count as evidence for ``query`` at ``level``.

    Returns ids, not units: a builder intersects them with each unit's
    ``source_turn_ids``, which is the only thing both a raw turn and an
    extracted fact can be compared on. A unit whose provenance is empty can
    therefore never be gold at any level — see qrels_has_answer, which keeps
    the system's own stored flag for "strict" for exactly that reason.

    On a sample that localises evidence per question, the query's own turns
    replace the sample's ``has_answer`` flags at every level — see
    query_evidence_turn_ids. The level semantics are unchanged by the
    substitution: "strict" is the evidence turns, "session" widens them to their
    whole sessions, and "turn" sits between. Inert on LongMemEval, which does
    not set that flag.
    """
    if level in RELABEL_LEVELS:
        raise ValueError(f"{level!r} needs a relabel label set — use relabel_evidence()")
    if level not in GOLD_LEVELS:
        raise ValueError(f"gold level must be one of {GOLD_LEVELS}, got {level!r}")

    per_session: dict[str, list] = {}
    flagged: set[str] = set()
    for session in sample.sessions:
        per_session[session.session_id] = session.turns
        flagged |= {turn_id(session.session_id, i)
                    for i, t in enumerate(session.turns) if t.has_answer}

    # Replaces the flags rather than joining them. On a multi-query sample what
    # the flags say is the union over every question, so a union of the two
    # would widen each question's gold straight back out to it.
    named = query_evidence_turn_ids(sample, query)
    if named is not None:
        flagged = named

    if level == "strict":
        return flagged

    # See the module docstring: the union covers both a benchmark that names its
    # evidence sessions and one that only flags turns.
    evidence_sessions = set(query.evidence_session_ids)
    evidence_sessions |= {tid.split(":t")[0] for tid in flagged}

    def whole(session_id: str) -> set[str]:
        return {turn_id(session_id, i) for i in range(len(per_session.get(session_id) or []))}

    if level == "session":
        return set().union(*(whole(s) for s in evidence_sessions)) if evidence_sessions else set()

    # "turn": an evidence session that flags nothing contributes all of itself —
    # it is evidence the dataset asserts at session granularity only, and
    # dropping it is what leaves _abs prompts empty and multi-hop chains short.
    ids = set(flagged)
    for session_id in evidence_sessions:
        if not (whole(session_id) & flagged):
            ids |= whole(session_id)
    return ids


@dataclass
class RelabelLabels:
    """One relabel label set, loaded once: ``query_id -> {turn_id: grade}``.

    Read from ``<relabel root>/<dataset>/labels/<label_set>/labels.jsonl`` — the
    file holds a record for EVERY judged turn, grade 0 included. The turn ids are
    already in this module's ``<session_id>:t<i>`` namespace, so no translation
    happens here.

    ``sha256`` is of the file as read, so a qrels manifest can name the exact
    labels its files were built from.
    """
    dataset: str
    label_set: str
    path: str
    sha256: str
    grades: dict[str, dict[str, int]] = field(repr=False)

    @classmethod
    def load(cls, dataset: str, label_set: str) -> "RelabelLabels":
        root = os.environ.get("EVALGRID_RELABEL_ROOT") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs", "_relabel")
        path = os.path.join(root, dataset, "labels", label_set, "labels.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No relabel label set at {path}.")
        grades: dict[str, dict[str, int]] = defaultdict(dict)
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for raw in fh:
                digest.update(raw)
                if not raw.strip():
                    continue
                record = json.loads(raw)
                grades[record["query_id"]][record["turn_id"]] = int(record["grade"])
        return cls(dataset, label_set, path, digest.hexdigest(), dict(grades))

    def grades_for(self, query_id: str) -> dict[str, int]:
        """Every judged turn of ``query_id``, grade 0 included.

        A question absent from the file RAISES. It was never judged, which is a
        different thing from a question the judge found no evidence for (those
        are present, at grade 0) — and returning an empty set would write an
        empty gold file that reads exactly like the second.
        """
        if query_id not in self.grades:
            raise KeyError(
                f"question {query_id!r} is not in relabel set {self.label_set!r} "
                f"({self.path}): it was never judged. Judge it before building qrels "
                f"from this set, or build from a set that covers it.")
        return self.grades[query_id]

    def provenance(self) -> dict:
        """What a qrels manifest records about the labels it was built from."""
        return {"relabel_dataset": self.dataset, "relabel_label_set": self.label_set,
                "relabel_labels_file": self.path, "relabel_labels_sha256": self.sha256,
                "relabel_questions": len(self.grades), "relabel_levels": list(RELABEL_LEVELS)}


def relabel_evidence(sample: Sample, query: Query, level: str,
                     labels: RelabelLabels) -> dict[str, int]:
    """``turn_id -> grade`` for ``query`` at a relabel level (see RELABEL_LEVELS).

    Graded, unlike evidence_turn_ids: a builder gives each unit the highest
    grade among its source turns (grade_units).
    """
    if level not in RELABEL_LEVELS:
        raise ValueError(f"relabel level must be one of {RELABEL_LEVELS}, got {level!r}")
    judged = {tid: grade for tid, grade in labels.grades_for(query.query_id).items()
              if grade >= 1}
    if level == "relabel":
        return judged
    if level == "relabel_union_insession":
        # "session" is every turn of every evidence session, so its session ids
        # are exactly the evidence sessions "turn" is built over.
        sessions = {tid.rsplit(":t", 1)[0]
                    for tid in evidence_turn_ids(sample, query, "session")}
        judged = {tid: grade for tid, grade in judged.items()
                  if tid.rsplit(":t", 1)[0] in sessions}
    union = dict(judged)
    for tid in evidence_turn_ids(sample, query, "turn"):
        union.setdefault(tid, 1)   # benchmark evidence, judged Non Relevant: kept at grade 1
    return union


def grade_units(units, evidence: dict[str, int]) -> dict[str, int]:
    """``unit_id -> grade`` for every unit that meets ``evidence``, in unit order.

    A unit's grade is the highest grade among its ``source_turn_ids`` — a fact
    extracted from a Relevant turn and a Partial one is as good as the Relevant
    turn. The same rule as the binary levels (a unit is relevant iff its
    provenance meets the evidence), with the grade carried through.
    """
    graded = {}
    for unit in units:
        grade = max((evidence[t] for t in unit.source_turn_ids if t in evidence), default=0)
        if grade:
            graded[unit.unit_id] = grade
    return graded
