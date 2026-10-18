"""
evalgrid/benchmarks/adapters/beam.py
------------------------------------
Adapter for BEAM (local parquet).

Files:
  data/BEAM/data/100K-00000-of-00001.parquet   <- the tier this repo stages
  data/BEAM/data/500K-00000-of-00001.parquet
  data/BEAM/data/1M-00000-of-00001.parquet

One parquet ROW is one CONVERSATION, and one conversation is one Sample
carrying TWENTY queries — 2 for each of ten memory abilities. That is the shape
every decision below follows from, and it is the shape LongMemEval does not
have. ``evalgrid.tools.beam_stats`` measures these properties; this file
only says what the harness has to do about them.

Four things this adapter owns, because nothing downstream can recover them:

1. PER-QUESTION EVIDENCE. ``source_chat_ids`` names individual messages, so each
   of the 20 questions gets its own gold set through ``Query.evidence_turn_ids``
   (evalgrid/benchmarks/gold.py). The sample-level ``Turn.has_answer`` flag is
   still set — to the UNION over the conversation's questions, which is the only
   thing a per-sample flag can honestly mean — but it must not be read as gold
   here: the union is a median 45 turns of 271 where one question's evidence is
   a median 2. Anything scoring BEAM off the flag is scoring a 22x-wide gold
   set. gold.py prefers the per-query ids wherever they are populated.

2. TURN IDS. A BEAM message ``id`` is a per-conversation counter — "0", "1", …
   contiguous across the whole chat, restarting at 0 in the NEXT conversation
   (388 of the 100K tier's 392 distinct ids are shared by two or more
   conversations). It is carried in ``Turn.turn_id`` and resolved to the
   canonical ``<session_id>:t<i>`` namespace by gold.py, per sample, so the
   collision across samples never matters and no arithmetic maps one to the
   other.

3. THE ``->->`` MARKER. 74.9% of user messages end with a trailing ``->-> i,j``
   — the message's own ``index`` field, leaked by the generation pipeline.
   Stripped here so no system indexes an artefact.

4. THE GOLD ANSWER, which has three different names and is often absent. See
   ``_gold_answer``.

Not handled here, on purpose: the 500K/1M tiers exceed the ingest unit (every
session at 1M is over a 32 K window). They load through this same adapter, but
treat them as a length-stress study, not a comparison row.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime
from typing import Any, Iterator, Optional

from evalgrid.benchmarks.gold import turn_id as _turn_id
from evalgrid.benchmarks.types import Session, Sample, Query, Turn

#: The trailing pipeline marker leaked into released message text, e.g.
#: "...meet my deadlines by then? ->-> 1,1". Same regex as
#: evalgrid/tools/beam_stats.py, which is what measured its 74.9% incidence.
#:
#: WIDER THAN `[\d,]+`, which is what the first
#: version of this file used. Three variants leaked past that and survived into
#: the corpus — measured by dumping the tier with evalgrid.tools.beam_to_jsonl
#: and grepping, 8 user turns of the 100K tier's 2,866:
#:     "->-> 2,N/A"      an index the generator could not fill
#:     "->-> 5,14)"      a stray bracket after it
#:     "->-> 2,22, 24"   a space after the comma
#: All three are still the message's own `index` field, all still at end of
#: string, so they are the same artefact and are stripped the same way. The
#: alternation is deliberately narrow — digits or the literal N/A, nothing else
#: — so this cannot eat a sentence that merely ends in a number.
_MARKER = re.compile(r"\s*->->\s*(?:\d+|N/A)(?:\s*,\s*(?:\d+|N/A))*\s*\)?\s*$")

#: BEAM's names for the SHORT gold answer, narrowest first. Which one appears
#: depends on the ability: `answer` for five of them, `ideal_answer` for
#: contradiction_resolution, `ideal_response` for abstention — and NONE of the
#: three for instruction_following, preference_following and summarization.
#: That is 120 of the 100K tier's 400 questions with no short gold answer,
#: so ``Query.answer`` is None there and a judge for this
#: benchmark MUST be able to score against the rubric instead.
#:
#: Those 120 are not goldless — besides the `rubric` every question carries,
#: they ship a BEHAVIORAL gold under an ability-specific name: `ideal_summary`
#: (summarization, 40) and `expected_compliance` (instruction_following and
#: preference_following, 80). Deliberately NOT listed here. BEAM's own judge
#: interpolates only <rubric_item> and <llm_response> (judges/beam.py), so no
#: score can read either field, and leaving them out is what keeps
#: ``Query.answer is None`` an honest marker of "graded behaviorally, not
#: against a string". Adding them would also have to move the same count in
#: tools/beam_stats.py::ANSWER_FIELDS and the six
#: experiments/*/beam_100k.yaml headers, which all quote the 120.
_ANSWER_FIELDS = ("answer", "ideal_answer", "ideal_response")

#: ``time_anchor`` is set on exactly one message per session, in %B-%d-%Y
#: ("March-15-2024") — verified on all 90 sessions of the 100K tier. Normalised
#: to ISO so it parses in the systems' own timestamp readers (lightmem's
#: _INPUT_FMTS accepts %Y-%m-%d; nothing accepts BEAM's spelling).
_ANCHOR_FMT = "%B-%d-%Y"


def _clean(text: str) -> str:
    """One message's text, with the leaked index marker removed."""
    return _MARKER.sub("", text or "")


def _null(value: Any) -> Optional[Any]:
    """BEAM writes the STRING "None" where a field is unset, not a null.

    Every parquet column is typed str, so an absent ``time_anchor`` arrives as
    the four characters "None" and is truthy. Filtering it at the one place it
    enters is cheaper than remembering it at each use.
    """
    return None if value in (None, "", "None") else value


def _timestamp(session: list[dict]) -> Optional[str]:
    """The session's date, as ISO ``YYYY-MM-DD``, or None if it carries none."""
    for message in session:
        anchor = _null(message.get("time_anchor"))
        if anchor:
            try:
                return datetime.strptime(anchor.strip(), _ANCHOR_FMT).strftime("%Y-%m-%d")
            except ValueError:
                # An unparseable anchor is worth keeping verbatim rather than
                # dropping: a system's own reader may still make sense of it,
                # and a silently absent date is indistinguishable from a session
                # BEAM never dated.
                return anchor
    return None


def _literal(value: Any) -> Any:
    """Parse a Python-repr STRING, twice where the value is a repr of a repr.

    ``probing_questions`` is a repr string rather than JSON, and some rows store
    it double-encoded, so one literal_eval returns a str instead of a dict.
    ``rubric`` is the same shape one level down. Anything that
    does not parse comes back untouched — a malformed rubric is a question worth
    keeping without one, not a conversation worth losing.
    """
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            break
    return value


def _expand(entry: Any) -> list[Any]:
    """One ``source_chat_ids`` entry, as the id(s) it names.

    An entry is normally a single id. In three event_ordering rows it is a
    NESTED LIST of ids — ``[136, 138]`` sitting where every other row has a bare
    ``136`` — so this flattens, recursively, and returns anything else as the
    one value it was.

    Not ``_literal``: nothing here is a repr string, and parsing one would be
    free to rewrite an id that was already correct.
    """
    if isinstance(entry, (list, tuple)):
        return [i for sub in entry for i in _expand(sub)]
    return [entry]


def _evidence_ids(item: dict) -> list[str]:
    """Flatten ``source_chat_ids`` into a list of message ids.

    Three shapes in the 100K tier, and an adapter that assumes one gets a
    different subset of the benchmark wrong in each case:
      * a flat list — event_ordering, information_extraction,
        instruction_following, multi_session_reasoning, preference_following,
        summarization;
      * a dict of named lists — contradiction_resolution
        (first_statement/second_statement), knowledge_update and
        temporal_reasoning. BEAM documents only the first of those three;
        measured, all three are dicts;
      * absent entirely — every abstention question, by design (a question with
        no answer in the corpus has no evidence in it either), plus a handful of
        rows where ``source_chat_ids`` is simply one of the optional keys.

    A FOURTH shape hides inside the first: three event_ordering rows put a
    NESTED LIST of ids where every other row puts a bare id — ``[136, 138]``
    rather than ``136, 138`` — one nesting per group of events. Unflattened,
    ``str([136, 138])`` is ``"[136, 138]"``, which names no message and was
    dropped by the ``i in canonical`` filter in ``_queries``. That cost 14 ids
    across ``100K_4_event_ordering_1`` (2), ``100K_13_event_ordering_1`` (5) and
    ``100K_18_event_ordering_1`` (7) — the last left with an EMPTY gold set, so
    it was un-scoreable in retrieval and answered from nothing under
    ``--retriever oracle``.
    ``_expand`` flattens them. That reads BEAM's own data; it does not change
    BEAM's protocol — the ids were always there, one level down.

    Ids stay STRINGS: they are matched against ``Turn.turn_id``, and BEAM writes
    them as strings in one place and ints in another.
    """
    source = item.get("source_chat_ids")
    if isinstance(source, dict):
        flat: list = []
        for value in source.values():
            flat += list(value) if isinstance(value, (list, tuple)) else [value]
    elif isinstance(source, (list, tuple)):
        flat = list(source)
    else:
        return []
    return [str(i) for entry in flat for i in _expand(entry) if i is not None]


def _gold_answer(item: dict) -> Optional[str]:
    """The question's gold answer string, or None where it ships none.

    None is a real state here, not a parse failure: a third of BEAM is graded
    against a ``rubric`` and never had an answer string to return. Returning ""
    instead would make those questions look answerable-and-empty to a judge,
    which scores every one of them wrong.
    """
    for field in _ANSWER_FIELDS:
        value = _null(item.get(field))
        if value is not None:
            return value
    return None


class BEAMAdapter:
    """One parquet row is one conversation; one conversation is one Sample."""

    def __init__(self, parquet_path: str, tier: str):
        import pandas as pd

        self.tier = tier
        self.data = pd.read_parquet(parquet_path)
        missing = [c for c in ("conversation_id", "chat", "probing_questions")
                   if c not in self.data.columns]
        if missing:
            raise ValueError(
                f"{parquet_path} is missing column(s) {missing} — the upstream BEAM "
                f"schema changed. Re-run evalgrid.tools.beam_stats before trusting any number from it.")

    def _sessions(self, chat) -> tuple[list[Session], dict[str, tuple[int, int]]]:
        """Build the Sample's sessions, and the message-id -> (session, turn) map.

        ``Turn.turn_id`` IS THE CANONICAL ``<session_id>:t<i>`` ID, not BEAM's
        message id, and that is load-bearing. Every memory system records a
        fact's provenance as ``turn.turn_id or f"{session_id}:t{i}"`` — see
        a_mem/longmemeval.py:85, mem0/longmemeval.py:92, lightmem's pairing
        loop — so whatever this field holds is what lands in the saved store and
        is later intersected with the gold set. LongMemEval leaves it None and
        gets the synthesised canonical id; if BEAM put its own "0"/"1"/"2" here,
        every system would record provenance in BEAM's namespace while gold.py
        emits canonical ids, the intersection would be empty, and
        qrels_has_answer would write an EMPTY gold set for all six systems
        without failing.

        BEAM's own message id is kept in ``Turn.extra`` — it is what
        ``source_chat_ids`` names and the only way back to the release — and the
        map returned here is what translates one to the other.
        """
        sessions: list[Session] = []
        located: dict[str, tuple[int, int]] = {}
        for index, messages in enumerate(chat):
            session_id = f"session_{index + 1}"
            turns = []
            for position, message in enumerate(messages):
                message_id = str(message["id"])
                located[message_id] = (index, position)
                turns.append(Turn(
                    role=message["role"],
                    content=_clean(message["content"]),
                    turn_id=_turn_id(session_id, position),
                    extra={"beam_message_id": message_id},
                ))
            sessions.append(Session(session_id=session_id, turns=turns,
                                    timestamp=_timestamp(list(messages))))
        return sessions, located

    def _queries(self, sample_id: str, probing, canonical: dict[str, str]) -> list[Query]:
        """The conversation's 20 questions — 2 per ability, ability-ordered.

        ``question_type`` is the ability name, which is what the judge aggregates
        per-category on and what per-ability tables are keyed by.
        There is no per-question date in BEAM, so ``question_date`` stays None
        and the answer prompts fall through to their undated branch.

        ``evidence_turn_ids`` is translated into canonical turn ids HERE rather
        than left in BEAM's message-id namespace, so the ids a query names and
        the ids a system records are the same strings. An id naming no message
        is dropped — 0.6% of the 100K tier's.
        """
        queries: list[Query] = []
        for ability, items in sorted(probing.items()):
            for position, item in enumerate(items):
                queries.append(Query(
                    query_id=f"{sample_id}_{ability}_{position}",
                    question=item.get("question", ""),
                    answer=_gold_answer(item),
                    question_type=ability,
                    evidence_turn_ids=[canonical[i] for i in _evidence_ids(item)
                                       if i in canonical],
                    extra={
                        # The rubric is the ONLY gold for instruction_following,
                        # preference_following and summarization, and is present
                        # beside the answer elsewhere — so a judge reads it here
                        # rather than deciding per ability whether to look.
                        "rubric": _literal(item.get("rubric")) or [],
                        "difficulty": _null(item.get("difficulty")),
                        "tier": self.tier,
                    },
                ))
        return queries

    def samples(self) -> Iterator[Sample]:
        seen: set[str] = set()
        for _, row in self.data.iterrows():
            sample_id = f"{self.tier}_{row['conversation_id']}"
            if sample_id in seen:
                raise ValueError(
                    f"duplicate conversation_id {row['conversation_id']!r} in tier "
                    f"{self.tier} — sample ids key one output directory each, so a "
                    f"collision would have two conversations overwrite each other's "
                    f"memory store.")
            seen.add(sample_id)

            sessions, located = self._sessions(row["chat"])
            probing = _literal(row["probing_questions"])
            if not isinstance(probing, dict):
                raise ValueError(
                    f"{sample_id}: probing_questions did not parse to a dict after two "
                    f"literal_eval passes (got {type(probing).__name__}).")
            canonical = {message_id: _turn_id(sessions[s].session_id, t)
                         for message_id, (s, t) in located.items()}
            queries = self._queries(sample_id, probing, canonical)

            # The sample-level flag, set to the UNION over the conversation's
            # questions. It is what the memory systems stamp onto each unit at
            # ingest, so leaving it False would make plugin.records_has_answer
            # report no provenance and qrels_has_answer skip every sample. It is
            # NOT gold: read the docstring at the top of this file, and see
            # gold.py, which prefers Query.evidence_turn_ids at every level.
            by_turn_id = {t.turn_id: t for s in sessions for t in s.turns}
            for query in queries:
                for tid in query.evidence_turn_ids:
                    by_turn_id[tid].has_answer = True

            yield Sample(sample_id=sample_id, source="beam",
                         sessions=sessions, queries=queries,
                         # BEAM localises evidence per QUESTION. Declaring it
                         # here is what stops the 40 abstention questions — which
                         # name no evidence because they have none — from falling
                         # back to the union flag set above. See types.Sample.
                         evidence_is_per_query=True)
