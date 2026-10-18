"""
evalgrid/systems/memorybank/store.py
------------------------------------
MemoryBank's on-disk memory, its unit rendering, and its forgetting curve —
rewritten (see the package docstring for why upstream cannot be imported), with
the strings kept byte-identical to what upstream indexes.

Pure stdlib on purpose: the plugin flattens a saved store into corpus units
through this module, and that has to work on a machine that cannot run the
system.

The schema is upstream's, one dict per user
(`forget_memory.py::initial_load_forget_and_save` is the authority — it is the
loader that assigns ids):

    {"<user>": {
        "name": "<user>",
        "history":     {"<bucket>": [ {query, response, memory_id, source,
                                       memory_strength, last_recall_date, ...}, ... ]},
        "summary":     {"<bucket>": {"content": ..., "memory_id": ..., ...}},
        "personality": {"<bucket>": "<text>"},
        "overall_history": "...", "overall_personality": "..."}}

Three deviations from upstream, each deliberate:

1. **Buckets are sessions, not dates.** Upstream's only grouping key is
   ``time.strftime("%Y-%m-%d")`` — *wall-clock today*, not any date carried by
   the data (`memory_utils.py::save_local_memory`). That is wrong twice over
   here: it would stamp every sample with the run date, and LongMemEval puts 50+
   sessions on far fewer distinct dates, so a "daily" summary would span
   unrelated conversations. Keying by ``session_id`` keeps one session = one
   summary and gives stable unit ids. ``bucket_by: date`` restores date buckets,
   using the session's own date rather than today's.

2. **Provenance rides on the turn-pair dict** (``session_id`` / ``turn_ids`` /
   ``has_answer`` / ``timestamp``). Upstream injects ``memory_id`` /
   ``memory_strength`` / ``last_recall_date`` into these same dicts lazily, so
   extra keys are the schema's own extension mechanism.

3. **``source`` is the bucket, not the memory id.** Upstream's two loaders
   disagree: ``JsonMemoryLoader`` sets ``metadata={"source": date}``
   (local_doc_qa.py:24) while ``initial_load_forget_and_save`` sets
   ``"source": memory_id`` (forget_memory.py:112). Everything downstream is
   written for the first: ``search_memory`` sorts by ``source`` and merges hits
   sharing one, and strips a ``时间{source}的对话内容：`` prefix — both no-ops
   when ``source`` is a unique per-unit id. We take the id scheme from
   forget_memory.py and the ``source`` semantics from local_doc_qa.py, which is
   the combination that makes the chronological regrouping actually do
   something.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Optional

# Upstream's EN speaker markers (forget_memory.py:95-96, local_doc_qa.py:39-40 —
# the two loaders agree on English and differ only in the Chinese AI marker).
USER_MARKER = "[|User|]:"
AI_MARKER = "[|AI|]:"


def render_turn_unit(bucket: str, query: str, response: str) -> str:
    """The indexed string for one turn-pair.

    Byte-identical to ``forget_memory.py:94-110``:
        f'Conversation content on {date}:' + f'{user_kw} {query.strip()}; '
                                           + f'{ai_kw} {response.strip()}'
    Note there is no space after the bucket's colon — that is upstream's string,
    not a typo here. Putting the date inside the indexed text (rather than only
    in metadata) is deliberate upstream: it gives a lexical retriever something
    to match dates on, and a dense one temporal signal for free.
    """
    return (f"Conversation content on {bucket}:"
            f"{USER_MARKER} {query.strip()}; "
            f"{AI_MARKER} {response.strip()}")


def render_summary_unit(bucket: str, summary: str) -> str:
    """The indexed string for one bucket summary (forget_memory.py:137)."""
    return f"The summary of the conversation on {bucket} is: {summary.strip()}"


def pair_turns(turns: list, query_role: Optional[str] = None) -> list[tuple]:
    """Fold a session's turns into MemoryBank's ``{query, response}`` pairs.

    Upstream assumes strict alternation because its own data is generated that
    way; benchmark sessions are mostly alternating but nothing guarantees it, so
    an unmatched turn becomes a pair with an empty other side rather than being
    dropped — the user turn is the one that carries the evidence.

    query_role  the role that opens a pair ("user" on LongMemEval). None pairs
                by alternation instead, for benchmarks whose roles are speaker
                names (LoCoMo), where no fixed role means "the asker".

    Returns (query_turn|None, response_turn|None, [original turn indices]).
    """
    pairs: list[tuple] = []
    i = 0
    while i < len(turns):
        turn = turns[i]
        following = turns[i + 1] if i + 1 < len(turns) else None
        opens_pair = query_role is None or turn.role == query_role
        if opens_pair and following is not None and following.role != turn.role:
            pairs.append((turn, following, [i, i + 1]))
            i += 2
        elif opens_pair:
            pairs.append((turn, None, [i]))
            i += 1
        else:
            # A response-role turn with no query before it (a session opening
            # with the assistant).
            pairs.append((None, turn, [i]))
            i += 1
    return pairs


def forgetting_curve(days: float, strength: float, *, variant: str = "paper") -> float:
    """Retention probability after ``days``, for a memory of given ``strength``.

    Two variants, because upstream's code does not implement its own paper:

    paper  R = e^(-t/S)     — arXiv:2305.10250 §Method. Higher strength means
                              slower forgetting, which is the Ebbinghaus model
                              the paper describes.
    code   R = e^(-t*S/5)   — what ``forget_memory.py:36`` actually computes.
                              It is written ``math.exp(-t / 5*S)``, and Python
                              parses that as ``exp((-t/5)*S)``, so the exponent
                              is inverted: higher strength means *faster*
                              forgetting, and recalling a memory (which
                              increments strength) makes it more likely to be
                              deleted. Kept selectable so the shipped behaviour
                              can be reproduced on purpose.

    Either way, note the scale mismatch: with S=1 and t=60 days the paper's
    curve retains 8e-27. The mechanism is calibrated for a 10-day companion
    setting and does not transfer to a months-long haystack without retuning S.
    """
    if variant == "paper":
        return math.exp(-days / max(strength, 1e-9))
    if variant == "code":
        return math.exp(-days * strength / 5)
    raise ValueError(f"forgetting curve variant must be 'paper' or 'code', got {variant!r}")


class MemoryBankStore:
    """One sample's memory, in upstream's per-user dict shape."""

    def __init__(self, user: str = "user"):
        self.user = user
        self.data: dict = {
            user: {"name": user, "history": {}, "summary": {}, "personality": {},
                   "overall_history": "", "overall_personality": ""}
        }

    # -- accessors -------------------------------------------------------

    @property
    def memory(self) -> dict:
        return self.data[self.user]

    @property
    def history(self) -> dict:
        return self.memory["history"]

    @property
    def summaries(self) -> dict:
        return self.memory["summary"]

    def buckets(self) -> list[str]:
        """Bucket keys in insertion order — which is ingest order, i.e. time."""
        return list(self.history)

    def turn_count(self) -> int:
        return sum(len(pairs) for pairs in self.history.values())

    # -- write path (no LLM, as upstream) --------------------------------

    def add_pair(self, bucket: str, query: str, response: str, *,
                 session_id: str, turn_ids: list[str], has_answer: bool,
                 timestamp: Optional[str]) -> dict:
        """Append one turn-pair. This is the whole of upstream's write path.

        ``memory_id`` follows ``forget_memory.py:108`` — ``{user}_{bucket}_{i}``
        — with ``i`` the index within the bucket.
        """
        pairs = self.history.setdefault(bucket, [])
        index = len(pairs)
        record = {
            "query": query,
            "response": response,
            "memory_id": f"{self.user}_{bucket}_{index}",
            "source": bucket,
            "memory_strength": 1,
            "last_recall_date": timestamp,
            # ours — see the module docstring
            "session_id": session_id,
            "turn_ids": turn_ids,
            "has_answer": bool(has_answer),
            "timestamp": timestamp,
        }
        pairs.append(record)
        return record

    def set_summary(self, bucket: str, content: str) -> None:
        pairs = self.history.get(bucket) or []
        self.summaries[bucket] = {
            "content": content,
            "memory_id": f"{self.user}_{bucket}_summary",
            "source": bucket,
            "memory_strength": 1,
            "last_recall_date": pairs[0]["timestamp"] if pairs else None,
            "session_ids": sorted({p["session_id"] for p in pairs}),
            "turn_ids": [tid for p in pairs for tid in p["turn_ids"]],
            "has_answer": any(p["has_answer"] for p in pairs),
            "timestamp": pairs[0]["timestamp"] if pairs else None,
        }

    # -- forgetting ------------------------------------------------------

    def apply_forgetting(self, now: str, *, variant: str = "paper", seed: int = 0,
                         day_difference) -> int:
        """Mark units as forgotten. Returns how many were marked.

        Two deliberate departures from ``initial_load_forget_and_save``:

        * **Non-destructive.** Upstream pops the turn out of the dict and
          rewrites the source JSON, so the evidence is gone from disk and the
          run is unreproducible. Here a unit is flagged ``forgotten: true`` and
          skipped by ``iter_units``; the store still holds it, so the same
          memory can be re-flattened with forgetting off and the two corpora
          diffed.
        * **Seeded.** Upstream calls a bare ``random.random()``. On a benchmark
          where the answer lives in exactly one session, an unseeded coin flip
          makes the score irreproducible.

        A bucket whose every turn is forgotten also loses its summary, as
        upstream drops the date's summary when the date empties
        (forget_memory.py:129-131).
        """
        rng = random.Random(seed)
        forgotten = 0
        for bucket, pairs in self.history.items():
            for record in pairs:
                days = day_difference(record.get("last_recall_date"), now)
                retention = forgetting_curve(days, record.get("memory_strength", 1),
                                             variant=variant)
                record["forgotten"] = rng.random() > retention
                forgotten += bool(record["forgotten"])
            if pairs and all(p.get("forgotten") for p in pairs):
                if bucket in self.summaries:
                    self.summaries[bucket]["forgotten"] = True
        return forgotten

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return self.data

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryBankStore":
        store = cls(user=next(iter(data)))
        store.data = data
        return store


def iter_store_units(data: dict) -> Iterable[tuple]:
    """Flatten a saved store into (unit_id, text, granularity, record) tuples.

    Emission order is upstream's: for each bucket in ingest order, its
    turn-pairs, then its summary — so both granularities land in one flat
    stream, competing for the same top-k, which is MemoryBank's actual design.
    Forgotten units are skipped.

    Shared by the plugin (corpus building) and the system (native index), so the
    two can never disagree about what a unit is.
    """
    for user_memory in data.values():
        history = user_memory.get("history") or {}
        summaries = user_memory.get("summary") or {}
        for bucket, pairs in history.items():
            for record in pairs:
                if record.get("forgotten"):
                    continue
                yield (record["memory_id"],
                       render_turn_unit(bucket, record.get("query") or "",
                                        record.get("response") or ""),
                       "turn", record)
            summary = summaries.get(bucket)
            if summary and not summary.get("forgotten") and summary.get("content"):
                yield (summary["memory_id"],
                       render_summary_unit(bucket, summary["content"]),
                       "summary", summary)
