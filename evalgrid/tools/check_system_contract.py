#!/usr/bin/env python3
"""
check_system_contract.py
------------------------
Verifies the SystemPlugin contract against synthetic fixtures — no GPU, no
models, no API, no outputs/ tree. Runs in about a second.

It pins the bytes the pipeline emits. The reference values inlined below are
the pre-plugin implementations, copied verbatim from the code that produced the
paper's results, so a regression in a plugin shows up as a byte diff rather
than as a silently different number three stages downstream:

  * iter_units            -> corpus.jsonl / qrels gold-file bytes
  * answer_prompt         -> the exact message list sent to the generator
  * qrels_turns           -> the raw-turn corpus, built through the same type
  * records_has_answer    -> distinguishes "no evidence" from "legacy store"
  * config_path           -> variant resolves to a file that exists

Run it after touching a plugin, benchmarks/qrels.py, or the answer prompt:

    python -m evalgrid.tools.check_system_contract
"""
import json
import os
import sys
import tempfile

import evalgrid  # bootstrap (.env, sys.path)
from evalgrid.lib.paths import ROOT
from evalgrid.benchmarks.qrels import write_beir_sample
from evalgrid.benchmarks.types import MemorySystem, Query, Session, Turn, Sample
from evalgrid.systems.base import shared_answer_prompt
from evalgrid.systems.registry import get_plugin, available_systems
from evalgrid.pipeline.answer import resolve_prompt
from evalgrid.systems.lightmem.plugin import _ADAPTER_KEYS as _LIGHTMEM_ADAPTER_KEYS
from evalgrid.benchmarks.gold import (GOLD_LEVELS, STAMPED_GOLD_LEVEL,
                                      gold_filename)
from evalgrid.pipeline.qrels_turns import _build_sample, _gold

failures = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}\n    got : {got!r}\n    want: {want!r}")


def main() -> int:
    # --- fixture: a LightMem points.jsonl exactly as save_memory writes it -------
    # NOTE the provenance key is "turn_id", a SCALAR — that is what
    # vendor/lightmem/memory/lightmem.py:439 writes. This fixture used to say
    # "source_ids": [...], a key nothing in the tree has ever written, so the
    # provenance check below passed against a fiction while every real lightmem
    # unit carried source_turn_ids == [].
    POINTS = [
        {"id": "p1", "payload": {"memory": "User adopted a cat named Mochi.",
                                 "time_stamp": "2023-05-20T01:33:00.000",
                                 "has_answer": True, "turn_id": "s1:t0",
                                 "float_time_stamp": 1.0}},
        {"id": "p2", "payload": {"memory": "User dislikes cilantro.",
                                 "time_stamp": "2023-06-01T09:00:00.000",
                                 "has_answer": False, "float_time_stamp": 2.0}},
    ]

    tmp = tempfile.mkdtemp()
    sample_mem = os.path.join(tmp, "sample")
    os.makedirs(sample_mem)
    with open(os.path.join(sample_mem, "points.jsonl"), "w") as f:
        for p in POINTS:
            f.write(json.dumps(p) + "\n")

    plugin = get_plugin("lightmem")

    # --- 1. iter_units reproduces the old (id, memory, time_stamp, has_answer) ---
    units = list(plugin.iter_units(sample_mem))
    check("iter_units ids", [u.unit_id for u in units], [p["id"] for p in POINTS])
    check("iter_units text", [u.text for u in units], [p["payload"]["memory"] for p in POINTS])
    check("iter_units timestamps", [u.timestamp for u in units],
          [p["payload"]["time_stamp"] for p in POINTS])
    check("iter_units has_answer", [u.has_answer for u in units], [True, False])
    # LightMem prices a fact to ONE source turn (utils.py:355 reads
    # turn_id_list[source_id * 2] — the user side of the pair), so a scalar
    # becomes a one-element list, and a point without one stays empty rather
    # than fabricating a link.
    check("iter_units provenance reads the key lightmem writes",
          units[0].source_turn_ids, ["s1:t0"])
    check("a point with no turn_id claims no provenance",
          units[1].source_turn_ids, [])
    # Regression guard: "source_ids" is the key this used to read, and nothing
    # writes it. If it ever creeps back, provenance silently empties again.
    stray = os.path.join(tmp, "stray")
    os.makedirs(stray)
    with open(os.path.join(stray, "points.jsonl"), "w") as f:
        f.write(json.dumps({"id": "p1", "payload": {
            "memory": "m", "source_ids": ["s9:t9"]}}) + "\n")
    check("the never-written 'source_ids' key is not honoured",
          list(plugin.iter_units(stray))[0].source_turn_ids, [])

    # old: relevant_ids = [p["id"] for p in points if p["payload"].get("has_answer")]
    check("relevant ids", [u.unit_id for u in units if u.has_answer],
          [p["id"] for p in POINTS if p["payload"].get("has_answer")])

    # --- 2. corpus.jsonl bytes are unchanged ------------------------------------
    q = Query(query_id="q1", question="What is the cat called?", answer="Mochi",
              question_date="2023/07/01 (Sat) 10:00", question_type="single-session-user")
    out_new = os.path.join(tmp, "out_new")
    write_beir_sample(out_new, units, [(q, ["p1"])])

    # what the OLD write_beir_sample emitted, inlined verbatim from git history
    old_corpus = "\n".join(json.dumps({
        "_id": p["id"], "title": "",
        "text": p["payload"]["memory"],
        "metadata": {"time_stamp": p["payload"].get("time_stamp")},
    }, ensure_ascii=False) for p in POINTS) + "\n"
    with open(os.path.join(out_new, "corpus.jsonl")) as f:
        check("corpus.jsonl bytes", f.read(), old_corpus)
    with open(os.path.join(out_new, "qrels", gold_filename(STAMPED_GOLD_LEVEL))) as f:
        check("qrels stamped-level bytes", f.read(), "query-id\tcorpus-id\tscore\nq1\tp1\t1\n")

    # --- 3. answer prompt is byte-identical to the old build_messages ------------
    facts = ["[2023/05/20 (Sat) 01:33] User adopted a cat named Mochi.",
             "[2023/06/01 (Thu) 09:00] User dislikes cilantro."]


    def old_build_messages(question, question_date, facts):
        header = f"Question time: {question_date}\nQuestion: {question}" if question_date \
            else f"Question: {question}"
        facts_text = "\n".join(facts) if facts else "(no relevant memories found)"
        user_content = (f"{header}\n\n"
                        f"Please answer the question based on the following memories:\n{facts_text}")
        return [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content}]


    # What --prompt shared resolves to, for every system including LightMem.
    # This is the one the paper's numbers were produced under, so it is pinned
    # against the pre-plugin bytes; LightMem's *native* prompt (which does differ
    # from this, deliberately) is checked in §12.
    for label, qd, fl in [("with date", q.question_date, facts),
                          ("no date", None, facts),
                          ("no facts", q.question_date, [])]:
        check(f"prompt via resolve_prompt shared ({label})",
              resolve_prompt("lightmem", "shared")(q.question, qd, fl),
              old_build_messages(q.question, qd, fl))
        check(f"prompt via shared ({label})",
              shared_answer_prompt(q.question, qd, fl), old_build_messages(q.question, qd, fl))

    # --- 4. qrels_turns corpus unchanged ----------------------------------------
    sample = Sample(sample_id="s", source="longmemeval", queries=[q], sessions=[
        Session(session_id="s1", timestamp="2023/06/23 (Fri) 06:57", turns=[
            Turn(role="user", content="I adopted a cat.", has_answer=True),
            Turn(role="assistant", content="Lovely!"),
            Turn(role="user", content="Her name is Mochi.", has_answer=False),
        ])])
    # roles="user" is the pre---roles behaviour, which is what the reference
    # bytes below were captured under; roles="all" is checked after it.
    # _build_sample returns the corpus; the gold set is _gold at a LEVEL, and
    # "strict" is the rule that predates levels — the has_answer turns, the
    # exchanges containing one. These checks pin that equivalence.
    turn_units = _build_sample(sample, "user")
    turn_rel = _gold(turn_units, sample, "strict")[0][1]
    check("qrels_turns ids", [u.unit_id for u in turn_units], ["s1:t0", "s1:t2"])
    check("qrels_turns text", [u.text for u in turn_units], ["I adopted a cat.", "Her name is Mochi."])
    check("qrels_turns timestamps", [u.timestamp for u in turn_units],
          ["2023/06/23 (Fri) 06:57"] * 2)
    check("qrels_turns relevant", turn_rel, ["s1:t0"])

    # roles="all" keeps every turn and moves the speaker into the TEXT (the
    # `all_turns` baseline). Turn ids index the ORIGINAL turn list under both,
    # so an id means the same turn in either corpus — which is what lets the
    # two baselines share qrels with a system's memory corpus.
    all_units = _build_sample(sample, "all")
    all_rel = _gold(all_units, sample, "strict")[0][1]
    check("qrels_turns roles=all ids", [u.unit_id for u in all_units],
          ["s1:t0", "s1:t1", "s1:t2"])
    check("qrels_turns roles=all text", [u.text for u in all_units],
          ["user: I adopted a cat.", "assistant: Lovely!", "user: Her name is Mochi."])
    check("qrels_turns roles=all relevant", all_rel, ["s1:t0"])

    # roles="pair" keeps the same turns as "all" but makes the (user, assistant)
    # EXCHANGE the document. The id is the opening turn's, so pair ids stay in
    # the turn-id namespace; the trailing unpaired user turn survives as a
    # one-sided doc (dropping it is F1, the bug this mode must not inherit).
    pair_units = _build_sample(sample, "pair")
    pair_rel = _gold(pair_units, sample, "strict")[0][1]
    check("qrels_turns roles=pair ids", [u.unit_id for u in pair_units],
          ["s1:t0", "s1:t2"])
    check("qrels_turns roles=pair text", [u.text for u in pair_units],
          ["user: I adopted a cat.\nassistant: Lovely!", "user: Her name is Mochi."])
    check("qrels_turns roles=pair provenance", [u.source_turn_ids for u in pair_units],
          [["s1:t0", "s1:t1"], ["s1:t2"]])
    check("qrels_turns roles=pair relevant", pair_rel, ["s1:t0"])
    check("qrels_turns roles=pair keeps every turn",
          sum(len(u.source_turn_ids) for u in pair_units), len(sample.sessions[0].turns))

    # A pair is relevant if EITHER side is: an assistant-only evidence turn makes
    # its exchange gold, which is exactly what an utterance-level corpus cannot
    # express. This is also why pair gold counts differ from all_turns'.
    asst_gold = Sample(sample_id="s2", source="longmemeval", queries=[q], sessions=[
        Session(session_id="s2", timestamp="2023/06/23 (Fri) 06:57", turns=[
            Turn(role="user", content="Which lens?", has_answer=False),
            Turn(role="assistant", content="The 35mm.", has_answer=True),
        ])])
    asst_rel = _gold(_build_sample(asst_gold, "pair"), asst_gold, "strict")[0][1]
    check("qrels_turns roles=pair inherits assistant-side evidence", asst_rel, ["s2:t0"])
    # Both sides flagged collapse to ONE gold doc where all_turns has two.
    both_gold = Sample(sample_id="s3", source="longmemeval", queries=[q], sessions=[
        Session(session_id="s3", timestamp="2023/06/23 (Fri) 06:57", turns=[
            Turn(role="user", content="Which lens?", has_answer=True),
            Turn(role="assistant", content="The 35mm.", has_answer=True),
        ])])
    check("qrels_turns roles=pair collapses both-sides gold",
          (len(_gold(_build_sample(both_gold, "pair"), both_gold, "strict")[0][1]),
           len(_gold(_build_sample(both_gold, "all"), both_gold, "strict")[0][1])),
          (1, 2))
    # A session opening with the assistant must not lose that turn either.
    leading_asst = Sample(sample_id="s4", source="longmemeval", queries=[q], sessions=[
        Session(session_id="s4", timestamp="2023/06/23 (Fri) 06:57", turns=[
            Turn(role="assistant", content="Welcome back!"),
            Turn(role="user", content="Hi.", has_answer=True),
        ])])
    lead_units = _build_sample(leading_asst, "pair")
    lead_rel = _gold(lead_units, leading_asst, "strict")[0][1]
    check("qrels_turns roles=pair keeps a leading assistant turn",
          [u.text for u in lead_units], ["assistant: Welcome back!", "user: Hi."])
    check("qrels_turns roles=pair leading-assistant relevant", lead_rel, ["s4:t1"])

    # --- 4a. gold LEVELS widen that set, and only that set ----------------------
    # One session, one flagged turn, so every level is hand-checkable:
    #   strict  the flagged turn                              s1:t0
    #   turn    + nothing (this session DOES flag something)   s1:t0
    #   session + every turn of the evidence session          s1:t0, s1:t1, s1:t2
    q_ev = Query(query_id="q1", question=q.question, answer=q.answer,
                 question_date=q.question_date, question_type=q.question_type,
                 evidence_session_ids=["s1"])
    ev_sample = Sample(sample_id="s", source="longmemeval", queries=[q_ev],
                       sessions=sample.sessions)
    for level, expected in [("strict", ["s1:t0"]), ("turn", ["s1:t0"]),
                            ("session", ["s1:t0", "s1:t1", "s1:t2"])]:
        check(f"gold level {level} over roles=all",
              _gold(all_units, ev_sample, level)[0][1], expected)

    # The level that exists for the 94 evidence sessions LongMemEval declares but
    # never flags: with nothing flagged, "turn" must take the whole session —
    # this is what stops an abstention question's oracle prompt being empty.
    unflagged = Sample(sample_id="s5", source="longmemeval", sessions=[
        Session(session_id="s5", timestamp="2023/06/23 (Fri) 06:57", turns=[
            Turn(role="user", content="Anything on file about my cat?"),
            Turn(role="assistant", content="Nothing recorded."),
        ])], queries=[Query(query_id="q1", question=q.question, answer=q.answer,
                            evidence_session_ids=["s5"])])
    unflagged_units = _build_sample(unflagged, "all")
    check("gold strict is EMPTY for an unflagged evidence session",
          _gold(unflagged_units, unflagged, "strict")[0][1], [])
    for level in ("turn", "session"):
        check(f"gold {level} takes the whole unflagged evidence session",
              _gold(unflagged_units, unflagged, level)[0][1], ["s5:t0", "s5:t1"])

    out_turns = os.path.join(tmp, "out_turns")
    write_beir_sample(out_turns, turn_units, [(q, turn_rel)])
    old_turn_corpus = "\n".join(json.dumps({
        "_id": tid, "title": "", "text": txt,
        "metadata": {"time_stamp": "2023/06/23 (Fri) 06:57"},
    }, ensure_ascii=False) for tid, txt in
        [("s1:t0", "I adopted a cat."), ("s1:t2", "Her name is Mochi.")]) + "\n"
    with open(os.path.join(out_turns, "corpus.jsonl")) as f:
        check("qrels_turns corpus.jsonl bytes", f.read(), old_turn_corpus)

    # extra_gold adds FILES, never rows to test.tsv: a reader that predates gold
    # levels must find the same bytes it always found.
    out_levels = os.path.join(tmp, "out_levels")
    write_beir_sample(out_levels, all_units, [(q_ev, ["s1:t0"])],
                      extra_gold={"session": [(q_ev, ["s1:t0", "s1:t1", "s1:t2"])]})
    with open(os.path.join(out_levels, "qrels", gold_filename(STAMPED_GOLD_LEVEL))) as f:
        check("extra_gold leaves the stamped level alone", f.read(),
              "query-id\tcorpus-id\tscore\nq1\ts1:t0\t1\n")
    with open(os.path.join(out_levels, "qrels", gold_filename("session"))) as f:
        check("extra_gold writes the level's own file", f.read(),
              "query-id\tcorpus-id\tscore\nq1\ts1:t0\t1\nq1\ts1:t1\t1\nq1\ts1:t2\t1\n")
    check("a level not passed writes no file",
          os.path.exists(os.path.join(out_levels, "qrels", gold_filename("turn"))), False)
    # Pinned deliberately: a "pair" level was built and removed — widening a
    # flagged turn to its exchange scores no better than adding a RANDOM turn of
    # the same role, because has_answer already flags the assistant turn where
    # the assistant turn is the evidence. See gold.py's module docstring before
    # re-adding it.
    check("gold levels", GOLD_LEVELS, ("strict", "turn", "session"))

    # --- 5. contract behaviours -------------------------------------------------
    check("records_has_answer true", plugin.records_has_answer(sample_mem), True)
    legacy = os.path.join(tmp, "legacy")
    os.makedirs(legacy)
    with open(os.path.join(legacy, "points.jsonl"), "w") as f:
        f.write(json.dumps({"id": "x", "payload": {"memory": "old", "time_stamp": None}}) + "\n")
    check("records_has_answer false (legacy store)", plugin.records_has_answer(legacy), False)
    check("iter_units on empty dir", list(plugin.iter_units(os.path.join(tmp, "nope"))), [])
    check("registry lists every system", available_systems(),
          ["a_mem", "hypermem", "lightmem", "mem0", "memorybank", "simplemem"])
    # Every registered system must resolve a default config and be reachable
    # through the registry — the two failures that otherwise surface only when
    # an ingest job has already started.
    check("every system has a default config",
          {n: os.path.isfile(get_plugin(n).config_path()) for n in available_systems()},
          {n: True for n in available_systems()})
    check("config_path default",
          os.path.relpath(plugin.config_path(), ROOT),
          "evalgrid/configs/lightmem/default.yaml")
    check("config_path variant",
          os.path.relpath(plugin.config_path("naive_query"), ROOT),
          "evalgrid/configs/lightmem/naive_query.yaml")
    check("config files exist",
          [os.path.isfile(plugin.config_path(v))
           for v in (None, "naive_query")], [True, True])

    for pseudo in ("all_turns", "user_turns", "pair_turns"):
        try:
            get_plugin(pseudo)
            check(f"{pseudo} rejected with a useful message", "no error", "KeyError")
        except KeyError as e:
            check(f"{pseudo} error mentions qrels_turns", "qrels_turns" in str(e), True)

    # --- 6. simplemem -----------------------------------------------------------
    # No reference bytes to pin (nothing preceded the plugin), so these check the
    # two things that would otherwise fail three stages downstream: that a saved
    # entry becomes the corpus doc the retriever is meant to rank, and that the
    # window loop puts the right turns behind each entry's provenance.
    ENTRIES = [
        {"entry_id": "e1", "lossless_restatement": "Alice adopted a cat named Mochi on 2023-05-20.",
         "keywords": ["Alice", "cat", "Mochi"], "event_timestamp": "2023-05-20T00:00:00",
         "location": None, "persons": ["Alice"], "entities": [], "topic": "Pet adoption",
         "window_timestamp": "2023/05/20 (Sat) 01:33", "session_ids": ["s1"],
         "source_turn_ids": ["s1:t0", "s1:t2"], "has_answer": True},
        {"entry_id": "e2", "lossless_restatement": "Alice dislikes cilantro.",
         "keywords": ["Alice", "cilantro"], "event_timestamp": None,
         "location": None, "persons": ["Alice"], "entities": [], "topic": "Food preference",
         "window_timestamp": "2023/06/01 (Thu) 09:00", "session_ids": ["s2"],
         "source_turn_ids": ["s2:t0"], "has_answer": False},
    ]
    sm_mem = os.path.join(tmp, "simplemem_sample")
    os.makedirs(sm_mem)
    with open(os.path.join(sm_mem, "entries.jsonl"), "w") as f:
        for e in ENTRIES:
            f.write(json.dumps(e) + "\n")

    sm = get_plugin("simplemem")
    sm_units = list(sm.iter_units(sm_mem))
    check("simplemem iter_units ids", [u.unit_id for u in sm_units], ["e1", "e2"])
    # text is lossless_restatement alone — the only field SimpleMem itself indexes
    check("simplemem iter_units text", [u.text for u in sm_units],
          [e["lossless_restatement"] for e in ENTRIES])
    # the corpus is stamped with conversation time, not the extractor's event time
    check("simplemem iter_units timestamps", [u.timestamp for u in sm_units],
          ["2023/05/20 (Sat) 01:33", "2023/06/01 (Thu) 09:00"])
    check("simplemem iter_units has_answer", [u.has_answer for u in sm_units], [True, False])
    check("simplemem iter_units provenance", sm_units[0].source_turn_ids, ["s1:t0", "s1:t2"])
    check("simplemem records_has_answer true", sm.records_has_answer(sm_mem), True)

    sm_legacy = os.path.join(tmp, "simplemem_legacy")
    os.makedirs(sm_legacy)
    with open(os.path.join(sm_legacy, "entries.jsonl"), "w") as f:
        f.write(json.dumps({"entry_id": "x", "lossless_restatement": "old"}) + "\n")
    check("simplemem records_has_answer false (legacy store)",
          sm.records_has_answer(sm_legacy), False)
    check("simplemem iter_units on empty dir",
          list(sm.iter_units(os.path.join(tmp, "nope"))), [])
    # One config, deliberately: the windowing variants were folded into
    # default.yaml so that "SimpleMem" names one system rather than three.
    check("simplemem config files exist",
          sorted(os.listdir(os.path.dirname(sm.config_path()))), ["default.yaml"])

    # Window loop + provenance. SimpleMemBase imports nothing from
    # vendor/simplemem at module scope, so the windowing can be exercised with
    # _extract_window stubbed — no lancedb, no embedder, no API.
    from evalgrid.systems.simplemem.system import SimpleMemBase

    class _WindowProbe(SimpleMemBase):
        """Records the windows the loop would have sent to the extractor."""

        def __init__(self, **cfg):
            self.windows = []
            self.ingest_policy = cfg.get("ingest_policy", "session")
            self.window_size = cfg.get("window_size", 4)
            self.overlap_size = cfg.get("overlap_size", 1)
            self._pending = []
            self._entries = ["stub"]          # non-empty so _finalize won't raise
            self.empty_windows = 0

        def _extract_window(self, window):
            if window:
                self.windows.append([d for _, d in window])

    def _turns(session_id, n):
        """n turns of one session, as (dialogue, provenance) pairs — the shape
        make_dialogue returns, with the Dialogue stubbed out as None."""
        return [(None, {"turn_id": f"{session_id}:t{i}", "session_id": session_id,
                        "timestamp": f"day-{session_id}", "has_answer": i == 1})
                for i in range(n)]

    # session policy: one window per session (sessions shorter than window_size),
    # and no window ever spans two sessions
    probe = _WindowProbe(ingest_policy="session", window_size=4, overlap_size=1)
    probe.ingest_dialogues(_turns("s1", 3))
    probe.ingest_dialogues(_turns("s2", 2), is_last=True)
    check("simplemem session policy: one window per session",
          [[p["turn_id"] for p in w] for w in probe.windows],
          [["s1:t0", "s1:t1", "s1:t2"], ["s2:t0", "s2:t1"]])

    # window policy: windows fill to window_size across session boundaries,
    # advance by step_size (= window_size - overlap_size), tail flushed at is_last
    probe = _WindowProbe(ingest_policy="window", window_size=4, overlap_size=1)
    probe.ingest_dialogues(_turns("s1", 3))
    probe.ingest_dialogues(_turns("s2", 3), is_last=True)
    check("simplemem window policy: overlapping windows across sessions",
          [[p["turn_id"] for p in w] for w in probe.windows],
          [["s1:t0", "s1:t1", "s1:t2", "s2:t0"],   # full window, spans the boundary
           ["s2:t0", "s2:t1", "s2:t2"]])           # tail, retaining 1 of overlap

    # provenance collapses a window: has_answer is the OR, timestamp is the first
    prov = SimpleMemBase._window_provenance(_turns("s1", 3) + _turns("s2", 1))
    check("simplemem window provenance turn ids", prov["source_turn_ids"],
          ["s1:t0", "s1:t1", "s1:t2", "s2:t0"])
    check("simplemem window provenance sessions", prov["session_ids"], ["s1", "s2"])
    check("simplemem window provenance has_answer (OR over window)",
          prov["has_answer"], True)
    check("simplemem window provenance timestamp (first turn)",
          prov["timestamp"], "day-s1")
    check("simplemem window provenance has_answer (none in window)",
          SimpleMemBase._window_provenance(_turns("s3", 1))["has_answer"], False)

    # --- 7. hypermem ------------------------------------------------------------
    # The store is upstream's own hypergraph_conv_N.json plus our provenance
    # sidecar, so these pin the flattening: which levels become units, the
    # chronological interleave, and the fact->episode->session provenance chain.
    HYPERGRAPH = {
        "facts": {
            "fact_aaa": {"content": "Alice adopted a cat named Mochi",
                         "episode_ids": ["s1"], "topic_id": "topic_1",
                         "temporal": "yesterday (May 19, 2023)", "spatial": "the shelter",
                         "keywords": ["cat"], "query_patterns": ["What pet?"],
                         "confidence": 0.9},
            "fact_bbb": {"content": "Alice dislikes cilantro",
                         "episode_ids": ["s2"], "topic_id": "topic_2",
                         "temporal": None, "spatial": None, "keywords": [],
                         "query_patterns": [], "confidence": 0.8},
        },
        "episodes": {
            "s1": {"subject": "Adopting Mochi", "summary": "Alice adopts a cat.",
                   "episode_description": "On May 20 2023 Alice adopted a cat named Mochi.",
                   "keywords": ["cat"], "original_data": []},
            "s2": {"subject": "Food dislikes", "summary": "Alice on cilantro.",
                   "episode_description": "On June 1 2023 Alice said she dislikes cilantro.",
                   "keywords": [], "original_data": []},
        },
        "topics": {"topic_1": {"title": "Pets", "summary": "Alice's pets",
                               "keywords": ["cat"], "episode_ids": ["s1"]}},
        "fact_hyperedges": {}, "episode_hyperedges": {},
    }
    PROVENANCE = {
        "episodes": {
            "s1": {"session_id": "s1", "source_turn_ids": ["s1:t0", "s1:t2"],
                   "has_answer": True, "timestamp": "2023/05/20 (Sat) 01:33"},
            "s2": {"session_id": "s2", "source_turn_ids": ["s2:t0"],
                   "has_answer": False, "timestamp": "2023/06/01 (Thu) 09:00"},
        },
        "facts": {
            "fact_aaa": {"session_ids": ["s1"], "source_turn_ids": ["s1:t0", "s1:t2"],
                         "has_answer": True, "timestamp": "2023/05/20 (Sat) 01:33"},
            "fact_bbb": {"session_ids": ["s2"], "source_turn_ids": ["s2:t0"],
                         "has_answer": False, "timestamp": "2023/06/01 (Thu) 09:00"},
        },
    }

    def write_hypermem_store(dirname, levels="facts+episodes", provenance=PROVENANCE):
        d = os.path.join(tmp, dirname)
        os.makedirs(d)
        with open(os.path.join(d, "hypergraph.json"), "w") as fh:
            json.dump(HYPERGRAPH, fh)
        if provenance is not None:
            with open(os.path.join(d, "provenance.json"), "w") as fh:
                json.dump(provenance, fh)
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump({"corpus_levels": levels}, fh)
        return d

    hm = get_plugin("hypermem")
    hm_mem = write_hypermem_store("hypermem_sample")
    hm_units = list(hm.iter_units(hm_mem))
    # each episode's facts, then the episode itself, in session order
    check("hypermem iter_units ids", [u.unit_id for u in hm_units],
          ["fact::fact_aaa", "ep::s1", "fact::fact_bbb", "ep::s2"])
    # fact text is FactNode.to_text(): "content (Time: ...; Location: ...)"
    check("hypermem fact text", hm_units[0].text,
          "Alice adopted a cat named Mochi (Time: yesterday (May 19, 2023); "
          "Location: the shelter)")
    check("hypermem fact text without temporal/spatial", hm_units[2].text,
          "Alice dislikes cilantro")
    check("hypermem episode text is episode_description", hm_units[1].text,
          "On May 20 2023 Alice adopted a cat named Mochi.")
    check("hypermem timestamps are conversation time",
          [u.timestamp for u in hm_units],
          ["2023/05/20 (Sat) 01:33"] * 2 + ["2023/06/01 (Thu) 09:00"] * 2)
    check("hypermem has_answer", [u.has_answer for u in hm_units],
          [True, True, False, False])
    check("hypermem provenance reaches turn ids", hm_units[0].source_turn_ids,
          ["s1:t0", "s1:t2"])
    check("hypermem unit levels", [u.metadata["level"] for u in hm_units],
          ["fact", "episode", "fact", "episode"])

    # corpus_levels comes from the STORE, not the live config
    facts_only = write_hypermem_store("hypermem_facts_only", levels="facts")
    check("hypermem corpus_levels=facts", [u.unit_id for u in hm.iter_units(facts_only)],
          ["fact::fact_aaa", "fact::fact_bbb"])
    with_topics = write_hypermem_store("hypermem_all", levels="facts+episodes+topics")
    check("hypermem corpus_levels adds topics",
          [u.unit_id for u in hm.iter_units(with_topics)][-1], "topic::topic_1")

    check("hypermem records_has_answer true", hm.records_has_answer(hm_mem), True)
    legacy_hm = write_hypermem_store("hypermem_legacy", provenance=None)
    check("hypermem records_has_answer false (no provenance sidecar)",
          hm.records_has_answer(legacy_hm), False)
    check("hypermem iter_units on empty dir",
          list(hm.iter_units(os.path.join(tmp, "nope"))), [])
    check("hypermem config file exists", os.path.isfile(hm.config_path()), True)

    # Session timestamps must parse, not silently become "now" — HyperMem
    # resolves relative time against them in two different prompts.
    from evalgrid.systems.hypermem.system import parse_session_timestamp
    check("hypermem parses LongMemEval timestamps",
          parse_session_timestamp("2023/05/20 (Sat) 01:33").isoformat(),
          "2023-05-20T01:33:00")
    check("hypermem parses LoCoMo timestamps",
          parse_session_timestamp("1:56 pm on 8 May, 2023").isoformat(),
          "2023-05-08T13:56:00")
    for bad, label in [(None, "missing"), ("not a date", "unparseable")]:
        try:
            parse_session_timestamp(bad)
            check(f"hypermem rejects {label} timestamp", "no error", "ValueError")
        except ValueError:
            check(f"hypermem rejects {label} timestamp", "ValueError", "ValueError")

    # --- 8. memorybank ----------------------------------------------------------
    # The unit strings are the load-bearing part here: MemoryBank is rewritten
    # rather than wrapped (its 2023 deps don't co-install), so these pin the
    # rendered text against upstream's loaders byte for byte.
    from evalgrid.systems.memorybank.store import (
        MemoryBankStore, forgetting_curve, pair_turns, render_summary_unit,
        render_turn_unit,
    )

    # forget_memory.py:94-110 / :137
    check("memorybank turn unit string",
          render_turn_unit("s1", "Hello, my name is Emily.", "Hello, Emily."),
          "Conversation content on s1:[|User|]: Hello, my name is Emily.; [|AI|]: Hello, Emily.")
    check("memorybank summary unit string",
          render_summary_unit("s1", "Emily introduced herself."),
          "The summary of the conversation on s1 is: Emily introduced herself.")

    # The two forgetting curves are genuinely different functions
    check("memorybank paper curve R(t=5,S=5)",
          round(forgetting_curve(5, 5, variant="paper"), 6), round(2.718281828 ** -1, 6))
    check("memorybank code curve is inverted (strength speeds forgetting)",
          forgetting_curve(5, 5, variant="code") < forgetting_curve(5, 1, variant="code"), True)

    class _T:
        def __init__(self, role, content, has_answer=False, turn_id=None):
            self.role, self.content = role, content
            self.has_answer, self.turn_id = has_answer, turn_id

    # user-opened pairing (LongMemEval) and alternation pairing (LoCoMo names)
    lme_turns = [_T("user", "q1"), _T("assistant", "a1"), _T("user", "q2")]
    check("memorybank pairs user->assistant",
          [(q.content if q else None, r.content if r else None, idx)
           for q, r, idx in pair_turns(lme_turns, "user")],
          [("q1", "a1", [0, 1]), ("q2", None, [2])])
    locomo_turns = [_T("Caroline", "hi"), _T("Melanie", "hey"), _T("Caroline", "bye")]
    check("memorybank pairs by alternation when roles are speaker names",
          [(q.content if q else None, r.content if r else None, idx)
           for q, r, idx in pair_turns(locomo_turns, None)],
          [("hi", "hey", [0, 1]), ("bye", None, [2])])
    check("memorybank keeps an assistant-opened turn",
          [(q.content if q else None, r.content if r else None)
           for q, r, _ in pair_turns([_T("assistant", "a0"), _T("user", "q0")], "user")],
          [(None, "a0"), ("q0", None)])

    # a store written the way the system writes one
    mb_store = MemoryBankStore()
    mb_store.add_pair("s1", "I adopted a cat.", "Lovely!", session_id="s1",
                      turn_ids=["s1:t0", "s1:t1"], has_answer=True,
                      timestamp="2023/05/20 (Sat) 01:33")
    mb_store.add_pair("s1", "Her name is Mochi.", "", session_id="s1",
                      turn_ids=["s1:t2"], has_answer=False,
                      timestamp="2023/05/20 (Sat) 01:33")
    mb_store.set_summary("s1", "Emily adopted a cat named Mochi.")
    mb_store.add_pair("s2", "I dislike cilantro.", "Noted.", session_id="s2",
                      turn_ids=["s2:t0", "s2:t1"], has_answer=False,
                      timestamp="2023/06/01 (Thu) 09:00")

    mb_mem = os.path.join(tmp, "memorybank_sample")
    os.makedirs(mb_mem)
    with open(os.path.join(mb_mem, "memory.json"), "w") as fh:
        json.dump(mb_store.to_dict(), fh)

    mb = get_plugin("memorybank")
    mb_units = list(mb.iter_units(mb_mem))
    check("memorybank iter_units ids", [u.unit_id for u in mb_units],
          ["user_s1_0", "user_s1_1", "user_s1_summary", "user_s2_0"])
    check("memorybank emits turn-pairs then that bucket's summary",
          [u.metadata["granularity"] for u in mb_units],
          ["turn", "turn", "summary", "turn"])
    check("memorybank unit text", mb_units[0].text,
          "Conversation content on s1:[|User|]: I adopted a cat.; [|AI|]: Lovely!")
    check("memorybank summary has_answer is the OR over its bucket",
          [u.has_answer for u in mb_units], [True, False, True, False])
    check("memorybank turn provenance", mb_units[0].source_turn_ids, ["s1:t0", "s1:t1"])
    check("memorybank summary provenance spans the bucket",
          mb_units[2].source_turn_ids, ["s1:t0", "s1:t1", "s1:t2"])
    check("memorybank timestamps", [u.timestamp for u in mb_units],
          ["2023/05/20 (Sat) 01:33"] * 3 + ["2023/06/01 (Thu) 09:00"])
    check("memorybank records_has_answer", mb.records_has_answer(mb_mem), True)

    # forgetting: non-destructive, seeded, and it removes units from the corpus
    forgotten = mb_store.apply_forgetting("2024/01/01 (Mon) 00:00", variant="paper",
                                          seed=0, day_difference=lambda a, b: 365.0)
    check("memorybank forgetting marks units at 365 days", forgotten, 3)
    mb_forget = os.path.join(tmp, "memorybank_forgotten")
    os.makedirs(mb_forget)
    with open(os.path.join(mb_forget, "memory.json"), "w") as fh:
        json.dump(mb_store.to_dict(), fh)
    check("memorybank forgotten units leave the corpus",
          [u.unit_id for u in mb.iter_units(mb_forget)], [])
    check("memorybank forgetting is non-destructive (store keeps the turns)",
          sum(len(v) for v in mb_store.history.values()), 3)

    mb_legacy = os.path.join(tmp, "memorybank_legacy")
    os.makedirs(mb_legacy)
    with open(os.path.join(mb_legacy, "memory.json"), "w") as fh:
        json.dump({"user": {"history": {"s1": [{"query": "q", "response": "r",
                                                "memory_id": "user_s1_0"}]}}}, fh)
    check("memorybank records_has_answer false (legacy store)",
          mb.records_has_answer(mb_legacy), False)
    check("memorybank iter_units on empty dir",
          list(mb.iter_units(os.path.join(tmp, "nope"))), [])
    check("memorybank config file exists", os.path.isfile(mb.config_path()), True)

    # --- 9. a_mem ---------------------------------------------------------------
    # The one thing worth pinning: corpus text is index_document() — paper Eq. 3,
    # concat(content, context, keywords, tags) — NOT the bare note content. If it
    # drifts to content-only, every A-Mem retrieval number silently changes while
    # nothing else in the pipeline errors.
    NOTES = [
        {"id": "n1", "content": "Speaker usersays : I adopted a cat named Mochi.",
         "context": "Pet adoption", "keywords": ["cat", "Mochi"], "tags": ["pets"],
         "timestamp": "2023/05/20 (Sat) 01:33", "links": ["n2"], "session_id": "s1",
         "source_turn_ids": ["s1:t0"], "has_answer": True},
        {"id": "n2", "content": "Speaker assistantsays : Lovely!",
         "context": "General", "keywords": [], "tags": [],
         "timestamp": "2023/05/20 (Sat) 01:33", "links": [], "session_id": "s1",
         "source_turn_ids": ["s1:t1"], "has_answer": False},
    ]
    am_mem = os.path.join(tmp, "a_mem_sample")
    os.makedirs(am_mem)
    with open(os.path.join(am_mem, "notes.jsonl"), "w") as fh:
        for n in NOTES:
            fh.write(json.dumps(n) + "\n")

    am = get_plugin("a_mem")
    am_units = list(am.iter_units(am_mem))
    check("a_mem iter_units ids", [u.unit_id for u in am_units], ["n1", "n2"])
    check("a_mem iter_units text is index_document, not bare content",
          am_units[0].text,
          "content:Speaker usersays : I adopted a cat named Mochi. context:Pet adoption"
          " keywords: cat, Mochi tags: pets")
    check("a_mem empty context defaults to 'General'",
          "context:General" in list(am.iter_units(am_mem))[1].text, True)
    check("a_mem iter_units timestamps", [u.timestamp for u in am_units],
          ["2023/05/20 (Sat) 01:33"] * 2)
    check("a_mem iter_units has_answer", [u.has_answer for u in am_units], [True, False])
    check("a_mem iter_units provenance", am_units[0].source_turn_ids, ["s1:t0"])
    # links are carried in metadata but NOT folded into text: the flat-corpus
    # protocol ranks independent units, and link expansion is the native path's.
    check("a_mem links stay out of the corpus text",
          ["n2" in u.text for u in am_units], [False, False])
    check("a_mem links survive in metadata", am_units[0].metadata["links"], ["n2"])
    check("a_mem records_has_answer true", am.records_has_answer(am_mem), True)

    am_legacy = os.path.join(tmp, "a_mem_legacy")
    os.makedirs(am_legacy)
    with open(os.path.join(am_legacy, "notes.jsonl"), "w") as fh:
        fh.write(json.dumps({"id": "x", "content": "old"}) + "\n")
    check("a_mem records_has_answer false (legacy store)",
          am.records_has_answer(am_legacy), False)
    check("a_mem iter_units on empty dir",
          list(am.iter_units(os.path.join(tmp, "nope"))), [])
    check("a_mem config file exists", os.path.isfile(am.config_path()), True)

    # --- 10. mem0 ---------------------------------------------------------------
    # Two things that would otherwise fail three stages downstream: the fact text
    # lives under payload["data"] (NOT "memory", which is the name it takes on
    # only after get_all()/search() reshape it), and the corpus stamp is the
    # benchmark's session_ts, not mem0's own created_at.
    M0_POINTS = [
        {"id": "u1", "payload": {"data": "Alice adopted a cat named Mochi.",
                                 "session_ts": "2023/05/20 (Sat) 01:33",
                                 "created_at": "2026-01-01T00:00:00",
                                 "session_id": "s1", "turn_ids": ["s1:t0"],
                                 "has_answer": True}},
        {"id": "u2", "payload": {"data": "Alice dislikes cilantro.",
                                 "session_ts": "2023/06/01 (Thu) 09:00",
                                 "created_at": "2026-01-01T00:01:00",
                                 "updated_at": "2026-01-01T00:02:00",
                                 "session_id": "s2", "turn_ids": ["s2:t0"],
                                 "has_answer": False}},
    ]
    m0_mem = os.path.join(tmp, "mem0_sample")
    os.makedirs(m0_mem)
    with open(os.path.join(m0_mem, "points.jsonl"), "w") as fh:
        for p in M0_POINTS:
            fh.write(json.dumps(p) + "\n")

    m0 = get_plugin("mem0")
    m0_units = list(m0.iter_units(m0_mem))
    check("mem0 iter_units ids", [u.unit_id for u in m0_units], ["u1", "u2"])
    check("mem0 iter_units text comes from payload['data']",
          [u.text for u in m0_units],
          [p["payload"]["data"] for p in M0_POINTS])
    check("mem0 iter_units timestamps are session_ts, not created_at",
          [u.timestamp for u in m0_units],
          ["2023/05/20 (Sat) 01:33", "2023/06/01 (Thu) 09:00"])
    check("mem0 iter_units has_answer", [u.has_answer for u in m0_units], [True, False])
    check("mem0 iter_units provenance", m0_units[0].source_turn_ids, ["s1:t0"])
    # 'updated' flags the facts Mem0's consolidation actually fired on — the
    # subset any UPDATE/DELETE analysis is about.
    check("mem0 flags consolidated facts", [u.metadata["updated"] for u in m0_units],
          [False, True])
    check("mem0 records_has_answer true", m0.records_has_answer(m0_mem), True)

    m0_legacy = os.path.join(tmp, "mem0_legacy")
    os.makedirs(m0_legacy)
    with open(os.path.join(m0_legacy, "points.jsonl"), "w") as fh:
        fh.write(json.dumps({"id": "x", "payload": {"data": "old"}}) + "\n")
    check("mem0 records_has_answer false (legacy store)",
          m0.records_has_answer(m0_legacy), False)
    check("mem0 iter_units on empty dir",
          list(m0.iter_units(os.path.join(tmp, "nope"))), [])
    check("mem0 config file exists", os.path.isfile(m0.config_path()), True)

    # --- 11. the native path ----------------------------------------------------
    # What turns a system's own retrieval loop into a run.tsv scorable against
    # the same qrels BM25 is scored against. The failure this catches is silent:
    # a trace whose ids don't reach run.tsv scores 0 recall and reads as a bad
    # retriever rather than as broken wiring.
    from evalgrid.pipeline.answer import ranked_from_trace

    check("native_unit_ids default reads the flat trace key",
          get_plugin("simplemem").native_unit_ids(
              {"query_id": "q1", "retrieved_unit_ids": ["e2", "e1"]}),
          ["e2", "e1"])
    check("native_unit_ids stringifies non-str ids",
          get_plugin("mem0").native_unit_ids({"retrieved_unit_ids": [1, 2]}), ["1", "2"])
    # HyperMem's trace is keyed by hypergraph level, not flat. Facts before
    # episodes; topics dropped (not in the corpus under either default).
    #
    # The ids are NAMESPACED, and that is the whole point of the check: the
    # trace carries upstream's raw ids while iter_units emits fact::/ep::, so
    # returning the raw form made every id in a native run.tsv unmatchable and
    # every hypermem native run scored 0 on every metric. The two vocabularies
    # are produced by one class and must not drift apart again.
    check("hypermem native_unit_ids flattens facts then episodes, dropping topics",
          hm.native_unit_ids({"retrieved_unit_ids": {
              "topics": ["t1"], "episodes": ["e1", "e2"], "facts": ["f1", "f2"]}}),
          ["fact::f1", "fact::f2", "ep::e1", "ep::e2"])
    # And they are the ids iter_units actually yields — asserted against the
    # store fixture rather than against a second hand-written list, so this
    # cannot pass while the corpus says something else.
    check("hypermem native ids are corpus ids",
          set(hm.native_unit_ids({"retrieved_unit_ids": {
              "episodes": ["s1"], "facts": ["fact_aaa"]}}))
          <= {u.unit_id for u in hm.iter_units(hm_mem)},
          True)
    check("hypermem native_unit_ids tolerates a flat trace",
          hm.native_unit_ids({"retrieved_unit_ids": ["f1"]}), ["f1"])

    # The ids have a matching TEXT vocabulary, and it must not drift either.
    # HyperMem renders a node in three places — upstream's formatter (the
    # unbudgeted answer prompt), iter_units (the flat corpus), and
    # HyperMemBase._unit_text (a --context-budget answer, and the token counts
    # the budget is measured against). An episode has no `content` key, so the
    # level-agnostic `content or text or summary` chain that used to build
    # retrieved_units fell through to `summary` — the one-line index field —
    # while the other two rendered the long `episode_description`. A budgeted
    # native run answered from different strings than an unbudgeted one over
    # the identical ids, and nothing in the output said so. Asserted against
    # iter_units rather than a hand-written expectation, so this cannot pass
    # while the corpus says something else.
    from evalgrid.systems.hypermem.system import HyperMemBase

    hm_by_id = {u.unit_id: u.text for u in hm.iter_units(hm_mem)}
    check("hypermem budgeted unit text matches the flat corpus",
          [HyperMemBase._unit_text("facts", HYPERGRAPH["facts"]["fact_aaa"]),
           HyperMemBase._unit_text("episodes", HYPERGRAPH["episodes"]["s1"])],
          [hm_by_id["fact::fact_aaa"], hm_by_id["ep::s1"]])
    check("hypermem episode text is the description, not the summary",
          HyperMemBase._unit_text("episodes", HYPERGRAPH["episodes"]["s1"]),
          HYPERGRAPH["episodes"]["s1"]["episode_description"])
    check("hypermem falls back to summary when extraction left no description",
          HyperMemBase._unit_text("episodes", {"summary": "only this"}), "only this")

    # A system whose answer() forgot to record ids must fail loudly here rather
    # than write an empty run.tsv that looks like a retrieval result.
    try:
        get_plugin("memorybank").native_unit_ids({"query_id": "q1"})
        check("missing retrieved_unit_ids raises", "no error", "NotImplementedError")
    except NotImplementedError as e:
        check("missing retrieved_unit_ids names the fix",
              "retrieved_unit_ids" in str(e), True)

    # run.tsv rows: de-duplicated (A-Mem's link expansion can surface a note
    # twice), first occurrence kept, scores strictly decreasing as 1/rank so
    # every rank-based metric eval_retrieval computes reads the intended order.
    check("ranked_from_trace de-dupes keeping best rank",
          ranked_from_trace(sm, {"retrieved_unit_ids": ["a", "b", "a", "c"]}),
          [("a", 1.0), ("b", 0.5), ("c", 1 / 3)])
    check("ranked_from_trace on an empty retrieval",
          ranked_from_trace(sm, {"retrieved_unit_ids": []}), [])

    # The ids a native run emits must be the ids iter_units yields, or they
    # match no corpus doc. Checked against the simplemem fixture from §6.
    check("native ids are corpus ids",
          [uid for uid, _ in ranked_from_trace(sm, {"retrieved_unit_ids": ["e1", "e2"]})],
          [u.unit_id for u in sm_units])

    # --- 12. answer-path segments -----------------------------------------------
    # The collision F11 was about: one system answered under both prompts must
    # land in two directories, and every pre-existing run must keep its path.
    from evalgrid.lib.paths import answers_dir, gen_segment, native_segment, retriever_name

    check("shared prompt keeps the historical path",
          os.path.relpath(answers_dir("lightmem", "longmemeval_s", "Qwen/Qwen3-30B",
                                      "bm25", "gpt-4o-mini", "top10"), ROOT),
          "outputs/_answers/lightmem/longmemeval_s/Qwen_Qwen3-30B/bm25/gpt-4o-mini/top10")
    check("oracle keeps its labeler level",
          os.path.relpath(answers_dir("lightmem", "longmemeval_s", "Qwen/Qwen3-30B",
                                      "oracle", "gpt-4o-mini", "all", labeler_seg="gpt-4o"), ROOT),
          "outputs/_answers/lightmem/longmemeval_s/Qwen_Qwen3-30B/oracle/gpt-4o/gpt-4o-mini/all")
    check("native prompt gets its own dir",
          answers_dir("lightmem", "longmemeval_s", "m", "bm25", "g", "top10", prompt="native")
          != answers_dir("lightmem", "longmemeval_s", "m", "bm25", "g", "top10"), True)
    check("native retrieval gets its own dir",
          os.path.basename(os.path.dirname(os.path.dirname(
              answers_dir("lightmem", "longmemeval_s", "m", "native", "g", "all",
                          prompt="native")))), "native")
    try:
        gen_segment("g", "bogus")
        check("gen_segment rejects an unknown prompt", "no error", "ValueError")
    except ValueError:
        check("gen_segment rejects an unknown prompt", True, True)

    # A native run's retriever is the system's own, configured in its YAML rather
    # than chosen with --retriever, so --native-variant is the only thing keeping
    # a sweep over those knobs from writing every setting to one directory (where
    # the answer stage's resumability would then skip all but the first).
    check("native variant suffixes the retriever segment",
          native_segment("bm25", "k30"), "native-bm25--k30")
    check("native variant is path-safe",
          native_segment("bm25", "k=30/top"), "native-bm25--k_30_top")
    check("two native settings get two dirs",
          len({os.path.relpath(answers_dir("a_mem", "longmemeval_s", "m",
                                           native_segment("bm25", v), "g", "all",
                                           prompt="native"), ROOT)
               for v in (None, "k30", "k50")}), 3)
    # A native run retrieves k units itself, so k changes its run.tsv and must
    # key the retrieval segment — unlike the flat path, where retrieve.py ranks
    # the whole corpus and only the answers dir needs a top<k> level.
    check("imposed top-k keys the native retrieval segment",
          [native_segment("bm25", None, k) for k in (None, 10, 30)],
          ["native-bm25", "native-bm25-top10", "native-bm25-top30"])
    check("encoder, top-k and variant compose",
          native_segment("emb-x", "nolinks", 10), "native-emb-x-top10--nolinks")
    # k lives in the retriever segment ONLY. <topk-seg> means answer-time
    # truncation, which a native run does none of (it answers from everything it
    # retrieved), so its answers dir writes "all" like oracle and full do —
    # carrying k in both places was redundant. And keeping it out of the
    # retrieval dir preserves the one-level-above-samples invariant that
    # tools/check_retrieval.py's "*/run.tsv" glob depends on.
    check("native answers dir carries k once, in the retriever segment",
          os.path.relpath(answers_dir("a_mem", "longmemeval_s", "m",
                                      native_segment("emb-x", None, 10), "g", "all",
                                      prompt="native"), ROOT),
          "outputs/_answers/a_mem/longmemeval_s/m/native-emb-x-top10/g--native-prompt/all")
    check("a native run dir stays one level above its samples",
          native_segment("emb-x", "v", 10).count("/"), 0)

    # The encoder must be DERIVED from each system's config, not passed by the
    # caller: an encoder swap in a YAML changes the ranking, and without this it
    # would land in the previous encoder's directory (the guarantee --emb-model
    # already gives a flat run).
    import yaml as _yaml
    native_names = {}
    for n in available_systems():
        with open(get_plugin(n).config_path()) as fh:
            native_names[n] = get_plugin(n).native_retriever_name(_yaml.safe_load(fh) or {})
    check("every system's native run names the retriever it ranks with",
          native_names,
          {"a_mem": "emb-sentence-transformers_all-MiniLM-L6-v2",
           "lightmem": "emb-sentence-transformers_all-MiniLM-L6-v2",
           "mem0": "emb-sentence-transformers_all-MiniLM-L6-v2",
           "memorybank": "emb-sentence-transformers_all-MiniLM-L6-v2",
           # SimpleMem is the odd one out — its native row confounds retriever
           # AND encoder against the other five unless deliberately aligned.
           "simplemem": "emb-Qwen_Qwen3-Embedding-0.6B",
           # HyperMem names its first stage, not an encoder, PLUS whether a
           # cross-encoder follows it. default.yaml is rrf (BM25 fused with the
           # propagated dense vectors) with `reranker.enabled: false`, the
           # repo-wide no-second-stage policy — so the
           # expected name carries the suffix. "bm25-norerank" here would mean
           # the config had regressed to keyword; a bare "rrf" would mean the
           # cross-encoder had been switched back on.
           "hypermem": "rrf-norerank"})
    # The name is in the flat side's vocabulary, so `native-emb-<m>` pairs with
    # `emb-<m>`: for lightmem/mem0/memorybank the two rankings should be equal.
    check("native encoder name matches the flat dense run's",
          native_names["lightmem"],
          retriever_name("dense", "sentence-transformers/all-MiniLM-L6-v2"))
    check("swapping a system's encoder moves its run dir",
          get_plugin("a_mem").native_retriever_name({"embedder": {"model": "naver/splade-v3"}})
          != native_names["a_mem"], True)
    check("hypermem names its retrieval_type, so rrf cannot clobber keyword runs",
          hm.native_retriever_name({"retrieval": {"retrieval_type": "keyword"}}), "bm25")
    # The cross-encoder is the second axis, and it has to be in the path for the
    # same reason the first one is: disabling it removes the rerank passes AND
    # narrows layers 1-2 from initial_candidates to their top_k (stage4
    # L552/L665), so the two settings are two rankings. Default = on = plain
    # name, per the repo's suffix convention.
    check("hypermem marks a rerank-off run, so it cannot clobber the default",
          [hm.native_retriever_name({"retrieval": {"retrieval_type": "rrf"},
                                     "reranker": {"enabled": on}}) for on in (True, False)],
          ["rrf", "rrf-norerank"])
    check("the rerank axis composes with a swapped layer scorer",
          hm.native_retriever_name({"native_retriever": {"name": "sparse",
                                                         "encoder": "naver/splade-v3"},
                                    "reranker": {"enabled": False}}),
          retriever_name("sparse", "naver/splade-v3") + "-norerank")
    try:
        get_plugin("a_mem").native_retriever_name({})
        check("a config with no encoder raises", "no error", "NotImplementedError")
    except NotImplementedError as e:
        check("a config with no encoder names the fix", "embedder.model" in str(e), True)

    # --- 14. retrieval / answer budget ------------------------------------------
    # Without these, a native-vs-flat comparison varies retrieval depth AND
    # output length as well as the prompt, and the gap has three explanations.
    for name, expect in [("a_mem", {"retrieve_k": 10}), ("mem0", {"retrieve_k": 10}),
                         ("memorybank", {"retrieve_k": 10}), ("lightmem", {"retrieve_k": 10})]:
        cfg = get_plugin(name).apply_retrieval_budget({}, 10)
        check(f"{name} top-k maps to retrieve_k", {k: cfg[k] for k in expect}, expect)

    # HyperMem retrieves per level; top_k sets the atomic (fact) one and the
    # coarser filters scale with it rather than being pinned flat.
    # Ratios come from the PAPER's depths (§4.1: 10 topics / 10 episodes / 30
    # facts), so both coarse levels sit at k/3 and the funnel keeps its
    # published shape as k sweeps. config.py's shipped 15/20/30 is a different
    # configuration, not the reported one — see plugin.apply_retrieval_budget.
    hm_cfg = hm.apply_retrieval_budget({}, 30)["retrieval"]["top_k"]
    check("hypermem top-k sets the fact level and scales the funnel",
          hm_cfg, {"fact_top_k": 30, "episode_top_k": 10, "topic_top_k": 10})
    check("hypermem leaves initial_candidates alone (recall ceiling, not context)",
          "initial_candidates" in hm_cfg, False)
    # SimpleMem's semantic path only: the keyword/structured paths are precision
    # side-channels upstream tuned at 5, and the union is uncapped regardless.
    check("simplemem top-k sets the semantic path only",
          sm.apply_retrieval_budget({}, 25)["retrieval"], {"semantic_top_k": 25})

    check("every system maps answer tokens to one key",
          {n: get_plugin(n).apply_answer_max_tokens({}, 2000)["answer_max_tokens"]
           for n in available_systems()},
          {n: 2000 for n in available_systems()})

    # lightmem used to hardcode _UNUSED_TOP_K=10 and never read a token budget:
    # dead on the flat path, wrong on the native one.
    check("lightmem reads retrieve_k / answer_max_tokens as adapter keys",
          all(k in _LIGHTMEM_ADAPTER_KEYS for k in ("retrieve_k", "answer_max_tokens")), True)

    # context_items is what makes the budget that CANNOT be equalised visible:
    # a_mem's link expansion and simplemem's union both exceed k.
    from evalgrid.benchmarks.runner import _add_trace_fields

    class _T:
        traces = [{"retrieved_unit_ids": ["a", "b", "a", "c"]}]
    rec: dict = {}
    _add_trace_fields(rec, _T())
    check("context_items counts distinct units the system actually showed",
          rec["context_items"], 3)

    class _TH:
        traces = [{"retrieved_unit_ids": {"topics": ["t"], "episodes": ["e"], "facts": ["f1", "f2"]}}]
    rec_h: dict = {}
    _add_trace_fields(rec_h, _TH())
    check("context_items handles hypermem's level-keyed trace", rec_h["context_items"], 4)

    # LightMem is the one system with a published prompt to reproduce: --prompt
    # native must actually differ from the shared one, or the comparison is a
    # no-op that looks like a result.
    lm_native = plugin.answer_prompt("What is the cat called?",
                                     "2023/07/01 (Sat) 10:00", facts)
    check("lightmem --prompt native differs from shared",
          lm_native != shared_answer_prompt("What is the cat called?",
                                            "2023/07/01 (Sat) 10:00", facts), True)
    check("lightmem native prompt is upstream's, verbatim",
          lm_native[1]["content"],
          "Question time:2023/07/01 (Sat) 10:00 and question:What is the cat called?\n"
          "Please answer the question based on the following memories: "
          + "\n".join(facts))
    # Every other system stays on the shared prompt unless it overrides, so
    # --prompt native is a no-op for them rather than a silent divergence.
    check("systems without a published prompt keep the shared one",
          {n: get_plugin(n).answer_prompt("q", None, ["f"]) == shared_answer_prompt("q", None, ["f"])
           for n in available_systems() if n != "lightmem"},
          {n: True for n in available_systems() if n != "lightmem"})

    # The raw-turn baselines have no plugin, so --prompt native is undefined for
    # them and must say so rather than fall through to the shared prompt.
    try:
        resolve_prompt("user_turns", "native")
        check("--prompt native rejected for a pseudo-system", "no error", "SystemExit")
    except SystemExit as e:
        check("--prompt native rejected for a pseudo-system",
              "no plugin" in str(e), True)
    check("--prompt shared works for a pseudo-system",
          resolve_prompt("user_turns", "shared") is shared_answer_prompt, True)

    # --- 13. answer-phase resumability ------------------------------------------
    # The failure mode here is data loss: resuming must APPEND to an existing
    # predictions.jsonl, never truncate it, and --overwrite must do the reverse.
    from evalgrid.benchmarks.runner import run as run_phase

    class _StubSystem(MemorySystem):
        """Answers "<query_id>!" from memory it never really loads."""
        supports_persistence = True

        def __init__(self):
            self.loaded = []
            self.traces = []

        def reset(self): pass
        def ingest(self, session, *, is_last=False): pass
        def save_memory(self, path): pass

        def load_memory(self, path):
            self.loaded.append(os.path.basename(path))

        def answer(self, query):
            self.traces.append({"query_id": query.query_id,
                                "retrieved_unit_ids": [f"{query.query_id}-u1"]})
            return f"{query.query_id}!"

    class _StubAdapter:
        def __init__(self, samples): self._samples = samples
        def samples(self): return iter(self._samples)

    res_samples = [
        Sample(sample_id=f"s{i}", source="longmemeval",
               queries=[Query(query_id=f"q{i}", question="?", answer="a")],
               sessions=[])
        for i in (1, 2)
    ]
    res_mem = os.path.join(tmp, "resume_mem")
    for s in res_samples:
        d = os.path.join(res_mem, s.sample_id)
        os.makedirs(d)
        with open(os.path.join(d, "notes.jsonl"), "w") as fh:
            fh.write("{}\n")
    res_out = os.path.join(tmp, "resume_predictions.jsonl")

    seen_runs: dict[str, list] = {}
    sys1 = _StubSystem()
    run_phase(sys1, _StubAdapter(res_samples[:1]), out_path=res_out, phase="answer",
              mem_dir=res_mem, on_sample_traces=lambda sid, tr: seen_runs.setdefault(sid, tr))
    check("answer phase writes the first sample",
          [json.loads(l)["question_id"] for l in open(res_out) if l.strip()], ["q1"])
    check("on_sample_traces receives that sample's traces",
          [t["query_id"] for t in seen_runs["s1"]], ["q1"])

    # Resume over BOTH samples: s1 is already answered, so it must be skipped
    # (its memory never loaded — the expensive part) and q1's row preserved.
    sys2 = _StubSystem()
    run_phase(sys2, _StubAdapter(res_samples), out_path=res_out, phase="answer",
              mem_dir=res_mem)
    check("resume appends without losing the earlier answers",
          [json.loads(l)["question_id"] for l in open(res_out) if l.strip()], ["q1", "q2"])
    check("resume skips the answered sample's memory load", sys2.loaded, ["s2"])

    # --overwrite starts clean.
    sys3 = _StubSystem()
    run_phase(sys3, _StubAdapter(res_samples), out_path=res_out, phase="answer",
              mem_dir=res_mem, overwrite=True)
    check("overwrite re-answers everything exactly once",
          [json.loads(l)["question_id"] for l in open(res_out) if l.strip()], ["q1", "q2"])
    check("overwrite reloads every sample's memory", sys3.loaded, ["s1", "s2"])

    # --- 14b. lightmem ingest provenance ----------------------------------------
    # The read side above is only half of it: LightMem also has to be GIVEN ids
    # at ingest. It used to pass `turn.turn_id or ""`, and LongMemEval ships no
    # turn ids, so every fact was stamped with "" no matter what iter_units read.
    from evalgrid.pipeline.qrels_turns import _turn_id as _qrels_turn_id
    from evalgrid.systems.lightmem.longmemeval import LightMemLongMemEval

    class _CapturingLightMem:
        """Stands in for the vendored LightMemory: records what it was handed."""

        class _Config:
            update_queue_top_k = 1
            update_queue_keep_top_n = 1
            update_sim_threshold = 0.5

        def __init__(self):
            self.pairs = []
            self.forced = []          # force_extract per call, in call order
            self.config = self._Config()

        def add_memory(self, messages, **kw):
            self.pairs.append(messages)
            self.forced.append(bool(kw.get("force_extract")))

        def construct_update_queue_all_entries(self, **kw):
            pass

        def offline_update_all_entries(self, **kw):
            pass

    def _new_lightmem_stub():
        """A LightMemLongMemEval with the real constructor skipped.

        The four attributes below are the ingest-path state the constructor
        would have set: the held add_memory call (see
        LightMemSystem.queue_add_memory) and the two turn ledgers — turns that
        never reached the model (dropped_turns) and turns that did
        (ingested_turn_ids, which save_memory differences against the saved
        points to find turns that were sent and yielded nothing).
        """
        sys_ = object.__new__(LightMemLongMemEval)
        sys_._lightmem = _CapturingLightMem()
        sys_._pending = None
        sys_.dropped_turns = 0
        sys_.ingested_turn_ids = set()
        return sys_

    def _ingest_ids(turns):
        sess = Session(session_id="s1", timestamp="2023/05/20 (Sat) 01:33", turns=turns)
        sys_ = _new_lightmem_stub()
        sys_.ingest(sess, is_last=False)
        # One add_memory call is always held back, so that the SAMPLE's last one
        # can be the forced one whichever session it belongs to. Release it
        # before reading what LightMem was handed.
        sys_.flush_add_memory()
        return [[m["turn_id"] for m in pair] for pair in sys_._lightmem.pairs], sess

    def _t(role, content="x"):
        return Turn(role=role, content=content)

    ids, sess = _ingest_ids([_t("user"), _t("assistant"), _t("user"), _t("assistant")])
    check("lightmem stamps synthesised turn ids on both sides of each pair",
          ids, [["s1:t0", "s1:t1"], ["s1:t2", "s1:t3"]])
    check("they are the ids qrels_turns builds, so units join to raw turns",
          ids[1], [_qrels_turn_id(sess, 2), _qrels_turn_id(sess, 3)])

    # The subtle one: a session opening with an assistant turn is skipped into,
    # so the first PAIR is not turn 0. Indexing a slice (what this did before)
    # would have called it s1:t0 and pointed provenance at the wrong turn.
    ids, sess = _ingest_ids([_t("assistant"), _t("user"), _t("assistant"), _t("user"),
                             _t("assistant")])
    check("a leading-assistant skip does not shift the ids",
          ids, [["s1:t1", "s1:t2"], ["s1:t3", "s1:t4"]])

    # And the pairs themselves must be unchanged by the reindexing — this fix
    # must not alter which turns lightmem ingests. 1,942 of 23,867 LongMemEval
    # sessions open with an assistant turn, so a silent shift here would be a
    # different store, not a provenance fix.
    ids, _ = _ingest_ids([_t("user"), _t("assistant"), _t("user")])
    check("an odd tail is still dropped, exactly as before", ids, [["s1:t0", "s1:t1"]])
    ids, _ = _ingest_ids([_t("user"), _t("user"), _t("assistant")])
    check("a non-alternating pair is still skipped whole", ids, [])

    # ...and the turns those two rules drop are now COUNTED. This is the one
    # system whose ingest can silently discard input (2,020 turns over
    # LongMemEval-S, 3 of them carrying has_answer), and until this counter a
    # store built from 2 of 3 turns was indistinguishable from a complete one.
    def _dropped(turns):
        sess = Session(session_id="s1", timestamp="2023/05/20 (Sat) 01:33", turns=turns)
        sys_ = _new_lightmem_stub()
        sys_.ingest(sess, is_last=True)
        return sys_.dropped_turns

    check("dropped_turns counts the odd tail",
          _dropped([_t("user"), _t("assistant"), _t("user")]), 1)
    check("dropped_turns counts a whole skipped session",
          _dropped([_t("user"), _t("user"), _t("assistant")]), 3)
    check("dropped_turns counts the leading-assistant skip",
          _dropped([_t("assistant"), _t("user"), _t("assistant")]), 1)
    check("dropped_turns is 0 on a clean session",
          _dropped([_t("user"), _t("assistant")]), 0)

    # The end-of-sample flush belongs to the SAMPLE, not to the last session.
    # LightMem buffers until force_extract/force_segment say otherwise, so the
    # sample's final add_memory call must carry them. Tying that to "the last
    # pair of the last session" meant a final session yielding no pair (a lone
    # assistant turn — 2 of the 500 LongMemEval-S samples) never flushed, and
    # everything still buffered was dropped with no error.
    def _forced_flags(sessions):
        sys_ = _new_lightmem_stub()
        for i, turns in enumerate(sessions):
            sys_.ingest(Session(session_id=f"s{i}", timestamp="2023/05/20 (Sat) 01:33",
                                turns=turns),
                        is_last=(i == len(sessions) - 1))
        return sys_._lightmem.forced

    check("exactly the sample's last add_memory call is forced",
          _forced_flags([[_t("user"), _t("assistant")],
                         [_t("user"), _t("assistant")]]),
          [False, True])
    check("a last session with no pairs does not swallow the flush",
          _forced_flags([[_t("user"), _t("assistant")],
                         [_t("assistant")]]),
          [True])
    check("a sample that produced no call at all needs no flush",
          _forced_flags([[_t("assistant")]]), [])

    # --- 15. answering-token budget ---------------------------------------------
    # Matching two systems at the same k does not match them at the same
    # evidence: a pair doc is ~2x a turn doc. fit_to_budget is the generalisation,
    # and its three rules are what the numbers depend on — pin them.
    from evalgrid.lib.paths import answer_depth_segment
    from evalgrid.lib.tokens import fit_to_budget, token_truncator

    cut = token_truncator("gpt-4o-mini")
    texts = ["a", "b", "c", "d"]
    kept, ctx = fit_to_budget(texts, [10, 20, 30, 40], None)
    check("unbudgeted keeps every fact", kept, texts)
    check("unbudgeted still measures the context (separators included)",
          ctx["context_tokens"], 10 + 20 + 30 + 40 + 3)

    kept, ctx = fit_to_budget(texts, [10, 20, 30, 40], 35)
    check("budget stops at the first fact that would cross it", kept, ["a", "b"])
    check("budget counts one separator per fact after the first",
          ctx["context_tokens"], 10 + 1 + 20)
    check("a cut ranking is flagged truncated", ctx["context_truncated"], True)
    check("a cut ranking is not under-filled", ctx["context_underfilled"], False)

    # Skip-and-continue would break the rank-prefix property: the context would
    # no longer be the top of the ranking, and short-unit systems would silently
    # be read deeper than long-unit ones.
    kept, _ = fit_to_budget(["big", "small"], [100, 1], 50)
    check("a fact that fits is NOT pulled in past one that doesn't", kept, ["big"])

    # Rule 2: a head fact that alone exceeds T is CUT to T, not admitted whole.
    # Admitting it whole reports its real size under the budget's label, which
    # is not a matched cell; dropping it answers from an empty context.
    from evalgrid.lib.tokens import token_counter
    count = token_counter("gpt-4o-mini")
    big = "MemoryBank block. " * 400
    kept, ctx = fit_to_budget([big, "b"], [count(big), 5], 200, truncate=cut)
    check("a head fact over budget is cut, not dropped", len(kept), 1)
    check("the cut lands ON the budget, measured not assumed",
          count(kept[0]), ctx["context_tokens"])
    check("the cut does not exceed the budget", ctx["context_tokens"] <= 200, True)
    check("the cut is flagged", ctx["context_unit_truncated"], True)
    check("the cut says how much it dropped",
          ctx["context_truncated_tokens"], count(big) - ctx["context_tokens"])
    check("a cut cell is not ALSO reported as under-filled",
          ctx["context_underfilled"], False)
    check("a filled budget is not flagged as cut",
          fit_to_budget(texts, [10, 20, 30, 40], 35, truncate=cut)[1]["context_unit_truncated"],
          False)

    # Rule 3: once something is kept, stop — do not fragment to top up. The
    # residual is bounded by one unit size and is measured, not assumed away.
    kept, ctx = fit_to_budget(["a", big], [10, count(big)], 200, truncate=cut)
    check("a later oversized fact is NOT cut to fill the remainder", kept, ["a"])
    check("that under-fill is reported rather than papered over",
          (ctx["context_tokens"], ctx["context_unit_truncated"]), (10, False))

    # Without a truncator the old keep-whole behaviour remains, and it must
    # announce itself rather than pass as a matched cell.
    kept, ctx = fit_to_budget([big, "b"], [count(big), 5], 200)
    check("no truncator degrades LOUDLY", (kept[0], ctx["context_overflow"]), (big, True))

    kept, ctx = fit_to_budget(["a"], [10], 500)
    check("a ranking that runs out early is flagged under-filled",
          (ctx["context_underfilled"], ctx["context_truncated"]), (True, False))

    check("empty ranking is measured, not crashed",
          fit_to_budget([], [], 500)[1]["context_items"], 0)

    # --- 15b. memory atoms: what BOTH budgets cut ------------------------------
    # k counts memories and a budget counts memories' tokens, so a system whose
    # unit is several memories (MemoryBank's date blocks) has to be expanded
    # before either cut — otherwise k=3 is unreachable and a token cut lands
    # inside a memory. The expansion must not change the prompt.
    from evalgrid.pipeline.answer import memory_atoms

    plain = {"retrieved_units": ["m1", "m2"], "retrieved_unit_ids": ["a", "b"]}
    check("an ungrouped system passes straight through",
          memory_atoms(plain, None), (["m1", "m2"], ["a", "b"]))

    # The members come from the corpus, so no system's own trace has to change
    # and every saved retrieval already on disk stays usable.
    grouped = {"retrieved_units": ["m1\nm2", "m3"],
               "retrieved_unit_ids": ["a", "b", "c"],
               "unit_id_groups": [["a", "b"], ["c"]]}
    corpus = {"a": "m1", "b": "m2", "c": "m3"}
    texts, ids = memory_atoms(grouped, corpus)
    check("a grouped unit expands to its memories", (texts, ids),
          (["m1", "m2", "m3"], ["a", "b", "c"]))
    check("expanding does not change the prompt (shared_answer_prompt joins on \\n)",
          "\n".join(texts), "\n".join(grouped["retrieved_units"]))

    # No corpus is a hard error, not a silent fall back to whole-unit cutting:
    # that would report k memories and hand the generator more.
    try:
        memory_atoms(grouped, None)
        check("a grouped unit with no member texts raises", "no raise", "RuntimeError")
    except RuntimeError:
        check("a grouped unit with no member texts raises", "RuntimeError", "RuntimeError")
    try:
        memory_atoms(grouped, {"a": "m1"})
        check("a corpus missing an id raises too", "no raise", "RuntimeError")
    except RuntimeError:
        check("a corpus missing an id raises too", "RuntimeError", "RuntimeError")

    # HyperMem's retrieved_unit_ids is a per-layer dict, not a flat list, so it
    # cannot be aligned to units and must come back as None rather than be
    # half-cut alongside them.
    layered = {"retrieved_units": ["f1", "f2"],
               "retrieved_unit_ids": {"topics": ["t"], "facts": ["f1", "f2"]}}
    check("unalignable ids come back as None", memory_atoms(layered, None)[1], None)

    # The depth segment is path IDENTITY: top<k> must stay byte-identical or a
    # GPU-day of judged runs under outputs/ is orphaned.
    check("top-k segment unchanged", answer_depth_segment(10), "top10")
    check("budget gets its own segment", answer_depth_segment(10, 500), "budget500")
    check("oracle/full stay uncapped", answer_depth_segment(10, 500, True), "all")

    # --- 16. query-time cost ----------------------------------------------------
    # An answering-token budget equalises what the generator READS and says
    # nothing about what was spent choosing it. That assumption holds for the
    # flat path (retrieval is an offline embedding pass) and is false for a_mem /
    # simplemem / hypermem, whose retrieval spends LLM calls per question.
    from evalgrid.lib.cost import query_cost, split_answer_layers, sum_cost
    from evalgrid.benchmarks.runner import _answer_record

    flat = query_cost(answer=(900, 40, 1))
    check("flat retrieval costs no LLM tokens, by construction",
          (flat["query_retrieval_tokens"], flat["query_retrieval_calls"]), (0, 0))
    check("flat query cost is the one answer call",
          (flat["query_llm_tokens"], flat["query_llm_calls"]), (940, 1))
    check("the answer prompt is broken out for the native path's C3",
          flat["query_answer_input_tokens"], 900)

    # a_mem/simplemem shape: the memory layer is query-time reasoning, not
    # construction — the runner snapshots per phase, so this is answer-phase only.
    by_layer = {"memory": {"input_tokens": 300, "output_tokens": 50, "calls": 6},
                "generation": {"input_tokens": 900, "output_tokens": 40, "calls": 1}}
    retrieval, answer = split_answer_layers(by_layer)
    check("the memory layer is charged to retrieval", retrieval, (300, 50, 6))
    check("the generation layer is charged to the answer", answer, (900, 40, 1))
    native = query_cost(retrieval, answer)
    check("a native loop's retrieval cost is visible",
          (native["query_retrieval_tokens"], native["query_retrieval_calls"]), (350, 6))
    check("query cost totals both", (native["query_llm_tokens"], native["query_llm_calls"]),
          (1290, 7))

    # Systems with no layer split report everything under "generation" (runner.
    # _layer_usage's fallback) — correct, since no memory layer means no
    # query-time memory calls to miss.
    check("a single-layer system charges nothing to retrieval",
          split_answer_layers({"generation": {"input_tokens": 5, "output_tokens": 1, "calls": 1}})[0],
          (0, 0, 0))
    check("an unknown extra layer counts as query-time reasoning, not as free",
          split_answer_layers({"rerank": {"input_tokens": 7, "output_tokens": 2, "calls": 1}})[0],
          (7, 2, 1))

    # The wiring, not just the helper: a native prediction record must carry the
    # same flat fields the flat path writes, or the two cannot share a cost axis.
    rec_cost = _answer_record(Query(query_id="q1", question="?", answer="a"),
                              Sample(sample_id="s1", source="longmemeval", queries=[], sessions=[]),
                              "hyp", 1.0, by_layer, (1200, 90, 7))
    check("native records carry the flat cost fields",
          (rec_cost["query_retrieval_calls"], rec_cost["query_llm_tokens"]), (6, 1290))
    check("sum_cost totals a run", sum_cost([flat, native])["query_llm_calls"], 8)

    # And the flat path's per-question record must carry the context measurement
    # — without it "matched at 500 tokens" is a claim, not a measurement.
    import asyncio as _asyncio
    from evalgrid.models.base import GenerationResult
    from evalgrid.pipeline.answer import _answer_one

    class _StubGen:
        async def generate_with_usage(self, messages, *, max_tokens=2000, **kw):
            return GenerationResult(text="ok", input_tokens=900, output_tokens=40)

    _, ans_stats, _ = _asyncio.run(_answer_one(
        _asyncio.Semaphore(1), _StubGen(), 2000, shared_answer_prompt,
        "q1", "?", None, ["f1", "f2"],
        {"context_items": 2, "context_tokens": 480, "context_budget": 500,
         "context_truncated": True, "context_underfilled": False, "context_overflow": False}))
    check("flat records carry the context measurement",
          (ans_stats["context_items"], ans_stats["context_tokens"],
           ans_stats["context_budget"], ans_stats["context_truncated"]),
          (2, 480, 500, True))
    check("flat records carry the cost fields",
          (ans_stats["query_retrieval_tokens"], ans_stats["query_llm_tokens"],
           ans_stats["query_llm_calls"]), (0, 940, 1))

    # --- 17. trajectory recording -----------------------------------------------
    # The feature's whole promise is that it is inert. These checks are the
    # promise, so they come before anything about what it records.
    from evalgrid.lib.trace import (
        NULL_RECORDER, STAGES, TRAJECTORY_FILENAMES, TraceRecorder, build_recorder,
        record_calls, trajectory_filename,
    )
    from evalgrid.models.base import TokenUsage

    check("off yields the shared null recorder", build_recorder("off") is NULL_RECORDER, True)
    check("a real level yields a real recorder",
          isinstance(build_recorder("calls"), TraceRecorder), True)
    check("null recorder reports itself disabled", NULL_RECORDER.enabled, False)

    class _Calc:
        """Stands in for an LLM adapter: a method plus a TokenUsage."""

        def __init__(self):
            self.token_usage = TokenUsage()
            self.seen = []

        def get_completion(self, prompt, temperature=None):
            self.seen.append((prompt, temperature))
            self.token_usage.add(10, 3)
            return f"<{prompt}>"

    # RULE 1 — at level off, record_calls must not wrap anything at all. A
    # wrapper that checks a flag is not the same as no wrapper.
    off = _Calc()
    check("off installs no wrapper",
          record_calls(off, "get_completion", recorder=NULL_RECORDER, step="x"), False)
    check("off leaves the method identical",
          off.get_completion.__func__ is _Calc.get_completion, True)

    # RULE 2 — recording must not change what the system computes. Same calls,
    # same return values, same arguments seen, byte for byte.
    traj = os.path.join(tmp, "traj")
    rec = build_recorder("steps", max_chars=20)
    on = _Calc()
    record_calls(on, "get_completion", recorder=rec, step="calc.get_completion")
    rec.begin_sample("s1", "ingest", traj)
    got_on = [on.get_completion("a", temperature=0.5), on.get_completion("bb")]
    got_off = [off.get_completion("a", temperature=0.5), off.get_completion("bb")]
    check("wrapped calls return exactly what unwrapped ones do", got_on, got_off)
    check("wrapped calls pass through their arguments unchanged", on.seen, off.seen)
    check("wrapped calls still account tokens exactly once",
          (on.token_usage.input_tokens, on.token_usage.calls), (20, 2))

    # RULE 3 — re-installing (a system that rebuilds its adapter every reset)
    # must not stack wrappers, or one call would be logged N times and priced N
    # times.
    record_calls(on, "get_completion", recorder=rec, step="calc.get_completion")
    on.get_completion("c")
    rec.end_sample()
    traj_f = os.path.join(traj, trajectory_filename("steps"))
    rows = [json.loads(l) for l in open(traj_f) if l.strip()]
    check("a re-install does not stack wrappers", len(rows), 3)

    r0 = rows[0]
    check("a call record carries the sample and phase",
          (r0["sample_id"], r0["phase"], r0["kind"]), ("s1", "ingest", "call"))
    check("cost is the per-call delta, not the running total",
          (r0["cost"]["input_tokens"], r0["cost"]["output_tokens"], r0["cost"]["calls"]),
          (10, 3, 1))
    check("an unclaimed call is stage 'llm', not a missing value", r0["stage"], "llm")
    check("seq numbers are per sample and monotone", [r["seq"] for r in rows], [1, 2, 3])

    # Truncation must be visible: a reader has to know the payload was cut and
    # how long the original was, or "chars" silently becomes a lie.
    long_rec = build_recorder("calls", max_chars=5)
    long_obj = _Calc()
    record_calls(long_obj, "get_completion", recorder=long_rec, step="calc")
    traj2 = os.path.join(tmp, "traj2")
    long_rec.begin_sample("s1", "ingest", traj2)
    long_obj.get_completion("x" * 100)
    long_rec.end_sample()
    row = json.loads(open(os.path.join(traj2, trajectory_filename("calls"))).read().strip())
    check("payloads are clipped to max_chars", len(row["in"]["text"]), 5)
    check("the original length survives clipping", row["in"]["chars"], 100)
    check("clipping is flagged", row["in"]["truncated"], True)

    # RULE 4 — a raising call is logged and re-raised. Losing it would make the
    # trajectory claim the step never happened.
    class _Boom:
        token_usage = TokenUsage()

        def go(self, prompt):
            raise ValueError("nope")

    boom_rec = build_recorder("calls")
    boom = _Boom()
    record_calls(boom, "go", recorder=boom_rec, step="boom")
    traj3 = os.path.join(tmp, "traj3")
    boom_rec.begin_sample("s1", "ingest", traj3)
    try:
        boom.go("q")
        check("a failing call still raises", "no error", "ValueError")
    except ValueError:
        check("a failing call still raises", True, True)
    boom_rec.end_sample()
    boom_row = json.loads(open(os.path.join(traj3, trajectory_filename("calls"))).read().strip())
    check("a failing call is recorded, with its error type",
          boom_row["meta"]["error"], "ValueError")

    # step(): the L2 layer. Its cost is the delta across the block, which is the
    # resolution (layer x phase) accounting cannot reach.
    usage = TokenUsage()
    step_rec = build_recorder("steps")
    traj4 = os.path.join(tmp, "traj4")
    step_rec.begin_sample("s2", "ingest", traj4)
    with step_rec.step("extract", "lm.extract", inputs="seg", usage=usage) as h:
        usage.add(100, 20)
        h.out("a fact")
        h.units(["u1", "u2"])
        h.meta(segment_idx=4)
    step_rec.end_sample()
    srow = json.loads(open(os.path.join(traj4, trajectory_filename("steps"))).read().strip())
    check("a step's cost is the delta across the block",
          (srow["cost"]["input_tokens"], srow["cost"]["output_tokens"]), (100, 20))
    check("a step records its outputs, units and meta",
          (srow["out"]["text"], srow["unit_ids"], srow["meta"]["segment_idx"]),
          ("a fact", ["u1", "u2"], 4))
    check("every stage used is in the closed vocabulary",
          all(r["stage"] in STAGES for r in rows + [srow]), True)

    # A call inside a step inherits that step's stage — which is what lets a
    # system with L2 attribute its LLM calls without naming them twice.
    nest_rec = build_recorder("steps")
    nest = _Calc()
    record_calls(nest, "get_completion", recorder=nest_rec, step="calc")
    traj5 = os.path.join(tmp, "traj5")
    nest_rec.begin_sample("s3", "ingest", traj5)
    with nest_rec.step("extract", "lm.extract") as _h:
        nest.get_completion("inner")
    nest_rec.end_sample()
    nrows = [json.loads(l) for l in open(os.path.join(traj5, trajectory_filename("steps"))) if l.strip()]
    inner = [r for r in nrows if r["kind"] == "call"][0]
    outer = [r for r in nrows if r["kind"] == "step"][0]
    check("a call inside a step inherits its stage", inner["stage"], "extract")
    check("nesting is recoverable from parent_seq", inner["parent_seq"], outer["seq"])

    # At level "calls" the step() block must still RUN (identical code path) but
    # write nothing — otherwise the level would change control flow.
    ran = []
    calls_rec = build_recorder("calls")
    traj6 = os.path.join(tmp, "traj6")
    calls_rec.begin_sample("s4", "ingest", traj6)
    with calls_rec.step("extract", "lm.extract") as _h:
        ran.append(True)
    calls_rec.end_sample()
    check("level 'calls' still runs the step body", ran, [True])
    check("level 'calls' writes no step record", open(os.path.join(traj6, trajectory_filename("calls"))).read().strip(), "")

    # ...but a call inside that block must still carry the stage. The stage is
    # L1 information; if it were dropped at "calls", the same run recorded at
    # two levels would disagree about what its calls were.
    lvl_rec = build_recorder("calls")
    lvl = _Calc()
    record_calls(lvl, "get_completion", recorder=lvl_rec, step="calc")
    traj6b = os.path.join(tmp, "traj6b")
    lvl_rec.begin_sample("s4b", "ingest", traj6b)
    with lvl_rec.step("extract", "lm.extract") as _h:
        lvl.get_completion("inner")
    lvl_rec.end_sample()
    lrow = json.loads(open(os.path.join(traj6b, trajectory_filename("calls"))).read().strip())
    check("level 'calls' still stages a call inside a step", lrow["stage"], "extract")
    check("level 'calls' leaves no parent pointing at an unemitted record",
          lrow["parent_seq"], None)

    # The prompt is captured BEFORE the call, so a method that mutates its
    # argument in place and returns it (LlmLingua2Compressor.compress) cannot
    # make "in" and "out" identical and the step read as a no-op it never was.
    class _Mutator:
        token_usage = TokenUsage()

        def compress(self, messages):
            for m in messages:
                m["content"] = m["content"].replace("verbose ", "")
            return messages

    mut_rec = build_recorder("full")
    mut = _Mutator()
    record_calls(mut, "compress", recorder=mut_rec, stage="compress", step="c")
    traj7 = os.path.join(tmp, "traj7")
    mut_rec.begin_sample("s5", "ingest", traj7)
    mut.compress([{"role": "user", "content": "a verbose sentence"}])
    mut_rec.end_sample()
    mrow = json.loads(open(os.path.join(traj7, trajectory_filename("full"))).read().strip())
    check("an in-place mutator's prompt is the PRE-call value",
          "verbose" in mrow["in"]["text"], True)
    check("an in-place mutator's response is the POST-call value",
          "verbose" in mrow["out"]["text"], False)

    # cost_of: for a method whose CALLER, not the method, posts its tokens
    # (lightmem's meta_text_extract), the delta across the method is always
    # zero and its calls would be priced nowhere.
    class _Deferred:
        token_usage = TokenUsage()   # never moves during the call

        def extract(self):
            return [{"output_prompt": "o", "usage": {"prompt_tokens": 40,
                                                     "completion_tokens": 9}}]

    cost_rec = build_recorder("calls")
    deferred = _Deferred()
    record_calls(deferred, "extract", recorder=cost_rec, stage="extract", step="e",
                 cost_of=lambda r: (sum(i["usage"]["prompt_tokens"] for i in r),
                                    sum(i["usage"]["completion_tokens"] for i in r),
                                    len(r)))
    traj8 = os.path.join(tmp, "traj8")
    cost_rec.begin_sample("s6", "ingest", traj8)
    deferred.extract()
    cost_rec.end_sample()
    crow = json.loads(open(os.path.join(traj8, trajectory_filename("calls"))).read().strip())
    check("cost_of prices a call the usage delta cannot see",
          (crow["cost"]["input_tokens"], crow["cost"]["output_tokens"], crow["cost"]["calls"]),
          (40, 9, 1))

    # stage_of/step_of: one choke-point method serving two stages (mem0).
    stage_rec = build_recorder("calls")
    multi = _Calc()
    record_calls(multi, "get_completion", recorder=stage_rec, step="m",
                 stage_of=lambda prompt, **k: "extract" if prompt == "x" else "update",
                 step_of=lambda prompt, **k: f"m.{prompt}")
    traj9 = os.path.join(tmp, "traj9")
    stage_rec.begin_sample("s7", "ingest", traj9)
    multi.get_completion("x")
    multi.get_completion("y")
    stage_rec.end_sample()
    srows = [json.loads(l) for l in open(os.path.join(traj9, trajectory_filename("calls"))) if l.strip()]
    check("stage_of names each call from its own arguments",
          [r["stage"] for r in srows], ["extract", "update"])
    check("step_of names each call from its own arguments",
          [r["step"] for r in srows], ["m.x", "m.y"])

    # record_steps: the L2 counterpart of record_calls, which is how a system
    # whose whole LLM interface is one method gets its calls staged and nested.
    from evalgrid.lib.trace import record_steps

    class _Pipeline:
        def __init__(self, calc):
            self.calc = calc

        def add(self, content):
            self.calc.get_completion("analyze " + content)
            self.evolve(content)
            return "id-1"

        def evolve(self, content):
            self.calc.get_completion("evolve " + content)
            return True

    check("off installs no step wrapper either",
          record_steps(_Pipeline(off), "add", recorder=NULL_RECORDER,
                       stage="extract", step="p.add"), False)

    calc = _Calc()
    pipe = _Pipeline(calc)
    steps_rec = build_recorder("steps")
    record_calls(calc, "get_completion", recorder=steps_rec, step="p.call")
    record_steps(pipe, "add", recorder=steps_rec, stage="extract", step="p.add",
                 usage=calc.token_usage)
    record_steps(pipe, "evolve", recorder=steps_rec, stage="update", step="p.evolve",
                 usage=calc.token_usage)
    traj10 = os.path.join(tmp, "traj10")
    steps_rec.begin_sample("s8", "ingest", traj10)
    check("record_steps returns the wrapped method's value untouched",
          pipe.add("note"), "id-1")
    steps_rec.end_sample()
    prows = [json.loads(l) for l in open(os.path.join(traj10, trajectory_filename("steps"))) if l.strip()]
    pcalls = [r for r in prows if r["kind"] == "call"]
    psteps = {r["step"]: r for r in prows if r["kind"] == "step"}
    check("record_steps stages the calls made inside it",
          [r["stage"] for r in pcalls], ["extract", "update"])
    check("a nested step parents to the enclosing one",
          psteps["p.evolve"]["parent_seq"], psteps["p.add"]["seq"])
    check("an enclosing step's cost sums every call beneath it",
          psteps["p.add"]["cost"]["calls"], 2)
    check("a nested step is priced on its own calls only",
          psteps["p.evolve"]["cost"]["calls"], 1)

    # --trace-samples: sampling decided in one place, invisible to the system.
    lim = build_recorder("calls", max_samples=1)
    check("the first sample records", lim.begin_sample("a", "ingest", os.path.join(tmp, "t_a")), True)
    lim.end_sample()
    check("a sample past --trace-samples does not",
          lim.begin_sample("b", "ingest", os.path.join(tmp, "t_b")), False)
    lim.call(step="dropped")   # must be a silent no-op, not a crash
    check("records past the sample limit are dropped, not raised",
          os.path.isdir(os.path.join(tmp, "t_b")), False)

    # Each level writes its OWN file. Sharing one name let an ingest resume keep
    # a file recorded at another level while the console announced the new one,
    # let --overwrite destroy it, and let a resumed answer run APPEND a "steps"
    # run into a "calls" file with nothing marking the boundary.
    check("each level names its own file",
          [trajectory_filename(l) for l in ("calls", "steps", "full")],
          ["trajectory-calls.jsonl", "trajectory-steps.jsonl", "trajectory-full.jsonl"])
    check("no two levels can collide",
          len(TRAJECTORY_FILENAMES), 3)
    check("a record says which level and clip produced it",
          (rows[0]["level"], rows[0]["max_chars"]), ("steps", 20))
    check("full records no clip, and says so",
          build_recorder("full").max_chars, None)
    check("the recorder owns the filename, not the caller",
          build_recorder("calls").filename, "trajectory-calls.jsonl")

    # Two levels over one directory must land in two files, not one.
    both = os.path.join(tmp, "both_levels")
    for lvl in ("calls", "steps"):
        r = build_recorder(lvl)
        obj = _Calc()
        record_calls(obj, "get_completion", recorder=r, step="calc")
        r.begin_sample("s1", "ingest", both)
        obj.get_completion("x")
        r.end_sample()
    check("two levels over one dir write two files",
          sorted(os.listdir(both)),
          ["trajectory-calls.jsonl", "trajectory-steps.jsonl"])

    # A recorder that cannot open its file must not take the run down.
    blocker = os.path.join(tmp, "not_a_dir")
    with open(blocker, "w") as fh:
        fh.write("x")
    bad = build_recorder("calls")
    check("an unopenable trajectory dir fails soft",
          bad.begin_sample("s", "ingest", os.path.join(blocker, "nope")), False)
    check("and is counted", bad.failures >= 1, True)

    # Every system must accept a recorder without one being required.
    check("MemorySystem defaults to the null recorder",
          _StubSystem().recorder is NULL_RECORDER, True)
    stub = _StubSystem()
    stub.install_recorder(build_recorder("calls"))
    check("install_recorder attaches it", stub.recorder.enabled, True)

    # The trajectory shares a directory with the memory it describes, and it is
    # opened BEFORE ingest runs. So a sample that crashed mid-ingest under
    # --trace-level must NOT look ingested to the resume check — otherwise
    # turning recording on would silently change which samples get built.
    from evalgrid.benchmarks.runner import _memory_saved

    crashed = os.path.join(tmp, "crashed_sample")
    os.makedirs(crashed)
    with open(os.path.join(crashed, trajectory_filename("steps")), "w") as fh:
        fh.write('{"seq": 1}\n')
    check("a trajectory alone does not mark a sample as ingested",
          _memory_saved(crashed), False)
    with open(os.path.join(crashed, "notes.jsonl"), "w") as fh:
        fh.write("{}\n")
    check("real memory beside it still does", _memory_saved(crashed), True)

    # Every REGISTERED system must actually install L1, or it silently produces
    # an empty trajectory that reads as "this system made no LLM calls".
    import importlib

    installed = {}
    for _name in available_systems():
        _mod = importlib.import_module(f"evalgrid.systems.{_name}.system")
        installed[_name] = any(
            isinstance(c, type) and c.__module__ == _mod.__name__
            and "_install_trace" in vars(c) and "install_recorder" in vars(c)
            for c in vars(_mod).values()
        )
    check("every registered system installs call recording",
          installed, {n: True for n in available_systems()})

    # --- has_answer label granularity -------------------------------------
    #
    # qrels_has_answer's labels only mean the same thing across two systems if
    # both label at the same granularity, so the value has to exist and has to
    # be one the reader of a qrels manifest can interpret. A typo'd or absent
    # value would silently read as "turn" (the base-class default) and claim a
    # session-level store labels turns.
    _GRANULARITIES = {"turn", "pair", "session", "window"}
    declared = {n: getattr(get_plugin(n), "has_answer_granularity", None)
                for n in available_systems()}
    check("every registered system declares a known has_answer granularity",
          {n: (g in _GRANULARITIES) for n, g in declared.items()},
          {n: True for n in available_systems()})
    # Pinned, not merely non-empty: these are the values the smoke4 label counts
    # were explained by (1-2 labels per evidence turn at "turn", 16-25 at
    # "session"), so a change here is a change to what a published recall number
    # means and should have to be made deliberately.
    check("granularities are the measured ones",
          declared,
          {"a_mem": "turn", "hypermem": "session", "lightmem": "turn",
           "mem0": "pair", "memorybank": "pair", "simplemem": "window"})

    # --- mem0: an empty extraction is not an empty add() ------------------
    #
    # The two were one counter until 2026-08-19 and the merged value overstated
    # lost facts by 21% on 58bf7951. What makes them separable
    # is reading the extraction reply itself, so that reader is pinned here —
    # including the two cases it must NOT claim: an unparseable reply (a JSON
    # failure, counted elsewhere) and a reply with facts in it.
    import types as _types

    from evalgrid.systems.mem0.system import _extracted_nothing

    def _choices(content):
        return [_types.SimpleNamespace(
            message=_types.SimpleNamespace(content=content), finish_reason="stop")]

    check("mem0 counts an empty facts list as an empty extraction",
          _extracted_nothing(_choices('{"facts": []}')), True)
    check("mem0 sees through upstream's ```json fence",
          _extracted_nothing(_choices('```json\n{"facts": []}\n```')), True)
    check("mem0 does not count a reply that carried facts",
          _extracted_nothing(_choices('{"facts": ["a"]}')), False)
    check("mem0 does not count an unparseable reply",
          _extracted_nothing(_choices("not json")), False)
    check("mem0 does not count a response with no choices",
          _extracted_nothing([]), False)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("All equivalence checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
