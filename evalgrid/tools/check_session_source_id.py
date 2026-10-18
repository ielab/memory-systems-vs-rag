#!/usr/bin/env python3
"""
check_session_source_id.py
==========================
Standalone verification for the LightMem ``session_id`` plumbing and the
``source_id`` batch-alignment fix. Exercises the *real* functions in
``lightmem.memory.utils`` with synthetic data — no GPU, no models, no API.

    python -m evalgrid.tools.check_session_source_id

Exit code 0 = all checks passed, 1 = a check failed.

``utils.py`` imports ``tiktoken`` / ``transformers`` only for an unrelated
tokenizer helper that none of the tested functions call, so if those packages
are missing (e.g. on a laptop) we stub them and run anyway.
"""

import logging
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # evalgrid/tools/ -> repo root
sys.path.insert(0, str(REPO / "vendor"))


def _stub_missing_tokenizer_deps():
    try:
        import tiktoken  # noqa: F401
        import transformers.tokenization_utils_fast  # noqa: F401
        import transformers.tokenization_utils  # noqa: F401
        return
    except ImportError:
        pass
    tk = types.ModuleType("tiktoken")
    tk.Encoding = type("Encoding", (), {})
    sys.modules["tiktoken"] = tk
    sys.modules.setdefault("transformers", types.ModuleType("transformers"))
    m1 = types.ModuleType("transformers.tokenization_utils_fast")
    m1.PreTrainedTokenizerFast = type("PreTrainedTokenizerFast", (), {})
    sys.modules["transformers.tokenization_utils_fast"] = m1
    m2 = types.ModuleType("transformers.tokenization_utils")
    m2.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    sys.modules["transformers.tokenization_utils"] = m2


_stub_missing_tokenizer_deps()

from lightmem.memory.utils import (  # noqa: E402
    MemoryEntry,
    assign_sequence_numbers_with_timestamps,
    convert_extraction_results_to_memory_entries,
)

logger = logging.getLogger("check")
logging.basicConfig(level=logging.WARNING, format="    [util] %(levelname)s %(message)s")

_failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


def _msg(session_time, weekday, session_id, role, speaker, turn_id="", has_answer=False):
    return {
        "role": role,
        "content": f"{speaker}:{role}",
        "session_time": session_time,
        "weekday": weekday,
        "session_id": session_id,
        "speaker_id": speaker,
        "speaker_name": speaker,
        "turn_id": turn_id,
        "has_answer": has_answer,
    }


def build_fixture():
    """Two batches, one topic segment each, four (user, assistant) messages each.

    Global sequence numbers 0..7; user turns on even indices (the fact sources).
      batch 0 -> session_0  (source_ids 0, 1 -> seq 0, 2)
      batch 1 -> session_1  (source_ids 2, 3 -> seq 4, 6)
    Turn-level ground truth: the FIRST user turn of each session is the evidence
    turn (has_answer=True), the second is not -> tests per-turn attribution.
    """
    s0t, s1t = "2023/05/20 (Sat) 10:00", "2023/05/21 (Sun) 11:00"
    batch0 = [[
        _msg(s0t, "Sat", "session_0", "user", "Alice", turn_id="t0", has_answer=True),
        _msg(s0t, "Sat", "session_0", "assistant", "Alice"),
        _msg(s0t, "Sat", "session_0", "user", "Alice", turn_id="t2", has_answer=False),
        _msg(s0t, "Sat", "session_0", "assistant", "Alice"),
    ]]
    batch1 = [[
        _msg(s1t, "Sun", "session_1", "user", "Bob", turn_id="t4", has_answer=True),
        _msg(s1t, "Sun", "session_1", "assistant", "Bob"),
        _msg(s1t, "Sun", "session_1", "user", "Bob", turn_id="t6", has_answer=False),
        _msg(s1t, "Sun", "session_1", "assistant", "Bob"),
    ]]
    return [batch0, batch1], [[10], [20]]  # extract_list, topic_id_mapping


def run():
    extract_list, topic_id_mapping = build_fixture()

    (extract_list, timestamps_list, weekday_list, speaker_list,
     session_list, turn_id_list, has_answer_list,
     topic_id_map) = assign_sequence_numbers_with_timestamps(
        extract_list, offset_ms=500, topic_id_mapping=topic_id_mapping
    )
    max_source_ids = [
        max((m["sequence_number"] // 2 for seg in batch for m in seg), default=0)
        for batch in extract_list
    ]

    # ---- 1. assign_* collects per-message provenance aligned with sequence ----
    print("\n# provenance collection")
    check("assign_* returns a session_list", session_list is not None)
    check("session_list length == 8", len(session_list) == 8, f"got {len(session_list)}")
    check("session_list aligned to sequence_number",
          session_list == ["session_0"] * 4 + ["session_1"] * 4, f"got {session_list}")
    check("turn_id_list aligned to sequence_number",
          turn_id_list == ["t0", "", "t2", "", "t4", "", "t6", ""], f"got {turn_id_list}")
    check("has_answer_list aligned to sequence_number",
          has_answer_list == [True, False, False, False, True, False, False, False],
          f"got {has_answer_list}")
    check("max_source_ids == [1, 3]", max_source_ids == [1, 3], f"got {max_source_ids}")

    # ---- 2. MemoryEntry carries the new provenance fields ----
    print("\n# MemoryEntry fields")
    check("MemoryEntry has session_id", hasattr(MemoryEntry(), "session_id"))
    check("MemoryEntry has turn_id", hasattr(MemoryEntry(), "turn_id"))
    check("MemoryEntry has has_answer", hasattr(MemoryEntry(), "has_answer"))

    # ---- 3. Normal case: every fact mapped to its own source turn ----
    print("\n# normal extraction -> correct session / turn / has_answer attribution")
    res = [
        {"cleaned_result": [{"source_id": 0, "fact": "f0"}, {"source_id": 1, "fact": "f1"}]},
        {"cleaned_result": [{"source_id": 2, "fact": "f2"}, {"source_id": 3, "fact": "f3"}]},
    ]
    entries = convert_extraction_results_to_memory_entries(
        extracted_results=res, timestamps_list=timestamps_list, weekday_list=weekday_list,
        speaker_list=speaker_list, session_list=session_list,
        turn_id_list=turn_id_list, has_answer_list=has_answer_list,
        topic_id_map=topic_id_map, max_source_ids=max_source_ids, logger=logger,
    )
    by_fact = {e.memory: e for e in entries}
    check("all 4 facts produced", len(entries) == 4, f"got {len(entries)}")
    check("each fact -> right session",
          {k: v.session_id for k, v in by_fact.items()} ==
          {"f0": "session_0", "f1": "session_0", "f2": "session_1", "f3": "session_1"},
          f"got {{k: v.session_id ...}}")
    # source_id 0->seq0(t0), 1->seq2(t2), 2->seq4(t4), 3->seq6(t6)
    check("each fact -> right source turn_id",
          {k: v.turn_id for k, v in by_fact.items()} ==
          {"f0": "t0", "f1": "t2", "f2": "t4", "f3": "t6"},
          f"got {{k: v.turn_id ...}}")
    # only the first user turn of each session is the evidence turn
    check("has_answer flags the ground-truth-turn facts only",
          {k: v.has_answer for k, v in by_fact.items()} ==
          {"f0": True, "f1": False, "f2": True, "f3": False},
          f"got {{k: v.has_answer ...}}")

    # ---- 4. THE BUG FIX: an empty leading batch must not mis-clamp ----
    # batch 0 yields nothing (None); batch 1's facts have source_ids 2,3.
    # Buggy code indexed max_source_ids by the *filtered* position (0 -> max 1),
    # clamping source_ids 2,3 down to 1 and mis-attributing them to session_0.
    print("\n# empty leading batch -> no mis-clamp (the bug fix)")
    res_bug = [
        None,
        {"cleaned_result": [{"source_id": 2, "fact": "f2"}, {"source_id": 3, "fact": "f3"}]},
    ]
    entries_bug = convert_extraction_results_to_memory_entries(
        extracted_results=res_bug, timestamps_list=timestamps_list, weekday_list=weekday_list,
        speaker_list=speaker_list, session_list=session_list,
        turn_id_list=turn_id_list, has_answer_list=has_answer_list,
        topic_id_map=topic_id_map, max_source_ids=max_source_ids, logger=logger,
    )
    got_bug = {e.memory: e.session_id for e in entries_bug}
    got_bug_turn = {e.memory: e.turn_id for e in entries_bug}
    check("surviving batch produced 2 facts", len(entries_bug) == 2, f"got {len(entries_bug)}")
    check("source_ids NOT clamped -> still session_1",
          got_bug == {"f2": "session_1", "f3": "session_1"},
          f"got {got_bug}  (buggy code would give session_0)")
    check("source_ids NOT clamped -> right turn_id",
          got_bug_turn == {"f2": "t4", "f3": "t6"},
          f"got {got_bug_turn}  (buggy code would give t2)")

    print()
    if _failures:
        print(f"RESULT: FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(run())
