#!/usr/bin/env python3
"""
beam_to_jsonl.py
================
Dump a BEAM tier as JSONL so it can be read, grepped and ``jq``-ed.

The parquet is awkward to inspect for the same reason PersonaMem's CSV is
(see personamem_to_jsonl.py): it holds Python *repr* strings rather than data.
``probing_questions`` is a stringified dict — sometimes a repr OF a repr — and
each question's ``rubric`` is a stringified list one level below that. Nothing
short of parsing it twice shows you the questions.

THIS DOES NOT REPLACE THE PARQUET, and nothing in the pipeline reads what it
writes. ``data/`` is gitignored and the parquet is not tracked, so the release
stays byte-reproducible from ``hf download``;
a converted copy would quietly become a second source of truth that no longer
matches upstream. The adapter reads the parquet directly in about a second.
This is a reading tool.

TWO VIEWS, and the default is the one that answers "is the harness seeing what
I think it is":

  adapter (default)  what evalgrid/benchmarks/adapters/beam.py produces — the
                     ``->->`` marker stripped, session ids and turn ids in the
                     canonical ``<session_id>:t<i>`` namespace, evidence
                     resolved into that namespace with dangling ids dropped,
                     rubric parsed, gold answer picked from whichever of the
                     three field names the ability uses.
  --raw              the parquet as released, one row per conversation, with
                     only the repr-strings parsed. Use it to check the adapter
                     against upstream rather than to check upstream.

GRANULARITY IS THE POINT. A conversation is ~700 KB, so a conversation-per-line
file is 20 lines of unreadable width and no easier than the parquet — which is
why ``--unit`` defaults to ``question``.

  --unit question      400 lines: the question, ability, gold answer, rubric,
                       and its evidence turn ids. The view for checking gold.
  --unit turn        5,732 lines: one utterance each, with its turn id, role,
                       session and has_answer flag. The view for grepping text.
  --unit conversation   20 lines: whole Samples, nested. The faithful shape,
                       and the one you want only for ``jq`` on structure.

Usage
-----
    python -m evalgrid.tools.beam_to_jsonl                       # 100K, questions
    python -m evalgrid.tools.beam_to_jsonl --unit turn
    python -m evalgrid.tools.beam_to_jsonl --with-evidence       # inline the evidence text
    python -m evalgrid.tools.beam_to_jsonl --dataset beam_1m --unit question
    python -m evalgrid.tools.beam_to_jsonl --raw --out /tmp/beam_raw.jsonl

Then, e.g.:
    jq -r 'select(.ability=="abstention") | .question' beam_100k.questions.jsonl | head
    jq -r 'select(.gold_answer==null) | .ability' beam_100k.questions.jsonl | sort | uniq -c
    jq -r 'select(.n_evidence==0) | [.query_id,.ability] | @tsv' beam_100k.questions.jsonl
    jq -r 'select(.role=="user" and .has_answer) | .content' beam_100k.turns.jsonl | head
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Beside the dataset these views are derived from, not under ``outputs/``:
#: ``outputs/`` is for what a RUN produced and is keyed by
#: system/data/mem-model, while these are a reading of the release itself and
#: belong with it. ``data/`` is gitignored, so they stay untracked exactly as
#: the parquet does.
#:
#: Deliberately NOT ``data/BEAM/data/``, which is what
#: ``hf download --local-dir data/BEAM`` populates and what its
#: ``.cache/huggingface`` tracks. Keeping derived files out of that directory is
#: what preserves this property: the release stays
#: byte-comparable with upstream and can be re-fetched without first working out
#: which files are ours.
#:
#: One directory for all three tiers — the filenames carry the tier.
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "data", "BEAM", "inspect")


def _samples(dataset, limit):
    from evalgrid.benchmarks.adapters import build_adapter

    out = []
    for i, sample in enumerate(build_adapter(dataset).samples()):
        if limit is not None and i >= limit:
            break
        out.append(sample)
    return out


def rows_question(samples, with_evidence):
    """One line per question — the view for checking gold.

    ``evidence_turn_ids`` is what gold.py will actually intersect against a
    unit's provenance, already resolved and already dropped of dangling ids, so
    ``n_evidence == 0`` here is exactly the set of questions no retrieval can
    ever score: the 40 abstention questions by design, plus a handful whose ids
    resolve to nothing.
    """
    for sample in samples:
        text = {t.turn_id: t for s in sample.sessions for t in s.turns}
        for q in sample.queries:
            row = {
                "query_id": q.query_id,
                "sample_id": sample.sample_id,
                "ability": q.question_type,
                "question": q.question,
                # None, not "", where the ability ships only a rubric — 120 of
                # the 100K tier's 400. The distinction is the whole reason
                # BEAM needs its own judge, so it must survive the dump.
                "gold_answer": q.answer,
                "rubric": q.extra.get("rubric") or [],
                "n_rubric_items": len(q.extra.get("rubric") or []),
                "difficulty": q.extra.get("difficulty"),
                "evidence_turn_ids": q.evidence_turn_ids,
                "n_evidence": len(q.evidence_turn_ids),
            }
            if with_evidence:
                row["evidence"] = [
                    {"turn_id": tid, "role": text[tid].role, "content": text[tid].content}
                    for tid in q.evidence_turn_ids if tid in text
                ]
            yield row


def rows_turn(samples):
    """One line per utterance — the view for grepping text.

    ``has_answer`` is the SAMPLE-level union over the conversation's 20
    questions, which is all a per-turn flag can mean on BEAM. It is not any one
    question's gold; for that, join on ``evidence_turn_ids`` in the question
    view. See adapters/beam.py.
    """
    for sample in samples:
        for session in sample.sessions:
            for i, t in enumerate(session.turns):
                yield {
                    "turn_id": t.turn_id,
                    "sample_id": sample.sample_id,
                    "session_id": session.session_id,
                    "timestamp": session.timestamp,
                    "index_in_session": i,
                    "role": t.role,
                    "content": t.content,
                    "has_answer_any_question": t.has_answer,
                    "beam_message_id": (t.extra or {}).get("beam_message_id"),
                }


def rows_conversation(samples):
    """One line per Sample — the faithful nested shape."""
    for sample in samples:
        yield {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "evidence_is_per_query": sample.evidence_is_per_query,
            "sessions": [
                {"session_id": s.session_id, "timestamp": s.timestamp,
                 "turns": [{"turn_id": t.turn_id, "role": t.role, "content": t.content,
                            "has_answer_any_question": t.has_answer} for t in s.turns]}
                for s in sample.sessions],
            "queries": [
                {"query_id": q.query_id, "ability": q.question_type, "question": q.question,
                 "gold_answer": q.answer, "rubric": q.extra.get("rubric") or [],
                 "evidence_turn_ids": q.evidence_turn_ids}
                for q in sample.queries],
        }


def rows_raw(dataset, limit):
    """The parquet as released, repr-strings parsed and nothing else touched.

    For checking the ADAPTER against upstream — markers still attached, message
    ids still BEAM's own, ``source_chat_ids`` still in whichever of its three
    shapes the ability uses.
    """
    import pandas as pd

    from evalgrid.benchmarks.adapters import DATASETS
    from evalgrid.benchmarks.adapters.beam import _literal

    df = pd.read_parquet(DATASETS[dataset])
    for i, (_, r) in enumerate(df.iterrows()):
        if limit is not None and i >= limit:
            break
        row = {}
        for col in df.columns:
            v = r[col]
            if col in ("probing_questions", "chat", "user_questions"):
                v = _literal(v)
            if hasattr(v, "tolist"):
                v = v.tolist()
            row[col] = v
        yield row



# ---------------------------------------------------------------------------
# Output formats. JSONL is the default and the only lossless one; the other two
# exist because reading 400 questions is a different job from querying them, and
# `jq` is not always installed.
# ---------------------------------------------------------------------------

#: Order the flat views put their columns in. Not alphabetical: identity first,
#: then the question, then what it is scored against, then its evidence — which
#: is the order you read a row in when you are checking one.
_COLUMNS = {
    "question": ["query_id", "sample_id", "ability", "difficulty", "question",
                 "gold_answer", "n_rubric_items", "rubric", "n_evidence",
                 "evidence_turn_ids"],
    "turn": ["turn_id", "sample_id", "session_id", "timestamp", "index_in_session",
             "role", "has_answer_any_question", "beam_message_id", "content"],
}


def _flat(v):
    """A cell value for CSV: lists joined, None kept distinguishable from ""."""
    if v is None:
        # "" would be indistinguishable from a gold answer that is genuinely
        # empty, and the whole point of the gold_answer column is that 120 of
        # the 400 questions HAVE no gold answer. Spell it.
        return "(none)"
    if isinstance(v, (list, tuple)):
        return " | ".join(str(x) for x in v)
    return str(v)


def write_jsonl(rows, out):
    n = 0
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def write_csv(rows, out, unit):
    """One row per record, lists joined with " | ". Opens in any spreadsheet.

    Only the flat units get a stable column set; a conversation is nested and
    has no honest CSV shape, so it is refused rather than flattened into
    something that looks tabular and is not.
    """
    import csv as _csv

    cols = _COLUMNS.get(unit)
    if cols is None:
        sys.exit(f"--format csv needs a flat unit (question|turn), not {unit!r}. "
                 f"A conversation is nested; use --format jsonl for it.")
    n = 0
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([_flat(row.get(c)) for c in cols])
            n += 1
    return n


def write_txt(rows, out, unit):
    """A readable block per record — for reading, not querying.

    Wrapped at 88 columns so it is legible in an editor beside code, and each
    block opens with the id so a grep hit can be traced back.
    """
    import textwrap

    def para(label, text, indent=2):
        body = textwrap.fill(str(text), width=88,
                             initial_indent=" " * indent, subsequent_indent=" " * indent)
        return f"{label}\n{body}" if label else body

    n = 0
    with open(out, "w") as f:
        for row in rows:
            if unit == "question":
                f.write(f"{'=' * 88}\n")
                f.write(f"{row['query_id']}   [{row['ability']}]"
                        f"   difficulty={row.get('difficulty')}\n")
                f.write(f"{'=' * 88}\n")
                f.write(para("QUESTION:", row["question"]) + "\n\n")
                gold = row.get("gold_answer")
                f.write(para("GOLD ANSWER:",
                             gold if gold is not None
                             else "(none — this ability is scored against the rubric only)")
                        + "\n\n")
                if row.get("rubric"):
                    f.write("RUBRIC:\n")
                    for i, item in enumerate(row["rubric"], 1):
                        f.write(para("", f"{i}. {item}", indent=2) + "\n")
                    f.write("\n")
                ids = row.get("evidence_turn_ids") or []
                f.write(f"EVIDENCE ({len(ids)} turn(s)): "
                        f"{', '.join(ids) if ids else '(none)'}\n")
                for ev in row.get("evidence") or []:
                    f.write(para(f"  [{ev['turn_id']}] {ev['role']}:",
                                 ev["content"], indent=4) + "\n")
                f.write("\n")
            elif unit == "turn":
                flag = "  *EVIDENCE*" if row.get("has_answer_any_question") else ""
                f.write(f"--- {row['turn_id']}  ({row['sample_id']}, "
                        f"{row['timestamp']}) {row['role']}{flag}\n")
                f.write(para("", row["content"]) + "\n\n")
            else:
                sys.exit("--format txt needs --unit question or turn.")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="beam_100k",
                    choices=["beam_100k", "beam_500k", "beam_1m"])
    ap.add_argument("--unit", default="question",
                    choices=["question", "turn", "conversation"],
                    help="one JSON object per what (default: question)")
    ap.add_argument("--raw", action="store_true",
                    help="dump the parquet as released instead of the adapter's view")
    ap.add_argument("--with-evidence", action="store_true",
                    help="--unit question: inline each evidence turn's text")
    ap.add_argument("--format", default="jsonl", choices=["jsonl", "csv", "txt"],
                    help="jsonl (default, lossless, for jq/python) | csv (opens in a "
                         "spreadsheet) | txt (a readable block per record)")
    ap.add_argument("--out",
                    help="default: data/BEAM/inspect/<dataset>.<unit>.<format>")
    ap.add_argument("--limit", type=int, help="only the first N conversations")
    a = ap.parse_args()

    if a.raw:
        rows = rows_raw(a.dataset, a.limit)
        unit = "raw"
    else:
        samples = _samples(a.dataset, a.limit)
        unit = a.unit
        if a.unit == "question":
            rows = rows_question(samples, a.with_evidence)
        elif a.unit == "turn":
            rows = rows_turn(samples)
        else:
            rows = rows_conversation(samples)

    ext = a.format
    out = a.out or os.path.join(DEFAULT_OUT_DIR, f"{a.dataset}.{unit}.{ext}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if a.format == "csv":
        n = write_csv(rows, out, unit)
    elif a.format == "txt":
        n = write_txt(rows, out, unit)
    else:
        n = write_jsonl(rows, out)

    size = os.path.getsize(out)
    print(f"{a.dataset}  [{unit}, {a.format}]\n  -> {out}")
    print(f"  lines written  {n:,}")
    print(f"  size           {size / 1e6:.1f} MB")
    if not a.raw and a.unit == "question" and a.format == "jsonl":
        # python, not jq: jq is not installed everywhere and this needs no deps.
        print(f"\n  python -c \"import json,collections as c; print(c.Counter("
              f"r['ability'] for r in map(json.loads, open('{out}')) "
              f"if r['gold_answer'] is None))\"")


if __name__ == "__main__":
    main()
