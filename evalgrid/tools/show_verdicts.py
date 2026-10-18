"""
evalgrid/tools/show_verdicts.py
-------------------------------
Read a ``judge_verdicts.<judge-slug>.jsonl`` — the judge's own words behind
each score — and print it in the shape a person investigating a cell needs.

The metrics file says a question scored 0.06. This says which rubric criterion
the judge failed the answer on and why it said so, and on ``event_ordering``
which alignment pairs came back NO. That second family is the one worth
looking at first when a question's rubric verdicts are all 1.0 and its
``final_score`` is near zero: ``final_score = tau_norm * f1``, and both terms
come out of the equivalence calls, not the graded ones.

Usage
-----
    # the questions that scored worst, with every call behind them
    python -m evalgrid.tools.show_verdicts <verdicts.jsonl> --worst 5

    # one question by id
    python -m evalgrid.tools.show_verdicts <verdicts.jsonl> --qid 100K_1_event_ordering_1

    # where the graded judge and the alignment classifier DISAGREE — a question
    # whose criteria all passed but whose alignment matched almost nothing
    python -m evalgrid.tools.show_verdicts <verdicts.jsonl> --disagreements

Scores come from the metrics file beside the log when one is there, so --worst
and --disagreements can rank; without it the log still prints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import textwrap


def load(path: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["question_id"]] = rec["calls"]
    return out


def load_scores(verdicts_path: str) -> tuple[dict[str, float], dict[str, dict]]:
    """(question_id -> score, question_id -> event_ordering breakdown).

    Found by rewriting judge_verdicts.<slug>.jsonl to metrics.<slug>.json in
    the same directory — the two are written by the same run, so the slug that
    names one names the other.
    """
    m = re.match(r"judge_verdicts\.(.+)\.jsonl$", os.path.basename(verdicts_path))
    if not m:
        return {}, {}
    metrics = os.path.join(os.path.dirname(verdicts_path), f"metrics.{m.group(1)}.json")
    if not os.path.isfile(metrics):
        return {}, {}
    with open(metrics) as f:
        d = json.load(f)
    scores = {q["question_id"]: q["score"] for q in d.get("per_question", [])
              if q.get("score") is not None}
    return scores, (d.get("beam") or {}).get("event_ordering", {})


def wrap(text: str, indent: str = " " * 8) -> str:
    return textwrap.fill(str(text), width=100, initial_indent=indent,
                         subsequent_indent=indent) if text else indent + "(none)"


def show(qid: str, calls: list[dict], scores, ordering) -> None:
    head = f"{qid}"
    if qid in scores:
        head += f"   score={scores[qid]:.4f}"
    if qid in ordering:
        o = ordering[qid]
        head += (f"   [judge={o['llm_judge_score']:.2f} f1={o['f1']:.2f} "
                 f"tau_norm={o['tau_norm']:.2f} final={o['final_score']:.2f}]")
    print("=" * 100)
    print(head)
    for c in calls:
        if c["kind"] == "rubric":
            mark = "FAIL" if c.get("error") else f"{c['score']:.1f} "
            print(f"  [{mark}] rubric[{c['rubric_index']}] {c['criterion']}")
            print(wrap(c.get("error") or c.get("reason")))
        elif c["kind"] == "equivalence":
            mark = "YES" if c["same"] else "no "
            print(f"  {mark}  rubric[{c['rubric_index']}] {c['criterion']!r}")
            print(f"        <- {c['line'].strip()!r}")
        else:                                   # a binary judge's single call
            print(f"  [{c.get('score')}] {c.get('reply', '')!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verdicts", help="path to judge_verdicts.<judge-slug>.jsonl")
    p.add_argument("--qid", action="append", help="show only these question ids (repeatable)")
    p.add_argument("--worst", type=int, metavar="N",
                   help="show the N lowest-scoring questions (needs the metrics file)")
    p.add_argument("--disagreements", action="store_true",
                   help="event_ordering questions whose rubric verdicts all passed but "
                        "whose alignment f1 is below --f1-max — the graded judge and the "
                        "equivalence classifier reading the same answer differently")
    p.add_argument("--f1-max", type=float, default=0.5,
                   help="the f1 below which --disagreements calls it a disagreement "
                        "(default: 0.5)")
    args = p.parse_args()

    calls = load(args.verdicts)
    scores, ordering = load_scores(args.verdicts)

    if args.qid:
        qids = [q for q in args.qid if q in calls]
        missing = [q for q in args.qid if q not in calls]
        if missing:
            print(f"!!! not in this log: {', '.join(missing)}")
    elif args.disagreements:
        qids = [q for q, o in ordering.items()
                if o["llm_judge_score"] >= 1.0 and o["f1"] < args.f1_max]
        print(f"{len(qids)} event_ordering question(s) where every rubric criterion "
              f"passed but alignment f1 < {args.f1_max}")
    elif args.worst is not None:
        if not scores:
            p.error("--worst needs the metrics file beside the log to rank by")
        qids = sorted(scores, key=lambda q: scores[q])[:args.worst]
    else:
        qids = list(calls)

    for qid in qids:
        show(qid, calls[qid], scores, ordering)


if __name__ == "__main__":
    main()
