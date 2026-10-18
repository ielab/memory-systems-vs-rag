#!/usr/bin/env python3
"""
dataset_stats.py
================
Compute corpus statistics for a LongMemEval JSON release (haystack size, turn
and session distributions, token counts, ``has_answer`` evidence density, gold
annotation coverage) and print them as JSON.

Every LongMemEval statistic quoted in this repo comes from this script.

Usage
-----
    python -m evalgrid.tools.dataset_stats                       # both local files
    python -m evalgrid.tools.dataset_stats --dataset longmemeval_s
    python -m evalgrid.tools.dataset_stats --path some/other.json

Tokens are counted with tiktoken's ``cl100k_base``, which is a proxy: the
generator/embedder tokenisers used by the pipeline will differ by a few percent.

Output
------
A JSON list, one object per dataset analysed. ``--summary`` prints a short
human-readable digest instead.
"""

import argparse
import collections
import datetime as dt
import json
import statistics as st
import sys

from evalgrid.benchmarks.adapters import DATASETS


def _percentiles(vals):
    """min/p5/p25/median/mean/p75/p95/max/total for a list of numbers."""
    v = sorted(vals)
    n = len(v)
    if not n:
        return None
    at = lambda f: v[min(n - 1, int(f * n))]
    return dict(n=n, min=v[0], p5=at(.05), p25=at(.25), median=at(.5),
                mean=round(sum(v) / n, 1), p75=at(.75), p95=at(.95),
                max=v[-1], total=sum(v))


def _parse_date(s):
    """'2023/04/10 (Mon) 17:50' -> datetime. The weekday is redundant."""
    head, _, tail = s.partition(" (")
    return dt.datetime.strptime(head + tail.partition(")")[2], "%Y/%m/%d %H:%M")


def analyse(path, name, encoding="cl100k_base", threads=8):
    import tiktoken
    enc = tiktoken.get_encoding(encoding)

    with open(path) as f:
        data = json.load(f)
    out = {"name": name, "path": path, "n_instances": len(data),
           "tokeniser": encoding}

    # Tokenise every turn once, keeping (sample, session, turn) coordinates so
    # the counts can be regrouped without re-encoding.
    flat, coord = [], []
    for i, x in enumerate(data):
        for j, sess in enumerate(x["haystack_sessions"]):
            for k, turn in enumerate(sess):
                flat.append(turn.get("content") or "")
                coord.append((i, j, k))
    counts = [len(t) for t in enc.encode_ordinary_batch(flat, num_threads=threads)]
    print(f"[{name}] tokenised {len(flat)} turns", file=sys.stderr)

    per_sample = collections.Counter()
    per_session = collections.Counter()
    role_turns = collections.Counter()
    role_tokens = collections.Counter()
    ev_tokens, ev_roles = [], collections.Counter()
    ev_per_sample = collections.Counter()
    for (i, j, k), n in zip(coord, counts):
        turn = data[i]["haystack_sessions"][j][k]
        per_sample[i] += n
        per_session[(i, j)] += n
        role_turns[turn["role"]] += 1
        role_tokens[turn["role"]] += n
        if turn.get("has_answer"):
            ev_tokens.append(n)
            ev_roles[turn["role"]] += 1
            ev_per_sample[i] += 1

    sample_tokens = [per_sample[i] for i in range(len(data))]
    ev_counts = [ev_per_sample[i] for i in range(len(data))]

    out["chars_total"] = sum(len(c) for c in flat)
    out["tokens_total"] = sum(counts)
    out["tokens_per_sample"] = _percentiles(sample_tokens)
    out["tokens_per_session"] = _percentiles(list(per_session.values()))
    out["tokens_per_turn"] = _percentiles(counts)
    out["sessions_per_sample"] = _percentiles([len(x["haystack_sessions"]) for x in data])
    out["turns_per_sample"] = _percentiles([sum(len(s) for s in x["haystack_sessions"]) for x in data])
    out["turns_per_session"] = _percentiles([len(s) for x in data for s in x["haystack_sessions"]])
    out["role_turns"] = dict(role_turns)
    out["role_tokens"] = dict(role_tokens)
    out["question_tokens"] = _percentiles([len(t) for t in enc.encode_ordinary_batch(
        [x["question"] for x in data], num_threads=threads)])
    out["answer_tokens"] = _percentiles([len(t) for t in enc.encode_ordinary_batch(
        [str(x["answer"]) for x in data], num_threads=threads)])

    # Evidence (has_answer) density.
    out["evidence_turns_per_sample"] = _percentiles(ev_counts)
    out["evidence_tokens"] = _percentiles(ev_tokens)
    out["evidence_roles"] = dict(ev_roles)
    out["needle_density_pct"] = round(100 * sum(ev_tokens) / sum(counts), 3)

    # Gold-annotation coverage: does every declared answer session actually
    # contain a has_answer turn? (It often does not.)
    empty_gold, partial_gold = [], []
    for i, x in enumerate(data):
        flagged = {sid for sid, sess in zip(x["haystack_session_ids"], x["haystack_sessions"])
                   if any(t.get("has_answer") for t in sess)}
        declared = set(x["answer_session_ids"])
        if not ev_counts[i]:
            empty_gold.append(x["question_id"])
        elif flagged != declared:
            partial_gold.append(x["question_id"])
    out["gold_coverage"] = {
        "empty_gold": len(empty_gold),
        "empty_gold_all_abstention": all(q.endswith("_abs") for q in empty_gold),
        "partial_gold": len(partial_gold),
        "partial_gold_non_abstention": sum(1 for q in partial_gold if not q.endswith("_abs")),
        "partial_gold_ids": sorted(partial_gold),
    }
    out["gold_sessions_per_sample"] = _percentiles([len(x["answer_session_ids"]) for x in data])

    # Question mix.
    out["types"] = dict(collections.Counter(x["question_type"] for x in data))
    out["n_abstention"] = sum(1 for x in data if x["question_id"].endswith("_abs"))
    by_type = collections.defaultdict(lambda: {"tok": [], "sess": [], "ev": []})
    for i, x in enumerate(data):
        b = by_type[x["question_type"]]
        b["tok"].append(sample_tokens[i])
        b["sess"].append(len(x["haystack_sessions"]))
        b["ev"].append(ev_counts[i])
    out["by_type"] = {k: {"n": len(v["tok"]),
                          "mean_tokens": round(st.mean(v["tok"])),
                          "median_tokens": int(st.median(v["tok"])),
                          "mean_sessions": round(st.mean(v["sess"]), 1),
                          "mean_evidence_turns": round(st.mean(v["ev"]), 2),
                          "zero_evidence": sum(1 for e in v["ev"] if e == 0)}
                      for k, v in sorted(by_type.items())}

    # Temporal structure.
    spans, lag, positions = [], [], []
    for x in data:
        dates = sorted(_parse_date(d) for d in x["haystack_dates"])
        spans.append((dates[-1] - dates[0]).days)
        if x.get("question_date"):
            lag.append((_parse_date(x["question_date"]) - dates[-1]).days)
        ids = x["haystack_session_ids"]
        for a in x["answer_session_ids"]:
            if a in ids and len(ids) > 1:
                positions.append(ids.index(a) / (len(ids) - 1))
    out["haystack_span_days"] = _percentiles(spans)
    out["days_question_after_last_session"] = _percentiles(lag)
    out["evidence_relative_position"] = _percentiles(positions)
    return out


def digest(o):
    ts, sp, ev = o["tokens_per_sample"], o["sessions_per_sample"], o["evidence_turns_per_sample"]
    g = o["gold_coverage"]
    return "\n".join([
        f"{o['name']}  ({o['n_instances']} instances, {o['tokens_total'] / 1e6:.2f}M tokens)",
        f"  tokens/sample   min {ts['min']:,}  median {ts['median']:,}  "
        f"mean {ts['mean']:,.0f}  max {ts['max']:,}",
        f"  sessions/sample min {sp['min']}  median {sp['median']}  max {sp['max']}",
        f"  evidence turns  median {ev['median']}  max {ev['max']}  "
        f"total {ev['total']}  ({o['needle_density_pct']}% of haystack tokens)",
        f"  gold coverage   {g['empty_gold']} empty, {g['partial_gold']} partial "
        f"({g['partial_gold_non_abstention']} non-abstention)",
    ])


def main():
    parser = argparse.ArgumentParser(description="LongMemEval corpus statistics")
    parser.add_argument("--dataset", choices=list(DATASETS), action="append",
                        help="Registered dataset key (repeatable). Default: all that exist on disk.")
    parser.add_argument("--path", action="append",
                        help="Analyse this JSON file directly (repeatable)")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    parser.add_argument("--summary", action="store_true", help="Print a digest instead of full JSON")
    parser.add_argument("--out", default=None, help="Write JSON here instead of stdout")
    args = parser.parse_args()

    targets = []
    if args.path:
        targets += [(p, p) for p in args.path]
    for key in args.dataset or []:
        targets.append((DATASETS[key], key))
    if not targets:
        import os
        # Only the LongMemEval releases share this schema; locomo does not.
        targets = [(p, k) for k, p in DATASETS.items()
                   if k.startswith("longmemeval") and os.path.isfile(p)]
        if not targets:
            sys.exit("No LongMemEval file found on disk — see the Data section of the README.")

    results = [analyse(p, name, args.encoding) for p, name in targets]
    if args.summary:
        print("\n\n".join(digest(r) for r in results))
        return
    text = json.dumps(results, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
