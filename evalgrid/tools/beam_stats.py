#!/usr/bin/env python3
"""
beam_stats.py
=============
Compute corpus statistics and run the shortcut audit for the BEAM release in
``data/BEAM/`` (per-tier conversation sizes, speaker balance, evidence
resolution and speaker location, needle density, probing-question schema
variance, generation artefacts) and print them as JSON.

This is the reproducer for our development notes — every number in that
document comes from this script.

Usage
-----
    python -m evalgrid.tools.beam_stats --summary       # digest
    python -m evalgrid.tools.beam_stats --out stats.json
    python -m evalgrid.tools.beam_stats --tiers 100K    # one tier (fast)

Schema note
-----------
One parquet row is one CONVERSATION: ``chat`` is a list of sessions, each a list
of ``{id, role, content, index, question_type, time_anchor}`` messages, and
``probing_questions`` is a *Python-repr string* holding a dict keyed by memory
ability. Each ability holds 2 questions per conversation. Gold evidence is
``source_chat_ids`` — message ``id``s, sometimes a flat list and sometimes a
dict of named lists — which is what makes turn-level qrels possible here.

The gold ANSWER field is not uniform: ``answer`` / ``ideal_answer`` /
``ideal_response`` depending on ability, and three abilities carry none at all,
only a ``rubric`` list. See ``gold_answer_field`` in the output.
"""

import argparse
import ast
import collections
import glob
import json
import os
import re
import statistics
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(_REPO_ROOT, "data", "BEAM")

#: Tier name -> the folder name upstream uses. The paper calls the first tier
#: "128K"; the release folder calls it "100K", and it measures ~120 K. Keep the
#: upstream name so a path is always reproducible from this file.
TIERS = ("100K", "500K", "1M")

#: Trailing pipeline marker leaked into the released message text, e.g.
#: "...meet my deadlines by then? ->-> 1,1" — the message's own `index` field.
#:
#: Kept identical to adapters/beam.py's `_MARKER`, and wider than the `[\d,]+`
#: beam.md §7 quotes: "->-> 2,N/A", "->-> 5,14)" and "->-> 2,22, 24" also occur
#: and are the same artefact. The percentages below therefore run very slightly
#: HIGHER than the ones in that document, which were measured with the narrow
#: form — 8 user turns of 2,866 at the 100K tier.
MARKER = re.compile(r"\s*->->\s*(?:\d+|N/A)(?:\s*,\s*(?:\d+|N/A))*\s*\)?\s*$")

#: A message is "markdown-structured" if it has >=3 heading/bullet/bold starts.
#: Separates short plain user turns from long formatted assistant expositions.
MARKDOWN = re.compile(r"^\s*(#{1,4} |\*\*|[-*] |\d+\. )", re.M)

#: The names BEAM uses for the gold answer, narrowest first. Three abilities
#: (instruction_following, preference_following, summarization) have NONE of
#: these and ship only a `rubric`, which is why an adapter cannot assume one.
ANSWER_FIELDS = ("answer", "ideal_answer", "ideal_response")


def _enc():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def load(root, tier):
    """Read one tier's parquet into a DataFrame."""
    import pandas as pd

    hits = glob.glob(os.path.join(root, "data", f"{tier}-*.parquet"))
    if not hits:
        sys.exit(f"No parquet for tier {tier!r} under {root}/data/ — see "
                 f"the README's Data section for the `hf download` command.")
    df = pd.concat([pd.read_parquet(h) for h in sorted(hits)], ignore_index=True)
    for col in ("chat", "probing_questions", "conversation_id"):
        if col not in df.columns:
            sys.exit(f"{hits[0]} is missing column {col!r} — the upstream schema "
                     f"changed.")
    return df


def probing(row):
    """Parse the probing_questions repr-string into {ability: [question, ...]}.

    Double-decoded on purpose: some rows store the dict as a repr *of a repr*,
    so one literal_eval returns a str rather than a dict.
    """
    d = row["probing_questions"]
    if isinstance(d, str):
        d = ast.literal_eval(d)
    if isinstance(d, str):
        d = ast.literal_eval(d)
    return d


def evidence_ids(item):
    """Flatten `source_chat_ids`, which is a list for some abilities and a dict
    of named lists (e.g. contradiction_resolution's first/second statement) for
    others. Missing on abstention by design, and on a few other rows by
    omission — both come back as an empty list."""
    sc = item.get("source_chat_ids")
    if isinstance(sc, dict):
        out = []
        for v in sc.values():
            out += v if isinstance(v, list) else [v]
        return out
    if isinstance(sc, list):
        return sc
    return []


def tier_stats(df, enc):
    msg_tokens, sess_tokens, conv_tokens = {}, [], []
    n_user = n_asst = tok_user = tok_asst = 0
    user_md = asst_md = user_mark = asst_mark = 0
    user_lens, asst_lens = [], []
    sessions_per_conv, msgs_per_conv = [], []

    for _, r in df.iterrows():
        cid = r["conversation_id"]
        total = 0
        sessions_per_conv.append(len(r["chat"]))
        msgs_per_conv.append(sum(len(s) for s in r["chat"]))
        for si, s in enumerate(r["chat"]):
            st_tok = 0
            for x in s:
                n = len(enc.encode(x["content"]))
                msg_tokens[(cid, str(x["id"]))] = (n, x["role"], si)
                st_tok += n
                is_md = len(MARKDOWN.findall(x["content"])) >= 3
                has_mark = bool(MARKER.search(x["content"]))
                if x["role"] == "user":
                    n_user += 1; tok_user += n; user_lens.append(n)
                    user_md += is_md; user_mark += has_mark
                else:
                    n_asst += 1; tok_asst += n; asst_lens.append(n)
                    asst_md += is_md; asst_mark += has_mark
            sess_tokens.append(st_tok)
            total += st_tok
        conv_tokens.append(total)

    corpus = tok_user + tok_asst

    # --- questions, evidence, schema
    n_q = collections.Counter()
    key_variants = collections.defaultdict(collections.Counter)
    answer_field = collections.Counter()
    with_ev = 0
    resolved = unresolved = 0
    ev_role = collections.Counter()
    ev_tok = collections.Counter()
    ev_role_by_ability = collections.defaultdict(collections.Counter)
    ev_per_q, densities, ev_sessions = [], [], []
    q_strings = collections.Counter()

    for _, r in df.iterrows():
        cid = r["conversation_id"]
        conv_total = sum(t for (c, _), (t, _, _) in msg_tokens.items() if c == cid) or 1
        for ability, items in probing(r).items():
            for it in items:
                n_q[ability] += 1
                key_variants[ability][tuple(sorted(it.keys()))] += 1
                answer_field[next((k for k in ANSWER_FIELDS if k in it), "NONE")] += 1
                q_strings[it.get("question", "")] += 1

                ids = evidence_ids(it)
                if ids:
                    with_ev += 1
                found, sess = [], set()
                for i in ids:
                    hit = msg_tokens.get((cid, str(i)))
                    if hit is None:
                        unresolved += 1
                        continue
                    resolved += 1
                    tok, role, si = hit
                    found.append(tok)
                    sess.add(si)
                    ev_role[role] += 1
                    ev_tok[role] += tok
                    ev_role_by_ability[ability][role] += 1
                if found:
                    ev_per_q.append(len(found))
                    densities.append(sum(found) / conv_total)
                    ev_sessions.append(len(sess))

    n_ev = sum(ev_role.values()) or 1
    q_total = sum(n_q.values()) or 1

    def dist(xs):
        return {"min": min(xs), "median": statistics.median(xs), "max": max(xs)}

    return {
        "conversations": len(df),
        "questions": q_total,
        "questions_per_conversation": q_total // max(1, len(df)),
        "questions_per_ability": dict(sorted(n_q.items())),
        "sessions_per_conversation": dist(sessions_per_conv),
        "messages_per_conversation": dist(msgs_per_conv),
        "tokens_per_conversation": dist(conv_tokens),
        "tokens_per_session": dist(sess_tokens),
        "sessions_over_32k_tokens": f"{sum(1 for t in sess_tokens if t > 32768)}/{len(sess_tokens)}",
        "corpus_tokens": corpus,
        "speakers": {
            "user_messages": n_user, "assistant_messages": n_asst,
            "strict_1to1": n_user == n_asst,
            "user_tokens": tok_user, "assistant_tokens": tok_asst,
            "user_token_share_pct": 100.0 * tok_user / max(1, corpus),
            "user_median_tokens": statistics.median(user_lens),
            "assistant_median_tokens": statistics.median(asst_lens),
            "assistant_max_tokens": max(asst_lens),
            "user_markdown_pct": 100.0 * user_md / max(1, n_user),
            "assistant_markdown_pct": 100.0 * asst_md / max(1, n_asst),
        },
        "artefacts": {
            "marker_on_user_pct": 100.0 * user_mark / max(1, n_user),
            "marker_on_assistant_pct": 100.0 * asst_mark / max(1, n_asst),
        },
        "evidence": {
            "questions_with_source_chat_ids": with_ev,
            "questions_without": q_total - with_ev,
            "ids_resolved": resolved, "ids_dangling": unresolved,
            "resolved_pct": 100.0 * resolved / max(1, resolved + unresolved),
            "messages_per_question": dist(ev_per_q),
            "sessions_per_question": dist(ev_sessions),
            "by_role": dict(ev_role),
            "user_share_of_evidence_pct": 100.0 * ev_role["user"] / n_ev,
            "tokens_by_role": dict(ev_tok),
            "pooled_density_pct": 100.0 * sum(ev_tok.values()) / max(1, corpus),
            "per_question_density_pct": {k: 100 * v for k, v in dist(densities).items()},
            "abilities_with_only_user_evidence": sorted(
                a for a in ev_role_by_ability if ev_role_by_ability[a]["assistant"] == 0),
        },
        "user_turn_shortcut": {
            "corpus_discarded_pct": 100.0 * tok_asst / max(1, corpus),
            "evidence_retained_pct": 100.0 * ev_role["user"] / n_ev,
        },
        "schema": {
            "gold_answer_field": dict(answer_field),
            "questions_with_no_gold_answer": answer_field["NONE"],
            "abilities_with_multiple_key_sets": {
                a: sorted(set().union(*[set(k) for k in c])
                          - set.intersection(*[set(k) for k in c]))
                for a, c in sorted(key_variants.items()) if len(c) > 1},
        },
        "duplicate_question_strings": sum(v - 1 for v in q_strings.values() if v > 1),
    }


def compute(root, tiers):
    enc = _enc()
    out = {"root": root, "tiers": {}}
    for t in tiers:
        out["tiers"][t] = tier_stats(load(root, t), enc)
    out["totals"] = {
        "conversations": sum(d["conversations"] for d in out["tiers"].values()),
        "questions": sum(d["questions"] for d in out["tiers"].values()),
        "corpus_tokens": sum(d["corpus_tokens"] for d in out["tiers"].values()),
    }
    out["totals"]["ingest_tokens_per_question"] = (
        out["totals"]["corpus_tokens"] / max(1, out["totals"]["questions"]))
    return out


def summary(s):
    for tier, d in s["tiers"].items():
        sp, ev, sc = d["speakers"], d["evidence"], d["schema"]
        c, t = d["tokens_per_conversation"], d["tokens_per_session"]
        print(f"BEAM-{tier}  ({d['conversations']} conversations, {d['questions']} questions, "
              f"{d['questions_per_conversation']}/chat)")
        print(f"  size        tokens/conv min {c['min']:,} med {c['median']:,.0f} max {c['max']:,}"
              f"  | corpus {d['corpus_tokens']:,}")
        print(f"              sessions/conv med {d['sessions_per_conversation']['median']:.0f} | "
              f"tokens/session med {t['median']:,.0f} | over 32k: "
              f"{d['sessions_over_32k_tokens']}")
        print(f"  speakers    user {sp['user_messages']:,} / assistant {sp['assistant_messages']:,} "
              f"(1:1 {sp['strict_1to1']}) | median tok user {sp['user_median_tokens']:.0f} "
              f"assistant {sp['assistant_median_tokens']:.0f} (max {sp['assistant_max_tokens']:,})")
        print(f"              markdown-structured: user {sp['user_markdown_pct']:.1f}% "
              f"assistant {sp['assistant_markdown_pct']:.1f}%")
        print(f"  TOKENS      user {sp['user_token_share_pct']:.1f}% / assistant "
              f"{100 - sp['user_token_share_pct']:.1f}%")
        print(f"  evidence    {ev['questions_with_source_chat_ids']}/{d['questions']} questions "
              f"have source_chat_ids | resolve {ev['ids_resolved']}/"
              f"{ev['ids_resolved'] + ev['ids_dangling']} = {ev['resolved_pct']:.1f}%")
        print(f"              by role: user {ev['user_share_of_evidence_pct']:.1f}% | "
              f"per-question density med {ev['per_question_density_pct']['median']:.3f}% | "
              f"{len(ev['abilities_with_only_user_evidence'])}/10 abilities 100% user")
        us = d["user_turn_shortcut"]
        print(f"  SHORTCUT    keeping ONLY user turns discards "
              f"{us['corpus_discarded_pct']:.1f}% of the corpus and retains "
              f"{us['evidence_retained_pct']:.1f}% of evidence")
        print(f"  artefacts   '->->' marker on {d['artefacts']['marker_on_user_pct']:.1f}% of user "
              f"msgs ({d['artefacts']['marker_on_assistant_pct']:.1f}% assistant)")
        print(f"  schema      gold answer field {sc['gold_answer_field']} | "
              f"{sc['questions_with_no_gold_answer']} questions have NO gold answer, rubric only")
        if sc["abilities_with_multiple_key_sets"]:
            print(f"              unstable keys: {sc['abilities_with_multiple_key_sets']}")
        print()
    t = s["totals"]
    print(f"TOTAL {t['conversations']} conversations, {t['questions']} questions, "
          f"{t['corpus_tokens']:,} tokens = {t['ingest_tokens_per_question']:,.0f} ingest "
          f"tokens/question")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--tiers", nargs="+", default=list(TIERS),
                    help=f"subset of {TIERS} (default: all)")
    ap.add_argument("--summary", action="store_true", help="print the digest instead of JSON")
    ap.add_argument("--out", help="write full JSON here")
    a = ap.parse_args()
    s = compute(a.root, a.tiers)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=2, default=str)
        print(f"wrote {a.out}")
    if a.summary or not a.out:
        summary(s)


if __name__ == "__main__":
    main()
