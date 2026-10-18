"""
evalgrid/benchmarks/judges/beam.py
----------------------------------
BEAM's own LLM-as-judge, reproduced from the released evaluation code.

Upstream: github.com/mohammadtavakoli78/BEAM, ``src/evaluation/compute_metrics.py``
(the ten ``evaluate_*`` functions) and ``src/prompts.py``
(``unified_llm_judge_base_prompt``, copied verbatim below).

THE PROTOCOL, and why it is not CORRECT/WRONG. BEAM does not grade an answer
against a gold string — a third of its questions have no gold string at all
. It grades against a ``rubric``: a list of
criteria the response should meet. The judge is asked about ONE criterion at a
time and answers ``{"score": 1.0 | 0.5 | 0.0}``; the question's score is the
MEAN over its rubric items, and a run's score is the mean over questions. All
ten abilities use the identical prompt and the identical averaging — the ten
``evaluate_*`` functions differ only in name.

So a BEAM item scores 0.5 when half its rubric is met, and this is the one
graded judge here (``Judge.graded``). Its metrics file says
``"metric": "mean_rubric_score"``, and a BEAM cell must not be put in a table
column with a LongMemEval accuracy without saying so.

NOT REPRODUCED, deliberately: ``event_ordering`` additionally computes a
Kendall-tau ordering score over the response's lines, and ``information_
extraction``/``summarization`` have unused fact-level BLEU/ROUGE/embedding
paths in the same file. Those are auxiliary metrics reported beside the judge,
not the judge, and two of them need sentence-transformers at judge time. The
``llm_judge_score`` implemented here is the one number every ability shares and
the one ``report_results.py`` averages.

THREE DELIBERATE DEPARTURES FROM ``main``. Both are recorded, with the evidence,
in our development notes. This benchmark is a MEASURING INSTRUMENT
for this repo's experiment, not a table to reproduce, so where upstream's
released code disagrees with upstream's own paper or its own prompt, this file
follows the paper and the prompt.

1. ``<question>`` IS SUBSTITUTED. The prompt tells the judge to score 0.0 if
   the response "does not address the QUESTION", and on ``main`` the question
   is never put in front of it — so the gate is unenforceable. This is a
   REGRESSION, not a design choice: the placeholder and its two anchoring lines
   existed at 3e12035 and were dropped by b2da22e ("Fixed evaluation scoring
   bug"), whose substance was the int()->float() accumulator fix. See upstream
   issue #15, which measures the effect at 160/1051 item verdicts (15.2%)
   against a 2.7% temperature-0 noise floor, moving different abilities in
   opposite directions. The prompt below is 3e12035's, i.e. the author's own
   pre-regression text, restored rather than rewritten — plus the one added
   line in (3).

3. ONE LINE IS OURS, in the RESPONSIVENESS REQUIREMENT block. Restoring the
   question alone INVERTS on abstention: its nuggets are absence statements
   ("Based on the provided chat, there is no information related to ..."), the
   prompt's own keyword rule classifies those as NEGATIVE CONSTRAINTS, and it
   gates negative-constraint credit on responsiveness — so a correct refusal
   reads as "does not address the QUESTION" and scores 0.0. Measured on the
   three oracle cells: abstention -0.192, with 11-12 of 40 questions scoring
   zero on verdicts whose stated reason is that the response "states a lack of
   information". The added line says a stated absence IS responsive, while
   leaving the rubric criterion to decide whether it is CORRECT — so "I do not
   know" still scores 0.0 against a criterion like "should state: 250ms".
   Isolated at fixed judge model, the gate is worth keeping: it gains
   +0.080 / +0.044 / +0.032 on contradiction_resolution, information_extraction
   and summarization, same-signed across all three corpora, and
   temporal_reasoning's -0.092 is the judge catching genuine arithmetic errors.

4. ``event_ordering`` IS SCORED BY ``final_score``, NOT ``tau_norm``. The paper
   says tau-b was chosen because it "considers both order and presence";
   ``event_ordering_score`` computes ``final_score = tau_norm * f1`` and
   ``report_results.py:43`` then publishes the bare ``tau_norm``, dropping the
   presence term. Published tau_norm pays 0.215 for an empty answer and 0.929
   for listing half the events, and the paper's own 128K row (0.162-0.205
   across twelve systems) sits at or below that empty-answer floor.
   ``final_score`` floors at 0 like the other nine abilities. Every component
   is kept in ``beam.event_ordering``, so a tau_norm column is still one field
   away.

Consequence: these scores are NOT comparable to BEAM's published table, by
intent. ``metrics.json`` records which metric each ability carries under
``beam.score_is``.
"""

from __future__ import annotations

import asyncio
import json
import re

from evalgrid.benchmarks.judges.base import _REPLY_CHARS, MISSING_FROM_BATCH, Judge
from evalgrid.benchmarks.types import Query

#: src/prompts.py::unified_llm_judge_base_prompt, byte-for-byte, including its
#: unsubstituted "QUESTION" references (see the module docstring) and its
#: trailing note. The two placeholders are filled by str.replace, as upstream
#: does — NOT by str.format, which would choke on the JSON braces below.
UNIFIED_LLM_JUDGE_BASE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- An explicit statement that the conversation contains no such information IS a way of addressing the QUESTION, not a failure to address it. Treat such a response as responsive and score it against the RUBRIC CRITERION as normal — it satisfies the criterion only if the criterion is itself about that absence.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

#: The only three values the prompt offers. A judge that answers "0.75" or "1"
#: is answering off-scale; see _score_of for what happens to it.
_VALID_SCORES = (0.0, 0.5, 1.0)


def _rubric_messages(question: str, rubric_item: str, hypothesis: str) -> list[dict]:
    """The one judge call that grades ONE rubric item.

    Module-level, and the only place the prompt is interpolated, because the
    live driver and the Batch job file both send it. Two copies of a
    str.replace pair is exactly the kind of drift the batch path exists not to
    have.

    ``<question>`` is substituted here and ``main`` does not substitute it —
    see the module docstring. That is the whole of the difference.
    """
    return [{"role": "user", "content": UNIFIED_LLM_JUDGE_BASE_PROMPT
             .replace("<question>", question or "")
             .replace("<rubric_item>", rubric_item)
             .replace("<llm_response>", hypothesis or "")}]


def parse_json_response(response: str):
    """Upstream ``compute_metrics.parse_json_response``, reproduced.

    Deliberately NOT ``judges/base.extract_json``, which greps the first ``{``
    to the last ``}``. The two disagree on malformed replies, and on this
    benchmark the parser is part of the protocol: it decides which items score
    and which are thrown away, so a different one is a different judge.

    Order, exactly as upstream: strip a ``\u0060\u0060\u0060json`` fence,
    try the whole string, then the first non-greedy ``{...}`` or ``[...]``.
    """
    response = response.strip()
    if response.startswith("```"):
        m = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL)
        if m:
            response = m.group(1).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*?\}|\[.*?\])", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception as exc:
            raise ValueError(f"Found possible JSON but failed to parse it: {exc}")
    raise ValueError("No valid JSON found in response.")


def _repair(response: str):
    """Upstream's last resort: ``json_repair.repair_json``.

    ``json-repair`` is pinned in requirements.txt (it is already every
    extractor's malformed-JSON fallback), so this is the SAME recovery upstream
    gets rather than an approximation of it. Guarded only so an environment
    without it degrades to a failed item instead of an ImportError mid-run.
    """
    try:
        from json_repair import repair_json
    except ImportError:
        raise ValueError("reply was not JSON and json-repair is not installed")
    return json.loads(repair_json(response))



# ---------------------------------------------------------------------------
# event_ordering — the one ability BEAM does NOT report the judge's score for
# ---------------------------------------------------------------------------
#
# src/evaluation/report_results.py:43 reads ``tau_norm`` for this ability and
# ``llm_judge_score`` for the other nine. That is not an oversight: the unified
# prompt scores ONE rubric item at a time, in isolation, so it cannot see
# sequence — and event_ordering's rubric IS a sequence (ordered labels like
# "Initial project setup" -> "Transaction CRUD implementation"), with all 40 of
# its questions asking for order explicitly. A response listing every event
# backwards satisfies every item individually and scores 1.0 under the judge
# while being maximally wrong; its tau_norm is 0.0.
#
# So this ability is scored by rank correlation, reproduced from
# compute_metrics.event_ordering_score / align_with_llm / llm_equivalence.

def _answer_lines(hypothesis: str) -> list[str]:
    """Upstream's own segmentation of the answer: a bare ``split("\\n")``.

    Empty lines included, because upstream includes them and asks the judge
    about them. See _order_score for why this crude split works at all.
    """
    return list((hypothesis or "").split("\n"))


# ---------------------------------------------------------------------------
# One-shot alignment — ours, replacing upstream's pairwise scan
# ---------------------------------------------------------------------------
#
# Upstream's ``align_with_llm`` asks "are these two the same event?" once per
# (line, unused nugget) pair and stops at the first YES: up to
# len(lines) x len(nuggets) calls, measured at 1,185 for the 40 event_ordering
# questions of one cell — 53% of the whole judge budget for 10% of the
# benchmark — and it aligns BADLY, f1 0.301 against a 0.996 positional ceiling
# on answers that are demonstrably right (oracle.md §9.3a).
#
# It aligns badly for a structural reason, not a prompt one. The scan is greedy
# and blind: each call sees one nugget and one line, so a line that vaguely
# matches an early nugget CONSUMES it and the later line that really was that
# nugget can no longer match. Showing the judge both lists at once lets it do a
# global 1-to-1 assignment with every candidate visible, which is the task.
#
# THE TRAP, and why the nuggets are shuffled. Both lists are ordered, and order
# is exactly what the metric measures. A model shown two ordered lists is primed
# to zip them positionally — and a positional alignment makes system_out ==
# reference for EVERY answer, i.e. tau_norm 1.0 even for a reversed one. The
# metric would silently become a constant. So the nuggets are presented SHUFFLED
# under stable ids and mapped back after: position carries nothing for the
# aligner to lean on, and the only way to align is by meaning.

_ALIGN_PROMPT = """You are aligning two lists that describe the same events.

REFERENCE ITEMS (unordered, each with an id):
<reference>

RESPONSE LINES (in the order the response gave them):
<lines>

For each response line, decide which ONE reference item it describes, matching by
MEANING rather than wording — a line may be far more specific than the reference
item (extra dates, names or detail) and still describe the same event.

Rules:
- Each reference id may be used AT MOST ONCE. If two lines fit the same item, give
  it to the better fit and mark the other null.
- If a line describes no reference item, use null.
- The two lists are NOT in the same order. Do not match by position.

Return ONLY a JSON object mapping each line number to a reference id or null:
{"0": "r3", "1": null, "2": "r1"}"""


def _align_messages(rubric: list[str], lines: list[str], order: list[int]) -> list[dict]:
    """The single alignment call. ``order`` is the shuffled nugget order."""
    ref = "\n".join(f"- r{i}: {rubric[i]}" for i in order)
    num = "\n".join(f"{j}. {l.strip()}" for j, l in enumerate(lines))
    return [{"role": "user", "content": _ALIGN_PROMPT
             .replace("<reference>", ref).replace("<lines>", num)}]


def _shuffled_order(rubric: list[str], hypothesis: str) -> list[int]:
    """A deterministic shuffle of nugget indices.

    Seeded from the content, so a question always presents its nuggets the same
    way: a judge run is reproducible, and a re-judge of the same predictions
    cannot move because the shuffle moved.
    """
    import random

    order = list(range(len(rubric)))
    seed = abs(hash(("beam-align", hypothesis[:200], tuple(rubric)))) & 0xFFFFFFFF
    random.Random(seed).shuffle(order)
    return order


def _aligned_from(reply: str | None, rubric: list[str], lines: list[str]) -> list[str]:
    """Parse the alignment reply into upstream's ``system_out`` shape.

    Same contract as ``align_with_llm``'s return — one entry per response line,
    the reference text where it matched and the raw line where it did not —
    which is what leaves the tau arithmetic below untouched.

    Defensive per line: a malformed id, an out-of-range index or a reused id
    degrades THAT line to unmatched rather than failing the question. An
    unparseable reply raises, and the caller records the question as unjudged.
    """
    payload = parse_json_response(reply or "")
    if not isinstance(payload, dict):
        raise ValueError(f"alignment reply was not an object: {(reply or '')[:200]!r}")
    used: set[int] = set()
    out: list[str] = []
    for j, line in enumerate(lines):
        rid = payload.get(str(j))
        idx = None
        if isinstance(rid, str) and rid.startswith("r") and rid[1:].isdigit():
            cand = int(rid[1:])
            if 0 <= cand < len(rubric) and cand not in used:
                idx = cand
        if idx is None:
            out.append(line)
        else:
            out.append(rubric[idx])
            used.add(idx)
    return out


def _kendall_tau_norm(reference: list[str], system: list[str]) -> dict:
    """compute_metrics.event_ordering_score's arithmetic, on ALIGNED lists.

    Returns precision/recall/f1, ``tau_norm`` and ``final_score`` exactly as
    upstream computes them. Note upstream defines ``final_score = tau_norm * f1``
    and then REPORTS the bare ``tau_norm`` — so the published event-ordering
    number ignores whether the right events were listed and measures only the
    relative order of whichever ones were. All four are kept so the column can
    be read either way.
    """
    from scipy.stats import kendalltau

    tp = len(set(reference) & set(system))
    fp = len([x for x in system if x not in reference])
    fn = len([x for x in reference if x not in system])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    union = list(dict.fromkeys(reference + system))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(to_rank(reference), to_rank(system),
                          variant="b", method="auto")
    tau_norm = (tau_b + 1) / 2 if tau_b is not None and tau_b == tau_b else 0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tau_norm": tau_norm, "final_score": tau_norm * f1}

class BEAMJudge(Judge):
    """Per-rubric-item compliance grading, averaged. See the module docstring."""

    graded = True
    metric_name = "mean_rubric_score"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: Scores the judge returned that are not one of 0.0/0.5/1.0. Upstream
        #: passes these straight through and so does this file; they are
        #: collected only so a run can say how often the judge went off the
        #: scale its own prompt defines, which is otherwise invisible.
        self.off_scale: list[float] = []
        #: query_id -> the full event_ordering breakdown (precision/recall/f1,
        #: tau_norm, final_score, llm_judge_score). The reported score for that
        #: ability is tau_norm; the rest is kept so the column can be read the
        #: other ways upstream computes but does not publish.
        self.extra_scores: dict[str, dict] = {}

    def build_metrics(self, items, labels, token_usage, errors=None) -> dict:
        m = super().build_metrics(items, labels, token_usage, errors)
        m["beam"] = {
            "rubric_items_scored": sum(len(self.rubric_of(q)) for q, _ in items),
            "off_scale_scores": len(self.off_scale),
            "off_scale_values": sorted(set(self.off_scale))[:20],
            # BEAM reports event_ordering as tau_norm and every other ability as
            # the mean rubric score (report_results.py:43). Named here so a
            # metrics file can never be read as ten columns of one metric.
            "score_is": {"event_ordering": "final_score",
                         "*": "mean_rubric_score"},
            "event_ordering": self.extra_scores,
        }
        return m

    def build_messages(self, query: Query, hypothesis: str) -> list[dict]:
        """Unused: this judge issues one call per RUBRIC ITEM, not one per item.

        Neither driver reaches it — live goes through ``_judge_one`` and the
        Batch path through ``batch_requests``, both overridden here. It stays
        implemented because ``Judge`` declares it abstract, and it raises rather
        than returning a plausible-looking single-call prompt that would
        silently score a whole question off one rubric line.
        """
        raise NotImplementedError(
            "BEAM grades one rubric item per call — see BEAMJudge._judge_one "
            "(live) and BEAMJudge.batch_requests (--batch). There is no "
            "single-call prompt for a whole question.")

    def parse_label(self, response: str) -> bool:
        raise NotImplementedError(
            "BEAM is a graded judge: a verdict is a score in [0, 1], not a bool. "
            "Use parse_score / _judge_one.")

    def rubric_of(self, query: Query) -> list[str]:
        """The question's rubric criteria, one judge call each.

        Populated by the adapter from BEAM's own ``rubric`` field, which ships
        as a Python-repr string and is parsed there. Every question in the 100K
        tier has at least one (median 2, max 9, 1,051 calls for the tier).
        """
        rubric = (query.extra or {}).get("rubric") or []
        if isinstance(rubric, str):
            rubric = [rubric]
        return [str(item) for item in rubric if str(item).strip()]

    def parse_verdict(self, response: str) -> tuple[float, str]:
        """One rubric item's grade AND the justification the prompt asked for.

        The prompt's OUTPUT FORMAT names two fields, ``score`` and ``reason``,
        and the protocol only ever consumed the first. The second is the only
        thing that can answer "why is this 0.0" without paying for the call
        again, so it is returned here and kept in ``verdicts`` — see
        ``Judge.record_verdict``. Nothing about the SCORE changes: this is the
        same ``float(payload["score"])`` as before, which is upstream's
        ``score += float(response['score'])`` exactly.

        An earlier version snapped an off-scale value to the nearest of
        0.0/0.5/1.0 on the grounds that the prompt offers only those three.
        That was a mistake: it silently rewrites the judge's output, so a run
        of this file and a run of BEAM's own evaluator could disagree on a
        number that both call the published metric. Off-scale values are
        counted instead (see ``off_scale``) so the thing snapping was worried
        about stays visible without being papered over.
        """
        try:
            payload = parse_json_response(response)
        except ValueError:
            payload = _repair(response)
        if not isinstance(payload, dict) or "score" not in payload:
            raise ValueError(f"no 'score' key in judge reply: {response[:200]!r}")
        value = float(payload["score"])
        if value not in _VALID_SCORES:
            self.off_scale.append(value)
        return value, str(payload.get("reason", ""))

    def parse_score(self, response: str) -> float:
        """``parse_verdict``'s score half — the hook the base class declares."""
        return self.parse_verdict(response)[0]

    def _record_rubric(self, query: Query, index: int, criterion: str,
                       score: float | None, error: str | None, reason: str) -> None:
        """One graded rubric call, as the live and batch paths both saw it.

        ``criterion`` is carried verbatim rather than by index because the log
        is read on its own, beside a metrics file that does not hold the
        rubric — and an index into a list you cannot see explains nothing.
        """
        self.record_verdict(query.query_id, kind="rubric", rubric_index=index,
                            criterion=criterion, score=score, error=error,
                            reason=reason[:_REPLY_CHARS])

    def _record_alignment(self, query_id: str, rubric: list[str],
                          lines: list[str], system_out: list[str]) -> None:
        """Store what the aligner decided, for the verdicts sidecar.

        The alignment is the whole event_ordering score, so an unreadable
        alignment is an unreadable column — keeping it means a suspicious
        tau can be traced to the line that failed to match.
        """
        self.record_verdict(
            query_id, kind="alignment",
            matched=sum(1 for x in system_out if x in rubric),
            of_lines=len(lines), of_nuggets=len(rubric),
            pairs=[{"line": l[:160], "matched_to": (o if o in rubric else None)}
                   for l, o in zip(lines, system_out)])

    def batch_requests(self, index: int, query: Query, hypothesis: str
                       ) -> list[tuple[str, list[dict]]]:
        """Every call this question needs, including ones the live path may skip.

        A question with no rubric returns [] — BEAM scores only against a rubric,
        and resolve_batch reports it unjudged with the same words _judge_one
        uses, rather than this sending a request whose answer means nothing.
        """
        rubric = self.rubric_of(query)
        if not rubric:
            return []
        requests = [(f"r:{index}:{j}", _rubric_messages(query.question, item, hypothesis))
                    for j, item in enumerate(rubric)]
        if query.question_type == "event_ordering":
            requests.append((f"a:{index}",
                             _align_messages(rubric, _answer_lines(hypothesis),
                                             _shuffled_order(rubric, hypothesis))))
        return requests

    def resolve_batch(self, index: int, query: Query, hypothesis: str,
                      replies: dict[str, tuple[str | None, str | None]]
                      ) -> tuple[float | None, str | None]:
        """_judge_one's arithmetic over replies that already came back.

        Same rule as live, for the same reason: a question is scored only if
        EVERY rubric item got a verdict, because a mean over whichever criteria
        happened to answer is a different, easier metric.
        """
        rubric = self.rubric_of(query)
        if not rubric:
            return None, "question carries no rubric, and BEAM scores only against a rubric"

        scores: list[float] = []
        failed: list[str] = []
        for j in range(len(rubric)):
            reply, error = replies.get(f"r:{index}:{j}", (None, MISSING_FROM_BATCH))
            if reply is None:
                failed.append(error or "no verdict returned")
                self._record_rubric(query, j, rubric[j], None,
                                    error or "no verdict returned", "")
                continue
            try:
                score, reason = self.parse_verdict(reply)
            except ValueError as exc:
                failed.append(str(exc)[:500])
                self._record_rubric(query, j, rubric[j], None, str(exc)[:500], "")
                continue
            scores.append(score)
            self._record_rubric(query, j, rubric[j], score, None, reason)
        if failed:
            return None, (f"{len(failed)}/{len(rubric)} rubric items got no verdict: "
                          f"{failed[0]}")
        judge_score = sum(scores) / len(scores)

        if query.question_type != "event_ordering":
            return judge_score, None

        try:
            ordering = self._order_score_from(replies, query, index, hypothesis, rubric)
        except Exception as exc:                       # noqa: BLE001
            return None, f"event_ordering alignment failed: {type(exc).__name__}: {exc}"[:500]
        ordering["llm_judge_score"] = judge_score
        self.extra_scores[query.query_id] = ordering
        return ordering["final_score"], None

    def _order_score_from(self, replies, query: Query, index: int, hypothesis: str,
                          rubric: list[str]) -> dict:
        """The batch half of _order_score: one reply, same parse, same arithmetic.

        One request per question now, so there is no scan to replay and no
        shared generator to drive — the two paths agree because both call the
        same _align_messages / _aligned_from pair.
        """
        reply, error = replies.get(f"a:{index}", (None, MISSING_FROM_BATCH))
        if reply is None:
            raise ValueError(f"alignment call a:{index}: {error}")
        lines = _answer_lines(hypothesis)
        system_out = _aligned_from(reply, rubric, lines)
        self._record_alignment(query.query_id, rubric, lines, system_out)
        return _kendall_tau_norm(rubric, system_out)


    async def _judge_one(self, sem: asyncio.Semaphore, query: Query,
                         hypothesis: str) -> tuple[float | None, str | None]:
        """Score one question: one call per rubric item, then the mean.

        The semaphore is taken per CALL, not per question, so a 9-item rubric
        does not hold a slot while it works through nine sequential requests —
        with rubrics of 1 to 9 items, per-question slots would leave most of the
        concurrency idle. The calls within one question run concurrently for the
        same reason.

        A question is scored only if EVERY item got a verdict. A partial mean
        over the items that happened to answer is a different, easier metric —
        it silently drops whichever criteria the judge found hardest — so a
        failed item fails the question, which the base class then excludes from
        both numerator and denominator and lists in judge_failures.jsonl.
        """
        rubric = self.rubric_of(query)
        if not rubric:
            return None, "question carries no rubric, and BEAM scores only against a rubric"

        async def grade(item: str) -> tuple[float | None, str | None, str]:
            async with sem:
                reply, error = await self._generate(
                    _rubric_messages(query.question, item, hypothesis))
            if error is not None:
                return None, error, ""
            try:
                score, reason = self.parse_verdict(reply)
                return score, None, reason
            except ValueError as exc:
                return None, str(exc)[:500], ""

        results = await asyncio.gather(*(grade(item) for item in rubric))
        # Recorded AFTER the gather, walking the rubric in its own order, so the
        # log reads in rubric order however the concurrent calls happened to
        # land. Failed items are recorded too: "the judge never answered this
        # criterion" is exactly what a reader of a failed question needs.
        for j, (item, (score, error, reason)) in enumerate(zip(rubric, results)):
            self._record_rubric(query, j, item, score, error, reason)
        scores = [s for s, _, _ in results]
        if any(s is None for s in scores):
            failed = [e for _, e, _ in results if e]
            return None, (f"{sum(s is None for s in scores)}/{len(rubric)} rubric items "
                          f"got no verdict: {failed[0] if failed else 'unknown'}")
        judge_score = sum(scores) / len(scores)

        if query.question_type != "event_ordering":
            return judge_score, None

        # Reported as final_score, not as the judge's score — see the module
        # docstring. The judge's own score is still computed (upstream computes
        # it too) and kept in `extra_scores` for reference.
        try:
            ordering = await self._order_score(sem, query, hypothesis, rubric)
        except Exception as exc:                       # noqa: BLE001
            return None, f"event_ordering alignment failed: {type(exc).__name__}: {exc}"[:500]
        ordering["llm_judge_score"] = judge_score
        self.extra_scores[query.query_id] = ordering
        return ordering["final_score"], None

    # --- batch protocol ----------------------------------------------------
    #
    # Judge.batch_requests/resolve_batch, for a judge whose item is 1-to-9
    # grading calls plus, for event_ordering, an alignment that decides its next
    # call from the last one's answer. Two request families, distinguished by
    # the custom_id's prefix:
    #
    #   r:<item>:<rubric index>          one rubric criterion, graded 0/0.5/1
    #   e:<item>:<line>:<rubric index>   "same event?", for the alignment
    #
    # Both are pure functions of (item index, question, hypothesis), which is
    # what lets a --batch-phase collect run rebuild them and find its replies.

    def batch_requests(self, index: int, query: Query, hypothesis: str
                       ) -> list[tuple[str, list[dict]]]:
        """Every call this question needs, including ones the live path may skip.

        A question with no rubric returns [] — BEAM scores only against a rubric,
        and resolve_batch reports it unjudged with the same words _judge_one
        uses, rather than this sending a request whose answer means nothing.
        """
        rubric = self.rubric_of(query)
        if not rubric:
            return []
        requests = [(f"r:{index}:{j}", _rubric_messages(query.question, item, hypothesis))
                    for j, item in enumerate(rubric)]
        if query.question_type == "event_ordering":
            requests.append((f"a:{index}",
                             _align_messages(rubric, _answer_lines(hypothesis),
                                             _shuffled_order(rubric, hypothesis))))
        return requests

    def resolve_batch(self, index: int, query: Query, hypothesis: str,
                      replies: dict[str, tuple[str | None, str | None]]
                      ) -> tuple[float | None, str | None]:
        """_judge_one's arithmetic over replies that already came back.

        Same rule as live, for the same reason: a question is scored only if
        EVERY rubric item got a verdict, because a mean over whichever criteria
        happened to answer is a different, easier metric.
        """
        rubric = self.rubric_of(query)
        if not rubric:
            return None, "question carries no rubric, and BEAM scores only against a rubric"

        scores: list[float] = []
        failed: list[str] = []
        for j in range(len(rubric)):
            reply, error = replies.get(f"r:{index}:{j}", (None, MISSING_FROM_BATCH))
            if reply is None:
                failed.append(error or "no verdict returned")
                self._record_rubric(query, j, rubric[j], None,
                                    error or "no verdict returned", "")
                continue
            try:
                score, reason = self.parse_verdict(reply)
            except ValueError as exc:
                failed.append(str(exc)[:500])
                self._record_rubric(query, j, rubric[j], None, str(exc)[:500], "")
                continue
            scores.append(score)
            self._record_rubric(query, j, rubric[j], score, None, reason)
        if failed:
            return None, (f"{len(failed)}/{len(rubric)} rubric items got no verdict: "
                          f"{failed[0]}")
        judge_score = sum(scores) / len(scores)

        if query.question_type != "event_ordering":
            return judge_score, None

        try:
            ordering = self._order_score_from(replies, query, index, hypothesis, rubric)
        except Exception as exc:                       # noqa: BLE001
            return None, f"event_ordering alignment failed: {type(exc).__name__}: {exc}"[:500]
        ordering["llm_judge_score"] = judge_score
        self.extra_scores[query.query_id] = ordering
        return ordering["final_score"], None

    async def _order_score(self, sem, query, hypothesis: str, rubric: list[str]) -> dict:
        """ONE alignment call, then event_ordering_score's arithmetic.

        ``hypothesis.split("\n")`` is upstream's own segmentation of the answer,
        and it works under any answer prompt because the event_ordering
        QUESTIONS carry the format instruction themselves ("Can you list the
        order in which ... Mention ONLY and ONLY three items").

        The alignment itself is NOT upstream's — see _align_messages. One call
        replaces up to len(lines) x len(rubric) pairwise ones.
        """
        lines = _answer_lines(hypothesis)
        order = _shuffled_order(rubric, hypothesis)
        async with sem:
            reply, error = await self._generate(_align_messages(rubric, lines, order))
        if error is not None:
            raise RuntimeError(error)
        system_out = _aligned_from(reply, rubric, lines)
        self._record_alignment(query.query_id, rubric, lines, system_out)
        return _kendall_tau_norm(rubric, system_out)
