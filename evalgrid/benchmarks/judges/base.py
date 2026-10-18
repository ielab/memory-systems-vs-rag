"""
evalgrid/benchmarks/judges/base.py
--------------------------
LLM-as-judge scoring for benchmark predictions.

A Judge reads (Query, hypothesis) pairs and asks a judge LLM how good each
hypothesis is, then aggregates overall + per-category accuracy. Each benchmark
has its OWN canonical judge (prompt + parsing), implemented in a subclass,
because the benchmarks define different evaluation protocols:
  locomo.py      — LoCoMoJudge: generous CORRECT/WRONG, category-tracked
  longmemeval.py — LongMemEvalJudge: official per-question-type prompts
  beam.py        — BEAMJudge: GRADED. Per-rubric-item 0.0/0.5/1.0, averaged

A VERDICT IS A FLOAT IN [0, 1], not a bool. Two of the three benchmarks here
grade pass/fail and emit 1.0 or 0.0, so "accuracy" is the mean verdict and means
exactly what it always did; BEAM grades a response against a rubric of several
criteria and emits their mean, so the same field is a mean rubric score. A judge
declares which it is with the ``graded`` flag, and metrics files say so in
``metric`` — the two must never be silently averaged into one table cell,
because 0.5 from a binary judge is impossible and 0.5 from BEAM is common.

Subclasses stay binary by overriding ``parse_label``; a graded judge overrides
``parse_score``. Nothing that existed before this split had to change.

Keep the judge fixed within a benchmark across all systems you compare; the
within-benchmark comparison is what's valid (cross-benchmark scores are not
comparable regardless of judge).
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

from tqdm import tqdm

from evalgrid.benchmarks.types import Query
from evalgrid.models.base import GeneratorModel, OutputCapError


def _is_output_cap_error(exc: Exception) -> bool:
    """True if the backend rejected the call because output hit the token cap.

    Two shapes: vLLM raises a 400 BadRequestError with this message when a
    reasoning model consumes the whole max_tokens budget before producing a
    final answer; OpenAI instead returns empty content with finish_reason=
    'length', which openai_gen normalises into OutputCapError. Both mean the
    same thing — retry with a bigger budget.
    """
    if isinstance(exc, OutputCapError):
        return True
    msg = str(exc).lower()
    return "max_tokens" in msg or "model output limit" in msg


#: What a custom_id that never came back is reported as. Lives here rather than
#: in judge_batch.py because every resolve_batch implementation needs the same
#: wording for the same condition, and a batch cancelled or expired part-way
#: through is the common way to meet it.
MISSING_FROM_BATCH = "missing from batch results (cancelled or expired?)"

#: How much of a judge reply / reason is kept in the verdict log. A binary
#: judge's reply is a word and a BEAM reason is a sentence or three, so this
#: truncates almost nothing — it is a bound on the pathological case (a judge
#: that ignores the output format and returns an essay) rather than a budget.
_REPLY_CHARS = 2000


def extract_json(text: str) -> Optional[dict]:
    """Best-effort: parse the first {...} block in the text as JSON."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def gold_to_str(answer) -> str:
    """Normalise a gold answer (str or list) to a single string."""
    if isinstance(answer, (list, tuple)):
        return "; ".join(str(a) for a in answer)
    return "" if answer is None else str(answer)


class Judge(ABC):
    """Base LLM-as-judge. Subclasses define the per-benchmark prompt & parsing."""

    def __init__(
        self,
        judge: GeneratorModel,
        *,
        max_tokens: int = 1024,
        concurrency: int = 8,
    ):
        self.judge = judge
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        #: query_id -> the calls that produced its score, in the order the
        #: protocol made them. THE JUDGE'S OWN WORDS, kept rather than dropped.
        #: Every judge prompt here asks for a justification alongside the
        #: verdict — BEAM's asks for a `reason` field explicitly — and the
        #: aggregation reads only the number, so before this the reasoning was
        #: generated, paid for (on BEAM ~70 output tokens per rubric call
        #: against ~1 for a binary judge) and thrown away. A score nothing can
        #: explain is a score nothing can be checked against: the only way to
        #: ask "why did this rubric item get 0.0" was to re-run the call and
        #: hope the judge said the same thing twice.
        #:
        #: Populated by whichever path actually made the call, so live and
        #: batch produce the same records; pipeline/judge.py writes them to
        #: judge_verdicts.<judge-slug>.jsonl beside the metrics. Kept OUT of
        #: metrics.json deliberately — that file is read whole by every table
        #: builder, and a BEAM cell's 1,051 reasons would multiply it by ~50.
        self.verdicts: dict[str, list[dict]] = {}

    def record_verdict(self, query_id: str, **entry) -> None:
        """Append one call's record to ``verdicts[query_id]``.

        Ordered by when the protocol asked, not by when the answer arrived, so
        a rubric's items read in rubric order however the calls were scheduled.
        """
        self.verdicts.setdefault(query_id, []).append(entry)

    # --- subclass hooks ----------------------------------------------------

    #: True when this judge returns a GRADED score rather than a verdict — a
    #: value that can land strictly between 0 and 1 for a single item. It
    #: changes nothing about how scores are aggregated (the mean is the mean);
    #: it changes what the aggregate is CALLED and whether a per-item ``correct``
    #: boolean is meaningful. BEAM is the only graded judge here: its score is
    #: the mean of per-rubric-item 0.0/0.5/1.0 compliance grades, so an item
    #: scoring 0.5 is half-right, not right or wrong.
    graded: bool = False

    #: What ``overall.accuracy`` is the mean OF, recorded in the metrics file so
    #: a reader never has to infer it from the judge's name.
    metric_name: str = "accuracy"

    @abstractmethod
    def build_messages(self, query: Query, hypothesis: str) -> list[dict]:
        """Return chat messages asking the judge to grade one prediction."""

    @abstractmethod
    def parse_label(self, response: str) -> bool:
        """Parse the judge's reply into a correct/incorrect boolean."""

    def parse_score(self, response: str) -> float:
        """Parse the judge's reply into a score in [0, 1].

        The one hook the aggregation actually calls. A binary judge does not
        implement it: the default defers to ``parse_label``, so CORRECT/WRONG
        arrives as 1.0/0.0 and every number downstream is what it was before
        scores were floats. A graded judge overrides this instead of
        ``parse_label`` and sets ``graded = True``.
        """
        return float(self.parse_label(response))

    def category_of(self, query: Query) -> str:
        """Bucket for per-category aggregation."""
        return query.question_type or "all"

    # --- batch protocol ----------------------------------------------------
    #
    # The Batch API path (benchmarks/judge_batch.py) cannot await anything: every
    # request must be known before the job file is uploaded, and the replies come
    # back in one lump keyed by custom_id. These two hooks are that shape —
    # declare the calls, then turn their replies into one score — and the default
    # pair below IS the live path's one-call-per-item protocol, so a binary judge
    # inherits batch support without writing a line.
    #
    # A judge that overrides ``_judge_one`` because its protocol is several calls
    # per item overrides these two as well, and the three must stay consistent.
    # The one thing the Batch API forbids that the live path allows is an
    # ADAPTIVE plan: a call whose existence depends on an earlier call's answer
    # cannot be written into the job file. The fix is to ask for every call the
    # loop could possibly want and decide offline which answers it consults —
    # see BEAMJudge, whose event-ordering alignment is exactly that shape.

    def batch_requests(self, index: int, query: Query, hypothesis: str
                       ) -> list[tuple[str, list[dict]]]:
        """The (custom_id, messages) pairs item ``index`` needs judged.

        custom_ids must be unique across the whole file and a pure function of
        (index, item): nothing stores them, so the collect phase of a two-phase
        run rebuilds them from the same predictions and must get the same
        strings. Returning [] means the item cannot be judged at all — it is
        reported unjudged rather than sent.
        """
        return [(str(index), self.build_messages(query, hypothesis))]

    def resolve_batch(self, index: int, query: Query, hypothesis: str,
                      replies: dict[str, tuple[str | None, str | None]]
                      ) -> tuple[float | None, str | None]:
        """This item's score from ``replies``: custom_id -> (text, error).

        Same contract as ``_judge_one``: (score, None) or (None, error). A
        custom_id absent from ``replies`` is a line the batch never returned —
        cancelled, expired, or dropped — and counts as no verdict.
        """
        content, error = replies.get(str(index), (None, MISSING_FROM_BATCH))
        if content is None:
            return None, error or "no verdict returned"
        # parse_score, not parse_label: it is the hook build_metrics reads, and
        # for a binary judge it is float(parse_label(...)) — the same verdict.
        score = self.parse_score(content)
        self.record_verdict(query.query_id, kind="verdict", score=score,
                            reply=(content or "")[:_REPLY_CHARS])
        return score, None

    # --- driver ------------------------------------------------------------

    async def _generate(self, messages: list[dict]) -> tuple[str | None, str | None]:
        """Run the judge; return (reply, error) with exactly one of them set.

        The judge prompts ask for a tiny verdict, but a reasoning-enabled judge
        (e.g. a Qwen3 thinking model behind vLLM) can spend the whole token
        budget inside <think> and emit no final answer. vLLM then 400s with
        "Could not finish the message because max_tokens ... was reached". We
        raise that ONE item's budget and send it again — a budget correction,
        not a retry of the same call — and if it still will not finish, the item
        is reported as UNJUDGED.

        THE ERROR IS RETURNED, NOT SWALLOWED INTO A VERDICT. This used to return
        "" on failure, which parse_label reads as False — so a judge call that
        never happened was indistinguishable from a system answering wrongly,
        and every infrastructure hiccup pushed accuracy DOWN by a silent amount.
        An unjudged item is missing data: it leaves the numerator and the
        denominator both, and is listed in judge_failures.jsonl.
        """
        budget = self.max_tokens
        for attempt in range(2):
            try:
                return await self.judge.generate(
                    messages, max_tokens=budget, temperature=0.0
                ), None
            except Exception as exc:  # noqa: BLE001 — a failed item must not kill the run
                if attempt == 0 and _is_output_cap_error(exc):
                    budget = max(budget * 8, 8192)
                    continue
                return None, f"{type(exc).__name__}: {exc}"[:500]
        return None, "output cap not cleared at 8x budget"

    async def _judge_one(self, sem: asyncio.Semaphore, query: Query,
                         hypothesis: str) -> tuple[float | None, str | None]:
        """One item's score in [0, 1], or (None, error) if it got no verdict.

        Overridable in full, not just through ``parse_score``: a judge whose
        protocol is several calls per item (BEAM asks once per rubric item and
        averages) cannot express itself as one build_messages/one parse, and
        should reuse ``self._generate`` inside its own fan-out so the
        output-cap retry and the error contract stay identical.
        """
        async with sem:
            messages = self.build_messages(query, hypothesis)
            resp, error = await self._generate(messages)
        if error is not None:
            return None, error
        score = self.parse_score(resp)
        self.record_verdict(query.query_id, kind="verdict", score=score,
                            reply=(resp or "")[:_REPLY_CHARS])
        return score, None

    async def _judge_all(self, items: list[tuple[Query, str]]
                         ) -> list[tuple[float | None, str | None]]:
        sem = asyncio.Semaphore(self.concurrency)
        with tqdm(total=len(items), desc="judging", unit="item") as bar:
            async def _wrapped(query: Query, hypothesis: str):
                result = await self._judge_one(sem, query, hypothesis)
                bar.update(1)
                return result

            return await asyncio.gather(*(_wrapped(q, h) for q, h in items))

    def build_metrics(self, items: list[tuple[Query, str]], labels: list[float | None],
                      token_usage, errors: list[str | None] | None = None) -> dict:
        """Aggregate per-item correctness ``labels`` into the metrics dict.

        Split out from score() so an alternative driver (e.g. the Batch-API
        judge in benchmarks/judge_batch.py) can reuse the exact same aggregation
        and output shape after gathering labels its own way. ``token_usage`` is
        anything carrying .input_tokens/.output_tokens/.calls (the live judge
        passes its generator's TokenUsage; the batch path sums the batch bodies).

        A label of ``None`` means THE JUDGE NEVER RETURNED A VERDICT for that
        item — a failed call live, an errored line in a batch. Those items are
        excluded from both the numerator and the denominator rather than counted
        as wrong, because "the grader did not answer" is not evidence about the
        system being graded, and scoring it as wrong biases every affected row
        downward by an amount nothing on disk records.

        The counts are reported so the exclusion can never be invisible:
        ``attempted`` is what was sent, ``total`` what came back with a verdict,
        ``failed`` the difference. ``complete`` at the top level is
        ``failed == 0`` — a metrics file for which that is false is a partial
        score and must not be read as a final number.
        """
        errors = errors or [None] * len(labels)
        per_cat: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])  # cat -> [score, total]
        per_question = []
        failures = []
        correct = 0.0
        judged = 0
        for (query, _), ok, err in zip(items, labels, errors):
            cat = self.category_of(query)
            if ok is None:
                # Not in per_cat: a category's denominator must match the
                # overall one, or the per-category rows will not reconcile.
                per_question.append({"question_id": query.query_id, "category": cat,
                                     "score": None, "correct": None, "judged": False,
                                     "error": err or "no verdict returned"})
                failures.append({"question_id": query.query_id, "category": cat,
                                 "error": err or "no verdict returned"})
                continue
            score = float(ok)
            per_cat[cat][0] += score
            per_cat[cat][1] += 1
            correct += score
            judged += 1
            # ``score`` is the value everything aggregates and the one a resume
            # pass reads back. ``correct`` is kept beside it for a binary judge
            # ONLY — it is what every existing metrics file carries and what
            # pipeline/judge.py --retry-failed used to key on. A graded judge
            # writes None there rather than rounding: BEAM's 0.5 is a half-met
            # rubric, and there is no honest bool for it.
            per_question.append(
                {"question_id": query.query_id, "category": cat, "score": score,
                 "correct": None if self.graded else bool(ok), "judged": True}
            )

        attempted = len(items)

        def _sum(value: float):
            """A binary judge's totals stay ints, as every existing file has them."""
            return round(value, 6) if self.graded else int(value)

        by_category = {
            cat: {"accuracy": c / t if t else 0.0, "correct": _sum(c), "total": t}
            for cat, (c, t) in sorted(per_cat.items())
        }
        return {
            "complete": not failures,
            # What "accuracy" is the mean of. "accuracy" for a pass/fail judge;
            # a graded judge names its own scale, so a BEAM row and a
            # LongMemEval row can never be read as the same measurement by
            # accident.
            "metric": self.metric_name,
            "overall": {
                "accuracy": correct / judged if judged else 0.0,
                "correct": _sum(correct),
                # `total` stays the accuracy DENOMINATOR, which is what every
                # reader of this field has always assumed it was.
                "total": judged,
                "attempted": attempted,
                "failed": len(failures),
            },
            "by_category": by_category,
            "judge_tokens": {
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "calls": token_usage.calls,
            },
            "judge_failures": failures,
            "per_question": per_question,
        }

    def label_items(self, items: list[tuple[Query, str]]
                    ) -> tuple[list[float | None], list[str | None]]:
        """Judge each pair and return the raw (labels, errors), unaggregated.

        Public because a caller may need to judge a SUBSET and merge the result
        into an existing metrics file — which is what `--retry-failed` does with
        the items a previous pass could not get a verdict for.
        """
        results = asyncio.run(self._judge_all(items))
        return [label for label, _ in results], [err for _, err in results]

    def score(self, items: list[tuple[Query, str]]) -> dict:
        """Judge every (Query, hypothesis) pair and return a metrics dict."""
        labels, errors = self.label_items(items)
        return self.build_metrics(items, labels, self.judge.token_usage, errors)
