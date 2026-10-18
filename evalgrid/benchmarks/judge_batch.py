"""
benchmarks/judge_batch.py
--------------------------
Batch-API variant of evalgrid/benchmarks/judges/base.py's LLM-as-judge scoring.

The live path (Judge.score in judges/base.py) makes one synchronous
chat-completions call per prediction, bounded by a concurrency semaphore. When
there are many predictions to grade — several retrievers x top-k sweeps x the
whole question set — that's slow and burns through a synchronous rate limit.
OpenAI's Batch API sidesteps the per-minute limit entirely (a separate, much
larger queue), costs half as much, and fits judging well since none of it is
latency sensitive.

Reuses the judge's OWN prompts and parsing unchanged, through the batch hooks on
judges/base.py (``batch_requests`` declares the calls, ``resolve_batch`` turns
their replies into a score) — only the transport (submit-a-job-file-and-poll vs.
call-and-await) differs, so live and batch judging can never drift on what they
actually ask the model — and the batch plumbing (submit/poll) from
benchmarks/openai_batch.py.

A JUDGE MAY NEED SEVERAL CALLS PER ITEM. The default hooks are one call with
``custom_id = str(i)``, which is what every binary judge here does and what this
module used to hardcode. BEAM is graded: it asks once per rubric item and, for
event_ordering, once per alignment pair, so its items are ~5 requests each under
ids like ``r:12:0`` and ``e:12:3:1``. Nothing in this file knows those shapes —
it moves lines and hands replies back to the judge that asked for them.

Resumable across process restarts the same way openai_batch is: the submitted
batch_id is saved to <work_dir>/<tag>_state.json before polling, so a re-run
re-attaches to the in-flight batch instead of submitting (and paying) again.
custom_ids are derived from the item's position, so this resume is only correct
when the re-run's ``items`` are IDENTICAL to the submitting run's — which holds
here because items come from the same predictions.jsonl filtered
deterministically. (Each predictions file gets its own <work_dir>, so different
runs never share state.)

A line that produces no verdict — an error, a non-2xx status, an empty or
max_tokens-truncated completion, or a custom_id missing from the results
entirely — is reported as UNJUDGED and excluded from the score. It is NOT
counted as a wrong answer; see Judge.build_metrics for why, and
judge_failures.jsonl / `--retry-failed` for what to do about it.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from evalgrid.benchmarks.judges.base import Judge
from evalgrid.benchmarks.openai_batch import (
    MAX_BATCH_REQUESTS,
    _batch_line,
    fetch_results,
    submit_batch,
    wait_for_batch,
)
from evalgrid.benchmarks.types import Query
from evalgrid.models.generators.openai_gen import OpenAIGenerator


def _request_body(
    messages: list[dict],
    model: str,
    max_tokens: int,
    *,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    seed: int | None = None,
    temperature: float = 0.0,
) -> dict:
    """One chat-completions request body for a single judge call.

    Mirrors the live judge: temperature is pinned to 0.0 (Judge._generate passes
    temperature=0.0), and the gpt-5-family token-kwarg/temperature/reasoning/
    verbosity quirks match OpenAIGenerator.generate() so live and batch grade
    identically. No response_format — the judges parse a free-text yes/no.
    ``temperature`` is overridable only for benchmarks/answer_batch.py, which
    shares this body so batch answering cannot drift from live answering
    either; it carries --gen-temperature there.
    """
    token_kwarg = (
        {"max_completion_tokens": max_tokens}
        if OpenAIGenerator._uses_max_completion_tokens(model)
        else {"max_tokens": max_tokens}
    )
    temperature_kwarg = (
        {} if OpenAIGenerator._uses_default_temperature(model) else {"temperature": temperature}
    )
    reasoning_kwarg = (
        {"reasoning_effort": reasoning_effort}
        if reasoning_effort and OpenAIGenerator._supports_reasoning_effort(model)
        else {}
    )
    verbosity_kwarg = (
        {"verbosity": verbosity} if verbosity and model.startswith("gpt-5") else {}
    )
    seed_kwarg = {"seed": seed} if seed is not None else {}
    return {
        "model": model,
        "messages": messages,
        **token_kwarg,
        **temperature_kwarg,
        **reasoning_kwarg,
        **verbosity_kwarg,
        **seed_kwarg,
    }


def _build_lines(
    judge: Judge,
    model: str,
    items: list[tuple[Query, str]],
    max_tokens: int,
    *,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    """The batch's request lines, from each item's ``judge.batch_requests``.

    ONE REQUEST PER ITEM IS NOT ASSUMED. A binary judge inherits the default
    hook and emits exactly one line per item with ``custom_id = str(i)``, byte
    for byte what this built before; a graded judge emits several — BEAM asks
    once per RUBRIC ITEM and again per alignment pair, so a 400-question cell is
    ~2,280 lines rather than 400. The judge owns the custom_id scheme because
    the judge is the only thing that can read it back (``resolve_batch``).

    Ids are checked for collisions here rather than trusted: two lines sharing a
    custom_id is a silent misattribution — the second reply overwrites the first
    in the results map — and it is a mistake a hook can make in one typo.
    """
    lines: list[dict] = []
    seen: dict[str, int] = {}
    for i, (query, hypothesis) in enumerate(items):
        for custom_id, messages in judge.batch_requests(i, query, hypothesis):
            if custom_id in seen:
                raise ValueError(
                    f"{type(judge).__name__}.batch_requests produced duplicate "
                    f"custom_id {custom_id!r} (items {seen[custom_id]} and {i}) — "
                    f"ids must be unique across the file or replies misattribute.")
            seen[custom_id] = i
            lines.append(_batch_line(custom_id, _request_body(
                messages, model, max_tokens,
                reasoning_effort=reasoning_effort, verbosity=verbosity, seed=seed,
            )))
    if len(lines) > MAX_BATCH_REQUESTS:
        raise ValueError(
            f"{len(lines)} judging requests for {len(items)} item(s) exceeds the "
            f"Batch API's {MAX_BATCH_REQUESTS}-request limit — judge in smaller "
            f"pieces (e.g. per retriever/top-k predictions file) instead of all "
            f"at once."
        )
    return lines


def _aggregate(judge: Judge, items: list[tuple[Query, str]],
               results: dict[str, tuple[str | None, dict | None, str | None]],
               tag: str) -> dict:
    """Score each item from its replies with judge.resolve_batch, and aggregate.

    A line that produced no verdict is passed to build_metrics as ``None``, not
    as ``False`` — it is UNJUDGED, and build_metrics drops it from both the
    numerator and the denominator. This used to score it incorrect "to mirror
    the live path"; the live path was wrong in the same way, and both are fixed
    together so batch and live still cannot drift.

    An item whose custom_ids are missing from ``results`` entirely (absent from
    both the output and error files) is also unjudged — that is what a batch
    cancelled or expired part-way through looks like. Whether ONE missing line
    sinks an item is the JUDGE's call, not this function's: BEAM fails a
    question when a rubric item goes unanswered, but not when the batch is
    missing an alignment pair its greedy scan never reaches.

    Tokens are summed over every RESULT LINE rather than per item, because for a
    graded judge the two differ by a factor of five and it is the lines that
    were paid for. ``calls`` counts lines that came back with content — for a
    one-line-per-item judge, exactly what it counted before.
    """
    in_tok = out_tok = calls = 0
    for _custom_id, (content, usage, _error) in results.items():
        if usage:
            in_tok += usage.get("prompt_tokens", 0)
            out_tok += usage.get("completion_tokens", 0)
        if content is not None:
            calls += 1
    replies = {custom_id: (content, error)
               for custom_id, (content, _usage, error) in results.items()}

    labels: list[float | None] = []
    errors: list[str | None] = []
    for i, (query, hypothesis) in enumerate(items):
        score, error = judge.resolve_batch(i, query, hypothesis, replies)
        labels.append(score)
        errors.append((error or "no verdict returned") if score is None else None)
    failed = sum(1 for e in errors if e)
    if failed:
        # Grouped, because 400 identical rate-limit lines and 400 different ones
        # are different problems and the count alone does not say which.
        from collections import Counter
        print(f"[{tag}] {failed} of {len(items)} item(s) returned NO VERDICT — "
              f"excluded from the score, not counted wrong:")
        for reason, n in Counter(e.split(":")[0] for e in errors if e).most_common(5):
            print(f"[{tag}]   {n:>5}  {reason}")
    usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok, calls=calls)
    return judge.build_metrics(items, labels, usage, errors)


def submit_batch_judging(
    client,
    judge: Judge,
    model: str,
    items: list[tuple[Query, str]],
    *,
    work_dir: str,
    tag: str = "judge",
    max_tokens: int = 1024,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    seed: int | None = None,
) -> str | None:
    """Build and submit this file's batch WITHOUT waiting, returning its
    batch_id (or None when there's nothing to judge).

    Splitting submit from collect lets a driver fire every predictions file's
    batch up front, so they run concurrently on OpenAI's side instead of one
    blocking the next. Idempotent: if a batch was already submitted for this
    work_dir (state file present), it resumes that one instead of resubmitting.
    """
    os.makedirs(work_dir, exist_ok=True)
    state_path = os.path.join(work_dir, f"{tag}_state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            batch_id = json.load(f)["batch_id"]
        print(f"[{tag}] resuming existing batch {batch_id}")
        return batch_id
    lines = _build_lines(judge, model, items, max_tokens,
                         reasoning_effort=reasoning_effort, verbosity=verbosity, seed=seed)
    if not lines:
        return None
    batch_id = submit_batch(client, lines, work_dir, tag)
    print(f"[{tag}] submitted batch {batch_id} ({len(lines)} requests)")
    return batch_id


def collect_batch_judging(
    client,
    judge: Judge,
    items: list[tuple[Query, str]],
    *,
    work_dir: str,
    tag: str = "judge",
    poll_interval: int = 30,
) -> dict:
    """Wait on the batch submitted for ``work_dir``, fetch + parse its results,
    and return the metrics dict.

    ``items`` MUST be the same, in the same order, as the submit call — custom_ids
    are built from each item's position and, for a multi-call judge, from the
    hypothesis itself. (Both phases rebuild items deterministically from the
    same predictions.jsonl, so they line up.)
    """
    state_path = os.path.join(work_dir, f"{tag}_state.json")
    if not os.path.isfile(state_path):
        if not items:
            zero = SimpleNamespace(input_tokens=0, output_tokens=0, calls=0)
            return judge.build_metrics(items, [], zero)
        raise RuntimeError(
            f"no submitted batch state in {work_dir} — run the submit phase first")
    with open(state_path) as f:
        batch_id = json.load(f)["batch_id"]
    batch = wait_for_batch(client, batch_id, poll_interval)
    results = fetch_results(client, batch)
    return _aggregate(judge, items, results, tag)


def run_batch_judging(
    client,
    judge: Judge,
    model: str,
    items: list[tuple[Query, str]],
    *,
    work_dir: str,
    tag: str = "judge",
    poll_interval: int = 30,
    max_tokens: int = 1024,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    seed: int | None = None,
) -> dict:
    """Submit + wait + aggregate in one call (single-file standalone path).

    The driver instead calls submit_batch_judging for every file, then
    collect_batch_judging for every file, so the jobs run concurrently.
    """
    submit_batch_judging(
        client, judge, model, items, work_dir=work_dir, tag=tag, max_tokens=max_tokens,
        reasoning_effort=reasoning_effort, verbosity=verbosity, seed=seed,
    )
    return collect_batch_judging(
        client, judge, items, work_dir=work_dir, tag=tag, poll_interval=poll_interval)
