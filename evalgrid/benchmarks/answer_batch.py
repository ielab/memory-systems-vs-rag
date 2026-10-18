"""
benchmarks/answer_batch.py
---------------------------
Batch-API variant of evalgrid/pipeline/answer.py's live generation loop
(``_run`` / ``_answer_one``), for OpenAI-hosted generators (``--batch``).

Answering a grid cell is 400-500 independent chat calls that nobody waits on,
which is what OpenAI's Batch API is for: half the price, and a separate queue
that the synchronous per-minute rate limit does not touch.

WHAT IS SHARED WITH THE LIVE PATH, AND WHAT IS NOT. The prompt is the caller's
``build_prompt`` over the same (facts, ctx) that live answering would send, and
the request body is benchmarks/judge_batch._request_body — the same gpt-5
token-kwarg/temperature/reasoning quirks as OpenAIGenerator — so a batch cell
asks the model exactly what a live cell would. The records written back have
the live schema, field for field, with two differences a reader must know:

  latency_sec is None. A batch line has no call duration; the time between
  submit and collect is queueing, not the model. Latency for an OpenAI
  generator has to come from a live run, and answer_summary.json's latency sums
  treat None as absent.

  reasoning_tokens is MEASURED (usage.completion_tokens_details), where the live
  OpenAI path has historically left it None.

custom_id IS THE question_id, not a position. openai_batch.py warns that
positional ids misattribute when a resumed run's item list differs from the
submitting run's; keying on question_id makes collect a lookup, so an item
missing from the results is an error for THAT question and never a shifted
answer for its neighbour.

State lives in <cell>/_batch_answer_<gen-slug>/answer_state.json. Submitting
again while it exists resumes that batch instead of paying twice. After the
caller has written the answers, ``retire_answer_batch`` renames it, so a later
``--batch`` pass over whatever failed submits a FRESH batch rather than
re-reading the old one's failures.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from evalgrid.benchmarks.judge_batch import _request_body
from evalgrid.benchmarks.openai_batch import (
    MAX_BATCH_REQUESTS,
    _batch_line,
    error_text,
    submit_batch,
    wait_for_batch,
)
from evalgrid.lib.cost import query_cost

TAG = "answer"

#: The Batch API's input-file cap. Budget cells are far below it (a 2000-token
#: budget x 500 questions is a few MB); a full-context row is not — ~110k tokens
#: x 500 LongMemEval questions is over 200 MB — so it is checked before upload
#: rather than discovered as an opaque upload error.
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024


def work_dir_for(out_run_dir: str, gen_slug: str) -> str:
    return os.path.join(out_run_dir, f"_batch_answer_{gen_slug}")


def _state_path(work_dir: str) -> str:
    return os.path.join(work_dir, f"{TAG}_state.json")


def submit_answer_batch(
    client,
    items: list[tuple],
    build_prompt,
    *,
    model: str,
    max_tokens: int,
    work_dir: str,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: float = 0.0,
) -> Optional[str]:
    """Build and submit one batch for ``items`` WITHOUT waiting; return its id.

    ``items`` are answer.py's (question_id, question, question_date, facts, ctx)
    tuples. Idempotent: an existing state file is resumed, not resubmitted.
    """
    state_path = _state_path(work_dir)
    if os.path.isfile(state_path):
        with open(state_path) as f:
            batch_id = json.load(f)["batch_id"]
        print(f"[{TAG}] resuming existing batch {batch_id}")
        return batch_id
    if not items:
        return None

    lines: list[dict] = []
    seen: set[str] = set()
    for question_id, question, question_date, facts, _ctx in items:
        if question_id in seen:
            raise ValueError(f"duplicate question_id {question_id!r} — it is the batch "
                             f"custom_id, so replies would misattribute")
        seen.add(question_id)
        body = _request_body(
            build_prompt(question, question_date, facts), model, max_tokens,
            reasoning_effort=reasoning_effort, verbosity=verbosity, seed=seed,
            temperature=temperature,
        )
        lines.append(_batch_line(question_id, body))
    if len(lines) > MAX_BATCH_REQUESTS:
        raise ValueError(f"{len(lines)} requests exceeds the Batch API's "
                         f"{MAX_BATCH_REQUESTS}-request limit")
    size = sum(len(json.dumps(line, ensure_ascii=False).encode()) + 1 for line in lines)
    if size > MAX_BATCH_FILE_BYTES:
        raise ValueError(
            f"batch input would be {size / 2**20:.0f} MB, over the Batch API's "
            f"{MAX_BATCH_FILE_BYTES // 2**20} MB file limit — answer this cell live, "
            f"or in pieces with --limit")

    batch_id = submit_batch(client, lines, work_dir, TAG)
    print(f"[{TAG}] submitted batch {batch_id} ({len(lines)} requests, {size / 2**20:.1f} MB)")
    return batch_id


def _fetch_rows(client, batch) -> dict[str, dict]:
    """custom_id -> raw result row, output file first (it wins over the error file)."""
    rows: dict[str, dict] = {}
    for file_id in (batch.output_file_id, batch.error_file_id):
        if not file_id:
            continue
        for line in client.files.content(file_id).text.splitlines():
            if line.strip():
                row = json.loads(line)
                rows.setdefault(row["custom_id"], row)
    return rows


def _resolve(question_id: str, row: Optional[dict], ctx: dict, batch_id: str,
             max_tokens: int, usage_sink) -> tuple:
    """One result row -> answer.py's (prediction, stats, error) triple."""
    if row is None:
        return None, None, {
            "question_id": question_id, "latency_sec": None, "batch_id": batch_id,
            "error": "missing from the batch's output and error files",
            "traceback": None, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": None,
        }
    response = row.get("response") or {}
    body = response.get("body") or {}
    status = response.get("status_code")
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    text = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason")
    usage = body.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens") if usage else None
    if usage:
        # Paid for whether or not it produced an answer — the live generator
        # counts a failed call the same way before raising OutputCapError.
        usage_sink.add(input_tokens, output_tokens)

    failed = row.get("error") or (status is not None and status >= 400) or not text.strip()
    if failed:
        if not row.get("error") and finish == "length":
            # The live path's OutputCapError, in the same words.
            error = (f"returned empty content with finish_reason='length' — the output "
                     f"budget (max_tokens={max_tokens}) was exhausted, likely inside "
                     f"reasoning. Raise the caller's max-tokens.")
        else:
            error = error_text(row)
        return None, None, {
            "question_id": question_id, "latency_sec": None, "batch_id": batch_id,
            "error": error, "traceback": None,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "reasoning_tokens": reasoning,
        }
    # A completion cut at max_tokens with text left is KEPT, as live keeps it
    # (OpenAIGenerator counts it in truncated_calls rather than raising).
    stats = {
        "question_id": question_id,
        "latency_sec": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reasoning_tokens": reasoning,
        **ctx,
        **query_cost(answer=(input_tokens, output_tokens, 1)),
        "batch_id": batch_id,
        "finish_reason": finish,
    }
    return {"question_id": question_id, "hypothesis": text}, stats, None


def collect_answer_batch(
    client,
    items: list[tuple],
    *,
    work_dir: str,
    max_tokens: int,
    usage_sink,
    poll_interval: int = 30,
) -> list[tuple]:
    """Wait on this cell's submitted batch and return one (prediction, stats,
    error) triple per item, in ``items`` order — the shape ``_run`` returns.

    ``usage_sink`` is the generator's TokenUsage, so answer_summary.json's
    generator_cumulative_tokens_this_invocation reads the same as a live run's.
    """
    state_path = _state_path(work_dir)
    if not os.path.isfile(state_path):
        raise RuntimeError(f"no submitted batch in {work_dir} — run --batch-phase submit first")
    with open(state_path) as f:
        batch_id = json.load(f)["batch_id"]
    batch = wait_for_batch(client, batch_id, poll_interval)
    rows = _fetch_rows(client, batch)
    results = [_resolve(qid, rows.get(qid), ctx, batch_id, max_tokens, usage_sink)
               for qid, _q, _d, _f, ctx in items]
    missing = sum(1 for qid, *_ in items if qid not in rows)
    if missing:
        print(f"[{TAG}] ! {missing} of {len(items)} question(s) absent from batch {batch_id} "
              f"(submitted for a different question set?) — recorded as errors")
    return results


def retire_answer_batch(work_dir: str) -> None:
    """Mark this cell's batch as consumed, AFTER its answers are on disk.

    Renamed rather than deleted: the batch id stays on record beside the answers
    it produced. Without this, a later --batch run to redo the failed questions
    would resume the old batch and read back the same failures.
    """
    state_path = _state_path(work_dir)
    if not os.path.isfile(state_path):
        return
    with open(state_path) as f:
        batch_id = json.load(f)["batch_id"]
    os.replace(state_path, os.path.join(work_dir, f"{TAG}_state.{batch_id}.done.json"))
