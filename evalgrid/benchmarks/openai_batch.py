"""
benchmarks/openai_batch.py
---------------------------
Transport-level helpers for OpenAI's Batch API — writing a job file, uploading
it, creating the batch, and polling to a terminal state.

The Batch API sidesteps the synchronous per-minute rate limit entirely (a
separate, much larger queue) and costs half as much, which suits any workload
here that is not latency sensitive.

Nothing in this module knows what is being asked of the model; callers build
their own request bodies and pass them through ``_batch_line``. Used by
benchmarks/judge_batch.py (``--batch`` on the judge stage).

Resumable across process restarts: ``submit_batch`` saves the batch_id to
<work_dir>/<tag>_state.json before polling begins, since the batch keeps running
on OpenAI's side regardless of whether this process stays attached — re-running
picks up the same batch instead of submitting (and paying) again.

This resume is only correct when the re-run's item set is IDENTICAL to the
submitting run's: custom_ids are positional, so if part of the work succeeded
and was written, the re-run's to_process shrinks and those same positional ids
map to different items than the resumed batch's results — silently
misattributing them. To retry after partial failures, drop the saved state
first so a fresh batch is submitted over only the unfinished items.
"""

from __future__ import annotations

import json
import os
import time

# OpenAI Batch API hard cap (confirmed via docs as of mid-2026): 50,000
# requests per batch. Splitting an oversized run is what --shard-id/
# --num-shards already does — no need for a second, parallel mechanism here.
MAX_BATCH_REQUESTS = 50_000

_TERMINAL_OK = {"completed"}
_TERMINAL_FAIL = {"failed", "expired", "cancelled"}


def _batch_line(custom_id: str, body: dict) -> dict:
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def submit_batch(client, lines: list[dict], work_dir: str, tag: str) -> str:
    """Write ``lines`` as a batch input file, upload it, create the batch, and
    save its state (batch_id) so a later run can resume instead of resubmitting."""
    os.makedirs(work_dir, exist_ok=True)
    input_path = os.path.join(work_dir, f"{tag}_input.jsonl")
    with open(input_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h",
    )

    state_path = os.path.join(work_dir, f"{tag}_state.json")
    with open(state_path, "w") as f:
        json.dump({"batch_id": batch.id, "input_file_id": uploaded.id}, f)
    return batch.id


def wait_for_batch(client, batch_id: str, poll_interval: int):
    """Block until ``batch_id`` reaches a terminal state, printing progress each tick."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [batch {batch_id}] {batch.status} "
              f"({counts.completed}/{counts.total} completed, {counts.failed} failed)")
        if batch.status in _TERMINAL_OK:
            return batch
        if batch.status in _TERMINAL_FAIL:
            raise RuntimeError(f"Batch {batch_id} ended with status {batch.status!r} — see {batch.errors}")
        time.sleep(poll_interval)


def error_text(row: dict) -> str:
    """A short, specific reason a batch line produced no verdict.

    Worth extracting rather than recording "failed": the three common causes
    need three different responses, and only the message tells them apart —
    a 429/500 is worth re-judging as-is, a content-filter refusal never will be,
    and finish_reason=length means the judge ran out of output budget and needs
    --judge-max-tokens raised before it can succeed.
    """
    err = row.get("error")
    if err:
        if isinstance(err, dict):
            return f"{err.get('code') or 'error'}: {err.get('message', '')}"[:500]
        return str(err)[:500]
    resp = row.get("response") or {}
    status = resp.get("status_code")
    body = resp.get("body") or {}
    if isinstance(body.get("error"), dict):
        e = body["error"]
        return f"{e.get('code') or status}: {e.get('message', '')}"[:500]
    choices = body.get("choices") or []
    if choices and choices[0].get("finish_reason") == "length":
        return ("finish_reason=length: the judge hit --judge-max-tokens before "
                "emitting a verdict")
    if choices and not (choices[0].get("message") or {}).get("content"):
        return f"empty content (finish_reason={choices[0].get('finish_reason')})"
    return f"HTTP {status}: no parseable reply"


def fetch_results(client, batch, *, with_body: bool = False):
    """Map custom_id -> (assistant reply text or None, usage dict or None, error or None).

    ``with_body=True`` appends the raw response body as a fourth element, for a
    caller that needs something this tuple does not carry, e.g. the
    ``system_fingerprint`` of the serving stack. Default stays a 3-tuple so the judges that
    unpack it are untouched.

    Fetches the batch's results and also keeps each line's token usage
    so the batch judge can report judge_tokens (the live path reads them off the
    generator; here they come from the batch response bodies).

    A line is a FAILURE, not a verdict, whenever it carries an error, a non-2xx
    status, or an empty/truncated completion. That last case is the one worth
    naming: a reply cut off at max_tokens comes back with content the parser
    reads as "not correct", so before this it scored as a wrong answer rather
    than as the budget problem it is.
    """
    out: dict[str, tuple[str | None, dict | None, str | None]] = {}
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row["custom_id"]
            body = (row.get("response") or {}).get("body") or {}
            status = (row.get("response") or {}).get("status_code")
            choices = body.get("choices") or []
            text = (choices[0].get("message") or {}).get("content") if choices else None
            truncated = bool(choices) and choices[0].get("finish_reason") == "length"
            failed = (row.get("error") or (status is not None and status >= 400)
                      or not text or truncated)
            entry = ((None, body.get("usage"), error_text(row)) if failed
                     else (text, body.get("usage"), None))
            out[custom_id] = (entry + (body,)) if with_body else entry
    if batch.error_file_id:
        content = client.files.content(batch.error_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            entry = (None, None, error_text(row))
            out.setdefault(row["custom_id"], (entry + ({},)) if with_body else entry)
    return out
