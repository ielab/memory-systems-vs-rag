"""
evalgrid/systems/simplemem/controller.py
----------------------------------------
An LLM client for the vendored SimpleMem, speaking the OpenAI-compatible API
this repo serves everything else through, and counting the tokens it spends.

SimpleMem's own ``LLMClient`` cannot be used as-is for two reasons.

1. **It cannot be metered.** ``USE_STREAMING`` defaults to True, so every call
   goes through ``_handle_streaming_response``, which concatenates deltas and
   drops the response object — and a stream carries no ``usage`` block at all
   unless ``stream_options={"include_usage": true}`` is requested, which
   upstream never does. This repo's cost accounting (``layer_usage`` ->
   ingest_usage.json) is built on token counts, and SimpleMem's whole claim is a
   cost profile, so unmetered calls would defeat the point of running it.
2. **It prints a bare newline after every call** (``llm_client.py:131``), on a
   path that already runs 1 call per window at ingest and 5-8 per question at
   answer time.

So this class implements the two methods the vendored components actually call
on their ``llm_client`` — ``chat_completion(messages, temperature,
response_format, max_retries) -> str`` and ``extract_json(text)`` — and is
assigned over ``SimpleMemSystem.llm_client`` plus the three components that hold
their own reference to it. Duck-typed rather than subclassed on purpose:
subclassing would make importing this module require SimpleMem's dependency
tree (lancedb, tantivy, sentence-transformers).

Non-streaming is the only behavioural difference from upstream's request. Same
model, same messages, same temperature, same ``response_format`` — only the
transport changes, and with it the availability of ``usage``.

``extract_json`` is delegated to a real upstream ``LLMClient`` instance rather
than reimplemented: it is 100 lines of fence-stripping and brace-balancing that
decides what does and does not become a memory, and a divergent copy here would
silently change what SimpleMem extracts. That instance's OpenAI handle is the
one this class calls through, so there is exactly one client.

JSON mode degrades rather than fails. When ``USE_JSON_FORMAT`` is on, upstream
asks for ``response_format={"type": "json_object"}``; OpenAI supports it and
most vLLM builds do, but not all. On the first rejection we drop the parameter
and stay in plain mode for the rest of the run — upstream's ``extract_json``
scrapes JSON out of surrounding text anyway. See the config for why that setting
is off by default.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from evalgrid.models.base import TokenUsage

# Errors that mean "this server will not do json_object", as opposed to a
# transient failure worth retrying. Matched on the message because the OpenAI
# client raises the same BadRequestError class for both.
_JSON_MODE_REJECTION_MARKERS = (
    "response_format",
    "json_object",
    "guided",
    "structured output",
)


class OpenAICompatLLMClient:
    """SimpleMem's ``LLMClient`` interface over an OpenAI-compatible endpoint.

    Args:
        model: model id the endpoint serves.
        base_url: e.g. http://localhost:8000/v1. None targets OpenAI proper.
        api_key: vLLM accepts any non-empty string; falls back to
            OPENAI_API_KEY, then to "token".
        max_tokens: per-call output budget, or None to send none — which is what
            upstream does. The extraction call returns a JSON array covering a
            whole window, and a budget that truncates it turns into a parse
            failure, three retries, and a silently dropped window (see
            SimpleMemBase.empty_windows).

            None does NOT avoid that, though, which is why the config no longer
            ships it: with no budget the context window becomes the budget, and
            the clip lands there instead — costing a full window of output
            tokens per attempt rather than the configured cap. Prefer a cap that
            clears the largest legitimate completion; see
            evalgrid/configs/simplemem/default.yaml for the sizing and
            our development notes for what makes the replies run away.
        json_mode: "auto" (send response_format when a caller asks for it, fall
            back on rejection), "on" (a rejection is an error), or "off" (never
            send it, whatever ``USE_JSON_FORMAT`` says).
        temperature: pins EVERY memory call to this value, overriding the one
            each vendored call site passes. None restores upstream's behaviour,
            where the temperature is a property of the call site rather than of
            the run (0.1 for extraction and the adequacy checks, 0.2-0.3 for
            query planning and the reflection rewrites).

            The config ships 0, from the paper's Appendix B.2 Table 7 —
            "Model Backend: gpt-4.1-mini (temperature = 0.0)". Table 7 states
            that for stage 1 only, and this pins the query-side calls too:
            those temperatures are undocumented in the paper, and a planner
            that rewrites a question differently on every run makes a native
            answer unreproducible for a reason nothing published asked for.
        max_retries: attempts per call on transient failures. Upstream's own
            default is 3, with 1s/2s/4s backoff; kept.
    """

    def __init__(self, model: str, *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, max_tokens: Optional[int] = None,
                 json_mode: str = "auto", temperature: Optional[float] = None,
                 max_retries: int = 1):
        # Imported here, not at module scope: reaching vendor/simplemem pulls in
        # lancedb, pyarrow and sentence-transformers through the package init,
        # none of which may be a prerequisite for running some other system.
        from simplemem.core.utils.llm_client import LLMClient

        if json_mode not in ("auto", "on", "off"):
            raise ValueError(
                f"json_mode must be 'auto', 'on' or 'off', got {json_mode!r}"
            )

        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.temperature = None if temperature is None else float(temperature)
        self.max_retries = max_retries
        self.token_usage = TokenUsage()
        #: Completions the server stopped at max_tokens. See _call.
        self.truncated_calls = 0
        #: Calls that exhausted their retries and raised. token_usage counts only
        #: calls that returned, so without this a run that rejected half its
        #: requests still reports a clean cost table.
        self.failed_calls = 0
        #: Non-empty completions that no JSON could be scraped out of. See
        #: extract_json — these are silently "no memories", not errors.
        self.json_failures = 0

        # Whether to attach response_format on the next call. In "auto" this
        # flips to False for good the first time a server rejects it.
        self._send_json_mode = json_mode != "off"

        # Attributes the vendored components read off their llm_client. Both are
        # False here by construction: streaming is what makes the call
        # unmeterable, and enable_thinking is only ever consulted to build the
        # dashscope-specific extra_body, which we never send.
        self.use_streaming = False
        self.enable_thinking = False

        # Upstream's client, held for extract_json and for its OpenAI handle.
        self._upstream = LLMClient(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "token",
            model=model,
            base_url=base_url,
            enable_thinking=False,
            use_streaming=False,
        )
        self.client = self._upstream.client

    # ------------------------------------------------------------------
    # Upstream's LLMClient interface
    # ------------------------------------------------------------------

    def chat_completion(self, messages: list[dict], temperature: float = 0.2,
                        response_format: Optional[dict] = None,
                        max_retries: Optional[int] = None) -> str:
        """Return the completion text. Signature matches upstream's client.

        Every vendored call site passes its own ``temperature`` (0.1 for
        extraction and the adequacy checks, 0.2-0.3 for query planning), so the
        signature's own default is never the one in force — it exists to keep
        this substitutable for upstream's client.

        ``self.temperature`` overrides all of them when the config sets one,
        which is how the paper's temperature = 0 reaches call sites that hold
        their value inline. Left None, each call site keeps its own.
        """
        if self.temperature is not None:
            temperature = self.temperature
        attempts = self.max_retries if max_retries is None else max_retries
        last_exc: Optional[Exception] = None

        # A dropped json_mode does not consume an attempt: it changes the
        # request rather than reacting to a flaky server.
        attempt = 0
        while attempt < attempts:
            try:
                return self._call(messages, response_format, temperature)
            except Exception as e:
                if self._is_json_mode_rejection(e) and self._send_json_mode:
                    if self.json_mode == "on":
                        raise RuntimeError(
                            f"json_mode='on' but the endpoint rejected "
                            f"response_format={{'type': 'json_object'}}: {e}. Set "
                            f"json_mode: auto in the simplemem config to fall back to "
                            f"plain JSON prompting."
                        ) from e
                    # "auto": stop asking for JSON mode and send this call plain.
                    self._send_json_mode = False
                    continue
                last_exc = e
                attempt += 1
                if attempt < attempts:
                    time.sleep(2 ** (attempt - 1))

        self.failed_calls += 1
        # Ordinary Exception, not LLMCallFailed — see the a_mem controller for
        # the reasoning. SimpleMem's only `while True` (evolver/manager.py:1704)
        # is a graph traversal, not a retry loop, so nothing upstream will
        # re-send this call. "Send it once" comes from max_retries=1.
        raise RuntimeError(
            f"SimpleMem memory LLM call failed after {attempts} attempt(s) and was "
            f"not retried further: {last_exc}"
        ) from last_exc

    def extract_json(self, text: str) -> Any:
        """Upstream's JSON scraper, unchanged — but counted. See the module docstring.

        Deliberately NOT made more robust. Every system here already has a
        repair path of its own (upstream's brace scan here, json_repair in
        hypermem, fence-stripping in a_mem and mem0), and adding a better one
        would change what the benchmark measures: a model that cannot hold a
        JSON contract under a system's prompt IS that system's result, and the
        system that recovers from it is doing real work the others are not.

        What was missing is the number. A scrape that comes back empty produces
        no entries and raises nothing, so a window whose extraction was
        unparseable and a window that genuinely held no facts were the same
        thing in the store. simplemem's meta.json reports ``empty_windows``,
        which conflated exactly those two; this splits the parse failures out.
        """
        result = self._upstream.extract_json(text)
        if result in (None, [], {}) and (text or "").strip():
            # Non-empty completion, nothing parseable out of it.
            self.json_failures += 1
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call(self, messages: list[dict], response_format: Optional[dict],
              temperature: float) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if response_format is not None and self._send_json_mode:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.token_usage.add(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        else:
            self.token_usage.calls += 1

        # Upstream retries a window three times on a JSON parse failure and
        # then returns [], which this adapter counts as an empty_window — but
        # a reply clipped at max_tokens fails to parse for a reason a retry
        # cannot fix, and the two are indistinguishable in that counter.
        if (response.choices[0].finish_reason or "") == "length":
            self.truncated_calls += 1

        return response.choices[0].message.content or ""

    @staticmethod
    def _is_json_mode_rejection(exc: Exception) -> bool:
        """Whether ``exc`` says the server won't honour response_format.

        Only 400-class errors qualify: a 500 or a connection drop while sending
        response_format is transient, and dropping JSON mode because of one
        would silently degrade every later call in the run.
        """
        status = getattr(exc, "status_code", None)
        if status is not None and not (400 <= int(status) < 500):
            return False
        message = str(exc).lower()
        return any(marker in message for marker in _JSON_MODE_REJECTION_MARKERS)
