"""
evalgrid/systems/hypermem/provider.py
-------------------------------------
An LLM provider for the vendored HyperMem, speaking the plain OpenAI-compatible
API this repo serves everything else through.

HyperMem's own ``OpenAIProvider`` cannot be used here. It is an OpenRouter
client written against ``aiohttp`` directly, and it puts two OpenRouter-specific
fields in every request body:

    "provider": <dict or None>       # OpenRouter routing directive
    "response_format": <dict or None>

Both are sent unconditionally, including as explicit ``null``. vLLM validates
its request bodies with pydantic and does not know ``provider``; a literal
``"response_format": null`` is likewise not what the OpenAI schema expects.
Whether a given vLLM build tolerates them is exactly the kind of thing that
fails 40 minutes into an ingest, so this provider sends neither. It also counts
tokens — upstream's does too, but only when ``enable_stats=True``, which no
caller sets, and this repo's cost accounting (``layer_usage`` ->
ingest_usage.json) is built on token counts.

The interface is upstream's ``LLMProvider`` protocol (``hypermem/llm/protocol.py``):
``async generate(prompt, temperature=None, extra_body=None, response_format=None)``
plus ``test_connection`` and ``__repr__``. Duck-typed on purpose — it is a
``typing.Protocol``, and subclassing the concrete provider would drag in aiohttp.

**Why a synchronous client inside an async method.** Every call site is
``await self.llm_provider.generate(...)`` and our orchestration is strictly
sequential (see system.py — upstream's ``asyncio.gather`` fan-out lives in the
stage scripts, not in the extractors). Driving a sync client means each
``asyncio.run`` in system.py is self-contained: no httpx connection pool
outliving the event loop that created it, which is the standard way an
``AsyncOpenAI`` client reused across ``asyncio.run`` calls dies with "Event loop
is closed". If concurrency is ever added here, this is the thing to revisit.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Optional

from evalgrid.lib.errors import LLMCallFailed
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

# 400s that mention a marker for a reason that has nothing to do with structured
# output. A context-length rejection names ``max_tokens`` and, on some builds,
# echoes the request body — so matching the markers alone would read "the server
# refuses json_object", latch _send_json_mode off for the whole process, and
# silently downgrade every later call to plain prompting. Checked first.
_NOT_A_JSON_MODE_REJECTION = (
    "maximum context length",
    "context length exceeded",
    "reduce the length of the",
    "please reduce",
)

# Substrings of a 400 that no number of retries can change: the request is
# arithmetically impossible against this server's window. See FatalLLMError.
_FATAL_MARKERS = (
    "maximum context length",
    "context length exceeded",
)


#: Backwards-compatible alias. The type now lives in evalgrid/lib/errors.py
#: because the no-retry rule is the harness's, not hypermem's.
FatalLLMError = LLMCallFailed


class OpenAICompatProvider:
    """HyperMem's LLMProvider over an OpenAI-compatible chat endpoint.

    Args:
        model: model id the endpoint serves.
        base_url: e.g. http://localhost:8000/v1. None targets OpenAI proper.
        api_key: vLLM accepts any non-empty string; falls back to
            OPENAI_API_KEY, then to "token".
        temperature: upstream's vllm profile uses 0 for every extraction call.
        max_tokens: per-call output budget. Upstream's vllm profile sets 20000;
            the extraction prompts return long JSON (a topic's whole fact set)
            and a budget that truncates one costs a full retry cycle.
        json_mode: "auto" (send response_format when a caller asks for it, fall
            back on rejection), "on" (a rejection is an error), or "off".
        max_retries: attempts per call on transient failures.
    """

    def __init__(self, model: str, *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: Optional[int] = 20000, json_mode: str = "auto",
                 max_retries: int = 1):
        from openai import OpenAI

        if json_mode not in ("auto", "on", "off"):
            raise ValueError(f"json_mode must be 'auto', 'on' or 'off', got {json_mode!r}")

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.token_usage = TokenUsage()
        #: Completions the server stopped at max_tokens. See _call.
        self.truncated_calls = 0
        #: generate() calls that exhausted their retries and raised. token_usage
        #: cannot show these — a failed request posts no usage — so without this
        #: a run that rejected half its calls reports a clean cost table.
        self.failed_calls = 0
        #: generate() calls abandoned as unsatisfiable. See FatalLLMError.
        self.fatal_calls = 0
        #: True once a json_mode="auto" run has fallen back to plain prompting.
        #: Reported in meta.json: which mode a store was built under changes what
        #: its extractions look like, and it used to be unknowable afterwards.
        self.json_mode_fallback = False

        self._send_json_mode = json_mode != "off"
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "token",
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # Upstream's LLMProvider protocol
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       extra_body: Optional[dict] = None,
                       response_format: Optional[dict] = None) -> str:
        """Return the completion text.

        ``extra_body`` is accepted because upstream's signature has it, and
        forwarded, but no vendored call site passes one.
        """
        # Load-bearing, not decorative. system.py bounds upstream's five
        # unbounded `while True:` retry loops by running each extractor call
        # under asyncio.wait_for — but a coroutine that never awaits anything
        # never returns control to the event loop, so the timeout could never
        # fire and the loop would still run forever. This yield is the point at
        # which the cancellation lands: between two consecutive LLM calls of a
        # retry loop. See system.py::HyperMemBase._run.
        await asyncio.sleep(0)

        temp = self.temperature if temperature is None else temperature
        budget = self.max_tokens if max_tokens is None else max_tokens
        last_exc: Optional[Exception] = None

        # A dropped json_mode does not consume one of the attempts. It changes
        # the request rather than reacting to a flaky server, so counting it as
        # a try left a real transient failure with two chances instead of three.
        attempt = 0
        while attempt < self.max_retries:
            try:
                return self._call(prompt, temp, budget, extra_body, response_format)
            except Exception as e:
                if self._is_fatal(e):
                    # Deterministic: every retry sends the same too-long request
                    # to the same window. Skip the remaining attempts entirely —
                    # they cannot do anything but fail identically.
                    self.fatal_calls += 1
                    self.failed_calls += 1
                    raise LLMCallFailed(
                        f"HyperMem memory LLM call is unsatisfiable against this endpoint "
                        f"and was not retried: {e}. The request reserves max_tokens="
                        f"{budget} out of the served window, so a prompt this size can "
                        f"never fit. Lower memory_llm.max_tokens in the hypermem config, "
                        f"or serve the model with a larger --max-model-len (see "
                        f"scripts/ingest.sh, CTX_LONG).",
                        system="hypermem", deterministic=True, cause=e,
                    ) from e
                if self._is_json_mode_rejection(e) and self._send_json_mode:
                    if self.json_mode == "on":
                        raise LLMCallFailed(
                            f"json_mode='on' but the endpoint rejected "
                            f"response_format={{'type': 'json_object'}}: {e}. Set "
                            f"json_mode: auto in the hypermem config to fall back to plain "
                            f"prompting — every extractor already repairs malformed JSON "
                            f"with json_repair.",
                            system="hypermem", deterministic=True, cause=e,
                        ) from e
                    # "auto": stop asking for JSON mode and send this call plain.
                    # Not a retry — it is a different request, so it does not
                    # consume an attempt.
                    self._send_json_mode = False
                    self.json_mode_fallback = True
                    print("[hypermem] endpoint rejected json_object; falling back to plain "
                          f"prompting for the rest of this run: {e}", file=sys.stderr)
                    continue
                last_exc = e
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))

        self.failed_calls += 1
        # LLMCallFailed, not RuntimeError: upstream's five `while True: ... except
        # Exception` loops would otherwise catch this and send the same request
        # again, which is exactly the behaviour max_retries is meant to bound.
        # With max_retries=1 (the default) this is the first and only attempt.
        raise LLMCallFailed(
            f"HyperMem memory LLM call failed after {self.max_retries} attempt(s) "
            f"and was not retried further: {last_exc}",
            system="hypermem", deterministic=False, cause=last_exc,
        ) from last_exc

    async def test_connection(self) -> bool:
        try:
            return len(await self.generate("Hello", temperature=0.1, max_tokens=8)) > 0
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"OpenAICompatProvider(model={self.model}, base_url={self.base_url})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call(self, prompt: str, temperature: float, max_tokens: Optional[int],
              extra_body: Optional[dict], response_format: Optional[dict]) -> str:
        # Upstream sends the prompt as a single user message; kept, since the
        # prompts are written as complete standalone instructions.
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None and self._send_json_mode:
            kwargs["response_format"] = response_format
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self._client.chat.completions.create(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.token_usage.add(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        else:
            self.token_usage.calls += 1

        # A completion stopped at max_tokens cannot be valid JSON, and every
        # consumer downstream runs it through json-repair — which returns
        # SOMETHING for almost any input, so a degenerate repetition that ran
        # the full budget is indistinguishable in the output from a clean
        # extraction. The smoke run had exactly one (20k output tokens of a
        # repeating list, 142s), and meta.json still reported zero failures.
        # Counted here so it reaches meta.json as a number.
        if (response.choices[0].finish_reason or "") == "length":
            self.truncated_calls += 1

        return response.choices[0].message.content or ""

    @staticmethod
    def _is_fatal(exc: Exception) -> bool:
        """Whether ``exc`` is a 400 that retrying cannot fix. See FatalLLMError."""
        status = getattr(exc, "status_code", None)
        if status is not None and not (400 <= int(status) < 500):
            return False
        message = str(exc).lower()
        return any(marker in message for marker in _FATAL_MARKERS)

    @staticmethod
    def _is_json_mode_rejection(exc: Exception) -> bool:
        """Whether ``exc`` says the server won't honour response_format.

        Only 400-class errors qualify: a 500 or a connection drop while sending
        response_format is transient, and dropping JSON mode because of one
        would silently degrade every later call in the run.

        The exclusion list is load-bearing. This sets a latch that is never
        unset, from a substring match over an error message the endpoint wrote —
        so a 400 that merely *mentions* one of the markers for an unrelated
        reason (a context-length rejection naming max_tokens, a build that echoes
        the request body) would turn structured output off for the whole process
        with nothing recording that it happened.
        """
        status = getattr(exc, "status_code", None)
        if status is not None and not (400 <= int(status) < 500):
            return False
        message = str(exc).lower()
        if any(marker in message for marker in _NOT_A_JSON_MODE_REJECTION):
            return False
        return any(marker in message for marker in _JSON_MODE_REJECTION_MARKERS)
