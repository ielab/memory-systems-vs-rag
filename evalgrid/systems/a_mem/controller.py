"""
evalgrid/systems/a_mem/controller.py
------------------------------------
An LLM controller for the vendored A-Mem, speaking the OpenAI-compatible API
this repo serves everything else through.

A-Mem's own controllers cannot be used here. ``OpenAIController`` builds
``OpenAI(api_key=...)`` with no ``base_url``, so it can only reach api.openai.com;
``SGLangController`` POSTs to SGLang's native ``/generate``, not the
OpenAI-compatible route vLLM exposes. Neither counts tokens, and this repo's
cost accounting (``layer_usage`` -> ingest_usage.json) is built on token counts.

So we implement upstream's ``BaseLLMController`` interface —
``get_completion(prompt, response_format, temperature) -> str`` — against the
same endpoint shape as evalgrid/models/, and assign an instance over
``AgenticMemorySystem.llm_controller.llm``. Duck-typed on purpose: subclassing
upstream's ABC would make importing this module require A-Mem's dependency tree.

Structured output degrades rather than fails. A-Mem asks for
``response_format={"type": "json_schema", ...}``, which OpenAI supports and most
vLLM builds support, but not all. On the first rejection we drop the parameter
and stay in plain mode for the rest of the run; upstream's parsers extract JSON
from surrounding text anyway (``process_memory`` brace-scans, and
``analyze_content`` strips markdown fences), so a model that emits clean JSON on
its own still works.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from evalgrid.models.base import TokenUsage

# Upstream sends this as the system message on every structured call. Kept
# byte-identical: it is part of the prompt A-Mem's published numbers came from.
_SYSTEM_MESSAGE = "You must respond with a JSON object."

# Errors that mean "this server will not do json_schema", as opposed to a
# transient failure worth retrying. Matched on the message because the OpenAI
# client raises the same BadRequestError class for both.
_SCHEMA_REJECTION_MARKERS = (
    "response_format",
    "json_schema",
    "guided",
    "structured output",
)

# 400s that mention a marker for a reason that has nothing to do with structured
# output. Checked first — see _is_schema_rejection.
_NOT_A_SCHEMA_REJECTION = (
    "maximum context length",
    "context length exceeded",
    "reduce the length of the",
    "please reduce",
)


class OpenAICompatController:
    """A-Mem's controller interface over an OpenAI-compatible chat endpoint.

    Args:
        model: model id the endpoint serves.
        base_url: e.g. http://localhost:8000/v1. None targets OpenAI proper.
        api_key: vLLM accepts any non-empty string; falls back to
            OPENAI_API_KEY, then to "token".
        max_tokens: per-call output budget. Upstream pins 1000 for every memory
            call; keep it unless you know why you are changing it.
        temperature: upstream's default for memory calls is 0.7.
        structured_output: "auto" (try json_schema, fall back on rejection),
            "on" (require it — a rejection is an error), or "off" (never send it).
        max_retries: attempts per call on transient failures.
    """

    def __init__(self, model: str, *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, max_tokens: int = 1000,
                 temperature: float = 0.7, structured_output: str = "auto",
                 max_retries: int = 1):
        from openai import OpenAI

        if structured_output not in ("auto", "on", "off"):
            raise ValueError(
                f"structured_output must be 'auto', 'on' or 'off', got {structured_output!r}"
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.structured_output = structured_output
        self.max_retries = max_retries
        self.token_usage = TokenUsage()
        #: Completions the server stopped at max_tokens. See _call.
        self.truncated_calls = 0
        #: Calls that exhausted their retries and raised. token_usage counts only
        #: calls that returned, so without this a run that rejected half its
        #: requests still reports a clean cost table.
        self.failed_calls = 0
        #: True once an "auto" run has fallen back to plain JSON prompting.
        self.schema_fallback = False

        # Whether to attach response_format on the next call. In "auto" this
        # flips to False for good the first time a server rejects it.
        self._send_schema = structured_output != "off"

        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "token",
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # Upstream's BaseLLMController interface
    # ------------------------------------------------------------------

    def get_completion(self, prompt: str, response_format: Optional[dict] = None,
                       temperature: Optional[float] = None) -> str:
        """Return the completion text. Signature matches upstream's controllers.

        ``response_format`` is optional here but positional in upstream's calls,
        which always pass it — the default exists only so this controller can
        also serve the plain-text call sites in memory_layer_robust.
        """
        temp = self.temperature if temperature is None else temperature
        last_exc: Optional[Exception] = None

        # A dropped schema does not consume one of the attempts: it changes the
        # request rather than reacting to a flaky server, so counting it as a try
        # left a real transient failure with two chances instead of three.
        attempt = 0
        while attempt < self.max_retries:
            try:
                return self._call(prompt, response_format, temp)
            except Exception as e:
                if self._is_schema_rejection(e) and self._send_schema:
                    if self.structured_output == "on":
                        raise RuntimeError(
                            f"structured_output='on' but the endpoint rejected json_schema: {e}. "
                            f"Set structured_output: auto in the a_mem config to fall back to "
                            f"plain JSON prompting."
                        ) from e
                    # "auto": stop asking for schemas and retry this call plain.
                    self._send_schema = False
                    self.schema_fallback = True
                    print("[a_mem] endpoint rejected json_schema; falling back to plain JSON "
                          f"prompting for the rest of this run: {e}", file=sys.stderr)
                    continue
                last_exc = e
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))

        self.failed_calls += 1
        # An ordinary Exception, NOT the harness's LLMCallFailed. A-Mem has no
        # unbounded retry loop for that to protect against (zero `while True:`
        # in vendor/a_mem), and upstream's `except Exception` at
        # memory_layer.py:393 is load-bearing: it is what stores the note with
        # empty keywords and increments degraded_notes. Raising past it would
        # turn one unusable metadata call into a lost sample of 550 notes.
        # "Send it once" is already guaranteed by max_retries=1 above.
        raise RuntimeError(
            f"A-Mem memory LLM call failed after {self.max_retries} attempt(s) and was "
            f"not retried further: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call(self, prompt: str, response_format: Optional[dict], temperature: float) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None and self._send_schema:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.token_usage.add(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        else:
            self.token_usage.calls += 1

        # A completion stopped at max_tokens is not an error at this layer and
        # not an error at the next one either: upstream json-parses the reply
        # inside a try/except and falls back to an empty result, so a clipped
        # evolution silently becomes "no evolution". Counted here (surfaced in
        # meta.json) so that failure mode has a number.
        if (response.choices[0].finish_reason or "") == "length":
            self.truncated_calls += 1

        return response.choices[0].message.content or ""

    @staticmethod
    def _is_schema_rejection(exc: Exception) -> bool:
        """Whether ``exc`` says the server won't honour response_format.

        Only 400-class errors qualify: a 500 or a connection drop while sending
        a schema is transient, and dropping structured output because of one
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
        if any(marker in message for marker in _NOT_A_SCHEMA_REJECTION):
            return False
        return any(marker in message for marker in _SCHEMA_REJECTION_MARKERS)
