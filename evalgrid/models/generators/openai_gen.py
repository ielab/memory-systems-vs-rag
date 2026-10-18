"""
models/generators/openai_gen.py
--------------------------------
Text generation via OpenAI chat-completions API.
Requires OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import os

import openai

from evalgrid.models.base import GenerationResult, GeneratorModel, OutputCapError


class OpenAIGenerator(GeneratorModel):
    """
    Args:
        model: e.g. "gpt-4o", "gpt-4o-mini", "gpt-5.5"
        api_key: defaults to OPENAI_API_KEY env var
        reasoning_effort: GPT-5 / o-series only — "none"|"low"|"medium"|"high"|"xhigh".
            None leaves it at the model default (medium). Ignored for models that
            don't support it, so it's safe to set unconditionally.
        verbosity: GPT-5 only — "low"|"medium"|"high"; constrains output length.
            None leaves it at the model default.
        seed: best-effort reproducibility — same seed + inputs + system_fingerprint
            yields (mostly) identical output. Accepted by all OpenAI chat models;
            None omits it. Reasoning models are still not bitwise-deterministic.
        temperature: sampling temperature; 0.0 = greedy. Used as the default for
            every generate() call unless a call passes its own. Dropped for
            gpt-5-family models, which only allow their default temperature.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        *,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        seed: int | None = None,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.seed = seed
        self.temperature = temperature
        self.client = openai.AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        """Newer OpenAI reasoning/GPT-5 chat models reject the legacy max_tokens field."""
        return model.startswith("o") or model.startswith("gpt-5")

    @staticmethod
    def _uses_default_temperature(model: str) -> bool:
        """Some newer OpenAI models only allow the default temperature."""
        return model.startswith("gpt-5")

    @staticmethod
    def _supports_reasoning_effort(model: str) -> bool:
        """GPT-5 family and o-series reasoning models accept reasoning_effort; older chat models reject it."""
        return model.startswith("gpt-5") or model.startswith("o")

    async def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2000,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        result = await self.generate_with_usage(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return result.text

    async def generate_with_usage(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2000,
        temperature: float | None = None,
        **kwargs,
    ) -> GenerationResult:
        # None = fall back to the instance temperature (the pinned default);
        # a per-call value still wins.
        temp = self.temperature if temperature is None else temperature
        token_kwarg = (
            {"max_completion_tokens": max_tokens}
            if self._uses_max_completion_tokens(self.model)
            else {"max_tokens": max_tokens}
        )
        temperature_kwarg = (
            {}
            if self._uses_default_temperature(self.model)
            else {"temperature": temp}
        )
        reasoning_kwarg = (
            {"reasoning_effort": self.reasoning_effort}
            if self.reasoning_effort and self._supports_reasoning_effort(self.model)
            else {}
        )
        verbosity_kwarg = (
            {"verbosity": self.verbosity}
            if self.verbosity and self.model.startswith("gpt-5")
            else {}
        )
        seed_kwarg = {"seed": self.seed} if self.seed is not None else {}
        # Merge so an explicit per-call kwarg overrides the instance default
        # rather than colliding into a duplicate-keyword TypeError.
        params = {
            **token_kwarg, **temperature_kwarg, **reasoning_kwarg,
            **verbosity_kwarg, **seed_kwarg, **kwargs,
        }
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )
        input_tokens = output_tokens = 0
        reasoning_tokens = None
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            details = getattr(response.usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            self.token_usage.add(input_tokens, output_tokens)
        choice = response.choices[0]
        text = choice.message.content or ""
        # The non-empty half of the same failure: a reply the model was cut off
        # mid-sentence still returns text, so nothing raises and nothing looks
        # wrong. OutputCapError below covers only the empty case, which is the
        # one that breaks callers immediately; this one just quietly stores a
        # half-finished answer, so it is counted rather than raised.
        if getattr(choice, "finish_reason", None) == "length":
            self.truncated_calls += 1
        # A reasoning model that spent max_completion_tokens inside its reasoning
        # returns empty content with finish_reason='length' and NO API error.
        # Silently reading that as "" corrupts downstream logic (an empty qrels
        # label, or a judge verdict scored WRONG). Normalise it to OutputCapError
        # so the retry/error paths can react instead of trusting the empty text.
        if not text.strip() and getattr(choice, "finish_reason", None) == "length":
            raise OutputCapError(
                f"OpenAI model {self.model!r} returned empty content with "
                f"finish_reason='length' — the output budget (max_tokens="
                f"{max_tokens}) was exhausted, likely inside reasoning. "
                f"Raise the caller's max-tokens.",
                input_tokens=input_tokens, output_tokens=output_tokens,
                reasoning_tokens=(reasoning_tokens if reasoning_tokens is not None
                                  else output_tokens),
            )
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )
