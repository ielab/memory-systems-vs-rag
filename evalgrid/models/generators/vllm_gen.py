"""
models/generators/vllm_gen.py
------------------------------
Text generation via a running vllm OpenAI-compatible server.

Start the server before using:
    vllm serve Qwen/Qwen3-8B --port 8000

The client speaks the OpenAI chat-completions API, so no vllm package
needs to be installed here — only `openai`.
"""

from __future__ import annotations

from pathlib import Path

import openai

from evalgrid.models.base import GenerationResult, GeneratorModel, OutputCapError

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_reasoning(text: str, finish_reason: str, opens_think: bool | None) -> str:
    """The answer part of a completion, with any reasoning trace removed.

    A thinking model served WITHOUT ``--reasoning-parser`` returns its reasoning
    inside ``content``. Every caller downstream (the judge, a memory system's
    parser, the relabeller) then reads the trace as though it were the answer.
    That was measured on Qwen3.8-27B's all_turns oracle (2026-09-19): 476/500
    LongMemEval and 359/400 BEAM predictions carried the whole trace. The BEAM
    event_ordering aligner saw 27 "answer lines" where 3 were written and scored
    the cell 0.16 against 0.64-0.71 for every non-thinking generator.

    Qwen3.8's chat template opens ``<think>`` inside the PROMPT, so the
    completion carries only the closing tag. That is why a completion stopped
    at max_tokens before ``</think>`` contains no tag at all and looks exactly
    like a plain answer. ``opens_think`` (from the chat template, see
    ``lib/tokens.prompt_opens_think``) is the only way to tell the two apart.
    All 65 such Qwen3.8 completions had hit the 2000-token cap.

    Returns the text after the last ``</think>``. Raises OutputCapError when the
    budget ran out inside the reasoning, because there is no answer to return and
    an empty or half-thought one must not be graded. Anything else is returned
    unchanged, so a non-thinking model's output is byte-identical.
    """
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1].strip()
    in_reasoning = text.lstrip().startswith(THINK_OPEN) or bool(opens_think)
    if in_reasoning and finish_reason == "length":
        raise OutputCapError(
            f"finish_reason='length' before {THINK_CLOSE}: the output budget was "
            f"exhausted inside reasoning, so there is no answer. Raise --max-tokens.")
    if text.lstrip().startswith(THINK_OPEN):
        # Opened and never closed, but the server says it stopped on its own.
        # Nothing after the tag is an answer either.
        raise OutputCapError(f"completion opened {THINK_OPEN} and never closed it "
                             f"(finish_reason={finish_reason!r}); no answer to return.")
    return text


class VLLMGenerator(GeneratorModel):
    """
    Args:
        model: model name as registered in the vllm server (e.g. "Qwen/Qwen3-8B")
        base_url: vllm server URL (default: http://localhost:8000/v1)
        api_key: vllm accepts any non-empty string
        seed: best-effort reproducibility — the vllm OpenAI-compatible server
            accepts the same `seed` field as OpenAI; same seed + inputs yields
            (mostly) identical output. None omits it.
        temperature: sampling temperature; 0.0 = greedy. Used as the default for
            every generate() call unless a call passes its own.
        opens_think: whether the chat template opens ``<think>`` in the prompt.
            None means detect it from the model's tokenizer on first use. When
            no tokenizer can be loaded, a cap-stopped completion without
            ``</think>`` is returned as it is (see ``split_reasoning``).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token",
        *,
        seed: int | None = None,
        temperature: float = 0.0,
        opens_think: bool | None = None,
    ):
        super().__init__()
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._opens_think = opens_think
        self._opens_think_checked = opens_think is not None

    @property
    def opens_think(self) -> bool | None:
        if not self._opens_think_checked:
            from evalgrid.lib.tokens import prompt_opens_think
            self._opens_think = prompt_opens_think(self.model)
            self._opens_think_checked = True
        return self._opens_think

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
        seed_kwarg = {"seed": self.seed} if self.seed is not None else {}
        # None = fall back to the instance temperature (the pinned default);
        # a per-call value still wins.
        temp = self.temperature if temperature is None else temperature
        # Merge so an explicit per-call kwarg overrides the instance default
        # rather than colliding into a duplicate-keyword TypeError.
        params = {
            "max_tokens": max_tokens,
            "temperature": temp,
            **seed_kwarg,
            **kwargs,
        }
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )
        input_tokens = output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self.token_usage.add(input_tokens, output_tokens)
        finish_reason = response.choices[0].finish_reason or ""
        if finish_reason == "length":
            self.truncated_calls += 1
        message = response.choices[0].message
        content = message.content or ""
        # Where a server run with --reasoning-parser puts the trace. The field
        # name differs across vLLM versions.
        parsed_reasoning = (getattr(message, "reasoning_content", None)
                            or getattr(message, "reasoning", None) or "")
        try:
            # With a parser, the trace is already out of content, so the only
            # failure left is running out of tokens while reasoning: content
            # comes back empty and nothing raises. Same normalisation as openai_gen.
            if not content.strip() and finish_reason == "length":
                raise OutputCapError(
                    f"empty completion with finish_reason='length' (max_tokens="
                    f"{max_tokens}), likely exhausted inside reasoning.")
            # Only the ambiguous case (stopped at the cap, no closing tag) needs
            # the chat template, so no other call has to load a tokenizer.
            ambiguous = finish_reason == "length" and THINK_CLOSE not in content
            text = split_reasoning(content, finish_reason,
                                   self.opens_think if ambiguous else None)
        except OutputCapError as e:
            # Every token of a call that never reached an answer was reasoning.
            e.input_tokens, e.output_tokens = input_tokens, output_tokens
            e.reasoning_tokens = output_tokens
            raise
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=self._reasoning_tokens(
                output_tokens, text, stripped=text != content,
                parsed=bool(parsed_reasoning.strip())),
        )

    def _reasoning_tokens(self, output_tokens: int, answer: str, *,
                          stripped: bool, parsed: bool) -> int | None:
        """How much of ``output_tokens`` the reasoning trace took.

        Computed as output minus the answer rather than by counting the trace,
        so the closing tag and the whitespace around it count as reasoning,
        which they are. The result is exactly output_tokens - answer_tokens.
        Returns 0 without a tokenizer when there was no trace, and None when there
        was one but the tokenizer could not be loaded to measure it.
        """
        if not (stripped or parsed):
            return 0
        from evalgrid.lib.tokens import try_count_tokens
        n = try_count_tokens(self.model, answer)
        return None if n is None else max(0, output_tokens - n)
