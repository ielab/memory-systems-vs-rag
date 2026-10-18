"""
evalgrid/lib/llm.py
------------------
One construction path for the pipeline's LLM roles (generator / labeler / judge).

build_generator drops the knobs that weren't set (None) and hands the rest to
models.factory.load_generator, which in turn drops any knob the chosen backend
doesn't accept (reasoning_effort/verbosity are OpenAI-only; base_url is
vllm-only; seed and temperature are honoured by both OpenAI and vllm). So a
stage can pass the same call for an OpenAI judge or a vllm generator without
per-backend guards.
"""

from __future__ import annotations

from evalgrid.models.factory import load_generator

# Pinned defaults shared by every pipeline stage (generator / labeler / judge) so
# runs are reproducible by default instead of each stage choosing its own. A stage
# overrides by passing the knob explicitly (e.g. --gen-seed / --judge-seed / --seed
# on the CLI).
#
# temperature is the other reproducibility knob, pinned at 0.0 (greedy). It travels
# the same construction path as seed (stored on the backend instance, applied at
# generate time), so it reaches both OpenAI and vLLM backends and is recorded in
# each stage's manifest.json via its --*-temperature flag.
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0


def build_generator(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
):
    """Construct a generator backend from a (possibly partial) model config.

    seed and temperature are pinned to their DEFAULT_* when a stage doesn't pass
    one, so runs are reproducible out of the box; pass them explicitly to
    override. (Both reach the OpenAI and vLLM backends.)
    """
    if seed is None:
        seed = DEFAULT_SEED
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    kwargs = {
        k: v for k, v in dict(
            # "" is a driver's unset GEN_BASE_URL, not an endpoint — passed on,
            # its mere presence would route an OpenAI model to vllm.
            base_url=base_url or None,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            seed=seed,
            temperature=temperature,
        ).items()
        if v is not None
    }
    return load_generator(model, **kwargs)
