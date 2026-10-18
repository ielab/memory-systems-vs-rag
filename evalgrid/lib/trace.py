"""
evalgrid/lib/trace.py
---------------------
Trajectory recording: what each system's pipeline actually did, step by step.

The design, in short:

**Two layers.** L1 is a *call log*: every system's memory LLM already passes
through an adapter class this repo owns (a_mem's OpenAICompatController,
simplemem's controller, hypermem's provider, memorybank's GeneratorModel,
mem0's vendored OpenAILLM, lightmem's vendored manager), so ``record_calls``
rebinds one method per system on the INSTANCE and logs prompt / response /
usage / latency with no vendor edit and no per-system semantics. L2 is
``recorder.step(...)``, which an adapter wraps around its own calls into
vendored code to name a stage — the only way to capture the steps that are not
LLM calls at all (lightmem's LLMLingua compression, its cosine boundary
detection, memorybank's forgetting curve).

**Cost rides along.** A step's cost is only computable at the seam that defines
the step: snapshot the layer's TokenUsage on entry and exit and the delta IS
that step's cost. This is what turns the (layer x phase) resolution of
benchmarks/runner.py into (layer x phase x step).

**The aggregate files stay the authority.** A trajectory can be sampled,
truncated or off, so no reported number may be derived from it. If a total from
a trajectory disagrees with ingest_usage.json, the aggregate is right.

**It must be inert.** Recording never mutates its arguments, is never read back
by a system, and never changes control flow; a recorder failure is counted, not
raised. ``off`` is the default and yields the NULL_RECORDER singleton, whose
methods return immediately and which ``record_calls`` refuses to install at all.
tools/check_system_contract.py runs a system with recording off and on and
compares the outputs byte for byte — that check is the actual guarantee.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

#: Recording levels, cheapest first.
#:   off    nothing; NULL_RECORDER, no file, no wrapping
#:   calls  L1 only — one record per LLM call, payloads truncated
#:   steps  L1 + L2 semantic steps, payloads truncated
#:   full   as steps, payloads NOT truncated (large; never a default)
LEVELS = ("off", "calls", "steps", "full")

#: Default payload truncation. A LongMemEval ingest is ~250k turns; without a
#: cap a single sample's trajectory outgrows the memory it describes.
DEFAULT_MAX_CHARS = 2000


def trajectory_filename(level: str) -> str:
    """The file a run at ``level`` writes — one per level, never shared.

    A different level is a different artifact, not a different verbosity of the
    same one, so it gets its own path — the same rule as ``top<k>`` vs
    ``budget<T>`` and ``--mem-variant``. Sharing one name had three failure
    modes, all silent: an ingest resume would keep a file recorded at another
    level while the console announced the new one; ``--overwrite`` would destroy
    it; and a resumed answer run appends, so a "steps" run would interleave into
    a "calls" file with nothing marking the boundary.
    """
    return f"trajectory-{level}.jsonl"


#: Every name trajectory_filename can produce. Callers that must recognise a
#: trajectory without knowing which level wrote it (runner._memory_saved, the
#: reader) match against this rather than hardcoding one name.
TRAJECTORY_FILENAMES = frozenset(trajectory_filename(l) for l in LEVELS if l != "off")

#: Stage vocabulary. Closed on purpose: it is what makes six differently-shaped
#: pipelines groupable ("what fraction of ingest cost is extraction?"). A step
#: that fits none of these is "other", which is a prompt to discuss the
#: vocabulary rather than to widen it silently.
INGEST_STAGES = ("receive", "compress", "segment", "extract", "update", "summarize", "store")
QUERY_STAGES = ("plan", "retrieve", "expand", "rerank", "render", "generate")
#: "llm" is the implicit stage of an L1 call made outside any step() — it says
#: "an LLM call happened here and nobody claimed it", which is a finding.
STAGES = INGEST_STAGES + QUERY_STAGES + ("llm", "other")


def _triple(usage: Any) -> tuple[int, int, int]:
    """Read (input, output, calls) off a TokenUsage, a callable, or nothing."""
    if usage is None:
        return (0, 0, 0)
    if callable(usage):
        return tuple(usage())  # type: ignore[return-value]
    return (int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
            int(getattr(usage, "calls", 0) or 0))


def _render(value: Any) -> str:
    """Flatten a prompt/response argument to text for the log.

    Handles the three shapes the adapters use: a bare string, a message list,
    and anything else (repr'd). Never raises — a payload that will not render
    must not take a run down.
    """
    try:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            return "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in value)
        return repr(value)
    except Exception:
        return "<unrenderable>"


class _Handle:
    """What ``step()`` yields: a place to attach the step's outputs.

    Setters, not attributes, so the null implementation can share the interface
    and so a system can never accidentally read state back out of the recorder.
    """

    __slots__ = ("_rec",)

    def __init__(self, rec: Optional[dict] = None):
        self._rec = rec

    def out(self, value: Any) -> None:
        if self._rec is not None:
            self._rec["_out"] = value

    def units(self, unit_ids) -> None:
        if self._rec is not None:
            self._rec["_units"] = unit_ids

    def meta(self, **kw) -> None:
        if self._rec is not None:
            self._rec.setdefault("_meta", {}).update(kw)


_NULL_HANDLE = _Handle(None)


class NullRecorder:
    """The off switch, as an object rather than a branch at every call site.

    Every method returns immediately. ``enabled`` is False, which is what
    ``record_calls`` checks before wrapping anything, so at level "off" no
    method is rebound and no file is opened.
    """

    enabled = False
    level = "off"
    failures = 0

    def begin_sample(self, sample_id: str, phase: str, out_dir: str, mode: str = "w") -> bool:
        return False

    def end_sample(self) -> None:
        pass

    def call(self, **kw) -> None:
        pass

    @contextmanager
    def step(self, stage: str, step: str, **kw):
        yield _NULL_HANDLE

    def close(self) -> None:
        pass


#: Shared singleton — systems default to this, so ``self.recorder.call(...)``
#: is always safe to write without a guard.
NULL_RECORDER = NullRecorder()


class TraceRecorder:
    """Writes one JSON line per step to a per-sample trajectory-<level>.jsonl.

    Flat and append-only rather than a nested tree: a tree would have to be held
    in memory until the sample ends (50 sessions of lightmem compression
    payloads) and would lose everything on a crash mid-sample. Nesting is
    reconstructed at read time from ``seq``/``parent_seq``.
    """

    enabled = True

    def __init__(self, *, level: str = "calls", max_chars: int = DEFAULT_MAX_CHARS,
                 max_samples: Optional[int] = None):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        if level == "off":
            raise ValueError("level='off' means NULL_RECORDER, not a TraceRecorder")
        self.level = level
        self.max_chars = None if level == "full" else max_chars
        self.max_samples = max_samples
        self.failures = 0

        self._file = None
        self._sample_id: Optional[str] = None
        self._phase: str = "ingest"
        self._seq = 0
        self._stack: list[tuple[int, str]] = []   # (seq, stage) of open steps
        self._samples_seen = 0
        self._t0 = 0.0
        # Guards _seq and the write. See _emit: lightmem's offline update calls
        # the memory LLM from a thread pool.
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    @property
    def filename(self) -> str:
        return trajectory_filename(self.level)

    def begin_sample(self, sample_id: str, phase: str, out_dir: str, mode: str = "w") -> bool:
        """Open this sample's trajectory in ``out_dir``. False = not recording it.

        A DIRECTORY, not a path: the filename is the recorder's to choose from
        its level, so no caller can name a file that misdescribes what is in it.

        Sampling lives here rather than at the call sites so there is one place
        that decides, and so a system never needs to know it is being sampled.
        """
        self.end_sample()
        if self.max_samples is not None and self._samples_seen >= self.max_samples:
            return False
        try:
            os.makedirs(out_dir or ".", exist_ok=True)
            self._file = open(os.path.join(out_dir, self.filename), mode)
        except Exception:
            self.failures += 1
            self._file = None
            return False
        self._samples_seen += 1
        self._sample_id = sample_id
        self._phase = phase
        self._seq = 0
        self._stack = []
        self._t0 = time.perf_counter()
        return True

    def end_sample(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                self.failures += 1
        self._file = None
        self._stack = []

    def close(self) -> None:
        self.end_sample()

    # -- emission ----------------------------------------------------------
    def _clip(self, text: str) -> tuple[str, int, bool]:
        n = len(text)
        if self.max_chars is None or n <= self.max_chars:
            return text, n, False
        return text[:self.max_chars], n, True

    def _payload(self, value: Any) -> dict:
        text, chars, truncated = self._clip(_render(value))
        return {"text": text, "chars": chars, "truncated": truncated}

    def _emit(self, record: dict) -> None:
        """Write one line. Never raises: a broken recorder must not end a run.

        ``level``/``max_chars`` go on every record so a file stays
        self-describing after it is copied out of the tree that named it. Both
        are needed to read a payload honestly: without ``max_chars`` a reader
        cannot tell a short prompt from one clipped at a setting it never saw.

        Locked because not every system is single-threaded: lightmem's offline
        update runs its LLM calls through a ``ThreadPoolExecutor(max_workers=5)``
        (vendor/lightmem/memory/lightmem.py:634). Two threads formatting a
        record and writing it without a lock can interleave inside one line and
        produce a file that is not JSONL any more — a corruption that shows up
        only when something later tries to read it.
        """
        if self._file is None:
            return
        record["level"] = self.level
        record["max_chars"] = self.max_chars
        try:
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            with self._lock:
                self._file.write(line)
        except Exception:
            self.failures += 1

    def _next_seq(self) -> int:
        # ``+= 1`` is a read-modify-write, so concurrent callers can be handed
        # the same seq — which would make two records claim one identity and
        # silently break parent_seq resolution for a reader.
        with self._lock:
            self._seq += 1
            return self._seq

    def _current_stage(self) -> str:
        return self._stack[-1][1] if self._stack else "llm"

    def _parent(self) -> Optional[int]:
        return self._stack[-1][0] if self._stack else None

    def call(self, *, step: str, prompt: Any = None, response: Any = None,
             cost: tuple[int, int, int] = (0, 0, 0), seconds: float = 0.0,
             stage: Optional[str] = None, meta: Optional[dict] = None) -> None:
        """L1: one LLM call.

        ``stage`` defaults to the enclosing step's stage, so an adapter that has
        L2 gets its calls attributed for free, and one that does not gets "llm"
        — which reads as "an LLM call nobody named", not as a missing value.
        """
        if self._file is None:
            return
        c_in, c_out, c_calls = cost
        self._emit({
            "sample_id": self._sample_id,
            "phase": self._phase,
            "seq": self._next_seq(),
            "parent_seq": self._parent(),
            "stage": stage or self._current_stage(),
            "step": step,
            "kind": "call",
            "t": round(time.perf_counter() - self._t0, 4),
            "seconds": round(seconds, 4),
            "in": self._payload(prompt),
            "out": self._payload(response),
            "cost": {"input_tokens": c_in, "output_tokens": c_out, "calls": c_calls},
            "meta": meta or {},
        })

    @contextmanager
    def step(self, stage: str, step: str, *, inputs: Any = None,
             usage: Any = None, meta: Optional[dict] = None):
        """L2: a named stage of the system's own pipeline.

        ``usage`` is the layer's TokenUsage (or a callable returning an
        (in, out, calls) triple); the delta across the block is the step's cost,
        which is the resolution nothing else in the repo can produce. Yields a
        handle for the step's outputs.

        Recorded only at level "steps"/"full" — at "calls" the block still runs
        (so the system's code path is identical) but no step record is written.
        The step's STAGE is still pushed there: a stage label is L1 information
        (it is what makes a call record readable as "extract" rather than "an
        LLM call nobody named"), it costs nothing, and dropping it would mean
        the same run recorded at two levels disagreed about what its calls were.
        The pushed frame carries the ENCLOSING parent, not a seq of its own, so
        a nested call never points at a record that was not emitted.
        """
        if self._file is None:
            yield _NULL_HANDLE
            return
        if self.level == "calls":
            self._stack.append((self._parent(), stage))
            try:
                yield _NULL_HANDLE
            finally:
                self._stack.pop()
            return

        seq = self._next_seq()
        parent = self._parent()
        rec: dict = {}
        before = _triple(usage)
        started = time.perf_counter()
        self._stack.append((seq, stage))
        try:
            yield _Handle(rec)
        finally:
            self._stack.pop()
            after = _triple(usage)
            self._emit({
                "sample_id": self._sample_id,
                "phase": self._phase,
                "seq": seq,
                "parent_seq": parent,
                "stage": stage,
                "step": step,
                "kind": "step",
                "t": round(started - self._t0, 4),
                "seconds": round(time.perf_counter() - started, 4),
                "in": self._payload(inputs),
                "out": self._payload(rec.get("_out")),
                "cost": {"input_tokens": after[0] - before[0],
                         "output_tokens": after[1] - before[1],
                         "calls": after[2] - before[2]},
                "unit_ids": rec.get("_units") or [],
                "meta": {**(meta or {}), **rec.get("_meta", {})},
            })


def build_recorder(level: str = "off", *, max_chars: int = DEFAULT_MAX_CHARS,
                   max_samples: Optional[int] = None):
    """The one constructor a pipeline stage should call."""
    if level == "off":
        return NULL_RECORDER
    return TraceRecorder(level=level, max_chars=max_chars, max_samples=max_samples)


#: Marks a method this module has already wrapped, so a re-install (a system
#: that rebuilds its controller every reset) cannot stack wrappers N deep.
_WRAPPED = "_evalgrid_trace_wrapped"


def record_calls(obj: Any, method_name: str, *, recorder, step: str,
                 stage: Optional[str] = None, usage: Any = None, prompt_arg: int = 0,
                 prompt_of: Optional[Callable] = None,
                 response_of: Optional[Callable] = None,
                 cost_of: Optional[Callable] = None,
                 stage_of: Optional[Callable] = None,
                 step_of: Optional[Callable] = None) -> bool:
    """Wrap ``obj.method_name`` to log every call. Returns whether it installed.

    Instance-level rebinding, never a class or module patch: vendor/ is
    byte-identical to upstream by design, and an instance rebind is the ADAPTER
    pattern this repo already uses (a_mem's find_related_memories for link_k).
    Nothing else on the object changes, and the wrapper returns the wrapped
    method's value untouched.

    ``usage`` is the counter to diff for this call's cost — an object with
    input_tokens/output_tokens/calls, or a callable returning that triple.
    Defaults to ``obj.token_usage`` when present, which is exactly what
    ``layer_usage()`` reads, so a call's cost and the run's total come from one
    source. ``cost_of(result)`` overrides the diff for the case the diff cannot
    see: a vendored method whose caller — not the method — posts the tokens to
    the counter, so the delta across the method itself is always zero. It
    returns an (input, output, calls) triple read off the return value.

    ``stage_of(*args, **kwargs)`` / ``step_of(*args, **kwargs)`` name the call
    from its own arguments, for a single choke-point method that serves more
    than one pipeline stage (mem0's ``generate_response`` is both the fact
    extraction and the ADD/UPDATE/DELETE resolution). Both fall back to the
    static ``stage`` / ``step``.

    Returns False without touching anything when recording is off, so "off"
    really is zero overhead rather than a wrapper that checks a flag.
    """
    if recorder is None or not getattr(recorder, "enabled", False):
        return False
    target = getattr(obj, method_name, None)
    if target is None or getattr(target, _WRAPPED, False):
        return False
    if usage is None:
        usage = getattr(obj, "token_usage", None)

    def _prompt(args, kwargs):
        if prompt_of is not None:
            try:
                return prompt_of(*args, **kwargs)
            except Exception:
                return "<prompt_of failed>"
        if len(args) > prompt_arg:
            return args[prompt_arg]
        return kwargs.get("prompt") or kwargs.get("messages")

    def _response(value):
        if response_of is None:
            return value
        try:
            return response_of(value)
        except Exception:
            return "<response_of failed>"

    def _named(hook, static, args, kwargs):
        if hook is None:
            return static
        try:
            return hook(*args, **kwargs) or static
        except Exception:
            return static

    def _cost(before, result, error):
        if cost_of is None or error is not None:
            return tuple(a - b for a, b in zip(_triple(usage), before))
        try:
            return tuple(int(x) for x in cost_of(result))
        except Exception:
            return (0, 0, 0)

    def _log(prompt, args, kwargs, before, started, result=None, error=None):
        recorder.call(step=_named(step_of, step, args, kwargs),
                      stage=_named(stage_of, stage, args, kwargs),
                      prompt=prompt,
                      response=f"<error> {error}" if error is not None else _response(result),
                      cost=_cost(before, result, error),
                      seconds=time.perf_counter() - started,
                      meta={"error": type(error).__name__} if error is not None else None)

    # The prompt is rendered to text BEFORE the call, not after it. Several
    # wrapped methods mutate their argument in place and return it
    # (LlmLingua2Compressor.compress rewrites each message's "content" and
    # hands back the same list), so a prompt captured by reference and rendered
    # at log time would render the POST-call object — making "in" and "out"
    # byte-identical and the step read as a no-op it never was.
    #
    # A raising call is logged and re-raised: a failed extraction is exactly
    # what a trajectory is read for, and losing it would make the log claim the
    # step never happened.
    if inspect.iscoroutinefunction(target):
        async def wrapper(*args, **kwargs):
            before, started = _triple(usage), time.perf_counter()
            prompt = _render(_prompt(args, kwargs))
            try:
                result = await target(*args, **kwargs)
            except BaseException as e:
                _log(prompt, args, kwargs, before, started, error=e)
                raise
            _log(prompt, args, kwargs, before, started, result=result)
            return result
    else:
        def wrapper(*args, **kwargs):
            before, started = _triple(usage), time.perf_counter()
            prompt = _render(_prompt(args, kwargs))
            try:
                result = target(*args, **kwargs)
            except BaseException as e:
                _log(prompt, args, kwargs, before, started, error=e)
                raise
            _log(prompt, args, kwargs, before, started, result=result)
            return result

    setattr(wrapper, _WRAPPED, True)
    setattr(obj, method_name, wrapper)
    return True


def record_steps(obj: Any, method_name: str, *, recorder, stage: str, step: str,
                 usage: Any = None, inputs_of: Optional[Callable] = None,
                 out_of: Optional[Callable] = None) -> bool:
    """Wrap ``obj.method_name`` in ``recorder.step`` — the L2 counterpart of
    ``record_calls``. Returns whether it installed.

    ``record_calls`` says "an LLM call happened"; this says "a named stage of
    the pipeline ran, and everything logged inside it belongs to that stage".
    The difference matters for the systems whose whole LLM interface is one
    choke-point method: wrapping the method logs N indistinguishable calls,
    whereas wrapping the pipeline boundary they are made from gives each of
    them a stage and a parent for free (``recorder.call`` defaults both from
    the open step).

    Same instance-level rebinding, same idempotence guard and same
    return-value-untouched contract as ``record_calls``; a step whose body
    raises is still emitted, because ``recorder.step`` records from a
    ``finally``.
    """
    if recorder is None or not getattr(recorder, "enabled", False):
        return False
    target = getattr(obj, method_name, None)
    if target is None or getattr(target, _WRAPPED, False):
        return False
    if usage is None:
        usage = getattr(obj, "token_usage", None)

    def _inputs(args, kwargs):
        if inputs_of is None:
            return None
        try:
            return inputs_of(*args, **kwargs)
        except Exception:
            return "<inputs_of failed>"

    def _out(handle, result):
        if out_of is None:
            return
        try:
            handle.out(out_of(result))
        except Exception:
            handle.out("<out_of failed>")

    if inspect.iscoroutinefunction(target):
        async def wrapper(*args, **kwargs):
            # Rendered before the call, for the same in-place-mutation reason
            # record_calls documents.
            inputs = _render(_inputs(args, kwargs))
            with recorder.step(stage, step, inputs=inputs, usage=usage) as handle:
                result = await target(*args, **kwargs)
                _out(handle, result)
                return result
    else:
        def wrapper(*args, **kwargs):
            inputs = _render(_inputs(args, kwargs))
            with recorder.step(stage, step, inputs=inputs, usage=usage) as handle:
                result = target(*args, **kwargs)
                _out(handle, result)
                return result

    setattr(wrapper, _WRAPPED, True)
    setattr(obj, method_name, wrapper)
    return True
