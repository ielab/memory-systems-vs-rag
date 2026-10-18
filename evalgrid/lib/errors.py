"""
evalgrid/lib/errors.py
----------------------
Failure types the harness raises, and the one rule they encode: **a failed call
is not retried**.

Why a ``BaseException``
-----------------------
Four of the six vendored systems retry a failed LLM call inside
``while True:`` with a bare ``except Exception`` and no attempt limit
(hypermem has five such loops). Against a *deterministic* failure — a prompt
that cannot fit the window, a model that will not satisfy a validator — those
loops never terminate on their own. One HyperMem topic sent 331 requests and
burned its whole 600s budget before anything stopped it, and the log said
nothing.

Deriving from ``BaseException`` is what makes "run it once" enforceable from
outside code we do not edit: upstream's ``except Exception`` cannot catch it, so
the failure leaves the retry loop on the first attempt. It is the same mechanism
``asyncio.CancelledError`` uses, and for the same reason.

The cost of that choice is that these errors also blow past every *ordinary*
handler, including the ingest loop's own ``except Exception``. So the rule is:

  * systems catch ``LLMCallFailed`` explicitly at whatever boundary should
    survive it (hypermem: the topic / episode), and
  * ``benchmarks/runner.py`` catches it as a last resort, so a failure can cost
    at most one sample and can never take down a run.

Both call sites name it. Anything that means to swallow one has to say so.
"""

from __future__ import annotations


class EvalgridError(BaseException):
    """Base for harness failures that must not be retried by vendored code.

    ``BaseException`` on purpose — see the module docstring. Never catch this
    with a bare ``except Exception``; it will not work, which is the point.
    """


class LLMCallFailed(EvalgridError):
    """One memory-LLM call failed. Do not send it again.

    Carries enough to explain the failure without the traceback: which system,
    which endpoint-reported reason, and whether the request was
    ``deterministic`` — i.e. resending it byte-for-byte is guaranteed to fail
    the same way (a context-length overflow), as opposed to a timeout or a
    dropped connection that merely *did* fail.

    The distinction is recorded rather than acted on: both are raised on the
    first failure. It is what lets a run's failures.jsonl separate "this
    configuration cannot work" from "the server had a bad minute", which need
    different fixes.
    """

    def __init__(self, message: str, *, system: str = "", deterministic: bool = False,
                 cause: BaseException | None = None):
        super().__init__(message)
        self.system = system
        self.deterministic = deterministic
        self.cause = cause

    def as_record(self) -> dict:
        """Flat dict for failures.jsonl."""
        return {
            "error": type(self).__name__,
            "system": self.system,
            "deterministic": self.deterministic,
            "message": str(self),
            "cause": f"{type(self.cause).__name__}: {self.cause}" if self.cause else None,
        }
