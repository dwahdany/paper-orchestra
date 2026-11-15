"""Telling failures that cost one piece of work from failures that end the run.

Agents tolerate a great deal on purpose: a figure that cannot be drawn is
skipped, a search task that fails costs a few references.  Three kinds of
failure must not be absorbed that way, because the next piece of work would
fail for the same reason or must not start at all: the run was stopped, a model
cannot be used as configured, or the compute backend cannot work.
"""

from __future__ import annotations

from collections.abc import Iterator


def chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and every exception it was raised from or while handling."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def stops_the_run(exc: BaseException) -> bool:
    """Whether ``exc`` means the run cannot, or must not, go on.

    True for a stop request (``RunCancelled``, ``KeyboardInterrupt``), a model
    configuration error (``LLMConfigError``) and a compute error marked
    ``unusable``, wherever they sit in the exception's chain: agents often wrap
    the original error in one of their own.
    """
    from paper_orchestra.llm.base import LLMConfigError
    from paper_orchestra.memory.harness import RunCancelled

    return any(
        isinstance(error, (RunCancelled, KeyboardInterrupt, LLMConfigError))
        or getattr(error, "unusable", False) is True
        for error in chain(exc)
    )
