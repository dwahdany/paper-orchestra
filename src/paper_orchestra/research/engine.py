"""The research engine: rounds of phases that pause at checkpoints.

``intake`` (once), then ``research``, ``hypothesize`` and ``design``, each ending
in a checkpoint where the person reviews what the agents proposed; then ``run``
(only what the person approved) followed by ``analyze``, another checkpoint, and
the next round.  :class:`Engine` carries a project through these phases one
``advance()`` at a time and writes what it does to ``state.json`` after every
transition, so that the web app, which only reads the project's files, can show
it and a later ``advance()`` can pick up where an interrupted one stopped.

One worker per project: ``advance()`` and ``run_experiment()`` hold a kernel lock
on ``.worker.lock`` and keep ``state.json["worker"]`` fresh with a heartbeat.

The model budget (``settings.limits.model_usd``) is looked at before every session
and every turn of one, and after every model request inside a turn: a turn that
takes the project past it ends at its next model call, and the phase stops at a
checkpoint that says it was cut short.  A phase that ends past the limit says so
at its checkpoint as well.

What ``advance()`` does when no phase is named: it first runs what the person
approved and analyses results that have no analysis yet, pauses after that
analysis when anything ran, and otherwise goes on with the phase that was next.
Under the classic runtime a run needs no model and never waits for one (under
``sdk`` an agent carries it out, and it waits like the rest).  Every other phase
does: where the project's models cannot be used, ``advance()`` stops before the first such phase,
at a checkpoint when something was done before it and with :class:`ModelNeeded`
(one sentence, nothing recorded) when nothing was.
Which phase is next is never changed by what an agent or a request says: more
research is asked for by naming the phase.

``state.json`` is the engine's account of where the project stands, and the
project's events are a second one: :func:`read_state` rebuilds round, phases
and what comes next from ``events.jsonl`` when the file is lost or damaged, and
:func:`state_problem` says so.  A project with work on file is never taken for
one that has not started.

:func:`next_step` says what an ``advance()`` without a phase would do, for whoever
shows the project: it asks the rules ``advance()`` itself goes by, reads only, and
needs neither the worker lock nor a model.  :func:`phase_in_words` is how a
sentence for the person names a phase.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import logging
import os
import re
import signal
import socket
import stat
import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_orchestra.compute import ComputeBackend, ComputeError, get_backend
from paper_orchestra.errors import chain, stops_the_run
from paper_orchestra.llm.base import Usage
from paper_orchestra.llm.pricing import ceiling_cost_usd, estimate_cost_usd, price_for
from paper_orchestra.memory.harness import ContextHarness, RunCancelled
from paper_orchestra.memory.lock import WikiLock

# The usage ledger is the paper pipeline's, so that usage.json has one shape and one
# way of pricing wherever it is written.
from paper_orchestra.pipeline import read_carried_usage, usage_report
from paper_orchestra.research import agents, autopilot, decisions, experiments
from paper_orchestra.research.agents import AgentContext, AgentReport
from paper_orchestra.research.model import (
    AUTOPILOT,
    FOR_ALL,
    SCHEMA,
    Limits,
    NotFound,
    ProjectError,
    Request,
    ResearchPage,
    format_stamp,
    is_agent,
    parse_stamp,
    phase_name,
)
from paper_orchestra.research.project import (
    EVENTS_FILE,
    EXPERIMENTS_DIR,
    STATE_FILE,
    Project,
    forget_worker_stop,
    record_worker_stop,
)
from paper_orchestra.research.sources import CutoffError
from paper_orchestra.utils.common import write_json
from paper_orchestra.utils.scholar import SemanticScholar

__all__ = [
    "PHASES",
    "Engine",
    "EngineError",
    "ModelNeeded",
    "WorkerBusy",
    "as_worker",
    "next_step",
    "phase_in_words",
    "phase_name",
    "read_state",
    "state_problem",
    "worker_alive",
]

log = logging.getLogger(__name__)

PHASES = ("intake", "research", "hypothesize", "design", "run", "analyze")

WORKER_LOCK_FILE = ".worker.lock"
USAGE_FILE = "usage.json"
#: A worker whose heartbeat is older than this, and whose process is gone, is dead.
STALE_HEARTBEAT_SECONDS = 60.0
#: How far ahead of this machine's clock a heartbeat may lie and still be a sign of life.
#: Further ahead, the clock was set back since (or the file is not an engine's), and the
#: beat says nothing.  A worker on this machine wrote it by this clock; one on another
#: machine wrote it by its own, which may differ by more.  The web server uses the same
#: two numbers.
FUTURE_HEARTBEAT_SECONDS = 5.0
FUTURE_HEARTBEAT_SECONDS_ELSEWHERE = 60.0
STATUSES = ("idle", "running", "checkpoint", "stopped", "failed")

_ROLE = {
    "intake": "intake",
    "research": "scout",
    "hypothesize": "theorist",
    "design": "designer",
    "run": "runner",
    "analyze": "analyst",
}
_NEXT = {
    "intake": "research",
    "research": "hypothesize",
    "hypothesize": "design",
    "design": "run",
    "run": "analyze",
    "analyze": "hypothesize",
}
_NOUN = {
    "intake": "intake",
    "research": "research",
    "hypothesize": "hypothesis",
    "design": "design",
    "run": "run",
    "analyze": "analysis",
}
_DOING = {
    "intake": "reading the attached files",
    "research": "research into what is already known",
    "hypothesize": "proposing hypotheses from what you accepted",
    "design": "designing experiments for the hypotheses you accepted",
    "run": "running the experiments you approved",
    "analyze": "analysing the new results",
}
#: The same where the person's standing rule (autopilot) decided: nothing here was
#: accepted or approved by them, and no sentence says it was.
_DOING_UNATTENDED = {
    **_DOING,
    "hypothesize": "proposing hypotheses from what is accepted",
    "design": "designing experiments for the accepted hypotheses",
    "run": "running the experiments that are approved",
}
#: Phases whose agent can act on a request of the person.
_TAKES_REQUESTS = ("research", "hypothesize", "design", "analyze")


def phase_in_words(name: Any) -> str:
    """A phase as a sentence for the person names it: ``the hypothesis phase`` for
    ``hypothesize``, ``the analysis phase`` for ``analyze``.

    The engine's own sentences (its events and checkpoints) use these words.
    Whoever else names a phase to the person (the web server, when it refuses a
    second worker) asks here, so that a phase has one name wherever it is read.
    What is no phase of the engine keeps the name it came with.

    Where a phase is named by itself and not in a sentence ("checkpoint after
    hypotheses", "next: analysis"), its name is ``phase_name`` (from
    ``research.model``, exported here as well): the words of the app.
    """
    return f"the {_NOUN.get(name, name) if isinstance(name, str) else name} phase"


#: The phases that follow an approval, not a checkpoint's "next": what the person approved
#: is run, and what ran is analysed, wherever in the round that falls.
_RUNS = ("run", "analyze")
_EXPERIMENT_DIR = re.compile(r"E\d+")


class EngineError(ProjectError):
    """The engine cannot do what was asked; the message says why and what to do."""


class WorkerBusy(EngineError):
    """Another worker is running this project."""


#: How the ledger rows end that hold spend an SDK agent never reported
#: (``research.sdk.session.UNREPORTED_SUFFIX`` after its label; not imported from
#: there, so that reading a ledger needs nothing of the sdk runtime).
UNREPORTED_LABEL = ".agent.unreported"


class ModelNeeded(EngineError):
    """The phase that would run needs a model, and the project's models cannot be used.

    Nothing is wrong with the project and nothing happened to it: the message is the
    one sentence that says which models, why, and what to set up.  ``need`` is what
    ``agents.model_need`` found, ``phase`` the phase that was not started.
    """

    def __init__(self, message: str, *, phase: str, need: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.phase = phase
        self.need = dict(need)


#: The counts of a row of the usage ledger.
_USAGE_FIELDS = tuple(Usage().to_dict())


class _BudgetSpent(Exception):
    """A limit of ``settings.limits`` is used up; the work stops at a checkpoint."""

    def __init__(self, limit: str, message: str, *, spent: float, allowed: float) -> None:
        super().__init__(message)
        self.limit = limit
        self.spent = spent
        self.allowed = allowed
        #: The same sentence for another amount spent, where the amount in it was a count
        #: made while a session was at work and the session's own figure came afterwards.
        self.say: Callable[[float], str] | None = None


# ---------------------------------------------------------------------------
# state.json
# ---------------------------------------------------------------------------


def _now() -> str:
    return format_stamp(datetime.now(UTC))


def _blank_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "idle",
        "round": 0,
        "phase": None,
        "next_phase": "intake",
        "checkpoint": None,
        "phases": [],
        "worker": None,
        "spent": {"model_usd": 0.0, "experiment_seconds": 0.0, "experiment_usd": 0.0},
        "error": None,
    }


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _state_on_file(project: Project) -> dict[str, Any] | None:
    """``state.json`` as the object it holds; ``None`` when it is missing or is not one."""
    try:
        data = json.loads((project.root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _records(value: Any) -> list[dict[str, Any]]:
    """The phase records in ``value``: objects that name a phase and a round.  Anything
    else in the list (a file edited by hand) is not a record."""
    return [
        record
        for record in (value if isinstance(value, list) else [])
        if isinstance(record, dict)
        and record.get("phase") in PHASES
        and _whole(record.get("round"))
        and record["round"] >= 0
    ]


def _recommendation(data: Any) -> dict[str, Any]:
    """The two fields of a checkpoint that carry the analyst's recommendation, read from
    a checkpoint or from the data of its event: ``recommends`` is the phase the round's
    journal recommends before the one that is next (only ``research`` can be) and
    ``recommends_page`` the journal page that says so.  Both are ``None`` when nothing
    is recommended, also for a checkpoint written before these fields existed."""
    data = data if isinstance(data, Mapping) else {}
    phase, page = data.get("recommends"), data.get("recommends_page")
    if phase != "research" or not isinstance(page, str) or not page:
        return {"recommends": None, "recommends_page": None}
    return {"recommends": phase, "recommends_page": page}


def _read_checkpoint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    pending = value.get("pending")
    counted = {
        str(kind): count
        for kind, count in (pending.items() if isinstance(pending, dict) else ())
        if _whole(count)
    }
    summary = value.get("summary")
    return {
        **value,
        "summary": summary if isinstance(summary, str) else "",
        "pending": counted,
        **_recommendation(value),
    }


def _read_leniently(data: Mapping[str, Any]) -> dict[str, Any]:
    """The state an object from ``state.json`` describes, every value of the type the
    engine and whoever shows the state count on.  What is of another type (a file
    edited by hand, or cut short and patched up) is replaced by what the rest implies."""
    state = _blank_state()
    spent = data.get("spent") if isinstance(data.get("spent"), Mapping) else {}
    state["spent"] = {name: _number(spent.get(name)) for name in state["spent"]}
    phases = state["phases"] = copy.deepcopy(_records(data.get("phases")))
    last = phases[-1] if phases else None
    rounds = [record["round"] for record in phases]
    round = data.get("round")
    if rounds:
        # The round is that of one of the phases on record: the last one's, unless the
        # file names another of them.
        state["round"] = round if _whole(round) and round in rounds else rounds[-1]
    else:
        state["round"] = round if _whole(round) and round >= 0 else 0
    phase, following = data.get("phase"), data.get("next_phase")
    state["phase"] = phase if phase in PHASES else (last["phase"] if last else None)
    state["next_phase"] = following if following in PHASES else (None if phases else "intake")
    state["checkpoint"] = _read_checkpoint(copy.deepcopy(data.get("checkpoint")))
    worker = data.get("worker")
    state["worker"] = copy.deepcopy(worker) if isinstance(worker, dict) else None
    error = data.get("error")
    state["error"] = error if isinstance(error, str) or error is None else _one_line(error)
    status = data.get("status")
    if status in STATUSES:
        state["status"] = status
    elif phases:
        # Nothing says how the last call ended: a checkpoint that is on file, else stopped.
        state["status"] = "checkpoint" if state["checkpoint"] else "stopped"
    stretch = autopilot.read_stretch(data.get(autopilot.STRETCH_KEY))
    if stretch is not None:
        # Only a project that ran unattended has the key: the others are as they were.
        state[autopilot.STRETCH_KEY] = stretch
    return state


def _unfinished(records: Sequence[dict[str, Any]], phase: str, round: int) -> dict | None:
    """The latest record of the phase in the round, when it has not completed."""
    for record in reversed(records):
        if record["phase"] == phase and record["round"] == round:
            return record if record["status"] != "completed" else None
    return None


def _from_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The state as the project's events tell it; ``None`` when they hold no phase.

    ``phase.started``, ``phase.finished``, ``phase.failed`` and ``checkpoint`` carry
    round, phase, what was produced, the error and what comes next.  A phase that
    was started and neither finished nor failed was interrupted: whoever reads a
    state from the events has no live worker's file to read it from.
    """
    state = _blank_state()
    records: list[dict[str, Any]] = state["phases"]
    for event in events:
        kind, phase, round = event.get("type"), event.get("phase"), event.get("round")
        if kind in (autopilot.STARTED, autopilot.CONTINUED, autopilot.STOPPED):
            # The record of an unattended stretch travels with its events.
            carried = event.get("data") if isinstance(event.get("data"), Mapping) else {}
            stretch = autopilot.read_stretch(carried.get("stretch"))
            if kind == autopilot.STARTED and isinstance(carried.get("caps"), Mapping):
                # Its beginning says what it was held to and when it began; the rest
                # (rounds, spend) is in the events that follow, where there are any.
                begun = autopilot.new_stretch(
                    carried["caps"],
                    started=str(event.get("ts") or ""),
                    spent_usd=0.0,
                    round=round if _whole(round) else 0,
                )
                begun["seq"] = event.get("seq") if _whole(event.get("seq")) else 0
                stretch = autopilot.read_stretch(begun)
            if stretch is not None:
                state[autopilot.STRETCH_KEY] = stretch
            continue
        if phase not in PHASES or not _whole(round):
            continue
        data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
        stamp = str(event.get("ts") or "")
        if kind == "phase.started":
            record = _unfinished(records, phase, round)
            if record is None:
                record = Engine._new_record(phase, round, None)
                record["started"] = stamp
            else:
                # Resumed: the engine lists the record of the phase in flight last, also
                # after a run and its analysis came in between.
                records[:] = [known for known in records if known is not record]
            records.append(record)
            word, given = data.get("instruction"), data.get("requests")
            record.update(status="running", finished=None, error=None)
            record["attempts"] += 1
            if isinstance(word, str) and word.strip():
                record["instruction"] = word
            if isinstance(given, list):
                record["requests"] = [str(item) for item in given]
            state.update(
                status="running", round=round, phase=phase, next_phase=phase, checkpoint=None
            )
            state["error"] = None
        elif kind in ("phase.finished", "phase.failed"):
            record = _unfinished(records, phase, round)
            if record is None:
                continue
            if kind == "phase.finished":
                produced = data.get("produced")
                record.update(
                    status="completed",
                    finished=stamp,
                    seconds=_number(data.get("seconds")),
                    produced=[str(path) for path in produced] if isinstance(produced, list) else [],
                )
                state.update(round=round, phase=phase, next_phase=_NEXT[phase])
            else:
                error, attempts = data.get("error"), data.get("attempts")
                error = error if isinstance(error, str) else _one_line(event.get("summary"))
                record.update(status="failed", finished=stamp, error=error)
                if _whole(attempts):
                    record["attempts"] = attempts  # a retry in the same call starts no event
                state.update(status="failed", round=round, phase=phase, next_phase=phase)
                state["error"] = error
        elif kind == "checkpoint" and records:
            following, pending = data.get("next"), data.get("pending")
            unfinished = _unfinished(records, phase, round)
            if unfinished is not None:
                # A checkpoint in the middle of a phase: a budget ended it early.
                unfinished.update(status="interrupted", finished=stamp)
            state.update(
                status="checkpoint",
                round=round,
                phase=phase,
                next_phase=following if following in PHASES else _NEXT[phase],
                checkpoint=_read_checkpoint(
                    {
                        "round": round,
                        "phase": phase,
                        "at": stamp,
                        "summary": event.get("summary"),
                        "pending": pending,
                        **_recommendation(data),
                    }
                ),
            )
            state["error"] = None
    if not records:
        return None
    for record in records:
        if record["status"] == "running":
            record.update(status="interrupted", finished=record["finished"] or record["started"])
    if state["status"] == "running":
        # Between two phases, or in one: either way no worker is there to go on.
        state["status"] = "stopped"
    stretch = state.get(autopilot.STRETCH_KEY)
    if stretch is not None and not stretch["ended"]:
        # Its worker is not there to go on either, and nothing starts it again.
        stretch.update(reason="worker_gone", said=None)
        stretch["ended"] = stretch["started"]
    return state


def _rebuilt(project: Project, written: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The state rebuilt from ``events.jsonl``, for a state file that holds no phase."""
    state = _from_events(project.events())
    if state is None:
        return None
    spent = written.get("spent") if written and isinstance(written.get("spent"), Mapping) else {}
    state["spent"] = {name: _number(spent.get(name)) for name in state["spent"]}
    kept = autopilot.read_stretch(written.get(autopilot.STRETCH_KEY)) if written else None
    if kept is not None:
        state[autopilot.STRETCH_KEY] = kept  # the file's own account of it, where it has one
    if not spent:
        # The ledger of the model spend is a file of its own; what the runs cost is
        # counted again from their records by the next worker.
        try:
            ledger = json.loads((project.root / USAGE_FILE).read_text(encoding="utf-8"))
            state["spent"]["model_usd"] = round(_number(ledger.get("estimated_cost_usd")), 4)
        except (OSError, ValueError, AttributeError):
            pass
    return state


def read_state(project: Project) -> dict[str, Any]:
    """The engine's state of ``project`` in the shape of ``state.json``.

    A project the engine has not touched yet reads as ``idle`` with ``intake``
    next.  A file that is missing keys, or holds values of the wrong type, is
    read as far as it goes.  A file that is missing, is not JSON or holds no
    phase record is not taken for "never started" when ``events.jsonl`` says
    that phases ran: round, phases, the checkpoint and what comes next are then
    rebuilt from the events (see :func:`state_problem`).  Nothing is written.
    """
    written = _state_on_file(project)
    state = _read_leniently(written or {})
    if not state["phases"]:
        return _rebuilt(project, written) or state
    return state


def _agents_wrote(project: Project) -> bool:
    """Whether the wiki holds what only a phase of the engine leaves behind: a page an
    agent wrote, or a source record.  (A result record does not say so: the person can
    have an experiment of their own run before any phase.)"""
    return any(is_agent(page.author) or page.kind == "source" for page in project.pages())


def state_problem(project: Project) -> str | None:
    """What is wrong with ``state.json``, for whoever shows the project; ``None`` when
    the file is the account of the phases that ran, or none has run.

    Two things can be wrong.  The file is lost or damaged but the project's
    events tell where it stands: :func:`read_state` then answers from the
    events, and the engine writes the file anew the next time it works.  Or the
    events do not tell either while the wiki holds what agents wrote: then
    nobody can say what was next, and ``advance()`` asks for a phase by name
    instead of starting from the beginning.
    """
    written = _state_on_file(project)
    if written is not None and _records(written.get("phases")):
        return None
    path = project.root / STATE_FILE
    what = (
        "holds no record of the phases that ran"
        if written is not None
        else "cannot be read"
        if path.exists()
        else "is missing"
    )
    if _from_events(project.events()) is not None:
        return (
            f"{STATE_FILE} {what}. Round, phases and what comes next are read from the "
            "project's events (events.jsonl) instead; the file is written anew the next "
            "time the agents work. Check that the round and the next phase are what you "
            "expect before you continue."
        )
    if _agents_wrote(project):
        return (
            f"{STATE_FILE} {what} and events.jsonl holds no record of a phase either, but "
            "the wiki holds pages the agents wrote: the engine cannot tell where the "
            "project stands, and does not start it from the beginning by itself. Name the "
            "phase to go on with."
        )
    return None


def next_step(
    project: Project,
    state: Mapping[str, Any] | None = None,
    *,
    client_for: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """What an ``advance()`` with no phase named would do as the project stands, for
    whoever shows it (the label of the app's Continue button)::

        {"will_run": ["run", "analyze"], "then": "hypothesize",
         "experiments": ["E1"], "results": 0,
         "checkpoint_of": None, "needs_phase": False, "blocked": None}

    ``will_run`` are the phases the call runs, in order, when each of them
    completes, and ``then`` is the phase its checkpoint names as next.  What the
    person approved comes first: ``experiments`` are the accepted experiments a
    run phase is owed (approved, or left queued or running by a worker that is
    gone), and ``results`` is how many result records have no analysis yet.
    ``checkpoint_of`` names a phase that completed without its checkpoint (its
    worker died right after it): the call writes that checkpoint first, and when
    ``will_run`` is empty that is all it does.  ``needs_phase`` says that the call
    would be refused, because nothing on file says where the project stands
    (:func:`state_problem`): a phase has to be named.

    ``blocked`` says where the call stops because a phase needs a model and the
    project's models cannot be used: ``{"phase", "needs": "model", "settings",
    "models", "message"}``, with the phase that is not started, the entries of
    ``settings.models`` concerned, their models, and the one sentence that says
    what is missing and what to do.  ``will_run`` then ends before that phase
    (what needs no model, a run, still happens) and ``then`` names it; with
    nothing in ``will_run`` the call would be refused with that sentence
    (:class:`ModelNeeded`).  The models are looked at only when ``client_for``
    is given: how a model's client is found, as :class:`Engine` takes it
    (``llm.router.get_client`` for the clients a worker would get).  Nothing is
    contacted for it and nothing spent.  Without it ``blocked`` is ``None``.

    The answer comes from the rules :meth:`Engine.advance` itself goes by, applied
    to a copy of the state.  It assumes that the experiments which wait do run:
    when none of them can (each changed since its approval, say), ``advance()``
    goes on with the phase that was next in the same call and its checkpoint says
    why.  What the budgets allow is not looked at: a call that is refused for
    that says so itself.

    For a project nobody is working on.  Nothing is written and no lock is taken;
    ``state`` is what :func:`read_state` returned, for a caller that holds it
    already (it is not changed).
    """
    mine = copy.deepcopy(dict(state)) if state is not None else read_state(project)
    return Engine._reading(project, mine, client_for)._forecast()


def _is_zombie(pid: int) -> bool:
    """Whether the process has exited and only waits for its parent to collect it."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:  # no /proc here, or the process is gone: nothing says it is a zombie
        return False
    # The state follows the command name, which is in brackets and may hold anything.
    return stat.rpartition(")")[2].split()[:1] == ["Z"]


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - Windows, where ``os.kill(pid, 0)`` is no test:
        # signal 0 is Ctrl-C there, and would interrupt the very worker that is asked about.
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x1000, False, pid)  # query limited information
        if not handle:
            return False
        code = ctypes.c_ulong()
        answered = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(answered) and code.value == 259  # still active
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # it exists, but is not ours to signal
        return True
    return not _is_zombie(pid)


def worker_alive(state: Mapping[str, Any], *, now: datetime | None = None) -> bool:
    """Whether the worker ``state`` names is still at work.

    It is dead when its heartbeat is older than a minute and its process is gone;
    on another machine only the heartbeat can be judged.  A heartbeat that lies in
    the future is no sign of life (by more than a few seconds for a worker on
    this machine, by more than a minute for one on another, whose clock is its
    own): after the clock was set back it would count as fresh for as long as
    the difference lasts, however long ago its writer died.  The process
    decides then.
    """
    worker = state.get("worker")
    if not isinstance(worker, Mapping):
        return False
    here = worker.get("host") == socket.gethostname()
    ahead = FUTURE_HEARTBEAT_SECONDS if here else FUTURE_HEARTBEAT_SECONDS_ELSEWHERE
    beat = parse_stamp(worker.get("heartbeat"))
    age = ((now or datetime.now(UTC)) - beat).total_seconds() if beat else float("inf")
    if -ahead <= age <= STALE_HEARTBEAT_SECONDS:
        return True
    return here and _pid_alive(worker.get("pid"))


#: The records a worker reads and writes by name in the project's directory.
_OWN_RECORDS = (STATE_FILE, EVENTS_FILE, USAGE_FILE)


def _refuse_unwritable_state(project: Project) -> None:
    """Say what is in the way when the place of a record the engine keeps (its state,
    the events, the usage ledger) is taken by something that is not a file of the
    project, before a worker does anything that could then not be recorded.

    A directory or a pipe cannot hold the record.  Through a symbolic link the engine
    would read where something else stands and write its own account there, and a
    project directory can come from anywhere: the app starts nothing in such a
    project, and neither does a worker.
    """
    for name in _OWN_RECORDS:
        path = project.root / name
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise EngineError(f"{path} cannot be looked at ({exc}).") from None
        if stat.S_ISREG(mode):
            continue
        what = "a symbolic link" if stat.S_ISLNK(mode) else "something else in its place"
        after = (
            "the state is then rebuilt from the project's events"
            if name == STATE_FILE
            else f"put the project's own {name} in its place if you have it"
        )
        raise EngineError(
            f"{path} is not a file ({what}), so the engine neither reads it nor keeps its "
            f"records there. Remove or rename it and run the command again; {after}."
        )


class _WorkerLock(WikiLock):
    """The kernel lock of ``memory.lock`` on the project's worker file; it never waits."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, wait_seconds=0.0, poll=0.05)
        self.path = Path(root) / WORKER_LOCK_FILE


class _Heartbeat(threading.Thread):
    """Keeps ``state.json["worker"]["heartbeat"]`` fresh while the worker is at work."""

    def __init__(self, beat: Callable[[], None], seconds: float) -> None:
        super().__init__(name="research-heartbeat", daemon=True)
        self._beat = beat
        self._seconds = seconds
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.wait(self._seconds):
            try:
                self._beat()
            except Exception:  # a heartbeat that fails once must not end the worker
                log.warning("The heartbeat could not be written.", exc_info=True)

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=5.0)


@contextlib.contextmanager
def as_worker(project: Project, *, kind: str) -> Iterator[None]:
    """Be the project's one worker for work that is not a phase of the engine.

    The write-up of the paper uses this.  It holds the worker lock and names
    this process under ``state.json["worker"]`` with a heartbeat and its
    ``kind``, as an engine does, so that the web server shows the worker and
    nothing else starts beside it.  The rest of the state is left as it is.
    Raises ``WorkerBusy`` when the project has a worker, and ``EngineError`` when
    the state file's place is taken by something that is not a file.
    """
    lock = _WorkerLock(project.root)
    with lock:
        if not lock.acquired:
            if lock.error:
                raise WorkerBusy(
                    f"The project at {project.root} could not be locked for a worker "
                    f"({lock.error}). Check that the directory can be written."
                )
            raise WorkerBusy(
                "Another worker is at work on this project: its agents, a run of an "
                "experiment, or a write-up that was started before. Wait for it to "
                "finish, or stop it, then start again; a project has one worker at a time."
            )
        _refuse_unwritable_state(project)
        path = project.root / STATE_FILE
        started = _now()
        mine = {"pid": os.getpid(), "host": socket.gethostname(), "started": started}
        written = threading.Lock()

        def name(*, gone: bool = False) -> None:
            # Holding the lock makes this process the only worker that writes the file.
            with written:
                state = read_state(project)
                state["worker"] = None if gone else {**mine, "heartbeat": _now(), "kind": kind}
                write_json(path, state)

        name()
        beat = _Heartbeat(name, Engine.HEARTBEAT_SECONDS)
        beat.start()
        try:
            yield
        finally:
            beat.stop()
            name(gone=True)


class _Compute:
    """The backend that runs a plan: the one the caller gave, else the one the person chose
    when approving the run, else the one an approval would choose (Modal for a GPU
    experiment, which no other backend can run; the project's for the rest)."""

    def __init__(self, project: Project, given: Any) -> None:
        self._project = project
        self._given = given
        self._made: dict[str, ComputeBackend] = {}

    def __call__(self, plan: Any = None) -> ComputeBackend:
        if isinstance(self._given, ComputeBackend):
            return self._given
        if self._given is not None:
            return self._given(plan)
        default = (
            "modal" if getattr(plan, "mode", None) == "gpu" else self._project.settings.compute
        )
        return self.named(self._approved_backend(plan) or default)

    def _approved_backend(self, plan: Any) -> str | None:
        """The backend the person's approval of the plan's experiment named, if it has one.

        Where an experiment runs is the person's choice, made in the project's
        settings or at approval.  A ``backend`` on a page that is not approved (a
        proposal the designer's check asks about) is not that choice, whoever wrote
        it there, and is not followed.
        """
        named, path = getattr(plan, "backend", None), getattr(plan, "path", None)
        if not named or not path:
            return None
        try:
            page = self._project.page(path)
        except ProjectError:
            return None
        return str(named) if page.state in ("approved", "queued", "running") else None

    def named(self, name: str) -> ComputeBackend:
        if name not in self._made:
            self._made[name] = get_backend(name)
        return self._made[name]

    def close(self) -> None:
        for backend in self._made.values():
            with contextlib.suppress(Exception):
                backend.close()
        self._made.clear()


@dataclass
class _Outcome:
    #: ``completed``, ``interrupted``, ``failed`` or ``budget``.
    status: str
    report: AgentReport | None = None
    #: Experiments the run phase executed.
    ran: int = 0


def _plural(count: int, noun: str) -> str:
    if count == 1:
        return f"1 {noun}"
    return f"{count} {noun[:-1] + 'ies' if noun.endswith('y') else noun + 's'}"


def _and(parts: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``; nothing for no parts."""
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _one_line(text: Any, limit: int = 400) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _sentence(text: Any, limit: int = 400) -> str:
    """``text`` on one line, ending the way a sentence ends."""
    line = _one_line(text, limit)
    return line if line[-1:] in ".!?…" else line + "."


def _money(amount: float) -> str:
    """``$0.34``; an amount under one cent with the digits that tell it from nothing
    (``$0.004``), so that a limit set that low does not read as "$0.00"."""
    if 0 < amount < 0.01:
        digits = f"{amount:.4f}".rstrip("0")
        return f"${digits}" if digits != "0." else "under $0.0001"
    return f"${amount:.2f}"


def _limit_money(limit: float) -> str:
    """A limit the person set, with the digits they gave it: ``$0.145`` and not
    ``$0.14``, beside which "$0.15 is spent" would read as an error of arithmetic.
    A limit in whole cents reads as any amount does."""
    if limit >= 0.01 and abs(limit * 100 - round(limit * 100)) > 1e-6:
        return "$" + f"{limit:.4f}".rstrip("0")
    return _money(limit)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class Engine:
    """Carries one project through its phases.  One instance per worker."""

    #: How often the worker's heartbeat (and usage.json) is written while it works.
    HEARTBEAT_SECONDS = 15.0

    def __init__(
        self,
        project: Project,
        *,
        harness: ContextHarness | None = None,
        compute: ComputeBackend | Callable[[Any], ComputeBackend] | None = None,
        client_for: Callable[[str], Any] | None = None,
        scholar: SemanticScholar | None = None,
        driver: Any = None,
    ) -> None:
        """``harness`` is the context harness every model call goes through; without one
        the engine builds the project's own (``client_for`` then chooses the clients).
        ``compute`` is a backend, or a callable choosing one for a plan; without one the
        backend named at approval, or in the project's settings, is used.  ``scholar``
        is the Semantic Scholar client of the research phase.  ``driver`` starts the
        agents of a project whose runtime is ``sdk`` (``research.sdk.driver``); without
        one the Claude Agent SDK is used, which is an optional package."""
        self.project = project
        self._driver = driver
        self._agents: Any = None
        self.harness = harness or agents.build_harness(project, client_for=client_for)
        self._client_for: Callable[[str], Any] | None = self.harness.client_for
        #: Whether the project's models can be used, asked once for each call that works
        #: (``settings`` asked about -> what ``agents.model_need`` found).
        self._needs: dict[tuple[str, ...], dict[str, Any] | None] = {}
        self._compute = _Compute(project, compute)
        self._scholar = scholar
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._state = read_state(project)
        self._carried: list[dict[str, Any]] = []
        self._baseline: list[dict[str, Any]] = []
        #: Model spend that state.json had recorded and the usage ledger no longer
        #: accounts for (the ledger was lost): carried, so that spending never goes down.
        self._carried_usd = 0.0
        #: Models that were used and have no known price: the model budget cannot count them.
        self._unpriced: list[str] = []
        #: Experiments the run phase in flight has executed.
        self._ran = 0
        #: Whether whoever drives this engine goes on with every experiment that is
        #: approved while it runs one (`research run --queue`); written with the
        #: worker's name, for those who approve meanwhile.
        self.drains = False
        #: What was wrong with state.json when this worker started, before it wrote the
        #: file anew (:func:`state_problem`); ``None`` when nothing was.
        self._lost: str | None = None
        #: The unattended stretch this worker is in (``state["autopilot"]``, its own to
        #: change), and the clock its hours are counted by; ``None`` outside of one.
        self._stretch: dict[str, Any] | None = None
        self._stretch_clock = 0.0
        #: How the latest pass of :meth:`advance` ended (see :meth:`_pass`).
        self._pass_end = ""
        #: What the standing rule would not approve in the run phase in flight.
        self._held: dict[str, str] = {}

    @classmethod
    def _reading(
        cls,
        project: Project,
        state: dict[str, Any],
        client_for: Callable[[str], Any] | None = None,
    ) -> Engine:
        """An engine that only reads: it holds ``project`` and ``state`` (its own to
        change) and nothing of what work needs, no harness, no backend, no lock.

        The rules that decide which phase comes next read the state and the
        project's pages and nothing else, so :func:`next_step` asks them here
        instead of repeating them.  With ``client_for`` it can also tell whether
        the models could be used; without, it does not look.
        """
        reader = cls.__new__(cls)
        reader.project = project
        reader._state = state
        reader._lock = threading.RLock()
        reader._client_for = client_for
        reader._needs = {}
        reader._driver = None
        reader._agents = None
        reader._stretch = None
        return reader

    # -- public ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """A copy of the state as this engine last wrote it."""
        with self._lock:
            return copy.deepcopy(self._state)

    @property
    def stop_requested(self) -> bool:
        """Whether this engine was asked to stop and has not begun other work since: a
        caller that runs experiments one after the other ends its line on it."""
        return self._stop.is_set()

    def stop(self) -> None:
        """Ask a running ``advance()`` to stop.

        The phase in flight ends at its next model call (a call that is under way
        finishes; what a session wrote until then is kept), a running experiment
        is cancelled, and the phase is recorded as interrupted; ``advance()`` then
        returns, and calling it again resumes that phase.
        """
        self._stop.set()

    def advance(
        self, *, phase: str | None = None, instruction: str | None = None
    ) -> dict[str, Any]:
        """Run from the next phase (or ``phase``) up to the next checkpoint; return the state.

        With no ``phase`` named, what the person approved comes first: experiments
        that are approved or queued are run and results without an analysis are
        analysed before the phase that was next.  When anything ran, the call ends
        at the checkpoint after that analysis and the phase that was next stays
        next; when nothing could run, it goes on with that phase in the same call.
        A run the person cancelled is no result: when that is all that ran, the
        call pauses after the run phase, and nothing is analysed.
        A named phase is run as named: a run phase asked for by name pauses also
        when nothing ran, and says so.

        ``instruction`` is the person's word for this run, handed to the agents of
        the phases it covers; a phase that is resumed without one runs under the
        instruction it was started with.  When the call ends after an analysis that
        came before the phase that was next, the checkpoint says that this phase
        has not had the instruction.

        The returned state says how it ended: ``checkpoint`` (also when a spent
        budget ended a phase early: its summary and a ``budget.warning`` event say
        so, and the same call resumes the phase once the limit is raised),
        ``stopped`` (interrupted; the same call resumes) or ``failed`` (a phase
        failed twice, or cannot run as the project is set; ``error`` says why).

        Nothing that needs no model waits for one.  Every phase but the run needs
        the project's models (under the ``sdk`` runtime the run does too: an agent
        carries it out); where they cannot be used (no credential, a model
        without tools), the call stops before the first phase that needs them.
        When a phase completed before that (the run of what was approved), it
        pauses at that phase's checkpoint, which says in one sentence what is
        missing and what to do.  When nothing was done, it raises
        :class:`ModelNeeded` with that sentence, and the project is left as it
        stood: no event, no phase on record, nothing spent.

        **Autopilot.**  In a project whose person switched it on
        (``settings.autonomy.mode: autopilot``) the call does not end at its first
        checkpoint: it is the start of an unattended stretch, in which this worker
        goes on with the next phase by itself, round after round, in this process
        (the web app may be restarted meanwhile).  The stretch ends, at a
        checkpoint and with an ``autopilot.stopped`` event that says why and what
        was decided without the person, when a cap of ``settings.autonomy`` is
        reached (rounds, model spend or hours of this stretch), when a limit of
        the project itself stops the work, when a phase fails, when the same
        experiment failed twice, when a model is missing, when nothing more can
        happen without the person, or on a stop request.  Its record is
        ``state["autopilot"]`` (``research.autopilot``).  Only this call starts a
        stretch: the next one begins when the person starts the agents again.

        Raises :class:`WorkerBusy` when another worker has the project,
        :class:`ModelNeeded` as said above, :class:`EngineError` when nothing on
        file says where the project stands and no phase is named, and
        ``KeyboardInterrupt`` after recording the interruption.
        """
        if phase is not None and phase not in PHASES:
            raise EngineError(f"{phase!r} is not a phase; the phases are: {', '.join(PHASES)}.")
        instruction = (instruction or "").strip() or None
        with self._working():
            # What autopilot approved for a stretch that is over runs in no other.
            self._take_back_rule_approvals()
            why: tuple[str, str] | None = None
            try:
                self._pass(phase, instruction)
                while self._stretch is not None:
                    why = self._stretch_over()
                    if why is not None:
                        break
                    self._say_it_goes_on()
                    # A stretch may last for hours: whether the models can be used is
                    # asked anew before each further pass, not once for all of it.
                    self._needs.clear()
                    try:
                        self._pass(None, None)
                    except ModelNeeded as exc:
                        why = ("needs_model", str(exc))
                        break
            except KeyboardInterrupt:
                why = ("stopped", "It was stopped.")
                raise
            except BaseException as exc:
                why = why or ("failed", _one_line(self._clean(exc), 300))
                raise
            finally:
                if self._stretch is not None:
                    self._end_stretch(why or ("stopped", "It was stopped."))
        return self.state

    def _pass(self, phase: str | None, instruction: str | None) -> None:
        """One run of the engine from the next phase (or ``phase``) up to its next pause:
        what ``advance()`` is in a supervised project, and one step of a stretch in
        autopilot.  How it ended is left in ``_pass_end``: ``checkpoint`` (the phases
        completed and the pause after them is written), ``needs_model``, ``budget:``
        and the limit that was reached, ``interrupted`` or ``failed``.
        """
        self._pass_end = "checkpoint"
        if phase is None:
            self._refuse_unknown_position()
        # A phase that had completed without its checkpoint counts as done in this call.
        done: str | None = self._unsettled_phase() if phase is None else None
        name: str | None = phase or self._first_phase()
        told: list[str] = []
        notes: list[str] = []
        back: str | None = None
        if phase is None and name:
            name, back = self._comes_first(name)
        while name:
            need = self._model_need(name)
            if need is not None:
                self._wait_for_a_model(name, need, done, told, notes)
                self._pass_end = "needs_model"
                break
            if self._stretch is None and self._autopilot_on():
                # Work begins: so does the stretch.  (A call that starts nothing, for
                # want of a model, leaves the project as it stood.)
                self._begin_stretch()
            done = name
            outcome = self._run_phase(name, instruction)
            if outcome.status != "completed" or outcome.report is None:
                if outcome.status != "budget":
                    self._pass_end = outcome.status
                break
            # What the phases of this call did and found is told at its checkpoint.
            told.append(outcome.report.summary)
            notes += outcome.report.notes
            if name == "analyze" and back is not None and instruction:
                notes.append(self._instruction_note(instruction, back))
            name = self._after(name, outcome, told, notes, back=back, named=phase is not None)

    # -- autopilot: the unattended stretch ---------------------------------------

    def _autopilot_on(self) -> bool:
        """Whether the person's standing rule holds (``settings.autonomy.mode``), as the
        project's file says now."""
        return bool(self.project.settings.autonomy.autopilot)

    def _caps(self) -> dict[str, Any]:
        """The caps of a stretch as the settings have them now: the person may tighten
        (or widen) one while a stretch runs, and nobody else can write them."""
        autonomy = self.project.settings.autonomy
        return {
            "rounds": int(autonomy.rounds),
            "model_usd": float(autonomy.model_usd),
            "hours": float(autonomy.hours),
            "network": bool(autonomy.network),
        }

    def _begin_stretch(self) -> None:
        """Begin an unattended stretch: its record in the state, and the event that says
        what it is held to.  Called from :meth:`advance` alone, which the person starts."""
        caps = self._caps()
        spent = self._write_usage()
        stretch = autopilot.new_stretch(
            caps,
            started=_now(),
            spent_usd=spent,
            round=self._state["round"],
            unpriced_usd=self._unpriced_count()[0],
            experiment_usd=self._state["spent"]["experiment_usd"],
        )
        network = (
            "runs it approves get the network access their plans ask for"
            if caps["network"]
            else "runs it approves get no network access"
        )
        limits = self.project.settings.limits
        runs = experiments.rule_leaves_runs(self.project)
        event = self.project.emit(
            autopilot.STARTED,
            f"Autopilot goes on from here without you: {autopilot.caps_in_words(caps)}; "
            f"{network}. "
            + (f"Runs on this machine it does not approve: {_sentence(runs, 600)} " if runs else "")
            + "What the experiments cost is no part of these caps: the project's own limit "
            f"holds it ({_money(self._state['spent']['experiment_usd'])} of "
            f"{_money(float(limits.experiment_usd))} spent, settings.limits.experiment_usd). "
            + "".join(f"{note} " for note in self._no_price_note())
            + "Stop ends it.",
            agent=AUTOPILOT,
            data={"by": AUTOPILOT, "caps": caps},
            round=self._state["round"],
        )
        stretch["seq"] = event["seq"]
        self._stretch_clock = time.monotonic()
        with self._lock:
            self._stretch = self._state[autopilot.STRETCH_KEY] = stretch
            self._save()

    def _unpriced_count(self) -> tuple[float, list[str]]:
        """What the ledger's tokens on models without a known price count for against the
        cap of a stretch, and those models: each token at the highest list price there
        is (``pricing.ceiling_cost_usd``).  The ledger itself holds them with nothing,
        and a cap that counted nothing for them would never be reached."""
        with self._lock:
            report = usage_report(self.harness.usage, self._carried, self._baseline)
        unpriced = {str(model) for model in report.get("unpriced_models") or []}
        counted = sum(
            ceiling_cost_usd(Usage(**{name: int(row.get(name) or 0) for name in _USAGE_FIELDS}))
            for row in report.get("by_label") or []
            if "cost_usd" not in row and str(row.get("model")) in unpriced
        )
        return counted, sorted(unpriced)

    def _no_price_note(self) -> list[str]:
        """Said where a stretch begins on models that have no known price (the classic
        runtime's, by ``settings.models``; an sdk session reports its own cost)."""
        if self.project.settings.agents.runtime == "sdk":
            return []
        models = self.project.settings.models
        names = sorted(
            {
                str(name)
                for name in (models.research, models.reasoning, models.coding, models.fast)
                if price_for(str(name)) is None
            }
        )
        if not names:
            return []
        return [
            f"No price is known for {_and(names)}: against the cap on model spend "
            f"{'its' if len(names) == 1 else 'their'} tokens count at the highest list "
            "price there is, and what they really cost is not known here."
        ]

    def _stretch_spent(
        self, more_usd: float = 0.0, stretch: Mapping[str, Any] | None = None
    ) -> float:
        """What counts against the dollar cap of this stretch so far (``more_usd`` on top:
        what a turn in flight has used and the ledger does not hold yet): what the
        ledger prices, and for a model without a price its tokens at the highest list
        price (:meth:`_unpriced_count`)."""
        stretch = self._stretch if stretch is None else stretch
        assert stretch is not None
        with self._lock:
            priced = self._ledger_usd() - stretch["spent_at_start"]
        return max(0.0, priced + self._stretch_unpriced(stretch) + more_usd)

    def _stretch_unpriced(self, stretch: Mapping[str, Any] | None = None) -> float:
        """The part of :meth:`_stretch_spent` that is a count of tokens without a price."""
        stretch = self._stretch if stretch is None else stretch
        assert stretch is not None
        return max(0.0, self._unpriced_count()[0] - _number(stretch.get("unpriced_at_start")))

    def _stretch_account(self, stretch: dict[str, Any]) -> None:
        """Write into the record of the stretch what it has cost so far."""
        unpriced = self._stretch_unpriced(stretch)
        stretch["spent_usd"] = autopilot.round_usd(self._stretch_spent(stretch=stretch))
        stretch["unpriced_usd"] = autopilot.round_usd(unpriced)
        stretch["unpriced_models"] = self._unpriced_count()[1] if unpriced else []
        stretch["experiment_usd"] = autopilot.round_usd(
            max(
                0.0,
                self._state["spent"]["experiment_usd"]
                - _number(stretch.get("experiment_at_start")),
            )
        )

    def _stretch_hours(self) -> float:
        return (time.monotonic() - self._stretch_clock) / 3600.0

    def _stretch_cap(
        self,
        more_usd: float = 0.0,
        who: str = "",
        *,
        unpriced_usd: float = 0.0,
        unpriced_model: str = "",
    ) -> _BudgetSpent | None:
        """The cap of the stretch that is reached, as the limit that stops the work, or
        ``None``: the dollars of this stretch, or its hours.  Asked wherever the model
        budget is asked (before a session, before a turn, after every model request),
        so that no agent works past a cap for longer than one request, and before
        every approval and start of an experiment (the run phase calls no model under
        the classic runtime, and would otherwise never ask).

        ``unpriced_usd`` is the part of ``more_usd`` that is a count of the tokens of
        ``unpriced_model``, a model without a price, at the highest list price."""
        if self._stretch is None:
            return None
        caps = self._caps()
        spent, hours = self._stretch_spent(more_usd), self._stretch_hours()
        ended = f" The work of {who} was ended at its next model call." if who else ""
        again = "Continue starts a new stretch; the phase then resumes where it stopped."
        if spent >= caps["model_usd"]:
            unpriced, models = self._stretch_unpriced() + unpriced_usd, self._unpriced_count()[1]
            if unpriced and unpriced_model and unpriced_model not in models:
                models = sorted({*models, unpriced_model})
            counted = (
                f" Of that, {_money(unpriced)} counts the tokens of {_and(models)}, for which "
                "no price is known, at the highest list price there is: what they really "
                "cost is not known here."
                if unpriced and models
                else ""
            )
            return _BudgetSpent(
                "autopilot.model_usd",
                f"Autopilot's cap on model spend for this stretch is reached: {_money(spent)} "
                f"of {_money(caps['model_usd'])} went to the models since it began "
                f"(settings.autonomy.model_usd).{counted}{ended} {again}",
                spent=spent,
                allowed=caps["model_usd"],
            )
        if hours >= caps["hours"]:
            return _BudgetSpent(
                "autopilot.hours",
                f"Autopilot's cap on time for this stretch is reached: it has run for "
                f"{autopilot._hours(hours)} of {autopilot._hours(caps['hours'])} "
                f"(settings.autonomy.hours).{ended} {again}",
                spent=hours,
                allowed=caps["hours"],
            )
        return None

    def _failed_twice(self) -> str | None:
        """The experiment that failed twice since the stretch began, if one did."""
        assert self._stretch is not None
        for page in self.project.pages(kind="experiment", review="accepted"):
            if not page.id:
                continue
            try:
                count = experiments.failures(
                    self.project, str(page.id), since=self._stretch["started"]
                )
            except (ProjectError, OSError, ValueError):
                continue
            if count >= 2:
                return str(page.id)
        return None

    def _stretch_over(self) -> tuple[str, str] | None:
        """Why the stretch ends after the pass that has just ended, as ``(reason code,
        sentence)``; ``None`` when the worker goes on with the next phase.

        Going on is the exception that has to be earned: the pass ended at the
        checkpoint after phases that completed, no cap is reached, the person has
        not asked to stop or switched autopilot off, no experiment failed twice,
        and what comes next is not the phase that has just found nothing to do.
        """
        stretch, state = self._stretch, self._state
        assert stretch is not None
        end, name = self._pass_end, state.get("phase")
        if end == "interrupted" or self._stop.is_set():
            return "stopped", "It was stopped."
        if end == "failed":
            error = _one_line(state.get("error") or "no error was recorded", 300)
            return "failed", f"{phase_in_words(name).capitalize()} failed: {error}"
        if end == "needs_model":
            return "needs_model", (
                f"{phase_in_words(state.get('next_phase')).capitalize()} needs a model that "
                "cannot be used; the checkpoint says what is missing."
            )
        if end.startswith("budget:"):
            limit = end.removeprefix("budget:")
            checkpoint = state.get("checkpoint") or {}
            said = _one_line(checkpoint.get("did") or "", 600)
            if limit == "autopilot.model_usd":
                return "model_usd", said or "The cap on model spend of this stretch was reached."
            if limit == "autopilot.hours":
                return "hours", said or "The cap on hours of this stretch was reached."
            return "limits", said or "A limit of the project was reached."
        if end != "checkpoint" or state.get("status") != "checkpoint":
            return "stopped", f"The work ended without a checkpoint ({end or 'no account'})."
        with contextlib.suppress(ProjectError):
            self.project.reload()
        if not self._autopilot_on():
            return "switched_off", "You switched autopilot off."
        caps = self._caps()
        with self._lock:
            stretch["passes"] += 1
            if name == "analyze":
                stretch["rounds"] += 1
            stretch["caps"] = caps
            self._stretch_account(stretch)
            self._save()
        twice = self._failed_twice()
        if twice:
            return "experiment_failed_twice", (
                f"{twice} failed twice in this stretch; autopilot does not try a third time."
            )
        if stretch["rounds"] >= caps["rounds"]:
            done = stretch["rounds"]
            return "rounds", (
                f"The cap on rounds was reached: {_plural(done, 'round')} completed in this "
                f"stretch, of at most {caps['rounds']} (settings.autonomy.rounds)."
            )
        cap = self._stretch_cap()
        if cap is not None:
            return cap.limit.removeprefix("autopilot."), str(cap)
        following = state.get("next_phase")
        if following == name:
            # The phase found nothing it could do (a run phase with nothing it may
            # run): doing it again would find the same.
            waits = [(eid, why) for eid, why in self._held.items() if why]
            return "needs_you", (
                "Nothing more can happen without you: "
                + (
                    f"autopilot did not approve {autopilot.grouped(waits)}."
                    if waits
                    else self._nothing_to_run()
                )
            )
        return None

    def _say_it_goes_on(self) -> None:
        """Say in the feed that the worker goes on by itself, and where the stretch stands."""
        stretch, state = self._stretch, self._state
        assert stretch is not None
        caps = stretch["caps"]
        following = str(state.get("next_phase"))
        self.project.emit(
            autopilot.CONTINUED,
            f"Autopilot goes on with {phase_in_words(following)}: "
            f"{_plural(stretch['rounds'], 'round')} of at most {caps['rounds']} completed in "
            f"this stretch, {_money(stretch['spent_usd'])} of {_money(caps['model_usd'])} spent "
            f"on models, {autopilot._hours(self._stretch_hours())} of "
            f"{autopilot._hours(caps['hours'])} used.",
            agent=AUTOPILOT,
            data={"by": AUTOPILOT, "next": following, "stretch": dict(stretch)},
            round=state["round"],
        )

    def _end_stretch(self, why: tuple[str, str]) -> None:
        """End the stretch: take back the approvals it did not get to run, write why it
        ended and what was decided without the person, and say so in the feed.

        Runs when the worker leaves ``advance()`` for whatever reason, so nothing in
        it may hide the reason: a failure to write the account is logged and the
        stretch is over all the same (the next worker finds it without an end and
        records that).
        """
        stretch, self._stretch = self._stretch, None
        if stretch is None:
            return
        code, said = why
        try:
            self._take_back_rule_approvals()
            with self._lock:
                stretch.update(ended=_now(), reason=code, said=_one_line(said, 900))
                stretch["caps"] = self._caps()
                self._write_usage()
                self._stretch_account(stretch)
                self._save()
            found = autopilot.digest(self.project, self._state)
            account = autopilot.summary(found) if found else ""
            self.project.emit(
                autopilot.STOPPED,
                _one_line(f"Autopilot stopped: {_sentence(said, 900)} {account}", 2400),
                agent=AUTOPILOT,
                data={
                    "by": AUTOPILOT,
                    "reason": code,
                    "stretch": dict(stretch),
                    "accepted": len(found["accepted"]) if found else 0,
                    "edits": len(found["edits"]) if found else 0,
                    "decisions": len(found["decisions"]) if found else 0,
                    "runs": [run["id"] for run in found["runs"]] if found else [],
                    "held": found["held"] if found else [],
                },
                round=self._state["round"],
            )
        except Exception:
            log.exception("The end of the unattended stretch could not be recorded.")

    def _take_back_rule_approvals(self) -> None:
        """Make a plan again of every experiment that autopilot approved and that has
        not run: its approval was for the stretch it was made in.

        A stretch that ends between an approval and its run (a cap, a stop, a worker
        that was killed) leaves such an experiment approved or in the queue.  Nobody
        but the worker of a stretch runs what the rule approved: without this, the
        next `research run`, or the app when it starts what waits in line, would run
        it with nobody having asked and no cap counting.  The next stretch approves
        it again.
        """
        try:
            waiting = self.project.pages(
                kind="experiment", review="accepted", state=["approved", "queued"]
            )
        except (ProjectError, OSError):
            return
        for page in waiting:
            if not experiments.approved_by_rule(page):
                continue
            with contextlib.suppress(ProjectError):
                self.project.set_state(
                    page.path,
                    "planned",
                    by="system",
                    reason="autopilot approved it for a stretch that ended before it ran; "
                    "the next stretch approves it again",
                )

    def run_experiment(
        self,
        experiment_ref: str,
        *,
        backend: ComputeBackend | str | None = None,
        limits: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one approved experiment now, outside the order of the phases.

        ``backend`` (a backend or its name) replaces the one chosen at approval for
        this run; ``limits`` replaces values of ``settings.limits`` for this run
        only.  Returns the final run record.  The experiment must be accepted and
        approved by the person: this runs what they approved, it approves nothing.
        """
        with self._working():
            self._recover()
            # What autopilot approved was for its stretch, and this is none.
            self._take_back_rule_approvals()
            page = self.project.page(experiment_ref)
            if page.kind != "experiment" or not page.id:
                raise EngineError(
                    f"{experiment_ref} is {page.kind} {page.label}, not an experiment."
                )
            if page.review != "accepted" or page.state not in ("approved", "queued"):
                raise EngineError(
                    f"{page.id} is {page.review} and {page.state}; only an experiment the "
                    "person accepted and approved is run. Approve it with `paper-orchestra "
                    "research approve` or in the app, then run it again."
                )
            self._tidy_hypotheses(release=False)
            named = backend if isinstance(backend, str) else getattr(backend, "name", None)
            if self._sdk() and self._agent_carries_out(page, named):
                # An agent carries the experiment out: it needs what every agent needs.
                # (A run approved for a remote backend runs there, without an agent.)
                need = self._agents_need()
                if need is not None:
                    sentence = agents.need_sentence(need, f"Carrying out {page.id}")
                    raise ModelNeeded(sentence, phase="run", need=need)
            try:
                self._check_experiment_budget(page, self._limits_with(limits))
            except _BudgetSpent as exc:
                self._warn_budget(exc, phase="run")
                raise EngineError(str(exc)) from None
            compute = self._compute.named(backend) if isinstance(backend, str) else backend
            round = max(1, self._state["round"])
            ctx = self._context("run", round, f"r{round}-run-{page.id}", (), None)
            try:
                return self._run_one(page, ctx, compute=compute, limits=limits)
            except _BudgetSpent as exc:
                self._warn_budget(exc, phase="run")
                raise EngineError(str(exc)) from None
            except agents.SessionStopped as exc:
                if isinstance(exc.reason, _BudgetSpent):
                    spent = self._as_spent(exc.reason)
                    self._warn_budget(spent, phase="run")
                    raise EngineError(str(spent)) from None
                # A stop request: the run went back to the queue, and its record says so.
                runs = experiments.runs(self.project, str(page.id))
                return runs[-1] if runs else {}

    # -- the worker --------------------------------------------------------------

    @contextlib.contextmanager
    def _working(self) -> Iterator[None]:
        """Hold the project's worker lock, with a heartbeat and a stop on SIGTERM."""
        lock = _WorkerLock(self.project.root)
        with lock:
            if not lock.acquired:
                raise WorkerBusy(self._busy_message(lock.error))
            _refuse_unwritable_state(self.project)
            self._stop.clear()
            forget_worker_stop(self.project.root)  # an earlier worker's, if any
            self.project.reload()
            self._needs.clear()  # the settings, and what is configured, as they are now
            # What writers that were killed left behind does not pile up crash by crash.
            swept = self.project.sweep_temporary_files()
            if swept:
                log.info("Removed %d temporary file(s) of processes that are gone.", len(swept))
            with self._lock:
                lost = self._lost = state_problem(self.project)
                self._state = read_state(self.project)
                self._settle_dead_worker()
                left = self._state.get(autopilot.STRETCH_KEY)
                if left is not None and not left["ended"]:
                    # A stretch whose worker went without ending it (it was killed, or
                    # the machine went down): it is over, and is not taken up again.
                    left.update(ended=_now(), reason="worker_gone", said=None)
                self._stretch = None
                self._count_runs()
                started = _now()
                self._state["worker"] = {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started": started,
                    "heartbeat": started,
                }
                if self.drains:
                    # `research run --queue`: whoever approves an experiment meanwhile
                    # can tell that this worker goes on to it.
                    self._state["worker"]["queue"] = True
                self._carried = read_carried_usage(self.project.root / USAGE_FILE)
                self._baseline = self.harness.usage.to_dict()["by_label"]
                missing = self._carry_recorded_spend()
                self._save()
            if lost and self._state["phases"]:
                self._say_rebuilt(lost)
            if missing:
                self._say_carried(missing)
            beat = _Heartbeat(self._beat, self.HEARTBEAT_SECONDS)
            beat.start()
            restore = self._stop_on_sigterm()
            stopped = False
            try:
                yield
            except KeyboardInterrupt:
                stopped = True
                raise
            finally:
                restore()
                beat.stop()
                self._compute.close()
                if stopped or self._stop.is_set():
                    # Said before the name is taken off: whoever watches this worker
                    # from outside (the app's server, for one started in a terminal)
                    # sees the name go, and must find by then that it was stopped.
                    record_worker_stop(self.project.root)
                with self._lock:
                    self._state["worker"] = None
                    self._write_usage()
                    self._save()

    def _busy_message(self, error: str | None) -> str:
        if error:
            return (
                f"The project at {self.project.root} could not be locked for a worker "
                f"({error}). Check that the directory can be written."
            )
        state = read_state(self.project)
        worker = state.get("worker") or {}
        who = f" (process {worker.get('pid')} on {worker.get('host')})" if worker else ""
        message = (
            f"Another worker is already running this project{who}. Wait until it reaches "
            "a checkpoint, or stop it first; a project has one worker at a time."
        )
        if not worker_alive(state):
            # Where the file system has no kernel locks, a lock outlives a worker that
            # crashed for a while; the person can tell that nothing runs, the code cannot.
            message += (
                f" If no worker is running (one that crashed can leave its lock behind on "
                f"some file systems), delete {self.project.root / WORKER_LOCK_FILE} and "
                "try again."
            )
        return message

    def _stop_on_sigterm(self) -> Callable[[], None]:
        """Turn SIGTERM into a stop request while this worker runs; returns the undo.

        Only where nothing handles the signal yet (it would end the process on the
        spot and leave the state at ``running``).  A program that has its own way of
        stopping, as the command line does by raising ``KeyboardInterrupt``, keeps it:
        the engine records an interrupt just the same.
        """
        if threading.current_thread() is not threading.main_thread():
            return lambda: None
        try:
            if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
                return lambda: None
            signal.signal(signal.SIGTERM, lambda *_: self.stop())
        except (ValueError, OSError):
            return lambda: None
        return lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL)

    def _settle_dead_worker(self) -> None:
        """A state left at ``running`` belongs to a worker that died.

        The phase it was in is recorded as interrupted and comes next again.  When
        that phase had already completed, the worker died between two phases:
        nothing was in flight, and what was to follow still follows.
        """
        state = self._state
        if state["status"] != "running" or state["phase"] not in PHASES:
            return
        state.update(status="stopped", error=None)
        worker = state.get("worker") if isinstance(state.get("worker"), Mapping) else {}
        record = self._last_record(state["phase"], state["round"])
        if record is not None and record.get("status") == "completed":
            return
        if record is None:
            record = self._new_record(state["phase"], state["round"], None)
            record.update(started=worker.get("started") or record["started"], attempts=1)
            state["phases"].append(record)
        record.update(
            status="interrupted",
            finished=worker.get("heartbeat") or _now(),
            error="The worker stopped before the phase finished.",
        )
        state["next_phase"] = state["phase"]

    def _say_rebuilt(self, problem: str) -> None:
        """Tell the feed that the state file was lost and what it was rebuilt as."""
        state = self._state
        following = state["next_phase"] if state["next_phase"] in PHASES else None
        self.project.emit(
            "error",
            f"{STATE_FILE} was lost or damaged ({_one_line(problem.split('. ')[0], 120)}) and "
            f"has been rebuilt from the project's events (events.jsonl): round "
            f"{state['round']}, {_plural(len(state['phases']), 'phase')} on record"
            + (f", next: {_DOING[following]}" if following else "")
            + ". Check that this is where the project stood.",
            round=state["round"],
            data={"rebuilt": STATE_FILE, "round": state["round"], "next": following},
        )

    def _ledger_usd(self) -> float:
        """What the usage ledger adds up to, with what was carried over a lost one."""
        report = usage_report(self.harness.usage, self._carried, self._baseline)
        return _number(report.get("estimated_cost_usd")) + self._carried_usd

    def _carry_recorded_spend(self) -> float:
        """Keep what the state file recorded as spent on models when the usage ledger
        accounts for less; return the difference (nothing when the ledger covers it).

        The ledger (usage.json) is where the spend is counted, and state.json only
        repeats its sum.  When the ledger is lost or cannot be read, that sum is
        all that is left of what the project spent: without it the count would
        start again at zero and the model budget would no longer mean the
        project's total.
        """
        try:
            ledger = json.loads((self.project.root / USAGE_FILE).read_text(encoding="utf-8"))
            self._carried_usd = _number(ledger.get("carried_usd"))
        except (OSError, ValueError, AttributeError):
            self._carried_usd = 0.0
        recorded = self._state["spent"]["model_usd"]
        missing = round(recorded - self._ledger_usd(), 4)
        # Both sums are estimates rounded to four places, priced from a table that may
        # have been corrected since: a difference that small is not a lost ledger.
        if missing <= 0.0002 + 0.001 * recorded:
            return 0.0
        self._carried_usd = round(self._carried_usd + missing, 4)
        return missing

    def _say_carried(self, missing: float) -> None:
        limit = float(self.project.settings.limits.model_usd)
        spent = self._write_usage()
        self.project.emit(
            "budget.warning",
            f"The usage ledger ({USAGE_FILE}) was lost or accounts for less than the "
            f"project had recorded as spent on models: {_money(missing)} of that record "
            f"is carried over, so the model budget still counts it ({_money(spent)} of "
            f"{_money(limit)} is spent). The token counts behind that amount are gone.",
            round=max(1, self._state["round"]),
            data={"limit": "model_usd", "spent": spent, "allowed": limit, "carried": missing},
        )

    def _beat(self) -> None:
        with self._lock:
            worker = self._state.get("worker")
            if not isinstance(worker, dict):
                return
            worker["heartbeat"] = _now()
            self._write_usage()
            self._save()

    def _save(self) -> None:
        with self._lock:
            write_json(self.project.root / STATE_FILE, self._state)

    def _write_usage(self) -> float:
        """Bring usage.json and ``spent.model_usd`` up to date; return the model spend.

        That is what the harness's ledger holds, which learns of a turn's usage when
        the turn ends; :meth:`_over_budget` adds what a turn in flight has used.
        """
        with self._lock:
            report = usage_report(self.harness.usage, self._carried, self._baseline)
            self._unpriced = [str(model) for model in report.get("unpriced_models") or []]
            spent = _number(report.get("estimated_cost_usd")) + self._carried_usd
            if self._carried_usd:
                # Spend the rows above do not account for: recorded before the ledger
                # was lost.  On file, so that the next worker carries it too.
                report["carried_usd"] = self._carried_usd
                report["estimated_cost_usd"] = round(spent, 4)
            # What agents that were stopped or killed never reported, counted from their
            # tokens at the list price: part of the sum, and a lower bound of what it
            # stands for.  Said beside the sum by whoever shows it.
            unreported = sum(
                _number(row.get("cost_usd"))
                for row in report.get("by_label") or []
                if str(row.get("label") or "").endswith(UNREPORTED_LABEL)
            )
            if unreported:
                report["unreported_usd"] = round(unreported, 4)
            # Such an agent on a model without a list price is in the sum with nothing:
            # named here, so that the sum is still shown as the lower bound it is.
            unpriced = sorted(
                {
                    str(row.get("model"))
                    for row in report.get("by_label") or []
                    if str(row.get("label") or "").endswith(UNREPORTED_LABEL)
                    and "cost_usd" not in row
                    and str(row.get("model")) in self._unpriced
                }
            )
            if unpriced:
                report["unreported_unpriced"] = unpriced
            write_json(self.project.root / USAGE_FILE, report)
            self._state["spent"]["model_usd"] = round(spent, 4)
            return spent

    def _emit(
        self,
        type: str,
        summary: str,
        *,
        phase: str,
        round: int,
        refs: Sequence[str] = (),
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.project.emit(
            type, summary, phase=phase, agent=_ROLE[phase], refs=refs, data=data, round=round
        )

    # -- what may run ------------------------------------------------------------

    def _clean(self, text: Any) -> str:
        """``text`` without the credentials this process holds.

        A library may quote the request it failed with, key and all, in its error.
        What the engine writes of an error goes into state.json and events.jsonl,
        which the app, the command line and later sessions read: the value is
        blanked before it gets there.
        """
        return experiments.mask_credentials(self.project, str(text))

    def _models_need(self, settings: Sequence[str] | None = None) -> dict[str, Any] | None:
        """What keeps the project's models (those of ``settings``; all of them without)
        from being used, as ``agents.model_need`` finds it; ``None`` when they can be,
        and for an engine that was not told how a client is found.  Asked once for
        each call that works: nothing is contacted, but a client is built for it."""
        if self._client_for is None:
            return None
        key = tuple(settings or ())
        if key not in self._needs:
            self._needs[key] = agents.model_need(self.project, self._client_for, settings=settings)
        return self._needs[key]

    def _model_need(self, name: str) -> dict[str, Any] | None:
        """What keeps the phase ``name`` from starting for want of a model; ``None`` when
        nothing does.

        The run phase needs none: it executes what the person approved.  Every
        other phase is an agent's session, and starts only when all the project's
        models can be used, as before: a round whose research can be paid for and
        whose design then cannot be done is refused before anything is spent.
        """
        if self._sdk():
            if name == "run" and not self._runs_need_agents():
                return None
            return self._agents_need()
        return None if name == "run" else self._models_need()

    @staticmethod
    def _agent_carries_out(page: ResearchPage, backend: Any = None) -> bool:
        """Whether a run of ``page`` under the sdk runtime is an agent's work: one that
        was approved for this machine.  A run approved for a remote backend runs there
        as it always did, and waits for nothing the agents need."""
        named = str(backend or page.meta.get("backend") or "local")
        return named == "local"

    def _runs_need_agents(self) -> bool:
        """Whether the run phase would start an agent: some approved run is for this
        machine (or nothing tells, and the gate then asks as for every phase)."""
        try:
            waiting = [
                page
                for page in self.project.pages(kind="experiment")
                if page.review == "accepted" and page.state in ("approved", "queued")
            ]
        except Exception:
            return True
        return not waiting or any(self._agent_carries_out(page) for page in waiting)

    def _sdk(self) -> bool:
        """Whether the project's agents are Claude Agent SDK agents (``settings.agents``)."""
        return bool(self.project.settings.agents.sdk)

    def _runtime(self) -> Any:
        """The ``sdk`` runtime of this engine: how its agents are started."""
        if self._agents is None:
            from paper_orchestra.research.sdk.session import AgentRuntime

            self._agents = AgentRuntime(self.project, self._driver)
        return self._agents

    def _agents_need(self) -> dict[str, Any] | None:
        """What keeps the project's SDK agents from starting, in the shape of
        :meth:`_models_need`; ``None`` when they can start, and for an engine that
        was not told to look.

        Under the ``sdk`` runtime this is the whole gate, for every phase: the
        models of ``settings.models`` are not used, so they are not asked about,
        and the run phase is no exception, because an approved experiment is
        carried out by an agent too.
        """
        if self._client_for is None:
            return None
        key = ("agents",)
        if key not in self._needs:
            self._needs[key] = self._runtime().need()
        return self._needs[key]

    def _budget_left(self) -> float:
        """What is left of the model budget, for a session that is told a sum."""
        with self._lock:
            return float(self.project.settings.limits.model_usd) - self._ledger_usd()

    def _wait_for_a_model(
        self,
        name: str,
        need: Mapping[str, Any],
        done: str | None,
        told: Sequence[str],
        notes: Sequence[str],
    ) -> None:
        """Stop before the phase ``name``, which needs a model that cannot be used.

        ``done`` is the phase this call completed last.  With one, the call pauses
        at its checkpoint, which keeps what the phases did (``told``, ``notes``)
        and says what is missing in one sentence; ``name`` is next.  With none,
        nothing was done and nothing is recorded: :class:`ModelNeeded` carries the
        sentence to whoever asked.
        """
        sentence = agents.need_sentence(need, phase_in_words(name).capitalize())
        if done is None:
            raise ModelNeeded(sentence, phase=name, need=need)
        self._checkpoint(done, self._state["round"], name, told, notes, missing=[sentence])

    def _position_unknown(self) -> bool:
        """Whether nothing on file says where the project stands although agents have
        worked on it: no phase on record, and pages in the wiki that only a phase
        leaves behind."""
        return not self._state["phases"] and _agents_wrote(self.project)

    def _refuse_unknown_position(self) -> None:
        """Refuse to choose a phase when nothing on file says where the project stands.

        Without phase records in state.json and without phase events, the first
        phase would be intake: on a project whose wiki holds the agents' pages that
        would do its first steps a second time, and pay for them.
        """
        if not self._position_unknown():
            return
        # (As the file was found: this worker has written an empty one since.)
        problem = self._lost or state_problem(self.project) or ""
        message = (
            f"{problem} For example `paper-orchestra research "
            f"advance {self.project.root} --phase hypothesize` (the phases are: "
            f"{', '.join(PHASES)}), or Continue with a phase in the app."
        ).strip()
        self.project.emit("error", _one_line(message, 600), data={"state": "unknown"})
        raise EngineError(message)

    def _guard(self) -> None:
        """Called before every model session and every turn of one: a stop request and
        the model budget."""
        self._heed_stop()
        limit = float(self.project.settings.limits.model_usd)
        spent = self._write_usage()
        if spent >= limit:
            raise _BudgetSpent(
                "model_usd",
                f"The model budget is used up: {_money(spent)} of {_limit_money(limit)} is spent. "
                "Raise settings.limits.model_usd in the project's settings to go on; the "
                "phase then resumes where it stopped.",
                spent=spent,
                allowed=limit,
            )
        cap = self._stretch_cap()
        if cap is not None:
            raise cap

    def _over_budget(
        self,
        model: str,
        usage: Usage,
        *,
        phase: str | None = None,
        experiment: str | None = None,
        more_usd: float = 0.0,
    ) -> _BudgetSpent | None:
        """Asked after every model request of a session's turn: is the model budget spent?

        ``phase`` is the phase whose agent is at work, and ``experiment`` the one an
        agent carries out where it is that agent: the sentence names who was
        stopped, and what happens to the work when the limit is raised.

        ``usage`` is what the turn has used so far on ``model``.  The ledger does not
        hold it yet, so it is priced here and added to what the ledger holds.  A turn
        that is past the limit is ended at its next model call (``AgentContext``) and
        the phase stops at a checkpoint, as it does when the limit is found spent
        before a session; without this, one turn of up to ``agents.MAX_TURNS``
        requests could spend many times the limit before anything looked.  A model
        without a known price counts as nothing here, as it does in the ledger.
        ``more_usd`` is what the turn cost on other models, priced by whoever asks.
        """
        limit = float(self.project.settings.limits.model_usd)
        priced = estimate_cost_usd(model, usage)
        in_flight = (priced or 0.0) + more_usd
        with self._lock:
            spent = self._ledger_usd() + in_flight
        if spent < limit:
            # The project's budget holds; the cap of an unattended stretch may not.
            who = (
                f"the agent that carries out {experiment}"
                if experiment
                else f"the {_ROLE.get(str(phase or self._state.get('phase')), 'agent')}"
            )
            # (Which counts a model without a price at the highest list price: a cap
            # that counted nothing for it would never end the stretch.)
            ceiling = ceiling_cost_usd(usage) if priced is None and self._stretch else 0.0
            return self._stretch_cap(
                in_flight + ceiling, who, unpriced_usd=ceiling, unpriced_model=str(model)
            )
        if experiment:
            work = f"the work of the agent that carries out {experiment}"
            then = f"{experiment} is in the queue again and is run when the work goes on."
        else:
            role = _ROLE.get(str(phase or self._state.get("phase")), "agent")
            work = f"the {role}'s work"
            then = "the phase then resumes where it stopped."

        def say(spent: float) -> str:
            return (
                f"The model budget ran out in the middle of {work}: {_money(spent)} of "
                f"{_limit_money(limit)} is spent, so its session was ended at its next model "
                "call. "
                "What it had written until then is kept. Raise settings.limits.model_usd in "
                f"the project's settings to go on; {then}"
            )

        exc = _BudgetSpent("model_usd", say(spent), spent=spent, allowed=limit)
        exc.say = say
        return exc

    def _short_of(
        self, needed: float, *, phase: str | None = None, experiment: str | None = None
    ) -> _BudgetSpent | None:
        """Asked before an agent is started (sdk runtime) with what the start of an agent
        was seen to cost in this project: is less than that left of the model budget?

        An agent's first request sends everything it is given, and in fast mode that
        alone was seen to cost most of a dollar.  Started with less than that left,
        the agent passes the limit within seconds, before any count can end it: it is
        not started, and the work stops at a checkpoint as for a budget that is spent.
        """
        limit = float(self.project.settings.limits.model_usd)
        with self._lock:
            spent = self._ledger_usd()
        if limit - spent >= needed:
            return self._stretch_short_of(needed, phase=phase, experiment=experiment)
        if experiment:
            who = f"the agent that carries out {experiment}"
            then = f"{experiment} keeps its approval and is run when the work goes on."
        else:
            role = _ROLE.get(str(phase or self._state.get("phase")), "agent")
            who = f"the {role}"
            then = "the phase then starts where it stopped."
        return _BudgetSpent(
            "model_usd",
            f"What is left of the model budget does not cover the start of {who}: "
            f"{_money(spent)} of {_limit_money(limit)} is spent, and the first request of an "
            f"agent was last seen to cost about {_money(needed)} here, so it was not "
            "started. Raise settings.limits.model_usd in the project's settings to go on; "
            f"{then}",
            spent=spent,
            allowed=limit,
        )

    def _stretch_short_of(
        self, needed: float, *, phase: str | None = None, experiment: str | None = None
    ) -> _BudgetSpent | None:
        """:meth:`_short_of` for the cap of an unattended stretch: an agent that would
        pass it with its first request is not started."""
        if self._stretch is None:
            return None
        cap = float(self._caps()["model_usd"])
        spent = self._stretch_spent()
        if cap - spent >= needed:
            return None
        who = (
            f"the agent that carries out {experiment}"
            if experiment
            else f"the {_ROLE.get(str(phase or self._state.get('phase')), 'agent')}"
        )
        return _BudgetSpent(
            "autopilot.model_usd",
            f"What is left of autopilot's cap on model spend for this stretch does not cover "
            f"the start of {who}: {_money(spent)} of {_money(cap)} went to the models since "
            f"it began (settings.autonomy.model_usd), and the first request of an agent was "
            f"last seen to cost about {_money(needed)} here, so it was not started. Continue "
            "starts a new stretch.",
            spent=spent,
            allowed=cap,
        )

    def _as_spent(self, exc: _BudgetSpent) -> _BudgetSpent:
        """``exc`` with the amount the ledger holds now.

        The amount in the sentence of a session that was ended in the middle of its
        turn is a count from tokens at the list price.  Under the sdk runtime the
        agent's own figure arrives when its turn has ended, and with fast mode or
        helpers it is several times that count: the sentence that stays on the
        checkpoint must name what the Spent line beside it names.
        """
        if exc.say is None:
            return exc
        spent = self._write_usage()
        # Only upwards: in the classic runtime the ledger never learns of a turn that
        # was cut short, and the count that ended it is then the larger and truer sum.
        if spent < exc.spent + 0.005:
            return exc
        again = _BudgetSpent(exc.limit, exc.say(spent), spent=spent, allowed=exc.allowed)
        again.say = exc.say
        return again

    def _heed_stop(self) -> None:
        if self._stop.is_set():
            raise RunCancelled("the engine was asked to stop")

    def _stopping(self) -> bool:
        return self._stop.is_set() or self.harness.cancelled

    def _check_experiment_budget(self, page: ResearchPage, limits: Limits | None = None) -> None:
        """Raise when what the project may spend on compute does not cover a run of ``page``.

        ``limits.experiment_usd`` is the total for all runs of the project.  The time
        limit (``experiment_seconds``) is a cap on one run, which
        ``research.experiments`` enforces; the time used is only recorded.
        """
        limits = limits or self.project.settings.limits
        usd, allowed = self._state["spent"]["experiment_usd"], float(limits.experiment_usd)
        estimate = _number(page.meta.get("est_usd"))
        if (usd > 0 and usd >= allowed) or (estimate > 0 and usd + estimate > allowed):
            planned = f" and {page.id} is estimated at {_money(estimate)}" if estimate else ""
            raise _BudgetSpent(
                "experiment_usd",
                f"The experiment cost budget does not cover {page.id}: {_money(usd)} of "
                f"{_money(allowed)} is spent{planned}, so it was not started. Raise "
                "settings.limits.experiment_usd in the project's settings to run it.",
                spent=usd,
                allowed=allowed,
            )

    def _limits_with(self, overrides: Mapping[str, Any] | None) -> Limits:
        """The project's limits with ``overrides`` in place of their values, for one run."""
        original = self.project.settings.limits
        if not overrides:
            return original
        problems: list[str] = []
        changed = Limits.load({**original.to_dict(), **overrides}, "limits", problems)
        if problems:
            raise EngineError("These limits cannot be used: " + "; ".join(problems) + ".")
        return changed

    def _warn_budget(self, exc: _BudgetSpent, *, phase: str) -> None:
        self._emit(
            "budget.warning",
            str(exc),
            phase=phase,
            round=max(1, self._state["round"]),
            data={"limit": exc.limit, "spent": exc.spent, "allowed": exc.allowed},
        )

    # -- which phase, which round ------------------------------------------------

    def _unsettled_phase(self) -> str | None:
        """The phase that completed without what follows it having been decided.

        A worker that dies between the last write of a phase and its checkpoint
        leaves the state at ``stopped`` (as this engine or the web server settles
        it) with that phase completed and no checkpoint.  The pause the person is
        owed must not be lost with the worker.
        """
        state = self._state
        last = state["phases"][-1] if state["phases"] else None
        owed = (
            state["status"] == "stopped"
            and state["checkpoint"] is None
            and last is not None
            and last.get("status") == "completed"
            and last.get("phase") in PHASES
            and last.get("phase") == state["phase"]
            and last.get("round") == state["round"]
        )
        return state["phase"] if owed else None

    def _first_phase(self) -> str | None:
        """The phase ``advance()`` starts with when none is named; ``None`` when all
        there was to do was the checkpoint of a phase whose worker died right after it."""
        owed = self._unsettled_phase()
        if owed is None:
            return self._next_phase()
        record = self._state["phases"][-1]
        self._complete_record(record)
        note = f"The worker stopped right after the {_NOUN[owed]} phase had finished."
        # What the worker may not have got round to: closing the requests the phase was
        # given, and saying in the feed that it finished.
        since = self._events_of(record)
        if not any(event.get("type") == "phase.finished" for event in since):
            self._announce(record)
        else:
            self._settle_requests(record)
        written = self._announced(since)
        if written is not None:
            # It had announced the pause and died before the state said so: one pause
            # is one event, so the event that is there is the checkpoint.
            self._adopt_checkpoint(written, note)
            return None
        return self._after(owed, _Outcome("completed"), [], [note])

    @staticmethod
    def _announced(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        """The checkpoint among the events of a phase's latest run, when it was announced."""
        written = [event for event in events if event.get("type") == "checkpoint"]
        return written[-1] if written else None

    @staticmethod
    def _next_of(event: Mapping[str, Any]) -> str:
        """The phase a checkpoint event names as next."""
        data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
        following = data.get("next")
        return str(following) if following in PHASES else _NEXT[str(event["phase"])]

    def _events_of(self, record: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The events of the phase ``record`` is of, from its latest start on."""
        name, round = record["phase"], record["round"]
        events = [
            event
            for event in self.project.events()
            if event.get("phase") == name and event.get("round") == round
        ]
        starts = [i for i, event in enumerate(events) if event.get("type") == "phase.started"]
        return events[starts[-1] :] if starts else events

    def _adopt_checkpoint(self, event: Mapping[str, Any], note: str) -> None:
        """Make the state say what a checkpoint event on file announced."""
        data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
        name, round = str(event["phase"]), int(event["round"])
        following = self._next_of(event)
        checkpoint = _read_checkpoint(
            {
                "round": round,
                "phase": name,
                "at": event.get("ts"),
                "summary": f"{note} {event.get('summary') or ''}".strip(),
                "pending": data.get("pending"),
                **_recommendation(data),
            }
        )
        with self._lock:
            self._state.update(
                status="checkpoint",
                round=round,
                phase=name,
                next_phase=following,
                error=None,
                checkpoint=checkpoint,
            )
            self._save()

    def _since_last_phase(self) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The latest record of a phase other than run and analysis, and the runs and
        analyses on record after it, oldest first."""
        phases = self._state["phases"]
        for index in range(len(phases) - 1, -1, -1):
            if phases[index].get("phase") not in _RUNS:
                return phases[index], phases[index + 1 :]
        return None, list(phases)

    @staticmethod
    def _analysed(records: Sequence[Mapping[str, Any]]) -> bool:
        """Whether an analysis among ``records`` has completed."""
        return any(
            record.get("phase") == "analyze" and record.get("status") == "completed"
            for record in records
        )

    def _return_phase(self) -> str | None:
        """The phase the work goes back to after the run and the analysis at the end of
        the records, when those were not the round's own; ``None`` when they were.

        The round's own run follows its design phase, and the first analysis that
        completes after it ends the round.  A run the person approved at any
        other point (an experiment run once more, one of their own) and its
        analysis are done where they fall, and afterwards the round is where it
        was: at the phase that was interrupted, at the one after the last that
        completed, at the first phase of the next round when the round's own
        analysis had completed already, or at intake when no phase has run yet.
        Read from the records, so that a stop or a failure in between does not
        lose it.
        """
        base, since = self._since_last_phase()
        if base is None:
            # Only runs and analyses so far: the project itself is where it started.
            return "intake"
        if base.get("phase") not in PHASES:
            return None
        if base.get("status") != "completed":
            return str(base["phase"])
        following = _NEXT[base["phase"]]
        if following != "run":
            return following
        # After the design phase.  The last record is the run or the analysis that is
        # asked about: it is the round's own unless the round's analysis completed
        # before it, at whose checkpoint the person then approved this one.
        return _NEXT["analyze"] if self._analysed(since[:-1]) else None

    def _after_last_record(self) -> str:
        """What comes next going by the phase records alone."""
        last = self._state["phases"][-1] if self._state["phases"] else None
        if last is None or last.get("phase") not in PHASES:
            return "intake"
        if last.get("status") != "completed":
            return str(last["phase"])
        if last["phase"] == "analyze":
            return self._return_phase() or _NEXT["analyze"]
        return _NEXT[last["phase"]]

    def _next_phase(self) -> str:
        """The phase ``advance()`` goes on with when none is named.

        What the state file says is next, as the last checkpoint told the person.
        Nothing else moves it: not a request, whatever its words, and not what an
        agent wrote.  More research is the person's choice, made by naming the phase.
        """
        name = self._state.get("next_phase")
        if name not in PHASES:
            # A state file that does not say (written by hand, or by an older version):
            # going back to intake would do the project's first steps a second time.
            name = self._after_last_record()
        return str(name)

    def _overdue(self, name: str) -> str | None:
        """What ``advance()`` does before ``name``, the phase that is next, when no phase
        was named: ``run`` when an experiment is approved (or was left queued or
        running by a worker that is gone), ``analyze`` when a result has no analysis
        yet, ``None`` when neither is the case or ``name`` is that phase already.

        The person approved a run for it to happen, and a result that nobody has
        looked at should not wait behind phases that would have built on it.  An
        analysis that is still to be finished waits for what was approved since, and
        then reads all of it.
        """
        if name == "run":
            return None
        if self._awaiting_run():
            return "run"
        return "analyze" if name != "analyze" and self._new_results() else None

    def _awaiting_run(self) -> list[ResearchPage]:
        """The accepted experiments a run phase is owed: approved, or left queued or
        running by a worker that is gone."""
        return self.project.pages(
            kind="experiment", review="accepted", state=["approved", "queued", "running"]
        )

    def _comes_first(self, name: str) -> tuple[str, str | None]:
        """Where a call that names no phase begins when ``name`` is the phase that is
        next: ``(the phase it starts with, the phase that was next and stays so)``.

        The second is ``None`` when the call starts with ``name`` itself, and also
        before an analysis that is still to be finished: the run is then the round's
        own or returns where the records say, and there is nothing to remember.
        """
        first = self._overdue(name)
        if first is None:
            return name, None
        return first, (None if name in _RUNS else name)

    @staticmethod
    def _instruction_note(instruction: str, back: str) -> str:
        """What the checkpoint says when a call that was given an instruction ended after
        the analysis of what the person had approved, before the phase that was next.

        The instruction went to the phases the call ran.  The person may have
        written it for the phase the last checkpoint named as next, which has not
        run: they are told, instead of finding out from what that phase proposes.
        """
        return (
            f"The instruction you gave with this run ({agents.said(instruction, 160)}) was "
            f"handed to the analysis. The {_NOUN[back]} phase, which comes next, has not had "
            "it: give it again when you continue, or leave it as a request, if it was meant "
            "for that phase."
        )

    def _open_record(self, name: str, round: int) -> dict[str, Any] | None:
        """The record of this phase in this round that is still to be completed.

        The latest record when it is this phase's, or the one that what the person
        approved meanwhile came in after: an analysis that was unfinished when a
        further experiment ran, any other phase that was unfinished when a run and
        its analysis did.  The phase is resumed in its own record, under the
        instruction it was started with.
        """
        came_between = () if name == "run" else ("run",) if name == "analyze" else _RUNS
        for record in reversed(self._state["phases"]):
            if record.get("phase") in came_between:
                continue
            mine = record.get("phase") == name and record.get("round") == round
            return record if mine and record.get("status") != "completed" else None
        return None

    def _last_record(self, name: str, round: int) -> dict[str, Any] | None:
        """The latest record, when it is of this phase in this round."""
        phases = self._state["phases"]
        last = phases[-1] if phases else None
        if isinstance(last, dict) and last.get("phase") == name and last.get("round") == round:
            return last
        return None

    def _round_for(self, name: str) -> int:
        """The round the phase ``name`` runs in.

        A new round begins with the first phase after the analysis that ended the
        last one: the analysis of the round's own run, which follows its design
        phase.  A run, an analysis, and whatever follows a run that was not the
        round's own stay in the round.  What was approved after the round's
        analysis (and ran, or could not run) does not keep the round open.
        """
        current = max(1, self._state["round"])
        if name in _RUNS:
            return current
        base, since = self._since_last_phase()
        ended = (
            base is not None
            and base.get("phase") == "design"
            and base.get("status") == "completed"
            and base.get("round") == current
            and self._analysed(since)
        )
        return current + 1 if ended else current

    @staticmethod
    def _new_record(name: str, round: int, instruction: str | None) -> dict[str, Any]:
        return {
            "round": round,
            "phase": name,
            "status": "running",
            "started": _now(),
            "finished": None,
            "seconds": 0.0,
            "attempts": 0,
            "error": None,
            "produced": [],
            "instruction": instruction,
            "requests": [],
        }

    def _complete_record(self, record: dict[str, Any]) -> None:
        """Give a record found in the state file the fields a resumed phase counts on."""
        for key, value in self._new_record(record["phase"], record["round"], None).items():
            record.setdefault(key, value)
        for key in ("produced", "requests"):
            if not isinstance(record[key], list):
                record[key] = []
        if not isinstance(record["attempts"], int) or isinstance(record["attempts"], bool):
            record["attempts"] = 0
        if not isinstance(record["instruction"], str) or not record["instruction"].strip():
            record["instruction"] = None

    def _session(self, name: str, round: int) -> str:
        """The session stem of this run of a phase: the same while it is resumed, new otherwise."""
        done = sum(
            1
            for record in self._state["phases"]
            if record.get("phase") == name
            and record.get("round") == round
            and record.get("status") == "completed"
        )
        return f"r{round}-{name}" + (f"-{done + 1}" if done else "")

    # -- requests ----------------------------------------------------------------

    def _requests_for(self, name: str) -> list[Request]:
        """The person's open requests that the agent of this phase is shown: the ones
        addressed to its role and the ones for every agent."""
        if name not in _TAKES_REQUESTS:
            return []
        mine = (FOR_ALL, _ROLE[name])
        requests = self.project.requests()
        return [request for request in requests if agents.addressee(request) in mine]

    def _settle_requests(self, record: Mapping[str, Any]) -> None:
        """Note on every request the completed phase was given that it had it, and close
        the ones nobody else is to be shown.

        A request for one role is closed by that role's phase.  A request for every
        agent stays open for the phases that follow and is closed by the analysis
        that ends the round.  Done again after a worker died in the middle, it
        changes nothing that was already done.
        """
        name, round = record["phase"], record["round"]
        given = {str(request_id) for request_id in record.get("requests") or []}
        ends_round = name == "analyze" and self._return_phase() is None
        for request in self.project.requests():
            if request.id not in given:
                continue
            whom = agents.addressee(request)
            close = whom == _ROLE[name] or (whom == FOR_ALL and ends_round)
            # (One the person withdrew while the phase ran is gone, which is as good.)
            with contextlib.suppress(NotFound):
                self.project.note_request(request.id, round=round, phase=name, close=close)

    # -- one phase ---------------------------------------------------------------

    def _context(
        self,
        name: str,
        round: int,
        session: str,
        requests: Sequence[Request],
        instruction: str | None,
    ) -> AgentContext:
        return AgentContext(
            project=self.project,
            harness=self.harness,
            round=round,
            phase=name,
            session=session,
            requests=requests,
            instruction=instruction,
            guard=self._guard,
            stopping=self._stopping,
            # (Named here: the state's phase is the last one that ran, which is another
            # one while a run is started by itself.)
            over_budget=functools.partial(self._over_budget, phase=name),
            scholar=self._scholar,
            compute=self._compute,
            handover=self._handover(name, round),
            runtime=self._runtime() if self._sdk() else None,
            left=self._budget_left,
            short_of=functools.partial(self._short_of, phase=name),
        )

    def _handover(self, name: str, round: int) -> tuple[str, ...]:
        """The roles whose undecided pages of this round the phase's agent may read.

        The scout that follows intake directly: intake wrote down what the
        person's files say, and the first look at the literature should know that,
        although the person has not reviewed those pages yet.  Every other agent
        builds only on what the person accepted.
        """
        if name != "research":
            return ()
        earlier = [
            record
            for record in self._state["phases"]
            if not (record.get("phase") == "research" and record.get("round") == round)
        ]
        last = earlier[-1] if earlier else None
        after_intake = (
            last is not None
            and last.get("phase") == "intake"
            and last.get("status") == "completed"
            and last.get("round") == round
        )
        return ("intake",) if after_intake else ()

    def _run_phase(self, name: str, instruction: str | None) -> _Outcome:
        """Run one phase to its end, retrying a failure once; record how it went."""
        self.project.reload()
        round = self._round_for(name)
        session = self._session(name, round)
        record = self._open_record(name, round)
        kept = False
        if record is None:
            record = self._new_record(name, round, instruction)
        else:
            self._complete_record(record)
            # A phase that is resumed runs under the word it was started with, unless
            # the person gives a new one: its record says so, and it must be true.
            kept = instruction is None and record["instruction"] is not None
            instruction = instruction or record["instruction"]
        requests = self._requests_for(name)
        self._enter(record, instruction, requests)
        if kept:
            self._emit(
                "agent.activity",
                f"The {_NOUN[name]} phase goes on under the instruction it was started "
                f"with: {agents.said(instruction, 200)}.",
                phase=name,
                round=round,
            )
        clock = time.monotonic()
        ctx = self._context(name, round, session, requests, instruction)
        self._ran = 0
        report: AgentReport | None = None
        for attempt in (1, 2):
            with self._lock:
                # On file before the attempt begins: a worker that is killed during it
                # can write nothing afterwards, and the attempt was made all the same.
                record["attempts"] += 1
                self._save()
            try:
                # Right after a run a verdict is still to come from the analysis, so only
                # the phases that follow a checkpoint give a hypothesis back.
                self._tidy_hypotheses(release=name not in ("run", "analyze"))
                report = self._execute(ctx)
                break
            except _BudgetSpent as exc:
                return self._budget_stop(record, clock, ctx, exc)
            except agents.SessionStopped as exc:
                if isinstance(exc.reason, _BudgetSpent):
                    # The turn in flight took the project past its model budget.
                    return self._budget_stop(record, clock, ctx, exc.reason)
                # A stop request, heard in the middle of a session's turn.
                return self._interrupted(record, clock, ctx)
            except KeyboardInterrupt:
                self._interrupted(record, clock, ctx)
                raise
            except Exception as exc:
                if self._stopping() or any(isinstance(e, RunCancelled) for e in chain(exc)):
                    return self._interrupted(record, clock, ctx)
                # (A setting that cannot be read is no better read a second time.)
                if attempt == 2 or stops_the_run(exc) or isinstance(exc, CutoffError):
                    return self._failed(record, clock, ctx, exc)
                # (An agent's own account of why it ended is a sentence for the person,
                # said in the feed below: a traceback of this process adds nothing to it.)
                log.warning(
                    "The %s phase failed; trying once more.",
                    name,
                    exc_info=not getattr(exc, "plain", False),
                )
                # An error can carry text a model wrote: it stands there as a quotation.
                self._emit(
                    "agent.activity",
                    f"The {_NOUN[name]} phase failed ({agents.said(self._clean(exc), 200)}); "
                    "trying it once more.",
                    phase=name,
                    round=round,
                )
        assert report is not None
        outcome = self._completed(record, clock, ctx, report)
        outcome.ran = self._ran if name == "run" else 0
        return outcome

    def _enter(
        self, record: dict[str, Any], instruction: str | None, requests: Sequence[Request]
    ) -> None:
        """Say in the state file that this run of a phase is in flight.

        Its record is listed from the start (as ``running``), with the instruction
        it runs under and the requests it is given, and the phase itself is what
        comes next for as long as it has not completed: whoever finds the state
        after this worker died resumes the phase by running ``advance()``.  The
        event says the same, so that a state rebuilt from the events knows it too.
        """
        name, round = record["phase"], record["round"]
        with self._lock:
            record.update(status="running", finished=None, error=None)
            record["instruction"] = instruction
            # Every request an attempt was given: a resumed phase adds to the list.
            given = dict.fromkeys([*record["requests"], *(request.id for request in requests)])
            record["requests"] = list(given)
            # The record of the phase in flight is the last one, also when the phase is
            # resumed after a run and its analysis came in between: whoever settles the
            # state after this worker died looks for it there.
            phases = self._state["phases"]
            phases[:] = [known for known in phases if known is not record] + [record]
            self._state.update(
                status="running",
                round=round,
                phase=name,
                next_phase=name,
                checkpoint=None,
                error=None,
            )
            self._save()
        said = {"instruction": instruction} if instruction else {}
        self._emit(
            "phase.started",
            f"Round {round}: the {_NOUN[name]} phase started.",
            phase=name,
            round=round,
            data={**said, "requests": record["requests"]} if record["requests"] else said,
        )

    def _execute(self, ctx: AgentContext) -> AgentReport:
        """Run the phase of ``ctx`` in the project's runtime.

        This is where the runtime is chosen.  With ``settings.agents.runtime`` set
        to ``sdk``, the context carries the runtime, and every role function below
        opens its session as a Claude Agent SDK agent (``agents._open_session``);
        the functions themselves, their messages and their checks are the same.
        Whatever the phase's agents still hold when it ends, also by a stop or a
        failure, is closed here.
        """
        ctx.runtime = self._runtime() if self._sdk() else None
        try:
            return self._execute_phase(ctx)
        finally:
            ctx.close_sessions()

    def _execute_phase(self, ctx: AgentContext) -> AgentReport:
        name = ctx.phase
        if name == "intake":
            return agents.run_intake(ctx)
        if name == "research":
            return agents.run_scout(ctx)
        if name == "hypothesize":
            return agents.run_theorist(ctx)
        if name == "design":
            return agents.run_designer(ctx, spent=self._state["spent"])
        if name == "run":
            return self._run_approved(ctx)
        results = self._new_results()
        report = agents.run_analyst(
            ctx, results=results, journal=self._journal_name(ctx), spent=self._state["spent"]
        )
        for result in results:
            # What marks a record as analysed has to survive the engine: it is on the record.
            # (A record the person deleted meanwhile needs no mark.)
            with contextlib.suppress(NotFound):
                self.project.update_page(
                    result.path, author="system", fields={"analysed": ctx.round}
                )
        return report

    def _settle(
        self, record: dict[str, Any], clock: float, ctx: AgentContext, status: str, error: Any
    ) -> None:
        """Write how this run of a phase ended into its record and the state file.

        Whatever it created is listed, also when it did not finish: a resumed phase
        adds to that list.
        """
        root = self.project.root
        produced = dict.fromkeys([*record["produced"], *ctx.produced])
        with self._lock:
            record.update(
                status=status,
                finished=_now(),
                seconds=round(_number(record.get("seconds")) + time.monotonic() - clock, 1),
                error=error,
                produced=[path for path in produced if (root / path).is_file()],
            )
            self._write_usage()
            self._save()

    def _completed(
        self, record: dict[str, Any], clock: float, ctx: AgentContext, report: AgentReport
    ) -> _Outcome:
        """Record the phase as completed, then do what goes with it.

        One save says that the phase completed and what follows it.  Closing the
        requests it was given and announcing it come right after; a worker that
        dies in between leaves both to the next one, which finds the completed
        record without its checkpoint and does them (:meth:`_first_phase`).
        """
        ctx.note(report.produced)
        with self._lock:
            self._state["next_phase"] = self._after_completed(record["phase"])
        self._settle(record, clock, ctx, "completed", None)
        self._announce(record, report.summary)
        return _Outcome("completed", report)

    def _after_completed(self, name: str) -> str:
        """What the state names as next once ``name`` has completed, until a checkpoint
        says it for the person: after an analysis that was not the round's own, the
        phase the round was at."""
        if name == "analyze":
            return self._return_phase() or _NEXT[name]
        return _NEXT[name]

    def _announce(self, record: Mapping[str, Any], summary: str | None = None) -> None:
        """Close the requests a completed phase was given and say that it finished."""
        name = record["phase"]
        self._settle_requests(record)
        self._emit(
            "phase.finished",
            summary or f"The {_NOUN[name]} phase finished.",
            phase=name,
            round=record["round"],
            refs=record["produced"][:50],
            data={"produced": record["produced"], "seconds": record["seconds"]},
        )

    def _interrupted(self, record: dict[str, Any], clock: float, ctx: AgentContext) -> _Outcome:
        name, round = record["phase"], record["round"]
        self._settle(record, clock, ctx, "interrupted", None)
        with self._lock:
            self._state.update(status="stopped", next_phase=name, error=None)
            self._save()
        self._emit(
            "agent.activity",
            f"The {_NOUN[name]} phase was stopped; running it again resumes it.",
            phase=name,
            round=round,
        )
        return _Outcome("interrupted")

    def _failed(
        self, record: dict[str, Any], clock: float, ctx: AgentContext, exc: BaseException
    ) -> _Outcome:
        name, round = record["phase"], record["round"]
        # What is written of an error is read by the app, the command line and later
        # sessions: no credential a library quoted in it gets that far.
        told = self._clean(exc)
        # (An error that is a whole sentence to the person stands without its class.)
        named = "" if getattr(exc, "whole_sentence", False) is True else f"{type(exc).__name__}: "
        error = f"{named}{_one_line(told, 1000)}"
        log.error("The %s phase failed.", name, exc_info=exc)
        self._settle(record, clock, ctx, "failed", error)
        with self._lock:
            self._state.update(status="failed", next_phase=name, error=error)
            self._save()
        again = " twice" if record["attempts"] > 1 else ""
        why = agents.said(_one_line(told, 300).rstrip("."), 300)
        self._emit(
            "phase.failed",
            f"The {_NOUN[name]} phase failed{again}: {why}. Run it again once the cause is "
            "put right; what it had written is kept.",
            phase=name,
            round=round,
            data={"error": error, "attempts": record["attempts"]},
        )
        return _Outcome("failed")

    def _budget_stop(
        self, record: dict[str, Any], clock: float, ctx: AgentContext, exc: _BudgetSpent
    ) -> _Outcome:
        name = record["phase"]
        exc = self._as_spent(exc)
        # Which limit it was decides whether an unattended stretch may go on (it may not).
        self._pass_end = f"budget:{exc.limit}"
        self._settle(record, clock, ctx, "interrupted", str(exc))
        self._warn_budget(exc, phase=name)
        extra = (
            " Results that exist can still be analysed: continue with the analyze phase."
            if name == "run" and self._new_results()
            else ""
        )
        self._checkpoint(
            name,
            record["round"],
            name,
            [],
            [str(exc) + extra],
            budget_said=exc.limit == "model_usd",
        )
        return _Outcome("budget")

    # -- between phases ----------------------------------------------------------

    def _after(
        self,
        name: str,
        outcome: _Outcome,
        told: Sequence[str],
        notes: Sequence[str],
        *,
        back: str | None = None,
        named: bool = False,
    ) -> str | None:
        """The phase that follows at once, or ``None`` after writing the checkpoint.

        ``named`` says that the person asked for the phase of this call by name: a
        run phase they named pauses also when nothing ran, instead of going on
        with a phase they did not ask for.

        ``back`` is the phase that was next when this call began with a run or an
        analysis that was not the round's own.  :meth:`_return_phase` knows that
        phase from the records; ``back`` stands in where the state file named
        another one than the records imply (a file edited by hand).
        """
        round = self._state["round"]
        at_once, following = self._follows(name, ran=bool(outcome.ran), back=back, named=named)
        if at_once is not None:
            return at_once
        if name == "analyze" and len(told) < 2:
            # The run phase reports the experiments that are planned again where its
            # checkpoint is this one.  Here it did not report in this call (the analysis
            # was resumed after a stop, a failure or a spent budget, or was asked for by
            # itself), and what it said then went with that call: said from the pages.
            notes = [*notes, *self._planned_again()]
        recommending = self._recommending(round) if name == "analyze" else None
        notes = [*notes, *self._journal_note(recommending)]
        self._checkpoint(name, round, str(following), told, notes, recommends=recommending)
        return None

    def _follows(
        self,
        name: str,
        *,
        ran: bool,
        back: str | None = None,
        named: bool = False,
        results: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """What follows the phase ``name`` once it has completed: ``(the phase that
        runs at once, None)``, or ``(None, the phase its checkpoint names as next)``
        when the call pauses there.

        ``ran`` says whether the run phase executed an experiment, and ``results``
        whether a result is there to analyse after it (looked up when it is not
        given; a forecast says what the runs it foretells will leave).  ``back`` and
        ``named`` are those of :meth:`_after`.  This is the rule alone, without the
        checkpoint: :meth:`_after` acts on it and :func:`next_step` foretells by it.
        """
        if name == "intake":
            return "research", None
        aside = self._return_phase() or back
        if name == "run":
            to_analyse = bool(self._new_results()) if results is None else results
            if to_analyse:
                return "analyze", None
            if aside is not None and not named and not ran:
                # Nothing ran, so there is nothing to pause for: what could not be run
                # is said at the checkpoint of the phase that was next.
                return aside, None
            # What ran left nothing to analyse (the person cancelled it): the call
            # pauses here instead of going on with a phase they did not ask for.
            return None, aside or "run"
        if name == "analyze":
            return None, aside or _NEXT[name]
        return None, _NEXT[name]

    def _forecast(self) -> dict[str, Any]:
        """What :meth:`advance` would do with no phase named, by its own rules; see
        :func:`next_step`.  Changes this engine's state as the phases would, and
        nothing on disk: only for an engine that reads (:meth:`_reading`)."""
        self._settle_dead_worker()
        waiting, held = self._will_start()
        said: dict[str, Any] = {
            "will_run": [],
            "then": None,
            "experiments": waiting,
            "not_running": held,
            "results": len(self._new_results()),
            "checkpoint_of": None,
            "needs_phase": False,
            "blocked": None,
        }
        if self._position_unknown():
            return {**said, "needs_phase": True}
        owed = self._unsettled_phase()
        name: str | None
        if owed is None:
            name = self._next_phase()
        else:
            # The pause a dead worker owed comes first, as in :meth:`_first_phase`: the
            # checkpoint it had announced already, or the one that follows the phase.
            said["checkpoint_of"] = owed
            written = self._announced(self._events_of(self._state["phases"][-1]))
            if written is not None:
                return {**said, "then": self._next_of(written)}
            name, said["then"] = self._follows(owed, ran=False)
        back = None
        if name:
            name, back = self._comes_first(name)
        while name and len(said["will_run"]) <= len(PHASES):
            need = self._model_need(name)
            if need is not None:
                # The call stops here, as :meth:`advance` does: before this phase.
                said["then"] = name
                said["blocked"] = {
                    "phase": name,
                    "needs": need["needs"],
                    "settings": list(need["settings"]),
                    "models": list(need["models"]),
                    "message": agents.need_sentence(need, phase_in_words(name).capitalize()),
                }
                break
            said["will_run"].append(name)
            self._assume_completed(name)
            # A run that starts is taken to leave a result, which is then analysed.
            runs = name == "run" and bool(waiting)
            name, said["then"] = self._follows(
                name, ran=runs, back=back, results=runs or bool(said["results"])
            )
        return said

    def _will_start(self) -> tuple[list[str], list[dict[str, str]]]:
        """Of the experiments a run phase is owed, the ones a run would start (their
        ids) and the ones it would not, each with the reason: ``[{"id", "reason"}]``.

        The run phase takes up every experiment that reads as approved, and
        ``research.experiments`` starts only what still is what the person
        approved; the others go back to planned there.  Whoever foretells a press
        must not count them as runs.
        """
        waiting: list[str] = []
        held: list[dict[str, str]] = []
        for page in self._awaiting_run():
            name = str(page.id or page.path)
            try:
                reason = experiments.will_not_run(self.project, page.path)
            except (ProjectError, OSError):
                reason = None  # it cannot be told here; the run says it
            if reason is None:
                waiting.append(name)
            else:
                held.append({"id": name, "reason": reason})
        return waiting, held

    def _assume_completed(self, name: str) -> None:
        """For a forecast: the records as :meth:`_run_phase` leaves them once the phase
        ``name`` has run to its end (its record last, and completed)."""
        round = self._round_for(name)
        record = self._open_record(name, round) or self._new_record(name, round, None)
        phases = self._state["phases"]
        phases[:] = [known for known in phases if known is not record] + [record]
        record["status"] = "completed"
        self._state.update(round=round, phase=name)

    def _recommending(self, round: int) -> ResearchPage | None:
        """The journal page of the round whose recommendation of more research still
        stands, the latest one when several do; ``None`` when none does.

        ``next: research`` in a journal page's front matter is that recommendation.
        A page the person rejected recommends nothing, and neither does one that
        was written before the research phase last completed in the round: the
        person followed it, and a later analysis of the round has no reason to say
        it again unless its own journal page asks anew.
        """
        asking = [
            page
            for page in self.project.pages(kind="journal", review=["accepted", "proposed"])
            if page.round == round
            and str(page.meta.get("next") or "").strip().lower() == "research"
            and not self._researched_since(page)
        ]
        return max(asking, key=lambda page: (page.created, page.path), default=None)

    def _researched_since(self, page: ResearchPage) -> bool:
        """Whether a research phase of the page's round completed after ``page`` was
        written.

        The phase records are in the order the phases ran, and each lists what it
        produced: a page of an analysis is told by the place of that analysis among
        them.  A page that is in no record (the person wrote it) is told by its
        stamp, and counts as written after a phase that ended in the same second.
        """
        phases = self._state["phases"]
        done = [
            index
            for index, record in enumerate(phases)
            if record.get("phase") == "research"
            and record.get("round") == page.round
            and record.get("status") == "completed"
        ]
        if not done:
            return False
        latest = done[-1]

        def wrote(record: Mapping[str, Any]) -> bool:
            produced = record.get("produced")
            return isinstance(produced, list) and page.path in produced

        if any(wrote(record) for record in phases[latest + 1 :]):
            return False
        if any(wrote(record) for record in phases[:latest]):
            return True
        written, finished = parse_stamp(page.created), parse_stamp(phases[latest].get("finished"))
        return written is not None and finished is not None and written < finished

    def _journal_note(self, recommending: ResearchPage | None) -> list[str]:
        """What the checkpoint of an analysis repeats when a journal page of the round
        recommends more research (``recommending``, from :meth:`_recommending`).

        The recommendation is the analyst's and changes nothing about what runs
        next: which phase runs is the person's to choose, and going back to the
        literature costs a session.
        """
        if recommending is None:
            return []
        return [
            "The journal page of this round recommends another look at the literature "
            "before new hypotheses (next: research). That is yours to decide and changes "
            "nothing by itself: to follow it, run the research phase next with "
            f"`paper-orchestra research advance {self.project.root} --phase research`, or "
            "Continue with research in the app."
        ]

    def _checkpoint(
        self,
        name: str,
        round: int,
        following: str,
        told: Sequence[str],
        notes: Sequence[str],
        *,
        budget_said: bool = False,
        recommends: ResearchPage | None = None,
        missing: Sequence[str] = (),
    ) -> None:
        """Pause for the person: say what was done, what needs their eye, what comes next.

        ``told`` is what the phases since the last checkpoint did (each has said so
        in the activity feed already); ``notes`` is what the person has to know
        beyond that.  The state keeps the whole account, the event the part that is
        new to the feed.  A model budget that the work has passed is said here too,
        unless ``notes`` already say it (``budget_said``).

        ``recommends`` is the journal page that recommends more research before the
        phase that is next (:meth:`_recommending`).  ``notes`` say so in a sentence;
        the checkpoint and its event also carry it as ``recommends`` and
        ``recommends_page``, for whoever shows the choice beside the recommendation.

        ``missing`` is what the phase that is next lacks to start, where that is why
        the call pauses here (a model that cannot be used, in one sentence).  It is
        said at the pause, in the event and in the checkpoint's summary.

        The checkpoint keeps the account by itself as ``did``: what was done and
        what the person has to know, without the count of what waited, without what
        was next and without ``missing``.  Those were true at this moment only;
        whoever shows the checkpoint later says them as they are then, beside
        ``did`` (a model may have been set up since: :func:`next_step` tells).
        """
        recommended = (
            {"recommends": "research", "recommends_page": recommends.path} if recommends else {}
        )
        if not budget_said:
            notes = [*notes, *self._overrun_note(name)]
        pending = self.project.counts()["pending"]
        waiting = [agents.count_of(count, kind) for kind, count in pending["by_kind"].items()]
        if pending["proposals"]:
            waiting.append(_plural(pending["proposals"], "suggested edit"))
        verb = "awaits" if pending["total"] == 1 else "await"
        review = (
            f"{_and(waiting)} {verb} your review." if waiting else "Nothing awaits your review."
        )
        # What the agents asked the person to choose.  It holds nothing up: the agents go
        # on with the default each question states, so it is said and not waited for.
        undecided = len(self.project.decisions(state="open"))
        resumes = " (it resumes where it stopped)" if following == name and name != "run" else ""
        account = [*notes, *self._waiting_for_research(following), *self._unpriced_note()]
        closing = [
            *account,
            *missing,
            review,
            decisions.waiting(undecided),
            f"Next: {self._doing(following)}{resumes}.",
        ]
        refs = [page.path for page in self.project.pages(review="proposed")][:50]
        event = self._emit(
            "checkpoint",
            " ".join(part.strip() for part in closing if part.strip()),
            phase=name,
            round=round,
            refs=refs,
            data={
                "pending": pending["by_kind"],
                "edits": pending["proposals"],
                "next": following,
                **({"decisions": undecided} if undecided else {}),
                **recommended,
            },
        )
        with self._lock:
            self._state.update(
                status="checkpoint",
                round=round,
                phase=name,
                next_phase=following,
                error=None,
                checkpoint={
                    "round": round,
                    "phase": name,
                    "at": event["ts"],
                    "summary": " ".join(
                        part.strip() for part in [*told, *closing] if part and part.strip()
                    ),
                    "did": " ".join(
                        part.strip() for part in [*told, *account] if part and part.strip()
                    ),
                    "pending": dict(pending["by_kind"]),
                    **_recommendation(recommended),
                },
            )
            self._save()

    def _doing(self, following: str) -> str:
        """What the phase that is next does, in words that are true of this project:
        where autopilot decided, nothing is called accepted or approved by the person."""
        return (_DOING_UNATTENDED if self._autopilot_on() else _DOING)[following]

    def _overrun_note(self, name: str) -> list[str]:
        """What a checkpoint says when the model spend has reached its limit.

        A phase that finishes is not cut short for it: the limit is looked at before a
        session, before each turn and after each model request, so the work that
        passed it has ended by now, and the last request may have gone well beyond
        it.  The person is told here, not only when the next phase finds it out.
        """
        limit = float(self.project.settings.limits.model_usd)
        spent = self._write_usage()
        if spent < limit:
            return []
        exc = _BudgetSpent(
            "model_usd",
            f"The model budget is used up: {_money(spent)} of {_limit_money(limit)} is spent. No "
            "agent starts again until settings.limits.model_usd is raised in the project's "
            "settings; what awaits your review can be decided all the same.",
            spent=spent,
            allowed=limit,
        )
        self._warn_budget(exc, phase=name)
        return [str(exc)]

    def _waiting_for_research(self, following: str) -> list[str]:
        """Said at every checkpoint while a request addressed to the scout is open and the
        research phase is not what comes next.

        A request for one role waits for that role's phase.  The theorist, the
        designer and the analyst come round by themselves; the scout does not,
        because more research is the person's choice.  Without this sentence
        such a request would wait for good, and nothing would say so.
        """
        if following == "research":
            return []
        scout = _ROLE["research"]
        waiting = [r for r in self.project.requests() if agents.addressee(r) == scout]
        if not waiting:
            return []
        count = len(waiting)
        return [
            f"{_plural(count, 'request')} for the {scout} {'waits' if count == 1 else 'wait'} "
            "for the research phase, which does not come round by itself: run it with "
            f"`paper-orchestra research advance {self.project.root} --phase research`, or "
            "Continue with research in the app."
        ]

    def _unpriced_note(self) -> list[str]:
        """Said at every checkpoint while it holds: a budget that cannot count is no limit."""
        if not self._unpriced:
            return []
        return [
            f"No price is known for {_and(self._unpriced)}, so what "
            f"{'it' if len(self._unpriced) == 1 else 'they'} cost is not part of the model "
            "spending shown, and settings.limits.model_usd does not limit it."
        ]

    # -- hypotheses under test ---------------------------------------------------

    def _tidy_hypotheses(self, *, release: bool) -> None:
        """Keep ``testing`` true to the experiments.

        A hypothesis is under test once an experiment that tests it is approved,
        queued or running.  With ``release`` it is open again when none is and no
        verdict is waiting for the person: the test is over and nothing was decided.
        Only a hypothesis that some experiment tests is given back: where there is
        none, it was the person who said it is under test, and that stands.
        """
        project = self.project
        active: set[str] = set()
        tested: set[str] = set()
        for experiment in project.pages(kind="experiment", review="accepted"):
            for target in experiment.relations.get("tests", []):
                path = project.resolve(target)
                if path:
                    tested.add(path)
                    if experiment.state in ("approved", "queued", "running"):
                        active.add(path)
        judged = {proposal.target for proposal in project.proposals() if "state" in proposal.fields}
        for hypothesis in project.pages(kind="hypothesis", review="accepted"):
            if hypothesis.state == "open" and hypothesis.path in active:
                project.set_state(
                    hypothesis.path,
                    "testing",
                    by="system",
                    reason="an experiment that tests it is approved",
                )
            elif (
                release
                and hypothesis.state == "testing"
                and hypothesis.path in tested
                and hypothesis.path not in active
                and hypothesis.path not in judged
            ):
                project.set_state(
                    hypothesis.path,
                    "open",
                    by="system",
                    reason="no approved experiment is testing it and no verdict is proposed",
                )

    # -- running experiments -----------------------------------------------------

    def _count_runs(self) -> None:
        """Bring the time and the cost of the project's runs up to date, from their records.

        They are read from what the runs left on disk, not added up as runs go by: a
        run that ended while no engine was looking (picked up after its worker died,
        cancelled from the web app, finished a moment before the worker was killed)
        cost what it cost all the same, and no run is ever counted twice.
        """
        seconds = usd = 0.0
        folder = self.project.root / EXPERIMENTS_DIR
        try:
            names = sorted(entry.name for entry in folder.iterdir()) if folder.is_dir() else []
            for name in names:
                if not _EXPERIMENT_DIR.fullmatch(name):
                    continue
                for run in experiments.runs(self.project, name):
                    seconds += _number(run.get("seconds"))
                    usd += _number(run.get("cost_usd"))
        except (OSError, ProjectError):
            # What was counted before stands; the next look may fare better.
            log.warning("The run records under %s could not be read.", folder, exc_info=True)
            return
        with self._lock:
            self._state["spent"].update(
                experiment_seconds=round(seconds, 3), experiment_usd=round(usd, 4)
            )

    def _recover(self) -> None:
        """Settle runs a dead worker left behind, before anything is selected to run.

        A remote job that outlived its worker is followed to its end here, and what it
        cost is the project's spending like that of any other run.
        """
        try:
            experiments.recover(self.project, compute=self._compute, should_stop=self._stopping)
        finally:
            self._count_runs()
            self._save()

    def _run_limits(self, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
        """What ``research.experiments.run`` may spend on the next run.

        ``experiment_usd`` is the project's total, and the experiments module checks
        one run against the amount it is given: it gets what is left of the total, so
        that a run whose worst case would pass the total is refused with the numbers.
        """
        allowed = float(self._limits_with(overrides).experiment_usd)
        left = max(0.0, allowed - self._state["spent"]["experiment_usd"])
        return {**(overrides or {}), "experiment_usd": round(left, 4)}

    def _run_one(
        self,
        page: ResearchPage,
        ctx: AgentContext,
        *,
        compute: Any = None,
        limits: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one approved experiment and return its final run record.

        The time and cost of every run it took (a repaired experiment runs more
        than once) are then part of what the project has spent, and the result
        records are noted in ``ctx.produced``, whatever way the run ended.
        """
        project, experiment = self.project, str(page.id)

        def settled() -> dict[Any, dict[str, Any]]:
            runs = experiments.runs(project, experiment)
            return {run.get("number"): run for run in runs if run.get("status") != "running"}

        before = set(settled())
        repair, unrepaired = self._repairer(ctx)
        carried_out: list[Any] = []
        chosen = compute or self._compute
        if self._sdk():
            # The seam of the sdk runtime: the job of the run is an agent's.  Its model
            # spend is the model budget's, which is looked at before it starts.
            ctx.runtime = self._runtime()
            ctx.over_budget = functools.partial(self._over_budget, experiment=experiment)
            ctx.short_of = functools.partial(self._short_of, experiment=experiment)
            self._guard()
            chosen = self._agent_backend(page, ctx, chosen, carried_out)
        try:
            record = experiments.run(
                project,
                experiment,
                compute=chosen,
                should_stop=self._stopping,
                repair=repair,
                unrepaired=unrepaired,
                limits=self._run_limits(limits),
            )
            for backend in carried_out:
                stopped = backend.stopped
                if isinstance(stopped, agents.SessionStopped):
                    # A stop request or the model budget ended the agent; the run is
                    # queued again, and the phase ends the way it does for any session.
                    raise stopped
            return record
        finally:
            ctx.close_sessions()
            self._count_runs()
            self._save()
            # Also a run found unfinished and settled on the way: it ended now.
            new = [run for number, run in settled().items() if number not in before]
            pages = [project.resolve(f"{experiment}-run{run.get('number')}") for run in new]
            ctx.note(path for path in pages if path)

    def _agent_backend(
        self, page: ResearchPage, ctx: AgentContext, compute: Any, made: list[Any]
    ) -> Callable[[Any], Any]:
        """The chooser of the backend for a run that an agent carries out.

        The backend is still the one the person approved; where that is the local
        one, an agent does its job (``research.sdk.experiment.AgentBackend``).  A
        run approved for a remote backend runs there as it always did: an agent
        works on this machine.
        """
        from paper_orchestra.research.sdk.experiment import AgentBackend

        def choose(plan: Any) -> Any:
            inner = compute if hasattr(compute, "run_job") else compute(plan)
            if getattr(inner, "name", "") != "local":
                return inner
            backend = AgentBackend(ctx, inner, page)
            made.append(backend)
            return backend

        return choose

    def _repairer(
        self, ctx: AgentContext
    ) -> tuple[Callable[..., Any] | None, str | Callable[[Mapping[str, Any]], str] | None]:
        """The callback that repairs a failed run, or ``None`` and why there is none.

        Under the classic runtime a run needs no model and never waits for one.  The
        repair of a run that failed for a programming error does, the runner's:
        where that model cannot be used, the run happens all the same, a failure
        stands, and its record says why nothing was tried
        (``experiments.run(..., unrepaired=...)``).  Under ``sdk`` nobody repairs
        afterwards, and why is said of the run that failed: an agent carried it
        out and repaired while it worked, or it ran where no agent works.
        """
        if self._sdk():

            def unrepaired(record: Mapping[str, Any]) -> str:
                if record.get("carried_out_by") == "agent":
                    return (
                        "under the sdk runtime the agent that carries out an experiment "
                        "repairs its code itself, so no separate repair is made."
                    )
                return (
                    f"this run was on the {record.get('backend') or 'remote'} backend, where "
                    "no agent works, and under the sdk runtime no separate repair is made "
                    "afterwards. Correct the code and approve the experiment again."
                )

            return None, unrepaired
        setting = agents.MODEL_ROLES["runner"]
        try:
            need = self._models_need((setting,))
        except Exception as exc:
            # Whatever is wrong with the model must not keep an approved run from running.
            log.warning("The model for repairs could not be asked about.", exc_info=True)
            why = self._clean(_one_line(f"{type(exc).__name__}: {exc}", 200))
            return None, (
                f"a repair needs a model, and the one set for {setting} could not be "
                f"looked at ({why})."
            )
        if need is None:
            return agents.make_repair(ctx), None
        return None, agents.need_sentence(need, "a repair")

    def _planned_again(self) -> list[str]:
        """One sentence for each accepted experiment that went back to ``planned`` with a
        reason on its page.

        That is how an experiment stands whose plan or code changed after the person
        approved it: what runs is what was approved, so it waits for a new approval,
        and the person should hear of it where they are told what ran.
        """
        planned = self.project.pages(kind="experiment", review="accepted", state="planned")
        again = [
            f"{page.id} is planned again: {_sentence(reason, 300)}"
            for page in planned
            if (reason := str(page.meta.get("state_reason") or "").strip())
        ]
        # A handful by name and of the rest the number, so that none goes unmentioned.
        more = len(again) - 6
        if more > 0:
            again[6:] = [f"{_plural(more, 'more experiment')} went back to planned as well."]
        return again

    def _nothing_to_run(self) -> str:
        project = self.project
        planned = project.pages(kind="experiment", review="accepted", state="planned")
        if planned and self._autopilot_on():
            # Under autopilot a plan that is still a plan is one the rule left alone.
            back = "".join(f" {sentence}" for sentence in self._planned_again())
            held = [
                (str(page.id), self._held.get(str(page.id)) or "it waits for your approval")
                for page in planned
            ]
            return (
                f"No experiment is approved to run.{back} Autopilot did not approve "
                f"{autopilot.grouped(held)}."
            )
        if planned:
            # A title is an agent's words more often than not: given as a quotation.
            names = _and([f"{page.id} {agents.said(page.title, 60)}" for page in planned[:6]])
            back = "".join(f" {sentence}" for sentence in self._planned_again())
            return (
                f"No experiment is approved to run.{back} Approve the ones you want run: "
                f"{names}{' and others' if len(planned) > 6 else ''}."
            )
        proposed = project.pages(kind="experiment", review="proposed")
        if proposed:
            return (
                f"No experiment is approved to run. {_plural(len(proposed), 'proposed experiment')} "
                "must be accepted and approved first."
            )
        return (
            "There is no experiment to run. Run the design phase to have some proposed, or "
            "add one yourself."
        )

    def _not_run(self, page: ResearchPage, exc: BaseException) -> str:
        """What the checkpoint says about an approved experiment that was refused a run.

        What runs is what the person approved: ``research.experiments`` refuses an
        experiment whose plan or code is no longer that, and puts it back to
        ``planned``.  The person has to look at it and approve it again, so the
        checkpoint names it, gives the reason and says where it stands.
        """
        told = self._clean(exc)  # a backend's error may quote what it was configured with
        with contextlib.suppress(ProjectError):
            if self.project.page(page.path).state == "planned":
                why = _sentence(told, 400)
                named = why if why.startswith(f"{page.id} ") else f"{page.id} was not run: {why}"
                return f"{named} It is planned again."
        return f"{page.id} could not be run: {_one_line(told, 300)}"

    def _refused_note(self, page: ResearchPage, record: Mapping[str, Any]) -> str | None:
        """What the checkpoint says about a run that was refused before any job started;
        ``None`` for a run that did start.

        ``research.experiments`` settles two refusals as a failed record: the worst
        case of the run's cost would pass what is left of the project's limit
        (``budget``), and the code changed between the look at the approval and the
        copy that was about to run (``approval``).  Neither is an experiment that ran.
        """
        kind = record.get("error_kind")
        if kind == "budget":
            limits = self.project.settings.limits
            return (
                f"{page.id} was not run: {_one_line(record.get('error'), 400)} (It was "
                f"measured against what is left of the project's "
                f"{_money(float(limits.experiment_usd))} for experiments, of which "
                f"{_money(self._state['spent']['experiment_usd'])} is spent.)"
            )
        if kind == "approval":
            return f"{page.id} was not run: {_sentence(record.get('error'), 400)}"
        return None

    @staticmethod
    def _unbounded_cost_note(page: ResearchPage, record: Mapping[str, Any]) -> list[str]:
        """Said when a run was billed on a GPU type that has no price: the cost limit
        for experiments rests on estimates, and for this run there was none."""
        plan = record.get("plan") if isinstance(record.get("plan"), Mapping) else {}
        gpu = str(plan.get("gpu") or "").strip()
        billed = record.get("backend") == "modal" and not _number(record.get("cost_usd"))
        if not gpu or not billed or agents.gpu_has_price(gpu):
            return []
        return [
            f"{page.id} ran on the GPU type {agents.named(gpu, 40)}, for which no price is "
            "known: what the run cost is not part of the experiment spending shown, and "
            "settings.limits.experiment_usd could not limit it."
        ]

    def _waiting_to_run(self, tried: Collection[str] = ()) -> list[ResearchPage]:
        """The approved experiments the run phase has not tried yet, in the order it
        runs them: those that wait in line in the order of their approvals (the order
        the app shows, see ``experiments.waiting_in_line``), then those that read
        approved and will not run as they stand, which are tried so that the
        checkpoint says why."""
        line = experiments.waiting_in_line(self.project, skip=tried, rule=self._stretch is not None)
        ahead = {page.path for page in line}
        rest = [
            page
            for page in self.project.pages(
                kind="experiment", review="accepted", state=["approved", "queued"]
            )
            if page.path not in tried and page.path not in ahead
        ]
        return line + rest

    def _run_approved(self, ctx: AgentContext) -> AgentReport:
        """Run every experiment the person approved, one after the other.

        In an unattended stretch their standing rule approves in their place: one
        experiment at a time, right before its run, so that a stretch that ends
        leaves no line of approved experiments behind.
        """
        self._recover()
        if self._stretch is None:
            # A stretch whose worker was killed in the middle of a run left the page as
            # running; the recovery has just put it back in the queue.  It was
            # autopilot's approval, for that stretch, and this is none.
            self._take_back_rule_approvals()
        self._tidy_hypotheses(release=False)
        self._held = {}
        cap = self._stretch_cap()
        if cap is not None:
            raise cap  # nothing is approved or started in a stretch that is over
        waiting = self._waiting_to_run()
        if not waiting:
            waiting = self._rule_approves_next()
        if not waiting:
            earlier = len(self._new_results())
            if not earlier:
                return AgentReport(summary=self._nothing_to_run())
            return AgentReport(
                summary=f"No experiment was waiting to run; {_plural(earlier, 'result')} of "
                "earlier runs had not been analysed yet.",
                notes=self._planned_again(),
            )
        outcomes: dict[str, list[str]] = {}
        notes: list[str] = self._planned_again()
        tried: set[str] = set()
        while waiting:
            if self._stretch_cap() is not None:
                # A cap of the stretch was reached while the last experiment ran: no
                # further one is started (what the person approved keeps its approval).
                # The phase ends here with what did run, and the stretch at its checkpoint.
                break
            page = waiting[0]
            tried.add(page.path)
            self._heed_stop()
            record = None
            try:
                self._check_experiment_budget(page)
                record = self._run_one(page, ctx)
            except _BudgetSpent as exc:
                if self._return_phase() is None or exc.limit.startswith("autopilot."):
                    raise  # the round's own run phase stops here, at a checkpoint
                # A run the person approved outside the order of the phases: what the
                # budget does not cover must not keep the phase that was next from running.
                self._warn_budget(exc, phase="run")
                notes.append(str(exc))
            except (ProjectError, ComputeError) as exc:
                if stops_the_run(exc):
                    raise
                # One experiment that cannot run does not keep the others from running.
                notes.append(self._not_run(page, exc))
            # Looked up again after every run: the person may approve one more while
            # another is running, and may cancel or change one that was waiting.
            waiting = self._waiting_to_run(tried) or self._rule_approves_next()
            if record is None:
                continue
            refused = self._refused_note(page, record)
            if refused is not None:
                # Said at the checkpoint, not only in the record of a run that never ran.
                notes.append(refused)
                continue
            self._ran += 1
            outcomes.setdefault(str(record.get("status") or "ended"), []).append(str(page.id))
            notes += self._unbounded_cost_note(page, record)
            # A run that a stop request cancelled is not a result to analyse.
            self._heed_stop()
        how = _and([f"{_and(ids)} {status}" for status, ids in outcomes.items()])
        summary = f"{_plural(self._ran, 'experiment')} ran: {how}." if how else "No experiment ran."
        notes += [
            f"Autopilot did not approve {name}: {_sentence(why, 400)}"
            for name, why in self._held.items()
            if why
        ]
        return AgentReport(produced=list(ctx.produced), summary=summary, notes=notes)

    def _rule_approves_next(self) -> list[ResearchPage]:
        """In an unattended stretch, when nothing waits to run: let the person's standing
        rule approve the next experiment it may, and return it as what waits (nothing
        otherwise, and always nothing outside a stretch).

        One at a time, and only when the line is empty: the approval is made right
        before the run it is for.  What the rule would not approve, or what the
        approval itself refused, is kept in ``_held`` with the reason, said at the
        checkpoint and tried no more in this run phase.
        """
        if self._stretch is None or not self._autopilot_on():
            return []
        if self._stretch_cap() is not None:
            # Past a cap the rule approves nothing more: an approval made now would be
            # a run started after the stretch should have ended.
            return []
        for page in experiments.rule_candidates(self.project):
            name = str(page.id)
            if name in self._held:
                continue
            self._heed_stop()
            try:
                held = experiments.rule_holds(self.project, page.path)
                if held is not None and page.state == "failed":
                    # It ran and failed, and is not tried again as it stands: that was
                    # said when it failed, and is no news at every later run phase.
                    self._held[name] = ""
                    continue
                if held is None:
                    self._check_experiment_budget(page)
                    approved = experiments.approve_by_rule(
                        self.project, page.path, backends=self._rule_backends(page)
                    )
                    return [approved]
            except _BudgetSpent as exc:
                held = str(exc)
            except (ProjectError, ComputeError) as exc:
                if stops_the_run(exc):
                    raise
                held = self._clean(exc)
            self._held[name] = _one_line(held, 500).rstrip(".")
        return []

    def _rule_backends(self, page: ResearchPage) -> Any:
        """The backend the approval asks whether it can run ``page``: the one this engine
        was given to run on, where it was given one; else the approval builds its own."""
        given = self._compute._given if hasattr(self._compute, "_given") else None
        if given is None or not hasattr(given, "run_job"):
            return None
        return {str(getattr(given, "name", "")): given}

    # -- analysis ----------------------------------------------------------------

    def _new_results(self) -> list[ResearchPage]:
        """Result records no analysis has been given yet.

        The record of a run the person cancelled is not among them: it is kept, as
        every settled run's is, but a run that was cut short on purpose measured
        nothing anyone should draw a finding from.
        """
        return [
            page
            for page in self.project.pages(kind="result")
            if not page.meta.get("analysed") and page.state != "cancelled"
        ]

    def _journal_name(self, ctx: AgentContext) -> str:
        """The journal page of this round: ``round-2``, or the next free name when an
        earlier analysis of the same round already wrote one."""
        own = ctx.session_id("analyst")
        number = 1
        while True:
            name = f"round-{ctx.round}" + (f"-{number}" if number > 1 else "")
            path = self.project.resolve(f"wiki/journal/{name}.md")
            if path is None or self.project.page(path).session == own:
                return name
            number += 1
