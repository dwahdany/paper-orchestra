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
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_orchestra.compute import ComputeBackend, ComputeError, get_backend
from paper_orchestra.errors import chain, stops_the_run
from paper_orchestra.memory.harness import ContextHarness, RunCancelled
from paper_orchestra.memory.lock import WikiLock

# The usage ledger is the paper pipeline's, so that usage.json has one shape and one
# way of pricing wherever it is written.
from paper_orchestra.pipeline import read_carried_usage, usage_report
from paper_orchestra.research import agents, experiments
from paper_orchestra.research.agents import AgentContext, AgentReport
from paper_orchestra.research.model import (
    SCHEMA,
    Limits,
    NotFound,
    ProjectError,
    Request,
    ResearchPage,
    format_stamp,
    parse_stamp,
)
from paper_orchestra.research.project import EXPERIMENTS_DIR, STATE_FILE, Project
from paper_orchestra.utils.common import write_json
from paper_orchestra.utils.scholar import SemanticScholar

__all__ = [
    "PHASES",
    "Engine",
    "EngineError",
    "WorkerBusy",
    "read_state",
    "worker_alive",
]

log = logging.getLogger(__name__)

PHASES = ("intake", "research", "hypothesize", "design", "run", "analyze")

WORKER_LOCK_FILE = ".worker.lock"
USAGE_FILE = "usage.json"
#: A worker whose heartbeat is older than this, and whose process is gone, is dead.
STALE_HEARTBEAT_SECONDS = 60.0

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
#: Phases whose agent can act on a request of the person.
_TAKES_REQUESTS = ("research", "hypothesize", "design", "analyze")
_EXPERIMENT_DIR = re.compile(r"E\d+")
#: A request that reads as asking for more of the literature sends the next round to the
#: research phase first, and stays open until that phase has had it.  The words are a
#: guess at what the person meant, so they never keep a request from another phase.
_ASKS_FOR_RESEARCH = re.compile(
    r"\b(more research|literature|prior work|related work|papers|sources|citations?|"
    r"references|look up|search for|survey)\b",
    re.IGNORECASE,
)


class EngineError(ProjectError):
    """The engine cannot do what was asked; the message says why and what to do."""


class WorkerBusy(EngineError):
    """Another worker is running this project."""


class _BudgetSpent(Exception):
    """A limit of ``settings.limits`` is used up; the work stops at a checkpoint."""

    def __init__(self, limit: str, message: str, *, spent: float, allowed: float) -> None:
        super().__init__(message)
        self.limit = limit
        self.spent = spent
        self.allowed = allowed


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


def read_state(project: Project) -> dict[str, Any]:
    """The engine's state of ``project`` in the shape of ``state.json``.

    A project the engine has not touched yet reads as ``idle`` with ``intake``
    next.  A file that is missing keys, or is not JSON, is read as far as it goes.
    """
    state = _blank_state()
    try:
        data = json.loads((project.root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data = data if isinstance(data, dict) else {}
    for key in state:
        if key in data and key != "spent":
            state[key] = copy.deepcopy(data[key])
    spent = data.get("spent") if isinstance(data.get("spent"), Mapping) else {}
    state["spent"] = {name: _number(spent.get(name)) for name in state["spent"]}
    phases = state["phases"] if isinstance(state["phases"], list) else []
    # A record is an object; anything else in the list (a file edited by hand) is not one.
    state["phases"] = [record for record in phases if isinstance(record, dict)]
    if not isinstance(state["round"], int) or isinstance(state["round"], bool):
        state["round"] = 0
    return state


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
    on another machine only the heartbeat can be judged.
    """
    worker = state.get("worker")
    if not isinstance(worker, Mapping):
        return False
    beat = parse_stamp(worker.get("heartbeat"))
    age = ((now or datetime.now(UTC)) - beat).total_seconds() if beat else float("inf")
    if age <= STALE_HEARTBEAT_SECONDS:
        return True
    return worker.get("host") == socket.gethostname() and _pid_alive(worker.get("pid"))


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
        return self.named(getattr(plan, "backend", None) or default)

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
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _one_line(text: Any, limit: int = 400) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _money(amount: float) -> str:
    return f"${amount:.2f}"


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
    ) -> None:
        """``harness`` is the context harness every model call goes through; without one
        the engine builds the project's own (``client_for`` then chooses the clients).
        ``compute`` is a backend, or a callable choosing one for a plan; without one the
        backend named at approval, or in the project's settings, is used.  ``scholar``
        is the Semantic Scholar client of the research phase."""
        self.project = project
        self.harness = harness or agents.build_harness(project, client_for=client_for)
        self._compute = _Compute(project, compute)
        self._scholar = scholar
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._state = read_state(project)
        self._carried: list[dict[str, Any]] = []
        self._baseline: list[dict[str, Any]] = []
        #: Models that were used and have no known price: the model budget cannot count them.
        self._unpriced: list[str] = []
        #: Experiments the run phase in flight has executed.
        self._ran = 0

    # -- public ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """A copy of the state as this engine last wrote it."""
        with self._lock:
            return copy.deepcopy(self._state)

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

        ``instruction`` is the person's word for this run, handed to the agents of
        the phases it covers.  The returned state says how it ended: ``checkpoint``,
        ``stopped`` (interrupted; the same call resumes) or ``failed`` (a phase failed
        twice; ``error`` says why).  Raises :class:`WorkerBusy` when another worker
        has the project, :class:`EngineError` when the project's models cannot serve
        the agents, and ``KeyboardInterrupt`` after recording the interruption.
        """
        if phase is not None and phase not in PHASES:
            raise EngineError(f"{phase!r} is not a phase; the phases are: {', '.join(PHASES)}.")
        instruction = (instruction or "").strip() or None
        with self._working():
            self._refuse_unusable_models()
            name: str | None = phase or self._first_phase()
            told: list[str] = []
            notes: list[str] = []
            while name:
                outcome = self._run_phase(name, instruction)
                if outcome.status != "completed" or outcome.report is None:
                    break
                # What the phases of this call did and found is told at its checkpoint.
                told.append(outcome.report.summary)
                notes += outcome.report.notes
                name = self._after(name, outcome, told, notes)
        return self.state

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
            try:
                self._check_experiment_budget(page, self._limits_with(limits))
            except _BudgetSpent as exc:
                self._warn_budget(exc, phase="run")
                raise EngineError(str(exc)) from None
            compute = self._compute.named(backend) if isinstance(backend, str) else backend
            round = max(1, self._state["round"])
            ctx = self._context("run", round, f"r{round}-run-{page.id}", (), None)
            return self._run_one(page, ctx, compute=compute, limits=limits)

    # -- the worker --------------------------------------------------------------

    @contextlib.contextmanager
    def _working(self) -> Iterator[None]:
        """Hold the project's worker lock, with a heartbeat and a stop on SIGTERM."""
        lock = _WorkerLock(self.project.root)
        with lock:
            if not lock.acquired:
                raise WorkerBusy(self._busy_message(lock.error))
            self._stop.clear()
            self.project.reload()
            with self._lock:
                self._state = read_state(self.project)
                self._settle_dead_worker()
                self._count_runs()
                started = _now()
                self._state["worker"] = {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started": started,
                    "heartbeat": started,
                }
                self._carried = read_carried_usage(self.project.root / USAGE_FILE)
                self._baseline = self.harness.usage.to_dict()["by_label"]
                self._save()
            beat = _Heartbeat(self._beat, self.HEARTBEAT_SECONDS)
            beat.start()
            restore = self._stop_on_sigterm()
            try:
                yield
            finally:
                restore()
                beat.stop()
                self._compute.close()
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
        """Bring usage.json and ``spent.model_usd`` up to date; return the model spend."""
        with self._lock:
            report = usage_report(self.harness.usage, self._carried, self._baseline)
            write_json(self.project.root / USAGE_FILE, report)
            self._unpriced = [str(model) for model in report.get("unpriced_models") or []]
            spent = _number(report.get("estimated_cost_usd"))
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

    def _refuse_unusable_models(self) -> None:
        problems = agents.model_problems(self.project, self.harness)
        if not problems:
            return
        message = " ".join(problems)
        self.project.emit("error", _one_line(message, 600), data={"problems": problems})
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
                f"The model budget is used up: {_money(spent)} of {_money(limit)} is spent. "
                "Raise settings.limits.model_usd in the project's settings to go on; the "
                "phase then resumes where it stopped.",
                spent=spent,
                allowed=limit,
            )

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

    def _research_requested(self) -> bool:
        return any(_ASKS_FOR_RESEARCH.search(r.text) for r in self.project.requests())

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
        note = f"The worker stopped right after the {_NOUN[owed]} phase had finished."
        return self._after(owed, _Outcome("completed"), [], [note])

    def _after_last_record(self) -> str:
        """What comes next going by the phase records alone."""
        last = self._state["phases"][-1] if self._state["phases"] else None
        if last is None or last.get("phase") not in PHASES:
            return "intake"
        return _NEXT[last["phase"]] if last.get("status") == "completed" else last["phase"]

    def _next_phase(self) -> str:
        """The phase ``advance()`` starts with when none is named."""
        name = self._state.get("next_phase")
        if name not in PHASES:
            # A state file that does not say (written by hand, or by an older version):
            # going back to intake would do the project's first steps a second time.
            name = self._after_last_record()
        # Asked for after the checkpoint was written: more of the literature comes first.
        if name == "hypothesize" and self._research_requested():
            return "research"
        return name

    def _last_record(self, name: str, round: int) -> dict[str, Any] | None:
        """The latest record, when it is of this phase in this round."""
        phases = self._state["phases"]
        last = phases[-1] if phases else None
        if isinstance(last, dict) and last.get("phase") == name and last.get("round") == round:
            return last
        return None

    def _round_for(self) -> int:
        """A new round begins with the first phase after a completed analysis."""
        current = max(1, self._state["round"])
        last = self._state["phases"][-1] if self._state["phases"] else None
        ended = (
            isinstance(last, dict)
            and last.get("phase") == "analyze"
            and last.get("status") == "completed"
            and last.get("round") == current
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
        }

    def _complete_record(self, record: dict[str, Any]) -> None:
        """Give a record found in the state file the fields a resumed phase counts on."""
        for key, value in self._new_record(record["phase"], record["round"], None).items():
            record.setdefault(key, value)
        if not isinstance(record["produced"], list):
            record["produced"] = []
        if not isinstance(record["attempts"], int) or isinstance(record["attempts"], bool):
            record["attempts"] = 0

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

    def _requests_for(self, name: str) -> list[Request]:
        """The person's open requests, for a phase whose agent can act on requests."""
        return list(self.project.requests()) if name in _TAKES_REQUESTS else []

    def _close_requests(self, requests: Sequence[Request], name: str, round: int) -> None:
        """Close what the completed phase was given.

        A request that reads as asking for more of the literature is closed by the
        research phase only: any other phase was shown it, in case it was meant for
        that phase after all, but only research can do what it most likely asks.
        """
        for request in requests:
            if name != "research" and _ASKS_FOR_RESEARCH.search(request.text):
                continue
            # (One the person withdrew while the phase ran is gone, which is as good.)
            with contextlib.suppress(NotFound):
                self.project.close_request(request.id, round=round, phase=name)

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
            scholar=self._scholar,
            compute=self._compute,
        )

    def _run_phase(self, name: str, instruction: str | None) -> _Outcome:
        """Run one phase to its end, retrying a failure once; record how it went."""
        self.project.reload()
        round = self._round_for()
        session = self._session(name, round)
        record = self._last_record(name, round)
        if record is None or record.get("status") == "completed":
            record = self._new_record(name, round, instruction)
        else:
            self._complete_record(record)
        requests = self._requests_for(name)
        self._enter(record, instruction)
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
            except agents.SessionStopped:
                # A stop request, heard in the middle of a session's turn.
                return self._interrupted(record, clock, ctx)
            except KeyboardInterrupt:
                self._interrupted(record, clock, ctx)
                raise
            except Exception as exc:
                if self._stopping() or any(isinstance(e, RunCancelled) for e in chain(exc)):
                    return self._interrupted(record, clock, ctx)
                if attempt == 2 or stops_the_run(exc):
                    return self._failed(record, clock, ctx, exc)
                log.warning("The %s phase failed; trying once more.", name, exc_info=True)
                self._emit(
                    "agent.activity",
                    f"The {_NOUN[name]} phase failed ({_one_line(exc, 200)}); trying it once more.",
                    phase=name,
                    round=round,
                )
        assert report is not None
        outcome = self._completed(record, clock, ctx, report, requests)
        outcome.ran = self._ran if name == "run" else 0
        return outcome

    def _enter(self, record: dict[str, Any], instruction: str | None) -> None:
        """Say in the state file that this run of a phase is in flight.

        Its record is listed from the start (as ``running``), and the phase itself
        is what comes next for as long as it has not completed: whoever finds the
        state after this worker died resumes the phase by running ``advance()``.
        """
        name, round = record["phase"], record["round"]
        with self._lock:
            record.update(status="running", finished=None, error=None)
            if instruction:
                record["instruction"] = instruction
            if not any(record is known for known in self._state["phases"]):
                self._state["phases"].append(record)
            self._state.update(
                status="running",
                round=round,
                phase=name,
                next_phase=name,
                checkpoint=None,
                error=None,
            )
            self._save()
        self._emit(
            "phase.started",
            f"Round {round}: the {_NOUN[name]} phase started.",
            phase=name,
            round=round,
        )

    def _execute(self, ctx: AgentContext) -> AgentReport:
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
        report = agents.run_analyst(ctx, results=results, journal=self._journal_name(ctx))
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
        self,
        record: dict[str, Any],
        clock: float,
        ctx: AgentContext,
        report: AgentReport,
        requests: Sequence[Request],
    ) -> _Outcome:
        name, round = record["phase"], record["round"]
        ctx.note(report.produced)
        self._settle(record, clock, ctx, "completed", None)
        with self._lock:
            self._state["next_phase"] = _NEXT[name]
            self._save()
        self._close_requests(requests, name, round)
        self._emit(
            "phase.finished",
            report.summary or f"The {_NOUN[name]} phase finished.",
            phase=name,
            round=round,
            refs=record["produced"][:50],
            data={"produced": record["produced"], "seconds": record["seconds"]},
        )
        return _Outcome("completed", report)

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
        error = f"{type(exc).__name__}: {_one_line(exc, 1000)}"
        log.error("The %s phase failed.", name, exc_info=exc)
        self._settle(record, clock, ctx, "failed", error)
        with self._lock:
            self._state.update(status="failed", next_phase=name, error=error)
            self._save()
        again = " twice" if record["attempts"] > 1 else ""
        why = _one_line(exc, 300).rstrip(".")
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
        self._settle(record, clock, ctx, "interrupted", str(exc))
        self._warn_budget(exc, phase=name)
        extra = (
            " Results that exist can still be analysed: continue with the analyze phase."
            if name == "run" and self._new_results()
            else ""
        )
        self._checkpoint(name, record["round"], name, [], [str(exc) + extra])
        return _Outcome("budget")

    # -- between phases ----------------------------------------------------------

    def _after(
        self, name: str, outcome: _Outcome, told: Sequence[str], notes: Sequence[str]
    ) -> str | None:
        """The phase that follows at once, or ``None`` after writing the checkpoint."""
        round = self._state["round"]
        if name == "intake":
            return "research"
        if name == "run":
            if outcome.ran or self._new_results():
                return "analyze"
            self._checkpoint(name, round, "run", told, notes)
            return None
        following = _NEXT[name]
        if name == "analyze" and self._journal_asks_for_research(round):
            following = "research"
        if following == "hypothesize" and self._research_requested():
            following = "research"
        self._checkpoint(name, round, following, told, notes)
        return None

    def _checkpoint(
        self, name: str, round: int, following: str, told: Sequence[str], notes: Sequence[str]
    ) -> None:
        """Pause for the person: say what was done, what needs their eye, what comes next.

        ``told`` is what the phases since the last checkpoint did (each has said so
        in the activity feed already); ``notes`` is what the person has to know
        beyond that.  The state keeps the whole account, the event the part that is
        new to the feed.
        """
        pending = self.project.counts()["pending"]
        waiting = [agents.count_of(count, kind) for kind, count in pending["by_kind"].items()]
        if pending["proposals"]:
            waiting.append(_plural(pending["proposals"], "suggested edit"))
        verb = "awaits" if pending["total"] == 1 else "await"
        review = (
            f"{_and(waiting)} {verb} your review." if waiting else "Nothing awaits your review."
        )
        resumes = " (it resumes where it stopped)" if following == name and name != "run" else ""
        closing = [*notes, *self._unpriced_note(), review, f"Next: {_DOING[following]}{resumes}."]
        refs = [page.path for page in self.project.pages(review="proposed")][:50]
        event = self._emit(
            "checkpoint",
            " ".join(part.strip() for part in closing if part.strip()),
            phase=name,
            round=round,
            refs=refs,
            data={"pending": pending["by_kind"], "edits": pending["proposals"], "next": following},
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
                    "pending": dict(pending["by_kind"]),
                },
            )
            self._save()

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
        try:
            return experiments.run(
                project,
                experiment,
                compute=compute or self._compute,
                should_stop=self._stopping,
                repair=agents.make_repair(ctx),
                limits=self._run_limits(limits),
            )
        finally:
            self._count_runs()
            self._save()
            # Also a run found unfinished and settled on the way: it ended now.
            new = [run for number, run in settled().items() if number not in before]
            pages = [project.resolve(f"{experiment}-run{run.get('number')}") for run in new]
            ctx.note(path for path in pages if path)

    def _nothing_to_run(self) -> str:
        project = self.project
        planned = project.pages(kind="experiment", review="accepted", state="planned")
        if planned:
            names = _and([f"{page.id} ({_one_line(page.title, 60)})" for page in planned[:6]])
            return (
                "No experiment is approved to run. Approve the ones you want run: "
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

    def _run_approved(self, ctx: AgentContext) -> AgentReport:
        """Run every experiment the person approved, one after the other."""
        self._recover()
        self._tidy_hypotheses(release=False)
        waiting = self.project.pages(
            kind="experiment", review="accepted", state=["approved", "queued"]
        )
        if not waiting:
            earlier = len(self._new_results())
            if not earlier:
                return AgentReport(summary=self._nothing_to_run())
            return AgentReport(
                summary=f"No experiment was waiting to run; {_plural(earlier, 'result')} of "
                "earlier runs had not been analysed yet."
            )
        outcomes: dict[str, list[str]] = {}
        notes: list[str] = []
        tried: set[str] = set()
        while waiting:
            page = waiting[0]
            tried.add(page.path)
            self._heed_stop()
            self._check_experiment_budget(page)
            try:
                record = self._run_one(page, ctx)
            except (ProjectError, ComputeError) as exc:
                if stops_the_run(exc):
                    raise
                # One experiment that cannot run does not keep the others from running.
                notes.append(f"{page.id} could not be run: {_one_line(exc, 300)}")
                record = None
            # Looked up again after every run: the person may approve one more while
            # another is running, and may cancel or change one that was waiting.
            waiting = [
                other
                for other in self.project.pages(
                    kind="experiment", review="accepted", state=["approved", "queued"]
                )
                if other.path not in tried
            ]
            if record is None:
                continue
            if record.get("error_kind") == "budget":
                # Refused before it started, by the worst case of its cost: said at the
                # checkpoint, not only in the record of a run that never ran.
                limits = self.project.settings.limits
                notes.append(
                    f"{page.id} was not run: {_one_line(record.get('error'), 400)} (It was "
                    f"measured against what is left of the project's "
                    f"{_money(float(limits.experiment_usd))} for experiments, of which "
                    f"{_money(self._state['spent']['experiment_usd'])} is spent.)"
                )
                continue
            self._ran += 1
            outcomes.setdefault(str(record.get("status") or "ended"), []).append(str(page.id))
            # A run that a stop request cancelled is not a result to analyse.
            self._heed_stop()
        how = _and([f"{_and(ids)} {status}" for status, ids in outcomes.items()])
        summary = f"{_plural(self._ran, 'experiment')} ran: {how}." if how else "No experiment ran."
        return AgentReport(produced=list(ctx.produced), summary=summary, notes=notes)

    # -- analysis ----------------------------------------------------------------

    def _new_results(self) -> list[ResearchPage]:
        """Result records no analysis has been given yet."""
        return [page for page in self.project.pages(kind="result") if not page.meta.get("analysed")]

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

    def _journal_asks_for_research(self, round: int) -> bool:
        """Whether a journal page of the round says the literature needs another look."""
        return any(
            str(page.meta.get("next") or "").strip().lower() == "research"
            for page in self.project.pages(kind="journal")
            if page.round == round
        )
