"""Engine processes, started and stopped by the web server.

The server never runs the engine itself.  For a project it starts
``python -m paper_orchestra research advance <project>`` (or ``research run``,
``research write``) as a process of its own and learns everything from the
project's files.  What this module guarantees:

* **One worker per project.**  A second start is refused while one runs.
* **A worker does not hang on the server.**  It has its own session and process
  group, writes to ``<project>/worker.log`` instead of a pipe, and is on record
  in the project directory (``.worker.json``, and the engine's own
  ``state.json["worker"]``).  A server that is killed leaves its workers
  running; the next server finds them again, shows them as running and can
  stop them.  A server that shuts down in an orderly way stops them: the ones
  it started and the ones a server that is gone left behind, not an engine the
  person started from a terminal (it is shown, and stopped only when asked).
* **Stopping is interrupt first, then kill.**  The interrupt lets the engine
  record the phase as interrupted.  Experiment jobs run in sessions of their
  own, so their process groups are looked up while the worker is still alive
  and signalled too; after the grace period everything is killed, whether or
  not the worker itself already went.  Only what the worker is and started is
  signalled: an engine that shares its process group with the script that
  runs it is reached by its pid.  All of that happens on a thread: a stop
  returns at once with the state ``stopping``.
* **A worker that dies is not shown as running.**  Its exit is observed; when it
  left ``state.json`` at ``running``, the phase is recorded as interrupted here.

No ``preexec_fn`` (unsafe in a threaded server) and no parent-death signal (it
follows the thread that started the child, and server threads end): the child
gets its session through ``start_new_session`` and sets up its own signal
handling.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_orchestra.research.model import format_stamp, parse_stamp
from paper_orchestra.research.project import EVENTS_FILE, STATE_FILE
from paper_orchestra.utils.common import write_json

__all__ = [
    "GRACE_SECONDS",
    "HEARTBEAT_STALE_SECONDS",
    "KINDS",
    "RECORD_FILE",
    "WORKER_LOG",
    "Launcher",
    "WorkerBusy",
    "WorkerError",
    "Workers",
    "default_command",
]

log = logging.getLogger(__name__)

#: What a worker can be asked to do: ``research advance``, ``research run``, ``research write``.
KINDS: tuple[str, ...] = ("advance", "run", "write")
#: The worker's combined output, in the project directory, appended to by every worker.
WORKER_LOG = "worker.log"
#: The server's record of the worker it started: pid, host, kind, command, start time.
RECORD_FILE = ".worker.json"
#: How long a worker gets after the interrupt before it is killed.  Longer than the ten
#: seconds the engine itself waits for model calls in flight when it is interrupted.
GRACE_SECONDS = 15.0
#: The same, when the server process itself is exiting and cannot wait long.
EXIT_GRACE_SECONDS = 2.0
#: A worker whose heartbeat in ``state.json`` is older than this and whose process is
#: gone is dead (spec 10.4).
HEARTBEAT_STALE_SECONDS = 60.0

_SETTLED_PHASES = frozenset({"completed", "failed", "interrupted", "skipped"})
_POLL_SECONDS = 0.1
#: How long a killed process is waited for before the stop gives up on seeing it go.
_KILL_WAIT_SECONDS = 5.0

#: ``(kind, project directory, options) -> argv`` of the worker process.
Launcher = Callable[[str, Path, Mapping[str, Any]], Sequence[str]]

#: What a worker of each kind is busy with, for a refusal a person reads.
_WORK = {"advance": "the agents", "run": "an experiment", "write": "the write-up of the paper"}


class WorkerError(RuntimeError):
    """A worker could not be started or stopped; the message says what to do."""


class WorkerBusy(WorkerError):
    """The project already has a worker.  ``status`` describes it; ``phase`` names what
    it is doing."""

    def __init__(self, status: Mapping[str, Any]) -> None:
        self.status = dict(status)
        self.phase = status.get("phase") or status.get("kind")
        doing = (
            f"the {self.phase} phase"
            if status.get("phase")
            else _WORK.get(str(status.get("kind")), "its work")
        )
        ending = " and is being stopped" if status.get("state") == "stopping" else ""
        super().__init__(
            f"This project already has a worker running {doing}{ending}. Wait for it to "
            "reach its checkpoint, or stop it first."
        )


def default_command(kind: str, root: Path, options: Mapping[str, Any]) -> list[str]:
    """The command line of a worker: this interpreter running the package's own CLI.

    ``options``: ``phase`` and ``instruction`` for ``advance``; ``experiments`` (ids)
    for ``run``; ``template`` and ``model`` for ``write``.  Values are passed as ``--name=value`` so
    that a value starting with a dash is still a value.
    """
    # A directory whose name starts with a dash, in a workspace given as ".", would read
    # as an option of the command.
    project = str(root) if not str(root).startswith("-") else f"./{root}"
    command = [sys.executable, "-m", "paper_orchestra", "research", kind, project]
    if kind == "advance":
        if options.get("phase"):
            command.append(f"--phase={options['phase']}")
        if options.get("instruction"):
            command.append(f"--instruction={options['instruction']}")
    elif kind == "run":
        command += [str(experiment) for experiment in options.get("experiments") or ()]
    elif kind == "write":
        for name in ("template", "model"):
            if options.get(name):
                command.append(f"--{name}={options[name]}")
    else:
        raise WorkerError(f"'{kind}' is not a kind of worker; the kinds are: {', '.join(KINDS)}.")
    return command


# ---------------------------------------------------------------------------
# Processes: who is alive, what did it start, how to signal it
# ---------------------------------------------------------------------------


def _process_table(proc: Path = Path("/proc")) -> list[tuple[int, int, int]]:
    """``(pid, parent pid, process group)`` of every process; empty when it cannot be read.

    Read from ``/proc`` where there is one (Linux), else from ``ps`` (macOS, BSD).
    """
    rows: list[tuple[int, int, int]] = []
    if (proc / "self" / "stat").is_file():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                # "pid (name) state ppid pgrp ...": the name may itself contain spaces and ")".
                fields = (entry / "stat").read_text(encoding="utf-8", errors="replace")
                _state, parent, group = fields.rsplit(")", 1)[1].split()[:3]
                rows.append((int(entry.name), int(parent), int(group)))
            except (OSError, ValueError, IndexError):
                continue  # the process ended while the table was being read
        return rows
    try:
        listing = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "pgid="],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("Could not list the running processes: %s", exc)
        return rows
    for line in listing.splitlines():
        fields_ = line.split()
        if len(fields_) == 3 and all(value.isdigit() for value in fields_):
            rows.append((int(fields_[0]), int(fields_[1]), int(fields_[2])))
    return rows


@dataclass
class _Targets:
    """What to signal to reach a worker and everything it started."""

    #: Process groups that hold nothing but the worker's own processes.
    groups: set[int] = field(default_factory=set)
    #: Single processes, each with its start token: the ones that sit in a group the
    #: worker did not make, next to processes that are not its own.
    processes: dict[int, str | None] = field(default_factory=dict)

    def add(self, other: _Targets) -> None:
        self.groups |= other.groups
        for pid, token in other.processes.items():
            self.processes.setdefault(pid, token)

    def signal(self, signum: int) -> None:
        for group in self.groups:
            try:
                os.killpg(group, signum)
            except OSError as exc:
                # Usually ProcessLookupError: the process group is already gone.
                log.debug("Could not signal the process group %d: %s", group, exc)
        for pid, token in self.processes.items():
            # A pid can be given out again within the grace period; a group cannot be
            # checked that way, a single process can.
            if not _same_process(pid, token):
                continue
            try:
                os.kill(pid, signum)
            except OSError as exc:
                log.debug("Could not signal the process %d: %s", pid, exc)


def _targets_of(pid: int) -> _Targets:
    """The worker ``pid`` and everything it started, directly or not (POSIX).

    The engine runs experiment jobs and LaTeX in sessions of their own so that it can
    kill them on a timeout.  Signalling the worker's own group therefore misses them,
    and once the worker is gone they can no longer be traced back to it: this has to
    be looked up while it is alive.

    A worker this server started leads its process group, and the whole group is
    its own.  An engine started from a script or a Makefile sits in the group of
    whatever started it: there the worker and its children are signalled one by
    one, so that a stop ends the engine and not the script around it.
    """
    children: dict[int, list[tuple[int, int]]] = {}
    for child, parent, group in _process_table():
        children.setdefault(parent, []).append((child, group))
    try:
        own = os.getpgid(pid)
    except OSError:
        own = None
    leads = own == pid
    targets = _Targets()
    if leads:
        targets.groups.add(pid)
    else:
        targets.processes[pid] = _start_token(pid)
    seen = {pid}
    queue = [pid]
    while queue:
        for child, group in children.get(queue.pop(), ()):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
            if group == own and not leads:
                targets.processes[child] = _start_token(child)
            else:
                targets.groups.add(group)
    # Never this server or its group, whatever a recycled pid may suggest.
    targets.groups -= {0, 1, os.getpgid(0)}
    for protected in (0, 1, os.getpid()):
        targets.processes.pop(protected, None)
    return targets


def _proc_stat(pid: int) -> tuple[str, int] | None:
    """``(state letter, start time in clock ticks)`` of a process on Linux, else ``None``."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        fields_ = text.rsplit(")", 1)[1].split()
        return fields_[0], int(fields_[19])
    except (OSError, ValueError, IndexError):
        return None


def _start_token(pid: int) -> str | None:
    """When the process started, as the system tells it: what distinguishes a process
    from a later one that was given the same pid.  ``None`` where that cannot be read.

    On Linux the kernel's own count; elsewhere what ``ps`` prints as the start time.
    """
    stat = _proc_stat(pid)
    if stat is not None:
        return str(stat[1])
    if os.name != "posix" or Path("/proc/self/stat").exists():
        return None
    try:
        listing = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return listing.strip() or None


def _pid_alive(pid: int) -> bool:
    """Whether a process with that pid exists and has not already exited."""
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - Windows; ``os.kill(pid, 0)`` would end it
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        alive = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(alive) and code.value == 259
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        pass  # it exists and is someone else's
    stat = _proc_stat(pid)
    return stat is None or stat[0] != "Z"  # a zombie only waits to be collected


def _same_process(pid: int, token: str | None) -> bool:
    """Whether ``pid`` is still the process that had it when ``token`` was read."""
    if not _pid_alive(pid):
        return False
    now = _start_token(pid)
    return token is None or now is None or now == token


def _mentions(pid: int, root: Path) -> bool:
    """Whether the command line of ``pid`` names the project: a guard against a pid from
    an old ``state.json`` that now belongs to an unrelated process.  True when the
    command line cannot be read."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True
    command = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    return str(root) in command or str(root.resolve()) in command or root.name in command


def _interrupt_signal(pid: int) -> int:
    """The signal that asks the worker to stop: the interrupt, as Ctrl-C would send it.

    A process started from a shell's background job inherits "ignore the interrupt",
    and so would the worker of a server started that way.  Where that can be seen
    (Linux), the termination signal is sent instead, which the CLI also turns into
    an orderly stop.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
        ignored = next(line for line in status.splitlines() if line.startswith("SigIgn:"))
        if int(ignored.split()[1], 16) & (1 << (signal.SIGINT - 1)):
            return signal.SIGTERM
    except (OSError, StopIteration, ValueError, IndexError):
        pass
    return signal.SIGINT


def _isolation() -> dict[str, Any]:
    """Popen options that give the worker its own session and process group, so that the
    whole tree can be stopped and a Ctrl-C in the server's terminal does not reach it."""
    if os.name == "nt":  # pragma: no cover - Windows
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


# ---------------------------------------------------------------------------
# state.json of a worker that is gone
# ---------------------------------------------------------------------------


def _now() -> str:
    return format_stamp(datetime.now(UTC))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_state(root: Path) -> dict[str, Any]:
    """``state.json`` of the project, or ``{}`` when the engine has not written one."""
    state = _read_json(Path(root) / STATE_FILE)
    return state if isinstance(state, dict) else {}


def _named_pid(state: Mapping[str, Any]) -> int | None:
    """The pid of the worker ``state.json`` names, if it names one."""
    worker = state.get("worker")
    pid = worker.get("pid") if isinstance(worker, dict) else None
    return pid if isinstance(pid, int) else None


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def mark_interrupted(root: Path, reason: str, *, by_person: bool) -> str | None:
    """Record in ``state.json`` that the phase in flight was interrupted.

    Only when the file still says ``running``: an engine that stopped in an orderly
    way has written its own account, and that stands.  Returns the name of the
    phase, or ``None`` when there was nothing to settle.  ``reason`` is kept as the
    error unless the person asked for the stop.  Call it only once the worker is gone.
    """
    path = Path(root) / STATE_FILE
    state = _read_json(path)
    if not isinstance(state, dict) or state.get("status") != "running":
        return None
    now = _now()
    phases = state.get("phases")
    for record in phases if isinstance(phases, list) else []:
        if isinstance(record, dict) and record.get("status") not in _SETTLED_PHASES:
            record["status"] = "interrupted"
            record["finished"] = record.get("finished") or now
            if not by_person:
                record["error"] = reason
    state["status"] = "stopped"
    state["worker"] = None
    state["error"] = None if by_person else reason
    write_json(path, state)
    return str(state.get("phase") or "")


def _take_name_off(root: Path, pid: int | None) -> None:
    """Take the name of a worker that is gone out of ``state.json`` (the one with ``pid``;
    whoever is named when ``pid`` is ``None``).

    An engine removes its name when it ends in an orderly way.  One that was killed
    leaves it behind with a heartbeat of a moment ago, which reads as a worker that
    is alive but cannot be seen.  Call it only once that worker is known to be gone.
    """
    path = Path(root) / STATE_FILE
    state = _read_json(path)
    if not isinstance(state, dict) or not isinstance(state.get("worker"), dict):
        return
    if pid is not None and _named_pid(state) != pid:
        return
    state["worker"] = None
    write_json(path, state)


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """One live worker of one project."""

    root: Path
    kind: str
    options: dict[str, Any]
    #: ``None`` for a worker on another machine that shares the project directory.
    pid: int | None
    started: str
    command: list[str] = field(default_factory=list)
    #: The process object when this server started it; ``None`` when it was found.
    process: subprocess.Popen[bytes] | None = None
    start_token: str | None = None
    host: str = ""
    stop_requested: bool = False
    stop_thread: threading.Thread | None = None
    #: When the kill is due (``time.monotonic()``); a later, more hurried stop moves it up.
    kill_at: float = 0.0
    #: When this server started it (``time.time()``), to tell what it wrote from what
    #: was there before.
    launched: float = 0.0
    #: Size of ``events.jsonl`` when this server started it, for the same purpose.
    events_size: int | None = None
    #: Known through ``state.json["worker"]`` alone: an engine started from a terminal.
    from_state: bool = False
    #: Whether a shutdown of this server stops it: true for a worker it started and for
    #: one a server that is gone left behind; false for an engine from a terminal and
    #: for the worker of another server that still runs.  A person's Stop reaches all.
    owned: bool = True
    done: threading.Event = field(default_factory=threading.Event)


_instances: weakref.WeakSet[Workers] = weakref.WeakSet()
_exit_hook = threading.Lock()
_exit_hook_registered = False


def _stop_all_at_exit() -> None:
    for workers in list(_instances):
        workers.shutdown(grace=EXIT_GRACE_SECONDS)


class Workers:
    """Starts, finds, stops and observes the worker processes of projects.

    Thread-safe.  ``launcher`` builds the command line (default:
    :func:`default_command`); tests substitute a small script.  ``grace`` is how
    long a worker gets between the interrupt and the kill.
    """

    def __init__(
        self,
        *,
        launcher: Launcher | None = None,
        grace: float = GRACE_SECONDS,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        global _exit_hook_registered
        self.launcher: Launcher = launcher or default_command
        self.grace = grace
        self._environ = environ
        self._guard = threading.RLock()
        self._entries: dict[Path, _Entry] = {}
        self._last_exit: dict[Path, dict[str, Any]] = {}
        self._closed = False
        self._host = socket.gethostname()
        with _exit_hook:
            # A server that exits without its shutdown having run (an exception on the
            # way out) must not leave workers behind either.
            if not _exit_hook_registered:
                atexit.register(_stop_all_at_exit)
                _exit_hook_registered = True
            _instances.add(self)

    # -- public ------------------------------------------------------------------

    def reopen(self) -> None:
        """Accept starts again after a :meth:`shutdown` (a server that is started twice)."""
        with self._guard:
            self._closed = False

    def start(self, root: str | Path, kind: str = "advance", **options: Any) -> dict[str, Any]:
        """Start a worker for the project in ``root`` and return its status.

        ``WorkerBusy`` when the project already has one (started here, found from an
        earlier server, or started from a terminal); ``WorkerError`` when the
        process cannot be started.  The project must exist on disk.
        """
        root = Path(root)
        if kind not in KINDS:
            raise WorkerError(
                f"'{kind}' is not a kind of worker; the kinds are: {', '.join(KINDS)}."
            )
        with self._guard:
            if self._closed:
                raise WorkerError("The server is shutting down; no worker was started.")
            current = self.status(root)
            if current["state"] != "idle":
                raise WorkerBusy(current)
            wanted = {key: value for key, value in options.items() if value not in (None, "")}
            entry = self._launch(root, kind, wanted)
            self._entries[self._key(root)] = entry
            return self._describe(entry)

    def stop(self, root: str | Path, *, grace: float | None = None) -> dict[str, Any]:
        """Ask the project's worker to stop; returns at once.

        The status returned says ``stopping`` while the interrupt, the grace period
        and the kill run on a thread, and ``idle`` when there was nothing to stop.
        """
        root = Path(root)
        with self._guard:
            status = self.status(root)
            entry = self._entries.get(self._key(root))
            if entry is None:
                return status
            self._begin_stop(entry, self.grace if grace is None else grace)
            return self._describe(entry)

    def status(self, root: str | Path) -> dict[str, Any]:
        """What the project's worker is doing.

        ``{"state": "idle" | "running" | "stopping", "kind", "phase", "pid",
        "started", "adopted", "host", "log", "last_exit"}``.  ``adopted`` says the
        worker was found, not started by this server.  ``last_exit`` describes
        the last worker seen to end: ``{"kind", "code", "at", "stopped",
        "interrupted_phase"}``.  Looking is also what notices a worker that
        died: its phase is then recorded as interrupted.
        """
        root = Path(root)
        with self._guard:
            key = self._key(root)
            entry = self._entries.get(key)
            if entry is not None and entry.process is None and not self._found_alive(entry):
                self._finished(entry, None)
                entry = None
            if entry is not None:
                return self._describe(entry)
            entry, dead_pid = self._adopt(root)
            return self._describe(entry) if entry is not None else self._idle(root, dead_pid)

    def live(self) -> int:
        """How many workers a :meth:`shutdown` of this server would stop."""
        with self._guard:
            return sum(
                1 for entry in self._entries.values() if entry.pid is not None and entry.owned
            )

    def known(self, root: str | Path) -> bool:
        """Whether a worker of the project is in this server's books (no file is read)."""
        return self._key(Path(root)) in self._entries

    def glance(self, root: str | Path) -> str:
        """``idle``, ``running`` or ``stopping`` from this server's books alone.

        Never waits: it is asked from the event loop many times a second, while a
        start or the settling of a dead worker may hold the books for a moment.
        :meth:`status` is what looks at the files.
        """
        entry = self._entries.get(self._key(Path(root)))
        if entry is None:
            return "idle"
        return "stopping" if entry.stop_thread is not None else "running"

    def wait(self, root: str | Path, timeout: float | None = None) -> bool:
        """Block until the project has no worker; ``False`` when ``timeout`` passed first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.status(root)["state"] != "idle":
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_SECONDS)
        return True

    def shutdown(self, *, grace: float | None = None) -> None:
        """Stop the workers that are this server's and wait until they are gone: the ones
        it started, and the ones a server that is gone left behind.  An engine the
        person started from a terminal, and the worker of another server that still
        runs, are left to whoever started them."""
        grace = self.grace if grace is None else grace
        with self._guard:
            self._closed = True
            entries = [
                entry for entry in self._entries.values() if entry.pid is not None and entry.owned
            ]
            for entry in entries:
                self._begin_stop(entry, grace)
        for entry in entries:
            if entry.stop_thread is not None:
                entry.stop_thread.join(timeout=grace + 2 * _KILL_WAIT_SECONDS)
            entry.done.wait(timeout=_KILL_WAIT_SECONDS)

    # -- starting ----------------------------------------------------------------

    @staticmethod
    def _key(root: Path) -> Path:
        return Path(os.path.realpath(root))

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ if self._environ is None else self._environ)
        # Unbuffered, so worker.log follows the worker; wide, so that the log lines of the
        # CLI are not folded at eighty columns (a folded line can cut a path in two).
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("COLUMNS", "200")
        return environment

    def _launch(self, root: Path, kind: str, options: dict[str, Any]) -> _Entry:
        command = [str(part) for part in self.launcher(kind, root, options)]
        if not command:
            raise WorkerError(f"The {kind} worker has no command to run.")
        started = _now()
        launched = time.time() - 1.0  # file times may be a little coarser than the clock
        events_size = _size(root / EVENTS_FILE)
        try:
            output = (root / WORKER_LOG).open("ab")
        except OSError as exc:
            raise WorkerError(
                f"The worker's log {root / WORKER_LOG} cannot be written ({exc}); no worker "
                "was started. Check that the project directory can be written to."
            ) from exc
        try:
            # What was asked is on disk before anything runs: the command line carries the
            # person's instruction, and nothing they typed may exist in memory only.
            header = f"\n[paper-orchestra] {started} starting: {shlex.join(command)}\n"
            output.write(header.encode("utf-8", errors="replace"))
            output.flush()
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=self._environment(),
                **_isolation(),
            )
        except (OSError, ValueError) as exc:
            raise WorkerError(
                f"The {kind} worker could not be started ({exc}). Nothing was changed."
            ) from exc
        finally:
            output.close()  # the worker has its own handle on the file
        entry = _Entry(
            root=root,
            kind=kind,
            options=options,
            pid=process.pid,
            started=started,
            launched=launched,
            events_size=events_size,
            command=command,
            process=process,
            start_token=_start_token(process.pid),
            host=self._host,
        )
        self._write_record(entry)
        threading.Thread(
            target=self._watch, args=(entry,), name=f"worker-{process.pid}", daemon=True
        ).start()
        return entry

    def _write_record(self, entry: _Entry) -> None:
        record = {
            "pid": entry.pid,
            "host": entry.host,
            "kind": entry.kind,
            "options": entry.options,
            "command": entry.command,
            "started": entry.started,
            "start_token": entry.start_token,
            "server_pid": os.getpid(),
            "server_token": _start_token(os.getpid()),
        }
        try:
            write_json(entry.root / RECORD_FILE, record)
        except OSError as exc:
            # The worker runs all the same; a later server finds it through state.json.
            log.warning("Could not write %s: %s", entry.root / RECORD_FILE, exc)

    def _watch(self, entry: _Entry) -> None:
        """Wait for a worker this server started, then settle what it left behind."""
        assert entry.process is not None
        code = entry.process.wait()
        self._finished(entry, code)

    # -- finding workers this server did not start ---------------------------------

    def _found_alive(self, entry: _Entry) -> bool:
        if entry.pid is None:  # another machine: only its heartbeat can be seen
            return _engine_alive(read_state(entry.root), entry.root, self._host)
        if entry.from_state and _named_pid(read_state(entry.root)) != entry.pid:
            # The engine takes its name out of state.json when it is done; a process that
            # lingers, or another one under its pid, no longer holds the project.
            return False
        return _same_process(entry.pid, entry.start_token)

    def _adopt(self, root: Path) -> tuple[_Entry | None, int | None]:
        """The worker a previous server (or a terminal) left running here, if any; and
        the pid of a worker an earlier server started that is known to be gone."""
        entry, dead_pid = self._from_record(root)
        if entry is None:
            entry = self._from_state(root, dead_pid)
        if entry is not None:
            self._entries[self._key(root)] = entry
        return entry, dead_pid

    def _from_record(self, root: Path) -> tuple[_Entry | None, int | None]:
        record = _read_json(root / RECORD_FILE)
        if not isinstance(record, dict):
            return None, None
        pid, token = record.get("pid"), record.get("start_token")
        token = token if isinstance(token, str) else None
        if record.get("host") != self._host or not isinstance(pid, int):
            return None, None
        if not _same_process(pid, token):
            # Its worker is gone; the record must not outlive it.
            with contextlib.suppress(OSError):
                (root / RECORD_FILE).unlink()
            return None, pid
        options = record.get("options")
        server, server_token = record.get("server_pid"), record.get("server_token")
        entry = _Entry(
            root=root,
            kind=str(record.get("kind") or "advance"),
            options=dict(options) if isinstance(options, dict) else {},
            pid=pid,
            started=str(record.get("started") or ""),
            command=[str(part) for part in record.get("command") or []],
            start_token=token,
            host=self._host,
            # The server that started it is still there (a second one on this workspace):
            # the worker is that server's to stop when it shuts down, not this one's.
            owned=not (
                isinstance(server, int)
                and server != os.getpid()
                and _same_process(server, server_token if isinstance(server_token, str) else None)
            ),
        )
        return entry, None

    def _from_state(self, root: Path, dead_pid: int | None) -> _Entry | None:
        """The engine ``state.json`` names as the project's worker, when it is alive.

        ``research advance`` sets the status to ``running`` while a phase runs;
        ``research run`` holds the project the same way (its name under ``worker``)
        while the status stays what it was.
        """
        state = read_state(root)
        worker = state.get("worker")
        if not isinstance(worker, dict) or not _engine_alive(state, root, self._host, dead_pid):
            return None
        host = str(worker.get("host") or self._host)
        pid = worker.get("pid") if host == self._host else None
        pid = pid if isinstance(pid, int) and _pid_alive(pid) else None
        return _Entry(
            root=root,
            kind="advance" if state.get("status") == "running" else "run",
            options={},
            pid=pid,
            started=str(worker.get("started") or ""),
            start_token=_start_token(pid) if pid is not None else None,
            host=host,
            from_state=True,
            owned=False,
        )

    # -- stopping ----------------------------------------------------------------

    def _begin_stop(self, entry: _Entry, grace: float) -> None:
        if entry.pid is None:
            raise WorkerError(
                f"The worker of this project runs on another machine ({entry.host}) or "
                "cannot be seen from here; stop it where it was started."
            )
        kill_at = time.monotonic() + grace
        if entry.stop_thread is not None:
            # Already being stopped.  A server that is exiting cannot wait as long as the
            # stop that was asked for earlier.
            entry.kill_at = min(entry.kill_at, kill_at)
            return
        entry.stop_requested = True
        entry.kill_at = kill_at
        entry.stop_thread = threading.Thread(
            target=self._stop_sequence, args=(entry,), name=f"stop-{entry.pid}", daemon=True
        )
        entry.stop_thread.start()

    def _gone(self, entry: _Entry, until: Callable[[], float]) -> bool:
        """Wait for the worker process itself to end, at most until ``until()`` (a
        ``time.monotonic()`` value, asked again at every look because it may move)."""
        while True:
            remaining = until() - time.monotonic()
            step = max(0.0, min(_POLL_SECONDS, remaining))
            if entry.process is not None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    entry.process.wait(timeout=step)
                    return True
            elif not self._found_alive(entry):
                return True
            else:
                time.sleep(step)
            if remaining <= 0:
                return False

    def _stop_sequence(self, entry: _Entry) -> None:
        pid = entry.pid
        assert pid is not None
        try:
            if os.name == "nt":  # pragma: no cover - Windows
                self._stop_on_windows(entry)
            else:
                targets = _targets_of(pid)
                targets.signal(_interrupt_signal(pid))
                if not self._gone(entry, lambda: entry.kill_at):
                    log.warning("The worker %d did not stop in time; killing it.", pid)
                    targets.add(_targets_of(pid))  # what it started in the meantime
                # Whether or not the worker itself has gone, nothing it started may
                # outlive the stop.
                targets.signal(signal.SIGKILL)
                killed = time.monotonic()
                self._gone(entry, lambda: killed + _KILL_WAIT_SECONDS)
        except Exception:
            log.exception("Stopping the worker %d failed.", pid)
        finally:
            if entry.process is None:
                # Nobody else observes the end of a worker this server did not start.
                self._finished(entry, None)

    def _stop_on_windows(self, entry: _Entry) -> None:  # pragma: no cover
        assert entry.pid is not None
        with contextlib.suppress(OSError):
            os.kill(entry.pid, signal.CTRL_BREAK_EVENT)
        if not self._gone(entry, lambda: entry.kill_at):
            subprocess.run(
                ["taskkill", "/PID", str(entry.pid), "/T", "/F"], capture_output=True, check=False
            )
            killed = time.monotonic()
            self._gone(entry, lambda: killed + _KILL_WAIT_SECONDS)

    # -- when a worker is gone -----------------------------------------------------

    def _finished(self, entry: _Entry, code: int | None) -> None:
        """Forget the worker, and settle ``state.json`` if it left the phase running.

        Several may notice the end of the same worker (a look at its status, the
        thread that stopped it); the first settles it, under the lock of the books,
        and the others find it done.
        """
        with self._guard:
            if entry.done.is_set():
                return
            try:
                self._close_books(entry, code)
            finally:
                # Whatever failed above, nobody waits for this worker any longer.
                entry.done.set()

    def _close_books(self, entry: _Entry, code: int | None) -> None:
        key = self._key(entry.root)
        if self._entries.get(key) is entry:
            del self._entries[key]
        record = _read_json(entry.root / RECORD_FILE)
        if isinstance(record, dict) and record.get("pid") == entry.pid:
            with contextlib.suppress(OSError):
                (entry.root / RECORD_FILE).unlink()
        phase = None
        unexpected = not entry.stop_requested
        try:
            phase = self._settle(entry, code)
        except Exception:
            log.exception("Could not settle the state of %s.", entry.root)
        self._last_exit[key] = {
            "kind": entry.kind,
            "code": code,
            "at": _now(),
            "stopped": entry.stop_requested,
            "interrupted_phase": phase,
        }
        if unexpected and (phase is not None or self._ended_badly(entry, code)):
            self._report_death(entry, code, phase)

    @staticmethod
    def _ended_badly(entry: _Entry, code: int | None) -> bool:
        """Whether a worker that nobody stopped ended in a way nothing else reports.

        Killed by a signal, always.  Otherwise a failure it left no account of.  For
        the engine: ``state.json`` untouched since the worker started means it
        stopped before the engine could record anything (no credentials, a broken
        project).  For a run or a write-up: no event since it started (the command
        refused, another worker holds the project); one that failed at its work is
        on record in its own events.  An interrupt from elsewhere (exit code 130)
        is somebody's orderly stop.
        """
        if code is None or code in (0, 130):
            return False
        if code < 0:
            return True
        if entry.kind != "advance":
            return entry.events_size is not None and _size(entry.root / EVENTS_FILE) == (
                entry.events_size
            )
        try:
            written = (entry.root / STATE_FILE).stat().st_mtime
        except OSError:
            return True
        return written < entry.launched

    def _settle(self, entry: _Entry, code: int | None) -> str | None:
        state = read_state(entry.root)
        if _named_pid(state) != entry.pid and _engine_alive(state, entry.root, self._host):
            # The state is another worker's by now (an engine started from a terminal
            # while a write-up ran, or right after this one ended): it is alive and
            # keeps its own account.
            return None
        if entry.stop_requested:
            phase = mark_interrupted(entry.root, "Stopped by you.", by_person=True)
        else:
            reason = _death_reason(entry.kind, code)
            phase = mark_interrupted(entry.root, reason, by_person=False)
        if phase is None:
            # No phase was running (a run of experiments holds the project without one).
            _take_name_off(entry.root, entry.pid)
        return phase

    def _report_death(self, entry: _Entry, code: int | None, phase: str | None) -> None:
        """Say in the activity stream that a worker ended without being asked to."""
        from paper_orchestra.research.project import Project, ProjectError

        summary = _death_reason(entry.kind, code)
        if phase:
            summary += f" The {phase} phase was recorded as interrupted; continuing resumes it."
        try:
            Project.open(entry.root).emit(
                "error", summary, phase=phase or "", data={"kind": entry.kind, "exit_code": code}
            )
        except (ProjectError, OSError) as exc:
            log.warning("Could not record the end of the worker in %s: %s", entry.root, exc)

    # -- describing --------------------------------------------------------------

    def _idle(self, root: Path, dead_pid: int | None = None) -> dict[str, Any]:
        """Nothing runs here.  A ``state.json`` that still says so is corrected: a phase
        left at running is recorded as interrupted, the name of a worker that is gone
        is taken off."""
        key = self._key(root)
        state = read_state(root)
        running = state.get("status") == "running"
        named = isinstance(state.get("worker"), dict)
        if (running or named) and not _engine_alive(state, root, self._host, dead_pid):
            if running:
                reason = "The worker of this phase is gone. Continuing resumes the phase."
                phase = mark_interrupted(root, reason, by_person=False)
                self._last_exit[key] = {
                    "kind": "advance",
                    "code": None,
                    "at": _now(),
                    "stopped": False,
                    "interrupted_phase": phase,
                }
            else:
                _take_name_off(root, None)
        return {
            "state": "idle",
            "kind": None,
            "phase": None,
            "pid": None,
            "started": None,
            "adopted": False,
            "host": None,
            "log": WORKER_LOG,
            "last_exit": self._last_exit.get(key),
        }

    def _describe(self, entry: _Entry) -> dict[str, Any]:
        state = read_state(entry.root)
        running_phase = state.get("phase") if state.get("status") == "running" else None
        return {
            "state": "stopping" if entry.stop_thread is not None else "running",
            "kind": entry.kind,
            "phase": running_phase or entry.options.get("phase"),
            "pid": entry.pid,
            "started": entry.started,
            "adopted": entry.process is None,
            "host": entry.host,
            "log": WORKER_LOG,
            "last_exit": self._last_exit.get(self._key(entry.root)),
        }


def _death_reason(kind: str, code: int | None) -> str:
    if code is None:
        how = "is gone"
    elif code < 0:
        how = f"was killed (signal {-code})"
    else:
        how = f"ended with exit code {code}"
    return f"The {kind} worker {how} without finishing. See {WORKER_LOG} in the project directory."


def _engine_alive(
    state: Mapping[str, Any], root: Path, this_host: str, dead_pid: int | None = None
) -> bool:
    """Whether the worker ``state.json`` names is alive (spec 10.4).

    It is dead when its heartbeat is older than a minute and its process is gone.
    A process on another machine cannot be looked at, so there the heartbeat alone
    decides.  A pid that exists while the heartbeat is stale must also look like a
    worker of this project: after a reboot the pid may be anyone's.  ``dead_pid``
    is a process this machine's server started and knows to have ended, however
    recent its last heartbeat.
    """
    worker = state.get("worker")
    if not isinstance(worker, dict):
        return False
    local = worker.get("host") in (None, "", this_host)
    if dead_pid is not None and worker.get("pid") == dead_pid and local:
        return False
    beat = parse_stamp(worker.get("heartbeat"))
    fresh = (
        beat is not None and (datetime.now(UTC) - beat).total_seconds() <= HEARTBEAT_STALE_SECONDS
    )
    pid = worker.get("pid")
    host = worker.get("host")
    if fresh or (host and host != this_host) or not isinstance(pid, int):
        return fresh
    return _pid_alive(pid) and _mentions(pid, root)
