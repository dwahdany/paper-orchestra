"""Engine processes, started and stopped by the web server.

The server never runs the engine itself.  For a project it starts
``python -P -m paper_orchestra research advance <project>`` (or ``research run``,
``research write``) as a process of its own and learns everything from the
project's files.  What this module guarantees:

* **One worker per project.**  A second start is refused while one runs.
* **A worker does not hang on the server.**  It has its own session and process
  group, writes to ``<project>/worker.log`` instead of a pipe, and is on record
  in the project directory (``.worker.json``, and the engine's own
  ``state.json["worker"]``).  The end of the server, however it comes, leaves
  its workers running; the next server finds them again, shows them as running
  and can stop them.  Only a server that was asked to (``stop_on_exit``) stops,
  when it ends, the workers that are its own: the ones it started and the ones
  a server that is gone left behind, not an engine the person started from a
  terminal (it is shown, and stopped only when asked).
* **Only a process that is shown to be this project's worker is signalled.**
  The files of a project directory may come from anywhere, and a process
  number in them may be anybody's by now.  The server signals a process it
  started itself and still holds, or one that it can show to be a worker of
  this very directory: the same process as when the record was written (its
  start time), running ``paper_orchestra research <kind>`` on this directory
  (its command line) or holding this directory's worker lock (the lock is
  held, and the process has that file open).  Anything else the files name is,
  at most, shown as a worker this server cannot verify: it has no Stop, and
  :meth:`Workers.forget` clears the name.  A copy of a project is not the
  original: when nobody holds the copy's worker lock, nobody works there,
  whatever the copied files say.
* **Workers run the installed package.**  They are started with ``python -P``
  and the project as an absolute path, so that no file of the directory the
  server was started in is imported in place of a module.
* **Stopping is interrupt first, then kill.**  The interrupt lets the engine
  record the phase as interrupted.  Experiment jobs run in sessions of their
  own, so their process groups are looked up while the worker is still alive
  and signalled too; after the grace period everything is killed, whether or
  not the worker itself already went.  Only what the worker is and started is
  signalled: an engine that shares its process group with the script that
  runs it is reached by its pid, and a process group is signalled only while
  a process that was seen in it is still there (the number of a group that
  emptied may be another group's by then).  All of that happens on a thread: a
  stop returns at once with the state ``stopping``.
* **A worker that dies is not shown as running.**  Its exit is observed; when it
  left ``state.json`` at ``running``, the phase is recorded as interrupted here,
  and the runs of experiments it left at ``running`` are settled through the
  experiments module's recovery, without anything being started.  That
  recovery ends what a run's record names as its job; the server lets it settle
  only where nothing so named is still running, because the record is a file
  like any other and the server settles at a mere look.
* **How a worker ended is on file, not in the server's memory.**  A worker that
  ended without an account of its own (the command refused, the process was
  killed) is given one in the project's events, with what it said last (the
  reason a command gives when it refuses); for a write-up that is a
  ``paper.finished`` event, so that the paper's own events say how the last
  write-up ended, to this server and to the next.
* **A worker that needed a model is not one that died.**  ``research advance``
  ends with its own exit code when the phase that would run needs a model and
  none can be used.  Nothing happened to the project, so no error is written:
  ``last_exit["reason"]`` is ``"needs_model"``, for whoever shows it as
  something to set up.  (What is missing is said by the engine's forecast,
  which the project view carries.)

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
import stat
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_orchestra.research.model import format_stamp, parse_stamp, quotable
from paper_orchestra.research.project import (
    EVENTS_FILE,
    STATE_FILE,
    forget_worker_stop,
    worker_stop_recorded,
)
from paper_orchestra.utils.common import write_json
from paper_orchestra.web.security import TOKEN_ENV, PathEscape, open_confined, open_for_append

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "GRACE_SECONDS",
    "HEARTBEAT_AHEAD_SECONDS",
    "HEARTBEAT_STALE_SECONDS",
    "KINDS",
    "NEEDS_A_MODEL_EXIT",
    "RECORD_FILE",
    "WORKER_LOCK_FILE",
    "WORKER_LOG",
    "Launcher",
    "WorkerBusy",
    "WorkerError",
    "Workers",
    "default_command",
    "experiments_in_line",
    "lock_held",
    "read_state",
    "read_state_file",
    "settle_stale_runs",
]

log = logging.getLogger(__name__)

#: What a worker can be asked to do: ``research advance``, ``research run``, ``research write``.
KINDS: tuple[str, ...] = ("advance", "run", "write")
#: The worker's combined output, in the project directory, appended to by every worker.
WORKER_LOG = "worker.log"
#: The server's record of the worker it started: pid, host, kind, command, start time.
RECORD_FILE = ".worker.json"
#: The file every worker of a project holds a kernel lock on while it works (the engine's
#: ``WORKER_LOCK_FILE``; named here so that the server need not import the engine).
WORKER_LOCK_FILE = ".worker.lock"
#: How long a worker gets after the interrupt before it is killed.  Longer than the ten
#: seconds the engine itself waits for model calls in flight when it is interrupted.
GRACE_SECONDS = 15.0
#: The same, when the server process itself is exiting and cannot wait long.
EXIT_GRACE_SECONDS = 2.0
#: A worker whose heartbeat in ``state.json`` is older than this and whose process is
#: gone is dead (spec 10.4).
HEARTBEAT_STALE_SECONDS = 60.0
#: A heartbeat further ahead of this machine's clock than this is not a sign of life: the
#: clock was set back, or the file says what it likes.  Written on this machine, a
#: heartbeat is at most a moment ahead; another machine's clock may differ by more.
HEARTBEAT_AHEAD_SECONDS = 5.0
HEARTBEAT_AHEAD_SECONDS_ELSEWHERE = 60.0

#: The exit code of ``research advance`` when it started nothing because the phase that
#: would run needs a model and none can be used (the command line's ``NEEDS_A_MODEL``;
#: named here so that the server need not import the command line).  Such an end is no
#: death: ``last_exit["reason"]`` says ``needs_model`` and no error is reported.
NEEDS_A_MODEL_EXIT = 3

_SETTLED_PHASES = frozenset({"completed", "failed", "interrupted", "skipped"})
_POLL_SECONDS = 0.1
#: How long a killed process is waited for before the stop gives up on seeing it go.
_KILL_WAIT_SECONDS = 5.0
#: How long a start waits for the runs of a dead worker to be settled before it goes on.
_SETTLE_WAIT_SECONDS = 20.0
#: How long the processes that a run's record names as its job are given to go before
#: the run is left unsettled: the job of a worker that was killed dies with it, a moment
#: after the server saw the worker go.
_JOB_GONE_WAIT_SECONDS = 3.0
#: The most a JSON file of a project may hold to be read here.  ``state.json`` with the
#: phases of hundreds of rounds stays far below it.
_MAX_JSON_BYTES = 32 * 1024 * 1024
#: No process has a number this large (Linux allows up to 2**22), and one that large in a
#: file cannot even be asked about.
_PID_LIMIT = 2**22
#: How much of the end of what a worker wrote is read for the reason it gave when it
#: ended, how much of that reason an event keeps, and how much of it a sentence quotes.
_LAST_WORDS_BYTES = 16 * 1024
_LAST_WORDS_CHARS = 1000
_QUOTED_CHARS = 600
#: How much of what was added to a project's events while a worker lived is looked
#: through for the worker's own account of its end.  A worker that left none ended
#: within moments of its start, and little was written beside it.
_EVENTS_SINCE_BYTES = 4 * 1024 * 1024

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
        if status.get("verified") is False:
            # It has no Stop here; the note says where it can be stopped, and that it
            # can be forgotten when nothing runs.
            super().__init__(f"Nothing was started. {status.get('note') or ''}".strip())
            return
        doing = (
            _phase_in_words(self.phase)
            if status.get("phase")
            else _WORK.get(str(status.get("kind")), "its work")
        )
        ending = " and is being stopped" if status.get("state") == "stopping" else ""
        super().__init__(
            f"This project already has a worker running {doing}{ending}. Wait for it to "
            "reach its checkpoint, or stop it first."
        )


def _phase_in_words(phase: Any) -> str:
    """The phase as a sentence for the person names it (``the hypothesis phase`` for
    ``hypothesize``): in the engine's words, which its own events and checkpoints use,
    so that a refusal of the server reads like everything else the person is told.
    The engine is asked when a sentence is made, not when this module is loaded."""
    try:
        from paper_orchestra.research.engine import phase_in_words
    except ImportError:  # an installation without the engine names the phase as it is
        return f"the {phase} phase"
    return phase_in_words(phase)


def default_command(kind: str, root: Path, options: Mapping[str, Any]) -> list[str]:
    """The command line of a worker: this interpreter running the package's own CLI.

    ``options``: ``phase`` and ``instruction`` for ``advance``; ``experiments`` (ids)
    and ``queue`` (go on with what is approved meanwhile) for ``run``; ``template`` and ``model`` for ``write``.  Values are passed as ``--name=value`` so
    that a value starting with a dash is still a value.

    ``-P`` keeps the directory the server was started in off the module search
    path: with a plain ``python -m`` a file there called ``secrets.py`` or
    ``types.py`` would be imported by every worker, next to the credentials.  The
    project is given as an absolute path, which says which directory is meant
    whatever the worker's working directory is, and never reads as an option.
    """
    project = os.path.abspath(root)
    command = [sys.executable, "-P", "-m", "paper_orchestra", "research", kind, project]
    if kind == "advance":
        if options.get("phase"):
            command.append(f"--phase={options['phase']}")
        if options.get("instruction"):
            command.append(f"--instruction={options['instruction']}")
    elif kind == "run":
        command += [str(experiment) for experiment in options.get("experiments") or ()]
        if options.get("queue"):
            # After those it goes on with whatever is approved meanwhile.
            command.append("--queue")
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


def _valid_pid(value: Any) -> int | None:
    """``value`` as the number of a process that a file may name, else ``None``.

    Not a bool, not 0 or 1 (the system's first process, the ancestor of nearly
    everything), nothing negative (to ``kill`` that is a whole process group or
    "everything") and nothing larger than a process number can be.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 1 < value < _PID_LIMIT else None


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


def _own_lineage(table: Sequence[tuple[int, int, int]] | None = None) -> set[int]:
    """This server and every process it descends from: never a worker, never signalled."""
    parents = {pid: parent for pid, parent, _ in (_process_table() if table is None else table)}
    lineage: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in lineage:
        lineage.add(pid)
        pid = parents.get(pid, 0)
    return lineage


def _group_of(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


@dataclass
class _Targets:
    """What to signal to reach a worker and everything it started."""

    #: Process groups that hold nothing but the worker's own processes, each with the
    #: processes seen in it (and their start tokens) when it was looked up.
    groups: dict[int, dict[int, str | None]] = field(default_factory=dict)
    #: Single processes, each with its start token: the ones that sit in a group the
    #: worker did not make, next to processes that are not its own.
    processes: dict[int, str | None] = field(default_factory=dict)

    def add(self, other: _Targets) -> None:
        for group, members in other.groups.items():
            known = self.groups.setdefault(group, {})
            for pid, token in members.items():
                known.setdefault(pid, token)
        for pid, token in other.processes.items():
            self.processes.setdefault(pid, token)

    def signal(self, signum: int) -> None:
        for group, members in self.groups.items():
            # The kill comes a grace period after the look.  A group whose processes
            # have all gone since may be somebody else's group under the same number by
            # now, and a group cannot be asked when it began: its processes can.
            if not any(
                _same_process(pid, token) and _group_of(pid) == group
                for pid, token in members.items()
            ):
                continue
            try:
                os.killpg(group, signum)
            except OSError as exc:
                # Usually ProcessLookupError: the process group is already gone.
                log.debug("Could not signal the process group %d: %s", group, exc)
        for pid, token in self.processes.items():
            # A pid can be given out again within the grace period.
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

    Nothing is returned for a process this server itself descends from, or for
    one that is no process a file may name: what such a process "started" is the
    person's whole session.
    """
    targets = _Targets()
    table = _process_table()
    lineage = _own_lineage(table)
    if _valid_pid(pid) is None or pid in lineage:
        return targets
    children: dict[int, list[tuple[int, int]]] = {}
    for child, parent, group in table:
        children.setdefault(parent, []).append((child, group))
    own = _group_of(pid)
    leads = own == pid
    if leads:
        targets.groups[pid] = {pid: _start_token(pid)}
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
                targets.groups.setdefault(group, {})[child] = _start_token(child)
    # Never this server, a process it descends from, or their groups, whatever a
    # recycled pid may suggest.
    for protected in (0, 1, *(_group_of(ancestor) or 0 for ancestor in lineage)):
        targets.groups.pop(protected, None)
    for protected in (0, 1, *lineage):
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
    stat_ = _proc_stat(pid)
    if stat_ is not None:
        return str(stat_[1])
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


def _pid_alive(pid: Any) -> bool:
    """Whether a process with that pid exists and has not already exited.  A value that
    is no process number (see :func:`_valid_pid`) names no process."""
    if _valid_pid(pid) is None:
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
    except (ProcessLookupError, OverflowError):
        return False
    except OSError:
        pass  # it exists and is someone else's
    stat_ = _proc_stat(pid)
    return stat_ is None or stat_[0] != "Z"  # a zombie only waits to be collected


def _same_process(pid: Any, token: str | None) -> bool:
    """Whether ``pid`` is still the process that had it when ``token`` was read."""
    if not _pid_alive(pid):
        return False
    now = _start_token(pid)
    return token is None or now is None or now == token


# -- is that process a worker of this project? -----------------------------------


def _command_line(pid: int) -> list[str] | None:
    """The arguments the process was started with, or ``None`` when they cannot be read.

    Exact on Linux.  Elsewhere ``ps`` prints them joined by spaces, so an argument
    that contains a space comes back in pieces (a project in such a directory is
    then not recognised by its command line).
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        raw = None
    if raw is not None:
        parts = raw.split(b"\x00")
        if parts and not parts[-1]:
            parts.pop()
        return [part.decode("utf-8", errors="surrogateescape") for part in parts] or None
    if os.name != "posix" or Path("/proc/self/stat").exists():
        return None
    try:
        listing = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return listing.split() or None


def _working_directory(pid: int) -> str | None:
    """Where the process runs (Linux), for an argument that is a relative path."""
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def _is_cli(argument: str) -> bool:
    """Whether a command-line argument is this package's command, as a script."""
    name = os.path.basename(argument.replace("\\", "/")).lower()
    if name in ("paper-orchestra", "paper-orchestra.exe", "paper-orchestra-script.py"):
        return True
    return argument.replace("\\", "/").endswith("paper_orchestra/__main__.py")


def _worker_arguments(command: Sequence[str]) -> list[str] | None:
    """What follows ``research <kind>`` in the command line of a worker of this package
    (``python -m paper_orchestra research advance ...`` or the ``paper-orchestra``
    script); ``None`` when the command line is not one."""
    for index in range(1, len(command) - 1):
        if command[index] != "research" or command[index + 1] not in KINDS:
            continue
        before = command[:index]
        as_module = any(
            before[place] == "paper_orchestra" and before[place - 1] == "-m"
            for place in range(1, len(before))
        )
        if as_module or any(_is_cli(argument) for argument in before):
            return list(command[index + 2 :])
    return None


def _names_directory(arguments: Sequence[str], cwd: str | None, root: Path) -> bool:
    """Whether one of the arguments is the directory ``root``, however it is spelled:
    relative to where the process runs, or through a link."""
    target = os.path.realpath(root)
    for argument in arguments:
        if argument.startswith("-"):
            continue  # an option, with its value
        if os.path.isabs(argument):
            candidate = argument
        elif cwd is not None:
            candidate = os.path.join(cwd, argument)
        else:
            continue
        with contextlib.suppress(OSError, ValueError):
            if os.path.realpath(candidate) == target:
                return True
    return False


def _has_open(pid: int, path: Path) -> bool:
    """Whether the process has the file open under this very path (Linux; ``False``
    where that cannot be seen: another system, or a process of another user).

    By path, not by what the file is: a copy of a project made of hard links shares
    its lock file with the original, and a worker of the original has it open under
    the original's path.
    """
    wanted = os.path.join(os.path.realpath(path.parent), path.name)
    try:
        descriptors = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for descriptor in descriptors:
        with contextlib.suppress(OSError):
            if os.readlink(f"/proc/{pid}/fd/{descriptor}") == wanted:
                return True
    return False


def lock_held(path: Path) -> bool | None:
    """Whether some process holds the kernel lock on the lock file ``path``.

    ``True``: somebody does.  ``False``: nobody does, whatever the files next to it
    say (the kernel drops a lock with its holder, however that died, and a copy of
    the file is another file that nobody locked).  ``None``: it cannot be told here
    (no such locks on this system or file system, or the file cannot be opened).

    This is how the engine's lock on a project (``.worker.lock``) and the lock of
    whoever runs an experiment (``.run.lock`` in its directory) are looked at
    from outside.  The lock is tried without waiting and as a shared one, so that
    two servers looking at once do not see each other, and given back at once.
    """
    if fcntl is None:  # pragma: no cover - Windows
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    except OSError:
        return None  # no kernel locks here: the engine then goes by the file itself
    finally:
        os.close(descriptor)


def _lock_held(root: Path) -> bool | None:
    """Whether some process holds the project's worker lock: somebody is at work in
    this directory (see :func:`lock_held`)."""
    return lock_held(root / WORKER_LOCK_FILE)


def _is_worker_of(pid: int, root: Path, expected: Sequence[str] | None = None) -> bool:
    """Whether the live process ``pid`` can be shown to be a worker of the project in
    ``root``: the only kind of process this server signals without having started it.

    Shown by its command line: exactly ``expected`` (what this server starts for
    the project), or ``paper_orchestra research <kind>`` with this directory among
    its arguments.  A worker that was given the project in another way (by its
    id) is shown by the lock every worker holds: the lock of this directory is
    held, and the process has that file open.  A file's word is never enough, and
    neither is a directory name that merely occurs in a command line.  Never this
    server or a process it descends from.
    """
    if _valid_pid(pid) is None or pid in _own_lineage():
        return False
    command = _command_line(pid)
    if command is None:
        return False
    if expected:
        wanted = [str(part) for part in expected]
        # Joined as well: where the arguments come from ``ps``, they come in pieces.
        if command == wanted or " ".join(command) == " ".join(wanted):
            return True
    arguments = _worker_arguments(command)
    if arguments is None:
        return False
    if _names_directory(arguments, _working_directory(pid), root):
        return True
    return _has_open(pid, root / WORKER_LOCK_FILE) and _lock_held(root) is True


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


def _uninstalled_package_path() -> str | None:
    """The directory this server imports the package from, when it can do so only
    because it was started there (``cd src && python -m paper_orchestra ui`` in a
    checkout that is not installed); ``None`` for an installed package.

    A worker is started with ``-P`` and would not find the package then.  It is to
    run the code its server runs, so that one directory is named to it outright.
    """
    import paper_orchestra

    if getattr(sys.flags, "safe_path", False) or not sys.path:
        return None
    home = os.path.dirname(os.path.dirname(os.path.abspath(paper_orchestra.__file__)))
    first = os.path.abspath(sys.path[0] or os.getcwd())
    return home if os.path.realpath(first) == os.path.realpath(home) else None


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


def _read_bytes(path: Path, *, follow_links: bool = False) -> bytes | None:
    """What a file of the project holds; ``None`` when it is absent or not an ordinary
    file.

    Not read through a symbolic link unless ``follow_links`` says so: what
    ``state.json`` holds is shown in the app, and a project directory that came
    from somebody else could name any JSON file of the person's that way.  A file
    far larger than any of these files gets is not taken into memory: they are
    read for every project at every look at the library.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    flags |= 0 if follow_links else getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        found = os.fstat(descriptor)
        if not stat.S_ISREG(found.st_mode) or found.st_size > _MAX_JSON_BYTES:
            return None
        chunks = []
        taken = 0
        while taken <= _MAX_JSON_BYTES and (chunk := os.read(descriptor, 1 << 20)):
            chunks.append(chunk)
            taken += len(chunk)
        if taken > _MAX_JSON_BYTES:
            return None  # it grew while it was read
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _as_json(data: bytes | None) -> Any:
    """The JSON value ``data`` is, or ``None`` when it is none."""
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):
        # RecursionError: brackets nested deeper than the reader follows.
        return None


def _read_json(path: Path, *, follow_links: bool = False) -> Any:
    """What a JSON file of the project holds; ``None`` when it is absent, not JSON, or
    not an ordinary file (see :func:`_read_bytes` for what is not read)."""
    return _as_json(_read_bytes(path, follow_links=follow_links))


def read_state_file(root: Path) -> tuple[dict[str, Any], bytes | None]:
    """``state.json`` of the project as :func:`read_state` gives it, and the bytes it was
    read from (``None`` when there is no such file of the project's own): what tells
    one content of the file from the next, for whoever keeps a reading of it."""
    data = _read_bytes(Path(root) / STATE_FILE)
    state = _as_json(data)
    return (state if isinstance(state, dict) else {}), data


def read_state(root: Path) -> dict[str, Any]:
    """``state.json`` of the project as the file holds it, or ``{}`` when the engine has
    not written one (or what is there is not the project's own file).

    This is what the supervisor goes by: who the file names as the worker, and
    whether a phase reads as running.  Where a project stands is the engine's
    reading, which knows what to make of a file that is lost or damaged; the API
    asks the engine for that and shows no state from here alone.
    """
    return read_state_file(root)[0]


def _named_pid(state: Mapping[str, Any]) -> int | None:
    """The pid of the worker ``state.json`` names, if it names a process."""
    worker = state.get("worker")
    return _valid_pid(worker.get("pid")) if isinstance(worker, dict) else None


def _heartbeat_fresh(worker: Mapping[str, Any], *, ahead: float) -> bool:
    """Whether the worker's last heartbeat is a sign of life: written within the last
    minute, and not more than ``ahead`` seconds after now.

    A heartbeat from the future says that the clock was set back since, or that
    the file is not an engine's: it would otherwise count as fresh for as long as
    the difference lasts, however long ago its writer died.
    """
    beat = parse_stamp(worker.get("heartbeat"))
    if beat is None:
        return False
    age = (datetime.now(UTC) - beat).total_seconds()
    return -ahead <= age <= HEARTBEAT_STALE_SECONDS


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _event_since(root: Path, offset: int, prefix: str) -> bool:
    """Whether the project's events hold, from byte ``offset`` on, an event whose type
    begins with ``prefix``.  Also true where that cannot be told (the events are not
    the project's own file, or more was written since than is looked through): the
    caller then adds no account of its own to a stream it cannot read."""
    try:
        with open_confined(root, EVENTS_FILE) as handle:
            if os.fstat(handle.fileno()).st_size < offset:
                return True  # another file than the one that was measured
            handle.seek(offset)
            written = handle.read(_EVENTS_SINCE_BYTES + 1)
    except (OSError, ValueError):  # PathEscape is a ValueError
        return True
    if len(written) > _EVENTS_SINCE_BYTES:
        return True
    for line in written.splitlines():
        event = _as_json(line)
        kind = event.get("type") if isinstance(event, dict) else None
        if isinstance(kind, str) and kind.startswith(prefix):
            return True
    return False


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
    if not isinstance(state, dict) or state.get("worker") is None:
        return
    if pid is not None and _named_pid(state) != pid:
        return
    state["worker"] = None
    write_json(path, state)


def settle_stale_runs(root: str | Path) -> list[dict[str, Any]]:
    """Settle the runs of experiments that a worker which died left at ``running``.

    Goes through the experiments module's recovery (``experiments.recover`` with
    ``follow=False``), which starts nothing and waits for nothing: a run nobody
    is running any more is recorded as ended because its worker stopped, and its
    experiment goes back to the queue (or is failed after several such deaths),
    so that the app no longer shows a run in progress with a log that never
    moves.  Experiments a live process is running are left alone, and so is a
    run that has a job on a remote backend: that job may still be alive, and
    following it to its end is a worker's part.  The other runs of the same
    project are settled all the same.

    Recovery also cleans up after a local job: the backend ends the processes that
    are provably what the job left behind (they carry the mark of a working
    directory the backend made, which a file cannot fake) and removes that
    directory.  The server settles at a mere look, and it ends no process at a
    look: while such processes are still running, the project's runs are left as
    they are, for the next worker (the person's Run) or the person's Cancel.  The
    job of a worker that was killed is killed with it, a moment later, and is
    waited for that long.

    Returns the records of the runs that were settled.
    """
    from paper_orchestra.research import experiments
    from paper_orchestra.research.project import Project

    project = Project.open(Path(root))
    stale = False
    jobs: list[Any] = []
    for page in project.pages(kind="experiment", state=("queued", "running")):
        if not page.id:
            continue
        running = [run for run in experiments.runs(project, page.id) if _is_running(run)]
        if running:
            jobs += [run.get("job") for run in running]
            jobs += _job_files(project.experiment_dir(page.id) / experiments.RUNS_DIR)
        stale = stale or bool(running) or page.state == "running"
    if not stale:
        return []
    if _job_still_running(jobs):
        log.info(
            "A run in %s names processes that are still running as its job. The server "
            "ends no process because a file names it, so the run is left as it is: the "
            "next worker of the project settles it, or cancelling the experiment does.",
            project.root,
        )
        return []
    return list(experiments.recover(project, follow=False))


def experiments_in_line(root: str | Path) -> list[str]:
    """The ids of the experiments that wait for their run in the project in ``root``,
    in the order they are run in (``experiments.waiting_in_line``)."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research.project import Project

    return [str(page.id) for page in experiments.waiting_in_line(Project.open(Path(root)))]


def _is_running(run: Any) -> bool:
    return isinstance(run, Mapping) and run.get("status") == "running"


def _job_files(runs: Path) -> list[Any]:
    """What the ``job.json`` files of an experiment's run directories hold: the record a
    local job leaves when it starts, which recovery reads where ``run.json`` has none.
    Read as recovery reads them, through a link too."""
    from paper_orchestra.research import experiments

    try:
        directories = [entry for entry in runs.iterdir() if entry.is_dir()]
    except OSError:
        return []
    return [_read_json(entry / experiments.JOB_FILE, follow_links=True) for entry in directories]


def _job_still_running(jobs: Sequence[Any]) -> bool:
    """Whether processes are still running that recovery would end for one of the job
    records, once they had a moment to go.

    What that is, is the local backend's to say (``LocalBackend.job_processes``: what
    its ``end_job`` would end), so that the server's look and the backend's clean-up
    cannot disagree: processes that carry the mark of a working directory the
    backend made and nobody holds any more.  A session or a directory that a record
    merely names is nothing to wait for, and nothing the backend would touch.
    """
    from paper_orchestra.compute.local import LocalBackend

    def running() -> bool:
        return any(LocalBackend.job_processes(job) for job in jobs if job is not None)

    deadline = time.monotonic() + _JOB_GONE_WAIT_SECONDS
    while running():
        if time.monotonic() >= deadline:
            return True
        time.sleep(_POLL_SECONDS)
    return False


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """One live worker of one project."""

    root: Path
    kind: str
    options: dict[str, Any]
    #: ``None`` for a worker this server cannot verify: one on another machine that
    #: shares the project directory, or one the files name without proof.
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
    #: Where in ``worker.log`` what this worker writes begins (after the line that says
    #: what was started); ``None`` for a worker that was found, whose output is not told
    #: from that of the workers before it.
    log_offset: int | None = None
    #: Known through ``state.json["worker"]`` alone: an engine started from a terminal.
    from_state: bool = False
    #: The server that started it, as its record says (pid and start token): while that
    #: server runs, the worker is its to stop at exit.  ``None`` for a worker this
    #: server started and for an engine from a terminal.
    server: tuple[int, str | None] | None = None
    #: Why this server cannot verify the worker, for the person; ``None`` for a worker
    #: it started or has shown to be this project's.  Such a worker is never signalled.
    unverified: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


_instances: weakref.WeakSet[Workers] = weakref.WeakSet()
_exit_hook = threading.Lock()
_exit_hook_registered = False


def _stop_all_at_exit() -> None:
    """For a server that was asked to stop its workers when it ends and exits without
    its shutdown having run (an exception on the way out).  Every other server leaves
    its workers running."""
    for workers in list(_instances):
        if workers.stop_on_exit:
            workers.shutdown(grace=EXIT_GRACE_SECONDS)


class Workers:
    """Starts, finds, stops and observes the worker processes of projects.

    Thread-safe.  ``launcher`` builds the command line (default:
    :func:`default_command`); tests substitute a small script.  ``grace`` is how
    long a worker gets between the interrupt and the kill.  ``stop_on_exit`` says
    whether the end of this server stops its workers (see :meth:`shutdown`); by
    default it does not (see :meth:`close`).  ``settle_runs`` is called with a
    project directory when a worker of it is seen to be gone (default:
    :func:`settle_stale_runs`).
    """

    def __init__(
        self,
        *,
        launcher: Launcher | None = None,
        grace: float = GRACE_SECONDS,
        environ: Mapping[str, str] | None = None,
        stop_on_exit: bool = False,
        settle_runs: Callable[[Path], Any] | None = None,
        waiting_runs: Callable[[Path], Sequence[str]] | None = None,
    ) -> None:
        global _exit_hook_registered
        self.launcher: Launcher = launcher or default_command
        self.grace = grace
        self.stop_on_exit = stop_on_exit
        self._settle_runs: Callable[[Path], Any] = settle_runs or settle_stale_runs
        self._environ = environ
        self._guard = threading.RLock()
        self._entries: dict[Path, _Entry] = {}
        self._last_exit: dict[Path, dict[str, Any]] = {}
        self._settling: dict[Path, threading.Thread] = {}
        self._waiting_runs: Callable[[Path], Sequence[str]] = waiting_runs or experiments_in_line
        #: Experiments whose run the person asked for while the project's worker was at
        #: other work: when that worker ends by itself (nobody stopped or killed it,
        #: here or anywhere else: see :meth:`_line_ended`), a worker is started for
        #: what still waits.  In memory only: a server that is restarted
        #: has forgotten it, and the approval then waits for Continue or Run now.
        self._promised: dict[Path, set[str]] = {}
        #: Projects whose files could not be judged, so that the log says it once.
        self._complained: set[Path] = set()
        self._closed = False
        self._host = socket.gethostname()
        with _exit_hook:
            if not _exit_hook_registered:
                atexit.register(_stop_all_at_exit)
                _exit_hook_registered = True
            _instances.add(self)

    # -- public ------------------------------------------------------------------

    def reopen(self) -> None:
        """Accept starts again after a :meth:`shutdown` or :meth:`close` (a server that
        is started twice)."""
        with self._guard:
            self._closed = False

    def start(self, root: str | Path, kind: str = "advance", **options: Any) -> dict[str, Any]:
        """Start a worker for the project in ``root`` and return its status.

        ``WorkerBusy`` when the project already has one (started here, found from an
        earlier server, started from a terminal, or one this server cannot
        verify); ``WorkerError`` when the process cannot be started.  The project
        must exist on disk.
        """
        root = Path(root)
        if kind not in KINDS:
            raise WorkerError(
                f"'{kind}' is not a kind of worker; the kinds are: {', '.join(KINDS)}."
            )
        # What a worker that died left half done is put in order first, so that the new
        # one does not find an experiment held by the server.  A look comes before the
        # wait: it is what notices a worker that died while no server watched it.
        self.status(root)
        self.settled(root, timeout=_SETTLE_WAIT_SECONDS)
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

    def run_in_line(self, root: str | Path, experiment: str) -> dict[str, Any]:
        """Have an approved experiment run without another request: now, in a worker of
        its own, or after the work the project's worker is doing.  Returns the worker's
        status with ``waits``.

        With no worker a ``run`` worker is started for the experiment, which then
        goes on with whatever else is approved until nothing waits (``waits`` is
        ``None``).  With a worker at work nothing is started and nothing is
        refused.  ``waits`` is ``"line"`` when that worker runs experiments and
        goes on with every approved one (``queue`` in its status, or the run phase
        of an ``advance``): it reaches this one by itself.  It is ``"worker"`` for any other worker (a phase, a
        write-up, a run that was started for its experiments only): the
        experiment is remembered, and when that worker has ended a worker is
        started for what still waits then.  That also covers a worker that was
        just past its last look at the line when the approval came.

        A stop ends the line, wherever it comes from: the Stop of the app, Ctrl-C or
        ``kill`` in the terminal the worker was started in, a kill of the process.
        After a worker that ended that way nothing is started; the approvals stand,
        and Continue or Run now runs them.

        ``WorkerBusy`` for a worker that is being stopped
        and for one this server cannot verify, whose end it cannot see either;
        ``WorkerError`` when a worker cannot be started.
        """
        root = Path(root)
        self.status(root)
        self.settled(root, timeout=_SETTLE_WAIT_SECONDS)
        with self._guard:
            if self._closed:
                raise WorkerError("The server is shutting down; no worker was started.")
            current = self.status(root)
            key = self._key(root)
            if current["state"] == "idle":
                entry = self._launch(root, "run", {"experiments": [experiment], "queue": True})
                self._entries[key] = entry
                return {**self._describe(entry), "waits": None}
            if current["state"] != "running" or current.get("verified") is not True:
                raise WorkerBusy(current)
            # Remembered in either case: a worker that drains the line may be past its
            # last look at it by now, and the one started after it finds what is left.
            self._promised.setdefault(key, set()).add(str(experiment))
            in_run_phase = current.get("kind") == "advance" and current.get("phase") == "run"
            return {
                **self._describe(self._entries[key]),
                "waits": "line" if current.get("queue") or in_run_phase else "worker",
            }

    def stop(self, root: str | Path, *, grace: float | None = None) -> dict[str, Any]:
        """Ask the project's worker to stop; returns at once.

        The status returned says ``stopping`` while the interrupt, the grace period
        and the kill run on a thread, and ``idle`` when there was nothing to stop.
        ``WorkerError`` for a worker this server cannot verify (``verified`` is
        false in its status): it is not signalled; see :meth:`forget`.
        """
        root = Path(root)
        with self._guard:
            status = self.status(root)
            entry = self._entries.get(self._key(root))
            if entry is None:
                return status
            # A stop ends the line: what was to start after this worker does not.
            self._promised.pop(self._key(root), None)
            self._begin_stop(entry, self.grace if grace is None else grace)
            return self._describe(entry)

    def forget(self, root: str | Path) -> dict[str, Any]:
        """Clear the name of a worker this server cannot verify, and return the status.

        For a project whose files name a worker that is not there, or not one
        this server can see: the name is taken out of ``.worker.json`` and
        ``state.json``, a phase that read as running is recorded as interrupted
        (continuing resumes it), and the project can be started again.  Nothing is
        signalled.  If a worker is in fact at work elsewhere, it writes its name
        back with its next heartbeat and is shown again.

        ``WorkerError`` for a worker this server started or has verified: that
        one is stopped, not forgotten.  With nothing to forget the status is
        returned as it is.
        """
        root = Path(root)
        with self._guard:
            status = self.status(root)
            key = self._key(root)
            entry = self._entries.get(key)
            if entry is None:
                return status
            if entry.unverified is None:
                raise WorkerError(
                    f"The worker of this project (process {entry.pid}) is running and is "
                    "this server's to stop: stop it instead. It is not forgotten while "
                    "it runs."
                )
            del self._entries[key]
            self._promised.pop(key, None)
            record = _read_json(root / RECORD_FILE)
            if not isinstance(record, dict) or record.get("host") == self._host:
                _remove_record(root)  # not the record of another machine's server
            phase = mark_interrupted(root, "Forgotten by you.", by_person=True)
            if phase is None:
                _take_name_off(root, None)
            self._last_exit[key] = {
                "kind": entry.kind,
                "code": None,
                "at": _now(),
                "stopped": False,
                "interrupted_phase": phase,
                "reason": None,
                "forgotten": True,
            }
            self._settle_later(root)
            return self._nothing(root)

    def status(self, root: str | Path) -> dict[str, Any]:
        """What the project's worker is doing.

        ``{"state": "idle" | "running" | "stopping", "kind", "phase", "pid",
        "started", "adopted", "host", "log", "verified", "note", "last_exit",
        "queue", "then_runs"}``.  ``queue`` says the worker runs experiments and
        goes on with every one that is approved meanwhile; ``then_runs`` are the
        experiments a worker is started for when this one has ended (see
        :meth:`run_in_line`).
        ``adopted`` says the worker was found, not started by this server.
        ``verified`` is ``True`` for a worker this server started or has shown to
        be this project's (it can be stopped), ``False`` for one the project's
        files name without such proof (``pid`` is then ``None``, ``note`` says why
        and what to do, :meth:`stop` refuses and :meth:`forget` clears it), and
        ``None`` when the project is idle.  ``last_exit`` describes the last
        worker seen to end: ``{"kind", "code", "at", "stopped",
        "interrupted_phase", "reason"}`` (and ``"forgotten": True`` after
        :meth:`forget`).  ``reason`` is ``"needs_model"`` when the agents were
        not started because the phase that would run needs a model and none
        can be used (no worker died: it is something to set up), else ``None``.
        Looking is also what notices a worker that died: its phase is then
        recorded as interrupted.

        Whatever the project's files hold, this does not raise: a file of one
        project must not break the listing of all of them.
        """
        root = Path(root)
        with self._guard:
            key = self._key(root)
            entry = self._entries.get(key)
            if entry is not None and entry.process is None:
                if entry.unverified is not None:
                    entry = None  # judged anew at every look, from the files as they are
                elif not self._found_alive(entry):
                    self._finished(entry, None)
                    entry = None
            if entry is not None:
                return self._describe(entry)
            try:
                entry = self._adopt(root)
                status = self._describe(entry) if entry is not None else self._idle(root)
                self._complained.discard(key)
                return status
            except Exception:
                # A file that says something nobody thought of, or a directory that
                # cannot be written: the project reads as idle, and the log says why
                # (once, not at every look).
                self._entries.pop(key, None)
                if key not in self._complained:
                    self._complained.add(key)
                    log.exception("Could not tell whether %s has a worker.", root)
                return self._nothing(root)

    def live(self) -> int:
        """How many workers a :meth:`shutdown` of this server would stop."""
        with self._guard:
            return sum(1 for entry in self._entries.values() if self._owned(entry))

    def running(self) -> list[dict[str, Any]]:
        """The workers in this server's books that it started or verified, each as its
        status plus ``dir`` (the project directory): what goes on when the server ends."""
        with self._guard:
            return [
                {**self._describe(entry), "dir": str(entry.root)}
                for entry in self._entries.values()
                if entry.unverified is None and entry.pid is not None
            ]

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

    def settled(self, root: str | Path, timeout: float | None = None) -> bool:
        """Block until the runs a dead worker of the project left behind are settled;
        ``False`` when ``timeout`` passed first.  Returns at once when nothing is being
        settled."""
        thread = self._settling.get(self._key(Path(root)))
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def stopping(self) -> int:
        """How many workers are being stopped at the moment because somebody asked."""
        with self._guard:
            return sum(1 for entry in self._entries.values() if entry.stop_thread is not None)

    def close(self) -> list[dict[str, Any]]:
        """The server ends and leaves its workers running: accept no more starts, and
        return the workers that go on (as :meth:`running` gives them).

        They belong to their projects.  Each is on record in its project
        directory, so the next server finds it again, shows it and can stop it.
        A stop that was asked for before is carried out first, to its kill if
        need be: the person was told the worker is stopping.
        """
        with self._guard:
            self._closed = True
            asked = [entry for entry in self._entries.values() if entry.stop_thread is not None]
        for entry in asked:
            thread = entry.stop_thread
            if thread is not None:
                remaining = max(0.0, entry.kill_at - time.monotonic())
                thread.join(timeout=remaining + 2 * _KILL_WAIT_SECONDS)
            entry.done.wait(timeout=_KILL_WAIT_SECONDS)
        with self._guard:
            return self.running()

    def shutdown(self, *, grace: float | None = None) -> None:
        """Stop the workers that are this server's and wait until they are gone: the ones
        it started, and the ones a server that is gone by now left behind.  An engine
        the person started from a terminal, the worker of another server that still
        runs, and a worker this server cannot verify are left to whoever started them.

        This is what ``stop_on_exit`` asks for at the end of the server; without it
        the end of the server is :meth:`close`.
        """
        grace = self.grace if grace is None else grace
        with self._guard:
            self._closed = True
            entries = [entry for entry in self._entries.values() if self._owned(entry)]
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
        # The app's access token is the server's alone: whoever holds it uses the app as
        # the person, and neither a worker nor the jobs it starts have a use for it.
        environment.pop(TOKEN_ENV, None)
        # Unbuffered, so worker.log follows the worker; wide, so that the log lines of the
        # CLI are not folded at eighty columns (a folded line can cut a path in two).
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("COLUMNS", "200")
        # As ``-P`` on the command line: the worker imports the installed package, not a
        # file of the directory the server happens to run in.
        environment["PYTHONSAFEPATH"] = "1"
        source = _uninstalled_package_path()
        if source is not None:
            known = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, known]))
        return environment

    def _launch(self, root: Path, kind: str, options: dict[str, Any]) -> _Entry:
        command = [str(part) for part in self.launcher(kind, root, options)]
        if not command:
            raise WorkerError(f"The {kind} worker has no command to run.")
        started = _now()
        launched = time.time() - 1.0  # file times may be a little coarser than the clock
        events_size = _size(root / EVENTS_FILE)
        try:
            output = open_for_append(root, WORKER_LOG)
        except PathEscape as exc:
            raise WorkerError(
                f"The worker's log cannot be used ({exc}); no worker was started. Remove "
                f"or rename {root / WORKER_LOG} and start again: the server then makes a "
                "new log."
            ) from exc
        except OSError as exc:
            raise WorkerError(
                f"The worker's log {root / WORKER_LOG} cannot be written ({exc}); no worker "
                "was started. Check that the project directory can be written to."
            ) from exc
        forget_worker_stop(root)  # the word of a worker that was stopped earlier
        try:
            # What was asked is on disk before anything runs: the command line carries the
            # person's instruction, and nothing they typed may exist in memory only.
            header = f"\n[paper-orchestra] {started} starting: {shlex.join(command)}\n"
            output.write(header.encode("utf-8", errors="replace"))
            output.flush()
            log_offset = os.fstat(output.fileno()).st_size
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
            log_offset=log_offset,
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
        """Whether a worker that was found and verified is still there."""
        if entry.from_state and _named_pid(read_state(entry.root)) != entry.pid:
            # The engine takes its name out of state.json when it is done; a process that
            # lingers, or another one under its pid, no longer holds the project.
            return False
        return _same_process(entry.pid, entry.start_token)

    def _owned(self, entry: _Entry) -> bool:
        """Whether a :meth:`shutdown` of this server stops the worker: one it started, or
        one whose server is gone by now.  Asked when the shutdown comes, not when the
        worker was found: the server that started it may have ended since."""
        if entry.unverified is not None or entry.pid is None or entry.from_state:
            return False
        if entry.process is not None or entry.server is None:
            return True
        server, token = entry.server
        return server == os.getpid() or not _same_process(server, token)

    def _adopt(self, root: Path) -> _Entry | None:
        """The worker a previous server (or a terminal) left running here, if any.  It
        takes the place of whatever the books held for the project (a worker that
        could not be verified at the last look)."""
        entry = self._from_record(root)
        if entry is None or entry.unverified is not None:
            # What the record could not show, the engine's own name in state.json may.
            entry = self._from_state(root) or entry
        key = self._key(root)
        if entry is not None:
            self._entries[key] = entry
        else:
            self._entries.pop(key, None)
        return entry

    def _expected_command(self, kind: str, root: Path, options: Mapping[str, Any]) -> list[str]:
        """What this server would start for the project, to compare a process with."""
        try:
            return [str(part) for part in self.launcher(kind, root, options)]
        except Exception:  # a record can ask for anything; a launcher need not take it
            return []

    def _from_record(self, root: Path) -> _Entry | None:
        """The worker ``.worker.json`` names, when it is on this machine and alive.

        The record lies in the project directory, which may have come from anywhere
        (a copy, an archive, a shared folder): it is believed as far as the live
        process bears it out.  A record whose process is gone, or which names
        nothing that can be a process, is removed.
        """
        record = _read_json(root / RECORD_FILE)
        if not isinstance(record, dict):
            if (root / RECORD_FILE).is_file():
                _remove_record(root)  # not a record: nothing a server wrote
            return None
        if record.get("host") != self._host:
            return None  # another machine's server wrote it, and it is that server's
        pid = _valid_pid(record.get("pid"))
        token = record.get("start_token")
        token = token if isinstance(token, str) and token else None
        if pid is None or not _same_process(pid, token):
            _remove_record(root)  # its worker is gone; the record must not outlive it
            return None
        named = record.get("kind")
        kind = named if isinstance(named, str) and named in KINDS else "advance"
        options = record.get("options")
        options = dict(options) if isinstance(options, dict) else {}
        command = record.get("command")
        entry = _Entry(
            root=root,
            kind=kind,
            options=options,
            pid=pid,
            started=str(record.get("started") or ""),
            command=[str(part) for part in command] if isinstance(command, list) else [],
            start_token=token,
            host=self._host,
        )
        server = _valid_pid(record.get("server_pid"))
        if server is not None:
            server_token = record.get("server_token")
            entry.server = (server, server_token if isinstance(server_token, str) else None)
        # The start time says it is the process the record was written for; the command
        # line says that process works on this directory and no other.
        expected = self._expected_command(kind, root, options)
        if token is not None and _is_worker_of(pid, root, expected or None):
            return entry
        held = _lock_held(root)
        if held is False:
            # Nobody works in this directory: the record came with a copy of another
            # project, or names a process that has nothing to do with this one.
            _remove_record(root)
            return None
        entry.pid = None
        entry.unverified = _unverified_note(pid, held=bool(held))
        return entry

    def _from_state(self, root: Path) -> _Entry | None:
        """The engine ``state.json`` names as the project's worker, when it is alive.

        ``research advance`` sets the status to ``running`` while a phase runs;
        ``research run`` holds the project the same way (its name under ``worker``)
        while the status stays what it was.

        On this machine the process and the project's worker lock decide, not the
        heartbeat: a worker is this project's when its command line or the lock
        says so, and when nobody holds the lock nobody works here, however recent
        the heartbeat and whatever process has the number by now.  On another
        machine neither can be looked at, and the heartbeat alone tells: such a
        worker is shown, and is not this server's to stop.
        """
        state = read_state(root)
        worker = state.get("worker")
        if not isinstance(worker, dict):
            return None
        host = worker.get("host")
        host = host if isinstance(host, str) and host else self._host
        # A worker that is not an engine phase says what it is (a write-up); otherwise the
        # status tells: `advance` sets it to running, `run` leaves it as it was.
        named = worker.get("kind")
        inferred = "advance" if state.get("status") == "running" else "run"
        entry = _Entry(
            root=root,
            kind=named if isinstance(named, str) and named in KINDS else inferred,
            # `research run --queue` says with its name that it goes on with what is
            # approved meanwhile.
            options={"queue": True} if worker.get("queue") is True else {},
            pid=None,
            started=str(worker.get("started") or ""),
            host=host,
            from_state=True,
        )
        pid = _valid_pid(worker.get("pid"))
        alive = pid is not None and _pid_alive(pid)
        if pid is not None and alive and _is_worker_of(pid, root):
            # Whatever machine the file names: the process is here and is this project's
            # (a laptop is called something else on another network).
            entry.pid = pid
            entry.start_token = _start_token(pid)
            entry.host = self._host
            return entry
        if host != self._host:
            if not _heartbeat_fresh(worker, ahead=HEARTBEAT_AHEAD_SECONDS_ELSEWHERE):
                return None
            entry.unverified = (
                f"The project's files name a worker on another machine ({host}), which "
                "this server can neither see nor stop. Stop it where it was started. If "
                "nothing runs there any more, forget it here and the project can be "
                "continued."
            )
            return entry
        held = _lock_held(root)
        if held is False:
            return None
        if held is None and not (alive and _heartbeat_fresh(worker, ahead=HEARTBEAT_AHEAD_SECONDS)):
            return None
        entry.unverified = _unverified_note(pid, held=bool(held))
        return entry

    # -- stopping ----------------------------------------------------------------

    def _begin_stop(self, entry: _Entry, grace: float) -> None:
        if entry.unverified is not None or entry.pid is None:
            raise WorkerError(
                "This server does not signal a process it cannot show to be this "
                f"project's worker. {entry.unverified or ''}".strip()
            )
        if entry.pid in _own_lineage():
            raise WorkerError(
                f"Process {entry.pid} is this server or a process it was started from; "
                "it is not a worker and is not signalled."
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
                self._after_stop_of_found(entry)

    def _after_stop_of_found(self, entry: _Entry) -> None:
        """Close the books on a worker this server did not start, once it was stopped:
        nobody else observes its end.  One that could not be ended (it runs as another
        user) is still the project's worker and is shown as running again; recording
        its phase as interrupted would say something that is not so."""
        if not self._found_alive(entry):
            self._finished(entry, None)
            return
        log.warning(
            "The worker %s of %s could not be stopped; it is still running.",
            entry.pid,
            entry.root,
        )
        with self._guard:
            entry.stop_thread = None
            entry.stop_requested = False

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
            _remove_record(entry.root)
        phase = None
        unexpected = not entry.stop_requested
        line_ended = self._line_ended(entry, code)  # before the name is taken off
        try:
            phase = self._settle(entry, code)
        except Exception:
            log.exception("Could not settle the state of %s.", entry.root)
        # The command's own word for "the next phase needs a model, nothing was started".
        # With a phase left running it is not that, whatever the number.
        # (`run` says it too: under the sdk runtime an agent carries a run out.)
        needs_model = (
            unexpected
            and entry.kind in ("advance", "run")
            and code == NEEDS_A_MODEL_EXIT
            and phase is None
        )
        self._last_exit[key] = {
            "kind": entry.kind,
            "code": code,
            "at": _now(),
            "stopped": entry.stop_requested,
            "interrupted_phase": phase,
            "reason": "needs_model" if needs_model else None,
        }
        promised = self._promised.pop(key, None)
        if needs_model:
            return  # nothing died, and nothing ran that could have left a run behind
        if promised and not line_ended and not self._closed:
            # The person asked for a run while this worker was at work: it is started
            # now, for whatever still waits.  Not after a stop, which ends the line.
            threading.Thread(
                target=self._start_what_waits,
                args=(entry.root, set(promised), entry.pid),
                name="run-next",
                daemon=True,
            ).start()
        if unexpected and (phase is not None or self._ended_badly(entry, code)):
            self._report_death(entry, code, phase)
        if entry.kind in ("advance", "run") and code != 0:
            # It may have been in the middle of a run of an experiment, which only the
            # process that runs it settles: killed, it left the run at "running".
            self._settle_later(entry.root)

    @staticmethod
    def _line_ended(entry: _Entry, code: int | None) -> bool:
        """Whether the worker that is gone was stopped or killed, by anybody: nothing is
        started after such a worker, since a stop ends the line of experiments (and
        the worker started next would run again the one that was just stopped).

        The Stop of this server; an interrupt from elsewhere, which a worker command
        answers with exit code 130 (Ctrl-C or ``kill`` in a terminal); a signal that
        ended the process.  Of a worker this server did not start no exit code is
        known: the command leaves word of a stop in the project
        (``project.record_worker_stop``), and one that was killed leaves its name in
        ``state.json``, which a worker that ends by itself takes off.  Asked before
        that name is taken off for it.
        """
        if entry.stop_requested or worker_stop_recorded(entry.root, entry.pid):
            return True
        if code is not None:
            return code == 130 or code < 0
        return entry.pid is not None and _named_pid(read_state(entry.root)) == entry.pid

    @staticmethod
    def _ended_badly(entry: _Entry, code: int | None) -> bool:
        """Whether a worker that nobody stopped ended in a way nothing else reports.

        Killed by a signal, always.  Otherwise a failure it left no account of.  For
        the engine: ``state.json`` untouched since the worker started means it
        stopped before the engine could record anything (a broken project, a
        command that could not start).  For a run: no event since it started (the command refused,
        another worker holds the project); one that failed at its work is on record
        in its own events.  For a write-up: no event of the paper's since it
        started.  A write-up tells its start and its end in ``paper.`` events and
        nothing else does, so what anybody else wrote in that time (the person
        decided a page a moment after starting it) is no account of it.  An
        interrupt from elsewhere (exit code 130) is somebody's orderly stop.
        """
        if code is None or code in (0, 130):
            return False
        if code < 0:
            return True
        if entry.kind == "write":
            return entry.events_size is not None and not _event_since(
                entry.root, entry.events_size, "paper."
            )
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
        if _named_pid(state) != entry.pid and self._from_state(entry.root) is not None:
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
        """Say in the activity stream that a worker ended without being asked to, with
        what it said last when it ended by itself (the reason a command gives when it
        refuses).  The event is the account of that end that outlasts this server.

        A write-up is told as a ``paper.finished`` event, as the write-up tells its
        own end: whoever shows the paper reads how the last one went from the
        paper's events.  Any other worker is an ``error`` event.
        """
        from paper_orchestra.research.project import Project, ProjectError

        # A worker that was killed had no chance to say why: what it printed is no reason.
        said = _last_words(entry.root, entry.log_offset) if code is not None and code > 0 else ""
        if entry.kind == "write":
            kind, summary, data = (
                "paper.finished",
                _write_up_ended(code, said),
                _paper_data(code, said),
            )
        else:
            summary = _death_reason(entry.kind, code)
            if phase:
                summary += (
                    f" {_phase_in_words(phase).capitalize()} was recorded as interrupted; "
                    "continuing resumes it."
                )
            if said:
                summary += f" It said: “{quotable(said, _QUOTED_CHARS)}”"
            kind, data = "error", {"kind": entry.kind, "exit_code": code, "said": said}
        try:
            Project.open(entry.root).emit(kind, summary, phase=phase or "", data=data)
        except (ProjectError, OSError) as exc:
            log.warning("Could not record the end of the worker in %s: %s", entry.root, exc)

    def _start_what_waits(
        self, root: Path, promised: Collection[str], ended: int | None = None
    ) -> None:
        """Start a worker for the experiments that wait in line, after a worker ended
        while the person had asked for a run.  On a thread of its own: looking at the
        line reads pages and code, and starting waits for what the worker left to be
        settled.  With nothing waiting (the worker that ended ran it, or the person
        took the approval back) nothing is started.  The worker starts with the first
        of ``promised`` that still waits, and goes on with everything else in line.
        ``ended`` is the pid of the worker that ended: word that it was stopped may
        have come a moment after its end was seen, and then nothing is started."""
        try:
            waiting = [str(name) for name in self._waiting_runs(root)]
            asked = [name for name in waiting if name in promised]
            if asked and worker_stop_recorded(root, ended):
                log.info("The runs waiting in %s were not started: the worker was stopped.", root)
            elif asked:
                self.run_in_line(root, asked[0])
        except WorkerBusy as exc:
            # Somebody started a worker first: the run phase of an advance runs what is
            # approved, and so does a worker that runs experiments.
            log.info("The runs waiting in %s were not started: %s", root, exc)
        except Exception:
            # The approvals stand: Continue, or Run now, runs them.
            log.exception("Could not start the runs waiting in %s.", root)

    def _settle_later(self, root: Path) -> None:
        """Settle, on a thread of its own, the runs the project's dead worker left at
        ``running``.  Not under the lock of the books: recovery reads pages, writes
        records and may end what is left of a job, and no look at another project
        should wait for that.  A start of the same project does (:meth:`settled`)."""
        key = self._key(root)

        def settle() -> None:
            try:
                self._settle_runs(root)
            except Exception:
                # The project stays as it was: the next worker's own recovery, or the
                # person's Run, settles what this could not.
                log.exception("Could not settle the runs a worker left behind in %s.", root)
            finally:
                with self._guard:
                    if self._settling.get(key) is threading.current_thread():
                        del self._settling[key]

        with self._guard:
            earlier = self._settling.get(key)
            if earlier is not None and earlier.is_alive():
                return  # one look at the project's runs is under way
            thread = threading.Thread(target=settle, name="settle-runs", daemon=True)
            self._settling[key] = thread
            thread.start()

    # -- describing --------------------------------------------------------------

    def _idle(self, root: Path) -> dict[str, Any]:
        """Nothing runs here.  A ``state.json`` that still says so is corrected: a phase
        left at running is recorded as interrupted, the name of a worker that is gone
        is taken off, and the runs it left at running are settled."""
        key = self._key(root)
        state = read_state(root)
        running = state.get("status") == "running"
        named = state.get("worker") is not None
        # A phase that reads as running without a name: unless somebody holds the
        # project (an engine between taking the lock and writing its name), it is over.
        if (running or named) and not (running and not named and _lock_held(root)):
            if running:
                reason = "The worker of this phase is gone. Continuing resumes the phase."
                phase = mark_interrupted(root, reason, by_person=False)
                self._last_exit[key] = {
                    "kind": "advance",
                    "code": None,
                    "at": _now(),
                    "stopped": False,
                    "interrupted_phase": phase,
                    "reason": None,
                }
            else:
                _take_name_off(root, None)
            self._settle_later(root)
        return self._nothing(root)

    def _nothing(self, root: Path) -> dict[str, Any]:
        return {
            "state": "idle",
            "kind": None,
            "phase": None,
            "pid": None,
            "started": None,
            "adopted": False,
            "host": None,
            "log": WORKER_LOG,
            "verified": None,
            "note": None,
            "last_exit": self._last_exit.get(self._key(root)),
            "queue": False,
            "then_runs": [],
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
            "verified": entry.unverified is None,
            "note": entry.unverified,
            "last_exit": self._last_exit.get(self._key(entry.root)),
            "queue": entry.kind == "run" and bool(entry.options.get("queue")),
            "then_runs": sorted(self._promised.get(self._key(entry.root), ())),
        }


def _remove_record(root: Path) -> None:
    with contextlib.suppress(OSError):
        (root / RECORD_FILE).unlink()


def _unverified_note(pid: int | None, *, held: bool) -> str:
    """Why a worker the files name is shown without a Stop, and what the person can do."""
    who = f"process {pid}" if pid is not None else "a process"
    sign = (
        "Something holds the project's worker lock, so work is going on here"
        if held
        else "It showed a sign of life a moment ago"
    )
    return (
        f"The project's files name {who} as its worker, but this server cannot show that "
        f"it is this project's worker, so it will not signal it. {sign}: if it runs in a "
        "terminal, stop it there. If nothing is running, forget it here and the project "
        "can be continued."
    )


def _death_reason(kind: str, code: int | None) -> str:
    if code is None:
        how = "is gone"
    elif code < 0:
        how = f"was killed (signal {-code})"
    else:
        how = f"ended with exit code {code}"
    return f"The {kind} worker {how} without finishing. See {WORKER_LOG} in the project directory."


def _last_words(root: Path, offset: int | None) -> str:
    """What a worker said last, from what it wrote to ``worker.log`` from ``offset`` on:
    the refusal of the command it ran (the last ``Error: ...`` it printed, which may be
    folded over several lines), else its last line (the error a traceback ends in).
    On one line and cut to what an event keeps; empty when it wrote nothing, when the
    log is not the project's own file, or when ``offset`` is not known.
    """
    if offset is None:
        return ""
    try:
        with open_confined(root, WORKER_LOG) as handle:
            size = os.fstat(handle.fileno()).st_size
            handle.seek(max(offset, size - _LAST_WORDS_BYTES))
            written = handle.read(_LAST_WORDS_BYTES).decode("utf-8", errors="replace")
    except (OSError, ValueError):  # PathEscape is a ValueError
        return ""
    lines = [line.strip() for line in written.splitlines() if line.strip()]
    refusals = [number for number, line in enumerate(lines) if line.startswith("Error:")]
    if refusals:
        words = " ".join(lines[refusals[-1] :])[len("Error:") :]
    else:
        words = lines[-1] if lines else ""
    words = " ".join(words.split())
    return words if len(words) <= _LAST_WORDS_CHARS else words[: _LAST_WORDS_CHARS - 1] + "…"


def _write_up_ended(code: int | None, said: str) -> str:
    """One sentence for a write-up whose worker ended without an account of its own."""
    if code is not None and code < 0:
        return (
            "The write-up ended before the paper was finished: its worker was killed "
            f"(signal {-code}). Starting it again continues it. See {WORKER_LOG} in the "
            "project directory."
        )
    start = f"The write-up did not start: its worker ended with exit code {code}"
    if said:
        # Its words stand as a quotation: they are the command's, not the server's.
        return f"{start} and said: “{quotable(said, _QUOTED_CHARS)}”"
    return f"{start} without saying why. See {WORKER_LOG} in the project directory."


def _paper_data(code: int | None, said: str) -> dict[str, Any]:
    """The data of the ``paper.finished`` event the server writes for such a write-up, in
    the shape of the write-up's own (``success``, ``interrupted``, ``error``), with the
    exit code.  ``started`` is false where the worker ended by itself without having
    written any event: it was refused before it began."""
    data: dict[str, Any] = {"success": False, "interrupted": False}
    if code is not None and code > 0:
        data["started"] = False
    return {**data, "error": said or None, "exit_code": code}
