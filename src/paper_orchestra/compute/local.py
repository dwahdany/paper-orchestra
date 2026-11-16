"""Run jobs as subprocesses on this machine.

This backend needs TeX Live on ``PATH`` for LaTeX.  Every kind of job gets a
minimal environment without credentials, its own process group, a timeout,
and bounded output capture; model-written Python and experiment jobs
additionally run in a throw-away working directory under resource limits, and
TeX may only read and write inside its working directory.  That contains
accidents, but it is **not** a security sandbox: the code still runs as your
user with network access.  Use the Modal backend for isolation.

An experiment job runs in a copy of its source directory with the packages
that are installed here.  It has no GPU, and its network access cannot be
blocked; ``supports`` and the notes of a result say so.

What a job starts is ended with it: its process group, its session and, on
Linux, every process that still carries the job's mark in its environment
(``JOB_ENV``), wherever it went, with whatever shares a session with one.  A
process that left the session *and* dropped its environment cannot be found;
that is the limit of what is tracked here.

``job.json`` in the run directory records the job's working directory as soon as
it exists, so that :meth:`LocalBackend.end_job` can clean up after a worker that
was killed and could not.  That record is a file of a project directory, and a
project may have come from somebody else, so nothing is done on its word: it
only says where to look.  A process is the job's by the mark it carries itself,
a directory by being one that this backend makes, in the place where it makes
them, and both only once nobody works there any more: the backend holds the
kernel's lock on a working directory for as long as it uses it, and a job whose
processes descend from a living ``paper-orchestra research`` process is that
worker's.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import importlib.metadata
import importlib.util
import itertools
import math
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, NamedTuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from paper_orchestra.compute.base import (
    JOB_NAME,
    BackendUnusable,
    CompileResult,
    ComputeBackend,
    ComputeError,
    ExecResult,
    JobLog,
    JobResult,
    JobSpec,
    JobStopped,
    SourceReport,
    choose_outputs,
    clear_outputs,
    leftover_notes,
    open_run,
    refuse_unsafe_document,
    requirement_name,
    safe_relative,
    scan_outputs,
    spec_problems,
    supervise,
    walk_source,
    write_record,
)
from paper_orchestra.utils.latex import latex_commands, summarize_build

_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TZ")
#: The variable every process of a job inherits, set to the job's working directory
#: (a name nobody else has).  It is the mark by which what a job started is found once
#: it has left the job's process group and session.
JOB_ENV = "PAPER_ORCHESTRA_JOB"
#: How the working directories of jobs are named, wherever they are made.
SCRATCH_PREFIX = "paper-orchestra-job-"
#: How often in a row the processes of a job are looked up and killed: one that forks
#: while the others are killed is found the next time.
_SWEEPS = 3
#: Seconds processes that were killed are waited for before their directory is removed.
_REAP_SECONDS = 3.0
#: Seconds the lock on a working directory is tried for when somebody holds it.  Whoever
#: merely looks at a directory holds its lock for an instant; a backend that works in it
#: holds it for as long as its job takes.
_BUSY_SECONDS = 1.0
#: The kill signal in the masks of pending signals that /proc shows.
_KILL_PENDING = 1 << (signal.SIGKILL - 1)
#: Why a path cannot be opened as a directory when there is no directory by that name:
#: nothing there, something else there, or a link.
_NO_DIRECTORY = (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
#: Characters of stdout/stderr kept per stream.  Only the tail is kept, so a job
#: that prints in a loop cannot exhaust the orchestrator's memory.
_MAX_CAPTURE = 200_000


def scrubbed_env(home: Path) -> dict[str, str]:
    """A minimal environment with no credentials for untrusted children."""
    env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ}
    env.setdefault("PATH", os.defpath)
    env.update(
        HOME=str(home),
        TMPDIR=str(home),
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(home / ".mplconfig"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
    )
    return env


def latex_env(home: Path) -> dict[str, str]:
    """Environment for TeX: no credentials, no inherited TeX search paths, a private
    home directory, and file access restricted to the working directory tree
    (``openin_any`` / ``openout_any`` = paranoid), so a document cannot ``\\input``
    files such as ``/proc/self/environ`` or anything given by absolute or parent path.
    Those limits do not cover every pdfTeX primitive, which is why documents are also
    screened by :func:`refuse_unsafe_document` before they are compiled."""
    env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ}
    env.setdefault("PATH", os.defpath)
    env.update(HOME=str(home), openout_any="p", openin_any="p")
    return env


_PR_SET_PDEATHSIG = 1


def _prctl() -> Any:
    """libc's ``prctl`` on Linux, else ``None``.  Looked up before a fork, never after."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        return ctypes.CDLL(None).prctl
    except (ImportError, OSError, AttributeError):  # pragma: no cover - unusual libc
        return None


def _program_exists(program: str, env: Mapping[str, str], cwd: Path) -> bool:
    """Whether there is a file the command's first word names, as far as can be seen.

    The system says "permission denied" for a program that is nowhere when the search
    along PATH met a directory it may not read.  A shell looks for itself and says
    "not found"; so does this.
    """
    if os.sep in program:
        return (cwd / program).exists()
    folders = env.get("PATH", os.defpath).split(os.pathsep)
    return any(os.path.exists(os.path.join(folder or ".", program)) for folder in folders)


def _limit_resources(cpu_seconds: int, memory_bytes: int | None, *, die_with_parent: bool = False):
    """What a child does before it becomes the job: take its resource limits.

    ``die_with_parent`` additionally has the kernel kill the child when the thread
    that started it ends (Linux only), so that a job does not run on after an
    orchestrator that was killed and could not stop it.
    """
    # Imported here, before the fork: an import in the child can wait forever for a lock
    # that another thread of the orchestrator held at that moment.
    import resource

    prctl = _prctl() if die_with_parent else None

    def apply() -> None:  # pragma: no cover - runs in the child
        # A limit above what this process is allowed itself cannot be set; the job then
        # runs under the limit it inherits.
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        if memory_bytes:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if prctl is not None:
            prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)

    return apply


class _Tail:
    """Keeps the last ``limit`` bytes written to it."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: deque[bytes] = deque()
        self._size = 0

    def pump(self, stream: IO[bytes]) -> None:
        with contextlib.suppress(OSError, ValueError):
            for chunk in iter(lambda: stream.read(65536), b""):
                self._chunks.append(chunk)
                self._size += len(chunk)
                while self._size - len(self._chunks[0]) >= self._limit:
                    self._size -= len(self._chunks.popleft())

    def text(self) -> str:
        data = b"".join(self._chunks)[-self._limit :]
        return data.decode("utf-8", errors="replace")


def _kill_group(process: subprocess.Popen[bytes], sig: int = signal.SIGKILL) -> None:
    """Send ``sig`` to the process group of ``process``, which it leads."""
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
    else:  # pragma: no cover - non-posix
        with contextlib.suppress(OSError):
            process.kill()


def _on_linux() -> bool:
    return sys.platform.startswith("linux")


def _stat_fields(pid: int | str) -> list[bytes] | None:
    """The fields of ``/proc/<pid>/stat`` from the state on; ``None`` if it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as stat_file:
            # "pid (name) state ppid pgrp session ...": the name may hold anything.
            fields = stat_file.read().rsplit(b")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return fields if len(fields) > 19 else None


class _Process(NamedTuple):
    """What the process table says of one process."""

    parent: int
    session: int


def _table() -> dict[int, _Process]:
    """The processes of this machine that are still alive, by their id (read from
    /proc: empty where there is none)."""
    try:
        entries = [entry.name for entry in os.scandir("/proc") if entry.name.isdigit()]
    except OSError:
        return {}
    table: dict[int, _Process] = {}
    for name in entries:
        fields = _stat_fields(name)
        # A zombie only waits to be collected: there is nothing left of it to end.
        if fields is None or fields[0] == b"Z":
            continue
        with contextlib.suppress(ValueError):
            table[int(name)] = _Process(parent=int(fields[1]), session=int(fields[3]))
    return table


def _marks(pids: Iterable[int]) -> Iterator[tuple[int, bytes]]:
    """``(pid, mark)`` for those of ``pids`` that carry the mark of a job in their own
    environment, with the mark as it stands there.

    A process inherits its environment whatever group or session it moves to; only
    one that replaced its environment is not found.  Environments of other users'
    processes cannot be read and are not a job's.
    """
    variable = os.fsencode(f"{JOB_ENV}=")  # as the environment itself was encoded
    for pid in pids:
        try:
            with open(f"/proc/{pid}/environ", "rb") as environ:
                entries = environ.read().split(b"\0")
        except OSError:
            continue  # not ours to read, or gone
        for entry in entries:
            if entry.startswith(variable):
                yield pid, entry[len(variable) :]


def _marked(mark: str, pids: Iterable[int]) -> set[int]:
    """Those of ``pids`` whose own environment says they belong to the job ``mark``,
    letter for letter: for the backend that set the mark itself."""
    wanted = os.fsencode(mark)
    found = {pid for pid, value in _marks(pids) if value == wanted}
    found.discard(os.getpid())
    return found


def _names(mark: bytes, scratch: str) -> bool:
    """Whether the mark of a process names the working directory ``scratch``, which
    exists: the same path, or another path to the same directory that ends in the same
    name (a temporary directory has two spellings where it is reached through a link)."""
    try:
        path = os.fsdecode(mark)
        if path == scratch:
            return True
        elsewhere = os.path.isabs(path) and os.path.basename(path) == os.path.basename(scratch)
        return elsewhere and os.path.samefile(path, scratch)
    except (OSError, ValueError):  # it leads nowhere, or is no name a file can have
        return False


def _marked_for(scratch: str, pids: Iterable[int]) -> set[int]:
    """Those of ``pids`` whose mark names the working directory ``scratch``, however it
    is spelled.  For whoever cleans up by a record: the record's spelling of the
    directory is a string in a file, and need not be the one its job was given."""
    return {pid for pid, value in _marks(pids) if _names(value, scratch)}


def _with_their_sessions(marked: set[int], table: Mapping[int, _Process]) -> set[int]:
    """``marked`` and every process of ``table`` that shares a session with one of them.

    A process cannot join a session, only start one.  So whatever is in a session with
    a process of the job was started by the job, whether or not it still carries the
    mark: the job's own session, and any session something of the job went on to start.
    """
    sessions = {table[pid].session for pid in marked if pid in table} - {0, 1}
    return marked | {pid for pid, process in table.items() if process.session in sessions}


def _being_killed(pid: int) -> bool:
    """Whether the kill signal is already pending on the process ``pid``.

    That signal cannot be held back: such a process is ending, and nothing is left of
    it to end.  The command of a job is in this state for a moment after its worker
    died (the kernel sends it the signal), and so is whatever was just killed here.
    ``False`` for a process that is gone, or where /proc does not say.
    """
    try:
        with open(f"/proc/{pid}/status", "rb") as status:
            lines = status.read().splitlines()
    except OSError:
        return False
    for line in lines:
        name, _, pending = line.partition(b":")
        if name in (b"SigPnd", b"ShdPnd"):
            with contextlib.suppress(ValueError):
                if int(pending, 16) & _KILL_PENDING:
                    return True
    return False


def _kill(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return False  # it ended while we were looking
    return True


def _session_started(pid: int) -> str:
    """When the process ``pid`` was started, as this machine counts it since it booted,
    with the boot it was counted in; ``""`` where that cannot be told (not Linux, or no
    such process).  Two processes that were given the same id differ in this."""
    fields = _stat_fields(pid) if _on_linux() else None
    if fields is None:
        return ""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        boot = ""
    return f"{fields[19].decode('ascii', 'replace')}@{boot}"


def _kill_job(process: subprocess.Popen[bytes], mark: str = "") -> set[int]:
    """Kill the job ``process`` leads: its process group, what left the group and what
    left the session.  Returns the processes that were found outside the session.

    Tools such as coreutils ``timeout`` and shells with job control move what they
    start to a process group of its own, which a signal to the job's group does not
    reach.  The session is still the job's, so on Linux the processes of the session
    are looked up and killed too.  What started a session of its own (a daemon) is
    found by the job's ``mark`` in its environment, see :func:`_marked`, and with it
    what shares its session.  Call this before the process is reaped: until then its
    id cannot have been given to an unrelated process group or session.
    """
    _kill_group(process)
    strays: set[int] = set()
    if not _on_linux():
        return strays
    for _ in range(_SWEEPS):
        table = _table()
        found = _with_their_sessions(_marked(mark, table), table) if mark else set()
        killed = False
        for pid, entry in table.items():
            inside = entry.session == process.pid
            # Also what is dying already: while something of the job is there, something
            # may have been started that the next look finds.
            if (inside or pid in found) and pid != os.getpid() and _kill(pid):
                killed = True
                if not inside:
                    strays.add(pid)
        if not killed:
            break
    return strays


# -- cleaning up after a worker that died ------------------------------------------------
#
# The record of a job (job.json, or its copy in run.json) is a file in a project
# directory.  Nothing below acts on its word: it names a path, and what is done with
# that path is decided by what this machine shows.


def _is_command(argument: str) -> bool:
    """Whether a command-line argument is this package's command, as a script."""
    name = argument.replace("\\", "/")
    if os.path.basename(name).lower() in (
        "paper-orchestra",
        "paper-orchestra.exe",
        "paper-orchestra-script.py",
    ):
        return True
    return name.endswith("paper_orchestra/__main__.py")


def _reads_as_worker(command: Sequence[str]) -> bool:
    """Whether ``command`` is the command line of a worker of this package:
    ``... -m paper_orchestra research ...``, or the ``paper-orchestra`` command with
    ``research``.  Those are the processes that run the experiments of a project."""
    for index in range(1, len(command)):
        if command[index] != "research":
            continue
        before = command[:index]
        as_module = any(
            option == "-m" and module.split(".")[0] == "paper_orchestra"
            for option, module in itertools.pairwise(before)
        )
        if as_module or any(_is_command(argument) for argument in before):
            return True
    return False


def _is_worker(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as cmdline:
            parts = cmdline.read().split(b"\0")
    except OSError:
        return False
    return _reads_as_worker([part.decode("utf-8", "surrogateescape") for part in parts])


def _lineage(pid: int, table: Mapping[int, _Process]) -> list[int]:
    """The living processes ``pid`` descends from, nearest first."""
    found: list[int] = []
    while pid in table and table[pid].parent not in (0, pid, *found):
        pid = table[pid].parent
        if pid in table:
            found.append(pid)
    return found


def _living_worker(pids: Iterable[int], table: Mapping[int, _Process]) -> int:
    """The id of a living worker of this package that one of ``pids`` descends from,
    or ``0``.  The caller is not looked at, nor what it descends from itself: a job
    below the caller is the caller's own to end."""
    me = os.getpid()
    judged: dict[int, bool] = {}
    for pid in pids:
        for ancestor in _lineage(pid, table):
            if ancestor == me:
                break
            if ancestor not in judged:
                judged[ancestor] = _is_worker(ancestor)
            if judged[ancestor]:
                return ancestor
    return 0


class _Found(NamedTuple):
    """What is running of a job, as its mark shows it."""

    #: The processes of the job that would have to be ended.
    processes: set[int]
    #: Those that are being killed already and only have to be waited for.
    dying: set[int]
    #: A living worker that some of them descend from (``0``: none): its job, not ours.
    worker: int


def _find(scratch: str) -> _Found:
    """The processes of the job whose working directory is ``scratch`` (it exists):
    those whose mark names that directory, by whatever spelling, and what shares a
    session with one.  Never the caller or a process it descends from.  Linux only;
    nothing elsewhere."""
    if not _on_linux():
        return _Found(set(), set(), 0)
    table = _table()
    mine = {os.getpid(), *_lineage(os.getpid(), table)}
    found = _with_their_sessions(_marked_for(scratch, table), table) - mine
    dying = {pid for pid in found if _being_killed(pid)}
    return _Found(found - dying, dying, _living_worker(found, table))


class DirectoryInUse(Exception):
    """Somebody holds the lock on a working directory: a backend is at work in it."""


def is_own_directory(path: str) -> bool:
    """Whether ``path`` is a real directory (not a link to one) that this user owns."""
    try:
        status = os.lstat(path)
    except (OSError, ValueError):  # no such thing, or no name a file can have
        return False
    mine = not hasattr(os, "geteuid") or status.st_uid == os.geteuid()
    return stat.S_ISDIR(status.st_mode) and mine


def lock_directory(path: str, *, shared: bool = False, patience: float = 0.0) -> int | None:
    """Take the kernel's lock on the directory ``path`` and return the descriptor that
    holds it; closing the descriptor gives the lock back, and the kernel does that for
    a process that dies, however it dies.

    A backend holds this lock on the working directory of a job for as long as it uses
    it.  That is what tells a directory nobody works in any more from one that is in
    use: a file cannot say it, and a lock cannot be planted.  ``shared`` is for a mere
    look, so that two who look at once do not see each other.  Raises :class:`DirectoryInUse`
    when somebody else still holds it after ``patience`` seconds.  ``None`` where there
    is nothing to hold or no way of holding it: no such directory (any more), or no
    such locks on this system or file system.  A directory that is there and cannot be
    opened raises the :class:`OSError`: whether somebody holds it is then not known,
    which is not the same as nobody holding it.
    """
    if fcntl is None:  # pragma: no cover - Windows
        return None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in _NO_DIRECTORY:
            return None
        raise
    deadline = time.monotonic() + patience
    while True:
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            return descriptor
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(descriptor)
                raise DirectoryInUse(path) from None
            time.sleep(0.02)
        except OSError:
            os.close(descriptor)
            return None


@contextlib.contextmanager
def holding(directory: str) -> Iterator[None]:
    """Hold the lock on a job's working directory while a backend works in it (see
    :func:`lock_directory`)."""
    try:
        descriptor = lock_directory(directory, patience=_BUSY_SECONDS)
    except (DirectoryInUse, OSError):
        # Somebody keeps looking at it, or it cannot be opened: the job runs without the
        # lock, as it does where there are no such locks.
        descriptor = None
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _shown(path: str, limit: int = 300) -> str:
    """A path out of a record, for a note: on one line and not endless."""
    text = "".join(char if char.isprintable() else "?" for char in path)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _made_by_backend(scratch: str) -> bool:
    """Whether ``scratch`` is, by its name and by what is there, a working directory as
    this backend makes them: an absolute path spelled as the backend spells it, with
    the backend's prefix, to a real directory (not a link to one) of this user's."""
    plain = bool(scratch) and "\0" not in scratch and os.path.isabs(scratch)
    if not plain or os.path.normpath(scratch) != scratch:
        return False
    return os.path.basename(scratch).startswith(SCRATCH_PREFIX) and is_own_directory(scratch)


def _scratch_roots(scratch_dir: Path | None) -> list[str]:
    """Where working directories are made: the directory a backend was told to use,
    and the system's temporary directory (as they really are, links resolved)."""
    places = [str(scratch_dir)] if scratch_dir is not None else []
    with contextlib.suppress(OSError):
        places.append(tempfile.gettempdir())
    return list(dict.fromkeys(os.path.realpath(place) for place in places))


def _held(scratch: str, *, to_end: bool) -> int | None:
    """The lock on the working directory ``scratch`` for whoever examines it, as
    :func:`lock_directory` gives it: shared for a look, and tried for a moment when
    the directory is to be cleaned up."""
    patience = _BUSY_SECONDS if to_end else 0.0
    try:
        return lock_directory(scratch, shared=not to_end, patience=patience)
    except PermissionError:
        # Its job took the access to it away.  It is the user's own directory (the caller
        # saw to that), and gets the access again that it was made with.
        os.chmod(scratch, 0o700)
        return lock_directory(scratch, shared=not to_end, patience=patience)


class _Leftover:
    """What a job record names, as far as this machine bears it out."""

    def __init__(self) -> None:
        #: The job's working directory, when it is this backend's and nobody's any more.
        self.scratch = ""
        #: What still runs of the job and would be ended, and what is ending already.
        self.processes: set[int] = set()
        self.dying: set[int] = set()
        #: Why nothing is to be ended or removed although the directory exists.
        self.reason = ""


@contextlib.contextmanager
def _examined(record: Any, roots: Sequence[str], *, to_end: bool) -> Iterator[_Leftover]:
    """Look at what the job record ``record`` names.  The lock on the working directory
    is held while the caller works with the answer (shared, unless ``to_end``)."""
    scratch = _job_fields(record)[2]
    left = _Leftover()
    if not _made_by_backend(scratch):
        yield left  # nothing of this backend's is named, or nothing is left of it
        return
    named = f"The run's record names the working directory {_shown(scratch)}."
    if os.path.realpath(os.path.dirname(scratch)) not in roots:
        left.reason = (
            f"{named} The local backend makes its working directories directly in "
            f"{' or '.join(roots) or 'the temporary directory'}, and cleans up only there. "
            "Nothing was stopped or removed. If that directory is what the run left behind "
            "(a run that was made with another TMPDIR or scratch directory), remove it yourself."
        )
        yield left
        return
    try:
        descriptor = _held(scratch, to_end=to_end)
    except DirectoryInUse:
        left.reason = (
            f"{named} A job is running in it now (its backend holds the directory), so it "
            "is not what this run left behind. Nothing was stopped or removed."
        )
        yield left
        return
    except OSError as exc:
        left.reason = (
            f"{named} Whether a job is running in it could not be found out: the directory "
            f"could not be opened ({exc.strerror or type(exc).__name__}). Nothing was stopped "
            "or removed. If that directory is what the run left behind, remove it yourself."
        )
        yield left
        return
    try:
        found = _find(scratch)
        if found.worker:
            left.reason = (
                f"{named} The job in it is running under another worker (process "
                f"{found.worker}), so it is not what this run left behind. Nothing was "
                "stopped or removed."
            )
        else:
            left.scratch, left.processes, left.dying = scratch, found.processes, found.dying
        yield left
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _end_processes(left: _Leftover) -> int:
    """Kill what still runs of the job ``left`` describes, wait for it to be gone, and
    return how many processes were killed.  Looked up more than once: what forks while
    the others are killed is found the next time."""
    killed: set[int] = set()
    waiting = set(left.dying)
    targets = left.processes
    for _ in range(_SWEEPS):
        hit = {pid for pid in targets if _kill(pid)}
        if not hit:
            break
        killed |= hit
        found = _find(left.scratch)
        waiting |= found.dying
        targets = found.processes
    _wait_gone(killed | waiting, _REAP_SECONDS)
    return len(killed)


def _job_record(scratch: str, session: int = 0) -> dict[str, Any]:
    """What ``job.json`` holds of a job that works in ``scratch``: the session its
    command leads and when that was started, or ``0`` and ``""`` while the command has
    not been started and the source is still being copied."""
    started = _session_started(session) if session else ""
    return {"session": session, "session_started": started, "scratch": scratch}


def _job_fields(record: Any) -> tuple[int, str, str]:
    """``(session, session_started, scratch)`` of a job record, or of a record that holds
    one under ``"job"``; ``0`` and ``""`` for what is missing or of another type."""
    job = record.get("job") if isinstance(record, Mapping) else None
    job = job if isinstance(job, Mapping) else record if isinstance(record, Mapping) else {}
    session, started, scratch = (job.get(key) for key in ("session", "session_started", "scratch"))
    return (
        session if isinstance(session, int) and not isinstance(session, bool) else 0,
        started if isinstance(started, str) else "",
        scratch if isinstance(scratch, str) else "",
    )


def _wait_gone(pids: Iterable[int], seconds: float) -> None:
    """Give killed processes a moment to end, so that they hold no file any more."""
    deadline = time.monotonic() + seconds
    waiting = set(pids)
    while waiting and time.monotonic() < deadline:
        waiting &= _table().keys()
        if waiting:
            time.sleep(0.05)


def remove_tree(path: Path) -> bool:
    """Remove the directory ``path`` and everything in it, also what a job made
    unreadable; returns whether it is gone.  Links are removed, never followed."""
    stack = [str(path)]
    while stack:
        directory = stack.pop()
        with contextlib.suppress(OSError):
            os.chmod(directory, 0o700)
            with os.scandir(directory) as entries:
                stack.extend(e.path for e in entries if e.is_dir(follow_symlinks=False))
    shutil.rmtree(path, ignore_errors=True)
    return not os.path.lexists(path)


def _spawn(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    preexec: Any = None,
    merge_stderr: bool = False,
) -> subprocess.Popen[bytes]:
    """Start ``command`` without stdin, in a session and process group of its own, so
    that it and everything it starts can be signalled together."""
    posix = os.name == "posix"
    return subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        start_new_session=posix,
        preexec_fn=preexec if posix else None,
    )


def _has_exited(process: subprocess.Popen[bytes]) -> bool:
    """Whether the child has ended, without reaping it.

    Until it is reaped its process group id cannot be handed to another process, so
    the rest of the group can still be killed without hitting a stranger.
    """
    if not hasattr(os, "waitid"):  # pragma: no cover - Windows, macOS before Python 3.13
        return process.poll() is not None
    try:
        return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


def _pipe_chunks(stream: IO[bytes]) -> Iterator[bytes]:
    """The output of a child as it arrives, for :func:`supervise`.

    Yields an empty chunk whenever nothing arrived for a moment, which is what lets
    the reading thread end when a process that escaped its group keeps the pipe open.
    """
    fd = stream.fileno()
    if os.name != "posix":  # pragma: no cover - pipes cannot be polled there
        yield from iter(lambda: os.read(fd, 65536), b"")
        return
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            if not selector.select(0.2):
                yield b""
                continue
            data = os.read(fd, 65536)
            if not data:
                return
            yield data


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    preexec: Any = None,
) -> tuple[int, str, str, bool]:
    """Run ``command`` in its own process group and return its bounded output.

    The whole group is killed on timeout and also when this call is interrupted
    (for example by Ctrl-C), so no compile or script outlives the pipeline.
    """
    process = _spawn(command, cwd=cwd, env=env, preexec=preexec)
    out, err = _Tail(_MAX_CAPTURE), _Tail(_MAX_CAPTURE)
    readers = [
        threading.Thread(target=out.pump, args=(process.stdout,), daemon=True),
        threading.Thread(target=err.pump, args=(process.stderr,), daemon=True),
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(process)
        process.wait()
    except BaseException:
        _kill_group(process)
        process.wait()
        raise
    finally:
        # A grandchild that escaped the group could keep a pipe open; do not wait forever.
        for reader in readers:
            reader.join(timeout=5)
    return process.returncode, out.text(), err.text(), timed_out


def copy_source(
    source: Path, dest: Path, should_stop: Callable[[], bool] | None = None
) -> list[str]:
    """Copy ``source`` into the new directory ``dest`` for a job to work in.

    What goes in is decided by :func:`walk_source`; the notes say what was left out.
    Files are copied, never linked: whatever the job does to its working directory,
    the source stays as it was.  Raises :class:`JobStopped` when ``should_stop``
    turns true on the way.
    """
    report = SourceReport()
    dest.mkdir(parents=True)
    for relative, path, kind in walk_source(source, report, should_stop):
        target = dest / relative
        if kind == "dir":
            target.mkdir()
        elif kind == "link":
            os.symlink(os.readlink(path), target)
        else:
            shutil.copy2(path, target, follow_symlinks=False)
    return report.notes()


def collect_outputs(work: Path, outputs: Sequence[str], run_dir: Path) -> list[str]:
    """Copy the requested outputs that exist from ``work`` to ``run_dir``.

    Returns notes about what was not copied (see :func:`choose_outputs`).
    """
    # More than can be copied is never listed: a job may have left millions of files.
    listing = scan_outputs(str(work), list(outputs), max_entries=100_000)
    chosen, notes = choose_outputs(listing["entries"], truncated=listing["truncated"])
    for relative, _ in chosen:
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / relative, target, follow_symlinks=False)
    return notes


#: Import names of well-known distributions whose name does not give them away.  Only
#: needed for what is not installed: an installed distribution lists its own modules.
_IMPORT_NAMES = {
    "attrs": "attr",
    "beautifulsoup4": "bs4",
    "faiss-cpu": "faiss",
    "faiss-gpu": "faiss",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "protobuf": "google.protobuf",
    "python-dateutil": "dateutil",
    "pyyaml": "yaml",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
}


def _canonical(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def missing_packages(requirements: Iterable[str]) -> list[str]:
    """The requirements whose distribution cannot be imported in this interpreter.

    The modules of a distribution are taken from its metadata when it is installed
    and guessed from its name otherwise.  Versions are not compared.  Strings that
    are not plain requirements are skipped: ``spec_problems`` reports those.
    """
    requirements = list(requirements)
    if not requirements:
        return []  # most jobs: reading every installed distribution would be wasted
    modules: dict[str, list[str]] = {}
    for module, distributions in importlib.metadata.packages_distributions().items():
        for distribution in distributions:
            modules.setdefault(_canonical(distribution), []).append(module)
    missing: list[str] = []
    for requirement in requirements:
        name = requirement_name(requirement)
        if name is None:
            continue
        key = _canonical(name)
        candidates = modules.get(key) or [_IMPORT_NAMES.get(key, key.replace("-", "_"))]
        if not any(_importable(module) for module in candidates):
            missing.append(requirement.strip())
    return missing


class _on_class_or_instance:
    """Makes a method one that can be called on the class as well as on an instance:
    the function is given the instance it was called on, or ``None``."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function
        functools.update_wrapper(self, function)

    def __get__(self, instance: Any, owner: type | None = None) -> Callable[..., Any]:
        return functools.partial(self._function, instance)


class LocalBackend(ComputeBackend):
    name = "local"

    def __init__(
        self,
        *,
        python: str | None = None,
        memory_limit_mb: int | None = 4096,
        scratch_dir: str | Path | None = None,
    ) -> None:
        """``python`` runs figure code and stands for ``python`` in a job's command;
        ``memory_limit_mb`` caps the address space of figure code (``None``: no cap; a
        job is capped only by its own ``JobSpec.memory_mb``); ``scratch_dir`` is where the working
        directories of jobs are made and removed (default: the system's temporary
        directory, which on some machines is small or kept in memory)."""
        self.python = python or sys.executable
        self.memory_limit_mb = memory_limit_mb
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else None
        self._tex_home: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.Lock()

    def _private_tex_home(self) -> Path:
        """An empty home directory for TeX, so ``~`` never resolves to the user's files."""
        with self._lock:
            if self._tex_home is None:
                self._tex_home = tempfile.TemporaryDirectory(prefix="paper-orchestra-texhome-")
            return Path(self._tex_home.name)

    def close(self) -> None:
        with self._lock:
            if self._tex_home is not None:
                self._tex_home.cleanup()
                self._tex_home = None

    # -- python ----------------------------------------------------------------

    def run_python(
        self,
        code: str,
        *,
        inputs: Mapping[str, bytes] | None = None,
        outputs: Sequence[str] = (),
        timeout: float = 120.0,
    ) -> ExecResult:
        input_paths = {name: safe_relative(name) for name in (inputs or {})}
        output_paths = {name: safe_relative(name) for name in outputs}
        try:
            with tempfile.TemporaryDirectory(prefix="paper-orchestra-exec-") as tmp:
                workdir = Path(tmp)
                for name, relative in input_paths.items():
                    target = workdir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((inputs or {})[name])
                script = workdir / "_script.py"
                script.write_text(code, encoding="utf-8")

                preexec = None
                if os.name == "posix":
                    memory = self.memory_limit_mb * 1024 * 1024 if self.memory_limit_mb else None
                    preexec = _limit_resources(int(timeout) + 5, memory)
                # -I: isolated mode, so PYTHON* variables and the user site are ignored.
                returncode, out, err, timed_out = _run(
                    [self.python, "-I", "-B", script.name],
                    cwd=workdir,
                    env=scrubbed_env(workdir),
                    timeout=timeout,
                    preexec=preexec,
                )
                files = {
                    name: (workdir / relative).read_bytes()
                    for name, relative in output_paths.items()
                    if (workdir / relative).is_file()
                }
        except OSError as exc:
            raise ComputeError(f"could not run the script locally: {exc}") from exc
        return ExecResult(
            returncode=returncode, stdout=out, stderr=err, timed_out=timed_out, files=files
        )

    # -- latex -----------------------------------------------------------------

    @staticmethod
    def latex_available() -> bool:
        return bool(shutil.which("pdflatex") and shutil.which("bibtex"))

    def compile_latex(
        self,
        workdir: str | Path,
        tex_name: str,
        *,
        output_pdf: str | Path | None = None,
        timeout: float = 240.0,
    ) -> CompileResult:
        work = Path(workdir)
        if not (work / f"{tex_name}.tex").is_file():
            raise ComputeError(f"{work / (tex_name + '.tex')} does not exist")
        produced = work / f"{tex_name}.pdf"
        produced.unlink(missing_ok=True)
        # Screened first: a document that must not be compiled is refused the same way
        # whether or not TeX is installed here.
        refused = refuse_unsafe_document(work, tex_name)
        if refused is not None:
            return refused
        if not self.latex_available():
            raise BackendUnusable(
                "pdflatex/bibtex not found on PATH. Install TeX Live, or run with "
                "--compute modal to compile in a Modal sandbox."
            )
        env = latex_env(self._private_tex_home())

        deadline = time.monotonic() + timeout
        steps: list[tuple[str, str]] = []
        transcript: list[str] = []
        timed_out = False
        for command in latex_commands(tex_name):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                _, out, err, step_timed_out = _run(command, cwd=work, env=env, timeout=remaining)
            except OSError as exc:
                raise ComputeError(f"could not run {command[0]}: {exc}") from exc
            steps.append((command[0], out + err))
            transcript.append(f"$ {' '.join(command)}\n\n{out}{err}")
            if step_timed_out:
                timed_out = True
                break

        errors, warnings = summarize_build(steps)
        ok = produced.is_file() and not timed_out
        pdf_path: Path | None = None
        if ok:
            pdf_path = Path(output_pdf) if output_pdf else produced
            if pdf_path != produced:
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(produced), pdf_path)
        return CompileResult(
            ok=ok,
            pdf_path=pdf_path,
            errors=errors,
            warnings=warnings,
            log_tail="\n".join(transcript)[-6000:],
            timed_out=timed_out,
        )

    # -- experiment jobs -------------------------------------------------------

    def supports(self, spec: JobSpec) -> list[str]:
        reasons = spec_problems(spec)
        if spec.gpu:
            reasons.append(
                f"It asks for a GPU ({spec.gpu}), and the local backend runs on this machine "
                "without one. Use the Modal backend for GPU jobs."
            )
        missing = missing_packages(spec.packages)
        if missing:
            reasons.append(
                "These packages are not installed in the Python that runs local jobs: "
                f"{', '.join(missing)}. Install them there, or use the Modal backend, which "
                "installs what a job asks for."
            )
        return reasons

    def run_job(
        self,
        spec: JobSpec,
        *,
        source_dir: Path,
        run_dir: Path,
        on_log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        mask: Sequence[str] = (),
    ) -> JobResult:
        source, run_dir = Path(source_dir), Path(run_dir)
        log_path = open_run(run_dir, spec.outputs)
        result = JobResult(returncode=None, backend=self.name, log_path=log_path)
        result.outputs_dir = run_dir
        with JobLog(log_path, on_log, mask=mask) as log:
            reasons = self.supports(spec)
            if not reasons and not source.is_dir():
                reasons = [f"The source directory {source} does not exist."]
            scratch_root = self.scratch_dir or Path(tempfile.gettempdir())
            if not reasons and scratch_root.resolve().is_relative_to(source.resolve()):
                # The copy would contain itself, and so on until the path is too long.
                reasons = [
                    f"Working directories are made in {scratch_root}, which is inside the "
                    f"source directory {source}. Set scratch_dir (or TMPDIR) to a directory "
                    "outside of it."
                ]
            if reasons:
                result.error = "The local backend cannot run this job. " + " ".join(reasons)
                log.note(result.error)
                return result
            result.notes.extend(self._caveats(spec))
            # The source may be the person's repository: the job only ever sees a copy,
            # which is gone when this returns, however it returns.
            try:
                if self.scratch_dir is not None:
                    self.scratch_dir.mkdir(parents=True, exist_ok=True)
                # Spelled out in full: the path is the job's mark and goes on record, and
                # whoever cleans up after it compares both with what is on this machine.
                where = os.path.abspath(self.scratch_dir) if self.scratch_dir is not None else None
                scratch = tempfile.TemporaryDirectory(
                    prefix=SCRATCH_PREFIX, dir=where, ignore_cleanup_errors=True
                )
            except OSError as exc:
                result.error = f"The local backend could not make a working directory: {exc}."
                log.note(result.error)
                return result
            try:
                # Held until the directory is gone again: nobody who finds its name in a
                # record takes it for a leftover while this backend works in it.
                with holding(scratch.name), scratch as name:
                    work, home = Path(name, "work"), Path(name, "home")
                    home.mkdir()
                    # From here on a worker that is killed leaves this directory behind:
                    # the record says where it is before anything is copied into it.
                    write_record(run_dir / JOB_NAME, _job_record(name))
                    self._note(result, log, copy_source(source, work, should_stop))
                    # A result is what the command writes, not what its source brought.
                    leftovers = clear_outputs(work, spec.outputs)
                    self._note(result, log, leftover_notes(leftovers))
                    self._execute(spec, work, home, run_dir, log, should_stop, result)
                    self._note(result, log, collect_outputs(work, spec.outputs, run_dir))
            except JobStopped:
                result.cancelled = True
                log.note("The job was stopped on request, before it had started.")
            except (OSError, subprocess.SubprocessError) as exc:
                result.error = f"The local backend could not run the job: {exc}."
                log.note(result.error)
            if os.path.lexists(scratch.name):
                self._note(
                    result, log, [f"The working directory {scratch.name} could not be removed."]
                )
            result.notes.extend(log.notes())
        return result

    @staticmethod
    def _note(result: JobResult, log: JobLog, notes: Sequence[str]) -> None:
        for note in notes:
            log.note(note)
        result.notes.extend(notes)

    @staticmethod
    def _caveats(spec: JobSpec) -> list[str]:
        """What a job asks for that this backend cannot give, without that being a reason
        to refuse it: the local backend is documented as not being a sandbox."""
        caveats: list[str] = []
        if not spec.network:
            caveats.append(
                "The job was to run without network access. The local backend cannot block the "
                "network (it is not a sandbox); use the Modal backend where that matters."
            )
        if spec.cpu is not None:
            caveats.append(
                f"The limit of {spec.cpu:g} CPU cores is not applied by the local backend."
            )
        return caveats

    @_on_class_or_instance
    def job_processes(self: LocalBackend | None, record: Any) -> set[int]:
        """The processes that :meth:`end_job` would end for the job record ``record``.

        Empty unless the record names a working directory that this backend makes
        (see :meth:`end_job`), that no backend holds any more, and whose processes
        descend from no living worker.  Linux only; empty elsewhere.  Callable on
        the class, or on a backend that was told where to make its working
        directories.
        """
        roots = _scratch_roots(self.scratch_dir if self is not None else None)
        with _examined(record, roots, to_end=False) as left:
            return set(left.processes)

    @_on_class_or_instance
    def end_job(self: LocalBackend | None, record: Any) -> list[str]:
        """Clean up after a local job whose worker died before it could, and return
        notes on what was done.

        ``record`` is what the job wrote to ``job.json`` in its run directory, or a
        record that holds it under ``"job"``.  It lies in a project directory, where
        anybody may have put it, so only its ``scratch`` is read, as the place to
        look, and nothing is done on its word:

        * The directory must be one this backend makes: named with its prefix,
          directly in the system's temporary directory or in the directory this
          backend was told to use, a real directory and not a link, and the user's
          own.  Otherwise nothing is ended or removed.
        * Nothing is touched while the job may be running: a backend holds the lock
          on the directory (or the directory cannot be opened to see whether one
          does), or processes of the job descend from a living
          ``paper-orchestra research`` process other than the caller.
        * The processes that are ended are those whose mark, in their own
          environment, names the directory (however the record or the mark spell
          the way to it), with what shares a session with one of them (Linux only).
          No session is ended because the record names it.

        Then the directory is removed.  The notes say what was done, or why a
        directory that exists was left alone.  Nothing is done, and nothing returned,
        for a record that names no such directory; calling this twice is safe.  Call
        it on the class, or on a backend that was told where to make its working
        directories (``scratch_dir``), which then knows that place as well.
        """
        roots = _scratch_roots(self.scratch_dir if self is not None else None)
        with _examined(record, roots, to_end=True) as left:
            if not left.scratch:
                return [left.reason] if left.reason else []
            notes: list[str] = []
            if _on_linux():
                killed = _end_processes(left)
                if killed:
                    notes.append(
                        f"Stopped {killed} process{'es' if killed > 1 else ''} that the "
                        "run had left running on this machine when its worker stopped."
                    )
            elif _job_fields(record)[0] > 1:
                notes.append(
                    "Whether processes of the run are still running could not be checked on "
                    "this system (they are found through /proc, on Linux). Look for them "
                    "yourself."
                )
            path = Path(left.scratch)
            if remove_tree(path):
                notes.append(f"Removed the working directory {path} that the run had left behind.")
            else:
                notes.append(
                    f"The working directory {path} that the run left behind could not be "
                    "removed. Remove it yourself."
                )
            return notes

    def _execute(
        self,
        spec: JobSpec,
        work: Path,
        home: Path,
        run_dir: Path,
        log: JobLog,
        should_stop: Callable[[], bool] | None,
        result: JobResult,
    ) -> None:
        """Run the command in ``work`` and record how it ended in ``result``."""
        # `python` is the interpreter whose packages `supports` looked at, not whichever
        # one PATH happens to find first.
        argv = list(spec.command)
        if argv[0] in ("python", "python3"):
            argv[0] = self.python
        env = scrubbed_env(home)
        env["PYTHONUNBUFFERED"] = "1"  # prints reach the person watching as they happen
        env.update(spec.env)
        # The mark of this job: the directory made for it, which nothing else is called.
        mark = str(work.parent)
        env[JOB_ENV] = mark
        preexec = None
        if os.name == "posix":
            # Only a limit the job asked for.  The default cap is sized for a plotting
            # script; numerical libraries reserve far more address space than they use,
            # and would die of it with a MemoryError that says nothing about the job.
            memory_mb = spec.memory_mb
            # CPU time, not wall-clock time: a job on every core uses that many seconds
            # per second.  The wall-clock limit is kept by supervise(); this one only
            # ends a job that nobody supervises any more.
            cpu_seconds = math.ceil(spec.timeout) * (os.cpu_count() or 1) + 5
            preexec = _limit_resources(
                cpu_seconds, memory_mb * 1024 * 1024 if memory_mb else None, die_with_parent=True
            )
        started = time.monotonic()
        try:
            process = _spawn(argv, cwd=work, env=env, preexec=preexec, merge_stderr=True)
        except OSError as exc:
            # The command is the job's own, so this is the job failing, with the status
            # a shell would give it: 126 for a program that may not be run, 127 for one
            # that is not there.
            denied = isinstance(exc, PermissionError) and _program_exists(argv[0], env, work)
            log.note(f"The command could not be started: {exc}.")
            result.returncode = 126 if denied else 127
            return
        assert process.stdout is not None
        strays: set[int] = set()

        def kill() -> None:
            strays.update(_kill_job(process, mark))

        try:
            # For whoever settles this run if this process is killed: what to end, and
            # what to remove (see end_job).
            write_record(run_dir / JOB_NAME, _job_record(mark, process.pid))
            outcome = supervise(
                _pipe_chunks(process.stdout),
                log,
                timeout=spec.timeout,
                should_stop=should_stop,
                interrupt=lambda: _kill_group(process, signal.SIGTERM),
                kill=kill,
                exited=lambda: _has_exited(process),
            )
        finally:
            # Also on Ctrl-C and on errors of the callbacks: nothing of the job outlives
            # this call.  Killed before it is reaped, see _kill_job.
            kill()
            process.wait()
            process.stdout.close()
        result.seconds = time.monotonic() - started
        code = process.returncode
        result.returncode = code if code >= 0 else 128 - code
        if outcome.timed_out:
            result.timed_out = True
            log.note(
                "The job was stopped because it ran longer than its time limit of "
                f"{spec.timeout:g} s."
            )
        elif outcome.cancelled:
            result.cancelled = True
            log.note("The job was stopped on request.")
        if strays:
            count = len(strays)
            self._note(
                result,
                log,
                [
                    f"{count} process{'es' if count > 1 else ''} that the job had started "
                    "outside its own session (they would have run on after it) "
                    f"{'were' if count > 1 else 'was'} stopped when the job ended."
                ],
            )
        if outcome.abandoned:
            self._note(
                result,
                log,
                [
                    "Something the job started kept its output open after the job had ended "
                    "and could not be found to stop it (it left the job's session and no "
                    "longer carries the job's mark in its environment). It may still be "
                    "running on this machine; what it prints from here on is not in the log."
                ],
            )
        if outcome.failure is not None:
            raise outcome.failure

    def check(self, *, python: bool = True, latex: bool = True) -> list[str]:
        if latex and not self.latex_available():
            return [
                "pdflatex/bibtex were not found on PATH, so the paper cannot be compiled. "
                "Install TeX Live, or run with --compute modal to compile in a Modal sandbox."
            ]
        return []

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "python": self.python,
            "pdflatex": shutil.which("pdflatex"),
            "bibtex": shutil.which("bibtex"),
            "isolation": "none (subprocess with a minimal environment)",
            "jobs": {
                "supported": True,
                "gpu": False,
                "network_isolation": False,
                "packages": False,
                "reattach": False,
                "summary": (
                    "Runs on this machine as your user, with the packages installed here. "
                    "No GPU, and network access cannot be blocked: it is not a sandbox, and "
                    "a job can read what you can read."
                ),
            },
        }
