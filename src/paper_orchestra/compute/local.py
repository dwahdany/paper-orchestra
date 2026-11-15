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
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from paper_orchestra.compute.base import (
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
    open_run,
    refuse_unsafe_document,
    requirement_name,
    safe_relative,
    scan_outputs,
    spec_problems,
    supervise,
    walk_source,
)
from paper_orchestra.utils.latex import latex_commands, summarize_build

_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TZ")
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


def _kill_job(process: subprocess.Popen[bytes]) -> None:
    """Kill the job ``process`` leads: its process group, and what left the group.

    Tools such as coreutils ``timeout`` and shells with job control move what they
    start to a process group of its own, which a signal to the job's group does not
    reach.  The session is still the job's, so on Linux the processes of the session
    are looked up and killed too.  Only what starts a session of its own (a daemon)
    gets away.  Call this before the process is reaped: until then its id cannot
    have been given to an unrelated process group or session.
    """
    _kill_group(process)
    if not sys.platform.startswith("linux"):
        return
    with contextlib.suppress(OSError):
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/stat", "rb") as stat_file:
                    # "pid (name) state ppid pgrp session ...": the name may hold anything.
                    fields = stat_file.read().rsplit(b")", 1)[1].split()
                if int(fields[3]) == process.pid:
                    os.kill(int(entry.name), signal.SIGKILL)
            except (OSError, IndexError, ValueError):
                continue  # it ended while we were looking


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
    ) -> JobResult:
        source, run_dir = Path(source_dir), Path(run_dir)
        log_path = open_run(run_dir, spec.outputs)
        result = JobResult(returncode=None, backend=self.name, log_path=log_path)
        result.outputs_dir = run_dir
        with JobLog(log_path, on_log) as log:
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
                scratch = tempfile.TemporaryDirectory(
                    prefix="paper-orchestra-job-", dir=self.scratch_dir, ignore_cleanup_errors=True
                )
            except OSError as exc:
                result.error = f"The local backend could not make a working directory: {exc}."
                log.note(result.error)
                return result
            try:
                with scratch as name:
                    work, home = Path(name, "work"), Path(name, "home")
                    home.mkdir()
                    self._note(result, log, copy_source(source, work, should_stop))
                    self._execute(spec, work, home, log, should_stop, result)
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

    def _execute(
        self,
        spec: JobSpec,
        work: Path,
        home: Path,
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
            # a shell would give it.
            log.note(f"The command could not be started: {exc}.")
            result.returncode = 126 if isinstance(exc, PermissionError) else 127
            return
        assert process.stdout is not None
        try:
            outcome = supervise(
                _pipe_chunks(process.stdout),
                log,
                timeout=spec.timeout,
                should_stop=should_stop,
                interrupt=lambda: _kill_group(process, signal.SIGTERM),
                kill=lambda: _kill_job(process),
                exited=lambda: _has_exited(process),
            )
        finally:
            # Also on Ctrl-C and on errors of the callbacks: nothing of the job outlives
            # this call.  Killed before it is reaped, see _kill_job.
            _kill_job(process)
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
        if outcome.abandoned:
            self._note(
                result,
                log,
                [
                    "Something the job started kept its output open after the job had ended "
                    "and could not be stopped (it runs in a session of its own). It may still "
                    "be running on this machine; what it prints from here on is not in the log."
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
                    "No GPU, and network access cannot be blocked: it is not a sandbox."
                ),
            },
        }
