"""Compute backend contract.

Three kinds of work leave the orchestrator process:

* running model-written Python (figure code), which is untrusted,
* compiling model-written LaTeX, which needs a TeX distribution, and
* experiment jobs: a command run in a copy of a source directory, whose output a
  person watches while it runs and whose result files are copied back.

``LocalBackend`` runs all three as subprocesses on this machine.  ``ModalBackend``
runs them in Modal Sandboxes, so nothing model-written executes locally, no TeX
install is needed, and a job can have a GPU and packages of its own.

The second half of this module is the plumbing of jobs that both backends share:
the log, the supervision of a running job, and the rules for what goes into a
working directory and what comes back out of it.
"""

from __future__ import annotations

import codecs
import contextlib
import fnmatch
import inspect
import math
import os
import queue
import re
import shutil
import stat
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, NamedTuple


class ComputeError(RuntimeError):
    """The backend itself failed (as opposed to the job exiting non-zero)."""

    #: True when the backend cannot work as configured (a missing tool, no credentials), so
    #: trying again cannot help and the run should stop with the message.
    unusable: bool = False


class BackendUnusable(ComputeError):
    """The backend cannot work as configured; retrying will not change that."""

    unusable = True


def safe_relative(name: str) -> Path:
    """Reject absolute paths and ``..`` in caller-supplied file names."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ComputeError(f"unsafe file name: {name!r}")
    return path


@dataclass
class ExecResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    #: Requested output files that existed after the run, keyed by relative path.
    files: dict[str, bytes] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass
class CompileResult:
    ok: bool
    pdf_path: Path | None = None
    #: LaTeX error messages (``! ...`` blocks with their ``l.<n>`` context).
    errors: list[str] = field(default_factory=list)
    #: Undefined citations / references and other warnings worth fixing.
    warnings: list[str] = field(default_factory=list)
    #: Tail of the combined tool output, for humans and for repair prompts.
    log_tail: str = ""
    timed_out: bool = False

    def feedback(self, max_chars: int = 6000) -> str:
        """Compact description of what went wrong, suitable for a repair prompt."""
        lines: list[str] = []
        if self.timed_out:
            lines.append("Compilation timed out.")
        if self.errors:
            lines.append("LaTeX errors:")
            lines.extend(f"- {e}" for e in self.errors[:40])
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {w}" for w in self.warnings[:40])
        if not self.ok and not self.errors and self.log_tail:
            lines.append("Log tail:")
            lines.append(self.log_tail[-2500:])
        text = "\n".join(lines)
        return text[:max_chars]


def refuse_unsafe_document(workdir: str | Path, tex_name: str) -> CompileResult | None:
    """A failed result for a document that must not be compiled, else ``None``.

    Every backend calls this before running TeX.  TeX's own file-access limits do
    not cover every primitive (pdfTeX can embed any readable file in the PDF), so
    a document that uses file-reading, file-writing or code-running constructs is
    turned down and the reasons are returned as errors for the author to fix.
    """
    from paper_orchestra.utils.latex import unsafe_constructs

    source = Path(workdir) / f"{tex_name}.tex"
    try:
        problems = unsafe_constructs(source.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    if not problems:
        return None
    return CompileResult(
        ok=False,
        errors=problems,
        log_tail="The document was not compiled because it uses constructs that are not allowed.",
    )


# -- experiment jobs: the contract -------------------------------------------------------

#: Where a backend writes the combined output of a job, inside the run directory.
LOG_NAME = "log.txt"
#: Where a backend with remote jobs writes the job's handle as soon as it has one.
HANDLE_NAME = "handle.txt"
#: Files of a run directory that the backend or the engine writes.  A job's outputs may
#: not replace them.
RESERVED_RUN_FILES = frozenset({LOG_NAME, HANDLE_NAME, "run.json"})
#: Starts every line that a backend adds to a job's log itself.
NOTE_PREFIX = "[paper-orchestra] "

#: What ``describe()["jobs"]`` holds for a backend that runs no jobs.
NO_JOB_FEATURES: Mapping[str, bool] = {
    "supported": False,
    "gpu": False,
    "network_isolation": False,
    "packages": False,
    "reattach": False,
}


@dataclass
class JobSpec:
    """What to run.  ``command`` is an argv, run in a copy of the source directory."""

    command: list[str]
    #: Seconds of wall-clock time the command may take.
    timeout: float = 900.0
    #: A GPU type such as ``"T4"``, optionally with a count (``"A100:2"``); ``None`` = CPU.
    gpu: str | None = None
    #: Whether the job may use the network.
    network: bool = False
    #: pip requirements, each a name with optional extras and version (``"torch>=2.4"``).
    packages: tuple[str, ...] = ()
    #: Extra environment variables.  Non-secret values only: the job is untrusted.
    env: dict[str, str] = field(default_factory=dict)
    #: CPU cores and memory (MiB) the job may use; ``None`` leaves it to the backend.
    cpu: float | None = None
    memory_mb: int | None = None
    #: Files and directories, relative to the working directory, copied to the run directory.
    outputs: tuple[str, ...] = ("metrics.json", "artifacts")


@dataclass
class JobResult:
    """How a job went.

    Read ``error`` first (the backend failed: the job is not to blame, and repairing
    its code will not help), then ``cancelled`` and ``timed_out``, then ``returncode``.
    """

    #: Exit status of the command; ``128 + n`` when signal ``n`` ended it, which is also
    #: what a job that was stopped usually shows.  ``None`` when the backend has none:
    #: it refused the job, lost it, or ended the sandbox it ran in.
    returncode: int | None
    timed_out: bool = False
    cancelled: bool = False
    #: Wall-clock seconds the command ran.
    seconds: float = 0.0
    #: Combined stdout and stderr as UTF-8 text (bytes that are not are replaced).
    log_path: Path | None = None
    #: The run directory, where the requested outputs that existed were copied.
    outputs_dir: Path | None = None
    backend: str = ""
    #: Remote id of the job, for ``ComputeBackend.attach``.
    handle: str | None = None
    #: Estimated cost; ``None`` when unknown or when nothing is billed.
    cost_usd: float | None = None
    #: The backend's own failure, as a message for a person.  Not the job's.
    error: str | None = None
    #: What the person should know that is neither output nor a failure: outputs that
    #: were not copied back, isolation the backend could not give, files left out.
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """The command ran to its end and exited with status 0."""
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.cancelled
            and self.error is None
        )


class ComputeBackend(ABC):
    name: str = ""

    @abstractmethod
    def run_python(
        self,
        code: str,
        *,
        inputs: Mapping[str, bytes] | None = None,
        outputs: Sequence[str] = (),
        timeout: float = 120.0,
    ) -> ExecResult:
        """Run ``code`` as a script in a fresh, empty working directory.

        ``inputs`` are written into that directory first.  Each relative path in
        ``outputs`` that exists afterwards is returned in ``ExecResult.files``.
        The code must be treated as untrusted: no credentials in its
        environment, and no network where the backend can enforce it.
        """

    @abstractmethod
    def compile_latex(
        self,
        workdir: str | Path,
        tex_name: str,
        *,
        output_pdf: str | Path | None = None,
        timeout: float = 240.0,
    ) -> CompileResult:
        """Compile ``<workdir>/<tex_name>.tex`` (pdflatex, bibtex, pdflatex x2).

        Shell escape is always disabled.  On success the PDF is at
        ``output_pdf`` (default ``<workdir>/<tex_name>.pdf``).  A PDF produced
        despite recoverable LaTeX errors still counts as ``ok`` with the errors
        listed, matching what a human author would get from ``nonstopmode``.
        """

    def run_job(
        self,
        spec: JobSpec,
        *,
        source_dir: Path,
        run_dir: Path,
        on_log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> JobResult:
        """Run ``spec.command`` in a private copy of ``source_dir`` and report how it went.

        ``source_dir`` is never run in place and never changed.  The combined
        stdout and stderr of the job go to ``run_dir/log.txt`` as they are produced
        and, line by line, to ``on_log`` (called on the caller's thread).  The job is
        stopped when ``should_stop()`` turns true (asked at least once a second) or
        when ``spec.timeout`` seconds have passed.  Afterwards the paths in
        ``spec.outputs`` that exist are copied into ``run_dir``; outputs of an earlier
        run that were there are removed first, so what is found there is this run's.

        Every outcome that follows from the job, or from the backend failing while it
        worked, is a :class:`JobResult`; ``error`` is set when the backend refused the
        job, could not run it or lost it.  Only three things raise:
        :class:`BackendUnusable` (no credentials: trying again cannot help),
        :class:`ComputeError` when ``run_dir`` cannot be written, and whatever
        ``on_log`` or ``should_stop`` raise, including ``KeyboardInterrupt``.  In every
        case the job has been stopped and its working directory is gone when this
        returns.
        """
        raise ComputeError(f"The {self.name or 'compute'} backend does not run experiment jobs.")

    def supports(self, spec: JobSpec) -> list[str]:
        """Why this backend cannot run ``spec``, as sentences for a person; empty if it can.

        Cheap and offline: it looks at the request and at this machine only.
        ``run_job`` applies the same test and answers a refused job with a result
        whose ``error`` holds these reasons.
        """
        return [f"The {self.name or 'compute'} backend does not run experiment jobs."]

    def attach(
        self,
        handle: str,
        *,
        run_dir: Path,
        on_log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> JobResult:
        """Pick up the job ``handle`` (``JobResult.handle``) after the process that ran it died.

        Only backends whose jobs outlive the orchestrator have something to pick up;
        ``describe()["jobs"]["reattach"]`` says whether this one does.
        """
        raise ComputeError(
            f"Jobs of the {self.name or 'compute'} backend cannot be picked up again: they end "
            "with the process that started them. Run the job again."
        )

    def check(self, *, python: bool = True, latex: bool = True) -> list[str]:
        """Why this backend cannot do the named kinds of work, as messages for the user.

        Called once before any model call is paid for; an empty list means it is
        ready.  Implementations keep this cheap: look for the tools or the
        credentials, without starting a job.
        """
        return []

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release remote resources.  Safe to call more than once."""

    def describe(self) -> dict[str, Any]:
        """Non-secret diagnostics for ``paper-orchestra doctor`` and the setup page.

        ``jobs`` says what experiment jobs can ask of this backend: ``supported``,
        ``gpu``, ``network_isolation`` (a job without network access really has
        none), ``packages`` (it installs what a job asks for) and ``reattach``.
        """
        return {"backend": self.name, "jobs": dict(NO_JOB_FEATURES)}

    def __enter__(self) -> ComputeBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# -- experiment jobs: plumbing shared by the backends ------------------------------------

#: Bytes and files copied back from one job.  Outputs are figures and tables for the wiki,
#: not datasets or checkpoints, and the project directory is meant to stay small.
MAX_OUTPUT_BYTES = 500 * 1024 * 1024
MAX_OUTPUT_FILES = 2000
#: Bytes of output a log keeps from the start of a job, and from its end once that is
#: exceeded, so that a job printing in a loop can fill neither the disk nor the memory.
MAX_LOG_BYTES = 64 * 1024 * 1024
LOG_TAIL_BYTES = 256 * 1024
#: Longest piece of output handed to ``on_log`` at once.
MAX_LINE_CHARS = 4000
#: Seconds a job has between being asked to stop and being killed.
KILL_GRACE = 5.0
#: Seconds without output after which a killed job's output is given up, and bytes of it
#: that are read at most.  What was waiting to be read when the job was killed is less
#: than that and ends by itself; only something that got away from the kill and still
#: holds the output open stays silent for longer, or prints more.
DRAIN_SECONDS = 5.0
DRAIN_BYTES = 16 * 1024 * 1024
#: How long supervision sleeps when no output arrives, and how often it asks ``should_stop``.
TICK_SECONDS = 0.2
CHECK_SECONDS = 0.5
#: Chunks of output that may wait for the supervising thread; the reader blocks beyond it.
QUEUE_CHUNKS = 64

#: Names that never go into a job's working directory, matched against each file and
#: directory name: version-control data and caches (large, and not what the job is about),
#: environments (rebuilt from ``JobSpec.packages``) and ``.env`` files, which hold the
#: credentials a job must not see.
SOURCE_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".ipynb_checkpoints",
    ".DS_Store",
    ".env",
    ".env.*",
)

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GPU = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*!?(?::[1-9][0-9]?)?")
_VERSION = r"(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+"
# A name, optional extras, optional versions.  Nothing else pip accepts (options, URLs,
# paths, markers): the strings are written by a model and end up in an image build.
_REQUIREMENT = re.compile(
    r"\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"\s*(?:\[\s*[A-Za-z0-9._-]+(?:\s*,\s*[A-Za-z0-9._-]+)*\s*\])?"
    rf"\s*(?:{_VERSION}(?:\s*,\s*{_VERSION})*)?\s*"
)
_LINE_END = re.compile(r"[\r\n]")


def human_size(count: int) -> str:
    if count < 10_000:
        return f"{count} bytes"
    if count < 1024 * 1024:
        return f"{count / 1024:.0f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


def _listed(names: Sequence[str], limit: int = 5) -> str:
    more = len(names) - limit
    return ", ".join(names[:limit]) + (f" and {more} more" if more > 0 else "")


def requirement_name(requirement: str) -> str | None:
    """The distribution a pip requirement names, or ``None`` if it is not a plain one."""
    match = _REQUIREMENT.fullmatch(requirement) if isinstance(requirement, str) else None
    return match.group("name") if match else None


def _output_problem(name: object) -> str | None:
    if isinstance(name, str) and "\0" not in name:
        with contextlib.suppress(ComputeError):
            if safe_relative(name).parts[0] not in RESERVED_RUN_FILES:
                return None
            return (
                f"Output {name!r} would replace a record of the run "
                f"({', '.join(sorted(RESERVED_RUN_FILES))}); give the output another name."
            )
    return f"Output {name!r} must be a relative path inside the working directory."


def spec_problems(spec: JobSpec) -> list[str]:
    """What is wrong with ``spec`` whatever the backend, as sentences for a person."""
    problems: list[str] = []
    command = spec.command
    if not isinstance(command, list | tuple) or not command:
        problems.append("The job has no command to run.")
    elif not all(isinstance(arg, str) and "\0" not in arg for arg in command) or not command[0]:
        problems.append("The command of the job must be a list of strings, the program first.")
    timeout = spec.timeout
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not 0 < timeout < math.inf
    ):
        problems.append(
            f"The time limit of the job must be a positive number of seconds, not {timeout!r}."
        )
    for field_name in ("outputs", "packages"):
        if not isinstance(getattr(spec, field_name), list | tuple):
            # A bare string would be taken apart into its characters.
            problems.append(f"The {field_name} of the job must be a list, not a single string.")
            return problems
    problems.extend(filter(None, map(_output_problem, spec.outputs)))
    for name, value in spec.env.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            problems.append(
                f"Environment variable name {name!r} is not valid: use letters, digits and "
                "underscores, and do not start with a digit."
            )
        elif not isinstance(value, str) or "\0" in value:
            problems.append(f"The value of environment variable {name!r} must be a string.")
    if spec.cpu is not None and not 0 < spec.cpu < math.inf:
        problems.append(f"The number of cpu cores must be positive, not {spec.cpu!r}.")
    if spec.memory_mb is not None and not 16 <= spec.memory_mb < math.inf:
        problems.append(f"The memory of the job must be at least 16 MB, not {spec.memory_mb!r}.")
    if spec.gpu is not None and not (isinstance(spec.gpu, str) and _GPU.fullmatch(spec.gpu)):
        problems.append(
            f"The GPU request {spec.gpu!r} is not a GPU type. Give a type such as T4 or "
            "A100-80GB, optionally with a count as in H100:2."
        )
    for requirement in spec.packages:
        if requirement_name(requirement) is None:
            problems.append(
                f"Package {requirement!r} is not a plain requirement. Give a package name with "
                "an optional version, such as numpy or torch>=2.4; options, URLs and paths are "
                "not accepted."
            )
    return problems


def open_run(run_dir: Path, outputs: Sequence[str]) -> Path:
    """Make ``run_dir`` ready for a new run and return the path of its log.

    The log is emptied, and the requested outputs and the handle that an earlier
    run left there are removed (links are removed, never followed), so that what is
    found afterwards is this run's.  Outputs that ``spec_problems`` objects to are
    left alone.  Raises :class:`ComputeError` when the directory cannot be used.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / LOG_NAME).write_bytes(b"")
        stale = [run_dir / HANDLE_NAME]
        for name in () if isinstance(outputs, str) else outputs:
            if _output_problem(name) is None:
                stale.append(run_dir / safe_relative(name))
        for path in stale:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    except OSError as exc:
        raise ComputeError(
            f"The run directory {run_dir} cannot be used: {exc}. Check that it can be written."
        ) from exc
    return run_dir / LOG_NAME


class _Lines:
    """Cuts text into the pieces that ``on_log`` receives.

    A piece ends at a line feed or carriage return (so a progress bar that redraws
    itself arrives as separate pieces) or after ``MAX_LINE_CHARS`` characters.  Empty
    pieces are dropped.  The pieces depend only on the text, never on how it was cut
    into chunks; replaying a log relies on that.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._pending = ""

    def feed(self, text: str) -> None:
        pending = self._pending + text
        start = 0
        for match in _LINE_END.finditer(pending):
            self._pieces(pending[start : match.start()])
            start = match.end()
        rest = pending[start:]
        whole = len(rest) - len(rest) % MAX_LINE_CHARS
        self._pieces(rest[:whole])
        self._pending = rest[whole:]

    def flush(self) -> None:
        pending, self._pending = self._pending, ""
        self._pieces(pending)

    def _pieces(self, line: str) -> None:
        for start in range(0, len(line), MAX_LINE_CHARS):
            self._emit(line[start : start + MAX_LINE_CHARS])


def _utf8_decoder() -> codecs.IncrementalDecoder:
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


class JobLog:
    """The log of one job: a file that grows as output arrives, and lines for ``on_log``.

    The file is UTF-8 text whatever the job printed (other bytes are replaced), and
    holds everything received so far after every ``feed``, so a person can follow it
    while the job runs.  Beyond ``MAX_LOG_BYTES`` only the end of the output is kept
    (in memory, bounded) and written when the log is closed, and ``on_log`` is told
    once and then left alone.  ``note`` adds a line of the backend's own.  A file that
    cannot be written (a full disk) raises :class:`ComputeError`.

    ``skip_lines`` withholds that many lines from ``on_log``: a log that is rebuilt
    from the start of a job reports only what was not reported before.  ``append``
    keeps what the file holds.  Not thread-safe: one thread feeds it.
    """

    def __init__(
        self,
        path: Path,
        on_log: Callable[[str], None] | None = None,
        *,
        skip_lines: int = 0,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        #: Bytes of job output received, before any replacement or truncation.
        self.raw_bytes = 0
        #: Whether output was left out of the file because the job printed too much.
        self.truncated = False
        self._on_log = on_log
        self._skip = skip_lines
        self._lines = _Lines(self._deliver)
        self._decoder = _utf8_decoder()
        self._file: IO[bytes] = open(self.path, "ab" if append else "wb")  # noqa: SIM115
        self._at_line_start = not append or _ends_with_newline(self.path)
        self._written = 0
        self._tail: deque[bytes] = deque()
        self._tail_bytes = 0
        self._beyond = 0
        self._late_notes: list[str] = []
        self._closed = False

    def feed(self, data: bytes) -> None:
        """Take the next bytes of the job's output."""
        self.raw_bytes += len(data)
        self._accept(self._decoder.decode(data))

    def note(self, message: str) -> None:
        """Add a line of the backend's own, on a line of its own."""
        line = NOTE_PREFIX + message
        if self.truncated:
            self._late_notes.append(line)
        else:
            self._write(("" if self._at_line_start else "\n") + line + "\n")
        if self._on_log is not None:
            self._lines.flush()
            self._on_log(line)

    def notes(self) -> list[str]:
        """What the person should know about the log itself, for ``JobResult.notes``."""
        if not self.truncated:
            return []
        return [
            f"The job printed more than {human_size(MAX_LOG_BYTES)}; its log keeps the "
            "beginning and the end of the output."
        ]

    def close(self) -> None:
        """Write what is still held back and close the file.  Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        try:
            self._accept(self._decoder.decode(b"", final=True))
            if self.truncated:
                self._write_tail()
            if self._on_log is not None:
                self._lines.flush()
        finally:
            # Closing writes what an earlier write could not; that failure was reported.
            with contextlib.suppress(OSError):
                self._file.close()

    def __enter__(self) -> JobLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _accept(self, text: str) -> None:
        if not text:
            return
        data = text.encode("utf-8")
        if not self.truncated and self._written + len(data) > MAX_LOG_BYTES:
            self.truncated = True
            if self._on_log is not None:
                self._lines.flush()
                self._on_log(
                    f"{NOTE_PREFIX}The job has printed more than {human_size(MAX_LOG_BYTES)}; "
                    "its output is no longer shown live. The end of it is kept in the log."
                )
        if self.truncated:
            self._beyond += len(data)
            self._tail.append(data)
            self._tail_bytes += len(data)
            while self._tail_bytes - len(self._tail[0]) >= LOG_TAIL_BYTES:
                self._tail_bytes -= len(self._tail.popleft())
            return
        self._written += len(data)
        self._write(text)
        if self._on_log is not None:
            self._lines.feed(text)

    def _write(self, text: str) -> None:
        try:
            self._file.write(text.encode("utf-8"))
            # Flushed every time: the file is what a person watching the job reads.
            self._file.flush()
        except OSError as exc:
            raise ComputeError(
                f"The log {self.path} cannot be written: {exc}. Check the disk that holds "
                "the run directory."
            ) from exc
        self._at_line_start = text.endswith("\n")

    def _write_tail(self) -> None:
        tail = b"".join(self._tail)[-LOG_TAIL_BYTES:]
        self._tail.clear()
        marker = (
            f"{NOTE_PREFIX}{human_size(self._beyond - len(tail))} of output are left out "
            f"here: a log keeps the first {human_size(MAX_LOG_BYTES)} and the end."
        )
        # The cut may have gone through a character; what is left of it is dropped.
        end = tail.decode("utf-8", errors="ignore")
        self._write(("" if self._at_line_start else "\n") + marker + "\n" + end)
        for line in self._late_notes:
            self._write(("" if self._at_line_start else "\n") + line + "\n")

    def _deliver(self, piece: str) -> None:
        if not piece:
            return
        if self._skip > 0:
            self._skip -= 1
            return
        assert self._on_log is not None
        self._on_log(piece)


def _ends_with_newline(path: Path) -> bool:
    try:
        with open(path, "rb") as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                return True
            stream.seek(-1, os.SEEK_END)
            return stream.read(1) == b"\n"
    except OSError:
        return True


def delivered_lines(log_path: Path) -> int:
    """How many lines of job output the log at ``log_path`` has handed to ``on_log``.

    The backend's own notes and an unfinished last line do not count.  This is the
    ``skip_lines`` of a :class:`JobLog` that rebuilds the same log from the start.
    """
    count = 0

    def emit(piece: str) -> None:
        nonlocal count
        if piece and not piece.startswith(NOTE_PREFIX):
            count += 1

    lines = _Lines(emit)
    decoder = _utf8_decoder()
    try:
        with open(log_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                lines.feed(decoder.decode(chunk))
    except OSError:
        return 0
    return count


_END = object()


class _Failure(NamedTuple):
    error: BaseException


class _Pump(threading.Thread):
    """Reads a job's output on a thread of its own.

    Reading blocks while a job is silent, and the thread that supervises the job has
    to stay free to look at the clock and at ``should_stop``.  The queue between the
    two is bounded, so a job that prints faster than it is read is held back instead
    of filling the memory.
    """

    def __init__(self, chunks: Iterable[bytes]) -> None:
        super().__init__(name="paper-orchestra-job-output", daemon=True)
        self._chunks = chunks
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=QUEUE_CHUNKS)
        self._abandoned = threading.Event()

    def run(self) -> None:
        try:
            for chunk in self._chunks:
                # An empty chunk is the heartbeat of a stream that can be polled; it is
                # what lets this thread notice that nobody is listening any more.
                if self._abandoned.is_set() or (chunk and not self._offer(chunk)):
                    return
        except BaseException as exc:  # handed to the supervising thread
            self._offer(_Failure(exc))
        else:
            self._offer(_END)

    def _offer(self, item: Any) -> bool:
        while not self._abandoned.is_set():
            try:
                self._queue.put(item, timeout=0.1)
            except queue.Full:
                continue
            return True
        return False

    def get(self, timeout: float) -> Any:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def waiting(self) -> bool:
        """Whether output that was read is waiting to be taken with ``get``."""
        return not self._queue.empty()

    def abandon(self) -> None:
        self._abandoned.set()


@dataclass
class Supervised:
    """How supervision ended, unless the output simply came to its end."""

    timed_out: bool = False
    cancelled: bool = False
    #: What reading the output raised.  The job may still be running.
    failure: BaseException | None = None
    #: The job was killed and its output stayed open all the same: something it started
    #: got away and holds it.  What that prints from then on is not in the log.
    abandoned: bool = False


def supervise(
    chunks: Iterable[bytes],
    log: JobLog,
    *,
    timeout: float,
    interrupt: Callable[[], None],
    should_stop: Callable[[], bool] | None = None,
    kill: Callable[[], None] | None = None,
    exited: Callable[[], bool] | None = None,
    grace: float | None = None,
    drain: float | None = None,
) -> Supervised:
    """Feed a running job's output to ``log`` until the output ends, and stop the job
    when it should not go on.

    ``chunks`` is the job's output; iterating it may block.  After ``timeout``
    seconds, or once ``should_stop()`` is true (asked every ``CHECK_SECONDS``, with
    or without output arriving), ``interrupt()`` asks the job to end; ``kill()`` ends
    it for certain ``grace`` seconds later, or as soon as ``exited()`` says the
    command is gone.  ``exited()`` turning true by itself means the command has ended
    while something it started still holds its output open: that is killed too.

    After a kill, what the job had printed is still read to its end, however long
    ``on_log`` takes over it.  Reading is only given up (``abandoned``) when the output
    is still open ``drain`` seconds later with nothing left of it waiting to be read,
    or when more than ``DRAIN_BYTES`` have come since the kill: something survived it
    and holds the output.

    Returns when the output has ended, was given up, or failed.  The caller still
    owns the job: it makes sure that the job is dead, whatever this returns or raises.
    """
    grace = KILL_GRACE if grace is None else grace
    drain = DRAIN_SECONDS if drain is None else drain
    outcome = Supervised()
    pump = _Pump(chunks)
    pump.start()
    deadline = time.monotonic() + timeout
    next_check = 0.0
    kill_at: float | None = None  # set once the job was asked to stop
    give_up_at: float | None = None  # set once it was killed
    drained = 0  # bytes of output read since then
    last_look = False  # all that was waiting has been read; what comes next decides
    try:
        while True:
            item = pump.get(TICK_SECONDS)
            if item is _END:
                break
            if isinstance(item, _Failure):
                outcome.failure = item.error
                break
            if item is not None:
                log.feed(item)
                if give_up_at is not None:
                    drained += len(item)
            now = time.monotonic()
            if give_up_at is not None:
                if last_look or drained > DRAIN_BYTES:
                    outcome.abandoned = True
                    break
                # Not while output is waiting to be read: a slow `on_log` must not cost
                # the end of the log, which is where the result and the traceback are.
                # Once nothing is waiting, the end of the output is all that can still
                # come from the job itself; the next look tells (it broke out above).
                last_look = now >= give_up_at and not pump.waiting()
            elif kill_at is not None:
                if now >= kill_at or (exited is not None and exited()):
                    if kill is not None:
                        kill()
                    give_up_at = now + drain
            elif now >= next_check or now >= deadline:
                next_check = now + CHECK_SECONDS
                if exited is not None and exited():
                    if kill is not None:
                        kill()
                    give_up_at = now + drain
                elif should_stop is not None and should_stop():
                    outcome.cancelled = True
                elif now >= deadline:
                    outcome.timed_out = True
                if outcome.cancelled or outcome.timed_out:
                    interrupt()
                    kill_at = now + grace
    finally:
        pump.abandon()
        pump.join(timeout=0.5)
    return outcome


class OutputEntry(NamedTuple):
    """One thing found under a job's requested outputs."""

    #: Relative to the working directory, with forward slashes.
    path: str
    #: ``"file"``, ``"link"`` or ``"other"`` (a pipe, a socket, a device).
    kind: str
    size: int


def scan_outputs(
    root: str, outputs: list[str], max_entries: int = 100_000, max_chars: int = 10_000_000
) -> dict:
    """List what exists under the requested ``outputs`` of a job, following no links.

    Returns ``{"entries": [[path, kind, size], ...], "truncated": bool}`` with the
    fields of :class:`OutputEntry`.  An output that lies behind a link (``results`` is
    one, and ``results/metrics.json`` was asked for) is not looked at: the link is
    listed in its place.  Listing stops, and ``truncated`` is set, after
    ``max_entries`` entries or when the listing, written as JSON, would be longer
    than ``max_chars`` characters.

    Self-contained on purpose (its imports are inside, its annotations are builtins):
    the Modal backend runs this function's source inside the sandbox, so that both
    backends see a job's outputs in the same way.
    """
    import json
    import os
    import stat

    entries: list[list] = []
    seen: set[str] = set()
    chars = 0
    truncated = False

    def add(path: str) -> bool:
        nonlocal chars, truncated
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        if relative in seen:
            return True
        try:
            info = os.lstat(path)
        except OSError:
            return True
        cost = len(json.dumps(relative)) + 24  # what the entry takes in the listing
        if len(entries) >= max_entries or chars + cost > max_chars:
            truncated = True
            return False
        seen.add(relative)
        chars += cost
        if stat.S_ISREG(info.st_mode):
            entries.append([relative, "file", info.st_size])
        elif stat.S_ISLNK(info.st_mode):
            entries.append([relative, "link", 0])
        else:
            entries.append([relative, "other", 0])
        return True

    def link_above(output: str) -> str | None:
        current = root
        parts = [part for part in output.replace(os.sep, "/").split("/") if part not in ("", ".")]
        for part in parts[:-1]:
            current = os.path.join(current, part)
            if os.path.islink(current):
                return current
        return None

    for output in outputs:
        top = os.path.join(root, output)
        # A link on the way to an output leads out of the working directory just as one
        # at its end does.
        above = link_above(output)
        if above is not None:
            if not add(above):
                break
            continue
        if os.path.islink(top) or not os.path.isdir(top):
            if os.path.lexists(top) and not add(top):
                break
            continue
        for current, dirs, files in os.walk(top):
            dirs.sort()
            # os.walk does not descend into linked directories, but it names them here.
            links = [name for name in dirs if os.path.islink(os.path.join(current, name))]
            if not all(add(os.path.join(current, name)) for name in sorted(files + links)):
                break
        if truncated:
            break
    return {"entries": entries, "truncated": truncated}


def scan_outputs_script() -> str:
    """A standalone script that prints :func:`scan_outputs` of its working directory as
    JSON.  Its arguments are ``max_entries``, ``max_chars`` and then the outputs."""
    return (
        "from __future__ import annotations\n\n\n"
        + inspect.getsource(scan_outputs)
        + "\n\nimport json\nimport sys\n\n"
        "listing = scan_outputs('.', sys.argv[3:], int(sys.argv[1]), int(sys.argv[2]))\n"
        "print(json.dumps(listing))\n"
    )


def choose_outputs(
    entries: Iterable[Sequence[Any]],
    *,
    max_bytes: int | None = None,
    max_files: int | None = None,
    truncated: bool = False,
) -> tuple[list[tuple[str, int]], list[str]]:
    """Decide which of the scanned ``entries`` are copied to the run directory.

    Returns the chosen ``(path, size)`` pairs, in the order they were found, and
    notes for the person about everything that was left out: links (never followed,
    so no output can lead out of the working directory), things that are not regular
    files, files beyond ``max_files``, and files that no longer fit into ``max_bytes``
    in total.  ``truncated`` says that the scan itself stopped early.
    """
    max_bytes = MAX_OUTPUT_BYTES if max_bytes is None else max_bytes
    max_files = MAX_OUTPUT_FILES if max_files is None else max_files
    chosen: list[tuple[str, int]] = []
    seen: set[str] = set()
    remaining = max_bytes
    too_large: list[str] = []
    links: list[str] = []
    others: list[str] = []
    beyond = 0
    for path, kind, size in entries:
        if path in seen:
            continue
        seen.add(path)
        if kind == "link":
            links.append(path)
        elif kind != "file":
            others.append(path)
        elif len(chosen) >= max_files:
            beyond += 1
        elif size > remaining:
            too_large.append(f"{path} ({human_size(size)})")
        else:
            chosen.append((path, size))
            remaining -= size
    notes: list[str] = []
    if too_large:
        notes.append(
            f"Not copied (the outputs of a job are limited to {human_size(max_bytes)} in "
            f"total): {_listed(too_large)}."
        )
    if beyond:
        notes.append(
            f"Only the first {max_files} files of the outputs were copied; {beyond} more "
            "were left out."
        )
    if links:
        notes.append(f"Not copied (symbolic links are never followed): {_listed(links)}.")
    if others:
        notes.append(f"Not copied (not a regular file): {_listed(others)}.")
    if truncated:
        notes.append("The outputs hold too many entries to list; the rest was not looked at.")
    return chosen, notes


@dataclass
class SourceReport:
    """What :func:`walk_source` left out of a working directory."""

    excluded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def notes(self) -> list[str]:
        notes: list[str] = []
        if self.excluded:
            notes.append(
                "Left out of the working directory (version-control data, caches, "
                f"environments and .env files are not copied): {_listed(self.excluded)}."
            )
        if self.skipped:
            notes.append(
                "Left out of the working directory (links that lead outside the source "
                f"directory, and special files): {_listed(self.skipped)}."
            )
        return notes


def _excluded(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in SOURCE_EXCLUDES)


def _link_stays_inside(path: Path, relative: str) -> bool:
    """Whether the link at ``path`` is relative and still valid in a copy of the tree."""
    target = os.readlink(path)
    if os.path.isabs(target):
        return False
    resolved = os.path.normpath(os.path.join(os.path.dirname(relative), target))
    return resolved != ".." and not resolved.startswith(".." + os.sep)


class JobStopped(Exception):
    """``should_stop`` turned true before the job's command had been started."""


def walk_source(
    source: Path, report: SourceReport, should_stop: Callable[[], bool] | None = None
) -> Iterator[tuple[str, Path, str]]:
    """Yield what a job's working directory receives from ``source``.

    Each item is ``(relative posix path, path, kind)`` with kind ``"dir"``,
    ``"file"`` or ``"link"``, a directory before its contents.  Names matching
    ``SOURCE_EXCLUDES`` are left out, at any depth.  A relative link that stays
    inside the tree is kept as a link; any other link is left out, as are special
    files, so that nothing in the working directory leads back to the person's files.
    What was left out is recorded in ``report``.

    A large tree takes a while to copy or pack, so ``should_stop`` is asked every
    ``CHECK_SECONDS`` on the way, between files; :class:`JobStopped` is raised once it
    is true.
    """
    next_check = 0.0

    def hear_a_stop() -> None:
        nonlocal next_check
        if should_stop is not None and time.monotonic() >= next_check:
            if should_stop():
                raise JobStopped
            next_check = time.monotonic() + CHECK_SECONDS

    for current, dirs, files in os.walk(source):
        hear_a_stop()
        here = Path(current)
        descend: list[str] = []
        for name in sorted(dirs):
            path = here / name
            relative = path.relative_to(source).as_posix()
            if _excluded(name):
                report.excluded.append(relative)
            elif path.is_symlink():
                if _link_stays_inside(path, relative):
                    yield relative, path, "link"
                else:
                    report.skipped.append(relative)
            else:
                descend.append(name)
                yield relative, path, "dir"
        dirs[:] = descend
        for name in sorted(files):
            path = here / name
            relative = path.relative_to(source).as_posix()
            if _excluded(name):
                report.excluded.append(relative)
                continue
            try:
                mode = path.lstat().st_mode
            except OSError:  # removed while we were walking
                continue
            if stat.S_ISREG(mode):
                # Asked here too: one directory can hold hours of files to copy.
                hear_a_stop()
                yield relative, path, "file"
            elif stat.S_ISLNK(mode) and _link_stays_inside(path, relative):
                yield relative, path, "link"
            else:
                report.skipped.append(relative)
