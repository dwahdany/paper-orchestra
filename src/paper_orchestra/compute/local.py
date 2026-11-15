"""Run jobs as subprocesses on this machine.

This backend needs TeX Live on ``PATH`` for LaTeX.  Both kinds of job get a
minimal environment without credentials, their own process group, a timeout,
and bounded output capture; model-written Python additionally runs in a
throw-away working directory under resource limits, and TeX may only read and
write inside its working directory.  That contains accidents, but it is
**not** a security sandbox: the code still runs as your user with network
access.  Use the Modal backend for isolation.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from paper_orchestra.compute.base import (
    CompileResult,
    ComputeBackend,
    ComputeError,
    ExecResult,
    refuse_unsafe_document,
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


def safe_relative(name: str) -> Path:
    """Reject absolute paths and ``..`` in caller-supplied file names."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ComputeError(f"unsafe file name: {name!r}")
    return path


def _limit_resources(cpu_seconds: int, memory_bytes: int | None):  # pragma: no cover - child
    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        if memory_bytes:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

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


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    else:  # pragma: no cover - non-posix
        with contextlib.suppress(OSError):
            process.kill()


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
    posix = os.name == "posix"
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=posix,
        preexec_fn=preexec if posix else None,
    )
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


class LocalBackend(ComputeBackend):
    name = "local"

    def __init__(self, *, python: str | None = None, memory_limit_mb: int | None = 4096) -> None:
        self.python = python or sys.executable
        self.memory_limit_mb = memory_limit_mb
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
        if not self.latex_available():
            raise ComputeError(
                "pdflatex/bibtex not found on PATH. Install TeX Live, or run with "
                "--compute modal to compile in a Modal sandbox."
            )

        produced = work / f"{tex_name}.pdf"
        produced.unlink(missing_ok=True)
        refused = refuse_unsafe_document(work, tex_name)
        if refused is not None:
            return refused
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

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "python": self.python,
            "pdflatex": shutil.which("pdflatex"),
            "bibtex": shutil.which("bibtex"),
            "isolation": "none (subprocess with a minimal environment)",
        }
