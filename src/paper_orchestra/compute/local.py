"""Run jobs as subprocesses on this machine.

This backend needs TeX Live on ``PATH`` for LaTeX and gives model-written
Python a scrubbed environment, a throw-away working directory, resource limits
and a timeout.  That keeps credentials out of the child's environment and
contains accidents, but it is **not** a security sandbox: the code still runs
as your user with network access.  Use the Modal backend for isolation.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper_orchestra.compute.base import CompileResult, ComputeBackend, ComputeError, ExecResult
from paper_orchestra.utils.latex import latex_commands, parse_latex_log

_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TZ")
_TEX_ENV_TO_DROP = ("TEXINPUTS", "TEXMFHOME", "TEXMF", "TEXMFVAR", "TEXMFCONFIG", "BIBINPUTS")
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


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    preexec: Any = None,
) -> tuple[int, str, str, bool]:
    """Run ``command`` in its own process group; kill the group on timeout."""
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
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if posix:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - non-posix
            process.kill()
        out, err = process.communicate()

    def decode(data: bytes) -> str:
        text = data.decode("utf-8", errors="replace")
        return text if len(text) <= _MAX_CAPTURE else text[-_MAX_CAPTURE:]

    return process.returncode, decode(out), decode(err), timed_out


class LocalBackend(ComputeBackend):
    name = "local"

    def __init__(self, *, python: str | None = None, memory_limit_mb: int | None = 4096) -> None:
        self.python = python or sys.executable
        self.memory_limit_mb = memory_limit_mb

    # -- python ----------------------------------------------------------------

    def run_python(
        self,
        code: str,
        *,
        inputs: Mapping[str, bytes] | None = None,
        outputs: Sequence[str] = (),
        timeout: float = 120.0,
    ) -> ExecResult:
        with tempfile.TemporaryDirectory(prefix="paper-orchestra-exec-") as tmp:
            workdir = Path(tmp)
            for name, data in (inputs or {}).items():
                target = workdir / safe_relative(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            script = workdir / "_script.py"
            script.write_text(code, encoding="utf-8")

            preexec = None
            if os.name == "posix":
                memory = self.memory_limit_mb * 1024 * 1024 if self.memory_limit_mb else None
                preexec = _limit_resources(int(timeout) + 5, memory)
            try:
                # -I: isolated mode, so PYTHON* variables and the user site are ignored.
                returncode, out, err, timed_out = _run(
                    [self.python, "-I", "-B", script.name],
                    cwd=workdir,
                    env=scrubbed_env(workdir),
                    timeout=timeout,
                    preexec=preexec,
                )
            except OSError as exc:
                raise ComputeError(f"could not start python: {exc}") from exc

            files: dict[str, bytes] = {}
            for name in outputs:
                produced = workdir / safe_relative(name)
                if produced.is_file():
                    files[name] = produced.read_bytes()
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
        env = {k: v for k, v in os.environ.items() if k not in _TEX_ENV_TO_DROP}
        # Paranoid output: TeX may only write inside the working directory.
        env["openout_any"] = "p"

        deadline = time.monotonic() + timeout
        chunks: list[str] = []
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
            chunks.append(f"$ {' '.join(command)}\n{out}{err}")
            if step_timed_out:
                timed_out = True
                break

        output = "\n".join(chunks)
        errors, warnings = parse_latex_log(output)
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
            log_tail=output[-6000:],
            timed_out=timed_out,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "python": self.python,
            "pdflatex": shutil.which("pdflatex"),
            "bibtex": shutil.which("bibtex"),
            "isolation": "none (subprocess with scrubbed environment)",
        }
