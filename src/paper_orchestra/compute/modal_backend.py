"""Run model-written Python and LaTeX builds in Modal Sandboxes.

Model-written figure code is untrusted, so it never runs next to credentials.
A backend owns up to two Sandboxes (one for Python jobs, one for LaTeX builds).
Each is created on first use with outbound networking blocked, no Secrets and
no Modal identity token, is reused for later jobs (every job gets its own
scratch directory) and is replaced transparently once it has expired or been
collected for idleness.

``latex="local"`` keeps the Sandbox for Python but compiles with the TeX Live
install of the current machine.  ``paper_orchestra.modal_app`` uses that mode
because its image already ships TeX Live.

The ``modal`` package is imported inside functions so that importing this
module, and the rest of the package, works without the ``modal`` extra.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from paper_orchestra.compute.base import CompileResult, ComputeBackend, ComputeError, ExecResult
from paper_orchestra.compute.local import LocalBackend, safe_relative
from paper_orchestra.utils.latex import latex_commands, parse_latex_log

if TYPE_CHECKING:
    from types import ModuleType

    import modal

log = logging.getLogger(__name__)

APP_ENV = "PAPER_ORCHESTRA_MODAL_APP"
DEFAULT_APP_NAME = "paper-orchestra"
INSTALL_HINT = "Install the Modal SDK with `uv sync --extra modal`."

PYTHON_VERSION = "3.12"
PYTHON_PACKAGES = ("numpy>=1.26", "matplotlib>=3.8", "pandas", "seaborn", "scipy")

#: Covers the bundled templates and the usual conference styles.  ``plain-generic``
#: (soul, ulem) is only a Recommends of latex-extra, so it is listed explicitly.
TEXLIVE_PACKAGES = (
    "texlive-latex-base",
    "texlive-latex-recommended",
    "texlive-latex-extra",
    "texlive-fonts-recommended",
    "texlive-science",
    "texlive-bibtex-extra",
    "texlive-publishers",
    "texlive-plain-generic",
    "cm-super",
    "latexmk",
)
#: ``Image.apt_install`` cannot pass ``--no-install-recommends``; with recommends
#: the TeX Live layer roughly doubles in size.
TEXLIVE_INSTALL_COMMAND = (
    "apt-get update && apt-get install -y --no-install-recommends "
    + " ".join(TEXLIVE_PACKAGES)
    + " && rm -rf /var/lib/apt/lists/*"
)

SandboxKind = Literal["python", "latex"]
LatexMode = Literal["sandbox", "local"]

_ROOT = "/work"
_SCRIPT = "_script.py"
#: Hidden, so it can never collide with an uploaded file (hidden files are not uploaded).
_ARCHIVE = ".paper-orchestra-upload.tar"
_MAX_CAPTURE = 200_000
_MAX_FILE_BYTES = 50 * 1024 * 1024
#: Seconds between SIGTERM and SIGKILL when a job overruns its timeout.
_KILL_GRACE = 5
#: Modal's own exec deadline is only a backstop behind coreutils ``timeout``.
_EXEC_MARGIN = 15
#: Upload, download and round-trip time a job needs on top of its own timeout.
_JOB_OVERHEAD = 120
_UNPACK_TIMEOUT = 120
_MIN_LIFETIME = 300
_MAX_LIFETIME = 24 * 60 * 60
_LATEX_ARTIFACTS = (".log", ".blg")
#: Paranoid file access: TeX may only read and write inside the working directory
#: (and read its own distribution).
_TEX_ENV = {"openout_any": "p", "openin_any": "p"}

T = TypeVar("T")


def _import_modal() -> ModuleType:
    try:
        import modal
    except ModuleNotFoundError as exc:
        if exc.name != "modal":
            raise
        raise ComputeError(f"The Modal compute backend is not available. {INSTALL_HINT}") from exc
    return modal


def modal_token_configured() -> bool:
    """Whether the Modal SDK would find credentials.

    Reads the environment and ``~/.modal.toml`` only; nothing is sent anywhere.
    """
    try:
        from modal.config import config
    except ModuleNotFoundError:
        return False
    except Exception as exc:  # a malformed ~/.modal.toml raises at import
        log.warning("Could not read the Modal configuration: %s", exc)
        return False
    if config.get("token_id") and config.get("token_secret"):
        return True
    return bool(
        config.get("oauth_refresh_token")
        and config.get("oauth_client_id")
        and (config.get("oauth_client_secret") or config.get("oauth_jwt_key"))
    )


def python_image() -> modal.Image:
    """Image for model-written figure code: scientific Python and nothing else."""
    modal = _import_modal()
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .uv_pip_install(*PYTHON_PACKAGES)
        .env({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/opt/matplotlib"})
        # Importing pyplot once bakes the font cache, so no sandbox pays for rebuilding it.
        .run_commands("python -c 'import matplotlib.pyplot'")
    )


def latex_image() -> modal.Image:
    """Image with TeX Live for compiling papers."""
    modal = _import_modal()
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .run_commands(TEXLIVE_INSTALL_COMMAND)
        .env({"TEXMFVAR": "/tmp/texmf-var"})
    )


def pack_tree(root: Path, archive: Path, *, max_file_bytes: int) -> list[str]:
    """Write a tar of ``root`` to ``archive`` and return the relative paths left out.

    Hidden files and directories are omitted silently.  Files larger than
    ``max_file_bytes`` and symlinked directories are omitted and reported.
    """
    skipped: list[str] = []
    with tarfile.open(archive, "w", dereference=True) as tar:
        for current, dirs, files in os.walk(root):
            here = Path(current)
            visible = sorted(name for name in dirs if not name.startswith("."))
            skipped.extend(
                (here / name).relative_to(root).as_posix()
                for name in visible
                if (here / name).is_symlink()
            )
            dirs[:] = [name for name in visible if not (here / name).is_symlink()]
            for name in dirs:
                path = here / name
                tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
            for name in sorted(files):
                path = here / name
                if name.startswith(".") or not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                if path.stat().st_size > max_file_bytes:
                    skipped.append(relative)
                    continue
                tar.add(path, arcname=relative, recursive=False)
    return skipped


def _seconds(value: float) -> int:
    # Modal timeouts are uint32 protobuf fields: a float raises TypeError.
    return max(1, math.ceil(value))


def _decode(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    return text if len(text) <= _MAX_CAPTURE else text[-_MAX_CAPTURE:]


class _SandboxLost(Exception):
    """The sandbox stopped (lifetime, idle timeout, termination) while a job was using it."""


@dataclass
class _Completed:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass
class _Slot:
    lock: threading.Lock = field(default_factory=threading.Lock)
    sandbox: Any = None
    #: ``time.monotonic()`` value at which the sandbox reaches its lifetime.
    expires_at: float = 0.0


class ModalBackend(ComputeBackend):
    """Compute backend backed by Modal Sandboxes.

    ``cpu`` (cores) and ``memory_mb`` (MiB) are hard limits of each sandbox.
    ``sandbox_lifetime`` is the maximum age of a sandbox and ``idle_timeout``
    how long it may sit unused before Modal collects it; both are in seconds
    and bound what a crashed orchestrator can leave running.
    """

    name = "modal"

    def __init__(
        self,
        *,
        app_name: str | None = None,
        latex: LatexMode = "sandbox",
        cpu: float = 2.0,
        memory_mb: int = 4096,
        sandbox_lifetime: float = 3600.0,
        idle_timeout: float = 600.0,
    ) -> None:
        if latex not in ("sandbox", "local"):
            raise ValueError(f"latex must be 'sandbox' or 'local', got {latex!r}")
        if cpu <= 0:
            raise ValueError(f"cpu must be positive, got {cpu!r}")
        if memory_mb < 256:
            raise ValueError(f"memory_mb must be at least 256, got {memory_mb!r}")
        if not _MIN_LIFETIME <= sandbox_lifetime <= _MAX_LIFETIME:
            raise ValueError(
                f"sandbox_lifetime must be between {_MIN_LIFETIME} and {_MAX_LIFETIME} seconds, "
                f"got {sandbox_lifetime!r}"
            )
        if not 0 < idle_timeout <= sandbox_lifetime:
            raise ValueError(
                f"idle_timeout must be positive and at most sandbox_lifetime, got {idle_timeout!r}"
            )
        if latex == "local" and not LocalBackend.latex_available():
            raise ComputeError(
                "latex='local' needs pdflatex and bibtex on PATH. Install TeX Live, or use "
                "latex='sandbox' to compile in a Modal sandbox."
            )
        self._modal = _import_modal()
        self.app_name = app_name or os.environ.get(APP_ENV) or DEFAULT_APP_NAME
        self.latex: LatexMode = latex
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.sandbox_lifetime = sandbox_lifetime
        self.idle_timeout = idle_timeout
        self._local = LocalBackend() if latex == "local" else None
        self._slots: dict[SandboxKind, _Slot] = {"python": _Slot(), "latex": _Slot()}
        self._state_lock = threading.Lock()
        self._app: Any = None
        #: Sandboxes replaced before their lifetime ran out; other threads may still be
        #: finishing jobs on them, so they are only terminated in ``close``.
        self._retired: list[Any] = []

    # -- python ----------------------------------------------------------------

    def run_python(
        self,
        code: str,
        *,
        inputs: Mapping[str, bytes] | None = None,
        outputs: Sequence[str] = (),
        timeout: float = 120.0,
    ) -> ExecResult:
        files = {safe_relative(name).as_posix(): data for name, data in (inputs or {}).items()}
        wanted = [(name, safe_relative(name).as_posix()) for name in outputs]
        limit = _seconds(timeout)

        def job(sandbox: Any) -> ExecResult:
            job_dir = f"{_ROOT}/py-{uuid.uuid4().hex[:12]}"
            filesystem = sandbox.filesystem
            try:
                for relative, data in files.items():
                    filesystem.write_bytes(data, f"{job_dir}/{relative}")
                filesystem.write_text(code, f"{job_dir}/{_SCRIPT}")
                # -I: isolated mode, so PYTHON* variables and the user site are ignored.
                done = self._exec(
                    sandbox,
                    ["python", "-I", "-B", _SCRIPT],
                    cwd=job_dir,
                    limit=limit,
                    env={"HOME": job_dir, "TMPDIR": job_dir},
                )
                produced: dict[str, bytes] = {}
                notes: list[str] = []
                for name, relative in wanted:
                    self._collect(filesystem, f"{job_dir}/{relative}", name, produced, notes)
            finally:
                self._remove(sandbox, job_dir)
            return ExecResult(
                returncode=done.returncode,
                stdout=done.stdout,
                stderr=done.stderr + "".join(f"\n[paper-orchestra] {note}" for note in notes),
                timed_out=done.timed_out,
                files=produced,
            )

        return self._with_sandbox("python", limit, job)

    def _collect(
        self,
        filesystem: Any,
        remote: str,
        name: str,
        produced: dict[str, bytes],
        notes: list[str],
    ) -> None:
        """Fetch one requested output file if it exists and is small enough to hold in memory."""
        try:
            info = filesystem.stat(remote)
            if not info.is_file():
                return
            if info.size > _MAX_FILE_BYTES:
                notes.append(
                    f"output file {name!r} is {info.size / 1e6:.0f} MB, above the "
                    f"{_MAX_FILE_BYTES // (1024 * 1024)} MB limit, and was not returned."
                )
                return
            produced[name] = filesystem.read_bytes(remote)
        except self._modal.exception.SandboxFilesystemNotFoundError:
            return

    # -- latex -----------------------------------------------------------------

    def compile_latex(
        self,
        workdir: str | Path,
        tex_name: str,
        *,
        output_pdf: str | Path | None = None,
        timeout: float = 240.0,
    ) -> CompileResult:
        if self._local is not None:
            return self._local.compile_latex(
                workdir, tex_name, output_pdf=output_pdf, timeout=timeout
            )
        work = Path(workdir)
        source = work / safe_relative(f"{tex_name}.tex")
        if not source.is_file():
            raise ComputeError(f"{source} does not exist")
        produced = work / f"{tex_name}.pdf"
        target = Path(output_pdf) if output_pdf else produced
        limit = _seconds(timeout)

        def job(sandbox: Any) -> CompileResult:
            job_dir = f"{_ROOT}/tex-{uuid.uuid4().hex[:12]}"
            filesystem = sandbox.filesystem
            # A PDF left over from an earlier build must not pass for this one's output.
            produced.unlink(missing_ok=True)
            try:
                self._upload_tree(sandbox, work, job_dir)
                deadline = time.monotonic() + limit
                chunks: list[str] = []
                timed_out = False
                for command in latex_commands(tex_name):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    done = self._exec(
                        sandbox,
                        command,
                        cwd=job_dir,
                        limit=_seconds(remaining),
                        env=_TEX_ENV,
                    )
                    if done.returncode in (126, 127) and not done.timed_out:
                        raise ComputeError(
                            f"could not run {command[0]} in the Modal sandbox: "
                            f"{done.stderr.strip() or 'command not found'}"
                        )
                    chunks.append(f"$ {' '.join(command)}\n{done.stdout}{done.stderr}")
                    if done.timed_out:
                        timed_out = True
                        break
                pdf_path: Path | None = None
                if not timed_out and self._download(
                    filesystem, f"{job_dir}/{tex_name}.pdf", target
                ):
                    pdf_path = target
                for suffix in _LATEX_ARTIFACTS:
                    self._download(
                        filesystem, f"{job_dir}/{tex_name}{suffix}", work / f"{tex_name}{suffix}"
                    )
            finally:
                self._remove(sandbox, job_dir)
            output = "\n".join(chunks)
            errors, warnings = parse_latex_log(output)
            return CompileResult(
                ok=pdf_path is not None,
                pdf_path=pdf_path,
                errors=errors,
                warnings=warnings,
                log_tail=output[-6000:],
                timed_out=timed_out,
            )

        return self._with_sandbox("latex", limit, job)

    def _upload_tree(self, sandbox: Any, work: Path, job_dir: str) -> None:
        """Copy the visible files of ``work`` into ``job_dir`` with one upload."""
        remote_archive = f"{job_dir}/{_ARCHIVE}"
        with tempfile.TemporaryDirectory(prefix="paper-orchestra-upload-") as tmp:
            archive = Path(tmp) / "workdir.tar"
            skipped = pack_tree(work, archive, max_file_bytes=_MAX_FILE_BYTES)
            if skipped:
                log.warning(
                    "Not uploaded to the Modal sandbox (larger than %d MB or a symlinked "
                    "directory): %s",
                    _MAX_FILE_BYTES // (1024 * 1024),
                    ", ".join(skipped),
                )
            sandbox.filesystem.copy_from_local(archive, remote_archive)
        done = self._exec(
            sandbox, ["tar", "-xf", _ARCHIVE, "-C", "."], cwd=job_dir, limit=_UNPACK_TIMEOUT
        )
        if done.returncode != 0:
            raise ComputeError(
                "could not unpack the LaTeX workspace in the Modal sandbox: "
                f"{(done.stderr or done.stdout).strip()[-500:] or f'tar exited {done.returncode}'}"
            )

    def _download(self, filesystem: Any, remote: str, local: Path) -> bool:
        try:
            filesystem.copy_to_local(remote, local)
        except self._modal.exception.SandboxFilesystemNotFoundError:
            return False
        return True

    # -- sandbox plumbing --------------------------------------------------------

    def _exec(
        self,
        sandbox: Any,
        command: Sequence[str],
        *,
        cwd: str,
        limit: int,
        env: Mapping[str, str] | None = None,
    ) -> _Completed:
        """Run ``command`` under coreutils ``timeout`` and classify how it ended."""
        started = time.monotonic()
        process = sandbox.exec(
            "timeout",
            "--signal=TERM",
            f"--kill-after={_KILL_GRACE}",
            str(limit),
            *command,
            workdir=cwd,
            env=dict(env) if env else None,
            timeout=limit + _KILL_GRACE + _EXEC_MARGIN,
            # Modal's text mode decodes strictly and LaTeX logs are not always UTF-8.
            text=False,
        )
        stdout = _decode(process.stdout.read())
        stderr = _decode(process.stderr.read())
        returncode = process.wait()
        elapsed = time.monotonic() - started
        # -1: Modal's exec deadline fired.  137: SIGKILL, from `timeout`, the OOM killer or
        # the sandbox itself going away, which must not be reported as the job's own failure.
        if returncode in (-1, 137) and self._is_dead(sandbox):
            raise _SandboxLost("the sandbox stopped while the command was running")
        timed_out = returncode == -1 or (returncode in (124, 137) and elapsed >= limit)
        return _Completed(returncode, stdout, stderr, timed_out)

    def _with_sandbox(self, kind: SandboxKind, limit: int, job: Callable[[Any], T]) -> T:
        """Run ``job`` on the sandbox for ``kind``; retry once if that sandbox is gone."""
        retried = False
        while True:
            sandbox = self._acquire(kind, limit + _JOB_OVERHEAD)
            try:
                return job(sandbox)
            except (_SandboxLost, *self._transport_errors) as exc:
                if isinstance(exc, _SandboxLost) or self._is_dead(sandbox):
                    self._discard(kind, sandbox)
                    if not retried:
                        retried = True
                        log.warning(
                            "The Modal sandbox for %s jobs is no longer running (%s); "
                            "retrying on a fresh one.",
                            kind,
                            exc,
                        )
                        continue
                raise ComputeError(
                    f"Modal sandbox {kind} job failed: {type(exc).__name__}: {exc}"
                ) from exc

    def _acquire(self, kind: SandboxKind, needed: int) -> Any:
        if needed > self.sandbox_lifetime:
            raise ComputeError(
                f"a {kind} job with this timeout needs up to {needed} s, more than the sandbox "
                f"lifetime of {self.sandbox_lifetime:.0f} s; raise sandbox_lifetime"
            )
        slot = self._slots[kind]
        with slot.lock:
            if slot.sandbox is not None and slot.expires_at - time.monotonic() < needed:
                log.info("Replacing the Modal sandbox for %s jobs before it expires.", kind)
                with self._state_lock:
                    self._retired.append(slot.sandbox)
                slot.sandbox = None
            if slot.sandbox is None:
                slot.sandbox = self._create(kind)
                slot.expires_at = time.monotonic() + self.sandbox_lifetime
            return slot.sandbox

    def _create(self, kind: SandboxKind) -> Any:
        modal = self._modal
        log.info(
            "Starting a Modal sandbox for %s jobs (the first use builds its image, which "
            "can take several minutes).",
            kind,
        )
        try:
            return modal.Sandbox.create(
                app=self._lookup_app(),
                image=python_image() if kind == "python" else latex_image(),
                timeout=_seconds(self.sandbox_lifetime),
                idle_timeout=_seconds(self.idle_timeout),
                cpu=(min(0.5, self.cpu), self.cpu),
                memory=(min(1024, self.memory_mb), self.memory_mb),
                # Untrusted code: no way out, and nothing worth taking.  No `secrets=` either.
                block_network=True,
                include_oidc_identity_token=False,
            )
        except modal.exception.AuthError as exc:
            raise ComputeError(
                "Modal is not authenticated. Run `modal token new`, or set MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET."
            ) from exc
        except self._transport_errors as exc:
            raise ComputeError(
                f"could not start a Modal sandbox for {kind} jobs: {type(exc).__name__}: {exc}"
            ) from exc

    def _lookup_app(self) -> Any:
        """The App that owns the sandboxes; ``None`` selects the running Function's App."""
        if not self._modal.is_local():
            return None
        with self._state_lock:
            if self._app is None:
                self._app = self._modal.App.lookup(self.app_name, create_if_missing=True)
            return self._app

    @property
    def _transport_errors(self) -> tuple[type[BaseException], ...]:
        """What a Modal call raises when the sandbox or the connection to it fails."""
        return (self._modal.Error, TimeoutError, ConnectionError)

    def _is_dead(self, sandbox: Any) -> bool:
        try:
            return sandbox.poll() is not None
        except self._transport_errors:
            return True

    def _discard(self, kind: SandboxKind, sandbox: Any) -> None:
        slot = self._slots[kind]
        with slot.lock:
            if slot.sandbox is sandbox:
                slot.sandbox = None
        self._terminate(sandbox)

    def _terminate(self, sandbox: Any) -> None:
        try:
            sandbox.terminate()
        except self._transport_errors as exc:
            log.warning("Could not terminate a Modal sandbox: %s", exc)
        # Releases the client-side connection; the sandbox itself is already going away.
        with contextlib.suppress(*self._transport_errors):
            sandbox.detach()

    def _remove(self, sandbox: Any, job_dir: str) -> None:
        """Best-effort cleanup of a job directory; the sandbox is disposable anyway."""
        try:
            sandbox.filesystem.remove(job_dir, recursive=True)
        except self._transport_errors as exc:
            log.debug("Could not remove %s in the Modal sandbox: %s", job_dir, exc)

    # -- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        sandboxes: list[Any] = []
        for slot in self._slots.values():
            with slot.lock:
                if slot.sandbox is not None:
                    sandboxes.append(slot.sandbox)
                    slot.sandbox = None
        with self._state_lock:
            sandboxes.extend(self._retired)
            self._retired.clear()
        for sandbox in sandboxes:
            self._terminate(sandbox)

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": self.name,
            "app": self.app_name,
            "modal_version": getattr(self._modal, "__version__", None),
            "token_configured": modal_token_configured(),
            "inside_modal_function": not self._modal.is_local(),
            "isolation": "Modal Sandbox (network blocked, no secrets, no identity token)",
            "latex": self.latex,
            "cpu_limit": self.cpu,
            "memory_limit_mb": self.memory_mb,
            "sandbox_lifetime_s": self.sandbox_lifetime,
            "idle_timeout_s": self.idle_timeout,
            "sandboxes": {
                kind: getattr(slot.sandbox, "object_id", None) for kind, slot in self._slots.items()
            },
        }
        if self._local is not None:
            local = self._local.describe()
            info["pdflatex"] = local["pdflatex"]
            info["bibtex"] = local["bibtex"]
        return info
