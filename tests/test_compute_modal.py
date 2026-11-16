from __future__ import annotations

import contextlib
import errno
import inspect
import io
import itertools
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import tracemalloc
import types
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import paper_orchestra.compute.base as base
import paper_orchestra.compute.modal_backend as modal_backend
from paper_orchestra.compute import (
    BACKEND_NAMES,
    ComputeBackend,
    ComputeError,
    JobResult,
    JobSpec,
    get_backend,
)
from paper_orchestra.compute.base import BackendUnusable
from paper_orchestra.compute.local import DirectoryInUse, LocalBackend, lock_directory
from paper_orchestra.compute.modal_backend import (
    TEX_PACKAGES_ENV,
    ModalBackend,
    extra_tex_packages,
    modal_token_configured,
    pack_tree,
    texlive_install_command,
)
from paper_orchestra.utils.latex import latex_commands

needs_coreutils = pytest.mark.skipif(
    shutil.which("timeout") is None or shutil.which("tar") is None,
    reason="the fake sandbox runs commands with coreutils timeout and tar",
)

Handler = Callable[["FakeSandbox", list[str], Path, dict[str, str]], "FakeProcess | None"]

BANNER = b"This is pdfTeX, Version 3.141592653-2.6-1.40.25 (TeX Live 2024)\n"


class FakeStream:
    """An output stream of a process: iterating it yields the output in chunks."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self._data), 7):
            yield self._data[start : start + 7]


class FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self.stdout = FakeStream(stdout)
        self.stderr = FakeStream(stderr)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def kill_session(session: int) -> None:
    """Kill every process of a session.  coreutils timeout moves its command to a process
    group of its own, so the group of the shell that was started is not enough."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(session, signal.SIGKILL)
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            # "pid (name) state ppid pgrp session ...", and the name may hold anything.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == session:
                os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, IndexError, ValueError):
            continue


class LiveStream:
    """The output of a process that is still running: iterating blocks until more arrives."""

    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe

    def __iter__(self) -> Iterator[bytes]:
        while True:
            data = os.read(self._pipe.fileno(), 4096)
            if not data:
                return
            yield data


class LiveProcess:
    """A command that `Sandbox.exec` started and that runs on while the caller reads it."""

    def __init__(self, popen: subprocess.Popen) -> None:
        self.popen = popen
        self.stdout = LiveStream(popen.stdout)
        self.stderr = LiveStream(popen.stderr)

    def wait(self) -> int:
        code = self.popen.wait()
        return code if code >= 0 else 128 - code


class FakeFileInfo:
    def __init__(self, path: Path) -> None:
        self.size = path.stat().st_size
        self._file = path.is_file()

    def is_file(self) -> bool:
        return self._file


class FakeImage:
    def __init__(self, steps: list[tuple[str, tuple, dict]]) -> None:
        self.steps = steps

    def _then(self, name: str, args: tuple, kwargs: dict) -> FakeImage:
        return FakeImage([*self.steps, (name, args, kwargs)])

    def uv_pip_install(self, *args: Any, **kwargs: Any) -> FakeImage:
        return self._then("Image.uv_pip_install", args, kwargs)

    def run_commands(self, *args: Any, **kwargs: Any) -> FakeImage:
        return self._then("Image.run_commands", args, kwargs)

    def env(self, *args: Any, **kwargs: Any) -> FakeImage:
        return self._then("Image.env", args, kwargs)

    def named(self, name: str) -> list[tuple]:
        return [args for step, args, _ in self.steps if step == name]


class FakeFilesystem:
    """`sandbox.filesystem` backed by a directory on this machine."""

    def __init__(self, sandbox: FakeSandbox) -> None:
        self._sandbox = sandbox

    def _path(self, remote_path: str) -> Path:
        self._sandbox.require_running()
        if not remote_path.startswith("/"):
            raise self._sandbox.hub.exception.InvalidError(f"path must be absolute: {remote_path}")
        return self._sandbox.root / remote_path.lstrip("/")

    def _existing(self, remote_path: str) -> Path:
        path = self._path(remote_path)
        if not path.exists():
            raise self._sandbox.hub.exception.SandboxFilesystemNotFoundError(remote_path)
        return path

    def write_bytes(self, data: bytes, remote_path: str) -> None:
        self._sandbox.hub.record("filesystem.write_bytes", data, remote_path)
        path = self._path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(data))

    def write_text(self, data: str, remote_path: str) -> None:
        self._sandbox.hub.record("filesystem.write_text", data, remote_path)
        path = self._path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")

    def read_bytes(self, remote_path: str) -> bytes:
        self._sandbox.hub.record("filesystem.read_bytes", remote_path)
        return self._existing(remote_path).read_bytes()

    def stat(self, remote_path: str) -> FakeFileInfo:
        self._sandbox.hub.record("filesystem.stat", remote_path)
        return FakeFileInfo(self._existing(remote_path))

    def copy_from_local(self, local_path: str | os.PathLike, remote_path: str) -> None:
        self._sandbox.hub.record("filesystem.copy_from_local", local_path, remote_path)
        path = self._path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, path)

    def copy_to_local(self, remote_path: str, local_path: str | os.PathLike) -> None:
        self._sandbox.hub.record("filesystem.copy_to_local", remote_path, local_path)
        source = self._existing(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)

    def remove(self, remote_path: str, *, recursive: bool = False) -> None:
        self._sandbox.hub.record("filesystem.remove", remote_path, recursive=recursive)
        path = self._existing(remote_path)
        if path.is_dir():
            if not recursive:
                raise self._sandbox.hub.exception.SandboxFilesystemError("directory not empty")
            shutil.rmtree(path)
        else:
            path.unlink()


class FakeSandbox:
    def __init__(self, hub: FakeModal, index: int, kwargs: dict[str, Any]) -> None:
        self.hub = hub
        self.object_id = f"sb-fake-{index}"
        self.kwargs = kwargs
        self.root = hub.root / self.object_id
        self.root.mkdir(parents=True)
        self.filesystem = FakeFilesystem(self)
        self.exit_code: int | None = None
        self.terminated = False
        self.detached = False
        self.commands: list[list[str]] = []
        self.exec_kwargs: list[dict[str, Any]] = []
        #: Commands started through a shell, which run on after `exec` has returned.
        self.live: list[subprocess.Popen] = []

    def require_attached(self) -> None:
        if self.detached:
            raise self.hub.exception.ClientClosed(
                "Unable to perform operation on a detached sandbox"
            )

    def require_running(self) -> None:
        self.require_attached()
        if self.exit_code is not None:
            raise self.hub.exception.NotFoundError("The Sandbox is unavailable")

    def stop_processes(self) -> None:
        """What the end of a sandbox does to everything that runs in it."""
        for popen in self.live:
            kill_session(popen.pid)
            popen.wait()

    def exec(self, *args: str, **kwargs: Any) -> FakeProcess | LiveProcess:
        self.hub.record("Sandbox.exec", *args, **kwargs)
        self.require_running()
        if not isinstance(kwargs.get("timeout"), int):
            raise TypeError("'float' object cannot be interpreted as an integer")
        # A shell command is one that outlives the call (a job, or the reading of its
        # log); everything else runs under coreutils timeout and is waited for.
        live = args[0] == "sh"
        if not live:
            assert args[:3] == ("timeout", "--signal=TERM", "--kill-after=5"), args
        command = list(args) if live else list(args[4:])
        self.commands.append(command)
        self.exec_kwargs.append(kwargs)
        cwd = self.root / kwargs["workdir"].lstrip("/")
        if not cwd.is_dir():
            raise self.hub.exception.InvalidError(f"workdir does not exist: {kwargs['workdir']}")
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(self.root)}
        for key, value in (kwargs.get("env") or {}).items():
            env[key] = str(self.root / value.lstrip("/")) if value.startswith("/") else value
        for handler in self.hub.handlers:
            process = handler(self, command, cwd, env)
            if process is not None:
                return process
        if live:
            popen = subprocess.Popen(
                [sys.executable if a == "python" else a for a in args],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self.live.append(popen)
            return LiveProcess(popen)
        argv = [*args[:4], *(sys.executable if a == "python" else a for a in command)]
        try:
            done = subprocess.run(
                argv, cwd=cwd, env=env, capture_output=True, timeout=kwargs["timeout"]
            )
        except subprocess.TimeoutExpired as exc:
            return FakeProcess(exc.stdout or b"", exc.stderr or b"", -1)
        code = done.returncode if done.returncode >= 0 else 128 - done.returncode
        return FakeProcess(done.stdout, done.stderr, code)

    def poll(self) -> int | None:
        self.hub.record("Sandbox.poll")
        self.require_attached()
        return self.exit_code

    def terminate(self, *, wait: bool = False) -> None:
        self.hub.record("Sandbox.terminate", wait=wait)
        self.require_attached()
        self.terminated = True
        if self.exit_code is None:
            self.exit_code = 137
        self.stop_processes()

    def detach(self) -> None:
        self.hub.record("Sandbox.detach")
        self.detached = True


class FakeModal:
    """Stands in for the `modal` package: records every call and simulates sandboxes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, tuple, dict]] = []
        self.sandboxes: list[FakeSandbox] = []
        self.handlers: list[Handler] = []
        self.local = True
        self.settings: dict[str, str] = {}
        self.lookup_error: Exception | None = None
        self.create_error: Exception | None = None
        self.from_id_error: Exception | None = None
        #: When set, `Sandbox.create` blocks until the event is: an image is being built.
        self.create_gate: threading.Event | None = None

        exception = types.ModuleType("modal.exception")

        class Error(Exception):
            pass

        for name in (
            "AuthError",
            "ClientClosed",
            "InvalidError",
            "NotFoundError",
            "SandboxFilesystemError",
        ):
            setattr(exception, name, type(name, (Error,), {}))
        exception.Error = Error
        exception.SandboxFilesystemNotFoundError = type(
            "SandboxFilesystemNotFoundError", (exception.SandboxFilesystemError,), {}
        )
        self.exception = exception

        config = types.ModuleType("modal.config")
        config.config = types.SimpleNamespace(get=lambda key: self.settings.get(key))

        module = types.ModuleType("modal")
        module.__version__ = "0.0-fake"
        module.Error = Error
        module.exception = exception
        module.config = config
        module.is_local = lambda: self.local
        module.App = types.SimpleNamespace(lookup=self._lookup)
        module.Sandbox = types.SimpleNamespace(create=self._create, from_id=self._from_id)
        module.Image = types.SimpleNamespace(debian_slim=self._debian_slim)
        self.module = module
        self.modules = {"modal": module, "modal.exception": exception, "modal.config": config}

    def record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def named(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for call, args, kwargs in self.calls if call == name]

    def _lookup(self, *args: Any, **kwargs: Any) -> str:
        self.record("App.lookup", *args, **kwargs)
        if self.lookup_error is not None:
            raise self.lookup_error
        return f"app:{args[0]}"

    def _debian_slim(self, *args: Any, **kwargs: Any) -> FakeImage:
        return FakeImage([("Image.debian_slim", args, kwargs)])

    def _create(self, *args: Any, **kwargs: Any) -> FakeSandbox:
        self.record("Sandbox.create", *args, **kwargs)
        for key in ("timeout", "idle_timeout"):
            if not isinstance(kwargs.get(key), int):
                raise TypeError("'float' object cannot be interpreted as an integer")
        if self.create_gate is not None:
            assert self.create_gate.wait(30)
        if self.create_error is not None:
            raise self.create_error
        sandbox = FakeSandbox(self, len(self.sandboxes), kwargs)
        self.sandboxes.append(sandbox)
        return sandbox

    def _from_id(self, *args: Any, **kwargs: Any) -> FakeSandbox:
        """A new handle on a sandbox, as another process would get it."""
        self.record("Sandbox.from_id", *args, **kwargs)
        if self.from_id_error is not None:
            raise self.from_id_error
        for sandbox in self.sandboxes:
            if sandbox.object_id == args[0]:
                sandbox.detached = False
                return sandbox
        raise self.exception.NotFoundError(f"Sandbox {args[0]} not found")


@pytest.fixture
def real_sdk() -> types.SimpleNamespace:
    """The installed Modal SDK.  Request it before ``fake`` so the real package is imported."""
    modal = pytest.importorskip("modal")
    from modal.container_process import ContainerProcess
    from modal.io_streams import StreamReader
    from modal.sandbox.sandbox import SandboxFilesystem
    from modal.types import FileInfo

    return types.SimpleNamespace(
        modal=modal,
        ContainerProcess=ContainerProcess,
        StreamReader=StreamReader,
        SandboxFilesystem=SandboxFilesystem,
        FileInfo=FileInfo,
    )


@pytest.fixture(autouse=True)
def no_extra_tex_packages(monkeypatch) -> None:
    """The images are the default ones, whatever the shell that runs the tests has exported."""
    monkeypatch.delenv(TEX_PACKAGES_ENV, raising=False)


@pytest.fixture
def fake(monkeypatch, tmp_path) -> Iterator[FakeModal]:
    hub = FakeModal(tmp_path / "modal")
    for name, module in hub.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    yield hub
    for sandbox in hub.sandboxes:
        sandbox.stop_processes()


@pytest.fixture
def backend(fake) -> ModalBackend:
    return ModalBackend()


def scripted(**by_program: Callable[..., FakeProcess | None]) -> Handler:
    """A handler that answers commands by program name instead of running them."""

    def handler(sandbox, command, cwd, env):
        respond = by_program.get(command[0])
        return respond(sandbox, command, cwd, env) if respond else None

    return handler


def test_package_import_does_not_load_modal():
    code = (
        "import sys\n"
        "import paper_orchestra, paper_orchestra.compute, paper_orchestra.compute.modal_backend\n"
        "from paper_orchestra.compute import (CompileResult, ComputeBackend, ComputeError,\n"
        "    ExecResult, LocalBackend, ModalBackend, get_backend)\n"
        "assert issubclass(ModalBackend, ComputeBackend) and issubclass(LocalBackend, ComputeBackend)\n"
        "assert 'modal' not in sys.modules, 'modal was imported'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_get_backend_local_and_unknown_names():
    local = get_backend("local", memory_limit_mb=None)
    assert isinstance(local, LocalBackend) and local.memory_limit_mb is None
    assert BACKEND_NAMES == ("local", "modal")
    with pytest.raises(ComputeError, match=r"unknown compute backend 'docker'.*local, modal"):
        get_backend("docker")


def test_get_backend_modal_without_the_extra_gives_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    monkeypatch.setitem(sys.modules, "modal.config", None)
    with pytest.raises(ComputeError, match="uv sync --extra modal"):
        get_backend("modal")
    assert modal_token_configured() is False


def test_get_backend_modal_resolves_the_app_name(fake, monkeypatch):
    assert get_backend("modal").app_name == "paper-orchestra"
    monkeypatch.setenv("PAPER_ORCHESTRA_MODAL_APP", "from-env")
    assert get_backend("modal").app_name == "from-env"
    explicit = get_backend("modal", app_name="explicit", cpu=1.0, memory_mb=2048)
    assert isinstance(explicit, ModalBackend)
    assert (explicit.app_name, explicit.cpu, explicit.memory_mb) == ("explicit", 1.0, 2048)
    assert fake.calls == []  # nothing is created until a job needs it


@pytest.mark.parametrize(
    "options",
    [
        {"latex": "docker"},
        {"cpu": 0},
        {"memory_mb": 64},
        {"sandbox_lifetime": 30},
        {"sandbox_lifetime": 90_000},
        {"idle_timeout": 0},
        {"sandbox_lifetime": 600, "idle_timeout": 900},
    ],
)
def test_invalid_options_are_rejected(fake, options):
    with pytest.raises(ValueError):
        ModalBackend(**options)


@needs_coreutils
def test_run_python_round_trip_in_a_locked_down_sandbox(fake, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    backend = ModalBackend(cpu=1.5, memory_mb=3000, sandbox_lifetime=1800.5, idle_timeout=120.2)
    result = backend.run_python(
        "import os\n"
        "print(os.environ.get('ANTHROPIC_API_KEY'), os.environ.get('MODAL_IDENTITY_TOKEN'))\n"
        "print(open('data/in.txt').read())\n"
        "os.makedirs('out', exist_ok=True)\n"
        "open('out/result.bin', 'wb').write(bytes(range(256)))\n",
        inputs={"data/in.txt": b"payload"},
        outputs=["out/result.bin", "missing.txt"],
        timeout=30.4,
    )
    assert result.ok and result.returncode == 0 and not result.timed_out
    assert result.stdout.split() == ["None", "None", "payload"]
    assert result.files == {"out/result.bin": bytes(range(256))}

    assert fake.named("App.lookup") == [(("paper-orchestra",), {"create_if_missing": True})]
    [(args, created)] = fake.named("Sandbox.create")
    assert args == ("sleep", "infinity") and created["app"] == "app:paper-orchestra"
    assert created["block_network"] is True
    assert created["include_oidc_identity_token"] is False
    assert not created.get("secrets") and not created.get("env") and not created.get("volumes")
    assert not {k for k in created if "allowlist" in k or k.endswith("_ports")}
    assert (created["timeout"], created["idle_timeout"]) == (1801, 121)
    assert created["cpu"] == (0.5, 1.5) and created["memory"] == (1024, 3000)
    image = created["image"]
    assert image.steps[0] == ("Image.debian_slim", (), {"python_version": "3.12"})
    [packages] = image.named("Image.uv_pip_install")
    assert {p.split(">")[0] for p in packages} == {
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn",
        "scipy",
    }
    assert ({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/opt/matplotlib"},) in image.named("Image.env")

    [sandbox] = fake.sandboxes
    assert sandbox.commands == [["python", "-I", "-B", "_script.py"]]
    [(exec_args, exec_kwargs)] = fake.named("Sandbox.exec")
    job_dir = exec_kwargs["workdir"]
    assert exec_args[3] == "31" and exec_kwargs["timeout"] == 31 + 5 + 15
    assert exec_kwargs["text"] is False and "secrets" not in exec_kwargs
    assert job_dir.startswith("/work/py-")
    assert exec_kwargs["env"] == {"HOME": job_dir, "TMPDIR": job_dir}
    assert fake.named("filesystem.remove") == [((job_dir,), {"recursive": True})]
    assert list((sandbox.root / "work").iterdir()) == []


@needs_coreutils
def test_run_python_reports_failure_and_timeout(backend):
    failed = backend.run_python("import sys\nprint('partial')\nsys.exit('boom')")
    assert not failed.ok and failed.returncode == 1 and not failed.timed_out
    assert failed.stdout == "partial\n" and "boom" in failed.stderr
    slow = backend.run_python("import time\ntime.sleep(30)", timeout=1)
    assert slow.timed_out and not slow.ok and slow.returncode == 124


@pytest.mark.parametrize(
    ("returncode", "timed_out"),
    [(-1, True), (137, False), (124, False), (3, False)],
)
def test_exit_codes_are_classified(fake, backend, returncode, timed_out):
    """-1 is Modal's exec deadline; a quick 137 or 124 is the job's own exit, not a timeout."""
    fake.handlers.append(scripted(python=lambda *_: FakeProcess(b"out", b"err", returncode)))
    result = backend.run_python("pass", timeout=60)
    assert (result.returncode, result.timed_out) == (returncode, timed_out)
    assert (result.stdout, result.stderr, result.ok) == ("out", "err", False)
    assert len(fake.sandboxes) == 1


def test_output_is_decoded_leniently_and_capped(fake, backend, monkeypatch):
    monkeypatch.setattr(modal_backend, "_MAX_CAPTURE", 10)
    fake.handlers.append(scripted(python=lambda *_: FakeProcess(b"caf\xe9 " + b"x" * 50, b"")))
    result = backend.run_python("pass")
    assert result.stdout == "x" * 10
    fake.handlers[:] = [scripted(python=lambda *_: FakeProcess(b"caf\xe9", b""))]
    assert backend.run_python("pass").stdout == "caf\ufffd"


def test_runaway_output_is_drained_without_being_held_in_memory(monkeypatch):
    """Model-written code may print without bound; only the end of it is kept."""
    kilobyte = b"0123456789abcdef" * 64

    def endless() -> Iterator[bytes]:
        for index in range(20_000):  # 20 MB
            yield kilobyte if index else b"first" + kilobyte
        yield b"the end"

    tracemalloc.start()
    try:
        tail = modal_backend._read_tail(endless())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    budget = 4 * modal_backend._MAX_CAPTURE
    assert len(tail) == budget and tail.endswith(b"abcdefthe end")
    assert peak < 4 * budget

    monkeypatch.setattr(modal_backend, "_MAX_CAPTURE", 2)
    assert modal_backend._read_tail([b"ab", b"cd", b"efgh", b"i"]) == b"bcdefghi"
    assert modal_backend._read_tail([]) == b""
    assert modal_backend._read_tail([b"x" * 100]) == b"x" * 8


@pytest.mark.parametrize("name", ["/abs/path", "../up.txt", "a/../../b", ""])
def test_unsafe_file_names_never_reach_the_sandbox(fake, backend, name):
    with pytest.raises(ComputeError, match="unsafe file name"):
        backend.run_python("pass", inputs={name: b"x"})
    with pytest.raises(ComputeError, match="unsafe file name"):
        backend.run_python("pass", outputs=[name])
    assert fake.calls == []


@needs_coreutils
def test_oversized_and_non_file_outputs_are_not_returned(backend, monkeypatch):
    monkeypatch.setattr(modal_backend, "_MAX_FILE_BYTES", 1024)
    result = backend.run_python(
        "import os\n"
        "open('big.bin', 'wb').write(b'x' * 5000)\n"
        "open('small.bin', 'wb').write(b'ok')\n"
        "os.mkdir('a_dir')\n",
        outputs=["big.bin", "small.bin", "a_dir"],
    )
    assert result.ok and result.files == {"small.bin": b"ok"}
    assert "output file 'big.bin'" in result.stderr and "was not returned" in result.stderr


def test_sandboxes_are_reused_per_kind_and_terminated_on_close(fake, backend, tmp_path):
    fake.handlers.append(
        scripted(
            python=lambda *_: FakeProcess(),
            pdflatex=lambda *_: FakeProcess(),
            bibtex=lambda *_: FakeProcess(),
        )
    )
    (tmp_path / "paper.tex").write_text("x")
    for _ in range(3):
        backend.run_python("pass")
    backend.compile_latex(tmp_path, "paper")
    backend.compile_latex(tmp_path, "paper")
    python_sandbox, latex_sandbox = fake.sandboxes
    assert len(python_sandbox.commands) == 3
    assert len(latex_sandbox.commands) == 2 * (1 + len(latex_commands("paper")))
    assert len(fake.named("App.lookup")) == 1
    described = backend.describe()["sandboxes"]
    assert described == {"python": "sb-fake-0", "latex": "sb-fake-1"}

    backend.close()
    backend.close()
    assert all(s.terminated and s.detached for s in fake.sandboxes)
    assert len(fake.named("Sandbox.terminate")) == 2
    assert backend.describe()["sandboxes"] == {"python": None, "latex": None}

    with backend:
        backend.run_python("pass")
    assert len(fake.sandboxes) == 3 and fake.sandboxes[2].terminated


def test_latex_sandbox_image_has_texlive_without_recommends(fake, backend, tmp_path):
    fake.handlers.append(scripted(pdflatex=lambda *_: FakeProcess(), bibtex=lambda *_: None))
    fake.handlers.append(scripted(bibtex=lambda *_: FakeProcess()))
    (tmp_path / "paper.tex").write_text("x")
    backend.compile_latex(tmp_path, "paper")
    [(_, created)] = fake.named("Sandbox.create")
    image = created["image"]
    command = texlive_install_command()
    assert image.named("Image.run_commands") == [(command,)]
    assert "--no-install-recommends" in command
    assert "texlive-latex-extra" in command
    assert "texlive-plain-generic" in command
    assert created["block_network"] is True and not created.get("secrets")
    assert created["include_oidc_identity_token"] is False


def test_tex_images_carry_the_fonts_that_common_templates_need():
    r"""`\mathbbm`, inconsolata (ACL), newtx and libertine (acmart) and lmodern."""
    packages = texlive_install_command().split("--no-install-recommends ")[1].split(" && ")[0]
    assert {"texlive-fonts-extra", "lmodern", "texlive-publishers"} <= set(packages.split())
    assert len(packages.split()) == len(set(packages.split()))


def test_extra_tex_packages_come_from_the_environment(fake, tmp_path, monkeypatch):
    assert extra_tex_packages({}) == ()
    asked = " fonts-lato  texlive-lang-german\tlmodern fonts-lato libfoo++1.2 "
    assert extra_tex_packages({TEX_PACKAGES_ENV: asked}) == (
        "fonts-lato",
        "texlive-lang-german",  # lmodern is installed anyway, and nothing is listed twice
        "libfoo++1.2",
    )
    monkeypatch.setenv(TEX_PACKAGES_ENV, "fonts-lato texlive-music")
    assert extra_tex_packages() == ("fonts-lato", "texlive-music")
    fake.handlers.append(
        scripted(pdflatex=lambda *_: FakeProcess(), bibtex=lambda *_: FakeProcess())
    )
    (tmp_path / "paper.tex").write_text("x")
    ModalBackend().compile_latex(tmp_path, "paper")
    [(_, created)] = fake.named("Sandbox.create")
    [(command,)] = created["image"].named("Image.run_commands")
    assert command == texlive_install_command(("fonts-lato", "texlive-music"))
    assert " latexmk fonts-lato texlive-music && rm -rf " in command


@pytest.mark.parametrize(
    "value",
    [
        "lmodern; rm -rf /",
        "$(curl example.org)",
        "fonts-lato && true",
        "--allow-unauthenticated",
        "Texlive-Fonts-Extra",
        "a",
        "fonts-lato\nRUN true",
        "pkg=1.0",
    ],
)
def test_extra_tex_packages_must_be_package_names(fake, monkeypatch, value):
    """The names become part of a shell command of the image build."""
    with pytest.raises(ValueError, match="is not a Debian package name") as raised:
        extra_tex_packages({TEX_PACKAGES_ENV: value})
    assert TEX_PACKAGES_ENV in str(raised.value)
    # Reported when the backend is built, not by the first compile of a paid-for run.
    monkeypatch.setenv(TEX_PACKAGES_ENV, value)
    with pytest.raises(ValueError, match=TEX_PACKAGES_ENV):
        ModalBackend()
    assert fake.calls == []


def test_check_finds_a_missing_modal_token_without_contacting_modal(fake, backend, monkeypatch):
    """Preflight: no outline, literature search or draft is paid for before Modal is usable."""
    assert inspect.signature(ModalBackend.check) == inspect.signature(ComputeBackend.check)
    [problem] = backend.check()
    assert "Modal is not authenticated" in problem
    assert "sandboxes that compile the paper and run figure code" in problem
    assert "`modal setup`" in problem and problem.endswith("or use --compute local.")
    assert "experiment" not in problem and "--backend" not in problem
    # The paper itself is compiled in a sandbox, so this holds without figure code too.
    assert backend.check(python=False, latex=True) == [problem]
    assert backend.check(python=False, latex=False) == []

    fake.settings.update(token_id="ak-test", token_secret="as-test")
    assert backend.check() == []
    fake.settings.clear()
    # Inside a Modal Function the container's own identity starts the sandboxes.
    fake.local = False
    assert backend.check() == []
    fake.local = True

    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: True))
    compiles_here = ModalBackend(latex="local")
    assert compiles_here.check(python=False, latex=True) == []
    assert compiles_here.check(python=True, latex=True) == [problem]
    assert fake.calls == []  # nothing was looked up, created or started


def test_a_missing_modal_token_is_reported_for_an_experiment_in_its_own_words(
    fake, backend, job, monkeypatch
):
    """An experiment is approved and run without any paper.  The message used to speak
    of the sandboxes that compile the paper, and to offer ``--compute local``, which is
    not an option of ``research approve``."""
    [problem] = backend.check(python=True, latex=False)  # how an approval asks
    assert problem.startswith("Modal is not authenticated (no token is configured")
    assert "the sandbox that would run the experiment cannot be started" in problem
    assert "`modal setup`" in problem and "approve the experiment again" in problem
    assert "needs no GPU" in problem and "`--backend local`" in problem
    assert "in the app, choose the local backend" in problem
    assert "the paper" not in problem and "figure code" not in problem
    assert "--compute" not in problem

    # A token that Modal turns down shows when the job's sandbox is started, or when a
    # job that is under way is picked up again.
    source = job.source("print('never runs')\n")
    fake.create_error = fake.exception.AuthError("Token expired")
    with pytest.raises(BackendUnusable, match="Modal is not authenticated") as refused:
        run_main(backend, job, source)
    said = str(refused.value)
    assert "the experiment cannot run there" in said and "`modal token new`" in said
    assert "run the experiment again" in said and "`--backend local`" in said
    assert "--compute" not in said
    fake.create_error = None
    fake.from_id_error = fake.exception.AuthError("Token expired")
    with pytest.raises(BackendUnusable, match="Modal is not authenticated") as refused:
        backend.attach("sb-fake-0", run_dir=job.run)
    said = str(refused.value)
    assert "`modal token new`" in said and "is picked up where it is" in said
    assert "--backend local" not in said and "--compute" not in said  # it runs on Modal
    fake.from_id_error = None

    # The paper pipeline's own words are as they were.
    fake.create_error = fake.exception.AuthError("Token expired")
    with pytest.raises(BackendUnusable) as refused:
        backend.run_python("pass")
    assert str(refused.value) == (
        "Modal is not authenticated. Run `modal token new`, or set MODAL_TOKEN_ID and "
        "MODAL_TOKEN_SECRET."
    )


def test_expired_sandbox_is_replaced_and_the_job_retried(fake, backend):
    fake.handlers.append(scripted(python=lambda *_: FakeProcess(b"ok")))
    backend.run_python("pass")
    fake.sandboxes[0].exit_code = 124  # reached its lifetime, or was collected for idleness
    result = backend.run_python("pass", inputs={"in.txt": b"x"})
    assert result.ok and result.stdout == "ok"
    assert len(fake.sandboxes) == 2 and fake.sandboxes[0].detached
    assert fake.sandboxes[1].commands == [["python", "-I", "-B", "_script.py"]]


def test_sandbox_that_dies_mid_command_is_not_reported_as_a_job_failure(fake, backend):
    def killed_with_the_sandbox(sandbox, command, cwd, env):
        if sandbox is fake.sandboxes[0]:
            sandbox.exit_code = 137
            return FakeProcess(b"", b"", 137)
        return FakeProcess(b"second try")

    fake.handlers.append(scripted(python=killed_with_the_sandbox))
    result = backend.run_python("pass")
    assert result.ok and result.stdout == "second try"
    assert len(fake.sandboxes) == 2


def test_a_job_is_retried_only_once(fake, backend):
    def always_dies(sandbox, command, cwd, env):
        sandbox.exit_code = 137
        return FakeProcess(b"", b"", -1)

    fake.handlers.append(scripted(python=always_dies))
    with pytest.raises(ComputeError, match="python job failed"):
        backend.run_python("pass")
    assert len(fake.sandboxes) == 2
    assert backend.describe()["sandboxes"]["python"] is None


def test_sandbox_near_its_lifetime_is_replaced_before_a_job(fake, backend):
    fake.handlers.append(scripted(python=lambda *_: FakeProcess()))
    backend.run_python("pass")
    backend._slots["python"].expires_at = time.monotonic() + 30
    backend.run_python("pass", timeout=60)
    old, new = fake.sandboxes
    assert len(old.commands) == 1 and len(new.commands) == 1
    assert not old.terminated  # another thread may still be using it
    backend.close()
    assert old.terminated and new.terminated


def test_job_longer_than_the_sandbox_lifetime_is_refused(fake):
    backend = ModalBackend(sandbox_lifetime=300, idle_timeout=60)
    with pytest.raises(ComputeError, match="raise sandbox_lifetime"):
        backend.run_python("pass", timeout=400)
    assert fake.sandboxes == []


def test_modal_errors_become_compute_errors(fake, backend):
    fake.lookup_error = fake.exception.AuthError("Token missing")
    with pytest.raises(ComputeError, match="modal token new") as refused:
        backend.run_python("pass")
    # A token Modal turns down (expired, revoked, of another profile) passes check(), which
    # contacts nobody.  The error says that another attempt cannot help, so that the stage is
    # not paid for a second time.
    assert isinstance(refused.value, BackendUnusable) and refused.value.unusable is True
    fake.lookup_error = None

    fake.create_error = fake.exception.AuthError("Token expired")
    with pytest.raises(BackendUnusable, match="Modal is not authenticated"):
        backend.run_python("pass")
    fake.create_error = fake.exception.Error("Insufficient capacity")
    with pytest.raises(
        ComputeError, match=r"could not start a Modal sandbox.*Insufficient"
    ) as busy:
        backend.run_python("pass")
    assert busy.value.unusable is False  # this one may pass: the stage is tried again
    fake.create_error = ConnectionError("network down")
    with pytest.raises(ComputeError, match="ConnectionError: network down"):
        backend.run_python("pass")
    fake.create_error = None

    def broken(sandbox, command, cwd, env):
        raise fake.exception.Error("exec failed")

    fake.handlers.append(broken)
    with pytest.raises(ComputeError, match="python job failed: Error: exec failed") as raised:
        backend.run_python("pass")
    assert isinstance(raised.value.__cause__, fake.exception.Error)
    # The sandbox is still running, so the failure is not papered over by a retry.
    assert len(fake.sandboxes) == 1 and len(fake.sandboxes[0].commands) == 1


def test_inside_a_modal_function_the_container_app_owns_the_sandbox(fake, backend, tmp_path):
    fake.local = False
    fake.handlers.append(scripted(python=lambda *_: FakeProcess()))
    backend.run_python("pass")
    assert fake.named("App.lookup") == []
    [(_, created)] = fake.named("Sandbox.create")
    assert created["app"] is None and created["block_network"] is True
    assert backend.describe()["inside_modal_function"] is True

    # The container of a Function holds its credentials: the papers it compiles go to a
    # sandbox that has none of them and no way out, and one sandbox serves every build.
    fake.handlers.append(
        scripted(pdflatex=lambda *_: FakeProcess(), bibtex=lambda *_: FakeProcess())
    )
    (tmp_path / "paper.tex").write_text("x")
    for _ in range(3):
        backend.compile_latex(tmp_path, "paper")
    [_, (_, latex)] = fake.named("Sandbox.create")
    assert latex["app"] is None and latex["block_network"] is True
    assert latex["include_oidc_identity_token"] is False
    assert not latex.get("secrets") and not latex.get("env") and not latex.get("volumes")


@needs_coreutils
def test_concurrent_jobs_share_one_sandbox_without_interfering(fake, backend):
    def run(index: int) -> tuple[int, Any]:
        result = backend.run_python(
            "import pathlib\n"
            "value = pathlib.Path('in.txt').read_text()\n"
            "pathlib.Path('out.txt').write_text(value * 2)\n",
            inputs={"in.txt": str(index).encode()},
            outputs=["out.txt"],
        )
        return index, result

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, range(12)))
    assert all(result.ok for _, result in results)
    assert {i: r.files["out.txt"] for i, r in results} == {
        i: (str(i) * 2).encode() for i in range(12)
    }
    assert len(fake.sandboxes) == 1 and len(fake.named("App.lookup")) == 1
    workdirs = {kwargs["workdir"] for _, kwargs in fake.named("Sandbox.exec")}
    assert len(workdirs) == 12


def _latex_workspace(root: Path) -> Path:
    work = root / "latex_writeup"
    (work / "figures").mkdir(parents=True)
    (work / "empty_dir").mkdir()
    (work / ".git").mkdir()
    (work / "main.tex").write_text("\\documentclass{article}")
    (work / "references.bib").write_text("@article{a, title={T}}")
    (work / "figures" / "plot.png").write_bytes(b"\x89PNG")
    (work / "huge.bin").write_bytes(b"x" * 4096)
    (work / ".DS_Store").write_bytes(b"junk")
    (work / ".git" / "config").write_text("secret")
    (work / "main.pdf").write_bytes(b"%PDF-stale")
    return work


def test_compile_latex_uploads_the_workspace_and_fetches_the_outputs(
    fake, backend, tmp_path, monkeypatch, caplog
):
    # An earlier test may have configured the package logger not to propagate.
    monkeypatch.setattr(logging.getLogger("paper_orchestra"), "propagate", True)
    monkeypatch.setattr(modal_backend, "_MAX_FILE_BYTES", 1024)
    work = _latex_workspace(tmp_path)
    seen: list[tuple[list[str], set[str], tuple[str | None, str | None]]] = []

    def tex(sandbox, command, cwd, env):
        files = {p.relative_to(cwd).as_posix() for p in cwd.rglob("*")}
        seen.append((command, files, (env.get("openout_any"), env.get("openin_any"))))
        if command[0] == "bibtex":
            (cwd / "main.blg").write_text("This is BibTeX")
            return FakeProcess(b"This is BibTeX\n")
        (cwd / "main.pdf").write_bytes(b"%PDF-1.5 fresh")
        (cwd / "main.log").write_bytes(b"This is pdfTeX \xe9")
        return FakeProcess(
            BANNER + b"./main.tex:3: Undefined control sequence.\nl.3 \\foo\n"
            b"LaTeX Warning: Citation `Missing2024' on page 1 undefined on input line 3.\n"
            b"caf\xe9\n",
            b"",
            1,
        )

    fake.handlers.append(scripted(pdflatex=tex, bibtex=tex))
    target = tmp_path / "out" / "paper.pdf"
    with caplog.at_level("WARNING", logger="paper_orchestra"):
        result = backend.compile_latex(work, "main", output_pdf=target, timeout=100.5)

    assert result.ok and not result.timed_out and result.pdf_path == target
    assert target.read_bytes() == b"%PDF-1.5 fresh"
    assert not (work / "main.pdf").exists()
    assert (work / "main.log").read_bytes() == b"This is pdfTeX \xe9"
    assert (work / "main.blg").read_text() == "This is BibTeX"
    assert result.errors == ["./main.tex:3: Undefined control sequence. [l.3 \\foo]"]
    assert result.warnings == ["undefined citation 'Missing2024'"]
    assert result.log_tail.startswith("$ pdflatex -interaction=nonstopmode -no-shell-escape")
    assert "$ bibtex main\n\nThis is BibTeX" in result.log_tail and "caf\ufffd" in result.log_tail

    assert [command for command, _, _ in seen] == latex_commands("main")
    assert {policy for _, _, policy in seen} == {("p", "p")}  # paranoid TeX file access
    uploaded = seen[0][1] - {".paper-orchestra-upload.tar"}
    assert uploaded == {
        "main.tex",
        "references.bib",
        "figures",
        "figures/plot.png",
        "empty_dir",
    }
    assert "huge.bin" in caplog.text

    [sandbox] = fake.sandboxes
    assert sandbox.commands[0] == ["tar", "-xf", ".paper-orchestra-upload.tar", "-C", "."]
    kwargs = sandbox.exec_kwargs
    assert len({k["workdir"] for k in kwargs}) == 1 and kwargs[0]["workdir"].startswith(
        "/work/tex-"
    )
    assert all(isinstance(k["timeout"], int) and k["text"] is False for k in kwargs)
    limits = [int(args[3]) for args, _ in fake.named("Sandbox.exec")][1:]
    assert limits[0] == 101 and all(0 < limit <= 101 for limit in limits)
    assert list((sandbox.root / "work").iterdir()) == []


def test_compile_latex_defaults_to_the_workdir_and_reports_a_missing_pdf(fake, backend, tmp_path):
    (tmp_path / "paper.tex").write_text("x")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-stale")
    state = {"pdf": True}

    def pdflatex(sandbox, command, cwd, env):
        if state["pdf"]:
            (cwd / "paper.pdf").write_bytes(b"%PDF-new")
            return FakeProcess(BANNER + b"Output written on paper.pdf")
        return FakeProcess(BANNER + b"! Emergency stop.\nl.1 \\input{nofile}\n", b"", 1)

    fake.handlers.append(scripted(pdflatex=pdflatex, bibtex=lambda *_: FakeProcess()))
    built = backend.compile_latex(tmp_path, "paper")
    assert built.ok and built.pdf_path == tmp_path / "paper.pdf"
    assert built.pdf_path.read_bytes() == b"%PDF-new"

    state["pdf"] = False
    failed = backend.compile_latex(tmp_path, "paper")
    assert not failed.ok and failed.pdf_path is None and not failed.timed_out
    assert failed.errors == ["Emergency stop. [l.1 \\input{nofile}]"]
    assert not (tmp_path / "paper.pdf").exists()
    assert "Emergency stop" in failed.feedback()


def test_compile_latex_stops_at_the_deadline(fake, backend, tmp_path):
    (tmp_path / "paper.tex").write_text("x")

    def hangs(sandbox, command, cwd, env):
        (cwd / "paper.pdf").write_bytes(b"%PDF-partial")
        return FakeProcess(b"partial log", b"", -1)

    fake.handlers.append(scripted(pdflatex=hangs))
    result = backend.compile_latex(tmp_path, "paper", timeout=20)
    assert result.timed_out and not result.ok and result.pdf_path is None
    assert not (tmp_path / "paper.pdf").exists()
    assert fake.sandboxes[0].commands[1:] == latex_commands("paper")[:1]
    assert "partial log" in result.log_tail and "timed out" in result.feedback()


def test_compile_latex_failures_of_the_backend_itself_raise(fake, backend, tmp_path):
    with pytest.raises(ComputeError, match="does not exist"):
        backend.compile_latex(tmp_path, "nope")
    with pytest.raises(ComputeError, match="unsafe file name"):
        backend.compile_latex(tmp_path, "../escape")
    assert fake.calls == []

    (tmp_path / "paper.tex").write_text("x")
    missing = FakeProcess(b"", b"timeout: failed to run command 'pdflatex': No such file", 127)
    fake.handlers.append(scripted(pdflatex=lambda *_: missing))
    with pytest.raises(ComputeError, match=r"could not run pdflatex.*No such file"):
        backend.compile_latex(tmp_path, "paper")

    fake.handlers[:] = [scripted(tar=lambda *_: FakeProcess(b"", b"tar: short read", 2))]
    with pytest.raises(ComputeError, match=r"could not unpack.*short read"):
        backend.compile_latex(tmp_path, "paper")
    assert len(fake.sandboxes) == 1


def test_latex_local_mode_compiles_on_this_machine(fake, tmp_path, monkeypatch):
    calls: list[tuple] = []

    def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0):
        calls.append((workdir, tex_name, output_pdf, timeout))
        return "local result"

    monkeypatch.setattr(LocalBackend, "compile_latex", compile_latex)
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: False))
    with pytest.raises(ComputeError, match="latex='local' needs pdflatex and bibtex"):
        ModalBackend(latex="local")
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: True))
    backend = ModalBackend(latex="local")
    result = backend.compile_latex(tmp_path, "paper", output_pdf=tmp_path / "o.pdf", timeout=7)
    assert result == "local result"
    assert calls == [(tmp_path, "paper", tmp_path / "o.pdf", 7)]
    assert fake.calls == []
    described = backend.describe()
    assert described["latex"] == "local" and {"pdflatex", "bibtex"} <= set(described)

    fake.handlers.append(scripted(python=lambda *_: FakeProcess(b"sandboxed")))
    assert backend.run_python("pass").stdout == "sandboxed"
    [(_, created)] = fake.named("Sandbox.create")
    assert created["block_network"] is True


@needs_coreutils
@pytest.mark.latex
def test_real_tex_build_through_the_sandbox_interface(backend, tmp_path):
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "Hello \\cite{Missing2024}.\n\\bibliographystyle{plain}\\bibliography{refs}\n"
        "\\end{document}\n"
    )
    (tmp_path / "refs.bib").write_text("@article{Other2020, title={T}, author={A}, year={2020}}")
    result = backend.compile_latex(tmp_path, "paper", output_pdf=tmp_path / "build" / "p.pdf")
    assert result.ok, result.log_tail
    assert result.pdf_path.read_bytes().startswith(b"%PDF")
    assert "undefined citation 'Missing2024'" in result.warnings
    assert (tmp_path / "paper.log").is_file() and (tmp_path / "paper.blg").is_file()


def test_describe_reports_configuration_without_touching_modal(fake):
    backend = ModalBackend(app_name="my-app", cpu=3, memory_mb=2048)
    described = backend.describe()
    assert described["backend"] == "modal" and described["app"] == "my-app"
    assert described["modal_version"] == "0.0-fake"
    assert described["token_configured"] is False
    assert described["inside_modal_function"] is False
    assert (described["cpu_limit"], described["memory_limit_mb"]) == (3, 2048)
    assert (described["sandbox_lifetime_s"], described["idle_timeout_s"]) == (3600.0, 600.0)
    assert "network blocked" in described["isolation"]
    assert described["sandboxes"] == {"python": None, "latex": None}

    fake.settings.update(token_id="ak-test", token_secret="as-test")
    assert backend.describe()["token_configured"] is True
    assert "as-test" not in repr(backend.describe())
    fake.settings.clear()
    fake.settings.update(oauth_refresh_token="r", oauth_client_id="c", oauth_jwt_key="k")
    assert modal_token_configured() is True
    fake.settings.pop("oauth_jwt_key")
    assert modal_token_configured() is False
    assert fake.calls == []


def test_pack_tree_skips_hidden_oversized_and_symlinked_directories(tmp_path):
    root = tmp_path / "root"
    (root / "sub" / "deep").mkdir(parents=True)
    (root / ".hidden").mkdir()
    (root / "a.tex").write_text("a")
    (root / "sub" / "deep" / "b.png").write_bytes(b"b")
    (root / "sub" / "big.bin").write_bytes(b"x" * 100)
    (root / ".hidden" / "c.txt").write_text("c")
    (root / ".env").write_text("SECRET=1")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "o.txt").write_text("outside")
    (root / "linked_dir").symlink_to(outside, target_is_directory=True)
    (root / "linked_file.txt").symlink_to(outside / "o.txt")

    archive = tmp_path / "tree.tar"
    skipped = pack_tree(root, archive, max_file_bytes=50)
    assert sorted(skipped) == ["linked_dir", "sub/big.bin"]
    with tarfile.open(archive) as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert set(members) == {"a.tex", "sub", "sub/deep", "sub/deep/b.png", "linked_file.txt"}
        assert members["linked_file.txt"].isfile()
        assert tar.extractfile("linked_file.txt").read() == b"outside"


def test_token_detection_uses_the_installed_sdk_configuration(monkeypatch):
    pytest.importorskip("modal")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    assert modal_token_configured() is True


def replay_against_the_sdk(real_sdk: types.SimpleNamespace, fake: FakeModal) -> set[str]:
    """Bind every call recorded on the fake to the signature the installed SDK gives it.

    Returns the names of the calls that were replayed.
    """
    real = real_sdk.modal
    targets = {
        "App.lookup": real.App.lookup,
        "Sandbox.create": real.Sandbox.create,
        "Sandbox.from_id": real.Sandbox.from_id,
        "Sandbox.exec": real.Sandbox.exec,
        "Sandbox.poll": real.Sandbox.poll,
        "Sandbox.terminate": real.Sandbox.terminate,
        "Sandbox.detach": real.Sandbox.detach,
        "Image.debian_slim": real.Image.debian_slim,
        "Image.uv_pip_install": real.Image.uv_pip_install,
        "Image.run_commands": real.Image.run_commands,
        "Image.env": real.Image.env,
        **{
            f"filesystem.{name}": getattr(real_sdk.SandboxFilesystem, name)
            for name in (
                "write_bytes",
                "write_text",
                "read_bytes",
                "stat",
                "copy_from_local",
                "copy_to_local",
                "remove",
            )
        },
    }
    image_steps = [
        step for _, created in fake.named("Sandbox.create") for step in created["image"].steps
    ]
    replayed = set()
    for name, args, kwargs in [*fake.calls, *image_steps]:
        signature = inspect.signature(targets[name])
        if next(iter(signature.parameters)) == "self":
            args = (None, *args)
        signature.bind(*args, **kwargs)
        replayed.add(name)
    return replayed


def test_calls_match_the_installed_modal_sdk(real_sdk, fake, backend, tmp_path):
    """Replay every call the backend made on the fake against the real SDK signatures."""
    real = real_sdk.modal
    fake.handlers.append(
        scripted(
            python=lambda sandbox, command, cwd, env: (
                (cwd / "out.txt").write_text("x") and FakeProcess()
            ),
            tar=lambda *_: None,
            pdflatex=lambda sandbox, command, cwd, env: (
                (cwd / "paper.pdf").write_bytes(b"%PDF") and FakeProcess()
            ),
            bibtex=lambda *_: FakeProcess(),
        )
    )
    (tmp_path / "paper.tex").write_text("x")
    backend.run_python("pass", inputs={"in.txt": b"x"}, outputs=["out.txt"])
    assert backend.compile_latex(tmp_path, "paper").ok
    fake.sandboxes[0].exit_code = 124
    backend.run_python("pass")
    backend.close()

    assert replay_against_the_sdk(real_sdk, fake) == {
        "App.lookup",
        "Sandbox.create",
        "Sandbox.exec",
        "Sandbox.poll",
        "Sandbox.terminate",
        "Sandbox.detach",
        "Image.debian_slim",
        "Image.uv_pip_install",
        "Image.run_commands",
        "Image.env",
        "filesystem.write_bytes",
        "filesystem.write_text",
        "filesystem.read_bytes",
        "filesystem.stat",
        "filesystem.copy_from_local",
        "filesystem.copy_to_local",
        "filesystem.remove",
    }

    assert not hasattr(real.Sandbox, "__enter__")
    assert issubclass(real.exception.AuthError, real.Error)
    assert issubclass(real.exception.SandboxFilesystemNotFoundError, real.Error)
    assert callable(real.is_local) and callable(real.config.config.get)
    assert callable(real_sdk.FileInfo.is_file)
    assert "size" in real_sdk.FileInfo.__dataclass_fields__
    assert callable(real_sdk.ContainerProcess.wait)
    assert {"stdout", "stderr"} <= set(dir(real_sdk.ContainerProcess))
    assert callable(real_sdk.StreamReader.__iter__)  # output is read chunk by chunk


def test_a_document_the_screen_refuses_never_reaches_a_sandbox(fake, backend, tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF stale")
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\\begin{document}"
        "\\immediate\\pdfobj stream file {/etc/passwd}\\end{document}"
    )
    result = backend.compile_latex(tmp_path, "paper")
    assert not result.ok and result.pdf_path is None
    assert any("pdfobj is not allowed" in error for error in result.errors)
    assert fake.named("Sandbox.create") == []  # nothing was started, uploaded or run
    assert not (tmp_path / "paper.pdf").exists()


# -- experiment jobs ---------------------------------------------------------------------


class JobDirs:
    """A source directory, a run directory and the place temporary files are made in."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.run = root / "runs" / "1"
        self.scratch = root / "scratch"
        self.scratch.mkdir()

    def source(self, main: str = "", **files: str) -> Path:
        source = self.root / "code"
        source.mkdir(exist_ok=True)
        (source / "main.py").write_text(textwrap.dedent(main), encoding="utf-8")
        for name, text in files.items():
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            (source / name).write_text(text, encoding="utf-8")
        return source

    def log(self) -> str:
        return (self.run / "log.txt").read_bytes().decode("utf-8")

    def output_lines(self, seen: list[str]) -> list[str]:
        """What the job itself printed, without the lines the backend added."""
        return [line for line in seen if not line.startswith("[paper-orchestra]")]


@pytest.fixture
def job(tmp_path, monkeypatch) -> Iterator[JobDirs]:
    """Directories for one job.  Afterwards nothing temporary may be left on this machine."""
    dirs = JobDirs(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(dirs.scratch))
    yield dirs
    assert list(dirs.scratch.iterdir()) == [], "temporary files were left behind"


def run_main(
    backend: ModalBackend, job: JobDirs, source: Path, **options
) -> tuple[JobResult, list]:
    spec = JobSpec(command=["python", "main.py"], **options.pop("spec", {}))
    seen: list[str] = []
    options.setdefault("on_log", seen.append)
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, **options)
    return result, seen


def assert_gone(pid: int, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} is still running")


@needs_coreutils
def test_run_job_round_trip_in_a_sandbox_built_for_the_job(fake, backend, job):
    source = job.source(
        """
        import json, os, sys
        from pathlib import Path

        print("seed:", os.environ["SEED"], "unbuffered:", os.environ["PYTHONUNBUFFERED"])
        print("home:", Path.home().name, "cwd:", Path.cwd().name, "stdin:", repr(sys.stdin.read()))
        print(Path("data/in.txt").read_text())
        Path("data/in.txt").write_text("changed by the job")
        Path("artifacts/deep").mkdir(parents=True)
        Path("artifacts/curve.png").write_bytes(b"\\x89PNG")
        Path("artifacts/deep/table.csv").write_text("a,b\\n1,2\\n")
        Path("metrics.json").write_text(json.dumps({"accuracy": 0.91}))
        print("to stderr", file=sys.stderr)
        """,
        **{"data/in.txt": "payload"},
    )
    spec = JobSpec(
        command=["python", "main.py"],
        timeout=60.5,
        gpu="T4",
        packages=("torch>=2.4", "einops"),
        env={"SEED": "7"},
        cpu=4,
        memory_mb=8192,
    )
    seen: list[str] = []
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=seen.append)

    assert result.ok and result.returncode == 0, job.log()
    assert (result.timed_out, result.cancelled, result.error) == (False, False, None)
    assert (result.backend, result.handle) == ("modal", "sb-fake-0")
    assert result.log_path == job.run / "log.txt" and result.outputs_dir == job.run
    assert 0 < result.seconds < 30
    assert (job.run / "handle.txt").read_text() == "sb-fake-0\n"
    assert seen[0].startswith("[paper-orchestra] Starting a Modal sandbox")
    assert job.output_lines(seen) == [
        "seed: 7 unbuffered: 1",
        "home: home cwd: src stdin: ''",
        "payload",
        "to stderr",
    ]
    assert job.log().splitlines() == seen
    assert json.loads((job.run / "metrics.json").read_text()) == {"accuracy": 0.91}
    assert (job.run / "artifacts" / "curve.png").read_bytes() == b"\x89PNG"
    assert (job.run / "artifacts" / "deep" / "table.csv").read_text() == "a,b\n1,2\n"
    assert sorted(p.name for p in job.run.iterdir()) == [
        "artifacts",
        "cost.json",  # what the sandbox has cost, also for a run whose result is never read
        "handle.txt",
        "log.txt",
        "metrics.json",
    ]
    assert (source / "data" / "in.txt").read_text() == "payload"

    # One sandbox, made for this job and for nothing else, and gone afterwards.
    [(args, created)] = fake.named("Sandbox.create")
    assert args == ("sleep", "infinity") and created["app"] == "app:paper-orchestra"
    assert created["gpu"] == "T4"
    assert created["block_network"] is True
    assert created["include_oidc_identity_token"] is False
    assert not created.get("secrets") and not created.get("env") and not created.get("volumes")
    assert not {k for k in created if "allowlist" in k or k.endswith("_ports")}
    assert created["cpu"] == (0.5, 4) and created["memory"] == (1024, 8192)
    assert (created["timeout"], created["idle_timeout"]) == (61 + 900, 600)
    image = created["image"]
    assert image.steps[0] == ("Image.debian_slim", (), {"python_version": "3.12"})
    base_packages, job_packages = image.named("Image.uv_pip_install")
    assert "numpy>=1.26" in base_packages and job_packages == ("torch>=2.4", "einops")
    [sandbox] = fake.sandboxes
    assert sandbox.terminated and sandbox.detached
    assert len(fake.named("Sandbox.terminate")) == 1

    [runner] = [
        (a, k) for a, k in fake.named("Sandbox.exec") if a[:2] == ("sh", "-c") and "main.py" in a
    ]
    runner_args, runner_kwargs = runner
    assert runner_args[-4:] == ("61", "5", "python", "main.py")
    assert runner_kwargs["workdir"] == "/work/job/src"
    assert runner_kwargs["env"] == {
        "HOME": "/work/job/home",
        "TMPDIR": "/work/job/home",
        "PYTHONUNBUFFERED": "1",
        "SEED": "7",
    }
    assert "secrets" not in runner_kwargs and runner_kwargs["text"] is False
    assert all(isinstance(k["timeout"], int) for _, k in fake.named("Sandbox.exec"))

    # The estimate: the time the sandbox existed, at the prices of the table.
    per_second = 0.000164 + 4 * 0.00003942 + 8 * 0.00000667
    assert result.cost_usd is not None and 0 < result.cost_usd < 60 * per_second
    assert result.cost_usd >= round(result.seconds * per_second, 6)
    # ...and it is on file next to the run, as the last word on it.
    cost = json.loads((job.run / "cost.json").read_text())
    assert (cost["cost_usd"], cost["final"]) == (result.cost_usd, True)
    assert cost["seconds"] >= result.seconds


@needs_coreutils
def test_the_handle_is_on_disk_before_anything_runs_in_the_sandbox(fake, backend, job):
    recorded: list[str] = []

    def spy(sandbox, command, cwd, env):
        recorded.append((job.run / "handle.txt").read_text())

    fake.handlers.append(spy)
    result, _ = run_main(backend, job, job.source("print('ok')\n"))
    assert result.ok and len(recorded) >= 4
    assert set(recorded) == {"sb-fake-0\n"}
    # ...and before anything is written into it, too.
    first_write = next(i for i, call in enumerate(fake.calls) if call[0].startswith("filesystem."))
    created = next(i for i, call in enumerate(fake.calls) if call[0] == "Sandbox.create")
    assert first_write > created


@needs_coreutils
def test_the_packed_source_does_not_lie_on_this_machine_while_the_job_runs(fake, backend, job):
    """The archive of the source is a copy of the person's repository.  A worker that is
    killed removes nothing, and a job runs for hours: once the archive is uploaded
    there is none of it left here for a dead worker to leave behind."""
    source = job.source("import time\nprint('running', flush=True)\ntime.sleep(0.5)\n")
    while_running: list[list[str]] = []

    def on_log(line: str) -> None:
        if line == "running":
            while_running.append(sorted(p.name for p in job.scratch.rglob("*") if p.is_file()))

    result, _ = run_main(backend, job, source, on_log=on_log)
    assert result.ok and while_running == [[]]
    # It was made and uploaded, all the same.
    assert any(call[0] == "filesystem.copy_from_local" for call in fake.calls)


@needs_coreutils
def test_a_planted_job_record_cannot_have_the_directory_of_a_running_job_removed(
    fake, backend, job
):
    """While a job runs, its packed source and then its outputs pass through a directory
    in the temporary directory, where anybody can list it.  It was named like a working
    directory of the local backend, whose clean-up after dead workers removed whatever a
    job record of that name pointed it to: a record planted in another project had the
    directory of a running job removed, and with it the job's outputs."""
    source = job.source(
        """
        import json, time
        print('running', flush=True)
        time.sleep(0.5)
        json.dump({'accuracy': 0.5}, open('metrics.json', 'w'))
        """
    )
    planted: list[tuple[str, list[str], bool]] = []
    held: list[bool] = []

    def on_log(line: str) -> None:
        if line == "running":
            [scratch] = job.scratch.iterdir()
            record = {"session": 0, "session_started": "", "scratch": str(scratch)}
            planted.append((scratch.name, LocalBackend.end_job(record), scratch.is_dir()))
            try:
                os.close(lock_directory(str(scratch)))
                held.append(False)
            except DirectoryInUse:
                held.append(True)

    result, _ = run_main(backend, job, source, on_log=on_log)
    [(name, notes, still_there)] = planted
    assert still_there and notes == []
    # Not a name the local backend gives its working directories, and held while in use:
    # nothing that removes what dead workers left takes it for that.
    assert name.startswith("paper-orchestra-modal-") and held == [True]
    assert result.ok, result.error
    assert json.loads((job.run / "metrics.json").read_text()) == {"accuracy": 0.5}


@pytest.mark.skipif(os.name != "posix", reason="a directory in use is held by a kernel lock")
def test_what_a_killed_worker_left_in_the_temporary_directory_goes_with_the_next_job(
    fake, backend, job, monkeypatch
):
    """A worker that is killed while it packs or uploads the source of a job leaves the
    archive behind: a copy of the person's repository, up to a gigabyte, of which no
    record tells.  The next job of this backend on the machine removes what nobody
    holds any more."""

    def leftover(name: str, age: float) -> Path:
        directory = job.scratch / f"paper-orchestra-modal-{name}"
        directory.mkdir()
        (directory / "source.tar").write_bytes(b"\0" * 4096)
        os.utime(directory, (time.time() - age, time.time() - age))
        return directory

    dead = leftover("dead", 3600)
    in_use = leftover("in-use", 3600)
    just_made = leftover("new", 0)
    of_the_local_backend = job.scratch / "paper-orchestra-job-abc"
    of_the_local_backend.mkdir()
    not_a_directory = job.scratch / "paper-orchestra-modal-file"
    not_a_directory.write_text("x")
    link = job.scratch / "paper-orchestra-modal-link"
    precious = job.root / "precious"
    precious.mkdir()
    (precious / "thesis.tex").write_text("years of work")
    link.symlink_to(precious, target_is_directory=True)
    os.utime(link, (0, 0), follow_symlinks=False)
    holder = lock_directory(str(in_use))  # as the backend of a job that runs holds it
    assert holder is not None
    try:
        result, _ = run_main(backend, job, job.source("print('ok')\n"))
        assert result.ok, result.error
        assert not dead.exists()
        assert (in_use / "source.tar").is_file() and (just_made / "source.tar").is_file()
        assert of_the_local_backend.is_dir() and not_a_directory.is_file()
        assert link.is_symlink() and (precious / "thesis.tex").read_text() == "years of work"
    finally:
        os.close(holder)
        for path in (in_use, just_made, of_the_local_backend):
            shutil.rmtree(path, ignore_errors=True)
        not_a_directory.unlink()
        link.unlink()
    # Picking a job up again is the other moment a worker comes after a dead one.
    dead = leftover("dead-too", 3600)
    backend.attach("sb-nobody", run_dir=job.run)
    assert not dead.exists()

    # What somebody else left in the shared temporary directory is theirs to remove.
    theirs = leftover("of-another-user", 3600)
    with monkeypatch.context() as patch:
        patch.setattr(os, "geteuid", lambda: os.getuid() + 1)
        assert modal_backend.sweep_scratch() == []
    assert (theirs / "source.tar").is_file()

    # Nor is one removed that could not be opened to see whether somebody holds it:
    # that nobody does is then not known, and the job that asked goes on all the same.
    really_open = os.open

    def no_descriptor_left(path, flags, *args, **kwargs):
        if os.fspath(path) == str(theirs):
            raise OSError(errno.EMFILE, os.strerror(errno.EMFILE), str(theirs))
        return really_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", no_descriptor_left)
        assert modal_backend.sweep_scratch() == []
    assert (theirs / "source.tar").is_file()
    assert modal_backend.sweep_scratch() == [str(theirs)] and not theirs.exists()


@needs_coreutils
def test_a_job_is_not_started_next_to_the_archive_it_was_unpacked_from(
    fake, backend, job, monkeypatch
):
    """The archive holds the source as it was sent, also what the first step of a job
    takes out of its files before the command may read them (the token of a repository
    address).  If it cannot be removed from the sandbox, the command is not started."""
    remove = FakeFilesystem.remove

    def failing(self, remote_path: str, *, recursive: bool = False) -> None:
        if remote_path.endswith(".tar"):
            raise ConnectionError("the connection to the sandbox was lost")
        remove(self, remote_path, recursive=recursive)

    monkeypatch.setattr(FakeFilesystem, "remove", failing)
    result, seen = run_main(backend, job, job.source("print('the command ran')\n"))
    assert result.returncode is None and "could not run the job" in result.error
    assert "the connection to the sandbox was lost" in result.error
    assert "the command ran" not in job.log() and "the command ran" not in seen
    assert fake.sandboxes[0].terminated
    assert not any("main.py" in call[1] for call in fake.calls if call[0] == "Sandbox.exec")


@needs_coreutils
def test_the_network_is_open_only_when_the_job_asks_for_it(fake, job):
    backend = ModalBackend(cpu=1.5, memory_mb=3000, idle_timeout=120)
    source = job.source("print('ok')\n")
    result, _ = run_main(backend, job, source, spec={"network": True, "timeout": 30})
    assert result.ok
    [(_, created)] = fake.named("Sandbox.create")
    assert created["block_network"] is False and created["include_oidc_identity_token"] is False
    assert created.get("gpu") is None and not created.get("secrets")
    assert created["cpu"] == (0.5, 1.5) and created["memory"] == (1024, 3000)
    assert (created["timeout"], created["idle_timeout"]) == (930, 120)
    # No packages asked for: the image is the one figure code runs in.
    assert len(created["image"].named("Image.uv_pip_install")) == 1
    assert result.cost_usd is not None and result.cost_usd >= 0


@needs_coreutils
def test_job_output_is_streamed_while_the_job_runs(fake, backend, job):
    source = job.source(
        """
        import time
        print("first")
        time.sleep(1.5)
        print("second")
        """
    )
    seen: list[tuple[str, float, str]] = []

    def on_log(line: str) -> None:
        if not line.startswith("[paper-orchestra]"):
            seen.append((line, time.monotonic(), job.log()))

    result, _ = run_main(backend, job, source, on_log=on_log)
    assert result.ok
    (first, first_at, log_then), (second, second_at, _) = seen
    assert (first, second) == ("first", "second")
    assert second_at - first_at > 0.8
    assert log_then.endswith("first\n")


@needs_coreutils
def test_a_stop_request_terminates_the_sandbox(fake, backend, job):
    source = job.source("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(60)\n")
    seen: list[str] = []
    asked: list[float] = []

    def should_stop() -> bool:
        asked.append(time.monotonic())
        return bool(job.output_lines(seen)) and len(asked) > 3

    started = time.monotonic()
    result, _ = run_main(backend, job, source, on_log=seen.append, should_stop=should_stop)
    assert result.cancelled and not result.timed_out and not result.ok
    assert result.returncode is None and result.error is None
    assert time.monotonic() - started < 10
    assert max(b - a for a, b in itertools.pairwise(asked)) < 1.0
    [sandbox] = fake.sandboxes
    assert sandbox.terminated and len(fake.named("Sandbox.terminate")) == 1
    assert_gone(int(job.output_lines(seen)[0]))
    assert seen[-1] == "[paper-orchestra] The job was stopped on request."
    assert job.log().endswith(seen[-1] + "\n")
    assert result.cost_usd is not None
    # The time it ran counts against the project's budget although nothing came of it.
    assert 0.5 < result.seconds < 10


def test_a_stop_request_is_heard_while_the_sandbox_is_being_built(fake, backend, job):
    """Building an image takes minutes; the person who cancels does not wait for it."""
    fake.create_gate = threading.Event()
    asked: list[float] = []

    def should_stop() -> bool:
        asked.append(time.monotonic())
        return len(asked) >= 3

    started = time.monotonic()
    result, seen = run_main(
        backend, job, job.source("print('never runs')\n"), should_stop=should_stop
    )
    assert time.monotonic() - started < 5
    assert result.cancelled and result.returncode is None and result.error is None
    assert result.handle is None and result.cost_usd is None and result.seconds == 0.0
    assert seen[-1] == "[paper-orchestra] The job was stopped on request, before it had started."
    assert not (job.run / "handle.txt").exists() and fake.sandboxes == []

    # The sandbox that Modal hands over later has nobody waiting for it: it is ended.
    fake.create_gate.set()
    deadline = time.monotonic() + 5
    while not (fake.sandboxes and fake.sandboxes[0].terminated):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert fake.sandboxes[0].commands == [] and fake.sandboxes[0].detached

    # The same while the source is uploaded: the job never starts.
    fake.create_gate = None
    uploading = threading.Event()

    def slow_upload(sandbox, command, cwd, env):
        uploading.set()
        time.sleep(1.5)

    fake.handlers.append(slow_upload)
    result, _ = run_main(
        backend, job, job.source("print('never runs')\n"), should_stop=uploading.is_set
    )
    assert result.cancelled and result.handle == "sb-fake-1" and result.cost_usd is not None
    assert fake.sandboxes[1].terminated
    assert not any("main.py" in command for command in fake.sandboxes[1].commands)


def test_a_stop_request_is_heard_while_the_source_is_packed(fake, backend, job):
    result, seen = run_main(backend, job, job.source("print('x')\n"), should_stop=lambda: True)
    assert result.cancelled and result.error is None and result.handle is None
    assert seen == ["[paper-orchestra] The job was stopped on request, before it had started."]
    assert fake.calls == []


def test_an_interrupt_while_the_sandbox_is_being_built_does_not_leak_it(fake, backend, job):
    fake.create_gate = threading.Event()
    asked: list[int] = []

    def should_stop() -> bool:
        asked.append(1)
        if len(asked) > 1:  # the source is packed; Modal is building the image
            raise KeyboardInterrupt
        return False

    with pytest.raises(KeyboardInterrupt):
        run_main(backend, job, job.source("print('never runs')\n"), should_stop=should_stop)
    fake.create_gate.set()
    deadline = time.monotonic() + 5
    while not (fake.sandboxes and fake.sandboxes[0].terminated):
        assert time.monotonic() < deadline
        time.sleep(0.02)


@needs_coreutils
def test_a_job_that_runs_too_long_is_stopped_inside_the_sandbox(fake, backend, job):
    source = job.source(
        """
        import time
        open("metrics.json", "w").write('{"steps": 3}')
        for index in range(50):
            print("step", index)
        print("half a li", end="", flush=True)
        time.sleep(60)
        """
    )
    started = time.monotonic()
    result, seen = run_main(backend, job, source, spec={"timeout": 2})
    assert result.timed_out and not result.cancelled and not result.ok and result.error is None
    assert result.returncode == 124 and 2 <= result.seconds < 10
    assert time.monotonic() - started < 20
    lines = job.output_lines(seen)
    assert lines == [*(f"step {index}" for index in range(50)), "half a li"]
    assert seen[-1] == (
        "[paper-orchestra] The job was stopped because it ran longer than its time limit of 2 s."
    )
    # What it had written by then is still fetched.
    assert json.loads((job.run / "metrics.json").read_text()) == {"steps": 3}
    assert fake.sandboxes[0].terminated


@needs_coreutils
def test_an_interrupt_terminates_the_sandbox_and_leaves_nothing_behind(fake, backend, job):
    source = job.source("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(60)\n")
    pids: list[int] = []

    def on_log(line: str) -> None:
        if not line.startswith("[paper-orchestra]"):
            pids.append(int(line))
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_main(backend, job, source, on_log=on_log)
    [sandbox] = fake.sandboxes
    assert sandbox.terminated and sandbox.detached
    assert_gone(pids[0])
    assert list(job.scratch.iterdir()) == []
    assert (job.run / "handle.txt").read_text() == "sb-fake-0\n"


@needs_coreutils
def test_a_sandbox_that_is_lost_is_the_backends_failure_not_the_jobs(fake, backend, job):
    source = job.source("import time\nprint('running', flush=True)\ntime.sleep(60)\n")

    def on_log(line: str) -> None:
        if line == "running":
            # Preempted, or stopped from the dashboard: not by this process.
            fake.sandboxes[0].exit_code = 137
            fake.sandboxes[0].stop_processes()

    started = time.monotonic()
    result, _ = run_main(backend, job, source, on_log=on_log)
    assert time.monotonic() - started < 15
    assert result.returncode is None and not result.ok
    assert not result.timed_out and not result.cancelled
    assert "stopped before the job finished" in result.error and "137" in result.error
    assert result.error in job.log() and "running\n" in job.log()
    assert result.handle == "sb-fake-0" and fake.sandboxes[0].detached


def test_authentication_failures_stop_the_run_and_other_failures_are_results(fake, backend, job):
    source = job.source("print('never runs')\n")
    fake.create_error = fake.exception.AuthError("Token expired")
    with pytest.raises(BackendUnusable, match="Modal is not authenticated"):
        run_main(backend, job, source)
    fake.create_error = None
    fake.lookup_error = fake.exception.AuthError("Token missing")
    with pytest.raises(BackendUnusable, match="modal token new"):
        run_main(ModalBackend(), job, source)
    fake.lookup_error = None

    fake.create_error = fake.exception.Error("Insufficient capacity")
    result, seen = run_main(backend, job, source)
    assert result.returncode is None and result.handle is None and result.cost_usd is None
    assert result.error.startswith("The Modal backend could not run the job: ")
    assert "could not start a Modal sandbox" in result.error and "Insufficient" in result.error
    assert seen[-1] == f"[paper-orchestra] {result.error}"
    assert fake.sandboxes == []

    fake.create_error = None

    def broken(sandbox, command, cwd, env):
        raise fake.exception.Error("exec failed")

    fake.handlers.append(broken)
    result, _ = run_main(backend, job, source)
    assert "Error: exec failed" in result.error and result.handle == "sb-fake-0"
    assert fake.sandboxes[0].terminated  # nothing is left running after a failure
    assert result.cost_usd is not None


@needs_coreutils
def test_a_job_whose_orchestrator_died_can_be_picked_up_again(fake, job, monkeypatch):
    source = job.source(
        """
        import json, time
        print("one")
        print("two")
        time.sleep(1.5)
        print("three")
        json.dump({"accuracy": 0.5}, open("metrics.json", "w"))
        print("four")
        """
    )

    first = killed_worker(monkeypatch, job, source, "two", gpu="L4")
    [sandbox] = fake.sandboxes
    assert not sandbox.terminated and "four" not in job.log()
    handle = (job.run / "handle.txt").read_text().strip()

    # A new process finds the handle next to the run and picks the job up, as the job
    # that was approved and from the moment its run began.
    second: list[str] = []
    backend = ModalBackend()
    result = backend.attach(
        handle,
        run_dir=job.run,
        spec=JobSpec(command=["python", "main.py"], gpu="L4"),
        started=stamp(2),
        on_log=second.append,
    )
    assert result.ok and result.returncode == 0 and result.error is None
    assert (result.backend, result.handle) == ("modal", "sb-fake-0")
    assert result.seconds >= 1.5 and result.outputs_dir == job.run
    # Each line is reported once across the two processes...
    assert job.output_lines(first) == ["one", "two"]
    assert second == ["three", "four"]
    # ...and the log is whole.
    assert job.output_lines(job.log().splitlines()) == ["one", "two", "three", "four"]
    assert json.loads((job.run / "metrics.json").read_text()) == {"accuracy": 0.5}
    assert sandbox.terminated and sandbox.detached
    assert fake.named("Sandbox.from_id") == [(("sb-fake-0",), {})]
    # The price is that of the sandbox the job was started in, since it was started.
    assert result.cost_usd is not None and result.cost_usd > 0.000222 * 1.5


def test_a_job_that_was_never_started_is_not_waited_for(fake, job, monkeypatch):
    """The orchestrator died between making the sandbox and starting the command."""

    class Died(BaseException):
        pass

    def dies_during_the_upload(sandbox, command, cwd, env):
        raise Died

    fake.handlers.append(dies_during_the_upload)
    with monkeypatch.context() as crash:
        crash.setattr(ModalBackend, "_terminate", lambda self, sandbox: None)
        with pytest.raises(Died):
            run_main(ModalBackend(), job, job.source("print('never runs')\n"))
    fake.handlers.clear()
    [sandbox] = fake.sandboxes
    assert not sandbox.terminated

    started = time.monotonic()
    result = ModalBackend().attach("sb-fake-0", run_dir=job.run)
    assert time.monotonic() - started < 5
    assert result.returncode is None and "had not been started" in result.error
    assert "run the experiment again" in result.error
    assert sandbox.terminated  # it was ours, and of no use


@needs_coreutils
def test_attaching_to_a_job_that_cannot_be_recovered_keeps_the_log(fake, backend, job):
    job.run.mkdir(parents=True)
    (job.run / "log.txt").write_text("epoch 1\nepo")
    seen: list[str] = []

    gone = backend.attach("sb-unknown", run_dir=job.run, on_log=seen.append)
    assert gone.returncode is None and gone.handle == "sb-unknown" and not gone.ok
    assert "no sandbox sb-unknown" in gone.error and "Run the experiment again" in gone.error
    assert job.log() == f"epoch 1\nepo\n[paper-orchestra] {gone.error}\n"
    assert seen == [f"[paper-orchestra] {gone.error}"]

    result, _ = run_main(backend, job, job.source("print('done')\n"))
    assert result.ok
    ended = backend.attach(result.handle, run_dir=job.run)
    assert "is no longer running" in ended.error and "137" in ended.error
    assert "done\n" in job.log()  # the log of the run is still there

    # A sandbox of this backend that holds no job (it runs figure code) is left alone.
    backend.run_python("pass")
    other = fake.sandboxes[-1]
    foreign = backend.attach(other.object_id, run_dir=job.run)
    assert "does not hold a job" in foreign.error and not other.terminated

    fake.from_id_error = fake.exception.AuthError("Token expired")
    with pytest.raises(BackendUnusable, match="Modal is not authenticated"):
        backend.attach("sb-fake-0", run_dir=job.run)
    fake.from_id_error = ConnectionError("network down")
    offline = backend.attach("sb-fake-0", run_dir=job.run)
    assert "ConnectionError: network down" in offline.error


@needs_coreutils
def test_outputs_of_a_sandbox_are_fetched_with_the_same_rules_as_local_ones(
    fake, backend, job, monkeypatch
):
    monkeypatch.setattr(base, "MAX_OUTPUT_BYTES", 20_000)
    source = job.source(
        """
        import os
        os.makedirs("artifacts")
        open("metrics.json", "w").write('{"loss": 0.5}')
        open("artifacts/a_huge.bin", "wb").write(b"x" * 500_000)
        open("artifacts/b_small.txt", "w").write("small")
        open("artifacts/-starts-with-a-dash.txt", "w").write("dash")
        os.symlink("../../job.json", "artifacts/leak.txt")
        os.symlink("/", "artifacts/root")
        os.symlink("../job.json", "metrics2.json")
        os.mkfifo("artifacts/pipe")
        """
    )
    result, _ = run_main(
        backend, job, source, spec={"outputs": ("metrics.json", "metrics2.json", "artifacts")}
    )
    assert result.ok, job.log()
    copied = sorted(p.relative_to(job.run).as_posix() for p in job.run.rglob("*"))
    assert copied == [
        "artifacts",
        "artifacts/-starts-with-a-dash.txt",
        "artifacts/b_small.txt",
        "cost.json",
        "handle.txt",
        "log.txt",
        "metrics.json",
    ]
    assert not any(p.is_symlink() for p in job.run.rglob("*"))
    notes = "\n".join(result.notes)
    for name in ("artifacts/leak.txt", "artifacts/root", "metrics2.json", "artifacts/pipe"):
        assert name in notes
    assert "symbolic links are never followed" in notes
    assert "a_huge.bin (488 KB)" in notes and "limited to 20 KB in total" in notes
    assert "a_huge.bin (488 KB)" in job.log()
    # The big file never left the sandbox.
    [(_, local)] = [args for args, _ in fake.named("filesystem.copy_to_local")]
    assert Path(local).parent.parent == job.scratch


@needs_coreutils
def test_hard_linked_outputs_are_all_fetched(fake, backend, job):
    """tar stores the second name of a file as a link to the first, and links are not
    taken from an archive that untrusted code made."""
    source = job.source(
        """
        import os
        os.makedirs("artifacts")
        open("artifacts/a.csv", "w").write("a,b\\n")
        os.link("artifacts/a.csv", "artifacts/b.csv")
        """
    )
    result, _ = run_main(backend, job, source)
    assert result.ok, job.log()
    assert not any("could not be fetched" in note for note in result.notes), result.notes
    assert (job.run / "artifacts" / "a.csv").read_text() == "a,b\n"
    assert (job.run / "artifacts" / "b.csv").read_text() == "a,b\n"


@needs_coreutils
def test_an_output_behind_a_linked_directory_is_not_fetched(fake, backend, job):
    """The output is named with a directory, and the job made that directory a link."""
    source = job.source("import os\nos.symlink('..', 'results')\n")
    result, _ = run_main(backend, job, source, spec={"outputs": ("results/job.json",)})
    assert result.ok, job.log()
    assert sorted(p.name for p in job.run.iterdir()) == ["cost.json", "handle.txt", "log.txt"]
    [note] = [note for note in result.notes if "symbolic links are never followed" in note]
    assert "results" in note and note in job.log()
    assert fake.named("filesystem.copy_to_local") == []  # nothing was asked of the sandbox


def test_a_sandbox_cannot_talk_the_backend_into_writing_elsewhere(fake, backend, job):
    """Everything that comes back from a sandbox is the work of untrusted code."""
    job.run.mkdir(parents=True)
    (job.run / "run.json").write_text('{"status": "running"}')
    victim = job.root / "victim.txt"
    victim.write_text("untouched")

    def lying_scan(sandbox, command, cwd, env):
        if command[:2] != ["python", "-I"]:
            return None
        listing = [
            ["metrics.json", "file", 2],
            ["run.json", "file", 2],
            ["log.txt", "file", 2],
            ["../../victim.txt", "file", 2],
            ["/etc/passwd", "file", 2],
            ["artifacts/ok.txt", "file", 2],
            ["artifacts/grown.txt", "file", 2],
            ["artifacts/link.txt", "file", 2],
            ["elsewhere/x.txt", "file", 2],
        ]
        return FakeProcess(json.dumps({"entries": listing, "truncated": False}).encode())

    def lying_tar(sandbox, command, cwd, env):
        if command[:2] != ["tar", "-cf"]:
            return None
        with tarfile.open(cwd.parent / "outputs.tar", "w") as tar:

            def add(name: str, data: bytes = b"{}", **fields: Any) -> None:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                for key, value in fields.items():
                    setattr(info, key, value)
                tar.addfile(info, io.BytesIO(data) if info.isreg() else None)

            add("./metrics.json")
            add("./artifacts/ok.txt", b"ok")
            add("./artifacts/grown.txt", b"x" * 5000)
            add("./artifacts/link.txt", b"", type=tarfile.SYMTYPE, linkname=str(victim))
            add("./run.json", b"{}")
            add("../../victim.txt", b"overwritten")
            add(str(victim), b"overwritten")
            add("./artifacts/extra.txt", b"not listed")
        return FakeProcess()

    fake.handlers.extend(
        [
            scripted(sh=lambda sandbox, command, cwd, env: _finished_job(cwd, command)),
            lying_scan,
            lying_tar,
        ]
    )
    result, _ = run_main(backend, job, job.source("print('unused')\n"))
    assert result.returncode == 0 and result.error is None
    copied = sorted(p.relative_to(job.run).as_posix() for p in job.run.rglob("*") if p.is_file())
    assert copied == [
        "artifacts/ok.txt",
        "cost.json",
        "handle.txt",
        "log.txt",
        "metrics.json",
        "run.json",
    ]
    assert (job.run / "run.json").read_text() == '{"status": "running"}'
    assert victim.read_text() == "untouched"
    assert any("could not be fetched" in note for note in result.notes)
    # Only what was asked for was ever named to the sandbox.
    [(names, _)] = [
        args for args, _ in fake.named("filesystem.write_bytes") if args[1].endswith("outputs.list")
    ]
    assert names.split(b"\0") == [
        b"./metrics.json",
        b"./artifacts/ok.txt",
        b"./artifacts/grown.txt",
        b"./artifacts/link.txt",
        b"",
    ]


def test_only_a_link_is_accepted_above_a_requested_output(fake, backend, job):
    """A link on the way to an output is reported so that the person hears of it; a
    sandbox cannot use that to have a file of another name fetched."""
    listing: list[list] = []

    def lying_scan(sandbox, command, cwd, env):
        if command[:2] != ["python", "-I"]:
            return None
        return FakeProcess(json.dumps({"entries": listing, "truncated": False}).encode())

    fake.handlers.extend(
        [
            scripted(sh=lambda sandbox, command, cwd, env: _finished_job(cwd, command)),
            lying_scan,
        ]
    )
    source = job.source("print('unused')\n")
    spec = {"outputs": ("deep/metrics.json",)}

    listing[:] = [["deep", "file", 2], ["deeper", "link", 0], ["deep/metrics.json2", "link", 0]]
    result, _ = run_main(backend, job, source, spec=spec)
    assert result.returncode == 0 and result.error is None and result.notes == []

    listing[:] = [["deep", "link", 0]]
    result, _ = run_main(backend, job, source, spec=spec)
    assert result.returncode == 0 and result.error is None
    [note] = result.notes
    assert "symbolic links are never followed" in note and "deep" in note
    # Either way nothing was asked of the sandbox.
    assert fake.named("filesystem.copy_to_local") == []
    assert sorted(p.name for p in job.run.iterdir()) == ["cost.json", "handle.txt", "log.txt"]


# -- what a job costs is never lost, and a job is judged by this side's record --------------

H100_PER_SECOND = 0.001097 + 2 * 0.00003942 + 4 * 0.00000667


def cost_on_file(job: JobDirs) -> dict:
    return json.loads((job.run / "cost.json").read_text())


class Died(BaseException):
    """The worker is killed: nothing after this point runs in it."""


def killed_worker(
    monkeypatch, job: JobDirs, source: Path, at: str, mask: tuple[str, ...] = (), **spec
) -> list[str]:
    """Start the job in ``source`` and have the worker that follows it die when the job
    prints the line ``at``.  A killed process stops nothing and settles nothing: the
    sandbox stays up and the records next to the run are as they were at that moment.
    Returns what the worker had passed on."""
    seen: list[str] = []

    def on_log(line: str) -> None:
        seen.append(line)
        if line == at:
            raise Died

    with monkeypatch.context() as crash:
        crash.setattr(ModalBackend, "_terminate", lambda self, sandbox: None)
        crash.setattr(ModalBackend, "_settle", lambda self, sandbox, job, result, run_dir: None)
        with pytest.raises(Died):
            run_main(ModalBackend(), job, source, on_log=on_log, spec=spec, mask=mask)
    return seen


def stamp(seconds_ago: float) -> str:
    """A moment as the engine writes it into ``run.json``."""
    moment = time.time() - seconds_ago
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))


@needs_coreutils
def test_the_cost_of_a_sandbox_is_on_file_from_the_moment_it_exists(fake, backend, job):
    seen_by_the_sandbox: list[dict] = []

    def spy(sandbox, command, cwd, env):
        seen_by_the_sandbox.append(cost_on_file(job))

    fake.handlers.append(spy)
    result, _ = run_main(backend, job, job.source("print('ok')\n"), spec={"gpu": "H100"})
    assert result.ok
    # Before anything ran in the sandbox there was a cost on file, not yet the final one.
    first = seen_by_the_sandbox[0]
    assert first["final"] is False and first["cost_usd"] is not None and first["seconds"] >= 0
    assert all(cost["final"] is False for cost in seen_by_the_sandbox)
    last = cost_on_file(job)
    assert last["final"] is True and last["cost_usd"] == result.cost_usd > 0
    assert set(last) >= {"cost_usd", "seconds", "final"}


@needs_coreutils
def test_the_estimate_on_file_follows_a_job_that_runs(fake, backend, job, monkeypatch):
    """A worker that is killed outright writes nothing more: what is on file by then
    is what the run is known to have cost."""
    monkeypatch.setattr(modal_backend, "_COST_EVERY", 0.2)
    source = job.source(
        """
        import time
        print("first", flush=True)
        time.sleep(2.0)
        print("second", flush=True)
        """
    )
    estimates: list[dict] = []

    def on_log(line: str) -> None:
        if not line.startswith("[paper-orchestra]"):
            estimates.append(cost_on_file(job))

    result, _ = run_main(backend, job, source, on_log=on_log, spec={"gpu": "H100"})
    assert result.ok
    early, late = estimates
    assert early["final"] is False and late["final"] is False
    assert late["seconds"] - early["seconds"] > 1.0 and late["cost_usd"] > early["cost_usd"]
    assert late["cost_usd"] == pytest.approx(late["seconds"] * H100_PER_SECOND, rel=0.01)


@needs_coreutils
def test_an_interrupted_run_leaves_its_cost_on_file(fake, backend, job, monkeypatch):
    """Ctrl-C, or the Stop button, reaches the worker as an interrupt: ``run_job`` raises
    and its caller never sees a result.  The sandbox was billed all the same."""
    source = job.source("import time\nprint('started', flush=True)\ntime.sleep(60)\n")
    real_time = time.time

    def on_log(line: str) -> None:
        if line == "started":
            # The job has been running for a while by the backend's clock.
            monkeypatch.setattr(modal_backend.time, "time", lambda: real_time() + 6000)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_main(backend, job, source, on_log=on_log, spec={"gpu": "H100", "timeout": 7200})
    monkeypatch.undo()
    assert fake.sandboxes[0].terminated
    cost = cost_on_file(job)
    assert cost["final"] is True and 6000 <= cost["seconds"] < 6100
    assert cost["cost_usd"] == pytest.approx(cost["seconds"] * H100_PER_SECOND, rel=0.001)
    assert cost["cost_usd"] > 6.0


@needs_coreutils
def test_a_sandbox_that_could_not_be_ended_is_charged_for_as_long_as_it_can_live(
    fake, backend, job, monkeypatch
):
    """The call that ends a sandbox can fail like any other.  The sandbox then lives, and
    is billed, until Modal ends it: the cost on record is that of its whole life, not of
    the time until the call failed."""

    def unreachable(self, *, wait: bool = False) -> None:
        raise ConnectionError("the connection to Modal was lost")

    monkeypatch.setattr(FakeSandbox, "terminate", unreachable)
    source = job.source("print('ok')\n")
    result, _ = run_main(backend, job, source, spec={"gpu": "H100", "timeout": 600})
    assert result.ok  # the job itself went well
    assert result.cost_usd == pytest.approx((600 + 900) * H100_PER_SECOND, rel=0.001)
    cost = cost_on_file(job)
    assert cost["final"] is True and cost["cost_usd"] == result.cost_usd
    assert cost["seconds"] == 1500
    [note] = [note for note in result.notes if "could not be ended" in note]
    assert "1500 s" in note


@needs_coreutils
def test_a_job_whose_sandbox_is_gone_is_charged_for_the_time_since_it_started(
    fake, job, monkeypatch
):
    """The worker died, the sandbox ran to its end unattended and was collected."""
    source = job.source("import time\nprint('started', flush=True)\ntime.sleep(60)\n")
    spec = {"gpu": "H100", "timeout": 3600}
    killed_worker(monkeypatch, job, source, "started", **spec)
    [sandbox] = fake.sandboxes
    on_death = cost_on_file(job)
    assert on_death["final"] is False  # all a killed worker left behind
    sandbox.exit_code = 0  # it reached the end of its life while nobody looked
    sandbox.stop_processes()
    approved = JobSpec(command=["python", "main.py"], **spec)
    seen: list[str] = []

    result = ModalBackend().attach(
        "sb-fake-0", run_dir=job.run, spec=approved, started=stamp(1800), on_log=seen.append
    )

    assert result.returncode is None and "is no longer running" in result.error
    assert result.cost_usd == pytest.approx(1800 * H100_PER_SECOND, rel=0.01)
    cost = cost_on_file(job)
    assert cost["final"] is True and cost["cost_usd"] == result.cost_usd
    assert 1799 <= cost["seconds"] <= 1805
    [note] = [note for note in result.notes if "charged" in note]
    assert (
        "since it started (18" in note and "4500 s" in note and f"[paper-orchestra] {note}" in seen
    )
    assert note in job.log() and "started\n" in job.log()  # the log of the run is kept

    # However long ago that was: a sandbox is not charged for longer than it can live.
    (job.run / "cost.json").write_text(json.dumps(on_death))
    late = ModalBackend().attach("sb-fake-0", run_dir=job.run, spec=approved, started=stamp(90000))
    assert late.cost_usd == pytest.approx((3600 + 900) * H100_PER_SECOND, rel=0.001)

    # A sandbox Modal no longer knows is gone just the same.
    (job.run / "cost.json").write_text(json.dumps(on_death))
    unknown = ModalBackend().attach("sb-nobody", run_dir=job.run, spec=approved, started=stamp(60))
    assert "no sandbox sb-nobody" in unknown.error
    assert unknown.cost_usd == pytest.approx(60 * H100_PER_SECOND, rel=0.05)

    # Without being told, the backend goes by what it wrote down when it made the sandbox.
    (job.run / "cost.json").write_text(json.dumps(on_death))
    untold = ModalBackend().attach("sb-fake-0", run_dir=job.run)
    assert untold.cost_usd is not None and 0 < untold.cost_usd < 60 * H100_PER_SECOND

    # Once a final cost is on file it stands: nothing is charged twice.
    again = ModalBackend().attach("sb-fake-0", run_dir=job.run, spec=approved, started=stamp(4000))
    assert again.cost_usd == untold.cost_usd and not any("charged" in n for n in again.notes)


def test_a_sandbox_that_cannot_be_reached_is_charged_for_as_long_as_it_can_live(fake, job):
    """It may still be running, and nobody will look at it again."""
    job.run.mkdir(parents=True)
    fake.from_id_error = ConnectionError("network down")
    approved = JobSpec(command=["python", "main.py"], gpu="H100", timeout=3600)
    result = ModalBackend().attach("sb-fake-0", run_dir=job.run, spec=approved, started=stamp(30))
    assert "ConnectionError: network down" in result.error
    assert result.cost_usd == pytest.approx(4500 * H100_PER_SECOND, rel=0.001)
    [note] = result.notes
    assert "Whether the sandbox still runs is not known" in note and "4500 s" in note
    assert cost_on_file(job)["cost_usd"] == result.cost_usd

    # A job of which nothing is known has no price: none is made up.
    (job.run / "cost.json").unlink()
    unknown = ModalBackend().attach("sb-fake-0", run_dir=job.run)
    assert unknown.cost_usd is None and unknown.notes == []
    assert not (job.run / "cost.json").exists()


@needs_coreutils
def test_picking_a_job_up_goes_by_this_sides_record_not_by_the_sandbox(fake, job, monkeypatch):
    """The job rewrites the record in its sandbox: more outputs, and a sandbox that
    "was made" in the future, so that it would be billed for no time at all."""
    source = job.source(
        """
        import json, os, time
        record = json.load(open("../job.json"))
        record["outputs"] = ["metrics.json", "artifacts", "code", "run.json"]
        record["created"] = time.time() + 290
        record["timeout"] = 1
        record["gpu"] = None
        json.dump(record, open("../job.json", "w"))
        os.makedirs("code")
        open("code/main.py", "w").write("# an honest-looking experiment\\n")
        open("run.json", "w").write('{"status": "succeeded", "cost_usd": 0}')
        print("started", flush=True)
        time.sleep(1.5)
        json.dump({"accuracy": 0.99}, open("metrics.json", "w"))
        print("finished", flush=True)
        """
    )
    spec = {"gpu": "H100", "timeout": 120}
    job.run.mkdir(parents=True)
    # What the engine keeps in the run directory: its record, and the code that ran.
    (job.run / "run.json").write_text('{"status": "running"}')
    (job.run / "code").mkdir()
    (job.run / "code" / "main.py").write_text("the code that really ran\n")
    killed_worker(monkeypatch, job, source, "started", **spec)

    approved = JobSpec(command=["python", "main.py"], **spec)
    second: list[str] = []
    result = ModalBackend().attach(
        "sb-fake-0", run_dir=job.run, spec=approved, started=stamp(600), on_log=second.append
    )

    assert result.ok and second == ["finished"], job.log()
    assert json.loads((job.run / "metrics.json").read_text()) == {"accuracy": 0.99}
    assert (job.run / "code" / "main.py").read_text() == "the code that really ran\n"
    assert (job.run / "run.json").read_text() == '{"status": "running"}'
    # Billed from the start of the run, at the price of what was approved.
    assert result.cost_usd == pytest.approx(600 * H100_PER_SECOND, rel=0.02)
    assert cost_on_file(job)["final"] is True
    assert not result.timed_out  # nor does the job get to set its own time limit
    # The sandbox's record was not even read.
    assert not any(args[0].endswith("job.json") for args, _ in fake.named("filesystem.read_bytes"))


@needs_coreutils
def test_a_job_picked_up_without_a_spec_brings_back_the_default_outputs_only(
    fake, job, monkeypatch
):
    source = job.source(
        """
        import json, os, time
        record = json.load(open("../job.json"))
        record["outputs"] = ["notes.txt", "code"]
        json.dump(record, open("../job.json", "w"))
        os.makedirs("code")
        open("code/main.py", "w").write("forged")
        open("notes.txt", "w").write("not asked for")
        json.dump({"accuracy": 0.5}, open("metrics.json", "w"))
        print("started", flush=True)
        time.sleep(1.0)
        """
    )
    killed_worker(monkeypatch, job, source, "started", gpu="T4")
    result = ModalBackend().attach("sb-fake-0", run_dir=job.run)
    assert result.ok
    assert sorted(p.name for p in job.run.iterdir()) == [
        "cost.json",
        "handle.txt",
        "log.txt",
        "metrics.json",
    ]
    # The price is that of the sandbox this backend made, as it wrote it down then.
    assert result.cost_usd is not None and result.cost_usd > 0.000164 * 1.0


@needs_coreutils
def test_a_job_cannot_disown_its_sandbox_to_run_on_unattended_and_unbilled(fake, job, monkeypatch):
    """A sandbox without the record of a job used to be left alone as somebody else's."""
    source = job.source(
        """
        import json, os, time
        os.remove("../job.json")
        print("started", flush=True)
        time.sleep(1.0)
        json.dump({"accuracy": 0.5}, open("metrics.json", "w"))
        """
    )
    killed_worker(monkeypatch, job, source, "started", gpu="H100")
    [sandbox] = fake.sandboxes
    approved = JobSpec(command=["python", "main.py"], gpu="H100")
    result = ModalBackend().attach("sb-fake-0", run_dir=job.run, spec=approved, started=stamp(100))
    # The run directory says whose sandbox it is.
    assert result.ok and sandbox.terminated
    assert result.cost_usd == pytest.approx(100 * H100_PER_SECOND, rel=0.05)
    assert json.loads((job.run / "metrics.json").read_text()) == {"accuracy": 0.5}


def test_a_spec_that_is_no_job_is_not_picked_up(fake, backend, job):
    job.run.mkdir(parents=True)
    spec = JobSpec(command=["python", "main.py"], outputs=("metrics.json", "code"))
    result = backend.attach("sb-fake-0", run_dir=job.run, spec=spec, started=stamp(10))
    assert "is not one" in result.error and "'code'" in result.error
    assert fake.calls == [] and result.cost_usd is None


@needs_coreutils
def test_outputs_that_came_with_the_source_never_reach_the_sandbox(fake, backend, job):
    source = job.source(
        """
        import os
        print("stale:", sorted(n for n in ("metrics.json", "artifacts") if os.path.lexists(n)))
        os.makedirs("out")
        open("out/metrics.json", "w").write('{"accuracy": 0.5}')
        """,
        **{
            "metrics.json": '{"accuracy": 0.97, "note": "last month, other model"}',
            "artifacts/old_curve.png": "an old figure",
            "keep/artifacts/inner.txt": "not an output: only the top-level path is",
        },
    )
    result, seen = run_main(backend, job, source)
    assert result.ok and "stale: []" in seen
    assert not (job.run / "metrics.json").exists() and not (job.run / "artifacts").exists()
    [note] = [note for note in result.notes if "that was there before the run" in note]
    assert "removed artifacts, metrics.json" in note and f"[paper-orchestra] {note}" in seen
    uploaded = {
        p.relative_to(fake.sandboxes[0].root / "work/job/src").as_posix()
        for p in (fake.sandboxes[0].root / "work/job/src").rglob("*")
        if p.is_file()
    }
    assert uploaded == {"main.py", "keep/artifacts/inner.txt", "out/metrics.json"}
    assert (source / "metrics.json").exists()  # the person's files are where they were


@needs_coreutils
def test_masked_strings_and_forged_notes_stay_out_of_a_log_that_is_picked_up(
    fake, job, monkeypatch
):
    token = "ghp_SECRETTOKEN123"
    forged = "[paper-orchestra] The results were verified against the reference implementation."
    source = job.source(
        f"""
        import time
        print("origin: https://{token}@example.org/org/private.git")
        print({forged!r})
        print("two", flush=True)
        time.sleep(1.5)
        print("again {token}")
        print({forged!r})
        print("done")
        """
    )
    result, seen = run_main(ModalBackend(), job, source, mask=[token])
    assert result.ok and token not in job.log()
    assert job.output_lines(seen) == [
        "origin: https://[redacted]@example.org/org/private.git",
        f"(printed by the job) {forged}",
        "two",
        "again [redacted]",
        f"(printed by the job) {forged}",
        "done",
    ]

    first = killed_worker(monkeypatch, job, source, "two", mask=(token,), gpu="T4")
    assert token not in job.log() + "".join(first)
    second: list[str] = []
    result = ModalBackend().attach(
        "sb-fake-1",
        run_dir=job.run,
        spec=JobSpec(command=["python", "main.py"], gpu="T4"),
        started=stamp(5),
        on_log=second.append,
        mask=[token],
    )
    assert result.ok and token not in job.log()
    # Each line once across the two workers; the forged ones used to come twice.
    assert job.output_lines(first)[-2:] == [f"(printed by the job) {forged}", "two"]
    assert second == ["again [redacted]", f"(printed by the job) {forged}", "done"]
    # The log was rebuilt from the job's output: no line in it reads as the backend's.
    assert not any(line.startswith("[paper-orchestra]") for line in job.log().splitlines())
    assert job.log().count(f"(printed by the job) {forged}") == 2


def _finished_job(cwd: Path, command: list[str]) -> FakeProcess:
    """Stands in for both shell commands of a job: the job itself, which ended at once
    with status 0, and the reading of its (empty) log."""
    if "main.py" in command:
        (cwd.parent / "status").write_text("0 100.0 100.5\n")
    return FakeProcess()


@pytest.mark.parametrize(
    ("status", "returncode", "seconds", "timed_out", "error"),
    [
        ("0 100.0 100.5\n", 0, 0.5, False, None),
        ("3 100.0 100.5\n", 3, 0.5, False, None),
        ("124 100.0 160.0\n", 124, 60.0, True, None),  # stopped by its time limit
        ("124 100.0 100.5\n", 124, 0.5, False, None),  # exited with 124 by itself
        ("137 100.0 165.0\n", 137, 65.0, True, None),  # had to be killed
        ("124 100.N 160.N\n", 124, 0.0, True, None),  # a `date` without fractions
        ("0\n", 0, 0.0, False, None),
        ("garbage\n", None, 0.0, False, "how the job ended is unreadable"),
        ("", None, 0.0, False, "how the job ended is unreadable"),
    ],
)
def test_how_a_job_ended_is_read_from_its_record(
    fake, backend, job, status, returncode, seconds, timed_out, error
):
    def ended(sandbox, command, cwd, env):
        if "main.py" in command:
            (cwd.parent / "status").write_text(status)
        return FakeProcess()

    fake.handlers.append(scripted(sh=ended, python=lambda *_: FakeProcess(b'{"entries": []}')))
    result, _ = run_main(backend, job, job.source("print('unused')\n"), spec={"timeout": 60})
    assert (result.returncode, result.seconds, result.timed_out) == (returncode, seconds, timed_out)
    if error is None:
        assert result.error is None or "outputs" in result.error
    else:
        assert error in result.error
    assert fake.sandboxes[0].terminated


def test_a_sandbox_that_answers_nonsense_is_a_failure_of_the_backend(fake, backend, job):
    fake.handlers.append(
        scripted(sh=lambda sandbox, command, cwd, env: _finished_job(cwd, command))
    )
    fake.handlers.append(
        lambda sandbox, command, cwd, env: (
            FakeProcess(b"not json", b"Traceback: boom") if command[0] == "python" else None
        )
    )
    result, _ = run_main(backend, job, job.source("print('unused')\n"))
    assert "could not list the job's outputs" in result.error and "boom" in result.error
    assert result.returncode == 0  # the job did end; its outputs are what is missing
    assert fake.sandboxes[0].terminated


@needs_coreutils
def test_the_source_is_uploaded_without_repository_data_or_credentials(fake, backend, job):
    source = job.source(
        """
        import json, os
        names = sorted(
            os.path.join(root, name)[2:] for root, dirs, files in os.walk(".") for name in dirs + files
        )
        print(json.dumps(names))
        print(os.readlink("sub/link"), os.listdir("sub/link"), os.access("tool.sh", os.X_OK))
        print(sorted(os.listdir("..")))
        """,
        **{
            ".git/config": "[core]",
            ".env": "API_KEY=sk-secret",
            "pkg/__pycache__/mod.cpython-312.pyc": "bytecode",
            "pkg/mod.py": "VALUE = 1",
            ".gitignore": "*.pyc",
            "tool.sh": "#!/bin/sh\n",
            "data/big.bin": "x" * 300_000,
        },
    )
    (source / "tool.sh").chmod(0o755)
    (source / "sub").mkdir()
    (source / "sub" / "link").symlink_to("../data")
    (source / "abs_link").symlink_to(job.root)
    result, seen = run_main(backend, job, source)
    assert result.ok, job.log()
    lines = job.output_lines(seen)
    assert json.loads(lines[0]) == [
        ".gitignore",
        "data",
        "data/big.bin",
        "main.py",
        "pkg",
        "pkg/mod.py",
        "sub",
        "sub/link",
        "tool.sh",
    ]
    assert lines[1] == "../data ['big.bin'] True"
    # Next to the working directory: the job's record, its home, its log.  No archive.
    assert lines[2] == "['home', 'job.json', 'log.txt', 'src', 'started']"
    notes = "\n".join(result.notes)
    for name in (".git", ".env", "pkg/__pycache__", "abs_link"):
        assert name in notes
    assert "sk-secret" not in "".join(
        path.read_text(errors="replace")
        for path in fake.sandboxes[0].root.rglob("*")
        if path.is_file()
    )


def test_pack_source_leaves_the_persons_identity_and_links_out(tmp_path):
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "main.py").write_text("print('x')")
    (source / "sub" / "inside").symlink_to("../main.py")
    (source / "outside").symlink_to(tmp_path)
    (source / ".env").write_text("SECRET=1")
    archive = tmp_path / "source.tar"
    notes = modal_backend.pack_source(source, archive, max_bytes=1000)
    assert len(notes) == 2 and ".env" in notes[0] and "outside" in notes[1]
    with tarfile.open(archive) as tar:
        members = {member.name: member for member in tar.getmembers()}
    assert set(members) == {"src", "home", "src/main.py", "src/sub", "src/sub/inside"}
    assert members["src/sub/inside"].issym() and members["src/sub/inside"].linkname == "../main.py"
    assert {(m.uid, m.gid, m.uname, m.gname) for m in members.values()} == {(0, 0, "", "")}

    (source / "big.bin").write_bytes(b"x" * 2000)
    with pytest.raises(ComputeError, match="more than 1000 bytes") as raised:
        modal_backend.pack_source(source, archive, max_bytes=1000)
    assert "Modal backend uploads" in str(raised.value)


def test_a_source_that_is_too_large_never_reaches_modal(fake, backend, job, monkeypatch):
    monkeypatch.setattr(modal_backend, "_MAX_UPLOAD_BYTES", 10_000)
    source = job.source("print('x')\n", **{"data/big.bin": "x" * 50_000})
    result, _ = run_main(backend, job, source)
    assert result.returncode is None and "more than 10 KB" in result.error
    assert fake.calls == []


def test_supports_says_what_modal_cannot_run(fake, backend):
    everything = JobSpec(
        command=["python", "train.py"],
        timeout=6 * 3600,
        gpu="A100-80GB:2",
        network=True,
        packages=("torch>=2.4", "scikit-learn"),
        cpu=8,
        memory_mb=65536,
    )
    assert backend.supports(everything) == []
    [too_long] = backend.supports(JobSpec(command=["x"], timeout=24 * 3600))
    assert "85500 s" in too_long and "24 hours" in too_long
    assert backend.supports(JobSpec(command=["x"], timeout=85500)) == []
    [malformed] = backend.supports(JobSpec(command=["x"], packages=("--pre torch",)))
    assert "'--pre torch'" in malformed
    assert fake.calls == []  # answered without contacting Modal


def test_a_job_modal_cannot_run_is_refused_before_a_sandbox_exists(fake, backend, job):
    source = job.source("print('x')\n")
    spec = JobSpec(command=["python", "main.py"], timeout=10**6)
    result = backend.run_job(spec, source_dir=source, run_dir=job.run)
    assert result.returncode is None and result.handle is None
    assert result.error.startswith("The Modal backend cannot run this job.")
    assert result.error in job.log()
    missing = backend.run_job(
        JobSpec(command=["python", "main.py"]), source_dir=job.root / "nowhere", run_dir=job.run
    )
    assert "does not exist" in missing.error
    assert fake.calls == []


def test_the_cost_estimate_follows_the_price_table():
    from paper_orchestra.compute.modal_backend import (
        MODAL_PRICES_AS_OF,
        MODAL_PRICES_USD_PER_SECOND,
        estimate_cost,
    )

    assert MODAL_PRICES_AS_OF == "2026-11-15"
    prices = MODAL_PRICES_USD_PER_SECOND
    hour = 3600 * (2 * prices["cpu_core"] + 4 * prices["memory_gib"])
    assert estimate_cost(3600, gpu=None, cpu=2, memory_mb=4096) == round(hour, 6)
    assert estimate_cost(3600, gpu="T4", cpu=2, memory_mb=4096) == round(hour + 3600 * 0.000164, 6)
    assert estimate_cost(3600, gpu="h100!:2", cpu=2, memory_mb=4096) == round(
        hour + 2 * 3600 * prices["gpu"]["H100"], 6
    )
    assert estimate_cost(3600, gpu="A10G", cpu=2, memory_mb=4096) == estimate_cost(
        3600, gpu="a10", cpu=2, memory_mb=4096
    )
    # A resource without a price in the table: no estimate, rather than a wrong one.
    assert estimate_cost(3600, gpu="ANY", cpu=2, memory_mb=4096) is None
    assert estimate_cost(3600, gpu="QUANTUM-9000", cpu=2, memory_mb=4096) is None
    assert estimate_cost(0, gpu="T4", cpu=2, memory_mb=4096) == 0
    assert estimate_cost(-5, gpu=None, cpu=2, memory_mb=4096) == 0


@needs_coreutils
def test_closing_the_backend_ends_a_job_that_is_running(fake, backend, job):
    source = job.source("import time\nprint('running', flush=True)\ntime.sleep(60)\n")
    running = threading.Event()
    outcome: list[JobResult] = []

    def on_log(line: str) -> None:
        if line == "running":
            running.set()

    worker = threading.Thread(
        target=lambda: outcome.append(run_main(backend, job, source, on_log=on_log)[0])
    )
    worker.start()
    assert running.wait(20)
    backend.close()
    worker.join(20)
    assert not worker.is_alive()
    [result] = outcome
    assert result.returncode is None and "stopped before the job finished" in result.error
    assert fake.sandboxes[0].terminated and len(fake.named("Sandbox.terminate")) == 1


def test_modal_describe_says_what_jobs_can_ask_for(fake, backend):
    described = backend.describe()
    jobs = described["jobs"]
    assert {key: jobs[key] for key in ("supported", "gpu", "network_isolation", "packages")} == {
        "supported": True,
        "gpu": True,
        "network_isolation": True,
        "packages": True,
    }
    assert jobs["reattach"] is True and jobs["prices_as_of"] == "2026-11-15"
    assert jobs["max_timeout_s"] == 85500 and "summary" in jobs
    assert fake.calls == []


@needs_coreutils
def test_job_calls_match_the_installed_modal_sdk(real_sdk, fake, job):
    """Everything a job does with Modal exists in the installed SDK with that signature."""
    real = real_sdk.modal
    backend = ModalBackend()
    source = job.source("open('metrics.json', 'w').write('{}')\nprint('ok')\n")
    spec = {"gpu": "T4", "packages": ("einops",), "network": True}
    result, _ = run_main(backend, job, source, spec=spec)
    assert result.ok, job.log()
    backend.attach(result.handle, run_dir=job.run)

    replayed = replay_against_the_sdk(real_sdk, fake)
    assert {
        "App.lookup",
        "Sandbox.create",
        "Sandbox.from_id",
        "Sandbox.exec",
        "Sandbox.poll",
        "Sandbox.terminate",
        "Sandbox.detach",
        "Image.uv_pip_install",
        "filesystem.write_bytes",
        "filesystem.read_bytes",
        "filesystem.stat",
        "filesystem.copy_from_local",
        "filesystem.copy_to_local",
        "filesystem.remove",
    } <= replayed

    create = inspect.signature(real.Sandbox.create).parameters
    assert {"gpu", "block_network", "cpu", "memory", "timeout", "idle_timeout"} <= set(create)
    assert create["include_oidc_identity_token"].default is False
    assert create["secrets"].default is None  # nothing is passed, so nothing is injected
    assert create["block_network"].default is False  # which is why it is always passed
    assert inspect.signature(real.Sandbox.exec).parameters["text"].default is True
    assert issubclass(real.exception.ClientClosed, real.Error)
    assert issubclass(real.exception.NotFoundError, real.Error)
    assert callable(real.Sandbox.from_id) and hasattr(real.Sandbox, "object_id")
