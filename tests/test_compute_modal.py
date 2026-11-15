from __future__ import annotations

import inspect
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
import tracemalloc
import types
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import paper_orchestra.compute.modal_backend as modal_backend
from paper_orchestra.compute import BACKEND_NAMES, ComputeError, get_backend
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.compute.modal_backend import (
    TEXLIVE_INSTALL_COMMAND,
    ModalBackend,
    modal_token_configured,
    pack_tree,
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

    def require_running(self) -> None:
        if self.exit_code is not None:
            raise self.hub.exception.NotFoundError("The Sandbox is unavailable")

    def exec(self, *args: str, **kwargs: Any) -> FakeProcess:
        self.hub.record("Sandbox.exec", *args, **kwargs)
        self.require_running()
        if not isinstance(kwargs.get("timeout"), int):
            raise TypeError("'float' object cannot be interpreted as an integer")
        assert args[:3] == ("timeout", "--signal=TERM", "--kill-after=5"), args
        command = list(args[4:])
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
        return self.exit_code

    def terminate(self, *, wait: bool = False) -> None:
        self.hub.record("Sandbox.terminate", wait=wait)
        self.terminated = True
        if self.exit_code is None:
            self.exit_code = 137

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

        exception = types.ModuleType("modal.exception")

        class Error(Exception):
            pass

        for name, base in (
            ("AuthError", Error),
            ("InvalidError", Error),
            ("NotFoundError", Error),
            ("SandboxFilesystemError", Error),
        ):
            setattr(exception, name, type(name, (base,), {}))
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
        module.Sandbox = types.SimpleNamespace(create=self._create)
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
        if self.create_error is not None:
            raise self.create_error
        sandbox = FakeSandbox(self, len(self.sandboxes), kwargs)
        self.sandboxes.append(sandbox)
        return sandbox


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


@pytest.fixture
def fake(monkeypatch, tmp_path) -> FakeModal:
    hub = FakeModal(tmp_path / "modal")
    for name, module in hub.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return hub


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
    assert image.named("Image.run_commands") == [(TEXLIVE_INSTALL_COMMAND,)]
    assert "--no-install-recommends" in TEXLIVE_INSTALL_COMMAND
    assert "texlive-latex-extra" in TEXLIVE_INSTALL_COMMAND
    assert "texlive-plain-generic" in TEXLIVE_INSTALL_COMMAND
    assert created["block_network"] is True and not created.get("secrets")
    assert created["include_oidc_identity_token"] is False


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
    with pytest.raises(ComputeError, match="modal token new"):
        backend.run_python("pass")
    fake.lookup_error = None

    fake.create_error = fake.exception.Error("Insufficient capacity")
    with pytest.raises(ComputeError, match=r"could not start a Modal sandbox.*Insufficient"):
        backend.run_python("pass")
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


def test_inside_a_modal_function_the_container_app_owns_the_sandbox(fake, backend):
    fake.local = False
    fake.handlers.append(scripted(python=lambda *_: FakeProcess()))
    backend.run_python("pass")
    assert fake.named("App.lookup") == []
    [(_, created)] = fake.named("Sandbox.create")
    assert created["app"] is None and created["block_network"] is True
    assert backend.describe()["inside_modal_function"] is True


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
    assert "$ bibtex main\nThis is BibTeX" in result.log_tail and "caf\ufffd" in result.log_tail

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

    targets = {
        "App.lookup": real.App.lookup,
        "Sandbox.create": real.Sandbox.create,
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
    assert replayed == set(targets)

    assert not hasattr(real.Sandbox, "__enter__")
    assert issubclass(real.exception.AuthError, real.Error)
    assert issubclass(real.exception.SandboxFilesystemNotFoundError, real.Error)
    assert callable(real.is_local) and callable(real.config.config.get)
    assert callable(real_sdk.FileInfo.is_file)
    assert "size" in real_sdk.FileInfo.__dataclass_fields__
    assert callable(real_sdk.ContainerProcess.wait)
    assert {"stdout", "stderr"} <= set(dir(real_sdk.ContainerProcess))
    assert callable(real_sdk.StreamReader.__iter__)  # output is read chunk by chunk
