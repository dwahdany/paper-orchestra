"""The web app end to end: a real server, a real browser, the scripted model.

``tests/e2e/web_check.js`` drives the frontend through the flows a person goes through
(see ``tests/e2e/README.md``).  Here it gets what it needs: a temporary workspace with the
demo project, the same project filled up to 150 pages and a third copy in a folder named
``nomodel``, and a server on a free loopback port whose workers use the scripted model
of ``tests/e2e/scripted_model.py``.  The server and the browser are child processes; this
process opens no socket (it learns the server's address from what the server prints).

One check looks at a project whose models cannot be used, which is where everybody is who
tries the demonstration without credentials.  The scripted model can be switched off for
that: while the file named by ``PAPER_ORCHESTRA_E2E_NO_MODEL`` exists, it offers no tools,
and the server and its workers say that a model is needed.  The fixture names such a file
(it does not create it) to the server and to the check, which writes and removes it.

The server asks for its access token, as every server does.  The fixture makes one, gives
it to the server in ``PAPER_ORCHESTRA_UI_TOKEN`` and hands it to the check the same way, so
nothing but this file is needed to run the test.

The checks pass in both colour schemes; this test runs them in the light one, and with
``PAPER_ORCHESTRA_E2E_SCHEME=dark`` in the dark one.

Skipped, with the reason, where Node, Playwright or a browser is missing.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

E2E_DIR = Path(__file__).resolve().parent / "e2e"
CHECK = E2E_DIR / "web_check.js"
#: The exit status of ``web_check.js`` when no browser could be started.
NO_BROWSER = 3
#: The address in the link the server prints once it listens (a server that could not
#: start names its address too, without the token: that is not a server to check).
ADDRESS = re.compile(r"(http://127\.0\.0\.1:\d+)/\?token=")
#: The variable that fixes the server's access token, and through which the check gets it.
TOKEN_ENV = "PAPER_ORCHESTRA_UI_TOKEN"
#: The variable that names the file whose presence switches the scripted model off.
NO_MODEL_ENV = "PAPER_ORCHESTRA_E2E_NO_MODEL"
#: The variable that names Modal's configuration file (``~/.modal.toml`` without it).
MODAL_CONFIG_ENV = "MODAL_CONFIG_PATH"
#: The option of ``paper-orchestra ui`` that ends the workers with the server.
STOP_WORKERS = "--stop-workers-on-exit"
#: ``dark`` runs the flows in the dark theme (``web_check.js --scheme dark``).
SCHEME = os.environ.get("PAPER_ORCHESTRA_E2E_SCHEME", "").strip().lower()


def _playwright_missing() -> str | None:
    """Why the browser test cannot run here, or ``None`` when it can."""
    if os.environ.get("PAPER_ORCHESTRA_SKIP_E2E"):
        return "PAPER_ORCHESTRA_SKIP_E2E is set"
    if shutil.which("node") is None:
        return "node is not installed"
    probe = subprocess.run(
        ["node", "-e", "require('playwright')"], capture_output=True, text=True, timeout=60
    )
    if probe.returncode != 0:
        return "the Node package 'playwright' is not installed"
    return None


def _run(command: list[str], *, env: dict[str, str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)


def _stop(process: subprocess.Popen) -> None:
    """Stop the server the way Ctrl-C does, then make sure."""
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _server_env(tmp_path: Path, no_model: Path) -> dict[str, str]:
    """The environment of the server of the test and of what builds its projects.

    The checks look at a machine on which only the local backend can run something (the
    demo's GPU experiment is "No backend can run this", and "run on" offers Local CPU and
    only that).  A developer's machine may be signed in to Modal, and the server would then
    offer Modal for them.  So Modal finds no credentials here: its token variables are left
    out and its configuration is a file that does not exist.  Nothing of the person's is
    read or changed.
    """
    return {
        **{
            name: value
            for name, value in os.environ.items()
            if name != TOKEN_ENV and not name.startswith("MODAL_")
        },
        "PYTHONPATH": os.pathsep.join(
            [str(E2E_DIR), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
        "PAPER_ORCHESTRA_SCRIPTED_CLIENT": "scripted_model:client",
        "PAPER_ORCHESTRA_E2E_PAUSE": "0.5",
        NO_MODEL_ENV: str(no_model),
        MODAL_CONFIG_ENV: str(tmp_path / "no-modal.toml"),
        "PYTHONUNBUFFERED": "1",
    }


def test_the_server_of_the_test_cannot_use_modal(tmp_path, monkeypatch):
    """The browser checks expect that nothing but this machine can run an experiment,
    whatever the person who runs them is signed in to."""
    pytest.importorskip("modal")
    signed_in = tmp_path / "modal.toml"
    signed_in.write_text('[default]\ntoken_id = "ak-test"\ntoken_secret = "as-test"\n')
    ask = (
        "from paper_orchestra.compute.modal_backend import modal_token_configured as c;"
        "print('modal:', c())"
    )
    monkeypatch.setenv(MODAL_CONFIG_ENV, str(signed_in))
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    # (this machine is signed in, as far as a child of this process can tell)
    signed = _run([sys.executable, "-c", ask], env=dict(os.environ), timeout=60)
    assert "modal: True" in signed.stdout, signed.stderr
    env = _server_env(tmp_path, tmp_path / "no-model")
    served = _run([sys.executable, "-c", ask], env=env, timeout=60)
    assert "modal: False" in served.stdout, served.stderr


@pytest.fixture
def server(tmp_path):
    """A server on a temporary demo workspace; yields its address, its access token and
    the file whose presence switches the scripted model off."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    token = secrets.token_urlsafe(24)
    # Not created here: the models work until the check that wants them gone writes it.
    no_model = tmp_path / "no-model"
    env = _server_env(tmp_path, no_model)
    # The demo project, and beside it the same project filled up to 150 pages: the checks
    # of a large project (the map folded, a large inbox decided at once, nothing typed in
    # one project turning up in another) need a second project to look at.  A third copy,
    # which no check touches before that one, is the project that is looked at without a model.
    for name, extra in (("demo", []), ("large", ["--pages", "150"]), ("nomodel", [])):
        built = _run(
            [
                sys.executable,
                "-m",
                "paper_orchestra",
                "research",
                "demo",
                "--dir",
                str(workspace / name),
                *extra,
            ],
            env=env,
            timeout=180,
        )
        assert built.returncode == 0, (
            f"the {name} project could not be built:\n{built.stdout}\n{built.stderr}"
        )
    # A server that is stopped leaves its workers running unless it is told otherwise. A
    # check that fails half way would leave one behind, so the server of the test takes
    # its workers with it where the command line has the option for that.
    described = _run(
        [sys.executable, "-m", "paper_orchestra", "ui", "--help"],
        env={**env, "COLUMNS": "200"},
        timeout=60,
    )
    # The help is drawn as a table whose cells may be wrapped: compare without the drawing.
    offered = STOP_WORKERS in re.sub(r"[^\w-]+", "", described.stdout)
    with_workers = [STOP_WORKERS] if offered else []
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "paper_orchestra",
            "ui",
            "--workspace",
            str(workspace),
            "--port",
            "0",
            "--no-browser",
            *with_workers,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**env, TOKEN_ENV: token},
    )
    lines: list[str] = []
    found = threading.Event()

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if ADDRESS.search(line):
                found.set()

    threading.Thread(target=read, daemon=True).start()
    try:
        deadline = time.monotonic() + 60
        while not found.is_set() and process.poll() is None and time.monotonic() < deadline:
            found.wait(0.2)
        assert found.is_set(), "the server did not say where it runs:\n" + "".join(lines)
        address = next(m.group(1) for m in map(ADDRESS.search, lines) if m)
        yield address, token, no_model
    finally:
        _stop(process)
        no_model.unlink(missing_ok=True)


def test_the_app_works_in_a_real_browser(server):
    reason = _playwright_missing()
    if reason:
        pytest.skip(reason)
    address, token, no_model = server
    scheme = ["--scheme", "dark"] if SCHEME == "dark" else []
    # The token goes to the check in its environment, not as an argument; so does the
    # name of the file that switches the scripted model off.
    result = _run(
        ["node", str(CHECK), "--base", address, *scheme],
        env={**os.environ, TOKEN_ENV: token, NO_MODEL_ENV: str(no_model)},
        timeout=1500,
    )
    if result.returncode == NO_BROWSER:
        pytest.skip("Playwright could not start a browser: " + result.stderr.strip()[-300:])
    assert result.returncode == 0, f"the browser checks failed:\n{result.stdout}\n{result.stderr}"
    assert "all checks passed" in result.stdout
