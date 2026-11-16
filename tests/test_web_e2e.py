"""The web app end to end: a real server, a real browser, the scripted model.

``tests/e2e/web_check.js`` drives the frontend through the flows a person goes through
(see ``tests/e2e/README.md``).  Here it gets what it needs: a temporary workspace with the
demo project, and a server on a free loopback port whose workers use the scripted model
of ``tests/e2e/scripted_model.py``.  The server and the browser are child processes; this
process opens no socket (it learns the server's address from what the server prints).

Skipped, with the reason, where Node, Playwright or a browser is missing.
"""

from __future__ import annotations

import os
import re
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
ADDRESS = re.compile(r"running at (http://127\.0\.0\.1:\d+)/")


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
    """Stop the server the way Ctrl-C does (it then stops its workers), then make sure."""
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture
def server(tmp_path):
    """A server on a temporary demo workspace; yields its address."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(E2E_DIR), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
        "PAPER_ORCHESTRA_SCRIPTED_CLIENT": "scripted_model:client",
        "PAPER_ORCHESTRA_E2E_PAUSE": "0.5",
        "PYTHONUNBUFFERED": "1",
    }
    built = _run(
        [
            sys.executable,
            "-m",
            "paper_orchestra",
            "research",
            "demo",
            "--dir",
            str(workspace / "demo"),
        ],
        env=env,
        timeout=180,
    )
    assert built.returncode == 0, (
        f"the demo project could not be built:\n{built.stdout}\n{built.stderr}"
    )
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
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
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
        yield address
    finally:
        _stop(process)


def test_the_app_works_in_a_real_browser(server):
    reason = _playwright_missing()
    if reason:
        pytest.skip(reason)
    result = _run(["node", str(CHECK), "--base", server], env=dict(os.environ), timeout=900)
    if result.returncode == NO_BROWSER:
        pytest.skip("Playwright could not start a browser: " + result.stderr.strip()[-300:])
    assert result.returncode == 0, f"the browser checks failed:\n{result.stdout}\n{result.stderr}"
    assert "all checks passed" in result.stdout
