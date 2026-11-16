"""The web server itself: what it lets in at the door, the files it opens, how it starts
and how it ends.

The API's endpoints are tested in ``test_web_api.py`` and the worker processes in
``test_web_workers.py``; here is what lies between them and the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from paper_orchestra.research import Project
from paper_orchestra.web import app as app_module
from paper_orchestra.web import security
from paper_orchestra.web.api import UploadLimits
from paper_orchestra.web.app import create_app
from paper_orchestra.web.security import CONTENT_SECURITY_POLICY, PathEscape, SecurityMiddleware

BASE_URL = "http://127.0.0.1:8600"
MEGABYTE = 1024 * 1024


def wait_for(condition, timeout: float = 60.0, message: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {message}")
        time.sleep(0.02)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "research"
    path.mkdir()
    return path


@pytest.fixture
def project(workspace: Path) -> Project:
    return Project.create(workspace / "warmup", title="Warmup", brief="Does warmup matter?")


@pytest.fixture
def client(workspace, tmp_path):
    app = create_app(workspace, static_dir=tmp_path / "no-frontend")
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as client:
        yield client


class Exchange:
    """One request put to an ASGI app by hand: the body arrives in the pieces given, and
    what the app took and answered is on record."""

    def __init__(
        self,
        app,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        pieces: list[bytes] | None = None,
        query: str = "",
    ) -> None:
        self.pieces = list(pieces or [])
        self.taken = 0
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.body = b""
        self.starts = 0
        #: What the app raised after (or instead of) answering, as a real server would log it.
        self.raised: BaseException | None = None
        sent = {"host": "127.0.0.1:8600", **(headers or {})}
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode("latin-1"),
            "root_path": "",
            "headers": [(k.lower().encode(), v.encode("latin-1")) for k, v in sent.items()],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8600),
        }
        try:
            asyncio.run(app(scope, self._receive, self._send))
        except Exception as exc:
            self.raised = exc

    async def _receive(self):
        if self.taken < len(self.pieces):
            self.taken += 1
            more = self.taken < len(self.pieces)
            return {"type": "http.request", "body": self.pieces[self.taken - 1], "more_body": more}
        if not self.pieces and not self.taken:
            self.taken += 1
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def _send(self, message):
        if message["type"] == "http.response.start":
            self.starts += 1
            self.status = message["status"]
            self.headers = {k.decode().lower(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")

    def error(self) -> dict[str, str]:
        return json.loads(self.body)["error"]


# ---------------------------------------------------------------------------
# Request bodies are bounded while they arrive
# ---------------------------------------------------------------------------


def test_a_body_that_announces_more_than_the_limit_is_refused_unread(workspace, tmp_path):
    app = create_app(workspace, static_dir=tmp_path / "none")
    pieces = [b"x" * MEGABYTE] * 64
    exchange = Exchange(
        app,
        "POST",
        "/api/render",
        headers={"content-type": "application/json", "content-length": str(64 * MEGABYTE)},
        pieces=pieces,
    )
    assert exchange.status == 413 and exchange.error()["code"] == "too_large"
    assert "MB" in exchange.error()["message"]
    assert exchange.taken == 0  # not one byte was taken in
    assert exchange.headers["x-content-type-options"] == "nosniff"


def test_a_body_without_a_length_is_cut_off_while_it_arrives(workspace, tmp_path):
    # Chunked, or a length that lies: what counts is what arrives.  The server used to
    # take the whole body into memory (three times over) before any endpoint looked at it.
    app = create_app(workspace, static_dir=tmp_path / "none")
    for headers in ({}, {"content-length": "10"}, {"content-type": "multipart/form-data"}):
        exchange = Exchange(
            app,
            "POST",
            "/api/render",
            headers={"content-type": "application/json", **headers},
            pieces=[b"x" * MEGABYTE] * 64,
        )
        assert exchange.status == 413 and exchange.error()["code"] == "too_large", headers
        assert exchange.starts == 1  # one answer, whatever the endpoint made of the cut
        assert exchange.taken <= security.MAX_BODY_BYTES // MEGABYTE + 1, headers
        assert exchange.headers["content-security-policy"]


def test_every_method_and_every_endpoint_is_bounded(workspace, project, tmp_path):
    app = create_app(workspace, static_dir=tmp_path / "none", max_body_bytes=2000)
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as client:
        big = {"text": "x" * 3000}
        base = f"/api/projects/{project.id}"
        for method, path in [
            ("post", "/api/render"),
            ("post", f"{base}/requests"),
            ("put", f"{base}/pages/idea"),
            ("patch", base),
            ("post", f"{base}/review"),
            ("post", f"{base}/advance"),
            ("post", "/api/nothing/here"),
            ("put", "/not-even-the-api"),
        ]:
            refused = getattr(client, method)(path, json=big)
            assert refused.status_code == 413, (method, path, refused.text)
        assert client.request("DELETE", f"{base}/pages/idea", json=big).status_code == 413
        assert project.requests() == []
        # What fits is taken as before, to the last byte of the limit.
        small = client.post(f"{base}/requests", json={"text": "Use three seeds."})
        assert small.status_code == 201, small.text
        exact = json.dumps({"markdown": "y" * 1000}).encode().ljust(2000)
        fits = client.post(
            "/api/render", content=exact, headers={"content-type": "application/json"}
        )
        assert fits.status_code == 200, fits.text


def test_the_upload_of_a_new_project_keeps_its_own_limit(workspace, tmp_path):
    limits = UploadLimits(file_bytes=40_000, total_bytes=60_000, text_bytes=10_000)
    app = create_app(workspace, static_dir=tmp_path / "none", max_body_bytes=2000, uploads=limits)
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/projects",
            data={"title": "With a file", "brief": "What is in it?"},
            files=[("files", ("data.csv", b"a,b\n" * 5000, "text/csv"))],
        )
        assert created.status_code == 201, created.text
        assert (workspace / "with-a-file" / "inputs" / "data.csv").stat().st_size == 20_000
        # ... and that limit is the upload's, not no limit at all.
        too_much = client.post(
            "/api/projects",
            data={"title": "Too much", "brief": "x"},
            files=[
                ("files", (f"f{n}.bin", b"\0" * 39_000, "application/octet-stream"))
                for n in range(80)
            ],
        )
        assert too_much.status_code == 413
        assert not (workspace / "too-much").exists()


def test_the_limit_is_kept_by_the_middleware_alone():
    # Whatever the application does with a body that was cut (ignore it, answer 400, fail),
    # the client is told that the body was too large, once.
    async def swallows(scope, receive, send):
        try:
            while (await receive()).get("more_body"):
                pass
        except Exception:
            pass
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"could not parse"})

    async def fails(scope, receive, send):
        while (await receive()).get("more_body"):
            pass

    async def ignores_the_body(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    for inner in (swallows, fails):
        guarded = SecurityMiddleware(inner, max_body_bytes=100)
        exchange = Exchange(guarded, "POST", "/api/x", pieces=[b"a" * 60, b"b" * 60, b"c" * 60])
        assert (exchange.status, exchange.starts) == (413, 1), inner.__name__
        assert exchange.error()["code"] == "too_large" and exchange.taken == 2
        assert b"could not parse" not in exchange.body
    untouched = Exchange(
        SecurityMiddleware(ignores_the_body, max_body_bytes=100), "POST", "/api/x", pieces=[b"a"]
    )
    assert untouched.status == 204
    # A length no number can hold is not a length.
    odd = Exchange(
        SecurityMiddleware(ignores_the_body, max_body_bytes=100),
        "POST",
        "/api/x",
        headers={"content-length": "9" * 5000},
    )
    assert odd.status == 413


# ---------------------------------------------------------------------------
# Headers and addresses that used to end in "unexpected error"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resumed", ["\u00b2", "\u00b9\u00b2", "9" * 5000, "1e3", "-1", " ", "12abc"]
)
def test_a_last_event_id_that_is_no_number_is_not_an_error(workspace, project, tmp_path, resumed):
    # Put to the app as a server hands it over: the bytes of the header, not what a test
    # client makes of them.  A superscript two is a digit to ``str.isdigit`` and no number.
    app = create_app(workspace, static_dir=tmp_path / "none")
    app.state.context.live.closing.set()  # the stream says hello and ends
    exchange = Exchange(
        app, "GET", f"/api/projects/{project.id}/events", headers={"last-event-id": resumed}
    )
    assert exchange.status == 200 and exchange.raised is None, exchange.body
    assert exchange.body.startswith(b"retry:")


def test_a_last_event_id_the_server_gave_out_is_still_honoured():
    seen = {}

    async def inner(scope, receive, send):
        seen["headers"] = dict(scope["headers"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guarded = SecurityMiddleware(inner)
    Exchange(guarded, "GET", "/api/projects/p/events", headers={"last-event-id": "41"})
    assert seen["headers"][b"last-event-id"] == b"41"
    Exchange(guarded, "GET", "/api/projects/p/events", headers={"last-event-id": "²"})
    assert b"last-event-id" not in seen["headers"]
    assert seen["headers"][b"host"] == b"127.0.0.1:8600"  # the others are as they came


@pytest.mark.parametrize(
    "path",
    [
        "/pages/" + "a" * 300,
        "/pages/topics/" + "b" * 5000,
        "/files/inputs/" + "c" * 300,
        "/files/inputs/" + "/".join(["d" * 200] * 40),
        "/experiments/" + "E" * 300,
        "/experiments/E1/runs/1/files/" + "f" * 300,
    ],
)
def test_an_address_longer_than_any_file_name_is_not_found(client, project, path):
    response = client.get(f"/api/projects/{project.id}{path}")
    assert response.status_code == 404, response.text
    assert set(response.json()["error"]) == {"code", "message"}
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_name_too_long_to_store_is_refused_not_an_error(client, project):
    changed = client.put(
        f"/api/projects/{project.id}/experiments/E1/code/" + "g" * 300 + ".py",
        json={"text": "print(1)\n"},
    )
    assert changed.status_code in (404, 422), changed.text
    assert changed.json()["error"]["code"] != "internal"


def test_any_other_error_of_the_system_is_still_an_unexpected_error(workspace, project, tmp_path):
    app = create_app(workspace, static_dir=tmp_path / "none")

    def denied() -> dict:
        raise PermissionError(13, "Permission denied")

    app.state.context.workspace.scan = denied
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as client:
        response = client.get("/api/projects")
        assert response.status_code == 500 and response.json()["error"]["code"] == "internal"
        assert "PermissionError" in response.json()["error"]["message"]
        assert response.headers["x-content-type-options"] == "nosniff"


# ---------------------------------------------------------------------------
# Files the server opens for a project
# ---------------------------------------------------------------------------


def test_a_file_is_opened_inside_the_project_or_not_at_all(tmp_path):
    root = tmp_path / "project"
    (root / "experiments" / "E1" / "runs" / "1").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SECRET")
    (root / "worker.log").write_text("the worker's own lines\n")
    log = root / "experiments" / "E1" / "runs" / "1" / "log.txt"
    log.symlink_to(outside)
    (root / "linked.log").symlink_to(outside)
    (root / "inside.log").symlink_to(root / "worker.log")

    with security.open_confined(root, "worker.log") as handle:
        assert handle.read() == b"the worker's own lines\n"
    with security.open_confined(root, "inside.log") as handle:  # a link that stays inside
        assert handle.read() == b"the worker's own lines\n"
    for escaping in ("experiments/E1/runs/1/log.txt", "linked.log", "../outside.txt"):
        with pytest.raises(PathEscape):
            security.open_confined(root, escaping)
    with pytest.raises(FileNotFoundError):
        security.open_confined(root, "missing.log")
    with pytest.raises(PathEscape):
        security.open_confined(root, "experiments")  # a directory is not a file to read


@pytest.mark.skipif(os.name != "posix", reason="links and named pipes")
def test_a_log_is_appended_to_without_following_a_link(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "bashrc"
    outside.write_text("export PATH=$PATH\n")

    with security.open_for_append(root, "worker.log") as handle:
        handle.write(b"first\n")
    with security.open_for_append(root, "worker.log") as handle:
        handle.write(b"second\n")
    assert (root / "worker.log").read_bytes() == b"first\nsecond\n"

    (root / "worker.log").unlink()
    (root / "worker.log").symlink_to(outside)
    with pytest.raises(PathEscape, match="link"):
        security.open_for_append(root, "worker.log")
    (root / "worker.log").unlink()
    (root / "worker.log").symlink_to(root / "elsewhere.log")  # dangling, and inside
    with pytest.raises(PathEscape):
        security.open_for_append(root, "worker.log")
    assert not (root / "elsewhere.log").exists()
    (root / "worker.log").unlink()
    os.link(outside, root / "worker.log")  # a second name for a file of somebody's
    with pytest.raises(PathEscape, match="another name"):
        security.open_for_append(root, "worker.log")
    (root / "worker.log").unlink()
    os.mkfifo(root / "worker.log")  # would block the server for ever
    with pytest.raises(PathEscape):
        security.open_for_append(root, "worker.log")
    (root / "worker.log").unlink()
    (root / "worker.log").mkdir()
    with pytest.raises(PathEscape):
        security.open_for_append(root, "worker.log")
    assert outside.read_text() == "export PATH=$PATH\n"
    for name in ("../bashrc", "sub/worker.log", "", ".hidden"):
        with pytest.raises(PathEscape):
            security.open_for_append(root, name)


# ---------------------------------------------------------------------------
# What an earlier server left behind
# ---------------------------------------------------------------------------


def test_uploads_an_earlier_server_left_half_staged_are_removed_at_start(workspace, tmp_path):
    staging = workspace / ".incoming"
    old = staging / "0123456789abcdef"
    (old / "folder").mkdir(parents=True)
    (old / "folder" / "big.bin").write_bytes(b"\0" * 4096)
    fresh = staging / "fedcba9876543210"
    fresh.mkdir()
    (fresh / "arriving.bin").write_bytes(b"\0" * 10)
    days_ago = time.time() - 2 * 24 * 3600
    for path in (old / "folder" / "big.bin", old / "folder", old):
        os.utime(path, (days_ago, days_ago))
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "mine.txt").write_text("not the server's")
    (staging / "a-link").symlink_to(keep)
    os.utime(staging / "a-link", (days_ago, days_ago), follow_symlinks=False)

    with TestClient(create_app(workspace, static_dir=tmp_path / "none"), base_url=BASE_URL):
        wait_for(lambda: not old.exists(), message="the stale upload to be removed")
    assert fresh.is_dir() and (fresh / "arriving.bin").exists()  # another server's, right now
    assert (keep / "mine.txt").read_text() == "not the server's"

    # A workspace whose staging directory is itself a link is not swept through it.
    other = tmp_path / "other-workspace"
    other.mkdir()
    target = tmp_path / "somewhere"
    (target / "0123456789abcdef").mkdir(parents=True)
    os.utime(target / "0123456789abcdef", (days_ago, days_ago))
    (other / ".incoming").symlink_to(target)
    assert app_module.sweep_staging(other) == []
    assert (target / "0123456789abcdef").is_dir()
    assert app_module.sweep_staging(tmp_path / "no-such-workspace") == []


# ---------------------------------------------------------------------------
# How the server ends
# ---------------------------------------------------------------------------

# A worker as far as the server can tell, started as ``paper-orchestra research <kind>
# <project>``: it records a running phase and stops in order on an interrupt.
WORKER_SCRIPT = r"""
import json, os, sys, time
from pathlib import Path

root = Path(sys.argv[3])
state = {"schema": 1, "status": "running", "round": 1, "phase": "research",
         "phases": [{"round": 1, "phase": "research", "status": "running"}],
         "worker": None, "error": None}
(root / "state.json").write_text(json.dumps(state))
started = {"argv": sys.argv, "environ": dict(os.environ)}
(root / "started-with.json").write_text(json.dumps(started))
(root / "ready").write_text(str(os.getpid()))
try:
    time.sleep(300)
except KeyboardInterrupt:
    state.update(status="stopped")
    state["phases"][0]["status"] = "interrupted"
    (root / "state.json").write_text(json.dumps(state))
    (root / "interrupted").write_text("yes")
    sys.exit(130)
"""


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    return True


def end(pid: int) -> None:
    for kill in (os.killpg, os.kill):
        with contextlib.suppress(OSError):
            kill(pid, signal.SIGKILL)


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "paper-orchestra"
    path.parent.mkdir()
    path.write_text(WORKER_SCRIPT, encoding="utf-8")
    return path


@pytest.fixture
def launcher(script):
    return lambda kind, root, options: [sys.executable, str(script), "research", kind, str(root)]


def state_of(project: Project) -> dict:
    return json.loads((project.root / "state.json").read_text())


pytestmark_posix = pytest.mark.skipif(os.name != "posix", reason="process groups and signals")


@pytestmark_posix
def test_the_end_of_the_app_leaves_its_workers_running_and_says_which(
    workspace, project, launcher, tmp_path
):
    said: list[str] = []
    app = create_app(workspace, launcher=launcher, static_dir=tmp_path / "none", report=said.append)
    with TestClient(app, base_url=BASE_URL) as client:
        started = client.post(f"/api/projects/{project.id}/advance")
        assert started.status_code == 202, started.text
        pid = started.json()["worker"]["pid"]
        wait_for(lambda: (project.root / "ready").exists(), message="the worker")
        assert said == []
    try:
        # The app has ended.  The worker belongs to its project and goes on.
        assert process_exists(pid) and not (project.root / "interrupted").exists()
        assert state_of(project)["status"] == "running"
        [message] = said
        assert "1 worker keeps running" in message and "next `paper-orchestra ui`" in message
        assert f"warmup: the agents (research phase), process {pid}" in message
        assert "press Stop" in message and "kill -INT" in message
        assert "--stop-workers-on-exit" in message

        # The next server finds it, shows it and can stop it.
        again = create_app(workspace, launcher=launcher, static_dir=tmp_path / "none", grace=10.0)
        with TestClient(again, base_url=BASE_URL) as client:
            worker = client.get(f"/api/projects/{project.id}").json()["worker"]
            assert worker["state"] == "running" and worker["adopted"] is True
            assert worker["pid"] == pid
            stopping = client.post(f"/api/projects/{project.id}/stop")
            assert stopping.json()["worker"]["state"] == "stopping"
            wait_for(
                lambda: (
                    client.get(f"/api/projects/{project.id}").json()["worker"]["state"] == "idle"
                ),
                message="the worker to stop",
            )
        assert not process_exists(pid) and (project.root / "interrupted").exists()
    finally:
        end(pid)


@pytestmark_posix
def test_an_app_that_was_asked_to_stops_its_workers_when_it_ends(
    workspace, project, launcher, tmp_path
):
    said: list[str] = []
    app = create_app(
        workspace,
        launcher=launcher,
        static_dir=tmp_path / "none",
        grace=10.0,
        stop_workers_on_exit=True,
        report=said.append,
    )
    with TestClient(app, base_url=BASE_URL) as client:
        pid = client.post(f"/api/projects/{project.id}/advance").json()["worker"]["pid"]
        wait_for(lambda: (project.root / "ready").exists(), message="the worker")
    try:
        assert not process_exists(pid) and (project.root / "interrupted").exists()
        state = state_of(project)
        assert state["status"] == "stopped" and state["phases"][0]["status"] == "interrupted"
        [message] = said
        assert "Stopping 1 running worker with the server" in message
    finally:
        end(pid)


def test_an_app_without_workers_ends_without_a_word(workspace, project, tmp_path):
    said: list[str] = []
    for stop in (False, True):
        app = create_app(
            workspace, static_dir=tmp_path / "none", stop_workers_on_exit=stop, report=said.append
        )
        with TestClient(app, base_url=BASE_URL) as client:
            assert client.get("/api/projects").status_code == 200
    assert said == []


def test_a_report_that_cannot_be_written_does_not_keep_the_app_from_ending(
    workspace, project, launcher, tmp_path
):
    # The terminal the server would write to may be the very thing that went away.
    def gone(message: str) -> None:
        raise OSError(5, "Input/output error")

    app = create_app(workspace, launcher=launcher, static_dir=tmp_path / "none", report=gone)
    with TestClient(app, base_url=BASE_URL) as client:
        pid = client.post(f"/api/projects/{project.id}/advance").json()["worker"]["pid"]
    try:
        assert process_exists(pid)
    finally:
        end(pid)


#: The variable that fixes the server's access token, and the one these tests give it.
TOKEN_ENV = "PAPER_ORCHESTRA_UI_TOKEN"
TOKEN = "tok-0123456789abcdefghijklmn"
#: The address a server prints at start, with the token when the link carries it.
LINK = re.compile(r"http://127\.0\.0\.1:(\d+)/(?:\?token=([\w.~-]+))?")

# The server as `paper-orchestra ui` runs it, with the stand-in as its worker.
SERVER_SCRIPT = r"""
import os, signal, sys

from paper_orchestra.web import app, workers

stand_in, workspace, mode = sys.argv[1:4]
# However the tests themselves were started (in the background, under nohup), the
# server begins as one started from a terminal does.
signal.signal(signal.SIGHUP, signal.SIG_DFL)
workers.default_command = lambda kind, root, options: [
    sys.executable, stand_in, "research", kind, str(root)
]
if os.environ.get("TEST_OPENER_SECONDS"):
    app.OPENER_SECONDS = float(os.environ["TEST_OPENER_SECONDS"])
app.serve(
    workspace,
    port=0,
    open_browser=bool(os.environ.get("BROWSER")),
    stop_workers_on_exit=(mode == "stop"),
)
"""

# The test process may not open connections (conftest); a process of its own asks.  The
# requests arrive on its input, never as arguments: some carry the token.
CLIENT_SCRIPT = r"""
import http.client, json, sys, time

spec = json.load(sys.stdin)
answers = []
for request in spec["requests"]:
    deadline = time.time() + 120
    while True:
        connection = http.client.HTTPConnection("127.0.0.1", spec["port"], timeout=120)
        try:
            connection.request(
                request.get("method", "GET"),
                request["path"],
                body=request.get("body"),
                headers=request.get("headers") or {},
            )
            response = connection.getresponse()
            # A stream never ends by itself: its first piece is enough.
            data = response.read1(65536) if request.get("stream") else response.read()
            answers.append(
                {
                    "status": response.status,
                    "headers": response.getheaders(),
                    "body": data.decode("utf-8", "replace"),
                }
            )
            break
        except ConnectionRefusedError:
            if time.time() > deadline:
                raise
            time.sleep(0.1)
        finally:
            connection.close()
json.dump(answers, sys.stdout)
"""


@dataclass
class Answer:
    status: int
    headers: dict[str, str]
    body: str


class Served:
    """A server in a process of its own: what it printed, and a way to ask it.

    ``token`` is what ``PAPER_ORCHESTRA_UI_TOKEN`` holds for it (``None``: the variable is
    not set, and the server makes a token of its own).  Its output goes to a file that
    only the person can read, as a terminal is, unless ``private`` is false.
    """

    def __init__(
        self,
        tmp_path: Path,
        command: list[str],
        *,
        token: str | None = TOKEN,
        env: dict[str, str] | None = None,
        private: bool = True,
        wait: bool = True,
        name: str = "server",
    ) -> None:
        self.output = tmp_path / f"{name}.out"
        self.worker: int | None = None
        scratch = tmp_path / f"{name}-tmp"
        scratch.mkdir()
        environment = {k: v for k, v in os.environ.items() if k not in (TOKEN_ENV, "BROWSER")}
        environment["TMPDIR"] = str(scratch)
        if token is not None:
            environment[TOKEN_ENV] = token
        environment.update(env or {})
        mode = 0o600 if private else 0o644
        descriptor = os.open(self.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        try:
            os.fchmod(descriptor, mode)  # whatever the umask of the test run is
            self.process = subprocess.Popen(
                command,
                stdout=descriptor,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        finally:
            os.close(descriptor)
        if not wait:
            return
        try:
            wait_for(
                lambda: LINK.search(self.said()) or self.process.poll() is not None,
                120,
                "the server to start",
            )
            assert LINK.search(self.said()), self.said()
        except BaseException:
            self.close()
            raise

    def said(self) -> str:
        return self.output.read_text(encoding="utf-8", errors="replace")

    @property
    def port(self) -> int:
        return int(LINK.search(self.said()).group(1))

    @property
    def token(self) -> str | None:
        """The token of the link the server printed, or ``None`` when it printed none."""
        found = [match.group(2) for match in LINK.finditer(self.said()) if match.group(2)]
        return found[0] if found else None

    def ask(self, *requests: dict) -> list[Answer]:
        done = subprocess.run(
            [sys.executable, "-c", CLIENT_SCRIPT],
            input=json.dumps({"port": self.port, "requests": requests}),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert done.returncode == 0, done.stderr
        return [
            Answer(answer["status"], {k.lower(): v for k, v in answer["headers"]}, answer["body"])
            for answer in json.loads(done.stdout)
        ]

    def ends_on(self, signum: int) -> int:
        os.kill(self.process.pid, signum)
        return self.process.wait(timeout=120)

    def close(self) -> None:
        end(self.process.pid)
        self.process.wait()
        if self.worker is not None:
            end(self.worker)


def script_server(tmp_path: Path, workspace: Path, script: Path, mode: str, **options) -> Served:
    """``serve()`` as the command line calls it, with the stand-in as its worker."""
    code = tmp_path / "serve.py"
    code.write_text(SERVER_SCRIPT, encoding="utf-8")
    command = [sys.executable, str(code), str(script), str(workspace), mode]
    return Served(tmp_path, command, **options)


def cli_server(tmp_path: Path, workspace: Path, **options) -> Served:
    """``paper-orchestra ui`` itself, on a free port and without a browser."""
    command = [sys.executable, "-m", "paper_orchestra", "ui", "--workspace", str(workspace)]
    return Served(tmp_path, [*command, "--port", "0", "--no-browser"], **options)


def bearer(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def running_server(
    tmp_path: Path, workspace: Path, project: Project, script: Path, mode: str, **options
) -> Served:
    """A server with one project whose worker runs."""
    server = script_server(tmp_path, workspace, script, mode, **options)
    try:
        [answer] = server.ask(
            {
                "method": "POST",
                "path": f"/api/projects/{project.id}/advance",
                "headers": {**bearer(server.token or TOKEN), "Content-Type": "application/json"},
                "body": "{}",
            }
        )
        assert answer.status == 202, answer.body
        server.worker = json.loads(answer.body)["worker"]["pid"]
        wait_for(lambda: (project.root / "ready").exists(), 120, "the worker")
    except BaseException:
        server.close()
        raise
    return server


@pytestmark_posix
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_however_the_server_ends_its_workers_go_on(tmp_path, workspace, project, script, signum):
    # Ctrl-C and a termination signal used to stop every worker (and put a run in flight
    # back in the queue); a closed terminal or a lost SSH session killed the server at
    # once and left them running.  Now there is one rule, and the server says so.
    server = running_server(tmp_path, workspace, project, script, "leave")
    try:
        assert "go on without it" in server.said()
        code = server.ends_on(signum)
        assert code in (0, -signal.SIGTERM), server.said()
        if signum != signal.SIGTERM:
            assert code == 0  # ended in order, not killed where it stood
        assert process_exists(server.worker) and not (project.root / "interrupted").exists()
        assert state_of(project)["status"] == "running"
        said = server.said()
        assert "1 worker keeps running" in said, said
        assert f"warmup: the agents (research phase), process {server.worker}" in said
        assert "kill -INT" in said
    finally:
        server.close()


@pytestmark_posix
@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM])
def test_a_server_asked_to_stop_its_workers_does_so_however_it_ends(
    tmp_path, workspace, project, script, signum
):
    server = running_server(tmp_path, workspace, project, script, "stop")
    try:
        assert "and with it the agents and experiments" in server.said()
        server.ends_on(signum)
        assert not process_exists(server.worker)
        assert (project.root / "interrupted").exists()
        assert state_of(project)["status"] == "stopped"
        assert "Stopping 1 running worker with the server" in server.said()
    finally:
        server.close()


@pytestmark_posix
def test_a_server_started_to_outlive_its_terminal_keeps_ignoring_the_hang_up():
    called: list[int] = []
    before = signal.getsignal(signal.SIGHUP)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as under nohup
        with app_module._hang_up_handled(lambda signum, frame: called.append(signum)):
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        with app_module._hang_up_handled(lambda signum, frame: called.append(signum)):
            signal.raise_signal(signal.SIGHUP)
        assert called == [signal.SIGHUP]
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL  # put back as it was
    finally:
        signal.signal(signal.SIGHUP, before)


# ---------------------------------------------------------------------------
# The access token: the person proves it is them, on loopback too
# ---------------------------------------------------------------------------

#: What a browser sends when a person opens an address.
HTML = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
SECURITY_HEADERS = {
    "content-security-policy": CONTENT_SECURITY_POLICY,
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cross-origin-resource-policy": "same-origin",
    "cache-control": "no-store",
}


@pytest.fixture
def guarded(workspace, project, tmp_path):
    """The app as ``serve()`` builds it: with an access token, and a frontend to keep."""
    static = tmp_path / "frontend"
    (static / "js").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>PaperOrchestra</title>")
    (static / "js" / "main.js").write_text("export {};\n")
    return create_app(workspace, token=TOKEN, static_dir=static)


def cookie_of(answer) -> str:
    """``name=value`` of the cookie an answer sets."""
    return answer.headers["set-cookie"].split(";", 1)[0]


def test_a_page_asked_for_without_the_token_says_where_the_link_is(guarded):
    page = Exchange(guarded, "GET", "/", headers=HTML)
    assert page.status == 401
    assert page.headers["content-type"] == "text/html; charset=utf-8"
    for name, value in SECURITY_HEADERS.items():
        assert page.headers[name] == value, name
    text = page.body.decode()
    assert text.lower().startswith("<!doctype html>")
    # Self-contained: the policy allows no inline style and no script, and every file of
    # the app needs the token as much as the page does.
    assert not re.search(r"<(script|style|link|img|iframe|form)\b", text, re.IGNORECASE)
    assert not re.search(r"\s(style|src|href|on\w+)\s*=", text, re.IGNORECASE)
    assert "paper-orchestra ui" in text and "terminal" in text
    assert "once" in text and "restart" in text.lower()
    assert TOKEN not in text
    # Every address a person may open is answered so, a route of the frontend included.
    for path in ("/p/p-1/wiki", "/index.html", "/anything/else"):
        other = Exchange(guarded, "GET", path, headers=HTML)
        assert (other.status, other.body) == (401, page.body), path
    # A script is not answered with a page, and the API answers as the API does.
    script = Exchange(guarded, "GET", "/js/main.js", headers={"accept": "*/*"})
    assert script.status == 401 and script.headers["content-type"].startswith("text/plain")
    assert "paper-orchestra ui" in script.body.decode()
    refused = Exchange(guarded, "GET", "/api/health", headers=HTML)
    assert refused.status == 401 and refused.headers["content-type"] == "application/json"
    assert refused.error()["code"] == "token_required" and set(refused.error()) == {
        "code",
        "message",
    }
    assert "paper-orchestra ui" in refused.error()["message"]
    assert "Authorization: Bearer" in refused.error()["message"]  # what a program does
    for answer in (script, refused):
        for name, value in SECURITY_HEADERS.items():
            assert answer.headers[name] == value, name


@pytest.mark.parametrize("path", ["/", "/js/main.js", "/api/health", "/api/projects"])
def test_a_wrong_token_is_answered_like_a_missing_one(guarded, path):
    missing = Exchange(guarded, "GET", path, headers=HTML)
    assert missing.status == 401
    nearly = TOKEN[:-1] + "x"
    attempts = [
        ({"cookie": f"port8600_po_token={nearly}"}, ""),
        ({"cookie": "port8600_po_token="}, ""),
        ({"cookie": f"po_token={nearly}"}, ""),
        ({"authorization": f"Bearer {nearly}"}, ""),
        ({"authorization": "Bearer"}, ""),
        ({"authorization": f"Basic {TOKEN}"}, ""),
        ({"authorization": f"Bearer {TOKEN} and more"}, ""),
        ({}, f"token={nearly}"),
        ({}, "token="),
        ({}, f"token={TOKEN}x"),
        ({}, f"token={TOKEN[:5]}"),
        ({}, f"Token={TOKEN}"),
    ]
    for headers, query in attempts:
        wrong = Exchange(guarded, "GET", path, headers={**HTML, **headers}, query=query)
        # Nothing in the answer tells a wrong token from none, or how wrong it was.
        assert (wrong.status, wrong.headers, wrong.body) == (
            missing.status,
            missing.headers,
            missing.body,
        ), (headers, query)
        assert TOKEN not in wrong.body.decode()


def test_the_token_opens_the_app_from_the_link_the_cookie_and_the_bearer_header(guarded, project):
    entered = Exchange(guarded, "GET", "/", query=f"token={TOKEN}&view=library", headers=HTML)
    assert entered.status == 303 and entered.headers["location"] == "/?view=library"
    assert entered.body == b""
    for name, value in SECURITY_HEADERS.items():
        assert entered.headers[name] == value, name
    name, _, value = cookie_of(entered).partition("=")
    assert (name, value) == ("port8600_po_token", TOKEN)
    attributes = {part.strip().lower() for part in entered.headers["set-cookie"].split(";")[1:]}
    assert attributes == {"path=/", "httponly", "samesite=strict"}

    cookie = {"cookie": f"{name}={value}"}
    base = f"/api/projects/{project.id}"
    for path in ("/", "/js/main.js", "/api/health", base, f"{base}/files/inputs/brief.md"):
        assert Exchange(guarded, "GET", path, headers=cookie).status == 200, path
    guarded.state.context.live.closing.set()  # the stream says hello and ends
    stream = Exchange(guarded, "GET", f"{base}/events", headers=cookie)
    assert stream.status == 200 and stream.headers["content-type"].startswith("text/event-stream")
    posted = Exchange(
        guarded,
        "POST",
        f"{base}/requests",
        headers={**cookie, "content-type": "application/json"},
        pieces=[b'{"text": "Use three seeds."}'],
    )
    assert posted.status == 201, posted.body

    header = {"authorization": f"Bearer {TOKEN}"}
    assert Exchange(guarded, "GET", "/api/health", headers=header).status == 200
    spaced = {"authorization": f"bearer   {TOKEN} "}
    assert Exchange(guarded, "GET", "/api/health", headers=spaced).status == 200
    # A cookie the server no longer takes does not stand in the way of the header.
    both = {**header, "cookie": f"{name}=of-an-earlier-server"}
    assert Exchange(guarded, "GET", "/api/health", headers=both).status == 200


def test_the_token_in_an_address_is_taken_only_where_a_person_opens_a_link(guarded, project):
    link = f"token={TOKEN}"
    # Never in the query of an API call: such an address ends up in histories and logs.
    for path in ("/api/health", f"/api/projects/{project.id}", "/api/openapi.json", "/api/"):
        refused = Exchange(guarded, "GET", path, query=link)
        assert refused.status == 401 and "set-cookie" not in refused.headers, path
    # A link is opened, not posted to.
    for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"):
        refused = Exchange(guarded, method, "/", query=link)
        assert refused.status == 401 and "set-cookie" not in refused.headers, method
    assert project.requests() == []


@pytest.mark.parametrize(
    ("path", "location"),
    [
        ("//evil.example/", "/evil.example/"),
        ("/\\evil.example/", "/evil.example/"),
        ("/\\/\\evil.example", "/evil.example"),
        ("/a\r\nSet-Cookie: x=y", "/a%0D%0ASet-Cookie%3A%20x%3Dy"),
        ("/a\tb\x00c", "/a%09b%00c"),
        ("/p/p-1/wiki", "/p/p-1/wiki"),
    ],
)
def test_the_link_leads_only_back_into_the_app_on_loopback_too(guarded, path, location):
    # The address of the request comes back as the Location of the redirect: it stays a
    # path of this server and one header line, whatever was asked for.
    query = f"token={TOKEN}&next=//evil.example&x=a%0D%0Ab"
    entered = Exchange(guarded, "GET", path, query=query, headers=HTML)
    assert entered.status == 303 and entered.starts == 1
    assert entered.headers["location"] == location + "?next=%2F%2Fevil.example&x=a%0D%0Ab"
    assert cookie_of(entered) == f"port8600_po_token={TOKEN}"
    assert TOKEN not in entered.headers["location"]


def test_a_link_another_site_led_to_is_answered_with_a_page_that_goes_on_by_itself(guarded):
    # A browser sends no SameSite=Strict cookie along with a navigation that a page of
    # another site started, the redirect that follows included: the person would arrive
    # without the cookie they were just given.  The file a browser is opened with is such
    # a page, and so are a terminal that runs in a browser and a link in a mail.
    query = f"token={TOKEN}&view=library&x=%22%3E%3Cscript%3E"
    browser = {**HTML, "sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate"}
    entered = Exchange(guarded, "GET", "/p/p-1/wiki", query=query, headers=browser)
    assert entered.status == 200
    assert entered.headers["content-type"] == "text/html; charset=utf-8"
    assert "location" not in entered.headers
    assert cookie_of(entered) == f"port8600_po_token={TOKEN}"
    attributes = {part.strip().lower() for part in entered.headers["set-cookie"].split(";")[1:]}
    assert attributes == {"path=/", "httponly", "samesite=strict"}
    for name, value in SECURITY_HEADERS.items():
        assert entered.headers[name] == value, name
    page = entered.body.decode()
    onward = "/p/p-1/wiki?view=library&amp;x=%22%3E%3Cscript%3E"
    assert f'<meta http-equiv="refresh" content="0; url={onward}">' in page
    assert f'<a href="{onward}">' in page
    assert TOKEN not in page and "<script" not in page
    # The address the page goes on to is a path of this server, whatever was asked for,
    # and nothing of it is read as markup.
    hostile = '/a"><meta http-equiv="refresh" content="0; url=//evil.example'
    for path, onward in (
        ("//evil.example/", "/evil.example/"),
        ("/\\evil.example/", "/evil.example/"),
        ("/a\r\nb", "/a%0D%0Ab"),
        (hostile, "/a%22%3E%3Cmeta%20http-equiv%3D%22refresh%22%20content%3D%220%3B%20url%3D//"),
    ):
        entered = Exchange(guarded, "GET", path, query=f"token={TOKEN}", headers=browser)
        page = entered.body.decode()
        assert entered.status == 200, path
        assert page.count("<meta http-equiv") == 1 and page.count("<a ") == 1, path
        assert f'<meta http-equiv="refresh" content="0; url={onward}' in page, path
        assert "url=//" not in page and 'href="//' not in page, path
    # Where the browser itself, or a page of the app, opened the link, it is the redirect.
    for site in ("none", "same-origin", "same-site"):
        direct = Exchange(
            guarded, "GET", "/", query=f"token={TOKEN}", headers={**HTML, "sec-fetch-site": site}
        )
        assert direct.status == 303 and direct.headers["location"] == "/", site
    # A wrong token is answered as everywhere.
    wrong = Exchange(guarded, "GET", "/", query="token=wrong", headers=browser)
    assert wrong.status == 401 and "set-cookie" not in wrong.headers


def test_the_cookie_is_kept_apart_from_that_of_a_server_on_another_port(workspace, tmp_path):
    # A browser keeps cookies by host name, not by port: two servers on one machine (two
    # workspaces, two forwarded ports) would otherwise overwrite each other's token, and
    # opening the one would close the other.
    other_token = "another-token-0123456789"
    first = create_app(workspace, token=TOKEN, static_dir=tmp_path / "none")
    second = create_app(workspace, port=8601, token=other_token, static_dir=tmp_path / "none")
    there = [(first, "127.0.0.1:8600", TOKEN), (second, "127.0.0.1:8601", other_token)]
    cookies = []
    for app, host, token in there:
        entered = Exchange(app, "GET", "/", query=f"token={token}", headers={"host": host})
        assert entered.status == 303
        cookies.append(cookie_of(entered))
    assert cookies == [f"port8600_po_token={TOKEN}", f"port8601_po_token={other_token}"]
    for order in (cookies, cookies[::-1]):
        jar = {"cookie": "; ".join(order)}
        for app, host, _ in there:
            answer = Exchange(app, "GET", "/api/health", headers={**jar, "host": host})
            assert answer.status == 200, (host, order)
    # The cookie of the other port is not this server's, whatever it holds.
    only_other = {"cookie": f"port8601_po_token={TOKEN}", "host": "127.0.0.1:8600"}
    assert Exchange(first, "GET", "/api/health", headers=only_other).status == 401
    # Reached without a port in the address (behind a proxy), the name says so.
    plain = Exchange(first, "GET", "/", query=f"token={TOKEN}", headers={"host": "localhost"})
    assert cookie_of(plain) == f"port80_po_token={TOKEN}"


def test_cookies_of_other_apps_on_the_machine_do_not_hide_the_token(guarded):
    # Everything on 127.0.0.1 shares one cookie jar, whatever port it listens on, and
    # what another app put there may not be well formed.
    crowded = (
        'weird[key]=1; other="half quoted; =nameless; lone; '
        f"port8600_po_token=stale; port8600_po_token={TOKEN}; trailing="
    )
    assert Exchange(guarded, "GET", "/api/health", headers={"cookie": crowded}).status == 200
    without = crowded.replace(TOKEN, "also-stale")
    assert Exchange(guarded, "GET", "/api/health", headers={"cookie": without}).status == 401
    # The pairs may arrive on several Cookie lines; the one that matters is found on any.
    guard = SecurityMiddleware(None, token=TOKEN)
    for order in (("a=b", f"port8600_po_token={TOKEN}"), (f"port8600_po_token={TOKEN}", "a=b")):
        lines = [(b"host", b"127.0.0.1:8600"), *[(b"cookie", line.encode()) for line in order]]
        assert guard._authorised({"type": "http", "headers": lines}, {"host": "127.0.0.1:8600"})


def test_a_websocket_is_refused_in_so_many_words(guarded):
    # Nothing in the app speaks WebSocket.  Left unanswered, such a connection made the
    # server answer 500 and write an error into the person's terminal, for anybody who
    # asked and as often as they asked.
    sent: list[dict] = []
    waiting = [{"type": "websocket.connect"}]

    async def receive():
        return waiting.pop(0) if waiting else {"type": "websocket.disconnect", "code": 1006}

    async def send(message):
        sent.append(message)

    headers = [(b"host", b"127.0.0.1:8600"), (b"cookie", f"port8600_po_token={TOKEN}".encode())]
    scope = {"type": "websocket", "path": "/api/projects/p/events", "headers": headers}
    asyncio.run(guarded(scope, receive, send))
    assert [message["type"] for message in sent] == ["websocket.close"]


@pytest.mark.parametrize(
    ("method", "headers", "code"),
    [
        # A "digit" that is no number to Python, and more digits than it converts.
        ("GET", {"host": "127.0.0.1:\xb2"}, "bad_host"),
        ("GET", {"host": "127.0.0.1:" + "1" * 5000}, "bad_host"),
        ("GET", {"host": "[::1]:\xb9"}, "bad_host"),
        ("GET", {"host": "[::1]:" + "8" * 5000}, "bad_host"),
        # An origin that is no address at all.
        ("POST", {"origin": "http://["}, "cross_origin"),
        ("POST", {"origin": "http://[::1"}, "cross_origin"),
        ("POST", {"origin": "http://127.0.0.1:" + "8" * 5000}, "cross_origin"),
        ("POST", {"origin": "http://127.0.0.1:\xb2"}, "cross_origin"),
    ],
    ids=[
        "host-odd-digit",
        "host-long-port",
        "host-v6-odd-digit",
        "host-v6-long-port",
        "origin-open-bracket",
        "origin-half-address",
        "origin-long-port",
        "origin-odd-digit",
    ],
)
def test_a_host_or_an_origin_that_cannot_be_read_is_refused_not_an_error(
    guarded, method, headers, code
):
    # These are read before the token is looked at, so anybody can send them.  The server
    # used to fail on them: 500, and a traceback in the person's terminal, as often as
    # they were sent.
    answer = Exchange(guarded, method, "/api/health", headers=headers)
    assert answer.raised is None, repr(answer.raised)
    assert answer.status == 403 and answer.error()["code"] == code
    for name, value in SECURITY_HEADERS.items():
        assert answer.headers[name] == value, name
    # With the token it is no different: the address is wrong, whoever asks.
    held = {**headers, "authorization": f"Bearer {TOKEN}"}
    assert Exchange(guarded, method, "/api/health", headers=held).status == 403


def test_what_a_host_header_names_is_read_as_before():
    split = security._split_host
    assert split("127.0.0.1:8600") == ("127.0.0.1", 8600)
    assert split("LocalHost.:80") == ("localhost", 80)
    assert split("[::1]:8600") == ("::1", 8600)
    assert split("[::1]") == ("::1", None)
    assert split("example.org") == ("example.org", None)
    assert split("::1") == ("::1", None)  # no brackets: a bare address, no port
    assert split("") == ("", None)
    # A port that cannot be read names no address: nothing accepts it.
    for unreadable in ("127.0.0.1:\xb2", "127.0.0.1:" + "9" * 6, "[::1]:\xb3", "[::1]:" + "9" * 40):
        assert split(unreadable) == ("", None), unreadable


# The framework traces every request by itself when OpenTelemetry is set up for the
# process, from outside the security layer.  A tracer that keeps what it is given stands
# for such a set-up; the app is asked without a network.
TRACED_SCRIPT = r"""
import asyncio, contextlib, json, sys
from pathlib import Path

from opentelemetry import trace

spans = []


class Tracer(trace.Tracer):
    def start_span(self, name, context=None, kind=trace.SpanKind.INTERNAL, attributes=None,
                   links=None, start_time=None, record_exception=True,
                   set_status_on_exception=True):
        spans.append({"name": name, **dict(attributes or {})})
        return trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)

    @contextlib.contextmanager
    def start_as_current_span(self, name, *args, **kwargs):
        yield self.start_span(name, attributes=kwargs.get("attributes"))


class Provider(trace.TracerProvider):
    def get_tracer(self, *args, **kwargs):
        return Tracer()


trace.set_tracer_provider(Provider())

from paper_orchestra.web.app import create_app

spec = json.load(sys.stdin)
app = create_app(Path(spec["workspace"]), token=spec["token"], static_dir=Path(spec["static"]))
statuses = []


async def ask(path, query, headers):
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": query.encode(), "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8600")]
        + [(key.encode(), value.encode()) for key, value in headers.items()],
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8600),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    await app(scope, receive, send)


async def main():
    for path, query, headers in spec["requests"]:
        await ask(path, query, headers)


asyncio.run(main())
json.dump({"statuses": statuses, "spans": spans}, sys.stdout)
"""


def test_the_frameworks_own_tracing_is_not_given_the_token(workspace, tmp_path):
    # The link carries the token in its address, and the framework's tracing records the
    # address of every request it sees and sends it where the person's OpenTelemetry
    # set-up says (by default a port on this machine that anybody may hold).
    pytest.importorskip("fastapi.telemetry", reason="this FastAPI does not trace by itself")
    static = tmp_path / "frontend"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>PaperOrchestra</title>")
    nearly = TOKEN[:-1] + "x"
    requests = [
        ["/", f"token={TOKEN}&view=library", {}],  # the link
        ["/p/p-1/wiki", f"view=x&token={TOKEN}", {"sec-fetch-site": "cross-site"}],
        ["/api/health", f"token={TOKEN}", {}],  # where it counts for nothing
        ["/", f"token={nearly}", {}],  # a guess that is one character off
        ["/api/health", "x=1", {"authorization": f"Bearer {TOKEN}"}],  # traced as ever
    ]
    done = subprocess.run(
        [sys.executable, "-c", TRACED_SCRIPT],
        input=json.dumps(
            {
                "workspace": str(workspace),
                "static": str(static),
                "token": TOKEN,
                "requests": requests,
            }
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["statuses"] == [303, 200, 401, 401, 200]
    recorded = json.dumps(seen["spans"])
    assert TOKEN not in recorded and nearly not in recorded, seen["spans"]
    # A request whose address offers no token is traced as the person set it up.
    assert any(span.get("url.query") == "x=1" for span in seen["spans"]), seen["spans"]


def test_only_a_framework_that_traces_is_told_what_not_to_trace(monkeypatch):
    # The package runs with FastAPI versions from before it traced anything: those do not
    # know the argument, and one that does not take unknown arguments would not start.
    class Older:
        def __init__(self, *, title: str = "", lifespan: object = None) -> None:
            pass

    class Newer:
        def __init__(self, *, telemetry: object = None, title: str = "") -> None:
            pass

    monkeypatch.setattr(app_module, "FastAPI", Older)
    assert app_module._untraced_link() == {}
    monkeypatch.setattr(app_module, "FastAPI", Newer)
    assert app_module._untraced_link() == {"telemetry": {"exclude": security.offers_token}}


@pytest.mark.parametrize(
    ("query", "offers"),
    [
        (b"token=abc", True),
        (b"view=library&token=", True),
        (b"%74oken=abc", True),  # read as the server reads it: the name percent-decoded
        (b"token", True),
        (b"", False),
        (b"view=token", False),
        (b"Token=abc&tokens=1&atoken=2", False),
        (b"\xff\xfe=1&x=%zz", False),  # nothing a query holds makes the question fail
    ],
)
def test_an_address_that_names_a_token_is_told_apart(query, offers):
    assert security.offers_token({"type": "http", "query_string": query}) is offers
    assert security.offers_token({"type": "websocket"}) is False


def test_the_token_can_be_fixed_in_the_environment():
    environ = {TOKEN_ENV: f"  {TOKEN}\n", "OTHER": "kept"}
    assert security.fixed_token(environ) == TOKEN
    # Taken out, so that nothing the server starts inherits it: workers, jobs, a browser.
    assert environ == {"OTHER": "kept"}
    assert security.fixed_token({}) is None
    for empty in ("", "   "):
        environ = {TOKEN_ENV: empty}
        assert security.fixed_token(environ) is None and TOKEN_ENV not in environ
    with pytest.raises(security.TokenError, match="16 characters") as short:
        security.fixed_token({TOKEN_ENV: "short-token"})
    assert "short-token" not in str(short.value) and TOKEN_ENV in str(short.value)
    for unusable in (
        "has a space in it 0123456789",
        "semi;colon-0123456789abcdef",
        'quote"0123456789abcdefgh',
        "ünïcode-0123456789abcdef",
        "a/b?c=d&e-0123456789abcdef",
    ):
        environ = {TOKEN_ENV: unusable}
        with pytest.raises(security.TokenError) as refused:
            security.fixed_token(environ)
        assert unusable not in str(refused.value) and TOKEN_ENV in str(refused.value)
        assert TOKEN_ENV not in environ  # not left for a child either
    # A job never gets it, even from a worker a person started by hand with the variable
    # set: what a job inherits is a short list of harmless variables.
    from paper_orchestra.compute.local import scrubbed_env

    os.environ[TOKEN_ENV] = TOKEN
    try:
        assert TOKEN not in json.dumps(scrubbed_env(Path("/nowhere")))
    finally:
        del os.environ[TOKEN_ENV]
    generated = {security.new_token() for _ in range(50)}
    assert len(generated) == 50 and all(len(token) >= 32 for token in generated)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", token) for token in generated)


class Opened:
    """Stands in for ``webbrowser.open``: what it was called with, and what the file it
    was pointed at looked like at that moment."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls: list[str] = []
        self.files: list[dict] = []
        self.done = threading.Event()

    def __call__(self, url: str, *args, **kwargs) -> bool:
        self.calls.append(url)
        if url.startswith("file:"):
            path = Path(url2pathname(urlsplit(url).path))
            self.files.append(
                {
                    "path": path,
                    "mode": stat.S_IMODE(path.stat().st_mode),
                    "directory_mode": stat.S_IMODE(path.parent.stat().st_mode),
                    "beside": sorted(entry.name for entry in path.parent.iterdir()),
                    "text": path.read_text(encoding="utf-8"),
                }
            )
        self.done.set()
        return self.answer


@pytest.fixture
def opened(monkeypatch, tmp_path):
    """The app's way to a browser, recorded; its temporary files are the test's."""
    import tempfile

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    recorder = Opened()
    monkeypatch.setattr(webbrowser, "open", recorder)
    return recorder


@pytestmark_posix
def test_the_browser_is_opened_with_a_file_only_the_person_can_read(
    workspace, tmp_path, monkeypatch, opened
):
    monkeypatch.setattr(app_module, "OPENER_SECONDS", 0.4, raising=False)
    said: list[str] = []
    app = create_app(
        workspace,
        token=TOKEN,
        open_browser=True,
        port=8612,
        static_dir=tmp_path / "none",
        report=said.append,
    )
    with TestClient(app, base_url=BASE_URL):
        assert opened.done.wait(20)
        # The token is never an argument of a process: the browser is given a file.
        [url] = opened.calls
        assert url.startswith("file://") and TOKEN not in url
        [file] = opened.files
        assert file["path"].is_relative_to(tmp_path / "tmp")
        assert file["mode"] == 0o600 and file["directory_mode"] == 0o700
        assert file["beside"] == [file["path"].name]
        link = f"http://127.0.0.1:8612/?token={TOKEN}"
        assert f'content="0; url={link}"' in file["text"] and f'href="{link}"' in file["text"]
        assert '<meta name="referrer" content="no-referrer">' in file["text"]
        assert file["text"].count(TOKEN) == 2 and "<script" not in file["text"]
        # ... which is gone a moment later, with its directory, while the app runs on.
        wait_for(lambda: not file["path"].parent.exists(), 20, "the file to be removed")
    assert said == []
    assert list((tmp_path / "tmp").iterdir()) == []


@pytestmark_posix
def test_the_file_for_the_browser_is_removed_when_the_app_ends(workspace, tmp_path, opened):
    app = create_app(workspace, token=TOKEN, open_browser=True, static_dir=tmp_path / "none")
    with TestClient(app, base_url=BASE_URL):
        assert opened.done.wait(20)
        [file] = opened.files
        assert file["path"].is_file()  # for half a minute, unless the app ends first
    assert not file["path"].parent.exists()
    assert list((tmp_path / "tmp").iterdir()) == []


def test_when_the_file_cannot_be_written_no_browser_is_opened_and_the_app_says_so(
    workspace, tmp_path, monkeypatch, opened
):
    import tempfile

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "mkdtemp", full)
    said: list[str] = []
    app = create_app(
        workspace,
        token=TOKEN,
        open_browser=True,
        static_dir=tmp_path / "none",
        report=said.append,
    )
    with TestClient(app, base_url=BASE_URL):
        wait_for(lambda: said, 20, "the app to say that no browser was opened")
    assert opened.calls == []
    [message] = said
    assert "browser was not opened" in message and "No space left on device" in message
    assert "link" in message and TOKEN not in message


def test_a_browser_that_cannot_be_started_leaves_no_file_behind(workspace, tmp_path, opened):
    opened.answer = False
    said: list[str] = []
    app = create_app(
        workspace,
        token=TOKEN,
        open_browser=True,
        static_dir=tmp_path / "none",
        report=said.append,
    )
    with TestClient(app, base_url=BASE_URL):
        wait_for(lambda: said, 20, "the app to say that no browser was opened")
        [file] = opened.files
        assert not file["path"].parent.exists()
    [message] = said
    assert "No browser" in message and "link" in message and TOKEN not in message


def test_without_a_token_the_app_of_a_test_opens_its_plain_address(workspace, tmp_path, opened):
    # ``create_app(..., token=None)`` is for in-process tests only; ``serve()`` always
    # makes a token.
    app = create_app(workspace, open_browser=True, port=8613, static_dir=tmp_path / "none")
    with TestClient(app, base_url=BASE_URL):
        assert opened.done.wait(20)
    assert opened.calls == ["http://127.0.0.1:8613/"]


def held(answer: Answer) -> dict[str, str]:
    """The cookie an answer set, as a browser sends it back."""
    return {"Cookie": cookie_of(answer)}


JSON_BODY = {"Content-Type": "application/json"}
PAGE = {"Accept": HTML["accept"]}


@pytestmark_posix
def test_on_loopback_nothing_is_reachable_without_the_token(tmp_path, workspace, project):
    # The audit's first finding: on loopback the server took every request without an
    # Origin as the person's, so any other user or process of the machine could list the
    # projects, write experiment code, approve it and run it as the person.
    server = cli_server(tmp_path, workspace, token=None)  # a token of its own, made at start
    try:
        token = server.token
        assert token is not None and len(token) >= 32
        wrong = "x" * len(token)
        base = f"/api/projects/{project.id}"
        closed = [
            {"path": "/", "headers": PAGE},
            {"path": "/"},
            {"path": "/index.html", "headers": PAGE},
            {"path": f"/p/{project.id}/wiki", "headers": PAGE},
            {"path": "/js/main.js"},
            {"path": "/css/app.css"},
            {"path": "/img/mark.svg"},
            {"path": "/api/health"},
            {"path": "/api/projects"},
            {"path": base},
            {"path": f"{base}/pages"},
            {"path": f"{base}/events", "headers": {"Accept": "text/event-stream"}},
            {"path": f"{base}/files/inputs/brief.md"},
            {"path": f"{base}/worker/log"},
            # What the framework adds by itself, and addresses that only look different.
            {"path": "/api/openapi.json"},
            {"path": "/openapi.json"},
            {"path": "/docs"},
            {"path": "/redoc"},
            {"path": "/api/health/"},
            {"path": "//api/health"},
            {"path": "/api//health"},
            {"path": "/js//main.js"},
            {"path": "/%61pi/health"},
            {"method": "HEAD", "path": "/"},
            {"method": "HEAD", "path": "/api/health"},
            {"method": "OPTIONS", "path": "/api/health"},
            {
                "method": "POST",
                "path": f"{base}/requests",
                "headers": JSON_BODY,
                "body": '{"text": "From another account."}',
            },
            {"method": "POST", "path": f"{base}/advance", "headers": JSON_BODY, "body": "{}"},
            {
                "method": "PUT",
                "path": f"{base}/experiments/E1/code/main.py",
                "headers": JSON_BODY,
                "body": '{"text": "print(1)"}',
            },
            {"method": "DELETE", "path": f"{base}/pages/idea"},
            # A wrong token wherever one is taken, and the right one where none is.
            {"path": "/api/health", "headers": bearer(wrong)},
            {"path": "/api/health", "headers": {"Cookie": f"port{server.port}_po_token={wrong}"}},
            {"path": f"/?token={wrong}", "headers": PAGE},
            {"path": f"/api/health?token={token}"},
            {"path": f"{base}?token={token}"},
        ]
        for request, answer in zip(closed, server.ask(*closed), strict=True):
            assert answer.status == 401, request
            assert token not in answer.body and "set-cookie" not in answer.headers, request
            for name, value in SECURITY_HEADERS.items():
                assert answer.headers[name] == value, (request, name)
        assert project.requests() == [] and not (project.root / "state.json").exists()

        # A person opens the link once; the browser keeps a cookie that scripts cannot read
        # and other sites cannot use, and is sent on without the token in the address.
        [entered] = server.ask({"path": f"/?token={token}", "headers": PAGE})
        assert entered.status == 303 and entered.headers["location"] == "/"
        assert "HttpOnly" in entered.headers["set-cookie"]
        assert "SameSite=Strict" in entered.headers["set-cookie"]
        cookie = held(entered)
        opened = [
            {"path": "/", "headers": {**PAGE, **cookie}},
            {"path": "/js/main.js", "headers": cookie},
            {"path": "/css/app.css", "headers": cookie},
            {"path": "/api/health", "headers": cookie},
            {"path": base, "headers": cookie},
            {"path": f"{base}/files/inputs/brief.md", "headers": cookie},
            {"path": f"{base}/events", "headers": cookie, "stream": True},
            {
                "method": "POST",
                "path": f"{base}/requests",
                "headers": {**cookie, **JSON_BODY},
                "body": '{"text": "Use three seeds."}',
            },
        ]
        answers = server.ask(*opened)
        assert [answer.status for answer in answers] == [200] * 7 + [201]
        assert "<title>PaperOrchestra</title>" in answers[0].body
        assert json.loads(answers[4].body)["id"] == project.id
        assert answers[6].headers["content-type"].startswith("text/event-stream")
        assert [request.text for request in project.requests()] == ["Use three seeds."]

        # A client that is no browser sends the token as a header.
        scripted = server.ask(
            {"path": "/api/health", "headers": bearer(token)},
            {"path": base, "headers": bearer(token)},
        )
        assert [answer.status for answer in scripted] == [200, 200]

        # The link cannot be made to lead anywhere but into the app.
        elsewhere = server.ask(
            {"path": f"//evil.example/?token={token}", "headers": PAGE},
            {"path": f"/a%0D%0ASet-Cookie:%20x=y?token={token}", "headers": PAGE},
        )
        assert [answer.status for answer in elsewhere] == [303, 303]
        assert elsewhere[0].headers["location"] == "/evil.example/"
        assert elsewhere[1].headers["location"] == "/a%0D%0ASet-Cookie%3A%20x%3Dy"
        assert cookie_of(elsewhere[1]) == cookie["Cookie"]

        # The link is printed once, and the token is written nowhere else: not on a command
        # line, not into the project.
        assert server.said().count(token) == 1
        assert f"http://127.0.0.1:{server.port}/?token={token}" in server.said()
        command_line = Path(f"/proc/{server.process.pid}/cmdline")
        if command_line.exists():
            assert token.encode() not in command_line.read_bytes()
        for path in workspace.rglob("*"):
            if path.is_file():
                assert token.encode() not in path.read_bytes(), path
    finally:
        server.close()


@pytestmark_posix
def test_a_token_fixed_in_the_environment_survives_a_restart_and_a_fresh_one_does_not(
    tmp_path, workspace, project
):
    health = {"path": "/api/health"}
    for name in ("first", "again"):
        server = cli_server(tmp_path, workspace, token=TOKEN, name=name)
        try:
            assert server.token == TOKEN and "PAPER_ORCHESTRA_UI_TOKEN" in server.said()
            kept = {"Cookie": f"port{server.port}_po_token={TOKEN}"}
            answers = server.ask({**health, "headers": bearer()}, {**health, "headers": kept})
            assert [answer.status for answer in answers] == [200, 200], name
        finally:
            server.close()

    # Without the variable (an empty value is none) every start makes a new token, and what
    # let a browser into the server before is refused by the next.
    seen: list[str] = []
    for name, value in (("fresh", None), ("restarted", "")):
        server = cli_server(tmp_path, workspace, token=value, name=name)
        try:
            token = server.token
            assert token and token not in (TOKEN, *seen) and len(token) >= 32
            requests = [{**health, "headers": bearer(token)}]
            for old in (TOKEN, *seen):
                stale = {"Cookie": f"port{server.port}_po_token={old}"}
                requests += [{**health, "headers": bearer(old)}, {**health, "headers": stale}]
            statuses = [answer.status for answer in server.ask(*requests)]
            assert statuses == [200] + [401] * (len(requests) - 1), name
            seen.append(token)
        finally:
            server.close()


@pytestmark_posix
@pytest.mark.parametrize(
    "value", ["too-short", "long enough but with spaces"], ids=["short", "spaces"]
)
def test_a_token_that_cannot_be_one_is_refused_at_start(tmp_path, workspace, value):
    server = cli_server(tmp_path, workspace, token=value, wait=False)
    try:
        assert server.process.wait(timeout=120) == 2
        said = server.said()
        assert "PAPER_ORCHESTRA_UI_TOKEN" in said and "16 characters" in said
        assert value not in said and "http://" not in said and "Traceback" not in said
    finally:
        server.close()


# `serve()` asked for a port that somebody else holds: the holder is this process itself,
# which stands for another server (another account's, on a shared machine).
HELD_PORT_SCRIPT = r"""
import os, socket, sys

from paper_orchestra.web import app

held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
held.bind(("127.0.0.1", 0))
held.listen(1)
print("held", held.getsockname()[1], flush=True)
app.serve(sys.argv[1], port=held.getsockname()[1], open_browser=bool(os.environ.get("BROWSER")))
"""


@pytestmark_posix
@pytest.mark.parametrize("fixed", [None, TOKEN], ids=["made-at-start", "from-the-environment"])
def test_the_link_is_said_only_once_the_server_holds_its_address(
    tmp_path, workspace, browser, fixed
):
    # The server used to say "PaperOrchestra is running" and print the link before it had
    # tried to listen.  Where another program holds the port, a person who then opened
    # the link, as the message told them to, handed the token to that program; a token
    # fixed in the environment is good for the next start as well.
    code = tmp_path / "held.py"
    code.write_text(HELD_PORT_SCRIPT, encoding="utf-8")
    command = [sys.executable, str(code), str(workspace)]
    server = Served(tmp_path, command, token=fixed, env=browser, wait=False)
    try:
        assert server.process.wait(timeout=120) not in (0, None)
        said = server.said()
        port = re.search(r"held (\d+)", said).group(1)
        assert "token=" not in said and "is running" not in said, said
        assert fixed is None or fixed not in said
        assert "Traceback" not in said
        # What happened, where, and what to do about it.
        assert "did not start" in said and f"127.0.0.1:{port}" in said and "--port" in said
        # No browser was sent to the other program either, and nothing is left behind.
        assert not Path(browser["TEST_BROWSER_RECORD"]).exists()
        assert list((tmp_path / "server-tmp").iterdir()) == []
    finally:
        server.close()


# Stands in for a browser: it records how it was started and what it was pointed at.
BROWSER_SCRIPT = r"""#!{python}
import json, os, stat, sys
from urllib.parse import urlsplit
from urllib.request import url2pathname

record = {{"argv": sys.argv, "environ": dict(os.environ), "pid": os.getpid()}}
with open("/proc/self/cmdline", "rb") as own:
    record["cmdline"] = own.read().decode("utf-8", "replace")
url = sys.argv[1] if len(sys.argv) > 1 else ""
if url.startswith("file:"):
    path = url2pathname(urlsplit(url).path)
    folder = os.path.dirname(path)
    with open(path, encoding="utf-8") as page:
        text = page.read()
    record["file"] = {{
        "path": path,
        "mode": stat.S_IMODE(os.stat(path).st_mode),
        "directory_mode": stat.S_IMODE(os.stat(folder).st_mode),
        "beside": sorted(os.listdir(folder)),
        "text": text,
    }}
out = os.environ["TEST_BROWSER_RECORD"]
with open(out + ".part", "w", encoding="utf-8") as handle:
    json.dump(record, handle)
os.replace(out + ".part", out)
"""


@pytest.fixture
def browser(tmp_path: Path) -> dict[str, str]:
    """The environment that makes a server open the stand-in instead of a browser."""
    program = tmp_path / "bin" / "stand-in-browser"
    program.parent.mkdir(exist_ok=True)
    program.write_text(BROWSER_SCRIPT.format(python=sys.executable), encoding="utf-8")
    program.chmod(0o755)
    return {"BROWSER": str(program), "TEST_BROWSER_RECORD": str(tmp_path / "browser.json")}


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="reads /proc")
# (The names of the cases must not hold the token: pytest puts them into the environment.)
@pytest.mark.parametrize("fixed", [None, TOKEN], ids=["made-at-start", "from-the-environment"])
def test_no_process_is_ever_started_with_the_token(
    tmp_path, workspace, project, script, browser, fixed
):
    # Bound beyond loopback, the server used to hand the link with the token to the
    # browser's opener as an argument, where every account of the machine could read it.
    record = Path(browser["TEST_BROWSER_RECORD"])
    server = running_server(
        tmp_path,
        workspace,
        project,
        script,
        "leave",
        token=fixed,
        env={**browser, "TEST_OPENER_SECONDS": "3"},
    )
    try:
        token = server.token
        assert token and (fixed is None or token == fixed)
        assert "opens in your browser" in server.said()
        wait_for(record.exists, 60, "the browser to be opened")
        seen = json.loads(record.read_text(encoding="utf-8"))
        # The browser: started with the name of a file, in an environment without the token.
        assert len(seen["argv"]) == 2 and seen["argv"][1].startswith("file://")
        assert token not in seen["cmdline"] and token not in json.dumps(seen["environ"])
        assert TOKEN_ENV not in seen["environ"]
        # The file: the person's alone, in a directory of its own, and it holds the link.
        file = seen["file"]
        assert Path(file["path"]).is_relative_to(tmp_path / "server-tmp")
        assert file["mode"] == 0o600 and file["directory_mode"] == 0o700
        assert file["beside"] == [Path(file["path"]).name]
        assert f"http://127.0.0.1:{server.port}/?token={token}" in file["text"]
        # The server and its worker: neither command line holds it, and the worker (which
        # starts the jobs) did not inherit it.
        started = json.loads((project.root / "started-with.json").read_text(encoding="utf-8"))
        assert token not in json.dumps(started) and TOKEN_ENV not in started["environ"]
        for pid in (server.process.pid, server.worker):
            assert token.encode() not in Path(f"/proc/{pid}/cmdline").read_bytes()
        assert token.encode() not in Path(f"/proc/{server.worker}/environ").read_bytes()
        # Nothing of the project holds it either: no log, no event, no record.
        for path in project.root.rglob("*"):
            if path.is_file():
                assert token.encode() not in path.read_bytes(), path
        assert server.said().count(token) == 1
        # The file is gone a moment later, and its directory with it.
        wait_for(lambda: not Path(file["path"]).parent.exists(), 60, "the file to be removed")
        assert list((tmp_path / "server-tmp").iterdir()) == []
        assert server.process.poll() is None
    finally:
        server.close()


@pytestmark_posix
def test_the_file_for_the_browser_does_not_outlive_the_server(tmp_path, workspace, script, browser):
    record = Path(browser["TEST_BROWSER_RECORD"])
    server = script_server(tmp_path, workspace, script, "leave", env=browser)
    try:
        wait_for(record.exists, 60, "the browser to be opened")
        file = Path(json.loads(record.read_text(encoding="utf-8"))["file"]["path"])
        assert file.is_file()  # kept for half a minute, unless the server ends first
        assert server.ends_on(signal.SIGINT) == 0
        assert not file.parent.exists()
        assert list((tmp_path / "server-tmp").iterdir()) == []
    finally:
        server.close()


@pytestmark_posix
def test_the_link_is_not_written_into_output_other_users_can_read(tmp_path, workspace, project):
    # `paper-orchestra ui > ui.log` on a shared machine: with the usual umask everybody
    # can read that file, and the link in it would let them in.
    made = cli_server(tmp_path, workspace, token=None, private=False, name="made")
    try:
        said = made.said()
        assert made.token is None and "?token=" not in said.replace("?token=<", "")
        assert "other users of this machine can read" in said
        assert "PAPER_ORCHESTRA_UI_TOKEN" in said and "terminal" in said
        [answer] = made.ask({"path": "/api/health"})
        assert answer.status == 401
    finally:
        made.close()
    fixed = cli_server(tmp_path, workspace, token=TOKEN, private=False, name="fixed")
    try:
        said = fixed.said()
        assert TOKEN not in said and "PAPER_ORCHESTRA_UI_TOKEN" in said
        [answer] = fixed.ask({"path": "/api/health", "headers": bearer()})
        assert answer.status == 200
    finally:
        fixed.close()


# What a real browser makes of the file it is opened with: where it ends up, and what it
# then holds.  The address of the file arrives in the environment.
OPENED_IN_A_BROWSER = r"""
const { chromium } = require('playwright');
(async () => {
  let browser;
  try { browser = await chromium.launch(); }
  catch (error) { console.error(String(error.message).split('\n')[0]); process.exit(3); }
  const context = await browser.newContext();
  const page = await context.newPage();
  const statuses = [];
  page.on('response', (r) => { if (r.request().isNavigationRequest()) statuses.push(r.status()); });
  await page.goto(process.env.OPENER_URL);
  const app = await page.waitForSelector('#view > *:not(.boot):not(.loading)', { timeout: 30000 })
    .then(() => true, () => false);
  const out = {
    app,
    url: page.url(),
    title: await page.title(),
    statuses,
    referrer: await page.evaluate(() => document.referrer),
    script_sees: await page.evaluate(() => document.cookie),
    cookies: (await context.cookies()).map((c) => [c.name, c.httpOnly, c.sameSite, c.value]),
  };
  await page.goBack().catch(() => {});
  out.back = page.url();
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


def real_browser_missing() -> str | None:
    """Why no real browser can be driven here, or ``None`` when one can."""
    if os.environ.get("PAPER_ORCHESTRA_SKIP_E2E"):
        return "PAPER_ORCHESTRA_SKIP_E2E is set"
    if shutil.which("node") is None:
        return "node is not installed"
    probe = subprocess.run(
        ["node", "-e", "require('playwright')"], capture_output=True, text=True, timeout=60
    )
    return None if probe.returncode == 0 else "the Node package 'playwright' is not installed"


@pytestmark_posix
def test_the_file_takes_a_real_browser_into_the_app(tmp_path, workspace, script, browser):
    # The file is a page of another site as far as the browser is concerned, and a browser
    # sends no SameSite=Strict cookie along with what such a page starts: answered with a
    # plain redirect, the link the file leads to set the cookie and then showed the page
    # that asks for the link.
    reason = real_browser_missing()
    if reason:
        pytest.skip(reason)
    record = Path(browser["TEST_BROWSER_RECORD"])
    # (The file is kept longer than it is in earnest: a browser that is driven by a test
    # may take a while to start.)
    patient = {**browser, "TEST_OPENER_SECONDS": "300"}
    server = script_server(tmp_path, workspace, script, "leave", token=None, env=patient)
    try:
        wait_for(record.exists, 60, "the browser to be opened")
        handed = json.loads(record.read_text(encoding="utf-8"))["argv"][1]
        token = server.token
        # (The browser's own temporary files stay with the test's.)
        scratch = tmp_path / "browser-tmp"
        scratch.mkdir()
        done = subprocess.run(
            ["node", "-e", OPENED_IN_A_BROWSER],
            env={**os.environ, "OPENER_URL": handed, "TMPDIR": str(scratch)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        if done.returncode == 3:
            pytest.skip("Playwright could not start a browser: " + done.stderr.strip()[-300:])
        assert done.returncode == 0, done.stderr
        seen = json.loads(done.stdout)
        # In the app, at an address without the token, and nothing was refused on the way.
        assert seen["app"] is True and seen["title"].endswith("PaperOrchestra"), seen
        assert seen["url"].split("#")[0] == f"http://127.0.0.1:{server.port}/"
        assert 401 not in seen["statuses"]
        # The browser holds the token as a cookie no script can read, and nothing else of
        # the visit holds it: not the address, not where the page came from, not the way back.
        assert seen["cookies"] == [[f"port{server.port}_po_token", True, "Strict", token]]
        assert seen["script_sees"] == "" and seen["referrer"] == ""
        assert token not in seen["url"] and token not in seen["back"]
    finally:
        server.close()
