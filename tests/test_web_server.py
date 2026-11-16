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
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from paper_orchestra.research import Project
from paper_orchestra.web import app as app_module
from paper_orchestra.web import security
from paper_orchestra.web.api import UploadLimits
from paper_orchestra.web.app import create_app
from paper_orchestra.web.security import PathEscape, SecurityMiddleware

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
            "query_string": b"",
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


# The server as `paper-orchestra ui` runs it, with the stand-in as its worker.
SERVER_SCRIPT = r"""
import signal, sys

from paper_orchestra.web import app, workers

stand_in, workspace, mode = sys.argv[1:4]
# However the tests themselves were started (in the background, under nohup), the
# server begins as one started from a terminal does.
signal.signal(signal.SIGHUP, signal.SIG_DFL)
workers.default_command = lambda kind, root, options: [
    sys.executable, stand_in, "research", kind, str(root)
]
app.serve(workspace, port=0, open_browser=False, stop_workers_on_exit=(mode == "stop"))
"""

# The test process may not open connections (conftest); a process of its own asks.
POST_SCRIPT = r"""
import sys, time, urllib.error, urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
request = urllib.request.Request(
    sys.argv[1], data=b"{}", method="POST", headers={"Content-Type": "application/json"}
)
deadline = time.time() + 120
while True:
    try:
        with opener.open(request, timeout=60) as response:
            print(response.read().decode())
            break
    except urllib.error.HTTPError as refused:
        print(refused.read().decode())
        sys.exit(1)
    except OSError:
        if time.time() > deadline:
            raise
        time.sleep(0.2)
"""


class RunningServer:
    """``serve()`` in a process of its own, with one project whose worker runs."""

    def __init__(self, tmp_path: Path, workspace: Path, project: Project, script: Path, mode: str):
        self.project = project
        self.output = tmp_path / "server.out"
        code = tmp_path / "serve.py"
        code.write_text(SERVER_SCRIPT, encoding="utf-8")
        with self.output.open("wb") as out:
            self.process = subprocess.Popen(
                [sys.executable, str(code), str(script), str(workspace), mode],
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.worker: int | None = None
        try:
            wait_for(lambda: "is running at http://" in self.said(), 120, "the server to start")
            address = self.said().split("is running at ", 1)[1].split()[0]
            answer = subprocess.run(
                [sys.executable, "-c", POST_SCRIPT, f"{address}api/projects/{project.id}/advance"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert answer.returncode == 0, answer.stdout + answer.stderr
            self.worker = json.loads(answer.stdout)["worker"]["pid"]
            wait_for(lambda: (project.root / "ready").exists(), 120, "the worker")
        except BaseException:
            self.close()
            raise

    def said(self) -> str:
        return self.output.read_text(encoding="utf-8", errors="replace")

    def ends_on(self, signum: int) -> int:
        os.kill(self.process.pid, signum)
        return self.process.wait(timeout=120)

    def close(self) -> None:
        end(self.process.pid)
        self.process.wait()
        if self.worker is not None:
            end(self.worker)


@pytestmark_posix
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_however_the_server_ends_its_workers_go_on(tmp_path, workspace, project, script, signum):
    # Ctrl-C and a termination signal used to stop every worker (and put a run in flight
    # back in the queue); a closed terminal or a lost SSH session killed the server at
    # once and left them running.  Now there is one rule, and the server says so.
    server = RunningServer(tmp_path, workspace, project, script, "leave")
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
    server = RunningServer(tmp_path, workspace, project, script, "stop")
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
