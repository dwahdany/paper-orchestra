"""The web app's HTTP API, in process: every endpoint, its refusals, and the live stream.

Requests go through ``fastapi.testclient.TestClient`` (no socket).  The event
stream is driven at the ASGI interface instead, because a test client reads a
response only once it has ended and a stream never does.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("markdown_it")

from fastapi.testclient import TestClient

from paper_orchestra.llm.base import LLMConfigError
from paper_orchestra.llm.fake import FakeLLM
from paper_orchestra.research import Project, ProjectError
from paper_orchestra.research.model import Conflict, ResearchPage
from paper_orchestra.web import api as api_module
from paper_orchestra.web import workers as workers_module
from paper_orchestra.web.api import LiveSettings, Services, UploadLimits, _Tail
from paper_orchestra.web.app import create_app
from paper_orchestra.web.security import (
    CONTENT_SECURITY_POLICY,
    PathEscape,
    confined,
    is_loopback,
    safe_upload_path,
)

BASE_URL = "http://127.0.0.1:8600"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

# A worker that behaves like the engine as far as the server can tell.
WORKER_SCRIPT = r"""
import json, os, sys, time
from pathlib import Path

root, kind = Path(sys.argv[1]), sys.argv[2]
(root / "launched.json").write_text(json.dumps(sys.argv[2:]))
state = {"schema": 1, "status": "running", "round": 1, "phase": "research",
         "phases": [{"round": 1, "phase": "research", "status": "running"}], "error": None}
if kind == "advance":
    (root / "state.json").write_text(json.dumps(state))
if "die" in sys.argv:
    os._exit(7)
(root / "ready").write_text("yes")
try:
    time.sleep(120)
except KeyboardInterrupt:
    if kind == "advance":
        state.update(status="stopped")
        state["phases"][0]["status"] = "interrupted"
        (root / "state.json").write_text(json.dumps(state))
    sys.exit(130)
"""


def wait_for(condition, timeout: float = 15.0, message: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {message}")
        time.sleep(0.02)


class FakeServices(Services):
    """Stand-in for ``research.experiments`` (spec 10.2), which is built separately:
    the same calls, working on the project's files in the simplest way."""

    def __init__(self) -> None:
        self.approvals: list[tuple[str, dict[str, Any]]] = []
        self.refuse_approval: str | None = None
        self.authors: dict[tuple[str, str], dict[str, str]] = {}
        self.paper_gap: str | None = None
        #: What finding a model's client raises here; ``None``: every model can be used.
        self.no_client: Exception | None = None

    def client_for(self, model):
        # The tests have no credentials; what a press will do is tested with models that
        # can be used, and without them where a test says so.
        if self.no_client is not None:
            raise self.no_client
        return FakeLLM("x")

    def load_plan(self, project, ref):
        page = project.page(ref)
        if page.meta.get("mode") not in ("script", "gpu", "repo"):
            raise ProjectError(f"{page.id}: '{page.meta.get('mode')}' is not a mode.")
        return {"id": page.id, "path": page.path, "title": page.title, "mode": page.meta["mode"]}

    def _code_dir(self, project, experiment_id) -> Path:
        return project.experiment_dir(experiment_id) / "code"

    def code_files(self, project, experiment_id):
        folder = self._code_dir(project, experiment_id)
        files = sorted(folder.rglob("*")) if folder.is_dir() else []
        return {f.relative_to(folder).as_posix(): f.read_text() for f in files if f.is_file()}

    def write_code(self, project, experiment_id, files, *, author):
        folder = self._code_dir(project, experiment_id)
        for name, text in files.items():
            target = folder / name
            # The API must never hand over a name that leaves the code directory.
            assert target.resolve().is_relative_to(folder.resolve()), name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            self.authors[experiment_id, name] = {
                "author": author,
                "written": "2026-11-15T12:30:00Z",
            }
        return self.check_code(project, experiment_id)

    def code_authors(self, project, experiment_id):
        return {name: entry for (eid, name), entry in self.authors.items() if eid == experiment_id}

    def check_code(self, project, experiment_id):
        files = self.code_files(project, experiment_id)
        return [] if "main.py" in files else ["The entry point main.py is missing."]

    def code_digest(self, project, experiment_id):
        files = self.code_files(project, experiment_id)
        return "digest-of-" + "|".join(
            f"{name}:{len(text)}" for name, text in sorted(files.items())
        )

    def approve(self, project, ref, **options):
        if self.refuse_approval:
            raise ProjectError(self.refuse_approval)
        page = project.page(ref)
        read = options.pop("code_digest", None)
        if read is not None and read != self.code_digest(project, page.id):
            raise Conflict("The code changed after the version that was read.", page=page)
        self.approvals.append((project.page(ref).id, options))
        if options:
            project.update_page(ref, author="human", fields=options)
        return project.set_state(ref, "approved", by="human")

    def request_cancel(self, project, ref):
        (project.experiment_dir(project.page(ref).id) / "cancel").write_text("requested")

    def runs(self, project, experiment_id):
        folder = project.experiment_dir(experiment_id) / "runs"
        found = [json.loads(f.read_text()) for f in folder.glob("*/run.json")]
        return sorted(found, key=lambda run: run["number"])

    def options(self, project, ref):
        return [
            {"backend": "local", "can_run": True, "reasons": [], "est_minutes": 2.0,
             "est_usd": None, "notes": ["The network cannot be blocked locally."]},
            {"backend": "modal", "can_run": False, "reasons": ["Modal is not set up."],
             "est_minutes": None, "est_usd": None, "notes": []},
        ]  # fmt: skip

    def number_check(self, project, page):
        return [
            {"number": "0.91", "found_in": ["wiki/results/E1-run1.md"]},
            {"number": "3.5", "found_in": []},
        ]

    def doctor(self, model=None):
        return {"version": "test", "ok": True, "model": model, "provider": {"usable": True}}

    def templates(self):
        return ["cvpr2025", "iclr2025"]

    def paper_missing(self, project):
        return self.paper_gap


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "research"
    path.mkdir()
    return path


@pytest.fixture
def launched(tmp_path: Path):
    """A launcher that runs the stand-in worker; ``launched.extra`` is added to its
    command line."""
    script = tmp_path / "stand_in_worker.py"
    script.write_text(WORKER_SCRIPT, encoding="utf-8")

    def launcher(kind, root, options):
        launcher.calls.append((kind, dict(options)))
        return [sys.executable, str(script), str(root), kind, *launcher.extra]

    launcher.calls = []
    launcher.extra = []
    return launcher


@pytest.fixture
def services() -> FakeServices:
    return FakeServices()


@pytest.fixture
def live() -> LiveSettings:
    return LiveSettings(poll_seconds=0.01, heartbeat_seconds=0.15, reconcile_seconds=0.05)


@pytest.fixture
def app(workspace, launched, services, live, tmp_path):
    return create_app(
        workspace,
        launcher=launched,
        services=services,
        live=live,
        grace=2.0,
        static_dir=tmp_path / "no-frontend",
        # A server leaves its workers running when it ends, unless it was asked to stop
        # them.  A test's server is asked to: no test leaves a worker behind.
        stop_workers_on_exit=True,
    )


@pytest.fixture
def client(app):
    with TestClient(app, base_url=BASE_URL) as client:
        yield client


@pytest.fixture
def project(workspace: Path) -> Project:
    """A project a few steps in: the person's pages, agents' proposals, a run."""
    project = Project.create(workspace / "warmup", title="Warmup", brief="Does warmup matter?")
    project.create_page(
        "topic",
        "Adaptive optimisers",
        "Adam divides by a running average.\n\nSee [[idea]].\n",
        author="human",
        summary="How Adam scales its steps.",
    )
    project.create_page(
        "hypothesis",
        "Warmup reduces early variance",
        "## Statement\n\nWarmup lowers the variance of the first steps.\n",
        author="agent:theorist",
        summary="Variance falls with warmup.",
        relations={"motivated_by": ["topics/adaptive-optimisers"]},
        round=1,
    )
    project.create_page(
        "experiment",
        "Variance with and without warmup",
        "## Objective\n\nMeasure it.\n",
        author="agent:designer",
        fields={"mode": "script", "entrypoint": "main.py"},
        relations={"tests": ["H1"]},
        round=1,
    )
    code = project.experiment_dir("E1") / "code"
    code.mkdir(parents=True)
    (code / "main.py").write_text("print('hello')\n")
    project.propose_edit(
        "topics/adaptive-optimisers",
        author="agent:analyst",
        summary="Say which average Adam uses",
        body="Adam divides by a running average of squared gradients.\n\nSee [[idea]].\n",
        fields={"summary": "How Adam scales its steps, precisely."},
        round=1,
    )
    project.add_request("Look at RAdam too.")
    return project


def add_run(project: Project, number: int = 1, status: str = "succeeded") -> Path:
    """What ``research.experiments.run`` leaves behind for one run of E1."""
    run_dir = project.experiment_dir("E1") / "runs" / str(number)
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "curve.png").write_bytes(PNG)
    (run_dir / "log.txt").write_text("step 1\nstep 2\ndone\n", encoding="utf-8")
    record = {
        "experiment": "E1", "number": number, "status": status, "backend": "local",
        "handle": None, "started": "2026-11-15T12:00:00Z", "finished": "2026-11-15T12:00:42Z",
        "seconds": 42.1, "exit_code": 0, "timed_out": False, "metrics": {"accuracy": 0.91},
        "artifacts": ["artifacts/curve.png"], "error": None, "cost_usd": None,
        "code_digest": "abc", "repairs": [],
    }  # fmt: skip
    (run_dir / "run.json").write_text(json.dumps(record))
    return run_dir


def url(project: Project, rest: str = "") -> str:
    return f"/api/projects/{project.id}{rest}"


def error_of(response) -> dict[str, str]:
    body = response.json()
    assert set(body["error"]) == {"code", "message"}, body
    return body["error"]


# ---------------------------------------------------------------------------
# Health, setup, templates, rendering
# ---------------------------------------------------------------------------


# Every endpoint of the specification (section 7.1), as method and path.
SPECIFIED = """
GET /api/health
GET /api/doctor
GET /api/templates
GET /api/projects
POST /api/projects
GET /api/projects/{pid}
PATCH /api/projects/{pid}
POST /api/projects/{pid}/advance
POST /api/projects/{pid}/stop
GET /api/projects/{pid}/events
GET /api/projects/{pid}/activity
GET /api/projects/{pid}/pages
GET /api/projects/{pid}/pages/{path}
PUT /api/projects/{pid}/pages/{path}
DELETE /api/projects/{pid}/pages/{path}
POST /api/projects/{pid}/pages
GET /api/projects/{pid}/pages/{path}/versions/{stamp}
POST /api/projects/{pid}/review
GET /api/projects/{pid}/proposals
GET /api/projects/{pid}/graph
GET /api/projects/{pid}/requests
POST /api/projects/{pid}/requests
DELETE /api/projects/{pid}/requests/{rid}
GET /api/projects/{pid}/experiments
GET /api/projects/{pid}/experiments/{eid}
PUT /api/projects/{pid}/experiments/{eid}/code/{name}
POST /api/projects/{pid}/experiments/{eid}/approve
POST /api/projects/{pid}/experiments/{eid}/cancel
GET /api/projects/{pid}/experiments/{eid}/runs/{number}
GET /api/projects/{pid}/experiments/{eid}/runs/{number}/log
POST /api/projects/{pid}/experiments/{eid}/runs/{number}/adopt-code
GET /api/projects/{pid}/files/{path}
POST /api/projects/{pid}/paper
GET /api/projects/{pid}/paper
GET /api/projects/{pid}/search
POST /api/render
"""


def test_every_specified_endpoint_exists(client):
    schema = client.get("/api/openapi.json")
    assert schema.status_code == 200
    served = {
        f"{method.upper()} {path}"
        for path, methods in schema.json()["paths"].items()
        for method in methods
    }
    assert set(SPECIFIED.split("\n")) - {""} <= served
    # What was added to it: running an approved experiment now, the worker's log,
    # forgetting a worker the server cannot verify, and the settings a project starts with.
    assert served - set(SPECIFIED.split("\n")) == {
        "POST /api/projects/{pid}/experiments/{eid}/run",
        "GET /api/projects/{pid}/worker/log",
        "POST /api/projects/{pid}/worker/forget",
        "GET /api/defaults",
    }


def test_health_says_what_is_running(client, workspace):
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["name"] == "PaperOrchestra"
    assert body["workspace"] == str(workspace)
    # What the end of this server does to its workers: the test's server was asked to
    # stop them, a server started without the option leaves them running.
    assert body["stops_workers_on_exit"] is True
    with TestClient(create_app(workspace), base_url=BASE_URL) as plain:
        assert plain.get("/api/health").json()["stops_workers_on_exit"] is False


def test_doctor_reports_what_the_command_reports(client):
    assert client.get("/api/doctor?model=some-model").json()["model"] == "some-model"


def test_the_real_doctor_never_returns_a_secret(workspace, monkeypatch):
    secret = "sk-ant-test-" + "x" * 40
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-" + "y" * 30)
    with TestClient(create_app(workspace), base_url=BASE_URL) as client:
        response = client.get("/api/doctor")
    assert response.status_code == 200
    report = response.json()
    assert {"version", "provider", "compute", "packages", "ok"} <= set(report)
    assert secret not in response.text and "y" * 30 not in response.text
    assert report["semantic_scholar_api_key"] is True


def test_templates_lists_the_bundled_ones(client, workspace):
    assert client.get("/api/templates").json() == {
        "templates": [{"name": "cvpr2025"}, {"name": "iclr2025"}]
    }
    with TestClient(create_app(workspace), base_url=BASE_URL) as real:
        names = [entry["name"] for entry in real.get("/api/templates").json()["templates"]]
    assert "cvpr2025" in names


def test_render_previews_with_the_page_renderer(client, project):
    plain = client.post("/api/render", json={"markdown": "# Hi\n\n<script>x</script> [[H1]]"})
    assert plain.status_code == 200
    html = plain.json()["html"]
    assert "<script" not in html and "wikilink-missing" in html
    inside = client.post("/api/render", json={"markdown": "[[H1]]", "project": project.id})
    assert f'href="#/p/{project.id}/wiki/' in inside.json()["html"]
    assert (
        client.post("/api/render", json={"markdown": "x", "project": "p-nope"}).status_code == 404
    )
    too_long = client.post("/api/render", json={"markdown": "x" * 1_000_001})
    assert too_long.status_code == 413 and error_of(too_long)["code"] == "too_large"


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def test_a_project_is_created_from_the_composer_with_its_files(client, workspace):
    assert client.get("/api/projects").json() == {"projects": [], "unreadable": []}
    response = client.post(
        "/api/projects",
        data={
            "title": "Warmup and batch size",
            "brief": "Does warmup matter at small batch sizes?",
            "settings": json.dumps({"limits": {"model_usd": 3}, "compute": "local"}),
        },
        files=[
            ("files", ("notes.md", b"# Notes\n", "text/markdown")),
            ("files", ("curves/loss curve.png", PNG, "image/png")),
        ],
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["title"] == "Warmup and batch size" and body["id"].startswith("p-")
    assert body["dir"] == "warmup-and-batch-size"
    assert body["settings"]["limits"]["model_usd"] == 3
    assert body["state"]["status"] == "idle" and body["checkpoint"] is None
    assert body["worker"]["state"] == "idle"
    assert body["counts"]["pages"] == 1
    assert [entry["name"] for entry in body["attachments"]] == ["curves/loss_curve.png", "notes.md"]
    assert body["uploads"] == [
        {"sent": "notes.md", "stored": "notes.md"},
        {"sent": "curves/loss curve.png", "stored": "curves/loss_curve.png"},
    ]

    root = workspace / "warmup-and-batch-size"
    assert (root / "inputs" / "brief.md").read_text() == "Does warmup matter at small batch sizes?"
    assert (root / "inputs" / "curves" / "loss_curve.png").read_bytes() == PNG
    assert sorted(entry.name for entry in workspace.iterdir()) == ["warmup-and-batch-size"]

    listed = client.get("/api/projects").json()["projects"]
    assert [(row["id"], row["title"], row["status"]) for row in listed] == [
        (body["id"], "Warmup and batch size", "idle")
    ]
    # Nothing waits in a project that has only its idea: counted, not unknown.
    assert listed[0]["worker"] == "idle" and listed[0]["pending"] == 0
    assert listed[0]["counts"]["by_kind"]["idea"] == 1
    detail = client.get(f"/api/projects/{body['id']}").json()
    assert detail["id"] == body["id"] and detail["usage"] is None and detail["problems"] == []


def test_a_second_project_with_the_same_title_gets_its_own_directory(client, workspace):
    first = client.post("/api/projects", data={"brief": "Same idea."}).json()
    second = client.post("/api/projects", data={"brief": "Same idea."}).json()
    assert first["id"] != second["id"]
    assert {first["dir"], second["dir"]} == {"same-idea", "same-idea-2"}
    assert first["title"] == "Same idea."


@pytest.mark.parametrize(
    "sent",
    [
        "../../../etc/passwd",
        "/etc/passwd",
        "..\\..\\windows\\system32\\evil.dll",
        "C:\\Users\\me\\secret.txt",
        ".env",
        ".ssh/id_rsa",
        "a/../../b.txt",
        "name\x00.txt",
        "con.txt",
        "~/x.txt",
        "brief.md",
        "manifest.json",
        "...",
    ],
)
def test_hostile_upload_names_stay_inside_the_projects_inputs(client, workspace, sent, tmp_path):
    response = client.post(
        "/api/projects",
        data={"title": "Names", "brief": "The idea."},
        files=[("files", (sent, b"payload", "application/octet-stream"))],
    )
    assert response.status_code == 201, response.text
    inputs = workspace / "names" / "inputs"
    written = sorted(
        p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file()
    )
    assert all(path.startswith("names/") for path in written)
    payloads = [p for p in inputs.rglob("*") if p.is_file() and p.read_bytes() == b"payload"]
    assert len(payloads) == 1  # kept, under a safe name
    stored = payloads[0].relative_to(inputs)
    assert ".." not in stored.parts and not any(part.startswith(".") for part in stored.parts)
    assert (inputs / "brief.md").read_text() == "The idea."  # not replaced by an upload
    assert json.loads((inputs / "manifest.json").read_text())["brief"] == "brief.md"
    # Nothing appeared next to the workspace or in the usual places.
    assert (
        sorted(p.name for p in tmp_path.iterdir() if p.name != "research") == ["no-frontend"]
        or not (tmp_path / "etc").exists()
    )
    assert not (workspace.parent / "etc").exists() and not (workspace / "etc").exists()


def test_two_uploads_with_the_same_name_are_both_kept(client, workspace):
    response = client.post(
        "/api/projects",
        data={"title": "Twins", "brief": "x"},
        files=[
            ("files", ("data.csv", b"a,b\n", "text/csv")),
            ("files", ("DATA.csv", b"c,d\n", "text/csv")),
            ("files[]", ("data.csv/inner.txt", b"nested", "text/plain")),
        ],
    )
    assert response.status_code == 201, response.text
    stored = [upload["stored"] for upload in response.json()["uploads"]]
    assert len({name.lower() for name in stored}) == 3
    inputs = workspace / "twins" / "inputs"
    assert sorted(p.read_bytes() for p in inputs.rglob("*") if p.suffix in (".csv", ".txt")) == [
        b"a,b\n",
        b"c,d\n",
        b"nested",
    ]


def test_upload_limits_are_enforced_and_nothing_is_left_behind(workspace, tmp_path):
    limits = UploadLimits(file_bytes=1000, total_bytes=2500)
    app = create_app(workspace, uploads=limits, static_dir=tmp_path / "none")
    with TestClient(app, base_url=BASE_URL) as client:
        too_big = client.post(
            "/api/projects",
            data={"title": "Big", "brief": "x"},
            files=[("files", ("big.bin", b"x" * 1001, "application/octet-stream"))],
        )
        assert too_big.status_code == 413
        assert "big.bin" in error_of(too_big)["message"]
        assert "No project was created" in error_of(too_big)["message"]

        too_many = client.post(
            "/api/projects",
            data={"title": "Many", "brief": "x"},
            files=[
                ("files", (f"f{n}.bin", b"x" * 900, "application/octet-stream")) for n in range(3)
            ],
        )
        assert too_many.status_code == 413 and error_of(too_many)["code"] == "too_large"

        huge = client.post(
            "/api/projects",
            data={"title": "Huge", "brief": "x"},
            files=[("files", ("huge.bin", b"x" * 6_000_000, "application/octet-stream"))],
        )
        assert huge.status_code == 413

        fits = client.post(
            "/api/projects",
            data={"title": "Fits", "brief": "x"},
            files=[
                ("files", (f"f{n}.bin", b"x" * 1000, "application/octet-stream")) for n in range(2)
            ],
        )
        assert fits.status_code == 201
    assert sorted(entry.name for entry in workspace.iterdir()) == ["fits"]


def test_an_upload_that_does_not_say_its_size_is_cut_off_at_the_limit(workspace, tmp_path):
    limits = UploadLimits(file_bytes=10_000_000, total_bytes=100_000, text_bytes=10_000)
    app = create_app(workspace, uploads=limits, static_dir=tmp_path / "none")
    boundary = "----test-boundary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="brief"\r\n\r\nAn idea.\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="big.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()

    def body():  # sent in pieces, with no Content-Length to check beforehand
        yield head
        for _ in range(40):
            yield b"x" * 65536
        yield f"\r\n--{boundary}--\r\n".encode()

    with TestClient(app, base_url=BASE_URL) as client:
        response = client.post(
            "/api/projects",
            content=body(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    assert response.status_code == 413 and error_of(response)["code"] == "too_large"
    # Refused as it arrived, before any of it was parsed or stored.
    assert "Attach fewer or smaller files" in error_of(response)["message"]
    assert "larger than the 0.1 MB allowed" in error_of(response)["message"]  # the limit itself
    assert list(workspace.iterdir()) == []


def test_a_project_that_cannot_be_created_leaves_nothing(client, workspace):
    nothing = client.post("/api/projects", data={"title": "", "brief": "  "})
    assert nothing.status_code == 422 and "nothing to start from" in error_of(nothing)["message"]
    bad_json = client.post("/api/projects", data={"brief": "x", "settings": "{not json"})
    assert bad_json.status_code == 422 and "not valid JSON" in error_of(bad_json)["message"]
    not_object = client.post("/api/projects", data={"brief": "x", "settings": "[1]"})
    assert not_object.status_code == 422
    invalid = client.post(
        "/api/projects",
        data={"brief": "x", "settings": json.dumps({"compute": "abacus"})},
        files=[("files", ("a.txt", b"a", "text/plain"))],
    )
    assert invalid.status_code == 422 and "compute" in error_of(invalid)["message"]
    assert list(workspace.iterdir()) == []


def test_creating_with_start_runs_the_agents_once_the_project_is_on_disk(
    client, workspace, launched
):
    response = client.post("/api/projects", data={"brief": "Start now.", "start": "true"})
    assert response.status_code == 201
    body = response.json()
    assert body["worker"]["state"] == "running" and body["worker"]["kind"] == "advance"
    root = workspace / body["dir"]
    assert (root / "project.json").is_file()
    wait_for(lambda: (root / "ready").exists())
    assert launched.calls == [("advance", {})]


def test_a_project_whose_agents_cannot_start_is_created_all_the_same(workspace, tmp_path):
    def broken(kind, root, options):
        return ["/nonexistent/interpreter"]

    app = create_app(workspace, launcher=broken, static_dir=tmp_path / "none")
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.post("/api/projects", data={"brief": "Keep this.", "start": "1"})
    assert response.status_code == 201
    body = response.json()
    assert body["worker"]["state"] == "idle"
    assert body["worker_error"]["code"] == "worker_failed"
    assert "could not be started" in body["worker_error"]["message"]
    assert (workspace / body["dir"] / "inputs" / "brief.md").read_text() == "Keep this."


def test_unknown_projects_are_not_found(client, project):
    for path in ("", "/pages", "/graph", "/events", "/files/inputs/brief.md"):
        response = client.get(f"/api/projects/p-000000000000{path}")
        assert response.status_code == 404, path
        assert error_of(response)["code"] == "not_found"


def test_title_and_settings_can_be_changed(client, project):
    response = client.patch(
        url(project),
        json={"title": "Warmup, revisited", "settings": {"limits": {"model_usd": 40.0}}},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Warmup, revisited"
    assert response.json()["settings"]["limits"]["model_usd"] == 40.0
    assert response.json()["settings"]["limits"]["max_sources"] == 40  # the rest is kept
    reopened = Project.open(project.root)
    assert reopened.title == "Warmup, revisited" and reopened.settings.limits.model_usd == 40.0

    refused = client.patch(url(project), json={"settings": {"compute": "abacus"}})
    assert refused.status_code == 422 and "compute" in error_of(refused)["message"]
    assert client.get(url(project)).json()["settings"]["compute"] == "local"
    assert client.patch(url(project), json={"title": "  "}).status_code == 422
    assert client.get(url(project)).json()["title"] == "Warmup, revisited"


def test_a_change_made_by_another_process_is_seen(client, project):
    assert client.get(url(project)).json()["title"] == "Warmup"
    other = Project.open(project.root)
    other.title = "Renamed in a terminal"
    other.save_settings()
    assert client.get(url(project)).json()["title"] == "Renamed in a terminal"


def test_the_library_counts_what_waits_once_and_then_reads_no_page(client, workspace, monkeypatch):
    """A row says what waits in its project now (the audit: the number of the last
    checkpoint went stale and left edit proposals out).  Counting reads the pages of a
    project; it is done once, and again only when its pages or proposals changed."""
    for number in range(40):
        made = Project.create(workspace / f"p{number:02}", title=f"Project {number}", brief="x")
        if number % 2:
            state = {"status": "checkpoint", "round": 2, "phase": "hypothesize",
                     "next_phase": "design", "spent": {"model_usd": 1.5},
                     "checkpoint": {"round": 2, "phase": "hypothesize", "summary": "Five waiting.",
                                    "pending": {"hypothesis": 5, "question": 1}}}  # fmt: skip
            (made.root / "state.json").write_text(json.dumps(state))
            made.create_page("hypothesis", "One", "x", author="agent:theorist", round=2)
            topic = made.create_page("topic", "Mine", "x", author="human")
            made.propose_edit(topic.path, author="agent:analyst", summary="s", body="y", round=2)

    monkeypatch.setattr(api_module, "_COUNT_SECONDS", 60.0)
    rows = client.get("/api/projects").json()["projects"]
    assert len(rows) == 40
    waiting = [row for row in rows if row["status"] == "checkpoint"]
    # One proposed page and one edit proposal, not the six of a checkpoint long ago.
    assert len(waiting) == 20 and all(row["pending"] == 2 for row in waiting)
    assert waiting[0]["counts"]["pending"] == {
        "total": 2, "pages": 1, "proposals": 1, "by_kind": {"hypothesis": 1},
    }  # fmt: skip
    assert waiting[0]["counts"]["by_kind"]["topic"] == 1
    assert all(row["pending"] == 0 for row in rows if row["status"] != "checkpoint")
    assert waiting[0]["checkpoint"]["summary"] == "Five waiting."
    assert waiting[0]["spent"] == {
        "model_usd": 1.5,
        "experiment_seconds": 0.0,
        "experiment_usd": 0.0,
    }

    def no_pages(self, *args, **kwargs):
        raise AssertionError("the library must not read pages again")

    opened = []
    with monkeypatch.context() as patched:
        patched.setattr(Project, "_all_pages", no_pages)
        patched.setattr(Project, "_load_page", no_pages)
        patched.setattr(Project, "reload", lambda self: opened.append(self.root))
        again = client.get("/api/projects").json()["projects"]
    assert opened == []  # nothing changed, so nothing was read again
    assert [row["pending"] for row in again] == [row["pending"] for row in rows]

    # A decision changes what waits in that project, and the next listing says so.
    decided = Project.open(workspace / "p01")
    decided.review("H1", "accept")
    after = {row["dir"]: row for row in client.get("/api/projects").json()["projects"]}
    assert after["p01"]["pending"] == 1 and after["p03"]["pending"] == 2


def test_counting_for_the_library_is_bounded_and_caught_up_by_the_next_listings(
    client, workspace, monkeypatch
):
    for number in range(6):
        made = Project.create(workspace / f"p{number}", title=f"Project {number}", brief="x")
        made.create_page("hypothesis", "One", "x", author="agent:theorist", round=1)
        state = {"status": "checkpoint", "round": 1, "phase": "hypothesize",
                 "checkpoint": {"round": 1, "phase": "hypothesize", "summary": "s",
                                "pending": {"hypothesis": 7}}}  # fmt: skip
        (made.root / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(api_module, "_COUNT_SECONDS", 0.0)  # no time for any counting
    rows = client.get("/api/projects").json()["projects"]
    # Not counted yet: the row says so and falls back on what the checkpoint counted.
    assert all(row["counts"] is None and row["pending"] == 7 for row in rows)
    monkeypatch.setattr(api_module, "_COUNT_SECONDS", 60.0)
    rows = client.get("/api/projects").json()["projects"]
    assert all(row["counts"]["pending"]["total"] == 1 and row["pending"] == 1 for row in rows)


def test_a_copied_project_directory_is_still_reachable(client, workspace, project):
    import shutil

    shutil.copytree(project.root, workspace / "warmup-copy")
    rows = client.get("/api/projects").json()["projects"]
    assert sorted(row["dir"] for row in rows) == ["warmup", "warmup-copy"]
    ids = {row["dir"]: row["id"] for row in rows}
    assert ids["warmup"] == project.id and ids["warmup-copy"] != project.id
    assert client.get(f"/api/projects/{ids['warmup-copy']}").json()["dir"] == "warmup-copy"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def test_pages_are_listed_and_filtered(client, project):
    pages = client.get(url(project, "/pages")).json()["pages"]
    assert [page["path"] for page in pages] == [
        "wiki/idea.md",
        "wiki/topics/adaptive-optimisers.md",
        "wiki/hypotheses/H1-warmup-reduces-early-variance.md",
        "wiki/experiments/E1-variance-with-and-without-warmup.md",
    ]
    assert "body" not in pages[0] and pages[2]["id"] == "H1" and pages[2]["review"] == "proposed"

    def paths(query: str) -> list[str]:
        found = client.get(url(project, f"/pages?{query}")).json()["pages"]
        return [page["path"].rsplit("/", 1)[-1] for page in found]

    assert paths("kind=hypothesis") == ["H1-warmup-reduces-early-variance.md"]
    assert paths("kind=idea,topic&review=accepted") == ["idea.md", "adaptive-optimisers.md"]
    assert paths("review=proposed") == [
        "H1-warmup-reduces-early-variance.md",
        "E1-variance-with-and-without-warmup.md",
    ]
    assert paths("state=planned") == ["E1-variance-with-and-without-warmup.md"]
    assert paths("kind=finding") == []
    found = client.get(url(project, "/pages?q=running+average")).json()["pages"]
    assert [page["path"] for page in found] == ["wiki/topics/adaptive-optimisers.md"]
    assert "running average" in found[0]["snippet"]


def test_a_page_comes_with_what_the_wiki_view_shows(client, project):
    response = client.get(url(project, "/pages/wiki/topics/adaptive-optimisers.md"))
    assert response.status_code == 200
    view = response.json()
    page = view["page"]
    assert page["title"] == "Adaptive optimisers" and page["author"] == "human"
    assert page["body"].startswith("Adam divides") and page["meta"]["kind"] == "topic"
    assert f'href="#/p/{project.id}/wiki/wiki%2Fidea.md"' in view["html"]
    assert [(link["path"], link["type"]) for link in view["backlinks"]] == [
        ("wiki/hypotheses/H1-warmup-reduces-early-variance.md", "motivated_by")
    ]
    assert view["backlinks"][0]["label"] == "H1" and view["backlinks"][0]["review"] == "proposed"
    assert [(link["path"], link["type"]) for link in view["links"]] == [
        ("wiki/idea.md", "mentions")
    ]
    assert view["history"] == [] and "number_check" not in view

    (proposal,) = view["proposals"]
    assert proposal["id"].startswith("ep-") and proposal["author"] == "agent:analyst"
    assert proposal["summary"] == "Say which average Adam uses"
    assert "-Adam divides by a running average." in proposal["diff"].splitlines()
    assert "+Adam divides by a running average of squared gradients." in proposal["diff"]
    assert '<span class="diff-line diff-add">+Adam divides' in proposal["diff_html"]
    # The diff is headed in the words of the app: which side is the page, which the
    # suggestion and whose, and where the change is; not "agent:analyst" and "@@ -1 +1 @@".
    assert proposal["diff"].splitlines()[:2] == [
        "--- the page as it is",
        "+++ as the analyst suggests it",
    ]
    assert "− the page as it is" in proposal["diff_html"]
    assert "+ as the analyst suggests it" in proposal["diff_html"]
    assert "agent:" not in proposal["diff_html"] and "@@" not in proposal["diff_html"]
    assert '<span class="diff-line diff-hunk">From line 1</span>' in proposal["diff_html"]
    assert "average <ins>of squared gradients</ins>." in proposal["marked_html"]
    assert 'class="wikilink"' in proposal["marked_html"]  # the unchanged block, as it reads
    assert (proposal["words_added"], proposal["words_removed"]) == (3, 0)
    assert proposal["field_changes"] == [
        {
            "field": "summary",
            "from": "How Adam scales its steps.",
            "to": "How Adam scales its steps, precisely.",
        }
    ]
    assert proposal["page_changed"] is False


def test_a_page_is_found_by_id_and_link_target_too(client, project):
    by_id = client.get(url(project, "/pages/H1")).json()["page"]
    by_link = client.get(url(project, "/pages/hypotheses/H1-warmup-reduces-early-variance")).json()
    assert by_id["path"] == by_link["page"]["path"]
    missing = client.get(url(project, "/pages/wiki/topics/nothing.md"))
    assert missing.status_code == 404 and error_of(missing)["code"] == "not_found"
    for hostile in ("..%2f..%2fproject.json", "wiki/..%2f..%2fstate.json", "%2fetc%2fpasswd"):
        assert client.get(url(project, f"/pages/{hostile}")).status_code == 404


def test_a_finding_carries_its_number_check(client, project, services):
    project.create_page(
        "finding", "Warmup helps", "Accuracy was 0.91, up 3.5 points.\n", author="agent:analyst"
    )
    view = client.get(url(project, "/pages/F1")).json()
    assert view["number_check"] == [
        {"number": "0.91", "found_in": ["wiki/results/E1-run1.md"]},
        {"number": "3.5", "found_in": []},
    ]

    def broken(project, page):
        raise ProjectError("The records could not be read.")

    services.number_check = broken
    view = client.get(url(project, "/pages/F1")).json()
    assert view["number_check"] is None
    assert view["number_check_error"] == "The records could not be read."
    assert view["page"]["id"] == "F1"  # the page itself is still served


def test_the_person_edits_a_page_and_the_old_version_is_kept(client, project):
    path = "/pages/wiki/topics/adaptive-optimisers.md"
    before = client.get(url(project, path)).json()["page"]
    response = client.put(
        url(project, path),
        json={
            "title": "Adaptive optimisers and warmup",
            "body": "Rewritten by the person.\n",
            "fields": {"summary": "New summary."},
            "relations": {"cites": []},
            "base_updated": before["updated"],
        },
    )
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["page"]["title"] == "Adaptive optimisers and warmup"
    assert view["page"]["body"] == "Rewritten by the person.\n"
    assert view["page"]["summary"] == "New summary." and view["page"]["updated"] > before["updated"]
    assert "<p>Rewritten by the person.</p>" in view["html"]
    assert [version["stamp"] for version in view["history"]] == [before["updated"]]
    assert view["proposals"][0]["page_changed"] is True

    old = client.get(url(project, f"{path}/versions/{before['updated']}")).json()
    assert old["body"] == before["body"] and old["meta"]["title"] == "Adaptive optimisers"
    assert old["text"].startswith("---\n") and "<p>Adam divides" in old["html"]
    assert "-Adam divides by a running average." in old["diff"].splitlines()
    assert "+Rewritten by the person." in old["diff"].splitlines()
    assert 'class="diff-line diff-del"' in old["diff_html"]
    missing = client.get(url(project, f"{path}/versions/2001-01-01T00:00:00Z"))
    assert missing.status_code == 404


def test_a_stale_edit_is_refused_with_the_current_page(client, project):
    path = "/pages/wiki/topics/adaptive-optimisers.md"
    first = client.get(url(project, path)).json()["page"]
    client.put(url(project, path), json={"body": "Tab one.\n", "base_updated": first["updated"]})
    stale = client.put(
        url(project, path), json={"body": "Tab two.\n", "base_updated": first["updated"]}
    )
    assert stale.status_code == 409
    body = stale.json()
    assert body["error"]["code"] == "conflict" and "Reload the page" in body["error"]["message"]
    assert body["page"]["body"] == "Tab one.\n" and body["page"]["updated"] > first["updated"]
    assert client.get(url(project, path)).json()["page"]["body"] == "Tab one.\n"
    # With the version just handed back, the same edit goes through.
    again = client.put(
        url(project, path), json={"body": "Tab two.\n", "base_updated": body["page"]["updated"]}
    )
    assert again.status_code == 200 and again.json()["page"]["body"] == "Tab two.\n"


def test_editing_refuses_what_a_page_cannot_be(client, project):
    path = "/pages/wiki/topics/adaptive-optimisers.md"
    kind = client.put(url(project, path), json={"kind": "hypothesis", "body": "x"})
    assert kind.status_code == 422 and "stays one" in error_of(kind)["message"]
    assert client.put(url(project, path), json={"title": "  "}).status_code == 422
    assert (
        client.put(url(project, "/pages/wiki/topics/none.md"), json={"body": "x"}).status_code
        == 404
    )
    state = client.put(url(project, "/pages/E1"), json={"fields": {"state": "running"}})
    assert state.status_code in (400, 422)
    assert client.put(url(project, path), content=b"not json").status_code == 422


def test_editing_a_proposal_does_not_decide_it_and_save_and_accept_does(client, project):
    """Changed for the audit (truth 14): a person's edit used to accept the page."""
    kept = client.put(url(project, "/pages/H1"), json={"body": "Corrected, still open.\n"})
    assert kept.status_code == 200
    page = kept.json()["page"]
    assert page["review"] == "proposed" and page["body"] == "Corrected, still open.\n"
    assert "decided_by" not in page["meta"] and page["meta"]["updated_by"] == "human"

    # A rejected page stays rejected however it is edited.
    client.post(url(project, "/review"), json={"items": [{"ref": "E1", "decision": "reject"}]})
    edited = client.put(url(project, "/pages/E1"), json={"body": "## Objective\n\nAgain.\n"})
    assert edited.status_code == 200 and edited.json()["page"]["review"] == "rejected"

    accepted = client.put(
        url(project, "/pages/H1"),
        json={"body": "Corrected and accepted.\n", "accept": True, "base_updated": page["updated"]},
    )
    assert accepted.status_code == 200
    page = accepted.json()["page"]
    assert page["review"] == "accepted" and page["meta"]["decided_by"] == "human"
    assert page["body"] == "Corrected and accepted.\n"
    last = client.get(url(project, "/activity?limit=1")).json()["events"][0]
    assert last["type"] == "review.decided" and last["data"]["edited"] is True
    # Saving and accepting a version that is no longer the one that was read is refused.
    stale = client.put(
        url(project, "/pages/E1"),
        json={"body": "x\n", "accept": True, "base_updated": "2020-01-01T00:00:00Z"},
    )
    assert stale.status_code == 409
    assert client.get(url(project, "/pages/E1")).json()["page"]["review"] == "rejected"


def test_editing_a_page_of_ones_own_changes_nothing_about_its_review(client, project):
    edited = client.put(
        url(project, "/pages/topics/adaptive-optimisers"), json={"body": "Sharper.\n"}
    )
    assert edited.status_code == 200
    page = edited.json()["page"]
    assert page["review"] == "accepted" and page["author"] == "human"
    # "Save and accept" on a page that is accepted already is a plain save.
    again = client.put(
        url(project, "/pages/topics/adaptive-optimisers"),
        json={"body": "Sharper still.\n", "accept": True},
    )
    assert again.status_code == 200 and again.json()["page"]["review"] == "accepted"
    last = client.get(url(project, "/activity?limit=1")).json()["events"][0]
    assert last["type"] == "page.updated"


def test_the_person_adds_and_deletes_pages(client, project):
    created = client.post(
        url(project, "/pages"),
        json={
            "kind": "question",
            "title": "Does batch size change it?",
            "body": "Open.\n",
            "summary": "Batch size.",
            "relations": {"about": ["idea"]},
        },
    )
    assert created.status_code == 201, created.text
    page = created.json()["page"]
    assert page["id"] == "Q1" and page["review"] == "accepted" and page["author"] == "human"
    assert page["relations"] == {"about": ["idea"]} and page["state"] == "open"

    assert (
        client.post(url(project, "/pages"), json={"kind": "poem", "title": "x"}).status_code == 422
    )
    assert client.post(url(project, "/pages"), json={"kind": "topic"}).status_code == 422
    assert (
        client.post(url(project, "/pages"), json={"kind": "source", "title": "x"}).status_code
        == 422
    )

    deleted = client.delete(url(project, "/pages/Q1"))
    assert deleted.status_code == 200 and deleted.json() == {"deleted": page["path"]}
    assert client.get(url(project, "/pages/Q1")).status_code == 404
    assert client.delete(url(project, "/pages/Q1")).status_code == 404
    idea = client.delete(url(project, "/pages/wiki/idea.md"))
    assert idea.status_code == 422 and "cannot be deleted" in error_of(idea)["message"]
    assert (project.root / "wiki" / "idea.md").is_file()


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


def test_review_applies_each_decision_and_reports_each_outcome(client, project):
    hypothesis = client.get(url(project, "/pages/H1")).json()["page"]
    response = client.post(
        url(project, "/review"),
        json={
            "items": [
                {"ref": "H1", "decision": "accept", "base_updated": hypothesis["updated"]},
                {"ref": "H9", "decision": "accept"},
                {"ref": "E1", "decision": "reject", "reason": "Too slow for a first test."},
                {"ref": "wiki/idea.md", "decision": "maybe"},
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is False
    accepted, missing, rejected, unclear = body["results"]
    assert accepted["ok"] is True and accepted["page"]["review"] == "accepted"
    assert accepted["page"]["meta"]["decided_by"] == "human" and "body" not in accepted["page"]
    assert missing == {
        "ref": "H9",
        "decision": "accept",
        "ok": False,
        "status": 404,
        "error": missing["error"],
    }
    assert missing["error"]["code"] == "not_found"
    assert rejected["ok"] is True and rejected["page"]["review"] == "rejected"
    assert rejected["page"]["meta"]["decision_reason"] == "Too slow for a first test."
    assert unclear["ok"] is False and unclear["status"] == 422
    assert project.page("H1").review == "accepted" and project.page("E1").review == "rejected"


def test_review_can_edit_while_accepting(client, project):
    response = client.post(
        url(project, "/review"),
        json={
            "items": [
                {
                    "ref": "H1",
                    "decision": "accept",
                    "edits": {
                        "body": "## Statement\n\nNarrower.\n",
                        "fields": {"title": "Narrower"},
                    },
                }
            ]
        },
    )
    assert response.json()["ok"] is True
    page = project.page("H1")
    assert page.title == "Narrower" and page.body == "## Statement\n\nNarrower.\n"
    assert page.review == "accepted"


def test_a_decision_on_a_version_the_person_did_not_see_is_a_conflict(client, project):
    seen = client.get(url(project, "/pages/H1")).json()["page"]["updated"]
    project.update_page("H1", author="agent:theorist", body="## Statement\n\nRevised since.\n")
    response = client.post(
        url(project, "/review"),
        json={"items": [{"ref": "H1", "decision": "accept", "base_updated": seen}]},
    )
    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["ok"] is False and result["status"] == 409
    assert result["error"]["code"] == "conflict" and result["applied"] is False
    assert result["page"]["updated"] > seen
    assert project.page("H1").review == "proposed"


def test_edit_proposals_are_listed_and_decided(client, project):
    listed = client.get(url(project, "/proposals")).json()["proposals"]
    assert len(listed) == 1
    proposal = listed[0]
    assert proposal["target"] == "wiki/topics/adaptive-optimisers.md"
    assert proposal["page"]["title"] == "Adaptive optimisers" and "body" not in proposal["page"]
    assert (proposal["lines_added"], proposal["lines_removed"]) == (1, 1)

    response = client.post(
        url(project, "/review"), json={"items": [{"ref": proposal["id"], "decision": "accept"}]}
    )
    (result,) = response.json()["results"]
    assert result["ok"] is True and result["proposal"]["review"] == "accepted"
    assert result["page"]["summary"] == "How Adam scales its steps, precisely."
    assert "squared gradients" in project.page("topics/adaptive-optimisers").body
    assert client.get(url(project, "/proposals")).json()["proposals"] == []
    assert len(client.get(url(project, "/proposals?review=accepted")).json()["proposals"]) == 1
    assert len(client.get(url(project, "/proposals?review=all")).json()["proposals"]) == 1
    view = client.get(url(project, "/pages/wiki/topics/adaptive-optimisers.md")).json()
    assert view["proposals"] == [] and len(view["history"]) == 1

    again = client.post(
        url(project, "/review"), json={"items": [{"ref": proposal["id"], "decision": "reject"}]}
    )
    assert again.json()["results"][0]["ok"] is False  # already decided


def test_an_edit_proposal_whose_page_changed_reports_the_conflict(client, project):
    proposal = client.get(url(project, "/proposals")).json()["proposals"][0]
    project.update_page("topics/adaptive-optimisers", author="human", body="The person's text.\n")
    response = client.post(
        url(project, "/review"), json={"items": [{"ref": proposal["id"], "decision": "accept"}]}
    )
    (result,) = response.json()["results"]
    assert result["ok"] is False and result["status"] == 409
    assert result["applied"] is True  # its field went in, its text did not
    assert result["page"]["summary"] == "How Adam scales its steps, precisely."
    assert project.page("topics/adaptive-optimisers").body == "The person's text.\n"
    rejected = client.post(
        url(project, "/review"),
        json={"items": [{"ref": "ep-00000000", "decision": "reject", "reason": "x"}]},
    )
    assert rejected.json()["results"][0]["status"] == 404


def test_a_plain_accept_leaves_an_experiment_planned(client, project, services):
    response = client.post(
        url(project, "/review"), json={"items": [{"ref": "E1", "decision": "accept"}]}
    )
    assert response.json()["results"][0]["page"]["state"] == "planned"
    assert services.approvals == []


def test_accept_and_run_approves_in_the_same_decision(client, project, services, launched):
    response = client.post(
        url(project, "/review"),
        json={
            "items": [
                {
                    "ref": "E1",
                    "decision": "accept",
                    "approve_run": {"backend": "local", "timeout_seconds": 300, "network": False},
                }
            ]
        },
    )
    (result,) = response.json()["results"]
    assert result["ok"] is True and result["approved"] is True
    assert result["page"]["review"] == "accepted" and result["page"]["state"] == "approved"
    assert services.approvals == [
        ("E1", {"backend": "local", "timeout_seconds": 300, "network": False})
    ]
    assert "worker" not in result and launched.calls == []  # approved, not started


def test_accept_and_run_with_defaults_and_with_start(client, project, services, launched):
    response = client.post(
        url(project, "/review"),
        json={"items": [{"ref": "E1", "decision": "accept", "approve_run": {"start": True}}]},
    )
    (result,) = response.json()["results"]
    assert result["ok"] is True and services.approvals == [("E1", {})]
    assert result["worker"]["state"] == "running" and result["worker"]["kind"] == "run"
    assert launched.calls == [("run", {"experiments": ["E1"]})]
    wait_for(lambda: (project.root / "ready").exists())

    project.create_page(
        "experiment", "Second", "x", author="agent:designer", fields={"mode": "script"}
    )
    busy = client.post(
        url(project, "/review"),
        json={"items": [{"ref": "E2", "decision": "accept", "approve_run": {"start": True}}]},
    )
    (second,) = busy.json()["results"]
    # Approved all the same; it waits for the worker that is running.
    assert second["ok"] is True and second["approved"] is True
    assert second["worker_error"]["code"] == "worker_running"


def test_accept_and_run_says_when_only_the_accept_went_through(client, project, services):
    services.refuse_approval = "E1 cannot run: main.py does not parse."
    response = client.post(
        url(project, "/review"),
        json={"items": [{"ref": "E1", "decision": "accept", "approve_run": True}]},
    )
    (result,) = response.json()["results"]
    assert result["ok"] is False and result["approved"] is False and result["status"] == 422
    assert result["error"]["message"] == (
        "The plan was accepted, but its run was not approved: "
        "E1 cannot run: main.py does not parse."
    )
    assert result["page"]["review"] == "accepted" and result["page"]["state"] == "planned"


def test_approve_run_belongs_to_accepting_an_experiment(client, project, services):
    response = client.post(
        url(project, "/review"),
        json={
            "items": [
                {"ref": "H1", "decision": "accept", "approve_run": True},
                {"ref": "E1", "decision": "reject", "approve_run": True},
            ]
        },
    )
    first, second = response.json()["results"]
    assert first["status"] == 422 and second["status"] == 422
    assert project.page("H1").review == "proposed" and project.page("E1").review == "proposed"
    assert services.approvals == []


def test_a_malformed_review_changes_nothing(client, project):
    for body in (
        {},
        {"items": "H1"},
        {"items": [{"ref": "H1"}]},
        {"items": [{"decision": "accept"}]},
    ):
        response = client.post(url(project, "/review"), json=body)
        assert response.status_code == 422, body
        assert error_of(response)["code"] == "invalid_request"
    assert project.page("H1").review == "proposed"
    assert client.post(url(project, "/review"), json={"items": []}).json() == {
        "ok": True,
        "results": [],
    }


# ---------------------------------------------------------------------------
# Graph, requests, search, activity
# ---------------------------------------------------------------------------


def test_graph_is_the_projects_graph(client, project):
    graph = client.get(url(project, "/graph")).json()
    assert {node["id"] for node in graph["nodes"]} == {page.path for page in project.pages()}
    assert {
        "source": "wiki/experiments/E1-variance-with-and-without-warmup.md",
        "target": "wiki/hypotheses/H1-warmup-reduces-early-variance.md",
        "type": "tests",
    } in graph["edges"]
    assert graph["dangling"] == []
    project.review("E1", "reject", reason="no")
    assert len(client.get(url(project, "/graph")).json()["nodes"]) == 3
    assert len(client.get(url(project, "/graph?include_rejected=true")).json()["nodes"]) == 4


def test_requests_are_left_listed_and_withdrawn(client, project):
    listed = client.get(url(project, "/requests")).json()["requests"]
    assert [request["text"] for request in listed] == ["Look at RAdam too."]
    created = client.post(
        url(project, "/requests"), json={"text": "Design a smaller test.", "target": "H1"}
    )
    assert created.status_code == 201
    request = created.json()["request"]
    assert request["id"].startswith("rq-") and request["state"] == "open"
    assert request["target"] == "wiki/hypotheses/H1-warmup-reduces-early-variance.md"
    assert len(client.get(url(project, "/requests?state=open")).json()["requests"]) == 2
    assert client.get(url(project, "/requests?state=done")).json()["requests"] == []

    assert client.post(url(project, "/requests"), json={"text": "  "}).status_code == 422
    assert (
        client.post(url(project, "/requests"), json={"text": "x", "target": "H9"}).status_code
        == 404
    )
    assert client.delete(url(project, f"/requests/{request['id']}")).json() == {
        "deleted": request["id"]
    }
    # The withdrawal is in the project's activity, so that another window drops the row.
    last = client.get(url(project, "/activity?limit=1")).json()["events"][0]
    assert last["type"] == "request.withdrawn" and last["data"]["request"] == request["id"]
    assert last["summary"] == "You withdrew your request to the agents: Design a smaller test."
    assert client.delete(url(project, f"/requests/{request['id']}")).status_code == 404
    assert client.delete(url(project, "/requests/..%2f..%2fproject")).status_code == 404
    assert len(client.get(url(project, "/requests")).json()["requests"]) == 1


def test_search_covers_proposed_pages_too(client, project):
    body = client.get(url(project, "/search?q=variance")).json()
    assert body["query"] == "variance"
    assert {hit["id"] for hit in body["hits"]} == {"H1", "E1"}
    assert (
        body["hits"][0]["snippet"]
        and body["hits"][0]["score"] > 0
        and "body" not in body["hits"][0]
    )
    assert client.get(url(project, "/search?q=")).json()["hits"] == []
    assert client.get(url(project, "/search?q=zebra")).json()["hits"] == []


def test_activity_pages_back_through_past_events(client, project):
    events = project.events()
    assert len(events) >= 6
    latest = client.get(url(project, "/activity?limit=3")).json()
    assert [event["seq"] for event in latest["events"]] == [e["seq"] for e in events[-3:]]
    assert latest["has_more"] is True
    older = client.get(
        url(project, f"/activity?limit=3&before={latest['events'][0]['seq']}")
    ).json()
    assert [event["seq"] for event in older["events"]] == [e["seq"] for e in events[-6:-3]]
    everything = client.get(url(project, "/activity?limit=1000")).json()
    assert everything["events"] == events and everything["has_more"] is False
    assert client.get(url(project, "/activity?before=1")).json() == {
        "events": [],
        "has_more": False,
    }
    # A line that is still being written is not an event yet.
    with (project.root / "events.jsonl").open("ab") as handle:
        handle.write(b'{"seq": 999, "type": "agent.act')
    assert (
        client.get(url(project, "/activity?limit=1")).json()["events"][0]["seq"]
        == events[-1]["seq"]
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def test_experiments_are_listed_with_their_runs(client, project):
    rows = client.get(url(project, "/experiments")).json()["experiments"]
    assert [(row["id"], row["state"], row["runs"], row["last_run"]) for row in rows] == [
        ("E1", "planned", 0, None)
    ]
    add_run(project, 1, "failed")
    add_run(project, 2)
    (row,) = client.get(url(project, "/experiments")).json()["experiments"]
    assert row["runs"] == 2 and row["last_run"]["number"] == 2
    assert row["last_run"]["metrics"] == {"accuracy": 0.91}
    assert row["last_run"]["artifact_urls"] == {
        "artifacts/curve.png": f"/api/projects/{project.id}/files/experiments/E1/runs/2/artifacts/curve.png"
    }
    assert "body" not in row


def test_the_experiment_view_says_whether_network_access_grants_the_model_api(
    client, project, monkeypatch, tmp_path
):
    """With model access for jobs on, approving a local run with network access lets its
    code call the Anthropic API on the worker's account, counted nowhere.  The form says
    so where the approval is given, which needs the view to say whether it is on."""
    from paper_orchestra import cli

    assert client.get(url(project, "/experiments/E1")).json()["model_access"] is False
    token = tmp_path / "identity.jwt"
    token.write_text("jwt", encoding="utf-8")
    monkeypatch.setenv("PAPER_ORCHESTRA_JOB_MODEL_ACCESS", "1")
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "rule")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token))
    assert client.get(url(project, "/experiments/E1")).json()["model_access"] is True

    # The command line says the same when the approval is given.
    page = project.page("E1")
    page.meta.update(backend="local", network=True)
    assert cli._grants_model_access(page)
    assert "counted against no limit of the project" in cli.MODEL_ACCESS_SAID
    page.meta.update(network=False)
    assert not cli._grants_model_access(page)
    page.meta.update(backend="modal", network=True)
    assert not cli._grants_model_access(page)


def test_an_experiment_comes_with_plan_code_runs_and_options(client, project):
    add_run(project)
    view = client.get(url(project, "/experiments/E1")).json()
    assert view["page"]["id"] == "E1" and "<h2" in view["html"]
    assert view["plan"]["mode"] == "script" and view["plan_error"] is None
    assert view["code"] == {
        # The file was put there by hand: nobody is on record as its author.
        "files": [
            {
                "name": "main.py",
                "text": "print('hello')\n",
                "size": 15,
                "author": None,
                "written": None,
            }
        ],
        "problems": [],
        "digest": "digest-of-main.py:15",
    }
    assert [run["number"] for run in view["runs"]] == [1]
    assert view["runs"][0]["log_url"].endswith("/experiments/E1/runs/1/log")
    assert view["runs"][0]["result_page"] is None
    assert [(option["backend"], option["can_run"]) for option in view["options"]] == [
        ("local", True),
        ("modal", False),
    ]
    assert view["options"][1]["reasons"] == ["Modal is not set up."]
    assert client.get(url(project, "/experiments/e1")).status_code == 200

    project.update_page("E1", author="human", fields={"mode": "quantum"})
    broken = client.get(url(project, "/experiments/E1")).json()
    assert broken["plan"] is None and "not a mode" in broken["plan_error"]

    for missing in ("E9", "H1", "nonsense", "..%2fE1"):
        response = client.get(url(project, f"/experiments/{missing}"))
        assert response.status_code == 404, missing


def test_the_code_listing_says_who_wrote_each_file_and_when(client, project):
    client.put(url(project, "/experiments/E1/code/lib/util.py"), json={"text": "X = 1\n"})
    listed = client.get(url(project, "/experiments/E1")).json()["code"]["files"]
    files = {entry["name"]: entry for entry in listed}
    assert files["lib/util.py"]["author"] == "human"
    assert files["lib/util.py"]["written"] == "2026-11-15T12:30:00Z"
    assert files["main.py"]["author"] is None and files["main.py"]["written"] is None


def test_the_real_services_say_who_wrote_the_code_and_what_a_paper_lacks(project):
    real = Services()
    assert real.code_authors(project, "E1") == {}  # main.py was put there by hand
    real.write_code(project, "E1", {"main.py": "print(1)\n"}, author="human")
    entry = real.code_authors(project, "E1")["main.py"]
    assert entry["author"] == "human" and entry["written"].endswith("Z")

    # The write-up's own rule, asked without starting it: results first, then a finding.
    from paper_orchestra.research import writeup

    assert real.paper_missing(project) == "results"
    with pytest.raises(ProjectError, match="no accepted experiment has a run"):
        writeup._check_ready(project, *writeup._reportable(project))
    project.inputs_dir.mkdir(parents=True, exist_ok=True)
    (project.inputs_dir / "prior_run.csv").write_text("step,loss\n1,0.7\n", encoding="utf-8")
    (project.inputs_dir / "manifest.json").write_text(
        json.dumps({"files": [{"name": "prior_run.csv", "size": 18}]}), encoding="utf-8"
    )
    assert real.paper_missing(project) == "finding"
    with pytest.raises(ProjectError, match="no finding is accepted"):
        writeup._check_ready(project, *writeup._reportable(project))
    project.create_page("finding", "Warmup helps", "It does.\n", author="human")
    assert real.paper_missing(project) is None
    writeup._check_ready(project, *writeup._reportable(project))


def test_the_person_edits_code_and_gets_the_check_back(client, project):
    saved = client.put(url(project, "/experiments/E1/code/lib/util.py"), json={"text": "X = 1\n"})
    assert saved.status_code == 200 and saved.json() == {"name": "lib/util.py", "problems": []}
    assert (project.experiment_dir("E1") / "code" / "lib" / "util.py").read_text() == "X = 1\n"
    names = [f["name"] for f in client.get(url(project, "/experiments/E1")).json()["code"]["files"]]
    assert names == ["lib/util.py", "main.py"]
    for hostile in (
        "..%2f..%2f..%2fevil.py",
        "%2ftmp%2fpaper-orchestra-test-escape.py",
        "lib%2f%2futil.py",
        "lib/%2e/util.py",
        ".hidden.py",
        "a%5cb.py",
    ):
        refused = client.put(url(project, f"/experiments/E1/code/{hostile}"), json={"text": "x"})
        assert refused.status_code in (404, 422), hostile
    assert not Path("/tmp/paper-orchestra-test-escape.py").exists()
    assert (
        not (project.root / "evil.py").exists() and not (project.root.parent / "evil.py").exists()
    )
    assert client.put(url(project, "/experiments/E1/code/main.py"), json={}).status_code == 422


def test_approving_an_experiment_records_the_options(client, project, services):
    project.review("E1", "accept")
    response = client.post(
        url(project, "/experiments/E1/approve"),
        json={"backend": "modal", "gpu": "T4", "timeout_seconds": 3600, "network": True},
    )
    assert response.status_code == 200, response.text
    page = response.json()["page"]
    assert page["state"] == "approved" and page["meta"]["gpu"] == "T4"
    assert services.approvals == [
        ("E1", {"backend": "modal", "gpu": "T4", "timeout_seconds": 3600.0, "network": True})
    ]
    assert "worker" not in response.json()

    seen = client.post(
        url(project, "/experiments/E1/approve"), json={"base_updated": page["updated"]}
    )
    assert seen.status_code == 200
    assert services.approvals[-1] == ("E1", {"base_updated": page["updated"]})

    services.refuse_approval = "E1 is proposed, not accepted. Accept the page first."
    refused = client.post(url(project, "/experiments/E1/approve"))
    assert refused.status_code == 422 and "Accept the page first" in error_of(refused)["message"]
    assert client.post(url(project, "/experiments/E7/approve")).status_code == 404


def test_approving_code_that_changed_since_it_was_read_is_a_conflict(client, project, services):
    # The designer may still rewrite the code of a plan nobody decided on.  What the person
    # approves is what they read: the client sends the digest it was shown.
    read = client.get(url(project, "/experiments/E1")).json()["code"]["digest"]
    (project.experiment_dir("E1") / "code" / "main.py").write_text("import os; os.remove('x')\n")
    project.review("E1", "accept")

    stale = client.post(url(project, "/experiments/E1/approve"), json={"code_digest": read})
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "conflict"
    assert project.page("E1").state == "planned" and services.approvals == []

    seen = client.get(url(project, "/experiments/E1")).json()["code"]["digest"]
    assert seen != read
    fresh = client.post(url(project, "/experiments/E1/approve"), json={"code_digest": seen})
    assert fresh.status_code == 200 and fresh.json()["page"]["state"] == "approved"


def test_accept_and_run_refuses_code_that_changed_since_it_was_read(client, project, services):
    read = client.get(url(project, "/experiments/E1")).json()["code"]["digest"]
    (project.experiment_dir("E1") / "code" / "main.py").write_text("import os; os.remove('x')\n")
    body = {"items": [{"ref": "E1", "decision": "accept", "approve_run": {"code_digest": read}}]}
    (result,) = client.post(url(project, "/review"), json=body).json()["results"]
    assert result["ok"] is False and result["status"] == 409
    assert result["error"]["code"] == "conflict"
    assert "code of E1 changed" in result["error"]["message"]
    # One decision: the plan is not accepted with code the person did not read.
    page = project.page("E1")
    assert page.review == "proposed" and page.state == "planned" and services.approvals == []

    seen = client.get(url(project, "/experiments/E1")).json()["code"]["digest"]
    body["items"][0]["approve_run"] = {"code_digest": seen, "timeout_seconds": 60}
    (result,) = client.post(url(project, "/review"), json=body).json()["results"]
    assert result["ok"] is True and result["approved"] is True
    assert services.approvals == [("E1", {"timeout_seconds": 60})]


def test_an_approved_experiment_can_be_run_now(client, project, services, launched):
    planned = client.post(url(project, "/experiments/E1/run"))
    assert planned.status_code == 422 and "Approve it first" in error_of(planned)["message"]
    project.review("E1", "accept")
    started = client.post(url(project, "/experiments/E1/approve"), json={"start": True})
    assert started.json()["page"]["state"] == "approved"
    assert started.json()["worker"]["kind"] == "run"
    assert launched.calls == [("run", {"experiments": ["E1"]})]
    wait_for(lambda: (project.root / "ready").exists())

    second = client.post(url(project, "/experiments/E1/run"))
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "worker_running" and second.json()["phase"] == "run"
    assert client.post(url(project, "/stop")).json()["worker"]["state"] == "stopping"
    wait_for(lambda: client.get(url(project)).json()["worker"]["state"] == "idle")
    again = client.post(url(project, "/experiments/E1/run"))
    assert again.status_code == 202 and again.json()["worker"]["state"] == "running"


def test_cancel_asks_the_running_job_to_stop(client, project, services):
    # Nothing is approved, queued or running: there is nothing to cancel, and nobody is asked.
    nothing = client.post(url(project, "/experiments/E1/cancel"))
    assert nothing.status_code == 200, nothing.text
    assert nothing.json()["outcome"] == "nothing"
    assert nothing.json()["message"] == "E1 is planned: there is nothing to cancel."
    assert not (project.experiment_dir("E1") / "cancel").exists()

    running_without_a_worker(project, services)
    response = client.post(url(project, "/experiments/E1/cancel"))
    # Whoever runs it is told to stop (the stand-in leaves the flag and cancels nothing).
    assert response.status_code == 202, response.text
    body = response.json()
    assert (body["experiment"], body["outcome"], body["before"], body["state"]) == (
        "E1",
        "cancelling",
        "running",
        "running",
    )
    assert body["message"].startswith("E1 was told to stop.")
    assert body["page"]["id"] == "E1" and body["cancel_does"] == "cancel"
    assert (project.experiment_dir("E1") / "cancel").read_text() == "requested"
    assert client.post(url(project, "/experiments/E4/cancel")).status_code == 404


def test_a_run_is_served_with_the_end_of_its_log(client, project):
    run_dir = add_run(project)
    body = client.get(url(project, "/experiments/E1/runs/1")).json()
    assert body["run"]["status"] == "succeeded" and body["run"]["metrics"] == {"accuracy": 0.91}
    assert body["run"]["artifact_urls"]["artifacts/curve.png"].endswith(
        "/runs/1/artifacts/curve.png"
    )
    assert body["log"]["text"] == "step 1\nstep 2\ndone\n" and body["log"]["eof"] is True
    assert client.get(body["run"]["artifact_urls"]["artifacts/curve.png"]).content == PNG

    (run_dir / "log.txt").write_text("".join(f"line {n}\n" for n in range(5000)))
    tail = client.get(url(project, "/experiments/E1/runs/1")).json()["log"]
    assert tail["text"].endswith("line 4999\n") and tail["text"].startswith("line ")
    assert tail["offset"] > 0 and tail["next"] == tail["size"]

    assert client.get(url(project, "/experiments/E1/runs/2")).status_code == 404
    assert client.get(url(project, "/experiments/E1/runs/abc")).status_code == 422
    assert client.get(url(project, "/experiments/E9/runs/1")).status_code == 404


def test_a_log_is_followed_by_offset(client, project):
    log = add_run(project) / "log.txt"
    log.write_bytes(b"first\n")
    path = url(project, "/experiments/E1/runs/1/log")
    piece = client.get(path).json()
    assert piece == {
        "offset": 0,
        "next": 6,
        "size": 6,
        "text": "first\n",
        "eof": True,
        "reset": False,
    }
    assert client.get(f"{path}?offset=6").json()["text"] == ""

    # A character that arrives in two writes is handed out whole, once.
    snowman = "☃".encode()
    with log.open("ab") as handle:
        handle.write(b"temp " + snowman[:2])
    piece = client.get(f"{path}?offset=6").json()
    assert piece["text"] == "temp " and piece["next"] == 11
    # Nothing more can be read for now.  A client that reads on while eof is false must
    # not spin on a log whose writer was killed in the middle of a character.
    assert piece["eof"] is True and piece["size"] == 13
    waiting = client.get(f"{path}?offset=11").json()
    assert waiting["text"] == "" and waiting["next"] == 11 and waiting["eof"] is True
    with log.open("ab") as handle:
        handle.write(snowman[2:] + b" done\n")
    piece = client.get(f"{path}?offset={piece['next']}").json()
    assert piece["text"] == "☃ done\n" and piece["eof"] is True

    # The backend starts the log anew when it re-attaches: the client is told to start over.
    log.write_bytes(b"new\n")
    piece = client.get(f"{path}?offset=500").json()
    assert piece["reset"] is True and piece["offset"] == 0 and piece["text"] == "new\n"
    assert client.get(f"{path}?offset=-2").json()["text"] == ""  # starts at a line
    assert client.get(url(project, "/experiments/E1/runs/5/log")).json()["text"] == ""
    assert client.get(url(project, "/experiments/E1/runs/0/log")).status_code == 404


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_files_are_served_by_what_they_are(client, project):
    (project.inputs_dir / "notes.md").write_text("# Notes\n")
    (project.inputs_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    (project.inputs_dir / "data.bin").write_bytes(b"\x00\x01")
    run_dir = add_run(project)
    (run_dir / "artifacts" / "report.html").write_text("<script>alert(1)</script>")
    (run_dir / "artifacts" / "plot.svg").write_text("<svg onload='alert(1)'/>")

    brief = client.get(url(project, "/files/inputs/brief.md"))
    assert brief.status_code == 200 and brief.text == "Does warmup matter?"
    assert brief.headers["content-type"] == "text/plain; charset=utf-8"
    assert brief.headers["content-disposition"].startswith("inline")
    assert brief.headers["cache-control"] == "no-cache"

    image = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/curve.png"))
    assert image.headers["content-type"] == "image/png" and image.content == PNG

    pdf = client.get(url(project, "/files/inputs/paper.pdf"))
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.headers["x-frame-options"] == "SAMEORIGIN"  # the app may show it in a frame
    assert "frame-ancestors 'self'" in pdf.headers["content-security-policy"]

    # A page of HTML a run wrote is text; it never runs with the app's origin.
    page = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/report.html"))
    assert page.headers["content-type"] == "text/plain; charset=utf-8"
    assert page.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in page.headers["content-security-policy"]
    svg = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/plot.svg"))
    assert svg.headers["content-type"] == "image/svg+xml"
    assert "sandbox" in svg.headers["content-security-policy"]
    assert svg.headers["x-frame-options"] == "DENY"

    binary = client.get(url(project, "/files/inputs/data.bin"))
    assert binary.headers["content-type"] == "application/octet-stream"
    assert binary.headers["content-disposition"].startswith("attachment")
    forced = client.get(url(project, "/files/inputs/notes.md?download=1"))
    assert forced.headers["content-disposition"].startswith("attachment")

    part = client.get(url(project, "/files/inputs/brief.md"), headers={"Range": "bytes=5-10"})
    assert part.status_code == 206 and part.text == "warmup"


@pytest.mark.parametrize(
    "path",
    [
        "inputs/%2e%2e/project.json",
        "inputs/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
        "inputs/..%2fstate.json",
        "%2e%2e/%2e%2e/etc/passwd",
        "%2fetc%2fpasswd",
        "inputs/%2fetc%2fpasswd",
        "inputs%5c..%5cproject.json",
        "project.json",
        "state.json",
        "events.jsonl",
        "worker.log",
        ".worker.json",
        "wiki/idea.md",
        "transcripts/session.jsonl",
        "proposals/ep-1.json",
        "inputs",
        "inputs/",
        "inputs/missing.txt",
        "inputs/.secret",
        "experiments/E1/code/.env",
        "inputs/brief.md%00.png",
    ],
)
def test_files_outside_the_served_areas_are_not_found(client, project, path):
    (project.inputs_dir / ".secret").write_text("MARKER-ONE")
    code = project.experiment_dir("E1") / "code"
    (code / ".env").write_text("KEY=MARKER-TWO")
    response = client.get(url(project, f"/files/{path}"))
    assert response.status_code == 404, response.text
    assert error_of(response)["code"] == "not_found"
    assert "MARKER" not in response.text and project.id not in response.text


def test_symbolic_links_do_not_lead_out_of_the_project(client, project, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("MARKER-OUTSIDE")
    (project.inputs_dir / "link.txt").symlink_to(outside / "secret.txt")
    (project.inputs_dir / "folder").symlink_to(outside, target_is_directory=True)
    (project.inputs_dir / "state.txt").symlink_to(project.root / "project.json")
    (project.inputs_dir / "inside.txt").symlink_to(project.inputs_dir / "brief.md")

    for path in ("inputs/link.txt", "inputs/folder/secret.txt", "inputs/state.txt"):
        response = client.get(url(project, f"/files/{path}"))
        assert response.status_code == 404, path
        assert "MARKER-OUTSIDE" not in response.text and project.id not in response.text
    # A link that stays among the served files is just another name for one of them.
    assert client.get(url(project, "/files/inputs/inside.txt")).text == "Does warmup matter?"


def test_confined_resolves_inside_its_directory_or_refuses(tmp_path):
    root = tmp_path / "root"
    (root / "inputs" / "sub").mkdir(parents=True)
    (root / "inputs" / "a.txt").write_text("a")
    (tmp_path / "elsewhere").mkdir()
    (root / "inputs" / "out").symlink_to(tmp_path / "elsewhere")
    assert confined(root, "inputs/a.txt") == (root / "inputs" / "a.txt").resolve()
    assert confined(root, "inputs/sub/new.txt", areas=("inputs",)).name == "new.txt"
    for bad in ("", "/etc/passwd", "../x", "inputs/../../x", "inputs\\a.txt",
                "inputs/.hidden", "inputs/out/x.txt", "a\x00b", "wiki/idea.md"):  # fmt: skip
        with pytest.raises(PathEscape):
            confined(root, bad, areas=("inputs",))
    with pytest.raises(PathEscape):
        confined(root, "inputs/out")
    # A link with a plain name that leads to something hidden inside the directory.
    (root / ".cache").mkdir()
    (root / ".cache" / "token.txt").write_text("hidden")
    (root / "inputs" / "plain.txt").symlink_to(root / ".cache" / "token.txt")
    with pytest.raises(PathEscape):
        confined(root, "inputs/plain.txt")


def test_upload_names_are_reduced_to_safe_relative_paths():
    assert safe_upload_path("notes.md") == "notes.md"
    assert safe_upload_path("My Results (final).csv") == "My_Results_(final).csv"
    assert safe_upload_path("folder/sub/table.csv") == "folder/sub/table.csv"
    assert safe_upload_path("..\\..\\evil.dll") == "evil.dll"
    assert safe_upload_path("../../etc/passwd") == "etc/passwd"
    assert safe_upload_path("/abs/path.txt") == "abs/path.txt"
    assert safe_upload_path(".env") == "env"
    assert safe_upload_path("résumé.pdf") == "résumé.pdf"
    assert safe_upload_path("a\x00b\n.txt") == "a_b.txt"
    assert safe_upload_path("NUL.txt") == "_NUL.txt"
    assert safe_upload_path("x" * 300 + ".tar.gz").endswith(".gz")
    assert len(safe_upload_path("x" * 300 + ".png")) <= 120
    assert safe_upload_path("/".join(f"d{n}" for n in range(20)) + "/f.txt").count("/") == 7
    for nothing in ("", "...", "///", "..", "?", " . "):
        assert safe_upload_path(nothing) is None


# ---------------------------------------------------------------------------
# The agents: start, stop, death, shutdown
# ---------------------------------------------------------------------------


def test_advance_starts_a_worker_and_a_second_start_is_refused(client, project, launched):
    response = client.post(
        url(project, "/advance"), json={"phase": "research", "instruction": "Start with Adam."}
    )
    assert response.status_code == 202, response.text
    worker = response.json()["worker"]
    assert worker["state"] == "running" and worker["kind"] == "advance" and worker["pid"] > 0
    assert launched.calls == [("advance", {"phase": "research", "instruction": "Start with Adam."})]
    wait_for(lambda: (project.root / "ready").exists())
    assert json.loads((project.root / "launched.json").read_text()) == ["advance"]

    detail = client.get(url(project)).json()
    assert detail["worker"]["state"] == "running" and detail["state"]["status"] == "running"
    assert client.get("/api/projects").json()["projects"][0]["worker"] == "running"

    second = client.post(url(project, "/advance"), json={"phase": "design"})
    assert second.status_code == 409
    body = second.json()
    assert (
        body["error"]["code"] == "worker_running" and "research phase" in body["error"]["message"]
    )
    assert body["phase"] == "research" and body["worker"]["pid"] == worker["pid"]
    assert len(launched.calls) == 1
    paper = client.post(url(project, "/paper"), json={"template": "cvpr2025"})
    assert paper.status_code == 409  # one worker per project, whatever it does

    # The sentence names the phase as the engine's own sentences do, not by the word the
    # engine uses in its files; the field keeps that word for the client.
    state = json.loads((project.root / "state.json").read_text())
    state.update(phase="hypothesize")
    state["phases"] = [{"round": 1, "phase": "hypothesize", "status": "running"}]
    (project.root / "state.json").write_text(json.dumps(state))
    third = client.post(url(project, "/advance")).json()
    assert third["phase"] == "hypothesize"
    assert "a worker running the hypothesis phase." in third["error"]["message"]
    assert "hypothesize" not in third["error"]["message"]


def test_advance_needs_no_body_and_checks_what_it_is_given(client, project, launched):
    bad = client.post(url(project, "/advance"), json={"phase": "publish"})
    assert bad.status_code == 422 and "not a phase" in error_of(bad)["message"]
    long = client.post(url(project, "/advance"), json={"instruction": "x" * 20_001})
    assert long.status_code == 422 and "request" in error_of(long)["message"]
    assert launched.calls == []
    assert client.post(url(project, "/advance")).status_code == 202
    assert launched.calls == [("advance", {})]


def test_stop_returns_at_once_and_the_phase_ends_interrupted(client, project):
    client.post(url(project, "/advance"))
    wait_for(lambda: (project.root / "ready").exists())
    before = time.monotonic()
    response = client.post(url(project, "/stop"))
    assert time.monotonic() - before < 1.5  # the grace period alone is two seconds
    assert response.status_code == 202 and response.json()["worker"]["state"] == "stopping"
    wait_for(lambda: client.get(url(project)).json()["worker"]["state"] == "idle")
    detail = client.get(url(project)).json()
    assert detail["state"]["status"] == "stopped"
    assert detail["state"]["phases"][0]["status"] == "interrupted"
    assert detail["worker"]["last_exit"]["stopped"] is True
    idle = client.post(url(project, "/stop"))
    assert idle.status_code == 202 and idle.json()["worker"]["state"] == "idle"


def test_a_worker_that_dies_shows_as_interrupted_not_running(client, project, launched):
    launched.extra = ["die"]
    client.post(url(project, "/advance"))
    wait_for(lambda: client.get(url(project)).json()["worker"]["state"] == "idle")
    detail = client.get(url(project)).json()
    assert detail["state"]["status"] == "stopped"
    assert "exit code 7" in detail["state"]["error"]
    assert detail["worker"]["last_exit"]["code"] == 7
    assert detail["worker"]["last_exit"]["interrupted_phase"] == "research"
    assert client.get("/api/projects").json()["projects"][0]["status"] == "stopped"
    last = client.get(url(project, "/activity?limit=1")).json()["events"][0]
    assert last["type"] == "error" and "exit code 7" in last["summary"]
    launched.extra = []
    assert client.post(url(project, "/advance")).status_code == 202  # resumable


def test_the_workers_log_can_be_read_without_its_secrets(client, project, launched, monkeypatch):
    secret = "sk-ant-test-" + "z" * 40
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    empty = client.get(url(project, "/worker/log")).json()
    assert empty["text"] == "" and empty["eof"] is True
    (project.root / "worker.log").write_text(
        f"[paper-orchestra] starting\nAuthentication failed for key {secret}\nlast line\n"
    )
    piece = client.get(url(project, "/worker/log")).json()
    assert "Authentication failed for key [redacted]" in piece["text"]
    assert secret not in json.dumps(piece) and piece["text"].endswith("last line\n")
    assert client.get(url(project, f"/worker/log?offset={piece['next']}")).json()["text"] == ""
    assert client.get(url(project, "/worker/log?offset=0")).json()["text"].startswith("[paper-")


def test_a_server_asked_to_stops_its_workers_when_it_shuts_down(app, project):
    # The end of a server leaves its workers running (the next one finds them again);
    # it stops them when it was started to do so, as the app of these tests is.
    with TestClient(app, base_url=BASE_URL) as client:
        pid = client.post(url(project, "/advance")).json()["worker"]["pid"]
        wait_for(lambda: (project.root / "ready").exists())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    state = json.loads((project.root / "state.json").read_text())
    assert state["status"] == "stopped" and state["phases"][0]["status"] == "interrupted"


def test_a_restarted_server_finds_the_running_worker(workspace, project, launched, tmp_path):
    def serve():
        return create_app(workspace, launcher=launched, grace=2.0, static_dir=tmp_path / "none")

    killed = TestClient(serve(), base_url=BASE_URL)  # never shut down: as if killed
    pid = killed.post(url(project, "/advance")).json()["worker"]["pid"]
    wait_for(lambda: (project.root / "ready").exists())

    with TestClient(serve(), base_url=BASE_URL) as restarted:
        worker = restarted.get(url(project)).json()["worker"]
        assert worker["state"] == "running" and worker["adopted"] is True and worker["pid"] == pid
        assert restarted.post(url(project, "/advance")).status_code == 409
        assert restarted.post(url(project, "/stop")).json()["worker"]["state"] == "stopping"
        wait_for(lambda: restarted.get(url(project)).json()["worker"]["state"] == "idle")
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# ---------------------------------------------------------------------------
# The optional paper
# ---------------------------------------------------------------------------


def test_the_paper_step_starts_a_worker_and_reports_its_state(client, project, launched):
    assert client.get(url(project, "/paper")).json() == {
        "exists": False,
        "state": None,
        "running": False,
        "final_pdf": None,
        "missing": None,
        "missing_detail": "",
        "left_out": [],
        "last": None,
        "worker": client.get(url(project)).json()["worker"],
    }
    unknown = client.post(url(project, "/paper"), json={"template": "../../etc"})
    assert unknown.status_code == 422 and "cvpr2025" in error_of(unknown)["message"]
    assert client.post(url(project, "/paper"), json={}).status_code == 422

    started = client.post(url(project, "/paper"), json={"template": "iclr2025", "model": "m-1"})
    assert started.status_code == 202 and started.json()["worker"]["kind"] == "write"
    assert launched.calls == [("write", {"template": "iclr2025", "model": "m-1"})]
    paper = project.paper_dir
    paper.mkdir()
    (paper / "state.json").write_text(json.dumps({"status": "running", "stages": {}}))
    (paper / "final_paper.pdf").write_bytes(b"%PDF-1.4\n")
    body = client.get(url(project, "/paper")).json()
    assert body["exists"] is True and body["running"] is True
    assert body["state"] == {"status": "running", "stages": {}}
    assert body["final_pdf"] == "paper/final_paper.pdf"
    assert client.get(url(project, f"/files/{body['final_pdf']}")).content == b"%PDF-1.4\n"


# ---------------------------------------------------------------------------
# Live events
# ---------------------------------------------------------------------------


class Stream:
    """One request to the app's ASGI interface that stays open, as a browser's
    ``EventSource`` does.  ``frames()`` is what has arrived so far; ``close()`` hangs up."""

    def __init__(self, app, path: str, headers: dict[str, str] | None = None) -> None:
        self.app = app
        self.path, _, self.query = path.partition("?")
        self.headers = {"host": "127.0.0.1:8600", **(headers or {})}
        self.status: int | None = None
        self.response_headers: dict[str, str] = {}
        self._data = b""
        self._lock = threading.Lock()
        self._hang_up = threading.Event()
        self.ended = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._request())
        except BaseException as exc:
            self.error = exc
        finally:
            self.ended.set()

    async def _request(self) -> None:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": self.path,
            "raw_path": self.path.encode(),
            "query_string": self.query.encode(),
            "root_path": "",
            "headers": [
                (k.encode(), v if isinstance(v, bytes) else v.encode())
                for k, v in self.headers.items()
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8600),
        }
        asked = False

        async def receive():
            nonlocal asked
            if not asked:
                asked = True
                return {"type": "http.request", "body": b"", "more_body": False}
            while not self._hang_up.is_set():
                await asyncio.sleep(0.005)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.response_headers = {
                    k.decode().lower(): v.decode() for k, v in message["headers"]
                }
            elif message["type"] == "http.response.body":
                with self._lock:
                    self._data += message.get("body", b"")

        await self.app(scope, receive, send)

    def frames(self) -> list[dict[str, str]]:
        """Every complete frame so far: its fields, or ``{"comment": ...}``."""
        with self._lock:
            text = self._data.decode()
        frames = []
        for block in text.split("\n\n")[:-1]:
            frame: dict[str, str] = {}
            for line in block.split("\n"):
                name, _, value = line.partition(":")
                frame[name or "comment"] = value.strip()
            frames.append(frame)
        return frames

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(f["data"]) for f in self.frames() if "id" in f]

    def states(self) -> list[dict[str, Any]]:
        return [json.loads(f["data"]) for f in self.frames() if f.get("event") == "state"]

    def close(self) -> None:
        self._hang_up.set()
        assert self.ended.wait(10), "the stream did not end when the client hung up"
        if self.error is not None:
            raise self.error


@pytest.fixture
def stream(app):
    opened: list[Stream] = []

    def open_stream(path: str, headers: dict[str, str] | None = None) -> Stream:
        made = Stream(app, path, headers)
        opened.append(made)
        return made

    yield open_stream
    for made in opened:
        made._hang_up.set()
        made.ended.wait(10)


def test_the_stream_replays_then_follows_the_projects_events(stream, project):
    past = project.events()
    live = stream(url(project, "/events"))
    wait_for(lambda: len(live.events()) == len(past), message="the replay")
    assert live.status == 200
    assert live.response_headers["content-type"].startswith("text/event-stream")
    assert live.response_headers["cache-control"] == "no-cache"
    assert live.response_headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert live.events() == past
    ids = [int(frame["id"]) for frame in live.frames() if "id" in frame]
    assert ids == [event["seq"] for event in past]
    assert all("event" not in frame for frame in live.frames() if "id" in frame)
    assert live.frames()[0] == {"retry": "2000"}

    project.emit("agent.activity", "Reading the Adam paper.", agent="scout", phase="research")
    project.add_request("And AdamW.")
    wait_for(lambda: len(live.events()) == len(past) + 2, message="the new events")
    new = live.events()[-2:]
    assert [event["type"] for event in new] == ["agent.activity", "request.created"]
    assert new[0]["summary"] == "Reading the Adam paper." and new[0]["agent"] == "scout"
    live.close()


def test_the_stream_starts_after_what_the_client_already_has(stream, project):
    past = project.events()
    later = stream(url(project, f"/events?after={past[2]['seq']}"))
    wait_for(lambda: len(later.events()) == len(past) - 3)
    assert later.events() == past[3:]

    # A browser that reconnects says where it was; that outranks the address it used first.
    resumed = stream(url(project, "/events?after=1"), {"last-event-id": str(past[-2]["seq"])})
    wait_for(lambda: len(resumed.events()) == 1)
    assert resumed.events() == past[-1:]

    nothing_new = stream(url(project, f"/events?after={past[-1]['seq']}"))
    wait_for(lambda: nothing_new.states())
    assert nothing_new.events() == []
    project.emit("agent.activity", "One more.")
    wait_for(lambda: len(nothing_new.events()) == 1)
    assert nothing_new.events()[0]["summary"] == "One more."


def test_a_line_cut_off_by_a_writer_that_died_does_not_hide_the_next_event(stream, project):
    last = project.events()[-1]["seq"]
    with (project.root / "events.jsonl").open("ab") as handle:
        handle.write(b'{"seq": %d, "ts": "2026-11-15T12:00:00Z", "type": "agent.act' % (last + 1))
    live = stream(url(project, f"/events?after={last}"))
    wait_for(lambda: live.states())
    # The next writer starts a new line and, having found no whole event after the last
    # one, gives its event the same seq as the piece.
    project.emit("agent.activity", "The event after the cut.")
    wait_for(lambda: live.events(), message="the event after the cut")
    assert [(event["seq"], event["summary"]) for event in live.events()] == [
        (last + 1, "The event after the cut.")
    ]
    assert all(json.loads(frame["data"]) for frame in live.frames() if "id" in frame)


def test_the_stream_reports_state_changes(stream, project):
    live = stream(url(project, f"/events?after={project.events()[-1]['seq']}"))
    wait_for(lambda: live.states(), message="the first state")
    first = live.states()[0]
    assert first["state"]["status"] == "idle" and first["worker"]["state"] == "idle"
    assert all("id" not in frame for frame in live.frames() if frame.get("event") == "state")

    state = {"status": "checkpoint", "round": 1, "phase": "research", "next_phase": "hypothesize",
             "checkpoint": {"round": 1, "phase": "research", "summary": "12 sources."}}  # fmt: skip
    # As the engine writes it: in one step.  A file written in place is empty for a moment,
    # and a look in that moment would report a state of its own.
    written = project.root / "state.json.tmp"
    written.write_text(json.dumps(state))
    os.replace(written, project.root / "state.json")
    wait_for(lambda: len(live.states()) == 2, message="the changed state")
    second = live.states()[1]["state"]
    assert second["status"] == "checkpoint" and second["checkpoint"]["summary"] == "12 sources."
    assert second["spent"]["model_usd"] == 0.0  # every key of the state is there
    time.sleep(0.1)
    assert len(live.states()) == 2  # only when it changes


def test_the_stream_follows_a_worker_from_start_to_stop(stream, client, project):
    live = stream(url(project, "/events"))
    wait_for(lambda: live.states())
    client.post(url(project, "/advance"))
    wait_for(lambda: live.states()[-1]["worker"]["state"] == "running", message="running")
    wait_for(lambda: live.states()[-1]["state"]["status"] == "running")
    wait_for(lambda: (project.root / "ready").exists())
    client.post(url(project, "/stop"))
    wait_for(
        lambda: (
            live.states()[-1]["worker"]["state"] == "idle"
            and live.states()[-1]["state"]["status"] == "stopped"
        ),
        message="the stopped state",
    )
    # ("stopping" is passed through too quickly here to count on seeing it.)
    seen = [state["worker"]["state"] for state in live.states()]
    assert seen[0] == "idle" and "running" in seen and seen[-1] == "idle"


def test_the_stream_notices_a_worker_that_died_without_a_word(stream, project):
    # An engine from a terminal that is gone: state.json says running, nobody is.
    state = {"status": "running", "round": 1, "phase": "design",
             "phases": [{"round": 1, "phase": "design", "status": "running"}],
             "worker": {"pid": 2**22 + 12345, "host": "", "heartbeat": "2020-01-01T00:00:00Z"}}  # fmt: skip
    (project.root / "state.json").write_text(json.dumps(state))
    live = stream(url(project, "/events"))
    wait_for(lambda: live.states() and live.states()[-1]["state"]["status"] == "stopped")
    settled = live.states()[-1]["state"]
    assert settled["phases"][0]["status"] == "interrupted" and "gone" in settled["error"]


def sleeper(project: Project, kind: str = "run") -> subprocess.Popen:
    """A process that only sleeps, with the command line of a worker of ``project``
    (``-m paper_orchestra research <kind> <directory>``).  The server goes by that before
    it shows a process as a project's worker: what the project's files say of a process
    number is not enough."""
    asleep = [sys.executable, "-c", "import time; time.sleep(120)"]
    return subprocess.Popen([*asleep, "-m", "paper_orchestra", "research", kind, str(project.root)])


def test_an_experiment_run_from_a_terminal_shows_as_the_projects_worker(client, project):
    # `research run` in a terminal: state.json names the worker, the status stays as it was.
    engine = sleeper(project)
    try:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state = {"status": "checkpoint", "round": 1, "phase": "design", "phases": [],
                 "worker": {"pid": engine.pid, "host": socket.gethostname(),
                            "started": now, "heartbeat": now}}  # fmt: skip
        (project.root / "state.json").write_text(json.dumps(state))
        assert client.get("/api/projects").json()["projects"][0]["worker"] == "running"
        worker = client.get(url(project)).json()["worker"]
        assert worker["state"] == "running" and worker["kind"] == "run"
        assert worker["pid"] == engine.pid and worker["adopted"] is True
        refused = client.post(url(project, "/advance"))
        assert refused.status_code == 409 and "an experiment" in error_of(refused)["message"]
    finally:
        engine.kill()
        engine.wait()


def test_the_stream_notices_the_end_of_a_run_it_did_not_start(stream, project):
    # A run started by a server that was killed: on record in .worker.json, and nothing
    # in state.json says a phase is running.  Nobody but the stream is looking.
    run = sleeper(project)
    try:
        record = {"pid": run.pid, "host": socket.gethostname(), "kind": "run",
                  "options": {"experiments": ["E1"]}, "command": [], "started": "",
                  "start_token": workers_module._start_token(run.pid),
                  "server_pid": 2**22 + 4321}  # fmt: skip
        (project.root / ".worker.json").write_text(json.dumps(record))
        live = stream(url(project, "/events"))
        wait_for(lambda: live.states(), message="the first state")
        first = live.states()[0]["worker"]
        assert first["state"] == "running" and first["kind"] == "run" and first["adopted"] is True
    finally:
        run.kill()
        run.wait()
    wait_for(lambda: live.states()[-1]["worker"]["state"] == "idle", message="the end of the run")
    assert not (project.root / ".worker.json").exists()


def test_a_quiet_stream_sends_a_heartbeat_comment(stream, project):
    live = stream(url(project, f"/events?after={project.events()[-1]['seq']}"))

    def heartbeats() -> int:
        return live.frames().count({"comment": "keep-alive"})

    began = time.monotonic()
    wait_for(lambda: heartbeats() >= 3, message="three heartbeats")
    # One per 0.15 s of silence here, not one per look at the files (every 0.01 s).
    assert heartbeats() <= (time.monotonic() - began) / 0.15 + 1


def test_the_stream_ends_when_the_client_hangs_up_or_the_server_closes(stream, app, project):
    first, second = stream(url(project, "/events")), stream(url(project, "/events"))
    wait_for(lambda: first.events() and second.events())
    first.close()
    assert first.ended.is_set() and not second.ended.is_set()
    project.emit("agent.activity", "Still listening.")
    wait_for(lambda: second.events()[-1]["summary"] == "Still listening.")
    app.state.context.live.closing.set()
    assert second.ended.wait(5), "the stream did not end when the server closed"
    assert second.error is None


def test_a_client_that_does_not_read_holds_back_nobody(app, project):
    for number in range(300):
        project.emit("agent.activity", f"Event {number} " + "x" * 200)
    total = len(project.events())

    class Stalled(Stream):
        """Takes the first frames, then stops taking anything."""

        async def _request(self) -> None:
            stuck = asyncio.Event()
            sent = 0
            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1", "method": "GET", "scheme": "http", "path": self.path,
                "raw_path": self.path.encode(), "query_string": b"", "root_path": "",
                "headers": [(b"host", b"127.0.0.1:8600")],
                "client": ("127.0.0.1", 50001), "server": ("127.0.0.1", 8600),
            }  # fmt: skip

            async def receive():
                while not self._hang_up.is_set():
                    await asyncio.sleep(0.005)
                return {"type": "http.disconnect"}

            async def send(message):
                nonlocal sent
                sent += 1
                if sent > 5:
                    await stuck.wait()

            await self.app(scope, receive, send)

    stalled = Stalled(app, url(project, "/events"))
    reader = Stream(app, url(project, "/events"))
    try:
        wait_for(lambda: len(reader.events()) == total, message="the reader to get everything")
        project.emit("agent.activity", "After the stall.")
        wait_for(lambda: reader.events()[-1]["summary"] == "After the stall.")
        assert not stalled.ended.is_set()
    finally:
        reader.close()
        stalled.close()


def test_the_stream_is_refused_like_any_request(stream, project):
    missing = stream("/api/projects/p-000000000000/events")
    assert missing.ended.wait(5) and missing.status == 404
    rebound = stream(url(project, "/events"), {"host": "evil.example:8600"})
    assert rebound.ended.wait(5) and rebound.status == 403
    bad = stream(url(project, "/events?after=soon"))
    assert bad.ended.wait(5) and bad.status == 422


def test_tail_hands_out_complete_lines_only(tmp_path):
    path = tmp_path / "events.jsonl"
    tail = _Tail(path, chunk=64)
    assert tail.read() == []
    path.write_bytes(b'{"seq": 1}\n{"seq": 2}\n{"seq": 3')
    assert tail.read() == [b'{"seq": 1}', b'{"seq": 2}']
    assert tail.read() == []  # the third is still being written
    with path.open("ab") as handle:
        handle.write(b"}\n")
    assert tail.read() == [b'{"seq": 3}']
    # More than one piece: read in pieces, nothing lost.
    with path.open("ab") as handle:
        handle.write(b"".join(b'{"seq": %d}\n' % n for n in range(4, 40)))
    lines = []
    while piece := tail.read():
        lines += piece
    assert lines == [b'{"seq": %d}' % n for n in range(4, 40)]
    # A file that was replaced by a shorter one is read from its start.
    path.write_bytes(b'{"seq": 1}\n')
    assert tail.read() == [b'{"seq": 1}']
    # A line longer than a piece is skipped instead of stalling the stream for good.
    with path.open("ab") as handle:
        handle.write(b"x" * 200 + b'\n{"seq": 2}\n')
    lines = []
    for _ in range(6):
        lines += tail.read()
    assert lines[-1] == b'{"seq": 2}'


def test_event_lines_are_recognised_by_their_seq():
    assert api_module._seq(b'{"seq": 41, "ts": "x"}') == 41
    assert api_module._seq(b'{"ts": "x", "seq": 7}') == 7
    assert api_module._seq(b'{"seq": "7"}') is None
    assert api_module._seq(b"not json") is None and api_module._seq(b"[1]") is None


# ---------------------------------------------------------------------------
# Who may talk to the server
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8600",
        "127.0.0.1.evil.example",
        "localhost.evil.example",
        "192.168.1.20:8600",
        "",
        "127.0.0.1:8600@evil.example",
    ],
)
def test_requests_for_another_host_are_refused(client, project, host):
    for path in ("/api/health", url(project), "/", "/js/app.js"):
        response = client.get(path, headers={"Host": host})
        assert response.status_code == 403, (host, path)
    refused = client.get("/api/health", headers={"Host": host})
    assert error_of(refused)["code"] == "bad_host"
    assert refused.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert client.get("/", headers={"Host": host}).headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1:8600", "127.0.0.1", "localhost:8600", "LOCALHOST", "[::1]:8600", "localhost."],
)
def test_the_servers_own_names_are_accepted(client, host):
    assert client.get("/api/health", headers={"Host": host}).status_code == 200


def test_a_bind_address_and_extra_names_are_accepted_too(workspace, tmp_path):
    app = create_app(
        workspace, host="192.168.1.20", token="t", allowed_hosts=["research.lab"],
        static_dir=tmp_path / "none",
    )  # fmt: skip
    with TestClient(app, base_url="http://192.168.1.20:8600") as client:
        allowed = {"Authorization": "Bearer t"}
        assert client.get("/api/health", headers=allowed).status_code == 200
        assert (
            client.get("/api/health", headers={**allowed, "Host": "research.lab"}).status_code
            == 200
        )
        assert (
            client.get("/api/health", headers={**allowed, "Host": "other.lab"}).status_code == 403
        )


@pytest.mark.parametrize(
    ("headers", "allowed"),
    [
        ({}, True),  # not a browser
        ({"Origin": "http://127.0.0.1:8600"}, True),
        ({"Origin": "http://127.0.0.1:8600", "Sec-Fetch-Site": "same-origin"}, True),
        ({"Sec-Fetch-Site": "same-origin"}, True),
        ({"Sec-Fetch-Site": "none"}, True),
        ({"Origin": "http://evil.example"}, False),
        ({"Origin": "https://127.0.0.1:8600.evil.example"}, False),
        ({"Origin": "http://127.0.0.1:9999"}, False),  # another local app
        ({"Origin": "http://localhost:8600"}, False),  # not the address this page was opened at
        ({"Origin": "null"}, False),
        ({"Origin": ""}, False),
        ({"Sec-Fetch-Site": "cross-site"}, False),
        ({"Sec-Fetch-Site": "same-site"}, False),
    ],
)
def test_changes_from_another_origin_are_refused(client, project, headers, allowed):
    response = client.post(url(project, "/requests"), json={"text": "From here?"}, headers=headers)
    assert response.status_code == (201 if allowed else 403), headers
    if not allowed:
        assert error_of(response)["code"] == "cross_origin"
        assert len(project.requests()) == 1
        for method in ("put", "patch", "delete"):
            other = getattr(client, method)(url(project, "/pages/H1"), headers=headers)
            assert other.status_code == 403, method


def test_reading_from_another_origin_gets_no_permission_to_see_the_answer(client, project):
    response = client.get(url(project), headers={"Origin": "http://evil.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert response.headers["cross-origin-resource-policy"] == "same-origin"


def test_every_response_carries_the_security_headers(client, project):
    responses = [
        client.get("/api/health"),
        client.get(url(project, "/pages")),
        client.get("/api/projects/p-nope"),
        client.post(url(project, "/review"), json={}),
        client.get("/"),
        client.get("/anything/else"),
        client.get("/api/nothing"),
    ]
    for response in responses:
        assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
    policy = CONTENT_SECURITY_POLICY
    assert "default-src 'none'" in policy and "script-src 'self'" in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "frame-ancestors 'none'" in policy and "base-uri 'none'" in policy
    assert responses[0].headers["cache-control"] == "no-store"


def test_off_loopback_a_token_is_required(workspace, project, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>PaperOrchestra</title>")
    app = create_app(workspace, host="0.0.0.0", token="s3cret-token", static_dir=static)
    with TestClient(app, base_url="http://192.168.1.20:8600", follow_redirects=False) as client:
        for path in ("/api/health", url(project), "/", url(project, "/files/inputs/brief.md")):
            refused = client.get(path)
            assert refused.status_code == 401, path
            assert "s3cret-token" not in refused.text
        assert error_of(client.get("/api/health"))["code"] == "token_required"
        assert "?token=" in client.get("/").text
        assert client.post(url(project, "/requests"), json={"text": "x"}).status_code == 401
        assert client.get("/?token=wrong").status_code == 401
        assert client.get("/api/health?token=s3cret-token").status_code == 401  # pages only
        assert (
            client.get("/api/health", headers={"Authorization": "Bearer nope"}).status_code == 401
        )

        entered = client.get("/?token=s3cret-token&view=library")
        assert entered.status_code == 303 and entered.headers["location"] == "/?view=library"
        cookie = entered.headers["set-cookie"]
        assert "po_token=s3cret-token" in cookie and "HttpOnly" in cookie
        assert "SameSite=Strict" in cookie
        # The client keeps the cookie, as a browser does.
        assert client.get("/").status_code == 200
        assert client.get("/api/health").status_code == 200
        assert client.get(url(project)).json()["id"] == project.id

    with TestClient(app, base_url="http://192.168.1.20:8600") as script:
        bearer = {"Authorization": "Bearer s3cret-token"}
        assert script.get("/api/health", headers=bearer).status_code == 200


@pytest.mark.parametrize(
    ("path", "location"),
    [
        ("//evil.example/", "/evil.example/"),
        ("/\\evil.example/", "/evil.example/"),
        ("/a\r\nSet-Cookie: x=y", "/a%0D%0ASet-Cookie%3A%20x%3Dy"),
        ("/p/p-1/wiki", "/p/p-1/wiki"),
    ],
)
def test_the_token_link_only_leads_back_into_the_app(workspace, tmp_path, path, location):
    # The address of the request comes back as the Location of the redirect: it must stay
    # a path of this server and one header line, whatever was asked for.
    app = create_app(workspace, host="0.0.0.0", token="s3cret-token", static_dir=tmp_path / "x")
    entered = Stream(app, f"{path}?token=s3cret-token", {"host": "192.168.1.20:8600"})
    assert entered.ended.wait(5) and entered.status == 303
    assert entered.response_headers["location"] == location


def test_loopback_is_told_from_names_that_only_look_like_it():
    for name in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.53", "LOCALHOST"):
        assert is_loopback(name), name
    for name in ("0.0.0.0", "192.168.1.20", "127.0.0.1.evil.example", "example.org", "", "::"):
        assert not is_loopback(name), name


# ---------------------------------------------------------------------------
# The frontend's files
# ---------------------------------------------------------------------------


def test_without_the_frontend_files_the_root_says_so(client):
    response = client.get("/")
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("text/plain")
    assert "not installed" in response.text and "/api/" in response.text
    assert client.get("/api/health").status_code == 200  # the API works all the same


@pytest.fixture
def with_frontend(workspace, tmp_path):
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "css").mkdir()
    (static / "index.html").write_text("<!doctype html><title>PaperOrchestra</title>")
    (static / "js" / "app.js").write_text("export const ready = true;\n")
    (static / "css" / "app.css").write_text("body { margin: 0 }\n")
    (static / "icon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (tmp_path / "secret.txt").write_text("not part of the frontend")
    (static / "leak.txt").symlink_to(tmp_path / "secret.txt")
    with TestClient(create_app(workspace, static_dir=static), base_url=BASE_URL) as client:
        yield client


def test_the_frontend_is_served_with_the_right_types(with_frontend):
    client = with_frontend
    index = client.get("/")
    assert index.status_code == 200 and "<title>PaperOrchestra</title>" in index.text
    assert index.headers["content-type"] == "text/html; charset=utf-8"
    assert index.headers["cache-control"] == "no-store"
    script = client.get("/js/app.js")
    assert script.headers["content-type"] == "text/javascript; charset=utf-8"
    assert script.headers["cache-control"] == "no-cache" and script.headers["etag"]
    assert client.get("/css/app.css").headers["content-type"] == "text/css; charset=utf-8"
    assert client.get("/icon.svg").headers["content-type"] == "image/svg+xml"
    assert client.head("/js/app.js").status_code == 200

    unchanged = client.get("/js/app.js", headers={"If-None-Match": script.headers["etag"]})
    assert unchanged.status_code == 304 and unchanged.content == b""
    assert client.get("/", headers={"If-None-Match": index.headers["etag"]}).status_code == 200


def test_client_side_routes_get_the_page_and_missing_files_do_not(with_frontend):
    client = with_frontend
    for route in ("/library", "/p/p-123/wiki", "/some/deep/route"):
        response = client.get(route)
        assert response.status_code == 200 and "<title>PaperOrchestra</title>" in response.text
        assert response.headers["cache-control"] == "no-store"
    for missing in ("/js/missing.js", "/css/nope.css", "/favicon.ico"):
        response = client.get(missing)
        assert response.status_code == 404 and "<title>" not in response.text
    for hostile in (
        "/%2e%2e/secret.txt",
        "/js/%2e%2e/%2e%2e/secret.txt",
        "/leak.txt",
        "/.git/config",
    ):
        response = client.get(hostile)
        assert "not part of the frontend" not in response.text, hostile
    unknown = client.get("/api/unknown/route")
    assert unknown.status_code == 404 and error_of(unknown)["code"] == "not_found"
    assert client.post("/library").status_code == 405


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_errors_have_one_shape(client, project):
    cases = [
        (client.get("/api/projects/p-nope"), 404, "not_found"),
        (client.get(url(project, "/pages/H9")), 404, "not_found"),
        (client.post(url(project, "/pages"), json={"kind": "poem", "title": "x"}), 422, "invalid"),
        (client.post(url(project, "/pages"), json={"title": 3}), 422, "invalid_request"),
        (client.get(url(project, "/activity?limit=many")), 422, "invalid_request"),
        (client.delete(url(project)), 405, "method_not_allowed"),
        (client.post(url(project, "/advance"), json={"phase": "x"}), 422, "invalid"),
    ]
    for response, status, code in cases:
        assert response.status_code == status, response.text
        assert response.headers["content-type"] == "application/json"
        assert error_of(response)["code"] == code
        assert error_of(response)["message"]


def test_an_unexpected_error_does_not_leak_its_text(app, project, services):
    def explode(project, page):
        raise RuntimeError("token=super-secret-value in /home/someone/.config")

    services.options = explode
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as client:
        response = client.get(url(project, "/experiments/E1"))
    assert response.status_code == 500
    assert error_of(response)["code"] == "internal"
    assert "super-secret-value" not in response.text and "RuntimeError" in response.text
    # This answer is written outside the security layer; it carries the headers all the same.
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_the_api_works_without_the_experiments_module(workspace, project, monkeypatch, tmp_path):
    """Until ``research.experiments`` is installed, what does not need it still works and
    what does says so."""
    real = api_module.importlib.import_module

    def without_experiments(name, *args, **kwargs):
        if name == "paper_orchestra.research.experiments":
            raise ImportError("No module named 'paper_orchestra.research.experiments'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(api_module.importlib, "import_module", without_experiments)
    with TestClient(
        create_app(workspace, static_dir=tmp_path / "none"), base_url=BASE_URL
    ) as client:
        assert client.get(url(project, "/pages/H1")).status_code == 200
        # Nothing of a plan can be cancelled, which nobody has to be asked about ...
        nothing = client.post(url(project, "/experiments/E1/cancel"))
        assert nothing.status_code == 200 and nothing.json()["outcome"] == "nothing"
        # ... and taking back an approval is the experiments module's to do.
        project.review("E1", "accept")
        project.set_state("E1", "approved", by="human")
        refused = client.post(url(project, "/experiments/E1/cancel"))
        assert refused.status_code == 400 and error_of(refused)["code"] == "not_available"
        # Where the project stands is still said: the engine is not what is missing.
        assert client.get(url(project)).json()["state_source"] == "unknown"


# ---------------------------------------------------------------------------
# What runs is what the person approved (audit: containment 3, 4, 7; execution 1, 2, 7)
# ---------------------------------------------------------------------------

TOKEN = "ghp_FAKETOKEN123"
PRIVATE_URL = f"https://{TOKEN}@git.example.invalid/me/private.git"
SHOWN_URL = "https://***@git.example.invalid/me/private.git"


@pytest.fixture
def real(workspace, launched, live, tmp_path):
    """The API over the real experiments module, not its stand-in."""
    app = create_app(
        workspace,
        launcher=launched,
        services=Services(),
        live=live,
        grace=2.0,
        static_dir=tmp_path / "no-frontend",
        stop_workers_on_exit=True,
    )
    with TestClient(app, base_url=BASE_URL) as client:
        yield client


def second_experiment(project: Project, **fields: Any) -> ResearchPage:
    page = project.create_page(
        "experiment",
        "A second plan",
        "## Objective\n\nMeasure it again.\n",
        author="agent:designer",
        fields={"mode": "script", **fields},
        round=1,
    )
    code = project.experiment_dir(str(page.id)) / "code"
    code.mkdir(parents=True)
    (code / "main.py").write_text("print('hello')\n")
    return page


def test_the_api_never_grants_network_access_from_the_plan(real, project):
    """Whichever way an approval arrives, it states the network or runs without it: a
    plan that asks for the network is not approved by saying nothing."""
    project.update_page("E1", author="agent:designer", fields={"network": True})

    body = {"items": [{"ref": "E1", "decision": "accept", "approve_run": True}]}
    (result,) = real.post(url(project, "/review"), json=body).json()["results"]
    assert result["ok"] is False and result["approved"] is False and result["status"] == 422
    assert "The plan was accepted, but its run was not approved" in result["error"]["message"]
    assert "yours to grant" in result["error"]["message"]
    assert (project.page("E1").review, project.page("E1").state) == ("accepted", "planned")

    silent = real.post(url(project, "/experiments/E1/approve"), json={})
    assert silent.status_code == 422 and "yours to grant" in error_of(silent)["message"]
    assert project.page("E1").state == "planned"

    without = real.post(url(project, "/experiments/E1/approve"), json={"network": False})
    assert without.status_code == 200, without.text
    page = without.json()["page"]
    assert page["state"] == "approved" and page["meta"]["network"] is False
    assert page["meta"]["backend"] == "local" and len(page["meta"]["approved_digest"]) == 64

    real.post(url(project, "/experiments/E1/cancel"))
    granted = real.post(url(project, "/experiments/E1/approve"), json={"network": True})
    assert granted.status_code == 200 and granted.json()["page"]["meta"]["network"] is True

    # A plan that does not ask is approved without the network by "accept and run".
    second_experiment(project)
    body = {"items": [{"ref": "E2", "decision": "accept", "approve_run": True}]}
    (result,) = real.post(url(project, "/review"), json=body).json()["results"]
    assert result["ok"] is True and result["approved"] is True
    assert result["page"]["meta"]["network"] is False


def test_an_experiment_is_approved_through_its_approval_and_nothing_else(real, project):
    project.review("E1", "accept")
    edited = real.put(url(project, "/pages/E1"), json={"fields": {"state": "approved"}})
    assert edited.status_code == 422
    message = error_of(edited)["message"]
    assert "research approve" in message and "Approve in the app" in message
    assert project.page("E1").state == "planned"

    second_experiment(project, network=True)
    body = {
        "items": [{"ref": "E2", "decision": "accept", "edits": {"fields": {"state": "approved"}}}]
    }
    (result,) = real.post(url(project, "/review"), json=body).json()["results"]
    assert result["ok"] is False and result["status"] == 422
    assert "research approve" in result["error"]["message"]
    page = project.page("E2")
    assert (page.review, page.state) == ("proposed", "planned")

    created = real.post(
        url(project, "/pages"),
        json={"kind": "experiment", "title": "Mine", "body": "x", "fields": {"state": "approved"}},
    )
    assert created.status_code == 422 and "research approve" in error_of(created)["message"]


def test_an_approval_without_a_backend_goes_to_the_projects_not_the_pages(real, project):
    """The project is set to Modal (code the person would not run on their machine); a
    page that carries ``backend: local`` does not move the run onto it."""
    project.settings.update({"compute": "modal"})
    project.save_settings()
    project.update_page("E1", author="human", fields={"backend": "local"})
    project.review("E1", "accept")

    silent = real.post(url(project, "/experiments/E1/approve"), json={})

    # Modal is not set up in a test: the approval is refused for Modal, not made for local.
    assert silent.status_code == 422, silent.text
    assert "modal" in error_of(silent)["message"].lower()
    assert project.page("E1").state == "planned"
    chosen = real.post(url(project, "/experiments/E1/approve"), json={"backend": "local"})
    assert chosen.status_code == 200 and chosen.json()["page"]["meta"]["backend"] == "local"


def test_the_approval_form_is_offered_the_projects_backend_first(real, project):
    """A form offers the first backend that can run as its choice.  That must be what the
    person set for the project, not their own machine in place of a sandbox."""

    def offered() -> list[str]:
        view = real.get(url(project, "/experiments/E1")).json()
        return [option["backend"] for option in view["options"]]

    assert offered() == ["local", "modal"]
    project.settings.update({"compute": "modal"})
    project.save_settings()
    assert offered() == ["modal", "local"]


def test_the_experiment_view_says_what_was_approved_and_whether_it_still_is(real, project):
    never = real.get(url(project, "/experiments/E1")).json()
    assert never["approval"] == {
        "digest": None,
        "at": None,
        "matches": False,
        "changed": [],
        "repaired": False,
    }
    assert len(never["plan_digest"]) == 64 and never["plan_digest"] != never["code"]["digest"]

    project.review("E1", "accept")
    read = never["code"]["digest"]
    approved = real.post(url(project, "/experiments/E1/approve"), json={"code_digest": read})
    assert approved.status_code == 200, approved.text
    view = real.get(url(project, "/experiments/E1")).json()
    assert view["code"]["digest"] == read  # what an approval sends back, as before
    assert (
        view["approval"]["digest"] == view["plan_digest"] == view["page"]["meta"]["approved_digest"]
    )
    assert view["approval"]["at"] == view["page"]["meta"]["approved_at"]
    assert view["approval"]["matches"] is True and view["approval"]["changed"] == []

    # The person edits the code and the time limit afterwards: still "approved" on the
    # page, and the view says that it is no longer what was approved, and how.
    real.put(url(project, "/experiments/E1/code/main.py"), json={"text": "print('other')\n"})
    real.put(url(project, "/pages/E1"), json={"fields": {"timeout_seconds": 30}})
    view = real.get(url(project, "/experiments/E1")).json()
    assert view["page"]["state"] == "approved"
    assert view["approval"]["matches"] is False and view["approval"]["repaired"] is False
    assert view["plan_digest"] != view["approval"]["digest"]
    first, second = view["approval"]["changed"]
    assert first == "timeout_seconds: 900 when approved, now 30"
    assert second.startswith("main.py: changed after the approval (written by human at ")


def test_the_api_never_shows_the_token_of_the_repository_url(real, project):
    project.settings.update({"repo": {"path": PRIVATE_URL, "ref": "main"}})
    project.save_settings()
    project.update_page(
        "E1",
        author="agent:designer",
        fields={"mode": "repo", "command": "python train.py", "network": True},
    )

    seen = real.get(url(project)).json()
    assert seen["settings"]["repo"] == {"path": SHOWN_URL, "ref": "main"}
    view = real.get(url(project, "/experiments/E1"))
    assert view.json()["plan"]["repo"] == SHOWN_URL
    listed = real.get(url(project, "/experiments"))
    for response in (real.get(url(project)), view, listed, real.get("/api/projects")):
        assert TOKEN not in response.text

    # Settings sent back as they were shown do not replace the token by its mask ...
    kept = real.patch(url(project), json={"settings": {"repo": {"path": SHOWN_URL, "ref": "dev"}}})
    assert kept.status_code == 200 and TOKEN not in kept.text
    assert kept.json()["settings"]["repo"] == {"path": SHOWN_URL, "ref": "dev"}
    stored = Project.open(project.root).settings.repo
    assert (stored.path, stored.ref) == (PRIVATE_URL, "dev")
    # ... and another repository is another repository.
    real.patch(url(project), json={"settings": {"repo": {"path": "https://example.org/x.git"}}})
    assert Project.open(project.root).settings.repo.path == "https://example.org/x.git"


def test_logs_are_served_without_the_credentials_a_job_printed(real, project, monkeypatch):
    secret = "sk-ant-test-" + "z" * 40
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    project.settings.update({"repo": {"path": PRIVATE_URL}})
    project.save_settings()
    run_dir = add_run(project)
    (run_dir / "log.txt").write_text(
        f"step 1\nfrom my parent: ANTHROPIC_API_KEY={secret}\norigin {PRIVATE_URL} (fetch)\ndone\n"
    )
    (project.root / "worker.log").write_text(f"cloning {PRIVATE_URL}\nlast line\n")

    run = real.get(url(project, "/experiments/E1/runs/1"))
    followed = real.get(url(project, "/experiments/E1/runs/1/log"))
    worker = real.get(url(project, "/worker/log"))

    for response in (run, followed, worker):
        assert response.status_code == 200
        assert secret not in response.text and TOKEN not in response.text
    assert "ANTHROPIC_API_KEY=[redacted]" in run.json()["log"]["text"]
    assert "origin https://[redacted]@git.example.invalid/me/private.git" in followed.json()["text"]
    assert followed.json()["text"].endswith("done\n")
    # Offsets are those of the file: a client that follows the log misses nothing.
    assert followed.json()["next"] == (run_dir / "log.txt").stat().st_size
    assert "cloning https://[redacted]@git.example.invalid" in worker.json()["text"]


# ---------------------------------------------------------------------------
# The second round of the audit: what the API answers, whatever a project's files hold
# (web 5, 6, 7, 8, 11, 12, 13; integrity 4, 10, 11; truth 12, 14)
# ---------------------------------------------------------------------------

FAKE_KEY = "sk-audit-FAKE-0123456789abcdefFAKE"


def send(client, method: str, path: str, body: Any):
    """A JSON request whose text is sent as written: escapes stay escapes, so half a
    surrogate pair arrives the way a browser sends it."""
    return client.request(
        method,
        path,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )


def append_event(project: Project, **fields: Any) -> dict[str, Any]:
    """An event as a worker of an earlier version wrote it: straight into the file."""
    last = project.events()[-1]
    event = {**last, "seq": last["seq"] + 1, "refs": [], "data": {}, **fields}
    with (project.root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return event


def library(client) -> dict[str, str]:
    """``{directory: id}`` as the library lists the projects."""
    return {row["dir"]: row["id"] for row in client.get("/api/projects").json()["projects"]}


def fresh_client(workspace: Path, tmp_path: Path, **options: Any) -> TestClient:
    """A server that sees the workspace for the first time."""
    app = create_app(
        workspace, static_dir=tmp_path / "no-frontend", stop_workers_on_exit=True, **options
    )
    return TestClient(app, base_url=BASE_URL)


def test_an_id_keeps_naming_the_directory_it_named(client, workspace, project):
    """A copy whose name sorts before the original does not take the id over: a tab that
    was open on the project goes on reading and writing that project."""
    import shutil

    assert client.get(url(project)).json()["dir"] == "warmup"
    shutil.copytree(project.root, workspace / "a-branch-to-try-something")
    ids = library(client)
    assert ids["warmup"] == project.id
    assert ids["a-branch-to-try-something"] not in (project.id, None)
    assert client.get(url(project)).json()["dir"] == "warmup"

    before = len(list((workspace / "a-branch-to-try-something" / "requests").glob("*")))
    posted = client.post(url(project, "/requests"), json={"text": "typed in the tab on warmup"})
    assert posted.status_code == 201
    assert len(list((project.root / "requests").glob("*"))) == before + 1
    assert len(list((workspace / "a-branch-to-try-something" / "requests").glob("*"))) == before

    # The copy's own address holds as well, and a second copy does not move either.
    shutil.copytree(project.root, workspace / "a-a-second-copy")
    again = library(client)
    assert again["warmup"] == project.id
    assert again["a-branch-to-try-something"] == ids["a-branch-to-try-something"]
    assert len(set(again.values())) == 3
    copy = client.get(f"/api/projects/{ids['a-branch-to-try-something']}").json()
    assert copy["dir"] == "a-branch-to-try-something" and copy["id"] != project.id


def test_a_server_that_sees_a_project_and_its_copy_for_the_first_time_knows_the_original(
    workspace, project, tmp_path
):
    import shutil

    time.sleep(0.05)  # a copy is made after the project, however little
    shutil.copytree(project.root, workspace / "a-backup-of-warmup")  # keeps the files' dates
    with fresh_client(workspace, tmp_path) as client:
        ids = library(client)
        assert ids["warmup"] == project.id and ids["a-backup-of-warmup"] != project.id
        assert client.get(url(project)).json()["dir"] == "warmup"


def test_the_original_is_known_by_what_the_project_recorded_when_it_was_created(
    workspace, project, tmp_path
):
    """The command line tells a project from its copy by ``Project.at_home`` (the directory
    it was created as); the server goes by the same, before the dates of files."""
    import shutil

    shutil.copytree(project.root, workspace / "a-backup-of-warmup")
    time.sleep(0.05)
    brief = project.root / "inputs" / "brief.md"
    brief.write_text(brief.read_text())  # the original's files are now the younger ones
    assert project.at_home() is True
    assert Project.open(workspace / "a-backup-of-warmup").at_home() is False
    with fresh_client(workspace, tmp_path) as client:
        ids = library(client)
        assert ids["warmup"] == project.id and ids["a-backup-of-warmup"] != project.id


def test_a_project_from_before_that_record_is_told_from_its_copy_by_its_files(
    workspace, project, tmp_path
):
    import shutil

    from paper_orchestra.research.project import HOME_KEY

    file = project.root / "project.json"
    data = json.loads(file.read_text())
    del data[HOME_KEY]  # as a project of an earlier version has it
    file.write_text(json.dumps(data))
    time.sleep(0.05)  # a copy is made after the project, however little
    shutil.copytree(project.root, workspace / "a-backup-of-warmup")  # keeps the files' dates
    assert Project.open(project.root).at_home() is None
    with fresh_client(workspace, tmp_path) as client:
        ids = library(client)
        assert ids["warmup"] == project.id and ids["a-backup-of-warmup"] != project.id


def test_a_renamed_project_directory_keeps_its_id(client, workspace, project):
    assert library(client) == {"warmup": project.id}
    project.root.rename(workspace / "warmup-2026")
    assert library(client) == {"warmup-2026": project.id}
    assert client.get(url(project)).json()["dir"] == "warmup-2026"


def test_a_project_that_cannot_be_opened_is_listed_with_the_reason(client, workspace, project):
    Project.create(workspace / "second", title="Second", brief="x")
    assert set(library(client)) == {"warmup", "second"}
    file = project.root / "project.json"
    text = file.read_text()
    file.write_text(text[: len(text) // 2])  # cut short

    listing = client.get("/api/projects").json()
    assert [row["dir"] for row in listing["projects"]] == ["second"]
    (unreadable,) = listing["unreadable"]
    assert unreadable["dir"] == "warmup" and unreadable["id"] == project.id
    assert "project.json" in unreadable["message"] and "Repair the file" in unreadable["message"]

    refused = client.get(url(project))
    assert refused.status_code == 422
    assert error_of(refused)["code"] == "project_unreadable"
    assert "Repair the file" in error_of(refused)["message"]
    assert client.get(url(project, "/pages")).status_code == 422
    assert client.get("/api/projects/p-000000000000").status_code == 404  # still "no such project"

    file.write_text(text)  # repaired: the same address works again
    assert client.get(url(project)).json()["dir"] == "warmup"
    assert client.get("/api/projects").json()["unreadable"] == []


def test_a_damaged_project_is_listed_by_a_server_that_never_saw_it_whole(
    workspace, project, tmp_path
):
    (project.root / "project.json").write_text("{ not json")
    with fresh_client(workspace, tmp_path) as client:
        listing = client.get("/api/projects").json()
        assert listing["projects"] == []
        (unreadable,) = listing["unreadable"]
        assert unreadable["dir"] == "warmup" and unreadable["id"] is None
        assert "project.json" in unreadable["message"]
    # A directory that holds no project at all is not one that cannot be opened.
    (workspace / "notes").mkdir()
    with fresh_client(workspace, tmp_path) as client:
        assert [row["dir"] for row in client.get("/api/projects").json()["unreadable"]] == [
            "warmup"
        ]


def test_a_project_whose_row_cannot_be_built_does_not_take_the_library_down(
    client, workspace, project, monkeypatch
):
    Project.create(workspace / "second", title="Second", brief="x")
    build = api_module._summary

    def broken(context, key, opened):
        if opened.root.name == "warmup":
            raise TypeError("a value nobody thought of")
        return build(context, key, opened)

    monkeypatch.setattr(api_module, "_summary", broken)
    listing = client.get("/api/projects").json()
    assert [row["dir"] for row in listing["projects"]] == ["second"]
    (unreadable,) = listing["unreadable"]
    assert unreadable["dir"] == "warmup" and unreadable["id"] == project.id
    assert "TypeError" in unreadable["message"] and "nobody thought of" not in unreadable["message"]


def test_text_in_a_projects_files_that_cannot_be_sent_does_not_break_an_answer(
    client, workspace, project
):
    """Half of a surrogate pair is valid in a JSON file and cannot be sent as UTF-8: one
    such title must not cost the whole library."""
    Project.create(workspace / "second", title="Second", brief="x")
    file = project.root / "project.json"
    data = json.loads(file.read_text())
    data["title"] = "Warm\ud83dup"
    file.write_text(json.dumps(data))  # written as an escape, as any JSON writer does
    append_event(project, type="error", summary="cut \udc00 here", data={"error": "\ud800"})

    listing = client.get("/api/projects")
    assert listing.status_code == 200
    titles = {row["dir"]: row["title"] for row in listing.json()["projects"]}
    assert titles == {"warmup": "Warm�up", "second": "Second"}
    assert client.get(url(project)).json()["title"] == "Warm�up"
    last = client.get(url(project, "/activity?limit=1")).json()["events"][0]
    assert last["summary"] == "cut � here" and last["data"] == {"error": "�"}


def credential_in_the_records(project: Project, secret: str) -> None:
    """What a worker of an earlier version left behind when a provider's error quoted
    the key: the text in state.json, in an event and in a run record."""
    error = f"401 Unauthorized for url 'https://api.example.invalid/v1/messages?key={secret}'"
    state = {
        "schema": 1, "status": "failed", "round": 1, "phase": "research",
        "next_phase": "research", "error": error,
        "phases": [{"round": 1, "phase": "research", "status": "failed", "error": error}],
        "checkpoint": {"round": 1, "phase": "research", "summary": f"It failed: {error}",
                       "pending": {}},
    }  # fmt: skip
    (project.root / "state.json").write_text(json.dumps(state))
    append_event(project, type="error", summary=f"The research phase failed: {error}",
                 data={"error": error, "nested": [{"deep": error}]})  # fmt: skip
    run_dir = add_run(project, status="failed")
    record = json.loads((run_dir / "run.json").read_text())
    record.update(error=error, notes=[f"The backend said: {error}"])
    (run_dir / "run.json").write_text(json.dumps(record))
    usage = {"total": {"usd": 1.0}, "last_error": error}
    (project.root / "usage.json").write_text(json.dumps(usage))


def test_a_credential_in_a_projects_records_is_blanked_in_every_answer(
    client, stream, project, monkeypatch
):
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", FAKE_KEY)
    credential_in_the_records(project, FAKE_KEY)
    answers = {
        "the project": client.get(url(project)),
        "the library": client.get("/api/projects"),
        "the activity": client.get(url(project, "/activity?limit=50")),
        "the experiments": client.get(url(project, "/experiments")),
        "the experiment": client.get(url(project, "/experiments/E1")),
        "the run": client.get(url(project, "/experiments/E1/runs/1")),
    }
    for name, response in answers.items():
        assert response.status_code == 200, name
        assert FAKE_KEY not in response.text, name
        assert "key=[redacted]" in response.text, name
    assert answers["the project"].json()["state"]["error"].endswith("?key=[redacted]'")
    assert answers["the library"].json()["projects"][0]["error"].endswith("?key=[redacted]'")

    live = stream(url(project, "/events"))
    wait_for(lambda: len(live.events()) == len(project.events()) and live.states())
    sent = live._data.decode()
    assert FAKE_KEY not in sent and sent.count("key=[redacted]") >= 4
    assert live.events()[-1]["data"]["nested"][0]["deep"].endswith("?key=[redacted]'")
    assert live.states()[0]["state"]["error"].endswith("?key=[redacted]'")
    # An event without a credential is passed on as it was written, byte for byte.
    first = (project.root / "events.jsonl").read_bytes().split(b"\n")[0]
    assert first in live._data


def test_the_token_of_the_repository_is_blanked_in_events_and_state_too(client, stream, project):
    project.settings.update({"repo": {"path": PRIVATE_URL}})
    project.save_settings()
    said = f"fatal: could not read from {PRIVATE_URL}"
    append_event(project, type="error", summary=said, data={"error": said})
    (project.root / "state.json").write_text(json.dumps({"status": "failed", "error": said}))
    for response in (
        client.get(url(project)),
        client.get("/api/projects"),
        client.get(url(project, "/activity")),
    ):
        assert response.status_code == 200 and TOKEN not in response.text
    live = stream(url(project, "/events"))
    wait_for(lambda: len(live.events()) == len(project.events()) and live.states())
    assert TOKEN not in live._data.decode()


def test_a_log_piece_that_starts_inside_a_credential_shows_none_of_it(client, project, monkeypatch):
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", FAKE_KEY)
    log = f"line one\nthe key {FAKE_KEY} was refused\nlast line\n".encode()
    at = log.find(FAKE_KEY.encode())
    (project.root / "worker.log").write_bytes(log)
    (add_run(project) / "log.txt").write_bytes(log)
    for path in ("/worker/log", "/experiments/E1/runs/1/log"):
        for inside in (1, 3, len(FAKE_KEY) // 2, len(FAKE_KEY) - 1):
            piece = client.get(url(project, f"{path}?offset={at + inside}")).json()
            assert piece["text"] == "[redacted] was refused\nlast line\n", (path, inside)
            # The offsets stay those of the file: a client that follows it misses nothing.
            assert (piece["offset"], piece["next"]) == (at + inside, len(log))
        whole = client.get(url(project, f"{path}?offset=0")).json()
        assert whole["text"] == "line one\nthe key [redacted] was refused\nlast line\n"
        after = client.get(url(project, f"{path}?offset={at + len(FAKE_KEY)}")).json()
        assert after["text"] == " was refused\nlast line\n"


def test_a_log_piece_that_ends_inside_a_credential_waits_for_the_rest(client, project, monkeypatch):
    """A log is read while it is written: the end of the file may be the first half of a
    key.  That half is not shown; it is shown, blanked, once the rest is there."""
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", FAKE_KEY)
    log = project.root / "worker.log"
    log.write_bytes(b"line one\nthe key " + FAKE_KEY[:20].encode())
    piece = client.get(url(project, "/worker/log?offset=0")).json()
    assert piece["text"] == "line one\nthe key " and piece["eof"] is True
    assert piece["next"] == len(b"line one\nthe key ")
    with log.open("ab") as handle:
        handle.write(FAKE_KEY[20:].encode() + b" was refused\n")
    rest = client.get(url(project, f"/worker/log?offset={piece['next']}")).json()
    assert rest["text"] == "[redacted] was refused\n" and rest["next"] == log.stat().st_size

    # Asking again before anything was added gives nothing, however often.
    assert client.get(url(project, f"/worker/log?offset={rest['next']}")).json()["text"] == ""

    # A key that ends the file is all there: nothing of it is waited for, and a client
    # that asks again from where it is gets nothing more (the key here is one whose
    # end reads like its own beginning, which must not look like a key just begun).
    repeating = "abcabcabcabcabcabc"
    monkeypatch.setenv("AUDIT_OTHER_TOKEN", repeating)
    log.write_bytes(b"the token " + repeating.encode())
    whole = client.get(url(project, "/worker/log?offset=0")).json()
    assert whole["text"] == "the token [redacted]" and whole["next"] == log.stat().st_size
    more = client.get(url(project, f"/worker/log?offset={whole['next']}")).json()
    assert more["text"] == "" and more["next"] == whole["next"] and more["eof"] is True

    # A piece cut at the size limit inside a key: neither half is shown by either piece.
    run_log = add_run(project) / "log.txt"
    filler = b"x" * (256 * 1024 - 10)
    run_log.write_bytes(filler + FAKE_KEY.encode() + b"\nend\n")
    first = client.get(url(project, "/experiments/E1/runs/1/log?offset=0")).json()
    second = client.get(url(project, f"/experiments/E1/runs/1/log?offset={first['next']}")).json()
    joined = first["text"] + second["text"]
    assert not any(FAKE_KEY[i : i + 6] in joined for i in range(len(FAKE_KEY) - 5))
    assert joined.endswith("[redacted]\nend\n") and second["next"] == run_log.stat().st_size


def test_logs_are_not_read_through_a_link_that_leads_out_of_the_project(client, project, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "credentials").write_text("OUTSIDE-SECRET (stands for ~/.aws/credentials)\n")
    (outside / "log.txt").write_text("OUTSIDE-SECRET in a directory elsewhere\n")
    run_dir = add_run(project)
    (run_dir / "log.txt").unlink()
    (run_dir / "log.txt").symlink_to(outside / "credentials")
    (project.root / "worker.log").symlink_to(outside / "credentials")
    (project.experiment_dir("E1") / "runs" / "2").symlink_to(outside, target_is_directory=True)

    for path in (
        "/experiments/E1/runs/1",
        "/experiments/E1/runs/1/log",
        "/experiments/E1/runs/1/log?offset=-100",
        "/experiments/E1/runs/2/log",
        "/worker/log",
        "/worker/log?offset=0",
    ):
        response = client.get(url(project, path))
        assert response.status_code == 200, path
        assert "OUTSIDE-SECRET" not in response.text, path
    piece = client.get(url(project, "/worker/log")).json()
    assert piece["text"] == "" and piece["eof"] is True and piece["size"] == 0
    assert "worker.log" in piece["problem"] and "link" in piece["problem"]
    assert "problem" in client.get(url(project, "/experiments/E1/runs/1")).json()["log"]

    # A link that stays inside the project names a file of the project.
    (project.root / "worker.log").unlink()
    (project.root / "inputs" / "kept.log").write_text("kept inside\n")
    (project.root / "worker.log").symlink_to(project.root / "inputs" / "kept.log")
    inside = client.get(url(project, "/worker/log")).json()
    assert inside["text"] == "kept inside\n" and "problem" not in inside


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_log_that_is_not_a_file_is_not_waited_for(client, project):
    os.mkfifo(project.root / "worker.log")
    piece = client.get(url(project, "/worker/log")).json()  # would wait for a writer for ever
    assert piece["text"] == "" and "not a file" in piece["problem"]


def test_a_change_that_could_not_be_saved_is_not_kept_in_memory(client, project, monkeypatch):
    def disk_full(self):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patched:
        patched.setattr(Project, "_write_project_file", disk_full)
        with pytest.raises(OSError):
            client.patch(
                url(project),
                json={"title": "Never saved", "settings": {"limits": {"model_usd": 99.0}}},
            )
    seen = client.get(url(project)).json()
    assert seen["title"] == "Warmup" and seen["settings"]["limits"]["model_usd"] != 99.0
    assert client.get("/api/projects").json()["projects"][0]["title"] == "Warmup"
    on_disk = json.loads((project.root / "project.json").read_text())
    assert on_disk["title"] == "Warmup"


LONE = "cut in the middle of an emoji \ud83d"


def test_text_that_cannot_be_stored_is_refused_wherever_it_is_sent(client, project, launched):
    cases = [
        ("PATCH", url(project), {"title": LONE}),
        ("PATCH", url(project), {"settings": {"research_cutoff": LONE}}),
        ("POST", "/api/render", {"markdown": LONE}),
        ("POST", "/api/render", {"markdown": "fine", "path": LONE}),
        ("POST", url(project, "/pages"), {"kind": "topic", "title": LONE}),
        ("POST", url(project, "/pages"), {"kind": "topic", "title": "x", "fields": {"n": [LONE]}}),
        ("POST", url(project, "/pages"), {"kind": "topic", "title": "x", "fields": {LONE: 1}}),
        ("PUT", url(project, "/pages/H1"), {"body": LONE}),
        ("PUT", url(project, "/pages/H1"), {"relations": {"motivated_by": [LONE]}}),
        ("POST", url(project, "/requests"), {"text": LONE}),
        ("POST", url(project, "/review"),
         {"items": [{"ref": "H1", "decision": "reject", "reason": LONE}]}),
        ("POST", url(project, "/review"),
         {"items": [{"ref": "H1", "decision": "accept", "edits": {"body": LONE}}]}),
        ("PUT", url(project, "/experiments/E1/code/main.py"), {"text": LONE}),
        ("POST", url(project, "/experiments/E1/approve"), {"gpu": LONE}),
        ("POST", url(project, "/advance"), {"instruction": LONE}),
        ("POST", url(project, "/paper"), {"template": LONE}),
    ]  # fmt: skip
    for method, path, body in cases:
        response = send(client, method, path, body)
        assert response.status_code == 422, (method, path, response.text)
        error = error_of(response)
        assert error["code"] == "invalid_request" and "cannot be stored" in error["message"], path
    form = client.post(
        "/api/projects", data={"brief": "x", "settings": json.dumps({"research_cutoff": LONE})}
    )
    assert form.status_code == 422 and "cannot be stored" in error_of(form)["message"]

    # Nothing was changed or started, and every view still answers.
    assert launched.calls == []
    assert client.get("/api/projects").json()["projects"][0]["title"] == "Warmup"
    assert client.get(url(project, "/pages/H1")).json()["page"]["review"] == "proposed"
    assert len(client.get(url(project, "/requests")).json()["requests"]) == 1
    assert (project.experiment_dir("E1") / "code" / "main.py").read_text() == "print('hello')\n"


def test_an_upload_is_reported_under_the_name_it_has_in_inputs(client, workspace):
    sent = {
        "brief.md": b"my own notes, called brief",
        "manifest.json": b'{"mine": true}',
        "Brief.MD": b"and one that differs by case only",
        "brief.md/inside.txt": b"a folder called like the brief",
        "data.csv": b"a,b\n1,2\n",
    }
    response = client.post(
        "/api/projects",
        data={"title": "Names", "brief": "What is in the files?"},
        files=[("files", (name, content, "text/plain")) for name, content in sent.items()],
    )
    assert response.status_code == 201, response.text
    inputs = workspace / "names" / "inputs"
    uploads = {entry["sent"]: entry["stored"] for entry in response.json()["uploads"]}
    assert set(uploads) == set(sent)
    for name, stored in uploads.items():
        assert (inputs / stored).read_bytes() == sent[name], (name, stored)
    assert uploads["data.csv"] == "data.csv" and uploads["brief.md"] != "brief.md"
    assert sorted(uploads.values()) == sorted(a["name"] for a in response.json()["attachments"])
    # The brief typed into the form and the manifest keep their names.
    assert (inputs / "brief.md").read_text() == "What is in the files?"
    assert "files" in json.loads((inputs / "manifest.json").read_text())


def worker_elsewhere(project: Project) -> None:
    """state.json as an engine on another machine that shares the directory writes it."""
    from datetime import UTC, datetime

    from paper_orchestra.research.model import format_stamp

    state = {
        "status": "running", "round": 1, "phase": "design",
        "phases": [{"round": 1, "phase": "design", "status": "running"}],
        "worker": {"pid": 4242, "host": "other-machine",
                   "heartbeat": format_stamp(datetime.now(UTC))},
    }  # fmt: skip
    (project.root / "state.json").write_text(json.dumps(state))


def test_a_worker_the_server_cannot_verify_is_shown_as_that_and_can_be_forgotten(
    client, project, launched
):
    worker_elsewhere(project)
    seen = client.get(url(project)).json()["worker"]
    assert seen["state"] == "running" and seen["verified"] is False and seen["pid"] is None
    assert "other-machine" in seen["note"]
    row = client.get("/api/projects").json()["projects"][0]
    assert row["worker"] == "running" and row["worker_verified"] is False

    stop = client.post(url(project, "/stop"))
    assert stop.status_code == 400 and error_of(stop)["code"] == "worker_failed"
    start = client.post(url(project, "/advance"))
    assert start.status_code == 409
    body = start.json()
    assert body["error"]["code"] == "worker_running" and body["worker"]["verified"] is False
    assert "Nothing was started" in body["error"]["message"] and launched.calls == []

    forgotten = client.post(url(project, "/worker/forget"))
    assert forgotten.status_code == 200, forgotten.text
    worker = forgotten.json()["worker"]
    assert worker["state"] == "idle" and worker["last_exit"]["forgotten"] is True
    after = client.get(url(project)).json()
    assert after["worker"]["state"] == "idle" and after["state"]["status"] == "stopped"
    row = client.get("/api/projects").json()["projects"][0]
    assert row["worker"] == "idle" and row["worker_verified"] is None
    assert client.post(url(project, "/worker/forget")).json()["worker"]["state"] == "idle"
    assert client.post(url(project, "/advance")).status_code == 202  # it can go on


def test_a_worker_this_server_started_is_stopped_not_forgotten(client, project):
    client.post(url(project, "/advance"))
    wait_for(lambda: (project.root / "ready").exists())
    row = client.get("/api/projects").json()["projects"][0]
    assert row["worker"] == "running" and row["worker_verified"] is True
    refused = client.post(url(project, "/worker/forget"))
    assert refused.status_code == 400 and error_of(refused)["code"] == "worker_failed"
    assert "stop it" in error_of(refused)["message"]
    assert client.get(url(project)).json()["worker"]["state"] == "running"


def test_a_resume_point_that_is_no_number_is_ignored(stream, project):
    past = project.events()
    for value in ("٣", "²", "9" * 5000, "", "-1", "1e3"):
        live = stream(url(project, "/events"), {"last-event-id": value})
        wait_for(lambda live=live: len(live.events()) == len(past), message=repr(value))
        assert live.status == 200
    resumed = stream(url(project, "/events"), {"last-event-id": str(past[-2]["seq"])})
    wait_for(lambda: len(resumed.events()) == 1)
    assert resumed.events()[0]["seq"] == past[-1]["seq"]


# ---------------------------------------------------------------------------
# What the app needs for the rules of the loop (FIXES I and J on the API's side)
# ---------------------------------------------------------------------------


def test_the_default_settings_are_served(client):
    body = client.get("/api/defaults").json()
    from paper_orchestra.research.model import ProjectSettings

    assert body["settings"] == ProjectSettings().to_dict()
    assert body["settings"]["compute"] == "local" and "model_usd" in body["settings"]["limits"]
    assert body["phases"] == ["intake", "research", "hypothesize", "design", "run", "analyze"]
    assert body["request_for"] == ["all", "scout", "theorist", "designer", "analyst"]
    # The lists a form offers are the lists the settings accept: a client has none of its own.
    assert body["efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert body["compute_backends"] == ["local", "modal"]
    for effort in body["efforts"]:
        given = {"models": {"effort": effort}}
        assert ProjectSettings.from_dict(given, strict=True).models.effort == effort
    for backend in body["compute_backends"]:
        assert ProjectSettings.from_dict({"compute": backend}, strict=True).compute == backend
    # ... and the words the engine's own sentences use for the phases.
    assert body["phase_words"]["hypothesize"] == "the hypothesis phase"
    assert set(body["phase_words"]) == set(body["phases"])
    # What a phase is called by itself ("checkpoint after ..."): one list, for the app and
    # the command line alike.
    assert body["phase_names"] == {
        "intake": "intake",
        "research": "research",
        "hypothesize": "hypotheses",
        "design": "experiment design",
        "run": "runs",
        "analyze": "analysis",
    }
    # The kinds of page the person can write, as ``POST .../pages`` takes them, and which
    # agents write each: a lesson is the person's alone, and a form has to offer it.
    assert body["creatable_kinds"] == [
        "question", "topic", "hypothesis", "experiment", "finding", "journal", "lesson",
    ]  # fmt: skip
    assert body["kind_writers"]["lesson"] == []
    assert body["kind_writers"]["journal"] == ["analyst"]
    assert body["kind_writers"]["topic"] == ["intake", "scout"]


def test_every_kind_the_defaults_offer_can_be_created_and_a_lesson_among_them(client, project):
    for kind in client.get("/api/defaults").json()["creatable_kinds"]:
        made = client.post(
            url(project, "/pages"),
            json={"kind": kind, "title": f"My {kind}", "body": "Mine.\n"},
        )
        assert made.status_code == 201, (kind, made.text)
        assert made.json()["page"]["author"] == "human" and made.json()["page"]["kind"] == kind
    assert project.pages(kind="lesson")[0].review == "accepted"


def test_a_decided_edit_is_listed_with_the_decision_and_its_reason(client, project):
    (pending,) = project.proposals()
    path = "/pages/wiki/topics/adaptive-optimisers.md"
    assert client.get(url(project, path)).json()["decided_proposals"] == []
    project.review(pending.id, "reject", reason="The source says otherwise")
    second = project.propose_edit(
        "topics/adaptive-optimisers", author="agent:analyst", summary="Name RAdam",
        body="Adam divides by a running average. RAdam rectifies it.\n\nSee [[idea]].\n",
    )  # fmt: skip
    project.review(second.id, "accept")

    # For the inbox's Decided list: every decided edit with the page it was about.
    decided = client.get(url(project, "/proposals"), params={"review": "rejected"}).json()
    (row,) = decided["proposals"]
    assert (row["id"], row["review"], row["decided_by"]) == (pending.id, "rejected", "human")
    assert row["decision_reason"] == "The source says otherwise" and row["decided_at"]
    assert row["page"]["title"] == "Adaptive optimisers" and row["summary"]
    everything = client.get(url(project, "/proposals"), params={"review": "all"}).json()
    assert [r["review"] for r in everything["proposals"]] == ["rejected", "accepted"]

    # On the page itself: what was suggested and what became of it, the latest first.
    view = client.get(url(project, path)).json()
    assert view["proposals"] == []
    first, earlier = view["decided_proposals"]
    assert (first["id"], first["review"], first["decision_reason"]) == (second.id, "accepted", None)
    assert (earlier["id"], earlier["review"]) == (pending.id, "rejected")
    assert earlier["decision_reason"] == "The source says otherwise"
    assert earlier["author"] == "agent:analyst" and earlier["summary"]
    assert earlier["changes_text"] is True and earlier["fields"] == {
        "summary": "How Adam scales its steps, precisely."
    }
    assert "body" not in earlier  # a list of what was decided, not the texts again


def test_an_edit_of_the_persons_leaves_the_time_of_the_proposal_as_it_was(client, project):
    path = "/pages/wiki/hypotheses/H1-warmup-reduces-early-variance.md"
    before = client.get(url(project, path)).json()["page"]
    time.sleep(1.1)  # stamps have a resolution of a second

    edited = client.put(
        url(project, path),
        json={"body": "## Statement\n\nSharper.\n", "base_updated": before["updated"]},
    ).json()["page"]

    # Still a proposal of the theorist's, proposed when it was proposed: ``created``.
    # ``updated`` and ``meta.updated_by`` say who touched it last and when.
    assert (edited["review"], edited["author"]) == ("proposed", "agent:theorist")
    assert edited["created"] == before["created"] and edited["updated"] > before["updated"]
    assert edited["meta"]["updated_by"] == "human"
    row = next(r for r in client.get(url(project, "/pages")).json()["pages"] if r["id"] == "H1")
    assert row["created"] == before["created"] and row["meta"]["updated_by"] == "human"


def test_a_snippet_of_a_search_is_text_without_the_marks_of_a_page(client, project):
    project.create_page(
        "topic",
        "Spikes",
        "| run | value |\n|---|---|\n| spike_b16_no_warmup | 0.121 |\n\n"
        "## What it means\n\nThe **spike** of [[topics/adaptive-optimisers|Adam]] is in "
        "[[H1]], `main.py` says.\n",
        author="human",
        summary="Where the loss spikes.",
    )

    table = client.get(url(project, "/search"), params={"q": "spike_b16_no_warmup"}).json()
    prose = client.get(url(project, "/pages"), params={"q": "Adam says"}).json()

    assert table["hits"][0]["snippet"] == "spike_b16_no_warmup · 0.121"
    snippets = [row["snippet"] for row in prose["pages"] if row["title"] == "Spikes"]
    assert snippets == ["The spike of Adam is in H1, main.py says."]


def test_a_request_can_be_addressed_to_one_role(client, project):
    posted = client.post(
        url(project, "/requests"),
        json={"text": "Use three seeds.", "for": "designer", "target": "E1"},
    )
    assert posted.status_code == 201, posted.text
    request = posted.json()["request"]
    assert request["for"] == "designer" and request["handled_in"] == []
    assert request["target"] == "wiki/experiments/E1-variance-with-and-without-warmup.md"
    plain = client.post(url(project, "/requests"), json={"text": "Read more."}).json()["request"]
    assert plain["for"] == "all"
    said = client.post(url(project, "/requests"), json={"text": "x", "for": "all"})
    assert said.json()["request"]["for"] == "all"
    listed = client.get(url(project, "/requests")).json()["requests"]
    assert [item["for"] for item in listed] == ["all", "designer", "all", "all"]

    refused = client.post(url(project, "/requests"), json={"text": "x", "for": "runner"})
    assert refused.status_code == 422 and "designer" in error_of(refused)["message"]
    assert len(client.get(url(project, "/requests")).json()["requests"]) == 4


class Reporting(FakeServices):
    """Services whose number report says what it was asked to check."""

    def __init__(self) -> None:
        super().__init__()
        self.reported: list[ResearchPage] = []

    def number_report(self, project, page):
        self.reported.append(page)
        return {
            "checked": [{"number": "97.3", "found_in": [], "in_basis": False, "kinds": []}],
            "skipped": ["5"], "elsewhere": [], "unverified": ["97.3"],
            "summary": f"1 number was looked for in {page.title!r}.",
        }  # fmt: skip


def a_finding(project: Project) -> ResearchPage:
    add_run(project)
    result = project.write_record(
        "result", "E1-run1", "E1 run 1", "| metric | value |\n|---|---|\n| accuracy | 0.91 |\n",
        fields={"state": "succeeded"}, relations={"run_of": ["E1"]},
    )  # fmt: skip
    finding = project.create_page(
        "finding",
        "Warmup helps",
        "Accuracy reached 0.91.\n",
        author="human",
        relations={"based_on": [result.path]},
    )
    return finding


def test_a_suggested_edit_to_a_finding_comes_with_the_numbers_of_the_text_it_would_make(
    workspace, project, launched, live, tmp_path
):
    services = Reporting()
    finding = a_finding(project)
    proposal = project.propose_edit(
        finding.path,
        author="agent:analyst",
        summary="Report the better run",
        body="Accuracy reached 97.3%, 41.8 points above the baseline.\n",
        fields={"title": "Warmup helps a lot"},
        round=1,
    )
    app = create_app(workspace, launcher=launched, services=services, live=live,
                     static_dir=tmp_path / "none", stop_workers_on_exit=True)  # fmt: skip
    with TestClient(app, base_url=BASE_URL) as client:
        view = client.get(url(project, f"/pages/{finding.id}")).json()
    (shown,) = view["proposals"]
    assert shown["id"] == proposal.id
    assert shown["number_report"]["unverified"] == ["97.3"]
    assert shown["number_report"]["summary"] == "1 number was looked for in 'Warmup helps a lot'."
    # What was checked for the proposal is the page as it would read with the edit applied:
    # its text and title, with the records the finding is based on.
    would_be = next(page for page in services.reported if page.title == "Warmup helps a lot")
    assert would_be.body == "Accuracy reached 97.3%, 41.8 points above the baseline.\n"
    assert would_be.relations["based_on"] == finding.relations["based_on"]
    assert would_be.path == finding.path and would_be.kind == "finding"
    # The page itself is reported as it reads today.
    assert view["number_report"]["summary"] == "1 number was looked for in 'Warmup helps'."
    assert view["page"]["body"] == "Accuracy reached 0.91.\n"


def test_the_numbers_of_a_suggested_edit_are_checked_by_the_real_check(real, project):
    finding = a_finding(project)
    project.propose_edit(
        finding.path,
        author="agent:analyst",
        summary="Report the better run",
        body="Accuracy reached 0.91 in one run and 97.31% in another.\n",
        round=1,
    )
    view = real.get(url(project, f"/pages/{finding.id}")).json()
    report = view["proposals"][0]["number_report"]
    assert report["unverified"] == ["97.31"]
    found = {check["number"]: check for check in report["checked"]}
    assert found["0.91"]["in_basis"] is True and found["97.31"]["found_in"] == []
    assert "97.31" not in json.dumps(view["number_report"])  # the page does not say it yet
    assert view["number_report"]["unverified"] == []
    assert [check["number"] for check in view["number_check"]] == ["0.91"]
    # An edit to a page that is no finding has no numbers to trace.
    topic = real.get(url(project, "/pages/topics/adaptive-optimisers")).json()
    assert "number_report" not in topic["proposals"][0] and "number_report" not in topic


def test_a_report_that_cannot_be_made_is_said_not_hidden(client, project, services):
    finding = a_finding(project)
    project.propose_edit(finding.path, author="agent:analyst", summary="s", body="x 97.3\n")

    def broken(project, page):
        raise ProjectError("The evidence could not be read.")

    services.number_report = broken
    view = client.get(url(project, f"/pages/{finding.id}")).json()
    assert view["number_report"] is None
    assert view["number_report_error"] == "The evidence could not be read."
    assert view["proposals"][0]["number_report"] is None
    assert view["proposals"][0]["number_report_error"] == "The evidence could not be read."


def test_a_record_the_person_edited_is_marked(real, project):
    finding = a_finding(project)
    result = finding.relations["based_on"][0]
    assert real.get(url(project, f"/pages/{result}")).json()["edited_record"] is False
    assert real.get(url(project, f"/pages/{finding.id}")).json()["edited_record"] is False
    edited = real.put(
        url(project, f"/pages/{result}"),
        json={"body": "| metric | value |\n|---|---|\n| accuracy | 0.99 |\n"},
    )
    assert edited.status_code == 200 and edited.json()["edited_record"] is True
    assert real.get(url(project, f"/pages/{result}")).json()["edited_record"] is True


def test_rejecting_an_experiment_says_what_rests_on_it(real, project):
    project.review("H1", "accept")
    project.review("E1", "accept")
    finding = a_finding(project)
    project.update_page(finding.path, author="human", relations={"supports": ["H1"]})
    project.set_state("H1", "testing", by="system")
    project.set_state("H1", "supported", by="human")

    answer = real.post(
        url(project, "/review"),
        json={"items": [
            {"ref": "E1", "decision": "reject", "reason": "The setup was wrong."},
            {"ref": "H1", "decision": "accept"},
        ]},
    ).json()  # fmt: skip
    rejected, other = answer["results"]
    assert rejected["ok"] is True and rejected["page"]["review"] == "rejected"
    rests = rejected["dependents"]
    assert rests["experiment"] == "E1" and rests["results"] == ["wiki/results/E1-run1.md"]
    assert [(f["id"], f["only"]) for f in rests["findings"]] == [(finding.id, True)]
    assert [(h["id"], h["state"], h["through"]) for h in rests["hypotheses"]] == [
        ("H1", "supported", [finding.id])
    ]
    assert "dependents" not in other  # only a rejected experiment has something to say
    # The experiment's own view carries the same, so that the list is there after a
    # reload, in another window and on another device, for as long as it is rejected.
    assert real.get(url(project, "/experiments/E1")).json()["dependents"] == rests
    project.review("E1", "accept")
    assert real.get(url(project, "/experiments/E1")).json()["dependents"] is None


def running_without_a_worker(project: Project, services: FakeServices) -> None:
    """E1 as a worker that was killed in the middle of its run left it."""
    project.review("E1", "accept")
    services.approve(project, "E1")
    project.set_state("E1", "queued", by="system")
    project.set_state("E1", "running", by="system")
    add_run(project, status="running")


def test_an_interrupted_run_is_shown_as_that_and_run_starts_it_again(
    client, project, services, launched
):
    running_without_a_worker(project, services)
    view = client.get(url(project, "/experiments/E1")).json()
    assert view["page"]["state"] == "running" and view["interrupted"] is True
    assert "interrupted" in view["interrupted_note"] and "Run it again" in view["interrupted_note"]
    (row,) = client.get(url(project, "/experiments")).json()["experiments"]
    assert row["interrupted"] is True

    started = client.post(url(project, "/experiments/E1/run"))
    assert started.status_code == 202, started.text
    assert started.json()["worker"]["kind"] == "run"
    assert launched.calls == [("run", {"experiments": ["E1"]})]
    wait_for(lambda: (project.root / "ready").exists())
    # With the worker at it, the run is not "interrupted", and a second start is refused.
    view = client.get(url(project, "/experiments/E1")).json()
    assert view["interrupted"] is False and view["interrupted_note"] is None
    assert client.get(url(project, "/experiments")).json()["experiments"][0]["interrupted"] is False
    again = client.post(url(project, "/experiments/E1/run"))
    assert again.status_code == 409 and again.json()["error"]["code"] == "worker_running"


def test_a_run_another_process_holds_is_neither_interrupted_nor_started(
    client, project, services, launched
):
    fcntl = pytest.importorskip("fcntl")
    running_without_a_worker(project, services)
    lock = project.experiment_dir("E1") / ".run.lock"
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # what whoever runs the experiment holds
        view = client.get(url(project, "/experiments/E1")).json()
        assert view["page"]["state"] == "running" and view["interrupted"] is False
        refused = client.post(url(project, "/experiments/E1/run"))
        assert refused.status_code == 409 and error_of(refused)["code"] == "conflict"
        assert "run lock" in error_of(refused)["message"] and launched.calls == []
    assert client.get(url(project, "/experiments/E1")).json()["interrupted"] is True


def test_an_experiment_that_is_not_running_is_never_interrupted(client, project):
    view = client.get(url(project, "/experiments/E1")).json()
    assert view["interrupted"] is False and view["interrupted_note"] is None
    planned = client.post(url(project, "/experiments/E1/run"))
    assert planned.status_code == 422 and "Approve it first" in error_of(planned)["message"]


def test_the_project_view_says_what_is_wrong_with_the_projects_files(client, project):
    assert client.get(url(project)).json()["problems"] == []
    (project.root / "wiki" / "hypotheses" / "H7-by-hand.md").write_text(
        "---\ntitle: By hand\ncites: [sources/nowhere]\n---\nText.\n"
    )
    problems = client.get(url(project)).json()["problems"]
    assert any("nowhere" in problem["message"] for problem in problems)
    assert all(set(problem) == {"path", "message"} for problem in problems)


def test_no_record_of_a_project_is_read_through_a_link_that_leads_out_of_it(
    client, stream, project, tmp_path
):
    """Siblings of the linked logs: the usage, the paper's state and the events are files
    of the project too, and a project directory may come from somebody else."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.json").write_text(json.dumps({"OUTSIDE-SECRET": "a file of the person"}))
    (outside / "private.jsonl").write_text(
        json.dumps({"seq": 1, "type": "note", "summary": "OUTSIDE-SECRET in a line"}) + "\n"
    )
    (project.root / "usage.json").symlink_to(outside / "private.json")
    (project.root / "paper").symlink_to(outside, target_is_directory=True)
    (outside / "state.json").write_text(json.dumps({"OUTSIDE-SECRET": "paper state"}))
    events = project.root / "events.jsonl"
    events.unlink()
    events.symlink_to(outside / "private.jsonl")

    for path in ("", "/paper", "/activity"):
        response = client.get(url(project, path))
        assert response.status_code == 200, path
        assert "OUTSIDE-SECRET" not in response.text, path
    assert client.get(url(project)).json()["usage"] is None
    assert client.get(url(project, "/paper")).json()["state"] is None
    assert client.get(url(project, "/activity")).json() == {"events": [], "has_more": False}
    live = stream(url(project, "/events"))
    wait_for(lambda: live.states())
    time.sleep(0.1)  # a few looks at the file
    assert live.events() == [] and b"OUTSIDE-SECRET" not in live._data


def test_a_text_file_is_served_without_the_credentials_in_it(client, project, monkeypatch):
    """The whole log of a run, and what a job wrote into an artifact, are opened as files:
    a credential in them is blanked there as in every other answer."""
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", FAKE_KEY)
    run_dir = add_run(project)
    big = b"x" * (1 << 20)  # the key lies across the pieces the file is read in
    (run_dir / "log.txt").write_bytes(big[:-7] + FAKE_KEY.encode() + b"\nlast line\n")
    (run_dir / "artifacts" / "env.txt").write_text(f"AUDIT_FAKE_API_KEY={FAKE_KEY}\n")
    (run_dir / "artifacts" / "clean.txt").write_text("nothing to hide\n")
    (run_dir / "artifacts" / "model.bin").write_bytes(b"\x00" + FAKE_KEY.encode())

    log = client.get(url(project, "/files/experiments/E1/runs/1/log.txt"))
    assert log.status_code == 200 and FAKE_KEY.encode() not in log.content
    assert log.content == big[:-7] + b"[redacted]\nlast line\n"
    assert log.headers["content-type"] == "text/plain; charset=utf-8"
    assert log.headers["content-disposition"] == 'inline; filename="log.txt"'
    assert log.headers["content-security-policy"].endswith("sandbox")
    env = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/env.txt?download=1"))
    assert env.text == "AUDIT_FAKE_API_KEY=[redacted]\n"
    assert env.headers["content-disposition"] == 'attachment; filename="env.txt"'

    # A file without one is served as before, with what a browser needs to resume it.
    clean = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/clean.txt"))
    assert clean.text == "nothing to hide\n" and clean.headers["content-length"] == "16"
    part = client.get(
        url(project, "/files/experiments/E1/runs/1/artifacts/clean.txt"),
        headers={"Range": "bytes=0-6"},
    )
    assert part.status_code == 206 and part.text == "nothing"
    # What is not text is the job's own data and is handed over as it is.
    binary = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/model.bin"))
    assert binary.content == b"\x00" + FAKE_KEY.encode()


def test_the_stream_checks_the_resume_point_itself(workspace, project, live):
    """The security layer drops a ``Last-Event-ID`` that is no number; the endpoint does
    not rely on it (a superscript two passes ``isdigit`` and is no ``int``)."""
    from fastapi import FastAPI

    from paper_orchestra.web.workers import Workers

    bare = FastAPI()  # the API alone, without the layer in front of it
    bare.include_router(api_module.router)
    bare.state.context = api_module.Context(
        workspace=api_module.Workspace(workspace), workers=Workers(), live=live
    )
    past = project.events()
    opened = []
    try:
        # The header as it arrives: bytes, which the framework reads as Latin-1.
        for value in (b"\xb2", b"\xb9\xb2\xb3", b"9" * 30, b" 12abc"):
            one = Stream(bare, url(project, "/events"), {"last-event-id": value})
            opened.append(one)
            wait_for(lambda one=one: len(one.events()) == len(past), message=repr(value))
            assert one.status == 200
        resumed = Stream(bare, url(project, "/events"), {"last-event-id": b"2"})
        opened.append(resumed)
        wait_for(lambda: len(resumed.events()) == len(past) - 2)
    finally:
        live.closing.set()
        for made in opened:
            made._hang_up.set()
            made.ended.wait(10)


def test_however_a_log_is_cut_into_pieces_no_part_of_a_credential_is_shown(tmp_path):
    """Many logs, keys and piece sizes: followed from the start, a log reads as itself
    with every key blanked, whatever the pieces were."""
    import random

    rng = random.Random(20261116)
    path = tmp_path / "log.txt"
    for case in range(300):
        keys = ["".join(rng.choices("ABCDEFGH0123456789", k=rng.randint(12, 30)))
                for _ in range(rng.randint(1, 3))]  # fmt: skip
        if case % 5 == 0:
            keys.append(keys[0][:6] * 3)  # one whose end reads like its beginning
        parts = []
        for _ in range(rng.randint(1, 8)):
            parts.append("".join(rng.choices("abc é\n", k=rng.randint(0, 40))))
            parts.append(rng.choice(keys))
        parts.append("".join(rng.choices("abc\n", k=rng.randint(1, 20))))
        text = "".join(parts)
        path.write_bytes(text.encode())
        limit = rng.randint(4, 64)  # a piece holds a whole character at least
        shown, offset = [], 0
        with path.open("rb") as handle:
            for _ in range(10_000):
                piece = api_module._read_log(handle, offset, limit, secrets=keys)
                shown.append(piece["text"])
                assert piece["offset"] == offset, (case, piece)
                if piece["next"] == offset:
                    break
                offset = piece["next"]
        joined = "".join(shown)
        assert offset == len(text.encode()), (case, limit)
        # Nothing of any key: their letters occur nowhere else in these logs.
        assert not set(joined) & set("ABCDEFGH0123456789"), (case, limit, joined)
        expected = text
        for key in sorted(keys, key=len, reverse=True):
            expected = expected.replace(key, "")
        assert joined.replace("[redacted]", "") == expected, (case, limit)


def test_however_a_file_is_read_in_pieces_every_credential_in_it_is_blanked(tmp_path):
    import random

    rng = random.Random(7)
    path = tmp_path / "artifact.txt"
    for case in range(200):
        keys = sorted(
            ("".join(rng.choices("ABCDEFGH0123456789", k=rng.randint(12, 30))) for _ in range(3)),
            key=len,
            reverse=True,
        )
        parts = []
        for _ in range(rng.randint(0, 6)):
            parts += ["".join(rng.choices("abc \n", k=rng.randint(0, 40))), rng.choice(keys)]
        text = "".join(parts) + "".join(rng.choices("abc\n", k=rng.randint(0, 20)))
        path.write_text(text)
        needles = [key.encode() for key in keys]
        piece = rng.randint(1, 48)
        assert api_module._holds_any(path, needles, piece) is any(key in text for key in keys)
        served = b"".join(api_module._blanked(path, needles, piece)).decode()
        expected = text
        for key in keys:
            expected = expected.replace(key, "[redacted]")
        assert served == expected, (case, piece)


def test_a_refusal_is_as_presentable_as_an_answer(client, project, monkeypatch):
    """What an error message quotes comes from the same files: a credential in it is
    blanked, and half a surrogate pair does not turn the refusal into a 500."""
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", FAKE_KEY)

    def refuse(self, ref, **changes):
        raise ProjectError(f"The page quotes {FAKE_KEY} and a cut \ud83d character.")

    monkeypatch.setattr(Project, "update_page", refuse)
    refused = client.put(url(project, "/pages/H1"), json={"body": "x"})
    assert refused.status_code == 422
    assert error_of(refused)["message"] == "The page quotes [redacted] and a cut � character."
    several = client.post(
        url(project, "/review"),
        json={"items": [{"ref": "H1", "decision": "accept", "edits": {"body": "x"}}]},
    )
    assert several.status_code == 200 and FAKE_KEY not in several.text


def test_run_waits_until_what_a_dead_worker_left_is_settled(client, project, services, launched):
    """The first look after a worker died sets the settling of its runs off; a Run that
    arrives in that moment is decided on the page as it is once they are settled, not
    refused because the settling holds the experiment."""
    from datetime import UTC, datetime, timedelta

    from paper_orchestra.research.model import format_stamp

    fcntl = pytest.importorskip("fcntl")
    running_without_a_worker(project, services)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    state = {
        "status": "checkpoint", "round": 1, "phase": "design", "phases": [],
        "worker": {"pid": gone.pid, "host": socket.gethostname(),
                   "heartbeat": format_stamp(datetime.now(UTC) - timedelta(minutes=5))},
    }  # fmt: skip
    (project.root / "state.json").write_text(json.dumps(state))

    workers = client.app.state.context.workers
    settle, entered = workers._settle_runs, threading.Event()

    def slow(root):
        # As the recovery does, only slower: the experiment is held while it is settled.
        with (project.experiment_dir("E1") / ".run.lock").open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            entered.set()
            time.sleep(0.5)
        return settle(root)

    workers._settle_runs = slow
    assert client.get(url(project)).json()["worker"]["state"] == "idle"  # the look
    assert entered.wait(10)
    started = client.post(url(project, "/experiments/E1/run"))
    assert started.status_code == 202, started.text
    assert launched.calls == [("run", {"experiments": ["E1"]})]
    # The unfinished run was settled before the worker was started, not beside it.
    view = client.get(url(project, "/experiments/E1")).json()
    assert [(run["number"], run["status"]) for run in view["runs"]] == [(1, "failed")]
    assert "worker stopped" in view["runs"][0]["error"]
    assert view["page"]["state"] == "queued" and view["interrupted"] is False


def test_a_credential_is_blanked_before_text_is_rendered(client, project, monkeypatch):
    """Rendering escapes and marks up what it is given: a credential that holds such
    characters would stand in the HTML in a form a search for it does not find."""
    secret = "p&ss<w>*or*d-0123456789"
    monkeypatch.setenv("AUDIT_FAKE_PASSWORD", secret)
    said = f"The server answered that {secret} was refused.\n"
    project.update_page("topics/adaptive-optimisers", author="human", body=said)
    project.propose_edit(
        "topics/adaptive-optimisers",
        author="agent:analyst",
        summary="Say it again",
        body=said + f"\nAgain: {secret}.\n",
        round=1,
    )
    project.update_page("E1", author="agent:designer", body=f"## Objective\n\nUse {secret}.\n")
    stamp = project.history("topics/adaptive-optimisers")[0].stamp
    answers = [
        client.get(url(project, "/pages/topics/adaptive-optimisers")),
        client.get(url(project, f"/pages/topics/adaptive-optimisers/versions/{stamp}")),
        client.get(url(project, "/experiments/E1")),
    ]
    for response in answers:
        assert response.status_code == 200
        for piece in (secret, "p&amp;ss", "ss&lt;w&gt;", "<em>or</em>", "d-0123456789"):
            assert piece not in response.text, piece
    page = answers[0].json()
    assert "that [redacted] was refused" in page["html"]
    assert page["proposals"][-1]["marked_html"].count("[redacted]") == 2
    assert "[redacted]" in page["proposals"][-1]["diff_html"]
    assert "Use [redacted]." in answers[2].json()["html"]


# ---------------------------------------------------------------------------
# What the review of these repairs found
# ---------------------------------------------------------------------------


def test_a_piece_cut_by_its_size_shows_nothing_of_a_credential_still_being_written(tmp_path):
    """The file ends in the first part of a key (its writer is in the middle of the line),
    and the size of a piece cuts the read inside that part: the piece ends before it, as
    one that reaches the end of the file does."""
    key = "sk-audit-FAKE-0123456789abcdefFAKE"
    path = tmp_path / "log.txt"
    for written in (8, 20, len(key) - 1):
        for inside in (1, 5, written - 1):
            before = b"x" * 40 + b"\n" + b"y" * (59 - inside)
            path.write_bytes(before + key[:written].encode())
            with path.open("rb") as handle:
                first = api_module._read_log(handle, 0, 100, secrets=[key])
                assert first["text"] == before.decode(), (written, inside)
                assert first["next"] == len(before)
                # Nothing more can be read for now: a client that follows the log waits.
                assert first["eof"] is True
                again = api_module._read_log(handle, first["next"], 100, secrets=[key])
                assert again["text"] == "" and again["next"] == first["next"] and again["eof"]
            with path.open("ab") as handle:  # the writer finishes the line
                handle.write(key[written:].encode() + b" was refused\n")
            with path.open("rb") as handle:
                rest = api_module._read_log(handle, first["next"], 100, secrets=[key])
            assert rest["text"] == "[redacted] was refused\n", (written, inside)


def test_a_credential_is_blanked_as_json_writes_it_in_a_file_and_in_a_log(
    client, project, monkeypatch
):
    """A record is a JSON file: a password with a quote, a backslash or a letter beyond
    ASCII stands in it escaped.  The run's view blanks it; so does the same file
    served as a file, and a log a job printed JSON into."""
    secret = 'p"ss\\wörd-0123456789'
    monkeypatch.setenv("AUDIT_FAKE_PASSWORD", secret)
    run_dir = add_run(project, status="failed")
    record = json.loads((run_dir / "run.json").read_text())
    record["error"] = f"login failed for {secret}"
    for name, ascii_only in (("run.json", True), ("artifacts/as-utf8.json", False)):
        (run_dir / name).write_text(json.dumps(record, ensure_ascii=ascii_only), encoding="utf-8")
    (run_dir / "log.txt").write_text(
        f"plain: {secret}\n{json.dumps({'error': secret})}\n"
        f"{json.dumps({'error': secret}, ensure_ascii=False)}\nlast line\n",
        encoding="utf-8",
    )
    (project.root / "worker.log").write_text(json.dumps({"error": secret}) + "\n")

    shown = client.get(url(project, "/experiments/E1/runs/1")).json()
    assert shown["run"]["error"] == "login failed for [redacted]"
    assert shown["log"]["text"].count("[redacted]") == 3
    answers = [
        client.get(url(project, "/files/experiments/E1/runs/1/run.json")),
        client.get(url(project, "/files/experiments/E1/runs/1/artifacts/as-utf8.json")),
        client.get(url(project, "/files/experiments/E1/runs/1/log.txt")),
        client.get(url(project, "/experiments/E1/runs/1/log")),
        client.get(url(project, "/experiments/E1/runs/1/log?offset=-30")),
        client.get(url(project, "/worker/log")),
    ]
    for response in answers:
        assert response.status_code == 200
        for piece in ("0123456789", "ss\\\\w", "w\\u00f6rd", "wörd"):
            assert piece not in response.text, (response.url, piece)
        assert "[redacted]" in response.text or "last line" in response.text
    served = json.loads(answers[0].text)  # still the record, with the value blanked
    assert served["error"] == "login failed for [redacted]"
    assert answers[2].text.count("[redacted]") == 3
    # Every offset of the log: no piece shows part of the value, in either form.
    size = (run_dir / "log.txt").stat().st_size
    for offset in range(size):
        text = client.get(url(project, f"/experiments/E1/runs/1/log?offset={offset}")).json()
        assert "012345" not in text["text"] and "456789" not in text["text"], offset


def test_a_number_that_is_no_number_is_refused_wherever_it_is_sent(client, project):
    """``NaN`` and ``Infinity`` pass for numbers where the body is read and are none a
    file can hold: a limit of NaN would be no limit at all, and say nothing."""
    before = (project.root / "project.json").read_text()
    for method, path, raw in (
        ("PATCH", url(project), '{"settings": {"limits": {"model_usd": NaN}}}'),
        ("PATCH", url(project), '{"settings": {"limits": {"experiment_usd": Infinity}}}'),
        ("PATCH", url(project), '{"settings": {"limits": {"experiment_seconds": 1e999}}}'),
        ("PATCH", url(project), '{"settings": {"limits": {"model_usd": -Infinity}}}'),
        ("POST", url(project, "/experiments/E1/approve"), '{"timeout_seconds": NaN}'),
        ("POST", url(project, "/experiments/E1/approve"), '{"timeout_seconds": 1e999}'),
        ("POST", url(project, "/pages"),
         '{"kind": "topic", "title": "x", "fields": {"weights": [1.5, NaN]}}'),
        ("PUT", url(project, "/pages/H1"), '{"fields": {"confidence": Infinity}}'),
        ("POST", url(project, "/review"),
         '{"items": [{"ref": "E1", "decision": "accept", "approve_run": {"timeout_seconds": NaN}}]}'),
    ):  # fmt: skip
        response = client.request(
            method, path, content=raw.encode(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 422, (path, raw, response.text)
        error = error_of(response)
        assert error["code"] == "invalid_request", raw
        assert "cannot be stored" in error["message"] and "number" in error["message"], raw
    form = client.post(
        "/api/projects", data={"brief": "x", "settings": '{"limits": {"model_usd": NaN}}'}
    )
    assert form.status_code == 422 and "cannot be stored" in error_of(form)["message"]
    assert (project.root / "project.json").read_text() == before
    assert client.get(url(project, "/pages/E1")).json()["page"]["review"] == "proposed"
    assert len(client.get("/api/projects").json()["projects"]) == 1
    # An ordinary number is one, whatever its size.
    fine = client.patch(url(project), json={"settings": {"limits": {"model_usd": 1e6}}})
    assert fine.status_code == 200 and fine.json()["settings"]["limits"]["model_usd"] == 1e6


def outside_code(tmp_path: Path) -> Path:
    """A directory elsewhere that reads like the code of an experiment."""
    outside = tmp_path / "elsewhere"
    (outside / "code").mkdir(parents=True)
    (outside / "code" / "main.py").write_text("# OUTSIDE-SECRET (a file of the person)\n")
    return outside


@pytest.mark.parametrize("linked", ["code", "experiment", "experiments"])
def test_the_code_of_an_experiment_is_not_read_through_a_link_that_leads_out_of_the_project(
    real, project, tmp_path, linked
):
    """A sibling of the linked logs: the experiment's view shows its code, and a project
    directory that came from somebody else may have that directory linked to one of the
    person's.  Nothing of the experiment is shown, changed, approved or run then (the
    experiments module refuses; this holds the endpoints to it)."""
    import shutil

    project.review("E1", "accept")
    outside = outside_code(tmp_path)
    home = project.experiment_dir("E1")
    if linked == "code":
        shutil.rmtree(home / "code")
        (home / "code").symlink_to(outside / "code", target_is_directory=True)
    elif linked == "experiment":
        shutil.rmtree(home)
        home.symlink_to(outside, target_is_directory=True)
    else:
        (tmp_path / "all").mkdir()
        outside.rename(tmp_path / "all" / "E1")
        shutil.rmtree(project.root / "experiments")
        (project.root / "experiments").symlink_to(tmp_path / "all", target_is_directory=True)

    view = real.get(url(project, "/experiments/E1"))
    assert view.status_code == 422 and "OUTSIDE-SECRET" not in view.text
    message = error_of(view)["message"]
    assert "link" in message and "experiments/E1" in message and "Remove the link" in message
    attempts = [
        real.post(url(project, "/experiments/E1/approve"), json={"backend": "local"}),
        real.put(url(project, "/experiments/E1/code/main.py"), json={"text": "print(1)\n"}),
    ]
    for response in attempts:
        assert response.status_code == 422 and "OUTSIDE-SECRET" not in response.text
    for path in ("/experiments/E1/runs/1", "/experiments/E1/runs/1/log"):
        assert "OUTSIDE-SECRET" not in real.get(url(project, path)).text
    assert real.post(url(project, "/experiments/E1/run")).status_code == 422  # not approved
    decided = real.post(
        url(project, "/review"),
        json={"items": [{"ref": "E1", "decision": "accept", "approve_run": True}]},
    ).json()
    assert decided["ok"] is False and decided["results"][0]["status"] == 422
    assert "OUTSIDE-SECRET" not in json.dumps(decided)
    assert project.page("E1").state == "planned"
    code = (tmp_path / "all" / "E1" if linked == "experiments" else outside) / "code"
    assert (code / "main.py").read_text() == "# OUTSIDE-SECRET (a file of the person)\n"
    # The rest of the project is read as before, and the listing says nothing of the code.
    assert real.get(url(project)).status_code == 200
    listed = real.get(url(project, "/experiments"))
    assert listed.status_code == 200 and "OUTSIDE-SECRET" not in listed.text


@pytest.mark.parametrize("linked", ["version", "page", "history"])
def test_an_earlier_version_is_not_read_through_a_link_that_leads_out_of_the_project(
    client, project, tmp_path, linked
):
    import shutil

    project.update_page("topics/adaptive-optimisers", author="human", body="A second text.\n")
    version = project.history("topics/adaptive-optimisers")[0]
    kept = project.root / version.path
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    if linked == "version":
        (outside / "private.md").write_text("OUTSIDE-SECRET (a file of the person)\n")
        kept.unlink()
        kept.symlink_to(outside / "private.md")
    else:
        # The same names as in the project, in a directory of somebody else's making.
        place = kept.parent if linked == "page" else project.root / "history"
        (outside / kept.relative_to(place)).parent.mkdir(parents=True, exist_ok=True)
        (outside / kept.relative_to(place)).write_text("OUTSIDE-SECRET (a file of the person)\n")
        shutil.rmtree(place)
        place.symlink_to(outside, target_is_directory=True)

    for stamp in {version.stamp, *(v.stamp for v in project.history("topics/adaptive-optimisers"))}:
        response = client.get(url(project, f"/pages/topics/adaptive-optimisers/versions/{stamp}"))
        assert response.status_code in (404, 422), response.text
        assert "OUTSIDE-SECRET" not in response.text
    page = client.get(url(project, "/pages/topics/adaptive-optimisers"))
    assert page.status_code == 200 and "OUTSIDE-SECRET" not in page.text
    assert page.json()["page"]["body"] == "A second text.\n"


def test_a_version_kept_in_the_project_is_still_read(client, project):
    """The guard of the test above lets the project's own history through."""
    project.update_page("topics/adaptive-optimisers", author="human", body="A second text.\n")
    version = project.history("topics/adaptive-optimisers")[0]
    shown = client.get(url(project, f"/pages/topics/adaptive-optimisers/versions/{version.stamp}"))
    assert shown.status_code == 200
    assert shown.json()["body"].startswith("Adam divides by a running average.")
    current = project.page("topics/adaptive-optimisers").updated
    now = client.get(url(project, f"/pages/topics/adaptive-optimisers/versions/{current}"))
    assert now.status_code == 200 and now.json()["body"] == "A second text.\n"


def test_the_server_asks_again_about_the_places_the_store_refuses_as_links():
    """The store decides which of a project's own places may not be a link out of it;
    the server only has to know when to ask it again.  The two lists are one list."""
    from paper_orchestra.research import project as store

    assert set(api_module._OWN_PLACES) == set(store._OWN_ENTRIES)


@pytest.mark.parametrize("name", ["wiki", "proposals", "requests", "project.json"])
def test_a_project_whose_own_files_are_a_link_out_of_it_is_not_opened(
    client, workspace, project, tmp_path, name
):
    """A project directory with its wiki (or its proposals, its requests, its
    project.json) linked to a directory of the person's: the app would show what is
    there as the project's, and write there.  The store does not open such a project;
    neither does a server that had it open before the link was put there.  It is listed
    as a project that cannot be opened, with the reason."""
    import shutil

    Project.create(workspace / "second", title="Second", brief="x")
    assert set(library(client)) == {"warmup", "second"}
    outside = tmp_path / "elsewhere"
    shutil.copytree(project.root, outside)
    for page in (outside / "wiki").rglob("*.md"):
        page.write_text(page.read_text() + "\nOUTSIDE-SECRET (a file of the person)\n")
    for kept in [*(outside / "proposals").glob("*.json"), *(outside / "requests").glob("*.json")]:
        kept.write_text(kept.read_text().replace("RAdam", "OUTSIDE-SECRET").replace(
            "Say which average", "OUTSIDE-SECRET"))  # fmt: skip
    title = json.loads((outside / "project.json").read_text())
    (outside / "project.json").write_text(json.dumps({**title, "title": "OUTSIDE-SECRET"}))
    target = project.root / name
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    target.symlink_to(outside / name, target_is_directory=(outside / name).is_dir())

    listing = client.get("/api/projects")
    assert listing.status_code == 200 and "OUTSIDE-SECRET" not in listing.text
    assert [row["dir"] for row in listing.json()["projects"]] == ["second"]
    (unreadable,) = listing.json()["unreadable"]
    assert unreadable["dir"] == "warmup" and unreadable["id"] == project.id
    assert name in unreadable["message"] and "link" in unreadable["message"]
    for path in ("", "/pages", "/pages/wiki/idea.md", "/proposals", "/requests", "/graph"):
        refused = client.get(url(project, path))
        assert refused.status_code == 422 and "OUTSIDE-SECRET" not in refused.text, path
        assert error_of(refused)["code"] == "project_unreadable"
    written = client.post(url(project, "/pages"), json={"kind": "topic", "title": "Through it"})
    asked = client.post(url(project, "/requests"), json={"text": "Through it"})
    assert written.status_code == 422 and asked.status_code == 422
    assert not list(outside.rglob("through-it*"))
    assert "Through it" not in "".join(p.read_text() for p in (outside / "requests").glob("*"))

    # With the directory itself back in place the project is read again, under its id.
    target.unlink()
    if (outside / name).is_dir():
        shutil.copytree(outside / name, target)
    else:
        shutil.copy(outside / name, target)
    assert client.get(url(project)).status_code == 200
    assert client.get("/api/projects").json()["unreadable"] == []


def test_a_link_that_stays_inside_the_project_is_no_reason_to_refuse_it(client, project):
    """Only a link that leads out is one: a directory kept under another name inside the
    project directory is the project's own."""
    (project.root / "requests").rename(project.root / "requests-kept")
    (project.root / "requests").symlink_to("requests-kept", target_is_directory=True)
    assert client.get("/api/projects").json()["unreadable"] == []
    assert len(client.get(url(project, "/requests")).json()["requests"]) == 1


def test_the_original_takes_its_id_back_from_a_copy_put_under_its_name(client, workspace, project):
    """The directory is renamed and a copy of it put under the old name between two looks
    of the server.  The id goes on naming the directory it named, which is the renamed
    one, not whatever stands under its old name now: a tab that was open on the project
    writes into the project, also when its request is the first thing the server hears
    of the change."""
    import shutil

    assert library(client) == {"warmup": project.id}
    project.root.rename(workspace / "warmup-2026")
    shutil.copytree(workspace / "warmup-2026", workspace / "warmup")
    # The open tab asks first: nobody looked at the library in between.
    posted = client.post(url(project, "/requests"), json={"text": "typed in the open tab"})
    assert posted.status_code == 201
    assert len(list((workspace / "warmup-2026" / "requests").glob("*"))) == 2
    assert len(list((workspace / "warmup" / "requests").glob("*"))) == 1
    assert client.get(url(project)).json()["dir"] == "warmup-2026"
    ids = library(client)
    assert ids["warmup-2026"] == project.id
    assert ids["warmup"] not in (project.id, None)
    # And it stays so: the next looks change nothing, and the copy has its own address.
    assert library(client) == ids
    assert client.get(f"/api/projects/{ids['warmup']}").json()["dir"] == "warmup"


def test_a_copy_that_was_addressed_by_the_id_keeps_it_when_the_original_comes_back(
    client, workspace, project, tmp_path
):
    """The other way round: the server knew only the copy under the id (the original was
    kept elsewhere) and the original is moved in.  The tab that is open shows the copy
    and goes on writing into the copy; the newcomer gets the derived id."""
    import shutil

    project.root.rename(tmp_path / "kept-elsewhere")
    shutil.copytree(tmp_path / "kept-elsewhere", workspace / "working-copy")
    assert library(client) == {"working-copy": project.id}
    (tmp_path / "kept-elsewhere").rename(workspace / "a-original")
    posted = client.post(url(project, "/requests"), json={"text": "typed in the open tab"})
    assert posted.status_code == 201
    assert len(list((workspace / "working-copy" / "requests").glob("*"))) == 2
    ids = library(client)
    assert ids["working-copy"] == project.id and ids["a-original"] not in (project.id, None)


def test_an_unreadable_project_keeps_its_address_when_its_directory_is_renamed(
    client, workspace, project
):
    assert library(client) == {"warmup": project.id}
    file = project.root / "project.json"
    whole = file.read_text()
    file.write_text("{ cut short")
    project.root.rename(workspace / "warmup-2026")
    (unreadable,) = client.get("/api/projects").json()["unreadable"]
    assert unreadable["dir"] == "warmup-2026" and unreadable["id"] == project.id
    assert client.get(url(project)).status_code == 422
    (workspace / "warmup-2026" / "project.json").write_text(whole)
    assert client.get(url(project)).json()["dir"] == "warmup-2026"


def test_a_refusal_blanks_the_token_of_the_projects_repository_too(client, project):
    """A stale edit is answered with the page as it is now; a refused review with the
    page it was about.  Those hold the project's text like any view of the page."""
    project.settings.update({"repo": {"path": PRIVATE_URL}})
    project.save_settings()
    said = f"fatal: could not read from {PRIVATE_URL}\n"
    project.update_page("topics/adaptive-optimisers", author="human", body=said)
    assert TOKEN not in client.get(url(project, "/pages/topics/adaptive-optimisers")).text
    stale = client.put(
        url(project, "/pages/topics/adaptive-optimisers"),
        json={"body": "mine", "base_updated": "2020-01-01T00:00:00Z"},
    )
    assert stale.status_code == 409 and TOKEN not in stale.text
    assert "[redacted]" in stale.json()["page"]["body"]
    missing = client.get(url(project, f"/pages/{TOKEN}-is-no-page"))
    assert missing.status_code == 404 and TOKEN not in missing.text


def test_folders_that_differ_by_case_alone_are_one_folder_under_the_name_reported(
    client, workspace
):
    """Two folders picked from different places, ``Data`` and ``data``: the project keeps
    no two top-level names that differ by case alone and would rename one, so the
    answer would name files that are not there.  They are one folder, under the
    spelling it was first sent with, and every file is where the answer says."""
    sent = [
        ("Data/a.txt", b"the first folder"),
        ("data/b.txt", b"the second folder, another spelling"),
        ("DATA/a.txt", b"a third, with a file called like one of the first"),
        ("data", b"a file called like the folder"),
        ("Data/Deep/x.txt", b"deeper"),
        ("data/deep/y.txt", b"deeper, another spelling"),
    ]
    response = client.post(
        "/api/projects",
        data={"title": "Cases", "brief": "What is in the folders?"},
        files=[("files", (name, content, "text/plain")) for name, content in sent],
    )
    assert response.status_code == 201, response.text
    inputs = workspace / "cases" / "inputs"
    uploads = response.json()["uploads"]
    assert [entry["sent"] for entry in uploads] == [name for name, _ in sent]
    for entry, (_, content) in zip(uploads, sent, strict=True):
        assert (inputs / entry["stored"]).read_bytes() == content, entry
    stored = [entry["stored"] for entry in uploads]
    assert stored[:3] == ["Data/a.txt", "Data/b.txt", "Data/a-2.txt"]
    assert stored[3] == "data-2" and stored[4:] == ["Data/Deep/x.txt", "Data/Deep/y.txt"]
    assert sorted(stored) == sorted(a["name"] for a in response.json()["attachments"])
    assert response.json()["skipped"] == []


def test_the_paper_view_names_the_findings_a_paper_would_leave_out(real, project):
    """The store asked for it: before the write-up is started, the app can say which
    accepted findings would not be reported (every record they rest on belongs to an
    experiment that was rejected or is gone), in the words the write-up's refusal uses."""
    project.review("E1", "accept")
    finding = a_finding(project)
    before = real.get(url(project, "/paper")).json()
    assert before["left_out"] == [] and before["missing_detail"] == ""
    project.review("E1", "reject", reason="The setup was wrong.")

    paper = real.get(url(project, "/paper")).json()
    assert paper["missing"] in ("results", "finding")
    (left,) = paper["left_out"]
    assert left["id"] == finding.id and left["path"] == finding.path
    assert "rejected" in left["reason"]
    assert finding.id in paper["missing_detail"] and "rejected" in paper["missing_detail"]


def test_a_drawing_is_text_too_and_is_served_without_the_credentials_in_it(
    client, project, monkeypatch
):
    """An SVG figure is shown as an image and is text all the same: a label a job wrote
    into it is read by whoever looks at the figure."""
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", FAKE_KEY)
    run_dir = add_run(project)
    drawing = f'<svg xmlns="http://www.w3.org/2000/svg"><text>key {FAKE_KEY}</text></svg>'
    (run_dir / "artifacts" / "figure.svg").write_text(drawing)
    (run_dir / "artifacts" / "clean.svg").write_text(drawing.replace(FAKE_KEY, "none"))
    shown = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/figure.svg"))
    assert shown.status_code == 200 and shown.headers["content-type"] == "image/svg+xml"
    assert shown.text == drawing.replace(FAKE_KEY, "[redacted]")
    assert shown.headers["content-security-policy"].endswith("sandbox")
    assert shown.headers["content-disposition"] == 'inline; filename="figure.svg"'
    clean = client.get(url(project, "/files/experiments/E1/runs/1/artifacts/clean.svg"))
    assert clean.text == drawing.replace(FAKE_KEY, "none") and "content-length" in clean.headers


def test_a_repository_address_no_file_can_hold_does_not_cost_the_logs_or_the_stream(
    client, stream, project
):
    """``project.json`` written by hand, with half a surrogate pair in the token of the
    repository's address: what is blanked is looked for as bytes, and looking for it
    must not fail the answer (the log, a served file, every event)."""
    file = project.root / "project.json"
    data = json.loads(file.read_text())
    data["settings"]["repo"] = {"path": "https://tok\ud83den-0123456789@git.example.invalid/x.git"}
    file.write_text(json.dumps(data))  # written as an escape, as any JSON writer does
    run_dir = add_run(project)
    (project.root / "worker.log").write_text("it said tok\\ud83den-0123456789 and went on\n")
    append_event(project, type="note", summary="it said tok\ud83den-0123456789")

    for path in (
        "/worker/log",
        "/experiments/E1/runs/1/log",
        "/experiments/E1/runs/1",
        "/files/experiments/E1/runs/1/log.txt",
        "/activity",
        "",
    ):
        assert client.get(url(project, path)).status_code == 200, path
    assert (run_dir / "log.txt").read_text() in client.get(
        url(project, "/files/experiments/E1/runs/1/log.txt")
    ).text
    # Where the file holds it as JSON writes it, it is blanked like any other.
    assert (
        client.get(url(project, "/worker/log")).json()["text"] == "it said [redacted] and went on\n"
    )
    live = stream(url(project, "/events"))
    wait_for(lambda: len(live.events()) == len(project.events()), message="every event")
    assert live.events()[-1]["summary"] == "it said [redacted]"


def test_the_server_itself_puts_back_what_could_not_be_saved(client, project, monkeypatch):
    """The store reads the file back when a save of its own fails; the server does not
    lean on that.  Whatever stopped the change (here a save that fails and leaves the
    object as it was set), the project it keeps says what is on disk again."""

    def fails(self):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patched:
        patched.setattr(Project, "save_settings", fails)
        with pytest.raises(OSError):
            client.patch(
                url(project),
                json={"title": "Never saved", "settings": {"limits": {"model_usd": 99.0}}},
            )
    seen = client.get(url(project)).json()
    assert seen["title"] == "Warmup" and seen["settings"]["limits"]["model_usd"] != 99.0
    assert client.get("/api/projects").json()["projects"][0]["title"] == "Warmup"


def test_the_token_of_a_repository_an_experiment_names_itself_is_blanked_too(
    client, stream, project
):
    """An experiment may name a repository of its own on its page, token and all (only the
    person can).  That address is shown like the project's: without the token, in the
    page as in every answer that quotes it."""
    project.review("E1", "accept")
    project.update_page("E1", author="human", fields={"mode": "repo", "repo": PRIVATE_URL})
    said = f"fatal: could not read from {PRIVATE_URL}"
    append_event(project, type="error", summary=said, data={"error": said})
    (project.root / "worker.log").write_text(said + "\n")
    shown = "https://[redacted]@git.example.invalid/me/private.git"

    answers = {
        "the page": client.get(url(project, "/pages/E1")),
        "the pages": client.get(url(project, "/pages?kind=experiment")),
        "the experiments": client.get(url(project, "/experiments")),
        "the experiment": client.get(url(project, "/experiments/E1")),
        "the activity": client.get(url(project, "/activity")),
        "the worker's log": client.get(url(project, "/worker/log")),
        "a stale edit": client.put(
            url(project, "/pages/E1"), json={"body": "x", "base_updated": "2020-01-01T00:00:00Z"}
        ),
    }
    for name, response in answers.items():
        assert TOKEN not in response.text, name
    assert answers["the page"].json()["page"]["meta"]["repo"] == shown
    assert answers["a stale edit"].status_code == 409
    live = stream(url(project, "/events"))
    wait_for(lambda: len(live.events()) == len(project.events()), message="every event")
    assert TOKEN not in live._data.decode()

    # The address as it was shown, sent back with an edit, is the address that is kept.
    saved = client.put(
        url(project, "/pages/E1"), json={"fields": {"repo": shown, "timeout_seconds": 60}}
    )
    assert saved.status_code == 200, saved.text
    assert project.page("E1").meta["repo"] == PRIVATE_URL
    assert project.page("E1").meta["timeout_seconds"] == 60
    project.propose_edit("E1", author="agent:designer", summary="A note", body="x\n", round=1)
    for ref in ("E1", project.proposals(review="proposed")[-1].id):
        decided = client.post(
            url(project, "/review"),
            json={
                "items": [{"ref": ref, "decision": "accept", "edits": {"fields": {"repo": shown}}}]
            },
        ).json()
        assert TOKEN not in json.dumps(decided), ref
    assert project.page("E1").meta["repo"] == PRIVATE_URL
    # Another address is another address: the person names a repository as before.
    other = "https://git.example.invalid/me/public.git"
    assert (
        client.put(url(project, "/pages/E1"), json={"fields": {"repo": other}}).status_code == 200
    )
    assert project.page("E1").meta["repo"] == other
    # The library still reads no page for it.
    assert client.get("/api/projects").status_code == 200


# ---------------------------------------------------------------------------
# The seams between the API and the app: the server says it, the app shows it
# ---------------------------------------------------------------------------


def record_of(phase: str, round: int = 1, status: str = "completed") -> dict[str, Any]:
    """A phase record as the engine writes it into state.json."""
    return {
        "round": round, "phase": phase, "status": status,
        "started": "2026-11-15T12:00:00Z", "finished": "2026-11-15T12:00:05Z",
        "seconds": 5.0, "attempts": 1, "error": None, "produced": [],
        "instruction": None, "requests": [],
    }  # fmt: skip


def phases_ran(project: Project, *phases: str, following: str = "hypothesize") -> dict[str, Any]:
    """The project as the engine leaves it at the checkpoint after ``phases`` (intake and
    research when none are named): their events, and the state.json that goes with them."""
    phases = phases or ("intake", "research")
    for phase in phases:
        project.emit("phase.started", f"Round 1: the {phase} phase started.", phase=phase, round=1)
        project.emit(
            "phase.finished",
            f"The {phase} phase finished.",
            phase=phase,
            round=1,
            data={"produced": [], "seconds": 5.0},
        )
    paused = project.emit(
        "checkpoint",
        "2 topics await your review. Next: what comes next.",
        phase=phases[-1],
        round=1,
        data={"pending": {"topic": 2}, "edits": 0, "next": following},
    )
    state = {
        "schema": 1, "status": "checkpoint", "round": 1, "phase": phases[-1],
        "next_phase": following,
        "checkpoint": {"round": 1, "phase": phases[-1], "at": paused["ts"],
                       "summary": paused["summary"], "pending": {"topic": 2},
                       "recommends": None, "recommends_page": None},
        "phases": [record_of(phase) for phase in phases], "worker": None,
        "spent": {"model_usd": 0.75, "experiment_seconds": 0.0, "experiment_usd": 0.0},
        "error": None,
    }  # fmt: skip
    (project.root / "state.json").write_text(json.dumps(state))
    return state


def files_of(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under ``root`` with what tells one version of it from the next."""
    found = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in found
    }


def row_of(client, directory: str) -> dict[str, Any]:
    rows = client.get("/api/projects").json()["projects"]
    return next(row for row in rows if row["dir"] == directory)


def test_a_demonstration_is_marked_where_its_cost_is_shown(client, workspace, project):
    demo = Project.create(
        workspace / "demo", title="Demonstration: warmup", brief="x", demonstration=True
    )
    phases_ran(demo)  # $0.75 "spent" on models, which no model was paid

    row = row_of(client, "demo")
    view = client.get(url(demo)).json()

    assert row["demonstration"] is True and view["demonstration"] is True
    assert row["spent"]["model_usd"] == 0.75
    # Any other project says that it is none.
    assert row_of(client, "warmup")["demonstration"] is False
    assert client.get(url(project)).json()["demonstration"] is False


def test_a_lost_state_is_shown_as_the_engine_reads_it_with_the_problem_in_a_field_of_its_own(
    client, workspace
):
    project = Project.create(workspace / "lost", title="Lost", brief="Does warmup matter?")
    written = phases_ran(project)
    view = client.get(url(project)).json()
    assert (view["state_source"], view["state_problem"]) == ("file", None)
    assert view["state"] == written and view["checkpoint"] == written["checkpoint"]

    (project.root / "state.json").unlink()
    view = client.get(url(project)).json()

    # Not round 0, $0.00 and idle beside a reported problem: where the project stands, read
    # from its events as the engine reads it, and the engine's sentence about that.
    state = view["state"]
    assert (state["status"], state["round"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        1,
        "research",
        "hypothesize",
    )
    assert [(p["phase"], p["status"]) for p in state["phases"]] == [
        ("intake", "completed"),
        ("research", "completed"),
    ]
    assert view["checkpoint"]["pending"] == {"topic": 2}
    assert view["state_source"] == "events"
    assert view["state_problem"].startswith(
        "state.json is missing. Round, phases and what comes next are read from the "
        "project's events (events.jsonl) instead"
    )
    # Said once, in its own field: the list of what was read around does not repeat it.
    assert view["problems"] == [] and view["problems_total"] == 0
    # What a press of Continue does comes from the same reading.
    assert view["next_step"]["will_run"] == ["hypothesize"]
    assert view["next_step"]["then"] == "design"

    # The library row does not claim "not started" where the project page says otherwise.
    row = row_of(client, "lost")
    assert (row["status"], row["round"], row["phase"], row["next_phase"]) == (
        "checkpoint",
        1,
        "research",
        "hypothesize",
    )
    assert row["checkpoint"]["pending"] == {"topic": 2}
    assert (row["state_source"], row["state_problem"]) == ("events", view["state_problem"])

    # A file that is there and does not say: the same, with the engine's word for it.
    (project.root / "state.json").write_text("{}")
    view = client.get(url(project)).json()
    assert view["state_source"] == "events" and view["state"]["status"] == "checkpoint"
    assert view["state_problem"].startswith("state.json holds no record of the phases that ran.")


def test_where_a_project_stands_is_said_not_guessed(client, workspace, project):
    # Never started: nothing is wrong, and intake is what comes first.
    fresh = Project.create(workspace / "fresh", title="Fresh", brief="An idea.")
    view = client.get(url(fresh)).json()
    assert (view["state_source"], view["state_problem"]) == ("not_started", None)
    assert (view["state"]["status"], view["state"]["next_phase"]) == ("idle", "intake")
    assert view["next_step"] == {
        "will_run": ["intake", "research"],
        "then": "hypothesize",
        "experiments": [],
        "not_running": [],
        "results": 0,
        "checkpoint_of": None,
        "needs_phase": False,
        "blocked": None,
    }
    row = row_of(client, "fresh")
    assert (row["status"], row["state_source"], row["state_problem"]) == (
        "idle",
        "not_started",
        None,
    )

    # Agents wrote pages, and neither state.json nor the events say where that was: the
    # engine cannot tell, and a press of Continue has to name the phase.
    view = client.get(url(project)).json()
    assert view["state_source"] == "unknown"
    assert "cannot tell where the project stands" in view["state_problem"]
    assert view["state"]["status"] == "idle" and view["state"]["phases"] == []
    assert view["next_step"]["needs_phase"] is True and view["next_step"]["will_run"] == []
    row = row_of(client, "warmup")
    assert (row["state_source"], row["state_problem"]) == ("unknown", view["state_problem"])

    # With the engine's account on file there is nothing to say.
    phases_ran(project, "intake", "research", "hypothesize", following="design")
    view = client.get(url(project)).json()
    assert (view["state_source"], view["state_problem"]) == ("file", None)
    assert (view["next_step"]["will_run"], view["next_step"]["then"]) == (["design"], "run")
    assert row_of(client, "warmup")["state_source"] == "file"


def test_the_library_replays_no_events_for_a_project_whose_state_reads_fine(
    client, workspace, monkeypatch
):
    for number in range(6):
        made = Project.create(workspace / f"p{number}", title=f"Project {number}", brief="x")
        phases_ran(made)
    lost = Project.open(workspace / "p5")
    (lost.root / "state.json").unlink()
    replayed: list[str] = []
    events = Project.events

    def counting(self, *args, **kwargs):
        replayed.append(self.root.name)
        return events(self, *args, **kwargs)

    monkeypatch.setattr(Project, "events", counting)
    rows = client.get("/api/projects").json()["projects"]
    assert {row["status"] for row in rows} == {"checkpoint"} and len(rows) == 6
    # Only the project whose state.json does not say where it stands is read from its events.
    assert set(replayed) == {"p5"}
    del replayed[:]

    def no_pages(self, *args, **kwargs):
        raise AssertionError("the library must not read pages again")

    with monkeypatch.context() as patched:
        patched.setattr(Project, "_all_pages", no_pages)
        patched.setattr(Project, "_load_page", no_pages)
        again = client.get("/api/projects").json()["projects"]
    # ... and only once: nothing changed in it, so nothing is replayed at the next look.
    assert replayed == []
    assert [row["state_source"] for row in again] == [row["state_source"] for row in rows]

    # An event more, and the next look reads it again.
    lost.emit("agent.activity", "Something happened.")
    assert row_of(client, "p5")["state_source"] == "events" and replayed == ["p5", "p5"]


@pytest.mark.parametrize("source", ["file", "events"])
def test_looking_at_a_project_writes_nothing_and_takes_no_lock(client, workspace, source):
    fcntl = pytest.importorskip("fcntl")
    project = Project.create(workspace / "lost", title="Lost", brief="Does warmup matter?")
    phases_ran(project)
    if source == "events":
        (project.root / "state.json").unlink()
    locks = [project.root / ".worker.lock", project.root / ".project.lock"]
    for lock in locks:
        lock.touch()
    before = files_of(project.root)

    held = [lock.open("r+") for lock in locks]
    try:
        for handle in held:
            # Somebody has the project (a worker) and somebody is writing to it.
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        view = client.get(url(project)).json()
        rows = client.get("/api/projects").json()["projects"]
        for path in ("/pages", "/experiments", "/proposals", "/paper", "/requests", "/graph"):
            assert client.get(url(project, path)).status_code == 200, path
        assert time.monotonic() - started < 5  # nothing waited for either lock
    finally:
        for handle in held:
            handle.close()

    assert view["state_source"] == source and rows[0]["state_source"] == source
    assert view["next_step"]["will_run"] == ["hypothesize"]
    assert files_of(project.root) == before


def test_where_a_project_stands_is_not_read_through_what_is_no_file_of_its_own(
    client, stream, workspace, tmp_path
):
    project = Project.create(workspace / "linked", title="Linked", brief="x")
    outside = tmp_path / "elsewhere.json"
    outside.write_text(
        json.dumps(
            {
                "status": "failed",
                "round": 7,
                "error": "OUTSIDE-SECRET",
                "phase": "design",
                "phases": [record_of("design", 7, "failed")],
            }
        )
    )
    (project.root / "state.json").symlink_to(outside)

    view = client.get(url(project))
    assert "OUTSIDE-SECRET" not in view.text and view.json()["state"]["round"] == 0
    assert view.json()["state_source"] == "unknown" and view.json()["next_step"] is None
    assert view.json()["state_problem"].startswith(
        "state.json of this project is not an ordinary file"
    )
    row = client.get("/api/projects")
    assert "OUTSIDE-SECRET" not in row.text
    assert row.json()["projects"][0]["state_source"] == "unknown"
    live = stream(url(project, "/events"))
    wait_for(lambda: live.states())
    assert b"OUTSIDE-SECRET" not in live._data
    assert live.states()[0]["state_source"] == "unknown"

    # Something that is no file at all (a pipe would be waited for without end).
    (project.root / "state.json").unlink()
    os.mkfifo(project.root / "events.jsonl.pipe")
    (project.root / "events.jsonl").unlink()
    os.rename(project.root / "events.jsonl.pipe", project.root / "events.jsonl")
    started = time.monotonic()
    view = client.get(url(project)).json()
    assert time.monotonic() - started < 5
    assert view["state_source"] == "unknown"
    assert view["state_problem"].startswith("events.jsonl of this project is not an ordinary file")


def test_no_agent_is_started_where_the_view_says_that_the_state_is_not_read(
    client, workspace, launched, tmp_path
):
    project = Project.create(workspace / "linked", title="Linked", brief="x")
    written = phases_ran(project)
    outside = tmp_path / "elsewhere.json"
    (project.root / "state.json").rename(outside)
    (project.root / "state.json").symlink_to(outside)
    view = client.get(url(project)).json()
    assert (view["state_source"], view["next_step"]) == ("unknown", None)

    # The view says that where the project stands is not read, and foretells no press. A
    # press must not then start an engine that reads the state through that very link: the
    # answer is the sentence of the view, and nothing is started, with a phase named or not.
    for body in ({}, {"phase": "hypothesize"}):
        refused = client.post(url(project, "/advance"), json=body)
        assert refused.status_code == 409, refused.text
        assert error_of(refused) == {
            "code": "conflict",
            "message": f"Nothing was started. {view['state_problem']}",
        }
    assert launched.calls == []
    assert client.get(url(project)).json()["worker"]["state"] == "idle"

    # Put right (the file is the project's own again), the same press starts the agents.
    (project.root / "state.json").unlink()
    outside.rename(project.root / "state.json")
    view = client.get(url(project)).json()
    assert (view["state_source"], view["state"]) == ("file", written)
    assert client.post(url(project, "/advance")).status_code == 202
    assert [kind for kind, _ in launched.calls] == ["advance"]


def test_a_link_and_the_file_it_led_to_are_told_apart_at_the_next_look(
    client, stream, workspace, launched, tmp_path
):
    project = Project.create(workspace / "linked", title="Linked", brief="x")
    phases_ran(project)
    (project.root / "state.json").unlink()  # lost: where it stands is read from the events
    events, outside = project.root / "events.jsonl", tmp_path / "events.jsonl"
    assert client.get(url(project)).json()["state_source"] == "events"
    live = stream(url(project, f"/events?after={project.events()[-1]['seq']}"))
    wait_for(lambda: live.states(), message="the first state")

    events.rename(outside)
    events.symlink_to(outside)
    problem = client.get(url(project)).json()["state_problem"]
    assert problem.startswith("events.jsonl of this project is not an ordinary file")
    # Put right by moving the file itself back: under the project's name stands what the
    # link led to, with the same inode, time and size.  The next look must not go on
    # saying that it is no file of the project's (nor go on refusing to start anything).
    events.unlink()
    outside.rename(events)
    view = client.get(url(project)).json()
    assert view["state_source"] == "events" and view["next_step"]["will_run"] == ["hypothesize"]
    assert client.post(url(project, "/advance")).status_code == 202

    # The same for a window that is open: the stream says so without being asked.
    project = Project.create(workspace / "watched", title="Watched", brief="x")
    phases_ran(project)
    state, outside = project.root / "state.json", tmp_path / "state.json"
    live = stream(url(project, f"/events?after={project.events()[-1]['seq']}"))
    wait_for(lambda: live.states(), message="the first state")
    assert live.states()[-1]["state_source"] == "file"
    state.rename(outside)
    state.symlink_to(outside)
    wait_for(lambda: live.states()[-1]["state_source"] == "unknown", message="the link is seen")
    state.unlink()
    outside.rename(state)
    wait_for(lambda: live.states()[-1]["state_source"] == "file", message="the file is seen")


def test_the_stream_says_where_a_project_with_a_lost_state_stands(stream, workspace):
    project = Project.create(workspace / "lost", title="Lost", brief="Does warmup matter?")
    written = phases_ran(project)
    (project.root / "state.json").unlink()

    live = stream(url(project, f"/events?after={project.events()[-1]['seq']}"))
    wait_for(lambda: live.states(), message="the first state")
    first = live.states()[0]
    # What the stream hands the app replaces what the project view gave it: the same reading.
    assert first["state"]["status"] == "checkpoint" and first["state"]["round"] == 1
    assert first["state_source"] == "events" and "events.jsonl" in first["state_problem"]

    # The engine writes its file anew when it next works: the stream says so.
    temporary = project.root / "state.json.tmp"
    temporary.write_text(json.dumps(written))
    os.replace(temporary, project.root / "state.json")
    wait_for(lambda: len(live.states()) == 2, message="the state on file again")
    second = live.states()[1]
    assert (second["state_source"], second["state_problem"]) == ("file", None)
    assert second["state"] == written


def approved_experiment(project: Project, services: FakeServices) -> None:
    """H1 and E1 accepted and the run of E1 approved, as the person does it: through the
    experiments module, which records what was approved.  (What a press will run is
    measured against that record; the stand-in's approval leaves none.)"""
    from paper_orchestra.research import experiments

    project.review("H1", "accept")
    project.review("E1", "accept")
    experiments.approve(project, "E1")


def test_the_project_says_what_a_press_of_continue_will_do(client, project, services, caplog):
    from paper_orchestra.research import engine

    phases_ran(project, "intake", "research", "hypothesize", following="design")
    said = client.get(url(project)).json()["next_step"]
    assert (said["will_run"], said["then"], said["experiments"]) == (["design"], "run", [])

    # What the person approved comes first, and the phase that was next stays next.
    approved_experiment(project, services)
    said = client.get(url(project)).json()["next_step"]
    assert said == {
        "will_run": ["run", "analyze"],
        "then": "design",
        "experiments": ["E1"],
        "not_running": [],
        "results": 0,
        "checkpoint_of": None,
        "needs_phase": False,
        "blocked": None,
    }
    # It is the engine's own answer, not a second rule of the server's.
    assert said == engine.next_step(Project.open(project.root))

    # A result nobody has analysed comes before the phase that was next as well.
    project.set_state("E1", "queued", by="system")
    project.set_state("E1", "running", by="system")
    project.set_state("E1", "succeeded", by="system")
    a_finding(project)
    said = client.get(url(project)).json()["next_step"]
    assert (said["will_run"], said["then"], said["results"]) == (["analyze"], "design", 1)

    # A forecast that fails costs the forecast, not the project view, and the log is
    # told once however often the project is looked at.
    def broken(project, state):
        raise RuntimeError("no forecast today")

    working = services.next_step
    services.next_step = broken
    for _ in range(3):
        view = client.get(url(project))
        assert view.status_code == 200 and view.json()["next_step"] is None
    assert len([r for r in caplog.records if "would do could not be told" in r.message]) == 1
    services.next_step = working

    # While somebody works on the project, nothing is foretold: the worker decides.
    client.post(url(project, "/advance"))
    wait_for(lambda: (project.root / "ready").exists())
    assert client.get(url(project)).json()["next_step"] is None
    # The library does not ask at all: its rows stay cheap.
    assert "next_step" not in client.get("/api/projects").json()["projects"][0]


NO_CREDENTIALS = LLMConfigError(
    "No Anthropic credentials were found. Configure one of:\n  1. API key: set "
    "ANTHROPIC_API_KEY (for example in a .env file).\n  2. Workload Identity Federation."
)


def test_the_project_says_beforehand_that_a_press_needs_a_model(client, project, services):
    phases_ran(project, "intake", "research", "hypothesize", following="design")
    assert client.get(url(project)).json()["next_step"]["blocked"] is None

    services.no_client = NO_CREDENTIALS
    said = client.get(url(project)).json()["next_step"]

    # Nothing would run, and the reason is a field with the one sentence to show: no
    # worker has to be started and refused to find it out.
    assert (said["will_run"], said["then"]) == ([], "design")
    blocked = said["blocked"]
    assert (blocked["phase"], blocked["needs"]) == ("design", "model")
    assert blocked["settings"] == ["research", "reasoning", "coding"]
    assert blocked["message"].startswith(
        "The design phase needs a model, and the model set for research, reasoning and coding"
    )
    assert "set ANTHROPIC_API_KEY" in blocked["message"]
    assert "\n" not in blocked["message"] and "run the command" not in blocked["message"]
    # The library does not ask at all: its rows stay cheap.
    assert "next_step" not in client.get("/api/projects").json()["projects"][0]


def test_whether_a_model_can_be_used_is_found_out_once_in_a_while_not_at_every_look(
    real, project, monkeypatch
):
    """The server's own answer (no stand-in): with no credential configured, as here,
    making a model's client fails anew every time it is tried, and costs as much as
    making one.  A project is looked at often; the refusal is remembered for a while."""
    from paper_orchestra.llm import router

    phases_ran(project, "intake", "research", "hypothesize", following="design")
    asked: list[str] = []
    real_client = router.get_client

    def counting(model: str):
        asked.append(model)
        return real_client(model)

    monkeypatch.setattr(router, "get_client", counting)
    router.reset_clients()

    first = real.get(url(project)).json()["next_step"]
    for _ in range(5):
        assert real.get(url(project)).json()["next_step"] == first

    assert first["blocked"]["needs"] == "model" and first["will_run"] == []
    assert "no Anthropic credentials were found" in first["blocked"]["message"]
    assert len(asked) == 1  # three roles, one model, one look for six views


def test_a_remembered_refusal_does_not_keep_what_earlier_looks_left_on_it():
    """The refusal the server remembers is one exception, raised at every look.  Raised as
    it is, it would gather the frames of every look in its traceback, and hold on to
    what they held (a project, a request) for as long as it is remembered."""
    from paper_orchestra.config import DEFAULT_MODEL
    from paper_orchestra.llm import router
    from paper_orchestra.llm.base import LLMConfigError

    router.reset_clients()
    services = Services()

    def depth() -> int:
        """How many frames the refusal carries when it arrives."""
        with pytest.raises(LLMConfigError) as refusal:
            services.client_for(DEFAULT_MODEL)
        frames, at = 0, refusal.value.__traceback__
        while at is not None:
            frames, at = frames + 1, at.tb_next
        return frames

    depth()  # the look that finds out, with the frames of making a client
    remembered = [depth() for _ in range(6)]
    assert remembered == [remembered[0]] * 6


def test_the_checkpoint_carries_the_recommendation_for_as_long_as_it_stands(
    client, stream, workspace
):
    project = Project.create(
        workspace / "rec", title="Rec", brief="x", settings={"review": {"journal": "review"}}
    )
    journal = project.create_page(
        "journal", "Round 1", "Schedule-free optimisers were not looked at.\n",
        author="agent:analyst", fields={"next": "research"}, round=1,
    )  # fmt: skip
    state = phases_ran(project, "intake", "research", "hypothesize", "design", "run", "analyze")
    state["checkpoint"].update(recommends="research", recommends_page=journal.path)
    (project.root / "state.json").write_text(json.dumps(state))

    view = client.get(url(project)).json()
    assert view["checkpoint"]["recommends"] == "research"
    assert view["checkpoint"]["recommends_page"] == journal.path
    assert view["state"]["checkpoint"] == view["checkpoint"]
    live = stream(url(project, "/events"))
    wait_for(lambda: live.states())
    assert live.states()[0]["state"]["checkpoint"]["recommends_page"] == journal.path

    # A journal page the person rejected since recommends nothing any more: the client
    # is told so, and does not have to look the page up to find out.
    project.review(journal.path, "reject", reason="Not now.")
    view = client.get(url(project)).json()
    assert view["checkpoint"]["recommends"] is None
    assert view["checkpoint"]["recommends_page"] is None
    assert view["state"]["checkpoint"] == view["checkpoint"]
    assert "2 topics await your review." in view["checkpoint"]["summary"]  # the rest stands
    # The row of the library claims nothing about it either way.
    assert "recommends" not in row_of(client, "rec")["checkpoint"]

    # A checkpoint that names a page which is gone, or names nothing a file can name.
    for page in ("wiki/journal/nowhere.md", "../../etc/passwd", 7):
        state["checkpoint"].update(recommends="research", recommends_page=page)
        (project.root / "state.json").write_text(json.dumps(state))
        assert client.get(url(project)).json()["checkpoint"]["recommends"] is None


# -- what will run, and what a cancelled run leaves ----------------------------------


def test_an_approved_experiment_that_changed_is_listed_as_one_that_will_not_run(real, project):
    phases_ran(project, "intake", "research", "hypothesize", following="design")
    project.review("H1", "accept")
    project.review("E1", "accept")
    assert real.post(url(project, "/experiments/E1/approve")).status_code == 200

    def everywhere() -> list[Any]:
        """What each place that lists the experiment says about its run."""
        row = real.get(url(project, "/experiments")).json()["experiments"][0]
        page = real.get(url(project, "/pages"), params={"kind": "experiment"}).json()["pages"][0]
        view = real.get(url(project, "/experiments/E1")).json()
        assert row["state"] == page["state"] == view["page"]["state"] == "approved"
        return [row["will_not_run"], page["will_not_run"], view["will_not_run"]]

    assert everywhere() == [None, None, None]
    said = real.get(url(project)).json()["next_step"]
    assert (said["experiments"], said["not_running"]) == (["E1"], [])

    # The person edits the code after approving it.
    changed = real.put(url(project, "/experiments/E1/code/main.py"), json={"text": "print(2)\n"})
    assert changed.status_code == 200

    reasons = everywhere()
    assert len(set(reasons)) == 1 and reasons[0].startswith("it is not what was approved at ")
    assert "main.py" in reasons[0]
    # ... and the press is not said to run it: the forecast names it with the reason.
    said = real.get(url(project)).json()["next_step"]
    assert said["experiments"] == []
    assert said["not_running"] == [{"id": "E1", "reason": reasons[0]}]
    assert said["will_run"][0] == "run" and "analyze" not in said["will_run"]
    # A look wrote nothing: the page still reads approved, as the person left it.
    assert project.page("E1").state == "approved"

    # Approved again as it is now, it runs.
    assert real.post(url(project, "/experiments/E1/approve")).status_code == 200
    assert everywhere() == [None, None, None]


def test_a_cancelled_experiment_whose_earlier_run_succeeded_says_so(client, project):
    project.review("E1", "accept")
    a_finding(project)  # run 1 succeeded, with its result record
    add_run(project, 2, status="cancelled")
    for state in ("approved", "queued", "running", "cancelled"):
        project.set_state("E1", state, by="human" if state == "approved" else "system")

    row = client.get(url(project, "/experiments")).json()["experiments"][0]
    page = client.get(url(project, "/pages"), params={"kind": "experiment"}).json()["pages"][0]
    view = client.get(url(project, "/experiments/E1")).json()

    # "cancelled" is how its last run ended; that an earlier one succeeded is said with it.
    success = {"number": 1, "result_page": "wiki/results/E1-run1.md"}
    assert row["state"] == page["state"] == "cancelled"
    assert row["last_success"] == page["last_success"] == view["last_success"] == success
    assert row["last_run"]["number"] == 2 and row["last_run"]["status"] == "cancelled"

    # An experiment that reads succeeded says it by its state; one that never succeeded
    # has nothing to add.
    second = second_experiment(project)
    rows = client.get(url(project, "/experiments")).json()["experiments"]
    assert next(r for r in rows if r["id"] == second.id)["last_success"] is None


# -- taking an approval back ---------------------------------------------------------


def test_cancel_takes_back_an_approval_that_has_not_started(real, project):
    project.review("E1", "accept")
    assert real.get(url(project, "/experiments/E1")).json()["cancel_does"] is None
    assert real.post(url(project, "/experiments/E1/approve")).status_code == 200
    # What a press of the button does is said before it is pressed.
    assert real.get(url(project, "/experiments/E1")).json()["cancel_does"] == "withdraw"
    assert real.get(url(project, "/experiments")).json()["experiments"][0]["cancel_does"] == (
        "withdraw"
    )
    seen = project.events()[-1]["seq"]

    taken = real.post(url(project, "/experiments/E1/cancel"), json={"expect": "withdraw"})

    # As `paper-orchestra research cancel` does it: the experiment is a plan again.
    assert taken.status_code == 200, taken.text
    body = taken.json()
    assert (body["experiment"], body["outcome"], body["before"], body["state"]) == (
        "E1",
        "withdrawn",
        "approved",
        "planned",
    )
    assert body["message"] == (
        "E1 is a plan again: its approval was taken back and nothing ran. Approve it again "
        "when you want it run."
    )
    assert body["page"]["state"] == "planned" and body["cancel_does"] is None
    page = Project.open(project.root).page("E1")
    assert (page.state, page.meta["state_reason"]) == (
        "planned",
        "You withdrew the approval of its run.",
    )
    # Nothing was cancelled and nothing is on record as a run.
    assert real.get(url(project, "/experiments/E1")).json()["runs"] == []
    assert not (project.experiment_dir("E1") / ".cancel").exists()
    assert [event["type"] for event in project.events(after=seen)] == ["experiment.state"]

    again = real.post(url(project, "/experiments/E1/cancel"), json={"expect": "withdraw"})
    assert again.status_code == 200 and again.json()["outcome"] == "nothing"
    assert again.json()["message"] == "E1 is planned: there is nothing to cancel."

    # Without a word about what was expected, the state decides, as on the command line.
    assert real.post(url(project, "/experiments/E1/approve")).status_code == 200
    plain = real.post(url(project, "/experiments/E1/cancel"))
    assert plain.status_code == 200 and plain.json()["outcome"] == "withdrawn"
    assert (
        real.post(url(project, "/experiments/E1/cancel"), json={"expect": "x"}).status_code == 422
    )


def test_a_queued_experiment_is_cancelled_as_before(real, project):
    project.review("E1", "accept")
    real.post(url(project, "/experiments/E1/approve"))
    project.set_state("E1", "queued", by="system")
    assert real.get(url(project, "/experiments/E1")).json()["cancel_does"] == "cancel"

    cancelled = real.post(url(project, "/experiments/E1/cancel"), json={"expect": "cancel"})

    # Nobody is running it: it is cancelled here and now.
    assert cancelled.status_code == 200, cancelled.text
    body = cancelled.json()
    assert (body["outcome"], body["before"], body["state"]) == ("cancelled", "queued", "cancelled")
    assert body["message"] == "E1 is cancelled. Approve it again when you want it run."
    assert Project.open(project.root).page("E1").state == "cancelled"


def test_the_press_does_what_the_person_was_told_or_nothing(real, project):
    fcntl = pytest.importorskip("fcntl")
    project.review("E1", "accept")
    real.post(url(project, "/experiments/E1/approve"))

    # Told "Cancel" (an older view of the experiment), and there is no run to cancel.
    stale = real.post(url(project, "/experiments/E1/cancel"), json={"expect": "cancel"})
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "conflict"
    assert "has not started" in stale.json()["error"]["message"]
    assert stale.json()["cancel_does"] == "withdraw" and stale.json()["page"]["state"] == "approved"
    assert Project.open(project.root).page("E1").state == "approved"

    # A process took the experiment up meanwhile (it holds the run lock): taking the
    # approval back must not cancel, requeue or otherwise touch the run in flight.
    lock = project.experiment_dir("E1") / ".run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert real.get(url(project, "/experiments/E1")).json()["cancel_does"] == "cancel"
        before = files_of(project.root)
        late = real.post(url(project, "/experiments/E1/cancel"), json={"expect": "withdraw"})
        assert late.status_code == 409 and "cannot be taken back" in late.json()["error"]["message"]
        assert late.json()["cancel_does"] == "cancel"
        assert files_of(project.root) == before  # no flag for the runner, no state moved
        # Asked to cancel the run, the server does: whoever holds the lock sees the flag.
        told = real.post(url(project, "/experiments/E1/cancel"), json={"expect": "cancel"})
        assert told.status_code == 202 and told.json()["outcome"] == "cancelling"
        assert (project.experiment_dir("E1") / ".cancel").exists()
        assert Project.open(project.root).page("E1").state == "approved"


class Racing(Services):
    """Services that see no process running an experiment: the moment between a look at
    the run lock and the withdrawal, in which a worker takes the experiment up."""

    def run_held(self, project, experiment_id):
        return False


def test_an_approval_a_run_took_up_in_that_moment_is_not_taken_back(
    workspace, project, launched, live, tmp_path
):
    fcntl = pytest.importorskip("fcntl")
    app = create_app(workspace, launcher=launched, services=Racing(), live=live,
                     static_dir=tmp_path / "none", stop_workers_on_exit=True)  # fmt: skip
    project.review("E1", "accept")
    with TestClient(app, base_url=BASE_URL) as client:
        assert client.post(url(project, "/experiments/E1/approve")).status_code == 200
        lock = project.experiment_dir("E1") / ".run.lock"
        with lock.open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            before = files_of(project.root)
            for body in ({"expect": "withdraw"}, {}):
                refused = client.post(url(project, "/experiments/E1/cancel"), json=body)
                assert refused.status_code == 409, refused.text
                assert "is being run right now" in refused.json()["error"]["message"]
                assert refused.json()["page"]["state"] == "approved"
            assert files_of(project.root) == before  # the run in flight was not touched


def test_an_approval_whose_run_is_over_is_said_to_be_used_up(client, project, services):
    approved_experiment(project, services)

    def withdraw(project, ref):
        # As the experiments module finds it: the approval was used by a run that is over,
        # and the page is moved to how that run ended.
        project.set_state(ref, "queued", by="system")
        project.set_state(ref, "running", by="system")
        project.set_state(ref, "succeeded", by="system")
        raise ProjectError("E1 is succeeded; there is no approval to withdraw.")

    services.withdraw = withdraw
    answer = client.post(url(project, "/experiments/E1/cancel"), json={"expect": "withdraw"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert (body["outcome"], body["before"], body["state"]) == ("ended", "approved", "succeeded")
    assert body["message"] == (
        "E1 is succeeded: its run had ended before the approval could be taken back."
    )


def test_an_approval_a_run_took_up_and_moved_on_with_is_not_said_to_have_ended(
    client, project, services
):
    approved_experiment(project, services)

    def withdraw(project, ref):
        # A worker took the experiment up between the look and the withdrawal and has
        # moved its page on: the experiments module finds it running.
        project.set_state(ref, "queued", by="system")
        project.set_state(ref, "running", by="system")
        raise ProjectError("E1 is running; there is no approval to withdraw. Cancel its run.")

    services.withdraw = withdraw
    seen = project.events()[-1]["seq"]
    answer = client.post(url(project, "/experiments/E1/cancel"), json={"expect": "withdraw"})

    # Not "its run had ended": the run is under way, and the press did nothing to it.
    assert answer.status_code == 409, answer.text
    assert error_of(answer) == {
        "code": "conflict",
        "message": "E1 is running and has been taken up by a run since you looked: its "
        "approval is in use and cannot be taken back. Nothing was changed. Cancel its run if "
        "you want it stopped.",
    }
    assert (answer.json()["page"]["state"], answer.json()["cancel_does"]) == ("running", "cancel")
    assert not (project.experiment_dir("E1") / ".cancel").exists()
    moves = [event["data"]["state"] for event in project.events(after=seen)]
    assert moves == ["queued", "running"]  # the worker's own, and nothing after them


def test_an_approval_somebody_else_took_back_in_that_moment_leaves_nothing_to_do(
    client, project, services
):
    approved_experiment(project, services)

    def withdraw(project, ref):
        # Another window took the approval back between the look and this press.
        project.set_state(ref, "planned", by="human", reason="You withdrew the approval.")
        raise ProjectError("E1 is planned; there is no approval to withdraw.")

    services.withdraw = withdraw
    answer = client.post(url(project, "/experiments/E1/cancel"), json={"expect": "withdraw"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert (body["outcome"], body["before"], body["state"]) == ("nothing", "approved", "planned")
    # The sentence is about the experiment as it is, not as the press found it.
    assert body["message"] == "E1 is planned: there is nothing to cancel."


# -- an interrupted run, wherever a page is listed -------------------------------------


def test_an_interrupted_experiment_is_marked_wherever_its_page_is_listed(client, project, services):
    rows = {row["path"]: row for row in client.get(url(project, "/pages")).json()["pages"]}
    experiment = next(row for row in rows.values() if row["kind"] == "experiment")
    assert experiment["interrupted"] is False
    assert all("interrupted" not in row for row in rows.values() if row["kind"] != "experiment")

    running_without_a_worker(project, services)
    # The glyph in the rail and on the map is drawn from this list: it says what the
    # experiment's own view says in words.
    listed = client.get(url(project, "/pages?kind=experiment")).json()["pages"]
    assert [(row["id"], row["state"], row["interrupted"]) for row in listed] == [
        ("E1", "running", True)
    ]
    found = client.get(url(project, "/pages?q=variance")).json()["pages"]
    assert next(row for row in found if row["id"] == "E1")["interrupted"] is True


# -- the problems of a project's files ---------------------------------------------------


def test_a_problem_names_its_file_once_and_a_capped_list_says_how_many_there_are(client, workspace):
    project = Project.create(workspace / "odd", title="Odd", brief="x")
    # Phases ran by what the events hold, but not which: the store reports the state as
    # lost.  Its sentence begins with the name of the file; the entry in the list of
    # problems, which names the file already, goes on from there.
    project.emit("checkpoint", "A pause.", phase="research", round=1)
    assert project.state_lost().startswith("state.json is missing")
    assert project.problems() == [
        {"path": "state.json", "message": project.state_lost()[len("state.json ") :]}
    ]

    view = client.get(url(project)).json()
    assert view["state_problem"] is None
    assert view["problems"] == [
        {"path": "state.json", "message": project.state_lost()[len("state.json ") :]}
    ]
    assert view["problems"][0]["message"].startswith("is missing or cannot be read")
    assert view["problems_total"] == 1

    folder = project.root / "wiki" / "hypotheses"
    folder.mkdir(parents=True, exist_ok=True)
    for number in range(60):
        (folder / f"H{number + 1}-by-hand.md").write_text(
            "---\ntitle: By hand\ncites: [sources/nowhere]\n---\nText.\n"
        )
    view = client.get(url(project)).json()
    assert len(view["problems"]) == 50 and view["problems_total"] >= 61


# -- how the last write-up went, from what is on disk ---------------------------------------


def paper_started(project: Project) -> dict[str, Any]:
    return project.emit(
        "paper.started",
        "Writing a paper from the wiki with the iclr2025 template: 1 succeeded run, 1 "
        "accepted finding.",
        data={"template": "iclr2025", "dir": "paper", "findings": ["wiki/findings/F1-x.md"]},
    )


def restarted(workspace, launched, live, tmp_path) -> TestClient:
    """Another server on the same workspace: it knows only what is on disk."""
    return TestClient(
        create_app(workspace, launcher=launched, services=FakeServices(), live=live,
                   static_dir=tmp_path / "none", stop_workers_on_exit=True),
        base_url=BASE_URL,
    )  # fmt: skip


def test_how_the_last_write_up_went_is_read_from_the_projects_events(
    client, project, workspace, launched, live, tmp_path
):
    assert client.get(url(project, "/paper")).json()["last"] is None
    started = paper_started(project)
    failed = project.emit(
        "paper.finished",
        "No paper was written: the outline stage failed.",
        data={"success": False, "pdf": None, "error": "the outline stage failed", "cost_usd": 0.2},
    )

    last = client.get(url(project, "/paper")).json()["last"]

    assert last == {
        "outcome": "failed",
        "started_at": started["ts"],
        "started_summary": started["summary"],
        "template": "iclr2025",
        "ended_at": failed["ts"],
        "summary": "No paper was written: the outline stage failed.",
        "error": "the outline stage failed",
        "exit_code": None,
        "pdf": None,
    }

    # A write-up that was refused before it began: its worker's end is the server's event.
    refused = project.emit(
        "paper.finished",
        "The write-up did not start: its worker ended with exit code 1 and said: “There is "
        "nothing to write a paper from yet: no finding is accepted.”",
        data={"success": False, "interrupted": False, "started": False, "exit_code": 1,
              "error": "There is nothing to write a paper from yet: no finding is accepted."},
    )  # fmt: skip
    last = client.get(url(project, "/paper")).json()["last"]
    assert last == {
        "outcome": "refused",
        "started_at": None,
        "started_summary": None,
        "template": None,
        "ended_at": refused["ts"],
        "summary": refused["summary"],
        "error": "There is nothing to write a paper from yet: no finding is accepted.",
        "exit_code": 1,
        "pdf": None,
    }
    # A server that was started afterwards says the same: nothing of it was in memory.
    with restarted(workspace, launched, live, tmp_path) as other:
        assert other.get(url(project, "/paper")).json()["last"] == last

    # The next write-up: running, stopped, and written.
    started = paper_started(project)
    last = client.get(url(project, "/paper")).json()["last"]
    assert (last["outcome"], last["started_at"], last["ended_at"], last["summary"]) == (
        "unfinished",  # no worker writes it, and none said how it ended
        started["ts"],
        None,
        None,
    )
    stopped = project.emit(
        "paper.finished",
        "The write-up was stopped before the paper was finished; starting it again continues it.",
        data={"success": False, "interrupted": True, "error": None},
    )
    last = client.get(url(project, "/paper")).json()["last"]
    assert (last["outcome"], last["ended_at"]) == ("stopped", stopped["ts"])
    paper_started(project)
    project.emit(
        "paper.finished",
        "The paper is written: paper/final_paper.pdf.",
        data={"success": True, "pdf": "paper/final_paper.pdf", "error": None, "cost_usd": 1.5},
    )
    last = client.get(url(project, "/paper")).json()["last"]
    assert (last["outcome"], last["pdf"]) == ("written", "paper/final_paper.pdf")


def test_a_write_up_that_is_running_is_said_to_be(client, project, launched):
    client.post(url(project, "/paper"), json={"template": "iclr2025"})
    wait_for(lambda: (project.root / "ready").exists())
    paper_started(project)
    paper = client.get(url(project, "/paper")).json()
    assert paper["running"] is True and paper["last"]["outcome"] == "running"


def test_a_refused_write_up_is_shown_with_its_reason_also_by_the_next_server(
    workspace, project, live, tmp_path
):
    refusing = tmp_path / "refusing.py"
    refusing.write_text(
        "import sys\n"
        "print('Error: There is nothing to write a paper from yet: no finding is accepted.',"
        " file=sys.stderr)\n"
        "sys.exit(1)\n"
    )

    def serve() -> TestClient:
        app = create_app(
            workspace,
            launcher=lambda kind, root, options: [sys.executable, str(refusing)],
            services=FakeServices(),
            live=live,
            static_dir=tmp_path / "none",
            stop_workers_on_exit=True,
        )
        return TestClient(app, base_url=BASE_URL)

    with serve() as first:
        assert first.post(url(project, "/paper"), json={"template": "iclr2025"}).status_code == 202
        wait_for(lambda: first.get(url(project)).json()["worker"]["state"] == "idle")
        wait_for(lambda: first.get(url(project, "/paper")).json()["last"] is not None)
        last = first.get(url(project, "/paper")).json()["last"]
    assert last["outcome"] == "refused" and last["exit_code"] == 1
    assert last["error"] == "There is nothing to write a paper from yet: no finding is accepted."
    assert last["summary"].startswith("The write-up did not start: its worker ended with exit")
    assert last["started_at"] is None and last["ended_at"]
    # Not from the server's memory, and not from the end of worker.log, which the next
    # worker writes over what this one said.
    (project.root / "worker.log").write_text("something else entirely\n")
    with serve() as second:
        assert second.get(url(project)).json()["worker"]["last_exit"] is None
        assert second.get(url(project, "/paper")).json()["last"] == last


def test_the_paper_events_are_read_once_and_then_only_what_was_added(client, project, monkeypatch):
    started = paper_started(project)
    assert client.get(url(project, "/paper")).json()["last"]["started_at"] == started["ts"]
    read: list[int] = []
    opened = api_module._open_plain

    def counting(path):
        handle = opened(path)
        if Path(path).name == "events.jsonl":
            read.append(os.fstat(handle.fileno()).st_size)
        return handle

    monkeypatch.setattr(api_module, "_open_plain", counting)
    assert client.get(url(project, "/paper")).json()["last"]["started_at"] == started["ts"]
    assert read == []  # nothing was added: the file is not opened again
    project.emit("agent.activity", "Something else happened.")
    client.get(url(project, "/paper"))
    assert len(read) == 1
    # A stream that was written anew (restored from a copy) is read from its beginning.
    (project.root / "events.jsonl").write_text("")
    assert client.get(url(project, "/paper")).json()["last"] is None


# -- the number check of a suggested edit, where the list decides ------------------------------


def test_a_suggested_edit_to_a_finding_is_listed_with_what_its_numbers_check_found(
    workspace, project, launched, live, tmp_path, monkeypatch
):
    services = Reporting()
    finding = a_finding(project)
    proposal = project.propose_edit(
        finding.path, author="agent:analyst", summary="Report the better run",
        body="Accuracy reached 97.3%, 41.8 points above the baseline.\n", round=1,
    )  # fmt: skip
    app = create_app(workspace, launcher=launched, services=services, live=live,
                     static_dir=tmp_path / "none", stop_workers_on_exit=True)  # fmt: skip
    with TestClient(app, base_url=BASE_URL) as client:
        rows = {
            row["id"]: row for row in client.get(url(project, "/proposals")).json()["proposals"]
        }
        # What the quick accept in the list decides on: the check of the text as it would
        # read, in the row itself, as the detail shows it.
        row = rows[proposal.id]
        assert row["number_report"]["unverified"] == ["97.3"]
        assert row["number_report_error"] is None
        detail = client.get(url(project, f"/pages/{finding.id}")).json()["proposals"][0]
        assert row["number_report"] == detail["number_report"]
        assert (
            services.reported[0].body == "Accuracy reached 97.3%, 41.8 points above the baseline.\n"
        )
        # An edit to a page that is no finding has no numbers to trace.
        other = next(row for key, row in rows.items() if key != proposal.id)
        assert "number_report" not in other

        # The listing spends a bounded time on it: an edit it does not get to says so by
        # carrying no report, and the app opens the detail before it decides.
        monkeypatch.setattr(api_module, "_ROW_CHECK_SECONDS", 0.0)
        del services.reported[:]
        late = client.get(url(project, "/proposals")).json()["proposals"]
        row = next(row for row in late if row["id"] == proposal.id)
        assert row["number_report"] is None and row["number_report_error"] is None
        assert services.reported == []
        monkeypatch.setattr(api_module, "_ROW_CHECK_SECONDS", 60.0)

        # A report that cannot be made is said, as in the detail.
        def broken(project, page):
            raise ProjectError("The evidence could not be read.")

        services.number_report = broken
        row = client.get(url(project, "/proposals")).json()["proposals"][-1]
        assert row["number_report"] is None
        assert row["number_report_error"] == "The evidence could not be read."

        # One that is decided has nothing left to check.
        services.number_report = Reporting().number_report
        client.post(
            url(project, "/review"), json={"items": [{"ref": proposal.id, "decision": "reject"}]}
        )
        decided = client.get(url(project, "/proposals?review=all")).json()["proposals"]
        assert "number_report" not in next(row for row in decided if row["id"] == proposal.id)


# ---------------------------------------------------------------------------
# How the agents are run (settings.agents), their lines in the feed, their runs
# ---------------------------------------------------------------------------

AGENT_RUNTIME_ENV = "PAPER_ORCHESTRA_AGENT_RUNTIME"


def test_the_defaults_offer_the_runtimes_and_credentials_the_settings_accept(client, monkeypatch):
    from paper_orchestra.research.model import ProjectSettings

    monkeypatch.delenv(AGENT_RUNTIME_ENV, raising=False)
    body = client.get("/api/defaults").json()
    assert body["agent_runtimes"] == ["classic", "sdk"]
    assert body["agent_credentials"] == ["login", "federation"]
    for runtime in body["agent_runtimes"]:
        for credential in body["agent_credentials"]:
            given = {"agents": {"runtime": runtime, "credential": credential}}
            agents = ProjectSettings.from_dict(given, strict=True).agents
            assert (agents.runtime, agents.credential) == (runtime, credential)
    # Every field a form shows has its default here: a client carries none of its own.
    assert body["settings"]["agents"] == {
        "runtime": "classic",
        "model": "claude-fable-5-2",
        "fast": True,
        "ultracode": True,
        "credential": "login",
        "sandbox": True,
    }


def test_a_new_project_gets_the_runtime_the_servers_environment_names(
    client, workspace, monkeypatch
):
    def created(settings: dict[str, Any] | None) -> dict[str, Any]:
        data = {"brief": "An idea."}
        if settings is not None:
            data["settings"] = json.dumps(settings)
        response = client.post("/api/projects", data=data)
        assert response.status_code == 201, response.text
        pid = response.json()["id"]
        shown = client.get(f"/api/projects/{pid}").json()["settings"]["agents"]
        # ... and it is written into the project: the environment of whoever opens it
        # later changes nothing.
        on_file = json.loads((workspace / response.json()["dir"] / "project.json").read_text())
        assert on_file["settings"]["agents"] == shown
        return shown

    monkeypatch.delenv(AGENT_RUNTIME_ENV, raising=False)
    assert created(None)["runtime"] == "classic"

    monkeypatch.setenv(AGENT_RUNTIME_ENV, "sdk")
    assert client.get("/api/defaults").json()["settings"]["agents"]["runtime"] == "sdk"
    assert created(None)["runtime"] == "sdk"
    # A form that says something else about the agents, and nothing about the runtime,
    # still gets the environment's.
    partly = created({"agents": {"fast": False, "credential": "federation"}})
    assert (partly["runtime"], partly["fast"], partly["credential"]) == ("sdk", False, "federation")
    # What the form names wins.
    assert created({"agents": {"runtime": "classic"}})["runtime"] == "classic"
    refused = client.post(
        "/api/projects",
        data={"brief": "x", "settings": json.dumps({"agents": {"runtime": "abacus"}})},
    )
    assert refused.status_code == 422 and "runtime" in error_of(refused)["message"]

    monkeypatch.setenv(AGENT_RUNTIME_ENV, "nonsense")
    assert created(None)["runtime"] == "classic"


def test_how_the_agents_are_run_can_be_changed_and_a_refused_change_changes_nothing(
    client, project, monkeypatch
):
    monkeypatch.delenv(AGENT_RUNTIME_ENV, raising=False)
    assert client.get(url(project)).json()["settings"]["agents"]["runtime"] == "classic"
    changed = client.patch(
        url(project),
        json={
            "settings": {
                "agents": {
                    "runtime": "sdk",
                    "model": "claude-haiku-5-5",
                    "fast": False,
                    "ultracode": False,
                    "credential": "federation",
                    "sandbox": False,
                }
            }
        },
    )
    assert changed.status_code == 200, changed.text
    wanted = {
        "runtime": "sdk",
        "model": "claude-haiku-5-5",
        "fast": False,
        "ultracode": False,
        "credential": "federation",
        "sandbox": False,
    }
    assert changed.json()["settings"]["agents"] == wanted
    assert Project.open(project.root).settings.agents.sdk
    # One field alone: the others stay.
    one = client.patch(url(project), json={"settings": {"agents": {"sandbox": True}}})
    assert one.json()["settings"]["agents"] == {**wanted, "sandbox": True}

    before = (project.root / "project.json").read_text()
    for wrong, named in (
        ({"runtime": "abacus"}, "runtime"),
        ({"credential": "password"}, "credential"),
        ({"fast": "yes"}, "fast"),
        ({"model": " "}, "model"),
        ({"sandbox": True, "ultracode": 1}, "ultracode"),
        # A name the group does not have: stored, it would silently do nothing.
        ({"bogus": 1}, "bogus"),
        ({"sandbx": False}, "sandbx"),
        # Not the name of a model: it goes to the agents' command line as it stands.
        ({"model": "claude-haiku-5-5 ; rm -rf"}, "model"),
    ):
        refused = client.patch(url(project), json={"settings": {"agents": wrong}})
        assert refused.status_code == 422, (wrong, refused.text)
        assert named in error_of(refused)["message"]
        assert (project.root / "project.json").read_text() == before
        assert client.get(url(project)).json()["settings"]["agents"] == {**wanted, "sandbox": True}


def test_the_person_can_make_the_code_an_agent_left_the_experiments_code(real, project):
    """An agent that carries a run out repairs the code in its own folder; the run's
    record names what it changed, and the person can take that code over as their edit."""
    from paper_orchestra.research import experiments

    page = second_experiment(project)
    eid = str(page.id)
    runs = project.experiment_dir(eid) / "runs" / "1"
    (runs / "code").mkdir(parents=True)
    (runs / "code" / "main.py").write_text("print('hello')\n")
    (runs / "code-as-run").mkdir()
    (runs / "code-as-run" / "main.py").write_text("print('hello, repaired')\n")
    (runs / "code-as-run" / "util.py").write_text("X = 1\n")
    record = {
        "experiment": eid,
        "number": 1,
        "status": "succeeded",
        "backend": "local",
        "carried_out_by": "agent",
        "exit_code": None,
        "code_as_run": {
            "digest": "d" * 64,
            "changed": ["main.py"],
            "added": ["util.py"],
            "removed": [],
        },
    }
    (runs / "run.json").write_text(json.dumps(record))
    shown = real.get(f"{url(project)}/experiments/{eid}/runs/1").json()["run"]
    assert shown["code_as_run"]["changed"] == ["main.py"] and shown["exit_code"] is None
    # The file as the agent left it can be opened from the run.
    kept = real.get(f"{url(project)}/files/experiments/{eid}/runs/1/code-as-run/main.py")
    assert kept.status_code == 200 and "repaired" in kept.text

    adopted = real.post(f"{url(project)}/experiments/{eid}/runs/1/adopt-code")
    assert adopted.status_code == 200, adopted.text
    assert adopted.json()["files"] == ["main.py", "util.py"] and adopted.json()["run"] == 1
    assert experiments.code_files(project, eid) == {
        "main.py": "print('hello, repaired')\n",
        "util.py": "X = 1\n",
    }
    assert experiments.code_authors(project, eid)["main.py"]["author"] == "human"
    # A run whose agent changed nothing has nothing to take over; neither has a run there is not.
    record["code_as_run"] = {"digest": "d" * 64, "changed": [], "added": [], "removed": []}
    (runs / "run.json").write_text(json.dumps(record))
    assert real.post(f"{url(project)}/experiments/{eid}/runs/1/adopt-code").status_code in (
        409,
        422,
    )
    assert real.post(f"{url(project)}/experiments/{eid}/runs/9/adopt-code").status_code == 404


def test_the_doctor_says_what_this_machine_has_for_agents_with_a_shell(
    workspace, monkeypatch, tmp_path
):
    """Setup shows it: the package and its command line, a login, the federation
    settings.  It is information; it never makes the report not ok, and no value of a
    setting is in it."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    for name in ("ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_IDENTITY_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-must-not-be-shown")
    with TestClient(create_app(workspace), base_url=BASE_URL) as client:
        report = client.get("/api/doctor").json()
        agents = report["agents"]
        assert agents["login"] in (False, None) and agents["federation"] is False
        assert "ANTHROPIC_FEDERATION_RULE_ID" in agents["federation_missing"]
        assert isinstance(agents["package"], bool) and isinstance(agents["command_line"], bool)
        assert "org-must-not-be-shown" not in json.dumps(report)
        (home / ".claude" / ".credentials.json").write_text("{}")
        assert client.get("/api/doctor").json()["agents"]["login"] is True


def test_the_feed_carries_the_parts_of_an_agents_line(client, project):
    """What an agent with a shell did reaches the app as the sentence and as its parts
    (``data``), over the listing and over the stream alike; a credential is in neither."""
    project.emit(
        "agent.activity",
        "The scout ran “ls -la && cat main.py” in its shell.",
        phase="research",
        agent="scout",
        round=1,
        data={"did": "shell", "text": "ls -la && cat main.py"},
    )
    project.emit(
        "agent.activity",
        "The scout read wiki/idea.md.",
        phase="research",
        agent="scout",
        round=1,
    )
    told, plain = client.get(url(project, "/activity?limit=2")).json()["events"]
    assert told["agent"] == "scout" and told["summary"].startswith("The scout ran")
    assert told["data"] == {"did": "shell", "text": "ls -la && cat main.py"}
    # A line of the tool loop, and one written before lines had parts, has none.
    assert not plain["data"]


def test_a_run_an_agent_carried_out_says_so_wherever_the_run_is_shown(client, project):
    run_dir = add_run(project)
    record = json.loads((run_dir / "run.json").read_text())
    by_hand = client.get(url(project, "/experiments/E1")).json()["runs"][0]
    assert "carried_out_by" not in by_hand  # a run of the runner's says nothing of the kind

    (run_dir / "run.json").write_text(json.dumps({**record, "carried_out_by": "agent"}))
    view = client.get(url(project, "/experiments/E1")).json()
    assert view["runs"][0]["carried_out_by"] == "agent"
    # ... with its log and what it left, like any run.
    assert view["runs"][0]["log_url"] and view["runs"][0]["artifact_urls"]
    log = client.get(url(project, "/experiments/E1/runs/1/log")).json()
    assert "done" in log["text"]
    (row,) = client.get(url(project, "/experiments")).json()["experiments"]
    assert row["last_run"]["carried_out_by"] == "agent"
