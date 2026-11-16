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
    # What was added to it: running an approved experiment now, and the worker's log.
    assert served - set(SPECIFIED.split("\n")) == {
        "POST /api/projects/{pid}/experiments/{eid}/run",
        "GET /api/projects/{pid}/worker/log",
    }


def test_health_says_what_is_running(client, workspace):
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["name"] == "PaperOrchestra"
    assert body["workspace"] == str(workspace)


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
    assert client.get("/api/projects").json() == {"projects": []}
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
    assert listed[0]["worker"] == "idle" and listed[0]["pending"] is None
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


def test_the_library_reads_no_page(client, workspace, monkeypatch):
    for number in range(40):
        made = Project.create(workspace / f"p{number:02}", title=f"Project {number}", brief="x")
        if number % 2:
            state = {"status": "checkpoint", "round": 2, "phase": "hypothesize",
                     "next_phase": "design", "spent": {"model_usd": 1.5},
                     "checkpoint": {"round": 2, "phase": "hypothesize", "summary": "Five waiting.",
                                    "pending": {"hypothesis": 5, "question": 1}}}  # fmt: skip
            (made.root / "state.json").write_text(json.dumps(state))

    def no_pages(self, *args, **kwargs):
        raise AssertionError("the library must not read pages")

    monkeypatch.setattr(Project, "_all_pages", no_pages)
    monkeypatch.setattr(Project, "_load_page", no_pages)
    rows = client.get("/api/projects").json()["projects"]
    assert len(rows) == 40
    waiting = [row for row in rows if row["status"] == "checkpoint"]
    assert len(waiting) == 20 and all(row["pending"] == 6 for row in waiting)
    assert waiting[0]["checkpoint"]["summary"] == "Five waiting."
    assert waiting[0]["spent"] == {
        "model_usd": 1.5,
        "experiment_seconds": 0.0,
        "experiment_usd": 0.0,
    }

    opened = []
    monkeypatch.setattr(Project, "reload", lambda self: opened.append(self.root))
    client.get("/api/projects")
    assert opened == []  # nothing changed, so nothing was read again


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


def test_editing_an_agents_proposal_accepts_it_as_the_persons(client, project):
    edited = client.put(url(project, "/pages/H1"), json={"body": "## Statement\n\nSharper.\n"})
    assert edited.status_code == 200
    page = edited.json()["page"]
    assert page["review"] == "accepted" and page["meta"]["decided_by"] == "human"


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


def test_cancel_asks_the_running_job_to_stop(client, project):
    response = client.post(url(project, "/experiments/E1/cancel"))
    assert response.status_code == 202 and response.json() == {"cancelling": "E1"}
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
            "headers": [(k.encode(), v.encode()) for k, v in self.headers.items()],
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
        refused = client.post(url(project, "/experiments/E1/cancel"))
        assert refused.status_code == 400 and error_of(refused)["code"] == "not_available"


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
