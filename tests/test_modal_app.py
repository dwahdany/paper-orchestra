from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from paper_orchestra.compute.base import ComputeError
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.compute.modal_backend import ModalBackend
from paper_orchestra.config import DEFAULT_MODEL, LAB_MEMORY_ENV, TEMPLATES_ENV
from paper_orchestra.context import RunPaths
from paper_orchestra.llm.base import LLMConfigError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.llm.router import register_client, reset_clients
from paper_orchestra.memory.harness import ContextHarness
from paper_orchestra.memory.wiki import Wiki

modal = pytest.importorskip("modal")

import paper_orchestra.modal_app as modal_app  # noqa: E402
from paper_orchestra.modal_app import (  # noqa: E402
    JobError,
    Settings,
    StageCheckpoint,
    batch_results,
    build_job,
    check_batch,
    check_credentials,
    commit_volume,
    consolidate_lab,
    container_path,
    download_outputs,
    failure_summary,
    function_env,
    job_spec,
    launch,
    load_settings,
    new_run_id,
    run_job,
    upload_inputs,
    validate_run_id,
    volume_path,
    wait_for_result,
)

WIF_IDS = {
    "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_123",
    "ANTHROPIC_ORGANIZATION_ID": "org-uuid",
}


class FakeBatch:
    def __init__(self, volume: FakeVolume) -> None:
        self._volume = volume

    def put_file(self, local_file: Path, remote_path: str, mode: int | None = None) -> None:
        assert remote_path.startswith("/"), "uploads name their target from the volume root"
        target = self._volume.root / remote_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_file, target)
        self._volume.uploaded.append(remote_path)


class FakeVolume:
    """A `modal.Volume` whose contents are a local directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.uploaded: list[str] = []
        self.forced: list[bool] = []
        self.commits = 0
        self.reloads = 0
        self.commit_errors: list[Exception] = []
        self.reload_error: Exception | None = None

    @contextmanager
    def batch_upload(self, force: bool = False):
        self.forced.append(force)
        yield FakeBatch(self)

    def read_file(self, path: str):
        target = self.root / path
        if not target.is_file():
            raise FileNotFoundError(path)
        data = target.read_bytes()
        for start in range(0, len(data), 5):
            yield data[start : start + 5]

    def commit(self) -> None:
        self.commits += 1
        if self.commit_errors:
            raise self.commit_errors.pop(0)

    def reload(self) -> None:
        self.reloads += 1
        if self.reload_error is not None:
            raise self.reload_error


class FakeCall:
    object_id = "fc-fake"

    def __init__(self, run, pending: int) -> None:
        self._run = run
        self._pending = pending
        self.timeouts: list[float | None] = []

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        self.timeouts.append(timeout)
        if self._pending:
            self._pending -= 1
            raise TimeoutError
        return self._run()


class FakeFunction:
    """`write_paper` as seen from the launcher: spawn runs the job against the fake volume."""

    def __init__(self, volume: FakeVolume, pending: int = 2) -> None:
        self._volume = volume
        self._pending = pending
        self.jobs: list[dict[str, Any]] = []
        self.calls: list[FakeCall] = []

    def spawn(self, job: dict[str, Any]) -> FakeCall:
        self.jobs.append(job)
        wire = json.loads(json.dumps(job))  # a job crosses a process boundary on Modal
        call = FakeCall(
            lambda: run_job(
                wire,
                commit=self._volume.commit,
                reload=self._volume.reload,
                mount=self._volume.root,
            ),
            self._pending,
        )
        self.calls.append(call)
        return call


class StubPipeline:
    """Stands in for `paper_orchestra.pipeline`: copies inputs, resumes from state.json.

    Like the real pipeline it refreshes the harness when a stage has settled,
    on a worker thread for the stages that run side by side.
    """

    def __init__(self) -> None:
        self.prepared: list[tuple[Any, Path | None]] = []
        self.runs: list[dict[str, Any]] = []
        self.success = True
        self.error: Exception | None = None
        #: Called after each stage has settled; its value is recorded in ``checkpoints``.
        self.observe = lambda: None
        self.checkpoints: list[tuple[str, Any]] = []
        #: Overrides the stages that settle.
        self.only_stages: list[str] | None = None
        self.refresh_errors: list[Exception] = []

    def prepare_run_dir(self, cfg, raw_materials_dir):
        self.prepared.append((cfg, raw_materials_dir))
        paths = RunPaths(cfg.base_dir, cfg.idea_filename, cfg.experimental_log_filename)
        if raw_materials_dir is not None:
            shutil.copytree(raw_materials_dir, paths.materials_dir, dirs_exist_ok=True)
        if not paths.idea.is_file():
            raise FileNotFoundError(f"required file not found: {paths.idea}")
        return paths

    def stages(self, cfg) -> list[str]:
        if self.only_stages is not None:
            return self.only_stages
        middle = ["literature", "plotting"] if cfg.use_plotting else ["literature"]
        return ["ingest", "outline", *middle, "section_writing", "refinement", "finalize"]

    def settle(self, name: str, cfg, harness) -> None:
        def refresh() -> None:
            try:
                harness.refresh()
            except Exception as exc:  # the real pipeline guards its memory upkeep too
                self.refresh_errors.append(exc)

        if cfg.use_plotting and name in ("literature", "plotting"):
            with ThreadPoolExecutor(1, thread_name_prefix="stage") as pool:
                pool.submit(refresh).result()
        else:
            refresh()
        self.checkpoints.append((name, self.observe()))

    def run_paper(self, cfg, *, harness=None, compute=None):
        paths = RunPaths(cfg.base_dir, cfg.idea_filename, cfg.experimental_log_filename)
        self.runs.append(
            {
                "cfg": cfg,
                "harness": harness,
                "compute": compute,
                "env": {
                    name: os.environ.get(name)
                    for name in (LAB_MEMORY_ENV, "openin_any", "ANTHROPIC_FEDERATION_RULE_ID")
                },
            }
        )
        if self.error is not None:
            raise self.error
        resumed = cfg.resume and paths.state.is_file()
        attempts = json.loads(paths.state.read_text())["attempts"] if resumed else 0
        paths.state.write_text(json.dumps({"attempts": attempts + 1}))
        if self.success:
            paths.final_pdf.write_bytes(b"%PDF-1.5 " + paths.idea.read_bytes())
            template = (cfg.template_dir / "template.tex").read_text()
            paths.final_tex.write_text(f"% built from: {template}")
            paths.usage.write_text(json.dumps({"total": {"requests": 3}}))
        for name in self.stages(cfg):
            self.settle(name, cfg, harness)
        return types.SimpleNamespace(
            success=self.success,
            pdf_path=paths.final_pdf if self.success else None,
            base_dir=cfg.base_dir,
            stages={"outline": {"status": "done", "attempts": attempts + 1, "error": None}},
            usage={"total": {"requests": 3}, "report": paths.usage},
            error=None if self.success else "stage refinement failed: boom",
        )


@pytest.fixture
def pipeline(monkeypatch) -> StubPipeline:
    stub = StubPipeline()
    module = types.ModuleType("paper_orchestra.pipeline")
    module.prepare_run_dir = stub.prepare_run_dir
    module.run_paper = stub.run_paper
    monkeypatch.setitem(sys.modules, "paper_orchestra.pipeline", module)
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: True))
    monkeypatch.setattr(modal_app, "_COMMIT_RETRY_DELAY", 0.0)
    # run_job sets these for the pipeline; registering them here restores them afterwards.
    monkeypatch.setenv(LAB_MEMORY_ENV, "before")
    monkeypatch.setenv("openin_any", "before")
    return stub


@pytest.fixture
def volume(tmp_path) -> FakeVolume:
    return FakeVolume(tmp_path / "vol")


@pytest.fixture
def logs(caplog, monkeypatch):
    """Records of `paper_orchestra.*`, even if an earlier test stopped the logger propagating."""
    monkeypatch.setattr(logging.getLogger("paper_orchestra"), "propagate", True)
    caplog.set_level(logging.INFO, logger="paper_orchestra")
    return caplog


def stage_inputs(volume: FakeVolume, run_id: str = "run-1") -> dict[str, Any]:
    """Put a run's inputs on the fake volume and return the job that uses them."""
    inputs = volume.root / "inputs" / run_id
    (inputs / "raw_materials").mkdir(parents=True)
    (inputs / "template").mkdir()
    (inputs / "raw_materials" / "idea_sparse.md").write_text("the idea")
    (inputs / "raw_materials" / "experimental_log.md").write_text("the log")
    (inputs / "template" / "template.tex").write_text("TEMPLATE")
    return build_job(
        run_id=run_id,
        raw_materials_dir=f"inputs/{run_id}/raw_materials",
        template_dir=f"inputs/{run_id}/template",
        research_cutoff="2025-06",
    )


def local_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    (raw / "figures").mkdir(parents=True)
    (raw / ".git").mkdir()
    (raw / "idea_sparse.md").write_text("local idea")
    (raw / "experimental_log.md").write_text("local log")
    (raw / "figures" / "plot.png").write_bytes(b"\x89PNG")
    (raw / ".env").write_text("SECRET=1")
    (raw / ".git" / "config").write_text("x")
    template = tmp_path / "templates" / "mytemplate"
    template.mkdir(parents=True)
    (template / "template.tex").write_text("MY TEMPLATE")
    (template / "guidelines.md").write_text("Eight pages.")
    (template / "style.sty").write_text("% style")
    return raw, template


# -- settings and credentials -------------------------------------------------------


def test_settings_defaults_and_overrides():
    assert load_settings({}) == Settings(
        app_name="paper-orchestra",
        volume_name="paper-orchestra",
        secret_name="paper-orchestra",
        timeout=6 * 3600,
        max_containers=4,
    )
    custom = load_settings(
        {
            "PAPER_ORCHESTRA_MODAL_APP": "my.app",
            "PAPER_ORCHESTRA_MODAL_VOLUME": "papers_v2",
            "PAPER_ORCHESTRA_MODAL_SECRET": " anthropic-wif ",
            "PAPER_ORCHESTRA_MODAL_TIMEOUT": "7200",
            "PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS": "2",
        }
    )
    assert custom == Settings("my.app", "papers_v2", "anthropic-wif", 7200, 2)
    for spelling in ("none", "NONE", "", "  "):
        assert load_settings({"PAPER_ORCHESTRA_MODAL_SECRET": spelling}).secret_name is None


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("PAPER_ORCHESTRA_MODAL_APP", "bad name", "not a valid Modal name"),
        ("PAPER_ORCHESTRA_MODAL_VOLUME", "a/b", "not a valid Modal name"),
        ("PAPER_ORCHESTRA_MODAL_SECRET", "x" * 65, "not a valid Modal name"),
        ("PAPER_ORCHESTRA_MODAL_TIMEOUT", "2h", "is not an integer"),
        ("PAPER_ORCHESTRA_MODAL_TIMEOUT", "90000", "must be between 60 and 86400"),
        ("PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS", "0", "must be between 1 and 1000"),
    ],
)
def test_invalid_settings_name_the_variable(variable, value, message):
    with pytest.raises(ValueError, match=message) as raised:
        load_settings({variable: value})
    assert variable in str(raised.value)


def test_function_env_forwards_ids_but_never_secrets():
    settings = load_settings({"PAPER_ORCHESTRA_MODAL_SECRET": "none"})
    local = {
        **WIF_IDS,
        "ANTHROPIC_WORKSPACE_ID": "wrkspc_1",
        "ANTHROPIC_SERVICE_ACCOUNT_ID": "",
        "PAPER_ORCHESTRA_STRUCTURED_OUTPUTS": "0",
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "ANTHROPIC_AUTH_TOKEN": "token",
        "ANTHROPIC_IDENTITY_TOKEN": "jwt",
        "SEMANTIC_SCHOLAR_API_KEY": "s2",
        "MODAL_TOKEN_SECRET": "as-secret",
    }
    env = function_env(settings, local)
    assert env == {
        "PAPER_ORCHESTRA_MODAL_APP": "paper-orchestra",
        "PAPER_ORCHESTRA_MODAL_VOLUME": "paper-orchestra",
        "PAPER_ORCHESTRA_MODAL_SECRET": "none",
        "PAPER_ORCHESTRA_MODAL_TIMEOUT": "21600",
        "PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS": "4",
        "PAPER_ORCHESTRA_MEMORY_DIR": "/vol/memory",
        "PAPER_ORCHESTRA_TEMPLATES": "/root/templates",
        **WIF_IDS,
        "ANTHROPIC_WORKSPACE_ID": "wrkspc_1",
        "PAPER_ORCHESTRA_STRUCTURED_OUTPUTS": "0",
    }
    assert all(value for value in env.values())  # Modal may drop empty values


@pytest.mark.parametrize(
    "local",
    [
        {},
        {"PAPER_ORCHESTRA_MODAL_SECRET": "none", "PAPER_ORCHESTRA_MODAL_APP": "other-app"},
        {
            "PAPER_ORCHESTRA_MODAL_SECRET": "wif-ids",
            "PAPER_ORCHESTRA_MODAL_VOLUME": "v",
            "PAPER_ORCHESTRA_MODAL_TIMEOUT": "600",
            "PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS": "9",
        },
    ],
)
def test_the_container_resolves_the_same_settings_as_the_launcher(local):
    """Modal rejects a Function whose Secrets differ between the launcher and the container."""
    settings = load_settings(local)
    assert load_settings(function_env(settings, local)) == settings


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, "none"),
        ({"ANTHROPIC_API_KEY": "sk"}, "api_key"),
        ({"ANTHROPIC_AUTH_TOKEN": "t"}, "auth_token"),
        ({"ANTHROPIC_API_KEY": "sk", "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_1"}, "api_key"),
        ({**WIF_IDS, "MODAL_IDENTITY_TOKEN_FILE": "/run/token"}, "workload_identity"),
        ({**WIF_IDS, "MODAL_IDENTITY_TOKEN": "jwt"}, "workload_identity"),
        ({**WIF_IDS, "ANTHROPIC_IDENTITY_TOKEN_FILE": "/t"}, "workload_identity"),
    ],
)
def test_credential_mode(environ, expected):
    assert check_credentials(environ) == expected


@pytest.mark.parametrize(
    "static", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"]
)
def test_a_static_credential_next_to_wif_is_refused(static):
    environ = {**WIF_IDS, "MODAL_IDENTITY_TOKEN_FILE": "/run/token", static: "sk-ant-secret"}
    with pytest.raises(JobError, match="exactly one credential") as raised:
        check_credentials(environ)
    assert static in str(raised.value) and "sk-ant-secret" not in str(raised.value)


def test_wif_without_a_modal_identity_token_is_refused():
    with pytest.raises(JobError, match="no identity token"):
        check_credentials(WIF_IDS)


# -- jobs and paths -------------------------------------------------------------------


def test_run_ids():
    assert new_run_id(datetime(2026, 11, 15, 9, 30, 5, tzinfo=UTC)).startswith("20261115-093005-")
    assert validate_run_id(new_run_id()) and new_run_id() != new_run_id()
    for good in ("a", "cvpr2025_0b62029e18", "run.1-b", "9" * 80):
        assert validate_run_id(good) == good
    for bad in ("", ".hidden", "../up", "a/b", "a b", "-x", "x" * 81, None, 7):
        with pytest.raises(JobError, match="run_id"):
            validate_run_id(bad)


def test_volume_and_container_paths(tmp_path):
    inside = container_path("runs/r1/final_paper.pdf", tmp_path)
    assert inside == tmp_path / "runs" / "r1" / "final_paper.pdf"
    assert volume_path(inside, tmp_path) == "runs/r1/final_paper.pdf"
    for bad in ("/etc/passwd", "runs/../../etc", "", ".", 5, None):
        with pytest.raises(JobError, match="not a path inside the volume"):
            container_path(bad, tmp_path)
    with pytest.raises(ValueError):
        volume_path(Path("/elsewhere/file"), tmp_path)


def test_build_job_keeps_only_what_was_asked_for():
    job = build_job(run_id="r1", template="cvpr2025", research_cutoff="2024-11")
    assert job == {
        "run_id": "r1",
        "template": "cvpr2025",
        "idea_filename": "idea_sparse.md",
        "experimental_log_filename": "experimental_log.md",
        "research_cutoff": "2024-11",
        "use_plotting": False,
        "resume": True,
        "models": {},
        "memory": {"enabled": True, "lab": True, "dream": "stages"},
    }
    full = build_job(
        run_id="r2",
        template_dir="inputs/r2/template",
        raw_materials_dir="inputs/r2/raw_materials",
        idea_filename="idea.md",
        experimental_log_filename="log.md",
        use_plotting=True,
        plotting_max_critic_rounds=5,
        writer_model="claude-fable-5-2",
        reflection_model="gemini-3.1-pro-preview",
        fast_model="claude-haiku-4-5",
        image_model="gemini-3-pro-image-preview",
        effort="xhigh",
        memory=False,
        lab_memory=False,
        dream="off",
        resume=False,
    )
    assert full["models"] == {
        "writer": "claude-fable-5-2",
        "reflection": "gemini-3.1-pro-preview",
        "fast": "claude-haiku-4-5",
        "image": "gemini-3-pro-image-preview",
        "effort": "xhigh",
    }
    assert full["memory"] == {"enabled": False, "lab": False, "dream": "off"}
    assert full["plotting_max_critic_rounds"] == 5 and full["use_plotting"] is True
    assert "template" not in full and full["resume"] is False
    assert json.loads(json.dumps(full)) == full
    # The cutoff is fixed when the job is built, so a retry next month sees the same one.
    assert build_job(run_id="r3", template="t")["research_cutoff"] == datetime.now().strftime(
        "%Y-%m"
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"effort": "extreme"}, "effort must be one of low, medium, high, xhigh, max"),
        ({"dream": "always"}, "dream must be one of off, end, stages"),
        ({"research_cutoff": "2024-13"}, "research_cutoff must look like 2024-11"),
        ({"research_cutoff": "Nov 2024"}, "research_cutoff must look like 2024-11"),
        ({"run_id": "../x"}, "run_id must be"),
    ],
)
def test_build_job_rejects_invalid_options(options, message):
    with pytest.raises(JobError, match=message):
        build_job(**{"run_id": "r1", "template": "cvpr2025", **options})


def test_job_spec_maps_a_job_onto_container_paths():
    spec = job_spec(build_job(run_id="r1", template="iclr2025", research_cutoff="2024-11"))
    cfg = spec.config
    assert spec.run_id == "r1" and spec.raw_materials_dir is None
    assert cfg.base_dir == Path("/vol/runs/r1")
    assert cfg.template_dir == Path("/root/templates/iclr2025")
    assert cfg.project_memory_dir == Path("/vol/runs/r1/memory")
    assert cfg.memory.resolved_lab_dir() == Path("/vol/memory")
    assert cfg.compute == "modal" and cfg.research_cutoff == "2024-11"
    assert cfg.models.writer == DEFAULT_MODEL and cfg.resume is True


def test_job_spec_applies_overrides_and_uploaded_inputs(tmp_path):
    job = build_job(
        run_id="r2",
        template_dir="inputs/r2/template",
        raw_materials_dir="inputs/r2/raw_materials",
        use_plotting=True,
        writer_model="claude-opus-4-8",
        effort="medium",
        lab_memory=False,
        dream="end",
    )
    job.update(max_reflections=1, review_ensemble=2, latex_timeout=90.0)
    spec = job_spec(job, tmp_path)
    cfg = spec.config
    assert cfg.base_dir == tmp_path / "runs" / "r2"
    assert cfg.template_dir == tmp_path / "inputs" / "r2" / "template"
    assert spec.raw_materials_dir == tmp_path / "inputs" / "r2" / "raw_materials"
    assert (cfg.models.writer, cfg.models.effort) == ("claude-opus-4-8", "medium")
    assert cfg.models.reflection == DEFAULT_MODEL
    assert cfg.use_plotting and cfg.memory.dream == "end"
    assert cfg.memory.resolved_lab_dir() is None
    assert (cfg.max_reflections, cfg.review_ensemble, cfg.latex_timeout) == (1, 2, 90.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"base_dir": "/tmp/elsewhere"}, "unknown job option"),
        ({"compute": "local"}, "unknown job option"),
        ({"typo_option": 1}, "unknown job option.*typo_option"),
        ({"template": None}, "exactly one of 'template'"),
        ({"template_dir": "inputs/r/template"}, "exactly one of 'template'"),
        ({"template": "../../etc"}, "must be the name of a bundled template"),
        ({"template": None, "template_dir": "/abs"}, "not a path inside the volume"),
        ({"raw_materials_dir": "a/../../b"}, "not a path inside the volume"),
        ({"run_id": "a/b"}, "run_id must be"),
        ({"memory": {"lab_dir": "/host/memory"}}, r"memory\.lab_dir cannot be set"),
        ({"memory": {"project_dir": "/x", "lab_dir": "/y"}}, r"memory\.lab_dir and memory\.proj"),
        ({"memory": "on"}, "'memory' must be a mapping"),
        ({"models": {"wrtier": "claude-fable-5-2"}}, "invalid job options.*wrtier"),
        ({"memory": {"dreams": "off"}}, "invalid job options.*dreams"),
    ],
)
def test_job_spec_rejects_what_the_container_cannot_honour(change, message):
    job = {"run_id": "r1", "template": "cvpr2025", **change}
    job = {key: value for key, value in job.items() if value is not None}
    with pytest.raises(JobError, match=message):
        job_spec(job)


def test_job_spec_rejects_non_mappings():
    with pytest.raises(JobError, match="a job must be a dict, got list"):
        job_spec(["run_id"])


# -- running a job --------------------------------------------------------------------


def test_run_job_runs_the_pipeline_on_the_volume(pipeline, volume, monkeypatch, logs):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    job = stage_inputs(volume)
    pipeline.observe = lambda: volume.commits
    summary = run_job(job, commit=volume.commit, reload=volume.reload, mount=volume.root)

    assert summary == {
        "success": True,
        "run_id": "run-1",
        "error": None,
        "volume": modal_app.SETTINGS.volume_name,
        "run_dir": "runs/run-1",
        "files": {
            "pdf": "runs/run-1/final_paper.pdf",
            "tex": "runs/run-1/final_paper.tex",
            "usage": "runs/run-1/usage.json",
        },
        "stages": {"outline": {"status": "done", "attempts": 1, "error": None}},
        "usage": {
            "total": {"requests": 3},
            "report": str(volume.root / "runs/run-1/usage.json"),
        },
        "seconds": summary["seconds"],
    }
    assert json.loads(json.dumps(summary)) == summary and summary["seconds"] >= 0
    assert (volume.root / "runs/run-1/final_paper.pdf").read_bytes() == b"%PDF-1.5 the idea"
    # One commit as each stage settles, and a last one before the summary is returned.
    assert pipeline.checkpoints == [
        ("ingest", 1),
        ("outline", 2),
        ("literature", 3),
        ("section_writing", 4),
        ("refinement", 5),
        ("finalize", 6),
    ]
    assert (volume.reloads, volume.commits) == (1, 7)

    [(prepared_cfg, raw_materials_dir)] = pipeline.prepared
    assert raw_materials_dir == volume.root / "inputs/run-1/raw_materials"
    [run] = pipeline.runs
    cfg = run["cfg"]
    assert cfg is prepared_cfg and cfg.base_dir == volume.root / "runs/run-1"
    assert cfg.template_dir == volume.root / "inputs/run-1/template"
    assert cfg.compute == "modal" and cfg.research_cutoff == "2025-06"
    assert cfg.memory.resolved_lab_dir() == volume.root / "memory"
    # LaTeX is compiled in the container, figure code in a nested sandbox.
    assert isinstance(run["compute"], ModalBackend) and run["compute"].latex == "local"
    assert run["env"][LAB_MEMORY_ENV] == str(volume.root / "memory")
    assert run["env"]["openin_any"] == "p"
    assert "Anthropic credential in use: api_key." in logs.text
    assert "sk-ant-secret" not in logs.text and "sk-ant-secret" not in json.dumps(summary)


def test_run_job_is_resumable(pipeline, volume):
    job = stage_inputs(volume)
    first = run_job(job, commit=volume.commit, mount=volume.root)
    again = run_job(job, commit=volume.commit, mount=volume.root)
    assert first["stages"]["outline"]["attempts"] == 1
    assert again["stages"]["outline"]["attempts"] == 2  # continued from state.json
    assert again["success"] and again["files"] == first["files"]
    assert volume.commits == 2 * (6 + 1) and volume.reloads == 0

    fresh = run_job({**job, "resume": False}, commit=volume.commit, mount=volume.root)
    assert fresh["stages"]["outline"]["attempts"] == 1


def test_run_job_reports_a_failed_pipeline(pipeline, volume):
    pipeline.success = False
    summary = run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and summary["error"] == "stage refinement failed: boom"
    assert summary["files"] == {} and summary["run_dir"] == "runs/run-1"
    assert volume.commits == 6 + 1  # partial progress is still made durable


def test_run_job_keyless_with_modal_identity(pipeline, volume, monkeypatch, logs):
    for name, value in {**WIF_IDS, "MODAL_IDENTITY_TOKEN_FILE": "/run/modal/token"}.items():
        monkeypatch.setenv(name, value)
    summary = run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert summary["success"]
    assert pipeline.runs[0]["env"]["ANTHROPIC_FEDERATION_RULE_ID"] == "fdrl_123"
    assert "Anthropic credential in use: workload_identity." in logs.text


def test_run_job_warns_when_no_anthropic_credential_is_configured(pipeline, volume, logs):
    assert run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)["success"]
    assert "No Anthropic credential is configured" in logs.text


def test_run_job_refuses_an_api_key_in_keyless_mode(pipeline, volume, monkeypatch):
    for name, value in {
        **WIF_IDS,
        "MODAL_IDENTITY_TOKEN_FILE": "/run/modal/token",
        "ANTHROPIC_API_KEY": "sk-ant-secret",
    }.items():
        monkeypatch.setenv(name, value)
    summary = run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and summary["run_id"] == "run-1"
    assert (
        "ANTHROPIC_API_KEY is set next to the Workload Identity Federation ids" in summary["error"]
    )
    assert "sk-ant-secret" not in json.dumps(summary)
    assert pipeline.prepared == [] and pipeline.runs == []
    assert volume.commits == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"run_id": "../escape"}, "run_id must be"),
        ({"unknown": 1}, "unknown job option"),
        ({"template_dir": "inputs/run-1/no-such-template"}, "LaTeX template not found"),
        ({"raw_materials_dir": "inputs/run-1/empty"}, "required file not found"),
    ],
)
def test_run_job_returns_unfixable_problems_instead_of_raising(pipeline, volume, change, message):
    job = stage_inputs(volume)
    (volume.root / "inputs/run-1/empty").mkdir()
    summary = run_job({**job, **change}, commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and message in summary["error"]
    assert summary["files"] == {} and pipeline.runs == []
    assert json.loads(json.dumps(summary)) == summary
    assert volume.commits == 1


def test_run_job_rejects_a_job_that_is_not_a_dict(pipeline, volume):
    summary = run_job(None, commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and summary["run_id"] is None
    assert "a job must be a dict" in summary["error"]


def test_run_job_checks_for_tex_before_any_model_call(pipeline, volume, monkeypatch):
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: False))
    summary = run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and "pdflatex and bibtex" in summary["error"]
    assert pipeline.prepared == [] and pipeline.runs == []


def test_run_job_lets_crashes_propagate_so_modal_retries(pipeline, volume):
    pipeline.error = RuntimeError("worker died")
    with pytest.raises(RuntimeError, match="worker died"):
        run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert volume.commits == 1

    volume.commit_errors = [RuntimeError("volume busy")]
    with pytest.raises(RuntimeError, match="worker died"):
        run_job(stage_inputs(volume, "run-2"), commit=volume.commit, mount=volume.root)


def test_checkpoints_wait_for_stages_that_run_side_by_side(pipeline, volume):
    job = {**stage_inputs(volume), "use_plotting": True}
    pipeline.observe = lambda: volume.commits
    assert run_job(job, commit=volume.commit, mount=volume.root)["success"]
    # No commit while a sibling stage may still be writing files; the next boundary has it.
    assert pipeline.checkpoints == [
        ("ingest", 1),
        ("outline", 2),
        ("literature", 2),
        ("plotting", 2),
        ("section_writing", 3),
        ("refinement", 4),
        ("finalize", 5),
    ]
    assert volume.commits == 5 + 1


def test_stage_checkpoint_commits_on_the_driving_thread_only_and_never_raises(logs):
    commits: list[str] = []
    checkpoint = StageCheckpoint(lambda: commits.append("commit"))
    worker = threading.Thread(target=checkpoint)
    worker.start()
    worker.join()
    assert commits == []
    checkpoint()
    assert commits == ["commit"] and "Committed the volume between stages." in logs.text

    def busy() -> None:
        raise RuntimeError("there are open files preventing the operation")

    StageCheckpoint(busy)()
    assert "Could not commit the volume between stages: there are open files" in logs.text


def test_the_pipeline_gets_a_harness_that_checkpoints_after_every_refresh(
    pipeline, volume, monkeypatch
):
    job = stage_inputs(volume)
    assert run_job(job, commit=volume.commit, mount=volume.root)["success"]
    harness = pipeline.runs[0]["harness"]
    assert isinstance(harness, ContextHarness)
    # The harness the pipeline would have built itself: both wikis live on the volume.
    assert harness.project.root == volume.root / "runs/run-1/memory"
    assert harness.lab.root == volume.root / "memory"
    assert harness.consolidate_lab is False  # consolidate_memory is the one writer of lab pages
    assert (harness.project.root / "MEMORY.md").is_file()  # the real refresh ran as well

    def broken(self) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ContextHarness, "refresh", broken)
    before = volume.commits
    assert run_job(job, commit=volume.commit, mount=volume.root)["success"]
    assert [type(error) for error in pipeline.refresh_errors] == [OSError] * 6
    assert volume.commits == before + 6 + 1  # a failed refresh does not cost the checkpoint

    assert run_job(
        {**job, "memory": {"enabled": False}}, commit=volume.commit, mount=volume.root
    )["success"]
    assert pipeline.runs[-1]["harness"].enabled is False


def test_run_job_tolerates_failed_checkpoints_and_reloads(pipeline, volume, logs):
    volume.reload_error = RuntimeError("there are open files preventing the operation")
    volume.commit_errors = [RuntimeError("volume busy")]
    job = stage_inputs(volume)
    assert run_job(job, commit=volume.commit, reload=volume.reload, mount=volume.root)["success"]
    assert "Could not reload the volume" in logs.text
    assert "Could not commit the volume between stages: volume busy" in logs.text
    assert volume.commits == 6 + 1


def test_run_job_insists_on_the_final_commit(pipeline, volume):
    pipeline.only_stages = []
    job = stage_inputs(volume)
    volume.commit_errors = [RuntimeError("busy 1"), RuntimeError("busy 2")]
    assert run_job(job, commit=volume.commit, mount=volume.root)["success"]
    assert volume.commits == 3

    volume.commit_errors = [RuntimeError(f"busy {n}") for n in range(3)]
    with pytest.raises(RuntimeError, match="busy 2"):
        run_job(job, commit=volume.commit, mount=volume.root)


@pytest.mark.parametrize(
    "error",
    [
        LLMConfigError("the writer model 'claude-fable-5-2' cannot be used: no credentials"),
        ComputeError("could not start a Modal sandbox"),
        ValueError("The idea file is empty"),
    ],
)
def test_run_job_does_not_retry_configuration_errors_from_the_pipeline(pipeline, volume, error):
    pipeline.error = error
    summary = run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and summary["error"] == str(error)
    assert volume.commits == 1


def test_commit_volume_retries_then_gives_up(monkeypatch, logs):
    monkeypatch.setattr(modal_app, "_COMMIT_RETRY_DELAY", 0.0)
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise RuntimeError("open files")

    commit_volume(flaky)
    assert len(attempts) == 3 and logs.text.count("Committing the volume failed") == 2

    def broken() -> None:
        raise OSError("no connection")

    with pytest.raises(OSError, match="no connection"):
        commit_volume(broken)


def test_failure_summary_has_the_shape_of_a_summary():
    summary = failure_summary("r1", "boom", 1.26)
    assert summary == {
        "success": False,
        "run_id": "r1",
        "error": "boom",
        "volume": modal_app.SETTINGS.volume_name,
        "run_dir": None,
        "files": {},
        "stages": {},
        "usage": {},
        "seconds": 1.3,
    }
    assert failure_summary({"not": "a string"}, "bad")["run_id"] is None


# -- batches --------------------------------------------------------------------------


def test_batch_helpers():
    jobs = [{"run_id": "a"}, {"run_id": "b"}, {"run_id": "c"}]
    check_batch(jobs)
    with pytest.raises(JobError, match="'a' appears more than once"):
        check_batch([*jobs, {"run_id": "a"}])
    with pytest.raises(JobError, match="run_id must be"):
        check_batch([{"template": "cvpr2025"}])
    with pytest.raises(JobError, match="run_id must be"):
        check_batch(["a"])

    ok = {"success": True, "run_id": "a"}
    results = batch_results(jobs, iter([ok, TimeoutError("took too long"), KeyError("x")]))
    assert results[0] is ok
    assert results[1]["success"] is False and results[1]["run_id"] == "b"
    assert results[1]["error"] == "TimeoutError: took too long"
    assert results[2]["error"] == "KeyError: 'x'" and results[2]["run_id"] == "c"
    with pytest.raises(ValueError):
        batch_results(jobs, [ok])


def test_write_batch_fans_out_and_returns_exceptions_as_failures(monkeypatch):
    monkeypatch.setattr(modal_app, "setup_logging", lambda **_: None)
    mapped: list[tuple] = []

    class FakeWritePaper:
        def map(self, jobs, **kwargs):
            mapped.append((jobs, kwargs))
            for job in jobs:
                if job["run_id"] == "bad":
                    yield RuntimeError("container crashed")
                else:
                    yield {"success": True, "run_id": job["run_id"]}

    monkeypatch.setattr(modal_app, "write_paper", FakeWritePaper())
    jobs = [{"run_id": "one"}, {"run_id": "bad"}, {"run_id": "three"}]
    results = modal_app.write_batch.local(jobs)
    assert mapped == [(jobs, {"return_exceptions": True})]
    assert [r["run_id"] for r in results] == ["one", "bad", "three"]
    assert [r["success"] for r in results] == [True, False, True]
    assert results[1]["error"] == "RuntimeError: container crashed"

    with pytest.raises(JobError, match="more than once"):
        modal_app.write_batch.local([{"run_id": "one"}, {"run_id": "one"}])
    assert len(mapped) == 1


def test_write_paper_commits_and_reloads_the_mounted_volume(monkeypatch, tmp_path):
    monkeypatch.setattr(modal_app, "setup_logging", lambda **_: None)
    fake_volume = FakeVolume(tmp_path / "vol")
    monkeypatch.setattr(modal_app, "volume", fake_volume)
    seen: dict[str, Any] = {}

    def fake_run_job(job, *, commit, reload, consolidate):
        commit()
        reload()
        seen["job"] = job
        seen["report"] = consolidate("some-model")
        return {"success": True}

    def fake_consolidate_lab(model, *, commit, reload):
        commit()
        reload()
        return {"model": model}

    monkeypatch.setattr(modal_app, "run_job", fake_run_job)
    monkeypatch.setattr(modal_app, "consolidate_lab", fake_consolidate_lab)
    # Inside a container `.remote` reaches the consolidation Function; here it runs in-process.
    monkeypatch.setattr(
        modal_app,
        "consolidate_memory",
        types.SimpleNamespace(remote=modal_app.consolidate_memory.local),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # .local() warns that the Volume is not mounted
        assert modal_app.write_paper.local({"run_id": "r"}) == {"success": True}
    assert seen == {"job": {"run_id": "r"}, "report": {"model": "some-model"}}
    assert (fake_volume.commits, fake_volume.reloads) == (2, 2)


# -- launching ------------------------------------------------------------------------


def test_upload_inputs_sends_visible_files_only(volume, tmp_path):
    raw, template = local_inputs(tmp_path)
    targets = upload_inputs(volume, "r1", raw, template)
    assert targets == ("inputs/r1/raw_materials", "inputs/r1/template")
    assert sorted(volume.uploaded) == [
        "/inputs/r1/raw_materials/experimental_log.md",
        "/inputs/r1/raw_materials/figures/plot.png",
        "/inputs/r1/raw_materials/idea_sparse.md",
        "/inputs/r1/template/guidelines.md",
        "/inputs/r1/template/style.sty",
        "/inputs/r1/template/template.tex",
    ]
    assert volume.forced == [True]  # a resumed run uploads the same files again


def test_download_outputs_fetches_what_the_summary_lists(volume, tmp_path, logs):
    (volume.root / "runs/r1").mkdir(parents=True)
    (volume.root / "runs/r1/final_paper.pdf").write_bytes(b"%PDF-1.5 " + b"x" * 100)
    (volume.root / "runs/r1/usage.json").write_text("{}")
    summary = {
        "files": {
            "pdf": "runs/r1/final_paper.pdf",
            "tex": "runs/r1/final_paper.tex",
            "usage": "runs/r1/usage.json",
        }
    }
    out = tmp_path / "out" / "nested"
    saved = download_outputs(volume, summary, out)
    assert saved == {"pdf": out / "final_paper.pdf", "usage": out / "usage.json"}
    assert saved["pdf"].read_bytes() == b"%PDF-1.5 " + b"x" * 100
    assert sorted(p.name for p in out.iterdir()) == ["final_paper.pdf", "usage.json"]
    assert "runs/r1/final_paper.tex is not on the volume" in logs.text
    assert download_outputs(volume, {"success": False}, out) == {}


def test_wait_for_result_polls_until_the_call_finishes(logs):
    call = FakeCall(lambda: {"success": True}, pending=3)
    assert wait_for_result(call, poll_seconds=0.5) == {"success": True}
    assert call.timeouts == [0.5] * 4
    assert logs.text.count("still in progress") == 3

    def crashed():
        raise RuntimeError("remote failure")

    with pytest.raises(RuntimeError, match="remote failure"):
        wait_for_result(FakeCall(crashed, pending=0))


def test_launch_uploads_runs_and_downloads(pipeline, volume, tmp_path, logs):
    raw, template = local_inputs(tmp_path)
    function = FakeFunction(volume)
    out = tmp_path / "paper"
    summary = launch(
        volume=volume,
        function=function,
        raw_materials_dir=raw,
        template=template,
        output_dir=out,
        run_id="demo-1",
        poll_seconds=0.01,
        use_plotting=True,
        writer_model="claude-fable-5-2",
        research_cutoff="2025-01",
    )
    assert summary["success"] and summary["run_id"] == "demo-1"
    assert summary["downloaded"] == {
        "pdf": str(out / "final_paper.pdf"),
        "tex": str(out / "final_paper.tex"),
        "usage": str(out / "usage.json"),
    }
    # The PDF is derived from the uploaded idea and the LaTeX from the uploaded template.
    assert (out / "final_paper.pdf").read_bytes() == b"%PDF-1.5 local idea"
    assert (out / "final_paper.tex").read_text() == "% built from: MY TEMPLATE"
    assert json.loads((out / "usage.json").read_text()) == {"total": {"requests": 3}}

    [job] = function.jobs
    assert job["raw_materials_dir"] == "inputs/demo-1/raw_materials"
    assert job["template_dir"] == "inputs/demo-1/template" and "template" not in job
    assert job["use_plotting"] is True and job["models"] == {"writer": "claude-fable-5-2"}
    assert function.calls[0].timeouts == [0.01] * 3
    cfg = pipeline.runs[0]["cfg"]
    assert cfg.use_plotting and cfg.research_cutoff == "2025-01"
    assert (cfg.base_dir / "raw_materials" / "figures" / "plot.png").is_file()
    assert not (cfg.base_dir / "raw_materials" / ".env").exists()
    assert f"Paper written to {out / 'final_paper.pdf'}" in logs.text
    assert "function call fc-fake" in logs.text

    again = launch(
        volume=volume,
        function=function,
        raw_materials_dir=raw,
        template=template,
        output_dir=out,
        run_id="demo-1",
        poll_seconds=0.01,
    )
    assert again["success"] and again["stages"]["outline"]["attempts"] == 2


def test_launch_resolves_a_template_by_name_and_defaults_the_output_dir(
    pipeline, volume, tmp_path, monkeypatch
):
    raw, template = local_inputs(tmp_path)
    monkeypatch.setenv(TEMPLATES_ENV, str(template.parent))
    monkeypatch.chdir(tmp_path)
    summary = launch(
        volume=volume,
        function=FakeFunction(volume, pending=0),
        raw_materials_dir=str(raw),
        template="mytemplate",
    )
    run_id = summary["run_id"]
    assert summary["success"] and validate_run_id(run_id)
    out = Path(f"paper_output_{run_id}")
    assert summary["downloaded"]["pdf"] == str(out / "final_paper.pdf")
    assert (tmp_path / out / "final_paper.tex").read_text() == "% built from: MY TEMPLATE"


def test_launch_validates_everything_before_uploading(pipeline, volume, tmp_path):
    raw, template = local_inputs(tmp_path)
    function = FakeFunction(volume)
    common = {"volume": volume, "function": function, "template": template}
    with pytest.raises(FileNotFoundError, match="raw materials directory not found"):
        launch(raw_materials_dir=tmp_path / "missing", **common)
    with pytest.raises(FileNotFoundError, match=r"required file not found.*other_idea\.md"):
        launch(raw_materials_dir=raw, idea_filename="other_idea.md", **common)
    with pytest.raises(JobError, match="effort must be one of"):
        launch(raw_materials_dir=raw, effort="ludicrous", **common)
    with pytest.raises(JobError, match="run_id must be"):
        launch(raw_materials_dir=raw, run_id="no/slash", **common)
    with pytest.raises(FileNotFoundError, match="LaTeX template 'nope' not found"):
        launch(raw_materials_dir=raw, **{**common, "template": "nope"})
    (tmp_path / "not_a_template").mkdir()
    with pytest.raises(FileNotFoundError, match=r"not_a_template has no template\.tex"):
        launch(raw_materials_dir=raw, **{**common, "template": tmp_path / "not_a_template"})
    (tmp_path / "not_a_template" / "template.tex").write_text("x")
    with pytest.raises(FileNotFoundError, match=r"not_a_template has no guidelines\.md"):
        launch(raw_materials_dir=raw, **{**common, "template": tmp_path / "not_a_template"})
    assert volume.uploaded == [] and function.jobs == []


def test_launch_reports_a_failed_run_and_fetches_nothing(pipeline, volume, tmp_path, logs):
    raw, template = local_inputs(tmp_path)
    pipeline.success = False
    summary = launch(
        volume=volume,
        function=FakeFunction(volume, pending=0),
        raw_materials_dir=raw,
        template=template,
        output_dir=tmp_path / "out",
        run_id="failing",
    )
    assert summary["success"] is False and summary["downloaded"] == {}
    assert "Run failing did not produce a paper: stage refinement failed: boom" in logs.text


# -- the Modal definitions ------------------------------------------------------------


def test_modal_objects_are_defined_without_credentials():
    assert isinstance(modal_app.app, modal.App)
    assert isinstance(modal_app.write_paper, modal.Function)
    assert isinstance(modal_app.write_batch, modal.Function)
    assert isinstance(modal_app.volume, modal.Volume)
    assert isinstance(modal_app.image, modal.Image)
    assert type(modal_app.main).__name__ == "LocalEntrypoint"
    assert isinstance(modal_app.SETTINGS, Settings)


def test_fakes_match_the_modal_sdk_calls_the_app_makes():
    from modal.volume import AbstractVolumeUploadContextManager

    calls = [
        (modal.Volume.batch_upload, (), {"force": True}),
        (AbstractVolumeUploadContextManager.put_file, (Path("idea.md"), "/inputs/r/idea.md"), {}),
        (modal.Volume.read_file, ("runs/r/final_paper.pdf",), {}),
        (modal.Volume.commit, (), {}),
        (modal.Volume.reload, (), {}),
        (modal.Function.spawn, ({"run_id": "r"},), {}),
        (modal.Function.map, ([{"run_id": "r"}],), {"return_exceptions": True}),
        (modal.FunctionCall.get, (), {"timeout": 60.0}),
        (modal.Image.build, (modal_app.app,), {}),
    ]
    for function, args, kwargs in calls:
        inspect.signature(function).bind(None, *args, **kwargs)
    assert hasattr(modal.FunctionCall, "object_id")


def test_pipeline_entry_points_accept_what_run_job_passes():
    """Guards the lazily imported pipeline contract against drift."""
    pipeline = pytest.importorskip("paper_orchestra.pipeline")
    config = job_spec({"run_id": "r1", "template": "cvpr2025"}).config
    inspect.signature(pipeline.prepare_run_dir).bind(config, None)
    inspect.signature(pipeline.run_paper).bind(config, harness=object(), compute=object())
    result_fields = {field.name for field in dataclasses.fields(pipeline.PaperResult)}
    assert {"success", "error", "stages", "usage"} <= result_fields
    # The volume is committed between stages from the memory refresh that follows each one.
    assert "harness.refresh" in inspect.getsource(pipeline)


def test_main_builds_the_sandbox_image_then_launches(monkeypatch):
    monkeypatch.setattr(modal_app, "setup_logging", lambda **_: None)
    built: list[Any] = []
    monkeypatch.setattr(
        modal_app, "python_image", lambda: types.SimpleNamespace(build=built.append)
    )
    seen: dict[str, Any] = {}

    def fake_launch(**kwargs):
        seen.update(kwargs)
        downloaded = {} if kwargs["run_id"] == "not-downloaded" else {"pdf": "final_paper.pdf"}
        return {"success": kwargs["run_id"] != "failing", "downloaded": downloaded}

    monkeypatch.setattr(modal_app, "launch", fake_launch)
    modal_app.main(
        raw_materials_dir="raw",
        template="cvpr2025",
        output_dir="out",
        run_id="r1",
        plotting=True,
        writer_model="claude-fable-5-2",
        effort="low",
        dream="end",
        lab_memory=False,
    )
    assert built == [modal_app.app]
    assert seen["volume"] is modal_app.volume and seen["function"] is modal_app.write_paper
    launcher_options = ("volume", "function", "raw_materials_dir", "template", "output_dir")
    assert [seen.pop(name) for name in launcher_options[2:]] == ["raw", "cvpr2025", "out"]
    job = build_job(**{k: v for k, v in seen.items() if k not in launcher_options})
    assert job["run_id"] == "r1" and job["use_plotting"] is True
    assert job["models"] == {"writer": "claude-fable-5-2", "effort": "low"}
    assert job["memory"] == {"enabled": True, "lab": False, "dream": "end"}

    for run_id in ("failing", "not-downloaded"):
        with pytest.raises(SystemExit) as raised:
            modal_app.main(raw_materials_dir="raw", template="cvpr2025", run_id=run_id)
        assert raised.value.code == 1


def test_local_entrypoint_flags_are_understood_by_the_modal_cli():
    cli = pytest.importorskip("modal.cli.run")
    build = getattr(cli, "_get_click_command_for_local_entrypoint", None)
    if build is None:
        pytest.skip("this Modal version builds entrypoint commands differently")
    from click.testing import CliRunner

    result = CliRunner().invoke(build(modal_app.app, modal_app.main), ["--help"], obj={})
    assert result.exit_code == 0, result.output
    options = {line.split("  ")[1]: line for line in result.output.splitlines() if "--" in line}
    assert set(options) == {
        "--raw-materials-dir TEXT",
        "--template TEXT",
        "--output-dir TEXT",
        "--run-id TEXT",
        "--plotting / --no-plotting",
        "--idea-filename TEXT",
        "--experimental-log-filename TEXT",
        "--research-cutoff TEXT",
        "--writer-model TEXT",
        "--reflection-model TEXT",
        "--plotting-model TEXT",
        "--fast-model TEXT",
        "--image-model TEXT",
        "--effort TEXT",
        "--plotting-max-critic-rounds INTEGER",
        "--memory / --no-memory",
        "--lab-memory / --no-lab-memory",
        "--dream [off|end|stages]",
        "--resume / --no-resume",
        "--help",
    }
    required = {name for name, line in options.items() if "[required]" in line}
    assert required == {"--raw-materials-dir TEXT", "--template TEXT"}


def test_module_imports_cleanly_anywhere_without_modal_credentials(tmp_path):
    """Definition smoke test: no deprecated Modal parameters, no cwd or credential dependence."""
    code = (
        "import json, paper_orchestra.modal_app as m\n"
        "print(json.dumps({'settings': list(vars(m.SETTINGS).values()), 'app': m.app.name}))\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MODAL_", "PAPER_ORCHESTRA_", "ANTHROPIC_"))
    }
    env.update(
        MODAL_CONFIG_PATH=str(tmp_path / "no-such-config.toml"),
        PAPER_ORCHESTRA_MODAL_APP="smoke.app",
        PAPER_ORCHESTRA_MODAL_SECRET="none",
        PAPER_ORCHESTRA_TEMPLATES="/root/templates",
    )
    done = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {
        "settings": ["smoke.app", "paper-orchestra", None, 21600, 4],
        "app": "smoke.app",
    }
