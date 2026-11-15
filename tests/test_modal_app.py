from __future__ import annotations

import base64
import dataclasses
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
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
from paper_orchestra.compute.modal_backend import TEX_PACKAGES_ENV, ModalBackend
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
    LEASE_NAME,
    JobError,
    LeaseHeld,
    Settings,
    StageCheckpoint,
    VolumeLease,
    batch_results,
    build_job,
    check_batch,
    check_credentials,
    commit_volume,
    consolidate_lab,
    container_path,
    download_outputs,
    explain_credential_failure,
    failure_summary,
    federation_advice,
    function_env,
    identity_token_claims,
    job_spec,
    latex_mode,
    launch,
    load_settings,
    new_run_id,
    project_root,
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
        self.removed: list[str] = []
        self.commits = 0
        self.reloads = 0
        #: What the next commits raise, in order; ``None`` lets a commit through.
        self.commit_errors: list[Exception | None] = []
        self.reload_error: Exception | None = None
        self.upload_error: Exception | None = None

    @contextmanager
    def batch_upload(self, force: bool = False):
        self.forced.append(force)
        if self.upload_error is not None:
            raise self.upload_error
        yield FakeBatch(self)

    def listdir(self, path: str, *, recursive: bool = False):
        """Entries under ``path``, named from the root of the volume as Modal names them."""
        assert recursive, "the launcher lists a run's inputs as a whole"
        kinds = modal.volume.FileEntryType
        return [
            types.SimpleNamespace(
                path=found.relative_to(self.root).as_posix(),
                type=kinds.DIRECTORY if found.is_dir() else kinds.FILE,
            )
            for found in sorted((self.root / path).rglob("*"))
        ]

    def read_file(self, path: str):
        target = self.root / path
        if not target.is_file():
            raise FileNotFoundError(path)
        data = target.read_bytes()
        for start in range(0, len(data), 5):
            yield data[start : start + 5]

    def remove_file(self, path: str, recursive: bool = False) -> None:
        target = self.root / path
        if not target.exists():
            raise FileNotFoundError(path)
        self.removed.append(path)
        if target.is_dir():
            assert recursive, "a directory is only removed recursively"
            shutil.rmtree(target)
        else:
            target.unlink()

    def commit(self) -> None:
        self.commits += 1
        error = self.commit_errors.pop(0) if self.commit_errors else None
        if error is not None:
            raise error

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
        self.failure = "stage refinement failed: boom"
        self.error: Exception | None = None
        #: Called after each stage has settled; its value is recorded in ``checkpoints``.
        self.observe = lambda: None
        self.checkpoints: list[tuple[str, Any]] = []
        #: Overrides the stages that settle.
        self.only_stages: list[str] | None = None
        self.refresh_errors: list[Exception] = []
        #: Lessons the run marks as portable, as ``(page, text)``; the end of the run
        #: shares them with the lab memory the way the real pipeline does.
        self.lessons: list[tuple[str, str]] = []
        #: What the run looks up and keeps for later runs, as ``(key, value)``.
        self.lookups: list[tuple[str, Any]] = []

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
                "lease": (cfg.base_dir / LEASE_NAME).is_file(),
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
        for key, value in self.lookups:
            harness.cache.put("literature", key, value)
        if self.lessons:
            for page, text in self.lessons:
                harness.write_page(page, text, {"portable": True})
            harness.finish()
        return types.SimpleNamespace(
            success=self.success,
            pdf_path=paths.final_pdf if self.success else None,
            base_dir=cfg.base_dir,
            stages={"outline": {"status": "done", "attempts": attempts + 1, "error": None}},
            usage={"total": {"requests": 3}, "report": paths.usage},
            error=None if self.success else self.failure,
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


@pytest.fixture(autouse=True)
def no_settle_delay(monkeypatch) -> None:
    """A call that took a lease normally waits a moment before it checks that it kept it.

    And one that finds the lease of another call looks at it every few seconds.
    """
    monkeypatch.setattr(modal_app, "_LEASE_SETTLE", 0.0)
    monkeypatch.setattr(modal_app, "_LEASE_POLL", 0.01)


@pytest.fixture(autouse=True)
def default_modal_settings(monkeypatch) -> None:
    """The tests see the defaults, whatever the shell that runs them has exported."""
    monkeypatch.delenv(modal_app.LATEX_ENV, raising=False)
    monkeypatch.delenv(TEX_PACKAGES_ENV, raising=False)
    # Read from the environment when the module was imported.
    monkeypatch.setattr(
        modal_app,
        "SETTINGS",
        dataclasses.replace(modal_app.SETTINGS, latex="sandbox", tex_packages=()),
    )


@pytest.fixture
def volume(tmp_path) -> FakeVolume:
    return FakeVolume(tmp_path / "vol")


@pytest.fixture
def logs(caplog, monkeypatch):
    """Records of `paper_orchestra.*`, even if an earlier test stopped the logger propagating."""
    monkeypatch.setattr(logging.getLogger("paper_orchestra"), "propagate", True)
    caplog.set_level(logging.INFO, logger="paper_orchestra")
    return caplog


def lab_wiki(volume: FakeVolume) -> Wiki:
    """The lab wiki of a volume that has been used before."""
    return Wiki(volume.root / "memory", name="lab")


def identity_token(**claims: Any) -> str:
    """An unsigned stand-in for the JWT Modal gives a container."""

    def part(data: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{part({'alg': 'RS256', 'kid': 'k'})}.{part(claims)}.c2lnbmF0dXJl"


MODAL_CLAIMS = {
    "iss": "https://oidc.modal.com",
    "aud": "oidc.modal.com",
    "sub": "modal:workspace_id:ac-1:environment_name:main:app_name:paper-orchestra",
    "workspace_id": "ac-1",
    "environment_name": "main",
    "app_name": "paper-orchestra",
    "function_name": "consolidate_memory",
    "container_id": "ta-1",
    "iat": 1_800_000_000,
    "exp": 1_800_172_800,
    "jti": "one-time",
}


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
        # Model-written LaTeX is compiled away from the credentials unless asked otherwise.
        latex="sandbox",
        tex_packages=(),
    )
    custom = load_settings(
        {
            "PAPER_ORCHESTRA_MODAL_APP": "my.app",
            "PAPER_ORCHESTRA_MODAL_VOLUME": "papers_v2",
            "PAPER_ORCHESTRA_MODAL_SECRET": " anthropic-wif ",
            "PAPER_ORCHESTRA_MODAL_TIMEOUT": "7200",
            "PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS": "2",
            "PAPER_ORCHESTRA_MODAL_LATEX": " Local ",
            "PAPER_ORCHESTRA_MODAL_TEX_PACKAGES": "texlive-lang-german  fonts-lato lmodern",
        }
    )
    assert custom == Settings(
        "my.app",
        "papers_v2",
        "anthropic-wif",
        7200,
        2,
        latex="local",
        tex_packages=("texlive-lang-german", "fonts-lato"),  # lmodern is installed anyway
    )
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
        ("PAPER_ORCHESTRA_MODAL_LATEX", "docker", "must be one of: sandbox, local"),
        ("PAPER_ORCHESTRA_MODAL_TEX_PACKAGES", "lmodern; curl x", "is not a Debian package name"),
        ("PAPER_ORCHESTRA_MODAL_TEX_PACKAGES", "$(id)", "is not a Debian package name"),
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
        "PAPER_ORCHESTRA_MODAL_LATEX": "sandbox",
        "PAPER_ORCHESTRA_MEMORY_DIR": "/vol/memory",
        "PAPER_ORCHESTRA_TEMPLATES": "/root/templates",
        **WIF_IDS,
        "ANTHROPIC_WORKSPACE_ID": "wrkspc_1",
        "PAPER_ORCHESTRA_STRUCTURED_OUTPUTS": "0",
    }
    assert all(value for value in env.values())  # Modal may drop empty values
    # The container builds the image of its LaTeX sandbox from the same package list.
    extra = load_settings({"PAPER_ORCHESTRA_MODAL_TEX_PACKAGES": "fonts-lato texlive-music"})
    assert function_env(extra, {})["PAPER_ORCHESTRA_MODAL_TEX_PACKAGES"] == (
        "fonts-lato texlive-music"
    )


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
        {
            "PAPER_ORCHESTRA_MODAL_LATEX": "local",
            "PAPER_ORCHESTRA_MODAL_TEX_PACKAGES": "fonts-lato texlive-lang-german",
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
        keep_provided_figures=True,
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
    assert full["keep_provided_figures"] is True
    assert job_spec(full).config.keep_provided_figures is True
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
    # Runs only add notes to the shared wiki; consolidate_memory is its one page writer.
    assert cfg.memory.lab_consolidate is False
    assert job_spec({"run_id": "r1", "template": "t", "memory": {"lab_consolidate": False}})
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
        ({"memory": {"lab_consolidate": True}}, r"memory\.lab_consolidate cannot be set"),
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
    # One commit for the lease on the run, one as each stage settles, and a last one before
    # the summary is returned.
    assert pipeline.checkpoints == [
        ("ingest", 2),
        ("outline", 3),
        ("literature", 4),
        ("section_writing", 5),
        ("refinement", 6),
        ("finalize", 7),
    ]
    # Reloaded on arrival, and once more to see whether another call took the run meanwhile.
    assert (volume.reloads, volume.commits) == (2, 8)

    [(prepared_cfg, raw_materials_dir)] = pipeline.prepared
    assert raw_materials_dir == volume.root / "inputs/run-1/raw_materials"
    [run] = pipeline.runs
    cfg = run["cfg"]
    assert cfg is prepared_cfg and cfg.base_dir == volume.root / "runs/run-1"
    assert cfg.template_dir == volume.root / "inputs/run-1/template"
    assert cfg.compute == "modal" and cfg.research_cutoff == "2025-06"
    assert cfg.memory.resolved_lab_dir() == volume.root / "memory"
    # Nothing model-written runs next to the credentials: LaTeX is compiled, and figure code
    # is run, in nested sandboxes.
    assert isinstance(run["compute"], ModalBackend) and run["compute"].latex == "sandbox"
    # The run held the lease on its directory while it worked and gave it up at the end.
    assert run["lease"] is True and not (volume.root / "runs/run-1" / LEASE_NAME).exists()
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
    assert volume.commits == 2 * (1 + 6 + 1) and volume.reloads == 0

    fresh = run_job({**job, "resume": False}, commit=volume.commit, mount=volume.root)
    assert fresh["stages"]["outline"]["attempts"] == 1


def test_run_job_reports_a_failed_pipeline(pipeline, volume):
    pipeline.success = False
    summary = run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and summary["error"] == "stage refinement failed: boom"
    assert summary["files"] == {} and summary["run_dir"] == "runs/run-1"
    assert volume.commits == 1 + 6 + 1  # partial progress is still made durable


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
    ("change", "message", "commits"),
    [
        ({"run_id": "../escape"}, "run_id must be", 1),
        ({"unknown": 1}, "unknown job option", 1),
        ({"template_dir": "inputs/run-1/no-such-template"}, "LaTeX template not found", 1),
        # Found once the run directory is being prepared, so after the lease was committed.
        ({"raw_materials_dir": "inputs/run-1/empty"}, "required file not found", 2),
    ],
)
def test_run_job_returns_unfixable_problems_instead_of_raising(
    pipeline, volume, change, message, commits
):
    job = stage_inputs(volume)
    (volume.root / "inputs/run-1/empty").mkdir()
    summary = run_job({**job, **change}, commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and message in summary["error"]
    assert summary["files"] == {} and pipeline.runs == []
    assert json.loads(json.dumps(summary)) == summary
    assert volume.commits == commits
    assert not list(volume.root.rglob(LEASE_NAME))


def test_run_job_rejects_a_job_that_is_not_a_dict(pipeline, volume):
    summary = run_job(None, commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and summary["run_id"] is None
    assert "a job must be a dict" in summary["error"]


def test_latex_is_compiled_in_the_container_only_when_asked(pipeline, volume, monkeypatch):
    """PAPER_ORCHESTRA_MODAL_LATEX=local trades the sandbox for speed."""
    assert latex_mode({}) == "sandbox" and latex_mode({modal_app.LATEX_ENV: "LOCAL"}) == "local"
    monkeypatch.setenv(modal_app.LATEX_ENV, "local")
    assert run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)["success"]
    assert pipeline.runs[0]["compute"].latex == "local"
    assert pipeline.runs[0]["env"]["openin_any"] == "p"

    # In that mode a missing TeX Live is found before any model call.
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: False))
    summary = run_job(stage_inputs(volume, "run-2"), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and "pdflatex and bibtex" in summary["error"]
    assert len(pipeline.prepared) == 1 and len(pipeline.runs) == 1

    monkeypatch.setenv(modal_app.LATEX_ENV, "docker")
    summary = run_job(stage_inputs(volume, "run-3"), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and "PAPER_ORCHESTRA_MODAL_LATEX" in summary["error"]


def test_run_job_lets_crashes_propagate_so_modal_retries(pipeline, volume):
    pipeline.error = RuntimeError("worker died")
    with pytest.raises(RuntimeError, match="worker died"):
        run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert volume.commits == 1 + 1  # the lease, then what was written before the crash
    # The retry must not find the run taken by the attempt that crashed.
    assert not (volume.root / "runs/run-1" / LEASE_NAME).exists()

    volume.commit_errors = [None, RuntimeError("volume busy")]
    with pytest.raises(RuntimeError, match="worker died"):
        run_job(stage_inputs(volume, "run-2"), commit=volume.commit, mount=volume.root)


def test_checkpoints_wait_for_stages_that_run_side_by_side(pipeline, volume):
    job = {**stage_inputs(volume), "use_plotting": True}
    pipeline.observe = lambda: volume.commits
    assert run_job(job, commit=volume.commit, mount=volume.root)["success"]
    # No commit while a sibling stage may still be writing files; the next boundary has it.
    assert pipeline.checkpoints == [
        ("ingest", 2),
        ("outline", 3),
        ("literature", 3),
        ("plotting", 3),
        ("section_writing", 4),
        ("refinement", 5),
        ("finalize", 6),
    ]
    assert volume.commits == 1 + 5 + 1


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
    assert volume.commits == before + 1 + 6 + 1  # a failed refresh does not cost the checkpoint

    assert run_job({**job, "memory": {"enabled": False}}, commit=volume.commit, mount=volume.root)[
        "success"
    ]
    assert pipeline.runs[-1]["harness"].enabled is False


def test_run_job_tolerates_failed_checkpoints_and_reloads(pipeline, volume, logs):
    volume.reload_error = RuntimeError("there are open files preventing the operation")
    volume.commit_errors = [None, RuntimeError("volume busy")]
    job = stage_inputs(volume)
    assert run_job(job, commit=volume.commit, reload=volume.reload, mount=volume.root)["success"]
    assert "Could not reload the volume" in logs.text
    assert "Could not commit the volume between stages: volume busy" in logs.text
    assert volume.commits == 1 + 6 + 1


def test_run_job_insists_on_the_final_commit(pipeline, volume):
    pipeline.only_stages = []
    job = stage_inputs(volume)
    volume.commit_errors = [None, RuntimeError("busy 1"), RuntimeError("busy 2")]
    assert run_job(job, commit=volume.commit, mount=volume.root)["success"]
    assert volume.commits == 1 + 3

    volume.commit_errors = [None, *(RuntimeError(f"busy {n}") for n in range(3))]
    with pytest.raises(RuntimeError, match="busy 2"):
        run_job(job, commit=volume.commit, mount=volume.root)


def test_run_job_has_the_lab_memory_consolidated_after_a_successful_run(pipeline, volume, logs):
    calls: list[tuple[str, int]] = []

    def consolidate(model: str) -> dict[str, Any]:
        calls.append((model, volume.commits))
        return {"notes_archived": 2, "pages_changed": ["wiki/lessons/latex.md"], "error": None}

    lab_wiki(volume)
    job = {**stage_inputs(volume), "models": {"reflection": "claude-opus-4-8"}}
    summary = run_job(job, commit=volume.commit, consolidate=consolidate, mount=volume.root)
    assert summary["success"]
    # With the reflection model, and only after the final commit made the run's notes visible.
    assert calls == [("claude-opus-4-8", 1 + 6 + 1)] and volume.commits == 1 + 6 + 1
    assert "Lab memory consolidated: 2 note(s) folded in, 1 page(s) changed." in logs.text


@pytest.mark.parametrize(
    ("change", "pipeline_succeeds"),
    [
        ({"memory": {"lab": False}}, True),
        ({"memory": {"enabled": False}}, True),
        ({"memory": {"dream": "off"}}, True),
        ({}, False),
        ({"run_id": "../escape"}, True),
    ],
)
def test_run_job_skips_the_consolidation_when_there_is_nothing_to_fold_in(
    pipeline, volume, change, pipeline_succeeds
):
    pipeline.success = pipeline_succeeds
    lab_wiki(volume)
    calls: list[str] = []
    summary = run_job(
        {**stage_inputs(volume), **change},
        commit=volume.commit,
        consolidate=calls.append,
        mount=volume.root,
    )
    assert calls == [] and summary["success"] is (pipeline_succeeds and "run_id" not in change)


def test_run_job_survives_a_failed_consolidation(pipeline, volume, logs):
    def crashing(model: str) -> dict[str, Any]:
        raise TimeoutError("took too long")

    lab_wiki(volume)
    job = stage_inputs(volume)
    assert run_job(job, commit=volume.commit, consolidate=crashing, mount=volume.root)["success"]
    assert "Could not consolidate the lab memory (TimeoutError: took too long)" in logs.text

    busy = {"error": "another consolidation of this wiki is in progress"}
    summary = run_job(job, commit=volume.commit, consolidate=lambda model: busy, mount=volume.root)
    assert summary["success"]
    assert "The lab memory was not consolidated: another consolidation" in logs.text


@pytest.fixture
def scripted_model():
    """Serve model ids starting with "fake" from a scripted client; yields its installer."""
    yield lambda responder: register_client("fake", FakeLLM(responder))
    reset_clients()


def test_consolidate_lab_folds_the_notes_of_earlier_runs_into_pages(volume, scripted_model):
    lab = Wiki(volume.root / "memory", name="lab")
    note = lab.add_note(
        "Load cleveref after hyperref.", stage="promotion", kind="lesson", title="cleveref order"
    )
    seen: dict[str, Any] = {}

    def responder(request: FakeRequest) -> str:
        seen.update(label=request.label, model=request.model)
        seen["notes"] = request.memory.view("/memories/notes")
        request.memory.create(
            "/memories/wiki/lessons/latex.md",
            "---\nsummary: LaTeX pitfalls\n---\nLoad cleveref after hyperref.",
        )
        return "Folded one lesson into the LaTeX page."

    scripted_model(responder)
    report = consolidate_lab(
        "fake-dreamer", commit=volume.commit, reload=volume.reload, mount=volume.root
    )
    assert json.loads(json.dumps(report)) == report
    assert (report["scope"], report["error"], report["llm_used"]) == ("lab", None, True)
    assert (report["notes_seen"], report["notes_archived"]) == (1, 1)
    assert report["pages_changed"] == ["wiki/lessons/latex.md"]
    assert report["summary"] == "Folded one lesson into the LaTeX page."
    assert (seen["label"], seen["model"]) == ("dream.lab", "fake-dreamer")
    assert Path(note).name in seen["notes"]
    page = volume.root / "memory" / "wiki" / "lessons" / "latex.md"
    assert "Load cleveref after hyperref." in page.read_text()
    assert lab.pending_notes() == []
    # It saw the notes other containers committed, and its pages are durable when it returns.
    # The other reload and commit belong to its lease on the wiki, which it gave up.
    assert (volume.reloads, volume.commits) == (2, 2)
    assert not (volume.root / "memory" / LEASE_NAME).exists()


def test_consolidate_lab_without_pending_notes_asks_no_model(volume, scripted_model):
    requests: list[FakeRequest] = []
    scripted_model(lambda request: requests.append(request) or "nothing to do")
    report = consolidate_lab("fake", commit=volume.commit, mount=volume.root)
    assert (report["notes_seen"], report["llm_used"], report["error"]) == (0, False, None)
    assert requests == [] and volume.commits == 1 + 1


def test_consolidate_lab_reports_a_model_that_cannot_be_used(volume, monkeypatch):
    Wiki(volume.root / "memory", name="lab").add_note("A lesson.", kind="lesson")
    report = consolidate_lab("no-such-provider-1", commit=volume.commit, mount=volume.root)
    assert "LLMConfigError" in report["error"] and report["notes_archived"] == 0
    assert len(Wiki(volume.root / "memory", name="lab").pending_notes()) == 1

    for name, value in {
        **WIF_IDS,
        "MODAL_IDENTITY_TOKEN": "jwt",
        "ANTHROPIC_API_KEY": "sk",
    }.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(JobError, match="exactly one credential"):
        consolidate_lab("fake", commit=volume.commit, mount=volume.root)
    assert volume.commits == 1 + 1  # nothing after the first consolidation


def test_consolidations_of_two_apps_do_not_write_the_lab_wiki_together(
    volume, scripted_model, monkeypatch
):
    """One container at a time holds within an App; two Apps can share a volume."""
    monkeypatch.setattr(modal_app, "_LEASE_REFRESH", 0.01)
    lab_wiki(volume).add_note("Load cleveref after hyperref.", kind="lesson")
    other: dict[str, Any] = {}

    def responder(request: FakeRequest) -> str:
        # While this consolidation is in its session, the one of another App starts.
        other["report"] = consolidate_lab(
            "fake-dreamer",
            commit=volume.commit,
            reload=volume.reload,
            owner="fc-other-app",
            mount=volume.root,
        )
        request.memory.create(
            "/memories/wiki/lessons/latex.md",
            "---\nsummary: LaTeX pitfalls\n---\nLoad cleveref after hyperref.",
        )
        return "Folded one lesson."

    scripted_model(responder)
    report = consolidate_lab(
        "fake-dreamer",
        commit=volume.commit,
        reload=volume.reload,
        owner="fc-this-app",
        mount=volume.root,
    )
    assert (report["error"], report["notes_archived"]) == (None, 1)
    refused = other["report"]
    assert (
        "the lab memory is already being written by another call (fc-this-app" in (refused["error"])
    )
    assert refused["notes_archived"] == 0 and refused["pages_changed"] == []
    assert json.loads(json.dumps(refused)) == refused
    assert not (volume.root / "memory" / LEASE_NAME).exists()

    # Afterwards the wiki is free again.
    again = consolidate_lab(
        "fake-dreamer", commit=volume.commit, owner="fc-other-app", mount=volume.root
    )
    assert again["error"] is None


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
    assert volume.commits == 1 + 1  # the lease and its release
    assert not (volume.root / "runs/run-1" / LEASE_NAME).exists()


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


# -- one run per run id ---------------------------------------------------------------


def test_a_run_that_is_in_flight_is_not_started_a_second_time(pipeline, volume, monkeypatch):
    """Two calls for one run id must not drive the same directory at the same time."""
    # The second call is turned away when it sees the heartbeat of the first one move.
    monkeypatch.setattr(modal_app, "_LEASE_REFRESH", 0.01)
    job = stage_inputs(volume)
    second: dict[str, Any] = {}

    def the_same_job_arrives_again() -> None:
        # While the first call is working, a second submission reaches another container.
        if not second:
            second["summary"] = run_job(
                job,
                commit=volume.commit,
                reload=volume.reload,
                mount=volume.root,
                owner="fc-second",
            )

    pipeline.observe = the_same_job_arrives_again
    first = run_job(
        job, commit=volume.commit, reload=volume.reload, mount=volume.root, owner="fc-first"
    )

    assert first["success"]
    assert len(pipeline.prepared) == 1 and len(pipeline.runs) == 1  # the paper ran once
    refused = second["summary"]
    assert refused["success"] is False and refused["run_id"] == "run-1"
    assert "run run-1 is already being written by another call (fc-first" in refused["error"]
    assert json.loads(json.dumps(refused)) == refused
    assert first["stages"]["outline"]["attempts"] == 1

    # Once the first call is over the run id can be submitted again, which resumes the run.
    again = run_job(job, commit=volume.commit, mount=volume.root, owner="fc-third")
    assert again["success"] and again["stages"]["outline"]["attempts"] == 2


def test_the_lease_is_committed_before_the_run_and_released_with_its_results(pipeline, volume):
    lease = volume.root / "runs/run-1" / LEASE_NAME
    at_commit: list[bool] = []

    def commit() -> None:
        at_commit.append(lease.is_file())
        volume.commit()

    pipeline.observe = lambda: json.loads(lease.read_text())
    started = time.time()
    summary = run_job(stage_inputs(volume), commit=commit, mount=volume.root, owner="fc-1")
    assert summary["success"]
    # Other containers see the lease from the first commit on, and see it gone in the commit
    # that makes the results visible.
    assert at_commit == [True] * 7 + [False]
    held = [value for _, value in pipeline.checkpoints]
    assert {value["owner"] for value in held} == {"fc-1"}
    assert all(started <= value["at"] <= time.time() for value in held)
    assert sorted(path.name for path in lease.parent.iterdir() if path.name.startswith(".")) == []


def test_a_retry_of_the_same_call_takes_its_own_lease_over_at_once(pipeline, volume):
    job = stage_inputs(volume)
    lease = volume.root / "runs/run-1" / LEASE_NAME
    lease.parent.mkdir(parents=True)
    # What a preempted container leaves behind: a fresh lease that nobody released.
    lease.write_text(json.dumps({"owner": "fc-1", "at": time.time()}))
    assert run_job(job, commit=volume.commit, mount=volume.root, owner="fc-1")["success"]
    assert not lease.exists()

    # Another call has to wait until that heartbeat is old.
    lease.write_text(json.dumps({"owner": "fc-1", "at": time.time()}))
    refused = run_job(job, commit=volume.commit, mount=volume.root, owner="fc-2")
    assert refused["success"] is False and "already being written" in refused["error"]
    assert json.loads(lease.read_text())["owner"] == "fc-1"  # not released by the refused call
    assert len(pipeline.runs) == 1


def test_a_lease_whose_heartbeat_stopped_is_taken_over(pipeline, volume, logs):
    job = stage_inputs(volume)
    lease = volume.root / "runs/run-1" / LEASE_NAME
    lease.parent.mkdir(parents=True)
    almost = time.time() - modal_app._LEASE_STALE + 20
    lease.write_text(json.dumps({"owner": "fc-gone", "at": almost}))
    refused = run_job(job, commit=volume.commit, mount=volume.root, owner="fc-new")
    assert refused["success"] is False and "fc-gone" in refused["error"]
    assert f"once the heartbeat is {modal_app._LEASE_STALE:.0f} s old" in refused["error"]

    lease.write_text(json.dumps({"owner": "fc-gone", "at": time.time() - modal_app._LEASE_STALE}))
    assert run_job(job, commit=volume.commit, mount=volume.root, owner="fc-new")["success"]
    assert "Taking over run run-1: the call that held it (fc-gone)" in logs.text

    # A lease that cannot be read holds nobody back for ever.
    lease.write_text("{not json")
    assert run_job(job, commit=volume.commit, mount=volume.root, owner="fc-new")["success"]
    assert len(pipeline.runs) == 2 and not lease.exists()


def test_a_run_whose_container_was_stopped_is_resumed_without_asking_twice(
    pipeline, volume, monkeypatch, logs
):
    """Ctrl-C on `modal run`, or `--cancel`, stops a container before it can give up its lease.

    Launching the run id again is how such a run is resumed, so that launch must not be
    turned away because the heartbeat the container left still looks fresh.
    """
    job = stage_inputs(volume)
    lease = volume.root / "runs/run-1" / LEASE_NAME
    lease.parent.mkdir(parents=True)
    monkeypatch.setattr(modal_app, "_LEASE_STALE", 1.0)
    lease.write_text(json.dumps({"owner": "fc-stopped", "at": time.time()}))
    resumed = run_job(
        job, commit=volume.commit, reload=volume.reload, mount=volume.root, owner="fc-new"
    )
    assert resumed["success"] and len(pipeline.runs) == 1
    # It watched the lease until the heartbeat that never moved was stale.
    assert volume.reloads > 2
    assert "watching whether that call is still at work" in logs.text
    assert "Taking over run run-1: the call that held it (fc-stopped)" in logs.text
    assert not lease.exists()

    # A holder that is at work shows a new heartbeat, and the run is left to it.
    monkeypatch.setattr(modal_app, "_LEASE_STALE", 150.0)
    monkeypatch.setattr(modal_app, "_LEASE_REFRESH", 0.01)
    holder = VolumeLease.acquire(
        lease.parent, "fc-working", subject="run run-1", commit=volume.commit
    )
    try:
        refused = run_job(
            job, commit=volume.commit, reload=volume.reload, mount=volume.root, owner="fc-new"
        )
    finally:
        holder.release()
    assert refused["success"] is False and len(pipeline.runs) == 1
    assert "run run-1 is already being written by another call (fc-working" in refused["error"]

    # So does a lease that changes hands: another call took the run over while this one
    # was watching.
    taken = {"owner": "fc-quicker", "at": time.time()}
    lease.write_text(json.dumps({"owner": "fc-stopped", "at": taken["at"]}))
    looks: list[int] = []

    def another_call_takes_the_run() -> None:
        looks.append(len(looks))
        if len(looks) > 1:  # the first look is the one run_job takes on arrival
            lease.write_text(json.dumps(taken))

    refused = run_job(
        job, commit=volume.commit, reload=another_call_takes_the_run, mount=volume.root
    )
    assert refused["success"] is False and "(fc-quicker" in refused["error"]
    assert len(pipeline.runs) == 1

    # The clocks of two containers need not agree.  A heartbeat stamped by one that runs
    # ahead never looks old, so the watch itself ends after the time a lease takes to go
    # stale: a heartbeat that has not moved by then is not going to.
    monkeypatch.setattr(modal_app, "_LEASE_STALE", 0.5)
    lease.write_text(json.dumps({"owner": "fc-stopped", "at": time.time() + 3600}))
    resumed = run_job(
        job, commit=volume.commit, reload=volume.reload, mount=volume.root, owner="fc-new"
    )
    assert resumed["success"] and len(pipeline.runs) == 2
    assert "its heartbeat did not move" in logs.text


def test_the_heartbeat_keeps_the_lease_of_a_long_run_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_app, "_LEASE_REFRESH", 0.01)
    run_dir = tmp_path / "runs" / "r1"
    path = run_dir / LEASE_NAME
    commits: list[int] = []
    lease = VolumeLease.acquire(run_dir, "fc-1", subject="run r1", commit=lambda: commits.append(1))
    try:
        first = json.loads(path.read_text())
        assert first["owner"] == "fc-1"
        deadline = time.monotonic() + 10
        while json.loads(path.read_text())["at"] == first["at"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert json.loads(path.read_text())["at"] > first["at"]
        # The heartbeat thread leaves committing to the thread that drives the pipeline.
        assert commits == [1]
        with pytest.raises(LeaseHeld, match="run r1 is already being written by another call"):
            VolumeLease.acquire(run_dir, "fc-2", subject="run r1", commit=lambda: None)
    finally:
        lease.release()
    lease.release()
    assert list(run_dir.iterdir()) == []

    # A lease that was taken over meanwhile is not the old holder's to remove.
    lease = VolumeLease.acquire(run_dir, "fc-1", subject="run r1", commit=lambda: None)
    path.write_text(json.dumps({"owner": "fc-2", "at": time.time()}))
    lease.release()
    assert json.loads(path.read_text())["owner"] == "fc-2"


def test_an_interrupted_run_leaves_its_lease_to_the_retry_of_the_same_call(pipeline, volume):
    """After a preemption the next attempt may already hold the lease this one would remove."""
    job = stage_inputs(volume)
    lease = volume.root / "runs/run-1" / LEASE_NAME
    pipeline.error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_job(job, commit=volume.commit, mount=volume.root, owner="fc-1")
    assert json.loads(lease.read_text())["owner"] == "fc-1"

    pipeline.error = None
    refused = run_job(job, commit=volume.commit, mount=volume.root, owner="fc-2")
    assert refused["success"] is False and "already being written" in refused["error"]
    # The attempt Modal starts after a preemption is the same call: it goes ahead at once.
    assert run_job(job, commit=volume.commit, mount=volume.root, owner="fc-1")["success"]
    assert not lease.exists()
    assert run_job(job, commit=volume.commit, mount=volume.root, owner="fc-2")["success"]


def test_two_calls_that_start_in_the_same_moment_do_not_both_run(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    path = run_dir / LEASE_NAME

    def reload_shows_the_other_commit() -> None:
        # Both found the run free and wrote a lease; the later commit won the file.
        path.write_text(json.dumps({"owner": "fc-other", "at": time.time()}))

    with pytest.raises(LeaseHeld, match="fc-other"):
        VolumeLease.acquire(
            run_dir,
            "fc-mine",
            subject="run r1",
            commit=lambda: None,
            reload=reload_shows_the_other_commit,
        )
    assert json.loads(path.read_text())["owner"] == "fc-other"

    # The call whose commit came last sees its own lease and goes ahead.
    lease = VolumeLease.acquire(
        run_dir, "fc-other", subject="run r1", commit=lambda: None, reload=lambda: None
    )
    lease.release()
    assert not path.exists()


# -- the lab memory is written by one Function only --------------------------------------


def lab_files(volume: FakeVolume) -> dict[str, bytes]:
    root = volume.root / "memory"
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_a_run_adds_notes_to_the_lab_memory_and_changes_nothing_else_there(pipeline, volume):
    """Runs share the lab wiki without locking, so only consolidate_memory may rewrite it."""
    lab = lab_wiki(volume)
    lab.write("wiki/lessons/latex.md", "Load cleveref after hyperref.", {"summary": "LaTeX"})
    lab.set_standing_notes("- Compile twice after changing the bibliography.")
    lab.rebuild_index()
    # A page the index does not list yet and a note that waits for the consolidation: a run
    # that rebuilt the index, or tidied the wiki, would show.
    lab.write("wiki/lessons/figures.md", "Prefer vector graphics.", {"summary": "Figures"})
    lab.add_note("Keep tables narrow.", stage="promotion", kind="lesson")
    before = lab_files(volume)
    assert "MEMORY.md" in before and "figures" not in before["MEMORY.md"].decode()

    pipeline.lessons = [("wiki/lessons/tables.md", "Rules from booktabs read better.")]
    pipeline.lookups = [("doi:10.1000/182", {"title": "A paper the run looked up"})]
    job = {**stage_inputs(volume), "memory": {"dream": "off"}}
    summary = run_job(job, commit=volume.commit, reload=volume.reload, mount=volume.root)
    assert summary["success"]

    after = lab_files(volume)
    # The index, every page and every earlier note are byte for byte what they were.
    assert {path: after[path] for path in before} == before
    # What the run left is in files of its own, which no other container writes: its lesson
    # as a note, and what it looked up as an entry of the cache beside the wiki.
    entry, note = sorted(set(after) - set(before))
    assert note.startswith("notes/") and entry.startswith("cache/literature/")
    assert "Rules from booktabs read better." in after[note].decode()
    assert "A paper the run looked up" in after[entry].decode()
    harness = pipeline.runs[0]["harness"]
    assert harness.lab.root == volume.root / "memory" and harness.consolidate_lab is False


def test_the_lab_wiki_of_a_new_volume_is_created_by_its_one_writer(
    pipeline, volume, scripted_model
):
    """Not even the first, empty index is written by a run: the last commit of a file wins."""
    scripted_model(lambda request: "nothing to do")
    index = volume.root / "memory" / "MEMORY.md"
    calls: list[tuple[str, int]] = []
    found: list[tuple[bytes, int] | None] = []
    left: list[tuple[bytes, int]] = []

    def state() -> tuple[bytes, int] | None:
        return (index.read_bytes(), index.stat().st_mtime_ns) if index.is_file() else None

    def consolidate(model: str) -> dict[str, Any]:
        calls.append((model, volume.reloads))
        found.append(state())
        report = consolidate_lab(model, commit=volume.commit, mount=volume.root)
        left.append(state())
        return report

    job = {**stage_inputs(volume), "models": {"reflection": "fake-dreamer"}}
    summary = run_job(
        job, commit=volume.commit, reload=volume.reload, consolidate=consolidate, mount=volume.root
    )
    assert summary["success"]
    # Asked for before the run opened the wiki, and again after it for the run's lessons.
    assert calls == [("fake-dreamer", 1), ("fake-dreamer", 3)]
    assert found[0] is None and left[0] is not None
    # In between, the run reloaded the volume to see the new wiki and left its index alone.
    assert found[1] == left[0]

    again = run_job(
        stage_inputs(volume, "run-2"),
        commit=volume.commit,
        consolidate=consolidate,
        mount=volume.root,
    )
    assert again["success"] and len(calls) == 3  # the wiki exists: only the final call

    # A run that does not use the lab memory does not have it created.
    fresh = FakeVolume(volume.root.parent / "other-volume")
    job = {**stage_inputs(fresh), "memory": {"lab": False}}
    assert run_job(job, commit=fresh.commit, consolidate=consolidate, mount=fresh.root)["success"]
    assert len(calls) == 3 and not (fresh.root / "memory").exists()


# -- keyless credentials -----------------------------------------------------------------


def test_identity_token_claims_describe_the_token_without_revealing_it(tmp_path):
    token = identity_token(**MODAL_CLAIMS)
    token_file = tmp_path / "token"
    token_file.write_text(token + "\n")
    expected = {
        "iss": "https://oidc.modal.com",
        "aud": "oidc.modal.com",
        "workspace_id": "ac-1",
        "environment_name": "main",
        "app_name": "paper-orchestra",
        "function_name": "consolidate_memory",
        "lifetime_seconds": 172_800,
        "single_use": True,
    }
    assert identity_token_claims({"MODAL_IDENTITY_TOKEN_FILE": str(token_file)}) == expected
    assert identity_token_claims({"MODAL_IDENTITY_TOKEN": token}) == expected
    # A token chosen explicitly is the one the SDK presents.
    other = identity_token(iss="https://issuer.example", iat=0, exp=600)
    assert identity_token_claims(
        {"ANTHROPIC_IDENTITY_TOKEN": other, "MODAL_IDENTITY_TOKEN": token}
    ) == {"iss": "https://issuer.example", "lifetime_seconds": 600, "single_use": False}
    for unreadable in (
        {},
        {"MODAL_IDENTITY_TOKEN": "not-a-jwt"},
        {"MODAL_IDENTITY_TOKEN": "a.b.c"},
        {"MODAL_IDENTITY_TOKEN": "e30.WyJhIl0.c2ln"},  # the payload is a list
        {"MODAL_IDENTITY_TOKEN_FILE": str(tmp_path / "missing")},
    ):
        assert identity_token_claims(unreadable) == {}


def test_a_refused_identity_token_comes_with_what_to_check():
    token = identity_token(**MODAL_CLAIMS)
    environ = {**WIF_IDS, "MODAL_IDENTITY_TOKEN": token}
    refused = (
        "stage 'outline' failed: LLMConfigError: Anthropic credentials could not be used: "
        "WorkloadIdentityError: 401 Authentication failed."
    )
    explained = explain_credential_failure(refused, environ)
    assert explained.startswith(refused + "\n") and explained.endswith(federation_advice(environ))
    for expected in (
        "federation history of the Anthropic Console",
        "https://oidc.modal.com with audience oidc.modal.com",
        "172800 s for this one",
        "workspace_id=ac-1",
        "app_name=paper-orchestra",
        "function_name=consolidate_memory",
        "must admit both write_paper and consolidate_memory",
        "can be exchanged only once",
        "PAPER_ORCHESTRA_MODAL_TIMEOUT",
    ):
        assert expected in explained, expected
    assert token not in explained and "one-time" not in explained

    # Without a readable token the advice still names what to compare.
    assert "it could not be read" in federation_advice(WIF_IDS)
    # Only a federated credential that was not accepted gets the advice.
    assert explain_credential_failure("stage 'outline' failed: boom", environ) == (
        "stage 'outline' failed: boom"
    )
    assert explain_credential_failure(refused, {"ANTHROPIC_API_KEY": "sk"}) == refused
    assert explain_credential_failure(None, environ) is None


def test_run_job_explains_a_refused_identity_token(pipeline, volume, monkeypatch, logs):
    claims = {**MODAL_CLAIMS, "function_name": "write_paper"}
    for name, value in {**WIF_IDS, "MODAL_IDENTITY_TOKEN": identity_token(**claims)}.items():
        monkeypatch.setenv(name, value)
    pipeline.success = False
    pipeline.failure = (
        "stage 'outline' failed: LLMConfigError: Anthropic credentials could not be used "
        "(writer): WorkloadIdentityError: 401 Authentication failed."
    )
    summary = run_job(stage_inputs(volume), commit=volume.commit, mount=volume.root)
    assert summary["success"] is False and summary["error"].startswith(pipeline.failure)
    assert "function_name=write_paper" in summary["error"]
    assert "federation history of the Anthropic Console" in summary["error"]
    assert "Anthropic credential in use: workload_identity." in logs.text
    assert "Identity token: iss=https://oidc.modal.com" in logs.text
    assert "lifetime_seconds=172800" in logs.text

    # The same when the credential is found unusable before the first stage.
    pipeline.error = LLMConfigError(
        "Anthropic credentials could not be used: IdentityTokenFileError: no such file"
    )
    summary = run_job(stage_inputs(volume, "run-2"), commit=volume.commit, mount=volume.root)
    assert "must admit both write_paper and consolidate_memory" in summary["error"]

    # Other failures of a keyless run are left as they are.
    pipeline.error = None
    pipeline.failure = "stage refinement failed: boom"
    summary = run_job(stage_inputs(volume, "run-3"), commit=volume.commit, mount=volume.root)
    assert summary["error"] == "stage refinement failed: boom"


def test_consolidate_memory_checks_and_explains_credentials_like_write_paper(
    volume, monkeypatch, scripted_model, logs
):
    """It authenticates with a token of its own: a rule pinned to write_paper rejects it."""
    lab_wiki(volume).add_note("A lesson.", kind="lesson")

    def refused(request: FakeRequest) -> str:
        raise LLMConfigError(
            "Anthropic credentials could not be used (dream.lab): WorkloadIdentityError: "
            "401 Authentication failed."
        )

    scripted_model(refused)
    report = consolidate_lab("fake-dreamer", commit=volume.commit, mount=volume.root)
    assert "No Anthropic credential is configured in this container" in logs.text
    assert report["notes_archived"] == 0 and "WorkloadIdentityError" in report["error"]
    assert "federation history" not in report["error"]  # not a keyless container

    for name, value in {**WIF_IDS, "MODAL_IDENTITY_TOKEN": identity_token(**MODAL_CLAIMS)}.items():
        monkeypatch.setenv(name, value)
    report = consolidate_lab("fake-dreamer", commit=volume.commit, mount=volume.root)
    assert "Anthropic credential in use: workload_identity." in logs.text
    assert "function_name=consolidate_memory" in logs.text
    assert "WorkloadIdentityError" in report["error"]
    assert "function_name=consolidate_memory" in report["error"]
    assert "must admit both write_paper and consolidate_memory" in report["error"]
    assert json.loads(json.dumps(report)) == report
    assert len(lab_wiki(volume).pending_notes()) == 1  # the lesson waits for the next attempt


# -- the image -----------------------------------------------------------------------------


def test_the_image_is_built_from_a_source_checkout(tmp_path, repo_root):
    assert project_root(Path(modal_app.__file__)) == repo_root
    checkout = tmp_path / "checkout"
    module = checkout / "src" / "paper_orchestra" / "modal_app.py"
    module.parent.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\n")
    (checkout / "uv.lock").write_text("version = 1\n")
    assert project_root(module) == checkout

    (checkout / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError, match="run from a source checkout") as raised:
        project_root(module)
    assert f"uv.lock is not in {checkout}" in str(raised.value)

    wheel = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "paper_orchestra"
    with pytest.raises(FileNotFoundError) as raised:
        project_root(wheel / "modal_app.py")
    assert f"pyproject.toml and uv.lock are not in {wheel.parents[1]}" in str(raised.value)
    assert "installed wheel" in str(raised.value)
    assert "uv sync --extra modal" in str(raised.value)


def test_an_installed_copy_of_the_package_refuses_to_define_the_modal_app(tmp_path):
    """Instead of an image build that fails on a path inside the virtual environment."""
    site = tmp_path / "site-packages"
    shutil.copytree(
        Path(modal_app.__file__).parent,
        site / "paper_orchestra",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MODAL_", "PAPER_ORCHESTRA_", "ANTHROPIC_"))
    }
    env.update(MODAL_CONFIG_PATH=str(tmp_path / "no-such-config.toml"), PYTHONPATH=str(site))
    code = (
        "import paper_orchestra, paper_orchestra.modal_batch\n"
        f"assert paper_orchestra.__file__.startswith({str(site)!r}), paper_orchestra.__file__\n"
        "import paper_orchestra.modal_app\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env=env
    )
    assert done.returncode != 0
    assert "FileNotFoundError: The Modal app is run from a source checkout" in done.stderr
    assert f"are not in {tmp_path}" in done.stderr


def test_the_container_does_not_look_for_a_checkout(monkeypatch):
    """Inside the container the package lives at /root and no image is ever built."""

    def looked(package_file: Path) -> Path:
        raise AssertionError("looked for a checkout inside the container")

    monkeypatch.setattr(modal_app.modal, "is_local", lambda: False)
    monkeypatch.setattr(modal_app, "project_root", looked)
    assert isinstance(modal_app._pipeline_image(modal_app.SETTINGS), modal.Image)


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

    def fake_run_job(job, *, commit, reload, consolidate, owner):
        commit()
        reload()
        seen["job"] = job
        seen["owner"] = owner
        seen["report"] = consolidate("some-model")
        if job["run_id"] == "unreachable":
            raise modal.exception.ConnectionError("could not commit the volume")
        if job["run_id"] == "timed-out":
            raise TimeoutError("no answer within 60 s")
        return {"success": True}

    def fake_consolidate_lab(model, *, commit, reload, owner):
        commit()
        reload()
        return {"model": model, "owner": owner}

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
        # The lease on the run is held under the id of the function call (none out here).
        assert seen == {
            "job": {"run_id": "r"},
            "owner": modal.current_function_call_id(),
            "report": {"model": "some-model", "owner": modal.current_function_call_id()},
        }
        assert (fake_volume.commits, fake_volume.reloads) == (2, 2)
        # A Modal error raised by the run is the run's failure.  Passed on as it is, whoever
        # waits for the call would read it as "Modal could not be asked about the call".
        with pytest.raises(RuntimeError, match="ConnectionError: could not commit") as raised:
            modal_app.write_paper.local({"run_id": "unreachable"})
        assert not isinstance(raised.value, modal.exception.Error)
        # FunctionCall.get raises the builtin TimeoutError while a call is running.  A call
        # that ended with one would look as if it were still running, for ever.
        with pytest.raises(RuntimeError, match="TimeoutError: no answer within 60 s") as raised:
            modal_app.write_paper.local({"run_id": "timed-out"})
        assert not isinstance(raised.value, TimeoutError)


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

    # Launching the run id again replaces its inputs: what is no longer there locally must
    # not stay on the volume and find its way into the paper.
    (raw / "figures" / "plot.png").unlink()
    (raw / "figures" / "new.png").write_bytes(b"\x89PNG new")
    (volume.root / "inputs/other-run").mkdir()
    upload_inputs(volume, "r1", raw, template)
    uploaded = volume.root / "inputs/r1/raw_materials"
    assert sorted(path.name for path in (uploaded / "figures").iterdir()) == ["new.png"]
    assert (uploaded / "idea_sparse.md").read_text() == "local idea"
    assert (volume.root / "inputs/other-run").is_dir()  # other runs are left alone
    # Only what is no longer among the inputs was removed, once the new files were there.  A
    # run may be reading its inputs (a second launch is turned away by the run itself, after
    # this upload), and it must never find them missing.
    assert volume.removed == ["inputs/r1/raw_materials/figures/plot.png"]

    # A directory that is gone locally goes with everything in it.
    shutil.rmtree(raw / "figures")
    upload_inputs(volume, "r1", raw, template)
    assert not (uploaded / "figures").exists()
    assert volume.removed[1:] == ["inputs/r1/raw_materials/figures"]

    # An upload that fails leaves the inputs of the earlier launch where they were.
    volume.upload_error = modal.exception.ConnectionError("Deadline exceeded")
    with pytest.raises(modal.exception.ConnectionError):
        upload_inputs(volume, "r1", raw, template)
    assert (uploaded / "idea_sparse.md").read_text() == "local idea"
    assert sorted(path.name for path in (volume.root / "inputs/r1/template").iterdir()) == [
        "guidelines.md",
        "style.sty",
        "template.tex",
    ]
    volume.upload_error = None

    # Whatever a listing says, nothing outside the inputs of this run is ever removed.
    (volume.root / "runs/r1").mkdir(parents=True)
    (volume.root / "runs/r1/state.json").write_text("{}")
    kinds = modal.volume.FileEntryType
    volume.listdir = lambda path, recursive=False: [
        types.SimpleNamespace(path="/runs/r1/state.json", type=kinds.FILE),
        types.SimpleNamespace(path="runs/r1", type=kinds.DIRECTORY),
        types.SimpleNamespace(path="inputs/r1/../../runs/r1/state.json", type=kinds.FILE),
        types.SimpleNamespace(path="inputs/other-run", type=kinds.DIRECTORY),
        types.SimpleNamespace(path="/inputs/r1/raw_materials/idea_sparse.md", type=kinds.FILE),
    ]
    upload_inputs(volume, "r1", raw, template)
    assert len(volume.removed) == 2
    assert (volume.root / "runs/r1/state.json").is_file()
    assert (volume.root / "inputs/other-run").is_dir()
    assert (uploaded / "idea_sparse.md").is_file()  # named with a leading slash, still known


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
    assert isinstance(modal_app.consolidate_memory, modal.Function)
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
        (modal.Volume.remove_file, ("inputs/r",), {"recursive": True}),
        (modal.Volume.listdir, ("inputs/r",), {"recursive": True}),
        (modal.Volume.commit, (), {}),
        (modal.Volume.reload, (), {}),
        (modal.Function.spawn, ({"run_id": "r"},), {}),
        (modal.Function.remote, (DEFAULT_MODEL,), {}),
        (modal.Function.map, ([{"run_id": "r"}],), {"return_exceptions": True}),
        (modal.FunctionCall.get, (), {"timeout": 60.0}),
        (modal.Image.build, (modal_app.app,), {}),
    ]
    for function, args, kwargs in calls:
        inspect.signature(function).bind(None, *args, **kwargs)
    assert hasattr(modal.FunctionCall, "object_id")
    # What a listing of the volume is made of.
    entry = {field.name for field in dataclasses.fields(modal.volume.FileEntry)}
    assert {"path", "type"} <= entry
    assert {"FILE", "DIRECTORY"} <= set(modal.volume.FileEntryType.__members__)
    inspect.signature(modal.current_function_call_id).bind()
    assert modal.is_local() is True


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


def test_main_builds_the_sandbox_images_then_launches(monkeypatch):
    monkeypatch.setattr(modal_app, "setup_logging", lambda **_: None)
    built: list[Any] = []

    def image(kind: str):
        return lambda *args: types.SimpleNamespace(build=lambda app: built.append((kind, app)))

    monkeypatch.setattr(modal_app, "python_image", image("python"))
    monkeypatch.setattr(modal_app, "latex_image", image("latex"))
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
    # Both where the build log is visible: the image LaTeX is compiled in, and the one for
    # figure code.
    assert built == [("latex", modal_app.app), ("python", modal_app.app)]
    assert seen["volume"] is modal_app.volume and seen["function"] is modal_app.write_paper
    launcher_options = ("volume", "function", "raw_materials_dir", "template", "output_dir")
    assert [seen.pop(name) for name in launcher_options[2:]] == ["raw", "cvpr2025", "out"]
    job = build_job(**{k: v for k, v in seen.items() if k not in launcher_options})
    assert job["run_id"] == "r1" and job["use_plotting"] is True
    # Naming only the writer keeps discovery on the writer's model, as `write` does.
    assert job["models"] == {
        "writer": "claude-fable-5-2",
        "fast": "claude-fable-5-2",
        "effort": "low",
    }
    assert job["memory"] == {"enabled": True, "lab": False, "dream": "end"}

    del built[:]
    for run_id in ("failing", "not-downloaded"):
        with pytest.raises(SystemExit) as raised:
            modal_app.main(raw_materials_dir="raw", template="cvpr2025", run_id=run_id)
        assert raised.value.code == 1
    # Without --plotting no figure code runs, so only the LaTeX image is built.
    assert built == [("latex", modal_app.app)] * 2

    # Compiling in the container needs no LaTeX sandbox, hence no image for one.
    del built[:]
    monkeypatch.setattr(
        modal_app, "SETTINGS", dataclasses.replace(modal_app.SETTINGS, latex="local")
    )
    modal_app.main(raw_materials_dir="raw", template="cvpr2025", run_id="r2")
    assert built == []


def test_main_takes_one_model_for_every_role(monkeypatch):
    """`--model`, as `paper-orchestra write` has it; the options of a role still win."""
    monkeypatch.setattr(modal_app, "setup_logging", lambda **_: None)
    monkeypatch.setattr(
        modal_app, "latex_image", lambda *_: types.SimpleNamespace(build=lambda app: None)
    )
    seen: dict[str, Any] = {}

    def fake_launch(**kwargs):
        seen.update(kwargs)
        return {"success": True, "downloaded": {"pdf": "final_paper.pdf"}}

    monkeypatch.setattr(modal_app, "launch", fake_launch)
    modal_app.main(
        raw_materials_dir="raw",
        template="cvpr2025",
        run_id="r1",
        model="claude-opus-4-8",
        fast_model="claude-haiku-4-5",
    )
    roles = {name: seen[f"{name}_model"] for name in ("writer", "reflection", "plotting", "fast")}
    assert roles == {
        "writer": "claude-opus-4-8",
        "reflection": "claude-opus-4-8",
        "plotting": "claude-opus-4-8",
        "fast": "claude-haiku-4-5",
    }
    assert seen["image_model"] is None

    modal_app.main(raw_materials_dir="raw", template="cvpr2025", run_id="r2")
    assert {seen[f"{name}_model"] for name in roles} == {None}  # the pipeline's defaults


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
        "--keep-provided-figures / --no-keep-provided-figures",
        "--idea-filename TEXT",
        "--experimental-log-filename TEXT",
        "--research-cutoff TEXT",
        "--model TEXT",
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
    code = """
import json, modal
functions = {}
decorator_for = modal.App.function
def recording(self, **options):
    decorate = decorator_for(self, **options)
    def wrap(function):
        functions[function.__name__] = options
        return decorate(function)
    return wrap
modal.App.function = recording
import paper_orchestra.modal_app as m
print(json.dumps({
    "settings": list(vars(m.SETTINGS).values()),
    "app": m.app.name,
    "functions": {
        name: {
            "volumes": sorted(options.get("volumes", {})),
            "secrets": len(options["secrets"]),
            "max_containers": options.get("max_containers"),
            "single_use": options.get("single_use_containers", False),
            "retries": options.get("retries") is not None,
            "nonpreemptible": options.get("nonpreemptible", False),
            "timeout": options["timeout"],
            "pipeline_image": options["image"] is m.image,
        }
        for name, options in functions.items()
    },
}))
"""
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
    described = json.loads(done.stdout)
    functions = described.pop("functions")
    assert described == {
        "settings": ["smoke.app", "paper-orchestra", None, 21600, 4, "sandbox", []],
        "app": "smoke.app",
    }
    assert functions["write_paper"] == {
        "volumes": ["/vol"],
        "secrets": 1,  # no named Secret was asked for: only the forwarded settings
        "max_containers": 4,
        "single_use": True,
        "retries": True,  # a retry, or a restart after preemption, resumes the run
        "nonpreemptible": False,
        "timeout": 21600,
        "pipeline_image": True,
    }
    # One container at a time, each starting from the committed volume: a single page writer.
    assert functions["consolidate_memory"] == {
        "volumes": ["/vol"],
        "secrets": 1,
        "max_containers": 1,
        "single_use": True,
        "retries": False,
        "nonpreemptible": False,
        "timeout": 3600,
        "pipeline_image": True,
    }
    # The batch driver holds no credentials or volume, and is not restarted halfway through.
    assert functions["write_batch"] == {
        "volumes": [],
        "secrets": 1,
        "max_containers": None,
        "single_use": False,
        "retries": False,
        "nonpreemptible": True,
        "timeout": 86400,
        "pipeline_image": True,
    }
