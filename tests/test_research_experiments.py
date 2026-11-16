"""Experiments: plans, code, approval, runs and their records (``research/experiments.py``)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from paper_orchestra.compute import BackendUnusable, ComputeError, JobResult
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.research import experiments
from paper_orchestra.research.model import Conflict, NotAllowed, NotFound, ProjectError
from paper_orchestra.research.project import Project

BRIEF = "Does learning-rate warmup reduce the variance of early updates?"
BODY = "## Objective\n\nMeasure it.\n\n## Method\n\nSweep the warmup length.\n"

#: A one-pixel PNG, written by experiments as their figure.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000001e221bc330000000049454e44ae426082"
)

GOOD = f"""\
import json
import os

import numpy as np

rng = np.random.default_rng(0)
values = rng.normal(size=100)
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/values.csv", "w") as handle:
    handle.write("index,value\\n0,%.4f\\n" % values[0])
with open("artifacts/curve.png", "wb") as handle:
    handle.write(bytes.fromhex("{PNG.hex()}"))
print("mean of the values computed for", len(values), "draws")
with open("metrics.json", "w") as handle:
    handle.write('{{"accuracy": 0.9100, "f1": 0.8765, "steps": 100, "converged": true, '
                 '"note": "ok", "lr": 1e-05}}')
"""

QUICK = """\
import json

print("working")
with open("metrics.json", "w") as handle:
    json.dump({"loss": 0.25}, handle)
"""

BROKEN = QUICK.replace('print("working")', 'print("working", undefined_name)')

SLOW = """\
import time

print("started", flush=True)
time.sleep(60)
open("metrics.json", "w").write("{}")
"""

RECORD_KEYS = {
    "experiment",
    "number",
    "status",
    "backend",
    "handle",
    "started",
    "finished",
    "seconds",
    "exit_code",
    "timed_out",
    "metrics",
    "artifacts",
    "error",
    "cost_usd",
    "code_digest",
    "repairs",
}


@pytest.fixture
def project(tmp_path) -> Project:
    return Project.create(tmp_path / "p", title="Warmup and variance", brief=BRIEF)


@pytest.fixture
def backend(tmp_path) -> LocalBackend:
    return LocalBackend(scratch_dir=tmp_path / "scratch")


def planned(project: Project, files=None, *, accept=True, hypothesis=None, **fields):
    """An experiment as the designer proposes it, with its code, accepted by the person."""
    page = project.create_page(
        "experiment",
        "Sweep warmup length",
        BODY,
        author="agent:designer",
        fields={"mode": "script", **fields},
        relations={"tests": [hypothesis]} if hypothesis else None,
    )
    if files:
        experiments.write_code(project, page.id, files, author="agent:designer")
    if accept:
        project.review(page.path, "accept")
    return project.page(page.path)


def approved(project: Project, backend, files, **fields):
    page = planned(project, files, **fields)
    name = getattr(backend, "name", "local")
    return experiments.approve(project, page.id, backends={name: backend})


def events_of(project: Project, type: str) -> list[dict]:
    return [event for event in project.events() if event["type"] == type]


def states(project: Project) -> list[str]:
    return [event["data"]["state"] for event in events_of(project, "experiment.state")]


def run_dir(project: Project, number: int = 1, experiment: str = "E1") -> Path:
    return project.root / "experiments" / experiment / "runs" / str(number)


def script_writing(metrics: str) -> str:
    return f"open('metrics.json', 'w').write({metrics!r})\n"


class FakeModal:
    """What the engine sees of a remote backend: a handle at once, a job that outlives it."""

    name = "modal"
    cpu = 2.0
    memory_mb = 4096

    def __init__(self, *, attach_error: str | None = None) -> None:
        self.attached: list[str] = []
        self.specs: list = []
        self.during: dict = {}
        self.attach_error = attach_error

    def supports(self, spec):
        return []

    def check(self, **kwargs):
        return []

    def close(self):
        pass

    def describe(self):
        jobs = {"supported": True, "gpu": True, "network_isolation": True, "reattach": True}
        return {"backend": self.name, "jobs": jobs}

    def _finish(self, directory: Path, handle: str) -> JobResult:
        (directory / "log.txt").write_text("epoch 1\nepoch 2\n")
        (directory / "metrics.json").write_text('{"loss": 0.25}')
        return JobResult(
            returncode=0,
            seconds=1.5,
            backend=self.name,
            handle=handle,
            cost_usd=0.0123,
            log_path=directory / "log.txt",
            outputs_dir=directory,
            notes=["The image was built for this job."],
        )

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None):
        self.specs.append(spec)
        directory = Path(run_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "handle.txt").write_text("sb-123\n")
        record: dict = {}
        for _ in range(100):  # a silent job: the engine is only asked whether to stop
            should_stop()
            record = json.loads((directory / "run.json").read_text())
            if record["handle"]:
                break
            time.sleep(0.02)
        self.during = record
        on_log("epoch 1")
        return self._finish(directory, "sb-123")

    def attach(self, handle, *, run_dir, on_log=None, should_stop=None):
        self.attached.append(handle)
        if self.attach_error:
            return JobResult(
                returncode=None, backend=self.name, handle=handle, error=self.attach_error
            )
        return self._finish(Path(run_dir), handle)


# -- plans -------------------------------------------------------------------------------


def test_a_plan_is_read_from_the_front_matter_of_the_page(project):
    page = planned(
        project,
        mode="gpu",
        gpu="T4",
        timeout_seconds=600,
        network=True,
        packages=["torch>=2.4", "numpy"],
        est_minutes=12,
        est_usd=0.4,
        entrypoint="train.py",
    )
    assert experiments.load_plan(project, page.id) == experiments.ExperimentPlan(
        id="E1",
        path=page.path,
        title="Sweep warmup length",
        mode="gpu",
        entrypoint="train.py",
        command=None,
        timeout_seconds=600.0,
        gpu="T4",
        network=True,
        packages=("torch>=2.4", "numpy"),
        repo=None,
        repo_ref=None,
        backend=None,
        est_minutes=12.0,
        est_usd=0.4,
    )
    assert experiments.load_plan(project, page.path).id == "E1"  # any reference to the page


def test_a_plan_has_defaults_and_its_timeout_never_exceeds_the_projects_limit(tmp_path, project):
    page = planned(project)
    plan = experiments.load_plan(project, page.id)
    assert (plan.mode, plan.entrypoint, plan.timeout_seconds) == ("script", "main.py", 900.0)
    assert (plan.gpu, plan.network, plan.packages, plan.command) == (None, False, (), None)
    assert experiments.job_spec(plan).command == ["python", "main.py"]

    short = Project.create(
        tmp_path / "short", title="Short", settings={"limits": {"experiment_seconds": 300}}
    )
    assert experiments.load_plan(short, planned(short).id).timeout_seconds == 300.0


def test_a_page_written_by_hand_is_read_leniently(project):
    text = (
        "---\ntitle: Hand-made\nkind: experiment\nid: E7\nreview: accepted\nauthor: human\n"
        'timeout_seconds: "120"\nnetwork: yes\ngpu: none\n'
        "packages:\n- numpy\n- scipy>=1.10\ncommand: python main.py --seed 3\n---\n\nBody.\n"
    )
    file = project.root / "wiki" / "experiments" / "E7-hand-made.md"
    file.parent.mkdir(parents=True)
    file.write_text(text)
    plan = experiments.load_plan(project, "E7")
    assert plan.mode == "script" and plan.timeout_seconds == 120.0 and plan.network is True
    assert plan.gpu is None and plan.packages == ("numpy", "scipy>=1.10")
    assert plan.command == ["python", "main.py", "--seed", "3"]
    assert experiments.job_spec(plan).command == ["python", "main.py", "--seed", "3"]

    inline = text.replace("packages:\n- numpy\n- scipy>=1.10\n", "packages: [numpy, scipy]\n")
    file.write_text(inline)
    assert experiments.load_plan(project, "E7").packages == ("numpy", "scipy")


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"mode": "notebook"}, "script, gpu, repo"),
        ({"mode": "gpu", "gpu": "T4", "backend": "local"}, "GPU"),
        ({"mode": "gpu"}, "names its GPU"),
        ({"mode": "script", "gpu": "T4"}, "mode: gpu"),
        ({"timeout_seconds": 5000}, "900"),
        ({"timeout_seconds": "soon"}, "number of seconds"),
        ({"timeout_seconds": -5}, "number of seconds"),
        ({"packages": ["torch; rm -rf /"]}, "plain requirement"),
        ({"est_usd": "cheap"}, "est_usd"),
        ({"backend": "cluster"}, "local, modal"),
        ({"entrypoint": "../main.py"}, "entrypoint"),
        ({"network": "maybe"}, "true or false"),
    ],
)
def test_a_plan_that_cannot_run_is_refused_with_the_reason(project, fields, expected):
    page = planned(project, **fields)
    with pytest.raises(ProjectError, match=re.escape(expected)) as refused:
        experiments.load_plan(project, page.id)
    assert "E1" in str(refused.value)


def test_every_problem_of_a_plan_is_named_at_once(project):
    page = planned(project, mode="notebook", timeout_seconds=-5, est_minutes="long")
    with pytest.raises(ProjectError) as refused:
        experiments.load_plan(project, page.id)
    message = str(refused.value)
    assert "notebook" in message and "-5" in message and "est_minutes" in message


def test_only_an_experiment_has_a_plan(project):
    with pytest.raises(ProjectError, match="not an experiment"):
        experiments.load_plan(project, "wiki/idea.md")
    with pytest.raises(NotFound):
        experiments.load_plan(project, "E9")


# -- code --------------------------------------------------------------------------------


def test_code_is_written_under_the_experiment_and_earlier_versions_are_kept(project):
    planned(project, accept=False)
    files = {"main.py": QUICK, "lib/helpers.py": "X = 1\n"}
    assert experiments.write_code(project, "E1", files, author="agent:designer") == []
    code = project.root / "experiments" / "E1" / "code"
    assert experiments.code_dir(project, "E1") == code
    assert experiments.code_files(project, "e1") == {"lib/helpers.py": "X = 1\n", "main.py": QUICK}

    experiments.write_code(project, "E1", {"lib/helpers.py": "X = 2\n"}, author="human")
    experiments.write_code(project, "E1", {"lib/helpers.py": "X = 2\n"}, author="human")
    experiments.write_code(project, "E1", {"lib/helpers.py": "X = 3\n"}, author="human")
    kept = sorted((code / ".history" / "lib" / "helpers.py").iterdir())
    assert [path.read_text() for path in kept] == ["X = 1\n", "X = 2\n"]
    assert kept[0].name.endswith("-agent-designer.py") and kept[1].name.endswith("-human.py")
    # A file someone put there by hand is theirs when it is replaced.
    (code / "notes.py").write_text("# by hand\n")
    experiments.write_code(project, "E1", {"notes.py": "# rewritten\n"}, author="agent:designer")
    [by_hand] = (code / ".history" / "notes.py").iterdir()
    assert by_hand.read_text() == "# by hand\n" and by_hand.name.endswith("-human.py")
    experiments.write_code(project, "E1", {"notes.py": QUICK}, author="agent:designer")
    # What was kept is not code of the experiment.
    assert set(experiments.code_files(project, "E1")) == {"lib/helpers.py", "main.py", "notes.py"}
    assert (code / "lib" / "helpers.py").read_text() == "X = 3\n"


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/abs.py",
        "a/../../b.py",
        "",
        ".hidden.py",
        "lib/.env",
        ".history/main.py",
        "metrics.json",
        "artifacts/plot.png",
        "back\\slash.py",
        "new\nline.py",
        "lib/",
    ],
)
def test_only_safe_relative_names_are_written(project, name):
    planned(project, accept=False)
    with pytest.raises(ProjectError):
        experiments.write_code(project, "E1", {"main.py": QUICK, name: "x = 1\n"}, author="human")
    # Nothing of the call was written, not even the file with a good name.
    assert experiments.code_files(project, "E1") == {}
    assert not (project.root / "escape.py").exists() and not (project.root / "b.py").exists()


def test_code_is_not_written_through_a_link_that_leads_elsewhere(project, tmp_path):
    planned(project, {"main.py": QUICK}, accept=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    (experiments.code_dir(project, "E1") / "lib").symlink_to(outside)
    files = {"main.py": QUICK + "# changed\n", "lib/helpers.py": "X = 1\n"}
    with pytest.raises(ProjectError, match="link"):
        experiments.write_code(project, "E1", files, author="agent:designer")
    assert list(outside.iterdir()) == []
    assert experiments.code_files(project, "E1") == {"main.py": QUICK}  # all or nothing


def test_code_is_text_of_a_sensible_size_for_an_experiment_that_exists(project):
    planned(project, accept=False)
    with pytest.raises(ProjectError, match="text"):
        experiments.write_code(project, "E1", {"main.py": b"x = 1"}, author="human")
    with pytest.raises(ProjectError, match="larger than"):
        experiments.write_code(project, "E1", {"main.py": "#" * 2_000_000}, author="human")
    with pytest.raises(ProjectError, match="No files"):
        experiments.write_code(project, "E1", {}, author="human")
    with pytest.raises(ProjectError, match="experiment"):
        experiments.write_code(project, "H1", {"main.py": QUICK}, author="human")
    with pytest.raises(ProjectError):
        experiments.write_code(project, "E1", {"main.py": QUICK}, author="nobody in particular")


def test_an_agent_cannot_change_the_code_of_a_plan_the_person_decided_on(project, backend):
    page = planned(project, {"main.py": QUICK})  # accepted by the person
    with pytest.raises(NotAllowed, match="accepted"):
        experiments.write_code(project, page.id, {"main.py": BROKEN}, author="agent:designer")
    experiments.approve(project, page.id, backends={"local": backend})
    with pytest.raises(NotAllowed):
        experiments.write_code(project, page.id, {"main.py": BROKEN}, author="agent:designer")
    assert experiments.code_files(project, page.id) == {"main.py": QUICK}
    # The runner's repair and the person's own edit are what may still change it.
    assert (
        experiments.write_code(project, page.id, {"main.py": QUICK + "#\n"}, author="agent:runner")
        == []
    )
    assert experiments.write_code(project, page.id, {"main.py": QUICK}, author="human") == []


def test_an_agent_writes_code_only_for_its_own_proposal(project):
    planned(project, accept=False)
    with pytest.raises(NotAllowed, match="agent:designer"):
        experiments.write_code(project, "E1", {"main.py": QUICK}, author="agent:analyst")
    assert experiments.write_code(project, "E1", {"main.py": QUICK}, author="agent:designer") == []


def test_an_agent_cannot_rewrite_code_approved_without_review(tmp_path, backend):
    settings = {"review": {"experiment": "auto"}}
    project = Project.create(tmp_path / "auto", title="Auto", settings=settings)
    page = planned(project, {"main.py": QUICK}, accept=False)
    assert page.review == "accepted" and not page.decided_by
    experiments.write_code(project, "E1", {"main.py": QUICK + "#\n"}, author="agent:designer")
    experiments.approve(project, "E1", backends={"local": backend})
    with pytest.raises(NotAllowed, match="approved"):
        experiments.write_code(project, "E1", {"main.py": BROKEN}, author="agent:designer")


def test_code_is_parsed_and_never_run(project, tmp_path, monkeypatch):
    planned(project, accept=False)
    monkeypatch.chdir(tmp_path)
    code = "open('ran.txt', 'w').write('x')\n" + QUICK
    assert experiments.write_code(project, "E1", {"main.py": code}, author="agent:designer") == []
    assert experiments.check_code(project, "E1") == []
    assert not list(tmp_path.rglob("ran.txt"))


def test_check_code_reports_what_cannot_run(project):
    planned(project, accept=False)
    [nothing] = experiments.check_code(project, "E1")
    assert "no code" in nothing and "main.py" in nothing

    problems = experiments.write_code(
        project, "E1", {"helper.py": "def broken(:\n    pass\n"}, author="agent:designer"
    )
    errors, _ = experiments.split_problems(problems)
    assert any("helper.py" in error and "line 1" in error for error in errors)
    assert any("main.py" in error and "entry point" in error for error in errors)

    fixed = {"helper.py": "def fine():\n    pass\n", "main.py": QUICK}
    assert experiments.write_code(project, "E1", fixed, author="agent:designer") == []


def test_the_entry_point_is_the_one_the_page_names(project):
    planned(project, accept=False, entrypoint="train.py")
    [missing] = experiments.write_code(project, "E1", {"main.py": QUICK}, author="agent:designer")
    assert "train.py" in missing
    assert experiments.write_code(project, "E1", {"train.py": QUICK}, author="agent:designer") == []


NETWORKED = """\
import json
import requests
from urllib import request
import urllib.request as fetch
from torchvision import datasets

data = datasets.MNIST("data", download=True)
json.dump({"n": 1}, open("metrics.json", "w"))
"""


def test_network_use_without_approval_is_a_warning_not_an_error(project):
    planned(project, accept=False)
    problems = experiments.write_code(
        project, "E1", {"main.py": NETWORKED}, author="agent:designer"
    )
    errors, warnings = experiments.split_problems(problems)
    assert errors == [] and all(item.startswith(experiments.WARNING) for item in problems)
    assert any("requests" in item and "line 2" in item for item in warnings)
    assert any("urllib.request" in item for item in warnings)
    assert any("download=True" in item and "line 7" in item for item in warnings)
    assert all("network: true" in item for item in warnings)


def test_network_use_is_fine_once_the_page_asks_for_the_network(project):
    planned(project, accept=False, network=True)
    assert (
        experiments.write_code(project, "E1", {"main.py": NETWORKED}, author="agent:designer") == []
    )


def test_code_that_never_mentions_metrics_gets_a_warning(project):
    planned(project, accept=False)
    [warning] = experiments.write_code(
        project, "E1", {"main.py": "print('hello')\n"}, author="agent:designer"
    )
    assert warning.startswith(experiments.WARNING) and "metrics.json" in warning


# -- approval ----------------------------------------------------------------------------


def test_approval_records_the_options_and_moves_the_experiment_and_its_hypothesis(project, backend):
    hypothesis = project.create_page(
        "hypothesis", "Warmup reduces variance", "## Statement\n\nIt does.", author="agent:theorist"
    )
    project.review(hypothesis.path, "accept")
    page = planned(project, {"main.py": QUICK}, hypothesis="H1")
    assert page.state == "planned" and project.page("H1").state == "open"

    done = experiments.approve(project, page.id, timeout_seconds=120, backends={"local": backend})

    assert done.state == "approved" and done.path == page.path
    assert done.meta["backend"] == "local" and done.meta["timeout_seconds"] == 120
    assert done.meta["network"] is False and "gpu" not in done.meta
    plan = experiments.load_plan(project, "E1")
    assert plan.backend == "local" and plan.timeout_seconds == 120.0
    assert project.page("H1").state == "testing"
    [moved] = events_of(project, "experiment.state")
    assert moved["data"]["state"] == "approved" and moved["data"]["by"] == "human"


def test_a_hypothesis_accepted_after_the_approval_is_under_test_once_the_run_is_queued(
    project, backend
):
    hypothesis = project.create_page(
        "hypothesis", "Warmup reduces variance", "## Statement\n\nIt does.", author="agent:theorist"
    )
    approved(project, backend, {"main.py": QUICK}, hypothesis="H1")
    assert project.page("H1").state == "open"  # still a proposal when the run was approved
    project.review(hypothesis.path, "accept")
    experiments.run(project, "E1", compute=backend)
    assert project.page("H1").state == "testing"


def test_accepting_a_page_does_not_approve_its_run(project, backend):
    page = planned(project, {"main.py": QUICK})
    assert page.review == "accepted" and page.state == "planned"
    with pytest.raises(ProjectError, match=r"[Aa]pprove"):
        experiments.run(project, page.id, compute=backend)
    assert experiments.runs(project, "E1") == [] and project.page("E1").state == "planned"


def test_approval_is_refused_with_the_reasons(project, backend):
    backends = {"local": backend}
    proposed = planned(project, {"main.py": QUICK}, accept=False)
    with pytest.raises(ProjectError, match="not accepted"):
        experiments.approve(project, proposed.id, backends=backends)
    project.review(proposed.path, "accept")

    with pytest.raises(ProjectError, match="900") as refused:
        experiments.approve(project, "E1", timeout_seconds=5000, backends=backends)
    assert "5000" in str(refused.value)
    with pytest.raises(ProjectError, match="GPU"):
        experiments.approve(project, "E1", gpu="T4", backends=backends)
    with pytest.raises(Conflict):
        experiments.approve(project, "E1", base_updated="2020-01-01T00:00:00Z", backends=backends)
    read = experiments.code_digest(project, "E1")
    experiments.write_code(project, "E1", {"main.py": QUICK + "# changed\n"}, author="human")
    with pytest.raises(Conflict, match="code of E1 changed"):
        experiments.approve(project, "E1", code_digest=read, backends=backends)

    experiments.write_code(project, "E1", {"main.py": "def broken(:\n"}, author="human")
    with pytest.raises(ProjectError, match=r"main\.py"):
        experiments.approve(project, "E1", backends=backends)

    page = project.page("E1")
    assert page.state == "planned" and "backend" not in page.meta
    assert events_of(project, "experiment.state") == []


def test_approval_is_refused_when_the_backend_cannot_run_the_plan(project, backend):
    planned(project, {"main.py": QUICK}, packages=["surely-not-installed-anywhere"])
    with pytest.raises(ProjectError, match="surely-not-installed-anywhere") as refused:
        experiments.approve(project, "E1", backends={"local": backend})
    assert "cannot be approved" in str(refused.value)
    with pytest.raises(ProjectError, match="tape"):
        experiments.approve(project, "E1", backend="tape", backends={"local": backend})


def test_an_experiment_without_code_is_not_approved(project, backend):
    planned(project)
    with pytest.raises(ProjectError, match="no code"):
        experiments.approve(project, "E1", backends={"local": backend})


def test_a_gpu_experiment_is_approved_for_modal_whatever_the_projects_default(project):
    assert project.settings.compute == "local"
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    done = experiments.approve(project, "E1", backends={"modal": FakeModal()})
    assert done.state == "approved" and done.meta["backend"] == "modal" and done.meta["gpu"] == "T4"


def test_whatever_asks_for_a_gpu_is_approved_for_modal_unless_the_person_says_otherwise(
    tmp_path, project
):
    assert project.settings.compute == "local"
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "train.py").write_text(QUICK)
    project.settings.update({"repo": {"path": str(repository)}})
    project.save_settings()
    modal = FakeModal()
    # In the person's repository, on a GPU the page names ...
    planned(project, mode="repo", command="python train.py", gpu="T4")
    done = experiments.approve(project, "E1", backends={"modal": modal})
    assert (done.meta["backend"], done.meta["gpu"]) == ("modal", "T4")
    # ... or one the person names when approving.
    planned(project, mode="repo", command="python train.py")
    done = experiments.approve(project, "E2", gpu="A10G", backends={"modal": modal})
    assert (done.meta["backend"], done.meta["gpu"]) == ("modal", "A10G")
    # Without a GPU it runs where the project's runs run.
    planned(project, mode="repo", command="python train.py")
    with pytest.raises(ProjectError, match="GPU"):
        experiments.approve(project, "E3", gpu="T4", backend="local", backends={"modal": modal})
    assert "backend" not in project.page("E3").meta


def test_approval_for_modal_needs_modal_to_be_usable(project):
    pytest.importorskip("modal")
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    # No backend is handed in: the real one is built, and it has no credentials here.
    with pytest.raises(ProjectError, match="not authenticated"):
        experiments.approve(project, "E1")
    [choice] = experiments.options(project, "E1", ["modal"])
    assert choice["can_run"] is False and choice["max_usd"] == pytest.approx(0.2426, abs=1e-3)
    assert project.page("E1").state == "planned"


def test_an_estimate_above_the_limit_is_refused_with_the_numbers(tmp_path, backend):
    project = Project.create(tmp_path / "p", title="Costly")
    planned(project, {"main.py": QUICK}, est_usd=9.5)
    with pytest.raises(ProjectError, match=re.escape("$9.50")) as refused:
        experiments.approve(project, "E1", backends={"local": backend})
    assert "$5.00" in str(refused.value) and "experiment_usd" in str(refused.value)


def test_what_a_run_could_cost_at_its_time_limit_is_checked_against_the_limit(tmp_path):
    limits = {"experiment_seconds": 3600, "experiment_usd": 0.1}
    project = Project.create(tmp_path / "p", title="Costly", settings={"limits": limits})
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="A100", timeout_seconds=3600)
    with pytest.raises(ProjectError, match=re.escape("$2.48")) as refused:
        experiments.approve(project, "E1", backends={"modal": FakeModal()})
    assert "$0.10" in str(refused.value) and "3600" in str(refused.value)


class Billing(FakeModal):
    """A remote backend whose every job costs the same."""

    def __init__(self, cost: float) -> None:
        super().__init__()
        self.cost = cost

    def _finish(self, directory: Path, handle: str) -> JobResult:
        result = super()._finish(directory, handle)
        result.cost_usd = self.cost
        return result


def test_the_cost_limit_is_for_all_the_runs_of_the_project_together(tmp_path, backend):
    limits = {"experiment_usd": 1.0}
    project = Project.create(tmp_path / "p", title="Costly", settings={"limits": limits})
    modal = Billing(0.7)
    first = planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4", est_usd=0.6)
    second = planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4", est_usd=0.6)
    for page in (first, second):
        experiments.approve(project, page.id, backends={"modal": modal})
    assert experiments.run(project, first.id, compute=modal)["cost_usd"] == 0.7

    # $0.70 is spent: another $0.60 would pass the dollar the project allows.
    record = experiments.run(project, second.id, compute=modal)
    assert (record["status"], record["error_kind"]) == ("failed", "budget")
    assert "$0.60" in record["error"] and "$0.70" in record["error"] and "$1.00" in record["error"]
    assert len(modal.specs) == 1  # the second job was never started
    [warning] = events_of(project, "budget.warning")
    assert warning["data"]["experiment"] == second.id
    with pytest.raises(ProjectError, match=re.escape("$0.70")):
        experiments.approve(project, second.id, backends={"modal": modal})
    [choice] = experiments.options(project, second.id, {"modal": modal})
    assert choice["can_run"] is False and "$0.70" in choice["reasons"][0]

    # Without an estimate of its own, it is what the run can cost at its time limit that
    # has to fit into what is left ($0.24 for 900 s of a T4, and $0.30 is left).
    third = planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, third.id, backends={"modal": modal})
    modal.cost = 0.1
    assert experiments.run(project, third.id, compute=modal)["status"] == "succeeded"
    fourth = planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    with pytest.raises(ProjectError, match=re.escape("$0.80")) as refused:
        experiments.approve(project, fourth.id, backends={"modal": modal})
    assert "$0.24" in str(refused.value)

    # A GPU without a known price cannot be bounded; it runs while something is left.
    unpriced = [planned(project, {"main.py": QUICK}, mode="gpu", gpu="X9000") for _ in range(3)]
    for page in unpriced[:2]:
        experiments.approve(project, page.id, backends={"modal": modal})
    modal.cost = 0.3
    assert experiments.run(project, unpriced[0].id, compute=modal)["status"] == "succeeded"
    record = experiments.run(project, unpriced[1].id, compute=modal)
    assert record["error_kind"] == "budget" and "$1.10 of $1.00 is spent" in record["error"]
    with pytest.raises(ProjectError, match="Nothing is left"):
        experiments.approve(project, unpriced[2].id, backends={"modal": modal})

    # A run that is not billed is not held back by money that was spent elsewhere.
    free = planned(project, {"main.py": QUICK})
    experiments.approve(project, free.id, backends={"local": backend})
    assert experiments.run(project, free.id, compute=backend)["status"] == "succeeded"


class FailingOnce(Billing):
    """A billed job that fails for an error of its code the first time it runs."""

    def _finish(self, directory: Path, handle: str) -> JobResult:
        result = super()._finish(directory, handle)
        if len(self.specs) == 1:
            (directory / "log.txt").write_text("NameError: name 'x' is not defined\n")
            (directory / "metrics.json").unlink()
            result.returncode = 1
        return result


def test_a_caller_that_keeps_the_accounts_says_what_a_call_may_spend(tmp_path):
    limits = {"experiment_usd": 1.0}
    project = Project.create(tmp_path / "p", title="Costly", settings={"limits": limits})
    modal = Billing(0.9)
    for _ in range(3):
        page = planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
        experiments.approve(project, page.id, backends={"modal": modal})
    assert experiments.run(project, "E1", compute=modal)["cost_usd"] == 0.9

    # The engine passes what is left of the total.  That amount is not reduced a second
    # time by what the records say was spent: $0.24 fits into the $0.50 it names.
    record = experiments.run(project, "E2", compute=modal, limits={"experiment_usd": 0.5})
    assert record["status"] == "succeeded"

    # The amount is for the whole call: what a failed attempt cost is gone when its
    # repair wants to run.
    failing = FailingOnce(0.4)
    record = experiments.run(
        project,
        "E3",
        compute=failing,
        repair=lambda *asked: {"main.py": QUICK + "# repaired\n"},
        limits={"experiment_usd": 0.5},
    )
    first, second = experiments.runs(project, "E3")
    assert (first["status"], first["error_kind"], first["cost_usd"]) == ("failed", "exit", 0.4)
    assert record == second and (second["status"], second["error_kind"]) == ("failed", "budget")
    assert "$0.40" in second["error"] and "$0.50" in second["error"]
    assert len(failing.specs) == 1

    # So is what a run cost that this call found unfinished and settled before its own.
    class Lost(Billing):
        def attach(self, handle, *, run_dir, on_log=None, should_stop=None):
            return JobResult(
                returncode=None, backend=self.name, error="The sandbox is gone.", cost_usd=0.4
            )

    page = planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    project.set_state(page.path, "approved", by="human")
    dead_run(project, experiment=str(page.id), backend="modal", handle="sb-4")
    lost = Lost(0.1)
    record = experiments.run(project, page.id, compute=lost, limits={"experiment_usd": 0.5})
    assert (record["number"], record["error_kind"]) == (2, "budget") and lost.specs == []

    # Nothing left, and a GPU without a known price: not started either.
    page = planned(project, {"main.py": QUICK}, mode="gpu", gpu="X9000")
    project.set_state(page.path, "approved", by="human")
    record = experiments.run(project, page.id, compute=modal, limits={"experiment_usd": 0})
    assert record["error_kind"] == "budget" and "Nothing is left" in record["error"]


def test_options_say_what_each_backend_can_do_with_the_plan(project, backend):
    planned(project, {"main.py": NETWORKED}, mode="gpu", gpu="T4", est_minutes=10)
    choices = experiments.options(project, "E1", {"local": backend, "modal": FakeModal()})
    assert [choice["backend"] for choice in choices] == ["local", "modal"]
    local, modal = choices
    assert {"backend", "can_run", "reasons", "est_minutes", "est_usd", "notes"} <= set(local)
    assert local["can_run"] is False and any("GPU" in reason for reason in local["reasons"])
    assert any("cannot block network" in note for note in local["notes"])
    assert modal["can_run"] is True and modal["reasons"] == []
    assert modal["est_minutes"] == 10
    assert modal["est_usd"] == pytest.approx(0.1617, abs=1e-3)  # ten minutes of a T4 sandbox
    assert any("requests" in note for note in modal["notes"])  # what the person should know
    assert not any("network: true" in note and "Approving" in note for note in modal["notes"])
    # Nothing was decided by asking.
    assert project.page("E1").state == "planned" and "backend" not in project.page("E1").meta


def test_options_take_backend_names_too(project):
    planned(project, {"main.py": QUICK}, est_minutes=1, est_usd=0, network=True)
    [choice] = experiments.options(project, "E1", ["local"])
    assert choice["backend"] == "local" and choice["can_run"] is True
    assert choice["est_usd"] == 0 and choice["est_minutes"] == 1
    assert any("Approving the run approves that" in note for note in choice["notes"])
    [unknown] = experiments.options(project, "E1", ["tape"])
    assert unknown["can_run"] is False and "tape" in unknown["reasons"][0]


# -- runs --------------------------------------------------------------------------------


def test_an_approved_experiment_runs_and_its_result_is_recorded(project, backend):
    approved(project, backend, {"main.py": GOOD})
    seen: list[str] = []

    record = experiments.run(project, "E1", compute=backend, on_log=seen.append)

    assert set(record) >= RECORD_KEYS
    assert (record["experiment"], record["number"], record["status"]) == ("E1", 1, "succeeded")
    assert (record["backend"], record["handle"], record["exit_code"]) == ("local", None, 0)
    assert record["timed_out"] is False and record["error"] is None and record["cost_usd"] is None
    assert record["metrics"] == {
        "accuracy": 0.91,
        "f1": 0.8765,
        "steps": 100,
        "converged": True,
        "note": "ok",
        "lr": 1e-05,
    }
    assert record["artifacts"] == ["artifacts/curve.png", "artifacts/values.csv"]
    assert record["repairs"] == [] and record["seconds"] > 0
    assert record["started"] <= record["finished"]
    assert re.fullmatch(r"[0-9a-f]{64}", record["code_digest"])
    assert record["code_digest"] == experiments.code_digest(project, "E1")
    assert json.loads((run_dir(project) / "run.json").read_text()) == record
    assert experiments.runs(project, "E1") == [record]
    assert (run_dir(project) / "artifacts" / "curve.png").read_bytes() == PNG
    assert (run_dir(project) / "code" / "main.py").read_text() == GOOD  # the code that ran
    assert any("mean of the values" in line for line in seen)

    assert project.page("E1").state == "succeeded"
    result = project.page("wiki/results/E1-run1.md")
    assert record["result"] == result.path
    assert (result.kind, result.state, result.review, result.author) == (
        "result",
        "succeeded",
        "accepted",
        "system",
    )
    assert result.relations == {"run_of": ["E1"]}
    # The values are the run's own text, not a number formatted again.
    assert "| accuracy | 0.9100 |" in result.body and "| steps | 100 |" in result.body
    assert "| converged | true |" in result.body and "| lr | 1e-05 (0.00001) |" in result.body
    assert "![curve.png](/files/experiments/E1/runs/1/artifacts/curve.png)" in result.body
    assert "[values.csv](/files/experiments/E1/runs/1/artifacts/values.csv)" in result.body
    assert "mean of the values" in result.body  # the end of the log
    assert "cannot block the network" in result.body  # what the backend noted
    assert "succeeded" in result.summary and "accuracy" in result.summary

    assert states(project) == ["approved", "queued", "running", "succeeded"]
    types = [event["type"] for event in project.events()]
    assert types.index("run.started") < types.index("run.log") < types.index("run.finished")
    [started], [finished] = events_of(project, "run.started"), events_of(project, "run.finished")
    assert started["data"] == {"experiment": "E1", "run": 1, "backend": "local"}
    assert finished["data"]["status"] == "succeeded" and result.path in finished["refs"]
    assert all(event["agent"] == "runner" for event in (started, finished))


def test_the_backend_can_be_chosen_per_plan(project, backend):
    approved(project, backend, {"main.py": QUICK})
    asked = []

    def choose(plan):
        asked.append((plan.id, plan.backend))
        return backend

    assert experiments.run(project, "E1", compute=choose)["status"] == "succeeded"
    assert asked == [("E1", "local")]


def test_an_experiment_runs_only_on_the_backend_it_was_approved_for(project, backend):
    approved(project, backend, {"main.py": QUICK})
    modal = FakeModal()
    record = experiments.run(project, "E1", compute=modal)
    assert record["status"] == "failed" and record["error_kind"] == "setup"
    assert "approved for the local backend" in record["error"] and modal.specs == []
    assert project.page("E1").state == "failed"


def test_a_programming_error_is_repaired_and_run_again_as_the_next_run(project, backend):
    approved(project, backend, {"main.py": BROKEN})
    calls = []

    def repair(plan, record, log_tail):
        calls.append((plan.id, record["number"], record["status"], log_tail))
        return {"main.py": QUICK}

    record = experiments.run(project, "E1", compute=backend, repair=repair)

    first, second = experiments.runs(project, "E1")
    assert record == second and (second["number"], second["status"]) == (2, "succeeded")
    assert (first["status"], first["exit_code"], first["error_kind"]) == ("failed", 1, "exit")
    assert "NameError" in first["error"] and first["repairs"] == [] and first["metrics"] == {}
    assert len(calls) == 1 and calls[0][:3] == ("E1", 1, "failed")
    assert "NameError: name 'undefined_name' is not defined" in calls[0][3]
    [made] = second["repairs"]
    assert made["round"] == 1 and "NameError" in made["reason"]
    assert '-print("working", undefined_name)' in made["diff"]
    assert '+print("working")' in made["diff"] and "main.py" in made["diff"]
    assert first["code_digest"] != second["code_digest"]
    # The correction is the runner's, and what the designer wrote is kept.
    [kept] = (experiments.code_dir(project, "E1") / ".history" / "main.py").iterdir()
    assert kept.read_text() == BROKEN and kept.name.endswith("-agent-designer.py")
    assert experiments.code_files(project, "E1") == {"main.py": QUICK}

    assert project.page("E1").state == "succeeded"
    assert project.page("E1-run1").state == "failed"
    repaired = project.page("E1-run2")
    assert repaired.state == "succeeded" and "## Repairs" in repaired.body
    assert "undefined_name" in repaired.body
    assert states(project) == ["approved", "queued", "running", "succeeded"]


def test_repairs_are_bounded_by_the_projects_limit(tmp_path, backend):
    settings = {"limits": {"repair_rounds": 2}}
    project = Project.create(tmp_path / "p", title="Bounded", settings=settings)
    approved(project, backend, {"main.py": BROKEN})
    rounds = []

    def repair(plan, record, log_tail):
        rounds.append(record["number"])
        return {"main.py": BROKEN + f"# attempt {len(rounds)}\n"}

    record = experiments.run(project, "E1", compute=backend, repair=repair)
    assert rounds == [1, 2] and (record["number"], record["status"]) == (3, "failed")
    assert [made["round"] for made in record["repairs"]] == [1, 2]
    assert project.page("E1").state == "failed"
    assert "NameError" in project.page("E1").meta["state_reason"]


def test_a_result_page_stays_within_the_size_of_a_page_however_much_was_repaired(tmp_path, backend):
    settings = {"limits": {"repair_rounds": 6}}
    project = Project.create(tmp_path / "p", title="Long repairs", settings=settings)
    approved(project, backend, {"main.py": BROKEN})
    rounds = []

    def repair(plan, record, log_tail):
        rounds.append(record["number"])
        filler = "".join(f"# line {len(rounds)}-{n} of a long correction\n" for n in range(2500))
        return {"main.py": (QUICK if len(rounds) == 6 else BROKEN) + filler}

    record = experiments.run(project, "E1", compute=backend, repair=repair)
    assert (record["number"], record["status"], len(record["repairs"])) == (7, "succeeded", 6)
    result = project.page(record["result"])
    assert len(result.body.encode()) < 100_000 and "run.json" in result.body
    assert all(len(made["diff"]) > 50_000 for made in record["repairs"])  # whole, on record


def test_a_failure_stands_when_nothing_repairs_it(tmp_path, backend):
    settings = {"limits": {"repair_rounds": 0}}
    project = Project.create(tmp_path / "p", title="No repairs", settings=settings)
    approved(project, backend, {"main.py": BROKEN})
    never = pytest.fail
    record = experiments.run(project, "E1", compute=backend, repair=lambda *a: never("no rounds"))
    assert (record["number"], record["status"]) == (1, "failed")

    for answer in (None, {}, {"main.py": BROKEN}):  # no correction, or one that changes nothing
        experiments.approve(project, "E1", backends={"local": backend})
        record = experiments.run(
            project,
            "E1",
            compute=backend,
            repair=lambda *a, answer=answer: answer,
            limits={"repair_rounds": 3},
        )
        assert record["status"] == "failed" and record["repairs"] == []
    assert len(experiments.runs(project, "E1")) == 4


def test_a_correction_that_does_not_parse_is_not_started(project, backend):
    approved(project, backend, {"main.py": BROKEN})
    answers = [{"main.py": "def broken(:\n"}, {"main.py": QUICK}]
    seen = []

    def repair(plan, record, log_tail):
        seen.append(record["error"])
        return answers.pop(0)

    record = experiments.run(project, "E1", compute=backend, repair=repair)
    first, second, third = experiments.runs(project, "E1")
    assert (second["status"], second["error_kind"], second["exit_code"]) == ("failed", "code", None)
    assert "main.py" in second["error"] and "line 1" in second["error"]
    assert seen == [first["error"], second["error"]]
    assert record == third and third["status"] == "succeeded" and len(third["repairs"]) == 2


def test_a_repairer_that_fails_leaves_the_failed_run_standing(project, backend):
    approved(project, backend, {"main.py": BROKEN})

    def repair(plan, record, log_tail):
        raise RuntimeError("the model is not available")

    record = experiments.run(project, "E1", compute=backend, repair=repair)
    assert record["status"] == "failed" and project.page("E1").state == "failed"
    assert any("the model is not available" in note for note in record["notes"])


def test_a_run_that_exceeds_its_time_limit_fails_and_is_not_repaired(project, backend):
    approved(project, backend, {"main.py": SLOW}, timeout_seconds=1)
    never = pytest.fail
    record = experiments.run(project, "E1", compute=backend, repair=lambda *a: never("a timeout"))
    assert record["status"] == "failed" and record["timed_out"] is True
    assert record["error_kind"] == "timeout" and "1 s" in record["error"]
    assert project.page("E1").state == "failed" and project.page("E1-run1").state == "failed"


def test_a_command_that_cannot_be_started_is_not_something_to_repair(project, backend):
    page = planned(project, {"main.py": QUICK}, command="no-such-program main.py")
    [warning] = experiments.check_code(project, page.id)
    assert warning.startswith(experiments.WARNING) and "no-such-program main.py" in warning
    experiments.approve(project, "E1", backends={"local": backend})
    never = pytest.fail
    record = experiments.run(project, "E1", compute=backend, repair=lambda *a: never("no code"))
    assert (record["status"], record["exit_code"], record["error_kind"]) == (
        "failed",
        127,
        "command",
    )
    assert "could not be started" in record["error"]


def test_the_time_limit_is_capped_when_the_projects_limit_was_lowered(project, backend):
    approved(project, backend, {"main.py": SLOW}, timeout_seconds=600)
    project.settings.update({"limits": {"experiment_seconds": 1}})
    project.save_settings()
    record = experiments.run(project, "E1", compute=backend)
    assert record["timed_out"] is True and record["seconds"] < 30
    assert any("600" in note and "1 s" in note for note in record["notes"])


def test_the_person_can_cancel_a_running_experiment(project, backend):
    approved(project, backend, {"main.py": SLOW})

    def on_log(line):
        if "started" in line:
            experiments.request_cancel(project, "E1")

    record = experiments.run(project, "E1", compute=backend, on_log=on_log)
    assert record["status"] == "cancelled" and record["requeued"] is False
    assert record["seconds"] < 30 and "cancelled" in record["error"]
    assert project.page("E1").state == "cancelled"
    assert project.page("E1-run1").state == "cancelled"
    # Nothing of the request is left to stop the next run.
    experiments.write_code(project, "E1", {"main.py": QUICK}, author="human")
    experiments.approve(project, "E1", backends={"local": backend})
    assert experiments.run(project, "E1", compute=backend)["status"] == "succeeded"


def test_an_experiment_that_waits_is_cancelled_at_once(project, backend):
    planned(project, {"main.py": QUICK})
    experiments.request_cancel(project, "E1")  # nothing was approved: nothing to cancel
    assert project.page("E1").state == "planned"
    experiments.approve(project, "E1", backends={"local": backend})
    experiments.request_cancel(project, "E1")
    assert project.page("E1").state == "cancelled" and experiments.runs(project, "E1") == []
    with pytest.raises(ProjectError):
        experiments.run(project, "E1", compute=backend)


def test_cancelling_an_experiment_whose_run_just_ended_changes_nothing(
    project, backend, monkeypatch
):
    approved(project, backend, {"main.py": QUICK})
    write_text = experiments.write_text

    def run_ends_first(path, text):
        # The run ends between the look at the page and the request being left for it.
        monkeypatch.setattr(experiments, "write_text", write_text)
        experiments.run(project, "E1", compute=backend)
        return write_text(path, text)

    monkeypatch.setattr(experiments, "write_text", run_ends_first)
    experiments.request_cancel(project, "E1")  # no error: there is nothing left to cancel
    assert project.page("E1").state == "succeeded"
    assert not (project.root / "experiments" / "E1" / ".cancel").exists()


def test_a_stopped_worker_leaves_the_experiment_queued_and_the_same_call_resumes_it(
    project, backend
):
    approved(project, backend, {"main.py": SLOW})
    heard = []
    record = experiments.run(
        project, "E1", compute=backend, should_stop=lambda: bool(heard), on_log=heard.append
    )
    assert record["status"] == "cancelled" and record["requeued"] is True
    assert record["result"] is None and project.resolve("wiki/results/E1-run1.md") is None
    assert project.page("E1").state == "queued"

    experiments.write_code(project, "E1", {"main.py": QUICK}, author="human")
    record = experiments.run(project, "E1", compute=backend)
    assert (record["number"], record["status"]) == (2, "succeeded")
    assert states(project) == ["approved", "queued", "running", "queued", "running", "succeeded"]


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_never_leaves_a_run_or_the_page_at_running(project, backend, stop):
    approved(project, backend, {"main.py": SLOW})

    def on_log(line):
        raise stop

    with pytest.raises(stop):
        experiments.run(project, "E1", compute=backend, on_log=on_log)
    [record] = experiments.runs(project, "E1")
    assert record["status"] == "cancelled" and record["requeued"] is True and record["finished"]
    assert project.page("E1").state == "queued"


class Raising(FakeModal):
    name = "local"

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None):
        raise self.error


def test_an_error_of_the_pipeline_fails_the_run_and_is_raised(project):
    backend = Raising(ComputeError("The run directory cannot be used."))
    approved(project, backend, {"main.py": QUICK})
    with pytest.raises(ComputeError):
        experiments.run(project, "E1", compute=backend)
    [record] = experiments.runs(project, "E1")
    assert record["status"] == "failed" and record["error_kind"] == "internal"
    assert "The run directory cannot be used." in record["error"]
    assert project.page("E1").state == "failed"


def test_a_backend_that_cannot_work_is_not_the_experiments_failure(project):
    backend = Raising(BackendUnusable("Modal is not authenticated."))
    approved(project, backend, {"main.py": QUICK})
    with pytest.raises(BackendUnusable):
        experiments.run(project, "E1", compute=backend)
    [record] = experiments.runs(project, "E1")
    assert record["status"] == "failed" and record["requeued"] is True
    assert "Modal is not authenticated." in record["error"]
    assert project.page("E1").state == "queued"  # it runs once the backend works again


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (None, "without writing metrics.json"),
        ("not json at all", "not valid JSON"),
        ("[1, 2]", "not an object"),
        ('{"per_seed": {"a": 1}}', '"per_seed"'),
        ('{"per_seed": [0.9, 0.8]}', "[0.9, 0.8]"),
        ('{"loss": NaN}', "NaN"),
        ('{"loss": 1e999}', "1e999"),
        ('{"loss": null}', "null"),
        ('{"note": "' + "x" * 300 + '"}', "longer than"),
        ("{}", "empty"),
    ],
)
def test_metrics_are_read_strictly(project, backend, metrics, expected):
    code = "print('done')\n" if metrics is None else script_writing(metrics)
    approved(project, backend, {"main.py": code + "# metrics.json\n"})
    record = experiments.run(project, "E1", compute=backend)
    assert (record["status"], record["error_kind"], record["exit_code"]) == ("failed", "metrics", 0)
    assert expected in record["error"] and record["metrics"] == {}
    assert project.page("E1").state == "failed"
    assert expected in project.page("E1-run1").body


def test_the_metrics_of_a_run_that_failed_are_not_reported(project, backend):
    code = QUICK + "raise SystemExit(3)\n"
    approved(project, backend, {"main.py": code})
    record = experiments.run(project, "E1", compute=backend)
    assert (record["status"], record["exit_code"], record["metrics"]) == ("failed", 3, {})
    body = project.page("E1-run1").body
    assert "| loss |" not in body and "did not finish" in body


def test_what_a_job_writes_is_only_text_on_its_result_page(project, backend):
    code = (
        "import json, os\nos.makedirs('artifacts')\n"
        "open('artifacts/a](evil)[b.csv', 'w').write('x')\n"
        "metrics = {'[click](http://evil.example)': '[[idea]] `x` <b>|', 'n': 1}\n"
        "json.dump(metrics, open('metrics.json', 'w'))\n"
    )
    approved(project, backend, {"main.py": code})
    record = experiments.run(project, "E1", compute=backend)
    assert record["metrics"] == {"[click](http://evil.example)": "[[idea]] `x` <b>|", "n": 1}
    body = project.page(record["result"]).body
    # Shown as the text it is: no link, no wiki link, no markup, no broken table.
    assert "| \\[click\\](http://evil.example) | \\[\\[idea\\]\\] \\`x\\` \\<b>\\| |" in body
    assert "[a)(evil)(b.csv](/files/experiments/E1/runs/1/artifacts/a%5D%28evil%29%5Bb.csv)" in body
    edges = [edge for edge in project.graph()["edges"] if edge["source"] == record["result"]]
    assert [(edge["type"], edge["target"]) for edge in edges] == [
        ("run_of", project.page("E1").path)
    ]


def test_the_name_of_an_artifact_cannot_write_lines_of_its_own_into_the_record(project, backend):
    code = (
        "import json, os\nos.makedirs('artifacts')\n"
        "open('artifacts/a\\n\\n## Verdict\\n\\nAccuracy was 99.87.png', 'wb').write(b'x')\n"
        "open('artifacts/t\\n- see [[idea]] \\\\.csv', 'w').write('x')\n" + QUICK
    )
    approved(project, backend, {"main.py": code})
    record = experiments.run(project, "E1", compute=backend)
    assert record["status"] == "succeeded" and len(record["artifacts"]) == 2
    body = project.page(record["result"]).body
    start = body.index("## Artifacts")
    listed = [line for line in body[start : body.index("## Notes")].splitlines()[1:] if line]
    assert len(listed) == 2 and listed[0].startswith("![a ## Verdict Accuracy was 99.87.png](")
    assert listed[1].startswith("- [t - see ((idea)) \\\\.csv](/files/experiments/E1/runs/1/")
    assert "\n## Verdict" not in body
    edges = [edge for edge in project.graph()["edges"] if edge["source"] == record["result"]]
    assert [edge["type"] for edge in edges] == ["run_of"]


def test_artifacts_are_listed_within_their_size_limit(project, backend, monkeypatch):
    monkeypatch.setattr(experiments, "MAX_ARTIFACT_BYTES", 100)
    code = (
        "import os\nos.makedirs('artifacts/deep')\n"
        "open('artifacts/deep/small.csv', 'w').write('a,b\\n')\n"
        "open('artifacts/huge.bin', 'w').write('x' * 5000)\n"
        "open('elsewhere.txt', 'w').write('not an artifact')\n" + QUICK
    )
    approved(project, backend, {"main.py": code})
    record = experiments.run(project, "E1", compute=backend)
    assert record["status"] == "succeeded"
    assert record["artifacts"] == ["artifacts/deep/small.csv"]
    assert not (run_dir(project) / "artifacts" / "huge.bin").exists()
    assert any("huge.bin" in note for note in record["notes"])


def test_a_code_directory_full_of_data_is_not_copied_for_a_run(project, backend, monkeypatch):
    approved(project, backend, {"main.py": QUICK})
    (experiments.code_dir(project, "E1") / "data.bin").write_bytes(b"0" * 5000)
    monkeypatch.setattr(experiments, "MAX_CODE_BYTES", 1000)
    record = experiments.run(project, "E1", compute=backend)
    assert (record["status"], record["error_kind"]) == ("failed", "setup")
    assert "keep data out of it" in record["error"]


def test_a_run_whose_files_cannot_be_put_in_place_fails_without_ending_the_worker(
    project, backend, monkeypatch
):
    approved(project, backend, {"main.py": QUICK})

    def full(source, target):
        raise OSError(28, "No space left on device", str(target))

    monkeypatch.setattr(experiments.shutil, "copyfile", full)
    record = experiments.run(project, "E1", compute=backend)  # a record, not an exception
    assert (record["status"], record["error_kind"]) == ("failed", "setup")
    assert "No space left on device" in record["error"]
    assert project.page("E1").state == "failed" and project.page("E1-run1").state == "failed"


def test_approving_again_gives_the_next_run_number(project, backend):
    approved(project, backend, {"main.py": QUICK})
    assert experiments.run(project, "E1", compute=backend)["number"] == 1
    with pytest.raises(ProjectError, match="succeeded"):
        experiments.run(project, "E1", compute=backend)  # one approval, one run
    experiments.approve(project, "E1", backends={"local": backend})
    assert experiments.run(project, "E1", compute=backend)["number"] == 2
    assert [record["number"] for record in experiments.runs(project, "E1")] == [1, 2]
    assert project.page("E1-run1").state == project.page("E1-run2").state == "succeeded"


def test_a_run_never_takes_the_number_of_a_result_that_is_on_record(project, backend):
    approved(project, backend, {"main.py": QUICK})
    experiments.run(project, "E1", compute=backend)
    first = project.page("E1-run1").body
    # The person cleared the run directories; the record of run 1 is still in the wiki,
    # and a finding may rest on it.
    shutil.rmtree(project.root / "experiments" / "E1" / "runs")
    experiments.write_code(
        project, "E1", {"main.py": QUICK.replace("0.25", "0.75")}, author="human"
    )
    experiments.approve(project, "E1", backends={"local": backend})
    record = experiments.run(project, "E1", compute=backend)
    assert (record["number"], record["result"]) == (2, "wiki/results/E1-run2.md")
    assert project.page("E1-run1").body == first and project.history("E1-run1") == []


def test_runs_lists_what_can_be_read(project, backend):
    assert experiments.runs(project, "E1") == []
    approved(project, backend, {"main.py": QUICK})
    experiments.run(project, "E1", compute=backend)
    runs = project.root / "experiments" / "E1" / "runs"
    (runs / "notes").mkdir()
    (runs / "7").mkdir()
    (runs / "7" / "run.json").write_text("{ not json")
    assert [record["number"] for record in experiments.runs(project, "e1")] == [1]


def test_log_events_are_throttled(project, backend):
    chatty = (
        "import time\nfor i in range(40):\n    print('step', i, flush=True)\n    time.sleep(0.01)\n"
    )
    approved(project, backend, {"main.py": chatty + QUICK})
    began = time.monotonic()
    experiments.run(project, "E1", compute=backend)
    took = time.monotonic() - began
    logs = events_of(project, "run.log")
    assert 1 <= len(logs) <= 1 + took // 2  # forty lines, and one event every two seconds
    assert logs[0]["data"]["experiment"] == "E1" and logs[0]["data"]["run"] == 1
    assert logs[0]["data"]["lines"] and "E1" in logs[0]["summary"]


def test_only_one_process_runs_an_experiment(project, backend):
    approved(project, backend, {"main.py": SLOW})
    started = threading.Event()
    outcome: dict = {}

    def first():
        outcome["record"] = experiments.run(
            project, "E1", compute=backend, on_log=lambda line: started.set()
        )

    worker = threading.Thread(target=first)
    worker.start()
    try:
        assert started.wait(30)
        with pytest.raises(ProjectError, match="already"):
            experiments.run(project, "E1", compute=backend)
        # A run that is alive is not something to recover.
        assert experiments.recover(project) == []
        assert project.page("E1").state == "running"
    finally:
        experiments.request_cancel(project, "E1")
        worker.join(60)
    assert outcome["record"]["status"] == "cancelled"


# -- remote jobs and recovery ------------------------------------------------------------


def test_the_handle_of_a_remote_job_is_recorded_while_it_runs(project):
    modal = FakeModal()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4", packages=["torch"], network=True)
    experiments.approve(project, "E1", backends={"modal": modal})
    record = experiments.run(project, "E1", compute=modal)

    assert modal.during["status"] == "running" and modal.during["handle"] == "sb-123"
    [spec] = modal.specs
    assert spec.command == ["python", "main.py"] and spec.gpu == "T4" and spec.network is True
    assert spec.packages == ("torch",) and spec.timeout == 900.0
    assert (record["status"], record["backend"], record["handle"]) == (
        "succeeded",
        "modal",
        "sb-123",
    )
    assert record["cost_usd"] == 0.0123 and record["seconds"] == 1.5
    body = project.page("E1-run1").body
    assert "The image was built for this job." in body and "$0.01" in body


def dead_run(
    project: Project, number: int = 1, *, experiment="E1", backend="local", handle=None
) -> Path:
    """Leave an experiment the way a worker that died mid-run leaves it."""
    if project.page(experiment).state == "approved":
        project.set_state(experiment, "queued", by="system")
    project.set_state(experiment, "running", by="system")
    directory = run_dir(project, number, experiment)
    directory.mkdir(parents=True)
    record = {
        "experiment": experiment,
        "number": number,
        "status": "running",
        "backend": backend,
        "handle": handle,
        "started": "2026-11-15T12:00:00Z",
        "finished": None,
        "seconds": None,
        "exit_code": None,
        "timed_out": False,
        "metrics": {},
        "artifacts": [],
        "error": None,
        "cost_usd": None,
        "code_digest": "0" * 64,
        "repairs": [],
    }
    (directory / "run.json").write_text(json.dumps(record))
    (directory / "log.txt").write_text("epoch 1\n")
    return directory


def test_recover_settles_a_run_whose_worker_died(project, backend):
    approved(project, backend, {"main.py": QUICK})
    dead_run(project)

    [record] = experiments.recover(project)

    assert record["status"] == "failed" and record["error_kind"] == "worker"
    assert "worker stopped" in record["error"] and record["requeued"] is True
    assert record["finished"] and set(record) >= RECORD_KEYS
    assert json.loads((run_dir(project) / "run.json").read_text()) == record
    assert project.page("E1").state == "queued"
    assert experiments.recover(project) == []  # nothing is left to settle

    resumed = experiments.run(project, "E1", compute=backend)
    assert (resumed["number"], resumed["status"]) == (2, "succeeded")


WORKER = """
import sys

from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.research import experiments
from paper_orchestra.research.project import Project

experiments.run(Project.open(sys.argv[1]), "E1", compute=LocalBackend(scratch_dir=sys.argv[2]))
"""


def test_a_worker_that_is_killed_leaves_a_run_that_recover_settles(project, backend, tmp_path):
    approved(project, backend, {"main.py": SLOW})
    worker = subprocess.Popen(
        [sys.executable, "-c", WORKER, str(project.root), str(tmp_path / "worker-scratch")]
    )
    try:
        log = run_dir(project) / "log.txt"
        deadline = time.monotonic() + 60
        while not (log.is_file() and "started" in log.read_text()):
            assert worker.poll() is None and time.monotonic() < deadline, "the worker never ran"
            time.sleep(0.05)
        # While the worker lives, its run is nobody else's to settle.
        assert experiments.recover(project) == []
    finally:
        worker.kill()
        worker.wait()
    [left] = experiments.runs(project, "E1")
    assert left["status"] == "running" and project.page("E1").state == "running"

    [record] = experiments.recover(project)
    assert record["status"] == "failed" and record["requeued"] is True
    assert "worker stopped" in record["error"] and project.page("E1").state == "queued"


def test_an_experiment_whose_worker_keeps_dying_is_not_started_again(project, backend):
    approved(project, backend, {"main.py": QUICK})
    for number in (1, 2, 3):
        dead_run(project, number)
        [record] = experiments.recover(project)
    assert record["requeued"] is False and record["status"] == "failed"
    assert project.page("E1").state == "failed"
    assert project.page("E1-run3").state == "failed"
    assert "3" in project.page("E1-run3").body


def test_a_page_left_at_running_without_a_run_goes_back_to_the_queue(project, backend):
    approved(project, backend, {"main.py": QUICK})
    project.set_state("E1", "queued", by="system")
    project.set_state("E1", "running", by="system")
    assert experiments.recover(project) == []
    assert project.page("E1").state == "queued"


def test_recover_picks_up_a_remote_job_that_outlived_its_worker(project):
    modal = FakeModal()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})
    directory = dead_run(project, backend="modal")
    (directory / "handle.txt").write_text("sb-9\n")  # the worker died before recording it

    [record] = experiments.recover(project, compute=modal)

    assert modal.attached == ["sb-9"]
    assert (record["status"], record["number"], record["handle"]) == ("succeeded", 1, "sb-9")
    assert record["metrics"] == {"loss": 0.25} and record["cost_usd"] == 0.0123
    assert project.page("E1").state == "succeeded"
    assert project.page("E1-run1").state == "succeeded"


def test_a_remote_job_that_is_gone_is_run_again(project):
    modal = FakeModal(attach_error="The sandbox sb-9 no longer exists.")
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})
    dead_run(project, backend="modal", handle="sb-9")

    [record] = experiments.recover(project, compute=lambda plan: modal)
    assert record["status"] == "failed" and record["requeued"] is True
    assert "sb-9 no longer exists" in record["error"] and "worker stopped" in record["error"]
    assert project.page("E1").state == "queued"

    resumed = experiments.run(project, "E1", compute=modal)
    assert (resumed["number"], resumed["status"], modal.attached) == (2, "succeeded", ["sb-9"])


def test_run_settles_what_a_dead_worker_left_before_it_starts(project):
    modal = FakeModal()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})
    dead_run(project, backend="modal", handle="sb-9")
    # Without recover(): the job that is still running is picked up, not started twice.
    record = experiments.run(project, "E1", compute=modal)
    assert (record["number"], record["status"]) == (1, "succeeded")
    assert modal.attached == ["sb-9"] and modal.specs == []


class Outliving(FakeModal):
    """A remote job that is still running when it is picked up, until it is stopped."""

    def __init__(self) -> None:
        super().__init__()
        self.following = threading.Event()
        self.stopped = False

    def attach(self, handle, *, run_dir, on_log=None, should_stop=None):
        self.attached.append(handle)
        self.following.set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                self.stopped = True  # the sandbox is ended, as the real backend does
                return JobResult(
                    returncode=None, cancelled=True, seconds=1.0, backend=self.name, handle=handle
                )
            time.sleep(0.02)
        return self._finish(Path(run_dir), handle)


def test_the_person_can_cancel_a_remote_job_while_recovery_follows_it(project):
    modal = Outliving()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})
    dead_run(project, backend="modal", handle="sb-9")
    settled: list[dict] = []
    worker = threading.Thread(
        target=lambda: settled.extend(experiments.recover(project, compute=modal))
    )
    worker.start()
    try:
        assert modal.following.wait(30)
        experiments.request_cancel(project, "E1")
    finally:
        worker.join(60)

    assert modal.stopped is True  # not left to run, and to be billed, to its end
    [record] = settled
    assert record["status"] == "cancelled" and record["requeued"] is False
    assert "person" in record["error"]
    assert project.page("E1").state == "cancelled"
    assert project.page("E1-run1").state == "cancelled"
    assert not (project.root / "experiments" / "E1" / ".cancel").exists()


def test_cancelling_a_run_whose_worker_died_settles_it(project, backend):
    approved(project, backend, {"main.py": QUICK})
    dead_run(project)
    experiments.request_cancel(project, "E1")
    [record] = experiments.runs(project, "E1")
    assert record["status"] == "cancelled" and record["requeued"] is False
    assert project.page("E1").state == "cancelled"


# -- the numbers of a finding ------------------------------------------------------------


def test_numbers_in_a_finding_are_traced_to_records_sources_and_inputs(tmp_path, backend):
    table = tmp_path / "table.csv"
    table.write_text("seed,loss\n0,0.4312\n")
    project = Project.create(
        tmp_path / "p",
        title="Warmup",
        brief=BRIEF + " Earlier runs reached 73.5% accuracy.",
        files=[table],
    )
    project.write_record(
        "source", "Smith2020Deep", "Deep nets", "Reports a 12.5x speed-up over 4096 steps."
    )
    approved(project, backend, {"main.py": GOOD})
    experiments.run(project, "E1", compute=backend)
    finding = project.create_page(
        "finding",
        "Warmup helps",
        "## What was observed\n\nAccuracy reached 0.91 after 100 steps and F1 was 0.88 (87.7%), "
        "up from 73.5% before; Smith reports 12.5x over 4,096 steps. The loss fell to 0.4312, "
        "a 98765.4-fold gain over 5 seeds in 2024, with a learning rate of 0.00001.\n",
        author="agent:analyst",
        relations={"based_on": ["E1-run1"]},
    )

    checks = {
        check["number"]: check["found_in"] for check in experiments.number_check(project, finding)
    }

    result = "wiki/results/E1-run1.md"
    assert checks == {
        "0.91": [result],
        "0.88": [result],  # 0.8765, rounded as a person would
        "87.7": [result],  # the same value as a percentage
        "73.5": ["inputs/brief.md"],
        "12.5": ["wiki/sources/Smith2020Deep.md"],
        "4,096": ["wiki/sources/Smith2020Deep.md"],
        "0.4312": ["inputs/table.csv"],
        "98765.4": [],
    }
    assert experiments.unverified_numbers(project, finding) == ["98765.4"]
    assert experiments.unverified_numbers(project, finding.path) == ["98765.4"]


def test_only_accepted_records_are_evidence(tmp_path):
    settings = {"review": {"source": "review"}}
    project = Project.create(tmp_path / "p", title="Vetted sources", settings=settings)
    source = project.write_record("source", "Lee2021Wide", "Wide nets", "A gain of 41.7 points.")
    assert source.review == "proposed"
    finding = project.create_page(
        "finding", "Gain", "The gain was 41.7 points.", author="agent:analyst"
    )
    assert experiments.unverified_numbers(project, finding) == ["41.7"]
    project.review(source.path, "accept")
    assert experiments.unverified_numbers(project, finding) == []
    # The page's own words are no evidence for themselves, and neither is another finding.
    echo = project.create_page("finding", "Echo", "As found, 55.5 points.", author="human")
    other = project.create_page("finding", "Other", "It was 55.5 points.", author="agent:analyst")
    assert experiments.unverified_numbers(project, other) == ["55.5"] and echo.review == "accepted"
