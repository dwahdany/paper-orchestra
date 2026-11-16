"""Experiments: plans, code, approval, runs and their records (``research/experiments.py``)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from paper_orchestra.compute import BackendUnusable, ComputeError, JobResult
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.research import experiments
from paper_orchestra.research.model import (
    Conflict,
    NotAllowed,
    NotFound,
    ProjectError,
    format_stamp,
    parse_stamp,
)
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
        "packages:\n- numpy\n- scipy>=1.10\n---\n\nBody.\n"
    )
    file = project.root / "wiki" / "experiments" / "E7-hand-made.md"
    file.parent.mkdir(parents=True)
    file.write_text(text)
    plan = experiments.load_plan(project, "E7")
    assert plan.mode == "script" and plan.timeout_seconds == 120.0 and plan.network is True
    assert plan.gpu is None and plan.packages == ("numpy", "scipy>=1.10")
    assert plan.command is None
    assert experiments.job_spec(plan).command == ["python", "main.py"]
    # A command is read from one line, as a shell would split it, where a plan may have
    # one: in the person's repository.
    (project.root / "repository").mkdir()
    project.settings.update({"repo": {"path": "repository"}})
    in_repo = text.replace(
        "---\n\nBody", "mode: repo\ncommand: python main.py --seed '3 4'\n---\n\nBody"
    )
    file.write_text(in_repo)
    plan = experiments.load_plan(project, "E7")
    assert plan.command == ["python", "main.py", "--seed", "3 4"]
    assert experiments.job_spec(plan).command == ["python", "main.py", "--seed", "3 4"]
    file.write_text(text)

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
    # A file someone put there by hand is theirs: not an agent's to replace, and kept
    # under their name when the runner's repair or they themselves replace it.
    (code / "notes.py").write_text("# by hand\n")
    with pytest.raises(NotAllowed, match=r"notes\.py"):
        experiments.write_code(
            project, "E1", {"notes.py": "# rewritten\n"}, author="agent:designer"
        )
    experiments.write_code(project, "E1", {"notes.py": "# rewritten\n"}, author="agent:runner")
    [by_hand] = (code / ".history" / "notes.py").iterdir()
    assert by_hand.read_text() == "# by hand\n" and by_hand.name.endswith("-human.py")
    experiments.write_code(project, "E1", {"notes.py": QUICK}, author="human")
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


def test_the_designer_supplies_the_code_of_an_accepted_plan_that_came_without_any(
    project, backend, monkeypatch
):
    # The analyst proposes an experiment as a next step: a plan, no code.  Once the person
    # accepts the plan the designer writes its code, which the person reads when approving
    # the run.  Nothing the person read is changed by that.
    page = project.create_page(
        "experiment", "Follow-up sweep", BODY, author="agent:analyst", fields={"mode": "script"}
    )
    page = project.review(page.path, "accept")
    with pytest.raises(NotAllowed, match="accepted"):
        experiments.write_code(project, page.id, {"main.py": QUICK}, author="agent:analyst")
    after = parse_stamp(page.meta["decided_at"]) + timedelta(minutes=5)
    monkeypatch.setattr(experiments, "_now", lambda: format_stamp(after))
    assert (
        experiments.write_code(project, page.id, {"main.py": QUICK}, author="agent:designer") == []
    )
    # In several calls, and it may correct what it wrote there.
    experiments.write_code(project, page.id, {"helper.py": "X = 1\n"}, author="agent:designer")
    experiments.write_code(project, page.id, {"main.py": QUICK + "#\n"}, author="agent:designer")
    assert sorted(experiments.code_files(project, page.id)) == ["helper.py", "main.py"]
    assert {
        name: entry["author"] for name, entry in experiments.code_authors(project, page.id).items()
    } == {
        "helper.py": "agent:designer",
        "main.py": "agent:designer",
    }
    # A file the person put there is theirs.
    experiments.write_code(project, page.id, {"notes.py": "Y = 2\n"}, author="human")
    with pytest.raises(NotAllowed, match="accepted"):
        experiments.write_code(project, page.id, {"helper.py": "X = 3\n"}, author="agent:designer")
    # And the approval of the run closes the code like any other.
    (experiments.code_dir(project, page.id) / "notes.py").unlink()
    experiments.write_code(project, page.id, {"helper.py": "X = 1\n"}, author="human")
    experiments.approve(project, page.id, backends={"local": backend})
    with pytest.raises(NotAllowed, match="approved"):
        experiments.write_code(project, page.id, {"main.py": BROKEN}, author="agent:designer")


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
    # All of them are approved while the project's dollar is still unspent.
    for gpu in ("T4", "T4", "T4", "T4", "X9000"):
        page = planned(project, {"main.py": QUICK}, mode="gpu", gpu=gpu)
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

    dead_run(project, experiment="E4", backend="modal", handle="sb-4")
    lost = Lost(0.1)
    record = experiments.run(project, "E4", compute=lost, limits={"experiment_usd": 0.5})
    assert (record["number"], record["error_kind"]) == (2, "budget") and lost.specs == []

    # Nothing left, and a GPU without a known price: not started either.
    record = experiments.run(project, "E5", compute=modal, limits={"experiment_usd": 0})
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
    assert not any("network: true" in note and "yours to grant" in note for note in modal["notes"])
    # Nothing was decided by asking.
    assert project.page("E1").state == "planned" and "backend" not in project.page("E1").meta


def test_options_take_backend_names_too(project):
    planned(project, {"main.py": QUICK}, est_minutes=1, est_usd=0, network=True)
    [choice] = experiments.options(project, "E1", ["local"])
    assert choice["backend"] == "local" and choice["can_run"] is True
    assert choice["est_usd"] == 0 and choice["est_minutes"] == 1
    assert any("yours to grant" in note for note in choice["notes"])
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


def test_a_command_that_cannot_be_started_is_not_something_to_repair(tmp_path, project, backend):
    (tmp_path / "repository").mkdir()
    (tmp_path / "repository" / "main.py").write_text(QUICK)
    project.settings.update({"repo": {"path": str(tmp_path / "repository")}})
    project.save_settings()
    page = planned(project, mode="repo", command="no-such-program main.py")
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
    # The same code runs both times (what was approved); it is slow until a file exists.
    resumed = project.root.parent / "resumed"
    slow_once = (
        "import os, time\n\nprint('started', flush=True)\n"
        f"if not os.path.exists({str(resumed)!r}):\n    time.sleep(60)\n" + QUICK
    )
    approved(project, backend, {"main.py": slow_once})
    heard = []
    record = experiments.run(
        project, "E1", compute=backend, should_stop=lambda: bool(heard), on_log=heard.append
    )
    assert record["status"] == "cancelled" and record["requeued"] is True
    assert record["result"] is None and project.resolve("wiki/results/E1-run1.md") is None
    assert project.page("E1").state == "queued"

    resumed.write_text("")
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
    planned(project, {"main.py": QUICK})
    (experiments.code_dir(project, "E1") / "data.bin").write_bytes(b"0" * 5000)
    experiments.approve(project, "E1", backends={"local": backend})
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
    experiments.approve(project, "E1", network=True, backends={"modal": modal})
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


# ---------------------------------------------------------------------------------------
# What runs is what the person approved (audit: containment 1-4, 6, 9, 10; execution 1, 2)
# ---------------------------------------------------------------------------------------

OTHER = QUICK.replace('print("working")', 'print("OTHER ENTRY POINT RAN")')
NEVER_APPROVED = {"digest": None, "at": None, "matches": False, "changed": [], "repaired": False}


def run_dirs(project: Project, experiment: str = "E1") -> list[str]:
    folder = project.root / "experiments" / experiment / "runs"
    return sorted(entry.name for entry in folder.iterdir()) if folder.is_dir() else []


def test_a_command_belongs_to_repo_mode_only(project, backend):
    """A script or gpu experiment runs its entry point: the code the person read."""
    page = planned(project, {"main.py": QUICK}, command=["python", "-c", "print('not main.py')"])
    with pytest.raises(ProjectError, match=r"command.*mode: repo") as refused:
        experiments.load_plan(project, page.id)
    assert "E1" in str(refused.value)
    with pytest.raises(ProjectError, match="command"):
        experiments.approve(project, page.id, backends={"local": backend})
    assert project.page("E1").state == "planned"
    # Whatever a plan object holds, only repo mode starts anything but the entry point.
    plan = experiments.ExperimentPlan(
        id="E1", path="", title="", mode="script", command=["sh", "-c", "anything"]
    )
    assert experiments.job_command(plan) == ["python", "main.py"]
    assert experiments.job_spec(plan).command == ["python", "main.py"]
    plan.mode = "repo"
    assert experiments.job_command(plan) == ["sh", "-c", "anything"]


def test_an_approval_records_exactly_what_may_run(project, backend):
    planned(project, {"main.py": QUICK})
    assert experiments.approval(project, "E1") == NEVER_APPROVED
    before = experiments.plan_digest(project, "E1")

    done = experiments.approve(project, "E1", timeout_seconds=120, backends={"local": backend})

    digest = experiments.plan_digest(project, "E1")
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest != before  # where it runs and for how long are part of what is approved
    assert digest != experiments.code_digest(project, "E1")
    assert done.meta["approved_digest"] == digest
    assert parse_stamp(done.meta["approved_at"]) is not None
    assert experiments.approval(project, "E1") == {
        "digest": digest,
        "at": done.meta["approved_at"],
        "matches": True,
        "changed": [],
        "repaired": False,
    }
    # Every field that decides what runs is part of it ...
    for fields in (
        {"network": True},
        {"timeout_seconds": 60},
        {"entrypoint": "other.py"},
        {"packages": ["numpy"]},
        {"backend": "modal"},
        {"mode": "gpu", "gpu": "T4"},
    ):
        project.update_page("E1", author="human", fields=fields)
        assert experiments.plan_digest(project, "E1") != digest, fields
        assert experiments.approval(project, "E1")["matches"] is False, fields
        project.update_page(
            "E1", author="human", fields={key: done.meta.get(key) for key in fields}
        )
        assert experiments.plan_digest(project, "E1") == digest, fields
    # ... and so is the code; what the page says in words is not.
    project.update_page(
        "E1", author="human", body="Other words.\n", fields={"title": "Renamed", "est_minutes": 3}
    )
    assert experiments.plan_digest(project, "E1") == digest
    experiments.write_code(project, "E1", {"lib.py": "X = 1\n"}, author="human")
    assert experiments.plan_digest(project, "E1") != digest


def test_an_approval_holds_however_the_path_of_the_project_is_written(
    tmp_path, backend, monkeypatch
):
    """A project's repository may be given relative to the project.  The app opens a
    project by its absolute path, a command line often by a relative one: the same
    directory is the same repository, and the approval made one way holds the other."""
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "train.py").write_text(QUICK)
    settings = {"repo": {"path": "../repository"}}
    project = Project.create(tmp_path / "p", title="Relative", settings=settings)
    planned(project, mode="repo", command="python train.py")
    experiments.approve(project, "E1", backends={"local": backend})

    monkeypatch.chdir(tmp_path)
    by_name = Project.open("p")
    assert experiments.plan_digest(by_name, "E1") == project.page("E1").meta["approved_digest"]
    assert experiments.approval(by_name, "E1")["matches"] is True
    monkeypatch.chdir(tmp_path / "p")
    assert experiments.run(Project.open("."), "E1", compute=backend)["status"] == "succeeded"

    # Another directory at that place is another repository.
    experiments.approve(project, "E1", backends={"local": backend})
    repository.rename(tmp_path / "kept")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "train.py").write_text(QUICK)
    repository.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    changed = experiments.approval(project, "E1")
    assert changed["matches"] is False and "elsewhere" in changed["changed"][0]


def test_an_accepted_edit_that_changes_what_runs_needs_a_new_approval(project, backend):
    """The agent's summary says "typo"; its fields change the entry point, the network
    and the time limit of a run the person already approved."""
    approved(project, backend, {"main.py": QUICK, "other.py": OTHER}, timeout_seconds=60)
    proposal = project.propose_edit(
        "E1",
        author="agent:analyst",
        summary="Fix a typo in the method section",
        fields={"network": True, "timeout_seconds": 600, "entrypoint": "other.py"},
    )
    project.review(proposal.id, "accept")

    status = experiments.approval(project, "E1")
    assert status["matches"] is False and status["repaired"] is False
    assert status["changed"] == [
        "entrypoint: main.py when approved, now other.py",
        "network: false when approved, now true",
        "timeout_seconds: 60 when approved, now 600",
    ]
    with pytest.raises(ProjectError, match="E1 was not run") as refused:
        experiments.run(project, "E1", compute=backend)
    message = str(refused.value)
    for said in ("network: false when approved, now true", "entrypoint", "timeout_seconds"):
        assert said in message
    assert "approve it again" in message and "research approve" in message

    page = project.page("E1")
    assert page.state == "planned" and "network" in page.meta["state_reason"]
    assert experiments.runs(project, "E1") == [] and run_dirs(project) == []  # nothing ran
    moved = events_of(project, "experiment.state")[-1]
    assert (moved["data"]["previous"], moved["data"]["state"]) == ("approved", "planned")
    assert "network" in moved["data"]["reason"]
    # A new approval is a decision on what is there now.
    experiments.approve(project, "E1", network=False, backends={"local": backend})
    record = experiments.run(project, "E1", compute=backend)
    assert record["status"] == "succeeded"
    assert "OTHER ENTRY POINT RAN" in (run_dir(project) / "log.txt").read_text()


def test_code_changed_after_the_approval_is_not_run(project, backend):
    """Also the person's own change: what runs is what was approved, by whoever changed."""
    approved(project, backend, {"main.py": QUICK, "lib.py": "X = 1\n"})
    experiments.write_code(project, "E1", {"main.py": OTHER, "extra.py": "Y = 2\n"}, author="human")
    (experiments.code_dir(project, "E1") / "lib.py").unlink()

    changed = experiments.approval(project, "E1")["changed"]
    assert len(changed) == 3
    assert changed[0].startswith("extra.py: added after the approval (written by human at ")
    assert changed[1] == "lib.py: removed after the approval"
    assert changed[2].startswith("main.py: changed after the approval (written by human at ")
    with pytest.raises(ProjectError, match=r"main\.py: changed after the approval"):
        experiments.run(project, "E1", compute=backend)
    assert project.page("E1").state == "planned" and run_dirs(project) == []


def test_an_experiment_that_reads_approved_without_a_record_does_not_run(project, backend):
    """A project from before approvals were recorded, or a state set by hand."""
    page = planned(project, {"main.py": QUICK})
    project.set_state(page.path, "approved", by="human")
    assert experiments.approval(project, "E1") == NEVER_APPROVED
    with pytest.raises(ProjectError, match="no record of what was approved") as refused:
        experiments.run(project, "E1", compute=backend)
    assert "approve it again" in str(refused.value)
    assert project.page("E1").state == "planned" and run_dirs(project) == []
    # The same for one that was queued when the record was lost.
    experiments.approve(project, "E1", backends={"local": backend})
    project.set_state("E1", "queued", by="system")
    project.record_approval("E1", {"approved_digest": None}, by="human")
    with pytest.raises(ProjectError, match="no record of what was approved"):
        experiments.run(project, "E1", compute=backend)
    assert project.page("E1").state == "planned"


def test_a_change_between_two_attempts_ends_the_repairs(project, backend):
    """The runner's repair is part of the approved run; anyone else's change is not."""
    approved(project, backend, {"main.py": BROKEN})

    def repair(plan, record, log_tail):
        # While the model was asked for a correction, the page was given the network.
        project.update_page("E1", author="human", fields={"network": True})
        return {"main.py": QUICK}

    record = experiments.run(project, "E1", compute=backend, repair=repair)

    assert (record["number"], record["status"]) == (1, "failed")
    assert experiments.code_files(project, "E1") == {"main.py": BROKEN}  # not repaired
    assert any("network: false when approved, now true" in note for note in record["notes"])
    assert run_dirs(project) == ["1"] and project.page("E1").state == "failed"


def test_a_failure_is_not_handed_to_the_repairer_once_the_experiment_changed(project, backend):
    """A repair is a model session, which costs money: it is not asked for on behalf of
    a run that cannot go on because the experiment is no longer what was approved."""
    approved(project, backend, {"main.py": BROKEN})
    edited: list[object] = []
    asked: list[int] = []

    def on_log(line):  # while the job runs, the person gives the page another time limit
        if not edited:
            edited.append(project.update_page("E1", author="human", fields={"timeout_seconds": 99}))

    def repair(plan, record, log_tail):
        asked.append(record["number"])
        return {"main.py": QUICK}

    record = experiments.run(project, "E1", compute=backend, repair=repair, on_log=on_log)

    assert edited and asked == []
    assert (record["number"], record["status"]) == (1, "failed")
    assert any("was not repaired" in note and "timeout_seconds" in note for note in record["notes"])


def test_code_written_between_the_check_of_the_approval_and_the_copy_is_not_run(project, backend):
    """The experiment is held against its approval when a run starts and once more on the
    copy of the code that is handed to the job: a write that lands in between is not run."""
    approved(project, backend, {"main.py": QUICK})
    marker = project.root / "unread-code-ran.txt"

    def compute(plan):
        # Asked for the backend once the approval has been checked: a late write lands.
        unread = f"open({str(marker)!r}, 'w').write('ran')\n" + QUICK
        (experiments.code_dir(project, "E1") / "main.py").write_text(unread)
        return backend

    record = experiments.run(project, "E1", compute=compute)

    assert (record["status"], record["error_kind"]) == ("failed", "approval")
    assert "main.py: changed after the approval" in record["error"]
    assert "approve it again" in record["error"]
    assert not marker.exists() and not (run_dir(project) / "metrics.json").exists()
    assert project.page("E1").state == "failed"


def test_the_runners_repair_is_recorded_and_the_next_approval_starts_afresh(project, backend):
    done = approved(project, backend, {"main.py": BROKEN})
    stopped: list[bool] = []

    def repair(plan, record, log_tail):
        stopped.append(True)  # the worker is told to stop while the repair is made
        return {"main.py": QUICK}

    with pytest.raises(KeyboardInterrupt):
        experiments.run(project, "E1", compute=_Interrupting(backend, after=2), repair=repair)
    page = project.page("E1")
    assert page.state == "queued" and stopped
    assert page.meta["approved_digest"] == done.meta["approved_digest"]
    repaired = page.meta["repaired_digest"]
    assert repaired == experiments.plan_digest(project, "E1") != page.meta["approved_digest"]
    status = experiments.approval(project, "E1")
    assert status["matches"] is True and status["repaired"] is True
    assert status["changed"] == [
        f"main.py: changed after the approval (written by agent:runner at "
        f"{experiments.code_authors(project, 'E1')['main.py']['written']})"
    ]
    # Resumed, the repaired code is what runs: it is the approved run going on.
    record = experiments.run(project, "E1", compute=backend)
    assert record["status"] == "succeeded" and project.page("E1").state == "succeeded"
    # One approval is one run; the next one pins the code as it is then.
    again = experiments.approve(project, "E1", backends={"local": backend})
    assert "repaired_digest" not in again.meta
    assert again.meta["approved_digest"] == repaired
    assert experiments.approval(project, "E1")["repaired"] is False


class _Interrupting:
    """A backend whose ``after``-th job is interrupted before it starts (Ctrl-C)."""

    def __init__(self, backend, *, after: int) -> None:
        self._backend, self._after, self._jobs = backend, after, 0
        self.name = backend.name

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def run_job(self, spec, **kwargs):
        self._jobs += 1
        if self._jobs >= self._after:
            raise KeyboardInterrupt
        return self._backend.run_job(spec, **kwargs)


def test_without_a_choice_a_run_goes_where_the_project_says_not_where_the_page_says(
    tmp_path, backend
):
    project = Project.create(tmp_path / "p", title="On Modal", settings={"compute": "modal"})
    modal = FakeModal()
    both = {"local": backend, "modal": modal}
    # A page from before agents were kept from writing ``backend``, or one edited by hand.
    planned(project, {"main.py": QUICK}, backend="local")

    done = experiments.approve(project, "E1", backends=both)

    assert done.meta["backend"] == "modal"
    [choice] = [c for c in experiments.options(project, "E1", both) if c["backend"] == "local"]
    assert choice["can_run"] is True  # the person may still choose it, by saying so
    # What the person chose is remembered by the approval, for the same experiment.
    planned(project, {"main.py": QUICK})
    first = experiments.approve(project, "E2", backend="local", backends=both)
    assert first.meta["backend"] == "local"
    assert experiments.run(project, "E2", compute=backend)["status"] == "succeeded"
    assert experiments.approve(project, "E2", backends=both).meta["backend"] == "local"
    # A value that only looks like it (the page changed after the approval) is not.
    experiments.request_cancel(project, "E1")
    project.update_page("E1", author="human", fields={"backend": "local"})
    assert experiments.approve(project, "E1", backends=both).meta["backend"] == "modal"


def test_network_access_is_granted_by_the_person_never_by_the_plan(project, backend):
    backends = {"local": backend}
    planned(project, {"main.py": QUICK}, network=True)
    with pytest.raises(ProjectError, match="yours to grant") as refused:
        experiments.approve(project, "E1", backends=backends)
    said = str(refused.value)
    assert "E1" in said and "network: true" in said
    assert "--network" in said and "--no-network" in said and "in the app" in said
    assert project.page("E1").state == "planned"
    assert "approved_digest" not in project.page("E1").meta

    without = experiments.approve(project, "E1", network=False, backends=backends)
    assert (without.state, without.meta["network"]) == ("approved", False)
    experiments.request_cancel(project, "E1")
    granted = experiments.approve(project, "E1", network=True, backends=backends)
    assert granted.meta["network"] is True
    # Granted for one run is not granted for the next: the page says true again.
    experiments.request_cancel(project, "E1")
    with pytest.raises(ProjectError, match="yours to grant"):
        experiments.approve(project, "E1", backends=backends)
    # A plan that does not ask runs without the network when nothing is said.
    planned(project, {"main.py": QUICK})
    assert experiments.approve(project, "E2", backends=backends).meta["network"] is False
    [choice] = experiments.options(project, "E1", backends)
    assert any("yours to grant" in note for note in choice["notes"])
    assert not any("Approving the run approves that" in note for note in choice["notes"])


def test_an_agents_page_runs_in_the_projects_repository_only(tmp_path):
    mine = "https://ghp_FAKETOKEN123@git.example.invalid/me/private.git"
    project = Project.create(tmp_path / "p", title="Repo", settings={"repo": {"path": mine}})
    foreign = planned(
        project,
        mode="repo",
        command="python train.py",
        network=True,
        repo="https://git.example.invalid/someone-else/train-with-extras.git",
    )
    with pytest.raises(ProjectError, match="someone-else") as refused:
        experiments.load_plan(project, foreign.id)
    said = str(refused.value)
    assert "not the repository this project was given" in said
    assert "ghp_FAKETOKEN123" not in said and "https://***@git.example.invalid/me" in said
    # Naming the project's own repository, as an agent sees it, is the project's repository.
    own = planned(
        project,
        mode="repo",
        command="python train.py",
        network=True,
        repo="https://git.example.invalid/me/private.git",
    )
    assert experiments.load_plan(project, own.id).repo == mine
    # The person's own page may name any repository: it is theirs to choose.
    theirs = project.create_page(
        "experiment",
        "In another repository",
        BODY,
        author="human",
        fields={
            "mode": "repo",
            "command": "python x.py",
            "network": True,
            "repo": "https://git.example.invalid/other/repo.git",
        },
    )
    assert experiments.load_plan(project, theirs.id).repo == (
        "https://git.example.invalid/other/repo.git"
    )

    # The same when the project's repository is a directory of this machine.
    directory = tmp_path / "repository"
    directory.mkdir()
    local = Project.create(
        tmp_path / "local", title="Repo", settings={"repo": {"path": str(directory)}}
    )
    page = planned(
        local,
        mode="repo",
        command="python train.py",
        network=True,
        repo="https://git.example.invalid/someone-else/train-with-extras.git",
    )
    with pytest.raises(ProjectError, match="not the repository this project was given"):
        experiments.load_plan(local, page.id)


def test_code_written_while_the_approval_is_under_way_is_not_approved(
    tmp_path, backend, monkeypatch
):
    """The designer may write until the approval; a write that lands in the middle of
    one must not become "approved" unread."""
    settings = {"review": {"experiment": "auto"}}
    project = Project.create(tmp_path / "auto", title="Auto", settings=settings)
    planned(project, {"main.py": QUICK}, accept=False)
    read = experiments.code_digest(project, "E1")
    unread = QUICK + "open('i-was-not-read.txt', 'w')\n"
    checked = experiments._refusals

    def then_the_designer_writes(*args, **kwargs):
        reasons = checked(*args, **kwargs)
        experiments.write_code(project, "E1", {"main.py": unread}, author="agent:designer")
        return reasons

    monkeypatch.setattr(experiments, "_refusals", then_the_designer_writes)
    for options in ({"code_digest": read}, {}):  # the app sends the digest, the CLI none
        with pytest.raises(Conflict, match="while it was being approved"):
            experiments.approve(project, "E1", backends={"local": backend}, **options)
        assert project.page("E1").state == "planned"
        experiments.write_code(project, "E1", {"main.py": QUICK}, author="agent:designer")


def test_writing_code_waits_for_whoever_holds_the_project(project):
    """``write_code`` checks who may write and writes under the project's lock, so an
    approval's last look and its change of state cannot be separated by a write."""
    planned(project, {"main.py": QUICK}, accept=False)
    finished = threading.Event()
    outcome: list[object] = []

    def write() -> None:
        try:
            outcome.append(
                experiments.write_code(project, "E1", {"main.py": OTHER}, author="agent:designer")
            )
        except ProjectError as exc:
            outcome.append(exc)
        finished.set()

    with project.locked():
        writer = threading.Thread(target=write)
        writer.start()
        assert not finished.wait(0.3)
        # What the holder decides meanwhile is what the writer finds.
        project.review("E1", "accept")
    writer.join(10)
    assert finished.is_set() and isinstance(outcome[0], NotAllowed)
    assert experiments.code_files(project, "E1") == {"main.py": QUICK}


def test_an_agent_does_not_replace_a_file_the_person_wrote(project):
    """The person corrected the code of a proposal; the designer's session goes on."""
    planned(project, {"main.py": BROKEN, "lib.py": "X = 1\n"}, accept=False)
    experiments.write_code(project, "E1", {"main.py": QUICK}, author="human")
    (experiments.code_dir(project, "E1") / "data.py").write_text("ROWS = []\n")  # put there by hand

    for name in ("main.py", "data.py"):
        with pytest.raises(NotAllowed, match=rf"{name}.*the person") as refused:
            experiments.write_code(project, "E1", {name: BROKEN}, author="agent:designer")
        assert "what should change" in str(refused.value)
    # Nothing of a refused call is written, not even the files that were the agent's.
    with pytest.raises(NotAllowed):
        experiments.write_code(
            project, "E1", {"lib.py": "X = 2\n", "main.py": BROKEN}, author="agent:designer"
        )
    files = experiments.code_files(project, "E1")
    assert files["main.py"] == QUICK and files["lib.py"] == "X = 1\n"
    assert experiments.code_authors(project, "E1")["main.py"]["author"] == "human"
    # Its own files, and new ones, are still the designer's to write; so is the person's
    # file when the text is the same (nothing is replaced).
    experiments.write_code(
        project, "E1", {"lib.py": "X = 3\n", "new.py": "Y = 1\n"}, author="agent:designer"
    )
    experiments.write_code(project, "E1", {"main.py": QUICK}, author="agent:designer")
    assert experiments.code_authors(project, "E1")["main.py"]["author"] == "human"
    # The runner's repair of an approved run is pipeline code, bounded and recorded.
    experiments.write_code(project, "E1", {"main.py": QUICK + "#\n"}, author="agent:runner")


# ---------------------------------------------------------------------------------------
# Jobs and their records (audit: execution 4, 5, 6, 8, 9; containment 7, 8, 14)
# ---------------------------------------------------------------------------------------

FAKE_KEY = "sk-ant-AUDIT-FAKE-KEY-000"
FAKE_TOKEN = "ghp_FAKETOKEN123"
PRIVATE_URL = f"https://{FAKE_TOKEN}@git.example.invalid/me/private.git"


class Recording(FakeModal):
    """A remote backend written against the contract of today: it is told which strings
    must never appear in a log, and what the job it picks up was approved to be."""

    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.masks: list[tuple[str, ...]] = []
        self.picked_up: list[dict] = []

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None, mask=()):
        self.masks.append(tuple(mask))
        return super().run_job(
            spec, source_dir=source_dir, run_dir=run_dir, on_log=on_log, should_stop=should_stop
        )

    def attach(
        self, handle, *, run_dir, spec=None, started=None, on_log=None, should_stop=None, mask=()
    ):
        self.picked_up.append({"spec": spec, "started": started, "mask": tuple(mask)})
        return super().attach(handle, run_dir=run_dir, on_log=on_log, should_stop=should_stop)


def test_the_backend_is_told_what_to_keep_out_of_the_log_and_what_job_it_picks_up(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    project = Project.create(
        tmp_path / "p", title="Private", settings={"repo": {"path": PRIVATE_URL}}
    )
    modal = Recording()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4", timeout_seconds=300)
    experiments.approve(project, "E1", backends={"modal": modal})

    assert experiments.run(project, "E1", compute=modal)["status"] == "succeeded"
    [mask] = modal.masks
    assert FAKE_KEY in mask and FAKE_TOKEN in mask

    # A job that outlived its worker is picked up as what was approved and recorded on
    # this side: nothing about it is asked of the place where it ran.
    experiments.approve(project, "E1", backends={"modal": modal})
    directory = dead_run(project, 2, backend="modal", handle="sb-9")
    record = json.loads((directory / "run.json").read_text())
    record["plan"] = {
        "mode": "gpu",
        "command": ["python", "main.py"],
        "timeout_seconds": 300.0,
        "gpu": "T4",
        "network": False,
        "packages": ["torch"],
    }
    (directory / "run.json").write_text(json.dumps(record))
    [settled] = experiments.recover(project, compute=modal)
    assert settled["status"] == "succeeded"
    [picked] = modal.picked_up
    assert picked["started"] == "2026-11-15T12:00:00Z"
    spec = picked["spec"]
    assert (spec.command, spec.timeout, spec.gpu, spec.network, spec.packages) == (
        ["python", "main.py"],
        300.0,
        "T4",
        False,
        ("torch",),
    )
    assert spec.outputs == ("metrics.json", "artifacts")
    assert FAKE_KEY in picked["mask"] and FAKE_TOKEN in picked["mask"]


class Leaky(FakeModal):
    """A job that found credentials (its parent holds them, its repository's URL carries
    one) and prints them; written like a backend from before ``mask`` existed."""

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None):
        self.specs.append(spec)
        directory = Path(run_dir)
        lines = [
            f"from the process that started me: ANTHROPIC_API_KEY={FAKE_KEY}",
            f"origin {PRIVATE_URL} (fetch)",
            f"RuntimeError: could not push to {PRIVATE_URL} with {FAKE_KEY}",
        ]
        (directory / "log.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            on_log(line)
        return JobResult(returncode=1, seconds=0.5, backend=self.name, log_path=None)


def test_what_a_job_prints_of_a_credential_is_masked_before_it_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    project = Project.create(
        tmp_path / "p", title="Private", settings={"repo": {"path": PRIVATE_URL}}
    )
    modal = Leaky()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})
    seen: list[str] = []

    record = experiments.run(project, "E1", compute=modal, on_log=seen.append)

    assert record["status"] == "failed" and "RuntimeError: could not push" in record["error"]
    result = project.page("E1-run1")
    kept = [
        json.dumps(record),
        result.body,
        result.summary,
        "\n".join(seen),
        (project.root / "events.jsonl").read_text(),
        (project.root / "MEMORY.md").read_text(),
    ]
    for text in kept:
        assert FAKE_KEY not in text and FAKE_TOKEN not in text
    assert "from the process that started me: ANTHROPIC_API_KEY=[redacted]" in result.body
    assert "origin https://[redacted]@git.example.invalid/me/private.git (fetch)" in result.body
    assert any("[redacted]" in line for line in seen)
    [logged] = events_of(project, "run.log")
    assert any("[redacted]" in line for line in logged["data"]["lines"])


class Billed(FakeModal):
    """A remote backend that leaves its estimate next to the run, as the contract says,
    and whose job is then interrupted: no result ever reaches the caller."""

    def __init__(self, cost: dict | None) -> None:
        super().__init__()
        self.cost = cost

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None, mask=()):
        self.specs.append(spec)
        (Path(run_dir) / "handle.txt").write_text("sb-77\n")
        if self.cost is not None:
            (Path(run_dir) / "cost.json").write_text(json.dumps(self.cost))
        raise KeyboardInterrupt  # Stop, Ctrl-C, SIGTERM: the sandbox was ended and billed


def test_a_run_that_ends_without_a_result_keeps_the_cost_the_backend_left(tmp_path):
    limits = {"experiment_usd": 0.5}
    project = Project.create(tmp_path / "p", title="Costly", settings={"limits": limits})
    modal = Billed({"cost_usd": 0.42, "seconds": 12.0, "final": True})
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})

    with pytest.raises(KeyboardInterrupt):
        experiments.run(project, "E1", compute=modal)

    [record] = experiments.runs(project, "E1")
    assert (record["status"], record["requeued"]) == ("cancelled", True)
    assert record["cost_usd"] == 0.42 and record["handle"] == "sb-77"
    assert project.page("E1").state == "queued"
    # It counts before the run is started again by itself: $0.42 of $0.50 is spent, and
    # 900 s of a T4 could cost another $0.24.
    again = experiments.run(project, "E1", compute=modal)
    assert (again["number"], again["error_kind"]) == (2, "budget")
    assert "$0.42" in again["error"] and len(modal.specs) == 1


def test_a_sandbox_that_is_gone_is_charged_for_the_time_since_it_started(project):
    """The worker died, the job ran on unattended and is gone: what it cost is unknown,
    so it is charged for as long as it can have run."""
    from paper_orchestra.compute.modal_backend import estimate_cost

    modal = Recording(attach_error="The sandbox sb-9 no longer exists.")
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4", timeout_seconds=600)
    experiments.approve(project, "E1", backends={"modal": modal})
    directory = dead_run(project, backend="modal", handle="sb-9")  # started long ago
    # The estimate from when the worker last looked: not the end of the story.
    (directory / "cost.json").write_text('{"cost_usd": 0.001, "seconds": 3.0, "final": false}')

    [record] = experiments.recover(project, compute=modal)

    assert (record["status"], record["error_kind"], record["requeued"]) == (
        "failed",
        "worker",
        True,
    )
    longest = 600 + experiments.REMOTE_MARGIN_SECONDS
    assert record["cost_usd"] == pytest.approx(
        estimate_cost(longest, gpu="T4", cpu=2.0, memory_mb=4096)
    )
    assert any("charged" in note and "sb-9" in note for note in record["notes"])
    # A run that never had a sandbox costs nothing, and a local one is never billed.
    planned(project, {"main.py": QUICK})
    experiments.approve(project, "E2", backends={"local": LocalBackend()})
    dead_run(project, experiment="E2")
    [other] = experiments.recover(project)
    assert other["experiment"] == "E2" and other["cost_usd"] is None


class LocalJobs:
    """What the engine sees of the local backend's jobs: a record of what a job started
    (``job.json``), and a way to end it for whoever finds it after its worker died."""

    name = "local"

    def __init__(self, scratch: Path) -> None:
        #: No such session exists (it is above any process id), and no such directory.
        self.job = {
            "session": 4_194_999,
            "session_started": "2026-11-15T12:00:00Z",
            "scratch": str(scratch / "paper-orchestra-job-gone"),
        }
        self.during: dict = {}
        self.ended: list[dict] = []

    def supports(self, spec):
        return []

    def check(self, **kwargs):
        return []

    def close(self):
        pass

    def describe(self):
        return {"backend": "local", "jobs": {"supported": True, "reattach": False}}

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None, mask=()):
        directory = Path(run_dir)
        (directory / "job.json").write_text(json.dumps(self.job))
        for _ in range(200):  # until the engine has seen the record of the job
            should_stop()
            self.during = json.loads((directory / "run.json").read_text())
            if self.during.get("job"):
                break
            time.sleep(0.02)
        (directory / "log.txt").write_text("done\n")
        (directory / "metrics.json").write_text('{"loss": 0.25}')
        return JobResult(returncode=0, seconds=0.2, backend=self.name)

    def end_job(self, record):
        self.ended.append(dict(record))
        return ["The processes the job left were ended and its working directory removed."]


def test_what_a_dead_workers_job_left_is_ended_and_removed_when_the_run_is_settled(
    tmp_path, project
):
    local = LocalJobs(tmp_path)
    planned(project, {"main.py": QUICK})
    experiments.approve(project, "E1", backends={"local": local})
    # While a job runs, what it started is copied into the run's record.
    assert experiments.run(project, "E1", compute=local)["status"] == "succeeded"
    assert local.during["job"] == local.job and local.ended == []

    # kill -9: the record still reads running, a copy of the repository is left behind.
    experiments.approve(project, "E1", backends={"local": local})
    directory = dead_run(project, 2)
    (directory / "job.json").write_text(json.dumps(local.job))
    (directory / "repo" / "data").mkdir(parents=True)
    (directory / "repo" / "data" / "big.bin").write_bytes(b"\0" * 1024)

    [record] = experiments.recover(project, compute=local)

    assert (record["status"], record["error_kind"]) == ("failed", "worker")
    assert local.ended == [local.job] and record["job"] == local.job
    assert not (directory / "repo").exists()
    assert any("were ended" in note for note in record["notes"])
    # The same when the person cancels what the dead worker left.
    directory = dead_run(project, 3)
    (directory / "job.json").write_text(json.dumps(local.job))
    (directory / "repo").mkdir()
    experiments.request_cancel(project, "E1")
    assert not (directory / "repo").exists()
    assert experiments.runs(project, "E1")[-1]["status"] == "cancelled"


def test_recovery_ends_the_job_as_the_backends_file_last_described_it(tmp_path, project):
    """The run's record holds a copy of the backend's job.json, made when the file first
    appeared.  A backend may write the file in steps (the working directory when it is
    made, the process session once there is one): the file is what counts."""
    local = LocalJobs(tmp_path)
    planned(project, {"main.py": QUICK})
    experiments.approve(project, "E1", backends={"local": local})
    directory = dead_run(project)
    record = json.loads((directory / "run.json").read_text())
    early = {"scratch": local.job["scratch"]}
    (directory / "run.json").write_text(json.dumps({**record, "job": early}))
    (directory / "job.json").write_text(json.dumps(local.job))

    [settled] = experiments.recover(project, compute=local)

    assert local.ended == [local.job] and settled["job"] == local.job


class ManyArtifacts(FakeModal):
    """A job that writes 150 artifacts with long names."""

    NAMES: tuple[str, ...] = tuple(
        "artifacts/" + "/".join(["ü" * 100] * 4) + f"/table-{index}.csv" for index in range(150)
    )

    def _finish(self, directory: Path, handle: str) -> JobResult:
        for name in self.NAMES:
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text("a,b\n1,2\n")
        return super()._finish(directory, handle)


def test_a_run_with_more_than_fits_on_a_page_still_gets_its_result_record(project):
    modal = ManyArtifacts()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})

    record = experiments.run(project, "E1", compute=modal)

    assert record["status"] == "succeeded" and len(record["artifacts"]) == 150
    assert record["result"] == "wiki/results/E1-run1.md"
    result = project.page("E1-run1")
    assert (result.state, result.review) == ("succeeded", "accepted")
    assert len((project.root / result.path).read_bytes()) < 200_000
    listed = result.body.count("/files/experiments/E1/runs/1/artifacts/")
    assert 0 < listed < 150
    assert f"and {150 - listed} more in run.json" in result.body
    assert "| loss | 0.25 |" in result.body  # what a finding rests on is all there


class LongEverything(ManyArtifacts):
    """A job that makes every part of its result page as long as that part may be: many
    artifacts, as many metrics as a run may have with long names and long texts, a log
    of wide characters."""

    def _finish(self, directory: Path, handle: str) -> JobResult:
        result = super()._finish(directory, handle)
        wordy = {f"{'ü' * 90}_{index}": "é" * 200 for index in range(199)}
        (directory / "metrics.json").write_text(json.dumps({"loss": 0.25, **wordy}))
        (directory / "log.txt").write_text("\n".join(["𝔁" * 600] * 60) + "\n")
        return result


def test_a_run_whose_abbreviated_page_is_still_too_long_gets_a_brief_one(project):
    """Each list on a result page is cut to what it may take; together they can still be
    more than a page holds.  Then the facts and the metrics are written alone."""
    modal = LongEverything()
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})

    record = experiments.run(project, "E1", compute=modal)

    assert record["status"] == "succeeded" and record["result"] == "wiki/results/E1-run1.md"
    result = project.page("E1-run1")
    assert len((project.root / result.path).read_bytes()) < 200_000
    assert "| loss | 0.25 |" in result.body and "| Command | python main.py |" in result.body
    assert "are not shown: together they did not fit on a page" in result.body
    assert "experiments/E1/runs/1/ (run.json, log.txt)" in result.body
    assert "## Log" not in result.body and events_of(project, "error") == []


def test_a_result_record_that_cannot_be_written_is_said_aloud(project, backend, monkeypatch):
    approved(project, backend, {"main.py": QUICK})

    def refuse(*args, **kwargs):
        raise ProjectError("wiki/results/E1-run1.md would be 371739 bytes; the limit is 200000.")

    monkeypatch.setattr(project, "write_record", refuse)
    record = experiments.run(project, "E1", compute=backend)

    assert record["status"] == "succeeded" and record["result"] is None
    assert any("could not be written" in note for note in record["notes"])
    [event] = events_of(project, "error")
    assert "E1" in event["summary"] and "result record" in event["summary"]
    assert "run.json" in event["summary"] and event["data"]["run"] == 1


def test_a_result_record_names_the_command_and_the_network_setting(tmp_path, project, backend):
    approved(project, backend, {"main.py": QUICK})
    experiments.run(project, "E1", compute=backend)
    body = project.page("E1-run1").body
    assert "| Command | python main.py |" in body
    assert "| Network access | not allowed |" in body

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "train.py").write_text(QUICK)
    project.settings.update({"repo": {"path": str(repository)}})
    project.save_settings()
    planned(project, mode="repo", command="python train.py --lr 0.1", network=True)
    experiments.approve(project, "E2", network=True, backends={"local": backend})
    record = experiments.run(project, "E2", compute=backend)
    assert record["status"] == "succeeded", record["error"]
    body = project.page("E2-run1").body
    assert "| Command | python train.py --lr 0.1 |" in body
    assert "| Network access | allowed |" in body


def test_what_came_with_the_repository_is_not_a_result_of_the_run(tmp_path, project, backend):
    """The person's repository holds a metrics.json and artifacts from earlier work; the
    command of the experiment writes neither."""
    repository = tmp_path / "repository"
    (repository / "artifacts").mkdir(parents=True)
    (repository / "train.py").write_text("print('training... (this script writes no metrics)')\n")
    (repository / "metrics.json").write_text('{"accuracy": 0.9912, "note": "last year"}')
    (repository / "artifacts" / "old_curve.png").write_bytes(PNG)
    project.settings.update({"repo": {"path": str(repository)}})
    project.save_settings()
    planned(project, mode="repo", command="python train.py")
    experiments.approve(project, "E1", backends={"local": backend})

    record = experiments.run(project, "E1", compute=backend)

    assert (record["status"], record["error_kind"]) == ("failed", "metrics")
    assert "ended without writing metrics.json" in record["error"]
    assert record["metrics"] == {} and record["artifacts"] == []
    assert any("metrics.json" in note and "before the run" in note for note in record["notes"])
    assert "0.9912" not in project.page("E1-run1").body
    # The repository itself is as it was.
    assert (repository / "metrics.json").exists() and (repository / "artifacts").is_dir()


class Collecting(FakeModal):
    """A backend that takes a job's outputs from the directory it was handed without
    clearing them first, and whose job writes none: whatever came with the source would
    be the result."""

    name = "local"

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None, mask=()):
        directory = Path(run_dir)
        (directory / "log.txt").write_text("training... (this script writes no metrics)\n")
        for name in ("metrics.json", "artifacts"):
            found = Path(source_dir) / name
            if found.is_dir():
                shutil.copytree(found, directory / name)
            elif found.is_file():
                shutil.copyfile(found, directory / name)
        return JobResult(returncode=0, seconds=0.1, backend=self.name)


def test_the_copy_of_the_repository_is_handed_over_without_old_results(tmp_path, project):
    """Not left to the backend alone: the copy that is made here for the job holds no
    metrics.json and no artifacts/ when the job gets it, and the record says so."""
    repository = tmp_path / "repository"
    (repository / "artifacts").mkdir(parents=True)
    (repository / "train.py").write_text("print('training')\n")
    (repository / "metrics.json").write_text('{"accuracy": 0.9912}')
    (repository / "artifacts" / "old_curve.png").write_bytes(PNG)
    project.settings.update({"repo": {"path": str(repository)}})
    project.save_settings()
    local = Collecting()
    planned(project, mode="repo", command="python train.py")
    experiments.approve(project, "E1", backends={"local": local})

    record = experiments.run(project, "E1", compute=local)

    assert (record["status"], record["error_kind"]) == ("failed", "metrics")
    assert record["metrics"] == {} and record["artifacts"] == []
    [note] = [note for note in record["notes"] if "before the run" in note]
    assert "metrics.json, artifacts/" in note and "copy of the repository" in note


@pytest.mark.parametrize("target", ["metrics.json", "artifacts/table.csv"])
def test_a_patch_cannot_put_results_into_the_repository(tmp_path, project, backend, target):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "train.py").write_text("print('nothing to do')\n")
    project.settings.update({"repo": {"path": str(repository)}})
    project.save_settings()
    patch = f'--- /dev/null\n+++ b/{target}\n@@ -0,0 +1 @@\n+{{"accuracy": 0.9731}}\n'
    planned(project, {"patch.diff": patch}, mode="repo", command="python train.py")

    errors, _ = experiments.split_problems(experiments.check_code(project, "E1"))
    assert any(target in error and "patch.diff" in error for error in errors)
    with pytest.raises(ProjectError, match=re.escape(target)):
        experiments.approve(project, "E1", backends={"local": backend})
    assert project.page("E1").state == "planned"


class Cloning(FakeModal):
    """A backend whose job is the clone step of a repo-mode plan followed by the plan's
    command: both write to the one log, the clone step first and on lines of its own."""

    def __init__(self, name: str, lines: list[str]) -> None:
        super().__init__()
        self.name = name
        self.lines = lines

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None, mask=()):
        self.specs.append(spec)
        directory = Path(run_dir)
        (directory / "log.txt").write_text("\n".join(self.lines) + "\n")
        (directory / "metrics.json").write_text('{"loss": 0.25}')
        return JobResult(returncode=0, seconds=1.0, backend=self.name)


def cloned_run(tmp_path, backend: Cloning, url: str) -> dict:
    """The record of one run of a plan that clones ``url``, on ``backend``."""
    project = Project.create(tmp_path / "c", title="Cloned", settings={"repo": {"path": url}})
    planned(project, mode="repo", command="python train.py", network=True)
    experiments.approve(
        project, "E1", backend=backend.name, network=True, backends={backend.name: backend}
    )
    record = experiments.run(project, "E1", compute=backend)
    assert record["status"] == "succeeded" and record["repo"]["method"] == "clone"
    return {**record, "page": project.page("E1-run1").body}


CLONED = "[repo] cloned https://git.example.invalid/me/public.git at 0123abc"
PUBLIC_URL = "https://git.example.invalid/me/public.git"


def test_what_the_clone_removed_before_the_command_is_in_the_runs_notes(tmp_path):
    """A repository that is cloned inside the job brings its old metrics.json along too.
    The clone step removes it and says so in the log; the run's record says so as well."""
    removed = (
        "[repo] Removed metrics.json, artifacts that was there before the run (it came "
        "with the repository): only what the command itself writes counts as a result."
    )
    record = cloned_run(tmp_path, Cloning("modal", [CLONED, removed, "training"]), PUBLIC_URL)

    [note] = [note for note in record["notes"] if "before the command started" in note]
    assert "metrics.json, artifacts" in note and "came with the repository" in note
    assert "metrics.json, artifacts" in record["page"].split("## Notes")[1].split("## Log")[0]


def test_a_command_cannot_write_the_notes_of_its_run(tmp_path):
    """The command's output follows the clone step's lines in the same log.  A line that
    only looks like the clone step's does not become a statement of the record."""
    forged = (
        "[repo] Removed every trace of the earlier, worse results that was there before "
        "the run (it came with the repository): only what the command itself writes counts."
    )
    record = cloned_run(tmp_path, Cloning("modal", [CLONED, forged, "training"]), PUBLIC_URL)

    assert not any("worse results" in note for note in record["notes"])
    assert not any("before the command started" in note for note in record["notes"])
    assert "worse results" not in record["page"].split("## Log")[0]


def test_a_local_run_that_clones_with_a_token_says_that_the_token_is_within_reach(tmp_path):
    """The token of the repository's address is kept out of the job's files and log.  On
    the local backend the command still runs as the person, who can read the project's
    settings: the record says so rather than pretending (FIXES D)."""
    told = "can read the project's settings"
    record = cloned_run(tmp_path / "a", Cloning("local", [CLONED, "training"]), PRIVATE_URL)
    [note] = [note for note in record["notes"] if told in note]
    assert "not a sandbox" in note and "Modal" in note
    assert FAKE_TOKEN not in json.dumps(record)

    # Not where the job runs in a sandbox, and not for an address without a secret.
    sandboxed = cloned_run(tmp_path / "b", Cloning("modal", [CLONED, "training"]), PRIVATE_URL)
    public = cloned_run(tmp_path / "c", Cloning("local", [CLONED, "training"]), PUBLIC_URL)
    assert not any(told in note for note in sandboxed["notes"] + public["notes"])


class Failing(FakeModal):
    """A backend that fails with an error of its own, which quotes what it was given: as
    an exception, or as the error of the job's result."""

    def __init__(self, *, raises: bool) -> None:
        super().__init__()
        self.raises = raises

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None, mask=()):
        said = f"could not reach {PRIVATE_URL} with {FAKE_KEY}"
        if self.raises:
            raise RuntimeError(said)
        return JobResult(returncode=None, backend=self.name, error=said)


@pytest.mark.parametrize("raises", [True, False])
def test_an_error_that_ends_a_run_is_kept_without_the_credentials_it_quotes(
    tmp_path, monkeypatch, raises
):
    """Not only what a job prints: the error that ends a run is written into the record,
    the page's state and the feed, and may quote the repository's address or a key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    project = Project.create(
        tmp_path / "p", title="Private", settings={"repo": {"path": PRIVATE_URL}}
    )
    modal = Failing(raises=raises)
    planned(project, {"main.py": QUICK}, mode="gpu", gpu="T4")
    experiments.approve(project, "E1", backends={"modal": modal})

    with pytest.raises(RuntimeError) if raises else contextlib.nullcontext():
        experiments.run(project, "E1", compute=modal)

    page = project.page("E1")
    assert page.state == "failed" and "could not reach" in page.meta["state_reason"]
    kept = [
        json.dumps(experiments.runs(project, "E1")),
        json.dumps(page.meta),
        project.page("E1-run1").body,
        (project.root / "events.jsonl").read_text(),
    ]
    for text in kept:
        assert FAKE_KEY not in text and FAKE_TOKEN not in text


# ---------------------------------------------------------------------------------------
# What a finding's numbers may rest on (audit: containment 11 and the measured 19 s)
# ---------------------------------------------------------------------------------------

BOASTFUL = """\
import json

print("expected accuracy 93.85")
with open("metrics.json", "w") as handle:
    handle.write('{"acc": 0.5125, "note": "expected 66.66", "p_at_77.77": 1.5}')
"""

FAILING = """\
import json

print("accuracy 88.12")
with open("metrics.json", "w") as handle:
    json.dump({"acc": 0.6644}, handle)
raise SystemExit(3)
"""


def test_only_the_metrics_of_a_run_that_succeeded_support_a_finding(project, backend):
    """Not the plan's title, not what the code printed, not a metric's name or note, and
    nothing of a run that failed."""
    page = project.create_page(
        "experiment",
        "Warmup reaches 97.31 percent accuracy",
        BODY,
        author="agent:designer",
        fields={"mode": "script"},
    )
    experiments.write_code(project, page.id, {"main.py": BOASTFUL}, author="agent:designer")
    project.review(page.path, "accept")
    experiments.approve(project, "E1", backends={"local": backend})
    assert experiments.run(project, "E1", compute=backend)["status"] == "succeeded"
    approved(project, backend, {"main.py": FAILING})
    assert experiments.run(project, "E2", compute=backend)["status"] == "failed"
    finding = project.create_page(
        "finding",
        "Warmup works",
        "Accuracy was 51.25% (0.5125), close to the 97.31% of the plan and the expected "
        "93.85; the second run saw 88.12 and 0.6644, a note said 66.66, at 77.77.\n",
        author="agent:analyst",
    )

    checks = {
        check["number"]: check["found_in"] for check in experiments.number_check(project, finding)
    }

    measured = ["wiki/results/E1-run1.md"]
    assert checks == {
        "51.25": measured,
        "0.5125": measured,
        "97.31": [],
        "93.85": [],
        "88.12": [],
        "0.6644": [],
        "66.66": [],
        "77.77": [],
    }
    assert experiments.unverified_numbers(project, finding) == [
        "97.31", "93.85", "88.12", "0.6644", "66.66", "77.77"
    ]  # fmt: skip


def test_the_number_check_stays_fast_on_a_large_project(project):
    """One finding of the largest size against 40 source records took 19 s: it is run
    on every look at the page, and in every analysis and write-up."""
    import random

    rng = random.Random(1)
    for index in range(40):
        body = " ".join(f"{rng.random() * 100:.4f}" for _ in range(3000))
        project.write_record("source", f"Src{index}", f"Source {index}", body + " 12.3456\n")
    numbers = " ".join(f"{rng.random() * 100:.3f}" for _ in range(24000))
    finding = project.create_page(
        "finding", "Many numbers", f"{numbers} and 12.346 but 123456.789\n", author="agent:analyst"
    )

    started = time.monotonic()
    checks = experiments.number_check(project, finding)
    elapsed = time.monotonic() - started

    assert len(checks) > 20_000
    assert elapsed < 6.0, f"the number check took {elapsed:.1f} s"
    by_number = {check["number"]: check["found_in"] for check in checks}
    assert sorted(by_number["12.346"]) == sorted(
        f"wiki/sources/Src{index}.md" for index in range(40)
    )
    assert by_number["123456.789"] == []
    # Asked again (the page is opened again, the write-up checks every finding), the
    # evidence that did not change is not read a second time.
    started = time.monotonic()
    assert experiments.number_check(project, finding) == checks
    assert time.monotonic() - started < elapsed + 1.0


WORKER = """\
import sys

from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.research import experiments
from paper_orchestra.research.project import Project

project = Project.open(sys.argv[1])
experiments.run(project, "E1", compute=LocalBackend(scratch_dir=sys.argv[2]))
"""


def _alive(pid: int) -> bool:
    """Whether the process exists and is not a zombie waiting to be reaped."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="processes are found via /proc")
def test_a_killed_workers_local_run_leaves_nothing_behind_once_it_is_settled(tmp_path, project):
    """kill -9 of the worker during a repo-mode run on this machine: the job's children
    go on running, its working directory and a copy of the repository stay.  Settling
    the run ends and removes them (the real local backend, a real worker process)."""
    repository = tmp_path / "repository"
    repository.mkdir()
    child_pid = tmp_path / "child.pid"
    (repository / "train.py").write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(child_pid)!r}, 'w').write(str(child.pid))\n"
        "print('training', flush=True)\n"
        "time.sleep(300)\n"
    )
    project.settings.update({"repo": {"path": str(repository)}})
    project.save_settings()
    scratch = tmp_path / "scratch"
    planned(project, mode="repo", command="python train.py")
    experiments.approve(project, "E1", backends={"local": LocalBackend(scratch_dir=scratch)})
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    worker = subprocess.Popen([sys.executable, str(script), str(project.root), str(scratch)])
    directory = run_dir(project)

    def under_way() -> bool:
        record = directory / "run.json"
        return (
            child_pid.exists()
            and bool(child_pid.read_text().strip())
            and record.exists()
            and bool(json.loads(record.read_text()).get("job"))
        )

    try:
        deadline = time.monotonic() + 60
        while not under_way() and time.monotonic() < deadline:
            assert worker.poll() is None, "the worker ended before its job was under way"
            time.sleep(0.05)
        assert under_way(), "the job did not record what it started"
        child = int(child_pid.read_text())
        job = json.loads((directory / "run.json").read_text())["job"]
        assert Path(job["scratch"]).is_dir() and (directory / "repo").is_dir()
        worker.kill()
        worker.wait(30)
        assert _alive(child), "the job's child outlives the worker that was killed"

        [record] = experiments.recover(project)

        assert (record["status"], record["error_kind"]) == ("failed", "worker")
        assert record["requeued"] is True and record["job"] == job
        deadline = time.monotonic() + 15
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child)
        assert not Path(job["scratch"]).exists() and not (directory / "repo").exists()
        assert any("left" in note for note in record["notes"])
        assert project.page("E1").state == "queued"
    finally:
        # Whatever failed above, nothing this test started stays behind.
        worker.kill()
        worker.wait(30)
        if child_pid.exists() and child_pid.read_text().strip().isdigit():
            with contextlib.suppress(OSError):
                os.kill(int(child_pid.read_text()), signal.SIGKILL)


def test_the_number_check_is_bounded_where_the_evidence_is_dense(tmp_path, monkeypatch):
    """A number of few digits against a file of 200,000 values: thousands of them round
    to it.  They are not all walked for every number, and the answer is the same."""
    import random

    rng = random.Random(7)
    dense = tmp_path / "dense.csv"
    dense.write_text("\n".join(f"{rng.random() * 10:.6f}" for _ in range(200_000)) + "\n")
    sparse = tmp_path / "sparse.csv"
    sparse.write_text("seed,loss\n3,7.12349\n")
    project = Project.create(tmp_path / "p", title="Dense", brief=BRIEF, files=[dense, sparse])
    stated = " ".join(f"{tenths / 10:.1f}" for tenths in range(11, 100) if tenths % 10)
    finding = project.create_page(
        "finding", "Few digits", f"Values {stated} and 123456.7.\n", author="agent:analyst"
    )

    started = time.monotonic()
    checks = {c["number"]: c["found_in"] for c in experiments.number_check(project, finding)}
    elapsed = time.monotonic() - started

    assert checks["7.1"] == ["inputs/dense.csv", "inputs/sparse.csv"]
    assert checks["3.3"] == ["inputs/dense.csv"] and checks["123456.7"] == []
    # Every value in reach walked, one by one: the same answers.
    monkeypatch.setattr(experiments, "MAX_EVIDENCE_WALK", 10**9)
    experiments._INDEX_KEPT.clear()
    walked = {c["number"]: c["found_in"] for c in experiments.number_check(project, finding)}
    assert walked == checks
    assert elapsed < 5.0, f"the number check took {elapsed:.1f} s"
