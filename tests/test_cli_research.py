"""The ``research`` commands and ``ui``: thin wrappers whose job is arguments in, the
right call, a message a person can act on, and the right exit code.

The engine, the experiments module and the web app are built next to the command
line.  The commands are tested against the contracts of those modules (spec 5.3,
10.2 and 7): each test puts a recording stand-in where the command imports from.
"""

from __future__ import annotations

import importlib
import importlib.abc
import json
import os
import signal
import subprocess
import sys
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from paper_orchestra import cli
from paper_orchestra.pipeline import PaperResult
from paper_orchestra.research import Project, ProjectError
from paper_orchestra.utils.common import read_json, write_json

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated():
    """No test leaves a signal handler behind (``research advance`` installs one)."""
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    yield
    for number, handler in before.items():
        if handler is not None:
            signal.signal(number, handler)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory; ``./research`` in it is the default workspace."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def stand_in(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: Any) -> None:
    """Make ``from <name> import <attribute>`` yield the given objects.

    Where the real module exists its attributes are replaced for the test; where it
    does not exist yet, a module holding just these attributes takes its place.
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    for key, value in attributes.items():
        monkeypatch.setattr(module, key, value, raising=False)


def checkpoint_state(**changes: Any) -> dict[str, Any]:
    state = {
        "schema": 1,
        "status": "checkpoint",
        "round": 1,
        "phase": "research",
        "next_phase": "hypothesize",
        "checkpoint": {
            "round": 1,
            "phase": "research",
            "at": "2026-11-15T12:00:00Z",
            "summary": "12 sources, 3 topics and 4 questions are in the wiki.",
            "pending": {"topic": 3, "question": 4},
        },
        "phases": [],
        "worker": None,
        "spent": {"model_usd": 3.21, "experiment_seconds": 120.4, "experiment_usd": 0.0},
        "error": None,
    }
    state.update(changes)
    return state


class FakeEngine:
    """Stands in for ``research.engine.Engine`` (spec 5.3) and records how it was used."""

    instances: ClassVar[list[FakeEngine]] = []
    #: What ``advance`` does: return this state, or raise it when it is an exception.
    outcome: Any = None
    #: What ``run_experiment`` returns per experiment id (an exception is raised).
    runs: ClassVar[dict[str, Any]] = {}

    def __init__(self, project: Project, **options: Any) -> None:
        self.project = project
        self.options = options
        self.calls: list[tuple[Any, ...]] = []
        FakeEngine.instances.append(self)

    def advance(self, *, phase: str | None = None, instruction: str | None = None) -> dict:
        self.calls.append(("advance", phase, instruction))
        self.project.emit(
            "phase.started", "The scout started looking for prior work.", phase="research", round=1
        )
        self.project.emit(
            "agent.activity",
            "Verified 12 of 15 candidate papers.",
            phase="research",
            agent="scout",
            round=1,
        )
        outcome = FakeEngine.outcome or checkpoint_state()
        if isinstance(outcome, BaseException):
            raise outcome
        write_json(self.project.root / "state.json", outcome)
        return outcome

    def run_experiment(self, experiment_ref: str, *, backend: Any = None, limits: Any = None):
        self.calls.append(("run_experiment", experiment_ref))
        self.project.emit("run.started", f"{experiment_ref} started on local.", phase="run")
        outcome = FakeEngine.runs[experiment_ref]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def stop(self) -> None:
        self.calls.append(("stop",))


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> type[FakeEngine]:
    FakeEngine.instances, FakeEngine.outcome, FakeEngine.runs = [], None, {}
    stand_in(monkeypatch, "paper_orchestra.research.engine", Engine=FakeEngine)
    return FakeEngine


@pytest.fixture
def project(workspace: Path) -> Project:
    """A project with something of every kind a command may list or decide."""
    project = Project.create(
        workspace / "research" / "warmup",
        title="Warmup for small batches",
        brief="With adaptive optimisers it is small batches that need warmup.",
    )
    for title in ("Second-moment noise explains the spike", "RAdam removes the need for warmup"):
        project.create_page(
            "hypothesis", title, "## Statement\n\nText.\n", author="agent:theorist", round=1
        )
    project.create_page(
        "question", "Does the effect hold for SGD?", "Open.\n", author="agent:scout", round=1
    )
    project.create_page(
        "experiment",
        "Warmup against none at batch 16",
        "## Objective\n\nCompare.\n",
        author="agent:designer",
        relations={"tests": ["H1"]},
        fields={"mode": "script"},
        round=1,
    )
    return project


def invoke(*args: str) -> Any:
    return runner.invoke(cli.app, ["research", *args])


def flat(text: str) -> str:
    return " ".join(text.split())


# -- new ---------------------------------------------------------------------------


def test_new_creates_the_project_and_says_where_it_is_and_what_comes_next(workspace, engine):
    brief = "Warmup matters most for small batches.\n\nEarly second-moment estimates are noisy."
    result = invoke("new", "--brief", brief, "--no-run")

    assert result.exit_code == 0, result.output
    root = Path("research") / "warmup-matters-most-for-small-batches"
    project = Project.open(root)
    assert project.title == "Warmup matters most for small batches."
    assert project.page("wiki/idea.md").body.strip() == brief
    out = flat(result.output)
    assert f'Created the research project "{project.title}" in {root}' in out
    assert f"paper-orchestra research advance {root}" in out
    assert "paper-orchestra ui" in out
    assert engine.instances == []  # --no-run: nothing was started


def test_new_takes_the_brief_from_a_file_and_attaches_what_is_named(workspace, engine):
    (workspace / "idea.txt").write_text("Small batches need warmup.\n", encoding="utf-8")
    (workspace / "notes.md").write_text("# Notes\n", encoding="utf-8")
    data = workspace / "data"
    data.mkdir()
    (data / "prior_run.csv").write_text("step,loss\n1,2.3\n", encoding="utf-8")

    result = invoke(
        "new", "notes.md", "data", "--brief-file", "idea.txt", "--title", "Warmup", "--dir", "w",
        "--no-run",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    project = Project.open("w")
    assert project.title == "Warmup"
    assert sorted(entry["name"] for entry in project.attachments()) == [
        "data/prior_run.csv",
        "notes.md",
    ]
    assert "2 files attached" in result.output


def test_new_names_the_directory_after_the_title_and_never_reuses_one(workspace, engine):
    first = invoke("new", "--brief", "An idea.", "--title", "Warmup: small batches!", "--no-run")
    second = invoke("new", "--brief", "An idea.", "--title", "Warmup: small batches!", "--no-run")

    assert first.exit_code == second.exit_code == 0, first.output + second.output
    assert Project.open("research/warmup-small-batches").title == "Warmup: small batches!"
    assert (Path("research") / "warmup-small-batches-2" / "project.json").is_file()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "Give the idea with --brief TEXT or --brief-file FILE."),
        (["--brief", "x", "--brief-file", "idea.txt"], "Give the idea with --brief TEXT or"),
        (["--brief", "   "], "The brief is empty"),
        (["--brief-file", "missing.txt"], "missing.txt"),
        (["nothing-here.csv", "--brief", "An idea."], "nothing-here.csv does not exist"),
    ],
)
def test_new_refuses_what_it_cannot_start_from(workspace, engine, args, message):
    (workspace / "idea.txt").write_text("An idea.\n", encoding="utf-8")

    result = invoke("new", *args, "--no-run")

    assert result.exit_code == 2, result.output
    assert message in flat(result.stderr)
    assert not (workspace / "research").exists()


def test_new_refuses_a_directory_that_is_in_use(workspace, engine):
    (workspace / "taken").mkdir()
    (workspace / "taken" / "file.txt").write_text("x")

    result = invoke("new", "--brief", "An idea.", "--dir", "taken", "--no-run")

    assert result.exit_code == 2
    assert "taken is not an empty directory" in flat(result.stderr)


def test_new_advances_to_the_first_checkpoint_unless_told_not_to(workspace, engine):
    result = invoke("new", "--brief", "An idea about warmup.")

    assert result.exit_code == 0, result.output
    (made,) = engine.instances
    assert made.calls == [("advance", None, None)]
    assert made.project.root == Path("research") / "an-idea-about-warmup"
    out = flat(result.stdout)
    assert "Checkpoint after research (round 1): 12 sources, 3 topics and 4 questions" in out
    # The project exists before anything is started, and stays when the engine fails.
    engine.outcome = checkpoint_state(status="failed", error="the scout could not reach a model")
    failed = invoke("new", "--brief", "Another idea.")
    assert failed.exit_code == 1
    assert Project.open("research/another-idea").title == "Another idea."


#: A model that answers every session offline: on the first turn it writes one topic page
#: through the memory tool, the way an agent does, and then says that it is done.
SCRIPTED_MODEL = '''
from paper_orchestra.llm.fake import FakeLLM

PAGE = """---
title: {title}
summary: What the scripted model had to say.
---
A short page written by the scripted model for {label}.
"""


def respond(request):
    if request.memory is not None and request.turn == 0:
        role = request.label.split(".")[-1]
        request.memory.create(
            f"/memories/wiki/topics/notes-of-the-{role}.md",
            PAGE.format(title=f"Notes of the {role}", label=request.label),
        )
    return "Done."


def client():
    return FakeLLM(respond)
'''


def test_new_takes_a_project_to_its_first_checkpoint_with_the_real_engine(workspace, monkeypatch):
    """The whole path of a worker, offline: the command line, the engine, agent sessions
    through the harness, the project store.  Only the model is scripted."""
    pytest.importorskip("paper_orchestra.research.engine")
    from paper_orchestra.llm.router import SCRIPTED_CLIENT_ENV, reset_clients

    (workspace / "scripted_model_for_cli.py").write_text(SCRIPTED_MODEL)
    monkeypatch.syspath_prepend(str(workspace))
    monkeypatch.setenv(SCRIPTED_CLIENT_ENV, "scripted_model_for_cli:client")
    reset_clients()
    try:
        started = invoke("new", "--brief", "Small batches need warmup under Adam.", "--dir", "w")
        listed = invoke("review", "w")
        decided = invoke("review", "w", "--accept-all", "topic")
        status = invoke("status", "w", "--json")
    finally:
        reset_clients()

    assert started.exit_code == 0, started.output
    assert "Checkpoint after research (round 1)" in started.stdout
    assert "Waiting for you: 2 topics" in started.stdout
    assert "paper-orchestra research review w" in flat(started.stdout)
    events = [line for line in started.stderr.splitlines() if line.strip()]
    assert any("intake" in line for line in events) and any("scout" in line for line in events)
    assert "2 waiting for your decision" in listed.stdout
    assert decided.exit_code == 0 and decided.stdout.count("Accepted wiki/topics/") == 2
    data = json.loads(status.stdout)
    assert data["state"]["status"] == "checkpoint" and data["state"]["phase"] == "research"
    assert data["counts"]["pending"]["total"] == 0
    assert data["counts"]["by_kind"]["topic"] == 2


# -- finding the project -------------------------------------------------------------


def test_a_project_is_named_by_its_directory_its_name_in_the_workspace_or_its_id(project):
    by_path = invoke("status", str(project.root), "--json")
    by_name = invoke("status", "warmup", "--json")
    by_id = invoke("status", project.id, "--json")

    for result in (by_path, by_name, by_id):
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["project"]["id"] == project.id


def test_an_unknown_project_is_a_usage_error_that_says_where_it_looked(workspace):
    result = invoke("status", "nowhere")

    assert result.exit_code == 2
    message = flat(result.stderr)
    assert "There is no research project at nowhere" in message
    assert "paper-orchestra research new" in message


# -- status --------------------------------------------------------------------------


def test_status_summarises_round_phase_what_is_waiting_experiments_and_cost(project):
    project.review("H1", "accept")
    project.review("E1", "accept")
    project.set_state("E1", "approved", by="human")
    project.add_request("Also look at schedule-free optimisers.")
    write_json(
        project.root / "state.json",
        checkpoint_state(
            round=2,
            phase="design",
            next_phase="run",
            checkpoint={
                "round": 2,
                "phase": "design",
                "at": "2026-11-15T12:00:00Z",
                "summary": "1 experiment is ready to approve.",
                "pending": {"hypothesis": 1},
            },
        ),
    )

    result = invoke("status", str(project.root))

    assert result.exit_code == 0, result.output
    out = flat(result.stdout)
    assert "Warmup for small batches" in out and project.id in out
    assert "Round 2, checkpoint after design; next: run" in out
    assert "1 experiment is ready to approve." in out
    assert "Waiting for you: 1 question, 1 hypothesis" in out
    assert "Experiments: 1 approved" in out
    assert "1 open request" in out
    assert "$3.21 of $25.00 on models" in out
    # The time limit is a cap on one run (spec 10.9), not a budget the runs use up.
    assert "$0.00 of $5.00 on experiments, which ran for 120 s in all" in out
    assert "(one run may take up to 900 s)" in out
    assert "120 s of 900 s" not in out
    assert f"paper-orchestra research review {project.root}" in out


def test_status_before_the_first_phase_says_how_to_start(project):
    result = invoke("status", str(project.root))

    assert result.exit_code == 0
    out = flat(result.stdout)
    assert "Not started" in out
    assert f"paper-orchestra research advance {project.root}" in out


def test_status_of_a_project_the_engine_only_opened_reads_as_not_started(project):
    """Running an experiment by hand leaves a state file in which no phase has run."""
    idle = checkpoint_state(status="idle", round=0, phase=None, next_phase="intake")
    write_json(project.root / "state.json", {**idle, "checkpoint": None})

    out = flat(invoke("status", str(project.root)).stdout)

    assert "Not started: no phase has run yet." in out and "idle" not in out
    assert f"paper-orchestra research advance {project.root}` starts the research" in out


def test_status_of_a_failed_project_shows_the_error_and_the_way_on(project):
    write_json(
        project.root / "state.json",
        checkpoint_state(status="failed", checkpoint=None, error="hypothesize: no usable model"),
    )

    out = flat(invoke("status", str(project.root)).stdout)

    assert "failed in research" in out
    assert "hypothesize: no usable model" in out
    assert f"paper-orchestra research advance {project.root}" in out


def test_status_tells_a_live_worker_from_one_that_died(project):
    """A worker that was killed leaves the state at running.  Whether it is still there
    is the engine's reading: a fresh heartbeat, or else a living process on this host."""
    pytest.importorskip("paper_orchestra.research.engine")
    import socket

    from paper_orchestra.research.model import format_stamp

    def worker(pid: int, beat: datetime) -> dict[str, Any]:
        return {
            "pid": pid,
            "host": socket.gethostname(),
            "started": format_stamp(beat),
            "heartbeat": format_stamp(beat),
        }

    long_ago = datetime.now(UTC) - timedelta(hours=1)
    alive = checkpoint_state(
        status="running", checkpoint=None, worker=worker(os.getpid(), long_ago)
    )
    write_json(project.root / "state.json", alive)
    running = flat(invoke("status", str(project.root)).stdout)
    assert f"Round 1, research is running (pid {os.getpid()})" in running
    assert json.loads(invoke("status", str(project.root), "--json").stdout)["worker_alive"] is True

    dead = checkpoint_state(status="running", checkpoint=None, worker=worker(2**22 - 3, long_ago))
    write_json(project.root / "state.json", dead)
    stopped = flat(invoke("status", str(project.root)).stdout)
    assert "Round 1, stopped during research (its worker is gone)" in stopped
    assert f"paper-orchestra research advance {project.root}` continues the phase" in stopped
    as_json = json.loads(invoke("status", str(project.root), "--json").stdout)
    assert as_json["worker_alive"] is False and as_json["state"]["status"] == "running"


def test_status_as_json_is_the_whole_picture_on_stdout(project):
    write_json(project.root / "state.json", checkpoint_state())

    result = invoke("status", str(project.root), "--json")

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["project"] == {
        "id": project.id,
        "title": "Warmup for small batches",
        "root": str(project.root),
        "created": project.created,
    }
    assert data["state"]["status"] == "checkpoint" and data["worker_alive"] is None
    assert data["counts"]["pending"]["total"] == 4
    assert data["limits"]["model_usd"] == 25.0
    assert sorted(item["ref"] for item in data["pending"]) == ["E1", "H1", "H2", "Q1"]


def test_status_without_a_project_lists_the_workspace(project, workspace):
    Project.create(workspace / "research" / "other", title="Another project", brief="x")

    listing = invoke("status")
    as_json = invoke("status", "--json")

    assert listing.exit_code == 0, listing.output
    lines = [line for line in listing.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "research/other" in lines[0] and "Another project" in lines[0]
    assert "research/warmup" in lines[1] and "4 waiting" in lines[1]
    assert [entry["project"]["title"] for entry in json.loads(as_json.stdout)] == [
        "Another project",
        "Warmup for small batches",
    ]


def test_status_lists_the_workspace_around_a_project_that_cannot_be_opened(project, workspace):
    broken = workspace / "research" / "broken"
    broken.mkdir()
    (broken / "project.json").write_text("{not json", encoding="utf-8")

    listing = invoke("status")
    as_json = invoke("status", "--json")

    assert listing.exit_code == as_json.exit_code == 1
    assert "research/warmup" in listing.stdout and "4 waiting" in listing.stdout
    assert "research/broken cannot be opened" in flat(listing.stderr)
    assert "research/broken" not in listing.stdout
    assert [entry["project"]["id"] for entry in json.loads(as_json.stdout)] == [project.id]
    assert "research/broken cannot be opened" in flat(as_json.stderr)


def test_status_inside_a_project_directory_is_about_that_project(project, monkeypatch):
    monkeypatch.chdir(project.root)

    result = invoke("status")

    assert result.exit_code == 0
    assert "Warmup for small batches" in result.stdout


def test_status_with_no_projects_says_how_to_make_one(workspace):
    result = invoke("status")

    assert result.exit_code == 0
    assert "paper-orchestra research new" in flat(result.stdout)


# -- advance -------------------------------------------------------------------------


def test_advance_stops_at_a_checkpoint_with_exit_0_and_one_line_per_event(project, engine):
    result = invoke("advance", str(project.root))

    assert result.exit_code == 0, result.output
    assert engine.instances[0].calls == [("advance", None, None)]
    events = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(events) == 2  # what happened during the command, nothing from before it
    assert events[0].endswith("research: The scout started looking for prior work.")
    assert events[1].endswith("research (scout): Verified 12 of 15 candidate papers.")
    out = flat(result.stdout)
    assert "Checkpoint after research (round 1)" in out
    assert "Waiting for you: 1 question, 2 hypotheses, 1 experiment" in out
    assert f"paper-orchestra research review {project.root}" in out


def test_advance_hands_the_phase_and_the_instruction_to_the_engine(project, engine):
    result = invoke(
        "advance", str(project.root), "--phase", "hypothesize", "--instruction", "Fewer, bolder."
    )

    assert result.exit_code == 0, result.output
    assert engine.instances[0].calls == [("advance", "hypothesize", "Fewer, bolder.")]


def test_advance_refuses_a_phase_that_does_not_exist(project, engine):
    result = invoke("advance", str(project.root), "--phase", "publish")

    assert result.exit_code == 2
    assert "hypothesize" in result.stderr  # the choices are listed
    assert engine.instances == []


def test_advance_exits_with_1_and_the_error_when_the_engine_failed(project, engine):
    engine.outcome = checkpoint_state(
        status="failed", checkpoint=None, error="research failed twice: no usable model"
    )

    result = invoke("advance", str(project.root))

    assert result.exit_code == 1
    message = flat(result.stderr)
    assert "Error: The research phase failed: research failed twice: no usable model" in message
    assert f"paper-orchestra research advance {project.root}" in message


@pytest.mark.parametrize(
    "outcome", [KeyboardInterrupt(), checkpoint_state(status="stopped", checkpoint=None)]
)
def test_advance_exits_with_130_when_it_was_stopped(project, engine, outcome):
    engine.outcome = outcome

    result = invoke("advance", str(project.root))

    assert result.exit_code == 130
    message = flat(result.stderr)
    assert "Stopped." in message
    assert f"paper-orchestra research advance {project.root}" in message


def test_advance_reports_what_the_engine_refuses(project, engine):
    engine.outcome = ProjectError("Another worker (pid 4242) is advancing this project.")

    result = invoke("advance", str(project.root))

    assert result.exit_code == 1
    assert "Error: Another worker (pid 4242) is advancing this project." in flat(result.stderr)


#: The real command line with an engine whose phase never ends: what the web server starts.
WORKER_DRIVER = """
import sys, time, types
from pathlib import Path

from paper_orchestra import cli
from paper_orchestra.utils.common import read_json, write_json

started = Path(sys.argv[1])


class Engine:
    def __init__(self, project, **options):
        self.project = project

    def advance(self, *, phase=None, instruction=None):
        state = self.project.root / "state.json"
        write_json(state, {"schema": 1, "status": "running", "round": 1, "phase": "research"})
        started.write_text("")
        try:
            time.sleep(120)
        except KeyboardInterrupt:
            write_json(state, {**read_json(state), "status": "stopped"})
            raise
        raise AssertionError("the engine was not interrupted")


engine = types.ModuleType("paper_orchestra.research.engine")
engine.Engine = Engine
sys.modules["paper_orchestra.research.engine"] = engine
cli.load_env = lambda: None
sys.argv = ["paper-orchestra", *sys.argv[2:]]
cli.main()
"""


def _ignore_interrupts() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX signals")
@pytest.mark.parametrize(
    ("stop", "inherit_ignored_interrupt"),
    [(signal.SIGTERM, False), (signal.SIGINT, False), (signal.SIGINT, True)],
)
def test_a_worker_takes_sigterm_and_the_interrupt_as_a_stop(
    project, tmp_path, stop, inherit_ignored_interrupt
):
    """The web server stops a worker with an interrupt.  A server that was itself started
    in the background ignores SIGINT, and its children inherit that."""
    driver, started = tmp_path / "driver.py", tmp_path / "started"
    driver.write_text(WORKER_DRIVER)
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(started), "research", "advance", str(project.root)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=_ignore_interrupts if inherit_ignored_interrupt else None,
    )
    try:
        deadline = time.monotonic() + 60
        while not started.exists():
            assert proc.poll() is None, proc.stderr.read()
            assert time.monotonic() < deadline
            time.sleep(0.05)
        proc.send_signal(stop)
        _, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert proc.returncode == 130, stderr
    assert read_json(project.root / "state.json")["status"] == "stopped"
    assert "Stopped." in stderr


# -- review --------------------------------------------------------------------------


def test_review_without_a_decision_lists_what_is_waiting(project):
    proposal = project.propose_edit(
        "wiki/idea.md", author="agent:analyst", summary="Sharpen the claim", body="New text.\n"
    )

    result = invoke("review", str(project.root))

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "5 waiting for your decision" in out
    for ref, text in (
        ("H1", "Second-moment noise explains the spike"),
        ("H2", "RAdam removes the need for warmup"),
        ("Q1", "Does the effect hold for SGD?"),
        ("E1", "Warmup against none at batch 16"),
        (proposal.id, "Sharpen the claim"),
    ):
        assert any(ref in line and text in line for line in out.splitlines()), ref
    assert "--accept" in out and "--reject" in out
    assert project.counts()["pending"]["total"] == 5  # listing decides nothing


def test_what_agents_and_runs_wrote_is_printed_without_control_characters(project, engine):
    """A title or a line of a run's output is data.  An escape sequence in it must not
    reach the terminal, where it could rewrite the list the person decides from."""
    project.create_page(
        "hypothesis",
        "Harmless\x1b[2K\rSPOOFED \x07title\x9b2J",
        "## Statement\n\nText.\n",
        author="agent:theorist",
        round=1,
    )

    listed = invoke("review", str(project.root))
    decided = invoke("review", str(project.root), "--accept", "H3")
    engine.outcome = checkpoint_state(
        status="failed", checkpoint=None, error="the model said \x1b[31mno\x1b[0m"
    )
    failed = invoke("advance", str(project.root))

    shown = listed.stdout + decided.stdout + failed.stdout + failed.stderr
    assert "Harmless[2K SPOOFED title2J" in listed.stdout
    assert 'Accepted H3: hypothesis "Harmless[2K SPOOFED title2J"' in decided.stdout
    assert "the model said [31mno[0m" in flat(failed.stderr)
    for character in ("\x1b", "\x07", "\r", "\x9b"):
        assert character not in shown, repr(character)
    # The page itself is not changed by being listed.
    assert "\x1b[2K" in project.page("H3").title


def test_events_are_printed_without_control_characters(project, engine, monkeypatch):
    class Chatty(FakeEngine):
        """An engine whose run prints what a job's output may hold."""

        def advance(self, **options: Any) -> dict:
            self.project.emit(
                "run.log", "E1 run 1: \x1b]0;evil\x07loss 0.4\x1b[0m", phase="run", agent="runner"
            )
            return super().advance(**options)

    stand_in(monkeypatch, "paper_orchestra.research.engine", Engine=Chatty)

    result = invoke("advance", str(project.root))

    assert result.exit_code == 0, result.output
    assert "run (runner): E1 run 1: ]0;evilloss 0.4[0m" in result.stderr
    assert "\x1b" not in result.stderr and "\x07" not in result.stderr


def test_review_with_nothing_waiting_says_so(project):
    for ref in ("H1", "H2", "Q1", "E1"):
        project.review(ref, "accept")

    result = invoke("review", str(project.root))

    assert result.exit_code == 0
    assert "Nothing is waiting for your decision" in result.stdout


def test_review_accepts_and_rejects_by_id_or_path_with_the_reason_on_the_rejection(project):
    question = project.page("Q1").path

    result = invoke(
        "review", str(project.root), "--accept", "h1", "--accept", question,
        "--reject", "H2", "--reason", "Ma and Yarats already tested this.",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert project.page("H1").review == "accepted"
    assert project.page("H1").decision_reason == ""
    assert project.page("Q1").review == "accepted"
    rejected = project.page("H2")
    assert rejected.review == "rejected"
    assert rejected.decision_reason == "Ma and Yarats already tested this."
    out = result.stdout
    assert 'Accepted H1: hypothesis "Second-moment noise explains the spike"' in out
    assert 'Rejected H2: hypothesis "RAdam removes the need for warmup"' in out
    assert "1 still waiting" in out


def test_review_accept_all_takes_one_kind_and_spares_what_was_rejected(project):
    result = invoke(
        "review", str(project.root), "--accept-all", "hypotheses", "--reject", "H2"
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert project.page("H1").review == "accepted"
    assert project.page("H2").review == "rejected"
    assert project.page("Q1").review == "proposed"
    assert project.page("E1").review == "proposed"
    # H2 was turned down and nothing else: it was at no moment an accepted hypothesis.
    assert "Accepted H2" not in result.stdout
    rejected = project.page("H2").path
    decisions = [
        event["data"]["decision"]
        for event in project.events()
        if event["type"] == "review.decided" and rejected in event["refs"]
    ]
    assert decisions == ["reject"]


def test_review_accept_all_says_when_there_was_nothing_of_that_kind(project):
    result = invoke("review", str(project.root), "--accept-all", "finding")

    assert result.exit_code == 0, result.output
    assert "No finding is waiting for a decision." in result.stdout
    assert "4 still waiting for your decision." in result.stdout


def test_review_decides_an_edit_proposal_by_its_id(project):
    proposal = project.propose_edit(
        "wiki/idea.md", author="agent:analyst", summary="Sharpen the claim", body="Sharper.\n"
    )

    result = invoke("review", str(project.root), "--accept", proposal.id)

    assert result.exit_code == 0, result.output
    assert project.page("wiki/idea.md").body == "Sharper.\n"
    assert f"Accepted {proposal.id}: Sharpen the claim" in result.stdout


def test_review_shows_what_an_edit_changes_before_and_when_it_is_accepted(project):
    """The agent's one-line summary is not what is decided: the fields and the text are."""
    project.review("H1", "accept")
    verdict = project.propose_edit(
        "H1",
        author="agent:analyst",
        summary="Add the missing citation",
        fields={"state": "supported", "summary": "Now with a citation"},
    )
    rewrite = project.propose_edit(
        "wiki/idea.md",
        author="agent:analyst",
        summary="Fix spelling",
        body="The agent's own idea, replacing the person's text.\n",
    )

    listing = invoke("review", str(project.root))

    assert listing.exit_code == 0, listing.output
    out = listing.stdout
    assert "state: open -> supported" in out
    assert "summary: (not set) -> Now with a citation" in out
    assert "-With adaptive optimisers it is small batches that need warmup." in out
    assert "+The agent's own idea, replacing the person's text." in out
    assert project.page("H1").state == "open"  # listing decides nothing

    accepted = invoke("review", str(project.root), "--accept", verdict.id, "--accept", rewrite.id)

    assert accepted.exit_code == 0, accepted.output
    out = accepted.stdout
    assert f"Accepted {verdict.id}: Add the missing citation" in out
    assert "state: open -> supported" in out
    assert f"Accepted {rewrite.id}: Fix spelling" in out
    assert "-With adaptive optimisers it is small batches that need warmup." in out
    assert "+The agent's own idea, replacing the person's text." in out
    assert project.page("H1").state == "supported"


def test_accepting_an_edit_to_an_approved_experiment_says_that_it_needs_approval_again(project):
    from paper_orchestra.research import experiments

    experiments.write_code(project, "E1", {"main.py": CODE}, author="agent:designer")
    project.review("E1", "accept")
    assert invoke("approve", str(project.root), "E1", "--timeout", "60").exit_code == 0
    edit = project.propose_edit(
        "E1",
        author="agent:analyst",
        summary="Fix a typo in the method section",
        fields={"network": True, "timeout_seconds": 600},
    )

    result = invoke("review", str(project.root), "--accept", edit.id)

    assert result.exit_code == 0, result.output
    out = flat(result.stdout)
    assert "network: false -> true" in out and "timeout_seconds: 60 -> 600" in out
    assert "E1 is no longer what you approved" in out
    assert "network: false when approved, now true" in out
    assert f"paper-orchestra research approve {project.root} E1" in out


def test_accepting_an_experiment_says_that_its_run_still_needs_approval(project):
    result = invoke("review", str(project.root), "--accept", "E1")

    assert result.exit_code == 0, result.output
    assert project.page("E1").state == "planned"
    out = flat(result.stdout)
    assert "E1 is accepted as a plan; nothing runs until you approve it" in out
    assert f"paper-orchestra research approve {project.root} E1" in out


def test_review_goes_on_after_a_decision_that_cannot_be_made_and_exits_with_1(project):
    result = invoke("review", str(project.root), "--accept", "H9", "--accept", "H1")

    assert result.exit_code == 1
    assert "H9" in result.stderr and "No page matches" in result.stderr
    assert project.page("H1").review == "accepted"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--accept-all", "paper"], "'paper' is not a kind of page"),
        (["--accept", "H1", "--reject", "H1"], "H1 is named for both"),
        # The same page, named by its id and by a link to it.
        (
            ["--accept", "h1", "--reject", "hypotheses/H1-second-moment-noise-explains-the-spike"],
            "h1 is named for both",
        ),
        (["--reason", "why"], "--reason goes with a decision"),
    ],
)
def test_review_usage_errors_decide_nothing(project, args, message):
    result = invoke("review", str(project.root), *args)

    assert result.exit_code == 2, result.output
    assert message in flat(result.stderr)
    assert project.counts()["pending"]["total"] == 4


# -- approve -------------------------------------------------------------------------


class Approvals:
    """Stands in for ``research.experiments.approve`` (addendum 10.2)."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.raises = raises

    def __call__(self, project: Project, ref: str, **options: Any) -> Any:
        self.calls.append((ref, options))
        if self.raises is not None:
            raise self.raises
        project.review(ref, "accept")
        return project.set_state(ref, "approved", by="human")


def test_approve_is_the_persons_decision_with_the_options_given(project, monkeypatch):
    approvals = Approvals()
    stand_in(monkeypatch, "paper_orchestra.research.experiments", approve=approvals)

    result = invoke(
        "approve", str(project.root), "E1", "--backend", "modal", "--gpu", "T4",
        "--timeout", "1800", "--network",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert approvals.calls == [
        ("E1", {"backend": "modal", "gpu": "T4", "timeout_seconds": 1800.0, "network": True})
    ]
    out = flat(result.stdout)
    assert "E1 is approved to run" in out
    assert f"paper-orchestra research run {project.root} E1" in out


def test_approve_leaves_unset_options_to_the_plan(project, monkeypatch):
    approvals = Approvals()
    stand_in(monkeypatch, "paper_orchestra.research.experiments", approve=approvals)

    assert invoke("approve", str(project.root), "E1").exit_code == 0
    assert approvals.calls == [
        ("E1", {"backend": None, "gpu": None, "timeout_seconds": None, "network": None})
    ]


CODE = "import json\n\njson.dump({'loss': 0.5}, open('metrics.json', 'w'))\n"


def test_a_plan_that_asks_for_the_network_is_approved_only_with_an_explicit_answer(project):
    """An agent writes the plan.  Whether its run may reach the network is the person's
    decision (spec 6.2), so the plan's own word is never what grants it.  The rule is
    the approval's own (``experiments.approve``), the same for the app: the command
    line only hands on what the person said."""
    from paper_orchestra.research import experiments

    project.update_page("E1", author="agent:designer", fields={"network": True})
    experiments.write_code(project, "E1", {"main.py": CODE}, author="agent:designer")
    project.review("E1", "accept")

    silent = invoke("approve", str(project.root), "E1")

    assert silent.exit_code == 1
    message = flat(silent.stderr)
    assert "The plan of E1 asks for network access" in message
    assert "--network" in message and "--no-network" in message
    assert project.page("E1").state == "planned"

    without = invoke("approve", str(project.root), "E1", "--no-network")
    assert without.exit_code == 0, without.output
    assert project.page("E1").meta["network"] is False
    assert "without network access" in flat(without.stdout)
    experiments.request_cancel(project, "E1")
    granted = invoke("approve", str(project.root), "E1", "--network")
    assert granted.exit_code == 0, granted.output
    assert project.page("E1").meta["network"] is True
    assert "and it may reach the network" in flat(granted.stdout)


def test_approve_says_what_was_approved(project, monkeypatch):
    def approve(project: Project, ref: str, **options: Any) -> Any:
        project.review(ref, "accept")
        fields = {"backend": "modal", "gpu": "T4", "timeout_seconds": 1800, "network": True}
        project.update_page(ref, author="human", fields=fields)
        return project.set_state(ref, "approved", by="human")

    stand_in(monkeypatch, "paper_orchestra.research.experiments", approve=approve)

    result = invoke("approve", str(project.root), "E1", "--network")

    assert result.exit_code == 0, result.output
    assert (
        "E1 is approved to run on modal with a T4 GPU, for up to 1800 s, and it may reach "
        "the network." in flat(result.stdout)
    )
    assert "Its code" not in result.stdout  # this plan has none to name


def test_approve_names_the_code_it_approved(project, monkeypatch):
    from paper_orchestra.research import experiments

    code = "import json\n\njson.dump({'loss': 0.5}, open('metrics.json', 'w'))\n"
    experiments.write_code(
        project, "E1", {"main.py": code, "lib/util.py": "X = 1\n"}, author="human"
    )
    project.review("E1", "accept")

    result = invoke("approve", str(project.root), "E1", "--backend", "local")

    assert result.exit_code == 0, result.output
    out = flat(result.stdout)
    # The digest is the approval's: of this code and of how it was approved to run.
    digest = project.page("E1").meta["approved_digest"]
    assert digest == experiments.plan_digest(project, "E1")
    assert f"Its code, in {project.root / 'experiments' / 'E1' / 'code'}:" in out
    assert f"lib/util.py (1 lines), main.py (3 lines); approved digest {digest[:12]}." in out


def test_approve_shows_what_will_run_and_what_the_code_check_warns_of(project):
    """What is approved is said in full: where, how long, with what reach, the command,
    the code, and what a reading of the code found worth knowing."""
    from paper_orchestra.research import experiments

    code = "import requests\n\nprint(requests.get('https://example.invalid').text)\n"
    experiments.write_code(project, "E1", {"main.py": code}, author="agent:designer")
    project.review("E1", "accept")

    result = invoke("approve", str(project.root), "E1", "--timeout", "60")

    assert result.exit_code == 0, result.output
    out = flat(result.stdout)
    assert "E1 is approved to run on local, for up to 60 s, without network access." in out
    assert "It runs `python main.py`." in out
    assert f"approved digest {project.page('E1').meta['approved_digest'][:12]}" in out
    assert "Warning: main.py, line 1: it imports requests, which uses the network" in out
    assert "Warning: no file of the code mentions metrics.json" in out


def test_approve_names_the_repository_and_the_command_of_a_repo_experiment(project):
    from paper_orchestra.research import experiments

    token = "ghp_FAKETOKEN123"
    project.settings.update(
        {"repo": {"path": f"https://{token}@git.example.invalid/me/private.git", "ref": "main"}}
    )
    project.save_settings()
    project.update_page(
        "E1",
        author="agent:designer",
        fields={"mode": "repo", "command": "sh -c 'python train.py --lr 0.1'", "network": True},
    )
    project.review("E1", "accept")

    result = invoke("approve", str(project.root), "E1", "--network")

    assert result.exit_code == 0, result.output
    out = flat(result.stdout + result.stderr)
    assert token not in out
    assert "E1 is approved to run on local" in out and "it may reach the network" in out
    assert (
        "It runs `sh -c 'python train.py --lr 0.1'` in a clone of "
        "https://***@git.example.invalid/me/private.git at main." in out
    )
    assert f"approved digest {project.page('E1').meta['approved_digest'][:12]}" in out
    assert "Warning: the run starts with" in out and "Read the command" in out
    assert experiments.approval(project, "E1")["matches"] is True


def test_approve_reports_a_refusal_and_exits_with_1(project, monkeypatch):
    refusal = ProjectError("E1 is proposed, not accepted. Accept the experiment's page first.")
    stand_in(monkeypatch, "paper_orchestra.research.experiments", approve=Approvals(refusal))

    result = invoke("approve", str(project.root), "E1")

    assert result.exit_code == 1
    assert "Error: E1 is proposed, not accepted." in flat(result.stderr)


# -- run -----------------------------------------------------------------------------


def run_record(experiment: str, number: int, status: str, **fields: Any) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "number": number,
        "status": status,
        "backend": "local",
        "seconds": 4.21,
        "exit_code": 0 if status == "succeeded" else 1,
        "metrics": {"final_loss": 0.412, "steps": 200} if status == "succeeded" else {},
        "error": None,
        **fields,
    }


@pytest.fixture
def approved(project: Project) -> Project:
    """E1 approved, E2 accepted and only planned."""
    project.review("E1", "accept")
    project.set_state("E1", "approved", by="human")
    project.create_page(
        "experiment", "Warmup length sweep", "## Objective\n\nSweep.\n", author="human", round=1
    )
    return project


def test_run_executes_what_is_approved_and_reports_each_run(approved, engine):
    engine.runs = {"E1": run_record("E1", 1, "succeeded")}

    result = invoke("run", str(approved.root))

    assert result.exit_code == 0, result.output
    assert engine.instances[0].calls == [("run_experiment", "E1")]
    out = flat(result.stdout)
    assert "E1 run 1 succeeded in 4.2 s" in out
    assert "Metrics: final_loss=0.412, steps=200." in out
    assert "wiki/results/E1-run1.md" in out
    assert f"paper-orchestra research advance {approved.root}" in out
    assert "E1 started on local." in result.stderr  # events, as for advance


def test_run_takes_named_experiments_and_exits_with_1_when_one_did_not_succeed(approved, engine):
    engine.runs = {
        "E2": ProjectError("E2 is planned, not approved. Approve it before running."),
        "E1": run_record(
            "E1", 2, "failed", error="exit code 1: NameError: name 'lr' is not defined"
        ),
    }

    result = invoke("run", str(approved.root), "E2", "E1")

    assert result.exit_code == 1
    assert engine.instances[0].calls == [("run_experiment", "E2"), ("run_experiment", "E1")]
    assert "E2 is planned, not approved." in flat(result.stderr)
    out = flat(result.stdout)
    assert "E1 run 2 failed after 4.2 s: exit code 1: NameError: name 'lr' is not defined" in out
    assert "experiments/E1/runs/2/log.txt" in out


def test_run_prints_a_few_metrics_and_points_to_the_record_for_the_rest(approved, engine):
    many = {f"metric_{index}": index for index in range(9)}
    engine.runs = {"E1": run_record("E1", 1, "succeeded", metrics=many)}

    out = flat(invoke("run", str(approved.root)).stdout)

    assert "metric_5=5 and 3 more. Result:" in out and "metric_6" not in out


def test_run_says_nothing_about_new_results_when_nothing_ran(approved, engine):
    engine.runs = {"E2": ProjectError("E2 is planned, not approved.")}

    result = invoke("run", str(approved.root), "E2")

    assert result.exit_code == 1
    assert result.stdout.strip() == ""


def test_run_says_once_that_the_project_has_a_worker(approved, engine):
    from paper_orchestra.research.engine import WorkerBusy

    busy = WorkerBusy("Another worker is already running this project (process 4242 on vm).")
    engine.runs = {"E1": busy, "E2": busy}

    result = invoke("run", str(approved.root), "E1", "E2")

    assert result.exit_code == 1
    assert engine.instances[0].calls == [("run_experiment", "E1")]
    assert result.stderr.count("Another worker is already running this project") == 1
    assert result.stdout.strip() == ""


def test_run_with_nothing_approved_says_how_to_approve(project, engine):
    result = invoke("run", str(project.root))

    assert result.exit_code == 0
    assert engine.instances == []
    out = flat(result.stdout)
    assert "No experiment is approved to run" in out
    assert f"paper-orchestra research approve {project.root}" in out


def test_run_exits_with_130_when_it_was_stopped(approved, engine):
    engine.runs = {"E1": KeyboardInterrupt()}

    result = invoke("run", str(approved.root))

    assert result.exit_code == 130
    message = flat(result.stderr)
    assert "Stopped." in message
    # `advance` would start the project's next phase, which need not be the run.
    assert f"`paper-orchestra research run {approved.root}` runs it again" in message
    assert "research advance" not in message


def test_run_takes_up_an_experiment_whose_worker_died_while_it_ran(approved, engine):
    """A killed worker leaves its experiment at running.  The engine settles such a run
    and starts it again; all the command has to do is hand it over."""
    approved.set_state("E1", "queued", by="system")
    approved.set_state("E1", "running", by="system")
    engine.runs = {"E1": run_record("E1", 2, "succeeded")}

    result = invoke("run", str(approved.root))

    assert result.exit_code == 0, result.output
    assert engine.instances[0].calls == [("run_experiment", "E1")]
    assert "E1 run 2 succeeded" in result.stdout


# -- write ---------------------------------------------------------------------------


class Writer:
    """Stands in for ``research.writeup.write_paper``."""

    def __init__(self, *, success: bool = True, raises: BaseException | None = None) -> None:
        self.success, self.raises = success, raises
        self.calls: list[tuple[Path, dict[str, Any]]] = []

    def __call__(self, project: Project, **options: Any) -> PaperResult:
        self.calls.append((project.root, options))
        if self.raises is not None:
            raise self.raises
        usage = {
            "total": {"requests": 7, "input_tokens": 1200, "output_tokens": 340},
            "estimated_cost_usd": 1.5,
        }
        base = project.paper_dir
        if self.success:
            return PaperResult(True, base / "final_paper.pdf", base, {}, usage)
        return PaperResult(False, None, base, {}, usage, error="stage 'outline' failed twice")


def test_write_hands_the_template_and_only_the_options_given(project, monkeypatch):
    writer = Writer()
    monkeypatch.setattr("paper_orchestra.research.writeup.write_paper", writer)

    plain = invoke("write", str(project.root), "--template", "cvpr2025")
    tuned = invoke(
        "write", str(project.root), "--template", "iclr2025", "--model", "test-model",
        "--effort", "low", "--plotting", "--compute", "modal", "--no-lab-memory", "--no-resume",
        "--research-cutoff", "2026-01",
    )  # fmt: skip

    assert plain.exit_code == tuned.exit_code == 0, plain.output + tuned.output
    assert writer.calls[0] == (project.root, {"template": "cvpr2025"})
    assert writer.calls[1] == (
        project.root,
        {
            "template": "iclr2025",
            "model": "test-model",
            "effort": "low",
            "use_plotting": True,
            "compute": "modal",
            "lab_memory": False,
            "resume": False,
            "research_cutoff": "2026-01",
        },
    )
    assert f"Paper written: {project.paper_dir / 'final_paper.pdf'}" in plain.stdout
    assert "7 requests, 1200 input and 340 output tokens, about $1.50" in plain.stdout


def test_write_needs_a_template(project):
    result = invoke("write", str(project.root))

    assert result.exit_code == 2
    assert "--template" in result.stderr


def test_write_reports_a_project_that_has_nothing_to_write_from(project, monkeypatch):
    refusal = ProjectError("No finding is accepted yet. Accept a finding, then write the paper.")
    monkeypatch.setattr("paper_orchestra.research.writeup.write_paper", Writer(raises=refusal))

    result = invoke("write", str(project.root), "--template", "cvpr2025")

    assert result.exit_code == 2
    assert "Error: No finding is accepted yet." in flat(result.stderr)


def test_write_exits_with_1_when_the_paper_was_not_written(project, monkeypatch):
    monkeypatch.setattr("paper_orchestra.research.writeup.write_paper", Writer(success=False))

    result = invoke("write", str(project.root), "--template", "cvpr2025")

    assert result.exit_code == 1
    message = flat(result.stderr)
    assert "the paper was not written: stage 'outline' failed twice" in message
    assert str(project.paper_dir / "state.json") in message


def test_write_exits_with_130_when_it_was_stopped(project, monkeypatch):
    stopped = Writer(raises=KeyboardInterrupt())
    monkeypatch.setattr("paper_orchestra.research.writeup.write_paper", stopped)

    result = invoke("write", str(project.root), "--template", "cvpr2025")

    assert result.exit_code == 130


# -- demo ----------------------------------------------------------------------------


def test_demo_builds_the_demonstration_project_and_says_how_to_look_at_it(workspace, monkeypatch):
    built: list[tuple[Path, dict[str, Any]]] = []

    def build(root: Path, **options: Any) -> Project:
        built.append((Path(root), options))
        return Project.create(root, title="Demonstration: warmup", brief="x")

    monkeypatch.setattr("paper_orchestra.research.demo.build_demo_project", build)

    default = invoke("demo")
    again = invoke("demo")
    large = invoke("demo", "--dir", "big", "--pages", "150")

    for result in (default, again, large):
        assert result.exit_code == 0, result.output
    assert built == [
        (Path("research") / "demo-warmup", {"pages": None}),
        (Path("research") / "demo-warmup-2", {"pages": None}),
        (Path("big"), {"pages": 150}),
    ]
    out = flat(default.stdout)
    assert "research/demo-warmup" in out
    assert "paper-orchestra ui" in out
    assert "paper-orchestra ui --workspace ." in flat(large.stdout)


def test_demo_reports_a_directory_it_cannot_use(workspace, monkeypatch):
    def build(root: Path, **options: Any) -> Project:
        raise ProjectError(f"{root} is not an empty directory.")

    monkeypatch.setattr("paper_orchestra.research.demo.build_demo_project", build)

    result = invoke("demo", "--dir", "taken")

    assert result.exit_code == 2
    assert "taken is not an empty directory." in flat(result.stderr)


# -- ui ------------------------------------------------------------------------------


def test_ui_serves_the_workspace_on_loopback_by_default(workspace, monkeypatch):
    served: list[tuple[Any, ...]] = []

    def serve(workspace: Path, host: str, port: int, open_browser: bool) -> None:
        served.append((workspace, host, port, open_browser))

    stand_in(monkeypatch, "paper_orchestra.web.app", serve=serve)

    default = runner.invoke(cli.app, ["ui"])
    chosen = runner.invoke(
        cli.app, ["ui", "--workspace", "w", "--host", "0.0.0.0", "--port", "9000", "--no-browser"]
    )

    assert default.exit_code == chosen.exit_code == 0, default.output + chosen.output
    assert served == [
        (Path("research"), "127.0.0.1", 8600, True),
        (Path("w"), "0.0.0.0", 9000, False),
    ]


class _MissingPackage(importlib.abc.MetaPathFinder):
    """Makes importing the web app fail the way it does without the ``ui`` extra."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        if fullname in ("paper_orchestra.web", "paper_orchestra.web.app"):
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return None


def test_ui_without_the_extra_fails_with_one_line_that_names_the_install_command(monkeypatch):
    for name in ("paper_orchestra.web", "paper_orchestra.web.app"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_MissingPackage(), *sys.meta_path])

    result = runner.invoke(cli.app, ["ui"])

    assert result.exit_code == 2
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "uv sync --extra ui" in lines[0] and "fastapi" in lines[0]


# -- the command line as a whole -------------------------------------------------------


def test_the_research_commands_are_the_ones_the_specification_lists():
    result = runner.invoke(cli.app, ["research", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    commands = {"new", "status", "advance", "review", "approve", "run", "write", "demo"}
    assert commands <= _first_words(result.output)

    top = runner.invoke(cli.app, ["--help"], env={"COLUMNS": "200"})
    # Everything that was there before is still there.
    assert {"write", "batch", "doctor", "memory", "modal", "research", "ui"} <= _first_words(
        top.output
    )


def _first_words(help_text: str) -> set[str]:
    """The first word of every line of a help screen, whatever frames the lines."""
    words = (line.strip(" \u2502|").split() for line in help_text.splitlines())
    return {line[0] for line in words if line}


def test_the_command_line_starts_without_the_research_modules_or_the_web_app():
    """Every command pays for what the module imports; research and the web app are
    loaded by the commands that need them."""
    code = (
        "import sys; from paper_orchestra import cli; "
        "loaded = [n for n in sys.modules if n.startswith(('paper_orchestra.research', "
        "'paper_orchestra.web', 'fastapi', 'uvicorn'))]; print(loaded)"
    )
    finished = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "[]"
