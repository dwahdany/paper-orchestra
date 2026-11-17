"""The ``sdk`` runtime in the engine, offline: every phase as an agent, the experiment
agent, the gate, stop, budget and the command line.

A scripted team of agents (``research.sdk.fake``) drives real projects through
the real engine.  No model client of the classic runtime is ever asked: the one
handed to the engine raises.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from paper_orchestra import cli
from paper_orchestra.llm.base import LLMConfigError
from paper_orchestra.research import experiments
from paper_orchestra.research.engine import Engine, EngineError, ModelNeeded, next_step, read_state
from paper_orchestra.research.model import AGENT_RUNTIME_ENV
from paper_orchestra.research.project import Project
from paper_orchestra.research.sdk.fake import Denied, FakeAgent, FakeDriver
from paper_orchestra.research.sdk.guard import FEDERATION_ENV, LAUNCHER
from test_research_agents import (
    EXPERIMENT,
    GOOD_CODE,
    HYPOTHESIS,
    LIBRARY,
    FakeBackend,
    accept_all,
    activity,
    created_id,
    make_project,
    make_scholar,
)

METRICS = {"variance_with_warmup": 0.42, "variance_without_warmup": 0.91, "seed": 0}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """A home directory of the test's own with a Claude Code login in it."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    for name in (
        *FEDERATION_ENV,
        AGENT_RUNTIME_ENV,
        "PAPER_ORCHESTRA_JOB_MODEL_ACCESS",
        "CLAUDE_CONFIG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def role_of(agent: FakeAgent) -> str:
    """Which role an agent was started as, as its system prompt says."""
    system = agent.options.system_prompt
    if "## Your role: carrying out one approved experiment" in system:
        return "experimenter"
    return re.search(r"^## Your role: (\w+)$", system, re.M).group(1)


class AgentTeam:
    """The scripted agents of a project.  Each does what a capable agent would: it looks
    at the index it was started with and does not write again what is there."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prompts: dict[str, list[str]] = {}
        self.folders: dict[str, Path] = {}
        #: role -> callable(agent) run in the middle of that role's work.
        self.during: dict[str, Any] = {}
        #: What every model request of an agent costs, as its process reports it.
        self.cost = 0.05
        self.metrics: dict[str, Any] | None = dict(METRICS)
        self.follow_up = False
        self.experimenter_answers: dict[str, Any] = {}

    def __call__(self, agent: FakeAgent) -> str:
        role = role_of(agent)
        self.calls.append(role)
        self.prompts.setdefault(role, []).append(agent.prompt)
        self.folders[role] = agent.cwd
        agent.use(cost_usd=self.cost, input_tokens=1000, output_tokens=100)
        return getattr(self, role)(agent)

    def _midway(self, agent: FakeAgent) -> None:
        hook = self.during.get(role_of(agent))
        if hook is not None:
            hook(agent)

    @staticmethod
    def _has(agent: FakeAgent, slug: str) -> bool:
        return slug in agent.options.system_prompt

    def intake(self, agent: FakeAgent) -> str:
        if not self._has(agent, "what-the-notes-say"):
            agent.memory(
                "create",
                path="/memories/wiki/topics/what-the-notes-say.md",
                file_text="---\ntitle: What the notes say\nsummary: Warmup of 4000 steps was "
                "used\n---\ninputs/notes.md states that warmup of 4000 steps was used.\n",
            )
        self._midway(agent)
        if not self._has(agent, "why-4000-steps"):
            agent.memory(
                "create",
                path="/memories/wiki/questions/why-4000-steps.md",
                file_text="---\ntitle: Why 4000 warmup steps?\nabout: [idea]\n---\nNot said.\n",
            )
        return "Wrote one topic and one question."

    def scout(self, agent: FakeAgent) -> str:
        agent.web_search("learning rate warmup variance")
        keys = []
        for paper in LIBRARY:
            text = agent.board("verify_paper", title=paper["title"])
            keys.append(re.search(r"ecorded as (\w+)", text).group(1))
        self._midway(agent)
        if not self._has(agent, "warmup-in-the-literature"):
            agent.memory(
                "create",
                path="/memories/wiki/topics/warmup-in-the-literature.md",
                file_text="---\ntitle: Warmup in the literature\nsummary: Warmup is described "
                f"as variance reduction\ncites: [{', '.join(keys)}]\n---\nThe abstract of "
                f"[[sources/{keys[1]}]] states that warmup reduces variance.\n",
            )
        return "Recorded the sources and wrote one topic."

    def theorist(self, agent: FakeAgent) -> str:
        question = re.search(r"questions/(Q\d+)-", agent.options.system_prompt).group(1)
        if not self._has(agent, "less-variance-with-warmup"):
            agent.memory(
                "create",
                path="/memories/wiki/hypotheses/less-variance-with-warmup.md",
                file_text=HYPOTHESIS.format(motivated_by=question),
            )
        self._midway(agent)
        return "Proposed one hypothesis."

    def designer(self, agent: FakeAgent) -> str:
        for hypothesis in re.findall(r'^- (H\d+) "', agent.prompt, re.M):
            agent.write("main.py", GOOD_CODE)  # developed in its own folder first
            experiment = created_id(
                agent.memory(
                    "create",
                    path=f"/memories/wiki/experiments/ablation-for-{hypothesis.lower()}.md",
                    file_text=EXPERIMENT.format(tests=hypothesis, mode="script"),
                )
            )
            self._midway(agent)
            agent.board("write_code", experiment=experiment, name="main.py", text=GOOD_CODE)
        return "Designed the experiments."

    def experimenter(self, agent: FakeAgent) -> str:
        answers = self.experimenter_answers
        answers["files"] = sorted(p.name for p in agent.cwd.iterdir())
        answers["temp"] = Path(agent.options.env["CLAUDE_CODE_TMPDIR"])
        answers["temp_was_there"] = answers["temp"].is_dir()
        answers["code"] = agent.read("main.py") if (agent.cwd / "main.py").is_file() else None
        self._midway(agent)
        if self.metrics is not None:
            metrics = json.dumps(self.metrics)
            agent.bash(
                "python main.py",
                does=lambda: (agent.cwd / "metrics.json").write_text(metrics, encoding="utf-8"),
            )
            agent.write("artifacts/curve.csv", "step,variance\n1,0.9\n")
        if self.follow_up:
            answers["follow_up"] = agent.memory(
                "create",
                path="/memories/wiki/experiments/more-seeds.md",
                file_text=EXPERIMENT.format(tests="H1", mode="script").replace(
                    "Warmup against no warmup on a tiny transformer", "The same with five seeds"
                ),
            )
        agent.say("The run measured both variances.")
        return "Ran the experiment as planned."

    def analyst(self, agent: FakeAgent) -> str:
        if "The project checked the numbers" in agent.prompt or agent.turn:
            return "The numbers stay as they are."
        results = re.findall(
            r"^- /memories/(wiki/results/(E\d+)-run\d+)\.md: .*tests (H\d+)", agent.prompt, re.M
        )
        for path, experiment, hypothesis in results:
            record = path.rsplit("/", 1)[1]
            agent.memory(
                "create",
                path=f"/memories/wiki/findings/what-{record.lower()}-shows.md",
                file_text=f"---\ntitle: What {experiment} shows\nsummary: The variance was "
                f"lower with warmup\nbased_on: [{record}]\nsupports: [{hypothesis}]\n---\n"
                "## What was observed\n\nWith warmup the variance was 0.42, without it 0.91."
                "\n\n## What it means\n\nThe prediction held.\n\n## Limits\n\nOne seed.\n",
            )
            agent.board(
                "propose_edit",
                target=hypothesis,
                summary=f"The run of {experiment} supports it",
                fields={"state": "supported"},
            )
        self._midway(agent)
        journal = re.search(r"goes to (/memories/wiki/journal/\S+\.md)", agent.prompt).group(1)
        agent.memory(
            "create",
            path=journal,
            file_text="---\ntitle: What this round did\nsummary: One result read\n---\n"
            "The run was read and a finding proposed.\n",
        )
        return "Wrote the finding and the journal."


def no_classic_model(name: str) -> Any:
    """The classic runtime's way to a model.  Under the sdk runtime nobody takes it."""
    raise LLMConfigError(f"No client for {name}: the classic runtime was asked for a model.")


@pytest.fixture
def team() -> AgentTeam:
    return AgentTeam()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def project(tmp_path) -> Project:
    notes = tmp_path / "notes.md"
    notes.write_text("Warmup of 4000 steps was used in the first runs.\n")
    return make_project(tmp_path, files=[notes], settings={"agents": {"runtime": "sdk"}})


def make_engine(project: Project, team: Any, backend: FakeBackend | None = None) -> Engine:
    driver = team if isinstance(team, FakeDriver) else FakeDriver(team)
    engine = Engine(
        project,
        compute=backend or FakeBackend(),
        client_for=no_classic_model,
        scholar=make_scholar(),
        driver=driver,
    )
    engine.driver = driver  # for the tests
    return engine


def advance(project: Project, team: Any, backend: FakeBackend | None = None, **options) -> dict:
    """One ``advance()`` by a fresh worker, as the command line starts one."""
    return make_engine(project, team, backend).advance(**options)


def approve(project: Project, ref: str, backend: FakeBackend | None = None, **options) -> None:
    experiments.approve(project, ref, backends={"local": backend or FakeBackend()}, **options)


def to_approved(project: Project, team: AgentTeam, backend: FakeBackend) -> None:
    """Research, a hypothesis and its experiment accepted, and the run approved."""
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    approve(project, "E1", backend)


def usage(project: Project) -> dict:
    return json.loads((project.root / "usage.json").read_text(encoding="utf-8"))


def events(project: Project, type: str) -> list[dict]:
    return [event for event in project.events() if event["type"] == type]


# ---------------------------------------------------------------------------
# Every phase as an agent
# ---------------------------------------------------------------------------


def test_a_project_goes_through_every_phase_with_agents_and_the_person_at_the_gate(
    project, team, backend
):
    state = advance(project, team, backend)
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "research",
        "hypothesize",
    )
    assert team.calls == ["intake", "scout"]
    # Everything the agents wrote is a proposal, made through the board.
    proposed = project.pages(review="proposed")
    assert sorted(page.kind for page in proposed) == ["question", "topic", "topic"]
    assert {page.author for page in proposed} == {"agent:intake", "agent:scout"}
    assert len(project.pages(kind="source")) == 2  # records, written by code after the check
    assert project.counts()["pending"]["total"] == 3
    # The scout read what intake wrote right before it, as in the classic runtime.
    assert "what-the-notes-say" in team.prompts["scout"][0]

    accept_all(project)
    state = advance(project, team, backend)
    assert (state["phase"], state["next_phase"]) == ("hypothesize", "design")
    accept_all(project)
    state = advance(project, team, backend)
    assert (state["phase"], state["next_phase"]) == ("design", "run")
    page = project.page("E1")
    assert (page.review, page.state, page.author) == ("proposed", "planned", "agent:designer")
    assert experiments.code_files(project, "E1") == {"main.py": GOOD_CODE}
    # Nothing ran: the designer's own trial is in its folder, and no run is on record.
    assert backend.jobs == [] and experiments.runs(project, "E1") == []
    assert (team.folders["designer"] / "main.py").is_file()

    accept_all(project)
    approve(project, "E1", backend)
    state = advance(project, team, backend)
    # The approved run was carried out by an agent, then analysed.
    assert team.calls[-2:] == ["experimenter", "analyst"]
    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    assert backend.jobs == []  # `python main.py` was not what ran it
    result = project.page("E1-run1")
    assert (result.kind, result.state, result.author, result.review) == (
        "result",
        "succeeded",
        "system",
        "accepted",
    )
    assert "| variance_with_warmup | 0.42 |" in result.body
    assert project.page("E1").state == "succeeded"
    finding = project.page("F1")
    assert (finding.review, finding.author) == ("proposed", "agent:analyst")
    # The number check reads the agent's metrics like any run's.
    assert experiments.number_report(project, finding)["unverified"] == []
    (verdict,) = project.proposals()
    assert verdict.fields == {"state": "supported"} and project.page("H1").state != "supported"
    assert [phase["phase"] for phase in state["phases"]] == [
        "intake",
        "research",
        "hypothesize",
        "design",
        "run",
        "analyze",
    ]
    assert all(phase["status"] == "completed" for phase in state["phases"])

    # What the agents cost is the project's model spend: their own figures, once each.
    rows = {row["label"]: row for row in usage(project)["by_label"]}
    assert set(rows) == {
        "intake.agent",
        "scout.agent",
        "theorist.agent",
        "designer.agent",
        "experimenter.agent",
        "analyst.agent",
    }
    assert all(row["cost_usd"] == pytest.approx(0.05) for row in rows.values())
    assert usage(project)["estimated_cost_usd"] == pytest.approx(0.30)
    assert state["spent"]["model_usd"] == pytest.approx(0.30)
    assert state["spent"]["experiment_seconds"] > 0


def test_the_agents_of_a_phase_are_closed_when_it_ends(project, team, backend):
    engine = make_engine(project, team, backend)
    engine.advance()
    assert len(engine.driver.conversations) == 2
    assert all(conversation.closed for conversation in engine.driver.conversations)


def test_a_classic_project_is_run_as_before_and_never_starts_an_agent(tmp_path):
    from paper_orchestra.llm.fake import FakeLLM
    from test_research_engine import Team

    project = make_project(tmp_path)
    driver = FakeDriver()
    engine = Engine(
        project,
        compute=FakeBackend(),
        client_for=lambda _: FakeLLM(Team()),
        scholar=make_scholar(),
        driver=driver,
    )
    state = engine.advance()
    assert (state["status"], state["phase"]) == ("checkpoint", "research")
    assert driver.opened == []
    assert not (project.root / "agents").exists()


def test_the_runtime_can_be_changed_for_a_project_that_exists(project, team, backend):
    from paper_orchestra.llm.fake import FakeLLM
    from test_research_engine import Team

    advance(project, team, backend)
    accept_all(project)
    project.settings.update({"agents": {"runtime": "classic"}})
    project.save_settings()
    driver = FakeDriver()
    classic = Team()
    engine = Engine(
        Project.open(project.root),
        compute=backend,
        client_for=lambda _: FakeLLM(classic),
        scholar=make_scholar(),
        driver=driver,
    )
    state = engine.advance()
    assert (state["status"], state["phase"]) == ("checkpoint", "hypothesize")
    assert driver.opened == [] and classic.calls == ["hypothesize.theorist"]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_a_project_set_to_sdk_without_the_package_is_told_how_to_install_it(project, team, backend):
    driver = FakeDriver(team)
    driver.missing = (
        "the Claude Agent SDK (the package claude-agent-sdk) is not installed",
        "install it with `uv sync --extra agents` in a checkout, or "
        "`pip install 'paper-orchestra[agents]'`",
    )
    engine = make_engine(project, driver, backend)
    with pytest.raises(ModelNeeded) as refused:
        engine.advance()
    assert str(refused.value) == (
        "The intake phase needs a model, and the project's agents (Claude Agent SDK, "
        "claude-fable-5-2) cannot start because the Claude Agent SDK (the package "
        "claude-agent-sdk) is not installed: install it with `uv sync --extra agents` in a "
        "checkout, or `pip install 'paper-orchestra[agents]'` (Setup in the app and "
        "`paper-orchestra doctor` show what is configured)."
    )
    assert refused.value.need["settings"] == ["agents"]
    # Nothing was started and nothing recorded.
    assert read_state(project)["phases"] == [] and driver.opened == []


def test_the_gate_of_the_sdk_runtime_does_not_ask_for_the_classic_models(project, team, backend):
    # No model of settings.models can be used here; the agents can start all the same.
    state = advance(project, team, backend)
    assert state["status"] == "checkpoint"


def test_the_gate_wants_the_credential_the_project_chose(project, team, backend, home):
    (home / ".claude" / ".credentials.json").unlink()
    with pytest.raises(ModelNeeded, match="no Claude Code login was found on this machine"):
        advance(project, team, backend)
    project.settings.update({"agents": {"credential": "federation"}})
    project.save_settings()
    with pytest.raises(ModelNeeded, match="Workload Identity Federation settings are not set"):
        advance(project, team, backend)


def test_the_forecast_agrees_with_the_gate(project, team, backend, home, monkeypatch):
    # (The forecast has no driver handed to it: it asks the real one, which is taken to
    # be installed here.)
    from paper_orchestra.research.sdk.claude import ClaudeDriver

    monkeypatch.setattr(ClaudeDriver, "unavailable", lambda self: None)
    assert next_step(project, client_for=no_classic_model)["blocked"] is None
    (home / ".claude" / ".credentials.json").unlink()
    blocked = next_step(project, client_for=no_classic_model)["blocked"]
    assert blocked["phase"] == "intake" and blocked["settings"] == ["agents"]
    assert "no Claude Code login was found" in blocked["message"]


def test_under_sdk_the_run_phase_waits_for_the_agents_too(project, team, backend, home):
    to_approved(project, team, backend)
    (home / ".claude" / ".credentials.json").unlink()
    with pytest.raises(ModelNeeded, match="The run phase needs a model"):
        advance(project, team, backend)
    engine = make_engine(project, team, backend)
    with pytest.raises(ModelNeeded, match="Carrying out E1 needs a model"):
        engine.run_experiment("E1")
    assert project.page("E1").state == "approved" and experiments.runs(project, "E1") == []


# ---------------------------------------------------------------------------
# Stop, in the middle of an agent's turn
# ---------------------------------------------------------------------------


def test_a_stop_request_interrupts_the_agent_and_the_phase_is_carried_on(project, team, backend):
    engine = make_engine(project, team, backend)
    team.during["intake"] = lambda agent: engine.stop()

    state = engine.advance()

    assert (state["status"], state["phase"], state["next_phase"]) == ("stopped", "intake", "intake")
    assert [(p["phase"], p["status"]) for p in state["phases"]] == [("intake", "interrupted")]
    # The page it had written is on record; the one after the stop was not written.
    assert state["phases"][0]["produced"] == ["wiki/topics/what-the-notes-say.md"]
    assert engine.driver.interrupts == 1 and engine.driver.conversations[0].closed
    assert any("was stopped; running it again resumes it" in line for line in activity(project))
    # What the interrupted turn cost is in the ledger.
    assert state["spent"]["model_usd"] == pytest.approx(0.05)

    del team.during["intake"]
    state = engine.advance()
    assert (state["status"], state["phase"]) == ("checkpoint", "research")
    assert [(p["phase"], p["status"], p["attempts"]) for p in state["phases"]] == [
        ("intake", "completed", 2),
        ("research", "completed", 1),
    ]
    # The second attempt carried the agent's own session on, in the same folder, and
    # wrote only what was missing.
    first, second = engine.driver.opened[:2]
    assert second.resume == "fake-1" and second.cwd == first.cwd
    assert sorted(page.kind for page in project.pages(review="proposed")) == [
        "question",
        "topic",
        "topic",
    ]
    assert (
        "The intake works as a Claude agent on claude-fable-5-2, in fast mode. It carries on "
        "the session that was interrupted."
    ) in activity(project)
    # Three turns were paid for: intake twice, the scout once.
    assert state["spent"]["model_usd"] == pytest.approx(0.15)


def test_ctrl_c_ends_the_agents_process_and_leaves_the_phase_interrupted(project, team, backend):
    def interrupt(agent: FakeAgent) -> None:
        raise KeyboardInterrupt  # what SIGINT, the web app's stop, raises in the worker

    team.during["scout"] = interrupt
    engine = make_engine(project, team, backend)
    with pytest.raises(KeyboardInterrupt):
        engine.advance()
    state = read_state(project)
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "stopped",
        "research",
        "research",
    )
    assert state["phases"][-1]["status"] == "interrupted"
    assert all(conversation.closed for conversation in engine.driver.conversations)
    # The turn that was killed is counted by its tokens: spend never goes unseen.
    labels = [row["label"] for row in usage(project)["by_label"]]
    assert labels == ["intake.agent", "scout.agent.unreported"]
    # ... and whoever shows the spend can say that this part of it is a lower bound.
    assert usage(project)["unreported_usd"] > 0
    status = runner.invoke(cli.app, ["research", "status", str(project.root)])
    assert "The model spend is at least that" in status.output
    # The sources the scout had recorded by then are on record as produced.
    assert len(state["phases"][-1]["produced"]) == 2

    del team.during["scout"]
    state = advance(project, team, backend)
    assert (state["status"], state["phase"]) == ("checkpoint", "research")
    assert state["phases"][-1]["attempts"] == 2


# ---------------------------------------------------------------------------
# Spend and the model budget
# ---------------------------------------------------------------------------


def test_the_spend_of_two_workers_adds_up_and_the_ledger_is_not_taken_for_lost(
    project, team, backend
):
    first = advance(project, team, backend)
    assert first["spent"]["model_usd"] == pytest.approx(0.10)
    accept_all(project)
    second = advance(project, team, backend)  # a fresh worker, a fresh ledger in memory
    assert second["spent"]["model_usd"] == pytest.approx(0.15)
    report = usage(project)
    assert report["estimated_cost_usd"] == pytest.approx(0.15) and "carried_usd" not in report
    assert {row["label"]: row["cost_usd"] for row in report["by_label"]} == {
        "intake.agent": pytest.approx(0.05),
        "scout.agent": pytest.approx(0.05),
        "theorist.agent": pytest.approx(0.05),
    }
    assert events(project, "budget.warning") == []


def test_the_model_budget_ends_an_agents_turn_in_the_middle_and_stops_at_a_checkpoint(
    project, team, backend
):
    project.settings.update({"limits": {"model_usd": 0.5}})
    project.save_settings()
    reached = []

    def expensive(agent: FakeAgent) -> None:
        for _ in range(40):
            # Each request is a few cents by the price list; the turn passes the limit
            # long before it would end by itself.
            agent.use(cost_usd=0.25, input_tokens=20_000, output_tokens=2_000)
            reached.append(1)

    team.during["scout"] = expensive
    engine = make_engine(project, team, backend)
    # (Not in fast mode: the count during the turn is then the list price alone.)
    engine.driver.started["fast_mode_state"] = "off"
    state = engine.advance()

    assert len(reached) < 40 and engine.driver.interrupts == 1
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "research",
        "research",
    )
    assert state["phases"][-1]["status"] == "interrupted"
    assert (
        "The model budget ran out in the middle of the scout's work" in state["phases"][-1]["error"]
    )
    (warning,) = events(project, "budget.warning")
    assert warning["data"]["limit"] == "model_usd"
    # What the agent's process said it cost is what counts, not the estimate that ended it.
    assert state["spent"]["model_usd"] == pytest.approx(0.05 + 0.05 + 0.25 * len(reached))
    assert state["spent"]["model_usd"] >= 0.5
    # The agent was told what was left when it started.
    scout = engine.driver.opened[1]
    assert scout.max_budget_usd == pytest.approx(0.45)

    # With the budget spent, the next call starts no agent.
    opened = len(engine.driver.opened)
    again = engine.advance()
    assert again["status"] == "checkpoint" and len(engine.driver.opened) == opened


def test_the_budget_sentence_on_the_checkpoint_names_what_the_agent_says_it_cost(
    project, team, backend
):
    """The turn is ended on a count from tokens at the list price.  In fast mode the
    agent's own figure, which arrives when the turn has ended, is several times that:
    the sentence that stays must not name the smaller sum beside the Spent line."""
    project.settings.update({"limits": {"model_usd": 0.004}})
    project.save_settings()

    team.cost = 0.34  # what the process says its one request cost; the list price is cents
    state = make_engine(project, team, backend).advance()
    assert (state["status"], state["phase"]) == ("checkpoint", "intake")
    spent = state["spent"]["model_usd"]
    assert spent == pytest.approx(0.34)
    said = state["phases"][-1]["error"]
    # A limit under one cent is shown with its digits, not as "$0.00".
    assert f"the intake's work: ${spent:.2f} of $0.004 is spent" in said
    assert said in json.dumps(state["checkpoint"])
    (warning,) = events(project, "budget.warning")
    assert warning["summary"] == said and warning["data"]["spent"] == pytest.approx(spent)
    status = runner.invoke(cli.app, ["research", "status", str(project.root)])
    assert f"Spent: ${spent:.2f} of $0.004 on models" in " ".join(status.output.split())


# ---------------------------------------------------------------------------
# The experiment agent
# ---------------------------------------------------------------------------


def test_an_approved_experiment_is_carried_out_by_an_agent_in_a_folder_of_its_own(
    project, team, backend
):
    to_approved(project, team, backend)
    engine = make_engine(project, team, backend)
    record = engine.run_experiment("E1")

    assert (record["status"], record["backend"], record["number"]) == ("succeeded", "local", 1)
    assert record["metrics"] == METRICS and record["error"] is None
    assert record["artifacts"] and record["cost_usd"] is None  # its model spend is the ledger's
    assert record["seconds"] > 0
    run_dir = project.root / "experiments" / "E1" / "runs" / "1"
    work = run_dir / "work"
    options = engine.driver.opened[-1]
    assert Path(options.cwd) == work
    # It started from the approved code, in its own folder; the snapshot stays beside it.
    assert team.experimenter_answers["code"] == GOOD_CODE
    assert team.experimenter_answers["files"] == ["main.py"]
    assert (run_dir / "code" / "main.py").read_text() == GOOD_CODE
    assert json.loads((run_dir / "metrics.json").read_text()) == METRICS
    assert (run_dir / "artifacts" / "curve.csv").is_file()
    # What it was told: the plan, the code, the time, what was granted.
    prompt = team.prompts["experimenter"][0]
    assert "Carry out E1" in prompt and "## Objective" in prompt
    assert "The code the researcher approved is in your folder (`main.py`)" in prompt
    assert "You have 300 seconds" in prompt
    assert "approved this run without network access" in prompt
    assert LAUNCHER not in prompt and not (work / LAUNCHER).exists()
    assert options.max_turns is None and [tool.name for tool in options.board] == [
        "memory",
        "wiki_search",
        "propose_edit",
        "fail_run",
    ]
    # Its temporary files were in a directory of its own, outside the run's folder
    # (nothing of them is taken for code or results), and that is gone with the agent.
    temp = team.experimenter_answers["temp"]
    assert team.experimenter_answers["temp_was_there"] and not temp.exists()
    assert (
        work not in temp.parents
        and str(temp) in options.settings["sandbox"]["filesystem"]["allowWrite"]
    )
    # The result page and the feed are those of any run.
    result = project.page("E1-run1")
    assert result.state == "succeeded" and "after" in result.body and "local backend" in result.body
    assert project.page("E1").state == "succeeded"
    log = (run_dir / "log.txt").read_text()
    assert "[paper-orchestra] An agent carries out E1; it has 300 seconds." in log
    assert "[paper-orchestra] The experimenter ran “python main.py” in its shell." in log
    assert "[paper-orchestra] The run measured both variances." in log
    assert [event["data"]["status"] for event in events(project, "run.finished")] == ["succeeded"]
    assert "The experimenter ran “python main.py” in its shell." in activity(project)
    # The run says of itself who did the job, in its record and where it starts: whoever
    # shows it (the app) must not ask the project's settings, which can change afterwards.
    assert record["carried_out_by"] == "agent"
    assert json.loads((run_dir / "run.json").read_text())["carried_out_by"] == "agent"
    assert [event["data"].get("carried_out_by") for event in events(project, "run.started")] == [
        "agent"
    ]
    assert engine.driver.conversations[-1].closed
    assert "experimenter.agent" in [row["label"] for row in usage(project)["by_label"]]


CHANGED_CODE = GOOD_CODE + "\n# The call that failed was corrected here.\n"


def test_the_record_of_an_agents_run_names_the_code_as_the_agent_left_it(project, team, backend):
    """The agent starts from the approved code and may change it.  The record must not
    name the approved code as what produced the metrics, nor an exit status and a
    command that nobody observed."""
    to_approved(project, team, backend)

    def repairs(agent: FakeAgent) -> None:
        agent.write("main.py", CHANGED_CODE)
        agent.write("helper.py", "SEEDS = 1\n")
        agent.write("notes.txt", "not code\n")
        (agent.cwd / "venv" / "lib").mkdir(parents=True)
        (agent.cwd / "venv" / "lib" / "site.py").write_text("x = 1\n")
        (agent.cwd / ".hidden.py").write_text("x = 1\n")
        (agent.cwd / "link.py").symlink_to(project.root / "project.json")

    team.during["experimenter"] = repairs
    record = make_engine(project, team, backend).run_experiment("E1")
    assert record["status"] == "succeeded"
    run_dir = project.root / "experiments" / "E1" / "runs" / "1"
    # The approved code is kept as it was, and its digest is still the approval's.
    assert (run_dir / "code" / "main.py").read_text() == GOOD_CODE
    assert record["code_digest"] == experiments.code_digest(project, "E1")
    # Beside it lies the code as the agent left it: the approved files and the scripts
    # it added, nothing hidden, no environment, no link, nothing that is not code.
    kept = run_dir / "code-as-run"
    assert sorted(p.relative_to(kept).as_posix() for p in kept.rglob("*") if p.is_file()) == [
        "helper.py",
        "main.py",
    ]
    assert (kept / "main.py").read_text() == CHANGED_CODE
    as_run = record["code_as_run"]
    assert (as_run["changed"], as_run["added"], as_run["removed"]) == (
        ["main.py"],
        ["helper.py"],
        [],
    )
    assert as_run["digest"] != record["code_digest"]
    assert any("changed main.py, added helper.py" in note for note in record["notes"])
    # No exit status was measured, and the record names none.
    assert record["exit_code"] is None
    assert json.loads((run_dir / "run.json").read_text())["code_as_run"] == as_run
    body = project.page("E1-run1").body
    assert "| Exit status | not measured: an agent carried the run out" in body
    assert "| Command of the plan | python main.py (the agent's log says what ran) |" in body
    assert f"sha256 `{as_run['digest'][:12]}` as the agent left it: it changed main.py" in body
    assert f"(approved: sha256 `{record['code_digest'][:12]}`)" in body
    assert "the model of the agent that carried it out counts as model spend" in body

    # The experiment's own code is still the approved one.  The person can make the
    # agent's code theirs; it is then their edit, and the old approval does not cover it.
    assert experiments.code_files(project, "E1")["main.py"] == GOOD_CODE
    adopted = experiments.adopt_run_code(project, "E1")
    assert (adopted["run"], adopted["files"]) == (1, ["helper.py", "main.py"])
    assert experiments.code_files(project, "E1")["main.py"] == CHANGED_CODE
    assert experiments.code_authors(project, "E1")["main.py"]["author"] == "human"
    assert experiments.code_digest(project, "E1") == as_run["digest"]
    with pytest.raises(experiments.ProjectError, match="nothing to adopt"):
        experiments.adopt_run_code(project, "E1", 7)

    # Three digests are now on record for E1: the first approval's, the second
    # approval's, and the code's.  The pages say which is which, and that the person
    # adopted the code between the runs: an analyst that finds digests that differ
    # must not take them for code of unknown origin.
    first_approval = record["approved_digest"]
    assert first_approval == project.page("E1").meta["approved_digest"]
    assert f"| Approval | sha256 `{first_approval[:12]}`, the experiment page's" in body
    assert "it covers the plan and the code together" in body and "Code history" not in body
    del team.during["experimenter"]
    approve(project, "E1", backend)
    second = make_engine(project, team, backend).run_experiment("E1")
    assert second["code_digest"] == as_run["digest"]
    assert second["approved_digest"] not in (first_approval, second["code_digest"])
    body = project.page("E1-run2").body
    assert f"| Approval | sha256 `{second['approved_digest'][:12]}`" in body
    assert (
        "| Code history | before this run the researcher made the code that the agent left "
        f"after run 1 the experiment's code (code sha256 `{record['code_digest'][:12]}` "
        f"before, `{as_run['digest'][:12]}` since) and approved the experiment again"
    ) in body
    # What an approval prints names both digests, and says what each is of.
    said = " ".join(cli._what_was_approved(project, project.page("E1")))
    assert f"approved digest {second['approved_digest'][:12]}" in said
    assert f"the code alone is sha256 {as_run['digest'][:12]}" in said


def test_an_agent_that_leaves_the_code_alone_ran_what_was_approved(project, team, backend):
    to_approved(project, team, backend)
    record = make_engine(project, team, backend).run_experiment("E1")
    as_run = record["code_as_run"]
    assert (as_run["changed"], as_run["added"], as_run["removed"]) == ([], [], [])
    assert as_run["digest"] == record["code_digest"]
    assert not experiments.run_code_changed(record)
    assert "as approved: the agent left it as it was" in project.page("E1-run1").body
    assert not any("changed" in note for note in record["notes"])
    with pytest.raises(experiments.ProjectError, match="nothing to adopt"):
        experiments.adopt_run_code(project, "E1")


def test_a_budget_that_runs_out_during_a_run_names_the_agent_that_carried_it_out(
    project, team, backend
):
    """`research run` starts a run outside the phases: the agent at work is the one that
    carries the experiment out, not the agent of whatever phase ran last."""
    to_approved(project, team, backend)
    spent = read_state(project)["spent"]["model_usd"]
    project.settings.update({"limits": {"model_usd": spent + 0.3}})
    project.save_settings()

    def expensive(agent: FakeAgent) -> None:
        for _ in range(40):
            agent.use(cost_usd=0.25, input_tokens=20_000, output_tokens=2_000)

    team.during["experimenter"] = expensive
    with pytest.raises(EngineError) as stopped:
        make_engine(project, team, backend).run_experiment("E1")
    said = str(stopped.value)
    assert "in the middle of the work of the agent that carries out E1" in said
    assert "E1 is in the queue again and is run when the work goes on" in said
    assert "designer" not in said and "the phase then resumes" not in said
    assert events(project, "budget.warning")[-1]["summary"] == said
    assert project.page("E1").state == "queued"


def dear_experimenter(project: Project, team: AgentTeam, backend: FakeBackend) -> FakeDriver:
    """An approved E1, 30 cents left of the model budget, and an experiment agent whose
    process says it cost 50: dollars far above the list price of its tokens (fast mode,
    helpers), so that the count from tokens does not end it first.  Its process ends the
    session itself at the sum it was told, as a real one does."""
    to_approved(project, team, backend)
    spent = read_state(project)["spent"]["model_usd"]
    project.settings.update({"limits": {"model_usd": spent + 0.3}})
    project.save_settings()
    team.during["experimenter"] = lambda agent: agent.use(
        cost_usd=0.5, input_tokens=10, output_tokens=10
    )
    team.metrics = None
    driver = FakeDriver(team)
    driver.ends_at_budget = True
    return driver


def test_an_agent_ended_by_its_own_budget_cap_leaves_the_run_in_the_queue(project, team, backend):
    """The agent's process ends its session at the sum it was told.  That is the model
    budget, not a failure of the experiment: the run goes back to the queue and keeps
    its approval, and the person is told that the budget is spent."""
    driver = dear_experimenter(project, team, backend)
    with pytest.raises(EngineError, match="The model budget is used up"):
        make_engine(project, driver, backend).run_experiment("E1")
    (run,) = experiments.runs(project, "E1")
    assert (run["status"], run["requeued"]) == ("cancelled", True)
    assert project.page("E1").state == "queued" and project.pages(kind="result") == []
    assert experiments.will_not_run(project, "E1") is None  # its approval stands
    assert len(events(project, "budget.warning")) == 1
    assert team.calls.count("experimenter") == 1


def test_the_run_phase_stops_at_the_budget_when_the_agents_process_ends_at_its_cap(
    project, team, backend
):
    driver = dear_experimenter(project, team, backend)
    state = make_engine(project, driver, backend).advance()
    assert (state["status"], state["phase"], state["next_phase"]) == ("checkpoint", "run", "run")
    assert state["phases"][-1]["status"] == "interrupted"
    assert "The model budget is used up" in state["phases"][-1]["error"]
    (run,) = experiments.runs(project, "E1")
    assert (run["status"], run["requeued"]) == ("cancelled", True)
    assert project.page("E1").state == "queued" and project.pages(kind="result") == []
    assert len(events(project, "budget.warning")) == 1
    assert "analyst" not in team.calls


def test_an_agent_whose_process_fails_once_is_followed_by_another_and_the_run_succeeds(
    project, team, backend
):
    """An overloaded API ends an agent's process with an error.  That says nothing about
    the experiment: a second agent takes the run up in the same folder."""
    to_approved(project, team, backend)
    failures = []

    def overloaded(agent: FakeAgent) -> None:
        if not failures:
            failures.append(1)
            raise RuntimeError("API Error: 529 Overloaded")

    team.during["experimenter"] = overloaded
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["number"]) == ("succeeded", 1)
    assert team.calls.count("experimenter") == 2
    assert team.prompts["experimenter"][1].startswith("Another agent began this run")
    log = (project.root / "experiments" / "E1" / "runs" / "1" / "log.txt").read_text()
    assert "The agent's process failed (RuntimeError: API Error: 529 Overloaded)" in log
    assert "another agent takes the run up once more" in log


def test_an_agent_whose_process_keeps_failing_does_not_cost_the_run_its_approval(
    project, team, backend
):
    """A failure of the agent's process is the backend being unusable, not the experiment
    failing: the run goes back to the queue, nothing is analysed, and the phase ends
    with the reason.  The agent is started twice and no more."""
    to_approved(project, team, backend)

    def overloaded(agent: FakeAgent) -> None:
        raise RuntimeError("API Error: 529 Overloaded")

    team.during["experimenter"] = overloaded
    state = advance(project, team, backend)
    assert team.calls.count("experimenter") == 2 and "analyst" not in team.calls
    assert (state["status"], state["phase"]) == ("failed", "run")
    assert "API Error: 529 Overloaded" in state["phases"][-1]["error"]
    assert "E1 keeps its approval" in state["phases"][-1]["error"]
    (run,) = experiments.runs(project, "E1")
    assert (run["status"], run["error_kind"], run["requeued"]) == ("failed", "backend", True)
    assert project.page("E1").state == "queued" and project.pages(kind="result") == []
    assert experiments.will_not_run(project, "E1") is None

    # `research run` ends the same way: with the reason, as a backend that cannot be used.
    from paper_orchestra.compute.base import BackendUnusable

    with pytest.raises(BackendUnusable, match="could not work"):
        make_engine(project, team, backend).run_experiment("E1")
    assert team.calls.count("experimenter") == 4
    assert project.page("E1").state == "queued" and project.pages(kind="result") == []

    # When the agents work again, the same approval is enough.
    del team.during["experimenter"]
    advance(project, team, backend)
    assert project.page("E1").state == "succeeded"


def test_a_run_whose_results_were_written_before_the_time_ran_out_keeps_them(
    project, team, backend, monkeypatch
):
    """The agent is told to keep a valid metrics.json in place.  One that was waiting
    (for a command, to write its reply) when its time ran out has finished the run."""
    to_approved(project, team, backend)
    from paper_orchestra.research.sdk import session as session_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock["now"])

    def waits(agent: FakeAgent) -> None:
        metrics = json.dumps(METRICS)
        agent.bash(
            "python main.py",
            does=lambda: (agent.cwd / "metrics.json").write_text(metrics, encoding="utf-8"),
        )
        clock["now"] += 301.0  # the plan gives 300 seconds
        agent.bash("sleep 60; cat run.log")

    team.during["experimenter"] = waits
    team.metrics = None
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["timed_out"], record["error"]) == ("succeeded", False, None)
    assert record["metrics"] == METRICS
    assert any("was up before it had said that it was done" in note for note in record["notes"])
    assert project.page("E1").state == "succeeded"
    assert "no file of the code was changed after it was written" in project.page("E1-run1").body


def test_results_that_a_change_of_the_code_followed_do_not_stand_when_the_time_runs_out(
    project, team, backend, monkeypatch
):
    to_approved(project, team, backend)
    from paper_orchestra.research.sdk import session as session_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock["now"])

    def changes(agent: FakeAgent) -> None:
        (agent.cwd / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
        agent.write("main.py", GOOD_CODE + "\n# another way of measuring\n")
        # (The clock of the files: the code is newer than the metrics.)
        os.utime(agent.cwd / "metrics.json", (1_000_000, 1_000_000))
        clock["now"] += 301.0
        agent.bash("python main.py")

    team.during["experimenter"] = changes
    team.metrics = None
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["error_kind"], record["timed_out"]) == (
        "failed",
        "timeout",
        True,
    )
    assert record["metrics"] == {}


def test_the_log_of_an_agents_run_holds_what_its_commands_printed(
    project, team, backend, monkeypatch
):
    """In the classic runtime the log of a run is the program's output.  An agent's run
    shows that too, under the line that names the command, with secrets masked and
    long output cut to its end."""
    to_approved(project, team, backend)
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret-0123456789abcdef")

    def noisy(agent: FakeAgent) -> None:
        agent.bash(
            "python main.py --trials 100",
            does=lambda: "\n".join(
                [f"[{n}/100] arm=direct trial={n}" for n in range(1, 101)]
                + ["key wandb-secret-0123456789abcdef", "trial failed: TypeError"]
            ),
        )

    team.during["experimenter"] = noisy
    record = make_engine(project, team, backend).run_experiment("E1")
    assert record["status"] == "succeeded"
    log = (project.root / "experiments" / "E1" / "runs" / "1" / "log.txt").read_text()
    lines = log.splitlines()
    told = next(i for i, line in enumerate(lines) if "python main.py --trials 100" in line)
    assert lines[told].startswith("[paper-orchestra] The experimenter ran")
    assert lines[told + 1] == "(62 earlier lines of this output are not shown)"
    assert "[100/100] arm=direct trial=100" in lines and "trial failed: TypeError" in lines
    assert "[1/100] arm=direct trial=1" not in lines
    assert "wandb-secret-0123456789abcdef" not in log
    # The program's lines are not marked as the project's own.
    assert not any(line.startswith("[paper-orchestra] [") for line in lines)


def test_a_command_of_an_experiment_agent_may_take_as_long_as_the_run(project, team, backend):
    """The agent's process moves a shell command to the background after two minutes
    unless it is told otherwise; an agent then waits with `sleep` and loses its time."""
    to_approved(project, team, backend)
    engine = make_engine(project, team, backend)
    engine.run_experiment("E1")
    options = engine.driver.opened[-1]
    assert options.env["BASH_DEFAULT_TIMEOUT_MS"] == options.env["BASH_MAX_TIMEOUT_MS"] == "300000"
    said = team.prompts["experimenter"][-1]
    assert "Run the code in the foreground" in said and "`nohup`" in said
    assert "unless a valid `metrics.json` is in place" in said
    # No other agent's shell is changed.
    assert all("BASH_DEFAULT_TIMEOUT_MS" not in o.env for o in engine.driver.opened[:-1])


def test_a_failed_run_on_a_remote_backend_does_not_say_that_an_agent_repairs_it(
    project, team, backend
):
    """Under sdk nobody repairs afterwards.  Why is said of the run that failed: an agent
    worked on it and repaired as it went, or it ran where no agent works."""
    to_approved(project, team, backend)

    class Remote(FakeBackend):
        name = "modal"

    remote = Remote()
    experiments.withdraw(project, "E1")
    experiments.write_code(project, "E1", {"main.py": GOOD_CODE + "\n# BUG\n"}, author="human")
    experiments.approve(project, "E1", backend="modal", backends={"modal": remote})
    engine = Engine(
        project,
        compute=remote,
        client_for=no_classic_model,
        scholar=make_scholar(),
        driver=FakeDriver(team),
    )
    record = engine.run_experiment("E1")
    assert (record["status"], record["error_kind"], record["backend"]) == (
        "failed",
        "exit",
        "modal",
    )
    assert team.calls.count("experimenter") == 0 and record["exit_code"] == 1
    (note,) = [note for note in record["notes"] if "not repaired" in note]
    assert "this run was on the modal backend, where no agent works" in note
    assert "repairs its code itself" not in note


def test_a_run_whose_code_can_call_the_model_api_says_that_those_calls_are_not_counted(
    project, team, backend, monkeypatch, tmp_path
):
    to_approved(project, team, backend)
    token = tmp_path / "identity.jwt"
    token.write_text("jwt", encoding="utf-8")
    for name in FEDERATION_ENV:
        monkeypatch.setenv(name, "id")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token))
    monkeypatch.setenv("PAPER_ORCHESTRA_JOB_MODEL_ACCESS", "1")
    experiments.withdraw(project, "E1")
    approve(project, "E1", backend, network=True)
    record = make_engine(project, team, backend).run_experiment("E1")
    assert record["status"] == "succeeded" and record["model_access"] is True
    body = project.page("E1-run1").body
    assert "its code could call the Anthropic API" in body and "counted nowhere here" in body
    assert "| Estimated cost | nothing is billed |" not in body


def test_a_run_without_model_access_does_not_say_so(project, team, backend):
    to_approved(project, team, backend)
    record = make_engine(project, team, backend).run_experiment("E1")
    assert "model_access" not in record
    assert "Anthropic API" not in project.page("E1-run1").body


def test_research_run_exits_as_advance_does_when_the_agents_cannot_start(
    project, team, backend, home
):
    """Under sdk a run needs an agent.  `research run` (which the app starts on approval)
    says so with the exit code `research advance` uses, not as an error of the project."""
    to_approved(project, team, backend)
    (home / ".claude" / ".credentials.json").unlink()
    ran = runner.invoke(cli.app, ["research", "run", str(project.root), "E1"])
    assert ran.exit_code == cli.NEEDS_A_MODEL, ran.output
    assert "Carrying out E1 needs a model" in ran.output and "Error:" not in ran.output
    advanced = runner.invoke(cli.app, ["research", "advance", str(project.root)])
    assert advanced.exit_code == cli.NEEDS_A_MODEL, advanced.output
    assert project.page("E1").state == "approved"


def test_an_agent_that_leaves_no_metrics_has_failed_and_the_run_says_so(project, team, backend):
    to_approved(project, team, backend)
    team.metrics = None
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["error_kind"]) == ("failed", "metrics")
    assert "metrics.json" in record["error"]
    # No model is asked for a repair: the agent was the one to repair, and it is done.
    assert team.calls.count("experimenter") == 1
    assert any(
        "the agent that carries out an experiment repairs its code itself" in note
        for note in record["notes"]
    )
    result = project.page("E1-run1")
    assert result.state == "failed" and project.page("E1").state == "failed"
    log = (project.root / "experiments" / "E1" / "runs" / "1" / "log.txt").read_text()
    assert "The agent finished without leaving metrics.json in its folder." in log


def test_metrics_that_are_not_a_flat_object_fail_the_run(project, team, backend):
    to_approved(project, team, backend)
    team.metrics = {"curve": [1, 2, 3]}
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["error_kind"]) == ("failed", "metrics")


def test_the_plans_time_limit_bounds_the_agents_work(project, team, backend, monkeypatch):
    to_approved(project, team, backend)
    from paper_orchestra.research.sdk import session as session_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock["now"])
    reached = []

    def slow(agent: FakeAgent) -> None:
        agent.tick()
        reached.append("in time")
        clock["now"] += 301.0  # the plan gives 300 seconds
        agent.tick()
        agent.bash("python main.py")
        reached.append("after the time was up")

    team.during["experimenter"] = slow
    engine = make_engine(project, team, backend)
    record = engine.run_experiment("E1")
    assert reached == ["in time"] and engine.driver.interrupts == 1
    assert (record["status"], record["error_kind"], record["timed_out"]) == (
        "failed",
        "timeout",
        True,
    )
    assert project.page("E1-run1").state == "failed"
    log = (project.root / "experiments" / "E1" / "runs" / "1" / "log.txt").read_text()
    assert "The agent's time (300 seconds) was up before it had finished." in log


def test_the_persons_cancel_ends_the_agent_and_the_run_is_cancelled(project, team, backend):
    to_approved(project, team, backend)
    reached = []

    def cancel(agent: FakeAgent) -> None:
        experiments.request_cancel(project, "E1")
        agent.tick()
        agent.bash("python main.py")
        reached.append("after the cancel")

    team.during["experimenter"] = cancel
    engine = make_engine(project, team, backend)
    record = engine.run_experiment("E1")
    assert reached == [] and engine.driver.interrupts == 1
    assert (record["status"], record["requeued"]) == ("cancelled", False)
    assert project.page("E1").state == "cancelled" and project.page("E1-run1").state == "cancelled"


def test_a_stop_request_ends_the_agent_and_the_run_goes_back_to_the_queue(project, team, backend):
    to_approved(project, team, backend)
    engine = make_engine(project, team, backend)

    def stop(agent: FakeAgent) -> None:
        engine.stop()
        agent.tick()
        agent.bash("python main.py")

    team.during["experimenter"] = stop
    state = engine.advance()
    assert (state["status"], state["phase"]) == ("stopped", "run")
    assert state["phases"][-1]["status"] == "interrupted"
    (run,) = experiments.runs(project, "E1")
    assert (run["status"], run["requeued"]) == ("cancelled", True)
    assert project.page("E1").state == "queued" and project.pages(kind="result") == []

    del team.during["experimenter"]
    again = make_engine(project, team, backend)
    state = again.advance()
    assert project.page("E1").state == "succeeded"
    assert [page.title.split(":")[0] for page in project.pages(kind="result")] == ["E1 run 2"]
    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    # The second run is a new agent in a new folder: the stopped one's session, which
    # worked in the first run's folder, is not carried on.
    second = again.driver.opened[0]
    assert second.resume is None and Path(second.cwd).parent.name == "2"


def test_every_run_of_a_run_phase_gets_an_agent_of_its_own(project, team, backend):
    to_approved(project, team, backend)
    other = project.create_page(
        "experiment",
        "The same with another seed",
        EXPERIMENT.format(tests="H1", mode="script").split("---\n", 2)[2],
        author="human",
        relations={"tests": ["H1"]},
        fields={"mode": "script", "timeout_seconds": 300},
    )
    experiments.write_code(project, str(other.id), {"main.py": GOOD_CODE}, author="human")
    approve(project, str(other.id), backend)

    engine = make_engine(project, team, backend)
    state = engine.advance()
    assert state["status"] == "checkpoint" and team.calls.count("experimenter") == 2
    first, second = (o for o in engine.driver.opened if Path(o.cwd).name == "work")
    # Both runs belong to one phase and so to one session id; neither agent carries the
    # other's session on, and each has its own folder, record and transcript.
    assert first.resume is None and second.resume is None
    assert Path(first.cwd).parts[-4:] == ("E1", "runs", "1", "work")
    assert Path(second.cwd).parts[-4:] == ("E2", "runs", "1", "work")
    kept = sorted(p.name for p in (project.root / "transcripts" / "agents").glob("*run*"))
    assert kept == [
        "r1-run-experimenter-E1-run1.json",
        "r1-run-experimenter-E1-run1.jsonl",
        "r1-run-experimenter-E2-run1.json",
        "r1-run-experimenter-E2-run1.jsonl",
    ]
    assert project.page("E1").state == project.page("E2").state == "succeeded"


def test_under_sdk_a_run_that_no_agent_carries_out_still_needs_code_that_is_sound(
    project, team, backend
):
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    for name, fields in (
        ("On a GPU", {"mode": "gpu", "gpu": "T4"}),
        ("On Modal", {"mode": "script", "backend": "modal"}),
    ):
        plan = project.create_page(
            "experiment",
            name,
            "## Objective\n\nx\n",
            author="human",
            relations={"tests": ["H1"]},
            fields=fields,
        )
        # Nobody writes or repairs code there: a plan without code is refused as ever.
        (problem,) = experiments.check_code(project, str(plan.id))
        assert problem.startswith(f"{plan.id} has no code yet: write its files"), problem


def test_an_experiment_agent_proposes_a_follow_up_and_cannot_decide_or_run_it(
    project, team, backend
):
    to_approved(project, team, backend)
    team.follow_up = True
    answers: dict[str, Any] = {}

    def tries(agent: FakeAgent) -> None:
        answers["approve"] = agent.bash("paper-orchestra research approve . E2")
        answers["page"] = agent.write(str(project.root / "wiki" / "experiments" / "E9.md"), "x")
        answers["record"] = agent.write("../run.json", "{}")
        answers["verdict"] = agent.memory(
            "str_replace",
            path="/memories/wiki/experiments/E1-ablation-for-h1.md",
            old_str="state: approved",
            new_str="state: succeeded",
        )
        answers["failed"] = agent.failed

    team.during["experimenter"] = tries
    record = make_engine(project, team, backend).run_experiment("E1")
    assert record["status"] == "succeeded"
    follow_up = project.page("E2")
    assert (follow_up.review, follow_up.state, follow_up.author) == (
        "proposed",
        "planned",
        "agent:experimenter",
    )
    assert "It is E2" in team.experimenter_answers["follow_up"]
    for name in ("approve", "page", "record"):
        assert isinstance(answers[name], Denied), name
    assert answers["failed"] is True
    # Nothing but the approved experiment ran, and the follow-up waits for the person.
    assert experiments.runs(project, "E2") == []
    assert json.loads((project.root / "experiments/E1/runs/1/run.json").read_text())["number"] == 1
    assert project.counts()["pending"]["by_kind"].get("experiment") == 1


def test_under_sdk_a_plan_without_code_can_be_approved_and_the_agent_writes_it(
    project, team, backend
):
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    plan = project.create_page(
        "experiment",
        "Warmup against no warmup",
        EXPERIMENT.format(tests="H1", mode="script").split("---\n", 2)[2],
        author="agent:analyst",
        relations={"tests": ["H1"]},
        fields={"mode": "script", "timeout_seconds": 300},
    )
    project.review(plan.path, "accept")
    (warning,) = experiments.check_code(project, "E1")
    assert warning.startswith("Warning: E1 has no code yet. The agent that carries out")
    approve(project, "E1", backend)
    record = make_engine(project, team, backend).run_experiment("E1")
    assert record["status"] == "succeeded"
    assert team.experimenter_answers["files"] == [] and team.experimenter_answers["code"] is None
    assert (
        "The plan came without code. Write it in your folder from the plan"
        in (team.prompts["experimenter"][0])
    )


def test_a_classic_project_still_refuses_a_plan_without_code(tmp_path):
    project = make_project(tmp_path)
    hypothesis = project.create_page("hypothesis", "H", "## Statement\n\nx\n", author="human")
    plan = project.create_page(
        "experiment",
        "No code",
        "## Objective\n\nx\n",
        author="human",
        relations={"tests": [str(hypothesis.id)]},
        fields={"mode": "script"},
    )
    (problem,) = experiments.check_code(project, str(plan.id))
    assert problem.startswith("E1 has no code yet: write its files")


def test_a_run_approved_with_network_and_model_access_gets_the_launcher_and_the_token(
    project, team, backend, monkeypatch, tmp_path
):
    token = tmp_path / "identity.jwt"
    token.write_text("jwt", encoding="utf-8")
    to_approved(project, team, backend)
    experiments.withdraw(project, "E1")
    project.update_page("E1", author="human", fields={"network": True})
    monkeypatch.setenv("PAPER_ORCHESTRA_JOB_MODEL_ACCESS", "1")
    values = {name: f"{name.lower()}-value" for name in FEDERATION_ENV}
    values["ANTHROPIC_IDENTITY_TOKEN_FILE"] = str(token)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    approve(project, "E1", backend, network=True)

    engine = make_engine(project, team, backend)
    record = engine.run_experiment("E1")
    assert record["status"] == "succeeded"
    options = engine.driver.opened[-1]
    work = Path(options.cwd)
    assert (work / LAUNCHER).is_file() and (work / LAUNCHER).stat().st_mode & 0o111
    prompt = team.prompts["experimenter"][0]
    assert "its code may call the Anthropic API" in prompt
    assert f"./{LAUNCHER} python main.py" in prompt
    assert "meant to start with `python main.py`" in prompt
    # The settings travel under other names; the agent itself still signs in by its login.
    for name, value in values.items():
        assert options.env[name] == ""
        assert options.env[f"PAPER_ORCHESTRA_JOB_{name}"] == value
    # Only this agent may read the token file.
    assert str(token) in options.settings["sandbox"]["filesystem"]["allowRead"]
    scout = engine.driver.opened[0] if len(engine.driver.opened) > 1 else None
    assert scout is None or str(token) not in scout.settings["sandbox"]["filesystem"]["allowRead"]
    # The launcher is not an output, and no value is in it.
    assert "-value" not in (work / LAUNCHER).read_text()


def test_without_the_persons_grant_no_agent_gets_the_federation_settings(
    project, team, backend, monkeypatch
):
    monkeypatch.setenv("PAPER_ORCHESTRA_JOB_MODEL_ACCESS", "1")
    for name in FEDERATION_ENV:
        monkeypatch.setenv(name, f"{name.lower()}-value")
    to_approved(project, team, backend)  # approved without network access
    engine = make_engine(project, team, backend)
    engine.run_experiment("E1")
    for options in engine.driver.opened:
        # (The worker's own switch is taken away like its other variables: empty.)
        assert not any(
            value for name, value in options.env.items() if name.startswith("PAPER_ORCHESTRA_JOB_")
        )
        assert all(options.env[name] == "" for name in FEDERATION_ENV)
    assert not (Path(engine.driver.opened[-1].cwd) / LAUNCHER).exists()


def test_a_spent_model_budget_keeps_an_experiment_agent_from_starting(project, team, backend):
    to_approved(project, team, backend)
    project.settings.update({"limits": {"model_usd": 0.1}})
    project.save_settings()
    engine = make_engine(project, team, backend)
    with pytest.raises(EngineError, match="The model budget is used up"):
        engine.run_experiment("E1")
    assert team.calls.count("experimenter") == 0
    assert project.page("E1").state == "approved" and experiments.runs(project, "E1") == []


def test_a_run_approved_for_a_remote_backend_is_not_an_agents(project, team, backend):
    to_approved(project, team, backend)

    class Remote(FakeBackend):
        name = "modal"

    remote = Remote()
    experiments.withdraw(project, "E1")
    experiments.approve(project, "E1", backend="modal", backends={"modal": remote})
    engine = Engine(
        project,
        compute=remote,
        client_for=no_classic_model,
        scholar=make_scholar(),
        driver=FakeDriver(team),
    )
    record = engine.run_experiment("E1")
    assert (record["status"], record["backend"]) == ("succeeded", "modal")
    assert len(remote.jobs) == 1 and team.calls.count("experimenter") == 0
    # ... and neither its record nor its start says that one did.
    assert "carried_out_by" not in record
    assert "carried_out_by" not in events(project, "run.started")[-1]["data"]


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

runner = CliRunner()


def invoke(*args: str) -> Any:
    return runner.invoke(cli.app, ["research", *args])


def test_new_writes_the_runtime_into_the_project_from_the_option_or_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    plain = invoke("new", "--brief", "Does warmup help?", "--dir", "plain", "--no-run")
    assert plain.exit_code == 0, plain.output
    assert Project.open(tmp_path / "plain").settings.agents.runtime == "classic"
    assert (
        json.loads((tmp_path / "plain" / "project.json").read_text())["settings"]["agents"][
            "runtime"
        ]
        == "classic"
    )

    monkeypatch.setenv(AGENT_RUNTIME_ENV, "sdk")
    by_env = invoke("new", "--brief", "Does warmup help?", "--dir", "by-env", "--no-run")
    assert by_env.exit_code == 0, by_env.output
    assert Project.open(tmp_path / "by-env").settings.agents.to_dict() == {
        "runtime": "sdk",
        "model": "claude-fable-5-2",
        "fast": True,
        "ultracode": True,
        "credential": "login",
        "sandbox": True,
    }
    named = invoke(
        "new", "--brief", "Does warmup help?", "--dir", "named", "--no-run", "--runtime", "classic"
    )
    assert named.exit_code == 0, named.output
    assert Project.open(tmp_path / "named").settings.agents.runtime == "classic"
    # The project that existed before the variable was set is what it was.
    assert Project.open(tmp_path / "plain").settings.agents.runtime == "classic"

    wrong = invoke("new", "--brief", "x y z", "--dir", "wrong", "--no-run", "--runtime", "turbo")
    assert wrong.exit_code == 2 and "settings.agents.runtime is 'turbo'" in wrong.output
    assert not (tmp_path / "wrong").exists()


def test_new_takes_the_settings_of_the_agents_before_the_first_of_them_starts(
    tmp_path, monkeypatch
):
    """Without these a project made on the command line starts its first session on the
    person's own account, on the dearest model, in fast mode, before any setting can be
    changed."""
    monkeypatch.chdir(tmp_path)
    made = invoke(
        "new", "--brief", "Does warmup help?", "--dir", "p", "--no-run", "--runtime", "sdk",
        "--credential", "federation", "--model", "claude-haiku-5-5", "--no-fast",
        "--no-ultracode",
    )  # fmt: skip
    assert made.exit_code == 0, made.output
    assert Project.open(tmp_path / "p").settings.agents.to_dict() == {
        "runtime": "sdk",
        "model": "claude-haiku-5-5",
        "fast": False,
        "ultracode": False,
        "credential": "federation",
        "sandbox": True,
    }
    said = " ".join(made.output.split())
    assert "The agents are Claude agents on claude-haiku-5-5, signed in with the worker's" in said
    assert "billed to that organization" in said and "fast mode" not in said.split("`")[0]

    # The defaults say whose account pays, and that fast mode is dear, before anything runs.
    monkeypatch.setenv(AGENT_RUNTIME_ENV, "sdk")
    plain = invoke("new", "--brief", "Does warmup help?", "--dir", "q", "--no-run")
    said = " ".join(plain.output.split())
    assert "Claude agents on claude-fable-5-2 in fast mode, signed in with your Claude Code" in said
    assert "billed to your own Claude account, and fast mode costs several times" in said
    assert said.index("billed to your own") < said.index("Next:")
    # A classic project is told nothing about agents' accounts.
    classic = invoke("new", "--brief", "x y z", "--dir", "c", "--no-run", "--runtime", "classic")
    assert "billed" not in classic.output

    wrong = invoke("new", "--brief", "x y z", "--dir", "w", "--no-run", "--credential", "mine")
    assert wrong.exit_code == 2 and "credential" in wrong.output
    assert not (tmp_path / "w").exists()


def test_settings_takes_model_as_another_name_for_agent_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    invoke("new", "--brief", "Does warmup help?", "--dir", "p", "--no-run", "--runtime", "sdk")
    changed = invoke("settings", "p", "--credential", "federation", "--model", "claude-haiku-5-5")
    assert changed.exit_code == 0, changed.output
    agents = Project.open(tmp_path / "p").settings.agents
    assert (agents.model, agents.credential) == ("claude-haiku-5-5", "federation")


def test_research_run_says_each_command_of_the_agent_once(project, team, backend, monkeypatch):
    """The feed says what the agent that carries a run out does, and the log of the run
    holds the same sentences: printed side by side, each command reads twice."""
    to_approved(project, team, backend)
    monkeypatch.setattr(experiments, "LOG_EVENT_SECONDS", 0.0)  # every line of the log is told
    printer = cli._EventPrinter(project)
    printed: list[str] = []
    monkeypatch.setattr(cli, "_say", lambda line, **how: printed.append(line))
    make_engine(project, team, backend).run_experiment("E1")
    printer.flush()
    ran = [line for line in printed if "python main.py" in line]
    assert len(ran) == 1 and "(experimenter): The experimenter ran" in ran[0]
    # Lines of the log that the feed does not have are still printed.
    assert any("An agent carries out E1" in line for line in printed)
    log_events = events(project, "run.log")
    assert any("The experimenter ran" in str(event["data"]["lines"]) for event in log_events)


def test_settings_shows_and_changes_how_the_agents_are_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    invoke("new", "--brief", "Does warmup help?", "--dir", "p", "--no-run")
    shown = invoke("settings", "p")
    assert shown.exit_code == 0 and "Agents: classic" in shown.output

    changed = invoke(
        "settings",
        "p",
        "--runtime",
        "sdk",
        "--credential",
        "federation",
        "--no-fast",
        "--agent-model",
        "claude-haiku-5-5",
    )
    assert changed.exit_code == 0, changed.output
    said = " ".join(changed.output.split())
    assert "Changed runtime, model, fast, credential" in said
    assert "Agents: sdk (Claude Agent SDK agents on claude-haiku-5-5; fast mode off;" in said
    assert "credential: the worker's federation; sandbox on" in said
    assert Project.open(tmp_path / "p").settings.agents.to_dict() == {
        "runtime": "sdk",
        "model": "claude-haiku-5-5",
        "fast": False,
        "ultracode": True,
        "credential": "federation",
        "sandbox": True,
    }
    as_json = invoke("settings", "p", "--no-sandbox", "--json")
    assert json.loads(as_json.output)["sandbox"] is False
    refused = invoke("settings", "p", "--runtime", "turbo")
    assert refused.exit_code == 2 and "must be one of: classic, sdk" in refused.output
    assert Project.open(tmp_path / "p").settings.agents.runtime == "sdk"


def test_the_web_app_gives_a_new_project_the_runtime_the_environment_names(monkeypatch):
    pytest.importorskip("fastapi")
    from paper_orchestra.web.api import Services

    assert Services.default_settings(None)["agents"]["runtime"] == "classic"
    monkeypatch.setenv(AGENT_RUNTIME_ENV, "sdk")
    assert Services.default_settings(None)["agents"]["runtime"] == "sdk"


# ---------------------------------------------------------------------------
# What the third review found
# ---------------------------------------------------------------------------


def test_a_run_the_agent_says_failed_is_recorded_as_failed_with_its_words(project, team, backend):
    """A valid metrics.json was all a run needed to be recorded as succeeded: code that
    caught its own errors wrote one that measured nothing, the agent said in its reply
    that the run had failed, and the board showed a green "succeeded"."""
    to_approved(project, team, backend)
    team.metrics = {"direct_trials": 0, "api_calls": 10, "api_errors": 10}
    why = "Every one of the 10 API calls failed, so no trial was scored and nothing was measured."

    def says_so(agent: FakeAgent) -> None:
        assert "fail_run" in agent.options.system_prompt and "fail_run" in agent.prompt
        short = agent.board("fail_run", reason="it failed")
        assert agent.failed and "Give the reason" in short
        said = agent.board("fail_run", reason=why)
        assert not agent.failed and "recorded as failed" in said

    team.during["experimenter"] = says_so
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["error_kind"], record["requeued"]) == (
        "failed",
        "declared",
        False,
    )
    assert record["error"] == f'The agent that carried out the run says that it failed: "{why}".'
    assert record["metrics"] == {} and record["repairs"] == []
    assert project.page("E1").state == "failed"
    result = project.page("E1-run1")
    assert result.state == "failed" and why in result.body
    # What the code wrote is kept as a file, and counts as no result.
    assert (project.root / "experiments" / "E1" / "runs" / "1" / "metrics.json").is_file()
    log = (project.root / "experiments" / "E1" / "runs" / "1" / "log.txt").read_text()
    assert f"The agent says that the run failed: {why}" in log
    assert team.calls.count("experimenter") == 1


def test_an_agent_that_got_the_run_to_work_after_all_takes_its_statement_back(
    project, team, backend
):
    to_approved(project, team, backend)

    def changed_its_mind(agent: FakeAgent) -> None:
        agent.board("fail_run", reason="The first attempt measured nothing at all, it seemed.")
        agent.board("fail_run", withdraw=True)

    team.during["experimenter"] = changed_its_mind
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["metrics"]) == ("succeeded", METRICS)


def test_a_statement_that_the_run_failed_stands_when_the_time_runs_out(
    project, team, backend, monkeypatch
):
    """Time that is up with a valid metrics.json in place records the run with those
    metrics.  Not when the agent had said that they are no result."""
    to_approved(project, team, backend)
    from paper_orchestra.research.sdk import experiment as experiment_module

    def late(agent: FakeAgent) -> None:
        (agent.cwd / "metrics.json").write_text(json.dumps({"trials": 0}), encoding="utf-8")
        agent.board("fail_run", reason="No trial ran: the data could not be downloaded in time.")
        # The time is up while it is still at work.
        monkeypatch.setattr(experiment_module.time, "monotonic", lambda: 10**9)
        agent.tick(3)

    team.metrics = None
    team.during["experimenter"] = late
    record = make_engine(project, team, backend).run_experiment("E1")
    assert (record["status"], record["error_kind"]) == ("failed", "declared")
    assert "No trial ran" in record["error"]


def test_under_sdk_a_run_for_a_remote_backend_does_not_wait_for_the_agents(
    project, team, backend, home
):
    """Such a run is carried out where it was approved for, without an agent.  It waited
    all the same for the agents' login, with a sentence about a model it does not use."""
    to_approved(project, team, backend)

    class Remote(FakeBackend):
        name = "modal"

    remote = Remote()
    experiments.withdraw(project, "E1")
    experiments.approve(project, "E1", backend="modal", backends={"modal": remote})
    (home / ".claude" / ".credentials.json").unlink()
    engine = Engine(
        project,
        compute=remote,
        client_for=no_classic_model,
        scholar=make_scholar(),
        driver=FakeDriver(team),
    )
    assert engine._model_need("run") is None
    assert engine._model_need("analyze") is not None  # the analyst is an agent
    record = engine.run_experiment("E1")
    assert (record["status"], record["backend"]) == ("succeeded", "modal")
    assert team.calls.count("experimenter") == 0

    # A run for this machine is an agent's, and waits for what the agents need.
    experiments.approve(project, "E1", backend="local", backends={"local": backend})
    local = make_engine(project, team, backend)
    assert local._model_need("run") is not None
    with pytest.raises(ModelNeeded, match="no Claude Code login"):
        local.run_experiment("E1")


def test_an_agent_that_could_not_work_is_not_called_a_compute_backend(project, team, backend):
    to_approved(project, team, backend)

    def overloaded(agent: FakeAgent) -> None:
        raise RuntimeError("API Error: 529 Overloaded")

    team.during["experimenter"] = overloaded
    state = advance(project, team, backend)
    (run,) = experiments.runs(project, "E1")
    assert run["error"].startswith("The agent that was to carry out E1 could not work (")
    assert "compute backend" not in run["error"] and run["requeued"] is True
    # The phase's record says it as a sentence, without the name of a class.
    assert state["phases"][-1]["error"].startswith("The agent that was to carry out E1")
    # The feed says that a second agent took the run up, as the log does.
    assert any(
        "another agent takes the run up once more" in line and "529" in line
        for line in activity(project)
    )


def test_without_the_sandbox_it_asks_for_no_experiment_agent_runs_and_the_approval_stays(
    project, team, backend
):
    to_approved(project, team, backend)
    from paper_orchestra.research.sdk.guard import sandbox_fault

    # The gate refuses before anything starts, in a sentence.
    gated = make_engine(project, team, backend)
    gated.driver.no_sandbox = sandbox_fault("/nowhere", platform="linux")
    with pytest.raises(ModelNeeded) as refused:
        gated.advance()
    assert "bwrap and socat are not installed" in str(refused.value)
    assert team.calls.count("experimenter") == 0 and project.page("E1").state == "approved"

    # Where the gate found nothing and the agent's own start does, the run is not
    # carried out, is not tried by a second agent, and keeps its approval.
    engine = make_engine(project, team, backend)
    engine._agents_need = lambda: None
    engine.driver.no_sandbox = gated.driver.no_sandbox
    from paper_orchestra.compute.base import BackendUnusable

    with pytest.raises(BackendUnusable, match="Nothing was run outside the sandbox"):
        engine.run_experiment("E1")
    assert team.calls.count("experimenter") == 0 and engine.driver.opened == gated.driver.opened
    (run,) = experiments.runs(project, "E1")
    assert run["requeued"] is True and project.page("E1").state == "queued"
    assert run["error"].startswith("The experimenter agent was not started:")
    assert "compute backend" not in run["error"]


def test_no_agent_is_started_with_less_left_than_a_start_was_seen_to_cost(project, team, backend):
    """In fast mode the first request of an agent was seen to cost most of a dollar: an
    agent started with fifty cents left took the project to $1.91 of $1.50 within
    seconds.  It is not started; the work stops at the checkpoint with the reason."""
    project.settings.update({"limits": {"model_usd": 1.5}})
    project.save_settings()
    team.cost = 0.976  # what the one request of each agent costs, as its process says
    engine = make_engine(project, team, backend)
    state = engine.advance()
    assert team.calls == ["intake"] and len(engine.driver.opened) == 1
    assert (state["status"], state["next_phase"]) == ("checkpoint", "research")
    said = state["phases"][-1]["error"]
    assert said.startswith("What is left of the model budget does not cover the start of the scout")
    assert "$0.98 of $1.50 is spent" in said and "so it was not started" in said
    assert state["spent"]["model_usd"] == pytest.approx(0.976)
    (warning,) = events(project, "budget.warning")
    assert warning["data"]["limit"] == "model_usd"
    # With the limit raised the phase starts.
    project.settings.update({"limits": {"model_usd": 5.0}})
    project.save_settings()
    again = advance(project, team, backend)
    assert "scout" in team.calls and again["phases"][-1]["status"] == "completed"


def test_a_limit_is_shown_with_the_digits_the_person_gave_it(project, team, backend):
    """A limit of $0.145 read "$0.15 of $0.14 is spent"."""
    from paper_orchestra.research.engine import _limit_money

    assert _limit_money(0.145) == "$0.145" and _limit_money(0.14) == "$0.14"
    assert _limit_money(25) == "$25.00" and _limit_money(0.004) == "$0.004"
    project.settings.update({"limits": {"model_usd": 0.145}})
    project.save_settings()
    team.cost = 0.15
    engine = make_engine(project, team, backend)
    engine.driver.started["fast_mode_state"] = "off"
    state = engine.advance()
    assert "$0.15 of $0.145 is spent" in json.dumps(state["checkpoint"])


def test_an_agent_on_an_unpriced_model_that_was_killed_makes_the_spend_a_lower_bound(
    project, team, backend
):
    """Such a turn was recorded as costing $0.00: status showed the spend as exact, and
    the ledger did not list the model among those it cannot price."""
    project.settings.update({"agents": {"model": "mystery-model-9"}})
    project.save_settings()

    def killed(agent: FakeAgent) -> None:
        agent.use(cost_usd=3.0, input_tokens=400_000, output_tokens=40_000)
        raise KeyboardInterrupt

    team.during["intake"] = killed
    with pytest.raises(KeyboardInterrupt):
        make_engine(project, team, backend).advance()
    report = usage(project)
    assert report["unpriced_models"] == ["mystery-model-9"]
    assert report["unreported_unpriced"] == ["mystery-model-9"]
    (row,) = [row for row in report["by_label"] if row["label"] == "intake.agent.unreported"]
    assert "cost_usd" not in row and row["input_tokens"] >= 400_000
    status = " ".join(
        runner.invoke(cli.app, ["research", "status", str(project.root)]).output.split()
    )
    assert "The model spend is at least that: an agent on 'mystery-model-9' was stopped" in status
    assert "has no list price here" in status


def test_the_command_line_says_that_the_agent_wrote_the_code_of_a_plan_without_code():
    """It said "changed the code it was given" of a plan that came with none."""
    wrote = {"approved": 0, "changed": [], "added": ["main.py"], "removed": []}
    assert cli._did_to_the_code(wrote) == "wrote the code for this run"
    changed = {"approved": 1, "changed": ["main.py"], "added": ["helper.py"], "removed": []}
    assert cli._did_to_the_code(changed) == "changed the code it was given"
    # A record from before the count of approved files was kept says what it always said.
    assert cli._did_to_the_code({"added": ["main.py"]}) == "changed the code it was given"
