"""The research engine, offline: rounds, checkpoints, resuming, budgets, the worker.

A scripted team of models drives real projects through the real engine, access
layer and harness.  The helpers and the stand-ins for the network, the compute
backend and (while it is missing) ``research.experiments`` come from
``test_research_agents.py``.
"""

from __future__ import annotations

import json
import os
import re
import signal
import time
import types
from pathlib import Path

import pytest

from paper_orchestra.compute import ComputeError, JobResult
from paper_orchestra.llm.base import Capabilities, LLMConfigError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.research import engine as engine_module
from paper_orchestra.research import experiments
from paper_orchestra.research.engine import (
    PHASES,
    Engine,
    EngineError,
    WorkerBusy,
    read_state,
    worker_alive,
)
from paper_orchestra.research.model import NotAllowed, ProjectError
from paper_orchestra.research.project import Project
from test_research_agents import (
    BRIEF,
    BROKEN_CODE,
    EXPERIMENT,
    FAILING_CODE,
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

STATE_KEYS = {
    "schema",
    "status",
    "round",
    "phase",
    "next_phase",
    "checkpoint",
    "phases",
    "worker",
    "spent",
    "error",
}
PHASE_KEYS = {
    "round",
    "phase",
    "status",
    "started",
    "finished",
    "seconds",
    "attempts",
    "error",
    "produced",
    "instruction",
}


class Team:
    """The scripted models of a project.  Each role answers the way a capable model would:
    it looks at the index it was given and does not write again what is already there."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sessions: list[str] = []
        self.texts: dict[str, list[str]] = {}
        #: label -> how many more times that role's session fails before it works.
        self.failures: dict[str, int] = {}
        self.error: Exception = RuntimeError("the provider had a bad moment")
        #: Label of the session in which the person presses Ctrl-C, after its first page.
        self.interrupt: str | None = None
        #: label -> callable run in the middle of that role's session.
        self.during: dict[str, object] = {}
        self.finding_text = "With warmup the variance was 0.42, without it 0.91."
        self.journal_meta = ""
        self.code = GOOD_CODE
        self.repaired = GOOD_CODE

    def __call__(self, request: FakeRequest) -> str:
        label = request.label
        self.calls.append(label)
        self.sessions.append(request.memory.session_id if request.memory else "")
        self.texts.setdefault(label, []).append(request.text)
        if self.failures.get(label):
            self.failures[label] -= 1
            raise self.error
        return getattr(self, label.replace(".", "_"))(request)

    def _midway(self, request: FakeRequest) -> None:
        """Between two pages of a session: where a test stops, interrupts or looks around."""
        hook = self.during.get(request.label)
        if hook is not None:
            hook()
        if self.interrupt == request.label:
            self.interrupt = None
            raise KeyboardInterrupt

    @staticmethod
    def _has(request: FakeRequest, slug: str) -> bool:
        return slug in (request.system or "")

    def intake_intake(self, request: FakeRequest) -> str:
        if not self._has(request, "what-the-notes-say"):
            request.memory.create(
                "/memories/wiki/topics/what-the-notes-say.md",
                "---\ntitle: What the notes say\nsummary: Warmup of 4000 steps was used\n---\n"
                "inputs/notes.md states that warmup of 4000 steps was used.\n",
            )
        self._midway(request)
        if not self._has(request, "why-4000-steps"):
            request.memory.create(
                "/memories/wiki/questions/why-4000-steps.md",
                "---\ntitle: Why 4000 warmup steps?\nabout: [idea]\n---\nThe notes do not say.\n",
            )
        return "Wrote one topic and one question."

    def research_scout(self, request: FakeRequest) -> str:
        keys = []
        for paper in LIBRARY:
            text = request.call_tool("verify_paper", {"title": paper["title"]})
            keys.append(re.search(r"ecorded as (\w+)", text).group(1))
        if not self._has(request, "warmup-in-the-literature"):
            request.memory.create(
                "/memories/wiki/topics/warmup-in-the-literature.md",
                "---\ntitle: Warmup in the literature\nsummary: Warmup is described as "
                f"variance reduction\ncites: [{', '.join(keys)}]\n---\nThe abstract of "
                f"[[sources/{keys[1]}]] states that warmup reduces variance.\n",
            )
        self._midway(request)
        if not self._has(request, "small-models"):
            request.memory.create(
                "/memories/wiki/questions/small-models.md",
                "---\ntitle: Does the effect hold for small transformers?\nabout: [idea]\n---\n"
                "Neither abstract speaks about model size.\n",
            )
        return "Recorded the sources, wrote one topic and one question."

    def hypothesize_theorist(self, request: FakeRequest) -> str:
        question = re.search(r"questions/(Q\d+)-", request.system).group(1)
        round = re.search(r"r(\d+)-hypothesize", request.memory.session_id).group(1)
        for slug, title in (
            (f"warmup-lowers-variance-r{round}", "Warmup lowers the variance of early updates"),
            (f"warmup-matters-less-when-small-r{round}", "Warmup matters less in small models"),
        ):
            if not self._has(request, slug):
                request.memory.create(
                    f"/memories/wiki/hypotheses/{slug}.md",
                    HYPOTHESIS.format(motivated_by=question).replace(
                        "Warmup lowers the variance of early updates", title, 1
                    ),
                )
            self._midway(request)
        return "Proposed two hypotheses."

    def design_designer(self, request: FakeRequest) -> str:
        if request.history:
            for experiment in re.findall(r"^(E\d+):$", request.text, re.M):
                request.call_tool(
                    "write_code", {"experiment": experiment, "name": "main.py", "text": GOOD_CODE}
                )
            return "Repaired the code."
        wanted = re.findall(r'^- (H\d+) "', request.text, re.M)
        for hypothesis in wanted:
            slug = f"ablation-for-{hypothesis.lower()}"
            known = re.search(rf"experiments/(E\d+)-{slug}", request.system or "")
            if known:
                experiment = known.group(1)
            else:
                experiment = created_id(
                    request.memory.create(
                        f"/memories/wiki/experiments/{slug}.md",
                        EXPERIMENT.format(tests=hypothesis, mode="script"),
                    )
                )
            self._midway(request)
            request.call_tool(
                "write_code", {"experiment": experiment, "name": "main.py", "text": self.code}
            )
        return f"Designed {len(wanted)} experiment(s)."

    def run_runner(self, request: FakeRequest) -> str:
        return json.dumps(
            {
                "explanation": "A stray raise ended the script before it measured anything.",
                "files": [{"name": "main.py", "text": self.repaired}],
            }
        )

    def analyze_analyst(self, request: FakeRequest) -> str:
        if request.history:
            return "The numbers stay as they are."
        results = re.findall(
            r"^- /memories/(wiki/results/(E\d+)-run\d+)\.md: .*tests (H\d+)", request.text, re.M
        )
        for path, experiment, hypothesis in results:
            slug = f"what-{path.rsplit('/', 1)[1].lower()}-shows"
            if not self._has(request, slug):
                request.memory.create(
                    f"/memories/wiki/findings/{slug}.md",
                    f"---\ntitle: What {experiment} shows\nsummary: The variance was lower "
                    f"with warmup\nbased_on: [{path.rsplit('/', 1)[1]}]\n"
                    f"supports: [{hypothesis}]\n---\n## What was observed\n\n"
                    f"{self.finding_text}\n\n## What it means\n\nThe prediction held.\n\n"
                    "## Limits\n\nOne seed.\n",
                )
                request.call_tool(
                    "propose_edit",
                    {
                        "target": hypothesis,
                        "summary": f"The run of {experiment} supports it",
                        "fields": {"state": "supported"},
                    },
                )
            self._midway(request)
        journal = re.search(r"goes to (/memories/wiki/journal/\S+\.md)", request.text).group(1)
        request.memory.create(
            journal,
            f"---\ntitle: What this round did\nsummary: {len(results)} result(s) read\n"
            f"{self.journal_meta}---\nThe runs were read and findings proposed.\n",
        )
        return "Wrote the findings and the journal."


@pytest.fixture
def team() -> Team:
    return Team()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def project(tmp_path) -> Project:
    notes = tmp_path / "notes.md"
    notes.write_text("Warmup of 4000 steps was used in the first runs.\n")
    return make_project(tmp_path, files=[notes])


def make_engine(project: Project, team: Team, backend: FakeBackend | None = None) -> Engine:
    client = FakeLLM(team)
    return Engine(
        project,
        compute=backend or FakeBackend(),
        client_for=lambda _: client,
        scholar=make_scholar(),
    )


def advance(project: Project, team: Team, backend: FakeBackend, **options) -> dict:
    """One ``advance()`` by a fresh worker, as the command line starts one."""
    return make_engine(project, team, backend).advance(**options)


def events(project: Project, type: str) -> list[dict]:
    return [event for event in project.events() if event["type"] == type]


def approve(project: Project, ref: str, backend: FakeBackend | None = None) -> None:
    """The person approves a run on the backend the tests stand in for."""
    experiments.approve(project, ref, backends={"local": backend or FakeBackend()})


def approve_all(project: Project) -> None:
    for page in project.pages(kind="experiment", review="accepted", state="planned"):
        approve(project, page.id)


def to_design_checkpoint(project: Project, team: Team, backend: FakeBackend) -> None:
    """Research accepted, one of two hypotheses accepted, its experiment designed and accepted."""
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    project.review("H1", "accept")
    project.review("H2", "reject", reason="Too vague to test")
    advance(project, team, backend)
    accept_all(project)


def wiki(project: Project) -> list[tuple[str, str]]:
    return sorted((page.kind, page.title) for page in project.pages())


# ---------------------------------------------------------------------------
# A round
# ---------------------------------------------------------------------------


def test_a_full_round_pauses_at_each_checkpoint_and_builds_on_what_was_accepted(
    project, team, backend
):
    assert read_state(project)["status"] == "idle"
    assert read_state(project)["next_phase"] == "intake"

    # Intake runs once, straight into research; then the person reviews.
    state = advance(project, team, backend)
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
    assert state["checkpoint"]["pending"] == {"topic": 2, "question": 2}
    assert "2 questions and 2 topics await your review." in state["checkpoint"]["summary"]
    assert "The scout recorded 2 sources" in state["checkpoint"]["summary"]
    assert state["checkpoint"]["summary"].endswith(
        "Next: proposing hypotheses from what you accepted."
    )
    assert len(project.pages(kind="source", review="accepted")) == 2
    assert state["worker"] is None
    accept_all(project)

    # Hypotheses: one accepted, one turned down with a reason.
    state = advance(project, team, backend)
    assert (state["phase"], state["next_phase"]) == ("hypothesize", "design")
    assert state["checkpoint"]["pending"] == {"hypothesis": 2}
    assert state["checkpoint"]["summary"] == (
        "The theorist proposed 2 hypotheses. 2 hypotheses await your review. "
        "Next: designing experiments for the hypotheses you accepted."
    )
    # The feed has heard what the phase did; the checkpoint adds what is waiting and what is next.
    assert events(project, "checkpoint")[-1]["summary"] == (
        "2 hypotheses await your review. "
        "Next: designing experiments for the hypotheses you accepted."
    )
    project.review("H1", "accept")
    project.review("H2", "reject", reason="Too vague to test")

    # Design: only for what the person accepted.
    state = advance(project, team, backend)
    assert (state["phase"], state["next_phase"]) == ("design", "run")
    assert state["checkpoint"]["pending"] == {"experiment": 1}
    plan = project.page("E1")
    assert plan.relations["tests"] == ["H1"] and plan.state == "planned"
    assert experiments.code_files(project, "E1") == {"main.py": GOOD_CODE}
    project.review("E1", "accept")

    # Accepted is not approved: nothing runs, and the checkpoint says what to approve.
    state = advance(project, team, backend)
    assert (state["status"], state["phase"], state["next_phase"]) == ("checkpoint", "run", "run")
    assert "No experiment is approved to run." in state["checkpoint"]["summary"]
    # (The title is the designer's: it stands there as a quotation.)
    assert "E1 “Warmup against no warmup" in state["checkpoint"]["summary"]
    assert backend.jobs == [] and "analyze.analyst" not in team.calls
    approve(project, "E1")

    # The approved experiment runs, and the analysis follows without a pause.
    state = advance(project, team, backend)
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "analyze",
        "hypothesize",
    )
    assert len(backend.jobs) == 1 and project.page("E1").state == "succeeded"
    result = project.page("E1-run1")
    assert result.meta["analysed"] == 1
    finding = project.page("F1")
    assert finding.review == "proposed" and finding.relations["supports"] == ["H1"]
    assert project.page("H1").state == "testing"
    assert state["checkpoint"]["pending"] == {"finding": 1}
    summary = state["checkpoint"]["summary"]
    assert "1 experiment ran: E1 succeeded." in summary
    assert "1 finding and 1 suggested edit await your review." in summary
    assert state["spent"]["experiment_seconds"] == 2.5 and state["spent"]["model_usd"] > 0
    assert project.page("wiki/journal/round-1.md").review == "accepted"
    project.review("F1", "accept")
    project.review(project.proposals()[0].id, "accept")
    assert project.page("H1").state == "supported"

    # The next round starts with new hypotheses, told what was rejected before.
    state = advance(project, team, backend)
    assert (state["round"], state["phase"], state["next_phase"]) == (2, "hypothesize", "design")
    told = team.texts["hypothesize.theorist"][-1]
    assert '"Warmup matters less in small models": Too vague to test' in told
    assert {page.round for page in project.pages(kind="hypothesis", review="proposed")} == {2}

    assert team.calls == [
        "intake.intake",
        "research.scout",
        "hypothesize.theorist",
        "design.designer",
        "analyze.analyst",
        "hypothesize.theorist",
    ]
    order = [(p["round"], p["phase"]) for p in state["phases"]]
    assert order == [
        (1, "intake"),
        (1, "research"),
        (1, "hypothesize"),
        (1, "design"),
        (1, "run"),
        (1, "run"),
        (1, "analyze"),
        (2, "hypothesize"),
    ]


def test_state_json_has_exactly_the_documented_shape(project, team, backend):
    advance(project, team, backend)

    state = json.loads((project.root / "state.json").read_text())

    assert set(state) == STATE_KEYS and state["schema"] == 1
    assert set(state["checkpoint"]) == {"round", "phase", "at", "summary", "pending"}
    assert set(state["spent"]) == {"model_usd", "experiment_seconds", "experiment_usd"}
    for record in state["phases"]:
        assert set(record) == PHASE_KEYS
        assert record["status"] == "completed" and record["attempts"] == 1
        assert record["error"] is None and record["instruction"] is None
        assert record["started"] <= record["finished"] and record["seconds"] >= 0
    intake, research = state["phases"]
    assert intake["produced"] == [
        "wiki/questions/Q1-why-4000-steps.md",
        "wiki/topics/what-the-notes-say.md",
    ]
    assert len(research["produced"]) == 4
    usage = json.loads((project.root / "usage.json").read_text())
    assert {row["label"] for row in usage["by_label"]} == {"intake.intake", "research.scout"}
    assert usage["estimated_cost_usd"] == pytest.approx(state["spent"]["model_usd"], abs=1e-4)


def test_every_phase_announces_itself_in_the_event_stream(project, team, backend):
    advance(project, team, backend)

    stream = [(e["type"], e["phase"], e["round"]) for e in project.events()]
    phases = [
        item for item in stream if item[0] in ("phase.started", "phase.finished", "checkpoint")
    ]
    assert phases == [
        ("phase.started", "intake", 1),
        ("phase.finished", "intake", 1),
        ("phase.started", "research", 1),
        ("phase.finished", "research", 1),
        ("checkpoint", "research", 1),
    ]
    finished = events(project, "phase.finished")[-1]
    assert finished["agent"] == "scout" and finished["summary"].startswith("The scout recorded")
    # Pages written during a phase carry it, because the state named it first.
    created = [e for e in events(project, "page.created") if e["agent"] == "scout"]
    assert created and all(e["phase"] == "research" and e["round"] == 1 for e in created)
    assert any("against Semantic Scholar" in line for line in activity(project))
    assert events(project, "checkpoint")[0]["data"]["next"] == "hypothesize"


def test_a_phase_can_be_named_and_an_unknown_one_is_refused(project, team, backend):
    with pytest.raises(EngineError, match="not a phase"):
        advance(project, team, backend, phase="publish")

    state = advance(project, team, backend, phase="research", instruction="Only the classics.")

    assert PHASES == ("intake", "research", "hypothesize", "design", "run", "analyze")
    assert [p["phase"] for p in state["phases"]] == ["research"]
    assert state["phases"][0]["instruction"] == "Only the classics."
    assert "<instruction>\nOnly the classics.\n</instruction>" in team.texts["research.scout"][0]
    assert team.calls == ["research.scout"]


# ---------------------------------------------------------------------------
# Interruptions
# ---------------------------------------------------------------------------


def run_round(project: Project, team: Team, backend: FakeBackend) -> list[dict]:
    """A whole round with the person accepting everything; an interrupted call is run again.

    Returns the states read after each interruption.
    """
    interruptions: list[dict] = []

    def step(**options) -> None:
        try:
            advance(project, team, backend, **options)
        except KeyboardInterrupt:
            interruptions.append(read_state(project))
            advance(project, team, backend, **options)

    step()
    accept_all(project)
    step()
    accept_all(project)
    step()
    accept_all(project)
    approve_all(project)
    step()
    return interruptions


@pytest.mark.parametrize(
    "label",
    [
        "intake.intake",
        "research.scout",
        "hypothesize.theorist",
        "design.designer",
        "analyze.analyst",
    ],
)
def test_an_interrupted_phase_resumes_without_writing_anything_twice(tmp_path, label):
    notes = tmp_path / "notes.md"
    notes.write_text("Warmup of 4000 steps was used in the first runs.\n")
    undisturbed = make_project(tmp_path, files=[notes], name="undisturbed")
    run_round(undisturbed, Team(), FakeBackend())
    project = make_project(tmp_path, files=[notes], name="interrupted")
    team = Team()
    team.interrupt = label
    phase = label.split(".")[0]

    (stopped,) = run_round(project, team, FakeBackend())

    # Ctrl-C was recorded before it was passed on, and the same call resumed the phase.
    assert (stopped["status"], stopped["phase"], stopped["next_phase"]) == ("stopped", phase, phase)
    assert stopped["phases"][-1]["status"] == "interrupted" and stopped["worker"] is None
    # What it had written by then is on record, and stays there.
    written = stopped["phases"][-1]["produced"]
    assert written and all((project.root / path).is_file() for path in written)
    final = read_state(project)
    records = [p for p in final["phases"] if p["phase"] == phase]
    assert [(p["status"], p["attempts"]) for p in records] == [("completed", 2)]
    assert set(written) < set(records[0]["produced"])
    assert team.calls.count(label) == 2
    assert len(set(team.sessions[i] for i, name in enumerate(team.calls) if name == label)) == 1
    assert wiki(project) == wiki(undisturbed)
    assert len(project.proposals(review=None)) == len(undisturbed.proposals(review=None))
    # The page written before the interruption is listed once, with what came after it.
    assert len(set(records[0]["produced"])) == len(records[0]["produced"])
    assert final["status"] == "checkpoint" and final["phase"] == "analyze"


def test_a_stop_request_ends_the_phase_at_its_next_model_call(project, team, backend):
    engine = make_engine(project, team, backend)
    team.during["intake.intake"] = engine.stop

    state = engine.advance()

    # The session had written one page; its next step (a tool call, which is where a
    # turn goes back to the model) was not taken, and no further session was opened.
    assert (state["status"], state["phase"], state["next_phase"]) == ("stopped", "intake", "intake")
    assert [(p["phase"], p["status"]) for p in state["phases"]] == [("intake", "interrupted")]
    assert state["phases"][0]["produced"] == ["wiki/topics/what-the-notes-say.md"]
    assert team.calls == ["intake.intake"]
    assert [page.kind for page in project.pages()] == ["idea", "topic"]
    assert any("was stopped; running it again resumes it" in line for line in activity(project))

    del team.during["intake.intake"]
    state = engine.advance()
    assert (state["status"], state["phase"]) == ("checkpoint", "research")
    assert [(p["phase"], p["status"], p["attempts"]) for p in state["phases"]] == [
        ("intake", "completed", 2),
        ("research", "completed", 1),
    ]
    # The resumed session found its page and wrote only what was missing.
    assert sorted(page.kind for page in project.pages(review="proposed")) == [
        "question",
        "question",
        "topic",
        "topic",
    ]


def test_a_stop_request_between_two_sessions_keeps_the_next_one_from_opening(
    project, team, backend
):
    engine = make_engine(project, team, backend)
    original = team.intake_intake

    def intake_then_stop(request):
        reply = original(request)
        engine.stop()  # after the session's last tool call: its turn ends by itself
        return reply

    team.intake_intake = intake_then_stop

    state = engine.advance()

    assert [(p["phase"], p["status"]) for p in state["phases"]] == [
        ("intake", "completed"),
        ("research", "interrupted"),
    ]
    assert (state["status"], state["next_phase"]) == ("stopped", "research")
    assert team.calls == ["intake.intake"]

    team.intake_intake = original
    state = engine.advance()
    assert (state["status"], state["phase"]) == ("checkpoint", "research")
    assert [(p["phase"], p["status"], p["attempts"]) for p in state["phases"]] == [
        ("intake", "completed", 1),
        ("research", "completed", 2),
    ]
    # Two calls of one engine: what the first one spent is counted once.
    usage = json.loads((project.root / "usage.json").read_text())
    assert usage["total"]["requests"] == 2
    assert [row["label"] for row in usage["by_label"]] == ["intake.intake", "research.scout"]


def test_sigterm_is_a_stop_request_while_the_worker_runs(project, team, backend):
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    team.during["intake.intake"] = lambda: signal.raise_signal(signal.SIGTERM)

    state = advance(project, team, backend)

    assert (state["status"], state["phase"]) == ("stopped", "intake")
    assert state["phases"][-1]["status"] == "interrupted"
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_a_program_that_handles_sigterm_itself_keeps_its_handler(project, team, backend):
    def interrupt(signum, frame):
        raise KeyboardInterrupt  # what the command line does

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        team.during["intake.intake"] = lambda: signal.raise_signal(signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            advance(project, team, backend)
        assert signal.getsignal(signal.SIGTERM) is interrupt
    finally:
        signal.signal(signal.SIGTERM, previous)

    state = read_state(project)
    assert (state["status"], state["phase"], state["next_phase"]) == ("stopped", "intake", "intake")
    assert state["phases"][-1]["status"] == "interrupted" and state["worker"] is None


def test_a_stop_request_cancels_the_running_experiment_and_the_run_resumes(project, team, backend):
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    engine = make_engine(project, team, backend)
    backend.before = engine.stop

    state = engine.advance()

    assert (state["status"], state["phase"], state["next_phase"]) == ("stopped", "run", "run")
    assert state["phases"][-1]["status"] == "interrupted"
    assert "analyze.analyst" not in team.calls
    assert project.page("E1").state in ("cancelled", "queued", "approved")

    backend.before = lambda: None
    if project.page("E1").state == "cancelled":
        approve(project, "E1")  # the person decides to run it after all
    state = advance(project, team, backend)

    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    assert project.page("E1").state == "succeeded"
    assert [(p["phase"], p["status"]) for p in state["phases"][-2:]] == [
        ("run", "completed"),
        ("analyze", "completed"),
    ]


def test_a_worker_that_died_left_its_phase_interrupted(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    state = read_state(project)
    state["phases"].append(
        {**state["phases"][-1], "phase": "hypothesize", "status": "running", "produced": []}
        | {"started": "2026-11-15T10:00:00Z", "finished": None}
    )
    state.update(
        status="running",
        phase="hypothesize",
        next_phase="hypothesize",
        checkpoint=None,
        worker={
            "pid": 2**22 + 12345,
            "host": os.uname().nodename,
            "started": "2026-11-15T10:00:00Z",
            "heartbeat": "2026-11-15T10:03:00Z",
        },
    )
    (project.root / "state.json").write_text(json.dumps(state))
    assert not worker_alive(read_state(project))

    state = advance(project, team, backend)

    assert (state["status"], state["phase"]) == ("checkpoint", "hypothesize")
    record = state["phases"][-1]
    assert (record["phase"], record["status"], record["attempts"]) == (
        "hypothesize",
        "completed",
        2,
    )
    assert record["started"] == "2026-11-15T10:00:00Z"
    assert len(project.pages(kind="hypothesis")) == 2


@pytest.mark.parametrize("marked_by_the_server", [False, True])
def test_a_phase_whose_worker_was_killed_is_resumed_by_the_next_one(
    project, team, backend, marked_by_the_server
):
    advance(project, team, backend)
    accept_all(project)

    class Killed(BaseException):
        """Nothing the engine handles: the process is gone before it can write anything."""

    def die() -> None:
        raise Killed

    team.during["hypothesize.theorist"] = die
    with pytest.raises(Killed):
        advance(project, team, backend)
    left = read_state(project)
    assert (left["status"], left["phase"], left["next_phase"]) == (
        "running",
        "hypothesize",
        "hypothesize",
    )
    assert left["phases"][-1]["status"] == "running" and not worker_alive(left)
    assert len(project.pages(kind="hypothesis")) == 1
    if marked_by_the_server:
        # What the web server does when it sees its worker die.
        left["phases"][-1].update(status="interrupted", error="The worker was killed.")
        left.update(status="stopped", worker=None, error="The worker was killed.")
        (project.root / "state.json").write_text(json.dumps(left))

    del team.during["hypothesize.theorist"]
    state = advance(project, team, backend)

    assert (state["status"], state["phase"], state["error"]) == ("checkpoint", "hypothesize", None)
    records = [p for p in state["phases"] if p["phase"] == "hypothesize"]
    assert [(p["status"], p["attempts"], p["error"]) for p in records] == [("completed", 2, None)]
    assert len(records[0]["produced"]) == 2 == len(project.pages(kind="hypothesis"))
    assert team.sessions[-2:] == ["r1-hypothesize-theorist"] * 2


def test_an_attempt_is_on_file_while_it_is_made(project, team, backend):
    seen: list[tuple[str, int]] = []

    def look() -> None:
        record = read_state(project)["phases"][-1]
        seen.append((record["status"], record["attempts"]))

    team.during["intake.intake"] = look

    advance(project, team, backend)

    # A worker that is killed writes nothing more: what it was doing must be there already.
    assert seen == [("running", 1)]


def test_a_worker_that_died_between_two_phases_did_not_leave_one_interrupted(
    project, team, backend
):
    state = advance(project, team, backend, phase="intake")
    assert (state["phase"], state["next_phase"]) == ("research", "hypothesize")
    # As the file stood when intake had completed and research had not begun.
    intake = state["phases"][0]
    state.update(status="running", phase="intake", next_phase="research", checkpoint=None)
    state["phases"] = [intake]
    state["worker"] = {"pid": 2**22 + 12345, "host": "gone", "started": "", "heartbeat": ""}
    (project.root / "state.json").write_text(json.dumps(state))
    calls = len(team.calls)

    state = advance(project, team, backend)

    assert team.calls[calls:] == ["research.scout"]
    assert [(p["phase"], p["status"]) for p in state["phases"]] == [
        ("intake", "completed"),
        ("research", "completed"),
    ]


DEAD_WORKER = {"pid": 2**22 + 12345, "host": "gone", "started": "", "heartbeat": ""}


@pytest.mark.parametrize("marked_by_the_server", [False, True])
def test_a_worker_that_died_before_writing_its_checkpoint_still_owes_the_pause(
    project, team, backend, marked_by_the_server
):
    state = advance(project, team, backend)
    assert (state["status"], state["phase"]) == ("checkpoint", "research")
    # As the file stood when research had completed and its checkpoint was not written.
    state.update(status="running", checkpoint=None, worker=DEAD_WORKER)
    if marked_by_the_server:
        state.update(status="stopped", worker=None, error="The worker was killed.")
    (project.root / "state.json").write_text(json.dumps(state))
    calls, checkpoints = len(team.calls), len(events(project, "checkpoint"))

    state = advance(project, team, backend)

    # No agent ran: the person is first given the pause the dead worker owed them.
    assert team.calls[calls:] == []
    assert (state["status"], state["phase"], state["next_phase"], state["error"]) == (
        "checkpoint",
        "research",
        "hypothesize",
        None,
    )
    summary = state["checkpoint"]["summary"]
    assert summary.startswith("The worker stopped right after the research phase had finished.")
    assert "2 questions and 2 topics await your review." in summary
    assert state["checkpoint"]["pending"] == {"topic": 2, "question": 2}
    assert len(events(project, "checkpoint")) == checkpoints + 1
    assert [(p["phase"], p["status"]) for p in state["phases"]] == [
        ("intake", "completed"),
        ("research", "completed"),
    ]

    accept_all(project)
    state = advance(project, team, backend)
    assert team.calls[calls:] == ["hypothesize.theorist"]
    assert (state["status"], state["phase"]) == ("checkpoint", "hypothesize")


def test_a_state_file_that_does_not_say_what_is_next_goes_on_from_its_records(
    project, team, backend
):
    advance(project, team, backend)
    accept_all(project)
    state = read_state(project)
    state["next_phase"] = None
    # A record without the fields a resumed phase counts on, and one that is no record.
    state["phases"] += ["scribble", {"round": 1, "phase": "hypothesize", "status": "interrupted"}]
    state.update(status="stopped", phase="hypothesize", checkpoint=None)
    (project.root / "state.json").write_text(json.dumps(state))
    calls = len(team.calls)

    state = advance(project, team, backend)

    # Not intake a second time: the phase the last record left unfinished.
    assert team.calls[calls:] == ["hypothesize.theorist"]
    assert [(p["phase"], p["status"]) for p in state["phases"]] == [
        ("intake", "completed"),
        ("research", "completed"),
        ("hypothesize", "completed"),
    ]
    record = state["phases"][-1]
    assert set(record) == PHASE_KEYS and record["attempts"] == 1 and len(record["produced"]) == 2

    # And after a completed one, the phase that follows it.
    state["next_phase"] = "publish"
    (project.root / "state.json").write_text(json.dumps(state))
    accept_all(project)
    state = advance(project, team, backend)
    assert team.calls[calls:] == ["hypothesize.theorist", "design.designer"]


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_a_failed_phase_is_tried_once_more(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    team.failures["hypothesize.theorist"] = 1

    state = advance(project, team, backend)

    record = state["phases"][-1]
    assert (state["status"], record["status"], record["attempts"]) == ("checkpoint", "completed", 2)
    assert state["error"] is None and record["error"] is None
    assert any(
        "failed (“the provider had a bad moment”); trying it once more" in line
        for line in activity(project)
    )
    assert events(project, "phase.failed") == []


def test_a_phase_that_fails_twice_stops_the_engine_with_the_error(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    team.failures["hypothesize.theorist"] = 2

    state = advance(project, team, backend)

    assert (state["status"], state["phase"], state["next_phase"]) == (
        "failed",
        "hypothesize",
        "hypothesize",
    )
    assert state["error"] == "RuntimeError: the provider had a bad moment"
    record = state["phases"][-1]
    assert (record["status"], record["attempts"], record["error"]) == (
        "failed",
        2,
        "RuntimeError: the provider had a bad moment",
    )
    (failed,) = events(project, "phase.failed")
    assert "failed twice: “the provider had a bad moment”. Run it again" in failed["summary"]
    assert failed["phase"] == "hypothesize" and state["worker"] is None
    assert events(project, "checkpoint")[-1]["phase"] == "research"

    # The same call picks the phase up once the cause is gone.
    state = advance(project, team, backend)
    record = state["phases"][-1]
    assert (state["status"], state["error"]) == ("checkpoint", None)
    assert (record["status"], record["attempts"], record["error"]) == ("completed", 3, None)
    assert len([p for p in state["phases"] if p["phase"] == "hypothesize"]) == 1


def test_a_failure_that_trying_again_cannot_help_is_not_retried(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    team.failures["hypothesize.theorist"] = 5
    team.error = LLMConfigError("the API key was revoked")

    state = advance(project, team, backend)

    assert state["status"] == "failed" and state["phases"][-1]["attempts"] == 1
    assert team.calls.count("hypothesize.theorist") == 1


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_a_spent_model_budget_stops_at_a_checkpoint_that_says_how_to_go_on(tmp_path, team, backend):
    project = make_project(tmp_path, settings={"limits": {"model_usd": 0.00001}})

    state = advance(project, team, backend)

    # Intake ran; before research opened its session the budget was found used up.
    assert team.calls == ["intake.intake"]
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "research",
        "research",
    )
    assert state["phases"][-1]["status"] == "interrupted"
    summary = state["checkpoint"]["summary"]
    assert "The model budget is used up" in summary and "settings.limits.model_usd" in summary
    (warning,) = events(project, "budget.warning")
    assert warning["data"]["limit"] == "model_usd" and warning["data"]["allowed"] == 0.00001
    assert warning["data"]["spent"] > 0.00001 and warning["phase"] == "research"

    project.settings.update({"limits": {"model_usd": 25}})
    project.save_settings()
    state = advance(project, team, backend)
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "research",
        "hypothesize",
    )
    assert state["phases"][-1]["status"] == "completed"


def test_a_turn_that_passes_the_model_budget_is_cut_short_at_a_checkpoint_that_says_so(
    tmp_path, team, backend
):
    project = make_project(tmp_path, settings={"limits": {"model_usd": 1.0}})
    original = team.research_scout

    def scout(request: FakeRequest) -> str:
        # What a provider's client reports after a model request: the first one of this
        # turn alone costs more than the project allows in all.
        request._emit({"type": "usage", "output_tokens": 100_000, "requests": 1})
        return original(request)

    team.research_scout = scout

    state = advance(project, team, backend)

    # The turn ended at its next model call: the scout had no paper verified yet.
    assert team.calls == ["intake.intake", "research.scout"]
    assert project.pages(kind="source") == []
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "research",
        "research",
    )
    record = state["phases"][-1]
    assert (record["phase"], record["status"]) == ("research", "interrupted")
    summary = state["checkpoint"]["summary"]
    assert "The model budget ran out in the middle of the scout's work" in summary
    assert "$4.00 of $1.00 is spent" in summary and "settings.limits.model_usd" in summary
    assert summary.endswith("(it resumes where it stopped).")
    (warning,) = events(project, "budget.warning")
    assert warning["phase"] == "research" and warning["data"]["limit"] == "model_usd"
    assert warning["data"]["spent"] > 4 and warning["data"]["allowed"] == 1.0
    assert "was stopped" not in " ".join(activity(project))

    # With a higher limit the same session goes on where it stopped.
    team.research_scout = original
    project.settings.update({"limits": {"model_usd": 25}})
    project.save_settings()
    state = advance(project, team, backend)
    record = state["phases"][-1]
    assert (state["status"], state["next_phase"]) == ("checkpoint", "hypothesize")
    assert (record["phase"], record["status"], record["attempts"]) == ("research", "completed", 2)
    assert team.sessions[-2:] == ["r1-research-scout"] * 2
    assert len(project.pages(kind="source")) == 2


def test_the_budget_ends_a_turn_of_the_anthropic_client_before_its_next_request(
    tmp_path, monkeypatch
):
    # The real client against a scripted Messages API: it reports the usage of every
    # request, and runs the tools a response asks for before it sends the next one.
    from paper_orchestra.llm.anthropic_client import AnthropicClient
    from test_llm_anthropic import FakeAPI, msg, text, tool_use

    api = FakeAPI()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-key-the-transport-is-scripted")
    client = AnthropicClient(http_client=api.http_client(), max_retries=0)
    project = make_project(tmp_path, settings={"limits": {"model_usd": 1.0}})
    costly = {"input_tokens": 1000, "output_tokens": 100_000}
    api.script = [
        msg([text("Nothing to add.")]),  # intake
        msg([tool_use("tu_1", "wiki_search", {"query": "warmup"})], "tool_use", usage=costly),
        msg([text("A request that is never sent.")]),
    ]
    engine = Engine(
        project, compute=FakeBackend(), client_for=lambda _: client, scholar=make_scholar()
    )

    state = engine.advance()

    # The scout's first request passed the limit; the tool it asked for was not run and
    # no second request went out.
    assert len(api.requests) == 2 and len(api.script) == 1
    assert not any("searched the wiki" in line for line in activity(project))
    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "research",
        "research",
    )
    assert state["phases"][-1]["status"] == "interrupted"
    assert "ran out in the middle of the scout's work" in state["checkpoint"]["summary"]
    (warning,) = events(project, "budget.warning")
    assert warning["data"]["limit"] == "model_usd" and warning["data"]["spent"] > 4
    # What the turn used is in the ledger, as the usage of any turn that ended.
    assert state["spent"]["model_usd"] == pytest.approx(warning["data"]["spent"], abs=1e-3)


@pytest.mark.parametrize("cause", ["budget", "stop"])
def test_a_paused_turn_of_the_anthropic_client_is_not_continued_once_it_must_end(
    tmp_path, monkeypatch, cause
):
    # A server tool loop that pauses (the scout's web searches do) is continued by the
    # client with a further request, and no tool of the session is called in between.
    import httpx2

    from paper_orchestra.llm.anthropic_client import AnthropicClient
    from test_llm_anthropic import SSE_HEADERS, FakeAPI, msg, sse, text

    api = FakeAPI()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-key-the-transport-is-scripted")
    client = AnthropicClient(http_client=api.http_client(), max_retries=0)
    limit = 1.0 if cause == "budget" else 100.0
    project = make_project(tmp_path, settings={"limits": {"model_usd": limit}})
    engine = Engine(
        project, compute=FakeBackend(), client_for=lambda _: client, scholar=make_scholar()
    )
    costly = {"input_tokens": 1000, "output_tokens": 100_000}

    def paused(request: httpx2.Request) -> httpx2.Response:
        if cause == "stop":
            engine.stop()  # the person presses Stop while the request is out
        message = msg([text("Searching.")], "pause_turn", usage=costly)
        return httpx2.Response(200, content=sse(message), headers=SSE_HEADERS)

    api.script = [
        msg([text("Nothing to add.")]),  # intake
        paused,
        msg([text("Still searching.")], "pause_turn", usage=costly),
        msg([text("Requests that are never sent.")], usage=costly),
    ]

    state = engine.advance()

    # Intake's request and the scout's first: its continuation did not go out.
    assert len(api.requests) == 2 and len(api.script) == 2
    record = state["phases"][-1]
    assert (record["phase"], record["status"], state["next_phase"]) == (
        "research",
        "interrupted",
        "research",
    )
    assert 4 < state["spent"]["model_usd"] < 5  # one request of the scout, not three
    if cause == "budget":
        assert state["status"] == "checkpoint"
        assert "ran out in the middle of the scout's work" in state["checkpoint"]["summary"]
        assert len(events(project, "budget.warning")) == 1
    else:
        assert state["status"] == "stopped" and events(project, "budget.warning") == []
        assert "The research phase was stopped; running it again resumes it." in activity(project)


def test_a_phase_that_ends_past_the_model_budget_says_so_at_its_checkpoint(tmp_path, team, backend):
    project = make_project(tmp_path, settings={"limits": {"model_usd": 1.0}})
    original = team.research_scout
    # A client that reports a turn's usage when the turn ends: 100,000 output tokens.
    team.research_scout = lambda request: original(request) + "x" * 400_000

    state = advance(project, team, backend)

    assert (state["status"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        "research",
        "hypothesize",
    )
    assert state["phases"][-1]["status"] == "completed" and state["spent"]["model_usd"] > 4
    summary = state["checkpoint"]["summary"]
    assert "The model budget is used up: $4.00 of $1.00 is spent." in summary
    assert "settings.limits.model_usd" in summary
    assert "2 questions and 2 topics await your review." in summary
    (warning,) = events(project, "budget.warning")
    assert warning["phase"] == "research" and warning["data"]["limit"] == "model_usd"

    # Nothing more starts until the limit is raised, as before.
    accept_all(project)
    state = advance(project, team, backend)
    assert state["phases"][-1]["status"] == "interrupted"
    assert "hypothesize.theorist" not in team.calls


def two_approved_experiments(
    project: Project, team: Team, backend: FakeBackend, *, name: str = "local"
) -> None:
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    assert [page.id for page in project.pages(kind="experiment")] == ["E1", "E2"]
    for page in project.pages(kind="experiment"):
        experiments.approve(project, page.id, backend=name, backends={name: backend})


def test_the_time_limit_is_a_cap_on_one_run_and_never_ends_the_run_phase(project, team, backend):
    two_approved_experiments(project, team, backend)
    # Less than the first run takes (2.5 s, as the stand-in reports it): were this a
    # total, the second would not start.  It is a cap on each run.
    project.settings.update({"limits": {"experiment_seconds": 2}})
    project.save_settings()

    state = advance(project, team, backend)

    assert len(backend.jobs) == 2
    assert (project.page("E1").state, project.page("E2").state) == ("succeeded", "succeeded")
    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    assert events(project, "budget.warning") == []
    # The time used is on record for the person, and is no budget.
    assert state["spent"]["experiment_seconds"] == 5.0
    # The cap reached the runs themselves: the pages asked for 300 seconds.
    assert [spec.timeout for spec in backend.specs] == [2.0, 2.0]


def test_the_experiment_cost_budget_counts_what_is_spent_and_what_a_run_is_estimated_at(
    project, team
):
    backend = FakeBackend(cost_usd=3.0)
    two_approved_experiments(project, team, backend)
    project.update_page("E2", author="human", fields={"est_usd": 2.5})

    state = advance(project, team, backend)

    assert len(backend.jobs) == 1 and state["spent"]["experiment_usd"] == 3.0
    assert (project.page("E1").state, project.page("E2").state) == ("succeeded", "approved")
    assert (state["status"], state["phase"], state["next_phase"]) == ("checkpoint", "run", "run")
    summary = state["checkpoint"]["summary"]
    assert "The experiment cost budget does not cover E2" in summary
    assert "$3.00 of $5.00 is spent and E2 is estimated at $2.50" in summary
    assert "settings.limits.experiment_usd" in summary
    assert "continue with the analyze phase" in summary
    assert events(project, "budget.warning")[0]["data"]["limit"] == "experiment_usd"

    # What ran can be analysed without raising the limit.
    state = advance(project, team, backend, phase="analyze")
    assert state["status"] == "checkpoint" and project.page("F1").relations["based_on"]


def test_a_run_is_given_what_is_left_of_the_cost_budget(project, team, monkeypatch):
    backend = FakeBackend(cost_usd=3.0)
    two_approved_experiments(project, team, backend)
    given: list[dict] = []
    real_run = experiments.run

    def run(project, ref, **options):
        given.append(dict(options["limits"]))
        return real_run(project, ref, **options)

    monkeypatch.setattr(experiments, "run", run)

    advance(project, team, backend)

    # The limit is the project's total; a run is checked against what remains of it.
    assert [limits["experiment_usd"] for limits in given] == [5.0, 2.0]


def test_a_run_whose_worst_case_would_pass_the_cost_budget_is_refused_with_the_numbers(
    project, team
):
    from paper_orchestra.compute.modal_backend import estimate_cost

    backend = FakeBackend(cost_usd=3.0)
    backend.name = "modal"
    two_approved_experiments(project, team, backend, name="modal")
    worst = estimate_cost(300.0, gpu=None, cpu=2.0, memory_mb=4096)
    assert worst and worst > 0.002
    # Enough for both when nothing is spent; after E1's 3.00 too little is left for E2.
    limit = round(3.0 + worst / 2, 4)
    project.settings.update({"limits": {"experiment_usd": limit}})
    project.save_settings()

    state = advance(project, team, backend)

    assert len(backend.jobs) == 1 and state["spent"]["experiment_usd"] == 3.0
    (refused,) = [run for run in experiments.runs(project, "E2")]
    assert (refused["status"], refused["error_kind"]) == ("failed", "budget")
    summary = state["checkpoint"]["summary"]
    assert "1 experiment ran: E1 succeeded." in summary
    assert "E2 was not run: Run to its time limit of 300 s it could cost about" in summary
    assert f"left of the project's ${limit:.2f} for experiments, of which $3.00 is spent" in summary
    assert events(project, "budget.warning")


class Reattachable(FakeBackend):
    """A backend whose jobs outlive the worker that started them, as Modal's do."""

    name = "modal"

    def __init__(self) -> None:
        super().__init__()
        self.attached: list[str] = []

    def describe(self) -> dict:
        described = super().describe()
        described["jobs"]["reattach"] = True
        return described

    def attach(
        self, handle, *, run_dir, spec=None, started=None, on_log=None, should_stop=None, mask=()
    ) -> JobResult:
        self.attached.append(handle)
        run_dir = Path(run_dir)
        (run_dir / "log.txt").write_text("measuring the variance of early updates\n")
        (run_dir / "metrics.json").write_text('{"variance_with_warmup": 0.42}')
        return JobResult(
            returncode=0,
            seconds=7.0,
            cost_usd=0.75,
            log_path=run_dir / "log.txt",
            outputs_dir=run_dir,
            backend=self.name,
            handle=handle,
        )


def test_a_job_picked_up_after_its_worker_died_is_the_projects_spending_too(project, team):
    backend = Reattachable()
    to_design_checkpoint(project, team, backend)
    experiments.approve(project, "E1", backend="modal", backends={"modal": backend})
    # What a worker that was killed during the run left behind: the run still reads as
    # running, and the handle of its remote job is on record.
    run_dir = project.experiment_dir("E1") / "runs" / "1"
    run_dir.mkdir(parents=True)
    left = {"experiment": "E1", "number": 1, "status": "running", "backend": "modal"}
    left.update(handle="sb-123", started="2026-11-15T10:00:00Z")
    (run_dir / "run.json").write_text(json.dumps(left))
    project.set_state("E1", "queued", by="system")
    project.set_state("E1", "running", by="system")

    state = advance(project, team, backend)

    assert backend.attached == ["sb-123"] and backend.jobs == []
    assert project.page("E1").state == "succeeded"
    assert [run["cost_usd"] for run in experiments.runs(project, "E1")] == [0.75]
    # It cost what it cost, whoever was there when it ended.
    assert state["spent"]["experiment_usd"] == 0.75
    assert state["spent"]["experiment_seconds"] == 7.0
    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    assert project.page("F1").relations["based_on"] == ["results/E1-run1"]


def test_what_the_runs_cost_is_read_from_their_records(project, team):
    backend = FakeBackend(cost_usd=1.25)
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    # A run that ended while no engine was looking: cancelled from the web app, or
    # finished a moment before its worker was killed.
    experiments.run(project, "E1", compute=backend)
    assert read_state(project)["spent"]["experiment_usd"] == 0

    # Whatever the next worker is started for, it finds out first.
    state = advance(project, team, backend, phase="design")
    assert state["spent"]["experiment_usd"] == 1.25
    assert "for all runs together: $3.75 of $5.00" in team.texts["design.designer"][-1]

    state = advance(project, team, backend)

    assert (state["phase"], state["spent"]["experiment_usd"]) == ("analyze", 1.25)
    assert state["spent"]["experiment_seconds"] == 2.5
    # However many workers come and go, a run is counted once.
    project.review("F1", "accept")
    state = advance(project, team, backend)
    assert (state["round"], state["spent"]["experiment_usd"]) == (2, 1.25)
    assert state["spent"]["experiment_seconds"] == 2.5


def test_a_model_without_a_known_price_is_named_at_the_checkpoint(tmp_path, team, backend):
    project = make_project(tmp_path, settings={"models": {"research": "mystery-model-1"}})

    state = advance(project, team, backend)

    usage = json.loads((project.root / "usage.json").read_text())
    assert usage["unpriced_models"] == ["mystery-model-1"] and state["spent"]["model_usd"] == 0
    # A budget that cannot count is no limit, and the person is told so.
    assert (
        "No price is known for mystery-model-1, so what it cost is not part of the model "
        "spending shown, and settings.limits.model_usd does not limit it."
    ) in state["checkpoint"]["summary"]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def test_open_requests_go_to_the_phase_that_runs_and_are_closed_when_it_completes(
    project, team, backend
):
    advance(project, team, backend)
    accept_all(project)
    request = project.add_request("Consider a batch size of 8 as well", target="Q1")
    team.failures["hypothesize.theorist"] = 2

    advance(project, team, backend)
    # A phase that did not complete has not handled anything.
    assert [r.id for r in project.requests()] == [request.id]

    advance(project, team, backend)
    told = team.texts["hypothesize.theorist"][-1]
    assert (
        "- Consider a batch size of 8 as well (about wiki/questions/Q1-why-4000-steps.md)" in told
    )
    assert project.requests() == []
    (done,) = project.requests(state="done")
    assert done.handled_in == {"round": 1, "phase": "hypothesize"}


def test_a_request_for_more_literature_waits_for_the_research_phase(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    asked = project.add_request("Please look up prior work on label smoothing")

    # Hypotheses were next; the request sends the scout out first.
    state = advance(project, team, backend)
    assert (state["phase"], state["next_phase"], state["round"]) == ("research", "hypothesize", 1)
    assert "label smoothing" in team.texts["research.scout"][-1]
    assert project.requests(state="done")[0].id == asked.id
    # A second run of the phase is a session of its own, not a resumed one.
    assert team.sessions[-2:] == ["r1-research-scout", "r1-research-2-scout"]

    state = advance(project, team, backend)
    assert state["phase"] == "hypothesize"


def test_a_request_is_never_kept_from_the_phase_that_runs(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    project.review("H1", "accept")
    project.review("H2", "reject", reason="Too vague to test")
    # Meant for the designer; "papers" also reads as a wish for more of the literature.
    meant_for_design = project.add_request("Set E1 up the way the two papers did")
    plain = project.add_request("Keep every run under a minute")
    withdrawn = project.add_request("Use a batch size of 8")
    team.during["design.designer"] = lambda: project.delete_request(withdrawn.id)

    state = advance(project, team, backend)

    told = team.texts["design.designer"][0]
    assert "Set E1 up the way the two papers did" in told
    assert "Keep every run under a minute" in told
    assert (state["status"], state["phase"]) == ("checkpoint", "design")
    # The designer's phase closes what only it was asked; the other stays open, so that
    # the next round begins with the literature, as the words may have meant.
    assert [request.id for request in project.requests()] == [meant_for_design.id]
    assert [request.id for request in project.requests(state="done")] == [plain.id]

    accept_all(project)
    approve_all(project)
    state = advance(project, team, backend)
    assert (state["phase"], state["next_phase"]) == ("analyze", "research")
    assert "Set E1 up the way the two papers did" in team.texts["analyze.analyst"][0]
    state = advance(project, team, backend)
    assert (state["round"], state["phase"]) == (2, "research")
    assert "Set E1 up the way the two papers did" in team.texts["research.scout"][-1]
    assert project.requests() == []
    done = {request.id: request.handled_in for request in project.requests(state="done")}
    assert done[meant_for_design.id] == {"round": 2, "phase": "research"}


def test_the_journal_can_send_the_next_round_back_to_the_literature(project, team, backend):
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    team.journal_meta = "next: research\n"

    state = advance(project, team, backend)

    assert (state["phase"], state["next_phase"]) == ("analyze", "research")
    assert state["checkpoint"]["summary"].endswith("Next: research into what is already known.")
    state = advance(project, team, backend)
    assert (state["round"], state["phase"], state["next_phase"]) == (2, "research", "hypothesize")


@pytest.mark.parametrize("decision", ["accept", "reject", "none"])
def test_only_an_accepted_journal_page_sends_a_round_back_to_the_literature(
    tmp_path, team, backend, decision
):
    # The person reviews journal pages themselves here.
    project = make_project(tmp_path, settings={"review": {"journal": "review"}})
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    team.journal_meta = "next: research\n"

    state = advance(project, team, backend)

    journal = project.page("wiki/journal/round-1.md")
    assert (journal.review, journal.meta["next"]) == ("proposed", "research")
    # What the analyst wrote is a suggestion: unreviewed, it sends the round nowhere.
    assert (state["phase"], state["next_phase"]) == ("analyze", "hypothesize")
    summary = state["checkpoint"]["summary"]
    assert "asks for another look at the literature" in summary
    assert "accept it and the next round starts with research" in summary
    assert summary.endswith("Next: proposing hypotheses from what you accepted.")

    if decision != "none":
        project.review(journal.path, decision, reason="Not now" if decision == "reject" else None)
    state = advance(project, team, backend)

    expected = "research" if decision == "accept" else "hypothesize"
    assert (state["round"], state["phase"]) == (2, expected)
    if decision == "accept":
        # Once, not for ever: after the research the round goes on to its hypotheses.
        state = advance(project, team, backend)
        assert (state["round"], state["phase"]) == (2, "hypothesize")


def test_a_journal_page_rejected_after_the_checkpoint_no_longer_sends_the_round_back(
    project, team, backend
):
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    team.journal_meta = "next: research\n"
    state = advance(project, team, backend)
    # Journal pages are accepted on creation in this project, so the checkpoint said research.
    assert (state["phase"], state["next_phase"]) == ("analyze", "research")

    project.review("wiki/journal/round-1.md", "reject", reason="The literature is covered")
    state = advance(project, team, backend)

    assert (state["round"], state["phase"]) == (2, "hypothesize")
    assert team.calls.count("research.scout") == 1


# ---------------------------------------------------------------------------
# Experiments: only what the person approved
# ---------------------------------------------------------------------------


def test_nothing_an_agent_writes_gets_an_experiment_run(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)  # the designer's page says `state: approved`

    proposed = project.page("E1")
    assert (proposed.review, proposed.state) == ("proposed", "planned")
    with pytest.raises(NotAllowed):
        project.set_state("E1", "approved", by="agent:designer")
    with pytest.raises(NotAllowed):
        project.review("E1", "accept", by="agent:designer")

    state = advance(project, team, backend)
    assert (
        backend.jobs == []
        and "must be accepted and approved first" in state["checkpoint"]["summary"]
    )

    project.review("E1", "accept")
    state = advance(project, team, backend)
    assert (
        backend.jobs == [] and "Approve the ones you want run: E1" in state["checkpoint"]["summary"]
    )


def test_a_failed_run_is_repaired_by_the_runner_and_run_again(project, team, backend):
    team.code = FAILING_CODE
    to_design_checkpoint(project, team, backend)
    approve_all(project)

    state = advance(project, team, backend)

    assert "run.runner" in team.calls
    assert project.page("E1").state == "succeeded"
    assert experiments.code_files(project, "E1")["main.py"] == GOOD_CODE
    assert [run["status"] for run in experiments.runs(project, "E1")] == ["failed", "succeeded"]
    # Both runs are the project's time, and both are results the analyst was given.
    assert state["spent"]["experiment_seconds"] == 3.0
    assert len(state["phases"][-2]["produced"]) == 2
    assert any("The runner corrected main.py of E1" in line for line in activity(project))
    usage = json.loads((project.root / "usage.json").read_text())
    assert "run.runner" in {row["label"] for row in usage["by_label"]}


def test_an_experiment_that_cannot_be_run_does_not_keep_the_others_from_running(project, team):
    backend = FakeBackend()
    two_approved_experiments(project, team, backend)

    def choose(plan):
        if plan is not None and plan.id == "E2":
            raise ComputeError("The backend chosen for E2 is not installed here.")
        return backend

    client = FakeLLM(team)
    engine = Engine(project, compute=choose, client_for=lambda _: client, scholar=make_scholar())

    state = engine.advance()

    assert (project.page("E1").state, project.page("E2").state) == ("succeeded", "approved")
    summary = state["checkpoint"]["summary"]
    assert "1 experiment ran: E1 succeeded." in summary
    assert "E2 could not be run: The backend chosen for E2 is not installed here." in summary
    assert state["phase"] == "analyze" and project.page("F1").relations["based_on"]


def refuse_changed(monkeypatch, changed: str, reason: str) -> None:
    """Have ``research.experiments.run`` refuse ``changed`` the way it refuses an experiment
    that is no longer what was approved: back to planned, with the reason, and an error."""
    real_run = experiments.run

    def run(project, ref, **options):
        if ref != changed:
            return real_run(project, ref, **options)
        project.set_state(ref, "planned", by="system", reason=f"Not run: {reason}")
        raise ProjectError(f"{ref} was not run: {reason} Approve it again.")

    monkeypatch.setattr(experiments, "run", run)


def test_an_experiment_that_changed_since_its_approval_is_named_and_the_others_run(
    project, team, monkeypatch
):
    backend = FakeBackend()
    two_approved_experiments(project, team, backend)
    refuse_changed(monkeypatch, "E1", "its plan changed: network was false and is true.")

    state = advance(project, team, backend)

    assert (project.page("E1").state, project.page("E2").state) == ("planned", "succeeded")
    assert len(backend.jobs) == 1
    summary = state["checkpoint"]["summary"]
    assert "1 experiment ran: E2 succeeded." in summary
    assert (
        "E1 was not run: its plan changed: network was false and is true. Approve it again. "
        "It is planned again."
    ) in summary
    assert state["phase"] == "analyze" and project.page("F1").relations["based_on"]


def test_code_changed_after_the_approval_is_not_run_and_the_checkpoint_says_so(project, team):
    # The same with nothing standing in: the person edits the code of E1 after approving it.
    backend = FakeBackend()
    two_approved_experiments(project, team, backend)
    experiments.write_code(project, "E1", {"main.py": GOOD_CODE + "# changed\n"}, author="human")

    state = advance(project, team, backend)

    assert (project.page("E1").state, project.page("E2").state) == ("planned", "succeeded")
    assert len(backend.jobs) == 1 and experiments.runs(project, "E1") == []
    summary = state["checkpoint"]["summary"]
    assert "1 experiment ran: E2 succeeded." in summary
    assert "E1 was not run: " in summary and "main.py" in summary
    assert "It is planned again." in summary
    assert state["phase"] == "analyze"

    # Approved again as it now stands, it runs.
    approve(project, "E1", backend)
    state = advance(project, team, backend, phase="run")
    assert project.page("E1").state == "succeeded" and len(backend.jobs) == 2


def test_a_changed_experiment_is_named_also_when_the_analysis_ends_in_a_later_call(project, team):
    backend = FakeBackend()
    two_approved_experiments(project, team, backend)
    experiments.write_code(project, "E1", {"main.py": GOOD_CODE + "# changed\n"}, author="human")
    # The run phase refuses E1 and runs E2; the analysis that follows fails twice, so the
    # call ends without a checkpoint and with it what the run phase had to say.
    team.failures["analyze.analyst"] = 2

    state = advance(project, team, backend)

    assert (state["status"], state["phase"]) == ("failed", "analyze")
    assert (project.page("E1").state, project.page("E2").state) == ("planned", "succeeded")

    state = advance(project, team, backend)

    # The checkpoint the person does get says where E1 stands, once.
    summary = state["checkpoint"]["summary"]
    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    assert summary.count("E1 is planned again: Not run: it is not what was approved") == 1
    assert "main.py" in summary and "Approve it again." in summary


def test_a_run_refused_while_it_was_being_set_up_did_not_run(project, team, monkeypatch):
    backend = FakeBackend()
    two_approved_experiments(project, team, backend)
    real_run = experiments.run
    error = (
        "The run was not started: it is not what was approved at 2026-11-16T03:56:12Z "
        "(main.py: changed after the approval). Look at the experiment as it is now and "
        "approve it again."
    )

    def run(project, ref, **options):
        if ref != "E1":
            return real_run(project, ref, **options)
        # As `research.experiments` settles a run whose code changed between its look at
        # the approval and the copy it was about to start: a record, and no job.
        project.set_state(ref, "failed", by="system", reason=error)
        return {
            "experiment": ref,
            "number": 1,
            "status": "failed",
            "backend": "local",
            "error": error,
            "error_kind": "approval",
            "cost_usd": None,
            "plan": {"mode": "script", "gpu": None},
        }

    monkeypatch.setattr(experiments, "run", run)

    state = advance(project, team, backend)

    assert len(backend.jobs) == 1 and project.page("E2").state == "succeeded"
    summary = state["checkpoint"]["summary"]
    # It is not counted among the experiments that ran, and the checkpoint says why.
    assert "1 experiment ran: E2 succeeded." in summary and "E1 failed" not in summary
    assert f"E1 was not run: {error}" in summary


def test_a_run_phase_that_only_found_a_changed_experiment_says_which_and_why(
    project, team, backend, monkeypatch
):
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    reason = "main.py was written by agent:designer after the approval."
    refuse_changed(monkeypatch, "E1", reason)

    state = advance(project, team, backend)

    assert backend.jobs == [] and project.page("E1").state == "planned"
    assert (state["status"], state["phase"], state["next_phase"]) == ("checkpoint", "run", "run")
    summary = state["checkpoint"]["summary"]
    assert summary.startswith(
        f"No experiment ran. E1 was not run: {reason} Approve it again. It is planned again."
    )

    # Looked at again later, the run phase still says why nothing is approved.
    state = advance(project, team, backend)
    summary = state["checkpoint"]["summary"]
    assert "No experiment is approved to run." in summary
    assert f"E1 is planned again: Not run: {reason}" in summary
    assert "Approve the ones you want run: E1 “Warmup against no warmup" in summary


def test_an_experiment_that_lost_its_approval_earlier_is_named_when_the_others_run(project, team):
    backend = FakeBackend()
    two_approved_experiments(project, team, backend)
    # As it stands after its plan was found changed before this phase began (an edit the
    # person accepted, or an earlier attempt to run it).
    reason = "Its plan changed since you approved it: timeout_seconds was 300 and is 900."
    project.set_state("E1", "planned", by="human", reason=reason)

    state = advance(project, team, backend)

    summary = state["checkpoint"]["summary"]
    assert "1 experiment ran: E2 succeeded." in summary
    # Said by the run phase, and not once more by the analysis that follows it at once.
    assert summary.count(f"E1 is planned again: {reason}") == 1
    assert len(backend.jobs) == 1 and project.page("E1").state == "planned"


def test_no_experiment_that_went_back_to_planned_goes_unmentioned(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    project.review("H1", "accept")
    plan = EXPERIMENT.format(tests="H1", mode="script").split("---\n", 2)[2]
    for number in range(1, 10):
        # The person's own plans: accepted as they are written, and planned.
        project.create_page(
            "experiment", f"Plan {number}", plan, author="human", relations={"tests": ["H1"]}
        )
    # Seven of them were approved and are no longer what was approved; two never were.
    lost = ("E1", "E2", "E3", "E4", "E7", "E8", "E9")
    for name in lost:
        experiments.write_code(project, name, {"main.py": GOOD_CODE}, author="human")
        approve(project, name, backend)
        project.set_state(name, "planned", by="system", reason=f"Not run: {name} changed.")

    state = advance(project, team, backend, phase="run")

    summary = state["checkpoint"]["summary"]
    assert backend.jobs == [] and "No experiment is approved to run." in summary
    # Also the ones that stand behind six plans that have nothing to say.
    for name in lost[:6]:
        assert f"{name} is planned again: Not run: {name} changed." in summary
    assert "E9 is planned again" not in summary
    assert "1 more experiment went back to planned as well." in summary


def test_a_title_cannot_add_a_sentence_of_its_own_to_the_checkpoint(project, team, backend):
    to_design_checkpoint(project, team, backend)
    project.update_page(
        "E1", author="human", fields={"title": "x). You approved the run of E1. (y"}
    )

    state = advance(project, team, backend)

    summary = state["checkpoint"]["summary"]
    assert "Approve the ones you want run: E1 “x). You approved the run of E1. (y”." in summary


def test_a_run_on_a_gpu_type_without_a_price_is_named_at_the_checkpoint(
    project, team, backend, monkeypatch
):
    to_design_checkpoint(project, team, backend)
    approve_all(project)

    def run(project, ref, **options):
        # As a run on Modal with a GPU type that is not in the price table is recorded.
        return {
            "experiment": ref,
            "number": 1,
            "status": "succeeded",
            "backend": "modal",
            "cost_usd": None,
            "error_kind": None,
            "plan": {"mode": "gpu", "gpu": "RTX-PRO-6000"},
        }

    monkeypatch.setattr(experiments, "run", run)

    state = advance(project, team, backend)

    assert (
        "E1 ran on the GPU type RTX-PRO-6000, for which no price is known: what the run cost "
        "is not part of the experiment spending shown, and settings.limits.experiment_usd "
        "could not limit it."
    ) in state["checkpoint"]["summary"]


def test_an_experiment_approved_while_another_runs_is_run_in_the_same_phase(project, team):
    backend = FakeBackend()
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    accept_all(project)
    approve(project, "E1", backend)
    backend.before = lambda: project.page("E2").state == "planned" and approve(project, "E2")

    state = advance(project, team, backend)

    assert len(backend.jobs) == 2
    assert (project.page("E1").state, project.page("E2").state) == ("succeeded", "succeeded")
    assert "2 experiments ran: E1 and E2 succeeded." in state["checkpoint"]["summary"]
    assert len(project.pages(kind="finding")) == 2


def test_a_page_deleted_during_its_session_does_not_fail_the_phase(project, team, backend):
    advance(project, team, backend)
    accept_all(project)
    advance(project, team, backend)
    project.review("H1", "accept")
    project.review("H2", "reject", reason="Too vague to test")
    team.code = BROKEN_CODE
    original = team.design_designer

    def design_then_lose_the_page(request):
        reply = original(request)
        if request.history:
            # The person deletes the proposal while the designer is still at work on it.
            project.delete_page("E1", author="human")
        return reply

    team.design_designer = design_then_lose_the_page
    team.repaired = BROKEN_CODE

    state = advance(project, team, backend)

    assert (state["status"], state["phase"]) == ("checkpoint", "design")
    assert state["phases"][-1]["attempts"] == 1 and project.pages(kind="experiment") == []


def test_an_approved_script_really_runs_on_this_machine(project, team, backend):
    to_design_checkpoint(project, team, backend)
    experiments.approve(project, "E1")
    client = FakeLLM(team)
    engine = Engine(project, client_for=lambda _: client, scholar=make_scholar())

    state = engine.advance()

    assert backend.jobs == [] and project.page("E1").state == "succeeded"
    (run,) = experiments.runs(project, "E1")
    assert (run["backend"], run["exit_code"], run["metrics"]["variance_with_warmup"]) == (
        "local",
        0,
        0.42,
    )
    result = project.page("E1-run1")
    assert "0.42" in result.body and "measuring the variance of early updates" in result.body
    assert state["spent"]["experiment_seconds"] > 0 and state["spent"]["experiment_usd"] == 0
    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    assert "unverified" not in project.page("F1").meta


def test_the_journal_of_a_round_never_replaces_a_page_that_is_there(project, team, backend):
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    own = project.create_page(
        "journal", "My notes on round 1", "Mine.", author="human", slug="round-1"
    )
    assert own.path == "wiki/journal/round-1.md"

    advance(project, team, backend)

    assert "goes to /memories/wiki/journal/round-1-2.md" in team.texts["analyze.analyst"][0]
    assert project.page("wiki/journal/round-1.md").body.strip() == "Mine."
    assert project.page("wiki/journal/round-1-2.md").author == "agent:analyst"


def test_hypotheses_follow_the_experiments_that_test_them(project, team, backend):
    to_design_checkpoint(project, team, backend)
    assert project.page("H1").state == "open"
    approve_all(project)

    advance(project, team, backend)
    assert project.page("H1").state == "testing"

    # The person turns the proposed verdict down: with nothing testing it, it is open again.
    for proposal in project.proposals():
        project.review(proposal.id, "reject", reason="One seed is not enough")
    # A hypothesis the person says is under test, with no experiment of the project's:
    # that is theirs to say, and to take back.
    own = project.create_page(
        "hypothesis", "Tested elsewhere", "## Statement\n\nx\n", author="human"
    )
    project.set_state(own.id, "testing", by="human")

    advance(project, team, backend)

    assert project.page("H1").state == "open"
    assert project.page("H1").meta["updated_by"] == "system"
    assert project.page(own.id).state == "testing"
    assert project.page(own.id).meta["updated_by"] == "human"


def test_a_hypothesis_accepted_after_its_experiment_was_approved_is_under_test_too(
    project, team, backend
):
    to_design_checkpoint(project, team, backend)
    late = project.create_page(
        "hypothesis", "A second claim", "## Statement\n\nx\n", author="agent:theorist"
    )
    project.update_page("E1", author="human", relations={"tests": ["H1", late.id]})
    approve(project, "E1")
    project.review(late.id, "accept")
    # The approval could only mark what was accepted then.
    assert (project.page("H1").state, project.page(late.id).state) == ("testing", "open")

    advance(project, team, backend, phase="design")

    assert project.page(late.id).state == "testing"
    assert project.page(late.id).meta["updated_by"] == "system"
    # So the designer is not asked for another experiment for it.
    listed = re.search(
        r"<hypotheses>\n(.*?)\n</hypotheses>", team.texts["design.designer"][-1], re.S
    )
    assert listed is None or late.id not in listed.group(1)


def test_unsupported_numbers_are_flagged_at_the_checkpoint(project, team, backend):
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    team.finding_text = "With warmup the variance was 0.42, which is 53.85 percent lower."

    state = advance(project, team, backend)

    assert project.page("F1").meta["unverified"] == ["53.85"]
    assert (
        "F1 states 1 number that no result record, source or input supports: 53.85."
        in state["checkpoint"]["summary"]
    )
    assert team.calls.count("analyze.analyst") == 3


def test_one_experiment_can_be_run_outside_the_order_of_the_phases(project, team, backend):
    to_design_checkpoint(project, team, backend)
    engine = make_engine(project, team, backend)

    with pytest.raises(EngineError, match="E1 is accepted and planned"):
        engine.run_experiment("E1")
    with pytest.raises(EngineError, match="not an experiment"):
        engine.run_experiment("H1")
    approve_all(project)
    other = FakeBackend(seconds=4.0)

    record = engine.run_experiment("E1", backend=other)

    assert record["status"] == "succeeded" and record["number"] == 1
    assert len(other.jobs) == 1 and backend.jobs == []
    state = read_state(project)
    # It was not a phase: the checkpoint stands, the time is counted, the worker is gone.
    assert (state["status"], state["phase"], state["next_phase"]) == ("checkpoint", "design", "run")
    assert state["spent"]["experiment_seconds"] == 4.0 and state["worker"] is None
    assert project.page("H1").state == "testing"

    # The run phase finds nothing left to run, but a result nobody has analysed.
    state = advance(project, team, backend)
    assert (state["status"], state["phase"]) == ("checkpoint", "analyze")
    assert state["checkpoint"]["summary"].startswith(
        "No experiment was waiting to run; 1 result of earlier runs had not been analysed yet."
    )
    assert project.page("F1").relations["based_on"] == ["results/E1-run1"]


def test_limits_can_be_replaced_for_one_run_only(project, team, backend):
    to_design_checkpoint(project, team, backend)
    approve_all(project)
    project.update_page("E1", author="human", fields={"est_usd": 2.5})
    project.settings.update({"limits": {"experiment_usd": 1}})
    project.save_settings()
    engine = make_engine(project, team, backend)

    with pytest.raises(EngineError, match="cost budget does not cover E1"):
        engine.run_experiment("E1")
    assert events(project, "budget.warning") and backend.jobs == []
    assert project.page("E1").state == "approved"
    with pytest.raises(EngineError, match="cannot be used"):
        engine.run_experiment("E1", limits={"experiment_usd": -5})

    record = engine.run_experiment("E1", limits={"experiment_usd": 10, "experiment_seconds": 60})

    assert record["status"] == "succeeded"
    assert backend.specs[0].timeout == 60.0
    assert project.settings.limits.experiment_usd == 1
    assert project.settings.limits.experiment_seconds == 900


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


def test_a_project_has_one_worker_at_a_time_and_its_heartbeat_is_kept_fresh(
    project, team, backend, monkeypatch
):
    engine = make_engine(project, team, backend)
    monkeypatch.setattr(engine, "HEARTBEAT_SECONDS", 0.02)
    beats: list[str] = []
    original = engine._beat
    monkeypatch.setattr(engine, "_beat", lambda: (original(), beats.append("beat")))
    seen: dict[str, object] = {}

    def look_around() -> None:
        # Once: were the lock ever missing, the second worker must not come back here.
        del team.during["intake.intake"]
        time.sleep(0.2)
        seen["state"] = read_state(project)
        try:
            make_engine(project, team, backend).advance()
        except WorkerBusy as exc:
            seen["busy"] = str(exc)
        try:
            make_engine(project, team, backend).run_experiment("E1")
        except WorkerBusy as exc:
            seen["busy_run"] = str(exc)

    team.during["intake.intake"] = look_around

    state = engine.advance()

    during = seen["state"]
    assert during["status"] == "running" and during["phase"] == "intake"
    # The phase in flight is on record, and is what a later worker would resume.
    assert during["next_phase"] == "intake"
    assert [(p["phase"], p["status"]) for p in during["phases"]] == [("intake", "running")]
    # The attempt is on file while it is made: a worker that is killed writes nothing more.
    assert during["phases"][0]["attempts"] == 1
    assert during["worker"]["pid"] == os.getpid()
    assert during["worker"]["host"] == os.uname().nodename
    assert during["worker"]["started"] <= during["worker"]["heartbeat"]
    assert worker_alive(during)
    assert len(beats) >= 3
    assert f"process {os.getpid()}" in seen["busy"] and "one worker at a time" in seen["busy"]
    assert "busy_run" in seen
    # The second worker changed nothing, and the first one finished its work.
    assert state["status"] == "checkpoint" and state["worker"] is None
    assert not worker_alive(state)
    assert team.calls == ["intake.intake", "research.scout"]
    assert (project.root / ".worker.lock").exists()

    # The lock went with the worker.
    accept_all(project)
    assert advance(project, team, backend)["status"] == "checkpoint"


def test_without_a_backend_of_its_own_the_engine_uses_the_one_an_approval_names(
    project, monkeypatch
):
    built: list[str] = []
    closed: list[str] = []

    class Named(FakeBackend):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            built.append(name)

        def close(self) -> None:
            closed.append(self.name)

    monkeypatch.setattr(engine_module, "get_backend", Named)
    compute = engine_module._Compute(project, None)
    pages = {}
    for name in ("planned", "approved"):
        page = project.create_page(
            "experiment", f"A {name} plan", "x", author="human", fields={"mode": "script"}
        )
        experiments.write_code(project, page.id, {"main.py": GOOD_CODE}, author="human")
        pages[name] = page.path
    experiments.approve(project, pages["approved"], backends={"local": FakeBackend()})

    def plan(mode: str, backend: str | None = None, page: str = "planned"):
        return types.SimpleNamespace(mode=mode, backend=backend, path=pages[page])

    assert compute().name == "local"  # the project's
    assert compute(plan("script")).name == "local"
    assert compute(plan("gpu")).name == "modal"  # what an approval would choose for a GPU
    # What the person chose when approving the run.
    assert compute(plan("script", "modal", "approved")).name == "modal"
    # A page that was never approved has no say in where it would run: the designer's
    # check asks the backend the project would use, whatever the page names.
    assert compute(plan("script", "modal", "planned")).name == "local"
    assert compute(types.SimpleNamespace(mode="script", backend="modal")).name == "local"
    assert built == ["local", "modal"]  # each is built once
    compute.close()
    assert sorted(closed) == ["local", "modal"]
    given = FakeBackend()
    assert engine_module._Compute(project, given)(plan("gpu")) is given
    assert engine_module._Compute(project, lambda p: given)(plan("gpu")) is given


def test_a_worker_is_alive_while_it_beats_or_its_process_exists():
    now = engine_module.parse_stamp("2026-11-15T12:00:00Z")
    host = os.uname().nodename

    def worker(pid: int, heartbeat: str, host: str = host) -> dict:
        return {"worker": {"pid": pid, "host": host, "started": heartbeat, "heartbeat": heartbeat}}

    gone = 2**22 + 12345
    assert worker_alive(worker(gone, "2026-11-15T11:59:30Z"), now=now)
    assert not worker_alive(worker(gone, "2026-11-15T11:58:00Z"), now=now)
    assert worker_alive(worker(os.getpid(), "2026-11-15T11:00:00Z"), now=now)
    # A process on another machine cannot be looked for; its heartbeat is all there is.
    assert not worker_alive(worker(os.getpid(), "2026-11-15T11:00:00Z", host="elsewhere"), now=now)
    assert not worker_alive({"worker": None}, now=now)


def test_models_that_cannot_use_the_wiki_are_refused_before_anything_is_spent(tmp_path):
    project = make_project(tmp_path, settings={"models": {"research": "gpt-5.5"}})
    plain = FakeLLM("x", capabilities=Capabilities(tools=True, memory_tool=False))
    capable = FakeLLM("x")
    engine = Engine(
        project,
        compute=FakeBackend(),
        client_for=lambda name: plain if name == "gpt-5.5" else capable,
    )

    with pytest.raises(EngineError) as refusal:
        engine.advance()

    message = str(refusal.value)
    assert "settings.models.research" in message and "gpt-5.5" in message
    assert "the memory tool" in message and "Nothing was spent" in message
    assert plain.requests == [] and capable.requests == []
    state = read_state(project)
    assert (state["status"], state["phases"], state["worker"]) == ("idle", [], None)
    assert state["spent"]["model_usd"] == 0
    (error,) = events(project, "error")
    assert "settings.models.research" in error["summary"]
    assert project.pages() == project.pages(kind="idea")


def test_the_brief_is_the_persons_and_stays_as_it_was(project, team, backend):
    run_round(project, team, backend)

    idea = project.page("wiki/idea.md")
    assert idea.body.strip() == BRIEF and idea.author == "human"
    assert project.history("wiki/idea.md") == []
