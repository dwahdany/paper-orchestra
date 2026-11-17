"""Decisions under the ``sdk`` runtime: an agent with a shell asks the person to choose
through the board, with the tool, the caps and the rules of the tool loop; it is told
what was asked and answered; and neither its shell nor its file tools answer one.

Everything runs against the scripted driver; no agent process is started.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import test_research_sdk
import test_research_sdk_engine
from paper_orchestra.research import agents, decisions, prompts
from paper_orchestra.research.sdk.driver import ToolUse
from paper_orchestra.research.sdk.fake import Denied, FakeAgent, FakeDriver
from paper_orchestra.research.sdk.guard import CLOSED_HOSTS
from paper_orchestra.research.sdk.session import AgentSession
from test_research_agents import activity
from test_research_decisions import OPTIONS, QUESTION, ask, call
from test_research_sdk import make_guard, run_scripted, sdk_context, sdk_project
from test_research_sdk_engine import advance, approve, make_engine, to_approved

# The fixtures of those two files, under their names: a home directory of the test's own
# for every test here, and a project with a team of scripted agents.
home = test_research_sdk.home
project = test_research_sdk_engine.project
team = test_research_sdk_engine.team
backend = test_research_sdk_engine.backend

#: Who asks under the sdk runtime: the roles of the tool loop, and the agent of a run.
ASKING = (*decisions.ASKING_ROLES, *decisions.AGENT_ASKING_ROLES)

#: A call as a model makes it, with everything the tool takes.
FULL = {
    "question": "Is the claim about Adam only, or about adaptive optimisers?",
    "context": "It decides **how many** optimisers the first sweep has.",
    "options": [
        {"label": "Adam only", "meaning": "One optimiser, one sweep."},
        {"label": "Adam and RAdam", "meaning": "Two sweeps."},
    ],
    "recommended": 1,
    "reason": "It is what the idea says.",
    "default": "The sweep is run with Adam only.",
    "page": "idea",
    "blocks": [],
}


# ---------------------------------------------------------------------------
# The tool on the board
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ASKING)
def test_an_agent_with_a_shell_asks_through_the_board_and_the_record_is_there(tmp_path, role):
    sdk = sdk_project(tmp_path)
    answers: list[tuple[str, bool]] = []

    def script(agent: FakeAgent) -> str:
        answers.append((agent.board("ask_decision", FULL), agent.failed))
        return "Asked."

    phase = "run" if role == "experimenter" else "research"
    _ctx, driver, _session, _said = run_scripted(sdk, role, script, phase=phase)

    (decision,) = sdk.decisions()
    said, failed = answers[0]
    assert not failed and said.startswith(f"Asked as {decision.id}.")
    assert "This session can ask 2 more." in said
    assert (decision.state, decision.answer, decision.recommended) == ("open", None, 1)
    assert decision.question == FULL["question"] and decision.page == sdk.page("idea").path
    assert [option.label for option in decision.options] == ["Adam only", "Adam and RAdam"]
    assert decision.raised_by == {
        "role": role,
        "round": 1,
        "phase": phase,
        "session": f"r1-{phase}-{role}",
    }
    assert (sdk.root / "decisions" / f"{decision.id}.json").is_file()
    # The person is told by the project's own event, once; the call has no line of its own.
    (asked,) = [e for e in sdk.events() if e["type"] == "decision.asked"]
    assert asked["agent"] == role and asked["data"]["decision"] == decision.id
    assert asked["summary"].startswith(f"The {role} asks you to decide:")
    assert not any("ask_decision" in line or "asks you" in line for line in activity(sdk))
    # It is the tool of the tool loop, word for word: one place says what may be asked.
    (offered,) = [tool for tool in driver.opened[0].board if tool.name == "ask_decision"]
    classic = decisions.ask_decision_tool(sdk, role, round=1, phase=phase, session="s")
    assert offered.input_schema == classic.input_schema
    assert offered.description == classic.description
    # Nothing on the board answers a decision, takes one back or changes one.
    names = [tool.name for tool in driver.opened[0].board]
    assert not [name for name in names if "answer" in name or "dismiss" in name or "decide" in name]
    assert set(offered.input_schema["properties"]) == set(FULL)


def test_both_runtimes_take_the_tool_from_one_place(tmp_path, monkeypatch):
    """Who may ask is decided once: a session of either kind offers what
    ``decisions.session_tools`` gives it for its role, and nothing where that is nothing."""
    sdk = sdk_project(tmp_path)
    asked_for: list[tuple[str, str]] = []
    real = decisions.session_tools

    def none_for_the_scout(project, role, **where):
        asked_for.append((role, where["session"]))
        return [] if role == "scout" else real(project, role, **where)

    monkeypatch.setattr(decisions, "session_tools", none_for_the_scout)
    driver = FakeDriver()
    ctx = sdk_context(sdk, driver)
    AgentSession(ctx, "scout").send("x")
    AgentSession(ctx, "theorist").send("x")
    scout, theorist = ([tool.name for tool in options.board] for options in driver.opened)
    assert "ask_decision" not in scout and "ask_decision" in theorist

    from test_research_agents import context, make_project

    classic_project = make_project(tmp_path / "classic")
    seen: list[list[str]] = []
    model = lambda request: seen.append(sorted(request.tools)) or "x"  # noqa: E731
    classic_ctx = context(classic_project, model, phase="research")
    agents._Session(classic_ctx, "scout").send("x")
    agents._Session(classic_ctx, "theorist").send("x")
    assert "ask_decision" not in seen[0] and "ask_decision" in seen[1]
    assert [role for role, _session in asked_for] == ["scout", "theorist", "scout", "theorist"]
    # No role that has no session asks, in either runtime.
    for role in ("runner", "engine", "human"):
        assert real(sdk, role, round=1, phase="run", session="s") == []


def test_the_cap_of_a_session_and_of_what_waits_hold_on_the_board(tmp_path):
    sdk = sdk_project(tmp_path)
    answers: list[tuple[str, bool]] = []

    def script(agent: FakeAgent) -> str:
        for number in range(1, 5):
            asked = agent.board("ask_decision", call(f"Is it choice number {number}?"))
            answers.append((asked, agent.failed))
        return "Asked."

    run_scripted(sdk, "theorist", script, phase="hypothesize")
    assert [failed for _said, failed in answers] == [False, False, False, True]
    assert "This session can ask no further decision." in answers[2][0]
    assert "which is the limit" in answers[3][0] and len(sdk.decisions()) == 3

    # A later attempt of the same session counts on from the files, not from zero ...
    again: list[tuple[str, bool]] = []

    def once_more(agent: FakeAgent) -> str:
        again.append((agent.board("ask_decision", call("And a fifth choice?")), agent.failed))
        return "Asked."

    run_scripted(sdk, "theorist", once_more, phase="hypothesize")
    assert again[0][1] and len(sdk.decisions()) == 3
    # ... what was asked is not asked twice, by whoever ...
    run_scripted(
        sdk,
        "analyst",
        lambda agent: (
            again.append(
                (agent.board("ask_decision", call("Is it choice number 1?")), agent.failed)
            )
            and "x"
        ),
        phase="analyze",
    )
    assert again[1][0].startswith("This was asked before") and len(sdk.decisions()) == 3
    # ... and nobody asks while twelve wait.
    for number in range(decisions.MAX_OPEN - 3):
        ask(sdk, f"What about question {number}?")
    run_scripted(
        sdk,
        "designer",
        lambda agent: (
            again.append(
                (agent.board("ask_decision", call("One more for the designer?")), agent.failed)
            )
            and "x"
        ),
        phase="design",
    )
    assert again[2][1] and "already wait for the researcher" in again[2][0]
    assert len(sdk.decisions(state="open")) == decisions.MAX_OPEN


def test_what_the_tool_refuses_in_the_tool_loop_it_refuses_on_the_board(tmp_path):
    sdk = sdk_project(tmp_path)
    answers: dict[str, tuple[str, bool]] = {}

    def script(agent: FakeAgent) -> str:
        for name, given in (
            ("one option", call(options=OPTIONS[:1])),
            ("no default", {"question": QUESTION, "options": OPTIONS}),
            ("no reason", call(recommended=1)),
            ("no such page", call(page="Q99")),
            ("an answer", {**call(), "answer": {"option": 1}}),
            ("a state", {**call(), "state": "answered"}),
        ):
            answers[name] = (agent.board("ask_decision", given), agent.failed)
        return "Tried."

    run_scripted(sdk, "scout", script)
    for name, (said, failed) in answers.items():
        assert failed, f"{name}: {said}"
    assert sdk.decisions() == []


def test_a_stop_request_keeps_an_agent_from_asking(tmp_path):
    sdk = sdk_project(tmp_path)
    stop: list[str] = []

    def script(agent: FakeAgent) -> str:
        stop.append("now")
        agent.board("ask_decision", call())
        raise AssertionError("the turn went on after the stop")

    driver = FakeDriver(script)
    ctx = sdk_context(sdk, driver, phase="hypothesize", stopping=lambda: bool(stop))
    with pytest.raises(agents.SessionStopped):
        AgentSession(ctx, "theorist").send("Go.")
    assert sdk.decisions() == []


# ---------------------------------------------------------------------------
# What the agents are told
# ---------------------------------------------------------------------------


def test_every_agent_with_a_shell_is_told_to_raise_what_is_the_researchers_choice():
    for role in ASKING:
        system = prompts.system_prompt(role, index="(empty)", runtime="sdk")
        plain = " ".join(system.split())
        # The shared part of both runtimes ...
        assert "## Decisions that are the researcher's" in system
        assert "A question page does not ask the researcher anything." in plain
        # ... and, where the board is described, that a choice is raised and not proposed.
        assert "If the choice is the researcher's" in plain
        assert "raise a decision with `mcp__board__ask_decision`" in plain
        assert "a page asks nobody" in plain
        assert "Raising it is all you can do with a decision" in plain
        assert plain.index("propose it on the board") < plain.index("If the choice is the")
    experimenter = " ".join(prompts.system_prompt("experimenter", index="x", runtime="sdk").split())
    assert "raise a decision with ask_decision that names the experiment as its `page`" in (
        experimenter
    )
    assert "the run does not wait for the answer" in experimenter
    # The tool loop's prompts say nothing of a board.
    assert "mcp__board__ask_decision" not in prompts.system_prompt("scout", index="x")


def test_every_agent_of_a_project_is_told_what_waits_and_what_was_answered(project, team, backend):
    """Through every phase, the run of an experiment among them: the opening message of
    each agent lists the decisions, as the tool loop's sessions are told them."""
    waiting = ask(project, "Which re-test channel is the object of study?", role="analyst")
    answered = ask(project, "Is the claim about Adam only?", role="theorist")
    project.answer_decision(answered.id, option=2, text="RAdam can wait.")
    left = ask(project, "Which seed?", role="designer")
    project.dismiss_decision(left.id)

    to_approved(project, team, backend)
    make_engine(project, team, backend).run_experiment("E1")
    advance(project, team, backend)

    assert set(team.prompts) == {*decisions.ASKING_ROLES, "experimenter"}
    for role, opened in team.prompts.items():
        message = opened[0]
        assert "<decided>" in message and f"- {answered.id}: Is the claim about Adam only?" in (
            message
        ), role
        assert 'They answered: option 2, "' in message and 'in their words: "RAdam can wait."' in (
            message
        ), role
        assert "<decisions_waiting>" in message and f"- {waiting.id}: Which re-test" in message
        assert "Do not ask them again." in message, role
        assert "<decisions_left_to_you>" in message and f"- {left.id}: Which seed?" in message
        assert "Only the decisions listed above have been put to the researcher." in message
    # The agent of a run is told its task first and what counts last, as before.
    run = team.prompts["experimenter"][0]
    assert run.startswith("Carry out E1") and run.index("<decided>") < run.index(
        "Leave `metrics.json`"
    )


def test_an_agent_is_told_that_nothing_was_asked_yet(project, team, backend):
    to_approved(project, team, backend)
    make_engine(project, team, backend).run_experiment("E1")
    for role in ("intake", "scout", "theorist", "designer", "experimenter"):
        assert (
            "No decision has been put to the researcher with ask_decision so far"
            in (team.prompts[role][0])
        ), role
    # Without the argument the message of a run says nothing of decisions.
    said = prompts.experiment_message(
        experiment="E1",
        title="x",
        plan="y",
        files=[],
        entrypoint="main.py",
        command="python main.py",
        timeout_seconds=60,
        network=False,
        models=False,
        launcher="with-models",
    )
    assert "ask_decision" not in said


# ---------------------------------------------------------------------------
# The agent that carries out an experiment
# ---------------------------------------------------------------------------


def test_the_agent_of_a_run_asks_and_the_run_does_not_wait(project, team, backend):
    to_approved(project, team, backend)
    answers: list[tuple[str, bool]] = []

    def ask_about_the_plan(agent: FakeAgent) -> None:
        for number in range(1, 5):
            given = {**FULL, "question": f"Does the plan mean reading {number}?"}
            if number > 1:
                del given["page"]
            answers.append((agent.board("ask_decision", given), agent.failed))
        # One that waits about a page is not asked about that page again.
        again = {**FULL, "question": "Is the plan about Adam?", "page": "idea"}
        answers.append((agent.board("ask_decision", again), False))

    team.during["experimenter"] = ask_about_the_plan
    engine = make_engine(project, team, backend)
    record = engine.run_experiment("E1")

    # The run is carried out as it would have been.
    assert record["status"] == "succeeded" and project.page("E1").state == "succeeded"
    # Three for the run, as for a session; the fourth is refused with the same words.
    assert [failed for _said, failed in answers] == [False, False, False, True, False]
    assert "which is the limit" in answers[3][0]
    assert "already asks the researcher about that page" in answers[4][0]
    asked = project.decisions()
    assert len(asked) == 3 and {decision.state for decision in asked} == {"open"}
    assert [decision.page for decision in asked] == [project.page("idea").path, None, None]
    assert asked[0].raised_by == {
        "role": "experimenter",
        "round": 1,
        "phase": "run",
        "session": "r1-run-E1-experimenter-E1-run1",
    }
    events = [e for e in project.events() if e["type"] == "decision.asked"]
    assert [e["agent"] for e in events] == ["experimenter"] * 3
    assert events[0]["summary"].startswith("The experimenter asks you to decide:")

    # The agent of the next run has three of its own: the cap is a run's, not the phase's.
    answers.clear()
    team.during["experimenter"] = lambda agent: answers.append(
        (agent.board("ask_decision", call("And for the second run?")), agent.failed)
    )
    approve(project, "E1", backend)
    engine = make_engine(project, team, backend)
    assert engine.run_experiment("E1")["number"] == 2
    assert answers[0][1] is False and len(project.decisions()) == 4
    assert project.decisions()[-1].raised_by["session"] == "r1-run-E1-experimenter-E1-run2"
    # ... and it was told what the first one asked.
    assert "Does the plan mean reading 1?" in team.prompts["experimenter"][-1]


def test_the_checkpoint_counts_what_an_agent_with_a_shell_asked(project, team, backend):
    team.during["scout"] = lambda agent: agent.board("ask_decision", FULL)
    state = advance(project, team, backend)
    assert state["status"] == "checkpoint"
    assert decisions.waiting(1) in state["checkpoint"]["summary"]
    assert len(project.decisions(state="open")) == 1


# ---------------------------------------------------------------------------
# The person answers: no shell command and no file of an agent does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "paper-orchestra research decide . 1 --option 2",
        "paper-orchestra research decide . d-0000aaaa --say 'take the second'",
        "paper-orchestra research decide . 1 --dismiss",
        "uv run paper-orchestra research decide . 1 --option 1",
        "cd .. && ../../.venv/bin/paper-orchestra research decide . 1 --option 1",
        "python -m paper_orchestra research decide . 1 --option 1",
        'python -c "from paper_orchestra.research.project import Project; '
        "Project.open('../..').answer_decision('d-0000aaaa', option=1)\"",
        "echo 1 | xargs paper-orchestra research decide .",
    ],
)
def test_the_guard_refuses_the_command_that_answers_a_decision(tmp_path, home, command):
    for sandbox in (True, False):
        guard = make_guard(tmp_path, home, sandbox=sandbox)
        for sub in (False, True):
            refused = guard.check(ToolUse("Bash", {"command": command}, sub))
            assert refused is not None, (command, sandbox, sub)
            # It says what to do instead, and whose the answer is.
            assert "decide" in refused and "ask_decision" in refused
            assert "The researcher decides and answers" in refused


def test_the_guard_closes_every_file_route_to_the_decisions(tmp_path, home):
    guard = make_guard(tmp_path, home)
    folder = guard.project / "decisions"
    folder.mkdir()
    record = folder / "d-0000aaaa.json"
    record.write_text('{"state": "open"}', encoding="utf-8")
    os.symlink(folder, guard.folder / "linked")
    os.symlink(record, guard.folder / "one.json")

    for tool in ("Write", "Edit", "MultiEdit"):
        for target in (
            str(record),
            str(folder / "d-0000ffff.json"),
            "../../decisions/d-0000aaaa.json",
            "linked/d-0000aaaa.json",
            "one.json",
        ):
            refused = guard.check(ToolUse(tool, {"file_path": target}))
            assert refused is not None and "only in your own folder" in refused, (tool, target)
    assert guard.check(ToolUse("NotebookEdit", {"notebook_path": str(record)})) is not None
    for use in (
        ToolUse("Read", {"file_path": str(record)}),
        ToolUse("Read", {"file_path": "../../decisions/d-0000aaaa.json"}),
        ToolUse("Read", {"file_path": "one.json"}),
        ToolUse("Read", {"file_path": "linked/d-0000aaaa.json"}),
        ToolUse("Grep", {"pattern": "open", "path": str(folder)}),
        ToolUse("Grep", {"pattern": "open", "path": str(guard.project)}),
        ToolUse("Grep", {"pattern": "open", "path": "linked"}),
        ToolUse("Glob", {"pattern": str(folder / "*.json")}),
        ToolUse("Glob", {"pattern": "../../decisions/*.json"}),
        ToolUse("Glob", {"pattern": "*.json", "path": str(folder)}),
        ToolUse("LS", {"path": str(folder)}),
    ):
        refused = guard.check(use)
        assert refused is not None and "not yours to read" in refused, use
        assert guard.check(ToolUse(use.name, dict(use.input), True)) is not None  # a helper's call
    # Its own folder stays its own.
    assert guard.check(ToolUse("Write", {"file_path": "notes/decisions.md"})) is None
    assert guard.check(ToolUse("Read", {"file_path": "notes/decisions.md"})) is None

    # The shell's sandbox hides the project's directory, the folder with it, and writes
    # only in the agent's own folder: `cat`, `sed -i` and `python` meet the same wall.
    files = guard.settings()["sandbox"]["filesystem"]
    hidden = [Path(place) for place in files["denyRead"] if place.startswith("/")]
    assert any(record.resolve().is_relative_to(place) for place in hidden)
    opened = [Path(place) for place in files["allowRead"]]
    assert not any(record.resolve().is_relative_to(place) for place in opened)
    assert not any(
        record.resolve().is_relative_to(Path(place)) for place in files.get("allowWrite", [])
    )
    assert guard.settings()["sandbox"]["allowUnsandboxedCommands"] is False
    # ... and the app, which answers a decision for whoever asks it, is not reached.
    assert {"localhost", "127.0.0.1", "::1"} <= set(CLOSED_HOSTS)
    for url in (
        "http://127.0.0.1:8501/api/projects/p/decisions/d-0000aaaa/answer",
        "http://localhost:8501/api/projects/p/decisions/d-0000aaaa/dismiss",
        "http://[::1]:8501/api/projects/p/decisions",
    ):
        assert guard.check(ToolUse("WebFetch", {"url": url})) is not None, url


def test_an_agent_that_tries_to_answer_a_decision_changes_nothing(tmp_path):
    """Every way an agent with a shell has: the command line, its file tools, the app,
    the board.  The record is afterwards what it was, and the feed shows the attempts."""
    sdk = sdk_project(tmp_path)
    decision = ask(sdk)
    record = sdk.root / "decisions" / f"{decision.id}.json"
    before = record.read_text(encoding="utf-8")
    forged = json.dumps({**decision.to_dict(), "state": "answered"})
    answers: dict[str, object] = {}
    ran: list[str] = []

    def script(agent: FakeAgent) -> str:
        answers["decide"] = agent.bash(
            f"paper-orchestra research decide . {decision.id} --option 1",
            does=lambda: ran.append("decide"),
        )
        answers["dismiss"] = agent.bash(
            "uv run paper-orchestra research decide . 1 --dismiss",
            does=lambda: ran.append("dismiss"),
            sub=True,
        )
        answers["module"] = agent.bash(
            "python -m paper_orchestra research decide . 1 --say yes",
            does=lambda: ran.append("module"),
        )
        answers["write"] = agent.write(str(record), forged)
        answers["edit"] = agent.write(f"../../decisions/{decision.id}.json", forged, tool="Edit")
        answers["new"] = agent.write(str(record.with_name("d-0000ffff.json")), forged)
        answers["read"] = agent.read(str(record))
        answers["glob"] = agent.tool("Glob", {"pattern": str(record.parent / "*.json")})
        answers["grep"] = agent.tool("Grep", {"pattern": "open", "path": str(record.parent)})
        answers["app"] = agent.fetch(
            f"http://127.0.0.1:8501/api/projects/p/decisions/{decision.id}/answer"
        )
        # The board reaches the wiki and the attached files, and no decision.
        path = f"/memories/decisions/{decision.id}.json"
        board = [
            (agent.memory("view", path=path), agent.failed),
            (agent.memory("create", path=path, file_text=forged), agent.failed),
            (
                agent.memory("str_replace", path=path, old_str='"open"', new_str='"answered"'),
                agent.failed,
            ),
            (agent.memory("delete", path=path), agent.failed),
        ]
        answers["board"] = board
        answers["own"] = agent.write("notes/what-i-would-answer.md", "option 1")
        return "Tried."

    _ctx, _driver, session, _said = run_scripted(sdk, "analyst", script, phase="analyze")

    for name in ("decide", "dismiss", "module", "write", "edit", "new", "read", "glob", "grep"):
        assert isinstance(answers[name], Denied), name
    assert isinstance(answers["app"], Denied)
    assert all(failed for _said, failed in answers["board"]), answers["board"]
    assert ran == []
    assert record.read_text(encoding="utf-8") == before
    assert sorted(path.name for path in record.parent.iterdir()) == [record.name]
    assert sdk.decisions() == [decision] and sdk.decision(decision.id).state == "open"
    assert not [e for e in sdk.events() if e["type"] in ("decision.answered", "decision.dismissed")]
    # What it writes in its own folder is nobody's answer.
    assert (session.folder / "notes" / "what-i-would-answer.md").is_file()
    assert "ask_decision" in answers["decide"] and "decide" in answers["decide"]
    feed = activity(sdk)
    assert any("research decide" in line and "it was refused" in line for line in feed)
    assert not any("ran “paper-orchestra" in line for line in feed)
