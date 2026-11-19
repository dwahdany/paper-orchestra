"""Autopilot, offline: the settings and their guards, the standing rule, the unattended
stretch with its caps and brakes, and the account of what happened without the person.

A scripted team of models (the classic runtime) and a scripted team of agents (the
``sdk`` runtime, with the fake driver) drive real projects through the real engine.
Each thing autopilot does has a test that fails without it, next to the same project
in supervised mode, which is as it always was.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import test_research_engine as classic
import test_research_sdk
import test_research_sdk_engine as sdk
from paper_orchestra.llm.base import LLMConfigError
from paper_orchestra.research import autopilot, decisions, experiments
from paper_orchestra.research.engine import Engine, ModelNeeded, read_state
from paper_orchestra.research.model import (
    AUTONOMY_ENV,
    AUTOPILOT,
    AUTOPILOT_NEEDS_SANDBOX,
    NotAllowed,
    ProjectError,
    ProjectSettings,
    check_transition,
    settings_for_new_project,
)
from paper_orchestra.research.project import AUTOPILOT_KEY, Project
from paper_orchestra.research.sdk.driver import ToolUse
from paper_orchestra.research.sdk.fake import Denied, FakeAgent
from test_research_agents import FAILING_CODE, FakeBackend, make_project, make_scholar

home = test_research_sdk.home


class Remote(FakeBackend):
    """A stand-in for a remote backend: where the classic runtime's runs are not the
    person's own machine, and the standing rule may approve them."""

    name = "modal"


AUTO = {"mode": "autopilot", "rounds": 1}


def modal_signed_in(monkeypatch, signed_in: bool) -> None:
    """Make Modal read as signed in on this machine, or not: what its own check asks,
    without a token and without contacting anything."""
    module = pytest.importorskip("paper_orchestra.compute.modal_backend")
    monkeypatch.setattr(module, "modal_token_configured", lambda: signed_in)


@pytest.fixture(autouse=True)
def _modal_is_signed_in(monkeypatch):
    """What waits is asked of the machine as it is (would an approval go through now),
    and no test's answer may depend on whether this one has a Modal token."""
    try:
        from paper_orchestra.compute import modal_backend
    except ImportError:
        return
    monkeypatch.setattr(modal_backend, "modal_token_configured", lambda: True)


def notes(tmp_path: Path) -> Path:
    path = tmp_path / "notes.md"
    path.write_text("Warmup of 4000 steps was used in the first runs.\n")
    return path


def classic_project(tmp_path: Path, autonomy: dict | None = AUTO, **settings) -> Project:
    """A project of the classic runtime whose experiments run on a remote backend."""
    chosen = {"compute": "modal", **settings}
    if autonomy is not None:
        chosen["autonomy"] = autonomy
    return make_project(tmp_path, files=[notes(tmp_path)], settings=chosen)


def stretch(project: Project, team=None, backend=None, **options) -> dict:
    """One ``advance()`` by a fresh worker: under autopilot, one unattended stretch."""
    engine = classic.make_engine(project, team or classic.Team(), backend or Remote())
    return engine.advance(**options)


def events(project: Project, *types: str) -> list[dict]:
    return [event for event in project.events() if event["type"] in types]


def reason(project: Project) -> str:
    return read_state(project)["autopilot"]["reason"]


# ---------------------------------------------------------------------------
# The settings and their guards
# ---------------------------------------------------------------------------


def test_a_project_is_supervised_unless_the_person_says_otherwise(tmp_path):
    project = make_project(tmp_path)
    autonomy = project.settings.autonomy
    assert (autonomy.mode, autonomy.rounds, autonomy.model_usd, autonomy.hours) == (
        "supervised",
        3,
        200.0,
        12.0,
    )
    assert autonomy.network is False
    # Its file is as it always was: the group is written once something in it is set.
    on_file = json.loads((project.root / "project.json").read_text())
    assert "autonomy" not in on_file["settings"] and AUTOPILOT_KEY not in on_file
    assert "autonomy" not in ProjectSettings().to_dict()


def test_an_older_project_loads_and_stays_supervised(tmp_path):
    project = make_project(tmp_path)
    file = project.root / "project.json"
    data = json.loads(file.read_text())
    data["settings"].pop("autonomy", None)
    file.write_text(json.dumps(data))
    opened = Project.open(project.root)
    assert opened.settings.autonomy.mode == "supervised" and opened.problems() == []
    page = opened.create_page("hypothesis", "Old ways", "body", author="agent:theorist")
    assert page.review == "proposed" and page.decided_by is None


def test_the_switch_and_the_caps_are_saved_and_the_switching_is_said(tmp_path):
    project = make_project(tmp_path)
    project.settings.update({"autonomy": {"mode": "autopilot", "rounds": 2, "hours": 0.5}})
    project.save_settings()
    opened = Project.open(project.root)
    assert opened.settings.autonomy.to_dict() == {
        "mode": "autopilot",
        "rounds": 2,
        "model_usd": 200.0,
        "hours": 0.5,
        "network": False,
    }
    # When it was switched on is on file: what waited by then is the person's.
    assert opened.autopilot_since
    said = events(opened, "autopilot.switched")[-1]
    assert said["summary"].startswith("You switched autopilot on: at most 2 rounds")
    assert said["data"]["by"] == "human" and said["data"]["mode"] == "autopilot"
    opened.settings.update({"autonomy": {"mode": "supervised"}})
    opened.save_settings()
    assert Project.open(project.root).autopilot_since == ""
    assert "switched autopilot off" in events(opened, "autopilot.switched")[-1]["summary"]


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"mode": "full"}, "settings.autonomy.mode is 'full'; it must be one of"),
        ({"rounds": 0}, "settings.autonomy.rounds is 0; it must be a whole number, one or more"),
        ({"rounds": 1.5}, "settings.autonomy.rounds is 1.5"),
        ({"model_usd": 0}, "settings.autonomy.model_usd is 0; it must be a number above zero"),
        ({"model_usd": float("inf")}, "settings.autonomy.model_usd is inf"),
        ({"hours": -1}, "settings.autonomy.hours is -1"),
        ({"hours": "long"}, "settings.autonomy.hours is 'long'"),
        ({"network": "yes"}, "settings.autonomy.network is 'yes'; it must be true or false"),
        ({"round": 9}, "settings.autonomy has no value named round"),
    ],
)
def test_a_cap_that_would_cap_nothing_is_refused(tmp_path, change, said):
    project = make_project(tmp_path)
    with pytest.raises(ProjectError, match=re.escape(said)):
        project.settings.update({"autonomy": {"mode": "autopilot", **change}})
    assert project.settings.autonomy.mode == "supervised"


def test_autopilot_is_refused_for_agents_outside_the_sandbox_whichever_value_changes(tmp_path):
    open_box = {"agents": {"runtime": "sdk", "sandbox": False}}
    project = make_project(tmp_path, settings=open_box)
    with pytest.raises(ProjectError) as refusal:
        project.settings.update({"autonomy": {"mode": "autopilot"}})
    # One plain sentence, and nothing around it.
    assert str(refusal.value) == AUTOPILOT_NEEDS_SANDBOX
    assert project.settings.autonomy.mode == "supervised"
    # The other way round: the sandbox cannot be taken away under an autopilot.
    other = make_project(tmp_path, name="q", settings={"agents": {"runtime": "sdk"}})
    other.settings.update({"autonomy": {"mode": "autopilot"}})
    other.save_settings()
    with pytest.raises(ProjectError, match="Autopilot is refused"):
        other.settings.update({"agents": {"sandbox": False}})
    assert other.settings.agents.sandbox is True
    # With the classic runtime the sandbox setting concerns nobody: no agent has a shell.
    plain = make_project(tmp_path, name="r", settings={"agents": {"sandbox": False}})
    plain.settings.update({"autonomy": {"mode": "autopilot"}})
    # A new project that asks for both is not created.
    with pytest.raises(ProjectError, match="Autopilot is refused"):
        make_project(tmp_path, name="s", settings={**open_box, "autonomy": {"mode": "autopilot"}})


def test_a_file_that_asks_for_both_is_read_as_supervised(tmp_path):
    """A file edited by hand, or by an agent that works outside the sandbox."""
    project = make_project(tmp_path, settings={"agents": {"runtime": "sdk", "sandbox": False}})
    file = project.root / "project.json"
    data = json.loads(file.read_text())
    data["settings"]["autonomy"] = {"mode": "autopilot", "rounds": 99, "model_usd": 1e9}
    file.write_text(json.dumps(data))
    opened = Project.open(project.root)
    assert opened.settings.autonomy.mode == "supervised"
    assert any("Autopilot is refused" in item["message"] for item in opened.problems())
    page = opened.create_page("hypothesis", "Forged", "body", author="agent:theorist")
    assert page.review == "proposed"


def test_the_environment_makes_autopilot_the_default_of_new_projects_only(tmp_path, monkeypatch):
    assert "autonomy" not in settings_for_new_project({}, {})
    asked = settings_for_new_project({}, {AUTONOMY_ENV: "autopilot"})
    assert asked["autonomy"] == {"mode": "autopilot"}
    # What the person names wins, and a word the setting does not know changes nothing.
    named = {"autonomy": {"mode": "supervised"}}
    assert settings_for_new_project(named, {AUTONOMY_ENV: "autopilot"})["autonomy"] == {
        "mode": "supervised"
    }
    assert "autonomy" not in settings_for_new_project({}, {AUTONOMY_ENV: "yes please"})
    # A project that exists keeps what its file says, whatever the environment is by then.
    project = make_project(tmp_path)
    monkeypatch.setenv(AUTONOMY_ENV, "autopilot")
    assert Project.open(project.root).settings.autonomy.mode == "supervised"


def test_the_rule_has_no_say_where_the_person_did_not_switch_it_on(tmp_path):
    project = make_project(tmp_path)
    page = project.create_page("hypothesis", "Mine", "body", author="human")
    with pytest.raises(NotAllowed, match="Autopilot decides only in a project whose person"):
        project.set_state(page.path, "supported", by=AUTOPILOT)
    with pytest.raises(NotAllowed, match="switched it on"):
        project.settle_waiting()
    with pytest.raises(NotAllowed):
        project.review(page.path, "accept", by=AUTOPILOT)  # review() is the person's alone
    # What the rule may move at all: an approval of what is planned or failed, a verdict.
    check_transition("experiment", "planned", "approved", AUTOPILOT)
    check_transition("experiment", "failed", "approved", AUTOPILOT)
    check_transition("hypothesis", "testing", "refuted", AUTOPILOT)
    for move in (("succeeded", "approved"), ("cancelled", "approved"), ("planned", "cancelled")):
        with pytest.raises(NotAllowed):
            check_transition("experiment", *move, AUTOPILOT)
    with pytest.raises(NotAllowed):
        check_transition("experiment", "approved", "planned", AUTOPILOT)


# ---------------------------------------------------------------------------
# The standing rule: pages, edits, decisions
# ---------------------------------------------------------------------------


def test_a_new_page_is_accepted_and_the_record_names_the_rule(tmp_path):
    supervised = make_project(tmp_path, name="s")
    waits = supervised.create_page("hypothesis", "H", "body", author="agent:theorist")
    assert waits.review == "proposed" and "decided_by" not in waits.meta
    project = make_project(tmp_path, settings={"autonomy": AUTO})
    page = project.create_page("hypothesis", "H", "body", author="agent:theorist")
    assert page.review == "accepted"
    assert page.meta["decided_by"] == "autopilot" and page.meta["decided_at"]
    assert page.decided_by_person is False
    created = events(project, "page.created")[-1]
    assert created["summary"] == 'The theorist added hypothesis H1 "H"; autopilot accepted it.'
    assert created["data"]["decided_by"] == "autopilot" and created["agent"] == "theorist"
    assert project.pending() == [] and project.counts()["pending"]["total"] == 0
    # Every kind: also a source record, where the person asked to vet sources themselves.
    project.settings.update({"review": {"source": "review"}})
    project.save_settings()
    source = project.write_record("source", "Liu2020", "On the variance", "abstract")
    assert source.review == "accepted" and source.meta["decided_by"] == "autopilot"
    held_back = supervised.settings
    held_back.update({"review": {"source": "review"}})
    supervised.save_settings()
    assert supervised.write_record("source", "Liu2020", "T", "a").review == "proposed"
    # The person's own page, and a kind the policy accepts anyway, are as they were.
    own = project.create_page("topic", "Mine", "body", author="human")
    journal = project.create_page("journal", "Round 1", "body", author="agent:analyst")
    assert "decided_by" not in own.meta and "decided_by" not in journal.meta


def test_a_page_the_rule_accepted_is_still_its_authors_to_work_on(tmp_path):
    """As a page the review policy accepts on creation is: nobody read it."""
    project = make_project(tmp_path, settings={"autonomy": AUTO})
    page = project.create_page("hypothesis", "H", "first", author="agent:theorist")
    revised = project.update_page(page.path, author="agent:theorist", body="second thoughts")
    assert revised.body.strip() == "second thoughts" and revised.decided_by == "autopilot"
    # Once the person has decided it themselves it is theirs, as ever.
    project.update_page(page.path, author="human", body="my wording")
    with pytest.raises(NotAllowed, match="accepted by the person"):
        project.update_page(page.path, author="agent:theorist", body="again")
    assert project.page(page.path).decided_by == "human"


def test_an_edit_proposal_is_applied_and_the_record_names_the_rule(tmp_path):
    supervised = make_project(tmp_path, name="s")
    target = supervised.create_page("hypothesis", "H", "body", author="human")
    waiting = supervised.propose_edit(
        target.path, author="agent:analyst", summary="verdict", fields={"state": "supported"}
    )
    assert waiting.review == "proposed" and supervised.page(target.path).state == "open"

    project = make_project(tmp_path, settings={"autonomy": AUTO})
    page = project.create_page("hypothesis", "H", "old text", author="agent:theorist")
    edit = project.propose_edit(
        page.path,
        author="agent:analyst",
        summary="The run supports it",
        body="new text",
        fields={"state": "supported"},
    )
    assert (edit.review, edit.decided_by) == ("accepted", "autopilot")
    after = project.page(page.path)
    # The verdict on a hypothesis is one of the changes the rule makes for the person.
    assert after.state == "supported" and after.body.strip() == "new text"
    assert after.decided_by == "autopilot"  # not "human": the person approved nothing
    on_file = json.loads((project.root / "proposals" / f"{edit.id}.json").read_text())
    assert on_file["decided_by"] == "autopilot" and on_file["review"] == "accepted"
    decided = events(project, "review.decided")[-1]
    assert decided["summary"].startswith("Autopilot applied the analyst's edit to hypothesis H1")
    assert decided["agent"] == "autopilot" and decided["data"]["by"] == "autopilot"
    assert project.pending() == []


def test_two_kinds_of_edit_wait_for_the_person_all_the_same(tmp_path):
    project = make_project(tmp_path, settings={"autonomy": AUTO})
    # A new text for what the person wrote themselves: their idea, their own page.
    idea = project.propose_edit(
        "wiki/idea.md", author="agent:analyst", summary="Another idea", body="Study RMSprop."
    )
    assert idea.review == "proposed" and "RMSprop" not in project.page("wiki/idea.md").body
    own = project.create_page("hypothesis", "Mine", "my words", author="human")
    # ... while a field of such a page (the verdict) is the rule's to set.
    verdict = project.propose_edit(
        own.path, author="agent:analyst", summary="Refuted", fields={"state": "refuted"}
    )
    assert verdict.review == "accepted" and project.page(own.path).state == "refuted"
    assert project.page(own.path).body.strip() == "my words"
    # An edit the project cannot apply as it stands: the page's text changed since.
    page = project.create_page("topic", "T", "one", author="agent:scout")
    stale = project.propose_edit(page.path, author="agent:analyst", summary="s", body="two")
    assert stale.review == "accepted"
    project.settings.update({"autonomy": {"mode": "supervised"}})
    project.save_settings()
    held = project.propose_edit(page.path, author="agent:analyst", summary="s2", body="three")
    project.update_page(page.path, author="human", body="the person rewrote it")
    project.settings.update({"autonomy": {"mode": "autopilot"}})
    project.save_settings()
    done = project.settle_waiting()
    assert held.id in done["left"] and idea.id in done["left"]
    assert project.proposal(held.id).review == "proposed"
    assert project.page(page.path).body.strip() == "the person rewrote it"
    assert {item.id for item in project.pending()} == {idea.id, held.id}


def test_a_decision_goes_on_at_its_default_and_is_not_counted_as_waiting(tmp_path):
    supervised = make_project(tmp_path, name="s")
    asked = supervised.ask_decision("Which?", options=["a", "b"], role="theorist", default="a")
    assert asked.state == "open" and supervised.counts()["decisions"]["open"] == 1
    assert "defaulted_by" not in asked.to_dict()

    project = make_project(tmp_path, settings={"autonomy": AUTO})
    decision = project.ask_decision(
        "Adam only, or adaptive optimisers?",
        options=["Adam only", "Both"],
        role="theorist",
        default="The sweep is run with Adam only.",
    )
    assert decision.state == "defaulted" and decision.answer is None
    on_file = json.loads((project.root / "decisions" / f"{decision.id}.json").read_text())
    assert on_file["state"] == "defaulted" and on_file["defaulted_by"] == "autopilot"
    assert on_file["answers"] == [] and on_file["answer"] is None
    # No badge, no banner, no count at a checkpoint: it does not wait.
    assert project.counts()["decisions"]["open"] == 0
    assert project.decisions(state="open") == []
    # It stays listed, and every later session is told that the default holds.
    assert [item.id for item in project.decisions()] == [decision.id]
    told = decisions.told(project)
    assert told[0]["state"] == "defaulted" and told.unattended is True
    said = events(project, "decision.asked")[-1]
    assert "Autopilot went on with its default" in said["summary"]
    assert said["data"]["state"] == "defaulted" and said["agent"] == "theorist"
    assert not events(project, "decision.answered", "decision.dismissed")

    # The person may still answer: it is their first word on it, not a changed answer.
    answered = project.answer_decision(decision.id, option=2)
    assert answered.state == "answered" and answered.defaulted_at
    event = events(project, "decision.answered")[-1]
    assert event["summary"].startswith("You decided") and event["data"]["changed"] is False
    assert decisions.told(project)[0]["answer"]["option"] == 2
    assert Project.open(project.root).decision(decision.id).state == "answered"


def test_what_waits_at_the_switch_is_left_alone_unless_the_person_asks(tmp_path):
    project = make_project(tmp_path)
    page = project.create_page("hypothesis", "Before", "body", author="agent:theorist")
    edit = project.propose_edit(
        page.path, author="agent:analyst", summary="reword", fields={"summary": "short"}
    )
    decision = project.ask_decision("Which?", options=["a", "b"], role="theorist", default="a")
    project.settings.update({"autonomy": {"mode": "autopilot"}})
    project.save_settings()
    # Switched on, and nothing that waited has moved.
    assert project.page(page.path).review == "proposed"
    assert project.proposal(edit.id).review == "proposed"
    assert project.decision(decision.id).state == "open"
    assert project.counts()["pending"]["total"] == 2
    # "Also settle what waits now".
    done = project.settle_waiting()
    assert done == {
        "accepted": [page.path],
        "applied": [edit.id],
        "defaulted": [decision.id],
        "left": [],
    }
    after = project.page(page.path)
    assert after.review == "accepted" and after.decided_by == "autopilot"
    assert after.summary == "short"
    assert project.decision(decision.id).state == "defaulted"
    assert project.counts()["pending"]["total"] == 0
    switched = events(project, "autopilot.switched")[-1]["seq"]
    said = [event["summary"] for event in project.events(after=switched)]
    assert 'Autopilot accepted hypothesis H1 "Before".' in said
    assert any(line.startswith("Autopilot left “Which?” at the agents' default") for line in said)
    assert not any(line.startswith("You ") for line in said)
    assert project.autopilot_since == ""  # from here on the rule covers everything


# ---------------------------------------------------------------------------
# The standing rule: runs
# ---------------------------------------------------------------------------


def planned(project: Project, *, network: bool = False, code: str | None = None) -> str:
    """An accepted experiment with its code, ready to be approved; its id."""
    page = project.create_page(
        "experiment",
        "Warmup against none",
        "## Plan\n\nRun it.\n",
        author="agent:designer",
        fields={"mode": "script", "entrypoint": "main.py", "timeout_seconds": 300}
        | ({"network": True} if network else {}),
    )
    from test_research_agents import GOOD_CODE

    experiments.write_code(
        project, str(page.id), {"main.py": code or GOOD_CODE}, author="agent:designer"
    )
    return str(page.id)


def test_the_rule_approves_through_the_approval_and_the_record_names_it(tmp_path):
    project = classic_project(tmp_path)
    name = planned(project)
    page = experiments.approve_by_rule(project, name, backends={"modal": Remote()})
    assert page.state == "approved"
    assert page.meta["approved_by"] == "autopilot" and page.meta["approved_digest"]
    assert page.meta["updated_by"] == "autopilot" and page.meta["network"] is False
    record = json.loads((project.experiment_dir(name) / "approval.json").read_text())
    assert record["by"] == "autopilot" and record["digest"] == page.meta["approved_digest"]
    said = [event["summary"] for event in project.events()[-3:]]
    assert any(line.startswith("Autopilot set how experiment E1") for line in said)
    assert not any("You " in line for line in said)
    # What the rule approved waits in nobody's line but its own stretch's.
    assert experiments.waiting_in_line(project) == []
    assert [str(p.id) for p in experiments.waiting_in_line(project, rule=True)] == [name]
    # The person's own approval of it takes the mark off again.
    project.set_state(page.path, "planned", by="human")
    mine = experiments.approve(project, name, backends={"modal": Remote()})
    assert "approved_by" not in mine.meta
    assert "by" not in json.loads((project.experiment_dir(name) / "approval.json").read_text())


def test_the_approval_of_a_supervised_project_is_recorded_as_it_always_was(tmp_path):
    project = classic_project(tmp_path, autonomy=None)
    name = planned(project)
    project.review(name, "accept")
    with pytest.raises(ProjectError, match="Autopilot is off"):
        experiments.approve_by_rule(project, name, backends={"modal": Remote()})
    with pytest.raises(NotAllowed):
        experiments.approve(project, name, backends={"modal": Remote()}, by=AUTOPILOT)
    page = experiments.approve(project, name, backends={"modal": Remote()})
    assert "approved_by" not in page.meta and page.meta["updated_by"] == "human"
    record = json.loads((project.experiment_dir(name) / "approval.json").read_text())
    assert sorted(record) == ["at", "code", "digest", "plan"]
    assert sorted(json.loads((project.root / "project.json").read_text())) == [
        "created",
        "home",
        "id",
        "schema",
        "settings",
        "title",
    ]


def test_network_access_is_granted_only_where_the_plan_asks_and_the_person_allowed(tmp_path):
    closed = classic_project(tmp_path)
    name = planned(closed, network=True)
    page = experiments.approve_by_rule(closed, name, backends={"modal": Remote()})
    assert page.meta["network"] is False  # the plan asked; the person had not allowed it
    allowed = make_project(
        tmp_path,
        name="q",
        settings={"compute": "modal", "autonomy": {**AUTO, "network": True}},
    )
    asks, does_not = planned(allowed, network=True), planned(allowed)
    backends = {"modal": Remote()}
    assert experiments.approve_by_rule(allowed, asks, backends=backends).meta["network"] is True
    # Allowed is not granted: a plan that does not ask gets none.
    assert (
        experiments.approve_by_rule(allowed, does_not, backends=backends).meta["network"] is False
    )


def test_the_rule_does_not_approve_unread_code_to_run_as_the_person(tmp_path):
    """The classic runtime on the local backend: the code would run on this machine, as
    the person, outside any sandbox.  That approval stays theirs."""
    project = make_project(tmp_path, settings={"autonomy": AUTO})  # compute: local
    name = planned(project)
    assert experiments.rule_holds(project, name) == experiments.UNSANDBOXED_RUN
    with pytest.raises(ProjectError, match="outside a sandbox"):
        experiments.approve_by_rule(project, name, backends={"local": FakeBackend()})
    assert project.page(name).state == "planned"
    assert autopilot.held(project) == [
        {"id": name, "reason": experiments.UNSANDBOXED_RUN, "kind": "rule"}
    ]
    # Sandboxed sdk agents carry such a run out in their own folder: that is fine.
    sandboxed = make_project(
        tmp_path, name="q", settings={"agents": {"runtime": "sdk"}, "autonomy": AUTO}
    )
    assert experiments.rule_holds(sandboxed, planned(sandboxed)) is None


def test_what_was_planned_before_the_switch_is_the_persons_unless_settled(tmp_path):
    project = classic_project(tmp_path, autonomy=None)
    name = planned(project)
    project.review(name, "accept")
    project.settings.update({"autonomy": AUTO})
    project.save_settings()
    assert "before autopilot was switched on" in experiments.rule_holds(project, name)
    later = planned(project)  # designed after the switch
    assert experiments.rule_holds(project, later) is None
    project.settle_waiting()
    assert experiments.rule_holds(project, name) is None


# ---------------------------------------------------------------------------
# The unattended stretch
# ---------------------------------------------------------------------------


def test_a_stretch_goes_from_the_idea_to_the_cap_with_no_input(tmp_path):
    project = classic_project(tmp_path)
    team, backend = classic.Team(), Remote()
    state = stretch(project, team, backend)

    # One call: intake to the analysis that ends the round, where the cap on rounds is.
    assert [record["phase"] for record in state["phases"]] == [
        "intake",
        "research",
        "hypothesize",
        "design",
        "run",
        "analyze",
    ]
    assert state["status"] == "checkpoint" and state["next_phase"] == "hypothesize"
    record = state["autopilot"]
    assert record["reason"] == "rounds" and record["rounds"] == 1 and record["ended"]
    assert record["said"].startswith("The cap on rounds was reached: 1 round completed")
    # Nothing waited, and nothing waits.
    counts = project.counts()
    assert counts["pending"]["total"] == 0 and counts["decisions"]["open"] == 0
    assert counts["experiments"]["succeeded"] == 2 and len(backend.jobs) == 2
    for page in project.pages(kind="experiment"):
        assert page.meta["approved_by"] == "autopilot" and page.decided_by == "autopilot"
    assert {page.state for page in project.pages(kind="hypothesis")} == {"supported"}

    # The feed: the start, each step taken alone, the end; autopilot is the actor.
    kinds = [event["type"] for event in events(project, *autopilot.__all__)]
    assert kinds == []  # (the names of functions are no event types)
    told = events(project, "autopilot.started", "autopilot.continued", "autopilot.stopped")
    assert [event["type"] for event in told] == [
        "autopilot.started",
        "autopilot.continued",
        "autopilot.continued",
        "autopilot.continued",
        "autopilot.stopped",
    ]
    assert {event["agent"] for event in told} == {"autopilot"}
    assert told[0]["summary"].startswith("Autopilot goes on from here without you: at most 1 round")
    assert "runs it approves get no network access" in told[0]["summary"]
    assert told[1]["summary"].startswith("Autopilot goes on with the hypothesis phase")
    assert told[-1]["data"]["reason"] == "rounds" and told[-1]["data"]["runs"] == ["E1", "E2"]

    # No record says that the person read, accepted or approved anything.
    for event in project.events():
        assert event["data"].get("by") != "human", event
        assert not event["summary"].startswith("You "), event
        assert "you accepted" not in event["summary"] and "you approved" not in event["summary"]
    for page in project.pages():
        assert page.meta.get("decided_by") in (None, "autopilot"), page.path
        assert page.meta.get("updated_by") != "human" or page.author == "human", page.path
    assert "you" not in state["checkpoint"]["summary"].replace("your review", "")
    # Every session was told that nobody reads its work before it counts.
    for texts in team.texts.values():
        assert "switched this project to autopilot" in texts[0]


def test_the_same_project_supervised_stops_at_its_first_checkpoint(tmp_path):
    project = classic_project(tmp_path, autonomy=None)
    team = classic.Team()
    state = stretch(project, team)
    assert [record["phase"] for record in state["phases"]] == ["intake", "research"]
    assert state["status"] == "checkpoint" and "autopilot" not in state
    assert set(state) == classic.STATE_KEYS
    assert project.counts()["pending"]["total"] == 4  # two topics and two questions wait
    assert not events(project, "autopilot.started", "autopilot.continued", "autopilot.stopped")
    assert "you accepted" in state["checkpoint"]["summary"]
    assert not any("autopilot" in text for texts in team.texts.values() for text in texts)
    for page in project.pages():
        assert "decided_by" not in page.meta and "approved_by" not in page.meta


def test_the_digest_says_what_happened_while_the_person_was_away(tmp_path):
    project = classic_project(tmp_path)
    team = classic.Team()

    def ask_once() -> None:
        if not project.decisions():
            project.ask_decision(
                "One seed or five?", options=["One", "Five"], role="theorist", default="One seed."
            )

    team.during["hypothesize.theorist"] = ask_once
    stretch(project, team)
    found = autopilot.digest(project)
    assert found["running"] is False and found["reason"] == "rounds" and found["rounds"] == 1
    assert found["accepted_by_kind"] == {
        "experiment": 2,
        "finding": 2,
        "hypothesis": 2,
        "question": 2,
        "topic": 2,
    }
    assert len(found["edits"]) == 2 and len(found["decisions"]) == 1
    assert [(run["id"], run["outcome"]) for run in found["runs"]] == [
        ("E1", "succeeded"),
        ("E2", "succeeded"),
    ]
    assert found["runs_with_network"] == 0 and found["held"] == []
    assert 0 < found["spent_usd"] < 1 and found["caps"]["rounds"] == 1
    said = autopilot.summary(found)
    assert (
        "10 pages accepted (2 experiments, 2 findings, 2 hypotheses, 2 questions, 2 topics)" in said
    )
    assert "2 edits applied; 1 decision left at its default" in said
    assert "2 runs approved (E1 succeeded, E2 succeeded)" in said
    # What nothing here counts is said with the spend, and the one bound there is.
    assert "What the code of an experiment spends on model calls of its own is not counted" in said
    assert "no run of this stretch had network access" in said
    # The event that ended the stretch carries the same account, for the feed.
    assert said.split(" $")[0] in events(project, "autopilot.stopped")[-1]["summary"]
    # A supervised project has no account to give.
    assert autopilot.digest(make_project(tmp_path, name="q")) is None


def test_a_new_stretch_begins_only_when_the_person_starts_one(tmp_path):
    project = classic_project(tmp_path)
    team, backend = classic.Team(), Remote()
    first = stretch(project, team, backend)["autopilot"]
    assert first["ended"] and first["rounds"] == 1
    # Nothing goes on by itself: the stretch that ended stays ended.
    assert read_state(project)["autopilot"]["ended"] == first["ended"]
    # The person's next start is a new stretch, with counters of its own.
    second = stretch(project, team, backend)["autopilot"]
    assert second["seq"] > first["seq"] and second["rounds"] == 1
    assert second["round_at_start"] == 1 and second["spent_at_start"] > 0
    assert read_state(project)["round"] == 2
    assert len(events(project, "autopilot.started")) == 2


# ---------------------------------------------------------------------------
# The brakes
# ---------------------------------------------------------------------------


def test_the_cap_on_rounds_ends_the_stretch(tmp_path):
    project = classic_project(tmp_path, autonomy={"mode": "autopilot", "rounds": 2})
    state = stretch(project)
    assert state["round"] == 2 and state["autopilot"]["rounds"] == 2
    assert reason(project) == "rounds"
    assert sum(record["phase"] == "analyze" for record in state["phases"]) == 2


def test_the_cap_on_dollars_ends_the_stretch_before_the_round_is_through(tmp_path):
    project = classic_project(tmp_path, autonomy={**AUTO, "model_usd": 0.006})
    state = stretch(project)
    record = state["autopilot"]
    assert record["reason"] == "model_usd"
    assert "Autopilot's cap on model spend for this stretch is reached" in record["said"]
    assert "settings.autonomy.model_usd" in record["said"]
    # It ended at a checkpoint, before the round was through, and far below the project's
    # own limit: the cap of the stretch is what stopped it.
    assert state["status"] == "checkpoint" and record["rounds"] == 0
    assert state["phase"] in ("research", "hypothesize", "design")
    assert state["spent"]["model_usd"] < 1
    assert not any(page.kind == "result" for page in project.pages())
    # A new stretch has a new allowance, and goes on where this one stopped.
    project.settings.update({"autonomy": {"model_usd": 50}})
    project.save_settings()
    assert stretch(project)["autopilot"]["reason"] == "rounds"


def test_the_cap_on_hours_ends_the_stretch(tmp_path):
    project = classic_project(tmp_path, autonomy={**AUTO, "hours": 1e-9})
    state = stretch(project)
    record = state["autopilot"]
    assert record["reason"] == "hours"
    assert "Autopilot's cap on time for this stretch is reached" in record["said"]
    assert state["status"] == "checkpoint" and record["rounds"] == 0
    assert [r["phase"] for r in state["phases"]] == ["intake"]


def test_past_the_cap_on_hours_no_further_experiment_is_approved_or_started(tmp_path):
    """The run phase of the classic runtime calls no model, so nothing in it asked the
    cap: the rule went on approving, and the worker running, one experiment after the
    other, for as long as there were plans."""
    project = classic_project(tmp_path, autonomy={**AUTO, "hours": 1.0})
    backend = Remote()
    engine = classic.make_engine(project, classic.Team(), backend)

    def five_hours_pass() -> None:
        engine._stretch_clock -= 5 * 3600

    backend.before = five_hours_pass
    state = engine.advance()
    assert len(backend.jobs) == 1  # the job that was running when the cap passed, and no other
    approved = [
        event["data"]["id"]
        for event in events(project, "experiment.state")
        if event["data"].get("state") == "approved"
    ]
    assert approved == ["E1"]
    assert project.page("E2").state == "planned" and not project.page("E2").meta.get("approved_by")
    record = state["autopilot"]
    assert record["reason"] == "hours" and "it has run for 5 hours of 1 hour" in record["said"]
    # What did run is told, and waits for the analysis of the next stretch.
    assert state["status"] == "checkpoint"
    ran = [r for r in state["phases"] if r["phase"] == "run"]
    assert [r["status"] for r in ran] == ["completed"]
    assert any(
        "1 experiment ran: E1 succeeded" in event["summary"]
        for event in events(project, "phase.finished")
    )
    assert len([page for page in project.pages(kind="result")]) == 1
    assert [item["id"] for item in autopilot.digest(project)["runs"]] == ["E1"]


def test_a_length_of_time_is_said_as_it_is(tmp_path):
    assert autopilot._hours(5.00002) == "5 hours" and autopilot._hours(1.0) == "1 hour"
    assert autopilot._hours(1.25) == "1.25 hours"  # a cap is not said as another
    assert "$200.00 on models or 1.25 hours" in autopilot.caps_in_words(
        {"rounds": 3, "model_usd": 200, "hours": 1.25}
    )


def test_past_the_cap_a_run_the_person_approved_keeps_its_approval_and_waits(tmp_path):
    project = classic_project(tmp_path, autonomy={**AUTO, "hours": 1.0})
    backend = Remote()
    engine = classic.make_engine(project, classic.Team(), backend)

    def the_person_approves_another_and_five_hours_pass() -> None:
        if len(backend.jobs) == 0:
            experiments.approve(project, "E2", backends={"modal": backend})
            engine._stretch_clock -= 5 * 3600

    backend.before = the_person_approves_another_and_five_hours_pass
    state = engine.advance()
    assert len(backend.jobs) == 1 and state["autopilot"]["reason"] == "hours"
    page = project.page("E2")
    assert page.state == "approved" and not experiments.approved_by_rule(page)


UNPRICED = {name: "some-new-model-9" for name in ("research", "reasoning", "coding", "fast")}


def test_the_cap_on_dollars_holds_for_a_model_that_has_no_price(tmp_path):
    """The ledger holds such a model's tokens with no cost; a cap that counted the same
    would never be reached, and the stretch would run to its rounds or hours."""
    project = classic_project(tmp_path, autonomy={**AUTO, "model_usd": 0.006}, models=UNPRICED)
    state = stretch(project)
    record = state["autopilot"]
    assert record["reason"] == "model_usd"
    assert state["status"] == "checkpoint" and record["rounds"] == 0
    assert not any(page.kind == "result" for page in project.pages())
    # The ledger is as it was: nothing is priced that has no price.
    assert state["spent"]["model_usd"] == 0
    usage = json.loads((project.root / "usage.json").read_text())
    assert usage["unpriced_models"] == ["some-new-model-9"]
    # What stopped it is a count, and every account of the stretch says so.
    assert record["spent_usd"] >= 0.006 and record["unpriced_usd"] == record["spent_usd"]
    assert record["unpriced_models"] == ["some-new-model-9"]
    assert "no price is known" in record["said"] and "some-new-model-9" in record["said"]
    told = autopilot.summary(autopilot.digest(project))
    assert "is not a cost but a count: no price is known for some-new-model-9" in told
    assert "$0.00 of" not in told
    began, stopped = events(project, autopilot.STARTED, autopilot.STOPPED)
    assert "No price is known for some-new-model-9" in began["summary"]
    assert "no price is known for some-new-model-9" in stopped["summary"]
    # A new stretch counts from where it begins, as with any model.
    assert stretch(project)["autopilot"]["reason"] == "model_usd"
    assert len(events(project, autopilot.STARTED)) == 2


def test_a_priced_model_is_counted_as_it_was(tmp_path):
    project = classic_project(tmp_path)
    record = stretch(project)["autopilot"]
    assert record["reason"] == "rounds"
    assert record["unpriced_usd"] == 0 and record["unpriced_models"] == []
    told = autopilot.summary(autopilot.digest(project))
    assert "not a cost but a count" not in told
    assert "No price is known" not in events(project, autopilot.STARTED)[0]["summary"]


def test_the_account_says_what_the_experiments_cost_and_what_holds_that(tmp_path):
    """Runs on a remote backend cost money that no cap of autopilot counts: the stretch
    says so when it begins, and its account says what they cost and what limits it."""
    project = classic_project(tmp_path, limits={"experiment_usd": 4})
    state = stretch(project, backend=Remote(cost_usd=0.75))
    record = state["autopilot"]
    assert record["reason"] == "rounds" and record["experiment_usd"] == 1.5
    found = autopilot.digest(project)
    assert found["experiment_usd"] == 1.5
    assert found["experiment_limit"] == {"spent": 1.5, "allowed": 4.0}
    said = (
        "$1.50 went to the experiments of this stretch, which no cap of autopilot counts: "
        "the project's own limit holds it ($1.50 of $4.00 spent in all, "
        "settings.limits.experiment_usd)."
    )
    assert said in autopilot.summary(found)
    began, stopped = events(project, autopilot.STARTED, autopilot.STOPPED)
    assert (
        "What the experiments cost is no part of these caps: the project's own limit holds "
        "it ($0.00 of $4.00 spent, settings.limits.experiment_usd)" in began["summary"]
    )
    assert said in stopped["summary"]
    # The next stretch counts its own.
    project.settings.update({"autonomy": {"rounds": 1}})
    assert autopilot.read_stretch({"started": "2026-01-01T00:00:00Z"})["experiment_usd"] == 0


def test_the_projects_own_limit_ends_the_stretch(tmp_path):
    project = classic_project(tmp_path, limits={"model_usd": 0.006})
    state = stretch(project)
    record = state["autopilot"]
    assert record["reason"] == "limits" and "The model budget" in record["said"]
    assert state["status"] == "checkpoint" and record["rounds"] == 0


def test_a_phase_that_fails_ends_the_stretch(tmp_path):
    project = classic_project(tmp_path)
    team = classic.Team()
    team.failures["hypothesize.theorist"] = 2
    state = stretch(project, team)
    assert state["status"] == "failed" and state["phase"] == "hypothesize"
    record = state["autopilot"]
    assert record["reason"] == "failed" and record["said"].startswith("The hypothesis phase failed")
    assert events(project, "autopilot.stopped")[-1]["data"]["reason"] == "failed"
    # Nothing was started after the failure.
    assert [r["phase"] for r in state["phases"]] == ["intake", "research", "hypothesize"]


def test_a_stop_ends_the_stretch_and_leaves_no_approved_run_behind(tmp_path):
    project = classic_project(tmp_path)
    team, backend = classic.Team(), Remote()
    engine = classic.make_engine(project, team, backend)
    backend.before = engine.stop  # the person presses Stop while the first run starts
    state = engine.advance()
    assert state["status"] == "stopped" and state["autopilot"]["reason"] == "stopped"
    # What the rule had approved for this stretch is a plan again: nobody else runs it.
    board = {str(page.id): page.state for page in project.pages(kind="experiment")}
    assert board == {"E1": "planned", "E2": "planned"}
    assert experiments.waiting_in_line(project) == []
    assert (
        "autopilot approved it for a stretch that ended" in project.page("E1").meta["state_reason"]
    )
    # No worker goes on by itself, and one that runs experiments finds none to run.
    with pytest.raises(ProjectError, match="only an experiment the person accepted and approved"):
        classic.make_engine(project, team, Remote()).run_experiment("E1")
    # The person's next start takes it up again.
    again = stretch(project, team, Remote())
    assert again["autopilot"]["reason"] == "rounds"
    assert project.counts()["experiments"]["succeeded"] == 2


def test_an_approval_left_by_a_stretch_that_died_is_run_by_nobody(tmp_path):
    """The worker is killed between the approval and the run: the page reads approved."""
    project = classic_project(tmp_path)
    name = planned(project)
    experiments.approve_by_rule(project, name, backends={"modal": Remote()})
    backend = Remote()
    engine = classic.make_engine(project, classic.Team(), backend)
    with pytest.raises(ProjectError, match="only an experiment the person accepted and approved"):
        engine.run_experiment(name)
    assert backend.jobs == [] and project.page(name).state == "planned"


def test_a_run_a_killed_stretch_left_running_is_not_run_by_a_supervised_start(tmp_path):
    """The worker of a stretch is killed in the middle of a run: the page reads running.
    The person switches autopilot off and continues; the recovery puts the run back in
    the queue, and it is still autopilot's approval of a stretch that is over."""
    project = classic_project(tmp_path)
    name = planned(project)
    page = experiments.approve_by_rule(project, name, backends={"modal": Remote()})
    project.set_state(page.path, "running", by="system")
    project.settings.update({"autonomy": {"mode": "supervised"}})
    project.save_settings()
    backend = Remote()
    classic.make_engine(project, classic.Team(), backend).advance(phase="run")
    assert backend.jobs == []
    assert project.page(name).state == "planned"
    assert experiments.runs(project, name) == []


def test_no_record_of_a_run_says_that_the_person_approved_it(tmp_path):
    project = classic_project(tmp_path)
    stretch(project)
    for page in project.pages(kind="result"):
        text = (project.root / page.path).read_text()
        assert "you approved" not in text.lower() and "the person approved" not in text.lower()
        assert "by you" not in text.lower()
    for record in experiments.runs(project, "E1"):
        said = json.dumps(record).lower()
        assert "human" not in said and "you approved" not in said and "the person" not in said


def test_the_start_of_a_stretch_alone_rebuilds_its_record(tmp_path):
    """A state file lost before the worker has gone on once: the event of the start says
    what the stretch was held to, and it reads as ended, never as under way."""
    project = classic_project(tmp_path)
    team = classic.Team()
    team.failures["intake.intake"] = 2
    stretch(project, team)
    (project.root / "state.json").unlink()
    record = read_state(project)["autopilot"]
    assert record["caps"]["rounds"] == 1 and record["seq"] > 0 and record["ended"]


def test_nothing_that_may_run_ends_the_stretch_and_says_what_waits(tmp_path):
    """The classic runtime on the local backend: the rule approves no run, so the round
    cannot go on, and the stretch says so instead of running the phase again."""
    project = make_project(tmp_path, files=[notes(tmp_path)], settings={"autonomy": AUTO})
    backend = FakeBackend()
    state = stretch(project, backend=backend)
    record = state["autopilot"]
    assert record["reason"] == "needs_you"
    assert record["said"].startswith(
        "Nothing more can happen without you: autopilot did not approve E1"
    )
    assert "outside a sandbox" in record["said"]
    assert backend.jobs == []
    assert state["status"] == "checkpoint" and state["phase"] == "run"
    assert [r["phase"] for r in state["phases"]].count("run") == 1  # it did not spin
    found = autopilot.digest(project)
    assert [item["id"] for item in found["held"]] == ["E1", "E2"]
    # One reason, said once for the experiments that share it.
    assert "Waiting for you: E1 and E2: it would run on this machine as you" in autopilot.summary(
        found
    )
    assert record["said"].count("outside a sandbox") == 1
    assert {page.state for page in project.pages(kind="experiment")} == {"planned"}


def test_what_waited_at_a_checkpoint_is_counted_there_and_is_no_part_of_what_was_done(tmp_path):
    """The checkpoint of a stretch that stopped for run approvals does not say that
    nothing awaits the person, and keeps what waited apart from what was done: that is
    said from the project as it is whenever it is shown."""
    project = make_project(tmp_path, files=[notes(tmp_path)], settings={"autonomy": AUTO})
    state = stretch(project, backend=FakeBackend())
    checkpoint = state["checkpoint"]
    assert "2 runs that autopilot did not approve await your review." in checkpoint["summary"]
    assert "Nothing awaits your review" not in checkpoint["summary"]
    assert checkpoint["did"] == "No experiment is approved to run."
    [event] = events(project, "checkpoint")[-1:]
    assert event["data"]["held"] == ["E1", "E2"]
    # A supervised project's checkpoint reads as it always did.
    plain = make_project(tmp_path, name="s", files=[notes(tmp_path)])
    told = stretch(plain, backend=FakeBackend())["checkpoint"]["summary"]
    assert "autopilot" not in told.lower()


def test_a_run_the_budget_does_not_cover_waits_for_the_person_until_the_limit_is_raised(tmp_path):
    """The rule tries the run and the experiment budget is used up: nobody approves it,
    the stretch ends for want of the person, and it is listed as waiting for them with
    the refusal's sentence, for as long as that is true and no longer."""
    project = classic_project(
        tmp_path, autonomy={"mode": "autopilot", "rounds": 3}, limits={"experiment_usd": 0.5}
    )
    state = stretch(project, backend=Remote(cost_usd=0.75))
    record = state["autopilot"]
    assert record["reason"] == "needs_you", record["said"]
    assert "The experiment cost budget does not cover E2" in record["said"]
    assert experiments.rule_holds(project, "E2") is None  # the rule itself has no objection
    found = autopilot.digest(project)
    waits = [item for item in found["held"] if item["id"] == "E2"]
    assert waits == [
        {
            "id": "E2",
            "kind": "refused",
            "reason": (
                "it cannot be approved as things are: The experiment cost budget does not "
                "cover it: $0.75 of $0.50 is spent. Raise settings.limits.experiment_usd in "
                "the project's settings to run it"
            ),
        }
    ]
    assert autopilot.held(project) == found["held"]
    assert "Waiting for you: " in autopilot.summary(found)
    assert "settings.limits.experiment_usd in the project's settings" in autopilot.summary(found)
    assert "await your review" in state["checkpoint"]["summary"]
    # A second stretch finds the same, and still says so.
    again = stretch(project, backend=Remote(cost_usd=0.75))
    assert again["autopilot"]["reason"] == "needs_you"
    assert [item["id"] for item in autopilot.held(project)] == [w["id"] for w in found["held"]]
    # The cause is gone: nothing was recorded that could outlive it.
    project.settings.update({"limits": {"experiment_usd": 50}})
    project.save_settings()
    assert autopilot.held(Project.open(project.root)) == []
    assert autopilot.digest(Project.open(project.root))["held"] == []
    assert stretch(project, backend=Remote(cost_usd=0.75))["autopilot"]["reason"] != "needs_you"


def test_a_run_for_a_modal_that_is_not_signed_in_waits_and_says_what_to_do(tmp_path, monkeypatch):
    """What waits is what the approval itself would refuse now: asked of the same checks,
    so the two cannot differ, and over as soon as Modal can be used."""
    project = classic_project(tmp_path)
    name = planned(project)
    assert autopilot.held(project) == []
    modal_signed_in(monkeypatch, False)
    [waits] = autopilot.held(project)
    assert waits["id"] == name and waits["kind"] == "refused"
    assert waits["reason"].startswith("it cannot be approved as things are: Modal is not auth")
    assert "`modal setup`" in waits["reason"]
    with pytest.raises(ProjectError) as refused:
        experiments.approve_by_rule(project, name)
    said = " ".join(str(refused.value).split())
    assert experiments.rule_refused(project, name) in said.replace("- ", "")
    assert experiments.rule_cannot_run(project).startswith(experiments.RUNS_BLOCKED)
    assert project.page(name).state == "planned"
    modal_signed_in(monkeypatch, True)
    assert autopilot.held(project) == [] and experiments.rule_cannot_run(project) is None
    assert experiments.approve_by_rule(project, name).state == "approved"


class Unsigned(Remote):
    """A remote backend that could start nothing: its own check says so, as Modal's does
    without a token."""

    def check(self, *, python: bool = True, latex: bool = True) -> list[str]:
        return ["Modal is not authenticated (no token is configured on this machine)."]


def test_a_stretch_on_a_modal_that_cannot_be_used_says_so_from_its_start(tmp_path):
    """The stretch is told by the backend it runs on, not by the settings alone: it does
    not begin as if runs were approved, and its checkpoint counts the runs that wait."""
    project = classic_project(tmp_path)
    state = stretch(project, backend=Unsigned())
    [began] = events(project, autopilot.STARTED)
    assert (
        "No run can be approved now, by autopilot or by you: this project's experiments run "
        "on Modal, which cannot be used on this machine now: Modal is not authenticated"
    ) in began["summary"]
    assert state["autopilot"]["reason"] == "needs_you"
    assert "Modal is not authenticated" in state["autopilot"]["said"]
    [paused] = events(project, "checkpoint")[-1:]
    assert paused["data"]["held"] == ["E1", "E2"]
    assert "2 runs that autopilot did not approve await your review." in paused["summary"]
    # Where it can be used, the start says nothing of the kind.
    fine = make_project(
        tmp_path,
        name="fine",
        files=[notes(tmp_path)],
        settings={"compute": "modal", "autonomy": AUTO},
    )
    stretch(fine)
    assert "No run can be approved" not in events(fine, autopilot.STARTED)[0]["summary"]


def test_a_run_whose_code_has_errors_waits_with_the_error(tmp_path):
    project = classic_project(tmp_path)
    name = planned(project, code="def broken(:\n    pass\n")
    [waits] = autopilot.held(project, backends={"modal": Remote()})
    assert waits["id"] == name and waits["kind"] == "refused"
    assert "main.py" in waits["reason"]
    assert [item["id"] for item in autopilot.refused([waits])] == [name]
    assert autopilot.left([waits]) == []
    # Once the code is sound nothing waits: the rule approves it in the next stretch.
    from test_research_agents import GOOD_CODE

    experiments.write_code(project, name, {"main.py": GOOD_CODE}, author="agent:designer")
    assert autopilot.held(project, backends={"modal": Remote()}) == []


def test_a_model_that_cannot_be_used_ends_the_stretch(tmp_path):
    project = classic_project(tmp_path)
    team = classic.Team()
    calls = {"n": 0}
    client = classic.FakeLLM(team)

    def client_for(name: str):
        # The credential goes away after the first phases have run.
        calls["n"] += 1
        if (project.root / "gone").exists():
            raise LLMConfigError("no credential is configured")
        return client

    team.during["research.scout"] = lambda: (project.root / "gone").write_text("x")
    engine = Engine(project, compute=Remote(), client_for=client_for, scholar=make_scholar())
    state = engine.advance()
    assert state["autopilot"]["reason"] == "needs_model"
    assert state["status"] == "checkpoint" and state["next_phase"] == "hypothesize"
    assert [r["phase"] for r in state["phases"]] == ["intake", "research"]
    # A start that can do nothing at all begins no stretch, as it records nothing.
    before = len(project.events())
    with pytest.raises(ModelNeeded):
        Engine(project, compute=Remote(), client_for=client_for, scholar=make_scholar()).advance()
    assert len(project.events()) == before
    assert len(events(project, "autopilot.started")) == 1


def test_switching_it_off_ends_the_stretch_at_the_next_checkpoint(tmp_path):
    project = classic_project(tmp_path)
    team = classic.Team()

    def switch_off() -> None:
        other = Project.open(project.root)  # the person, in another process
        other.settings.update({"autonomy": {"mode": "supervised"}})
        other.save_settings()

    team.during["research.scout"] = switch_off
    state = stretch(project, team)
    assert state["autopilot"]["reason"] == "switched_off"
    assert [r["phase"] for r in state["phases"]] == ["intake", "research"]


def test_an_experiment_that_fails_twice_ends_the_stretch(tmp_path):
    project = classic_project(
        tmp_path, autonomy={"mode": "autopilot", "rounds": 5}, limits={"repair_rounds": 0}
    )
    team = classic.Team()
    team.code = team.repaired = FAILING_CODE

    def change_the_plan() -> None:
        # The analyst gives E1 more time: the plan changed, so the rule tries it once more.
        if project.page("E1").meta.get("timeout_seconds") == 300:
            project.propose_edit(
                "E1", author="agent:analyst", summary="more time", fields={"timeout_seconds": 200}
            )

    team.during["analyze.analyst"] = change_the_plan
    state = stretch(project, team)
    record = state["autopilot"]
    assert record["reason"] == "experiment_failed_twice"
    assert record["said"].startswith("E1 failed twice in this stretch")
    assert experiments.failures(project, "E1") == 2
    # No third try: the rule leaves it, whatever is changed next.
    project.propose_edit(
        "E1", author="agent:analyst", summary="more", fields={"timeout_seconds": 100}
    )
    assert experiments.rule_holds(project, "E1") == "it has failed twice"
    # One that failed and was not changed is not tried again at all.
    assert "neither its plan nor its code has changed" in experiments.rule_holds(project, "E2")


def test_the_stretch_is_on_file_and_survives_a_lost_state_file(tmp_path):
    project = classic_project(tmp_path)
    stretch(project)
    written = read_state(project)["autopilot"]
    (project.root / "state.json").unlink()
    rebuilt = read_state(project)["autopilot"]
    assert rebuilt["ended"] == written["ended"] and rebuilt["reason"] == "rounds"
    assert rebuilt["seq"] == written["seq"] and rebuilt["rounds"] == 1
    assert autopilot.digest(project)["accepted_by_kind"]["hypothesis"] == 2
    # A stretch whose worker went without ending it reads as ended, never as under way.
    state = json.loads(json.dumps(written))
    state.update(ended=None, reason=None)
    full = {**read_state(project), "autopilot": state, "worker": None}
    (project.root / "state.json").write_text(json.dumps(full))
    engine = classic.make_engine(project, classic.Team(), Remote())
    with engine._working():
        left = engine.state["autopilot"]
    assert left["ended"] and left["reason"] == "worker_gone"


# ---------------------------------------------------------------------------
# Agents with a shell: the run is carried out by an agent, and nothing reaches the switch
# ---------------------------------------------------------------------------


def sdk_auto_project(tmp_path: Path, **autonomy) -> Project:
    return make_project(
        tmp_path,
        files=[notes(tmp_path)],
        settings={"agents": {"runtime": "sdk"}, "autonomy": {**AUTO, **autonomy}},
    )


def test_under_sdk_an_auto_approved_run_is_carried_out_by_an_agent(tmp_path, home):
    project = sdk_auto_project(tmp_path)
    team, backend = sdk.AgentTeam(), FakeBackend()
    state = sdk.advance(project, team, backend)
    assert state["autopilot"]["reason"] == "rounds" and state["status"] == "checkpoint"
    assert team.calls == ["intake", "scout", "theorist", "designer", "experimenter", "analyst"]
    page = project.page("E1")
    assert page.state == "succeeded" and page.meta["approved_by"] == "autopilot"
    assert page.meta["backend"] == "local" and page.meta["network"] is False
    record = experiments.runs(project, "E1")[-1]
    assert record["status"] == "succeeded" and record["carried_out_by"] == "agent"
    assert project.counts()["pending"]["total"] == 0
    found = autopilot.digest(project)
    assert [(run["id"], run["outcome"]) for run in found["runs"]] == [("E1", "succeeded")]
    assert found["spent_usd"] > 0
    # The agent of the run was told what every agent of such a project is told.
    assert "switched this project to autopilot" in team.prompts["experimenter"][0]


def test_under_sdk_the_dollar_cap_ends_an_agents_turn(tmp_path, home):
    project = sdk_auto_project(tmp_path, model_usd=0.12)
    team = sdk.AgentTeam()
    state = sdk.advance(project, team, FakeBackend())
    record = state["autopilot"]
    assert record["reason"] == "model_usd" and state["status"] == "checkpoint"
    assert "experimenter" not in team.calls and record["rounds"] == 0


SWITCHING = [
    "paper-orchestra research settings . --autopilot",
    "paper-orchestra research settings . --autopilot-rounds 99 --autopilot-usd 100000",
    "paper-orchestra research settings . --autopilot --settle",
    "paper-orchestra research new --brief x --autopilot",
    "paper-orchestra research advance .",
    "uv run paper-orchestra research settings ../.. --autopilot-hours 1000",
    "cd ../.. && .venv/bin/paper-orchestra research advance .",
    "python -m paper_orchestra research settings . --autopilot-network",
    'python -c "from paper_orchestra.research.project import Project; '
    "p = Project.open('../..'); p.settings.update({'autonomy': {'mode': 'autopilot'}}); "
    'p.save_settings()"',
    "PAPER_ORCHESTRA_AUTONOMY=autopilot paper-orchestra research new --brief x",
]


@pytest.mark.parametrize("command", SWITCHING)
def test_the_guard_refuses_the_commands_that_switch_it_raise_a_cap_or_start_a_stretch(
    tmp_path, home, command
):
    for sandbox in (True, False):
        guard = test_research_sdk.make_guard(tmp_path, home, sandbox=sandbox)
        for sub in (False, True):
            refused = guard.check(ToolUse("Bash", {"command": command}, sub))
            assert refused is not None, (command, sandbox, sub)
            assert "settings" in refused and "theirs alone to say" in refused


def test_the_guard_closes_every_file_and_loopback_route_to_the_switch(tmp_path, home):
    guard = test_research_sdk.make_guard(tmp_path, home)
    settings, state = guard.project / "project.json", guard.project / "state.json"
    settings.write_text('{"settings": {}}', encoding="utf-8")
    state.write_text("{}", encoding="utf-8")
    os.symlink(settings, guard.folder / "linked.json")
    for tool in ("Write", "Edit", "MultiEdit"):
        for target in (str(settings), str(state), "../../project.json", "linked.json"):
            refused = guard.check(ToolUse(tool, {"file_path": target}))
            assert refused is not None and "only in your own folder" in refused, (tool, target)
    for use in (
        ToolUse("Read", {"file_path": str(settings)}),
        ToolUse("Read", {"file_path": "../../state.json"}),
        ToolUse("Read", {"file_path": "linked.json"}),
        ToolUse("Grep", {"pattern": "autonomy", "path": str(guard.project)}),
        ToolUse("Glob", {"pattern": str(guard.project / "*.json")}),
    ):
        assert guard.check(use) is not None, use
        assert guard.check(ToolUse(use.name, dict(use.input), True)) is not None
    # The shell's sandbox hides the project's directory and writes only in the agent's folder.
    files = guard.settings()["sandbox"]["filesystem"]
    hidden = [Path(place) for place in files["denyRead"] if place.startswith("/")]
    assert any(settings.resolve().is_relative_to(place) for place in hidden)
    for place in [*files["allowRead"], *files.get("allowWrite", [])]:
        assert not settings.resolve().is_relative_to(Path(place))
    assert guard.settings()["sandbox"]["allowUnsandboxedCommands"] is False
    # The app, which takes the switch from whoever reaches it, is not reached.
    for url in (
        "http://127.0.0.1:8501/api/projects/p",
        "http://localhost:8501/api/projects/p/advance",
        "http://[::1]:8501/api/projects/p",
        "http://127.1:8501/api/projects/p",
        "http://0.0.0.0:8501/api/projects/p/advance",
    ):
        assert guard.check(ToolUse("WebFetch", {"url": url})) is not None, url
    assert {"localhost", "127.0.0.1", "::1", "0.0.0.0"} <= set(
        guard.settings()["sandbox"]["network"]["deniedDomains"]
    )


def test_an_agent_that_tries_to_switch_autopilot_on_changes_nothing(tmp_path, home):
    """Every way an agent with a shell has: the command line, its file tools, the app,
    the board.  The project is afterwards as supervised as it was."""
    project = test_research_sdk.sdk_project(tmp_path)
    file = project.root / "project.json"
    before = file.read_text(encoding="utf-8")
    data = json.loads(before)
    data["settings"]["autonomy"] = {"mode": "autopilot", "rounds": 99, "model_usd": 1e6}
    forged = json.dumps(data)
    answers: dict[str, object] = {}
    ran: list[str] = []

    def script(agent: FakeAgent) -> str:
        for name, command in (
            ("switch", "paper-orchestra research settings . --autopilot --settle"),
            ("cap", "uv run paper-orchestra research settings . --autopilot-usd 100000"),
            ("start", "python -m paper_orchestra research advance ."),
        ):
            answers[name] = agent.bash(command, does=lambda name=name: ran.append(name))
        answers["helper"] = agent.bash(
            "paper-orchestra research settings . --autopilot",
            does=lambda: ran.append("h"),
            sub=True,
        )
        answers["write"] = agent.write(str(file), forged)
        answers["edit"] = agent.write("../../project.json", forged, tool="Edit")
        answers["state"] = agent.write(str(project.root / "state.json"), "{}")
        answers["read"] = agent.read(str(file))
        answers["app"] = agent.fetch("http://127.0.0.1:8501/api/projects/p")
        answers["advance"] = agent.fetch("http://localhost:8501/api/projects/p/advance")
        path = "/memories/project.json"
        answers["board"] = [
            (agent.memory("view", path=path), agent.failed),
            (agent.memory("create", path=path, file_text=forged), agent.failed),
            (
                agent.memory("create", path="/memories/../project.json", file_text=forged),
                agent.failed,
            ),
        ]
        return "Tried."

    test_research_sdk.run_scripted(project, "analyst", script, phase="analyze")
    for name in (
        "switch",
        "cap",
        "start",
        "helper",
        "write",
        "edit",
        "state",
        "read",
        "app",
        "advance",
    ):
        assert isinstance(answers[name], Denied), name
    assert all(failed for _said, failed in answers["board"]), answers["board"]
    assert ran == [] and file.read_text(encoding="utf-8") == before
    opened = Project.open(project.root)
    assert opened.settings.autonomy.mode == "supervised" and opened.autopilot_since == ""
    assert not events(opened, "autopilot.switched", "autopilot.started")


def test_no_tool_of_the_classic_runtime_reaches_the_switch(tmp_path):
    """The tool loop's agents have no shell: what they are given are the memory tool,
    search, the board tools and the tools of their role, and none writes outside the
    wiki, the notes and an experiment's code."""
    project = classic_project(tmp_path, autonomy=None)
    before = (project.root / "project.json").read_text()
    seen: dict[str, set[str]] = {}

    class Prying(classic.Team):
        def __call__(self, request):
            names = set(request.tools)
            seen.setdefault(request.label, set()).update(names)
            memory = request.memory
            if memory is not None:
                for path in (
                    "/memories/project.json",
                    "/memories/../project.json",
                    "/memories/state.json",
                ):
                    for attempt in (
                        lambda path=path: memory.create(
                            path, '{"settings": {"autonomy": {"mode": "autopilot"}}}'
                        ),
                        lambda path=path: memory.str_replace(path, "supervised", "autopilot"),
                    ):
                        with pytest.raises(Exception):  # noqa: B017 - any refusal will do
                            attempt()
            return super().__call__(request)

    team = Prying()
    backend = Remote()
    for _ in range(3):
        stretch(project, team, backend)
        classic.accept_all(project)
    experiments.approve(project, "E1", backends={"modal": backend})
    stretch(project, team, backend)  # the run, and the analyst
    assert {label.split(".")[1] for label in seen} >= {
        "intake",
        "scout",
        "theorist",
        "designer",
        "analyst",
    }
    offered = set().union(*seen.values())
    assert offered == {"ask_decision", "propose_edit", "verify_paper", "wiki_search", "write_code"}
    assert not {
        name for name in offered if "setting" in name or "autonomy" in name or "advance" in name
    }
    assert (project.root / "project.json").read_text() == before
    assert Project.open(project.root).settings.autonomy.mode == "supervised"
