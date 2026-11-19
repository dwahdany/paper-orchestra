"""Autopilot as the person meets it: the command line and the API of the app.

The switch, the caps, "also settle what waits now", the refusal, where the stretch
stands and the account of what was decided without them; and that a stretch started
from the command line goes on in that worker, with only the model scripted.
"""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from paper_orchestra import cli
from paper_orchestra.research.engine import read_state
from paper_orchestra.research.experiments import RUNS_WAIT
from paper_orchestra.research.model import AUTONOMY_ENV, AUTOPILOT_NEEDS_SANDBOX
from paper_orchestra.research.project import Project
from paper_orchestra.web.app import create_app

runner = CliRunner()
BASE_URL = "http://127.0.0.1:8600"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No test leaves a signal handler behind, and none inherits a default mode."""
    monkeypatch.delenv(AUTONOMY_ENV, raising=False)
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    yield
    for number, handler in before.items():
        if handler is not None:
            signal.signal(number, handler)


@pytest.fixture(autouse=True)
def _modal_is_signed_in(monkeypatch):
    """What waits and what is promised are asked of the machine as it is; no test's
    answer may depend on whether this one has a Modal token."""
    try:
        from paper_orchestra.compute import modal_backend
    except ImportError:
        return
    monkeypatch.setattr(modal_backend, "modal_token_configured", lambda: True)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "research").mkdir()
    return tmp_path


def invoke(*args: str):
    return runner.invoke(cli.app, ["research", *args])


def flat(text: str) -> str:
    return " ".join(text.split())


def waiting_project(root: Path) -> Project:
    """A supervised project with a proposal, a suggested edit and a decision that wait."""
    project = Project.create(root, title="Warmup", brief="Does warmup matter?")
    page = project.create_page("hypothesis", "Warmup helps", "body", author="agent:theorist")
    project.propose_edit(
        page.path, author="agent:analyst", summary="shorter", fields={"summary": "short"}
    )
    project.ask_decision("Which?", options=["a", "b"], role="theorist", default="go with a")
    return project


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_settings_switches_autopilot_and_leaves_what_waits_unless_told_to_settle(workspace):
    project = waiting_project(workspace / "research" / "w")
    shown = invoke("settings", "w")
    assert "Autopilot: off (supervised: every proposal, edit, run and checkpoint waits" in flat(
        shown.stdout
    )
    switched = invoke(
        "settings", "w", "--autopilot", "--autopilot-rounds", "2", "--autopilot-usd", "5"
    )
    assert switched.exit_code == 0, switched.output
    said = flat(switched.stdout)
    assert "Changed autonomy.mode, autonomy.rounds, autonomy.model_usd for Warmup." in said
    assert "Autopilot: on." in said and "at most 2 rounds, $5.00 on models or 12 hours" in said
    assert "runs it approves get no network access" in said
    # What waited is left for the person, and the command says how to change that.
    assert "What waited before (3 to decide) is left for you" in said and "--settle" in said
    opened = Project.open(project.root)
    assert opened.settings.autonomy.mode == "autopilot" and opened.settings.autonomy.rounds == 2
    assert opened.counts()["pending"]["total"] == 2 and opened.counts()["decisions"]["open"] == 1

    settled = invoke("settings", "w", "--settle")
    assert settled.exit_code == 0, settled.output
    assert (
        "Autopilot settled what waited: 1 page accepted, 1 edit applied, 1 decision left at "
        "the default." in flat(settled.stdout)
    )
    opened = Project.open(project.root)
    assert opened.counts()["pending"]["total"] == 0 and opened.counts()["decisions"]["open"] == 0
    assert opened.page("H1").decided_by == "autopilot"

    as_json = json.loads(invoke("settings", "w", "--json").stdout)
    assert as_json["autonomy"] == {
        "mode": "autopilot",
        "rounds": 2,
        "model_usd": 5.0,
        "hours": 12.0,
        "network": False,
    }
    off = invoke("settings", "w", "--supervised")
    assert "Autopilot: off" in flat(off.stdout)
    assert Project.open(project.root).settings.autonomy.mode == "supervised"


def test_settle_without_autopilot_and_autopilot_without_a_sandbox_are_refused(workspace):
    project = waiting_project(workspace / "research" / "w")
    refused = invoke("settings", "w", "--settle")
    assert refused.exit_code == 2 and "Autopilot decides only in a project" in flat(refused.output)
    assert Project.open(project.root).counts()["pending"]["total"] == 2
    invoke("settings", "w", "--runtime", "sdk", "--no-sandbox")
    refused = invoke("settings", "w", "--autopilot")
    assert refused.exit_code == 2
    # One plain sentence, and nothing was changed.
    assert AUTOPILOT_NEEDS_SANDBOX in flat(refused.output)
    assert "not valid" not in refused.output
    assert Project.open(project.root).settings.autonomy.mode == "supervised"
    bad = invoke("settings", "w", "--autopilot-hours", "0")
    assert bad.exit_code == 2 and "settings.autonomy.hours is 0.0" in flat(bad.output)


def test_new_takes_the_switch_and_the_environment_gives_the_default(workspace, monkeypatch):
    made = invoke("new", "--brief", "An idea.", "--dir", "a", "--no-run", "--autopilot")
    assert made.exit_code == 0, made.output
    said = flat(made.stdout)
    assert "Autopilot: on." in said and "at most 3 rounds, $200.00 on models or 12 hours" in said
    assert "it goes on by itself until a cap of autopilot is reached" in said
    assert Project.open(workspace / "a").settings.autonomy.mode == "autopilot"
    plain = invoke("new", "--brief", "An idea.", "--dir", "b", "--no-run")
    assert "Autopilot" not in plain.stdout and "stops at the first checkpoint" in flat(plain.stdout)
    monkeypatch.setenv(AUTONOMY_ENV, "autopilot")
    invoke("new", "--brief", "An idea.", "--dir", "c", "--no-run")
    assert Project.open(workspace / "c").settings.autonomy.mode == "autopilot"
    invoke("new", "--brief", "An idea.", "--dir", "d", "--no-run", "--supervised")
    assert Project.open(workspace / "d").settings.autonomy.mode == "supervised"
    # The environment is asked when a project is created, and never of one that exists.
    assert Project.open(workspace / "b").settings.autonomy.mode == "supervised"


def test_a_stretch_started_from_the_command_line_goes_on_in_that_worker(workspace, monkeypatch):
    """The whole path of a worker, offline: the command, the engine, the sessions, the
    store; only the model is scripted.  One command goes from the idea through research,
    hypotheses and design to the run phase, where the classic runtime's experiments, which
    would run on this machine as the person, are the one thing that waits for them."""
    from paper_orchestra.llm.router import SCRIPTED_CLIENT_ENV, reset_clients

    monkeypatch.syspath_prepend(str(Path(__file__).parent / "e2e"))
    monkeypatch.setenv("PAPER_ORCHESTRA_E2E_PAUSE", "0")
    sys.modules.pop("scripted_model", None)
    monkeypatch.setenv(SCRIPTED_CLIENT_ENV, "scripted_model:client")
    reset_clients()
    try:
        started = invoke(
            "new",
            "--brief",
            "Small batches need warmup under Adam.",
            "--dir",
            "w",
            "--autopilot",
            "--autopilot-rounds",
            "1",
        )
        status = invoke("status", "w")
        data = json.loads(invoke("status", "w", "--json").stdout)
    finally:
        reset_clients()
        sys.modules.pop("scripted_model", None)

    assert started.exit_code == 0, started.output
    project = Project.open(workspace / "w")
    state = read_state(project)
    ran = [record["phase"] for record in state["phases"]]
    assert ran[:4] == ["intake", "research", "hypothesize", "design"] and ran[-1] == "run"
    assert state["autopilot"]["reason"] == "needs_you" and state["status"] == "checkpoint"
    out = flat(started.stdout)
    assert "The last stretch ended" in out
    assert "Nothing more can happen without you. Without you:" in out
    assert "Autopilot did not approve E1 and E2: it would run on this machine as you" in out
    assert "2 runs that autopilot did not approve await your review." in out
    assert "Nothing awaits your review" not in out
    assert "outside a sandbox" in out and "pages accepted" in out
    assert "starts a new stretch on autopilot" in out
    # What was promised before the first agent started is what happened: no run of this
    # project is approved without the person, and nothing said that one would be.
    assert "Runs on this machine are not approved without you" in out
    assert "runs approved and checkpoints passed" not in out
    # What is next is the one thing that lets a new stretch do anything.
    approve = (
        "Next: approve the runs autopilot left for you (E1 and E2) with "
        "`paper-orchestra research approve w E1`; then `paper-orchestra research advance w` "
        "starts a new stretch on autopilot"
    )
    assert approve in out
    # Nothing waits but what the rule left: no proposal, no decision, and the experiments.
    assert project.counts()["pending"]["total"] == 0
    assert {page.state for page in project.pages(kind="experiment")} == {"planned"}
    lines = flat(status.stdout)
    assert "Autopilot: on." in lines and "The last stretch ended" in lines
    # The runs the rule left wait for the person, and no line says that nothing does.
    assert "Waiting for you: 2 runs autopilot left for you (E1 and E2)" in lines
    assert "Nothing is waiting for you." not in lines
    assert approve in lines
    # Why they wait is said once, with them, and what to do stands right under it:
    # before the settings, the account of the stretch and what the checkpoint did.
    assert lines.count("it would run on this machine as you") == 1
    assert lines.count("outside a sandbox") == 1 and lines.count("did not approve") == 0
    assert "E1 and E2: it would run on this machine as you" in lines
    assert lines.count("Next:") == 1
    assert lines.index("Waiting for you:") < lines.index("Next:") < lines.index("Autopilot: on.")
    assert "Runs on this machine are not approved without you (why is said" in lines
    assert data["autonomy"]["mode"] == "autopilot"
    assert data["autopilot"]["reason"] == "needs_you" and data["autopilot"]["held"]
    assert data["autopilot"]["accepted_by_kind"]["hypothesis"] >= 1


def test_the_switch_promises_runs_only_where_the_rule_approves_them(workspace):
    """Under the classic runtime on the local backend the rule approves no run that
    needs no GPU; with Modal as the backend it does, and what those runs cost is held
    by the project's own limit, which the switch names."""
    local = flat(
        invoke("new", "--brief", "An idea.", "--dir", "a", "--no-run", "--autopilot").stdout
    )
    assert "Pages are accepted, edits applied and checkpoints passed without you." in local
    assert f"Runs on this machine are not approved without you: {RUNS_WAIT}." in local
    assert "runs approved" not in local
    project = Project.open(workspace / "a")
    project.settings.update({"compute": "modal"})
    project.save_settings()
    remote = flat(invoke("status", "a").stdout)
    assert "runs approved and checkpoints passed without you" in remote
    assert "not approved without you" not in remote
    # The advice names a path that exists: the backend is an option of the two commands.
    assert "Modal as the compute backend (`--compute modal`)" in local
    made = invoke(
        "new", "--brief", "Another.", "--dir", "m", "--no-run", "--autopilot", "--compute", "modal"
    )
    assert made.exit_code == 0, made.output
    assert Project.open(workspace / "m").settings.compute == "modal"
    assert "runs approved and checkpoints passed without you" in flat(made.stdout)
    back = invoke("settings", "m", "--compute", "local")
    assert back.exit_code == 0, back.output
    assert Project.open(workspace / "m").settings.compute == "local"
    assert "Changed compute for" in back.stdout
    assert "Runs on this machine are not approved without you" in flat(back.stdout)
    for said in (local, remote):
        assert (
            "What the experiments cost is no part of these caps: the project's own limit "
            "holds it ($5.00, settings.limits.experiment_usd)." in said
        )


def test_the_promise_of_runs_is_not_made_where_modal_is_not_signed_in(workspace, monkeypatch):
    """ "Runs approved" is true of the settings and of this machine, or it is not said."""
    import test_research_autopilot as offline

    made = invoke(
        "new", "--brief", "An idea.", "--dir", "m", "--no-run", "--autopilot", "--compute", "modal"
    )
    assert "runs approved and checkpoints passed without you" in flat(made.stdout)
    offline.modal_signed_in(monkeypatch, False)
    said = flat(invoke("settings", "m").stdout)
    assert "runs approved" not in said
    assert (
        "No run can be approved now, by autopilot or by you: this project's experiments run "
        "on Modal, which cannot be used on this machine now: Modal is not authenticated"
    ) in said
    assert "`modal setup`" in said and "Once it can be used, runs are approved." in said


def test_settle_says_what_becomes_of_the_planned_experiments_in_this_project(workspace):
    """On a project whose runs the rule does not approve, settling does not promise them."""
    project = waiting_project(workspace / "p")
    said = flat(invoke("settings", str(project.root), "--autopilot", "--settle").stdout)
    assert "Runs on this machine are not approved without you" in said
    assert "Experiments that were planned before still wait for your approval" in said
    assert "planned before are approved and run in the next stretch" not in said
    remote = waiting_project(workspace / "r")
    remote.settings.update({"compute": "modal"})
    remote.save_settings()
    said = flat(invoke("settings", str(remote.root), "--autopilot", "--settle").stdout)
    assert "planned before are approved and run in the next stretch." in said


def test_status_keeps_the_reason_of_the_settings_where_what_waits_has_another(workspace):
    """A plan from before the switch waits for that; why no run on this machine is
    approved is then said with the settings, in full, and not "said above"."""
    import test_research_autopilot as offline

    project = Project.create(workspace / "p", title="Warmup", brief="Does warmup matter?")
    name = offline.planned(project)
    project.review(name, "accept")
    assert invoke("settings", str(project.root), "--autopilot").exit_code == 0
    lines = flat(invoke("status", str(project.root)).stdout)
    assert f"{name}: it was planned before autopilot was switched on" in lines
    assert f"Runs on this machine are not approved without you: {RUNS_WAIT}." in lines
    assert "why is said with what waits" not in lines


def test_status_counts_a_run_the_approval_refuses_and_says_what_to_do(workspace):
    """The experiment budget is used up: the run waits for the person, is said to, and
    what is next is not "advance starts a new stretch" alone."""
    import test_research_autopilot as offline

    project = offline.classic_project(
        workspace, autonomy={"mode": "autopilot", "rounds": 3}, limits={"experiment_usd": 0.5}
    )
    offline.stretch(project, backend=offline.Remote(cost_usd=0.75))
    lines = flat(invoke("status", str(project.root)).stdout)
    assert "Nothing is waiting for you." not in lines
    assert "Waiting for you: 3 runs that cannot be approved as things are (E2, E3 and E4)" in lines
    # One reason for the three, said once.
    assert "E2, E3 and E4: it cannot be approved as things are: The experiment cost" in lines
    assert lines.count("Raise settings.limits.experiment_usd") == 1
    assert (
        "Next: deal with what keeps E2, E3 and E4 from being approved (said with what waits "
        "for you); autopilot approves them in the stretch after that; then `paper-orchestra "
        "research advance"
    ) in lines
    assert "research approve" not in lines  # nobody can approve it as things are
    data = json.loads(invoke("status", str(project.root), "--json").stdout)
    assert {item["kind"] for item in data["autopilot"]["held"]} == {"refused"}
    # With the limit raised nothing waits, and the next stretch is what is next.
    project.settings.update({"limits": {"experiment_usd": 50}})
    project.save_settings()
    lines = flat(invoke("status", str(project.root)).stdout)
    assert "Nothing is waiting for you." in lines and "cannot be approved" not in lines
    assert lines.rstrip().endswith("(it goes on by itself until a cap is reached).")


# ---------------------------------------------------------------------------
# The API of the app
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    app = create_app(root, static_dir=tmp_path / "no-frontend", stop_workers_on_exit=True)
    with TestClient(app, base_url=BASE_URL) as client:
        client.workspace = root
        yield client


def url(project: Project) -> str:
    listed = [project.root.name]
    return f"/api/projects/{listed[0]}"


def pid_of(client, project: Project) -> str:
    rows = client.get("/api/projects").json()["projects"]
    return next(row["id"] for row in rows if row["dir"] == project.root.name)


def test_the_view_carries_autonomy_and_patch_switches_it(client):
    project = waiting_project(client.workspace / "w")
    pid = pid_of(client, project)
    view = client.get(f"/api/projects/{pid}").json()
    # Always shown, also while the project's file leaves it out.
    assert view["settings"]["autonomy"] == {
        "mode": "supervised",
        "rounds": 3,
        "model_usd": 200.0,
        "hours": 12.0,
        "network": False,
    }
    assert view["autopilot"] == {
        "mode": "supervised",
        "refused": None,
        "since": "",
        "digest": None,
        "held": [],
        "runs_wait": RUNS_WAIT,
        "runs_blocked": None,
    }
    assert client.get("/api/defaults").json()["autonomy"]["mode"] == "supervised"
    rows = client.get("/api/projects").json()["projects"]
    assert rows[0]["autonomy"] == "supervised"

    patched = client.patch(
        f"/api/projects/{pid}", json={"settings": {"autonomy": {"mode": "autopilot", "rounds": 2}}}
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["settings"]["autonomy"]["mode"] == "autopilot" and "settled" not in body
    assert body["autopilot"]["mode"] == "autopilot" and body["autopilot"]["since"]
    # What waited is left alone: the count is what it was.
    assert body["counts"]["pending"]["total"] == 2 and body["counts"]["decisions"]["open"] == 1
    assert client.get("/api/projects").json()["projects"][0]["autonomy"] == "autopilot"

    settled = client.patch(f"/api/projects/{pid}", json={"settle": True}).json()
    assert settled["settled"]["accepted"] == ["wiki/hypotheses/H1-warmup-helps.md"]
    assert len(settled["settled"]["applied"]) == 1 and len(settled["settled"]["defaulted"]) == 1
    assert settled["counts"]["pending"]["total"] == 0
    assert settled["counts"]["decisions"]["open"] == 0
    listed = client.get(f"/api/projects/{pid}/decisions").json()["decisions"]
    assert [item["state"] for item in listed] == ["defaulted"]
    # The person's later answer holds, and is their first word on it.
    did = listed[0]["id"]
    answered = client.post(f"/api/projects/{pid}/decisions/{did}/answer", json={"option": 2})
    assert answered.status_code == 200 and answered.json()["decision"]["state"] == "answered"


def test_patch_refuses_what_the_settings_refuse_and_changes_nothing(client):
    project = waiting_project(client.workspace / "w")
    pid = pid_of(client, project)
    not_on = client.patch(f"/api/projects/{pid}", json={"settle": True})
    assert not_on.status_code == 400 and not_on.json()["error"]["code"] == "not_allowed"
    assert Project.open(project.root).counts()["pending"]["total"] == 2
    for change in ({"mode": "on"}, {"rounds": 0}, {"model_usd": -1}, {"cap": 5}):
        refused = client.patch(f"/api/projects/{pid}", json={"settings": {"autonomy": change}})
        assert refused.status_code == 422, (change, refused.text)
    client.patch(
        f"/api/projects/{pid}", json={"settings": {"agents": {"runtime": "sdk", "sandbox": False}}}
    )
    view = client.get(f"/api/projects/{pid}").json()
    assert view["autopilot"]["refused"] == AUTOPILOT_NEEDS_SANDBOX
    refused = client.patch(
        f"/api/projects/{pid}", json={"settings": {"autonomy": {"mode": "autopilot"}}}
    )
    assert (
        refused.status_code == 422 and AUTOPILOT_NEEDS_SANDBOX in refused.json()["error"]["message"]
    )
    after = Project.open(project.root)
    assert after.settings.autonomy.mode == "supervised" and after.counts()["pending"]["total"] == 2


def test_the_view_gives_the_account_of_the_last_stretch(client, tmp_path):
    """The digest of a stretch that a worker ran, as the overview shows it."""
    import test_research_autopilot as offline

    project = offline.classic_project(client.workspace)
    offline.stretch(project)
    pid = pid_of(client, project)
    view = client.get(f"/api/projects/{pid}").json()
    digest = view["autopilot"]["digest"]
    assert digest["reason"] == "rounds" and digest["running"] is False
    assert digest["accepted_by_kind"]["hypothesis"] == 2 and len(digest["edits"]) == 2
    assert [run["outcome"] for run in digest["runs"]] == ["succeeded", "succeeded"]
    assert digest["uncounted"].startswith("What the code of an experiment spends")
    assert view["state"]["autopilot"]["reason"] == "rounds"
    assert view["counts"]["pending"]["total"] == 0
    # The feed's events name autopilot as the actor.
    activity = client.get(f"/api/projects/{pid}/activity?limit=200").json()
    actors = {
        event["agent"] for event in activity["events"] if event["type"].startswith("autopilot.")
    }
    assert actors == {"autopilot"}


def test_the_library_counts_the_runs_autopilot_left_and_the_app_says_which_runs_wait(client):
    import test_research_autopilot as offline
    from test_research_agents import FakeBackend, make_project

    project = make_project(
        client.workspace,
        files=[offline.notes(client.workspace)],
        settings={"autonomy": offline.AUTO},
    )
    offline.stretch(project, backend=FakeBackend())
    rows = client.get("/api/projects").json()["projects"]
    row = next(row for row in rows if row["dir"] == project.root.name)
    # No proposal waits, and two runs do: the row counts them as the project's own page does.
    assert row["pending"] == 0 and row["held"] == 2
    view = client.get(f"/api/projects/{row['id']}").json()
    assert [item["id"] for item in view["autopilot"]["held"]] == ["E1", "E2"]
    assert view["autopilot"]["runs_wait"] == RUNS_WAIT
    assert view["autopilot"]["digest"]["experiment_limit"] == {"spent": 0.0, "allowed": 5.0}
    # A supervised project has none, and one whose runs the rule approves is told nothing.
    remote = make_project(
        client.workspace,
        name="r",
        files=[offline.notes(client.workspace)],
        settings={"compute": "modal", "autonomy": offline.AUTO},
    )
    rows = client.get("/api/projects").json()["projects"]
    other = next(row for row in rows if row["dir"] == remote.root.name)
    assert other["held"] == 0
    assert client.get(f"/api/projects/{other['id']}").json()["autopilot"]["runs_wait"] is None
    # What the composer says before a project exists, by runtime and backend.
    waits = client.get("/api/defaults").json()["autopilot_runs_wait"]
    assert waits["classic"] == {**dict.fromkeys(waits["classic"]), "local": RUNS_WAIT}
    assert not any(waits["sdk"].values())


def test_the_library_row_follows_a_change_of_the_settings_that_writes_no_event(client):
    """What the rule approves depends on the settings; the row's count is taken anew
    when they change, also where the change adds nothing to the events."""
    import test_research_autopilot as offline
    from test_research_agents import FakeBackend, make_project

    project = make_project(
        client.workspace,
        files=[offline.notes(client.workspace)],
        settings={"autonomy": offline.AUTO},
    )
    offline.stretch(project, backend=FakeBackend())

    def row():
        rows = client.get("/api/projects").json()["projects"]
        return next(row for row in rows if row["dir"] == project.root.name)

    assert row()["held"] == 2
    before = len(project.events())
    patched = client.patch(f"/api/projects/{row()['id']}", json={"settings": {"compute": "modal"}})
    assert patched.status_code == 200, patched.text
    assert len(Project.open(project.root).events()) == before  # nothing in the events
    assert row()["held"] == 0
    assert client.get(f"/api/projects/{row()['id']}").json()["autopilot"]["held"] == []


def test_the_app_counts_a_run_the_approval_refuses_and_does_not_promise_runs(client, monkeypatch):
    """Modal is not signed in: the runs of a project on Modal wait for the person, in the
    row, the view and the digest, each with what to do, and the switch and the composer
    do not say "runs approved"."""
    import test_research_autopilot as offline

    project = offline.classic_project(client.workspace)
    name = offline.planned(project)
    offline.modal_signed_in(monkeypatch, False)
    rows = client.get("/api/projects").json()["projects"]
    row = next(row for row in rows if row["dir"] == project.root.name)
    assert row["held"] == 1 and row["pending"] == 0
    view = client.get(f"/api/projects/{row['id']}").json()["autopilot"]
    [waits] = view["held"]
    assert (waits["id"], waits["kind"]) == (name, "refused") and "modal setup" in waits["reason"]
    assert view["runs_wait"] is None
    assert view["runs_blocked"].startswith("this project's experiments run on Modal, which")
    blocked = client.get("/api/defaults").json()["autopilot_runs_blocked"]
    assert blocked["local"] is None and "Modal is not authenticated" in blocked["modal"]
    offline.modal_signed_in(monkeypatch, True)
    assert client.get(f"/api/projects/{row['id']}").json()["autopilot"]["runs_blocked"] is None
    assert client.get("/api/defaults").json()["autopilot_runs_blocked"] == {
        "local": None,
        "modal": None,
    }


# ---------------------------------------------------------------------------
# The client: the switch, the digest, and who is said to have decided
# ---------------------------------------------------------------------------

CLIENT_SCRIPT = r"""
import { projectStore, page, checkpoint, said, times, ago } from './harness.js';
import { app } from './js/core.js';
import { overviewView } from './js/overview.js';
import { whoOf, foldRuns } from './js/shell.js';
import { who } from './js/page.js';
import { defaultedPanel, decidedDecisions, openDecisions, waitingCount, pageDecisions } from './js/decisions.js';
import { heldRuns, capsWords } from './js/autopilot.js';
import { rowState, startView, composer } from './js/home.js';

const autonomy = (mode) => ({ mode, rounds: 2, model_usd: 5, hours: 0.5, network: false });
const digest = { started: ago(900), ended: ago(60), running: false, reason: 'rounds', why: 'the cap on rounds was reached',
  said: 'The cap on rounds was reached: 2 rounds completed in this stretch, of at most 2 (settings.autonomy.rounds).',
  rounds: 2, caps: autonomy('autopilot'), spent_usd: 1.25, seconds: 840,
  accepted: [{ path: 'a', kind: 'hypothesis', id: 'H1' }, { path: 'b', kind: 'finding', id: 'F1' }, { path: 'c', kind: 'hypothesis', id: 'H2' }],
  accepted_by_kind: { hypothesis: 2, finding: 1 }, edits: [{ proposal: 'ep-1', path: 'a' }], decisions: [{ decision: 'd-1', page: null }],
  runs: [{ id: 'E1', path: 'wiki/experiments/E1.md', network: false, outcome: 'succeeded' }, { id: 'E2', path: 'x', network: true, outcome: 'failed' }],
  runs_with_network: 1, held: [{ id: 'E3', reason: 'it would run on this machine as you, outside a sandbox <b>' }],
  uncounted: 'What the code of an experiment spends on model calls of its own is not counted by anything here' };
const e1 = page('experiment', 'E1', { state: 'succeeded', meta: { decided_by: 'autopilot', decided_at: ago(800), approved_by: 'autopilot', approved_at: ago(700) } });
const e3 = page('experiment', 'E3', { state: 'planned' });
const decision = { id: 'd-1', question: 'One seed or five?', options: [{ label: 'One', meaning: '' }, { label: 'Five', meaning: '' }], default: 'One seed.', state: 'defaulted', defaulted_at: ago(500), created: ago(500), raised_by: { role: 'theorist', round: 1 }, answers: [], answer: null, page: e1.path };
const render = (store) => {
  const root = { innerHTML: '', querySelectorAll: () => [], querySelector: () => null };
  app.route = { name: 'project', pid: 'p-1', view: 'overview', ref: '', sel: '', sub: '', q: {} };
  overviewView.render(root);
  return root.innerHTML;
};
const out = {};
const on = projectStore({ pages: [e1, e3], experiments: [Object.assign({ runs: 1 }, e1), Object.assign({ runs: 0 }, e3)],
  project: { settings: { limits: { model_usd: 25, experiment_usd: 5 }, autonomy: autonomy('autopilot') }, state: checkpoint('analyze', 'hypothesize'), next_step: said({ will_run: ['hypothesize'], then: 'design' }),
    autopilot: { mode: 'autopilot', refused: null, since: '', digest, held: digest.held } } });
on.decisions = [decision];
on.activity = [
  { seq: 1, ts: ago(700), type: 'page.updated', agent: 'autopilot', summary: 'Autopilot set how experiment E1 "T" may run: on local, for up to 300 s, without network access.', data: { by: 'autopilot', approval: ['backend'], id: 'E1' }, refs: [] },
  { seq: 2, ts: ago(699), type: 'experiment.state', agent: 'autopilot', summary: 'Experiment E1 "T" is now approved.', data: { by: 'autopilot', state: 'approved', id: 'E1' }, refs: [] },
  { seq: 3, ts: ago(600), type: 'review.decided', agent: 'autopilot', summary: "Autopilot applied the analyst's edit to hypothesis H1.", data: { by: 'autopilot', target: 'proposal' }, refs: [] },
  { seq: 4, ts: ago(500), type: 'review.decided', agent: '', summary: 'You accepted hypothesis H2.', data: { by: 'human', target: 'page' }, refs: [] },
];
const html = render(on);
const section = html.slice(html.indexOf('class="ov-sec ov-auto"'), html.indexOf('Waiting for you<span>'));
out.on = {
  section: html.includes('data-autopilot="on"'), first: html.indexOf('ov-auto') < html.indexOf('Waiting for you<span>'),
  caps: section.includes('at most 2 rounds, $5.00 on models or 30 minutes, whichever comes first'),
  noNetwork: section.includes('Runs it approves get no network access.'),
  away: section.includes('While you were away'), why: section.includes('The cap on rounds was reached: 2 rounds completed'),
  accepted: section.includes('<b>3 pages accepted</b>') && section.includes('1 finding, 2 hypotheses'),
  edits: section.includes('<b>1 edit applied</b>'), decisions: section.includes("<b>1 decision</b> went on at the agents' default"),
  runs: section.includes('<b>2 runs approved</b>') && section.includes('E1 succeeded') && section.includes('E2 failed'),
  spent: section.includes('<b>$1.25</b> of $5.00'), uncounted: section.includes('is not counted by anything here; 1 run had network access'),
  held: section.includes('Waiting for you:') && section.includes('outside a sandbox'), escaped: !section.includes('sandbox <b>') && section.includes('&lt;b&gt;'),
  sub: html.includes('Autopilot stopped ') && html.includes('Continue starts a new stretch.'),
  heldRow: html.includes('autopilot did not approve it'),
  waiting: waitingCount(on), open: openDecisions(on).length, heldRuns: heldRuns(on).map((h) => h.id),
  defaulted: String(defaultedPanel(on)).includes('Went on at the default') && String(defaultedPanel(on)).includes('default: One seed.'),
  notYours: String(decidedDecisions(on)) === '', onPage: String(pageDecisions(on, e1.path)).includes('Went on at the default (autopilot)'),
  actors: on.activity.map(whoOf), folded: foldRuns(on.activity).map((ev) => ev.summary)[0],
  who: [who('autopilot'), who('human'), who('agent:scout')],
};
/* off, and never ran unattended: the switch is among the settings, and says what "on" would mean nowhere */
const off = projectStore({ pages: [e3], experiments: [Object.assign({ runs: 0 }, e3)],
  project: { settings: { limits: { model_usd: 25, experiment_usd: 5 }, autonomy: autonomy('supervised') }, state: checkpoint('design', 'run'), next_step: said({ will_run: ['run'], then: 'analyze' }),
    autopilot: { mode: 'supervised', refused: 'Autopilot is refused for this project: its agents run with the sandbox off.', since: '', digest: null, held: [] } } });
const plain = render(off);
out.off = { section: plain.includes('data-autopilot="off"'), after: plain.indexOf('ov-auto') > plain.indexOf('Waiting for you<span>'),
  says: plain.includes('<b>Off.</b> Every proposal, edit, run and checkpoint waits for you.'), refused: plain.includes('Autopilot is refused for this project'),
  noDigest: !plain.includes('While you were away'), yours: plain.includes('runs only after you approve it'), heldRuns: heldRuns(off).length, switch: plain.includes('Switch it on') };
/* the form: the server's values, nothing of the client's own */
app.ui.autopilotEdit = { on: true, rounds: '2', model_usd: '5', hours: '0.5', network: false, settle: false, error: null, saving: false };
const form = render(off);
out.form = { fields: ['auto-rounds', 'auto-model_usd', 'auto-hours', 'auto-network', 'auto-on'].every((id) => form.includes('id="' + id + '"')), settle: form.includes('Also settle what waits now'), only: form.includes('Only you can switch this on or raise a cap') };
/* a server that says nothing of autopilot (an older one) shows nothing of it */
app.ui.autopilotEdit = null;
out.older = render(projectStore({ pages: [], experiments: [], project: { state: checkpoint('design', 'run'), next_step: said({ will_run: ['run'], then: 'analyze' }) } })).includes('ov-auto');
out.caps = capsWords({ rounds: 1, model_usd: 200, hours: 12 });
/* a project whose runs the rule does not approve: the switch does not promise them, and gives the server's reason */
const WAIT = 'an experiment that needs no GPU would run on this machine as you <i>';
const local = projectStore({ pages: [e3], experiments: [Object.assign({ runs: 0 }, e3)],
  project: { settings: { limits: { model_usd: 25, experiment_usd: 5 }, autonomy: autonomy('autopilot') }, state: checkpoint('run', 'run'), next_step: said({ will_run: ['run'], then: 'analyze' }),
    autopilot: { mode: 'autopilot', refused: null, since: '', held: digest.held, runs_wait: WAIT,
      digest: Object.assign({}, digest, { reason: 'needs_you', why: 'nothing more can happen without you', runs: [], spent_usd: 0.02, unpriced_usd: 0.02, unpriced_models: ['some-new-model-9'], experiment_usd: 0, experiment_limit: { spent: 0, allowed: 5 } }) } } });
const waits = render(local), nowAt = waits.indexOf('class="auto-now"'), now = waits.slice(nowAt, waits.indexOf('</p>', nowAt));
out.local = { wait: now.includes('Runs on this machine are not approved without you: an experiment that needs no GPU'), escaped: now.includes('&lt;i&gt;') && !now.includes('you <i>'),
  noPromise: !now.includes('runs approved'), limit: now.includes("the project's own limit holds it ($5.00)"),
  sub: waits.includes('Approve the runs it left for you; Continue then starts a new stretch.') && !waits.includes('. Continue starts a new stretch.'),
  unpriced: waits.includes('<b>$0.02</b> of that is a count, not a cost: no price is known for some-new-model-9'), noExperiments: !waits.includes('data-auto="experiments"') };
/* where it approves them: the promise, what holds the cost of the runs, and what they cost in the stretch */
out.remote = { promise: section.includes('runs approved and checkpoints passed without you'), limit: section.includes("What the experiments cost is no part of these caps: the project's own limit holds it ($5.00)"),
  experiments: section.includes('data-auto="experiments"') && section.includes('which no cap of autopilot counts'), priced: !section.includes('data-auto="unpriced"') };
/* the library: the runs autopilot left are counted with what waits, and such a row is not "ready to continue" */
const row = { status: 'checkpoint', round: 2, phase: 'run', pending: 0, held: 3, counts: { pending: { total: 0 }, decisions: { open: 0 } } };
out.rows = [rowState(row)[2], rowState(Object.assign({}, row, { pending: 2 }))[2], rowState(Object.assign({}, row, { held: 0 }))[2]];
/* the composer: the tick promises what will happen, by the runtime and the backend the person chose */
const start = (compute) => {
  app.ui.defaults = { data: { settings: { compute: 'local', limits: { model_usd: 25 }, models: {}, agents: { runtime: 'classic', sandbox: true } }, autonomy: autonomy('supervised'),
    autopilot_runs_wait: { classic: { local: WAIT, modal: null }, sdk: { local: null, modal: null } }, compute_backends: ['local', 'modal'], efforts: [], agent_runtimes: ['classic', 'sdk'], agent_credentials: [] }, loading: false, error: null, at: Date.now() };
  const s = composer(); s.autopilot = true; s.adv = true; s.compute = compute;
  const root = { innerHTML: '', querySelectorAll: () => [], querySelector: () => null };
  startView.render(root, { q: {} });
  const at = root.innerHTML.indexOf('id="opt-autopilot"');
  return root.innerHTML.slice(at, root.innerHTML.indexOf('</label>', at));
};
const onLocal = start(''), onModal = start('modal');
out.composer = { wait: onLocal.includes('Runs on this machine are not approved without you: an experiment that needs no GPU'), noPromise: !onLocal.includes('runs approved'),
  escaped: onLocal.includes('&lt;i&gt;'), promise: onModal.includes('runs approved and checkpoints passed without you') && !onModal.includes('not approved') };
console.log(JSON.stringify(out));
"""


def test_the_client_shows_the_switch_the_digest_and_who_decided(tmp_path):
    import test_web_static

    if test_web_static.shutil.which("node") is None:
        pytest.skip("node is not installed")
    out = test_web_static._client(tmp_path, CLIENT_SCRIPT)
    on = out["on"]
    failed = [name for name, value in on.items() if value is False]
    assert not failed, failed
    # What autopilot left at its default does not wait; what it did not approve does,
    # and is counted with what waits.
    assert (on["waiting"], on["open"], on["heldRuns"]) == (1, 0, ["E3"])
    assert on["actors"] == ["autopilot", "autopilot", "autopilot", "you"]
    assert on["folded"].startswith('Autopilot approved experiment E1 "T" to run on this machine')
    assert on["who"] == ["autopilot", "you", "the scout"]
    off = out["off"]
    assert [name for name, value in off.items() if value is False] == []
    assert off["heldRuns"] == 0
    assert out["form"] == {"fields": True, "settle": False, "only": True}
    assert out["older"] is False
    assert out["caps"] == "at most 1 round, $200.00 on models or 12 hours, whichever comes first"
    for name in ("local", "remote", "composer"):
        assert [key for key, value in out[name].items() if value is False] == [], name
    assert out["rows"] == ["3 waiting for you", "5 waiting for you", "ready to continue"]


WAITING_SCRIPT = r"""
import { projectStore, page, checkpoint, said, ago } from './harness.js';
import { app } from './js/core.js';
import { overviewView } from './js/overview.js';
import { inboxView } from './js/inbox.js';
import { waitingSentence } from './js/shell.js';
import { waitingCount } from './js/decisions.js';
import { autopilotSub, heldAdvice, leftRuns, refusedRuns } from './js/autopilot.js';
import { startView, composer } from './js/home.js';

const autonomy = (mode) => ({ mode, rounds: 2, model_usd: 5, hours: 0.5, network: false });
const LEFT = { id: 'E1', kind: 'rule', reason: 'it would run on this machine as you, outside a sandbox' };
const REFUSED = { id: 'E2', kind: 'refused', reason: 'it cannot be approved as things are: The experiment cost budget does not cover it: $5.00 of $5.00 is spent. Raise settings.limits.experiment_usd <b>' };
const e1 = page('experiment', 'E1', { state: 'planned' }), e2 = page('experiment', 'E2', { state: 'planned' });
const digest = (reason, why, held) => ({ started: ago(900), ended: ago(60), running: false, reason, why, said: why, rounds: 1, caps: autonomy('autopilot'), spent_usd: 1, seconds: 800,
  accepted: [], accepted_by_kind: {}, edits: [], decisions: [], runs: [], runs_with_network: 0, held, uncounted: 'x' });
const store = (held, reason, why, extra) => projectStore({ pages: [e1, e2], experiments: [Object.assign({ runs: 0 }, e1), Object.assign({ runs: 0 }, e2)],
  project: { settings: { limits: { model_usd: 25, experiment_usd: 5 }, autonomy: autonomy('autopilot') }, state: checkpoint('run', 'run'), next_step: said({ will_run: ['run'], then: 'analyze' }),
    autopilot: Object.assign({ mode: 'autopilot', refused: null, since: '', digest: digest(reason, why, held), held }, extra || {}) } });
if (typeof window === 'undefined') globalThis.window = {};   /* the Inbox asks how wide the sheet is */
const show = (view, s, name) => {
  app.store = s;
  const root = { innerHTML: '', querySelectorAll: () => [], querySelector: () => null };
  app.route = { name: 'project', pid: 'p-1', view: name, ref: '', sel: '', sub: '', q: {} };
  view.render(root, app.route);
  return root.innerHTML;
};
const out = {};
/* a run the approval refuses: counted and listed as a run the rule left is, with what to do, and never "approve it" */
const refused = store([REFUSED], 'needs_you', 'nothing more can happen without you');
const e = { working: false };
const inbox = show(inboxView, refused, 'inbox'), over = show(overviewView, refused, 'overview');
out.refused = { count: waitingCount(refused), kinds: [leftRuns(refused).length, refusedRuns(refused).length],
  headline: waitingSentence(refused), sub: autopilotSub(refused, e), advice: heldAdvice(refused),
  inboxHead: inbox.includes('1 waiting for you') && !inbox.includes('Nothing is waiting'),
  inboxRow: inbox.includes('data-held="E2"') && inbox.includes('cannot be approved</span>') && inbox.includes('Raise settings.limits.experiment_usd'),
  inboxTitle: inbox.includes('Runs autopilot did not approve') && !inbox.includes('Runs autopilot left for you'),
  escaped: inbox.includes('&lt;b&gt;') && !inbox.includes('experiment_usd <b>'),
  /* (e) with a run waiting, the Inbox does not say that the checkpoint can go on */
  still: inbox.includes('Every proposal has a decision.') && inbox.includes('1 run above still waits for you: autopilot did not approve it.') && !inbox.includes('The checkpoint can go on'),
  overview: over.includes('data-held="E2"') && over.includes('autopilot did not approve it · it cannot be approved as things are') };
/* a run the rule left: as before */
const left = store([LEFT], 'needs_you', 'nothing more can happen without you');
const leftInbox = show(inboxView, left, 'inbox');
out.left = { headline: waitingSentence(left), sub: autopilotSub(left, e),
  inbox: leftInbox.includes('Runs autopilot left for you') && leftInbox.includes('not approved</span>') && !leftInbox.includes('The checkpoint can go on') };
/* both, after a stretch that ended at a cap: Continue alone is not what is said */
const both = store([LEFT, REFUSED], 'rounds', 'the cap on rounds was reached');
out.both = { headline: waitingSentence(both), sub: autopilotSub(both, e) };
/* nothing held: the checkpoint can go on, as it always could */
const none = store([], 'rounds', 'the cap on rounds was reached');
out.none = { headline: waitingSentence(none), sub: autopilotSub(none, e), inbox: show(inboxView, none, 'inbox').includes('Every proposal has a decision. The checkpoint can go on.') };
/* the switch where Modal cannot be used: no promise of runs */
const blocked = show(overviewView, store([REFUSED], 'needs_you', 'nothing more can happen without you', { runs_blocked: 'this project runs on Modal, which is not signed in <i>' }), 'overview');
const nowAt = blocked.indexOf('class="auto-now"'), now = blocked.slice(nowAt, blocked.indexOf('</p>', nowAt));
out.blocked = { says: now.includes('No run can be approved now, by autopilot or by you: this project runs on Modal'), noPromise: !now.includes('runs approved and'), escaped: now.includes('&lt;i&gt;') };
/* (d) the composer: the route says who decides as the tick stands */
const start = (ticked, compute) => {
  app.ui.defaults = { data: { settings: { compute: 'local', limits: { model_usd: 25 }, models: {}, agents: { runtime: 'classic', sandbox: true } }, autonomy: autonomy('supervised'),
    autopilot_runs_wait: { classic: { local: 'unread code would run as you', modal: null }, sdk: { local: null, modal: null } }, autopilot_runs_blocked: { local: null, modal: compute === 'blocked' ? 'Modal is not signed in' : null },
    compute_backends: ['local', 'modal'], efforts: [], agent_runtimes: ['classic', 'sdk'], agent_credentials: [] }, loading: false, error: null, at: Date.now() };
  const s = composer(); s.autopilot = ticked; s.adv = true; s.compute = compute === 'blocked' ? 'modal' : compute;
  const root = { innerHTML: '', querySelectorAll: () => [], querySelector: () => null };
  startView.render(root, { q: {} });
  const html = root.innerHTML, hints = {};
  html.split('class="smap-name">').slice(1).forEach((part) => { hints[part.slice(0, part.indexOf('<'))] = part.slice(part.indexOf('smap-hint">') + 11, part.indexOf('</div>', part.indexOf('smap-hint">'))); });
  const at = html.indexOf('id="opt-autopilot"');
  return { hypotheses: hints.Hypotheses, experiments: hints.Experiments, lede: html.includes('nothing is accepted, and nothing runs, without you'), tick: html.slice(at, html.indexOf('</label>', at)) };
};
const offC = start(false, ''), onLocal = start(true, ''), onModal = start(true, 'modal'), onBlocked = start(true, 'blocked');
out.composer = { off: [offC.hypotheses, offC.experiments, offC.lede], local: [onLocal.hypotheses, onLocal.experiments, onLocal.lede], modal: [onModal.hypotheses, onModal.experiments, onModal.lede],
  blocked: [onBlocked.hypotheses, onBlocked.experiments], blockedTick: onBlocked.tick.includes('No run can be approved now, by autopilot or by you: Modal is not signed in') && !onBlocked.tick.includes('runs approved and') };
console.log(JSON.stringify(out));
"""


def test_the_client_counts_refused_runs_and_says_who_decides_as_things_stand(tmp_path):
    import test_web_static

    if test_web_static.shutil.which("node") is None:
        pytest.skip("node is not installed")
    out = test_web_static._client(tmp_path, WAITING_SCRIPT)
    for name in ("refused", "left", "none", "blocked", "composer"):
        assert [key for key, value in out[name].items() if value is False] == [], name
    refused = out["refused"]
    assert refused["count"] == 1 and refused["kinds"] == [0, 1]
    assert refused["headline"] == (
        "1 run cannot be approved as things are, by autopilot or by you: what waits for you "
        "says what to change."
    )
    assert "Nothing is waiting" not in refused["headline"]
    assert refused["sub"].endswith(
        ": nothing more can happen without you. 1 run cannot be approved as things are: what "
        "waits for you says what to change, and autopilot approves it in the stretch after that."
    )
    assert "Approve the runs" not in refused["sub"]
    assert out["left"]["headline"] == (
        "1 run waits for your approval: autopilot did not approve it."
    )
    assert out["left"]["sub"].endswith(
        "Approve the runs it left for you; Continue then starts a new stretch."
    )
    assert out["both"]["headline"].startswith("1 run waits for your approval")
    assert "1 run cannot be approved as things are" in out["both"]["headline"]
    assert (
        "the cap on rounds was reached. Approve the runs it left for you. 1 run cannot"
        in (out["both"]["sub"])
    )
    assert out["none"]["headline"] == "Nothing is waiting for you."
    assert out["none"]["sub"].endswith(
        "the cap on rounds was reached. Continue starts a new stretch."
    )
    composer = out["composer"]
    assert composer["off"] == ["you accept or reject", "you approve each run", True]
    assert composer["local"] == ["accepted without you", "you approve each run", False]
    assert composer["modal"] == ["accepted without you", "approved without you", False]
    assert composer["blocked"] == ["accepted without you", "Modal is not set up"]
