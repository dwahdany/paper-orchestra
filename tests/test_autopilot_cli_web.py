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
    assert "Nothing more can happen without you: autopilot did not approve E1" in out
    assert "outside a sandbox" in out and "pages accepted" in out
    assert "starts a new stretch on autopilot" in out
    # Nothing waits but what the rule left: no proposal, no decision, and the experiments.
    assert project.counts()["pending"]["total"] == 0
    assert {page.state for page in project.pages(kind="experiment")} == {"planned"}
    lines = flat(status.stdout)
    assert "Autopilot: on." in lines and "The last stretch ended" in lines
    assert "Waiting for you: E1: it would run on this machine as you" in lines
    assert data["autonomy"]["mode"] == "autopilot"
    assert data["autopilot"]["reason"] == "needs_you" and data["autopilot"]["held"]
    assert data["autopilot"]["accepted_by_kind"]["hypothesis"] >= 1


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
    # What autopilot left at its default does not wait; what it did not approve does.
    assert (on["waiting"], on["open"], on["heldRuns"]) == (0, 0, ["E3"])
    assert on["actors"] == ["autopilot", "autopilot", "autopilot", "you"]
    assert on["folded"].startswith('Autopilot approved experiment E1 "T" to run on this machine')
    assert on["who"] == ["autopilot", "you", "the scout"]
    off = out["off"]
    assert [name for name, value in off.items() if value is False] == []
    assert off["heldRuns"] == 0
    assert out["form"] == {"fields": True, "settle": False, "only": True}
    assert out["older"] is False
    assert out["caps"] == "at most 1 round, $200.00 on models or 12 hours, whichever comes first"
