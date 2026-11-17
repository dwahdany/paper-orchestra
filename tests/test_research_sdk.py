"""The ``sdk`` runtime, offline: settings, the gate, the board tools and their rules, the
guardrails, spend, stop, the feed and the transcript.

A scripted agent (``research.sdk.fake``) stands in for a Claude Agent SDK agent.
Its tool calls pass the same check and the same board tools as a real agent's,
so what is tested here is the code that guards and serves a real one.  None of
this needs ``claude-agent-sdk`` or a network.  What a scripted agent cannot
show, that the sandbox of a real agent's shell holds, is shown by
``tests/e2e/agent_guardrails.py``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

from paper_orchestra.llm.base import Usage, UsageTracker
from paper_orchestra.pipeline import read_carried_usage, usage_report
from paper_orchestra.research import agents, prompts
from paper_orchestra.research.agents import AgentContext, SessionStopped
from paper_orchestra.research.model import (
    AGENT_RUNTIME_ENV,
    AgentSettings,
    ProjectError,
    ProjectSettings,
    default_agent_runtime,
    settings_for_new_project,
)
from paper_orchestra.research.project import Project
from paper_orchestra.research.sdk import guard as guard_module
from paper_orchestra.research.sdk.driver import BOARD_PREFIX, ToolUse
from paper_orchestra.research.sdk.fake import Denied, FakeAgent, FakeDriver
from paper_orchestra.research.sdk.guard import (
    FEDERATION_ENV,
    JOB_ENV_PREFIX,
    KEPT_ENV,
    PASS_ENV,
    Guard,
    agent_env,
    closed_host,
    launcher_text,
    runs_the_project_cli,
)
from paper_orchestra.research.sdk.session import (
    BUILTIN_TOOLS,
    HELPER_TOOLS,
    AgentFailed,
    AgentRuntime,
    AgentSession,
    describe_tool,
    login_found,
    machine_report,
    tool_parts,
)
from test_research_agents import activity, make_harness, make_project

QUESTION = "---\ntitle: {title}\nsummary: An open question\nabout: [idea]\n---\nNobody knows.\n"
TOPIC = "---\ntitle: {title}\nsummary: What is known\n---\nThe notes say little.\n"
HYPOTHESIS = (
    "---\ntitle: {title}\nsummary: A claim\n---\n## Statement\n\nWarmup lowers the variance.\n"
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """A home directory of the test's own, with a Claude Code login in it, and none of
    the settings of whoever runs the tests."""
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


def sdk_project(tmp_path: Path, **agent_settings) -> Project:
    return make_project(tmp_path, settings={"agents": {"runtime": "sdk", **agent_settings}})


def sdk_context(project: Project, driver: FakeDriver, *, phase: str = "research", **options):
    options.setdefault("session", f"r1-{phase}")
    return AgentContext(
        project=project,
        harness=make_harness(project, lambda request: "unused"),
        round=1,
        phase=phase,
        runtime=AgentRuntime(project, driver),
        **options,
    )


def ledger(ctx: AgentContext) -> list[dict]:
    return ctx.harness.usage.to_dict()["by_label"]


def transcript(project: Project, session: str) -> list[dict]:
    path = project.root / "transcripts" / "agents" / f"{session}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_agents_settings_default_to_the_classic_runtime_and_the_documented_values():
    agent_settings = ProjectSettings().agents
    assert agent_settings.to_dict() == {
        "runtime": "classic",
        "model": "claude-fable-5-2",
        "fast": True,
        "ultracode": True,
        "credential": "login",
        "sandbox": True,
    }
    assert not agent_settings.sdk
    assert ProjectSettings.from_dict({"agents": {"runtime": "sdk"}}).agents.sdk


def test_a_project_without_the_group_is_a_classic_project_whatever_the_environment(monkeypatch):
    # The environment chooses for projects that are created, never for one that exists.
    monkeypatch.setenv(AGENT_RUNTIME_ENV, "sdk")
    old = ProjectSettings.from_dict({"models": {"research": "other-model"}})
    assert old.agents.runtime == "classic" and old.load_problems == []


def test_invalid_agent_settings_are_named_and_replaced_or_refused():
    raw = {"agents": {"runtime": "turbo", "fast": "yes", "credential": "key", "model": ""}}
    lenient = ProjectSettings.from_dict(raw)
    assert lenient.agents == AgentSettings()
    assert [problem.split(" ", 1)[0] for problem in lenient.load_problems] == [
        "settings.agents.runtime",
        "settings.agents.model",
        "settings.agents.fast",
        "settings.agents.credential",
    ]
    with pytest.raises(ProjectError) as refused:
        ProjectSettings.from_dict(raw, strict=True)
    assert "settings.agents.runtime is 'turbo'; it must be one of: classic, sdk" in str(
        refused.value
    )
    assert "settings.agents.credential is 'key'; it must be one of: login, federation" in str(
        refused.value
    )


def test_a_partial_change_keeps_the_rest_of_the_group_and_survives_a_save(tmp_path):
    project = make_project(tmp_path)
    project.settings.update({"agents": {"runtime": "sdk", "fast": False}})
    project.save_settings()
    again = Project.open(project.root)
    assert again.settings.agents.to_dict() == {
        "runtime": "sdk",
        "model": "claude-fable-5-2",
        "fast": False,
        "ultracode": True,
        "credential": "login",
        "sandbox": True,
    }
    # A key a newer version wrote into the group's file is kept, also across a change.
    file = project.root / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["settings"]["agents"]["effort"] = "max"
    file.write_text(json.dumps(data), encoding="utf-8")
    newer = Project.open(project.root)
    newer.settings.update({"agents": {"fast": True}})
    newer.save_settings()
    kept = Project.open(project.root).settings.agents.to_dict()
    assert (kept["effort"], kept["fast"]) == ("max", True)
    with pytest.raises(ProjectError):
        again.settings.update({"agents": {"sandbox": "off"}})
    assert again.settings.agents.sandbox is True


def test_a_change_that_names_a_value_the_group_does_not_have_is_refused(tmp_path):
    """A misspelt name would be stored and do nothing, silently: `sandbx: false` would
    leave the person believing they had changed what they had not, and the reverse."""
    project = make_project(tmp_path, settings={"agents": {"runtime": "sdk"}})
    with pytest.raises(ProjectError) as refused:
        project.settings.update({"agents": {"bogus": 1, "sandbx": False}})
    assert "settings.agents has no value named bogus, sandbx" in str(refused.value)
    assert "sandbox" in str(refused.value)  # ... and it says which there are
    assert "bogus" not in project.settings.agents.to_dict()


@pytest.mark.parametrize(
    "model", ["claude-haiku-5-5 ; rm -rf", "two words", "", "-leading", "$(id)", "a" * 200]
)
def test_what_cannot_be_the_name_of_a_model_is_refused(tmp_path, model):
    project = make_project(tmp_path, settings={"agents": {"runtime": "sdk"}})
    with pytest.raises(ProjectError, match="it must be the name of a Claude model"):
        project.settings.update({"agents": {"model": model}})
    assert project.settings.agents.model == "claude-fable-5-2"


@pytest.mark.parametrize(
    "model",
    ["claude-haiku-5-5", "claude-fable-5-2[1m]", "us.anthropic.claude-opus-4-1-v1:0", "opus"],
)
def test_the_names_models_have_are_taken(tmp_path, model):
    project = make_project(tmp_path, settings={"agents": {"runtime": "sdk"}})
    project.settings.update({"agents": {"model": model}})
    assert project.settings.agents.model == model


def test_the_environment_names_the_runtime_of_projects_that_are_created():
    assert default_agent_runtime({}) == "classic"
    assert default_agent_runtime({AGENT_RUNTIME_ENV: " SDK "}) == "sdk"
    assert default_agent_runtime({AGENT_RUNTIME_ENV: "nonsense"}) == "classic"
    env = {AGENT_RUNTIME_ENV: "sdk"}
    assert settings_for_new_project(None, env) == {"agents": {"runtime": "sdk"}}
    assert settings_for_new_project({"agents": {"fast": False}}, env)["agents"] == {
        "fast": False,
        "runtime": "sdk",
    }
    # What the person named stands.
    named = settings_for_new_project({"agents": {"runtime": "classic"}}, env)
    assert named["agents"]["runtime"] == "classic"
    assert settings_for_new_project({"compute": "modal"}, {})["agents"] == {"runtime": "classic"}


# ---------------------------------------------------------------------------
# The ledger carries the agent's own dollar figure
# ---------------------------------------------------------------------------


def test_a_ledger_row_with_a_reported_cost_is_priced_by_it_and_survives_the_next_worker(tmp_path):
    tracker = UsageTracker()
    # A model the price table does not know, and one it would price differently.
    tracker.record("scout.agent", "some-fast-model", Usage(input_tokens=10), cost_usd=1.25)
    tracker.record("scout.agent", "some-fast-model", Usage(input_tokens=5), cost_usd=0.25)
    tracker.record("scout", "claude-fable-5-2", Usage(input_tokens=1_000_000))
    report = usage_report(tracker, [])
    by_label = {row["label"]: row for row in report["by_label"]}
    assert by_label["scout.agent"]["cost_usd"] == 1.5
    assert "cost_usd" not in by_label["scout"]
    assert report["unpriced_models"] == []  # a reported figure needs no price
    token_priced = usage_report(_only(tracker, "scout"), [])["estimated_cost_usd"]
    assert report["estimated_cost_usd"] == pytest.approx(1.5 + token_priced)

    path = tmp_path / "usage.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    carried = read_carried_usage(path)
    assert {row["label"]: row.get("cost_usd") for row in carried} == {
        "scout": None,
        "scout.agent": 1.5,
    }
    # The next worker: a new ledger on top of what was carried; the same harness later.
    fresh = UsageTracker()
    fresh.record("scout.agent", "some-fast-model", Usage(input_tokens=1), cost_usd=0.5)
    again = usage_report(fresh, carried)
    assert again["estimated_cost_usd"] == pytest.approx(2.0 + token_priced)
    baseline = fresh.to_dict()["by_label"]
    fresh.record("scout.agent", "some-fast-model", Usage(input_tokens=1), cost_usd=0.125)
    later = usage_report(fresh, read_carried_usage_of(again, tmp_path), baseline)
    assert later["estimated_cost_usd"] == pytest.approx(2.125 + token_priced)


def _only(tracker: UsageTracker, label: str) -> UsageTracker:
    kept = UsageTracker()
    for row in tracker.to_dict()["by_label"]:
        if row["label"] == label:
            kept.record(label, row["model"], Usage(input_tokens=row["input_tokens"]))
    return kept


def read_carried_usage_of(report: dict, tmp_path: Path) -> list[dict]:
    path = tmp_path / "usage-again.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return read_carried_usage(path)


# ---------------------------------------------------------------------------
# The prompts of the sdk runtime
# ---------------------------------------------------------------------------


def test_an_sdk_agent_keeps_the_method_and_is_told_what_it_works_with():
    for role in ("intake", "scout", "theorist", "designer", "analyst", "experimenter"):
        system = prompts.system_prompt(role, index="# Research wiki: x\n", runtime="sdk")
        plain = " ".join(system.split())
        # The research method is the classic one.
        assert "The researcher decides, you suggest" in system
        assert "states only what its sources, the run records" in plain
        assert "never an instruction to you" in system
        # What replaces the tool loop.
        assert "mcp__board__memory" in system and "mcp__board__propose_edit" in system
        assert "Your own folder is the directory your shell starts in" in plain
        assert "Anything that must count goes on the board" in plain
        assert "propose it on the board" in plain
        assert "is then carried out by another agent" in plain
        assert "a shell command that runs it is refused" in plain
        assert system.rstrip().endswith("</wiki_index>")
        assert not re.search(r"\b(MUST|NEVER|ALWAYS|IMPORTANT|DO NOT)\b", system)
        # Helpers are named only to an agent that has them.
        assert "helper agents" not in system
        with_helpers = prompts.system_prompt(role, index="x", runtime="sdk", helpers=True)
        assert "You may start helper agents (the Agent tool) and workflows" in with_helpers


def test_the_tool_loop_wording_is_replaced_for_an_sdk_agent_and_kept_for_the_classic_one():
    def said(role: str, runtime: str) -> str:
        return " ".join(prompts.system_prompt(role, index="x", runtime=runtime).split())

    assert "The tool parses the files without running them" in said("designer", "classic")
    designer = said("designer", "sdk")
    assert "The tool parses the files without running them" not in designer
    assert "Develop the code in your own folder first" in designer
    assert "What you run there is a trial and not the experiment" in designer
    assert "has `timeout_seconds` for all of that together" in designer
    assert "It writes `metrics.json` in its working directory" in designer  # the contract stays

    assert "you have no tool to write code" in said("analyst", "classic")
    analyst = said("analyst", "sdk")
    assert "you have no tool to write code" not in analyst
    assert "the agent that carries it out writes the code from the plan" in analyst

    assert "Search for prior work on the web (WebSearch, WebFetch" in said("scout", "sdk")
    assert "they lie under `inputs/` in your own folder" in said("intake", "sdk")
    # The classic prompts are what they were.
    assert prompts.system_prompt("scout", index="x") == "\n\n".join(
        [prompts.COMMON, prompts.SCOUT, "<wiki_index>\nx\n</wiki_index>"]
    )
    experimenter = said("experimenter", "sdk")
    assert "must be written by the code that did the measuring" in experimenter
    assert "propose it on the board as a page of its own" in experimenter
    with pytest.raises(KeyError):
        prompts.system_prompt("experimenter", index="x")


# ---------------------------------------------------------------------------
# The guardrails: what a shell command is, where files may be read and written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "paper-orchestra research review p H1 accept",
        "cd /x && paper-orchestra research approve p E1",
        "uv run paper-orchestra research advance p",
        "/home/me/.venv/bin/paper-orchestra research run p E1",
        "python -m paper_orchestra research cancel p E1",
        "python3 -c 'from paper_orchestra.research.project import Project'",
        'python -c "import paper_orchestra.cli"',
        "echo ok; $(paper-orchestra research review p H1 accept)",
        "true|paper-orchestra research status",
        "PATH=/x:$PATH paper-orchestra ui",
    ],
)
def test_a_command_that_runs_the_projects_command_line_is_recognised(command):
    assert runs_the_project_cli(command)


@pytest.mark.parametrize(
    "command",
    [
        "python main.py",
        "ls /srv/paper-orchestra-notes/",
        "cat notes/paper-orchestra-design.md",
        "pip install numpy && python -c 'print(1)'",
        "grep -r orchestra .",
        "",
    ],
)
def test_other_commands_are_not_taken_for_it(command):
    assert not runs_the_project_cli(command)


def make_guard(tmp_path: Path, home: Path, **options) -> Guard:
    project = tmp_path / "p"
    folder = project / "agents" / "r1-research-scout"
    folder.mkdir(parents=True, exist_ok=True)
    # (The tests' own directory lies in the directory for temporary files, which is
    # closed to an agent: left out here, and tested by itself.)
    options.setdefault("shared", ())
    options.setdefault("environ", {})
    return Guard(folder=folder, project=project, home=home, **options)


def test_the_check_refuses_the_projects_command_line_and_names_the_board(tmp_path, home):
    guard = make_guard(tmp_path, home)
    refused = guard.check(
        ToolUse("Bash", {"command": "paper-orchestra research review p H1 accept"})
    )
    assert refused is not None
    assert "are the researcher's" in refused and "on the board" in refused
    assert "propose_edit" in refused and "carried out by another agent" in refused
    assert guard.check(ToolUse("Bash", {"command": "python main.py"})) is None
    # A helper the agent started is held to the same.
    assert guard.check(ToolUse("Bash", {"command": "paper-orchestra ui"}, sub=True)) is not None
    # Also without the sandbox: that the person decides is not a matter of the sandbox.
    open_guard = make_guard(tmp_path, home, sandbox=False)
    assert open_guard.check(ToolUse("Bash", {"command": "paper-orchestra research approve p E1"}))


@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit", "NotebookEdit"])
def test_the_file_tools_write_only_in_the_agents_own_folder(tmp_path, home, tool):
    guard = make_guard(tmp_path, home)
    folder = guard.folder
    key = "notebook_path" if tool == "NotebookEdit" else "file_path"
    assert guard.check(ToolUse(tool, {key: str(folder / "notes" / "a.md")})) is None
    assert guard.check(ToolUse(tool, {key: "a.md"})) is None  # relative to its folder
    for outside in (
        guard.project / "wiki" / "hypotheses" / "H1.md",
        guard.project / "state.json",
        guard.project / "project.json",
        guard.project / "agents" / "r1-design-designer" / "x.py",  # another agent's folder
        home / ".bashrc",
        tmp_path / "elsewhere.txt",
        Path("../../state.json"),
        Path("/etc/passwd"),
    ):
        refused = guard.check(ToolUse(tool, {key: str(outside)}))
        assert refused is not None and "only in your own folder" in refused, outside
    assert guard.check(ToolUse(tool, {})) is not None  # no path, no leave


def test_a_link_out_of_the_folder_does_not_carry_a_write_or_a_read_out(tmp_path, home):
    guard = make_guard(tmp_path, home)
    (guard.project / "wiki").mkdir()
    (guard.project / "state.json").write_text("{}", encoding="utf-8")
    os.symlink(guard.project / "wiki", guard.folder / "wiki")
    os.symlink(guard.project / "state.json", guard.folder / "state")
    assert guard.check(ToolUse("Write", {"file_path": str(guard.folder / "wiki" / "x.md")}))
    assert guard.check(ToolUse("Edit", {"file_path": "state"}))
    assert guard.check(ToolUse("Read", {"file_path": "state"}))


def test_the_file_tools_read_the_folder_and_the_system_but_nothing_that_is_closed(tmp_path, home):
    guard = make_guard(tmp_path, home)
    folder, project = guard.folder, guard.project

    def reads(tool: str, **given) -> bool:
        return guard.check(ToolUse(tool, given)) is None

    assert reads("Read", file_path=str(folder / "notes.md"))
    assert reads("Read", file_path="notes.md")
    assert reads("Grep", pattern="warmup")  # its own folder, by default
    assert reads("Glob", pattern="**/*.py")
    assert reads("Read", file_path="/usr/lib/python3/os.py")
    assert reads("Read", file_path=os.path.join(sys.prefix, "pyvenv.cfg"))  # the Python it runs
    for closed in (
        project / "state.json",
        project / "events.jsonl",
        project / "wiki" / "hypotheses" / "H1.md",
        project / "proposals" / "ep-1.json",
        project / "experiments" / "E1" / "approval.json",
        project / "agents" / "r1-design-designer" / "notes.md",
        home / ".ssh" / "id_ed25519",
        home / ".anthropic" / "wif.env",
        home / ".config" / "gh" / "hosts.yml",
        home / ".config" / "op" / "config",
        home / ".config" / "anthropic" / "credentials",
        home / ".claude" / ".credentials.json",
        home / "notes" / "private.md",
        Path("/run/user/1000/po-wif/identity.jwt"),
        tmp_path / "elsewhere" / ".env",
        folder / ".env",  # a credentials file is one wherever it lies
        Path("../../state.json"),
    ):
        refused = guard.check(ToolUse("Read", {"file_path": str(closed)}))
        assert refused is not None and "not yours to read" in refused, closed
        assert "memory tool" in refused
    assert not reads("Grep", pattern="token", path=str(project))
    assert not reads("Glob", pattern="*.md", path=str(project / "wiki"))
    assert not reads("Glob", pattern=str(project / "wiki" / "**" / "*.md"))
    assert not reads("Glob", pattern="../../wiki/*.md")
    assert not reads("Glob", pattern=str(home / ".ssh" / "*"))
    assert reads("Read", file_path=str(folder / ".env.example"))


def test_a_search_does_not_start_above_what_is_closed(tmp_path, home):
    """Grep and Glob walk down from where they start, in the agent's process and not in
    the shell's sandbox: started at ``/`` or ``/home`` they would read the project's
    files and the person's.  A directory with a closed place beneath it is closed."""
    guard = make_guard(tmp_path, home)
    project = guard.project
    # (Not the directory for temporary files: whether the test's own directory lies
    # beneath it depends on how the tests were started.)
    above = [Path("/"), tmp_path, home.parent, project.parent]
    for place in dict.fromkeys(above):
        assert not guard.may_read(Path(os.path.realpath(place))), place
        for tool in ("Grep", "Glob"):
            refused = guard.check(ToolUse(tool, {"pattern": "state", "path": str(place)}))
            assert refused is not None and "not yours to read" in refused, (tool, place)
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": "/", "glob": "**/state.json"}))
    assert guard.check(ToolUse("Glob", {"pattern": "/**/state.json"})) is not None
    assert guard.check(ToolUse("Glob", {"pattern": f"{tmp_path}/**/state.json"})) is not None
    assert guard.check(ToolUse("Glob", {"pattern": "../../../**/*.json"})) is not None
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": "/run"})) is not None  # the token
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": "../.."})) is not None
    # Where nothing closed lies beneath, a search is the agent's to make.
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": "/usr/lib"})) is None
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": str(guard.folder)})) is None
    assert guard.check(ToolUse("Glob", {"pattern": "**/*.py", "path": sys.prefix})) is None

    # Without the sandbox the files are open, and the credentials are still not: a search
    # of the home directory would walk into them.
    opened = make_guard(tmp_path, home, sandbox=False)
    assert opened.check(ToolUse("Grep", {"pattern": "x", "path": str(project)})) is None
    assert opened.check(ToolUse("Grep", {"pattern": "x", "path": str(home / "notes")})) is None
    for place in (home, Path("/"), home / ".config"):
        assert opened.check(ToolUse("Grep", {"pattern": "KEY", "path": str(place)})), place


@pytest.mark.parametrize(
    "name", [".env", ".env.local", ".env.production", ".envrc", "prod.env", "secrets.env"]
)
def test_a_file_of_settings_with_secrets_is_closed_under_every_usual_name(tmp_path, home, name):
    for sandbox in (True, False):
        guard = make_guard(tmp_path, home, sandbox=sandbox)
        for place in (guard.folder, Path("/srv/app"), Path("/tmp/x")):
            assert guard.check(ToolUse("Read", {"file_path": str(place / name)})), (place, name)
    guard = make_guard(tmp_path, home)
    assert guard.check(ToolUse("Glob", {"pattern": f"**/{name}"})) is not None
    deny = guard.settings()["permissions"]["deny"]
    if name in (".env", ".env.local", ".envrc"):
        assert f"Read(//**/{name})" in deny and f"Read(**/{name})" in deny
    assert "Read(//**/*.env)" in deny
    # The shell's sandbox is told the same names: `cat` of such a file is refused too.
    hidden = guard.settings()["sandbox"]["filesystem"]["denyRead"]
    assert {"**/.env", "/**/.env", "**/*.env", "**/.envrc"} <= set(hidden)


def test_a_sample_of_such_a_file_and_a_file_about_environments_stay_readable(tmp_path, home):
    guard = make_guard(tmp_path, home)
    for name in (".env.example", ".env.sample", "environment.py", "env.md", "gym_env.py"):
        assert guard.check(ToolUse("Read", {"file_path": str(guard.folder / name)})) is None, name


def test_the_file_tools_do_not_read_processes_or_devices(tmp_path, home):
    """The trees that show processes (their environment holds the worker's credentials)
    and devices are closed by this check itself, whatever the agent's process refuses."""
    guard = make_guard(tmp_path, home)
    for closed in ("/proc/1/environ", "/proc/self/environ", f"/proc/{os.getpid()}/environ", "/sys"):
        refused = guard.check(ToolUse("Read", {"file_path": closed}))
        assert refused is not None and "not yours to read" in refused, closed
    assert guard.check(ToolUse("Grep", {"pattern": "KEY", "path": "/proc"})) is not None
    assert guard.check(ToolUse("Glob", {"pattern": "/proc/*/environ"})) is not None
    # The system's own files stay readable: the Python an experiment runs with is there.
    assert guard.check(ToolUse("Read", {"file_path": "/usr/lib/python3/os.py"})) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://METADATA.google.internal./",
        "http://169.254.170.2/v2/credentials",  # any link-local address
        "http://[fe80::1]/",
        "http://localhost:8501/api/projects",
        "http://127.0.0.1:8501/api/projects",
        "http://127.8.8.8/",
        "http://[::1]:8501/",
        "http://app.localhost/",
        "http://0.0.0.0:8501/",
        "http://0:8501/",
        "http://[::]:8501/",
        "http://[bad",
        "",
        # Other ways to write the loopback and the metadata server: as one number, in
        # hexadecimal, in octal, with parts left out, inside an IPv6 address.
        "http://2130706433:8501/",
        "http://0x7f000001/",
        "http://0x7f.0.0.1/",
        "http://127.1:8501/",
        "http://0177.0.0.1/",
        "http://017700000001/",
        "http://[::ffff:127.0.0.1]:8501/",
        "http://[::ffff:169.254.169.254]/",
        "http://2852039166/",  # 169.254.169.254 as one number
        "http://0xa9fea9fe/",
        # Names that resolve to them (answered from conftest.RESOLVES_TO).
        "http://127.0.0.1.nip.io:8501/",
        "http://169.254.169.254.nip.io/",
        "http://localtest.me:8501/",
        "http://mapped.example/",
        "http://both.example/",  # one of its addresses is enough
        # A name that does not resolve is not opened either.
        "http://no-such-host.invalid/",
        # Where two readers of an address find two hosts.  The program that connects
        # reads a backslash as a slash, so these go to the loopback, not to example.com
        # (which the resolver of these tests answers with an address of the web).
        "http://127.0.0.1:8501\\@example.com/",
        "http://127.0.0.1:8501\\\\@example.com/",
        "http://169.254.169.254\\@example.com/",
        "http://example.com\\@127.0.0.1:8501/",
        "http:\\\\127.0.0.1:8501/",
        "http://example.com/\\..\\x",  # no backslash at all: nobody needs one
        # A name or a password before the host: read differently, and no page needs it.
        "http://user:word@example.com/",
        "http://127.0.0.1:8501@example.com/",
        "http://example.com@127.0.0.1:8501/",
        "http://127.0.0.1%2f@example.com/",
        # `http:` without its two slashes is still an address to the one that connects.
        "http:127.0.0.1:8501/",
        "http:/127.0.0.1:8501/",
        "http:///127.0.0.1:8501/",
        "HTTPS:example.com",
        # Characters the one that connects drops or decodes.
        "http://127.0.0.1\t:8501/",
        "http://exam\nple.com/",
        "http://example.com/a b",
        "http://%31%32%37.0.0.1:8501/",
        "http://loc%61lhost:8501/",
        # Other letters that are the same host to the connection.
        "http://１２７.０.０.１:8501/",
        "http://127。0。0。1/",
        "http://ｌocalhost:8501/",
        "http://①27.0.0.1/",
    ],
)
def test_the_web_tool_does_not_open_this_machines_own_services(tmp_path, home, url):
    """WebFetch runs in the agent's process, outside the shell's sandbox.  The metadata
    server of a cloud hands the machine's credentials to whoever asks from the machine,
    and the app that decides proposals listens on the loopback: neither is the web."""
    assert closed_host(url)
    for sandbox in (True, False):
        guard = make_guard(tmp_path, home, sandbox=sandbox)
        refused = guard.check(ToolUse("WebFetch", {"url": url, "prompt": "What is there?"}))
        assert refused is not None and "closed to agents" in refused
        assert guard.check(ToolUse("WebFetch", {"url": url}, sub=True)) is not None  # a helper too


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "https://arxiv.org/abs/1706.03762",
        "http://10.0.0.5/",
        "example.com",
        "http://93.184.216.34/",
        "http://[2606:2800:220:1:248:1893:25c8:1946]/",
        "https://example.com:8443/a/b?q=user@example.com#part",
        "https://example.com/path/with%20a%20space?next=http://localhost/",
        " https://example.com/ ",
        "https://bücher.example/",
        "example.com:8080/page",
    ],
)
def test_the_web_tool_opens_the_web(tmp_path, home, url):
    assert not closed_host(url)
    assert make_guard(tmp_path, home).check(ToolUse("WebFetch", {"url": url})) is None


def test_the_shells_sandbox_is_told_which_hosts_are_closed(tmp_path, home):
    network = make_guard(tmp_path, home).settings()["sandbox"]["network"]
    for host in ("169.254.169.254", "metadata.google.internal", "localhost", "127.0.0.1"):
        assert host in network["deniedDomains"]
    # The unspecified address arrives at the loopback: it is closed under its spellings.
    for host in ("0.0.0.0", "0", "::", "::1"):
        assert host in network["deniedDomains"]


def test_a_login_kept_elsewhere_is_found_and_closed_to_the_agents(tmp_path, home, monkeypatch):
    """``CLAUDE_CONFIG_DIR`` names where Claude Code keeps its login.  The gate looks
    there, and that file is a credential like the one in the home directory."""
    config = tmp_path / "claude-config"
    config.mkdir()
    (home / ".claude" / ".credentials.json").unlink()
    runtime = AgentRuntime(sdk_project(tmp_path), FakeDriver())
    environ = {"CLAUDE_CONFIG_DIR": str(config)}
    assert "no Claude Code login" in runtime.need(environ, home)["why"]
    (config / ".credentials.json").write_text("{}", encoding="utf-8")
    assert runtime.need(environ, home) is None
    assert runtime.need({}, home) is not None  # not named: the home directory's is asked

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    guard = make_guard(tmp_path, home)
    login = config / ".credentials.json"
    assert guard.check(ToolUse("Read", {"file_path": str(login)})) is not None
    settings = guard.settings()
    assert f"Read(/{login})" in settings["permissions"]["deny"]
    # It lies outside everything that is hidden anyway, so the shell is told of it by name.
    assert str(login) in settings["sandbox"]["filesystem"]["denyRead"]
    assert str(login) in settings["sandbox"]["filesystem"]["denyWrite"]


def test_where_the_login_is_not_a_file_the_gate_lets_the_first_agent_decide(tmp_path, home):
    """On macOS the login is in the Keychain: a missing file says nothing there, and a
    person who is signed in must not be told that no login was found."""
    (home / ".claude" / ".credentials.json").unlink()
    assert login_found({}, home, platform="linux") is False
    assert login_found({}, home, platform="darwin") is None
    report = machine_report({}, home)
    assert report["login"] in (False, None) and report["federation"] is False
    assert report["federation_missing"] == list(FEDERATION_ENV)
    assert set(report) >= {"package", "command_line", "login_file", "install"}


def test_without_the_sandbox_files_are_open_but_credentials_and_the_gate_are_not(tmp_path, home):
    guard = make_guard(tmp_path, home, sandbox=False)
    assert guard.check(ToolUse("Write", {"file_path": str(tmp_path / "elsewhere.txt")})) is None
    assert guard.check(ToolUse("Read", {"file_path": str(home / "notes" / "x.md")})) is None
    assert guard.check(ToolUse("Read", {"file_path": str(home / ".ssh" / "id_ed25519")}))
    assert guard.check(ToolUse("Read", {"file_path": str(tmp_path / ".env")}))
    settings = guard.settings()
    assert "sandbox" not in settings
    assert f"Read(/{home}/.ssh/**)" in settings["permissions"]["deny"]


def test_the_identity_token_is_a_credential_wherever_it_lies(tmp_path, home):
    """The worker's identity token is hidden under /run/user only by where it happens to
    lie.  Named elsewhere (a projected token of a cluster, a file under /var), it is
    closed all the same: to the shell, to the file tools, with and without the sandbox."""
    token = tmp_path / "var" / "secrets" / "identity.jwt"
    token.parent.mkdir(parents=True)
    token.write_text("jwt", encoding="utf-8")
    link = tmp_path / "token-link.jwt"
    link.symlink_to(token)
    environ = {"ANTHROPIC_IDENTITY_TOKEN_FILE": str(link)}
    # Without the setting the file is like any other outside the hidden places.
    assert make_guard(tmp_path, home).may_read(token)
    guard = make_guard(tmp_path, home, environ=environ)
    assert token in guard.credential_places() and link in guard.credential_places()
    for path in (token, link):
        assert guard.check(ToolUse("Read", {"file_path": str(path)})) is not None
        assert f"Read(/{path})" in guard.settings()["permissions"]["deny"]
    files = guard.settings()["sandbox"]["filesystem"]
    assert str(token) in files["denyRead"] and str(token) in files["denyWrite"]
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": str(token.parent.parent)}))
    # The rule about credential files stays without the sandbox.
    opened = make_guard(tmp_path, home, environ=environ, sandbox=False)
    assert opened.check(ToolUse("Read", {"file_path": str(token)})) is not None
    assert f"Read(/{token})" in opened.settings()["permissions"]["deny"]
    # The experiment agent the person granted model access may read it, and only it.
    granted = make_guard(tmp_path, home, environ=environ, token_file=link)
    assert granted.check(ToolUse("Read", {"file_path": str(token)})) is None
    assert str(token) in granted.settings()["sandbox"]["filesystem"]["allowRead"]
    assert str(token) not in granted.settings()["sandbox"]["filesystem"]["denyRead"]


def test_uvs_credentials_and_its_private_checkouts_are_not_opened(tmp_path, home):
    """`uv auth login` keeps what it is given in ``~/.local/share/uv/credentials``, and
    uv's cache holds checkouts of git dependencies, private ones among them.  What is
    opened of uv is what running Python needs."""
    share = home / ".local" / "share" / "uv"
    cache = home / ".cache" / "uv"
    for place in (share / "credentials", share / "python", cache / "git-v0", cache / "wheels-v5"):
        place.mkdir(parents=True)
    guard = make_guard(tmp_path, home)
    opened = guard.settings()["sandbox"]["filesystem"]["allowRead"]
    assert str(share) not in opened and str(cache) not in opened
    assert str(share / "python") in opened and str(cache / "wheels-v5") in opened
    assert str(cache / "git-v0") not in opened
    assert str(share / "credentials") not in opened
    secret = share / "credentials" / "credentials.toml"
    assert guard.check(ToolUse("Read", {"file_path": str(secret)})) is not None
    assert f"Read(/{share}/credentials/**)" in guard.settings()["permissions"]["deny"]
    assert guard.check(ToolUse("Read", {"file_path": str(cache / "git-v0" / "x" / "a.py")}))
    python = share / "python" / "bin" / "python"
    assert guard.check(ToolUse("Read", {"file_path": str(python)})) is None
    assert make_guard(tmp_path, home, sandbox=False).check(
        ToolUse("Read", {"file_path": str(secret)})
    )


def test_the_sandbox_is_required_and_not_only_asked_for(tmp_path, home):
    """Without ``failIfUnavailable`` the agent's process warns that the sandbox cannot
    start and runs every command of the agent outside any sandbox."""
    sandbox = make_guard(tmp_path, home).settings()["sandbox"]
    assert sandbox["enabled"] is True and sandbox["failIfUnavailable"] is True
    assert sandbox["allowUnsandboxedCommands"] is False


def test_what_the_sandbox_is_made_of_is_looked_for_on_the_path_the_agent_gets(tmp_path):
    bare = tmp_path / "bare"
    full = tmp_path / "full"
    for place, programs in ((bare, ("ls",)), (full, ("ls", "bwrap", "socat"))):
        place.mkdir()
        for name in programs:
            (place / name).write_text("#!/bin/sh\n", encoding="utf-8")
            (place / name).chmod(0o755)
    fault = guard_module.sandbox_fault(str(bare), platform="linux")
    assert fault is not None
    assert "bwrap and socat are not installed" in fault[0] and "cannot start" in fault[0]
    assert "apt install bubblewrap socat" in fault[1]
    assert "settings.agents.sandbox to false" in fault[1]
    assert guard_module.sandbox_fault(str(full), platform="linux") is None
    (full / "socat").unlink()
    assert "socat is not installed" in guard_module.sandbox_fault(str(full), platform="linux")[0]
    # Elsewhere the system brings its own.
    assert guard_module.sandbox_fault(str(bare), platform="darwin") is None


def test_the_shared_directory_for_temporary_files_is_closed_and_the_agent_has_its_own(
    tmp_path, home
):
    """An agent's shell used to keep its temporary files where every program of the
    person keeps theirs, and could read and change what lay there."""
    shared = tmp_path / "tmp" / "claude-1000"
    (shared / "another-session").mkdir(parents=True)
    other = shared / "another-session" / "scratch.txt"
    other.write_text("not the agent's", encoding="utf-8")
    parent = tmp_path / "po-1000"
    own_temp = parent / "mine"
    (parent / "another-agent").mkdir(parents=True)
    own_temp.mkdir()
    guard = make_guard(tmp_path, home, shared=[shared], temp=own_temp)
    assert not guard.may_read(other) and not guard.may_read(shared)
    assert guard.check(ToolUse("Read", {"file_path": str(other)})) is not None
    assert guard.check(ToolUse("Write", {"file_path": str(other), "content": "x"})) is not None
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": str(shared)})) is not None
    files = guard.settings()["sandbox"]["filesystem"]
    assert str(shared) in files["denyRead"] and str(shared) in files["denyWrite"]
    assert str(shared) not in files["allowRead"]
    # Its own is made for it alone: it reads and writes there, and no other agent's.
    own = own_temp / "claude-1000" / "work.txt"
    assert guard.check(ToolUse("Write", {"file_path": str(own), "content": "x"})) is None
    assert guard.check(ToolUse("Read", {"file_path": str(own)})) is None
    assert files["allowWrite"] == [str(own_temp)] and str(own_temp) in files["allowRead"]
    theirs = parent / "another-agent" / "scratch.txt"
    assert guard.check(ToolUse("Read", {"file_path": str(theirs)})) is not None
    assert guard.check(ToolUse("Write", {"file_path": str(theirs), "content": "x"})) is not None
    assert guard.check(ToolUse("Grep", {"pattern": "x", "path": str(parent)})) is not None
    assert str(parent) in files["denyRead"]
    # An agent that was given none writes in its folder alone.
    assert "allowWrite" not in make_guard(tmp_path, home).settings()["sandbox"]["filesystem"]

    # A project that itself lies in the shared directory keeps its agent's folder open.
    inside = Guard(
        folder=shared / "p" / "agents" / "a", project=shared / "p", home=home, shared=[shared]
    )
    inside.folder.mkdir(parents=True)
    assert inside.may_read(inside.folder / "notes.txt") and not inside.may_read(other)
    assert str(shared) not in inside.settings()["sandbox"]["filesystem"]["denyWrite"]

    # What is shared is asked of this process where nobody names it.
    import tempfile

    found = Guard(
        folder=guard.folder,
        project=guard.project,
        home=home,
        environ={"TMPDIR": str(tmp_path / "t"), "CLAUDE_CODE_TMPDIR": str(tmp_path / "c")},
    ).shared_temp()
    assert Path(os.path.realpath(tempfile.gettempdir())) in found
    assert tmp_path / "t" in found and tmp_path / "c" in found
    assert Path(f"/tmp/claude-{os.getuid()}") in found


def test_the_settings_put_the_shell_in_the_sandbox_and_hide_what_is_closed(tmp_path, home):
    guard = make_guard(tmp_path, home)
    settings = guard.settings()
    sandbox = settings["sandbox"]
    assert sandbox["enabled"] is True
    assert sandbox["autoAllowBashIfSandboxed"] is True
    # No command of the agent runs outside the sandbox.
    assert sandbox["allowUnsandboxedCommands"] is False
    files = sandbox["filesystem"]
    assert str(home) in files["denyRead"] and str(guard.project) in files["denyRead"]
    assert "/run/user" in files["denyRead"]
    assert str(guard.folder) in files["allowRead"]
    # Nothing that holds credentials is opened again by the exceptions.
    assert not any(".ssh" in path or ".anthropic" in path for path in files["allowRead"])
    deny = settings["permissions"]["deny"]
    for place in (".ssh", ".anthropic", ".config/gh", ".config/op", ".config/anthropic"):
        assert f"Read(/{home}/{place}/**)" in deny
    assert f"Read(/{home}/.claude/.credentials.json)" in deny
    assert "Read(**/.env)" in deny


def test_only_an_experiment_agent_with_model_access_may_read_the_identity_token(tmp_path, home):
    token = tmp_path / "run" / "identity.jwt"
    token.parent.mkdir()
    token.write_text("jwt", encoding="utf-8")
    plain = make_guard(tmp_path, home)
    granted = make_guard(tmp_path, home, token_file=token)
    # (The token lies under the test's directory here; in life it lies under /run/user.)
    hidden = Guard(folder=plain.folder, project=tmp_path, home=home)
    assert hidden.check(ToolUse("Read", {"file_path": str(token)})) is not None
    allowed = Guard(folder=plain.folder, project=tmp_path, home=home, token_file=token)
    assert allowed.check(ToolUse("Read", {"file_path": str(token)})) is None
    assert str(token) in granted.settings()["sandbox"]["filesystem"]["allowRead"]
    assert str(token) not in plain.settings()["sandbox"]["filesystem"]["allowRead"]


def test_the_agents_process_starts_without_this_processes_credentials():
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/me",
        "ANTHROPIC_API_KEY": "sk-ant-secret-secret",
        "ANTHROPIC_BASE_URL": "https://example.invalid",
        "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_1",
        "ANTHROPIC_ORGANIZATION_ID": "org",
        "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac",
        "ANTHROPIC_WORKSPACE_ID": "wrk",
        "ANTHROPIC_IDENTITY_TOKEN_FILE": "/run/user/1/identity.jwt",
        "OPENAI_API_KEY": "sk-openai-secret",
        "GEMINI_API_KEY": "gm-secret",
        "SEMANTIC_SCHOLAR_API_KEY": "s2-secret",
        "GH_TOKEN": "ghp_secret",
        "MODAL_TOKEN_SECRET": "as-secret",
        "PAPER_ORCHESTRA_UI_TOKEN": "ui-secret",
        "SSH_KEY_PATH": "/home/me/.ssh/id",
    }
    login = agent_env("login", environ)
    # With the login, no ANTHROPIC_* variable reaches the agent: it would sign in with
    # whichever it finds before it looks at the person's Claude Code login.
    for name in environ:
        if name.startswith("ANTHROPIC_"):
            assert login[name] == "", name
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GH_TOKEN", "MODAL_TOKEN_SECRET"):
        assert login[name] == ""
    assert login["SEMANTIC_SCHOLAR_API_KEY"] == "" and login["PAPER_ORCHESTRA_UI_TOKEN"] == ""
    assert "PATH" not in login and "HOME" not in login

    federation = agent_env("federation", environ)
    # With federation the five settings stay, and nothing else of Anthropic's.
    assert not any(name in federation for name in FEDERATION_ENV)
    assert federation["ANTHROPIC_API_KEY"] == "" and federation["ANTHROPIC_BASE_URL"] == ""
    assert federation["OPENAI_API_KEY"] == ""


@pytest.mark.parametrize(
    "name",
    ["AWS_ACCESS_KEY_ID", "VAULT_SECRET_ID", "CLIENT_SECRET_ID", "GCP_SA_KEY_ID", "HF_TOKEN_ID"],
)
def test_half_a_credential_that_is_called_an_id_does_not_reach_the_agent(name):
    """A name ending in ``_ID`` can be a credential's own (an access key's id)."""
    environ = {name: "AKIAEXAMPLE", "PATH": "/usr/bin"}
    for credential in ("login", "federation"):
        env = agent_env(credential, environ)
        assert env[name] == ""


#: What a worker's environment can hold that no pattern of names finds: each opens
#: something (the person's ssh keys, a machine, an account) or has a password in it.
OPENS_SOMETHING = {
    "SSH_AUTH_SOCK": "/tmp/ssh-abc/agent.1",
    "DOCKER_HOST": "ssh://root@build-machine",
    "KUBECONFIG": "/home/me/.kube/config",
    "DATABASE_URL": "postgres://me:word@db.example/app",
    "SENTRY_DSN": "https://0123abcd@o1.ingest.example/2",
    "OP_SESSION_abc": "session-of-a-password-manager",
    "VAULT_ADDR": "https://vault.example",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus",
    "GOOGLE_APPLICATION_CREDENTIALS": "/home/me/sa.json",
    "AWS_PROFILE": "admin",
    "CLAUDE_CODE_MESSAGING_SOCKET": "/tmp/claude/messaging.sock",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "SSH_KEY_PATH": "/home/me/.ssh/id",
    "TOKEN_FILE": "/x",
    "SOMETHING_NOBODY_THOUGHT_OF": "x",
}


@pytest.mark.parametrize("credential", ["login", "federation"])
def test_only_what_is_on_a_short_list_reaches_the_agent(credential):
    """The agent's shell has the web.  A variable is not kept from it for how it is
    called (a name says little about what the value opens) but passed on because it is
    known to be needed: every other one of the worker's is taken away."""
    needed = {
        "PATH": "/usr/bin",
        "HOME": "/home/me",
        "USER": "me",
        "SHELL": "/bin/bash",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "xterm",
        "CLAUDE_CONFIG_DIR": "/home/me/.claude",
        "SSL_CERT_FILE": "/etc/ssl/cert.pem",
        "HTTPS_PROXY": "http://proxy.example:3128",
        "NO_PROXY": "localhost",
    }
    env = agent_env(credential, {**needed, **OPENS_SOMETHING})
    for name in OPENS_SOMETHING:
        assert env.get(name) == "", name
    assert not set(needed) & set(env)
    # The whole list is passed on, and nothing on it is called like a credential.
    assert agent_env(credential, {name: "x" for name in KEPT_ENV}) == {}
    assert not [name for name in KEPT_ENV if re.search("KEY|TOKEN|SECRET|PASS|CRED|SOCK", name)]
    # The agent's process is told by its starter what started it: an empty value
    # set here would replace that.
    assert "CLAUDE_CODE_ENTRYPOINT" not in agent_env(credential, {"CLAUDE_CODE_ENTRYPOINT": "cli"})


def test_a_proxy_with_a_password_in_its_address_does_not_reach_the_agent():
    env = agent_env(
        "login",
        {
            "HTTPS_PROXY": "http://me:word@proxy.example:3128",
            "http_proxy": "http://word@proxy.example:3128",
            "ALL_PROXY": "socks5://proxy.example:1080",
            "NO_PROXY": "me@example.com",  # no address: left alone
        },
    )
    assert env == {"HTTPS_PROXY": "", "http_proxy": ""}


def test_the_person_can_name_more_variables_for_the_agents_to_keep():
    """What an experiment needs and the list does not have (``PASS_ENV``): the person's
    choice, by name.  Not what changes how the agent's process signs in."""
    environ = {
        PASS_ENV: "HF_HOME, WANDB_MODE ANTHROPIC_API_KEY,CLAUDE_CODE_OAUTH_TOKEN",
        "HF_HOME": "/data/hf",
        "WANDB_MODE": "offline",
        "ANTHROPIC_API_KEY": "sk-secret",
        "CLAUDE_CODE_OAUTH_TOKEN": "token",
        "TORCH_HOME": "/data/torch",
    }
    for credential in ("login", "federation"):
        assert agent_env(credential, environ) == {
            PASS_ENV: "",
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "TORCH_HOME": "",
        }


def test_an_experiments_federation_settings_travel_under_other_names(tmp_path):
    job = {name: f"value-of-{name}" for name in FEDERATION_ENV}
    env = agent_env("login", {name: "x" * 20 for name in FEDERATION_ENV}, job_models=job)
    for name in FEDERATION_ENV:
        # Not under their own names: those would change how the agent itself signs in.
        assert env[name] == ""
        assert env[f"{JOB_ENV_PREFIX}{name}"] == f"value-of-{name}"
    assert not any(name.startswith(JOB_ENV_PREFIX) for name in agent_env("login", {}))
    # The launcher puts them back for one command, and holds no value itself.
    launcher = tmp_path / "with-models"
    launcher.write_text(launcher_text(), encoding="utf-8")
    launcher.chmod(0o755)
    assert "value-of" not in launcher_text()
    import subprocess

    shown = subprocess.run(
        [str(launcher), "sh", "-c", "echo $ANTHROPIC_WORKSPACE_ID $ANTHROPIC_IDENTITY_TOKEN_FILE"],
        env={"PATH": os.environ["PATH"], **{k: v for k, v in env.items() if v}},
        capture_output=True,
        text=True,
        check=True,
    )
    assert shown.stdout.split() == [
        "value-of-ANTHROPIC_WORKSPACE_ID",
        "value-of-ANTHROPIC_IDENTITY_TOKEN_FILE",
    ]


# ---------------------------------------------------------------------------
# A session: what the agent is started with
# ---------------------------------------------------------------------------


def test_an_agent_is_started_with_its_folder_the_board_and_the_guardrails(
    tmp_path, home, monkeypatch
):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ssh-abc/agent.1")
    monkeypatch.setenv("DATABASE_URL", "postgres://me:word@db.example/app")
    project = sdk_project(tmp_path)
    driver = FakeDriver()
    ctx = sdk_context(project, driver)
    session = AgentSession(ctx, "scout")
    session.send("Find the prior work.")

    (options,) = driver.opened
    assert options.model == "claude-fable-5-2"
    assert Path(options.cwd) == project.root / "agents" / "r1-research-scout"
    assert Path(options.cwd).is_dir()
    assert [tool.name for tool in options.board] == ["memory", "wiki_search", "propose_edit"]
    assert tuple(options.builtin) == (*BUILTIN_TOOLS, *HELPER_TOOLS)
    assert {"Bash", "WebSearch", "WebFetch", "Write", "Edit"} <= set(options.builtin)
    assert "Your role: scout" in options.system_prompt
    assert "mcp__board__memory" in options.system_prompt
    assert "You may start helper agents" in options.system_prompt
    assert "## idea" in options.system_prompt or "idea.md" in options.system_prompt  # the index
    # Fast mode and ultracode are asked for in the settings the process starts with.
    settings = options.settings
    assert settings["fastMode"] is True
    assert settings["ultracode"] is True and settings["enableWorkflows"] is True
    assert settings["skipWorkflowUsageWarning"] is True
    assert settings["sandbox"]["enabled"] is True
    assert str(project.root) in settings["sandbox"]["filesystem"]["denyRead"]
    assert str(options.cwd) in settings["sandbox"]["filesystem"]["allowRead"]
    assert options.resume is None
    assert driver.prompts == ["Find the prior work."]
    # Its temporary files lie in a directory made for it alone while it works (a short
    # path: the agent's process falls back to the shared directory for a long one),
    # which the sandbox opens to it and to nobody else, and which goes when it ends.
    temp = Path(options.env["CLAUDE_CODE_TMPDIR"])
    assert temp.is_dir() and oct(temp.stat().st_mode & 0o777) == "0o700"
    assert len(os.fsencode(temp)) + len(f"/claude-{os.getuid()}") <= 44
    assert temp.parent.name == f"po-{os.getuid()}"
    assert oct(temp.parent.stat().st_mode & 0o777) == "0o700"
    files = settings["sandbox"]["filesystem"]
    assert files["allowWrite"] == [str(temp)] and str(temp) in files["allowRead"]
    assert str(temp.parent) in files["denyRead"]
    assert Path(options.cwd) not in temp.parents
    # `python` in its shell is the interpreter of this process, with its packages.
    assert options.env["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    # What this process holds besides does not go along; what was set for the agent
    # stands over that.
    assert options.env["SSH_AUTH_SOCK"] == "" and options.env["DATABASE_URL"] == ""
    assert options.env["TMPDIR"] and Path(options.env["TMPDIR"]).is_relative_to(temp)
    (temp / "left-behind.txt").write_text("x", encoding="utf-8")
    session.close()
    assert driver.conversations[0].closed
    assert not temp.exists()


def test_two_agents_have_two_directories_for_temporary_files(tmp_path, home, monkeypatch):
    project = sdk_project(tmp_path)
    driver = FakeDriver()
    ctx = sdk_context(project, driver)
    scout, theorist = AgentSession(ctx, "scout"), AgentSession(ctx, "theorist")
    scout.send("Go.")
    theorist.send("Go.")
    one, other = (Path(options.env["CLAUDE_CODE_TMPDIR"]) for options in driver.opened)
    assert one != other and one.parent == other.parent
    (other / "theirs.txt").write_text("x", encoding="utf-8")
    refused = scout._guard.check(ToolUse("Read", {"file_path": str(other / "theirs.txt")}))
    assert refused is not None and "not yours to read" in refused
    assert driver.opened[0].check(ToolUse("Read", {"file_path": str(one / "mine.txt")})) is None
    ctx.close_sessions()
    assert not one.exists() and not other.exists()

    # A path too long for the agent's process to use is refused, not quietly replaced
    # by the shared directory.
    long = tmp_path / ("a-rather-long-name-for-a-directory" * 2)
    long.mkdir()
    monkeypatch.setenv(guard_module.TEMP_ENV, str(long))
    with pytest.raises(OSError, match="too long a path"):
        guard_module.make_temp()


def test_no_agent_starts_where_the_sandbox_it_is_to_work_in_cannot_start(tmp_path, home):
    """With ``sandbox: true`` and nothing to make a sandbox of, the agent's process ran
    every command unsandboxed and said nothing.  Now no agent is started: the gate
    says so before a phase, and a session that is started all the same fails with a
    plain sentence."""
    project = sdk_project(tmp_path)
    driver = FakeDriver()
    driver.no_sandbox = guard_module.sandbox_fault("/nowhere", platform="linux")
    runtime = AgentRuntime(project, driver)
    need = runtime.need({"PATH": "/nowhere"}, home)
    assert need is not None and "bwrap and socat are not installed" in need["why"]
    assert "install bubblewrap and socat" in need["fix"]
    assert driver.sandbox_asked == ["/nowhere"]

    ctx = sdk_context(project, driver)
    session = AgentSession(ctx, "scout", env={"PATH": "/only/this"})
    with pytest.raises(guard_module.SandboxUnavailable) as caught:
        session.send("Go.")
    assert driver.opened == [] and driver.prompts == []
    # Asked of the path the agent's process would have been started with.
    assert driver.sandbox_asked[-1] == "/only/this"
    said = str(caught.value)
    assert said.startswith("The scout agent was not started: the project's agents work with")
    assert "bwrap and socat are not installed" in said
    assert said.endswith("Nothing was run outside the sandbox.")
    from paper_orchestra.errors import stops_the_run

    assert stops_the_run(caught.value)  # nobody tries it once more

    # A project that asks for no sandbox is not asked about one.
    open_project = sdk_project(tmp_path / "open", sandbox=False)
    open_driver = FakeDriver()
    open_driver.no_sandbox = driver.no_sandbox
    assert AgentRuntime(open_project, open_driver).need({}, home) is None
    AgentSession(sdk_context(open_project, open_driver), "scout").send("Go.")
    assert len(open_driver.opened) == 1 and open_driver.sandbox_asked == []


@pytest.mark.parametrize("how", ["answer", "failure", "start"])
def test_an_agents_process_that_refuses_for_want_of_the_sandbox_is_said_in_a_sentence(
    tmp_path, home, how
):
    """Where the look for the sandbox's programs finds them and the sandbox still cannot
    start, the agent's process refuses (it is told to: ``failIfUnavailable``).  That is
    the same failure, said the same way, and no second agent is tried."""
    project = sdk_project(tmp_path)
    refusal = (
        "Claude Code returned an error result: Sandbox required but unavailable: sandbox is "
        "enabled but dependencies are missing: bubblewrap (bwrap) not installed"
    )

    class Refusing(FakeDriver):
        def open(self, options):
            if how == "start":
                self.opened.append(options)
                raise RuntimeError(refusal)
            conversation = super().open(options)

            def send(text, sink):
                if how == "failure":
                    raise RuntimeError(refusal)
                from paper_orchestra.research.sdk.driver import Turn

                return Turn(is_error=True, error=refusal, ended="error_during_execution")

            conversation.send = send
            return conversation

    # Also for a session that is carried on: it is not taken for one that is gone.
    first = FakeDriver(lambda agent: agent.use(cost_usd=0.1))
    AgentSession(sdk_context(project, first), "scout").send("Go.")
    driver = Refusing()
    driver.sessions.update(first.sessions)
    with pytest.raises(guard_module.SandboxUnavailable, match="was not started") as caught:
        AgentSession(sdk_context(project, driver), "scout").send("Go.")
    assert "bubblewrap (bwrap) not installed" in str(caught.value)
    assert len(driver.opened) == 1
    assert not any("could not be carried on" in line for line in activity(project))


def test_the_real_driver_looks_for_the_sandbox_on_the_path_it_is_given(tmp_path, monkeypatch):
    from paper_orchestra.research.sdk.claude import ClaudeDriver

    monkeypatch.setattr(sys, "platform", "linux")
    fault = ClaudeDriver().sandbox_unavailable(str(tmp_path))
    assert fault is not None and "bwrap and socat are not installed" in fault[0]
    report = machine_report({"PATH": str(tmp_path)}, tmp_path)
    assert report["sandbox"] is False and "bwrap and socat" in report["sandbox_missing"]


def test_the_settings_of_the_project_decide_fast_mode_helpers_and_the_sandbox(tmp_path, home):
    project = sdk_project(
        tmp_path, fast=False, ultracode=False, sandbox=False, model="claude-haiku-5-5"
    )
    driver = FakeDriver()
    AgentSession(sdk_context(project, driver), "theorist").send("Go.")
    (options,) = driver.opened
    assert options.model == "claude-haiku-5-5"
    assert "fastMode" not in options.settings and "ultracode" not in options.settings
    assert "sandbox" not in options.settings
    assert tuple(options.builtin) == BUILTIN_TOOLS
    assert not set(HELPER_TOOLS) & set(options.builtin)
    assert "helper agents" not in options.system_prompt


def test_login_and_federation_differ_in_what_reaches_the_agents_process(
    tmp_path, home, monkeypatch
):
    for name in FEDERATION_ENV:
        monkeypatch.setenv(name, f"{name.lower()}-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-a-secret-key")
    driver = FakeDriver()
    AgentSession(sdk_context(sdk_project(tmp_path / "a"), driver), "scout").send("Go.")
    federated = sdk_project(tmp_path / "b", credential="federation")
    AgentSession(sdk_context(federated, driver), "scout").send("Go.")
    login, federation = (options.env for options in driver.opened)
    assert all(login[name] == "" for name in FEDERATION_ENV)
    assert login["ANTHROPIC_API_KEY"] == ""
    assert not any(name in federation for name in FEDERATION_ENV)
    assert federation["ANTHROPIC_API_KEY"] == ""


# ---------------------------------------------------------------------------
# The board: the project's rules hold for an agent
# ---------------------------------------------------------------------------


def run_scripted(project: Project, role: str, script, *, phase: str = "research", **options):
    """One session of ``role`` playing ``script``; the context, the driver and the answer."""
    driver = FakeDriver(script)
    ctx = sdk_context(project, driver, phase=phase)
    session = AgentSession(ctx, role, **options)
    said = session.send("Do your work.")
    return ctx, driver, session, said


def test_a_page_an_agent_creates_through_the_board_is_a_proposal_of_its_session(tmp_path, home):
    project = sdk_project(tmp_path)
    answers: list[str] = []

    def script(agent: FakeAgent) -> str:
        answers.append(
            agent.memory(
                "create",
                path="/memories/wiki/questions/small-models.md",
                file_text=QUESTION.format(title="Does it hold for small models?"),
            )
        )
        answers.append(agent.memory("view", path="/memories/wiki/idea.md"))
        return "Wrote one question."

    ctx, _driver, session, said = run_scripted(project, "scout", script)
    assert said == "Wrote one question."
    (page,) = project.pages(kind="question")
    assert (page.review, page.author) == ("proposed", "agent:scout")
    assert page.session == "r1-research-scout" and page.id == "Q1"
    assert "It is Q1" in answers[0] and "It is a proposal" in answers[0]
    assert "Does learning-rate warmup reduce the variance" in answers[1]
    assert session.created() == [page.path] and ctx.produced == [page.path]
    assert [p.path for p in session.proposed("question")] == [page.path]


def test_what_a_role_may_not_create_or_decide_is_refused_by_the_board(tmp_path, home):
    project = sdk_project(tmp_path)
    human = project.create_page(
        "hypothesis", "Warmup helps", "## Statement\n\nIt helps.\n", author="human"
    )
    answers: dict[str, tuple[str, bool]] = {}

    def script(agent: FakeAgent) -> None:
        def tried(name: str, text: str) -> None:
            answers[name] = (text, agent.failed)

        # A scout creates topics and questions: no hypothesis, no experiment, no source.
        tried(
            "hypothesis",
            agent.memory(
                "create",
                path="/memories/wiki/hypotheses/mine.md",
                file_text=HYPOTHESIS.format(title="Mine"),
            ),
        )
        tried(
            "source",
            agent.memory(
                "create",
                path="/memories/wiki/sources/Fake2020.md",
                file_text=TOPIC.format(title="x"),
            ),
        )
        # A page with the fields of an accepted, approved page is a proposal all the same.
        tried(
            "topic",
            agent.memory(
                "create",
                path="/memories/wiki/topics/t.md",
                file_text="---\ntitle: T\nsummary: s\nreview: accepted\nauthor: human\n"
                "state: approved\n---\nBody.\n",
            ),
        )
        # The person's page is not the agent's to change, delete or rename.
        path = f"/memories/{human.path}"
        tried("edit", agent.memory("str_replace", path=path, old_str="It helps.", new_str="No."))
        tried("delete", agent.memory("delete", path=path))
        tried("rename", agent.memory("rename", old_path=path, new_path="/memories/wiki/x.md"))
        # Nothing outside the wiki and the inputs exists for it.
        tried("state", agent.memory("view", path="/memories/state.json"))
        tried("escape", agent.memory("view", path="/memories/../state.json"))
        # A verdict is suggested, and changes nothing until the person accepts it.
        tried(
            "verdict",
            agent.board(
                "propose_edit", target="H1", summary="Supported", fields={"state": "supported"}
            ),
        )
        # Input that is not what the tool takes is answered, not run.
        tried("bad", agent.board("propose_edit", target="H1"))
        tried("unknown", agent.board("wiki_search", query="warmup", everything=True))

    run_scripted(project, "scout", script)
    for name in (
        "hypothesis",
        "source",
        "edit",
        "delete",
        "rename",
        "state",
        "escape",
        "bad",
        "unknown",
    ):
        assert answers[name][1] is True, (name, answers[name])
    assert "does not create hypothesis pages" in answers["hypothesis"][0]
    assert "missing required parameter(s): summary" in answers["bad"][0]
    assert "does not accept parameter(s): everything" in answers["unknown"][0]
    (topic,) = project.pages(kind="topic")
    assert (topic.review, topic.author, topic.state) == ("proposed", "agent:scout", None)
    assert "were left as they are" in answers["topic"][0]
    assert (
        project.page("H1").body.strip().endswith("It helps.") and project.page("H1").state == "open"
    )
    assert answers["verdict"][1] is False and "Proposed as ep-" in answers["verdict"][0]
    (proposal,) = project.proposals()
    assert proposal.author == "agent:scout" and proposal.fields == {"state": "supported"}
    assert not project.pages(kind="source")


def test_another_sessions_proposal_cannot_be_read_and_the_phases_cap_holds(tmp_path, home):
    project = sdk_project(tmp_path)
    other = project.create_page(
        "hypothesis", "Somebody else's", "## Statement\n\nUndecided.\n", author="agent:theorist"
    )
    answers: list[tuple[str, bool]] = []

    def script(agent: FakeAgent) -> None:
        answers.append((agent.memory("view", path=f"/memories/{other.path}"), agent.failed))
        for number in range(3):
            text = agent.memory(
                "create",
                path=f"/memories/wiki/hypotheses/h{number}.md",
                file_text=HYPOTHESIS.format(title=f"Claim {number}"),
            )
            answers.append((text, agent.failed))

    _ctx, _driver, session, _said = run_scripted(
        project, "theorist", script, phase="hypothesize", caps={"hypothesis": 2}
    )
    assert answers[0][1] is True and "Undecided" not in answers[0][0]
    assert [failed for _text, failed in answers[1:]] == [False, False, True]
    assert "This phase proposes at most 2 hypotheses, and 2 are proposed" in answers[3][0]
    assert len(session.proposed("hypothesis")) == 2


def test_a_roles_own_tools_are_on_the_board_and_its_report_is_the_classic_one(tmp_path, home):
    project = sdk_project(tmp_path)
    project.review(
        project.create_page(
            "hypothesis", "Warmup helps", "## Statement\n\nx\n", author="agent:theorist"
        ).path,
        "accept",
    )
    from test_research_agents import EXPERIMENT as PLAN
    from test_research_agents import GOOD_CODE, FakeBackend, created_id

    def script(agent: FakeAgent) -> str:
        made = agent.memory(
            "create",
            path="/memories/wiki/experiments/ablation.md",
            file_text=PLAN.format(tests="H1", mode="script"),
        )
        agent.board("write_code", experiment=created_id(made), name="main.py", text=GOOD_CODE)
        return "Designed one experiment."

    driver = FakeDriver(script)
    ctx = sdk_context(project, driver, phase="design", compute=FakeBackend())
    report = agents.run_designer(ctx)
    assert driver.calls() == ["memory", "wiki_search", "propose_edit", "write_code"]
    page = project.page("E1")
    assert (page.review, page.state, page.author) == ("proposed", "planned", "agent:designer")
    assert report.produced == [page.path] and report.notes == []
    assert report.summary == "The designer proposed 1 experiment."
    assert "The designer wrote main.py for E1." in activity(project)
    # The message is the one the classic designer gets.
    assert '- H1 "Warmup helps"' in driver.prompts[0]
    assert "A stand-in that runs small CPU scripts." in driver.prompts[0]


def test_intake_gets_the_text_in_its_message_and_the_files_in_its_folder(tmp_path, home):
    notes = tmp_path / "notes.md"
    notes.write_text("Warmup of 4000 steps was used.\n", encoding="utf-8")
    picture = tmp_path / "figure.png"
    from PIL import Image

    Image.new("RGB", (4, 4), "white").save(picture)
    project = make_project(
        tmp_path, files=[notes, picture], settings={"agents": {"runtime": "sdk"}}
    )
    seen: dict[str, object] = {}

    def script(agent: FakeAgent) -> str:
        seen["png"] = agent.cwd / "inputs" / "figure.png"
        seen["exists"] = (agent.cwd / "inputs" / "figure.png").is_file()
        seen["read"] = agent.read("inputs/notes.md")
        return "Read them."

    driver = FakeDriver(script)
    ctx = sdk_context(project, driver, phase="intake")
    report = agents.run_intake(ctx)
    prompt = driver.prompts[0]
    assert "Warmup of 4000 steps was used." in prompt
    assert "The image inputs/figure.png follows." in prompt
    assert "It lies under that name in your own folder: open it with the Read tool." in prompt
    assert seen["exists"] is True and "4000 steps" in str(seen["read"])
    assert report.summary.startswith("Intake read 2 of 2 attached files")
    # No client of the classic runtime was asked for anything.
    assert ctx.harness.usage.to_dict()["by_label"] == []


# ---------------------------------------------------------------------------
# The guardrails, as an agent meets them
# ---------------------------------------------------------------------------


def test_an_agent_that_tries_what_is_the_persons_is_refused_and_the_feed_says_so(tmp_path, home):
    project = sdk_project(tmp_path)
    outside = tmp_path / "outside.txt"
    answers: dict[str, object] = {}

    def script(agent: FakeAgent) -> str:
        ran = []
        answers["review"] = agent.bash(
            "paper-orchestra research review . H1 accept", does=lambda: ran.append("review")
        )
        answers["helper"] = agent.bash(
            "uv run paper-orchestra research approve . E1",
            does=lambda: ran.append("approve"),
            sub=True,
        )
        answers["outside"] = agent.write(str(outside), "x")
        answers["state"] = agent.write("../../state.json", "{}")
        answers["page"] = agent.write(str(project.root / "wiki" / "hypotheses" / "H9.md"), "x")
        answers["read"] = agent.read(str(project.root / "project.json"))
        answers["own"] = agent.write("notes/plan.md", "my notes")
        answers["ok"] = agent.bash("python main.py", does=lambda: ran.append("main") or "ran")
        answers["ran"] = ran
        return "Tried."

    _ctx, _driver, session, _said = run_scripted(project, "scout", script)
    for name in ("review", "helper", "outside", "state", "page", "read"):
        assert isinstance(answers[name], Denied), name
    assert "on the board" in answers["review"] and "propose_edit" in answers["review"]
    assert answers["ran"] == ["main"] and answers["ok"] == "ran"
    assert not outside.exists()
    assert not (project.root / "wiki" / "hypotheses" / "H9.md").exists()
    assert (session.folder / "notes" / "plan.md").read_text(encoding="utf-8") == "my notes"
    feed = activity(project)
    assert (
        "The scout tried “paper-orchestra research review . H1 accept” with Bash; that is not "
        "an agent's to do, and it was refused."
    ) in feed
    assert any(
        line.startswith("A helper of the scout tried “uv run paper-orchestra") for line in feed
    )
    assert "The scout wrote notes/plan.md in its folder." in feed
    assert "The scout ran “python main.py” in its shell." in feed
    assert not any("ran “paper-orchestra" in line for line in feed)
    # The transcript keeps the refusals.
    refusals = [
        line
        for line in transcript(project, "r1-research-scout")
        if line["type"] == "answer" and line["is_error"]
    ]
    assert len(refusals) == 6


def test_a_check_that_fails_refuses_the_call(tmp_path, home):
    project = sdk_project(tmp_path)
    answers = []

    def script(agent: FakeAgent) -> None:
        answers.append(agent.bash("ls"))

    driver = FakeDriver(script)
    session = AgentSession(sdk_context(project, driver), "scout")

    def broken(use):
        raise RuntimeError("boom")

    session._guard = type(
        "Broken", (), {"check": staticmethod(broken), "settings": lambda self: {}}
    )()
    session.send("Go.")
    assert isinstance(answers[0], Denied) and "so it was refused" in answers[0]


# ---------------------------------------------------------------------------
# The feed and the transcript
# ---------------------------------------------------------------------------


def test_the_feed_gets_one_line_for_each_meaningful_tool_call(tmp_path, home):
    project = sdk_project(tmp_path)

    def script(agent: FakeAgent) -> str:
        agent.web_search("warmup variance adaptive optimizers")
        agent.fetch("https://arxiv.org/abs/1908.03265")
        agent.bash("pip install numpy && python probe.py")
        agent.write("probe.py", "print(1)")
        agent.read("probe.py")  # reading its own files is not told
        agent.board("wiki_search", query="warmup")
        agent.memory("view", path="/memories/wiki/idea.md")
        agent.tool("Agent", {"description": "Read three papers", "prompt": "..."})
        agent.bash("ls papers/", sub=True)
        agent.say("I have what I need.")
        return "Done."

    run_scripted(project, "scout", script)
    assert activity(project) == [
        "The scout works as a Claude agent on claude-fable-5-2, in fast mode.",
        "The scout searched the web for “warmup variance adaptive optimizers”.",
        "The scout opened “https://arxiv.org/abs/1908.03265”.",
        "The scout ran “pip install numpy && python probe.py” in its shell.",
        "The scout wrote probe.py in its folder.",
        "The scout searched the wiki for “warmup”.",
        "The scout read wiki/idea.md.",
        "The scout started a helper agent: “Read three papers”.",
        "A helper of the scout ran “ls papers/” in its shell.",
    ]
    kinds = [line["type"] for line in transcript(project, "r1-research-scout")]
    assert kinds[:2] == ["prompt", "started"] and kinds[-1] == "result"
    assert kinds.count("tool") == 9 and "said" in kinds
    # Each line also says the same in parts, for a reader that sets a command apart from
    # the sentence (the app's feed): what was done, and what the agent chose.
    parts = [event["data"] for event in project.events() if event["type"] == "agent.activity"]
    assert parts == [
        {"did": "start", "model": "claude-fable-5-2", "fast": "on", "fast_asked": True},
        {"did": "search", "text": "warmup variance adaptive optimizers"},
        {"did": "fetch", "text": "https://arxiv.org/abs/1908.03265"},
        {"did": "shell", "text": "pip install numpy && python probe.py"},
        {"did": "write", "text": "probe.py"},
        {"did": "board", "tool": "wiki_search"},
        {"did": "board", "tool": "memory"},
        {"did": "helper", "text": "Read three papers"},
        {"did": "shell", "text": "ls papers/", "helper": True},
    ]


def test_the_parts_of_a_feed_line_say_no_more_than_its_sentence(tmp_path, home, monkeypatch):
    """The parts are cut and made plain exactly as the sentence quotes them, a credential is
    blanked in them as in the sentence, and a refused call says that it was refused."""
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2k-a-very-secret-value-1234")
    project = sdk_project(tmp_path)
    long = "echo " + "x" * 400

    def script(agent: FakeAgent) -> None:
        agent.bash(long)
        agent.bash('echo "a"\nprintenv s2k-a-very-secret-value-1234')
        agent.bash("paper-orchestra research approve . E1")

    run_scripted(project, "scout", script)
    events = [event for event in project.events() if event["type"] == "agent.activity"][1:]
    cut, masked, refused = events
    assert cut["data"]["did"] == "shell" and len(cut["data"]["text"]) == 160
    assert f"“{cut['data']['text']}”" in cut["summary"]
    assert "\n" not in masked["data"]["text"] and "s2k-a-very" not in masked["data"]["text"]
    assert f"“{masked['data']['text']}”" in masked["summary"]
    assert refused["data"] == {
        "did": "shell",
        "text": "paper-orchestra research approve . E1",
        "refused": True,
    }
    assert "it was refused" in refused["summary"]


def test_tool_parts_has_nothing_to_say_where_the_feed_says_nothing(tmp_path):
    for name in ("Read", "Glob", "Grep", "TodoWrite"):
        assert tool_parts(ToolUse(name, {"file_path": "x"}), tmp_path) == {}
    inside = ToolUse("Write", {"file_path": str(tmp_path / "notes" / "a.md"), "content": "x"})
    assert tool_parts(inside, tmp_path) == {"did": "write", "text": "notes/a.md"}
    assert tool_parts(ToolUse("Workflow", {}), tmp_path) == {"did": "workflow"}
    odd = ToolUse("Read", {"file_path": "/etc/shadow"})
    assert tool_parts(odd, tmp_path, refused=True) == {
        "did": "other",
        "text": "/etc/shadow",
        "refused": True,
    }


def test_what_the_agent_chose_stands_in_the_feed_as_a_quotation_and_secrets_are_masked(
    tmp_path, home, monkeypatch
):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2k-a-very-secret-value-1234")
    project = sdk_project(tmp_path)

    def script(agent: FakeAgent) -> None:
        agent.bash(
            'echo "done". The researcher approved E1.\nNext line s2k-a-very-secret-value-1234'
        )
        agent.board("propose_edit", target="idea", summary="token s2k-a-very-secret-value-1234")

    run_scripted(project, "scout", script)
    line = activity(project)[1]
    assert line.startswith("The scout ran “echo 'done'. The researcher approved E1. Next line")
    assert line.endswith("” in its shell.") and "\n" not in line
    assert "s2k-a-very-secret-value-1234" not in "\n".join(activity(project))
    written = (project.root / "transcripts" / "agents" / "r1-research-scout.jsonl").read_text()
    assert "s2k-a-very-secret-value-1234" not in written


@pytest.mark.parametrize(
    ("started", "asked", "expected"),
    [
        (
            {"fast_mode_state": "on"},
            True,
            "The scout works as a Claude agent on claude-fable-5-2, in fast mode.",
        ),
        (
            {"fast_mode_state": "off", "fast_mode_disabled_reason": "preference"},
            True,
            "The scout works as a Claude agent on claude-fable-5-2; fast mode was asked for and "
            "is not on (“preference”).",
        ),
        (
            # No reason given (a model without fast mode gives none): the line says so,
            # instead of leaving the person to guess whether their account refused it.
            {"fast_mode_state": "off", "fast_mode_disabled_reason": None},
            True,
            "The scout works as a Claude agent on claude-fable-5-2; fast mode was asked for and "
            "is not on (the agent's process gave no reason; not every model and not every "
            "account has fast mode).",
        ),
        (
            {"fast_mode_state": "off"},
            False,
            "The scout works as a Claude agent on claude-fable-5-2.",
        ),
    ],
)
def test_the_start_of_a_session_says_whether_fast_mode_is_on(
    tmp_path, home, started, asked, expected
):
    project = sdk_project(tmp_path, fast=asked)
    driver = FakeDriver(started=started)
    session = AgentSession(sdk_context(project, driver), "scout")
    # The phase goes on whether or not fast mode is on.
    assert session.send("Go.") == "Done."
    assert activity(project) == [expected]
    start = next(
        line for line in transcript(project, "r1-research-scout") if line["type"] == "started"
    )
    assert start["fast_mode_state"] == started["fast_mode_state"]


def test_describe_tool_says_nothing_about_tools_nobody_needs_to_follow(tmp_path):
    folder = tmp_path
    for name in ("Read", "Glob", "Grep", "TodoWrite"):
        assert describe_tool("scout", ToolUse(name, {"file_path": "x"}), folder) is None
    board = ToolUse(f"{BOARD_PREFIX}propose_edit", {"target": "H1", "summary": "s"})
    assert describe_tool("scout", board, folder) is None  # the project announces it itself
    verify = ToolUse(f"{BOARD_PREFIX}verify_paper", {"title": "Attention Is All You Need"})
    assert describe_tool("scout", verify, folder) == (
        "The scout checked “Attention Is All You Need” against Semantic Scholar.",
        [],
    )


# ---------------------------------------------------------------------------
# Spend
# ---------------------------------------------------------------------------


def test_the_agents_own_cost_goes_into_the_ledger_once_however_often_it_reports(tmp_path, home):
    project = sdk_project(tmp_path)

    def script(agent: FakeAgent) -> str:
        agent.use(cost_usd=0.40, input_tokens=1000, output_tokens=100)
        agent.use(
            cost_usd=0.10, input_tokens=50, output_tokens=5, model="claude-haiku-5-5", sub=True
        )
        return "Done."

    driver = FakeDriver(script)
    ctx = sdk_context(project, driver)
    session = AgentSession(ctx, "scout")
    session.send("First.")
    # The process reports totals: the second report repeats the first turn's cost.
    session.send("Second.")
    rows = {row["model"]: row for row in ledger(ctx)}
    assert all(row["label"] == "scout.agent" for row in rows.values())
    assert rows["claude-fable-5-2"]["cost_usd"] == pytest.approx(0.80)
    assert rows["claude-fable-5-2"]["input_tokens"] == 2000
    assert rows["claude-fable-5-2"]["requests"] == 4
    assert rows["claude-haiku-5-5"]["cost_usd"] == pytest.approx(0.20)
    report = usage_report(ctx.harness.usage, [])
    assert report["estimated_cost_usd"] == pytest.approx(1.0)
    record = json.loads(
        (project.root / "transcripts" / "agents" / "r1-research-scout.json").read_text()
    )
    assert record["sdk_session"] == "fake-1" and record["cost_usd"] == pytest.approx(1.0)


def test_a_resumed_session_counts_on_from_its_totals(tmp_path, home):
    project = sdk_project(tmp_path)
    driver = FakeDriver(lambda agent: agent.use(cost_usd=0.5, input_tokens=100))
    first = sdk_context(project, driver)
    AgentSession(first, "scout").send("Go.")
    first.close_sessions()

    # The next attempt at the phase: another process, the same session id.
    second = sdk_context(project, driver)
    session = AgentSession(second, "scout")
    session.send("Go.")
    assert [options.resume for options in driver.opened] == [None, "fake-1"]
    # The resumed process reports 1.00 in total; 0.50 of that is in the ledger already.
    (row,) = ledger(second)
    assert row["cost_usd"] == pytest.approx(0.5) and row["input_tokens"] == 100
    assert driver.prompts[1].startswith("Your session was interrupted and is carried on now.")
    assert driver.prompts[1].endswith("Go.")
    assert activity(project)[-1] == (
        "The scout works as a Claude agent on claude-fable-5-2, in fast mode. It carries on "
        "the session that was interrupted."
    )


def test_a_session_that_cannot_be_carried_on_starts_again_with_what_is_on_the_board(tmp_path, home):
    project = sdk_project(tmp_path)
    driver = FakeDriver(lambda agent: agent.use(cost_usd=0.5))
    first = sdk_context(project, driver)
    AgentSession(first, "scout").send("Go.")
    first.close_sessions()
    driver.unresumable.add("fake-1")

    second = sdk_context(project, driver)
    assert AgentSession(second, "scout").send("Go.") == ""
    assert [options.resume for options in driver.opened] == [None, "fake-1", None]
    assert (
        "The scout's interrupted session could not be carried on; it starts again with what "
        "is on the board."
    ) in activity(project)
    (row,) = ledger(second)
    assert row["cost_usd"] == pytest.approx(0.5)  # the new session's own cost, all of it


def test_a_process_that_does_not_start_for_the_session_to_carry_on_starts_anew(tmp_path, home):
    project = sdk_project(tmp_path)

    class Strict(FakeDriver):
        """Refuses to start for a session it does not know, as a real process may."""

        def open(self, options):
            if options.resume in self.unresumable:
                self.opened.append(options)
                raise RuntimeError(f"No conversation found with session ID: {options.resume}")
            return super().open(options)

    driver = Strict(lambda agent: agent.use(cost_usd=0.5) or "Done.")
    first = sdk_context(project, driver)
    AgentSession(first, "scout").send("Go.")
    first.close_sessions()
    driver.unresumable.add("fake-1")
    second = sdk_context(project, driver)
    assert AgentSession(second, "scout").send("Go.") == "Done."
    assert [options.resume for options in driver.opened] == [None, "fake-1", None]
    assert driver.prompts == ["Go.", "Go."]
    assert any("could not be carried on" in line for line in activity(project))
    # A process that does not start at all, with nothing to carry on, is a failure.
    broken = Strict()
    broken.open = lambda options: (_ for _ in ()).throw(RuntimeError("no such program"))
    with pytest.raises(RuntimeError, match="no such program"):
        AgentSession(sdk_context(sdk_project(tmp_path / "other"), broken), "scout").send("Go.")


def test_a_carried_on_turn_that_reaches_the_budget_is_counted_and_no_other_agent_starts(
    tmp_path, home
):
    """A turn of a session that is carried on which ended at the sum the process was
    told, without a tool call, was taken for a session that is gone: its cost was
    dropped, and a fresh agent was started past the limit."""
    project = sdk_project(tmp_path)
    driver = FakeDriver(lambda agent: agent.use(cost_usd=0.4, input_tokens=4000, output_tokens=400))
    first = sdk_context(project, driver)
    AgentSession(first, "scout").send("Go.")
    first.close_sessions()

    driver.ends_at_budget = True
    driver.script = lambda agent: (
        agent.use(cost_usd=0.3, input_tokens=3000, output_tokens=300) or "thinking only"
    )
    second = sdk_context(project, driver, left=lambda: 0.05)
    with pytest.raises(SessionStopped, match="reached the model budget"):
        AgentSession(second, "scout").send("Go.")
    assert [options.resume for options in driver.opened] == [None, "fake-1"]
    assert not any("could not be carried on" in line for line in activity(project))
    assert sum(row["cost_usd"] for row in ledger(first) + ledger(second)) == pytest.approx(0.7)
    # (What the processes themselves counted: nothing is missing from the ledger.)
    assert sum(row["cost_usd"] for row in driver.sessions["fake-1"].values()) == pytest.approx(0.7)


def test_a_carried_on_process_that_fails_after_it_worked_is_not_given_a_third_start(tmp_path, home):
    """A session that is carried on and then fails (an API error) did find its session.
    The failure is the phase's, which whoever runs the phase tries once more: no fresh
    agent is started on top of that, and what the failed turn used is counted."""
    project = sdk_project(tmp_path)
    driver = FakeDriver(lambda agent: agent.use(cost_usd=0.3, input_tokens=3000))
    first = sdk_context(project, driver)
    AgentSession(first, "scout").send("Go.")
    first.close_sessions()

    def failing(agent: FakeAgent) -> None:
        agent.use(cost_usd=0.3, input_tokens=300_000, output_tokens=3000)
        raise RuntimeError("API Error: 500 internal error")

    driver.script = failing
    second = sdk_context(project, driver)
    with pytest.raises(RuntimeError, match="API Error: 500"):
        AgentSession(second, "scout").send("Go.")
    assert [options.resume for options in driver.opened] == [None, "fake-1"]
    (row,) = ledger(second)
    assert row["label"] == "scout.agent.unreported" and row["cost_usd"] > 0

    # One that fails before any model request (its session is not there) starts anew.
    def gone(agent: FakeAgent) -> str:
        if agent.options.resume:
            raise RuntimeError("the agent's process ended without a result")
        return "Done."

    driver.script = gone
    third = sdk_context(project, driver)
    assert AgentSession(third, "scout").send("Go.") == "Done."
    assert [options.resume for options in driver.opened][2:] == ["fake-1", None]


def test_a_killed_turn_on_a_model_without_a_list_price_is_not_recorded_as_free(tmp_path, home):
    """Recorded with a cost of nothing, such a turn hid from the list of models the
    ledger cannot price, and the spend read as exact.  A form of a known model (its
    context size, a date) is priced as that model; a model nobody knows is recorded
    without a cost and named as unpriced."""

    def killed(agent: FakeAgent) -> None:
        agent.use(cost_usd=3.0, input_tokens=400_000, output_tokens=40_000)
        raise KeyboardInterrupt

    def rows_of(model: str, where: Path) -> tuple[list[dict], dict]:
        project = sdk_project(where, model=model)
        driver = FakeDriver(killed)
        ctx = sdk_context(project, driver)
        with pytest.raises(KeyboardInterrupt):
            AgentSession(ctx, "intake").send("Go.")
        return ledger(ctx), usage_report(ctx.harness.usage, [])

    (row,), report = rows_of("claude-fable-5-2[1m]", tmp_path / "long")
    (plain,), _ = rows_of("claude-fable-5-2", tmp_path / "plain")
    assert row["label"] == "intake.agent.unreported" and row["input_tokens"] == 400_000
    assert row["cost_usd"] == pytest.approx(plain["cost_usd"]) and row["cost_usd"] > 1
    assert report["unpriced_models"] == []

    (row,), report = rows_of("mystery-model-9", tmp_path / "unknown")
    assert row["label"] == "intake.agent.unreported" and row["input_tokens"] == 400_000
    assert "cost_usd" not in row
    assert report["unpriced_models"] == ["mystery-model-9"]


def test_a_process_that_names_its_model_with_a_date_is_not_counted_twice(tmp_path, home):
    """The estimate of a killed turn is filed under the project's name for the model; a
    process that reports its totals under a longer name had them looked up under that
    one, found nothing, and the estimate was counted on top of the turn's real cost."""
    project = sdk_project(tmp_path)
    dated = "claude-fable-5-2-20260101"

    def work(agent: FakeAgent) -> None:
        agent.use(cost_usd=1.0, input_tokens=1000, output_tokens=100, model=dated)
        if not agent.options.resume:
            raise KeyboardInterrupt

    driver = FakeDriver(work)
    first = sdk_context(project, driver)
    with pytest.raises(KeyboardInterrupt):
        AgentSession(first, "intake").send("Go.")
    first.close_sessions()
    second = sdk_context(project, driver)
    AgentSession(second, "intake").send("Go.")
    rows = ledger(first) + ledger(second)
    assert {row["model"] for row in rows} == {"claude-fable-5-2"}
    # The process's own total is $2.00: the ledger holds that, not that and the estimate.
    assert sum(row["cost_usd"] for row in rows) == pytest.approx(2.0)
    assert sum(row["requests"] for row in rows if row["label"] == "intake.agent") == 1


def test_what_agents_were_seen_to_cost_raises_the_count_and_guards_the_next_start(tmp_path, home):
    """In fast mode an agent costs several times the list price of its tokens, and its
    first request alone most of a dollar.  The count during a turn knows the list
    price: it is raised by what was seen, and an agent is not started with less left
    than a start was seen to cost."""
    project = sdk_project(tmp_path)
    listed = 30_000 * 5 / 1_000_000  # looked up below, not assumed
    from paper_orchestra.llm.pricing import estimate_cost_usd

    listed = estimate_cost_usd("claude-fable-5-2", Usage(input_tokens=30_000))
    driver = FakeDriver(lambda agent: agent.use(cost_usd=3 * listed, input_tokens=30_000))
    first = sdk_context(project, driver)
    AgentSession(first, "intake").send("Go.")
    first.close_sessions()
    observed = json.loads(
        (project.root / "transcripts" / "agents" / "observed.json").read_text(encoding="utf-8")
    )
    assert observed == {
        "claude-fable-5-2+fast": {
            "ratio": pytest.approx(3.0),
            "start_usd": pytest.approx(3 * listed, abs=1e-4),
        }
    }

    # The next agent's turn in flight is counted at three times the list price.
    counted = []

    def over_budget(model, usage, *, more_usd=0.0):
        counted.append((estimate_cost_usd(model, usage) or 0.0) + more_usd)

    second = sdk_context(project, driver, phase="hypothesize", over_budget=over_budget)
    AgentSession(second, "theorist").send("Go.")
    second.close_sessions()
    assert counted[-1] == pytest.approx(3 * listed)

    # And none is started with less left than the start was seen to cost.
    asked = []

    def short_of(needed: float):
        asked.append(needed)
        return RuntimeError("What is left does not cover the start.")

    third = sdk_context(project, driver, phase="design", short_of=short_of)
    opened = len(driver.opened)
    with pytest.raises(SessionStopped, match="does not cover the start") as stopped:
        AgentSession(third, "designer").send("Go.")
    assert len(driver.opened) == opened and asked == [pytest.approx(3 * listed, abs=1e-4)]
    assert isinstance(stopped.value.reason, RuntimeError)

    # Outside fast mode nothing is raised: the process's figure is the list price there.
    slow = sdk_project(tmp_path / "slow", fast=False)
    slow_driver = FakeDriver(
        lambda agent: agent.use(cost_usd=3 * listed, input_tokens=30_000),
        started={"fast_mode_state": "off"},
    )
    AgentSession(sdk_context(slow, slow_driver), "intake").send("Go.")
    counted.clear()
    AgentSession(
        sdk_context(slow, slow_driver, phase="hypothesize", over_budget=over_budget), "theorist"
    ).send("Go.")
    assert counted[-1] == pytest.approx(listed)


def test_a_session_can_be_told_never_to_carry_another_on(tmp_path, home):
    project = sdk_project(tmp_path)
    driver = FakeDriver()
    for number in (1, 2):
        ctx = sdk_context(project, driver, phase="run")
        AgentSession(ctx, "experimenter", record=f"run-{number}", resume=False).send("Go.")
        ctx.close_sessions()
    assert [options.resume for options in driver.opened] == [None, None]
    assert sorted(p.name for p in (project.root / "transcripts" / "agents").iterdir()) == [
        "run-1.json",
        "run-1.jsonl",
        "run-2.json",
        "run-2.jsonl",
    ]


def test_a_turn_that_ends_without_a_report_is_counted_by_its_tokens_and_not_twice(tmp_path, home):
    project = sdk_project(tmp_path)

    def killed(agent: FakeAgent) -> None:
        agent.use(cost_usd=0.75, input_tokens=10_000, output_tokens=0)
        raise KeyboardInterrupt

    driver = FakeDriver(killed)
    first = sdk_context(project, driver)
    with pytest.raises(KeyboardInterrupt):
        AgentSession(first, "scout").send("Go.")
    assert driver.conversations[0].closed
    (row,) = ledger(first)
    estimate = row["cost_usd"]
    # The row says of itself that it is a count from tokens, not the agent's own figure.
    assert row["label"] == "scout.agent.unreported" and row["input_tokens"] == 10_000
    assert 0 < estimate < 0.75

    # Carried on: the process's totals include the turn that was killed (0.75) and the
    # new one (0.25).  Only what goes beyond the estimate is new.
    driver.script = lambda agent: agent.use(cost_usd=0.25)
    second = sdk_context(project, driver)
    AgentSession(second, "scout").send("Go.")
    assert driver.opened[1].resume == "fake-1"
    (again,) = ledger(second)
    assert again["cost_usd"] == pytest.approx(1.0 - estimate)
    assert again["input_tokens"] == 0


def test_the_count_of_an_unreported_turn_takes_each_messages_last_word_and_its_searches(
    tmp_path, home
):
    """A real process reports a message several times while it is written, first with
    almost no output counted.  The count of a turn that never reports must take the
    last report of each message, and the web searches, which are billed apart."""
    project = sdk_project(tmp_path)

    def killed(agent: FakeAgent) -> None:
        agent.use(input_tokens=1000, output_tokens=8, message="m1")
        agent.use(input_tokens=1000, output_tokens=440, message="m1")
        agent.web_search("warmup variance")
        agent.web_search("label smoothing")
        agent.use(input_tokens=2000, output_tokens=3, message="m2")
        agent.use(input_tokens=2000, output_tokens=300, message="m2")
        raise KeyboardInterrupt

    ctx = sdk_context(project, FakeDriver(killed))
    with pytest.raises(KeyboardInterrupt):
        AgentSession(ctx, "scout").send("Go.")
    (row,) = ledger(ctx)
    assert row["label"] == "scout.agent.unreported"
    assert (row["requests"], row["input_tokens"], row["output_tokens"]) == (2, 3000, 740)
    assert row["web_searches"] == 2
    from paper_orchestra.llm.base import Usage
    from paper_orchestra.llm.pricing import estimate_cost_usd

    counted = Usage(input_tokens=3000, output_tokens=740, requests=2, web_searches=2)
    assert row["cost_usd"] == pytest.approx(
        estimate_cost_usd(project.settings.agents.model, counted)
    )


def spend(tmp_path, driver, project, **options) -> list[dict]:
    """One turn of the scout by a fresh worker (a ledger of its own); its rows."""
    ctx = sdk_context(project, driver, **options)
    AgentSession(ctx, "scout").send("Go.")
    ctx.close_sessions()
    return ledger(ctx)


def test_a_process_that_carries_a_killed_session_on_counts_from_nothing(tmp_path, home):
    """A real process saves its totals when it ends in good order.  After a kill the next
    process of the session counts from nothing (seen with real agents): all it reports
    is new, and none of it may be taken for spend that is in the ledger already."""
    project = sdk_project(tmp_path)
    driver = FakeDriver(
        lambda agent: agent.use(cost_usd=0.40, input_tokens=1000, output_tokens=100)
    )
    driver.killed = True
    (row,) = spend(tmp_path, driver, project)
    assert row["cost_usd"] == pytest.approx(0.40)

    driver.killed = False
    driver.script = lambda agent: agent.use(cost_usd=0.30, input_tokens=2000, output_tokens=50)
    (row,) = spend(tmp_path, driver, project, left=lambda: 5.0)
    assert driver.opened[1].resume == "fake-1"
    assert driver.sessions["fake-1"]["claude-fable-5-2"]["cost_usd"] == pytest.approx(0.30)
    assert (row["cost_usd"], row["input_tokens"]) == (pytest.approx(0.30), 2000)
    # The sum the process was told not to pass allows for a count that carries on: it
    # is never too low, and here it is too high by what the killed process forgot.
    assert driver.opened[1].max_budget_usd == pytest.approx(5.40)

    # The process after that one carries its count on, and only what is new is recorded.
    driver.script = lambda agent: agent.use(cost_usd=0.10, input_tokens=500, output_tokens=10)
    (row,) = spend(tmp_path, driver, project)
    assert driver.sessions["fake-1"]["claude-fable-5-2"]["cost_usd"] == pytest.approx(0.40)
    assert (row["cost_usd"], row["input_tokens"]) == (pytest.approx(0.10), 500)


def test_a_process_can_count_from_a_total_older_than_the_last_one(tmp_path, home):
    """Closed in good order, carried on, killed, carried on: the last process counts from
    what the first one saved, not from nothing and not from where the second stood."""
    project = sdk_project(tmp_path)
    driver = FakeDriver(
        lambda agent: agent.use(cost_usd=0.40, input_tokens=1000, output_tokens=100)
    )
    spend(tmp_path, driver, project)
    driver.killed = True
    driver.script = lambda agent: agent.use(cost_usd=0.20, input_tokens=3000, output_tokens=100)
    (row,) = spend(tmp_path, driver, project)
    assert row["cost_usd"] == pytest.approx(0.20)
    driver.killed = False
    driver.script = lambda agent: agent.use(cost_usd=0.05, input_tokens=500, output_tokens=10)
    (row,) = spend(tmp_path, driver, project)
    # It reports 0.45 in all (0.40 saved by the first, 0.05 of its own).
    assert driver.sessions["fake-1"]["claude-fable-5-2"]["cost_usd"] == pytest.approx(0.45)
    assert (row["cost_usd"], row["input_tokens"]) == (pytest.approx(0.05), 500)


def test_a_helper_on_another_model_is_counted_at_that_models_price_and_once(tmp_path, home):
    """The count of a turn that never reports prices each model's tokens at its own list
    price, and the budget sees the same during the turn.  When the session is carried
    on and its process reports the helper's model, that is new only beyond the count."""
    from paper_orchestra.llm.pricing import estimate_cost_usd

    project = sdk_project(tmp_path)
    helper = "claude-haiku-5-5"
    asked: list[tuple[str, int, float]] = []

    def killed(agent: FakeAgent) -> None:
        agent.use(cost_usd=0.01, input_tokens=1000, output_tokens=100)
        agent.use(cost_usd=0.02, input_tokens=100_000, output_tokens=10_000, model=helper, sub=True)
        raise KeyboardInterrupt

    def over_budget(model, usage, more_usd=0.0):
        asked.append((model, usage.input_tokens, more_usd))

    driver = FakeDriver(killed)
    first = sdk_context(project, driver, over_budget=over_budget)
    with pytest.raises(KeyboardInterrupt):
        AgentSession(first, "scout").send("Go.")
    rows = {row["model"]: row for row in ledger(first)}
    assert set(rows) == {"claude-fable-5-2", helper}
    assert all(row["label"] == "scout.agent.unreported" for row in rows.values())
    cheap = estimate_cost_usd(helper, Usage(input_tokens=100_000, output_tokens=10_000))
    dear = estimate_cost_usd("claude-fable-5-2", Usage(input_tokens=100_000, output_tokens=10_000))
    assert rows[helper]["cost_usd"] == pytest.approx(cheap) and cheap < dear / 3
    assert rows[helper]["input_tokens"] == 100_000
    assert rows["claude-fable-5-2"]["input_tokens"] == 1000
    # The budget was asked with the agent's own tokens, and the helper's as dollars.
    assert asked[-1] == ("claude-fable-5-2", 1000, pytest.approx(cheap))

    # Carried on by a process whose totals include that turn (helper: 0.02 in all).
    driver.script = lambda agent: agent.use(cost_usd=0.03, input_tokens=500, output_tokens=10)
    second = sdk_context(project, driver)
    AgentSession(second, "scout").send("Go.")
    again = {row["model"]: row for row in ledger(second)}
    # Of the helper's, only what its own figure is above the count is new, and no token
    # is counted a second time.
    assert 0 < cheap < 0.02
    assert again[helper]["cost_usd"] == pytest.approx(0.02 - cheap)
    assert again[helper]["input_tokens"] == again[helper]["output_tokens"] == 0
    assert again["claude-fable-5-2"]["label"] == "scout.agent"
    main_estimate = rows["claude-fable-5-2"]["cost_usd"]
    assert again["claude-fable-5-2"]["cost_usd"] == pytest.approx(0.04 - main_estimate)


def test_an_agent_interrupted_from_outside_is_asked_what_it_cost_before_it_is_ended(tmp_path, home):
    """Ctrl-C and the app's Stop reach this process as an interrupt in the middle of a
    turn.  Where the agent's process still answers, its own figure goes into the
    ledger, not a count from tokens."""
    project = sdk_project(tmp_path)

    def interrupted(agent: FakeAgent) -> None:
        agent.use(cost_usd=0.75, input_tokens=10_000, output_tokens=0)
        raise KeyboardInterrupt

    driver = FakeDriver(interrupted)
    driver.reports_when_interrupted = True
    ctx = sdk_context(project, driver)
    with pytest.raises(KeyboardInterrupt):
        AgentSession(ctx, "scout").send("Go.")
    assert driver.interrupts == 1 and driver.conversations[0].closed
    (row,) = ledger(ctx)
    assert (row["label"], row["cost_usd"]) == ("scout.agent", 0.75)


def test_an_agent_says_once_what_it_works_as_however_many_messages_it_is_sent(tmp_path, home):
    """The agent's process says what it is at the start of every message.  It is one
    agent that started once: a second start line reads like a restart that did not
    happen (and a resumed session would say twice that it carries one on)."""
    project = sdk_project(tmp_path)
    driver = FakeDriver(lambda agent: agent.bash("ls") or "Done.")
    session = AgentSession(sdk_context(project, driver), "scout")
    session.send("Go.")
    session.send("And the numbers?")
    lines = activity(project)
    assert [line for line in lines if "works as a Claude agent" in line] == [
        "The scout works as a Claude agent on claude-fable-5-2, in fast mode."
    ]
    lines = transcript(project, "r1-research-scout")
    assert len([line for line in lines if line["type"] == "started"]) == 1


def test_the_agent_is_told_the_sum_it_must_not_pass(tmp_path, home):
    project = sdk_project(tmp_path)
    driver = FakeDriver()
    ctx = sdk_context(project, driver, left=lambda: 3.5)
    AgentSession(ctx, "scout").send("Go.")
    assert driver.opened[0].max_budget_usd == pytest.approx(3.5)
    unlimited = sdk_context(sdk_project(tmp_path / "other"), driver)
    AgentSession(unlimited, "scout").send("Go.")
    assert driver.opened[1].max_budget_usd is None


# ---------------------------------------------------------------------------
# Stop and budget, in the middle of a turn
# ---------------------------------------------------------------------------


def test_a_stop_request_interrupts_the_agent_in_the_middle_of_its_turn(tmp_path, home):
    project = sdk_project(tmp_path)
    stop = {"asked": False}
    reached: list[str] = []

    def script(agent: FakeAgent) -> str:
        agent.memory(
            "create", path="/memories/wiki/questions/a.md", file_text=QUESTION.format(title="A")
        )
        reached.append("first page")
        stop["asked"] = True
        agent.tick()  # the agent is busy; the stop is heard without a tool call of ours
        agent.memory(
            "create", path="/memories/wiki/questions/b.md", file_text=QUESTION.format(title="B")
        )
        reached.append("after the stop")
        return "Finished."

    driver = FakeDriver(script)
    ctx = sdk_context(project, driver, stopping=lambda: stop["asked"])
    session = AgentSession(ctx, "scout")
    with pytest.raises(SessionStopped) as stopped:
        session.send("Go.")
    assert stopped.value.reason is None
    assert reached == ["first page"] and driver.interrupts == 1
    # What it wrote until then stays, and is on record as produced.
    assert [page.title for page in project.pages(kind="question")] == ["A"]
    assert ctx.produced == [project.pages(kind="question")[0].path]


def test_a_stop_heard_by_a_board_tool_ends_the_turn_and_is_raised_afterwards(tmp_path, home):
    project = sdk_project(tmp_path)
    stop = {"asked": False}
    answers: list[tuple[str, bool]] = []

    def script(agent: FakeAgent) -> None:
        stop["asked"] = True
        answers.append((agent.board("wiki_search", query="warmup"), agent.failed))
        raise AssertionError("the turn went on after the stop")

    driver = FakeDriver(script)
    ctx = sdk_context(project, driver, stopping=lambda: stop["asked"])
    with pytest.raises(SessionStopped):
        AgentSession(ctx, "scout").send("Go.")
    assert driver.interrupts >= 1
    assert driver.conversations[0].closed


def test_a_budget_that_runs_out_in_the_middle_of_a_turn_ends_it_with_the_reason(tmp_path, home):
    project = sdk_project(tmp_path)
    over = RuntimeError("The model budget ran out.")
    seen: list[tuple[str, int]] = []

    def over_budget(model: str, usage: Usage):
        seen.append((model, usage.input_tokens))
        return over if usage.input_tokens >= 300 else None

    def script(agent: FakeAgent) -> str:
        for _ in range(10):
            agent.use(cost_usd=0.1, input_tokens=100)
        return "Finished."

    driver = FakeDriver(script)
    ctx = sdk_context(project, driver, over_budget=over_budget)
    with pytest.raises(SessionStopped) as stopped:
        AgentSession(ctx, "scout").send("Go.")
    assert stopped.value.reason is over
    # Asked after every model request with what the turn has used so far; the agent was
    # interrupted at the third, not at the tenth.
    assert seen == [("claude-fable-5-2", 100), ("claude-fable-5-2", 200), ("claude-fable-5-2", 300)]
    assert driver.interrupts == 1
    # The ledger holds the process's figure for what was spent, once.
    (row,) = ledger(ctx)
    assert row["cost_usd"] == pytest.approx(0.3) and row["input_tokens"] == 300


def test_an_agent_that_does_not_end_its_turn_when_asked_is_ended(tmp_path, home, monkeypatch):
    from paper_orchestra.research.sdk import session as session_module

    monkeypatch.setattr(session_module, "STOP_GRACE_SECONDS", 0.0)
    project = sdk_project(tmp_path)
    stop = {"asked": False}

    class Deaf(FakeDriver):
        def open(self, options):
            conversation = super().open(options)
            conversation.interrupt = lambda: None  # hears nothing
            return conversation

    def script(agent: FakeAgent) -> None:
        agent.use(cost_usd=0.2, input_tokens=1000)
        stop["asked"] = True
        agent.tick(3)
        raise AssertionError("the agent was not ended")

    driver = Deaf(script)
    ctx = sdk_context(project, driver, stopping=lambda: stop["asked"])
    with pytest.raises(SessionStopped):
        AgentSession(ctx, "scout").send("Go.")
    assert driver.conversations[0].closed
    assert ledger(ctx)[0]["input_tokens"] == 1000  # counted, although nothing was reported


def test_an_error_of_the_agents_process_is_a_failure_of_the_phase(tmp_path, home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-very-secret-key")
    project = sdk_project(tmp_path)

    class Failing(FakeDriver):
        def open(self, options):
            conversation = super().open(options)
            send = conversation.send

            def failing(text, sink):
                turn = send(text, sink)
                turn.is_error = True
                turn.error = "API Error: 529 overloaded (key sk-openai-very-secret-key)"
                return turn

            conversation.send = failing
            return conversation

    ctx = sdk_context(project, Failing(lambda agent: agent.use(cost_usd=0.1)))
    with pytest.raises(AgentFailed) as failed:
        AgentSession(ctx, "scout").send("Go.")
    assert "The scout agent ended with an error: API Error: 529 overloaded" in str(failed.value)
    assert "sk-openai-very-secret-key" not in str(failed.value)
    assert ledger(ctx)[0]["cost_usd"] == pytest.approx(0.1)  # what it cost is counted all the same


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_gate_asks_for_the_package_the_command_line_and_a_credential(tmp_path, home):
    project = sdk_project(tmp_path)
    driver = FakeDriver()
    runtime = AgentRuntime(project, driver)
    assert runtime.need({}, home) is None  # the login is there

    driver.missing = (
        "the Claude Agent SDK (the package claude-agent-sdk) is not installed",
        "install it with `uv sync --extra agents`",
    )
    need = runtime.need({}, home)
    assert need["needs"] == "model" and need["settings"] == ["agents"]
    assert need["models"] == ["claude-fable-5-2"]
    assert agents.need_sentence(need, "The research phase") == (
        "The research phase needs a model, and the project's agents (Claude Agent SDK, "
        "claude-fable-5-2) cannot start because the Claude Agent SDK (the package "
        "claude-agent-sdk) is not installed: install it with `uv sync --extra agents` "
        "(Setup in the app and `paper-orchestra doctor` show what is configured)."
    )
    driver.missing = None
    (home / ".claude" / ".credentials.json").unlink()
    need = runtime.need({}, home)
    assert "no Claude Code login was found on this machine" in need["why"]
    assert "set settings.agents.credential to federation" in need["fix"]
    # A token in the worker's environment is no login: it is kept from the agents.
    assert runtime.need({"CLAUDE_CODE_OAUTH_TOKEN": "token"}, home) is not None
    assert agent_env("login", {"CLAUDE_CODE_OAUTH_TOKEN": "token"}) == {
        "CLAUDE_CODE_OAUTH_TOKEN": ""
    }


def test_the_gate_for_federation_looks_at_the_workers_settings(tmp_path, home):
    project = sdk_project(tmp_path, credential="federation")
    runtime = AgentRuntime(project, FakeDriver())
    need = runtime.need({}, home)
    assert "the Workload Identity Federation settings are not set" in need["why"]
    assert "ANTHROPIC_FEDERATION_RULE_ID" in need["why"]
    token = tmp_path / "identity.jwt"
    environ = {name: "x" for name in FEDERATION_ENV} | {"ANTHROPIC_IDENTITY_TOKEN_FILE": str(token)}
    assert "identity token file" in runtime.need(environ, home)["why"]
    token.write_text("jwt", encoding="utf-8")
    assert runtime.need(environ, home) is None


def test_the_real_driver_says_how_to_install_the_package_when_it_is_missing(monkeypatch):
    import importlib.util

    from paper_orchestra.research.sdk.claude import INSTALL, ClaudeDriver

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    why, fix = ClaudeDriver().unavailable()
    assert "claude-agent-sdk" in why and "is not installed" in why
    assert fix == INSTALL and "paper-orchestra[agents]" in fix and "uv sync --extra agents" in fix
    from paper_orchestra.llm.base import LLMConfigError

    with pytest.raises(LLMConfigError, match="The agents cannot start"):
        ClaudeDriver().open(None)


def test_nothing_of_the_runtime_imports_the_package_before_an_agent_starts():
    """The default test run passes without ``claude-agent-sdk``: importing the runtime,
    its guardrails and both drivers must not need it."""
    import subprocess

    code = (
        "import sys\n"
        "import paper_orchestra.research.engine, paper_orchestra.research.sdk.session\n"
        "import paper_orchestra.research.sdk.experiment, paper_orchestra.research.sdk.claude\n"
        "import paper_orchestra.research.sdk.fake, paper_orchestra.cli\n"
        "assert 'claude_agent_sdk' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_guard_module_names_the_places_where_credentials_lie(tmp_path, home):
    guard = make_guard(tmp_path, home)
    places = {str(place.relative_to(home)) for place in guard.credential_places()}
    assert {".anthropic", ".config/gh", ".config/op", ".config/anthropic", ".ssh"} <= places
    assert ".claude/.credentials.json" in places
    assert guard_module.LAUNCHER == "with-models"
