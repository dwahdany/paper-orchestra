"""One role of a research project as a Claude Agent SDK agent.

:class:`AgentSession` is to the ``sdk`` runtime what ``research.agents._Session``
is to the classic one, with the same face (``send``, ``proposed``, ``created``,
``suggested``, ``backend``), so that the role functions, their user messages and
the checks they run afterwards are the same code in both runtimes.

What differs is what the session is: an agent with a shell in a folder of its
own, the web, and the *board tools*.  Those are the project's existing tools
(``ToolSpec`` objects: ``wiki_search``, ``propose_edit``, a role's own) and the
memory tool over the session's ``_SessionBackend``, offered to the agent as
in-process tools.  The wiki is reached through them alone, so every rule of the
project (who creates which kind, what a session may read, the phase's caps)
holds unchanged.

The session also does what the classic harness did for a turn: it counts what
the agent spends into the project's ledger (the agent's own dollar figure), ends
the agent when a stop is requested or the budget is spent, tells the activity
feed what the agent does, and keeps a transcript.  A session that was
interrupted is carried on where the agent's own session file allows it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from paper_orchestra.compute.base import write_record
from paper_orchestra.llm.anthropic_client import _check_tool_input, _result_text, run_memory_command
from paper_orchestra.llm.base import (
    Capabilities,
    Image,
    Part,
    Pdf,
    Text,
    ToolError,
    ToolSpec,
    Usage,
)
from paper_orchestra.llm.pricing import estimate_cost_usd
from paper_orchestra.memory.wiki import TRANSCRIPTS_DIR
from paper_orchestra.research import agents, experiments, prompts
from paper_orchestra.research.access import propose_edit_tool, search_tool
from paper_orchestra.research.model import EditProposal, ResearchPage
from paper_orchestra.research.project import Project
from paper_orchestra.research.sdk.driver import (
    BOARD_PREFIX,
    AgentConversation,
    AgentDriver,
    AgentOptions,
    Answered,
    BoardTool,
    Said,
    Started,
    Tick,
    ToolResult,
    ToolUse,
    Turn,
    Used,
    bare_name,
)
from paper_orchestra.research.sdk.guard import (
    FEDERATION_ENV,
    Guard,
    SandboxUnavailable,
    agent_env,
    claude_config_dir,
    make_temp,
    sandbox_fault,
)

__all__ = [
    "AGENTS_DIR",
    "BUILTIN_TOOLS",
    "HELPER_TOOLS",
    "AgentFailed",
    "AgentRuntime",
    "AgentSession",
    "describe_tool",
    "login_file",
    "login_found",
    "machine_report",
    "memory_tool",
    "output_tail",
    "tool_parts",
]

#: Where the agents' own folders lie in a project: ``agents/<session>/``.
AGENTS_DIR = "agents"
#: Where the transcripts and the session records of SDK agents lie, under ``transcripts/``.
SESSIONS_DIR = "agents"
#: The built-in tools every SDK agent has, and those ``ultracode`` adds.
BUILTIN_TOOLS: tuple[str, ...] = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
)
HELPER_TOOLS: tuple[str, ...] = ("Agent", "Task", "Workflow")
#: Seconds an agent that was asked to stop is given to end its turn before its process
#: is ended for it.
STOP_GRACE_SECONDS = 30.0
#: Seconds an agent whose turn was interrupted from outside (Ctrl-C, the app's Stop) is
#: given to say what its session cost.  Well below the time the app waits before it
#: kills a worker that was asked to stop.
INTERRUPT_REPORT_SECONDS = 5.0
#: What a ledger row of an SDK agent is labelled, after its role: ``scout.agent``.
LABEL_SUFFIX = ".agent"
#: What is added to that label for spend the agent's process never reported, because it
#: was stopped or killed first: ``scout.agent.unreported``.  Such a row is counted from
#: the tokens the turn's messages named, each model's at its own list price, and is a
#: lower bound (fast mode costs more than that, and a message can end longer than its
#: last report said).
UNREPORTED_SUFFIX = ".unreported"

_USAGE_NAMES = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
#: The file under ``transcripts/agents/`` that holds what the project's agents were
#: seen to cost (:meth:`AgentSession._observe`).
OBSERVED_FILE = "observed.json"
#: The most an agent's cost in fast mode is taken to be above the list price of its tokens.
_MAX_RATIO = 6.0
#: What the agent's process says when it was told to require the sandbox and cannot
#: start it.
_NO_SANDBOX = "Sandbox required but unavailable"
#: What it says of a session it is asked to carry on and does not have.
_NO_SESSION = "No conversation found"
#: What a row of a session's totals holds, per model.
_ROW_NAMES = (*_USAGE_NAMES, "web_searches", "cost_usd")
#: The counters of a row by which the point a process counts from is told: what went
#: into the model.  (Not the output: a message's last report can name less than it
#: came to.)
_INPUT_NAMES = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
#: How many earlier totals of a session are kept, per model, to tell that point by.
_MAX_HISTORY = 40


def _row(given: Any) -> dict[str, float]:
    given = given if isinstance(given, Mapping) else {}
    row = {}
    for name in _ROW_NAMES:
        try:
            row[name] = float(given.get(name) or 0)
        except (TypeError, ValueError):
            row[name] = 0.0
    return row


#: Lines of a shell command's output that go into the log of a run, and their length.
OUTPUT_LINES = 40
OUTPUT_LINE_CHARS = 400


def output_tail(text: str) -> str:
    """The end of what a shell command printed, for the run's log: whole lines, bounded."""
    lines = [line[:OUTPUT_LINE_CHARS] for line in str(text or "").splitlines()]
    if not lines:
        return ""
    left_out = len(lines) - OUTPUT_LINES
    if left_out > 0:
        lines = [f"({left_out} earlier lines of this output are not shown)", *lines[-OUTPUT_LINES:]]
    return "\n".join(lines) + "\n"


def _listed(model: str, usage: Usage) -> float | None:
    """What ``usage`` costs on ``model`` at the list price; ``None`` where there is none.

    A model can be named with more than the price list knows: the size of its
    context (``claude-fable-5-2[1m]``) or a date.  It is then priced as the model
    it is a form of, which is nearer the truth than counting it as nothing.
    """
    bare = re.sub(r"\[[^\]]*\]$", "", model)
    for name in dict.fromkeys((model, bare, re.sub(r"-\d{8}$", "", bare))):
        cost = estimate_cost_usd(name, usage)
        if cost is not None:
            return cost
    return None


def _row_of(given: Any, names: Sequence[str]) -> dict[str, float]:
    """The numbers ``names`` of a mapping read from a file; what is no number is left out."""
    row: dict[str, float] = {}
    if isinstance(given, Mapping):
        for name in names:
            value = given.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                row[name] = float(value)
    return row


def _with_temp(guard: Any, temp: Path | None) -> Any:
    """``guard`` knowing the agent's own directory for temporary files (a guard that is
    not this package's keeps its own counsel)."""
    if dataclasses.is_dataclass(guard) and not isinstance(guard, type):
        return dataclasses.replace(guard, temp=temp)
    return guard


def _cli_reason(text: str) -> str:
    """What the agent's process said of the sandbox it could not start, without what it
    says around that (its advice to allow commands outside the sandbox is not ours)."""
    said = text.split(_NO_SANDBOX, 1)[-1].lstrip(": ").split(" \u00b7 ", 1)[0].strip()
    return (
        "the agent's process could not start that sandbox"
        + (f" ({said.rstrip('.')})" if said else "")
        + ": install what it names, or set settings.agents.sandbox to false if you "
        "accept agents whose shell reaches everything yours does"
    )


def _no_sandbox_sentence(role: str, why: str) -> str:
    """What the person is told when an agent was not started, or was ended, for want of
    the sandbox the project asks for."""
    return (
        f"The {role} agent was not started: the project's agents work with their shell in "
        f"a sandbox (settings.agents.sandbox), and {_lower_first(why.rstrip('.'))}. Nothing "
        "was run outside the sandbox."
    )


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


class AgentFailed(RuntimeError):
    """The agent's process reported an error instead of an answer."""

    #: Its message is the whole of it (what the agent's process said): whoever logs it
    #: adds no traceback.
    plain = True


class _Unresponsive(Exception):
    """An agent that was asked to stop did not end its turn in time."""


def memory_tool(backend: Any) -> ToolSpec:
    """The memory tool as a board tool: the commands of the provider's memory tool over
    the session's backend, which decides what may be read and written."""
    return ToolSpec(
        name="memory",
        description=(
            "Read and write the project's wiki and read the researcher's attached files, "
            "under /memories/. Commands: view (a directory or a file; view_range gives "
            "lines), create (a new page, or the whole new text of a page of yours), "
            "str_replace, insert, delete. /memories/MEMORY.md is the index. What a "
            "command may do is decided by the project and said in its answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
                },
                "path": {"type": "string", "description": "A path under /memories/."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "For view: [first line, last line]; -1 is the end.",
                },
                "file_text": {"type": "string", "description": "For create: the whole text."},
                "old_str": {"type": "string", "description": "For str_replace."},
                "new_str": {"type": "string", "description": "For str_replace."},
                "insert_line": {"type": "integer", "description": "For insert."},
                "insert_text": {"type": "string", "description": "For insert."},
                "old_path": {"type": "string", "description": "For rename."},
                "new_path": {"type": "string", "description": "For rename."},
            },
            "required": ["command"],
        },
        handler=lambda tool_input: run_memory_command(backend, tool_input),
    )


def _folder_path(path: Any, folder: Path) -> str:
    """A path the agent named, as it reads in the feed: relative to its folder where it
    lies there."""
    text = str(path or "")
    with contextlib.suppress(ValueError):
        return str(Path(os.path.realpath(text)).relative_to(os.path.realpath(folder)))
    return text


def describe_tool(role: str, use: ToolUse, folder: Path) -> tuple[str, list[str]] | None:
    """One plain sentence for the feed about a tool call of an SDK agent, or ``None``.

    A board tool is described as the classic runtime describes it.  Of the
    built-in tools, what a person following the work wants to see: a shell
    command, a web search or a page opened, a file written, a helper started.
    Reading and searching its own files is not told.  Everything the agent
    chose stands in the sentence as a quotation or a plain name.
    """
    who = f"A helper of the {role}" if use.sub else f"The {role}"
    given = use.input if isinstance(use.input, Mapping) else {}
    name = use.name
    if name.startswith(BOARD_PREFIX):
        event = {"type": "tool_call", "name": bare_name(name), "input": given}
        return agents.describe_event(role, event)
    if name == "Bash":
        return f"{who} ran {agents.said(given.get('command'), 160)} in its shell.", []
    if name == "WebSearch":
        return f"{who} searched the web for {agents.said(given.get('query'), 100)}.", []
    if name == "WebFetch":
        return f"{who} opened {agents.said(given.get('url'), 160)}.", []
    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = given.get("file_path") or given.get("notebook_path")
        return f"{who} wrote {agents.named(_folder_path(path, folder), 100)} in its folder.", []
    if name in ("Agent", "Task"):
        about = given.get("description") or given.get("prompt")
        return f"{who} started a helper agent: {agents.said(about, 120)}.", []
    if name == "Workflow":
        return f"{who} started a workflow of helper agents.", []
    return None


#: What a built-in tool does, in one word, as ``data.did`` of the feed's line says it.
_DID = {
    "Bash": ("shell", "command"),
    "WebSearch": ("search", "query"),
    "WebFetch": ("fetch", "url"),
    "Write": ("write", "file_path"),
    "Edit": ("write", "file_path"),
    "MultiEdit": ("write", "file_path"),
    "NotebookEdit": ("write", "notebook_path"),
    "Agent": ("helper", "description"),
    "Task": ("helper", "description"),
    "Workflow": ("workflow", ""),
}


def tool_parts(use: ToolUse, folder: Path, *, refused: bool = False) -> dict[str, Any]:
    """The parts of the feed's sentence about a tool call (``data`` of the event), for a
    reader that sets a command apart from the words around it.

    ``did`` is what kind of thing the agent did (``shell``, ``search``, ``fetch``,
    ``write``, ``helper``, ``workflow``, ``board``), ``text`` what it chose (the
    command, the query, the address, the file), made plain and cut exactly as the
    sentence quotes it: the parts say nothing the sentence does not.  ``helper`` is
    set for a call of a helper agent, ``refused`` for a call that was not allowed.
    """
    given = use.input if isinstance(use.input, Mapping) else {}
    if use.name.startswith(BOARD_PREFIX):
        parts: dict[str, Any] = {"did": "board", "tool": bare_name(use.name)}
    else:
        did, key = _DID.get(use.name, ("", ""))
        if not did:
            if not refused:
                return {}
            did, key = "other", ""
        chosen = given.get(key) if key else None
        if did == "helper" and not chosen:
            chosen = given.get("prompt")
        if did == "write":
            chosen = _folder_path(chosen, folder)
        if refused and chosen is None:
            chosen = given.get("command") or given.get("file_path") or given.get("path")
        parts = {"did": did}
        if chosen is not None:
            # The words of a quotation without its marks: one line, cut, nothing hidden.
            parts["text"] = agents.said(chosen, 160)[1:-1]
    if use.sub:
        parts["helper"] = True
    if refused:
        parts["refused"] = True
    return parts


def login_file(environ: Mapping[str, str], home: Path) -> Path:
    """The file Claude Code keeps the person's login in, where it keeps it in a file:
    in its configuration directory, which ``CLAUDE_CONFIG_DIR`` can name."""
    return claude_config_dir(environ, home) / ".credentials.json"


def login_found(environ: Mapping[str, str], home: Path, platform: str | None = None) -> bool | None:
    """Whether there is a Claude Code login on this machine: ``True`` when its file is
    there, ``False`` when it is not, and ``None`` where that cannot be told from a file
    (on macOS the login is in the Keychain, and the first agent's start decides)."""
    if login_file(environ, home).is_file():
        return True
    return None if (platform or sys.platform) == "darwin" else False


def machine_report(
    environ: Mapping[str, str] | None = None, home: Path | None = None
) -> dict[str, Any]:
    """What this machine has of what SDK agents need, for ``paper-orchestra doctor`` and
    Setup in the app: the package, a command line, and each kind of credential.
    Nothing is imported and nothing contacted; no value of a setting is in it."""
    from paper_orchestra.research.sdk.claude import INSTALL, availability

    environ = os.environ if environ is None else environ
    home = home or Path.home()
    found = availability()
    missing = [name for name in FEDERATION_ENV if not environ.get(name)]
    token = environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE") or ""
    no_sandbox = sandbox_fault(environ.get("PATH"))
    return {
        "package": found["package"],
        "command_line": found["command_line"],
        "install": INSTALL,
        # Whether the sandbox of an agent's shell can start here, and if not, why and
        # what to do: a project with ``agents.sandbox`` on starts no agent without it.
        "sandbox": no_sandbox is None,
        "sandbox_missing": f"{no_sandbox[0]}: {no_sandbox[1]}" if no_sandbox else "",
        # true, false, or null where a file does not tell (macOS)
        "login": login_found(environ, home),
        "login_file": str(login_file(environ, home)),
        "federation": not missing and Path(token).is_file(),
        "federation_missing": missing,
        "federation_token_file": bool(token) and Path(token).is_file(),
    }


class AgentRuntime:
    """The ``sdk`` runtime of one project: how its agents are started.

    ``driver`` starts them; without one the real driver is used, which needs the
    optional package.  The engine holds one runtime and hands it to every role
    through ``AgentContext.runtime``.
    """

    #: What an SDK agent takes in a message: images and PDFs are read from its folder.
    capabilities = Capabilities(
        tools=True, memory_tool=True, web_search=True, pdf_input=True, image_input=True
    )

    def __init__(self, project: Project, driver: AgentDriver | None = None) -> None:
        self.project = project
        self._driver = driver

    @property
    def driver(self) -> AgentDriver:
        if self._driver is None:
            from paper_orchestra.research.sdk.claude import ClaudeDriver

            self._driver = ClaudeDriver()
        return self._driver

    @property
    def settings(self) -> Any:
        return self.project.settings.agents

    # -- the model gate ------------------------------------------------------------

    def sandbox_fault(self, path: str | None = None) -> tuple[str, str] | None:
        """Why the sandbox the project asks for cannot start for an agent whose process
        is started with ``path``; ``None`` when it can, and for a project that asks
        for none.  The driver answers: it knows what its agents run in."""
        if not self.settings.sandbox:
            return None
        asked = getattr(self.driver, "sandbox_unavailable", None)
        return asked(path) if asked is not None else None

    def _credential_fault(self, environ: Mapping[str, str], home: Path) -> tuple[str, str] | None:
        if self.settings.credential == "federation":
            missing = [name for name in FEDERATION_ENV if not environ.get(name)]
            if missing:
                return (
                    "the Workload Identity Federation settings are not set for this "
                    f"process ({', '.join(missing)})",
                    "set them where the worker starts, or set settings.agents.credential "
                    "to login to use your Claude Code login",
                )
            if not Path(environ["ANTHROPIC_IDENTITY_TOKEN_FILE"]).is_file():
                return (
                    "the identity token file that ANTHROPIC_IDENTITY_TOKEN_FILE names is not there",
                    "start whatever keeps that file fresh, or set "
                    "settings.agents.credential to login",
                )
            return None
        # (A token in this process's environment is no login for the agents: like every
        # credential here it is kept from the agent's process.)
        if login_found(environ, home) is False:
            return (
                "no Claude Code login was found on this machine",
                "run `claude` once and sign in, or set settings.agents.credential to "
                "federation to use the worker's Workload Identity Federation settings",
            )
        return None

    def need(
        self, environ: Mapping[str, str] | None = None, home: Path | None = None
    ) -> dict[str, Any] | None:
        """What keeps the project's SDK agents from starting, in the shape of
        ``agents.model_need`` (for ``agents.need_sentence``); ``None`` when they can.

        Four things are looked at, and nothing is contacted: the package can be
        imported, there is a command line for the agents to run in, the sandbox
        of their shell can start where the project asks for one, and there is a
        credential of the kind the project chose.
        """
        environ = os.environ if environ is None else environ
        fault = (
            self.driver.unavailable()
            or self.sandbox_fault(environ.get("PATH"))
            or self._credential_fault(environ, home or Path.home())
        )
        if fault is None:
            return None
        model = str(self.settings.model)
        return {
            "needs": "model",
            "settings": ["agents"],
            "models": [model],
            "why": f"the project's agents (Claude Agent SDK, {model}) cannot start because "
            f"{fault[0]}",
            "fix": fault[1],
        }


class AgentSession:
    """One role's agent: its memory backend, its board tools, its folder, what it wrote."""

    def __init__(
        self,
        ctx: agents.AgentContext,
        role: str,
        *,
        tools: Sequence[ToolSpec] | Callable[[Any], Sequence[ToolSpec]] = (),
        web_search: Any = None,
        caps: Mapping[str, int] | None = None,
        folder: Path | None = None,
        guard: Guard | None = None,
        job_models: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        deadline: float | None = None,
        on_line: Callable[[str], None] | None = None,
        on_output: Callable[[str], None] | None = None,
        max_turns: int | None = agents.MAX_TURNS,
        record: str | None = None,
        resume: bool = True,
    ) -> None:
        """``tools`` are the role's own, as for the classic session; ``web_search`` is
        accepted for the same reason and not needed (every SDK agent has the web).

        The rest is for an agent that is not a phase's role (the one that carries out
        an experiment): its ``folder`` and ``guard`` where they are not the usual
        ones, the federation settings its code may use (``job_models``), settings of
        the agent's process itself (``env``: how long a command may run), a further
        reason to stop (``should_stop``), the moment its time is up (``deadline``,
        on ``time.monotonic()``), ``on_line``, which is given every line the
        feed is given and what the agent says, and ``on_output``, which is given
        the end of what each of its shell commands printed (the log of a run is
        where a person looks for the experiment's own output).  ``record`` names the session's
        record and transcript where the session id does not tell this agent from
        another (every run of a run phase has the same), and ``resume=False``
        makes it start anew whatever is on file: a run begins in a fresh folder.
        """
        runtime = ctx.runtime
        if runtime is None:
            raise RuntimeError("an SDK agent needs the sdk runtime (AgentContext.runtime)")
        ctx.guard()
        self.ctx = ctx
        self.role = role
        self.runtime: AgentRuntime = runtime
        project = ctx.project
        self.session_id = ctx.session_id(role)
        self.backend = agents._SessionBackend(
            project,
            role,
            ctx.round,
            self.session_id,
            caps=caps,
            heed_stop=ctx.heed_stop,
            handover=ctx.handover,
        )
        if callable(tools):
            tools = tools(self.backend)
        #: Edit proposals that were on file before this attempt: what comes after is its own.
        self._known_edits = {proposal.id for proposal in project.proposals(review=None)}
        offered = [
            memory_tool(self.backend),
            search_tool(project),
            propose_edit_tool(project, role, ctx.round),
            *tools,
        ]
        self._board = [self._board_tool(agents._stoppable(tool, ctx.heed_stop)) for tool in offered]
        self.folder = (
            Path(folder) if folder is not None else project.root / AGENTS_DIR / self.session_id
        )
        self.folder.mkdir(parents=True, exist_ok=True)
        settings = runtime.settings
        self._guard = guard or Guard(
            folder=self.folder, project=project.root, sandbox=bool(settings.sandbox)
        )
        #: The agent's own directory for temporary files while its process runs.
        self._temp: Path | None = None
        self._job_models = dict(job_models or {})
        self._env = dict(env or {})
        self._should_stop = should_stop
        self._deadline = deadline
        self._on_line = on_line
        self._on_output = on_output
        #: The shell commands that are under way, by the id of their call.
        self._shell_calls: set[str] = set()
        self._max_turns = max_turns
        self._model = str(settings.model)
        self._label = f"{role}{LABEL_SUFFIX}"
        records = project.root / TRANSCRIPTS_DIR / SESSIONS_DIR
        named = record or self.session_id
        self._record_path = records / f"{named}.json"
        self._transcript_path = records / f"{named}.jsonl"
        self._may_resume = resume
        self._conversation: AgentConversation | None = None
        self._resumed = False
        #: The agent's own session, as its process names it: what a later attempt resumes.
        self._sdk_session = ""
        #: Totals the agent's processes have reported for this session, by model, oldest
        #: first, all of them in the ledger: the point a new process counts from is one
        #: of these, or nothing (:meth:`_counts_from`).
        self._history: dict[str, list[dict[str, float]]] = {}
        #: What the process that runs now last reported, by model; empty until it has.
        self._seen: dict[str, dict[str, float]] = {}
        #: The same for a process that reports one sum and no models.
        self._seen_total: float | None = None
        self._last_total = 0.0
        self._last = ""
        #: What the turn in flight has used so far, by model.
        self._turn: dict[str, Usage] = {}
        #: What each message of the turn in flight has used, as it was last reported,
        #: and the model it was written by.
        self._by_message: dict[str, tuple[str, Usage]] = {}
        self._searches = 0
        #: Whether this conversation has said what it works as (:meth:`_announce`).
        self._announced = False
        self._acted = False
        #: Whether the process has reported a model request since the last message.
        self._heard = False
        #: What the first model request of the process that runs now used.
        self._first: Usage | None = None
        #: How many times the list price this project's agents were seen to cost in fast
        #: mode (the least that was seen: the count during a turn must not run ahead of
        #: the truth), and whether the process that runs now says fast mode is on.
        self._ratio = max(1.0, self._observed().get("ratio") or 1.0)
        self._fast_on = False
        #: Set when the agent was asked to end its turn, and why.
        self._asked_at: float | None = None
        self._stopped: BaseException | None = None
        self.timed_out = False
        self.cancelled = False
        ctx.sessions.append(self)

    # -- the board -------------------------------------------------------------------

    def _board_tool(self, spec: ToolSpec) -> BoardTool:
        """``spec`` as the agent calls it: validated, answered in text, never raising.

        A stop request or a spent budget is raised by the project's tools as
        ``SessionStopped``, which is no ``Exception`` so that no client mistakes
        it for a tool's error.  Here it cannot travel through the agent's
        process: it is kept, the agent is asked to stop, and :meth:`send` raises
        it once the turn has ended.
        """

        def call(tool_input: dict[str, Any]) -> ToolResult:
            self._acted = True
            try:
                _check_tool_input(spec, tool_input)
                return ToolResult(_result_text(spec.handler(tool_input)))
            except ToolError as exc:
                return ToolResult(str(exc) or "The call was refused.", True)
            except agents.SessionStopped as exc:
                self._stopped = self._stopped or exc
                self._ask_to_stop()
                return ToolResult("The session is ending now; this call did nothing.", True)
            except Exception as exc:
                why = experiments.mask_credentials(self.ctx.project, f"{type(exc).__name__}: {exc}")
                return ToolResult(why, True)

        return BoardTool(spec.name, spec.description, spec.input_schema, call)

    # -- starting the agent ------------------------------------------------------------

    def _record(self) -> dict[str, Any]:
        try:
            data = json.loads(self._record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _options(self, resume: str | None) -> AgentOptions:
        settings = self.runtime.settings
        helpers = bool(settings.ultracode)
        # (Fast mode is only on when the settings the process is started with ask for
        # it; what came of it is said in the feed, :meth:`_announce`.)
        left = self.ctx.budget_left()
        temp = self._make_temp()
        process = dict(self._guard.settings())
        if settings.fast:
            process["fastMode"] = True
        if helpers:
            process.update(ultracode=True, enableWorkflows=True, skipWorkflowUsageWarning=True)
        places = {
            # `python` in the agent's shell is the interpreter this process runs with:
            # the packages the project brings are the ones its code finds.  (The
            # directory as it is named, not where it leads: a virtual environment is
            # told by where its `python` lies.)
            "PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), os.environ.get("PATH", os.defpath)]
            ),
            # Its temporary files lie in a directory made for it alone, not in the one
            # that every program of the person shares: the agent's process puts its
            # own there and names a directory in it to the shell as TMPDIR.
            "CLAUDE_CODE_TMPDIR": str(temp),
            "CLAUDE_TMPDIR": "",
            # The process's own, too: it keeps there the sockets that connect a
            # sandboxed command to the network, and the shared directory, where they
            # would lie otherwise, is closed to the sandbox (checked live: without
            # this the shell has no network).
            "TMPDIR": str(temp),
        }
        return AgentOptions(
            model=self._model,
            cwd=self.folder,
            system_prompt=prompts.system_prompt(
                self.role, index=self.backend.index(), runtime="sdk", helpers=helpers
            ),
            board=self._board,
            builtin=(*BUILTIN_TOOLS, *(HELPER_TOOLS if helpers else ())),
            settings=process,
            env={
                **agent_env(str(settings.credential), job_models=self._job_models),
                **places,
                **self._env,
            },
            # (Asked of the guard as it is when the call comes.)
            check=lambda use: self._guard.check(use),
            resume=resume,
            max_turns=self._max_turns,
            # The agent's process counts its own cost, sub-agents and fast mode
            # included, and ends the session when it has spent what is left.  The
            # process of a session that is carried on counts from a total an earlier
            # one left, at most the last one reported: with that added, the sum is
            # never too low, and it can be too high by what that process forgot
            # (:meth:`_counts_from`).  The count during a turn and the look at the
            # ledger between turns do not depend on it.
            max_budget_usd=None if left is None else max(0.01, left + self._last_reported()),
        )

    def _open(self, *, fresh: bool = False) -> AgentConversation:
        if self._conversation is not None:
            return self._conversation
        record = {} if fresh or not self._may_resume else self._record()
        resume = str(record.get("sdk_session") or "") or None
        self._resumed = resume is not None
        self._sdk_session = resume or ""
        self._history = self._history_of(record) if resume else {}
        self._last_total = float(record.get("cost_usd") or 0.0) if resume else 0.0
        self._seen, self._seen_total = {}, None
        self._announced = False
        self._first = None
        options = self._options(resume)
        # With the sandbox asked for, no agent starts where it cannot: its process would
        # run the agent's commands outside any sandbox.  (Asked of the path the
        # process is started with; the process is told to refuse besides.)
        missing = self.runtime.sandbox_fault(options.env.get("PATH"))
        if missing is not None:
            raise SandboxUnavailable(_no_sandbox_sentence(self.role, f"{missing[0]}: {missing[1]}"))
        self._short_of_a_start()
        try:
            self._conversation = self.runtime.driver.open(options)
        except Exception as exc:
            if _NO_SANDBOX in str(exc):
                why = experiments.mask_credentials(self.ctx.project, str(exc))
                raise SandboxUnavailable(_no_sandbox_sentence(self.role, _cli_reason(why))) from exc
            raise
        return self._conversation

    # -- what the project's agents were seen to cost ----------------------------------

    def _observed_key(self) -> str:
        return f"{self._model}{'+fast' if self.runtime.settings.fast else ''}"

    def _observed(self) -> dict[str, float]:
        """What agents of this project on this model were last seen to cost: how many
        times the list price of their tokens (``ratio``), and what the first model
        request of an agent came to (``start_usd``).  Empty before any was seen."""
        path = self._record_path.parent / OBSERVED_FILE
        try:
            found = json.loads(path.read_text(encoding="utf-8")).get(self._observed_key())
        except (OSError, ValueError, AttributeError):
            return {}
        return _row_of(found, ("ratio", "start_usd"))

    def _observe(self, cost_usd: float, turn: Mapping[str, Usage], first: Usage | None) -> None:
        """Take note of what a turn that has just been accounted for cost against the
        list price of its tokens.  Fast mode costs several times the list price,
        and the count during a turn knows only the list price: the next count is
        raised by what was seen here, and the next agent is not started with less
        left than a start was seen to cost."""
        used = turn.get(self._model)
        listed = _listed(self._model, used) if used is not None else None
        if not listed or cost_usd <= 0:
            return
        seen: dict[str, float] = {}
        before = self._observed()
        if self._fast_on:
            ratio = min(_MAX_RATIO, max(1.0, cost_usd / listed))
            seen["ratio"] = round(min(ratio, before.get("ratio") or ratio), 4)
        start = _listed(self._model, first) if first is not None else None
        if start:
            seen["start_usd"] = round(cost_usd * min(1.0, start / listed), 4)
        if not seen:
            return
        path = self._record_path.parent / OBSERVED_FILE
        with contextlib.suppress(OSError, ValueError):
            try:
                everything = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                everything = {}
            if not isinstance(everything, dict):
                everything = {}
            kept = everything.get(self._observed_key())
            everything[self._observed_key()] = {**(kept if isinstance(kept, dict) else {}), **seen}
            path.parent.mkdir(parents=True, exist_ok=True)
            write_record(path, everything)
        self._ratio = seen.get("ratio", self._ratio)

    def _short_of_a_start(self) -> None:
        """End the session before its agent starts when what is left of the model budget
        is less than the first request of an agent was seen to cost here: started all
        the same, it would pass the limit with its first request (in fast mode, by
        most of a dollar within seconds)."""
        start = self._observed().get("start_usd") or 0.0
        if start <= 0:
            return
        why = self.ctx.short_of(start)
        if why is not None:
            raise agents.SessionStopped(str(why), reason=why)

    @staticmethod
    def _history_of(record: Mapping[str, Any]) -> dict[str, list[dict[str, float]]]:
        """The totals on file for a session.  A record written before there was a
        history holds the last totals alone (``model_usage``)."""
        history = record.get("history")
        if isinstance(history, dict):
            return {
                str(model): [_row(row) for row in rows if isinstance(row, dict)]
                for model, rows in history.items()
                if isinstance(rows, list)
            }
        last = record.get("model_usage")
        if isinstance(last, dict):
            return {str(model): [_row(row)] for model, row in last.items()}
        return {}

    def _last_reported(self) -> float:
        """What the session had cost when a process of it last said so: the most a
        process that carries it on counts from."""
        by_model = sum(rows[-1]["cost_usd"] for rows in self._history.values() if rows)
        return max(float(by_model), self._last_total)

    def close(self) -> None:
        """End the agent's process.  What it wrote on the board stays; its session can be
        carried on by a later attempt at the phase.  Its temporary files go with it."""
        conversation, self._conversation = self._conversation, None
        if conversation is not None:
            with contextlib.suppress(Exception):
                conversation.close()
        temp, self._temp = self._temp, None
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
            self._guard = _with_temp(self._guard, None)

    def _make_temp(self) -> Path:
        """The agent's own directory for temporary files, made when its process is
        started and known to its guard from then on."""
        if self._temp is None:
            self._temp = make_temp()
            self._guard = _with_temp(self._guard, self._temp)
        return self._temp

    # -- following the agent -------------------------------------------------------------

    def _log(self, kind: str, **data: Any) -> None:
        """One line of the session's transcript; credentials of this process blanked."""
        line = json.dumps(
            {"at": time.time(), "type": kind, **data}, ensure_ascii=False, default=str
        )
        with contextlib.suppress(OSError):
            self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with self._transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(experiments.mask_credentials(self.ctx.project, line) + "\n")

    def _tell(
        self, sentence: str, refs: Sequence[str] = (), data: Mapping[str, Any] | None = None
    ) -> None:
        """One line for the activity feed (and the run's log), unless it repeats the last.
        ``data`` is the sentence in parts (:func:`tool_parts`)."""
        if sentence == self._last:
            return
        self._last = sentence
        self.ctx.activity(sentence, agent=self.role, refs=refs, data=data)
        if self._on_line is not None:
            self._on_line(sentence)

    def _announce(self, data: Mapping[str, Any]) -> None:
        """Say what the agent's process reports about itself: its model, and whether fast
        mode is on.  A project that asked for fast mode and did not get it is told why;
        the phase goes on all the same."""
        if data.get("session_id"):
            # On file from the start: a session that is killed can still be carried on.
            self._sdk_session = str(data["session_id"])
            self._save_record()
        if self._announced:
            # The agent's process says this again for every message it is sent.  It is
            # one agent that started once: a second line would read like a restart.
            return
        self._announced = True
        model = agents.named(data.get("model") or self._model, 60)
        state = str(data.get("fast_mode_state") or "off")
        self._fast_on = state == "on" and bool(self.runtime.settings.fast)
        said = f"The {self.role} works as a Claude agent on {model}"
        if not self.runtime.settings.fast:
            said += "."
        elif state == "on":
            said += ", in fast mode."
        else:
            reason = data.get("fast_mode_disabled_reason")
            # The process does not always give a reason (it gives none for a model
            # that has no fast mode): the line then says that it gave none.
            why = (
                f" ({agents.said(reason, 80)})"
                if reason
                else " (the agent's process gave no reason; not every model and not "
                "every account has fast mode)"
            )
            said += f"; fast mode was asked for and is not on{why}."
        if self._resumed:
            said += " It carries on the session that was interrupted."
        self._tell(
            said,
            data={
                "did": "start",
                "model": model,
                "fast": "on" if state == "on" else "off",
                "fast_asked": bool(self.runtime.settings.fast),
            },
        )
        self._log(
            "started",
            model=data.get("model"),
            fast_mode_state=state,
            fast_mode_disabled_reason=data.get("fast_mode_disabled_reason"),
            tools=data.get("tools"),
            resumed=self._resumed,
        )

    def _ending(self) -> bool:
        if self._stopped is not None or self.ctx.ending():
            return True
        if self._should_stop is not None and self._should_stop():
            self.cancelled = True
            return True
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self.timed_out = True
            return True
        return False

    def _ask_to_stop(self) -> None:
        """Ask the agent to end its turn; once is enough, whoever notices first."""
        if self._asked_at is not None:
            return
        self._asked_at = time.monotonic()
        conversation = self._conversation
        if conversation is not None:
            conversation.interrupt()

    def _watch(self, event: Any) -> None:
        """Follow the agent: count what it uses, tell the feed what it does, and ask it
        to stop when a stop was requested, the budget is spent or its time is up."""
        if isinstance(event, Started):
            self._announce(event.data)
        elif isinstance(event, Used):
            self._count(event)
        elif isinstance(event, ToolUse):
            self._acted = True
            if event.name == "WebSearch":
                # Billed apart from the tokens; counted for a turn that never reports.
                self._searches += 1
                self._turn.setdefault(self._model, Usage()).web_searches = self._searches
            self._log("tool", id=event.id, name=event.name, input=event.input, sub=event.sub)
            if event.name == "Bash" and event.id:
                self._shell_calls.add(event.id)
            if not self._ending():
                described = self._described(event)
                if described is not None:
                    self._tell(*described)
        elif isinstance(event, Answered):
            # Kept for whoever reads the transcript; long answers are cut.
            self._log(
                "answer",
                id=event.id,
                text=event.text[:4000],
                is_error=event.is_error,
                sub=event.sub,
            )
            if event.id in self._shell_calls:
                self._shell_calls.discard(event.id)
                if self._on_output is not None and event.text.strip():
                    self._on_output(output_tail(event.text))
        elif isinstance(event, Said):
            self._log("said", text=event.text, sub=event.sub)
            if self._on_line is not None and not event.sub:
                for line in event.text.strip().splitlines()[:40]:
                    self._on_line(line)
        if self._ending():
            asked = self._asked_at
            self._ask_to_stop()
            if (
                asked is not None
                and isinstance(event, Tick)
                and time.monotonic() - asked > STOP_GRACE_SECONDS
            ):
                raise _Unresponsive()

    def _described(self, use: ToolUse) -> tuple[str, list[str], dict[str, Any]] | None:
        """The feed's sentence about a tool call the agent announced, with what it
        refers to and its parts.  A call the guard refuses is told as refused: the
        person should see what was tried, and that it did not happen."""
        try:
            refused = self._guard.check(use)
        except Exception:
            refused = "refused"
        if refused is None:
            described = describe_tool(self.role, use, self.folder)
            if described is None:
                return None
            return described[0], described[1], tool_parts(use, self.folder)
        who = f"A helper of the {self.role}" if use.sub else f"The {self.role}"
        given = use.input if isinstance(use.input, Mapping) else {}
        tried = given.get("command") or given.get("file_path") or given.get("path") or use.name
        return (
            f"{who} tried {agents.said(tried, 140)} with {agents.named(use.name, 40)}; "
            "that is not an agent's to do, and it was refused.",
            [],
            tool_parts(use, self.folder, refused=True),
        )

    def _model_of(self, event: Used) -> str:
        """The model a message was written by, under the name the session's totals use:
        the agent's own model is named there as the project names it."""
        return self._named(str(event.model or ""))

    def _named(self, model: str) -> str:
        """``model`` under the name the session's totals use: the agent's own model is
        named as the project names it, whatever the process adds to the name (a
        date) or leaves out."""
        if not model or model.startswith(self._model) or self._model.startswith(model):
            return self._model
        return model

    def _by_name(self, usage: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        """The totals a process reported, by the names the session uses.  A process
        that names the agent's model otherwise than the project does would have its
        totals looked up under one name and filed under another, and what a
        killed turn was estimated at would be counted on top of its real cost."""
        named: dict[str, dict[str, float]] = {}
        for model, given in usage.items():
            row = _row(given)
            into = named.setdefault(self._named(str(model)), _row({}))
            for name in _ROW_NAMES:
                into[name] += row[name]
        return named

    def _count(self, event: Used) -> None:
        """Take note of what one model request used, and let the budget see the turn so
        far.  This is an estimate from tokens, for ending a long turn in time and for a
        turn that never reports; what goes into the ledger otherwise is the agent's
        own figure, when the turn has ended.

        The agent's process reports a message more than once while it is written,
        the first time with almost no output counted: the last report stands.  A
        helper can work on another model than the agent: each model's tokens are
        kept apart and priced at that model's own list price.
        """
        usage = event.usage
        self._heard = True
        named = self._model_of(event)
        used_now = Usage(
            **{name: agents._whole(usage.get(name)) for name in _USAGE_NAMES}, requests=1
        )
        if self._first is None and named == self._model and not event.sub:
            self._first = used_now
        self._by_message[event.message_id] = (named, used_now)
        turn: dict[str, Usage] = {self._model: Usage(web_searches=self._searches)}
        for model, used in self._by_message.values():
            turn.setdefault(model, Usage()).add(used)
        self._turn = turn
        others = sum(
            _listed(model, used) or 0.0 for model, used in turn.items() if model != self._model
        )
        # Whoever is asked prices the agent's own model by its name at the list price.
        # What it was seen to cost above that (fast mode) is added, and all of it where
        # the name has no price of its own: the budget must not learn of either only
        # when the turn has ended.
        own = turn[self._model]
        ratio = self._ratio if self._fast_on else 1.0
        above = ratio * (_listed(self._model, own) or 0.0) - (
            estimate_cost_usd(self._model, own) or 0.0
        )
        self.ctx.used(self._model, own, more_usd=others + max(0.0, above))

    def _counts_from(self, model: str, row: Mapping[str, float]) -> dict[str, float]:
        """The totals the process that runs now started its count of ``model`` from, told
        at its first report (``row``).

        A process that carries a session on does not always count on from where the
        last one stood.  It counts from the totals a process of the session saved
        when it ended in good order: after a process was killed that is an older
        total, or nothing (checked against real agents).  So the point is measured:
        what the report names, less what the messages of this turn used, is where
        the count began, and the latest total on file that is not beyond that point
        is taken for it.  Every total on file is in the ledger, so what the report
        names beyond the one taken is new.  Where the measure fails (a message
        whose usage was never reported), the latest total is taken and nothing is
        counted twice; what a killed process had not saved is then missed.
        """
        used = self._turn.get(model, Usage())
        began = {name: row[name] - getattr(used, name) for name in _INPUT_NAMES}
        # (A report that names no tokens at all is told by its dollars alone.  They are
        # not asked otherwise: a count from tokens on file can be above the process's
        # own figure for the same turn.)
        bare = not any(row[name] for name in _USAGE_NAMES)
        for known in reversed(self._history.get(model, [])):
            if (
                all(known[name] <= began[name] for name in _INPUT_NAMES)
                and known["output_tokens"] <= row["output_tokens"]
                and (not bare or known["cost_usd"] <= row["cost_usd"] + 1e-9)
            ):
                return known
        return _row({})

    def _remember(self, model: str, row: dict[str, float]) -> None:
        rows = self._history.setdefault(model, [])
        if not rows or rows[-1] != row:
            rows.append(row)
            del rows[:-_MAX_HISTORY]

    def _account(self, turn: Turn) -> None:
        """Put what the agent's session cost since it was last heard of into the ledger.

        The agent's process reports totals for its session, per model.  What is new
        since its last report is recorded once, with the dollar figure the process
        gave; at the first report of a process, since the totals it started its
        count from (:meth:`_counts_from`).
        """
        ledger = self.ctx.harness.usage
        for model, row in self._by_name(turn.model_usage).items():
            before = self._seen.get(model) or self._counts_from(model, row)
            new = {name: max(0.0, row[name] - before[name]) for name in _ROW_NAMES}
            self._seen[model] = row
            self._remember(model, row)
            if not any(new.values()):
                continue
            if model == self._model:
                self._observe(new["cost_usd"], self._turn, self._first)
            ledger.record(
                self._label,
                model,
                Usage(
                    **{name: int(new[name]) for name in _USAGE_NAMES},
                    web_searches=int(new["web_searches"]),
                    requests=turn.requests if model == self._model else 0,
                ),
                cost_usd=new["cost_usd"],
            )
        if not turn.model_usage and turn.cost_usd is not None:
            total = float(turn.cost_usd)
            before = self._seen_total
            if before is None:
                # A process that counts on names at least what the last one named.
                before = self._last_total if total >= self._last_total else 0.0
            if total > before:
                ledger.record(
                    self._label, self._model, Usage(requests=turn.requests), cost_usd=total - before
                )
            self._seen_total = self._last_total = total
        # What the ledger has of this turn is the process's figure now, not an estimate.
        self._turn = {}
        self._by_message = {}
        self._searches = 0
        # (Only the first turn of a process tells what a start costs.)
        self._first = Usage()
        if turn.session_id:
            self._sdk_session = turn.session_id
        self._save_record()

    # -- a turn ----------------------------------------------------------------------

    def _as_text(self, parts: Part | Sequence[Part] | str) -> str:
        """A message of the role function as the text an agent is sent.  An image or a
        PDF is not sent: it lies in the agent's folder, and the message says so."""
        if isinstance(parts, str):
            return parts
        if isinstance(parts, (Text, Image, Pdf)):
            parts = [parts]
        said = []
        for part in parts:
            if isinstance(part, Text):
                said.append(part.text)
            elif isinstance(part, str):
                said.append(part)
            else:
                said.append(
                    "(That file is not part of this message. It lies under that name in "
                    "your own folder: open it with the Read tool.)"
                )
        return "\n\n".join(said)

    def give_inputs(self) -> None:
        """Copy the person's attached files into the agent's folder (``inputs/``), where
        its file tools can open what a message cannot carry: images and PDFs."""
        source = self.ctx.project.inputs_dir
        target = self.folder / "inputs"
        if not source.is_dir() or target.exists():
            return
        shutil.copytree(
            source,
            target,
            symlinks=False,
            ignore=lambda folder, names: [
                n for n in names if n.startswith(".") or os.path.islink(os.path.join(folder, n))
            ],
        )

    def _exchange(self, text: str) -> Turn:
        conversation = self._open()
        try:
            return conversation.send(text, self._watch)
        except _Unresponsive:
            self.close()
            self._account_unreported()
            return Turn(ended="closed")
        except KeyboardInterrupt:
            # Ctrl-C, or the app's Stop: the agent is asked what its session cost before
            # its process is ended.  What it wrote on the board stays.
            turn = self._last_report(conversation)
            self.close()
            if turn is None:
                self._account_unreported()
            else:
                self._account(turn)
            self._log(
                "result",
                text="",
                ended=(turn.ended if turn is not None else None) or "interrupted",
                error=None,
                cost_usd=turn.cost_usd if turn is not None else None,
                session=turn.session_id if turn is not None else self._sdk_session,
            )
            raise
        except BaseException as exc:
            # A failure of the agent's process: it is ended, and what it used is counted
            # from what its messages said.
            self.close()
            self._account_unreported()
            if isinstance(exc, Exception) and _NO_SANDBOX in str(exc):
                why = experiments.mask_credentials(self.ctx.project, str(exc))
                raise SandboxUnavailable(_no_sandbox_sentence(self.role, _cli_reason(why))) from exc
            raise

    def _last_report(self, conversation: AgentConversation) -> Turn | None:
        """How the turn that was interrupted from outside ended, when the agent's process
        still says so within :data:`INTERRUPT_REPORT_SECONDS`; ``None`` when it does not
        (the interrupt reached it too, and it is gone) or the person interrupts again."""
        finish = getattr(conversation, "finish", None)
        if finish is None:
            return None

        def follow(event: Any) -> None:
            if isinstance(event, Used):
                self._count(event)

        try:
            conversation.interrupt()
            turn = finish(follow, INTERRUPT_REPORT_SECONDS)
        except BaseException:
            return None
        return turn if isinstance(turn, Turn) and turn.cost_usd is not None else None

    def _account_unreported(self) -> None:
        """Count a turn whose process ended without saying what it cost (it was killed,
        or it failed): by the tokens its messages reported and the web searches it
        made, each model's at its own list price.

        The rows say of themselves that they are such a count
        (:data:`UNREPORTED_SUFFIX`): a lower bound, since the list price knows
        nothing of fast mode, and whoever shows the spend says so.  The totals the
        process would have reported are put on file as it would have named them, so
        far as they can be told.  When the session is carried on by a process whose
        totals include this turn, only what goes beyond the estimate is new; when
        its totals do not (the process was killed, and counts from an older point),
        the estimate stands alone.  Nothing is counted twice either way.
        """
        ledger = self.ctx.harness.usage
        for model, used in self._turn.items():
            if not any(used.to_dict().values()):
                continue
            priced = _listed(model, used)
            estimate = priced or 0.0
            # A model without a list price is recorded without a cost, not with a cost
            # of nothing: the ledger then names it among the models it cannot price,
            # and whoever shows the spend says that the sum leaves it out.
            ledger.record(f"{self._label}{UNREPORTED_SUFFIX}", model, used, cost_usd=priced)
            rows = self._history.get(model) or []
            base = self._seen.get(model) or (rows[-1] if rows else _row({}))
            would = dict(base)
            for name in (*_USAGE_NAMES, "web_searches"):
                would[name] += getattr(used, name)
            would["cost_usd"] += estimate
            self._remember(model, would)
            self._last_total += estimate
        self._seen, self._seen_total = {}, None
        self._turn = {}
        self._by_message = {}
        self._searches = 0
        self._save_record()

    def _save_record(self) -> None:
        """Keep what a later attempt needs to carry this session on: the agent's own
        session, and what of its cost is in the ledger already."""
        if not self._sdk_session:
            return
        with contextlib.suppress(OSError):
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
            write_record(
                self._record_path,
                {
                    "session": self.session_id,
                    "role": self.role,
                    "sdk_session": self._sdk_session,
                    "model": self._model,
                    "folder": str(self.folder),
                    # What the session's processes said it had cost, last and before
                    # (:meth:`_counts_from`); all of it is in the ledger.
                    "cost_usd": self._last_reported(),
                    "model_usage": {m: rows[-1] for m, rows in self._history.items() if rows},
                    "history": self._history,
                    "transcript": self._transcript_path.name,
                },
            )

    def send(self, parts: Part | Sequence[Part] | str) -> str:
        self.ctx.guard()
        text = self._as_text(parts)
        self._turn = {}
        self._by_message = {}
        self._searches = 0
        self._asked_at = None
        self._acted = False
        self._heard = False
        try:
            return self._converse(text)
        finally:
            # Also when the turn was cut short: what it wrote by then is in the wiki.
            self.ctx.note(self.created())

    def _gone(self, turn: Turn) -> bool:
        """Whether ``turn``, the answer to the first message of a process that was to
        carry a session on, says that the session is not there any more: the process
        says so in words, or it answers with an error without having made one model
        request.  An error after the agent had worked (a failing API, a budget it
        reached) is none of that: its cost counts, and it ends the turn as any error
        does."""
        if not turn.is_error or self._acted or turn.ended == "error_max_budget_usd":
            return False
        said = str(turn.error or "")
        if _NO_SANDBOX in said:
            return False  # no other agent would fare better
        return _NO_SESSION in said or not (self._heard or turn.requests)

    def _first_exchange(self, text: str) -> Turn:
        """The first message of this process to the agent: to the session an earlier
        attempt left, where there is one and it can still be carried on, else to a new
        agent, which finds on the board what the earlier one wrote."""
        try:
            self._open()
            if not self._resumed:
                self._log("prompt", text=text)
                return self._exchange(text)
            carried_on = (
                "Your session was interrupted and is carried on now. What you had put on "
                "the board is still there: look at the wiki before you write anything "
                "again. The task, as it was given:\n\n" + text
            )
            self._log("prompt", text=carried_on)
            turn = self._exchange(carried_on)
            if not self._gone(turn):
                return turn
            # Let go of, and counted all the same where it says that it cost something.
            self._account(turn)
        except SandboxUnavailable:
            raise
        except Exception as exc:
            # (Not a stop and not Ctrl-C, which are no ``Exception``.)  An agent's process
            # that cannot find the session it is asked to carry on says so in one of
            # two ways: an error for an answer, or no start at all.  A process that
            # had made a model request before it failed did find its session: that
            # failure is the phase's, and whoever runs the phase tries again.
            if (
                not self._resumed
                or self._acted
                or self.ctx.ending()
                or (self._heard and _NO_SESSION not in str(exc))
            ):
                raise
        self.close()
        self._tell(
            f"The {self.role}'s interrupted session could not be carried on; it starts "
            "again with what is on the board."
        )
        self._open(fresh=True)
        self._log("prompt", text=text)
        return self._exchange(text)

    def _converse(self, text: str) -> str:
        if self._conversation is None:
            turn = self._first_exchange(text)
        else:
            self._log("prompt", text=text)
            turn = self._exchange(text)
        self._account(turn)
        self._log(
            "result",
            text=turn.text,
            ended=turn.ended,
            error=turn.error,
            cost_usd=turn.cost_usd,
            session=turn.session_id,
        )
        if self._stopped is not None:
            self.close()
            raise self._stopped
        if self.timed_out or self.cancelled:
            self.close()
            return turn.text
        # A stop request, or a budget the turn went past: said the way the engine hears it.
        self.ctx.heed_stop()
        if turn.ended == "error_max_budget_usd":
            # The agent's process counted itself past what was left.  The ledger now
            # holds its figure, so the engine's own check says so with the numbers.
            self.ctx.guard()
            raise agents.SessionStopped("the agent's session reached the model budget")
        if turn.is_error:
            why = experiments.mask_credentials(self.ctx.project, str(turn.error or turn.ended))
            if _NO_SANDBOX in why:
                self.close()
                raise SandboxUnavailable(_no_sandbox_sentence(self.role, _cli_reason(why)))
            raise AgentFailed(f"The {self.role} agent ended with an error: {why}")
        return turn.text

    # -- what it wrote ---------------------------------------------------------------

    def proposed(self, kind: str | None = None) -> list[ResearchPage]:
        return self.backend.proposed(kind)

    def created(self) -> list[str]:
        """Paths of the pages this session created and that are still there."""
        return [page.path for page in self.backend.proposed()]

    def suggested(self) -> list[EditProposal]:
        """The edits this attempt of the session suggested and the person has not decided."""
        return [
            proposal
            for proposal in self.ctx.project.proposals()
            if proposal.author == self.backend.author and proposal.id not in self._known_edits
        ]
