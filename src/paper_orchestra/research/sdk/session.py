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
import json
import os
import shutil
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
from paper_orchestra.research.sdk.guard import FEDERATION_ENV, Guard, agent_env

__all__ = [
    "AGENTS_DIR",
    "BUILTIN_TOOLS",
    "HELPER_TOOLS",
    "AgentFailed",
    "AgentRuntime",
    "AgentSession",
    "describe_tool",
    "memory_tool",
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
#: What a ledger row of an SDK agent is labelled, after its role: ``scout.agent``.
LABEL_SUFFIX = ".agent"

_USAGE_NAMES = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


class AgentFailed(RuntimeError):
    """The agent's process reported an error instead of an answer."""


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
        signed_in = (home / ".claude" / ".credentials.json").is_file() or bool(
            environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        )
        if not signed_in:
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

        Three things are looked at, and nothing is contacted: the package can be
        imported, there is a command line for the agents to run in, and there is a
        credential of the kind the project chose.
        """
        fault = self.driver.unavailable() or self._credential_fault(
            os.environ if environ is None else environ, home or Path.home()
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
        should_stop: Callable[[], bool] | None = None,
        deadline: float | None = None,
        on_line: Callable[[str], None] | None = None,
        max_turns: int | None = agents.MAX_TURNS,
    ) -> None:
        """``tools`` are the role's own, as for the classic session; ``web_search`` is
        accepted for the same reason and not needed (every SDK agent has the web).

        The rest is for an agent that is not a phase's role (the one that carries out
        an experiment): its ``folder`` and ``guard`` where they are not the usual
        ones, the federation settings its code may use (``job_models``), a further
        reason to stop (``should_stop``), the moment its time is up (``deadline``,
        on ``time.monotonic()``), and ``on_line``, which is given every line the
        feed is given and what the agent says.
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
        self._job_models = dict(job_models or {})
        self._should_stop = should_stop
        self._deadline = deadline
        self._on_line = on_line
        self._max_turns = max_turns
        self._model = str(settings.model)
        self._label = f"{role}{LABEL_SUFFIX}"
        records = project.root / TRANSCRIPTS_DIR / SESSIONS_DIR
        self._record_path = records / f"{self.session_id}.json"
        self._transcript_path = records / f"{self.session_id}.jsonl"
        self._conversation: AgentConversation | None = None
        self._resumed = False
        #: The agent's own session, as its process names it: what a later attempt resumes.
        self._sdk_session = ""
        #: What the agent's session had cost in total when this process last heard of it,
        #: by model: what a result reports beyond that is new.
        self._paid: dict[str, dict[str, float]] = {}
        self._paid_total = 0.0
        self._last = ""
        self._turn = Usage()
        self._seen: set[str] = set()
        self._acted = False
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
        process: dict[str, Any] = dict(self._guard.settings())
        if settings.fast:
            # Fast mode is only on when the settings the process is started with ask
            # for it; what came of it is said in the feed (:meth:`_announce`).
            process["fastMode"] = True
        if helpers:
            process.update(ultracode=True, enableWorkflows=True, skipWorkflowUsageWarning=True)
        left = self.ctx.budget_left()
        return AgentOptions(
            model=self._model,
            cwd=self.folder,
            system_prompt=prompts.system_prompt(
                self.role, index=self.backend.index(), runtime="sdk", helpers=helpers
            ),
            board=self._board,
            builtin=(*BUILTIN_TOOLS, *(HELPER_TOOLS if helpers else ())),
            settings=process,
            env=agent_env(str(settings.credential), job_models=self._job_models),
            check=self._guard.check,
            resume=resume,
            max_turns=self._max_turns,
            # The agent's process counts its own cost, sub-agents and fast mode
            # included, and ends the session when it has spent what is left.  A
            # resumed session counts on from what it had cost before.
            max_budget_usd=None if left is None else max(0.01, left + self._paid_total),
        )

    def _open(self, *, fresh: bool = False) -> AgentConversation:
        if self._conversation is not None:
            return self._conversation
        record = {} if fresh else self._record()
        resume = str(record.get("sdk_session") or "") or None
        self._resumed = resume is not None
        self._sdk_session = resume or ""
        paid = record.get("model_usage") if resume else None
        self._paid = {str(k): dict(v) for k, v in paid.items()} if isinstance(paid, dict) else {}
        self._paid_total = float(record.get("cost_usd") or 0.0) if resume else 0.0
        self._conversation = self.runtime.driver.open(self._options(resume))
        return self._conversation

    def close(self) -> None:
        """End the agent's process.  What it wrote on the board stays; its session can be
        carried on by a later attempt at the phase."""
        conversation, self._conversation = self._conversation, None
        if conversation is not None:
            with contextlib.suppress(Exception):
                conversation.close()

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

    def _tell(self, sentence: str, refs: Sequence[str] = ()) -> None:
        """One line for the activity feed (and the run's log), unless it repeats the last."""
        if sentence == self._last:
            return
        self._last = sentence
        self.ctx.activity(sentence, agent=self.role, refs=refs)
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
        model = agents.named(data.get("model") or self._model, 60)
        state = str(data.get("fast_mode_state") or "off")
        said = f"The {self.role} works as a Claude agent on {model}"
        if not self.runtime.settings.fast:
            said += "."
        elif state == "on":
            said += ", in fast mode."
        else:
            reason = data.get("fast_mode_disabled_reason")
            why = f" ({agents.said(reason, 80)})" if reason else ""
            said += f"; fast mode was asked for and is not on{why}."
        if self._resumed:
            said += " It carries on the session that was interrupted."
        self._tell(said)
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
            self._log("tool", id=event.id, name=event.name, input=event.input, sub=event.sub)
            if not self._ending():
                described = self._described(event)
                if described is not None:
                    self._tell(described[0], described[1])
        elif isinstance(event, Answered):
            # Kept for whoever reads the transcript; long answers are cut.
            self._log(
                "answer",
                id=event.id,
                text=event.text[:4000],
                is_error=event.is_error,
                sub=event.sub,
            )
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

    def _described(self, use: ToolUse) -> tuple[str, list[str]] | None:
        """The feed's sentence about a tool call the agent announced.  A call the guard
        refuses is told as refused: the person should see what was tried, and that it
        did not happen."""
        try:
            refused = self._guard.check(use)
        except Exception:
            refused = "refused"
        if refused is None:
            return describe_tool(self.role, use, self.folder)
        who = f"A helper of the {self.role}" if use.sub else f"The {self.role}"
        given = use.input if isinstance(use.input, Mapping) else {}
        tried = given.get("command") or given.get("file_path") or given.get("path") or use.name
        return (
            f"{who} tried {agents.said(tried, 140)} with {agents.named(use.name, 40)}; "
            "that is not an agent's to do, and it was refused.",
            [],
        )

    def _count(self, event: Used) -> None:
        """Add one model request to what the turn in flight has used, and let the budget
        see it.  This is an estimate from tokens, for ending a long turn in time; what
        goes into the ledger is the agent's own figure, when the turn has ended."""
        if event.message_id in self._seen:
            return
        self._seen.add(event.message_id)
        usage = event.usage
        self._turn.add(
            Usage(
                **{name: agents._whole(usage.get(name)) for name in _USAGE_NAMES},
                requests=1,
            )
        )
        self.ctx.used(self._model, self._turn)

    def _account(self, turn: Turn) -> None:
        """Put what the agent's session cost since it was last heard of into the ledger.

        The agent's process reports totals for its session, per model, and a resumed
        session counts on from where it stood.  What is new since the last report
        is recorded once, with the dollar figure the process gave.
        """
        ledger = self.ctx.harness.usage
        seen_models = False
        for model, row in turn.model_usage.items():
            before = self._paid.get(model, {})
            new = {
                name: max(0.0, float(row.get(name) or 0) - float(before.get(name) or 0))
                for name in (*_USAGE_NAMES, "web_searches", "cost_usd")
            }
            # Never below what is in the ledger already (an estimate for a turn that was
            # killed can be above what the process then reports for it).
            self._paid[model] = {
                name: max(float(row.get(name) or 0), float(before.get(name) or 0))
                for name in (*_USAGE_NAMES, "web_searches", "cost_usd")
            }
            seen_models = True
            if not any(new.values()):
                continue
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
        if seen_models:
            self._paid_total = sum(float(row.get("cost_usd") or 0.0) for row in self._paid.values())
        elif turn.cost_usd is not None:
            added = float(turn.cost_usd) - self._paid_total
            if added > 0:
                ledger.record(
                    self._label, self._model, Usage(requests=turn.requests), cost_usd=added
                )
                self._paid_total = float(turn.cost_usd)
        # What the ledger has of this turn is the process's figure now, not an estimate.
        self._turn = Usage()
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
        except BaseException:
            # A failure of the agent's process, or Ctrl-C in this one: the process is
            # ended; what the agent wrote on the board stays.
            self.close()
            self._account_unreported()
            raise

    def _account_unreported(self) -> None:
        """Count a turn whose process ended without saying what it cost (it was killed,
        or it failed): by the tokens its messages reported, at the price per token.

        The estimate goes where the process's own figures go, and is added to what
        the session counts as reported.  When the session is carried on and its
        process then reports totals that include this turn, only what goes beyond
        the estimate is new: nothing is counted twice.
        """
        if not any(self._turn.to_dict().values()):
            return
        estimate = estimate_cost_usd(self._model, self._turn) or 0.0
        self.ctx.harness.usage.record(self._label, self._model, self._turn, cost_usd=estimate)
        row = self._paid.setdefault(self._model, {})
        for name in _USAGE_NAMES:
            row[name] = float(row.get(name) or 0) + getattr(self._turn, name)
        row["cost_usd"] = float(row.get("cost_usd") or 0.0) + estimate
        self._paid_total += estimate
        self._turn = Usage()
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
                    "cost_usd": self._paid_total,
                    "model_usage": self._paid,
                    "transcript": self._transcript_path.name,
                },
            )

    def send(self, parts: Part | Sequence[Part] | str) -> str:
        self.ctx.guard()
        text = self._as_text(parts)
        self._turn = Usage()
        self._seen = set()
        self._asked_at = None
        self._acted = False
        try:
            return self._converse(text)
        finally:
            # Also when the turn was cut short: what it wrote by then is in the wiki.
            self.ctx.note(self.created())

    def _converse(self, text: str) -> str:
        first = self._conversation is None
        self._open()
        if first and self._resumed:
            text = (
                "Your session was interrupted and is carried on now. What you had put on "
                "the board is still there: look at the wiki before you write anything "
                "again. The task, as it was given:\n\n" + text
            )
        self._log("prompt", text=text)
        turn = self._exchange(text)
        if first and self._resumed and turn.is_error and not self._acted:
            # The agent's own session file is gone or cannot be read: start again, with
            # what is on the board.
            self.close()
            self._tell(
                f"The {self.role}'s interrupted session could not be carried on; it starts "
                "again with what is on the board."
            )
            self._open(fresh=True)
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
