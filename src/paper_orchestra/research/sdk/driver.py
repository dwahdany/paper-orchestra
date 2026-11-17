"""What stands between the research runtime and the Claude Agent SDK.

The runtime (``research.sdk.session``) says what an agent is: its model, its
folder, its system prompt, the board tools it may call, the settings and the
environment its process starts with, and the check every tool call passes
before it is made.  A driver turns that into a running agent and reports what
the agent does as plain events.  There are two: ``research.sdk.claude`` starts
a real agent through ``claude-agent-sdk``, and ``research.sdk.fake`` plays a
script, for tests that need neither the package nor a network.

Everything here is synchronous.  The engine is, and a driver that needs an
event loop keeps it to itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "BOARD_PREFIX",
    "BOARD_SERVER",
    "AgentConversation",
    "AgentDriver",
    "AgentOptions",
    "Answered",
    "BoardTool",
    "Said",
    "Started",
    "Tick",
    "ToolResult",
    "ToolUse",
    "Turn",
    "Used",
    "bare_name",
    "board_name",
]

#: The name of the in-process server the board tools are offered under, and the prefix
#: the agent therefore sees on each of them.
BOARD_SERVER = "board"
BOARD_PREFIX = f"mcp__{BOARD_SERVER}__"


def board_name(name: str) -> str:
    """``propose_edit`` -> ``mcp__board__propose_edit``: a board tool as the agent calls it."""
    return f"{BOARD_PREFIX}{name}"


def bare_name(name: str) -> str:
    """The name of a tool without the board's prefix (a built-in tool's name is itself)."""
    return name.removeprefix(BOARD_PREFIX)


@dataclass(frozen=True)
class ToolResult:
    """What a board tool answered: text, and whether it is an error the agent is told."""

    text: str
    is_error: bool = False


@dataclass(frozen=True)
class BoardTool:
    """One tool of the board, as a driver offers it: ``call`` is synchronous, may block
    (the project's lock, a bibliographic database) and never raises."""

    name: str
    description: str
    input_schema: dict[str, Any]
    call: Callable[[dict[str, Any]], ToolResult] = field(repr=False)


@dataclass(frozen=True)
class ToolUse:
    """A tool call an agent is about to make, or has made.  ``sub`` says that a helper
    the agent started (a sub-agent, a workflow) makes it, not the agent itself."""

    name: str
    input: Mapping[str, Any]
    sub: bool = False
    #: What tells this call from the others of the session, where the driver knows.
    id: str = ""


@dataclass(frozen=True)
class Answered:
    """What a tool call came to, as the agent was told: the text, and whether it was an
    error or a refusal.  ``id`` is the call's (:attr:`ToolUse.id`)."""

    id: str
    text: str
    is_error: bool = False
    sub: bool = False


@dataclass(frozen=True)
class Started:
    """The agent's process is up; ``data`` is what it says about itself (``model``,
    ``fast_mode_state``, ``fast_mode_disabled_reason``, ``tools``, ...)."""

    data: Mapping[str, Any]


@dataclass(frozen=True)
class Said:
    """Text the agent wrote between its tool calls."""

    text: str
    sub: bool = False


@dataclass(frozen=True)
class Used:
    """What one model request used, as the agent's process reports it per message.

    ``usage`` has the provider's names (``input_tokens``, ``output_tokens``,
    ``cache_read_input_tokens``, ``cache_creation_input_tokens``).  The same
    message can be reported more than once; ``message_id`` tells.
    """

    message_id: str
    model: str
    usage: Mapping[str, Any]
    sub: bool = False


@dataclass(frozen=True)
class Tick:
    """Nothing happened for a moment.  A driver sends these while an agent thinks or
    waits for a command, so that whoever listens can ask it to stop."""


@dataclass
class Turn:
    """How one message to the agent ended.

    ``cost_usd`` and ``model_usage`` are what the agent's process says the session
    has cost so far: totals, not what this turn added (a resumed session carries
    on from the totals it had).  ``model_usage`` maps a model to ``input_tokens``,
    ``output_tokens``, ``cache_read_input_tokens``, ``cache_creation_input_tokens``,
    ``web_searches`` and ``cost_usd``.  ``ended`` is why the turn ended when it did
    not simply complete (``aborted_tools`` after an interrupt, ``error_max_turns``,
    ``error_max_budget_usd``, ...).
    """

    text: str = ""
    session_id: str = ""
    cost_usd: float | None = None
    model_usage: dict[str, dict[str, float]] = field(default_factory=dict)
    is_error: bool = False
    error: str | None = None
    ended: str | None = None
    requests: int = 0


@dataclass
class AgentOptions:
    """Everything a driver needs to start one agent."""

    model: str
    #: The agent's own folder: where its shell starts and the only place it may write.
    cwd: Path
    system_prompt: str
    board: Sequence[BoardTool] = ()
    #: The built-in tools the agent has (``Bash``, ``Read``, ``WebSearch``, ...).
    builtin: Sequence[str] = ()
    #: The settings object the agent's process is started with (permissions, sandbox,
    #: fast mode, ultracode).  Nothing else is loaded: not the person's own settings,
    #: plugins, hooks or CLAUDE.md.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: Environment variables set for the agent's process on top of this process's own.
    #: An empty value takes a variable away from it.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Asked before every tool call, also those of helpers: why the call is refused, or
    #: ``None``.  A check that raises refuses the call.
    check: Callable[[ToolUse], str | None] = field(default=lambda use: None, repr=False)
    #: The session to carry on, as an earlier :class:`Turn` named it; ``None`` starts anew.
    resume: str | None = None
    max_turns: int | None = None
    #: The agent's process ends the session itself once it has cost this much.
    max_budget_usd: float | None = None


Sink = Callable[[Any], None]


class AgentConversation(Protocol):
    """One running agent."""

    def send(self, text: str, sink: Sink) -> Turn:
        """Give the agent a message and follow it until it has answered.

        Every event (:class:`Started`, :class:`ToolUse`, :class:`Answered`,
        :class:`Said`, :class:`Used`, :class:`Tick`) is handed to ``sink`` on the calling
        thread.  ``sink`` may call :meth:`interrupt`.
        """

    def interrupt(self) -> None:
        """Ask the agent to stop what it is doing.  Safe from any thread, and twice."""

    def close(self) -> None:
        """End the agent's process, whatever it is doing."""


class AgentDriver(Protocol):
    """Starts agents."""

    def unavailable(self) -> tuple[str, str] | None:
        """Why no agent can be started here and what to do about it (two clauses, as the
        rest of "..., and <why>: <fix>"), or ``None`` when one can.  Contacts nothing."""

    def open(self, options: AgentOptions) -> AgentConversation:
        """Start an agent.  Its first event is :class:`Started`."""
