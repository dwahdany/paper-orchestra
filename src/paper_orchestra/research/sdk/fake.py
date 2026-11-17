"""A scripted agent, for tests: no package, no process, no network.

``FakeDriver(script)`` plays ``script`` once for every message an agent is
sent.  The script gets a :class:`FakeAgent` and acts through it the way a real
agent acts through its tools: it calls board tools, runs shell commands, writes
files, searches the web.  Every one of those calls passes the same check
(``AgentOptions.check``) and the same board-tool wrapper as a real agent's, so a
test of a guardrail or of a rule of the board tests the code that guards a real
agent.  What the fake does not have is the sandbox of the shell: a command is
not executed unless the script says what it does.

It behaves like the real thing where the runtime depends on it: an interrupt
ends the turn at the agent's next action, the cost it reports is a total that
a resumed session carries on from, and the first event of a session is what
the process says about itself.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from paper_orchestra.research.sdk.driver import (
    AgentOptions,
    Said,
    Sink,
    Started,
    Tick,
    ToolResult,
    ToolUse,
    Turn,
    Used,
    bare_name,
    board_name,
)

__all__ = ["Denied", "FakeAgent", "FakeDriver"]


class _Interrupted(Exception):
    """The agent was asked to stop: the script ends where it stands."""


class Denied(str):
    """What a refused tool call answered: the reason, as the agent is told it."""


class FakeAgent:
    """What a script acts through, for one message."""

    def __init__(self, conversation: _FakeConversation, prompt: str, sink: Sink) -> None:
        self._conversation = conversation
        self._sink = sink
        #: The message the agent was sent, and how many it was sent before in this session.
        self.prompt = prompt
        self.turn = conversation.turns
        self.options = conversation.options
        self.cwd = Path(conversation.options.cwd)
        #: The calls that were refused, as ``(tool name, reason)``.
        self.denied: list[tuple[str, str]] = []
        self.requests = 0

    # -- what every action goes through ----------------------------------------

    def _alive(self) -> None:
        if self._conversation.interrupted:
            raise _Interrupted()

    def _refused(self, use: ToolUse) -> Denied | None:
        """Ask the check the way a real agent's process asks its hook."""
        self._alive()
        try:
            reason = self.options.check(use)
        except Exception as exc:  # a hook that fails refuses the call
            reason = f"The check of this call failed ({type(exc).__name__}), so it was refused."
        self._sink(use)
        if reason is None:
            return None
        self.denied.append((use.name, reason))
        return Denied(reason)

    def tool(self, name: str, tool_input: Mapping[str, Any], *, sub: bool = False) -> str | None:
        """Announce a call of the built-in tool ``name``; the refusal, or ``None``."""
        return self._refused(ToolUse(name, dict(tool_input), sub))

    # -- the board -----------------------------------------------------------------

    def board(self, name: str, tool_input: Mapping[str, Any] | None = None, **given: Any) -> str:
        """Call the board tool ``name``; the text it answered.  :attr:`failed` then says
        whether that was an error."""
        tool_input = {**(tool_input or {}), **given}
        refused = self._refused(ToolUse(board_name(name), tool_input))
        if refused is not None:
            self.failed = True
            return refused
        offered = {tool.name: tool for tool in self.options.board}
        if name not in offered:
            raise KeyError(f"board tool '{name}' was not offered (have: {sorted(offered)})")
        result: ToolResult = offered[name].call(dict(tool_input))
        self.failed = result.is_error
        self._alive()
        return result.text

    failed = False

    def memory(self, command: str, **given: Any) -> str:
        return self.board("memory", command=command, **given)

    # -- built-in tools --------------------------------------------------------------

    def bash(
        self, command: str, does: Callable[[], Any] | None = None, *, sub: bool = False
    ) -> str:
        """Run ``command`` in the agent's shell.  Nothing is executed: ``does`` is what the
        command would do, called when the call was let through."""
        refused = self._refused(ToolUse("Bash", {"command": command}, sub))
        if refused is not None:
            return refused
        return str(does() or "") if does is not None else ""

    def _path(self, path: str | os.PathLike[str]) -> Path:
        return Path(path) if os.path.isabs(path) else self.cwd / path

    def write(self, path: str | os.PathLike[str], text: str, *, tool: str = "Write") -> str:
        """Write a file with the built-in tool; the refusal, or an empty text."""
        refused = self._refused(ToolUse(tool, {"file_path": str(path), "content": text}))
        if refused is not None:
            return refused
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return ""

    def read(self, path: str | os.PathLike[str]) -> str:
        """Read a file with the built-in tool; its text, or the refusal."""
        refused = self._refused(ToolUse("Read", {"file_path": str(path)}))
        if refused is not None:
            return refused
        return self._path(path).read_text(encoding="utf-8")

    def web_search(self, query: str) -> str | None:
        return self.tool("WebSearch", {"query": query})

    def fetch(self, url: str) -> str | None:
        return self.tool("WebFetch", {"url": url, "prompt": "Summarise the page."})

    # -- what the process reports ----------------------------------------------------

    def say(self, text: str) -> None:
        self._alive()
        self._sink(Said(text))

    def use(
        self,
        *,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str | None = None,
        sub: bool = False,
    ) -> None:
        """One model request: what it used, and what the process says it cost."""
        self._alive()
        self.requests += 1
        model = model or self.options.model
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self._conversation.add(model, usage, cost_usd)
        self._sink(Used(f"msg-{self._conversation.driver.messages}", model, usage, sub))
        self._conversation.driver.messages += 1

    def tick(self, times: int = 1) -> None:
        """Time passes with nothing to report (a long command, a long thought)."""
        for _ in range(times):
            self._alive()
            self._sink(Tick())


Script = Callable[[FakeAgent], Any]


class _FakeConversation:
    def __init__(self, driver: FakeDriver, options: AgentOptions) -> None:
        self.driver = driver
        self.options = options
        self.turns = 0
        self.interrupted = False
        self.closed = False
        self._announced = False
        resumed = driver.sessions.get(options.resume or "")
        # A resumed session carries on from the totals it had, as the real one does.
        self.session_id = options.resume if resumed else f"fake-{len(driver.sessions) + 1}"
        self._totals: dict[str, dict[str, float]] = (
            {model: dict(row) for model, row in resumed.items()} if resumed else {}
        )
        driver.sessions[str(self.session_id)] = self._totals

    def add(self, model: str, usage: Mapping[str, Any], cost_usd: float) -> None:
        row = self._totals.setdefault(model, {})
        for name, value in usage.items():
            row[name] = row.get(name, 0) + value
        row["cost_usd"] = row.get("cost_usd", 0.0) + cost_usd

    def send(self, text: str, sink: Sink) -> Turn:
        if self.closed:
            raise RuntimeError("the conversation is closed")
        driver = self.driver
        driver.prompts.append(text)
        self.interrupted = False
        if not self._announced:
            self._announced = True
            sink(Started({"model": self.options.model, **driver.started}))
        if self.options.resume and self.options.resume in driver.unresumable:
            return Turn(
                session_id=str(self.session_id),
                is_error=True,
                error=f"No conversation found with session ID: {self.options.resume}",
                ended="error_during_execution",
            )
        agent = FakeAgent(self, text, sink)
        ended, said = None, ""
        try:
            answer = driver.script(agent)
            said = "" if answer is None else str(answer)
        except _Interrupted:
            ended = "aborted_tools"
        finally:
            self.turns += 1
        cost = sum(row.get("cost_usd", 0.0) for row in self._totals.values())
        return Turn(
            text=said,
            session_id=str(self.session_id),
            cost_usd=cost,
            model_usage={model: dict(row) for model, row in self._totals.items()},
            ended=ended,
            requests=agent.requests,
        )

    def interrupt(self) -> None:
        self.interrupted = True
        self.driver.interrupts += 1

    def close(self) -> None:
        self.closed = True


class FakeDriver:
    """Starts scripted agents; remembers what it was asked for."""

    def __init__(self, script: Script | None = None, *, started: Mapping[str, Any] | None = None):
        self.script: Script = script or (lambda agent: "Done.")
        #: What a session says about itself at its start, besides its model.
        self.started: dict[str, Any] = {"fast_mode_state": "on", **(started or {})}
        #: Why no agent can be started (``(why, fix)``), for tests of the gate.
        self.missing: tuple[str, str] | None = None
        self.opened: list[AgentOptions] = []
        self.conversations: list[_FakeConversation] = []
        self.prompts: list[str] = []
        self.interrupts = 0
        self.messages = 0
        #: session id -> the totals that session has reported.
        self.sessions: dict[str, dict[str, dict[str, float]]] = {}
        #: Sessions that can no longer be carried on (their file is gone).
        self.unresumable: set[str] = set()

    def unavailable(self) -> tuple[str, str] | None:
        return self.missing

    def open(self, options: AgentOptions) -> _FakeConversation:
        self.opened.append(options)
        conversation = _FakeConversation(self, options)
        self.conversations.append(conversation)
        return conversation

    def calls(self) -> list[str]:
        """The bare names of the tools every opened agent was offered on the board."""
        return [bare_name(tool.name) for options in self.opened for tool in options.board]
