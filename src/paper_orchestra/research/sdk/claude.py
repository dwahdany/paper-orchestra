"""The real driver: agents started through ``claude-agent-sdk``.

The package is an optional extra (``paper-orchestra[agents]``) and is imported
only when an agent is started or the gate asks whether one could be.  The SDK
is asynchronous and the engine is not: each conversation runs its own event
loop on a thread of its own, and hands what the agent does to the calling
thread through a queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import queue
import shutil
import threading
from collections import deque
from pathlib import Path
from typing import Any

from paper_orchestra.research.sdk.driver import (
    BOARD_SERVER,
    AgentOptions,
    Answered,
    BoardTool,
    Said,
    Sink,
    Started,
    Tick,
    ToolUse,
    Turn,
    Used,
    board_name,
)

__all__ = ["INSTALL", "ClaudeDriver"]

#: How the package is installed, said wherever it is found missing.
INSTALL = (
    "install it with `uv sync --extra agents` in a checkout, or "
    "`pip install 'paper-orchestra[agents]'`"
)
#: Seconds between two :class:`Tick` events while the agent reports nothing.
TICK_SECONDS = 0.5
#: Seconds the agent's process is given to start and to end.
START_SECONDS = 120.0
CLOSE_SECONDS = 15.0


def _model_usage(raw: Any) -> dict[str, dict[str, float]]:
    """The per-model totals of a result, under this package's names."""
    names = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "cacheReadInputTokens": "cache_read_input_tokens",
        "cacheCreationInputTokens": "cache_creation_input_tokens",
        "webSearchRequests": "web_searches",
        "costUSD": "cost_usd",
    }
    usage: dict[str, dict[str, float]] = {}
    for model, row in (raw or {}).items():
        if isinstance(row, dict):
            usage[str(model)] = {
                ours: row.get(theirs) or 0
                for theirs, ours in names.items()
                if isinstance(row.get(theirs) or 0, (int, float))
            }
    return usage


def _answer_text(content: Any) -> str:
    """The text of a tool's answer, whatever shape it came in."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
        return "\n".join(text for text in texts if text)
    return "" if content is None else str(content)


class _Conversation:
    """One agent process, driven from a loop on its own thread."""

    def __init__(self, options: AgentOptions) -> None:
        self._options = options
        self._events: queue.Queue[Any] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=20)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="paper-orchestra-agent", daemon=True
        )
        self._thread.start()
        self._client: Any = None
        self._closed = False
        try:
            self._call(self._connect(), START_SECONDS)
        except BaseException:
            self.close()
            raise

    # -- the loop's side -----------------------------------------------------------

    def _call(self, coroutine: Any, timeout: float | None) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)

    def _board(self) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server, tool

        def offered(board_tool: BoardTool) -> Any:
            async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
                # Off the loop: a board tool waits for the project's lock and for services.
                result = await asyncio.to_thread(board_tool.call, dict(arguments or {}))
                return {
                    "content": [{"type": "text", "text": result.text}],
                    "is_error": result.is_error,
                }

            return tool(board_tool.name, board_tool.description, board_tool.input_schema)(handler)

        return create_sdk_mcp_server(
            name=BOARD_SERVER, version="1", tools=[offered(item) for item in self._options.board]
        )

    async def _hook(self, given: Any, tool_use_id: Any, context: Any) -> dict[str, Any]:
        """The ``PreToolUse`` hook: this process's check of every tool call."""
        use = ToolUse(
            str(given.get("tool_name") or ""),
            given.get("tool_input") or {},
            sub=bool(given.get("agent_id")),
        )
        try:
            reason = self._options.check(use)
        except Exception as exc:  # a check that fails refuses: the gate fails closed
            reason = f"The check of this call failed ({type(exc).__name__}), so it was refused."
        if reason is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    async def _connect(self) -> None:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

        options = self._options
        tools = list(options.builtin)
        made = ClaudeAgentOptions(
            model=options.model,
            cwd=str(options.cwd),
            system_prompt=options.system_prompt,
            tools=tools,
            allowed_tools=[*tools, *(board_name(item.name) for item in options.board)],
            mcp_servers={BOARD_SERVER: self._board()},
            # The board is the only server: not the connectors of the person's account,
            # which a signed-in process would bring along.
            strict_mcp_config=True,
            # Nobody sits at this agent to answer a question: what it may do is decided
            # by the sandbox, the deny rules and the hook, not by a prompt.
            permission_mode="bypassPermissions",
            # Not the person's own Claude Code: none of their settings, plugins, hooks
            # or CLAUDE.md is loaded into a research agent.
            setting_sources=[],
            settings=json.dumps(dict(options.settings)),
            env=dict(options.env),
            hooks={"PreToolUse": [HookMatcher(hooks=[self._hook], on_failure="block")]},
            resume=options.resume,
            max_turns=options.max_turns,
            max_budget_usd=options.max_budget_usd,
            stderr=self._stderr.append,
        )
        self._client = ClaudeSDKClient(options=made)
        await self._client.connect()

    async def _turn(self, text: str) -> None:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        put = self._events.put
        requests = 0
        try:
            await self._client.query(text)
            async for message in self._client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    put(Started(dict(message.data)))
                elif isinstance(message, AssistantMessage):
                    sub = bool(message.parent_tool_use_id)
                    requests += 1
                    if message.usage:
                        put(
                            Used(
                                str(message.message_id or message.uuid or requests),
                                str(message.model),
                                dict(message.usage),
                                sub,
                            )
                        )
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            put(ToolUse(block.name, dict(block.input or {}), sub, str(block.id)))
                        elif isinstance(block, TextBlock) and block.text.strip():
                            put(Said(block.text, sub))
                elif isinstance(message, UserMessage) and isinstance(message.content, list):
                    sub = bool(message.parent_tool_use_id)
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            put(
                                Answered(
                                    str(block.tool_use_id),
                                    _answer_text(block.content),
                                    bool(block.is_error),
                                    sub,
                                )
                            )
                elif isinstance(message, ResultMessage):
                    failed = bool(message.is_error)
                    errors = "; ".join(str(error) for error in message.errors or [])
                    put(
                        Turn(
                            text=str(message.result or ""),
                            session_id=str(message.session_id or ""),
                            cost_usd=message.total_cost_usd,
                            model_usage=_model_usage(message.model_usage),
                            is_error=failed,
                            error=(errors or str(message.result or message.subtype))
                            if failed
                            else None,
                            # How it ended when it did not simply complete: the kind of
                            # error (``error_max_budget_usd``, ...), else why the loop
                            # stopped (``aborted_tools`` after an interrupt).
                            ended=message.subtype
                            if message.subtype != "success"
                            else (
                                message.terminal_reason
                                if message.terminal_reason not in (None, "completed")
                                else None
                            ),
                            requests=requests,
                        )
                    )
                    return
            raise RuntimeError("the agent's process ended without a result")
        except Exception as exc:
            put(exc)

    # -- the caller's side -----------------------------------------------------------

    def _failure(self, exc: BaseException) -> RuntimeError:
        said = " ".join(line.strip() for line in self._stderr if line.strip())[-600:]
        return RuntimeError(f"{exc}" + (f" (the agent's process said: {said})" if said else ""))

    def send(self, text: str, sink: Sink) -> Turn:
        if self._closed:
            raise RuntimeError("the conversation is closed")
        asyncio.run_coroutine_threadsafe(self._turn(text), self._loop)
        while True:
            try:
                event = self._events.get(timeout=TICK_SECONDS)
            except queue.Empty:
                sink(Tick())
                continue
            if isinstance(event, Turn):
                return event
            if isinstance(event, BaseException):
                raise self._failure(event) from event
            sink(event)

    def interrupt(self) -> None:
        if self._closed or self._client is None:
            return
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._client.interrupt(), self._loop)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._call(self._client.disconnect(), CLOSE_SECONDS)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(CLOSE_SECONDS)


class ClaudeDriver:
    """Starts Claude Agent SDK agents."""

    def unavailable(self) -> tuple[str, str] | None:
        try:
            spec = importlib.util.find_spec("claude_agent_sdk")
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.origin:
            return ("the Claude Agent SDK (the package claude-agent-sdk) is not installed", INSTALL)
        bundled = Path(spec.origin).parent / "_bundled"
        has_cli = bundled.is_dir() and any(bundled.glob("claude*"))
        if not has_cli and shutil.which("claude") is None:
            return (
                "the Claude Code command line that the agents run in was not found",
                "install Claude Code, or reinstall claude-agent-sdk, which brings its own",
            )
        return None

    def open(self, options: AgentOptions) -> _Conversation:
        missing = self.unavailable()
        if missing is not None:
            from paper_orchestra.llm.base import LLMConfigError

            raise LLMConfigError(f"The agents cannot start: {missing[0]}; {missing[1]}.")
        return _Conversation(options)
