"""The Anthropic client against the real SDK request/response path, offline.

A mock ``httpx2`` transport plays a scripted Messages API: every test sends
real SDK requests, inspects the JSON the SDK put on the wire and feeds back
server-sent events the SDK parses into messages.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import socket
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from paper_orchestra.llm import anthropic_client as ac
from paper_orchestra.llm.anthropic_client import (
    AnthropicClient,
    bridge_identity_token,
    describe_credentials,
    fit_image,
    json_output_format,
    normalize_fallback_turn,
    run_memory_command,
)
from paper_orchestra.llm.base import (
    Image,
    LLMConfigError,
    LLMError,
    LLMParseError,
    LLMRefusal,
    LLMTruncated,
    Pdf,
    ToolError,
    ToolSpec,
    WebSearch,
)
from paper_orchestra.llm.parsing import parse_json, parse_latex

MODEL = "claude-fable-5-2"
SSE_HEADERS = {"content-type": "text/event-stream"}
WIF_IDS = {"ANTHROPIC_FEDERATION_RULE_ID": "fdrl_1", "ANTHROPIC_ORGANIZATION_ID": "org-1"}


def _refuse_connection(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("a test tried to open a network connection")


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch, tmp_path):
    """No network, no profile on disk, no leftover switches; env edits die with the test."""
    # Anything that slips past the mock transport must fail, not reach the API.
    monkeypatch.setattr(socket.socket, "connect", _refuse_connection)
    # The client writes the bridged identity token into os.environ; a private
    # copy keeps that (and everything else) from leaking between tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in (
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_CUSTOM_HEADERS",
        "MODAL_IDENTITY_TOKEN_FILE",
        ac.STRUCTURED_OUTPUTS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ac, "STREAM_RESTART_DELAY_S", 0.0)


# ---------------------------------------------------------------------------
# A scripted Messages API
# ---------------------------------------------------------------------------


def msg(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    *,
    model: str = MODEL,
    usage: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }
    if stop_details is not None:
        message["stop_details"] = stop_details
    return message


def sse(message: dict[str, Any], *, first_events: int | None = None) -> bytes:
    """Encode a complete message as the event stream the API would send."""
    usage = message["usage"]
    start = {
        **message,
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
    }
    start.pop("stop_details", None)
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start}]
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        if kind == "text":
            opening = {**block, "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif kind == "thinking":
            opening = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif kind in ("tool_use", "server_tool_use"):
            opening = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        else:
            opening, deltas = block, []
        events.append({"type": "content_block_start", "index": index, "content_block": opening})
        events.extend({"type": "content_block_delta", "index": index, "delta": d} for d in deltas)
        events.append({"type": "content_block_stop", "index": index})
    delta: dict[str, Any] = {"stop_reason": message["stop_reason"], "stop_sequence": None}
    if "stop_details" in message:
        delta["stop_details"] = message["stop_details"]
    events.append({"type": "message_delta", "delta": delta, "usage": usage})
    events.append({"type": "message_stop"})
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events[:first_events]
    ).encode()


def api_error(status: int, kind: str, message: str, request_id: str = "req_err") -> httpx2.Response:
    body = {"type": "error", "error": {"type": kind, "message": message}, "request_id": request_id}
    return httpx2.Response(
        status, json=body, headers={"request-id": request_id, "retry-after-ms": "1"}
    )


class FakeAPI:
    """Mock transport handler: records request bodies, replies from a script."""

    def __init__(self) -> None:
        self.script: list[dict[str, Any] | Callable[[httpx2.Request], httpx2.Response]] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        self.paths.append(request.url.path)
        if not self.script:
            return api_error(500, "api_error", "the test script has no reply left")
        reply = self.script.pop(0)
        if callable(reply):
            return reply(request)
        headers = {**SSE_HEADERS, "request-id": f"req_{len(self.requests)}"}
        return httpx2.Response(200, content=sse(reply), headers=headers)

    def http_client(self) -> Any:
        return anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self))


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
def client(api, monkeypatch) -> AnthropicClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-test-key")
    return AnthropicClient(http_client=api.http_client(), max_retries=0)


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def thinking(signature: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": "", "signature": signature}


def tool_use(use_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": use_id, "name": name, "input": tool_input}


class DictMemory:
    """A minimal in-memory :class:`MemoryBackend`."""

    def __init__(self) -> None:
        self.files = {"/memories/MEMORY.md": "index\nsecond line\nthird line"}
        self.calls: list[tuple[Any, ...]] = []

    def _existing(self, path: str) -> str:
        if path not in self.files:
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
        return self.files[path]

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str:
        self.calls.append(("view", path, view_range))
        if path == "/memories":
            return "\n".join(sorted(self.files))
        lines = self._existing(path).split("\n")
        if view_range is not None:
            lines = lines[view_range[0] - 1 : None if view_range[1] == -1 else view_range[1]]
        return "\n".join(lines)

    def create(self, path: str, file_text: str) -> str:
        self.calls.append(("create", path, file_text))
        self.files[path] = file_text
        return f"File created successfully at: {path}"

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        self.calls.append(("str_replace", path, old_str, new_str))
        self.files[path] = self._existing(path).replace(old_str, new_str)
        return "The memory file has been edited."

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        self.calls.append(("insert", path, insert_line, insert_text))
        lines = self._existing(path).split("\n")
        lines.insert(insert_line, insert_text)
        self.files[path] = "\n".join(lines)
        return f"The file {path} has been edited."

    def delete(self, path: str) -> str:
        self.calls.append(("delete", path))
        del self.files[path]
        return f"Successfully deleted {path}"

    def rename(self, old_path: str, new_path: str) -> str:
        self.calls.append(("rename", old_path, new_path))
        self.files[new_path] = self.files.pop(old_path)
        return f"Successfully renamed {old_path} to {new_path}"


def png(width: int, height: int, color: str = "navy") -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def image_size(block: dict[str, Any]) -> tuple[int, int]:
    return PILImage.open(io.BytesIO(base64.b64decode(block["source"]["data"]))).size


class Outline(BaseModel):
    title: str
    sections: list[str]


# ---------------------------------------------------------------------------
# Plain text turns and the request shape
# ---------------------------------------------------------------------------


def test_text_turn_puts_the_recommended_request_on_the_wire(api, client):
    api.script.append(msg([thinking("S1"), text("Hello there.")]))
    conversation = client.conversation(
        model=MODEL, system="You write papers.", label="outline.writer", temperature=0.3
    )
    response = conversation.send("Say hello.")

    assert response.text == "Hello there."
    assert response.stop_reason == "end_turn"
    assert response.model == MODEL
    assert response.request_id == "req_1"
    assert response.tool_calls == 0
    assert response.usage.to_dict() == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "requests": 1,
        "web_searches": 0,
    }
    # Plain JSON only: no SDK object leaves the module.
    assert json.loads(json.dumps(response.raw))["content"][1] == text("Hello there.")

    body = api.requests[0]
    assert api.paths == ["/v1/messages"]
    assert body["model"] == MODEL
    assert body["stream"] is True
    assert body["max_tokens"] == 64000
    assert body["output_config"] == {"effort": "high"}
    assert body["fallbacks"] == "default"
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["system"] == [
        {"type": "text", "text": "You write papers.", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": [text("Say hello.")]}]
    # Everything Fable 5.2 rejects, or that would make history editable, stays out.
    for forbidden in ("thinking", "temperature", "top_p", "top_k", "tool_choice", "tools"):
        assert forbidden not in body
    assert "server-side-fallback-2026-07-01" in api.headers[0]["anthropic-beta"]


def test_effort_and_max_tokens_options(api, client):
    api.script += [msg([text("a")]), msg([text("b")])]
    conversation = client.conversation(model=MODEL, effort="low", max_tokens=500_000)
    conversation.send("one")
    conversation.send("two", max_tokens=1234)
    assert api.requests[0]["output_config"] == {"effort": "low"}
    assert api.requests[0]["max_tokens"] == 128_000  # capped at the model maximum
    assert api.requests[1]["max_tokens"] == 1234
    assert "system" not in api.requests[0]
    with pytest.raises(LLMConfigError, match="max_tokens"):
        client.conversation(model=MODEL, max_tokens=0)
    with pytest.raises(LLMError, match="empty message"):
        conversation.send(["", "  "])


def test_second_turn_extends_the_journal_and_usage_accumulates(api, client):
    first = [thinking("S1"), text("First answer.")]
    api.script += [
        msg(
            first, usage={"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 7}
        ),
        msg(
            [text("Second answer.")],
            usage={"input_tokens": 3, "output_tokens": 2, "cache_read_input_tokens": 17},
        ),
    ]
    conversation = client.conversation(model=MODEL, system="S")
    conversation.send("one")
    second = conversation.send("two")

    assert api.requests[1]["messages"] == [
        {"role": "user", "content": [text("one")]},
        {"role": "assistant", "content": first},
        {"role": "user", "content": [text("two")]},
    ]
    assert second.usage.requests == 1 and second.usage.cache_read_input_tokens == 17
    assert conversation.usage.to_dict() == {
        "input_tokens": 13,
        "output_tokens": 7,
        "cache_read_input_tokens": 17,
        "cache_creation_input_tokens": 7,
        "requests": 2,
        "web_searches": 0,
    }
    journal = conversation.journal
    assert [turn["role"] for turn in journal] == ["user", "assistant", "user", "assistant"]
    journal.clear()  # a copy: the conversation's own journal is unaffected
    assert len(conversation.journal) == 4


# ---------------------------------------------------------------------------
# The tool loop
# ---------------------------------------------------------------------------

LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {
        "q": {"type": "string", "description": "What to look up."},
        "k": {"type": "integer", "minimum": 1},
    },
    "required": ["q"],
    "additionalProperties": False,
}


def test_tool_loop_is_append_only_and_answers_every_call_in_one_message(api, client):
    memory = DictMemory()
    seen: list[dict[str, Any]] = []

    def lookup(tool_input: dict[str, Any]) -> str:
        seen.append(dict(tool_input))
        if tool_input["q"] == "missing":
            raise ToolError("No page matches 'missing'. Try a broader query.")
        if tool_input["q"] == "crash":
            raise RuntimeError("index is corrupt")
        tool_input["q"] = "mutated by the handler"
        return "page text"

    first = [
        thinking("S1"),
        text("Checking what I know."),
        tool_use("tu_1", "memory", {"command": "view", "path": "/memories"}),
        tool_use("tu_2", "lookup", {"q": "transformers", "k": 3}),
        tool_use("tu_3", "lookup", {"q": "missing"}),
        tool_use("tu_4", "lookup", {"q": "crash"}),
        tool_use("tu_5", "lookup", {"k": 2}),
        tool_use("tu_6", "lookup", {"q": 7}),
        tool_use("tu_7", "search_the_web", {"query": "x"}),
        tool_use(
            "tu_8",
            "memory",
            {"command": "str_replace", "path": "/memories/none.md", "old_str": "a"},
        ),
    ]
    second = [
        thinking("S2"),
        tool_use(
            "tu_9",
            "memory",
            {"command": "create", "path": "/memories/notes/n1.md", "file_text": "learned"},
        ),
    ]
    api.script += [msg(first, "tool_use"), msg(second, "tool_use"), msg([text("Final answer.")])]
    events: list[dict[str, Any]] = []
    conversation = client.conversation(
        model=MODEL,
        system="You write papers.",
        tools=[ToolSpec("lookup", "Look a term up in the wiki.", LOOKUP_SCHEMA, lookup)],
        memory=memory,
        label="writer",
        on_event=events.append,
    )
    response = conversation.send("Draft the related-work section.")

    assert response.text == "Final answer."
    assert response.tool_calls == 9
    assert response.usage.requests == 3 and conversation.usage.requests == 3
    assert memory.files["/memories/notes/n1.md"] == "learned"
    assert seen == [{"q": "transformers", "k": 3}, {"q": "missing"}, {"q": "crash"}]

    # Append-only: each request extends the previous one; system and tools never change.
    one, two, three = api.requests
    assert two["messages"][: len(one["messages"])] == one["messages"]
    assert three["messages"][: len(two["messages"])] == two["messages"]
    assert json.dumps(two["system"]) == json.dumps(one["system"]) == json.dumps(three["system"])
    assert json.dumps(two["tools"]) == json.dumps(one["tools"]) == json.dumps(three["tools"])

    # The assistant turn goes back exactly as received, even though a handler
    # modified the input dict it was given.
    assert two["messages"][1] == {"role": "assistant", "content": first}
    results = two["messages"][2]
    assert results["role"] == "user"
    assert [block["type"] for block in results["content"]] == ["tool_result"] * 8
    assert [block["tool_use_id"] for block in results["content"]] == [
        f"tu_{n}" for n in range(1, 9)
    ]
    by_id = {block["tool_use_id"]: block for block in results["content"]}
    assert "/memories/MEMORY.md" in by_id["tu_1"]["content"] and "is_error" not in by_id["tu_1"]
    assert by_id["tu_2"] == {"type": "tool_result", "tool_use_id": "tu_2", "content": "page text"}
    assert by_id["tu_3"]["is_error"] is True and "broader query" in by_id["tu_3"]["content"]
    assert by_id["tu_4"]["is_error"] is True
    assert by_id["tu_4"]["content"] == "RuntimeError: index is corrupt"
    assert (
        by_id["tu_5"]["is_error"] is True and "required parameter(s): q" in by_id["tu_5"]["content"]
    )
    assert (
        by_id["tu_6"]["is_error"] is True and "must be of type string" in by_id["tu_6"]["content"]
    )
    assert by_id["tu_7"]["is_error"] is True and "Unknown tool" in by_id["tu_7"]["content"]
    assert by_id["tu_8"]["is_error"] is True and "does not exist" in by_id["tu_8"]["content"]
    assert three["messages"][3] == {"role": "assistant", "content": second}
    assert three["messages"][4]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tu_9",
            "content": "File created successfully at: /memories/notes/n1.md",
        }
    ]

    kinds = [event["type"] for event in events]
    assert kinds[:3] == ["request", "assistant", "usage"]
    assert kinds.count("tool_call") == kinds.count("tool_result") == 9
    assert kinds[-2:] == ["assistant", "usage"]
    assert events[0] == {
        "type": "request",
        "label": "writer",
        "model": MODEL,
        "user": "Draft the related-work section.",
    }
    assert events[1]["text"] == "Checking what I know." and events[1]["stop_reason"] == "tool_use"
    call = next(event for event in events if event["type"] == "tool_call")
    assert call == {
        "type": "tool_call",
        "name": "memory",
        "input": {"command": "view", "path": "/memories"},
    }
    errors = [event["is_error"] for event in events if event["type"] == "tool_result"]
    assert errors == [False, False, True, True, True, True, True, True, False]
    json.dumps(events)  # the transcript layer stores events as JSON


def test_tool_declarations(api, client, monkeypatch):
    open_schema = {"type": "object", "properties": {"note": {"type": "string"}}}
    tools = [
        ToolSpec("lookup", "Look a term up.", LOOKUP_SCHEMA, lambda _: "x"),
        ToolSpec("save_note", "Save a note.", open_schema, lambda _: "x"),
    ]
    search = WebSearch(max_uses=3, allowed_domains=("arxiv.org", "openreview.net"))
    api.script.append(msg([text("ok")]))
    client.conversation(model=MODEL, tools=tools, memory=DictMemory(), web_search=search).send("go")

    declared = api.requests[0]["tools"]
    assert declared[0] == {"type": "memory_20250818", "name": "memory"}
    # A closed schema becomes a strict tool; constraints strict mode cannot
    # express move into the description, as the SDK's own transform does.
    assert declared[1] == {
        "name": "lookup",
        "description": "Look a term up.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "What to look up."},
                "k": {"type": "integer", "description": "{minimum: 1}"},
            },
            "required": ["q"],
            "additionalProperties": False,
        },
    }
    assert declared[2] == {
        "name": "save_note",
        "description": "Save a note.",
        "input_schema": open_schema,
    }
    assert declared[3] == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 3,
        "allowed_domains": ["arxiv.org", "openreview.net"],
    }
    assert "tool_choice" not in api.requests[0]

    # Strict tools are a structured-output feature and go away with the switch.
    monkeypatch.setenv(ac.STRUCTURED_OUTPUTS_ENV, "0")
    plain = AnthropicClient(http_client=api.http_client(), max_retries=0)
    api.script.append(msg([text("ok")]))
    plain.conversation(model=MODEL, tools=tools[:1]).send("go")
    assert api.requests[1]["tools"] == [
        {"name": "lookup", "description": "Look a term up.", "input_schema": LOOKUP_SCHEMA}
    ]


def test_strict_tools_are_dropped_when_the_first_request_is_rejected(api, client, caplog):
    def rejected(_: httpx2.Request) -> httpx2.Response:
        return api_error(400, "invalid_request_error", "tools.0.strict: not supported", "req_s")

    tools = [ToolSpec("lookup", "Look a term up.", LOOKUP_SCHEMA, lambda _: "page text")]
    plain = [{"name": "lookup", "description": "Look a term up.", "input_schema": LOOKUP_SCHEMA}]
    api.script += [
        rejected,
        msg([tool_use("tu_1", "lookup", {"q": "x"})], "tool_use"),
        msg([text("done")]),
    ]
    conversation = client.conversation(model=MODEL, tools=tools)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.anthropic_client"):
        assert conversation.send("go").text == "done"
    assert api.requests[0]["tools"][0]["strict"] is True
    assert api.requests[1]["tools"] == api.requests[2]["tools"] == plain
    assert api.requests[1]["messages"] == api.requests[0]["messages"]
    assert any("without strict tools" in record.getMessage() for record in caplog.records)

    # Only the first request of a conversation may change the tools; a later 400 is final.
    api.script.append(rejected)
    with pytest.raises(LLMError, match="req_s"):
        conversation.send("again")
    assert len(api.requests) == 4

    # Still rejected without strict tools: the tools were not the problem.
    api.script += [rejected, rejected]
    with pytest.raises(LLMConfigError, match="native structured output"):
        client.conversation(model=MODEL, tools=tools).send("go", schema=Outline)
    assert len(api.requests) == 6 and api.requests[5]["tools"] == plain

    # Without strict tools there is nothing to retry.
    api.script.append(rejected)
    open_tool = ToolSpec("note", "d", {"type": "object", "properties": {}}, lambda _: "x")
    with pytest.raises(LLMError, match="req_s"):
        client.conversation(model=MODEL, tools=[open_tool]).send("go")
    assert len(api.requests) == 7


def test_invalid_tool_declarations_are_config_errors(client):
    def tool(name: str, schema: Any = None) -> ToolSpec:
        return ToolSpec(name, "d", schema or {"type": "object", "properties": {}}, lambda _: "x")

    with pytest.raises(LLMConfigError, match="reserved"):
        client.conversation(model=MODEL, tools=[tool("memory")])
    with pytest.raises(LLMConfigError, match="reserved"):
        client.conversation(model=MODEL, tools=[tool("web_search")])
    with pytest.raises(LLMConfigError, match="unique"):
        client.conversation(model=MODEL, tools=[tool("a"), tool("a")])
    with pytest.raises(LLMConfigError, match="Invalid tool name"):
        client.conversation(model=MODEL, tools=[tool("has space")])
    with pytest.raises(LLMConfigError, match="type 'object'"):
        client.conversation(model=MODEL, tools=[tool("a", {"type": "string"})])
    with pytest.raises(LLMConfigError, match="not JSON-serialisable"):
        client.conversation(model=MODEL, tools=[tool("a", {"type": "object", "x": object()})])
    with pytest.raises(LLMConfigError, match="not both"):
        search = WebSearch(allowed_domains=("a.org",), blocked_domains=("b.org",))
        client.conversation(model=MODEL, web_search=search)
    with pytest.raises(LLMConfigError, match="max_turns"):
        client.conversation(model=MODEL, max_turns=0)


def test_memory_commands_are_validated_and_dispatched():
    memory = DictMemory()

    def run(**tool_input: Any) -> str:
        return run_memory_command(memory, tool_input)

    assert "second line" in run(command="view", path="/memories/MEMORY.md")
    assert run(command="view", path="/memories/MEMORY.md", view_range=[2, 2]) == "second line"
    assert run(command="view", path="/memories/MEMORY.md", view_range=[2, -1]).endswith(
        "third line"
    )
    run(command="create", path="/memories/a.md", file_text="alpha beta")
    run(command="create", path="/memories/empty.md", file_text="")
    run(command="str_replace", path="/memories/a.md", old_str="alpha", new_str="gamma")
    assert memory.files["/memories/a.md"] == "gamma beta"
    # new_str is optional in the documented contract: leaving it out deletes old_str.
    run(command="str_replace", path="/memories/a.md", old_str=" beta")
    run(command="str_replace", path="/memories/a.md", old_str="gam", new_str=None)
    assert memory.files["/memories/a.md"] == "ma"
    run(command="insert", path="/memories/a.md", insert_line=0, insert_text="top")
    assert memory.files["/memories/a.md"] == "top\nma"
    run(command="rename", old_path="/memories/a.md", new_path="/memories/b.md")
    run(command="delete", path="/memories/b.md")
    assert "/memories/a.md" not in memory.files and "/memories/b.md" not in memory.files
    assert memory.calls[1] == ("view", "/memories/MEMORY.md", (2, 2))

    dispatched = len(memory.calls)
    bad_inputs: list[Any] = [
        {"command": "view"},
        {"command": "view", "path": 3},
        {"command": "view", "path": "/memories", "view_range": [1]},
        {"command": "view", "path": "/memories", "view_range": "1-2"},
        {"command": "view", "path": "/memories", "view_range": [1, True]},
        {"command": "create", "path": "/memories/x.md"},
        {"command": "create", "path": "/memories/x.md", "file_text": None},
        {"command": "str_replace", "path": "/memories/x.md", "new_str": "b"},
        {"command": "str_replace", "path": "/memories/x.md", "old_str": "a", "new_str": 5},
        {"command": "insert", "path": "/memories/x.md", "insert_line": "3", "insert_text": "t"},
        {"command": "insert", "path": "/memories/x.md", "insert_line": True, "insert_text": "t"},
        {"command": "insert", "path": "/memories/x.md", "insert_line": 1},
        {"command": "delete"},
        {"command": "rename", "old_path": "/memories/x.md"},
        {"command": "format_disk", "path": "/memories"},
        {"path": "/memories"},
        "view /memories",
        None,
    ]
    for bad in bad_inputs:
        with pytest.raises(ToolError):
            run_memory_command(memory, bad)
    assert len(memory.calls) == dispatched  # nothing malformed reached the backend


def test_exceeding_max_turns_raises_and_leaves_the_journal_untouched(api, client):
    api.script += [
        msg([tool_use("tu_1", "ping", {})], "tool_use"),
        msg([tool_use("tu_2", "ping", {})], "tool_use"),
        msg([text("after the failure")]),
    ]
    tool = ToolSpec("ping", "Ping.", {"type": "object", "properties": {}}, lambda _: "pong")
    conversation = client.conversation(model=MODEL, tools=[tool], max_turns=2)
    with pytest.raises(LLMError, match="max_turns=2"):
        conversation.send("loop")
    assert conversation.journal == []
    assert conversation.usage.requests == 2
    conversation.send("again")
    assert api.requests[2]["messages"] == [{"role": "user", "content": [text("again")]}]


def test_answer_delivered_before_the_last_tool_call_is_still_returned(api, client):
    latex = "```latex\n\\documentclass{article}\n\\begin{document}Hi\\end{document}\n```"
    note = {"command": "create", "path": "/memories/notes/done.md", "file_text": "wrote the draft"}
    api.script += [
        msg([text(latex), tool_use("tu_1", "memory", note)], "tool_use"),
        msg([text("Saved the note.")]),
        msg(
            [
                text("Looking."),
                tool_use("tu_2", "memory", {"command": "view", "path": "/memories"}),
            ],
            "tool_use",
        ),
        msg([text("The summary.")]),
    ]
    conversation = client.conversation(model=MODEL, memory=DictMemory())
    response = conversation.send("Write the paper.", parser=parse_latex)
    assert response.parsed.startswith("\\documentclass")
    assert response.text.endswith("Saved the note.") and "\\documentclass" in response.text
    # Normally the text is the closing turn only, without the narration before tool calls.
    assert conversation.send("Summarise.").text == "The summary."


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def sdk_native_output_config(model: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    """What the SDK itself sends for ``output_format=model``."""
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(
            200, content=sse(msg([text(json.dumps(payload))])), headers=SSE_HEADERS
        )

    sdk = anthropic.Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    with sdk.beta.messages.stream(
        model=MODEL,
        max_tokens=100,
        output_format=model,
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": "x"}],
    ) as stream:
        stream.get_final_message()
    return seen["output_config"]


def test_pydantic_schema_uses_native_structured_output(api, client):
    payload = {"title": "T", "sections": ["Intro", "Method"]}
    api.script.append(msg([thinking("S1"), text(json.dumps(payload))]))
    response = client.conversation(model=MODEL).send("Outline it.", schema=Outline)

    assert response.parsed == Outline(title="T", sections=["Intro", "Method"])
    assert response.text == json.dumps(payload)
    sent = api.requests[0]["output_config"]
    assert sent == sdk_native_output_config(Outline, payload)
    assert sent["effort"] == "high"
    assert sent["format"]["type"] == "json_schema"
    assert sent["format"]["schema"]["additionalProperties"] is False


def test_dict_schema_returns_a_dict(api, client):
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    api.script.append(msg([text('{"score": 4}')]))
    response = client.conversation(model=MODEL).send("Score it.", schema=schema)
    assert response.parsed == {"score": 4}
    assert api.requests[0]["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {**schema, "additionalProperties": False},
    }
    assert "additionalProperties" not in schema  # the caller's schema is not modified


def test_structured_output_that_fails_validation_is_a_parse_error(api, client):
    class Review(BaseModel):
        score: int = Field(ge=1, le=5)

    api.script += [msg([text('{"score": 9}')]), msg([text('{"score": 9}')])]
    conversation = client.conversation(model=MODEL)
    with pytest.raises(LLMParseError, match="score") as caught:
        conversation.send("Review.", schema=Review)
    assert caught.value.raw_text == '{"score": 9}'
    # The exchange itself completed, so it is part of the conversation.
    assert len(conversation.journal) == 2
    lenient = conversation.send("Review again.", schema=Review, require_parsed=False)
    assert lenient.parsed is None and lenient.text == '{"score": 9}'


def test_truncated_structured_output_is_reported_as_truncation(api, client):
    api.script.append(msg([text('{"title": "T", "sec')], "max_tokens"))
    conversation = client.conversation(model=MODEL)
    with pytest.raises(LLMTruncated) as caught:
        conversation.send("Outline it.", schema=Outline, max_tokens=50)
    assert caught.value.partial_text == '{"title": "T", "sec'
    assert "max_tokens=50" in str(caught.value) and "req_1" in str(caught.value)
    assert conversation.journal == [] and conversation.usage.requests == 1


def test_schemas_native_output_cannot_express_are_rejected_before_any_request(api, client):
    class Scores(BaseModel):
        by_reviewer: dict[str, int]

    class Loose(BaseModel):
        payload: Any

    conversation = client.conversation(model=MODEL)
    with pytest.raises(LLMConfigError, match=r"\$\.by_reviewer"):
        conversation.send("x", schema=Scores)
    with pytest.raises(LLMConfigError, match="cannot be used"):
        conversation.send("x", schema=Loose)
    with pytest.raises(LLMConfigError, match="cannot be used"):
        conversation.send("x", schema="not a schema")  # type: ignore[arg-type]
    with pytest.raises(LLMConfigError, match="open-ended"):
        json_output_format({"type": "object", "properties": {"meta": {"type": "object"}}})
    assert api.requests == [] and conversation.journal == []


def test_schema_with_web_search_is_rejected_up_front_and_text_parsing_still_works(api, client):
    events: list[dict[str, Any]] = []
    conversation = client.conversation(
        model=MODEL, web_search=WebSearch(max_uses=2), on_event=events.append
    )
    with pytest.raises(LLMConfigError, match="web search"):
        conversation.send("Find papers.", schema=Outline)
    assert api.requests == [] and events == []

    api.script.append(msg([text('```json\n{"title": "T", "sections": []}\n```')]))
    response = conversation.send("Find papers.", parser=parse_json)
    assert response.parsed == {"title": "T", "sections": []}
    assert len(api.requests[0]["messages"]) == 1


def test_structured_outputs_switch_disables_schema_and_capability(api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-test-key")
    monkeypatch.setenv(ac.STRUCTURED_OUTPUTS_ENV, "0")
    client = AnthropicClient(http_client=api.http_client(), max_retries=0)
    assert client.capabilities.structured_output is False
    assert AnthropicClient.capabilities.structured_output is True
    assert client.capabilities.memory_tool and client.capabilities.tools
    with pytest.raises(LLMConfigError, match="PAPER_ORCHESTRA_STRUCTURED_OUTPUTS"):
        client.conversation(model=MODEL).send("x", schema=Outline)
    assert api.requests == []


def test_api_rejecting_the_format_is_a_config_error_and_the_caller_can_fall_back(api, client):
    api.script += [
        lambda _: api_error(
            400, "invalid_request_error", "output_config.format: unsupported", "req_fmt"
        ),
        msg([text('{"title": "T", "sections": []}')]),
    ]
    conversation = client.conversation(model=MODEL)
    with pytest.raises(LLMConfigError, match="req_fmt") as caught:
        conversation.send("Outline it.", schema=Outline)
    assert "output_config.format: unsupported" in str(caught.value)
    assert conversation.journal == []

    response = conversation.send("Outline it. Reply with JSON.", parser=parse_json)
    assert response.parsed == {"title": "T", "sections": []}
    assert "format" not in api.requests[1]["output_config"]
    assert len(api.requests[1]["messages"]) == 1  # the rejected turn was not kept


def test_structured_output_after_a_tool_loop_parses_the_closing_turn(api, client):
    payload = {"title": "T", "sections": ["Intro"]}
    api.script += [
        msg(
            [
                thinking("S1"),
                tool_use("tu_1", "lookup", {"q": "x", "scope": "Project"}),
                tool_use("tu_2", "lookup", {"q": "x", "scope": "galaxy"}),
            ],
            "tool_use",
        ),
        msg([text(json.dumps(payload))]),
    ]
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "scope": {"type": "string", "enum": ["project", "lab"]},
        },
        "required": ["q"],
    }
    seen: list[dict[str, Any]] = []

    def lookup(tool_input: dict[str, Any]) -> str:
        seen.append(tool_input)
        return "found"

    conversation = client.conversation(model=MODEL, tools=[ToolSpec("lookup", "d", schema, lookup)])
    response = conversation.send("Outline it.", schema=Outline)

    assert response.parsed == Outline(**payload) and response.tool_calls == 2
    # Enum casing is restored for the handler; the journal keeps what the model wrote.
    assert seen == [{"q": "x", "scope": "project"}]
    assert api.requests[1]["messages"][1]["content"][1]["input"] == {"q": "x", "scope": "Project"}
    rejected = api.requests[1]["messages"][2]["content"][1]
    assert rejected["is_error"] is True and "must be one of 'project', 'lab'" in rejected["content"]
    # The format stays on every request of the exchange.
    assert all("format" in body["output_config"] for body in api.requests)


def test_bad_request_after_the_format_was_accepted_is_not_blamed_on_the_schema(api, client):
    ping = ToolSpec("ping", "Ping.", {"type": "object", "properties": {}}, lambda _: "pong")
    api.script += [
        msg([tool_use("tu_1", "ping", {})], "tool_use"),
        lambda _: api_error(400, "invalid_request_error", "prompt is too long", "req_long"),
    ]
    with pytest.raises(LLMError, match="prompt is too long") as caught:
        client.conversation(model=MODEL, tools=[ping]).send("x", schema=Outline)
    assert not isinstance(caught.value, LLMConfigError)


# ---------------------------------------------------------------------------
# Refusals, truncation, pauses, fallbacks
# ---------------------------------------------------------------------------


def test_refusal_raises_with_category_and_appends_nothing(api, client):
    refusal = {"type": "refusal", "category": "cyber", "explanation": "declined by policy"}
    api.script += [msg([], "refusal", stop_details=refusal), msg([text("fine")])]
    events: list[dict[str, Any]] = []
    conversation = client.conversation(model=MODEL, label="lit.writer", on_event=events.append)
    with pytest.raises(LLMRefusal) as caught:
        conversation.send("Something the classifier dislikes.")
    assert caught.value.category == "cyber"
    for expected in ("cyber", "declined by policy", "req_1", "lit.writer"):
        assert expected in str(caught.value)
    assert conversation.journal == []
    assert conversation.usage.requests == 1  # the declined request still counts
    assert [event["type"] for event in events] == ["request", "assistant", "usage", "error"]
    assert events[-1]["error"].startswith("LLMRefusal:")

    assert conversation.send("A rephrased request.").text == "fine"
    assert api.requests[1]["messages"] == [
        {"role": "user", "content": [text("A rephrased request.")]}
    ]


def test_max_tokens_raises_truncated_with_the_partial_text(api, client):
    ran: list[Any] = []
    tool = ToolSpec("save", "Save.", {"type": "object", "properties": {}}, ran.append)
    api.script += [
        msg([thinking("S1"), text("The first half of")], "max_tokens"),
        msg([text("Saving."), tool_use("tu_1", "save", {})], "max_tokens"),
        msg([text("too long")], "model_context_window_exceeded"),
    ]
    conversation = client.conversation(model=MODEL, tools=[tool])
    with pytest.raises(LLMTruncated, match="before it was finished") as caught:
        conversation.send("Write a lot.")
    assert caught.value.partial_text == "The first half of"
    with pytest.raises(LLMTruncated, match="middle of a tool call"):
        conversation.send("Save it.")
    assert ran == []  # a truncated tool call is never executed
    with pytest.raises(LLMTruncated, match="context window"):
        conversation.send("More.")
    assert conversation.journal == []


def test_pause_turn_is_resumed_without_a_user_message(api, client, monkeypatch):
    paused = [
        thinking("S1"),
        text("Searching."),
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "q"},
        },
    ]
    citation = {
        "type": "web_search_result_location",
        "url": "https://e.org",
        "title": "E",
        "encrypted_index": "IDX",
        "cited_text": "c",
    }
    resumed = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://e.org",
                    "title": "E",
                    "encrypted_content": "ENC",
                    "page_age": None,
                }
            ],
        },
        {"type": "text", "text": "Related work: ", "citations": None},
        {"type": "text", "text": "E shows X.", "citations": [citation]},
    ]
    searched = {
        "input_tokens": 40,
        "output_tokens": 9,
        "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0},
    }
    api.script += [msg(paused, "pause_turn"), msg(resumed, usage=searched)]
    conversation = client.conversation(model=MODEL, web_search=WebSearch(max_uses=4))
    response = conversation.send("Survey the field.")

    # Citation-split text blocks are one passage; the paused narration is a separate one.
    assert response.text == "Searching.\n\nRelated work: E shows X."
    assert response.usage.web_searches == 2 and response.usage.requests == 2
    assert api.requests[1]["messages"] == [
        {"role": "user", "content": [text("Survey the field.")]},
        {"role": "assistant", "content": paused},
    ]
    # Search results and citations are replayed exactly (encrypted fields included).
    assert conversation.journal[2] == {"role": "assistant", "content": resumed}

    monkeypatch.setattr(ac, "MAX_PAUSES", 1)
    api.script += [msg(paused, "pause_turn"), msg(paused, "pause_turn")]
    with pytest.raises(LLMError, match="paused its turn"):
        conversation.send("Again.")


def test_fallback_turn_usage_sums_iterations_and_replay_drops_the_declined_thinking(api, client):
    fallback = {
        "type": "fallback",
        "from": {"model": MODEL},
        "to": {"model": "claude-opus-5"},
        "trigger": {"type": "refusal", "category": "bio"},
    }
    content = [thinking("FABLE"), text("partial"), fallback, thinking("OPUS"), text("The answer.")]
    usage = {
        "input_tokens": 20,
        "output_tokens": 30,
        "cache_read_input_tokens": 5,
        "iterations": [
            {
                "type": "message",
                "model": MODEL,
                "input_tokens": 100,
                "output_tokens": 4,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 25,
            },
            {
                "type": "fallback_message",
                "model": "claude-opus-5",
                "input_tokens": 20,
                "output_tokens": 30,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 0,
            },
        ],
    }
    api.script += [msg(content, model="claude-opus-5", usage=usage), msg([text("ok")])]
    conversation = client.conversation(model=MODEL)
    response = conversation.send("Question.")

    assert response.model == "claude-opus-5"
    assert response.text == "partial\n\nThe answer."
    assert response.usage.to_dict() == {
        "input_tokens": 120,
        "output_tokens": 34,
        "cache_read_input_tokens": 55,
        "cache_creation_input_tokens": 25,
        "requests": 1,
        "web_searches": 0,
    }
    conversation.send("Next.")
    assert api.requests[1]["messages"][1]["content"] == [
        text("partial"),
        fallback,
        thinking("OPUS"),
        text("The answer."),
    ]
    assert api.requests[1]["model"] == MODEL  # the conversation keeps asking for its own model


def test_normalize_fallback_turn_keeps_only_replayable_blocks():
    plain = [thinking("A"), text("x")]
    assert normalize_fallback_turn(plain) is plain
    fallback = {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}}
    turn = [
        thinking("A"),
        text("partial"),
        tool_use("t", "n", {}),
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
        {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {}},
        {"type": "mystery"},
        fallback,
        thinking("B"),
        tool_use("t2", "n", {}),
        text("answer"),
    ]
    kept = normalize_fallback_turn(turn)
    assert [(block["type"], block.get("id") or block.get("signature")) for block in kept] == [
        ("text", None),
        ("server_tool_use", "s1"),
        ("web_search_tool_result", None),
        ("fallback", None),
        ("thinking", "B"),
        ("tool_use", "t2"),
        ("text", None),
    ]


# ---------------------------------------------------------------------------
# Images and PDFs
# ---------------------------------------------------------------------------


def test_image_and_pdf_blocks_are_built_with_documents_first(api, client):
    small = png(64, 48)
    pdf_bytes = b"%PDF-1.4\n% not a real document\n"
    api.script.append(msg([text("ok")]))
    events: list[dict[str, Any]] = []
    conversation = client.conversation(model=MODEL, on_event=events.append)
    conversation.send(
        [
            "Figure 1:",
            Image.from_bytes(small),
            Pdf(pdf_bytes, title="paper.pdf"),
            Pdf(pdf_bytes),
            "Compare.",
        ]
    )

    content = api.requests[0]["messages"][0]["content"]
    pdf_source = {
        "type": "base64",
        "media_type": "application/pdf",
        "data": base64.standard_b64encode(pdf_bytes).decode(),
    }
    assert content == [
        {"type": "document", "source": pdf_source, "title": "paper.pdf"},
        {"type": "document", "source": pdf_source},
        text("Figure 1:"),
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(small).decode(),
            },
        },
        text("Compare."),
    ]
    assert events[0]["user"] == (
        f"Figure 1:\n[image image/png, {len(small)} bytes]\n"
        f"[pdf paper.pdf, {len(pdf_bytes)} bytes]\n[pdf , {len(pdf_bytes)} bytes]\nCompare."
    )


def test_oversized_image_is_downscaled_once_and_replayed_unchanged(api, client, monkeypatch):
    fitted: list[Image] = []
    real_fit = ac.fit_image

    def counting_fit(image: Image) -> Image:
        fitted.append(image)
        return real_fit(image)

    monkeypatch.setattr(ac, "fit_image", counting_fit)
    api.script += [msg([text("a wide figure")]), msg([text("still wide")])]
    conversation = client.conversation(model=MODEL)
    conversation.send([Image.from_bytes(png(5152, 1000)), "What is this?"])
    conversation.send("And now?")

    first = api.requests[0]["messages"][0]["content"][0]
    assert image_size(first) == (2576, 500)
    assert first["source"]["media_type"] == "image/png"
    # The journal replays the fitted bytes; the stored turn is never re-encoded.
    assert api.requests[1]["messages"][0]["content"][0] == first
    assert len(fitted) == 1


def test_fit_image_corrects_types_and_shrinks_to_the_byte_limit(monkeypatch):
    small = Image.from_bytes(png(32, 32))
    assert fit_image(small) is small

    jpeg = io.BytesIO()
    PILImage.new("RGB", (40, 30), "red").save(jpeg, format="JPEG")
    mislabeled = Image(jpeg.getvalue(), media_type="image/png")
    assert fit_image(mislabeled).media_type == "image/jpeg"
    assert fit_image(mislabeled).data == jpeg.getvalue()

    bmp = io.BytesIO()
    PILImage.new("RGB", (40, 30), "red").save(bmp, format="BMP")
    converted = fit_image(Image(bmp.getvalue(), media_type="image/bmp"))
    assert converted.media_type == "image/png"
    assert PILImage.open(io.BytesIO(converted.data)).size == (40, 30)

    # Noise does not compress, so only a smaller (and here JPEG) rendition fits.
    noisy = PILImage.frombytes("RGBA", (300, 300), os.urandom(300 * 300 * 4))
    buffer = io.BytesIO()
    noisy.save(buffer, format="PNG")
    monkeypatch.setattr(ac, "MAX_IMAGE_BASE64_BYTES", 60_000)
    shrunk = fit_image(Image.from_bytes(buffer.getvalue()))
    assert ac._base64_size(shrunk.data) <= 60_000
    assert max(PILImage.open(io.BytesIO(shrunk.data)).size) < 300

    with pytest.raises(LLMError, match="could not be decoded"):
        fit_image(Image(b"definitely not an image"))


# ---------------------------------------------------------------------------
# Errors and interrupted streams
# ---------------------------------------------------------------------------


def test_api_errors_map_to_llm_errors_with_the_request_id(api, client):
    api.script += [
        lambda _: api_error(400, "invalid_request_error", "prompt is too long", "req_400"),
        lambda _: api_error(401, "authentication_error", "invalid x-api-key", "req_401"),
        lambda _: api_error(403, "permission_error", "no access to this model", "req_403"),
        lambda _: api_error(429, "rate_limit_error", "slow down", "req_429"),
        lambda _: api_error(529, "overloaded_error", "overloaded", "req_529"),
    ]
    events: list[dict[str, Any]] = []
    conversation = client.conversation(model=MODEL, label="stage.agent", on_event=events.append)

    with pytest.raises(LLMError, match="prompt is too long") as bad_request:
        conversation.send("x")
    assert not isinstance(bad_request.value, LLMConfigError)
    assert "req_400" in str(bad_request.value) and "stage.agent" in str(bad_request.value)
    with pytest.raises(LLMConfigError, match="req_401"):
        conversation.send("x")
    with pytest.raises(LLMConfigError, match="no access to this model"):
        conversation.send("x")
    with pytest.raises(LLMError, match=r"429 \(rate_limit_error\).*req_429"):
        conversation.send("x")
    with pytest.raises(LLMError, match="529"):
        conversation.send("x")
    assert conversation.journal == [] and conversation.usage.requests == 0
    assert [event["type"] for event in events] == ["request", "error"] * 5
    assert "sk-secret-test-key" not in json.dumps(events)


def test_sdk_retries_transient_status_errors(api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-test-key")
    client = AnthropicClient(http_client=api.http_client(), max_retries=2)
    api.script += [
        lambda _: api_error(529, "overloaded_error", "overloaded"),
        lambda _: api_error(429, "rate_limit_error", "slow down"),
        msg([text("made it")]),
    ]
    assert client.conversation(model=MODEL).send("x").text == "made it"
    assert len(api.requests) == 3


class BrokenStream(httpx2.SyncByteStream):
    def __init__(self, data: bytes, error: Exception) -> None:
        self._data = data
        self._error = error

    def __iter__(self):
        yield self._data
        raise self._error


def dropped(error: Exception) -> Callable[[httpx2.Request], httpx2.Response]:
    partial = sse(msg([text("half an answer")]), first_events=3)
    return lambda _: httpx2.Response(200, stream=BrokenStream(partial, error), headers=SSE_HEADERS)


def test_interrupted_stream_is_reissued_unchanged(api, client, caplog):
    cut_short = sse(msg([text("half an answer")]), first_events=3)
    overloaded = (
        cut_short
        + b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}\n\n'
    )
    api.script += [
        dropped(httpx2.ReadError("connection reset")),
        dropped(httpx2.ReadTimeout("no data")),
        msg([text("whole answer")]),
        lambda _: httpx2.Response(200, content=cut_short, headers=SSE_HEADERS),
        lambda _: httpx2.Response(200, content=b"", headers=SSE_HEADERS),
        msg([text("second answer")]),
        lambda _: httpx2.Response(200, content=overloaded, headers=SSE_HEADERS),
        msg([text("third answer")]),
    ]
    conversation = client.conversation(model=MODEL)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.anthropic_client"):
        response = conversation.send("Question.")
    assert response.text == "whole answer"
    assert api.requests[0] == api.requests[1] == api.requests[2]
    assert response.usage.requests == 1  # interrupted attempts report no usage
    assert sum("sending the same request again" in r.getMessage() for r in caplog.records) == 2

    # A stream that just ends, an empty one, and a transient error event are the same case.
    assert conversation.send("Two.").text == "second answer"
    assert conversation.send("Three.").text == "third answer"
    assert len(api.requests) == 8
    assert [turn["role"] for turn in conversation.journal] == ["user", "assistant"] * 3


def test_stream_that_keeps_breaking_and_fatal_stream_errors_raise(api, client):
    invalid = (
        sse(msg([text("x")]), first_events=1)
        + b'event: error\ndata: {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}\n\n'
    )
    api.script += [dropped(httpx2.ReadError("reset"))] * 3
    api.script += [lambda _: httpx2.Response(200, content=invalid, headers=SSE_HEADERS)]
    conversation = client.conversation(model=MODEL)
    with pytest.raises(LLMError, match=r"broke 3 times in a row.*ReadError"):
        conversation.send("x")
    assert len(api.requests) == 3
    with pytest.raises(LLMError, match="invalid_request_error"):
        conversation.send("x")
    assert len(api.requests) == 4  # a non-transient error event is not retried
    assert conversation.journal == []


def test_connection_failure_is_an_llm_error(api, client):
    def refuse(_: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route to host")

    api.script.append(refuse)
    with pytest.raises(LLMError, match="Could not reach the Anthropic API"):
        client.conversation(model=MODEL).send("x")


def test_failing_event_sink_does_not_lose_the_response(api, client, caplog):
    def sink(event: dict[str, Any]) -> None:
        raise OSError("disk full")

    api.script.append(msg([text("answer")]))
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.anthropic_client"):
        response = client.conversation(model=MODEL, on_event=sink).send("x")
    assert response.text == "answer"
    assert any("event sink failed" in record.getMessage() for record in caplog.records)


def test_empty_response_is_returned_with_a_warning_and_keeps_the_journal_valid(api, client, caplog):
    api.script += [msg([]), msg([text("now with text")])]
    conversation = client.conversation(model=MODEL)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.anthropic_client"):
        assert conversation.send("one").text == ""
    assert any("without any text" in record.getMessage() for record in caplog.records)
    # An empty assistant message cannot be replayed, so only the user turn was kept.
    assert conversation.send("two").text == "now with text"
    assert api.requests[1]["messages"] == [
        {"role": "user", "content": [text("one")]},
        {"role": "user", "content": [text("two")]},
    ]
    with pytest.raises(LLMParseError):
        api.script.append(msg([]))
        conversation.send("three", parser=parse_json)


def test_generate_reasks_from_scratch_when_parsing_fails(api, client):
    api.script += [msg([text("no json here")]), msg([text('{"ok": true}')])]
    response = client.generate("Give me JSON.", model=MODEL, parser=parse_json, parse_retries=1)
    assert response.parsed == {"ok": True}
    assert [len(body["messages"]) for body in api.requests] == [1, 1]

    api.script.append(msg([text("still nothing")]))
    with pytest.raises(LLMParseError) as caught:
        client.generate("Give me JSON.", model=MODEL, parser=parse_json, parse_retries=0)
    assert caught.value.raw_text == "still nothing"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_missing_credentials_raise_with_both_setups(monkeypatch, api):
    with pytest.raises(LLMConfigError) as caught:
        AnthropicClient(http_client=api.http_client())
    message = str(caught.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "ANTHROPIC_FEDERATION_RULE_ID" in message and "ANTHROPIC_IDENTITY_TOKEN_FILE" in message
    assert "partly configured" not in message

    # Rule id and organization without a token do not activate WIF in the SDK.
    for name, value in WIF_IDS.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(LLMConfigError, match=r"partly configured.*ANTHROPIC_IDENTITY_TOKEN_FILE"):
        AnthropicClient(http_client=api.http_client())
    assert api.requests == []


def test_describe_api_key_and_auth_token(monkeypatch, api):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-test-key")
    info = AnthropicClient(http_client=api.http_client()).describe()
    assert info["provider"] == "anthropic"
    assert info["credential_source"] == "api_key"
    assert info["warnings"] == []
    assert info["base_url"] == "https://api.anthropic.com"
    assert info["structured_outputs"] is True and info["identity_token_bridge"] is None
    assert "sk-secret-test-key" not in json.dumps(info)

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-secret")
    both = AnthropicClient(http_client=api.http_client()).describe()
    assert both["credential_source"] == "api_key"
    assert any("the API key wins" in warning for warning in both["warnings"])

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    token = AnthropicClient(http_client=api.http_client()).describe()
    assert token["credential_source"] == "auth_token" and token["warnings"] == []
    assert "bearer-secret" not in json.dumps(token)
    assert api.requests == []  # describing never talks to the network


def test_describe_workload_identity_and_its_shadowing_traps(monkeypatch, api, tmp_path):
    token_file = tmp_path / "token.jwt"
    token_file.write_text("header.payload.signature")
    for name, value in WIF_IDS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_1")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_1")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token_file))

    info = AnthropicClient(http_client=api.http_client()).describe()
    assert info["credential_source"] == "workload_identity"
    assert info["federation_rule_id"] == "fdrl_1" and info["organization_id"] == "org-1"
    assert info["service_account_id"] == "svac_1" and info["workspace_id"] == "wrkspc_1"
    assert info["identity_token"] == f"file:{token_file}"
    assert info["warnings"] == []
    assert "header.payload.signature" not in json.dumps(info)

    token_file.unlink()
    missing = AnthropicClient(http_client=api.http_client()).describe()
    assert any("does not exist" in warning for warning in missing["warnings"])

    # A literal token works but cannot be renewed once its jti was used.
    monkeypatch.delenv("ANTHROPIC_IDENTITY_TOKEN_FILE")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "header.payload.signature")
    literal = AnthropicClient(http_client=api.http_client()).describe()
    assert literal["credential_source"] == "workload_identity"
    assert literal["identity_token"] == "env:ANTHROPIC_IDENTITY_TOKEN"
    assert any("jti" in warning for warning in literal["warnings"])
    assert "header.payload.signature" not in json.dumps(literal)

    # An API key next to the WIF variables silently wins in the SDK.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-test-key")
    shadowed = AnthropicClient(http_client=api.http_client()).describe()
    assert shadowed["credential_source"] == "api_key"
    assert any("ANTHROPIC_API_KEY is set and takes precedence" in w for w in shadowed["warnings"])
    assert api.requests == []


def test_describe_profile_and_broken_profile(monkeypatch, api, tmp_path):
    config_dir = tmp_path / "anthropic"
    (config_dir / "configs").mkdir(parents=True)
    profile = {
        "organization_id": "org-1",
        "authentication": {
            "type": "oidc_federation",
            "federation_rule_id": "fdrl_1",
            "identity_token": {"source": "file", "path": str(tmp_path / "token.jwt")},
        },
    }
    (config_dir / "configs" / "dev.json").write_text(json.dumps(profile))
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ANTHROPIC_PROFILE", "dev")
    info = AnthropicClient(http_client=api.http_client()).describe()
    assert info["credential_source"] == "profile" and info["profile"] == "dev"
    assert info["config_path"].endswith("dev.json") and info["warnings"] == []

    for name, value in WIF_IDS.items():
        monkeypatch.setenv(name, value)
    shadowed = AnthropicClient(http_client=api.http_client()).describe()
    assert any("ANTHROPIC_PROFILE is set and takes precedence" in w for w in shadowed["warnings"])

    monkeypatch.setenv("ANTHROPIC_PROFILE", "does-not-exist")
    with pytest.raises(LLMConfigError, match="could not be loaded"):
        AnthropicClient(http_client=api.http_client())


def test_describe_credentials_reports_none_and_hides_url_passwords():
    class Bare:
        api_key = None
        auth_token = None
        credentials = None
        base_url = "https://user:hunter2@proxy.example/v1"

    info = describe_credentials(Bare(), {})
    assert info["credential_source"] == "none" and info["warnings"] == []
    assert info["base_url"] == "https://proxy.example/v1"

    class Custom(Bare):
        credentials = object()

    assert describe_credentials(Custom(), {})["credential_source"] == "custom"


def test_bridge_identity_token_is_pure_and_prefers_the_refreshed_file():
    modal_file = {"MODAL_IDENTITY_TOKEN_FILE": "/run/modal/token"}
    modal_static = {"MODAL_IDENTITY_TOKEN": "header.payload.signature"}

    assert bridge_identity_token({}) == {}
    assert bridge_identity_token({**modal_file, **modal_static}) == {}
    assert bridge_identity_token({"ANTHROPIC_FEDERATION_RULE_ID": "fdrl_1", **modal_file}) == {}
    assert bridge_identity_token({"ANTHROPIC_ORGANIZATION_ID": "org-1", **modal_file}) == {}
    assert bridge_identity_token(WIF_IDS) == {}
    assert bridge_identity_token({**WIF_IDS, **modal_file, **modal_static}) == {
        "ANTHROPIC_IDENTITY_TOKEN_FILE": "/run/modal/token"
    }
    assert bridge_identity_token({**WIF_IDS, **modal_static}) == {
        "ANTHROPIC_IDENTITY_TOKEN": "header.payload.signature"
    }
    explicit_file = {**WIF_IDS, **modal_file, "ANTHROPIC_IDENTITY_TOKEN_FILE": "/mine"}
    explicit_token = {**WIF_IDS, **modal_file, "ANTHROPIC_IDENTITY_TOKEN": "mine"}
    assert bridge_identity_token(explicit_file) == {}
    assert bridge_identity_token(explicit_token) == {}

    environ = {**WIF_IDS, **modal_file}
    bridge_identity_token(environ)
    assert environ == {**WIF_IDS, **modal_file}  # the input mapping is not modified


def test_client_bridges_modal_identity_token_into_the_sdk_chain(monkeypatch, api, tmp_path, caplog):
    token_file = tmp_path / "modal-token"
    token_file.write_text("header.payload.signature")
    for name, value in WIF_IDS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN", "header.payload.signature")

    with caplog.at_level(logging.INFO, logger="paper_orchestra.llm.anthropic_client"):
        info = AnthropicClient(http_client=api.http_client()).describe()
    assert os.environ["ANTHROPIC_IDENTITY_TOKEN_FILE"] == str(token_file)
    assert "ANTHROPIC_IDENTITY_TOKEN" not in os.environ
    assert info["credential_source"] == "workload_identity"
    assert info["identity_token"] == f"file:{token_file}"
    assert info["identity_token_bridge"] == "MODAL_IDENTITY_TOKEN_FILE"
    assert info["warnings"] == []
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "MODAL_IDENTITY_TOKEN_FILE" in logged and "header.payload.signature" not in logged

    # Without the refreshed file the static token is used, with its caveat.
    monkeypatch.delenv("MODAL_IDENTITY_TOKEN_FILE")
    monkeypatch.delenv("ANTHROPIC_IDENTITY_TOKEN_FILE")
    static = AnthropicClient(http_client=api.http_client()).describe()
    assert static["identity_token_bridge"] == "MODAL_IDENTITY_TOKEN"
    assert static["identity_token"] == "env:ANTHROPIC_IDENTITY_TOKEN"
    assert any("jti" in warning for warning in static["warnings"])
    assert "header.payload.signature" not in json.dumps(static)
    assert api.requests == []


def test_workload_identity_exchanges_the_token_and_sends_the_access_token(
    monkeypatch, api, tmp_path
):
    token_file = tmp_path / "token.jwt"
    token_file.write_text("header.payload.signature")
    for name, value in WIF_IDS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN_FILE", str(token_file))
    exchanges: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/oauth/token":
            exchanges.append(json.loads(request.content))
            granted = {
                "access_token": "sk-ant-oat01-minted",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
            return httpx2.Response(200, json=granted)
        return api(request)

    def keyless_client() -> AnthropicClient:
        client = AnthropicClient(
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
            max_retries=0,
        )
        # The SDK exchanges the token over an HTTP client of its own (not the
        # one given to the API client), so that one is pointed at the mock too.
        exchange_client = httpx2.Client(transport=httpx2.MockTransport(handler))
        monkeypatch.setattr(client._sdk.credentials, "_http_client", exchange_client)
        return client

    api.script.append(msg([text("keyless hello")]))
    assert keyless_client().conversation(model=MODEL).send("hi").text == "keyless hello"
    assert exchanges == [
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": "header.payload.signature",
            "federation_rule_id": "fdrl_1",
            "organization_id": "org-1",
        }
    ]
    assert api.headers[0]["authorization"] == "Bearer sk-ant-oat01-minted"

    # A token file that disappears is a configuration problem, not an API error.
    token_file.unlink()
    with pytest.raises(LLMConfigError, match="credentials could not be used"):
        keyless_client().conversation(model=MODEL).send("hi")
    assert len(exchanges) == 1
