"""A scripted, offline :class:`LLMClient` for tests and dry runs.

``FakeLLM`` takes a *responder*: a callable that receives a
:class:`FakeRequest` and returns the assistant's final text.  The responder
may exercise tools and memory through the request object exactly as a model
would, so agent logic, the memory harness and the pipeline can be tested end
to end without a network call.

    def responder(req: FakeRequest) -> str:
        if req.label == "outline":
            return json.dumps({...})
        req.memory.create("/memories/notes/x.md", "learned something")
        return "```latex\\n...\\n```"

    client = FakeLLM(responder)
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from paper_orchestra.llm.base import (
    Capabilities,
    Conversation,
    Effort,
    EventSink,
    Image,
    LLMClient,
    LLMParseError,
    LLMResponse,
    MemoryBackend,
    Parser,
    Part,
    Pdf,
    Schema,
    Text,
    ToolError,
    ToolSpec,
    Usage,
    WebSearch,
    normalize_parts,
)
from paper_orchestra.llm.parsing import parse_json


@dataclass
class FakeRequest:
    """What the "model" sees for one user turn."""

    model: str
    system: str | None
    label: str
    parts: list[Text | Image | Pdf]
    turn: int
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    memory: MemoryBackend | None = None
    web_search: WebSearch | None = None
    effort: Effort | None = None
    schema: Schema | None = None
    #: Earlier ``(user_text, assistant_text)`` pairs of this conversation.
    history: list[tuple[str, str]] = field(default_factory=list)
    _events: EventSink | None = field(default=None, repr=False)
    tool_calls: int = 0

    @property
    def text(self) -> str:
        """All text parts of this turn, joined."""
        return "\n".join(p.text for p in self.parts if isinstance(p, Text))

    @property
    def images(self) -> list[Image]:
        return [p for p in self.parts if isinstance(p, Image)]

    @property
    def pdfs(self) -> list[Pdf]:
        return [p for p in self.parts if isinstance(p, Pdf)]

    def call_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Invoke a client tool the way the provider loop would."""
        if name not in self.tools:
            raise KeyError(f"tool '{name}' was not offered (have: {sorted(self.tools)})")
        self.tool_calls += 1
        self._emit({"type": "tool_call", "name": name, "input": tool_input})
        try:
            result, is_error = self.tools[name].handler(tool_input), False
        except ToolError as exc:
            result, is_error = str(exc), True
        self._emit({"type": "tool_result", "name": name, "is_error": is_error, "content": result})
        return result

    def _emit(self, event: dict[str, Any]) -> None:
        if self._events is not None:
            self._events(event)


Responder = Callable[[FakeRequest], str]


class _FakeConversation(Conversation):
    def __init__(self, client: FakeLLM, **options: Any) -> None:
        self._client = client
        self._options = options
        self.model = options["model"]
        self._usage = Usage()
        self._history: list[tuple[str, str]] = []

    @property
    def usage(self) -> Usage:
        return self._usage

    def send(
        self,
        parts: Part | Sequence[Part],
        *,
        parser: Parser | None = None,
        schema: Schema | None = None,
        require_parsed: bool = True,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        opts = self._options
        on_event: EventSink | None = opts.get("on_event")
        request = FakeRequest(
            model=self.model,
            system=opts.get("system"),
            label=opts.get("label") or "",
            parts=normalize_parts(parts),
            turn=len(self._history),
            tools={tool.name: tool for tool in opts.get("tools") or ()},
            memory=opts.get("memory"),
            web_search=opts.get("web_search"),
            effort=opts.get("effort"),
            schema=schema,
            history=list(self._history),
            _events=on_event,
        )
        request._emit(
            {
                "type": "request",
                "label": request.label,
                "model": self.model,
                "user": _describe_parts(request.parts),
            }
        )
        text = self._client._respond(request)
        self._history.append((request.text, text))

        usage = Usage(
            input_tokens=max(1, len(request.text) // 4),
            output_tokens=max(1, len(text) // 4),
            requests=1,
        )
        self._usage.add(usage)
        request._emit(
            {"type": "assistant", "text": text, "stop_reason": "end_turn", "model": self.model}
        )
        request._emit({"type": "usage", **usage.to_dict()})

        parsed: Any = None
        if schema is not None:
            parsed = _validate(schema, text)
        elif parser is not None:
            parsed = parser(text)
        if require_parsed and (schema is not None or parser is not None) and parsed is None:
            raise LLMParseError("fake response did not parse", raw_text=text)
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=self.model,
            usage=usage,
            stop_reason="end_turn",
            tool_calls=request.tool_calls,
        )


def _validate(schema: Schema, text: str) -> Any:
    data = parse_json(text)
    if data is None:
        return None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            return schema.model_validate(data)
        except ValidationError:
            return None
    return data


def _describe_parts(parts: Sequence[Text | Image | Pdf]) -> str:
    out = []
    for part in parts:
        if isinstance(part, Text):
            out.append(part.text)
        elif isinstance(part, Image):
            out.append(f"[image {part.media_type}, {len(part.data)} bytes]")
        else:
            out.append(f"[pdf {part.title or ''}, {len(part.data)} bytes]")
    return "\n".join(out)


class FakeLLM(LLMClient):
    """Scripted client.  Records every request in ``requests``."""

    provider = "fake"
    capabilities = Capabilities(
        tools=True,
        memory_tool=True,
        web_search=True,
        pdf_input=True,
        image_input=True,
        structured_output=True,
        image_generation=True,
    )

    def __init__(
        self,
        responder: Responder | Sequence[str] | Mapping[str, str | Responder],
        *,
        image: bytes | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        """``responder`` may be a callable, a queue of texts, or a ``label -> text`` map.

        With a mapping, the request label is matched exactly first and then by
        prefix; the ``"*"`` key is the default.
        """
        self._responder = responder
        self._queue = list(responder) if isinstance(responder, (list, tuple)) else None
        self._image = image
        self._lock = threading.Lock()
        self.requests: list[FakeRequest] = []
        self.image_prompts: list[str] = []
        if capabilities is not None:
            self.capabilities = capabilities

    def _respond(self, request: FakeRequest) -> str:
        with self._lock:
            self.requests.append(request)
            if self._queue is not None:
                if not self._queue:
                    raise AssertionError(f"FakeLLM ran out of responses at '{request.label}'")
                return self._queue.pop(0)
        responder = self._responder
        if isinstance(responder, Mapping):
            chosen = responder.get(request.label)
            if chosen is None:
                for key, value in responder.items():
                    if key != "*" and request.label.startswith(key):
                        chosen = value
                        break
            if chosen is None:
                chosen = responder.get("*")
            if chosen is None:
                raise AssertionError(f"FakeLLM has no response for label '{request.label}'")
            return chosen(request) if callable(chosen) else chosen
        assert callable(responder)
        result = responder(request)
        return result if isinstance(result, str) else json.dumps(result)

    def conversation(
        self,
        *,
        model: str,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        memory: MemoryBackend | None = None,
        web_search: WebSearch | None = None,
        effort: Effort | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_turns: int = 60,
        label: str = "",
        on_event: EventSink | None = None,
    ) -> Conversation:
        return _FakeConversation(
            self,
            model=model,
            system=system,
            tools=tools,
            memory=memory,
            web_search=web_search,
            effort=effort,
            label=label,
            on_event=on_event,
        )

    def generate_image(
        self,
        prompt: str,
        *,
        model: str,
        aspect_ratio: str = "16:9",
        system: str | None = None,
    ) -> bytes | None:
        with self._lock:
            self.image_prompts.append(prompt)
        return self._image

    def labels(self) -> list[str]:
        return [request.label for request in self.requests]
