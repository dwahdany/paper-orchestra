"""Provider-neutral LLM contract.

Every agent in the pipeline talks to a model through :class:`LLMClient` and
never imports a provider SDK directly.  A client hands out append-only
:class:`Conversation` objects; ``generate`` is the one-shot convenience.

Design rules that implementations must honour:

* **Append-only.**  A conversation never edits or drops earlier turns.  Claude
  Fable 5.2 validates replayed thinking blocks against the history that
  produced them, so the assistant content is appended back verbatim.
* **No import-time side effects.**  Constructing a client may read credentials;
  importing a module must not.
* **Capabilities are explicit.**  Callers branch on ``capabilities`` instead of
  sniffing model names.
"""

from __future__ import annotations

import base64
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for every error raised by the LLM layer."""


class LLMConfigError(LLMError):
    """Missing SDK, missing credentials, or an unsupported option."""


class LLMRefusal(LLMError):
    """The model (and any fallback) declined the request."""

    def __init__(self, message: str, *, category: str | None = None):
        super().__init__(message)
        self.category = category


class LLMParseError(LLMError):
    """The response could not be parsed into the requested shape."""

    def __init__(self, message: str, *, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


class LLMTruncated(LLMError):
    """The response hit ``max_tokens`` before finishing."""

    def __init__(self, message: str, *, partial_text: str = ""):
        super().__init__(message)
        self.partial_text = partial_text


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_media_type(data: bytes) -> str:
    """Return the media type implied by the leading magic bytes."""
    for magic, media_type in _MAGIC:
        if data.startswith(magic):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unrecognised image format (expected PNG, JPEG, GIF or WebP)")


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Image:
    data: bytes = field(repr=False)
    media_type: str = "image/png"

    @classmethod
    def from_bytes(cls, data: bytes) -> Image:
        return cls(data=data, media_type=sniff_image_media_type(data))

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_base64(cls, b64: str) -> Image:
        return cls.from_bytes(base64.b64decode(b64))

    def to_base64(self) -> str:
        return base64.standard_b64encode(self.data).decode("ascii")


@dataclass(frozen=True)
class Pdf:
    data: bytes = field(repr=False)
    title: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> Pdf:
        p = Path(path)
        return cls(data=p.read_bytes(), title=p.name)

    def to_base64(self) -> str:
        return base64.standard_b64encode(self.data).decode("ascii")


Part = str | Text | Image | Pdf


def normalize_parts(parts: Part | Sequence[Part]) -> list[Text | Image | Pdf]:
    """Accept a bare string / part / sequence and return a list of typed parts."""
    if isinstance(parts, (str, Text, Image, Pdf)):
        parts = [parts]
    out: list[Text | Image | Pdf] = []
    for part in parts:
        if isinstance(part, str):
            out.append(Text(part))
        elif isinstance(part, (Text, Image, Pdf)):
            out.append(part)
        else:
            raise TypeError(f"unsupported content part: {type(part).__name__}")
    return out


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """Token accounting, summed across requests."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    requests: int = 0
    web_searches: int = 0

    def add(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.requests += other.requests
        self.web_searches += other.web_searches
        return self

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "requests": self.requests,
            "web_searches": self.web_searches,
        }


class UsageTracker:
    """Thread-safe per-(label, model) usage ledger shared by a pipeline run.

    A row can also carry what its usage cost in dollars as whoever served it said
    (``cost_usd``): a Claude Agent SDK session reports its own figure, which covers
    what a price per token cannot know (fast mode, sub-agents).  Whoever prices the
    ledger takes that figure for such a row instead of an estimate from its tokens.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[tuple[str, str], Usage] = {}
        self._cost_by_key: dict[tuple[str, str], float] = {}

    def record(
        self, label: str, model: str, usage: Usage, *, cost_usd: float | None = None
    ) -> None:
        with self._lock:
            self._by_key.setdefault((label, model), Usage()).add(usage)
            if cost_usd is not None:
                key = (label, model)
                self._cost_by_key[key] = self._cost_by_key.get(key, 0.0) + float(cost_usd)

    def total(self) -> Usage:
        with self._lock:
            total = Usage()
            for usage in self._by_key.values():
                total.add(usage)
            return total

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            rows = []
            for (label, model), usage in sorted(self._by_key.items()):
                row: dict[str, Any] = {"label": label, "model": model, **usage.to_dict()}
                if (label, model) in self._cost_by_key:
                    row["cost_usd"] = round(self._cost_by_key[(label, model)], 6)
                rows.append(row)
        return {"total": self.total().to_dict(), "by_label": rows}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """A client-side tool the model may call.

    ``handler`` receives the validated input dict and returns the tool result
    text.  Raising :class:`ToolError` (or any exception) is reported back to the
    model as an error result rather than aborting the conversation.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str] = field(repr=False)


class ToolError(Exception):
    """Raised by a tool handler to return an ``is_error`` result to the model."""


@runtime_checkable
class MemoryBackend(Protocol):
    """File-like memory store behind the provider's native memory tool.

    Paths are the model-facing virtual paths (``/memories/...``).  Each method
    returns the text shown to the model and raises :class:`ToolError` on
    invalid input.
    """

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str: ...

    def create(self, path: str, file_text: str) -> str: ...

    def str_replace(self, path: str, old_str: str, new_str: str) -> str: ...

    def insert(self, path: str, insert_line: int, insert_text: str) -> str: ...

    def delete(self, path: str) -> str: ...

    def rename(self, old_path: str, new_path: str) -> str: ...


@dataclass(frozen=True)
class WebSearch:
    """Enable provider-hosted web search for a conversation."""

    max_uses: int = 5
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    text: str
    parsed: Any = None
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    request_id: str | None = None
    tool_calls: int = 0
    raw: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class Capabilities:
    tools: bool = False
    memory_tool: bool = False
    web_search: bool = False
    pdf_input: bool = False
    image_input: bool = True
    structured_output: bool = False
    image_generation: bool = False


# An event sink receives plain-JSON-serialisable dicts describing what happened
# inside a conversation (requests, assistant text, tool calls and results,
# usage).  The memory harness uses it to write the transcript layer.
EventSink = Callable[[dict[str, Any]], None]

Parser = Callable[[str], Any]
Schema = type[BaseModel] | Mapping[str, Any]


class Conversation(ABC):
    """An append-only, optionally tool-using, multi-turn exchange."""

    model: str

    @abstractmethod
    def send(
        self,
        parts: Part | Sequence[Part],
        *,
        parser: Parser | None = None,
        schema: Schema | None = None,
        require_parsed: bool = True,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Append a user turn and run to the model's next end of turn.

        Tool calls are executed and looped internally.  ``schema`` requests
        provider-native structured output (``parsed`` is then a validated
        pydantic instance or a dict); otherwise ``parser`` maps the final text
        to ``parsed``.  With ``require_parsed`` a ``None`` parse raises
        :class:`LLMParseError`.

        If the request itself fails (configuration error, refusal, API error),
        the conversation is left exactly as it was before the call, so the
        caller may send again.  A parse failure keeps the completed exchange.
        """

    @property
    @abstractmethod
    def usage(self) -> Usage:
        """Usage accumulated over every request made by this conversation."""


class LLMClient(ABC):
    """One provider.  Thread-safe; share a single instance across agents."""

    provider: str = ""
    capabilities: Capabilities = Capabilities()

    @abstractmethod
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
        single_turn: bool = False,
    ) -> Conversation:
        """Open a conversation.

        Options the provider or model does not support must either be ignored
        when harmless (``temperature`` on models that reject sampling
        parameters) or raise :class:`LLMConfigError` (``tools`` on a provider
        without tool support).

        ``single_turn`` says the caller sends exactly one user turn and then
        drops the conversation, so a provider may skip caching its tail.
        """

    def generate(
        self,
        parts: Part | Sequence[Part],
        *,
        model: str,
        system: str | None = None,
        parser: Parser | None = None,
        schema: Schema | None = None,
        require_parsed: bool = True,
        web_search: WebSearch | None = None,
        effort: Effort | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "",
        on_event: EventSink | None = None,
        parse_retries: int = 2,
    ) -> LLMResponse:
        """One-shot request.  Re-asks from scratch when parsing fails."""
        last_error: LLMParseError | None = None
        for _ in range(max(1, parse_retries + 1)):
            conversation = self.conversation(
                model=model,
                system=system,
                web_search=web_search,
                effort=effort,
                temperature=temperature,
                max_tokens=max_tokens,
                label=label,
                on_event=on_event,
                single_turn=True,
            )
            try:
                return conversation.send(
                    parts, parser=parser, schema=schema, require_parsed=require_parsed
                )
            except LLMParseError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def generate_image(
        self,
        prompt: str,
        *,
        model: str,
        aspect_ratio: str = "16:9",
        system: str | None = None,
    ) -> bytes | None:
        """Return encoded image bytes, or ``None`` when generation failed."""
        raise LLMConfigError(f"provider '{self.provider}' cannot generate images")

    def describe(self) -> dict[str, Any]:
        """Non-secret diagnostics for ``paper-orchestra doctor``."""
        return {"provider": self.provider}
