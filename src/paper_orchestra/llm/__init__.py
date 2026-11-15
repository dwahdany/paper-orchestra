"""Provider-neutral LLM layer.

Importing this package never imports a provider SDK: the provider clients are
resolved on first attribute access (``paper_orchestra.llm.AnthropicClient``) or
through :func:`get_client`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from paper_orchestra.llm.base import (
    Capabilities,
    Conversation,
    Effort,
    EventSink,
    Image,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMParseError,
    LLMRefusal,
    LLMResponse,
    LLMTruncated,
    MemoryBackend,
    Parser,
    Part,
    Pdf,
    Schema,
    Text,
    ToolError,
    ToolSpec,
    Usage,
    UsageTracker,
    WebSearch,
    normalize_parts,
    sniff_image_media_type,
)
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.llm.parsing import (
    identity,
    parse_json,
    parse_json_and_latex,
    parse_json_object,
    parse_latex,
    parse_python,
)
from paper_orchestra.llm.pricing import PRICES, estimate_cost_usd
from paper_orchestra.llm.router import (
    get_client,
    provider_for_model,
    register_client,
    reset_clients,
)

#: Provider clients and the module each one lives in, imported on first access.
#: Only the primary provider is part of ``__all__``: a star import must not
#: depend on the optional providers' modules.
_LAZY_CLIENTS = {
    "AnthropicClient": "paper_orchestra.llm.anthropic_client",
    "GeminiClient": "paper_orchestra.llm.gemini_client",
    "OpenAIClient": "paper_orchestra.llm.openai_client",
}

__all__ = [
    "PRICES",
    "AnthropicClient",
    "Capabilities",
    "Conversation",
    "Effort",
    "EventSink",
    "FakeLLM",
    "FakeRequest",
    "Image",
    "LLMClient",
    "LLMConfigError",
    "LLMError",
    "LLMParseError",
    "LLMRefusal",
    "LLMResponse",
    "LLMTruncated",
    "MemoryBackend",
    "Parser",
    "Part",
    "Pdf",
    "Schema",
    "Text",
    "ToolError",
    "ToolSpec",
    "Usage",
    "UsageTracker",
    "WebSearch",
    "estimate_cost_usd",
    "get_client",
    "identity",
    "normalize_parts",
    "parse_json",
    "parse_json_and_latex",
    "parse_json_object",
    "parse_latex",
    "parse_python",
    "provider_for_model",
    "register_client",
    "reset_clients",
    "sniff_image_media_type",
]


def __getattr__(name: str) -> Any:
    module = _LAZY_CLIENTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
