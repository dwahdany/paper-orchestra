"""Model id -> provider client.

Clients are built on first use and cached per provider, so a missing optional
SDK or credential for one provider never affects another.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from paper_orchestra.llm.base import LLMClient, LLMConfigError

#: Model id prefixes per provider, matched case-insensitively.
PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude-", "anthropic."),
    "gemini": ("gemini-", "imagen-"),
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "fake": ("fake",),
}

_clients: dict[str, LLMClient] = {}
_lock = threading.Lock()


def provider_for_model(model: str) -> str:
    """Return ``"anthropic"``, ``"gemini"``, ``"openai"`` or ``"fake"`` for a model id."""
    name = model.strip().lower() if isinstance(model, str) else ""
    for provider, prefixes in PROVIDER_PREFIXES.items():
        if name.startswith(prefixes):
            return provider
    supported = "; ".join(
        f"{provider}: {', '.join(prefixes)}" for provider, prefixes in PROVIDER_PREFIXES.items()
    )
    raise LLMConfigError(
        f"Cannot tell which provider serves model {model!r}. "
        f"Supported model id prefixes are {supported}."
    )


def _missing_sdk(provider: str, extra: str | None, exc: ImportError) -> LLMConfigError:
    command = f"uv sync --extra {extra}" if extra else "uv sync"
    return LLMConfigError(
        f"The {provider} provider needs a Python package that is not installed ({exc}). "
        f"Install it with `{command}`."
    )


def _import_guard(provider: str, extra: str | None, build: Callable[[], LLMClient]) -> LLMClient:
    try:
        return build()
    except ImportError as exc:
        # A missing module of this package is a packaging bug, not a missing extra.
        if (exc.name or "").startswith("paper_orchestra"):
            raise
        raise _missing_sdk(provider, extra, exc) from exc


def _build_anthropic() -> LLMClient:
    from paper_orchestra.llm.anthropic_client import AnthropicClient

    return AnthropicClient()


def _build_gemini() -> LLMClient:
    from paper_orchestra.llm.gemini_client import GeminiClient

    return GeminiClient()


def _build_openai() -> LLMClient:
    from paper_orchestra.llm.openai_client import OpenAIClient

    return OpenAIClient()


def _build(provider: str) -> LLMClient:
    if provider == "anthropic":
        return _import_guard(provider, None, _build_anthropic)
    if provider == "gemini":
        return _import_guard(provider, "gemini", _build_gemini)
    if provider == "openai":
        return _import_guard(provider, "openai", _build_openai)
    raise LLMConfigError(
        "No fake client is registered. Call "
        "paper_orchestra.llm.register_client('fake', FakeLLM(...)) before using a 'fake' model."
    )


def get_client(model: str) -> LLMClient:
    """Return the shared client of the provider that serves ``model``."""
    provider = provider_for_model(model)
    with _lock:
        client = _clients.get(provider)
        if client is None:
            client = _clients[provider] = _build(provider)
        return client


def register_client(provider: str, client: LLMClient) -> None:
    """Use ``client`` for every model of ``provider`` (tests, dependency injection)."""
    if provider not in PROVIDER_PREFIXES:
        raise LLMConfigError(
            f"Unknown provider {provider!r}; expected one of {', '.join(PROVIDER_PREFIXES)}."
        )
    with _lock:
        _clients[provider] = client


def reset_clients() -> None:
    """Forget every cached or registered client."""
    with _lock:
        _clients.clear()
