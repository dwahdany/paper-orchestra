"""Model id -> provider client.

Clients are built on first use and cached per provider, so a missing optional
SDK or credential for one provider never affects another.  Provider modules are
imported inside the builders: importing this module imports no SDK.
"""

from __future__ import annotations

import importlib
import os
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


def _build_anthropic() -> LLMClient:
    from paper_orchestra.llm.anthropic_client import AnthropicClient

    return AnthropicClient()


def _build_gemini() -> LLMClient:
    from paper_orchestra.llm.gemini_client import GeminiClient

    return GeminiClient()


def _build_openai() -> LLMClient:
    from paper_orchestra.llm.openai_client import OpenAIClient

    return OpenAIClient()


#: provider -> (optional-dependency extra that ships its SDK, builder).
_BUILDERS: dict[str, tuple[str | None, Callable[[], LLMClient]]] = {
    "anthropic": (None, _build_anthropic),
    "gemini": ("gemini", _build_gemini),
    "openai": ("openai", _build_openai),
}


def _build(provider: str) -> LLMClient:
    if provider not in _BUILDERS:
        raise LLMConfigError(
            f"No {provider} client is registered. Call paper_orchestra.llm.register_client("
            f"'{provider}', FakeLLM(...)) before using a '{provider}' model."
        )
    extra, build = _BUILDERS[provider]
    try:
        return build()
    except ImportError as exc:
        # A missing module of this package is a packaging bug, not a missing extra.
        if (exc.name or "").startswith("paper_orchestra"):
            raise
        command = f"uv sync --extra {extra}" if extra else "uv sync"
        raise LLMConfigError(
            f"The {provider} provider needs a Python package that is not installed ({exc}). "
            f"Install it with `{command}`."
        ) from exc


#: Names a function ``package.module:function`` that returns the client to use for every
#: model.  It exists so that the whole application (a server, its worker processes, a
#: browser) can be exercised offline with a scripted model; it is not a way to configure a
#: provider.
SCRIPTED_CLIENT_ENV = "PAPER_ORCHESTRA_SCRIPTED_CLIENT"


def _scripted_client() -> LLMClient | None:
    target = os.environ.get(SCRIPTED_CLIENT_ENV, "").strip()
    if not target:
        return None
    module_name, _, function_name = target.partition(":")
    try:
        client = getattr(importlib.import_module(module_name), function_name)()
    except Exception as exc:
        raise LLMConfigError(
            f"{SCRIPTED_CLIENT_ENV}={target!r} does not name a function that returns a "
            f"client ({type(exc).__name__}: {exc})."
        ) from exc
    if not isinstance(client, LLMClient):
        raise LLMConfigError(f"{SCRIPTED_CLIENT_ENV}={target!r} did not return a client.")
    return client


def get_client(model: str) -> LLMClient:
    """Return the shared client of the provider that serves ``model``."""
    with _lock:
        if "scripted" not in _clients and os.environ.get(SCRIPTED_CLIENT_ENV):
            scripted = _scripted_client()
            if scripted is not None:
                _clients["scripted"] = scripted
        if "scripted" in _clients:
            return _clients["scripted"]
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
