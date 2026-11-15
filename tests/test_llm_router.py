from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
import types

import pytest

import paper_orchestra.llm as llm
from paper_orchestra.llm import router
from paper_orchestra.llm.base import LLMConfigError
from paper_orchestra.llm.fake import FakeLLM
from paper_orchestra.llm.router import (
    get_client,
    provider_for_model,
    register_client,
    reset_clients,
)


def _refuse_connection(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("a test tried to open a network connection")


@pytest.fixture(autouse=True)
def _fresh_router(monkeypatch, tmp_path):
    monkeypatch.setattr(socket.socket, "connect", _refuse_connection)
    # No profile on disk may stand in for the credentials the tests remove.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_CONFIG_DIR", raising=False)
    monkeypatch.delenv("MODAL_IDENTITY_TOKEN_FILE", raising=False)
    reset_clients()
    yield
    reset_clients()


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-fable-5-2", "anthropic"),
        ("claude-haiku-5-5", "anthropic"),
        ("anthropic.claude-fable-5-2", "anthropic"),
        ("  Claude-Opus-5-5 ", "anthropic"),
        ("gemini-3-pro-preview", "gemini"),
        ("imagen-4.0-generate-001", "gemini"),
        ("gpt-5", "openai"),
        ("o1", "openai"),
        ("o3-pro", "openai"),
        ("o4-mini", "openai"),
        ("chatgpt-4o-latest", "openai"),
        ("fake", "fake"),
        ("fake-writer", "fake"),
    ],
)
def test_provider_for_model_routes_by_prefix(model, provider):
    assert provider_for_model(model) == provider


@pytest.mark.parametrize(
    "model", ["", "llama-3", "my-gpt-4-proxy", "nano-o3", "claude", "text-gemini-pro", None]
)
def test_unknown_models_are_rejected_instead_of_guessed(model):
    with pytest.raises(LLMConfigError) as caught:
        provider_for_model(model)
    for prefix in ("claude-", "anthropic.", "gemini-", "imagen-", "gpt-", "o1", "chatgpt-", "fake"):
        assert prefix in str(caught.value)


def test_anthropic_client_is_built_once_and_shared(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-test-key")
    client = get_client("claude-fable-5-2")
    assert type(client).__name__ == "AnthropicClient"
    assert client.provider == "anthropic"
    assert get_client("claude-haiku-5-5") is client
    assert get_client("anthropic.claude-fable-5-2") is client
    reset_clients()
    assert get_client("claude-fable-5-2") is not client


def test_missing_credentials_surface_and_a_failed_build_is_not_cached(monkeypatch):
    with pytest.raises(LLMConfigError, match="No Anthropic credentials"):
        get_client("claude-fable-5-2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-test-key")
    assert get_client("claude-fable-5-2").describe()["credential_source"] == "api_key"


def test_registered_clients_take_over_a_provider():
    fake = FakeLLM(["hello"])
    with pytest.raises(LLMConfigError, match="register_client"):
        get_client("fake")
    register_client("fake", fake)
    assert get_client("fake") is fake and get_client("fake-reviewer") is fake
    assert get_client("fake").generate("hi", model="fake").text == "hello"

    # Injecting a client for a real provider needs neither its SDK nor credentials.
    register_client("anthropic", fake)
    assert get_client("claude-fable-5-2") is fake
    with pytest.raises(LLMConfigError, match="Unknown provider"):
        register_client("mistral", fake)
    reset_clients()
    with pytest.raises(LLMConfigError):
        get_client("fake")


def _provider_module(monkeypatch, module_name: str, class_name: str, constructor) -> None:
    module = types.ModuleType(module_name)
    setattr(module, class_name, constructor)
    monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.parametrize(
    ("model", "module_name", "class_name", "sdk", "extra"),
    [
        ("gemini-3-pro", "paper_orchestra.llm.gemini_client", "GeminiClient", "google", "gemini"),
        ("gpt-5", "paper_orchestra.llm.openai_client", "OpenAIClient", "openai", "openai"),
    ],
)
def test_missing_optional_sdk_names_the_extra_to_install(
    monkeypatch, model, module_name, class_name, sdk, extra
):
    def constructor():
        raise ModuleNotFoundError(f"No module named '{sdk}'", name=sdk)

    _provider_module(monkeypatch, module_name, class_name, constructor)
    with pytest.raises(LLMConfigError) as caught:
        get_client(model)
    assert f"`uv sync --extra {extra}`" in str(caught.value)
    assert f"No module named '{sdk}'" in str(caught.value)
    assert isinstance(caught.value.__cause__, ImportError)

    # Once the provider can be built it is cached like any other.
    built = FakeLLM(["ok"])
    _provider_module(monkeypatch, module_name, class_name, lambda: built)
    assert get_client(model) is built and get_client(model) is built


def test_missing_module_of_this_package_is_not_reported_as_a_missing_extra(monkeypatch):
    def constructor():
        name = "paper_orchestra.utils.pdf"
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    _provider_module(monkeypatch, "paper_orchestra.llm.openai_client", "OpenAIClient", constructor)
    with pytest.raises(ModuleNotFoundError):
        get_client("gpt-5")


def test_concurrent_first_use_builds_one_client(monkeypatch):
    built: list[FakeLLM] = []

    def slow_builder() -> FakeLLM:
        time.sleep(0.05)
        built.append(FakeLLM(["x"]))
        return built[-1]

    monkeypatch.setitem(router._BUILDERS, "anthropic", (None, slow_builder))
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(get_client("claude-fable-5-2")))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(built) == 1
    assert len(results) == 8 and all(result is built[0] for result in results)


def test_package_exports_the_public_surface_and_resolves_clients_lazily(monkeypatch):
    for name in llm.__all__:
        if name not in ("AnthropicClient", "GeminiClient", "OpenAIClient"):
            assert hasattr(llm, name), name
    assert llm.get_client is get_client and llm.provider_for_model is provider_for_model
    assert llm.parse_json('{"a": 1}') == {"a": 1}

    from paper_orchestra.llm.anthropic_client import AnthropicClient

    assert llm.AnthropicClient is AnthropicClient
    with pytest.raises(AttributeError, match="NoSuchClient"):
        _ = llm.NoSuchClient


def test_importing_the_llm_package_imports_no_provider_sdk():
    script = (
        "import sys\n"
        "import paper_orchestra.llm\n"
        "import paper_orchestra.llm.anthropic_client\n"
        "import paper_orchestra.config\n"
        "roots = {name.split('.')[0] for name in sys.modules}\n"
        "leaked = roots & {'anthropic', 'httpx2', 'openai', 'google', 'modal', 'PIL'}\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
