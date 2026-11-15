from __future__ import annotations

import shutil
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    if shutil.which("pdflatex") and shutil.which("bibtex"):
        return
    skip = pytest.mark.skip(reason="pdflatex/bibtex not installed")
    for item in items:
        if "latex" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixed_clock():
    """A deterministic clock that advances one second per call."""
    state = {"n": 0}

    def clock() -> datetime:
        state["n"] += 1
        return datetime(2026, 1, 1, 0, 0, state["n"] % 60, tzinfo=UTC)

    return clock


@pytest.fixture(autouse=True)
def _restore_main_module(monkeypatch):
    """Give every test the ``__main__`` module it started with.

    Streamlit's test harness replaces it with the page under test.  Worker
    processes started with the spawn method re-import the parent's ``__main__``,
    so without this the outcome of later tests would depend on test order.
    """
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No test may open a network connection.

    Mock transports do not cover every path (for example an SDK's separate
    client for credential exchange), so connecting is blocked outright.
    """

    def refuse(self, address, *args, **kwargs):
        raise AssertionError(f"test tried to open a network connection to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture(autouse=True)
def _no_provider_credentials(monkeypatch, tmp_path_factory):
    """Tests never see real credentials, on-disk profiles or a custom API endpoint."""
    # SDKs fall back to profiles under the home directory; give every test an empty one.
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
        "ANTHROPIC_WORKSPACE_ID",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_CUSTOM_HEADERS",
        "PAPER_ORCHESTRA_STRUCTURED_OUTPUTS",
        "PAPER_ORCHESTRA_MODAL_APP",
        "PAPER_ORCHESTRA_MODAL_SECRET",
        "PAPER_ORCHESTRA_MODAL_VOLUME",
        "PAPER_ORCHESTRA_TEMPLATES",
        "MODAL_IDENTITY_TOKEN",
        "MODAL_IDENTITY_TOKEN_FILE",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "GEMINI_API_KEY",
        "VERTEX_AI_PROJECT",
        "VERTEX_AI_LOCATION",
        "OPENAI_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "PAPER_ORCHESTRA_MEMORY_DIR",
        "PAPER_BANANA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
