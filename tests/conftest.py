from __future__ import annotations

import shutil
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
def _no_provider_credentials(monkeypatch):
    """Tests never see real credentials or a custom API endpoint."""
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
        "MODAL_IDENTITY_TOKEN",
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
