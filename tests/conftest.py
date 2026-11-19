from __future__ import annotations

import os
import shutil
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Typer decides once, when its formatting module is first imported, whether to style help
# and error text for a terminal, and it does so on GitHub Actions (GITHUB_ACTIONS) and
# whenever colour is forced (FORCE_COLOR, PY_COLORS).  The tests compare that text as
# plain text, so the decision is taken here, before any test imports the command line,
# and no longer depends on where the tests run.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"


#: Set (to anything) where TeX Live is expected, such as CI: the tests that compile real
#: documents then fail loudly instead of being skipped when TeX is missing.
REQUIRE_TEX_ENV = "PAPER_ORCHESTRA_REQUIRE_TEX"


def pytest_collection_modifyitems(config, items):
    if shutil.which("pdflatex") and shutil.which("bibtex"):
        return
    if os.environ.get(REQUIRE_TEX_ENV):
        raise pytest.UsageError(
            f"{REQUIRE_TEX_ENV} is set but pdflatex/bibtex are not on PATH: the tests that "
            "compile real documents would be skipped."
        )
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
def _plain_output(monkeypatch):
    """Tables and messages the code prints itself are plain text too, however the shell
    that started the tests asks for colour."""
    for name in ("FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _supervised_unless_a_test_says_otherwise(monkeypatch):
    """A project a test creates is supervised unless the test itself asks for autopilot,
    whatever the shell that started the tests has set as the default of new projects."""
    monkeypatch.delenv("PAPER_ORCHESTRA_AUTONOMY", raising=False)


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


#: What the names resolve to that the tests of the sdk agents' guard ask about: no test
#: asks a name server.  A name that is not here resolves to an address of the web.
RESOLVES_TO = {
    "127.0.0.1.nip.io": ["127.0.0.1"],
    "169.254.169.254.nip.io": ["169.254.169.254"],
    "localtest.me": ["127.0.0.1", "::1"],
    "mapped.example": ["::ffff:127.0.0.1"],
    "both.example": ["93.184.216.34", "127.0.0.1"],
}


@pytest.fixture(scope="session")
def _agent_temp_base():
    """Where the scripted agents' own directories for temporary files are made during
    the tests: a short path (the real thing needs one), removed with whatever a test
    left in it."""
    import shutil
    import tempfile

    base = tempfile.mkdtemp(prefix="", dir="/tmp" if os.path.isdir("/tmp") else None)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(autouse=True)
def _agents_keep_temporary_files_apart(monkeypatch, _agent_temp_base):
    monkeypatch.setenv("PAPER_ORCHESTRA_AGENT_TMPDIR", _agent_temp_base)


@pytest.fixture(autouse=True)
def _names_resolve_offline(monkeypatch):
    """The check of a web address an agent asks for resolves the name; in a test it is
    answered from :data:`RESOLVES_TO`, and a name ending in ``.invalid`` does not resolve."""
    import socket

    from paper_orchestra.research.sdk import guard

    def addresses(host: str) -> list[str]:
        if host.endswith(".invalid"):
            raise socket.gaierror("Name or service not known")
        return RESOLVES_TO.get(host, ["93.184.216.34"])

    monkeypatch.setattr(guard, "_addresses", addresses, raising=False)


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
