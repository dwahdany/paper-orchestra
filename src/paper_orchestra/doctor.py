"""``paper-orchestra doctor``: is this machine ready to write a paper?

The report is assembled from the environment and local files only.  Nothing is
sent anywhere, and credentials are reported by where they come from, never by
value.  Every provider and backend is imported lazily and a failure to load
one becomes a row of the report, not a traceback.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from paper_orchestra import __version__
from paper_orchestra.config import DEFAULT_MODEL, LAB_MEMORY_ENV, default_lab_memory_dir

if TYPE_CHECKING:
    from rich.console import Console

S2_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"
REDACTED = "[redacted]"
#: What works with no credential at all, said where a first look finds none.
WITHOUT_CREDENTIALS = (
    "The demonstration and the app work without credentials: `paper-orchestra research "
    "demo` builds a project to look at and `paper-orchestra ui` opens it. Running "
    "approved experiments needs none either, unless the project's agents are Claude "
    "Agent SDK agents (an agent then carries each run out); the agents need a model."
)

_PACKAGES = (
    "anthropic",
    "modal",
    "google-genai",
    "openai",
    "pydantic",
    "typer",
    "rich",
    "pymupdf",
    "fastapi",
    "uvicorn",
    "markdown-it-py",
    "python-multipart",
)
#: Optional providers: the SDK module that must be importable, per provider.
_PROVIDER_SDKS = {"anthropic": "anthropic", "gemini": "google.genai", "openai": "openai"}
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_NOT_A_SECRET_SUFFIX = ("_FILE", "_PATH", "_DIR", "_ID")
#: Shorter values are too likely to be ordinary words to blank out everywhere.
_MIN_SECRET_LENGTH = 12


def secret_values(environ: Mapping[str, str]) -> list[str]:
    """Values of environment variables that hold credentials, longest first."""
    values = {
        value
        for name, value in environ.items()
        if _SECRET_NAME.search(name)
        and not name.upper().endswith(_NOT_A_SECRET_SUFFIX)
        and len(value) >= _MIN_SECRET_LENGTH
    }
    return sorted(values, key=len, reverse=True)


def scrub(value: Any, secrets: Sequence[str]) -> Any:
    """Blank out any credential value that found its way into the report."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, Mapping):
        return {str(key): scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [scrub(item, secrets) for item in value]
    return value


def _message(exc: BaseException) -> str:
    from paper_orchestra.compute.base import ComputeError
    from paper_orchestra.llm.base import LLMConfigError

    # These two are written for the user; anything else needs its type to make sense.
    if isinstance(exc, (LLMConfigError, ComputeError)):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _sdk_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _provider_report(model: str) -> dict[str, Any]:
    """Which credential the provider of ``model`` would use, from its ``describe()``."""
    row: dict[str, Any] = {
        "model": model,
        "provider": None,
        "usable": False,
        "credential_source": None,
        "warnings": [],
        "details": {},
        "error": None,
    }
    try:
        from paper_orchestra.llm.router import get_client, provider_for_model

        row["provider"] = provider_for_model(model)
        described = dict(get_client(model).describe())
    except Exception as exc:
        row["error"] = _message(exc)
        return row
    described.pop("provider", None)
    source = described.pop("credential_source", None)
    row["credential_source"] = source
    row["warnings"] = [str(warning) for warning in described.pop("warnings", None) or []]
    row["details"] = described
    row["usable"] = source != "none"
    if not row["usable"]:
        row["error"] = "No credentials are configured for this provider."
    return row


def environment_credential(provider: str, environ: Mapping[str, str]) -> str | None:
    """Which credential of ``provider`` the environment holds (``api_key``, ``auth_token``,
    ``workload_identity``, ``vertex_ai``), or ``None`` when it holds none.  A cheap look:
    no client is built, and a profile on disk is not seen."""
    if provider == "anthropic":
        if environ.get("ANTHROPIC_API_KEY"):
            return "api_key"
        if environ.get("ANTHROPIC_AUTH_TOKEN"):
            return "auth_token"
        if environ.get("ANTHROPIC_FEDERATION_RULE_ID") and environ.get("ANTHROPIC_ORGANIZATION_ID"):
            return "workload_identity"
    elif provider == "gemini":
        if environ.get("VERTEX_AI_PROJECT") and environ.get("VERTEX_AI_LOCATION"):
            return "vertex_ai"
        if environ.get("GEMINI_API_KEY"):
            return "api_key"
    elif provider == "openai" and environ.get("OPENAI_API_KEY"):
        return "api_key"
    return None


def _other_providers(default: str | None, environ: Mapping[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "provider": provider,
            "sdk_installed": _sdk_installed(module),
            "credential_source": environment_credential(provider, environ),
        }
        for provider, module in _PROVIDER_SDKS.items()
        if provider != default
    ]


def _compute_report(name: str) -> dict[str, Any]:
    try:
        from paper_orchestra.compute import get_backend

        backend = get_backend(name)
        try:
            info = dict(backend.describe())
            # The backend's own answer, the one a run is refused on before it starts.
            problems = [str(problem) for problem in backend.check()]
        finally:
            backend.close()
    except Exception as exc:
        return {"backend": name, "available": False, "error": _message(exc)}
    return {
        **info,
        "backend": name,
        "available": not problems,
        "problems": problems,
        "error": None,
    }


def _memory_report(environ: Mapping[str, str]) -> dict[str, Any]:
    lab = default_lab_memory_dir()
    wiki = lab / "wiki"
    return {
        "lab_dir": str(lab),
        "lab_dir_source": LAB_MEMORY_ENV if environ.get(LAB_MEMORY_ENV) else "default",
        "lab_exists": (lab / "MEMORY.md").is_file(),
        # Counted from the files: opening the wiki would create it.
        "lab_pages": sum(1 for _ in wiki.rglob("*.md")) if wiki.is_dir() else 0,
        "project_dir": "<run directory>/memory",
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect(model: str | None = None) -> dict[str, Any]:
    """Everything ``doctor`` reports, as plain JSON-serialisable data.

    ``ok`` is false when the provider of ``model`` (default: the default model)
    has no usable credentials; the other sections are informational.
    ``without_credentials`` is one line on what works all the same, for whoever
    shows a report that is not ``ok``.
    """
    environ = os.environ
    model = model or DEFAULT_MODEL
    provider = _provider_report(model)
    report = {
        "version": __version__,
        "python": platform.python_version(),
        "model": model,
        "provider": provider,
        "other_providers": _other_providers(provider["provider"], environ),
        "compute": {name: _compute_report(name) for name in ("local", "modal")},
        "semantic_scholar_api_key": bool(environ.get(S2_KEY_ENV)),
        "memory": _memory_report(environ),
        "agents": _agents_report(environ),
        "packages": _package_versions(),
        "ok": bool(provider["usable"]),
        "without_credentials": WITHOUT_CREDENTIALS,
    }
    return scrub(report, secret_values(environ))


def _agents_report(environ: Mapping[str, str]) -> dict[str, Any]:
    """What this machine has of what a project whose agents are Claude Agent SDK agents
    (``settings.agents.runtime: sdk``) needs.  Informational: it never makes the report
    not ``ok``, and the package itself is not imported."""
    try:
        from paper_orchestra.research.sdk.session import machine_report

        return machine_report(environ)
    except Exception as exc:  # the report must come out whatever is wrong here
        return {"error": f"{type(exc).__name__}: {exc}"}


def _agents_rows(agents: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """``(check, status, details)`` for the table: the package and its command line, and
    each of the two credentials an sdk project can choose."""
    if agents.get("error"):
        return [("agents: sdk runtime (optional)", "unavailable", str(agents["error"]))]
    rows = []
    if not agents.get("package"):
        rows.append(
            (
                "agents: sdk runtime (optional)",
                "not installed",
                f"the package claude-agent-sdk is missing: {agents.get('install')}",
            )
        )
    elif not agents.get("command_line"):
        rows.append(
            (
                "agents: sdk runtime (optional)",
                "missing",
                "claude-agent-sdk is installed, but the Claude Code command line the "
                "agents run in was not found",
            )
        )
    else:
        rows.append(
            ("agents: sdk runtime (optional)", "ok", "claude-agent-sdk and its command line")
        )
    login = agents.get("login")
    rows.append(
        (
            "agents: credential login",
            "ok" if login else "not configured" if login is False else "unknown",
            f"a Claude Code login is in {agents.get('login_file')}"
            if login
            else "no Claude Code login was found: run `claude` once and sign in"
            if login is False
            else "on this system the login is not kept in a file; the first agent's start "
            "shows whether you are signed in",
        )
    )
    missing = agents.get("federation_missing") or []
    rows.append(
        (
            "agents: credential federation",
            "ok" if agents.get("federation") else "not configured",
            "the Workload Identity Federation settings are set and the token file is there"
            if agents.get("federation")
            else f"not set: {', '.join(missing)}"
            if missing
            else "the file ANTHROPIC_IDENTITY_TOKEN_FILE names is not there",
        )
    )
    return rows


def _details(info: Mapping[str, Any], skip: Sequence[str] = ()) -> str:
    """``key: value`` pairs of the simple values in ``info``, for a table cell."""
    parts = []
    for key, value in info.items():
        if key in skip or value is None or isinstance(value, (Mapping, list)):
            continue
        parts.append(f"{key.replace('_', ' ')}: {value}")
    return ", ".join(parts)


def render(report: Mapping[str, Any], console: Console) -> None:
    """Print ``report`` (from :func:`collect`) as a table, then warnings and what to fix."""
    from rich.table import Table
    from rich.text import Text

    table = Table(title=f"PaperOrchestra {report['version']} (Python {report['python']})")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details", overflow="fold")

    def row(check: str, status: str, details: str = "") -> None:
        style = {"ok": "green", "missing": "red", "warning": "yellow"}.get(status, "dim")
        # Details are plain text: an error message may contain square brackets.
        table.add_row(check, Text(status, style=style), Text(details))

    provider = report["provider"]
    name = provider["provider"] or "unknown provider"
    if provider["usable"]:
        status = "warning" if provider["warnings"] else "ok"
        details = f"credentials: {provider['credential_source'] or 'configured'}"
        extra = _details(provider["details"])
        row(f"{name} ({report['model']})", status, f"{details}, {extra}" if extra else details)
    else:
        row(f"{name} ({report['model']})", "missing", "no usable credentials (see below)")
    for other in report["other_providers"]:
        if not other["sdk_installed"]:
            row(f"{other['provider']} (optional)", "not installed")
        elif other["credential_source"]:
            row(
                f"{other['provider']} (optional)",
                "ok",
                f"credentials: {other['credential_source']}",
            )
        else:
            row(f"{other['provider']} (optional)", "not configured")
    table.add_section()

    local = report["compute"]["local"]
    if local["available"]:
        row("compute: local", "ok", _details(local, skip=("backend", "available", "error")))
    else:
        row("compute: local", "missing", local.get("error") or " ".join(local["problems"]))
    modal = report["compute"]["modal"]
    if modal.get("error"):
        row("compute: modal (optional)", "unavailable", modal["error"])
    elif modal["available"]:
        # Inside a Modal Function the backend is ready without a token of its own.
        row(
            "compute: modal (optional)",
            "ok",
            "token configured" if modal.get("token_configured") else "ready",
        )
    else:
        row("compute: modal (optional)", "not configured", " ".join(modal["problems"]))
    table.add_section()

    for check, status, details in _agents_rows(report.get("agents") or {}):
        row(check, status, details)
    table.add_section()

    if report["semantic_scholar_api_key"]:
        row("Semantic Scholar API key", "ok", f"{S2_KEY_ENV} is set")
    else:
        row(
            "Semantic Scholar API key",
            "not configured",
            f"{S2_KEY_ENV} is not set: literature lookups use the slower public rate limit",
        )
    memory = report["memory"]
    lab_state = f"{memory['lab_pages']} page(s)" if memory["lab_exists"] else "not created yet"
    row("lab memory", "ok", f"{memory['lab_dir']} ({memory['lab_dir_source']}), {lab_state}")
    row("project memory", "ok", memory["project_dir"])
    table.add_section()

    installed = ", ".join(f"{k} {v}" for k, v in report["packages"].items() if v)
    absent = ", ".join(k for k, v in report["packages"].items() if not v)
    row("packages", "ok", installed + (f"; not installed: {absent}" if absent else ""))
    console.print(table)

    for warning in provider["warnings"]:
        console.print(Text(f"Warning: {warning}", style="yellow"))
    if not provider["usable"]:
        console.print(Text(f"{name} cannot be used: {provider['error']}", style="red"))
        # The newcomer runs this first: nothing here stops them from looking around.
        console.print(Text(str(report.get("without_credentials") or WITHOUT_CREDENTIALS)))
