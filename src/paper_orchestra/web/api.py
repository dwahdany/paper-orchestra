"""The HTTP API of the web app (spec 7.1 and 10.7).

Every endpoint works on the files of a project through
:class:`~paper_orchestra.research.project.Project`; nothing about a project is
kept in the server's memory, so any tab, after any reload or server restart,
sees the same thing.  The person is the author of every change made here
(``author="human"``).

Shapes, for whoever writes a client:

* a page is ``ResearchPage.to_dict()``: ``path, kind, id, label, title, summary,
  review, state, author, round, created, updated, relations, meta`` and, where
  the text is wanted, ``body``;
* an error is ``{"error": {"code", "message"}}`` with status 400, 404, 409 or
  422 (401 and 403 from the security layer, 413 for an upload over its limit);
  a 409 also carries what the client needs to recover: the current ``page``
  for a stale edit, the running ``worker`` for a second start;
* HTML in a field named ``html``, ``marked_html`` or ``diff_html`` was rendered
  by :mod:`paper_orchestra.web.render` and is the only text a client may insert
  as markup.

The engine, experiment execution and the paper write-up live in other modules
of the package and run in worker processes; what the API needs from them is
reached through :class:`Services`, which tests replace.

Two endpoints go beyond the list of the specification: ``POST
.../experiments/{eid}/run`` (run an approved experiment now) and ``GET
.../worker/log`` (what a worker wrote before it stopped).
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import dataclasses
import importlib
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from paper_orchestra import __version__
from paper_orchestra.memory.wiki import slugify, split_front_matter
from paper_orchestra.research.model import (
    SCHEMA,
    Conflict,
    EditProposal,
    NotAllowed,
    NotFound,
    ProjectError,
    ResearchPage,
    format_stamp,
    parse_id,
)
from paper_orchestra.research.project import (
    EVENTS_FILE,
    PAPER_DIR,
    PROJECT_FILE,
    STATE_FILE,
    Project,
)
from paper_orchestra.research.repo import display_url
from paper_orchestra.web.render import (
    diff_html,
    file_url,
    render_markdown,
    render_marked,
    unified_diff,
)
from paper_orchestra.web.security import (
    FILE_AREAS,
    FILE_CONTENT_SECURITY_POLICY,
    PDF_CONTENT_SECURITY_POLICY,
    PathEscape,
    confined,
    mask_secrets,
    safe_upload_path,
    security_headers,
)
from paper_orchestra.web.workers import (
    RECORD_FILE,
    WORKER_LOG,
    WorkerBusy,
    WorkerError,
    Workers,
    read_state,
)

__all__ = [
    "PHASES",
    "ApiError",
    "Context",
    "LiveSettings",
    "Services",
    "UploadLimits",
    "Workspace",
    "event_stream",
    "install_error_handlers",
    "router",
]

log = logging.getLogger(__name__)

#: The engine's phases (spec 5.3), for checking what a client asks to run.
PHASES: tuple[str, ...] = ("intake", "research", "hypothesize", "design", "run", "analyze")

_MAX_INSTRUCTION_CHARS = 20_000
_MAX_RENDER_CHARS = 1_000_000
_MAX_REVIEW_ITEMS = 500
_LOG_CHUNK_BYTES = 256 * 1024
_LOG_TAIL_BYTES = 16 * 1024
_STAGING_DIR = ".incoming"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """A refusal with its HTTP status.  ``extra`` is added next to ``error`` in the body."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}, **extra}, status_code=status)


def _classify(exc: Exception) -> tuple[int, str]:
    """HTTP status and error code of an exception raised while serving a request."""
    if isinstance(exc, ApiError):
        return exc.status, exc.code
    if isinstance(exc, NotFound):
        return 404, "not_found"
    if isinstance(exc, Conflict):
        return 409, "conflict"
    if isinstance(exc, NotAllowed):
        return 400, "not_allowed"
    if isinstance(exc, ProjectError):
        return 422, "invalid"
    if isinstance(exc, WorkerBusy):
        return 409, "worker_running"
    if isinstance(exc, WorkerError):
        return 400, "worker_failed"
    return 500, "internal"


_STATUS_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "too_large"}


def install_error_handlers(app: FastAPI) -> None:
    """Make every error of the app the JSON object the API promises."""

    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, **exc.extra)

    async def project_error(request: Request, exc: ProjectError) -> JSONResponse:
        status, code = _classify(exc)
        extra: dict[str, Any] = {}
        if isinstance(exc, Conflict):
            # What the client needs to show the person the version they did not see.
            if exc.page is not None:
                extra["page"] = exc.page.to_dict()
            if exc.proposal is not None:
                extra["proposal"] = exc.proposal.to_dict()
            extra["applied"] = exc.applied
        return _error(status, code, str(exc), **extra)

    async def worker_error(request: Request, exc: WorkerError) -> JSONResponse:
        status, code = _classify(exc)
        extra = {"worker": exc.status, "phase": exc.phase} if isinstance(exc, WorkerBusy) else {}
        return _error(status, code, str(exc), **extra)

    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = []
        for problem in exc.errors():
            where = ".".join(str(part) for part in problem.get("loc", ())[1:])
            problems.append(f"{where}: {problem.get('msg')}" if where else str(problem.get("msg")))
        message = "The request is not what this endpoint takes: " + "; ".join(problems)
        return _error(422, "invalid_request", message)

    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "error")
        return _error(exc.status_code, code, str(exc.detail))

    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The type only: the text of an unexpected error may quote anything.
        log.exception("Unhandled error while serving %s %s", request.method, request.url.path)
        message = (
            f"The server hit an unexpected error ({type(exc).__name__}). The details are "
            "in the terminal where it runs."
        )
        response = _error(500, "internal", message)
        # The framework sends this answer from outside every middleware, the security
        # layer included, so the headers that layer adds are put on here.
        for name, value in security_headers(api=request.url.path.startswith("/api/")):
            response.headers.setdefault(name.decode("latin-1"), value.decode("latin-1"))
        return response

    app.add_exception_handler(ApiError, api_error)
    app.add_exception_handler(ProjectError, project_error)
    app.add_exception_handler(WorkerError, worker_error)
    app.add_exception_handler(RequestValidationError, invalid_request)
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(Exception, unexpected)


# ---------------------------------------------------------------------------
# What the API needs from the rest of the package
# ---------------------------------------------------------------------------


class Services:
    """Calls into modules that are built apart from the web app.

    Each method is one call by contract (spec 10.2, 10.6); the module is imported
    when it is first needed, so the API starts without it and a test can put a
    stand-in in its place (``create_app(..., services=...)``).
    """

    def _experiments(self) -> Any:
        try:
            return importlib.import_module("paper_orchestra.research.experiments")
        except ImportError as exc:
            raise ApiError(
                400,
                "not_available",
                f"Experiments cannot be handled by this installation ({exc}). "
                "Reinstall the package.",
            ) from exc

    def load_plan(self, project: Project, ref: str) -> Any:
        return self._experiments().load_plan(project, ref)

    def code_files(self, project: Project, experiment_id: str) -> dict[str, str]:
        return dict(self._experiments().code_files(project, experiment_id))

    def write_code(
        self, project: Project, experiment_id: str, files: Mapping[str, str], *, author: str
    ) -> list[str]:
        return list(
            self._experiments().write_code(project, experiment_id, dict(files), author=author)
        )

    def check_code(self, project: Project, experiment_id: str) -> list[str]:
        return list(self._experiments().check_code(project, experiment_id))

    def code_digest(self, project: Project, experiment_id: str) -> str:
        """What identifies the code as it is now; ``approve`` takes it back as
        ``code_digest`` and refuses when the code changed in between."""
        return str(self._experiments().code_digest(project, experiment_id))

    def code_authors(self, project: Project, experiment_id: str) -> dict[str, dict[str, Any]]:
        """Who wrote the current version of each code file, and when:
        ``{name: {"author", "written"}}``, as ``write_code`` records it.  A file that
        was put there by hand has no entry."""
        return dict(self._experiments().code_authors(project, experiment_id))

    def plan_digest(self, project: Project, experiment_id: str) -> str:
        """What identifies everything a run would execute as things are now: the code
        and the run-defining fields of the plan.  An approval records it."""
        return str(self._experiments().plan_digest(project, experiment_id))

    def approval(self, project: Project, experiment_id: str) -> dict[str, Any]:
        """What was approved and whether the experiment still is that:
        ``{"digest", "at", "matches", "changed", "repaired"}``."""
        return dict(self._experiments().approval(project, experiment_id))

    def approve(self, project: Project, ref: str, **options: Any) -> ResearchPage:
        return self._experiments().approve(project, ref, **options)

    def masked(self, project: Project, text: str) -> str:
        """``text`` of a log without the credentials it may hold: those of the server's
        environment, which workers inherit, and the token in the URL of the project's
        repository, which a job that clones it is handed."""
        try:
            return str(self._experiments().mask_credentials(project, text))
        except ApiError:
            return mask_secrets(text)

    def request_cancel(self, project: Project, ref: str) -> None:
        self._experiments().request_cancel(project, ref)

    def runs(self, project: Project, experiment_id: str) -> list[dict[str, Any]]:
        return list(self._experiments().runs(project, experiment_id))

    def backend_names(self, project: Project) -> list[str]:
        """The compute backends the approval form asks about."""
        from paper_orchestra.compute import BACKEND_NAMES

        return list(BACKEND_NAMES)

    def options(self, project: Project, ref: str) -> list[dict[str, Any]]:
        # By name: the experiments module builds each backend itself, and one that cannot
        # be built here (Modal is not installed) comes back with that as its reason.
        return list(self._experiments().options(project, ref, self.backend_names(project)))

    def number_check(self, project: Project, page: ResearchPage) -> list[dict[str, Any]]:
        return list(self._experiments().number_check(project, page))

    def doctor(self, model: str | None = None) -> dict[str, Any]:
        from paper_orchestra import doctor

        return doctor.collect(model)

    def templates(self) -> list[str]:
        from paper_orchestra.config import bundled_templates

        return sorted(bundled_templates())

    def paper_missing(self, project: Project) -> str | None:
        """What the project still lacks for a paper, by the write-up's own rule:
        ``"results"`` (no accepted experiment has a run that succeeded, and nothing the
        person attached can be reported), ``"finding"`` (no finding is accepted), or
        ``None`` when a paper can be written.  The write-up decides; this only asks it
        without starting it, so that the page and a refused start never disagree."""
        try:
            writeup = importlib.import_module("paper_orchestra.research.writeup")
        except ImportError:
            return None  # Starting the write-up says what is wrong with the installation.
        return writeup.missing(project)


# ---------------------------------------------------------------------------
# The workspace: a directory of project directories
# ---------------------------------------------------------------------------


def _signature(path: Path) -> tuple[int, int, int] | None:
    """What tells one version of a file from the next, or ``None`` when it is absent."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


class Workspace:
    """Finds projects by the id in their ``project.json``.  Thread-safe.

    One :class:`Project` object is kept per directory (it caches parsed pages) and
    re-reads ``project.json`` when another process changed it.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._guard = threading.RLock()
        self._opened: dict[Path, tuple[tuple[int, int, int], Project]] = {}
        self._by_id: dict[str, Path] = {}

    def _open(self, directory: Path) -> Project | None:
        """The project in ``directory``, from the cache while its file is unchanged."""
        signature = _signature(directory / PROJECT_FILE)
        if signature is None:
            self._opened.pop(directory, None)
            return None
        cached = self._opened.get(directory)
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            if cached is not None:
                project = cached[1]
                project.reload()
            else:
                project = Project.open(directory)
        except (ProjectError, OSError) as exc:
            log.warning("Skipping %s: %s", directory, exc)
            self._opened.pop(directory, None)
            return None
        self._opened[directory] = (signature, project)
        return project

    def scan(self) -> dict[str, Project]:
        """Every project in the workspace by id.  Costs one ``stat`` per project that
        has not changed: no page is read."""
        with self._guard:
            found: dict[str, Project] = {}
            try:
                entries = sorted(os.scandir(self.root), key=lambda entry: entry.name)
            except OSError:
                entries = []
            for entry in entries:
                if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
                    continue
                project = self._open(Path(entry.path))
                if project is None:
                    continue
                key, copy = project.id, 1
                while key in found:
                    # A copied project directory keeps its id; both must stay reachable.
                    copy += 1
                    key = f"{project.id}-{slugify(entry.name)}" + (f"-{copy}" if copy > 2 else "")
                found[key] = project
            self._by_id = {key: project.root for key, project in found.items()}
            for directory in set(self._opened) - set(self._by_id.values()):
                del self._opened[directory]
            return found

    def project(self, pid: str) -> Project:
        """The project with that id; ``ApiError`` 404 when the workspace has none."""
        with self._guard:
            directory = self._by_id.get(pid)
            project = self._open(directory) if directory is not None else None
            if project is None or self.key_of(project) != pid:
                project = self.scan().get(pid)
            if project is None:
                raise ApiError(404, "not_found", f"There is no project {pid!r} in this workspace.")
            return project

    def key_of(self, project: Project) -> str:
        """The id the project is addressed by here (its own, unless a copy shares it)."""
        with self._guard:
            for key, directory in self._by_id.items():
                if directory == project.root:
                    return key
            return project.id

    def new_directory(self, title: str) -> Path:
        """A free directory name for a project called ``title``."""
        slug = slugify(title, 48)
        candidate, number = slug, 1
        while (self.root / candidate).exists():
            number += 1
            candidate = f"{slug}-{number}"
        return self.root / candidate


# ---------------------------------------------------------------------------
# What the endpoints share
# ---------------------------------------------------------------------------


@dataclass
class UploadLimits:
    """Sizes a new project's attachments may have."""

    file_bytes: int = 50 * 1024 * 1024
    total_bytes: int = 500 * 1024 * 1024
    files: int = 2000
    #: The brief and the other text fields of the form.
    text_bytes: int = 4 * 1024 * 1024


@dataclass
class LiveSettings:
    """How the event stream follows a project's files."""

    #: Seconds between two looks at the files (a ``stat`` each; nothing is read unless
    #: something changed).
    poll_seconds: float = 0.5
    #: Seconds of silence after which a comment line is sent, so proxies and the browser
    #: keep the connection.
    heartbeat_seconds: float = 15.0
    #: Seconds between two checks that a worker said to be running still is.
    reconcile_seconds: float = 5.0
    #: Set when the server shuts down: every stream ends at its next look.
    closing: threading.Event = field(default_factory=threading.Event)


@dataclass
class Context:
    """Everything an endpoint works with; one per app, at ``app.state.context``."""

    workspace: Workspace
    workers: Workers
    services: Services = field(default_factory=Services)
    live: LiveSettings = field(default_factory=LiveSettings)
    uploads: UploadLimits = field(default_factory=UploadLimits)


def _context(request: Request) -> Context:
    return request.app.state.context


Ctx = Annotated[Context, Depends(_context)]


def _project(pid: str, context: Ctx) -> Project:
    return context.workspace.project(pid)


Proj = Annotated[Project, Depends(_project)]

router = APIRouter(prefix="/api")


class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _many(value: str | None) -> list[str] | None:
    """A filter from a query string: ``a,b`` -> ``["a", "b"]``; empty -> no filter."""
    names = [name.strip() for name in (value or "").split(",") if name.strip()]
    return names or None


# ---------------------------------------------------------------------------
# Health, setup, templates, rendering
# ---------------------------------------------------------------------------


@router.get("/health")
def health(context: Ctx) -> dict[str, Any]:
    return {
        "ok": True,
        "name": "PaperOrchestra",
        "version": __version__,
        "workspace": str(context.workspace.root),
    }


@router.get("/doctor")
def doctor(context: Ctx, model: str | None = None) -> dict[str, Any]:
    """What ``paper-orchestra doctor --json`` reports: where credentials come from and
    which backends are usable, never a value."""
    return context.services.doctor(model)


@router.get("/templates")
def templates(context: Ctx) -> dict[str, Any]:
    return {"templates": [{"name": name} for name in context.services.templates()]}


class RenderBody(_Body):
    markdown: str = ""
    #: The project whose pages and files links resolve against, and the page being edited.
    project: str | None = None
    path: str | None = None


@router.post("/render")
def render(body: RenderBody, context: Ctx) -> dict[str, Any]:
    """Preview for the editor: the same renderer the pages go through."""
    if len(body.markdown) > _MAX_RENDER_CHARS:
        raise ApiError(413, "too_large", "The text is too long to preview at once.")
    project = context.workspace.project(body.project) if body.project else None
    pid = context.workspace.key_of(project) if project is not None else None
    html = render_markdown(body.markdown, project=project, pid=pid, page_path=body.path)
    return {"html": html}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

_NOTHING_SPENT = {"model_usd": 0.0, "experiment_seconds": 0.0, "experiment_usd": 0.0}


def _engine_state(root: Path) -> dict[str, Any]:
    """``state.json`` with every key of spec 5.2 present: a project that has not run has
    no file yet, and a client should not have to care."""
    written = read_state(root)
    spent = written.get("spent")
    return {
        "schema": SCHEMA,
        "status": "idle",
        "round": 0,
        "phase": None,
        "next_phase": None,
        "checkpoint": None,
        "phases": [],
        "worker": None,
        "error": None,
        **written,
        "spent": {**_NOTHING_SPENT, **(spent if isinstance(spent, dict) else {})},
    }


def _worker(context: Context, root: Path, state: Mapping[str, Any]) -> dict[str, Any] | None:
    """The worker's status, looked up only where there can be one: the listing of
    hundreds of projects must not read two more files for each."""
    # ``research run`` from a terminal names itself in state.json without a phase running.
    named = isinstance(state.get("worker"), dict)
    busy = state.get("status") == "running" or named or context.workers.known(root)
    if busy or (root / RECORD_FILE).exists():
        return context.workers.status(root)
    return None


def _state_and_worker(context: Context, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _engine_state(root)
    worker = _worker(context, root, state) or context.workers.status(root)
    if state["status"] == "running" and worker["state"] == "idle":
        state = _engine_state(root)  # the look has just recorded that its worker is gone
    return state, worker


def _modified(root: Path) -> str:
    """When the project last moved, as far as two ``stat`` calls can tell."""
    times = []
    for name in (STATE_FILE, EVENTS_FILE, PROJECT_FILE):
        with contextlib.suppress(OSError):
            times.append((root / name).stat().st_mtime)
    return format_stamp(datetime.fromtimestamp(max(times, default=0.0), UTC))


def _summary(context: Context, key: str, project: Project) -> dict[str, Any]:
    """One row of the library, from ``project.json`` and ``state.json`` alone."""
    state = _engine_state(project.root)
    worker = _worker(context, project.root, state)
    if worker is not None and state["status"] == "running" and worker["state"] == "idle":
        state = _engine_state(project.root)
    checkpoint = state.get("checkpoint") if isinstance(state.get("checkpoint"), dict) else None
    waiting = checkpoint.get("pending") if checkpoint else None
    return {
        "id": key,
        "title": project.title,
        "created": project.created,
        "dir": project.root.name,
        "status": state["status"],
        "round": state["round"],
        "phase": state["phase"],
        "next_phase": state["next_phase"],
        "checkpoint": checkpoint,
        "pending": sum(n for n in waiting.values() if isinstance(n, int))
        if isinstance(waiting, dict)
        else None,
        "spent": state["spent"],
        "error": state["error"],
        "updated": _modified(project.root),
        "worker": worker["state"] if worker else "idle",
    }


def _shown_settings(project: Project) -> dict[str, Any]:
    """The project's settings as they may leave the server: the URL of the repository
    without the password or token it carries, which stays in ``project.json``."""
    settings = project.settings.to_dict()
    if isinstance(settings.get("repo"), Mapping):
        repo = dict(settings["repo"])
        settings["repo"] = {**repo, "path": display_url(str(repo.get("path") or ""))}
    return settings


def _keep_repo_secret(project: Project, changes: dict[str, Any]) -> dict[str, Any]:
    """``changes`` to the settings, with a repository URL that came back as it was shown
    (masked) replaced by the stored one: showing the settings must not cost the token."""
    stored = project.settings.repo
    repo = changes.get("repo")
    sent = repo.get("path") if isinstance(repo, Mapping) else repo
    if stored and isinstance(sent, str) and sent.strip() == display_url(stored.path) != stored.path:
        kept = {**repo, "path": stored.path} if isinstance(repo, Mapping) else stored.path
        return {**changes, "repo": kept}
    return changes


def _project_view(context: Context, project: Project) -> dict[str, Any]:
    state, worker = _state_and_worker(context, project.root)
    usage = _read_json(project.root / "usage.json")
    return {
        "schema": SCHEMA,
        "id": context.workspace.key_of(project),
        "title": project.title,
        "created": project.created,
        "settings": _shown_settings(project),
        "dir": project.root.name,
        "state": state,
        "counts": project.counts(),
        "checkpoint": state.get("checkpoint"),
        "worker": worker,
        "attachments": project.attachments(),
        "usage": usage if isinstance(usage, dict) else None,
        "problems": project.problems()[:50],
    }


@router.get("/projects")
def list_projects(context: Ctx) -> dict[str, Any]:
    """Every project of the workspace, the most recently active first."""
    rows = [_summary(context, key, project) for key, project in context.workspace.scan().items()]
    rows.sort(key=lambda row: (row["updated"], row["created"]), reverse=True)
    return {"projects": rows}


def _upload_too_large(limits: UploadLimits) -> ApiError:
    return ApiError(
        413,
        "too_large",
        f"The upload is larger than the {_megabytes(limits.total_bytes)} allowed for one "
        "project. Attach fewer or smaller files.",
    )


def _limited(receive: Callable[[], Any], limit: int, refusal: ApiError) -> Callable[[], Any]:
    """``receive`` that raises ``refusal`` once the request body is larger than ``limit``
    bytes, while it arrives and whatever length the client announced."""
    seen = 0

    async def limited() -> Any:
        nonlocal seen
        message = await receive()
        if message.get("type") == "http.request":
            seen += len(message.get("body") or b"")
            if seen > limit:
                raise refusal
        return message

    return limited


def _megabytes(count: int) -> str:
    size = count / (1024 * 1024)
    return f"{size:.0f} MB" if size >= 10 else f"{size:.1f} MB"


def _free_path(name: str, files: set[str], folders: set[str]) -> str:
    """``name`` (a relative path), changed where needed so that it collides with nothing
    stored so far: not with a file of the same name, and not with a file where it needs
    a folder or a folder where it is a file.  Compared without case: the workspace may
    be on a file system that ignores it."""
    *parents, leaf = name.split("/")
    prefix = ""
    for part in parents:
        candidate, copy = part, 1
        while f"{prefix}{candidate}".lower() in files:
            copy += 1
            candidate = f"{part}-{copy}"
        prefix = f"{prefix}{candidate}/"
        folders.add(prefix.rstrip("/").lower())
    stem, suffix = os.path.splitext(leaf)
    candidate, copy = leaf, 1
    while f"{prefix}{candidate}".lower() in files | folders:
        copy += 1
        candidate = f"{stem}-{copy}{suffix}"
    files.add(f"{prefix}{candidate}".lower())
    return f"{prefix}{candidate}"


def _stage_uploads(
    uploads: list[UploadFile], staging: Path, limits: UploadLimits
) -> list[dict[str, str]]:
    """Write the uploaded files into ``staging`` under safe names; returns what each was
    called and what it is called now.  Nothing is written anywhere else."""
    stored: list[dict[str, str]] = []
    files: set[str] = set()
    folders: set[str] = set()
    total = 0
    for number, upload in enumerate(uploads, start=1):
        sent = upload.filename or ""
        size = upload.size or 0
        total += size
        if size > limits.file_bytes:
            raise ApiError(
                413,
                "too_large",
                f"{sent or 'A file'} is {_megabytes(size)}; a single file may be at most "
                f"{_megabytes(limits.file_bytes)}. No project was created.",
            )
        if total > limits.total_bytes:
            raise ApiError(
                413,
                "too_large",
                f"The files together are larger than {_megabytes(limits.total_bytes)}. "
                "No project was created.",
            )
        name = _free_path(safe_upload_path(sent) or f"file-{number}", files, folders)
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, handle)
        stored.append({"sent": sent, "stored": name})
    return stored


def _create_project(
    context: Context,
    title: str,
    brief: str,
    settings: Mapping[str, Any] | None,
    uploads: list[UploadFile],
) -> tuple[Project, list[dict[str, str]]]:
    workspace = context.workspace
    workspace.root.mkdir(parents=True, exist_ok=True)
    staging = workspace.root / _STAGING_DIR / secrets.token_hex(8)
    try:
        stored = _stage_uploads(uploads, staging, context.uploads) if uploads else []
        tops = sorted(staging.iterdir()) if staging.is_dir() else []
        name = title.strip() or brief.strip().split("\n", 1)[0] or "project"
        for _ in range(20):
            directory = workspace.new_directory(name)
            try:
                project = Project.create(
                    directory, title=title, brief=brief, files=tops, settings=settings
                )
                break
            except ProjectError:
                if not directory.exists():
                    raise  # refused for what was asked, not because the name was taken
        else:
            raise ApiError(409, "conflict", "No free directory name was found for the project.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        with contextlib.suppress(OSError):
            (workspace.root / _STAGING_DIR).rmdir()
    workspace.scan()
    return project, stored


@router.post("/projects", status_code=201)
async def create_project(request: Request, context: Ctx) -> dict[str, Any]:
    """Start a project from the composer: multipart with ``title`` (optional),
    ``brief``, ``files`` (any number; a folder's files carry their relative path as
    the file name), ``settings`` (a JSON object, partial) and ``start`` (``true`` to
    start the agents at once).

    The project is on disk, with the brief and every file, before anything else
    happens; a refusal leaves nothing behind.  The answer is the project as
    ``GET /api/projects/{pid}`` gives it, plus ``uploads`` (what each file was called
    and what it is called now) and ``skipped``.
    """
    limits = context.uploads
    announced = request.headers.get("content-length", "")
    slack = limits.text_bytes + 1024 * 1024
    # The form around the files (the brief, the part headers) is allowed for on top.
    if announced.isdigit() and int(announced) > limits.total_bytes + slack:
        raise _upload_too_large(limits)
    limited = Request(
        request.scope,
        _limited(request.receive, limits.total_bytes + slack, _upload_too_large(limits)),
    )
    async with limited.form(
        max_files=limits.files, max_fields=100, max_part_size=limits.text_bytes
    ) as form:
        title, brief, raw_settings = (form.get(name) for name in ("title", "brief", "settings"))
        # A text field sent as a file part is not text; it is taken as absent.
        title = title if isinstance(title, str) else ""
        brief = brief if isinstance(brief, str) else ""
        uploads = [
            item
            for name in ("files", "files[]")
            for item in form.getlist(name)
            if isinstance(item, UploadFile)
        ]
        settings: Any = None
        if isinstance(raw_settings, str) and raw_settings.strip():
            try:
                settings = json.loads(raw_settings)
            except ValueError as exc:
                raise ApiError(422, "invalid", f"settings is not valid JSON ({exc}).") from exc
            if not isinstance(settings, dict):
                raise ApiError(422, "invalid", "settings must be a JSON object.")
        if not brief.strip() and not uploads and not title.strip():
            raise ApiError(
                422,
                "invalid",
                "Write what the project is about, or attach a file: there is nothing to "
                "start from.",
            )
        start = _flag(str(form.get("start") or ""))
        project, stored = await run_in_threadpool(
            _create_project, context, title, brief, settings, uploads
        )
    started: dict[str, Any] = {}
    if start:
        try:
            worker = await run_in_threadpool(context.workers.start, project.root, "advance")
            started = {"worker": worker}
        except WorkerError as exc:
            # The project exists all the same; it can be started from its page.
            started = {"worker_error": _failure(exc)["error"]}
    view = await run_in_threadpool(_project_view, context, project)
    manifest = _read_json(project.inputs_dir / "manifest.json")
    skipped = manifest.get("skipped", []) if isinstance(manifest, dict) else []
    return {**view, "uploads": stored, "skipped": skipped, **started}


@router.get("/projects/{pid}")
def get_project(project: Proj, context: Ctx) -> dict[str, Any]:
    """``project.json``, the engine's ``state``, ``counts``, the ``checkpoint`` and the
    ``worker``."""
    return _project_view(context, project)


class ProjectPatch(_Body):
    title: str | None = None
    settings: dict[str, Any] | None = None


@router.patch("/projects/{pid}")
def patch_project(body: ProjectPatch, project: Proj, context: Ctx) -> dict[str, Any]:
    """Change the title and settings (a partial object is merged in, all or nothing)."""
    try:
        if body.settings is not None:
            project.settings.update(_keep_repo_secret(project, body.settings))
        if body.title is not None:
            project.title = body.title
        project.save_settings()
    except ProjectError:
        project.reload()  # what is in memory is again what is on disk
        raise
    return _project_view(context, project)


# ---------------------------------------------------------------------------
# The agents: start, stop
# ---------------------------------------------------------------------------


class AdvanceBody(_Body):
    phase: str | None = None
    instruction: str | None = None


@router.post("/projects/{pid}/advance", status_code=202)
def advance(project: Proj, context: Ctx, body: AdvanceBody | None = None) -> dict[str, Any]:
    """Start the engine from its next phase (or ``phase``) up to the next checkpoint.

    Returns at once; the worker reports through the event stream.  409 with the
    running ``worker`` and ``phase`` when the project already has one.
    """
    body = body or AdvanceBody()
    phase = (body.phase or "").strip() or None
    if phase is not None and phase not in PHASES:
        raise ApiError(
            422, "invalid", f"'{phase}' is not a phase; the phases are: {', '.join(PHASES)}."
        )
    instruction = (body.instruction or "").strip() or None
    if instruction and len(instruction) > _MAX_INSTRUCTION_CHARS:
        raise ApiError(
            422,
            "invalid",
            "The instruction is too long to hand to a phase; leave it as a request on the "
            "project instead.",
        )
    worker = context.workers.start(project.root, "advance", phase=phase, instruction=instruction)
    return {"worker": worker}


@router.post("/projects/{pid}/stop", status_code=202)
def stop(project: Proj, context: Ctx) -> dict[str, Any]:
    """Ask the project's worker to stop.  Returns at once: ``worker.state`` is
    ``stopping`` while it is interrupted (and killed if it does not go), or ``idle``
    when nothing was running."""
    return {"worker": context.workers.stop(project.root)}


@router.get("/projects/{pid}/worker/log")
def worker_log(project: Proj, context: Ctx, offset: int = -_LOG_TAIL_BYTES) -> dict[str, Any]:
    """What the project's workers wrote (``worker.log``): why one could not start, what
    it said before it died.  The same shape as a run's log; by default its end.  Values
    of the server's credentials and the token in the repository's URL are blanked out,
    should a library or a job have echoed one."""
    piece = _read_log(project.root / WORKER_LOG, offset, whole_lines=True)
    return {**piece, "text": context.services.masked(project, piece["text"])}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

_SEQ_FIRST = re.compile(rb'^\{"seq": (\d+)')


def _seq(raw: bytes) -> int | None:
    """The ``seq`` of an event line without parsing the rest (it is written first)."""
    match = _SEQ_FIRST.match(raw)
    if match:
        return int(match.group(1))
    try:
        seq = json.loads(raw).get("seq")
    except (ValueError, AttributeError):
        return None
    return seq if isinstance(seq, int) else None


def _is_event(raw: bytes) -> bool:
    """Whether the line is a whole event.  A writer that died in the middle of a line
    leaves a piece that starts like one; the event written next takes the same seq."""
    try:
        return isinstance(json.loads(raw), dict)
    except ValueError:
        return False


class _Tail:
    """Hands out the complete lines added to a file since the last look."""

    def __init__(self, path: Path, chunk: int = 1 << 20) -> None:
        self.path = path
        self.chunk = chunk
        self.offset = 0
        self._inode: int | None = None

    def grown(self) -> bool:
        """Whether there may be something to read: one ``stat``, nothing is opened."""
        try:
            stat = self.path.stat()
        except OSError:
            return False
        return stat.st_ino != self._inode or stat.st_size != self.offset

    def read(self) -> list[bytes]:
        try:
            stat = self.path.stat()
        except OSError:
            return []
        if stat.st_ino != self._inode or stat.st_size < self.offset:
            # Another file under the same name, or one cut short: start over.  The
            # caller skips what it has already sent by its seq.
            self._inode, self.offset = stat.st_ino, 0
        if stat.st_size == self.offset:
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read(self.chunk)
        end = data.rfind(b"\n")
        if end < 0:
            if len(data) >= self.chunk:
                self.offset += len(data)  # one line longer than a chunk: not an event
            return []  # the last line is still being written
        self.offset += end + 1
        return data[:end].split(b"\n")


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode()


async def event_stream(
    root: Path,
    after: int,
    live: LiveSettings,
    workers: Workers | None = None,
) -> AsyncIterator[bytes]:
    """The project's activity as server-sent events, for as long as the client listens.

    * every event of ``events.jsonl`` with ``seq`` greater than ``after``, in order:
      first those already there, then each new one as it is appended.  The frame
      is ``id: <seq>`` and ``data: <the event>`` with no ``event:`` line, so one
      ``onmessage`` handler receives them all and the browser resumes with
      ``Last-Event-ID``;
    * ``event: state`` with ``{"state": <state.json>, "worker": <worker status>}``
      at the start and whenever either changes;
    * a comment line after ``heartbeat_seconds`` of silence.

    Following costs two ``stat`` calls per look; files are read off the event loop
    and in bounded pieces, so one slow client holds back only itself.
    """
    tail = _Tail(root / EVENTS_FILE)
    last = after
    seen: Any = object()
    busy = False
    quiet_since = checked = time.monotonic()
    yield b"retry: 2000\n\n"
    while not live.closing.is_set():
        sent = False
        now = time.monotonic()
        if workers is not None and busy and now - checked >= live.reconcile_seconds:
            # Notices a worker that died without a word (an engine from a terminal, a run
            # an earlier server started): nothing else watches those.  The state then
            # changes.
            checked = now
            await asyncio.to_thread(workers.status, root)
        glance = workers.glance(root) if workers is not None else "idle"
        current = (_signature(root / STATE_FILE), glance)
        if current != seen:
            seen = current
            state = await asyncio.to_thread(_engine_state, root)
            worker = await asyncio.to_thread(workers.status, root) if workers else None
            busy = state["status"] == "running" or bool(worker and worker["state"] != "idle")
            yield _sse("state", {"state": state, "worker": worker})
            sent = True
        while tail.grown() and (lines := await asyncio.to_thread(tail.read)):
            for raw in lines:
                seq = _seq(raw)
                if seq is None or seq <= last or not _is_event(raw):
                    continue
                last = seq
                yield b"id: %d\ndata: %s\n\n" % (seq, raw.strip())
                sent = True
        if sent:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= live.heartbeat_seconds:
            quiet_since = time.monotonic()
            yield b": keep-alive\n\n"
        await asyncio.sleep(live.poll_seconds)


@router.get("/projects/{pid}/events")
async def events(pid: str, request: Request, context: Ctx, after: int = 0) -> StreamingResponse:
    """``text/event-stream`` of the project's activity (see :func:`event_stream`).

    Starts after ``after``, or after ``Last-Event-ID`` when the browser reconnects.
    """
    project = await run_in_threadpool(context.workspace.project, pid)
    resumed = request.headers.get("last-event-id", "").strip()
    start = max(after, int(resumed)) if resumed.isdigit() else after
    return StreamingResponse(
        event_stream(project.root, max(0, start), context.live, context.workers),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _lines_backwards(path: Path, chunk: int = 1 << 16) -> Iterator[bytes]:
    """The lines of a file, last first, reading only as far back as the caller goes."""
    try:
        handle = path.open("rb")
    except OSError:
        return
    with handle:
        position = handle.seek(0, os.SEEK_END)
        rest = b""
        while position > 0:
            size = min(chunk, position)
            position -= size
            handle.seek(position)
            lines = (handle.read(size) + rest).split(b"\n")
            rest = lines[0]  # may continue in the piece before this one
            for line in reversed(lines[1:]):
                if line.strip():
                    yield line
        if rest.strip():
            yield rest


@router.get("/projects/{pid}/activity")
def activity(project: Proj, before: int | None = None, limit: int = 100) -> dict[str, Any]:
    """Past events as JSON: the ``limit`` latest with ``seq`` below ``before`` (default:
    the latest of all), oldest first.  ``has_more`` says whether older ones exist."""
    limit = max(1, min(limit, 1000))
    found: list[dict[str, Any]] = []
    more = False
    for raw in _lines_backwards(project.root / EVENTS_FILE):
        seq = _seq(raw)
        if seq is None or (before is not None and seq >= before):
            continue
        if len(found) == limit:
            more = True
            break
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if isinstance(event, dict):
            found.append(event)
    found.reverse()
    return {"events": found, "has_more": more}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _node(node: Mapping[str, Any], relation: str) -> dict[str, Any]:
    return {
        "path": node["id"],
        "kind": node["kind"],
        "label": node["label"],
        "title": node["title"],
        "review": node["review"],
        "state": node["state"],
        "type": relation,
    }


def _proposal_view(
    context: Context, project: Project, proposal: EditProposal, page: ResearchPage
) -> dict[str, Any]:
    """An edit proposal with what a person needs to judge it: the unified diff against
    the page as it is, and the page as it would read with the changes marked."""
    proposed = page.body if proposal.body is None else proposal.body
    diff = unified_diff(
        page.body, proposed, old_label="current", new_label=f"proposed by {proposal.author}"
    )
    marked = render_marked(
        page.body,
        proposed,
        project=project,
        pid=context.workspace.key_of(project),
        page_path=page.path,
    )
    return {
        **proposal.to_dict(),
        "diff": diff,
        "diff_html": diff_html(diff),
        "marked_html": marked.html,
        "words_added": marked.words_added,
        "words_removed": marked.words_removed,
        "field_changes": [
            {"field": name, "from": page.meta.get(name), "to": value}
            for name, value in proposal.fields.items()
        ],
        # The page moved on since the edit was written; accepting says what still applies.
        "page_changed": bool(proposal.base_updated) and proposal.base_updated != page.updated,
    }


def _page_view(context: Context, project: Project, page: ResearchPage) -> dict[str, Any]:
    """A page as the wiki view shows it."""
    pid = context.workspace.key_of(project)
    graph = project.graph(include_rejected=True)
    nodes = {node["id"]: node for node in graph["nodes"]}
    view: dict[str, Any] = {
        "page": page.to_dict(),
        "html": render_markdown(page.body, project=project, pid=pid, page_path=page.path),
        "backlinks": [
            _node(nodes[edge["source"]], edge["type"])
            for edge in graph["edges"]
            if edge["target"] == page.path and edge["source"] in nodes
        ],
        "links": [
            _node(nodes[edge["target"]], edge["type"])
            for edge in graph["edges"]
            if edge["source"] == page.path and edge["target"] in nodes
        ],
        "history": [version.to_dict() for version in project.history(page.path)],
        "proposals": [
            _proposal_view(context, project, proposal, page)
            for proposal in project.proposals(review="proposed")
            if proposal.target == page.path
        ],
    }
    if page.kind == "finding":
        try:
            view["number_check"] = context.services.number_check(project, page)
        except (ApiError, ProjectError) as exc:
            # The page is still worth reading without the check; say that it is missing.
            view["number_check"] = None
            view["number_check_error"] = str(exc)
    return view


@router.get("/projects/{pid}/pages")
def list_pages(
    project: Proj,
    kind: str | None = None,
    review: str | None = None,
    state: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    """Pages without their text.  ``kind``, ``review`` and ``state`` take one value or
    several separated by commas; ``q`` keeps the pages that match a search, best first,
    each with its ``snippet``."""
    pages = project.pages(kind=_many(kind), review=_many(review), state=_many(state))
    if q and q.strip():
        hits = {hit.path: hit for hit in project.search(q, k=len(pages) or 1, review=None)}
        order = {path: position for position, path in enumerate(hits)}
        pages = sorted((p for p in pages if p.path in hits), key=lambda p: order[p.path])
        return {
            "pages": [{**p.to_dict(body=False), "snippet": hits[p.path].snippet} for p in pages]
        }
    return {"pages": [page.to_dict(body=False) for page in pages]}


class PageCreate(_Body):
    kind: str
    title: str
    body: str = ""
    summary: str = ""
    fields: dict[str, Any] | None = None
    relations: dict[str, Any] | None = None
    slug: str | None = None


@router.post("/projects/{pid}/pages", status_code=201)
def create_page(body: PageCreate, project: Proj, context: Ctx) -> dict[str, Any]:
    """A page written by the person: accepted on creation."""
    page = project.create_page(
        body.kind,
        body.title,
        body.body,
        author="human",
        summary=body.summary,
        fields=body.fields,
        relations=body.relations,
        slug=body.slug,
    )
    return _page_view(context, project, page)


@router.get("/projects/{pid}/pages/{path:path}/versions/{stamp}")
def get_version(path: str, stamp: str, project: Proj, context: Ctx) -> dict[str, Any]:
    """An earlier version of a page, with the diff from it to the page as it is now."""
    text = project.version(path, stamp)
    meta, body = split_front_matter(text)
    target = project.resolve(path)
    current = project.page(target).body if target else None
    diff = (
        unified_diff(body, current, old_label=f"version of {stamp}", new_label="current")
        if current is not None
        else ""
    )
    return {
        "path": target or path,
        "stamp": stamp,
        "text": text,
        "meta": meta,
        "body": body,
        "html": render_markdown(
            body, project=project, pid=context.workspace.key_of(project), page_path=target
        ),
        "diff": diff,
        "diff_html": diff_html(diff),
    }


@router.get("/projects/{pid}/pages/{path:path}")
def get_page(path: str, project: Proj, context: Ctx) -> dict[str, Any]:
    """``page`` (front matter and body), its ``html``, ``backlinks`` and ``links`` (each
    with the relation ``type``), ``history`` (earlier versions, newest first), the
    pending edit ``proposals`` for it (each with ``diff``, ``diff_html`` and
    ``marked_html``) and, for a finding, ``number_check``.  ``path`` may also be an id
    (``H2``) or a link target."""
    return _page_view(context, project, project.page(path))


class PageEdit(_Body):
    title: str | None = None
    kind: str | None = None
    body: str | None = None
    fields: dict[str, Any] | None = None
    relations: dict[str, Any] | None = None
    base_updated: str | None = None


@router.put("/projects/{pid}/pages/{path:path}")
def put_page(path: str, body: PageEdit, project: Proj, context: Ctx) -> dict[str, Any]:
    """The person's edit.  ``base_updated`` is the ``updated`` stamp of the version they
    started from: when the page changed since, the answer is 409 with the current
    ``page`` and nothing is written."""
    page = project.page(path)
    if body.kind and body.kind != page.kind:
        raise ApiError(
            422,
            "invalid",
            f"This page is {page.kind} and stays one: the kind follows from where a page "
            "is kept. Create a new page of the other kind instead.",
        )
    fields = dict(body.fields or {})
    if body.title is not None:
        fields["title"] = body.title
    updated = project.update_page(
        page.path,
        author="human",
        body=body.body,
        fields=fields or None,
        relations=body.relations,
        base_updated=body.base_updated,
    )
    return _page_view(context, project, updated)


@router.delete("/projects/{pid}/pages/{path:path}")
def delete_page(path: str, project: Proj) -> dict[str, Any]:
    """Remove a page; its last version stays under ``history/``."""
    page = project.page(path)
    project.delete_page(page.path, author="human")
    return {"deleted": page.path}


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


class ReviewItem(_Body):
    ref: str
    decision: str
    reason: str | None = None
    edits: dict[str, Any] | None = None
    #: ``true``, or ``{"backend", "gpu", "timeout_seconds", "network", "code_digest",
    #: "start"}``.
    approve_run: bool | dict[str, Any] | None = None
    base_updated: str | None = None


class ReviewBody(_Body):
    items: list[ReviewItem] = Field(max_length=_MAX_REVIEW_ITEMS)


#: What an approval may say: where and how the run may happen, and ``code_digest``, the
#: digest of the code as the person read it (``code.digest`` of the experiment view).
_APPROVAL_OPTIONS = ("backend", "gpu", "timeout_seconds", "network", "code_digest")


def _approval_options(given: Mapping[str, Any]) -> dict[str, Any]:
    return {name: given[name] for name in _APPROVAL_OPTIONS if given.get(name) is not None}


def _start_run(context: Context, project: Project, experiment_id: str) -> dict[str, Any]:
    return context.workers.start(project.root, "run", experiments=[experiment_id])


def _failure(exc: Exception) -> dict[str, Any]:
    status, code = _classify(exc)
    if status == 500:
        log.exception("Unexpected error while deciding a review item")
        return {"status": 500, "error": {"code": code, "message": type(exc).__name__}}
    failure: dict[str, Any] = {"status": status, "error": {"code": code, "message": str(exc)}}
    if isinstance(exc, Conflict):
        failure["applied"] = exc.applied
        if exc.page is not None:
            failure["page"] = exc.page.to_dict(body=False)
        if exc.proposal is not None:
            failure["proposal"] = exc.proposal.to_dict()
    return failure


def _decide(context: Context, project: Project, item: ReviewItem) -> dict[str, Any]:
    """Apply one decision and say how it went."""
    outcome: dict[str, Any] = {"ref": item.ref, "decision": item.decision, "ok": False}
    wants_run = item.approve_run not in (None, False)
    run_options = item.approve_run if isinstance(item.approve_run, dict) else {}
    try:
        if wants_run:
            target = project.page(item.ref)
            if target.kind != "experiment" or item.decision.strip().lower() not in (
                "accept",
                "accepted",
            ):
                raise ApiError(
                    422,
                    "invalid",
                    "approve_run goes with accepting an experiment: it approves the run of "
                    "the plan being accepted.",
                )
            # One decision: a plan whose code moved on since the person read it is not
            # accepted either (its author may still rewrite the code until then).
            read = run_options.get("code_digest")
            if read is not None and read != context.services.code_digest(project, str(target.id)):
                raise Conflict(
                    f"The code of {target.label} changed after the version that was read. "
                    "Look at the code as it is now, and decide again.",
                    page=target,
                )
        decided = project.review(
            item.ref,
            item.decision,
            reason=item.reason,
            edits=item.edits,
            base_updated=item.base_updated,
        )
    except Exception as exc:
        return {**outcome, **_failure(exc)}
    if isinstance(decided, EditProposal):
        outcome["proposal"] = decided.to_dict()
        with contextlib.suppress(ProjectError):
            outcome["page"] = project.page(decided.target).to_dict(body=False)
    else:
        outcome["page"] = decided.to_dict(body=False)
    if not wants_run:
        return {**outcome, "ok": True}
    assert isinstance(decided, ResearchPage)
    try:
        page = context.services.approve(project, decided.path, **_approval_options(run_options))
    except Exception as exc:
        # The plan is accepted all the same; only its run is not approved.
        failure = _failure(exc)
        failure["error"]["message"] = (
            f"The plan was accepted, but its run was not approved: {failure['error']['message']}"
        )
        return {**outcome, **failure, "approved": False}
    outcome.update(page=page.to_dict(body=False), approved=True, ok=True)
    if run_options.get("start"):
        try:
            outcome["worker"] = _start_run(context, project, page.id or decided.path)
        except WorkerError as exc:
            # Approved stays approved: the run phase picks it up.
            outcome["worker_error"] = _failure(exc)["error"]
    return outcome


@router.post("/projects/{pid}/review")
def review(body: ReviewBody, project: Proj, context: Ctx) -> dict[str, Any]:
    """The person's decisions, applied in order; one failing does not stop the others.

    Each item is ``{"ref", "decision": "accept" | "reject", "reason", "edits",
    "approve_run", "base_updated"}``; ``ref`` is a page (path or id) or the id of an
    edit proposal.  ``approve_run`` (``true``, or the approval options as an object,
    optionally with ``"start": true`` to run it at once) accepts an experiment and
    approves its run in one decision; a plain accept leaves it ``planned``.  With
    ``"code_digest"`` among the options (``code.digest`` of the experiment as it was
    read), code that changed since refuses the whole decision with 409.  A plan that
    asks for network access is approved only with ``"network": true`` or ``false``
    among the options: with ``true`` alone the plan is accepted and its run is not
    approved (422).  An experiment becomes ``approved`` this way or through its own
    ``approve`` endpoint only: a ``state`` among the ``edits`` is refused.

    Always 200 for a well-formed request: ``{"ok": <all succeeded>, "results":
    [...]}`` with, per item, ``ok`` and the decided ``page`` or ``proposal``, or
    ``status`` and ``error`` as the request would have got them on its own (409 for
    a version the person did not see, with the current ``page``).
    """
    results = [_decide(context, project, item) for item in body.items]
    return {"ok": all(result["ok"] for result in results), "results": results}


@router.get("/projects/{pid}/proposals")
def list_proposals(project: Proj, review: str = "proposed") -> dict[str, Any]:
    """Edit proposals, oldest first: the pending ones, or ``review=accepted``,
    ``rejected`` or ``all``.  Each carries the ``page`` it would change (without text)
    and how many lines it adds and removes."""
    wanted = None if review == "all" else review
    pages = {page.path: page for page in project.pages()}
    rows = []
    for proposal in project.proposals(review=wanted):
        page = pages.get(proposal.target)
        row: dict[str, Any] = {
            **proposal.to_dict(),
            "page": page.to_dict(body=False) if page else None,
        }
        if page is not None and proposal.body is not None:
            diff = unified_diff(page.body, proposal.body).split("\n")[2:]
            row["lines_added"] = sum(line.startswith("+") for line in diff)
            row["lines_removed"] = sum(line.startswith("-") for line in diff)
        rows.append(row)
    return {"proposals": rows}


# ---------------------------------------------------------------------------
# Graph, requests, search
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/graph")
def graph(project: Proj, include_rejected: str | None = None) -> dict[str, Any]:
    """``{"nodes", "edges", "dangling"}`` as ``Project.graph`` builds them."""
    return project.graph(include_rejected=_flag(include_rejected))


@router.get("/projects/{pid}/requests")
def list_requests(project: Proj, state: str = "all") -> dict[str, Any]:
    """The person's instructions to the agents, oldest first (``state=open`` or ``done``
    to narrow)."""
    wanted = None if state == "all" else state
    return {"requests": [request.to_dict() for request in project.requests(state=wanted)]}


class RequestBody(_Body):
    text: str
    target: str | None = None


@router.post("/projects/{pid}/requests", status_code=201)
def add_request(body: RequestBody, project: Proj) -> dict[str, Any]:
    return {"request": project.add_request(body.text, target=body.target or None).to_dict()}


@router.delete("/projects/{pid}/requests/{rid}")
def delete_request(rid: str, project: Proj) -> dict[str, Any]:
    project.delete_request(rid)
    return {"deleted": rid}


@router.get("/projects/{pid}/search")
def search(project: Proj, q: str = "", limit: int = 20) -> dict[str, Any]:
    """Keyword search over every page, whatever its review; best first."""
    if not q.strip():
        return {"query": q, "hits": []}
    pages = {page.path: page for page in project.pages()}
    hits = []
    for hit in project.search(q, k=max(1, min(limit, 100)), review=None):
        page = pages.get(hit.path)
        if page is not None:
            hits.append({**page.to_dict(body=False), "score": hit.score, "snippet": hit.snippet})
    return {"query": q, "hits": hits}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def _experiment(project: Project, eid: str) -> ResearchPage:
    """The experiment page ``eid`` names (``E3``); 404 for anything else."""
    parsed = parse_id(eid)
    page = project.page(f"{parsed[0]}{parsed[1]}".upper()) if parsed else None
    if page is None or page.kind != "experiment" or not page.id:
        raise ApiError(404, "not_found", f"There is no experiment {eid!r} in this project.")
    return page


def _run_dir(project: Project, experiment_id: str, number: int) -> Path:
    return project.experiment_dir(experiment_id) / "runs" / str(number)


def _run_view(
    context: Context, project: Project, experiment_id: str, run: Mapping[str, Any]
) -> dict[str, Any]:
    """A run record plus the addresses of what it produced."""
    pid = context.workspace.key_of(project)
    number = run.get("number")
    base = f"experiments/{experiment_id}/runs/{number}"
    result = project.resolve(f"results/{experiment_id}-run{number}")
    return {
        **run,
        "artifact_urls": {
            str(name): file_url(pid, f"{base}/{name}") for name in run.get("artifacts") or []
        },
        "log_url": f"/api/projects/{pid}/experiments/{experiment_id}/runs/{number}/log",
        "result_page": result,
    }


@router.get("/projects/{pid}/experiments")
def list_experiments(project: Proj, context: Ctx) -> dict[str, Any]:
    """Every experiment page (without text) with a summary of its runs: how many, and
    the latest one's record."""
    rows = []
    for page in project.pages(kind="experiment"):
        runs = context.services.runs(project, page.id) if page.id else []
        latest = max(runs, key=lambda run: run.get("number") or 0, default=None)
        rows.append(
            {
                **page.to_dict(body=False),
                "runs": len(runs),
                "last_run": _run_view(context, project, page.id, latest) if latest else None,
            }
        )
    return {"experiments": rows}


def _plan(context: Context, project: Project, page: ResearchPage) -> tuple[Any, str | None]:
    try:
        plan = context.services.load_plan(project, page.path)
    except ProjectError as exc:
        return None, str(exc)
    shown = dataclasses.asdict(plan) if dataclasses.is_dataclass(plan) else plan
    if isinstance(shown, dict) and shown.get("repo"):
        # The plan reads the repository from the settings, token and all; a view does not.
        shown = {**shown, "repo": display_url(str(shown["repo"]))}
    return shown, None


def _experiment_view(context: Context, project: Project, page: ResearchPage) -> dict[str, Any]:
    experiment_id = str(page.id)
    plan, plan_error = _plan(context, project, page)
    files = context.services.code_files(project, experiment_id)
    authors = context.services.code_authors(project, experiment_id)
    try:
        options, options_error = context.services.options(project, page.path), None
    except ProjectError as exc:
        options, options_error = [], str(exc)
    return {
        "page": page.to_dict(),
        "html": render_markdown(
            page.body, project=project, pid=context.workspace.key_of(project), page_path=page.path
        ),
        "plan": plan,
        "plan_error": plan_error,
        "code": {
            "files": [
                {
                    "name": name,
                    "text": text,
                    "size": len(text.encode("utf-8")),
                    # Who wrote this version and when (``None``: nobody recorded it), so
                    # that code which arrived after the plan was accepted can be told.
                    "author": authors.get(name, {}).get("author"),
                    "written": authors.get(name, {}).get("written"),
                }
                for name, text in sorted(files.items())
            ],
            "problems": context.services.check_code(project, experiment_id),
            # Sent back with an approval, so that what is approved is what was read.
            "digest": context.services.code_digest(project, experiment_id),
        },
        # The code and the run-defining fields of the plan as they are now, and what the
        # last approval recorded: when the two differ, the experiment changed since.
        "plan_digest": context.services.plan_digest(project, experiment_id),
        "approval": context.services.approval(project, experiment_id),
        "runs": [
            _run_view(context, project, experiment_id, run)
            for run in context.services.runs(project, experiment_id)
        ],
        "options": options,
        "options_error": options_error,
    }


@router.get("/projects/{pid}/experiments/{eid}")
def get_experiment(eid: str, project: Proj, context: Ctx) -> dict[str, Any]:
    """The experiment's ``page`` and ``html``, its ``plan`` as the runner reads it (or
    ``plan_error``), its ``code`` (``files``, each with the ``author`` of its current
    version and when it was ``written``; the ``problems`` a check finds; and
    the ``digest`` an approval sends back as ``code_digest``), its ``runs``, and
    ``options``: for each backend ``{"backend", "can_run",
    "reasons", "est_minutes", "est_usd", "notes"}``, so the approval form offers
    only what can run.

    ``approval`` is what the person approved: ``{"digest", "at", "matches",
    "changed", "repaired"}``.  ``digest`` and ``at`` are ``None`` for an experiment
    that was never approved; ``matches`` says whether a run would start (the
    experiment is what was approved, or the runner's recorded repair of it:
    ``repaired``); ``changed`` lists what differs since the approval, field by
    field and file by file.  ``plan_digest`` is the digest of the code and the
    run-defining plan fields as they are now, which an approval records.  The
    plan's ``repo`` is shown without the token of its URL."""
    return _experiment_view(context, project, _experiment(project, eid))


class CodeBody(_Body):
    text: str


@router.put("/projects/{pid}/experiments/{eid}/code/{name:path}")
def put_code(eid: str, name: str, body: CodeBody, project: Proj, context: Ctx) -> dict[str, Any]:
    """The person's edit of one code file; earlier versions are kept.  Returns the
    problems a check of the code finds afterwards."""
    page = _experiment(project, eid)
    try:
        # As written, not as a path library would tidy it: "//etc/x" must not pass as "etc/x"
        # here and reach the writer as an absolute name.
        if name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/")):
            raise PathEscape("a code file is named by a plain relative path")
        confined(project.root, f"experiments/{page.id}/code/{name}")
    except PathEscape as exc:
        raise ApiError(422, "invalid", f"{name!r} cannot name a code file: {exc}.") from exc
    problems = context.services.write_code(project, str(page.id), {name: body.text}, author="human")
    return {"name": name, "problems": problems}


class ApproveBody(_Body):
    backend: str | None = None
    gpu: str | None = None
    timeout_seconds: float | None = None
    network: bool | None = None
    #: The ``updated`` stamp of the plan as the person read it: a plan that changed since
    #: is not approved (409).
    base_updated: str | None = None
    #: ``code.digest`` of the experiment as the person read it: code that changed since
    #: is not approved (409).
    code_digest: str | None = None
    #: Also start the run now, in a worker of its own.
    start: bool = False


@router.post("/projects/{pid}/experiments/{eid}/approve")
def approve(
    eid: str, project: Proj, context: Ctx, body: ApproveBody | None = None
) -> dict[str, Any]:
    """The person approves the experiment's run, with where and how it may run.

    The page must be accepted.  What is left out is decided by the approval, never
    taken from the plan: without ``backend`` the run goes to the project's backend
    (Modal for a GPU), and without ``network`` it runs without network access; a plan
    that asks for the network (``network: true``) is refused (422) until the request
    says ``"network": true`` or ``false``.  With ``start`` the run begins at once
    (``worker``), unless the project already has a worker (``worker_error``): an
    approved experiment is then run by the next run phase.
    """
    body = body or ApproveBody()
    page = _experiment(project, eid)
    options = _approval_options(body.model_dump())
    if body.base_updated:
        options["base_updated"] = body.base_updated
    approved = context.services.approve(project, page.path, **options)
    answer: dict[str, Any] = {"page": approved.to_dict(body=False)}
    if body.start:
        try:
            answer["worker"] = _start_run(context, project, str(page.id))
        except WorkerError as exc:
            answer["worker_error"] = _failure(exc)["error"]
    return answer


@router.post("/projects/{pid}/experiments/{eid}/run", status_code=202)
def run_experiment(eid: str, project: Proj, context: Ctx) -> dict[str, Any]:
    """Run an approved experiment now, in a worker of its own (no model is called
    unless its code has to be repaired).  409 when the project already has a worker."""
    page = _experiment(project, eid)
    if page.state not in ("approved", "queued"):
        raise ApiError(
            422,
            "invalid",
            f"{page.id} is {page.state}; only an approved experiment can be run. Approve it first.",
        )
    return {"worker": _start_run(context, project, str(page.id))}


@router.post("/projects/{pid}/experiments/{eid}/cancel", status_code=202)
def cancel(eid: str, project: Proj, context: Ctx) -> dict[str, Any]:
    """Ask the running job of the experiment to stop; the run ends as ``cancelled``."""
    page = _experiment(project, eid)
    context.services.request_cancel(project, page.path)
    return {"cancelling": page.id}


def _find_run(
    context: Context, project: Project, experiment_id: str, number: int
) -> dict[str, Any]:
    for run in context.services.runs(project, experiment_id):
        if run.get("number") == number:
            return run
    raise ApiError(404, "not_found", f"{experiment_id} has no run {number}.")


def _read_log(
    path: Path, offset: int, limit: int = _LOG_CHUNK_BYTES, *, whole_lines: bool = False
) -> dict[str, Any]:
    """A piece of a log file from byte ``offset`` (negative: that many bytes from the
    end), cut at a character boundary so that pieces can be joined.  With
    ``whole_lines`` a piece that is not the end of the file ends at a line break.

    ``eof`` says that nothing more can be read at the moment: ask again later, from
    ``next``.  That is also so when the file ends in the middle of a character (it is
    being written, or its writer was killed): ``next`` then stays before that
    character, and a client that reads on while ``eof`` is false does not spin.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return {"offset": 0, "next": 0, "size": 0, "text": "", "eof": True, "reset": False}
    from_end = offset < 0
    start = max(0, size + offset) if from_end else offset
    reset = start > size  # the file was written anew since the client last read it
    if reset:
        start = 0
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(limit)
    if from_end and start > 0:
        # Begin at a line, not in the middle of one (or of a character).
        skip = data.find(b"\n") + 1
        start, data = start + skip, data[skip:]
    at_end = start + len(data) >= size
    if whole_lines and not at_end and b"\n" in data:
        data = data[: data.rfind(b"\n") + 1]
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(data, final=False)
    consumed = len(data) - len(decoder.getstate()[0])
    return {
        "offset": start,
        "next": start + consumed,
        "size": size,
        "text": text,
        "eof": at_end,
        "reset": reset,
    }


@router.get("/projects/{pid}/experiments/{eid}/runs/{number}")
def get_run(eid: str, number: int, project: Proj, context: Ctx) -> dict[str, Any]:
    """``run`` (the record of ``run.json`` with the addresses of its artifacts) and
    ``log``: the end of the log, with the offset to continue from."""
    page = _experiment(project, eid)
    run = _find_run(context, project, str(page.id), number)
    tail = _read_log(_run_dir(project, str(page.id), number) / "log.txt", -_LOG_TAIL_BYTES)
    tail["text"] = context.services.masked(project, tail["text"])
    return {"run": _run_view(context, project, str(page.id), run), "log": tail}


@router.get("/projects/{pid}/experiments/{eid}/runs/{number}/log")
def get_log(eid: str, number: int, project: Proj, context: Ctx, offset: int = 0) -> dict[str, Any]:
    """The run's log from byte ``offset``: ``{"offset", "next", "size", "text", "eof",
    "reset"}``.  Ask again with ``offset=next`` to follow it; ``reset`` says the log
    was started anew and ``text`` is from its beginning.  A negative ``offset`` reads
    that many bytes from the end.  Credentials a job printed are blanked out in
    ``text``; the offsets are those of the file."""
    page = _experiment(project, eid)
    if number < 1:
        raise ApiError(404, "not_found", f"{page.id} has no run {number}.")
    piece = _read_log(_run_dir(project, str(page.id), number) / "log.txt", offset)
    return {**piece, "text": context.services.masked(project, piece["text"])}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

_SHOWN_AS_IS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
}
#: Shown as plain text whatever they are (a page of HTML too): nothing a project holds is
#: ever run by the browser.
_SHOWN_AS_TEXT = frozenset(
    {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".log", ".py", ".ipynb"}
    | {".tex", ".bib", ".bst", ".sty", ".cls", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
    | {".sh", ".diff", ".patch", ".html", ".htm", ".xml", ".js", ".css", ".r", ".jl"}
)


@router.get("/projects/{pid}/files/{path:path}")
def get_file(path: str, project: Proj, download: str | None = None) -> FileResponse:
    """A file of the project under ``inputs/``, ``experiments/`` or ``paper/``.

    The path is resolved inside the project, symbolic links included.  Images and
    PDFs are shown; text of any kind is served as plain text; everything else is a
    download (as is anything with ``?download=1``).
    """
    try:
        real = confined(project.root, path, areas=FILE_AREAS)
    except PathEscape as exc:
        raise ApiError(404, "not_found", f"No such file: {exc}.") from exc
    if not real.is_file():
        raise ApiError(404, "not_found", f"There is no file {path!r} in this project.")
    suffix = real.suffix.lower()
    media_type = _SHOWN_AS_IS.get(suffix)
    attach = _flag(download)
    if media_type is None:
        media_type = "text/plain; charset=utf-8" if suffix in _SHOWN_AS_TEXT else None
    if media_type is None:
        media_type, attach = "application/octet-stream", True
    headers = {
        "Cache-Control": "no-cache",
        "Content-Security-Policy": FILE_CONTENT_SECURITY_POLICY,
    }
    if suffix == ".pdf":
        headers["Content-Security-Policy"] = PDF_CONTENT_SECURITY_POLICY
        headers["X-Frame-Options"] = "SAMEORIGIN"
    return FileResponse(
        real,
        media_type=media_type,
        filename=real.name,
        content_disposition_type="attachment" if attach else "inline",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# The optional paper
# ---------------------------------------------------------------------------


class PaperBody(_Body):
    template: str
    model: str | None = None


def _paper_view(context: Context, project: Project) -> dict[str, Any]:
    paper = project.root / PAPER_DIR
    state = _read_json(paper / STATE_FILE)
    worker = context.workers.status(project.root)
    final = paper / "final_paper.pdf"
    return {
        "exists": paper.is_dir(),
        "state": state if isinstance(state, dict) else None,
        "running": worker["state"] != "idle" and worker["kind"] == "write",
        "final_pdf": f"{PAPER_DIR}/final_paper.pdf" if final.is_file() else None,
        # What a write-up would refuse for ("results", "finding"), or None: it can start.
        "missing": context.services.paper_missing(project),
        "worker": worker,
    }


@router.post("/projects/{pid}/paper", status_code=202)
def start_paper(body: PaperBody, project: Proj, context: Ctx) -> dict[str, Any]:
    """Start the write-up of the wiki as a paper with the named bundled ``template``.
    409 when the project already has a worker."""
    known = context.services.templates()
    if body.template not in known:
        raise ApiError(
            422,
            "invalid",
            f"'{body.template}' is not a bundled template; choose one of: {', '.join(known)}.",
        )
    worker = context.workers.start(
        project.root, "write", template=body.template, model=(body.model or "").strip() or None
    )
    return {"worker": worker}


@router.get("/projects/{pid}/paper")
def get_paper(project: Proj, context: Ctx) -> dict[str, Any]:
    """The paper run: ``state`` is ``paper/state.json`` (``null`` before the first run),
    ``final_pdf`` the project-relative path of the finished paper for the files
    endpoint, ``missing`` what the project lacks for a paper (``"results"``,
    ``"finding"``) or ``null`` when it can be written."""
    return _paper_view(context, project)
