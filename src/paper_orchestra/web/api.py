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
  as markup;
* no answer carries a credential: every string that leaves the API (also in the
  event stream and in a piece of a log) has the values of the server's
  credentials and the token in the URL of the project's repository replaced by
  ``[redacted]``, whatever file of the project it was read from;
* text that cannot be stored (half of a surrogate pair) is refused with 422
  wherever a request carries it, and so is a number that is none (``NaN``, an
  infinity); where a project's file holds such text the answer shows U+FFFD in
  its place;
* nothing of a project is read or written through a symbolic link that leads
  out of its directory: a project whose ``project.json``, wiki, proposals,
  requests or inputs are such a link is listed as one that cannot be opened (the
  store refuses it, also when the link appears while the server runs), and an
  experiment's code, a log and an earlier version of a page are refused or left
  out one by one.

The engine, experiment execution and the paper write-up live in other modules
of the package and run in worker processes; what the API needs from them is
reached through :class:`Services`, which tests replace.

What a client would otherwise guess or work out a second time, the API says:

* where a project stands is the engine's reading of it (``state``, from
  ``state.json`` or, when that file does not say, rebuilt from the project's
  events), with what is wrong with the file in ``state_problem`` and where the
  reading came from in ``state_source``; the project view, the library row and
  the event stream say the same;
* what a press of Continue does is ``next_step`` of the project view, the
  engine's own answer; the analyst's recommendation of more research is
  ``recommends`` on the checkpoint, for as long as its journal page stands;
* where no model can be used, ``next_step.blocked`` says so before anything is
  pressed (which phase is not started, and the one sentence about what to set
  up), and a worker that ended for that reason is told from one that died by
  ``worker.last_exit.reason`` (``"needs_model"``), not by what it printed;
* an experiment that reads approved and will not run as it stands (it changed
  since the approval) carries ``will_not_run`` wherever it is listed and is
  named in ``next_step.not_running``; one that reads cancelled or failed
  although an earlier run of it succeeded carries ``last_success``;
* what the phases did before a checkpoint is ``checkpoint.did``, apart from the
  count of what waited then, from what was next and from a model the next phase
  lacked (all three are in ``summary``, as the checkpoint said them, and are
  shown live instead: the last one is ``next_step.blocked`` while it is true);
* what cancelling an experiment does is ``cancel_does`` (take the approval back,
  or cancel the run), and the answer to the cancel says which happened;
* an experiment whose run was interrupted is marked wherever its page is listed,
  what rests on a rejected experiment comes with its view, a suggested edit to a
  finding is listed with its number check, and how the last write-up went is
  read from the project's events.

Looking at a project writes nothing to it and takes none of its locks.  (The
one exception is older than this list: a look that finds the worker of a
phase gone records that phase as interrupted; see ``workers``.)

Four endpoints go beyond the list of the specification: ``POST
.../experiments/{eid}/run`` (run an approved experiment now), ``GET
.../worker/log`` (what a worker wrote before it stopped), ``POST
.../worker/forget`` (clear the name of a worker this server cannot verify) and
``GET /api/defaults`` (the settings a new project starts with).
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import dataclasses
import functools
import importlib
import inspect
import json
import logging
import math
import os
import re
import secrets
import shutil
import stat
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, BinaryIO
from urllib.parse import quote, unquote

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from paper_orchestra import __version__
from paper_orchestra.memory.wiki import slugify, split_front_matter
from paper_orchestra.research.model import (
    COMPUTE_BACKENDS,
    EFFORTS,
    FOR_ALL,
    HUMAN,
    KINDS,
    RECORD_KINDS,
    REQUEST_ROLES,
    ROLE_MAY_CREATE,
    ROLES,
    SCHEMA,
    SECTIONS,
    Conflict,
    EditProposal,
    NotAllowed,
    NotFound,
    ProjectError,
    ProjectSettings,
    ResearchPage,
    format_stamp,
    is_agent,
    normalize_author,
    page_from_parts,
    parse_id,
    phase_name,
    plain_links,
    role_of,
    settings_for_new_project,
)
from paper_orchestra.research.project import (
    BRIEF_FILE,
    EVENTS_FILE,
    INPUTS_DIR,
    MANIFEST_FILE,
    NOTES_DIR,
    PAPER_DIR,
    PROJECT_FILE,
    PROPOSALS_DIR,
    REQUESTS_DIR,
    STATE_FILE,
    TRANSCRIPTS_DIR,
    WIKI_DIR,
    Project,
)
from paper_orchestra.research.repo import display_url, userinfo
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
    open_confined,
    safe_upload_path,
    security_headers,
)
from paper_orchestra.web.workers import (
    RECORD_FILE,
    WORKER_LOG,
    WorkerBusy,
    WorkerError,
    Workers,
    lock_held,
    read_state_file,
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

#: Whom a request of the person can be for: every phase until the round's analysis has
#: completed, or the next phase of one role.
REQUEST_FOR: tuple[str, ...] = (FOR_ALL, *REQUEST_ROLES)

_MAX_INSTRUCTION_CHARS = 20_000
_MAX_RENDER_CHARS = 1_000_000
_MAX_REVIEW_ITEMS = 500
_LOG_CHUNK_BYTES = 256 * 1024
_LOG_TAIL_BYTES = 16 * 1024
_STAGING_DIR = ".incoming"
#: What stands in an answer where a credential stood (the word ``doctor`` uses).
_REDACTED = "[redacted]"
#: A shorter end of a log is not held back for looking like the start of a credential:
#: a few characters of one give nothing away, and a log may end in any letter.
_MIN_HELD_BACK_BYTES = 4
#: A shorter part of a URL is not blanked as its token: it would blank ordinary words.
_MIN_URL_SECRET_CHARS = 4
#: Seconds one listing of the library spends counting what waits in projects whose
#: pages changed; the projects it does not get to are counted by the next listing.
_COUNT_SECONDS = 0.3
#: Seconds a Run waits for the runs a dead worker left to be settled before it decides.
_SETTLE_SECONDS = 20.0
#: Seconds for which the server remembers that no client could be made for a model (no
#: credential) before it tries again: see ``Services.client_for``.
_REFUSAL_SECONDS = 20.0
#: Seconds one listing of the edit proposals spends on the number checks of suggested
#: edits to findings; an edit it does not get to is listed without a report.
_ROW_CHECK_SECONDS = 0.5
#: The most of a project's problems one answer lists; ``problems_total`` says how many
#: there are.
_MAX_PROBLEMS = 50
#: The ledger of what the models cost, which the engine reads when it rebuilds a state.
_USAGE_FILE = "usage.json"
#: What the engine opens by name to say where a project stands.  The server asks it only
#: where each of these is an ordinary file of the project itself (:func:`_from_elsewhere`).
_READ_BY_THE_ENGINE = (STATE_FILE, EVENTS_FILE, _USAGE_FILE)
#: A record of a project (usage, the paper's state, a manifest) is far smaller.
_MAX_JSON_BYTES = 32 * 1024 * 1024
#: The kinds of page a person writes: all but the idea (there is one, made with the
#: project) and the records, which code writes from what a run or a service returned.
_PERSON_KINDS: tuple[str, ...] = tuple(
    kind for kind in KINDS if kind != "idea" and kind not in RECORD_KINDS
)
#: How many decided edit proposals a page's view lists, the latest first.
_MAX_DECIDED_EDITS = 20
#: Half of a surrogate pair: valid in a JSON file, impossible to send or store as UTF-8.
_UNSENDABLE = re.compile("[\ud800-\udfff]")


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


def _error(
    status: int, code: str, message: str, hidden: Sequence[str] | None = None, **extra: Any
) -> JSONResponse:
    """A refusal as the API promises it.  What a message quotes, and what a refusal
    carries next to it (the page as it is now, for a stale edit), comes from the files
    of a project as much as an answer does: it is made presentable the same way.
    ``hidden`` is what it may not carry (see :func:`_hidden_from`); without it, the
    server's own credentials."""
    from paper_orchestra import doctor

    body = {"error": {"code": code, "message": message}, **extra}
    if hidden is None:
        hidden = doctor.secret_values(os.environ)
    return JSONResponse(_presentable(body, hidden), status_code=status)


def _hidden_from(request: Request) -> Sequence[str] | None:
    """The credentials the answer to ``request`` may not carry: the server's and, when
    the request is about a project, the token of that project's repository (the
    project is noted on the request where it is looked up).  ``None`` when that cannot
    be told; a refusal then blanks the server's own."""
    context = getattr(request.app.state, "context", None)
    project = getattr(request.state, "project", None)
    try:
        return context.services.secrets(project if isinstance(project, Project) else None)
    except Exception:  # a refusal is sent whatever else is wrong
        return None


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
        return _error(exc.status, exc.code, exc.message, _hidden_from(request), **exc.extra)

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
        return _error(status, code, str(exc), _hidden_from(request), **extra)

    async def worker_error(request: Request, exc: WorkerError) -> JSONResponse:
        status, code = _classify(exc)
        extra = {"worker": exc.status, "phase": exc.phase} if isinstance(exc, WorkerBusy) else {}
        return _error(status, code, str(exc), _hidden_from(request), **extra)

    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = []
        for problem in exc.errors():
            where = ".".join(str(part) for part in problem.get("loc", ())[1:])
            problems.append(f"{where}: {problem.get('msg')}" if where else str(problem.get("msg")))
        message = "The request is not what this endpoint takes: " + "; ".join(problems)
        return _error(422, "invalid_request", message, _hidden_from(request))

    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "error")
        return _error(exc.status_code, code, str(exc.detail), _hidden_from(request))

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
# What may leave the server, and what may come in
# ---------------------------------------------------------------------------


def _presentable(value: Any, secrets: Sequence[str] = ()) -> Any:
    """``value`` (JSON data) as it may leave the server.

    Every string in it, keys included, has each of ``secrets`` replaced by
    ``[redacted]`` and half surrogate pairs replaced by U+FFFD.  The files of a
    project are written by workers, by agents and by hand: a provider's error
    that quotes a key ends up in ``state.json`` and in an event, and a JSON file
    can hold text that cannot be sent at all, which would otherwise cost the
    whole answer (and, in the library, the answer for every project).
    """
    if isinstance(value, str):
        for secret in secrets:
            if secret in value:
                value = value.replace(secret, _REDACTED)
        return value if value.isascii() else _UNSENDABLE.sub("\ufffd", value)
    if isinstance(value, Mapping):
        return {
            (_presentable(key, secrets) if isinstance(key, str) else key): _presentable(
                item, secrets
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [_presentable(item, secrets) for item in value]
    return value


def _unstorable(value: Any, where: str = "") -> str | None:
    """Why ``value`` (JSON data) cannot be written to a file and read back as it is, as
    a sentence that names the place by its dotted path, or ``None`` when all of it can.

    Two things pass the reading of a request body and are none a file can hold: a
    string with half of a surrogate pair, and a number that is none (``NaN``, an
    infinity, a literal too large to be anything else), which would be written as a
    word no JSON reader but this one accepts, and as a limit would limit nothing.
    """
    if isinstance(value, str):
        if value.isascii() or not _UNSENDABLE.search(value):
            return None
        return _cannot_be_stored(where or "The text")
    if isinstance(value, float):
        return None if math.isfinite(value) else _is_no_number(where or "A value", value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key) if isinstance(key, str) and key.isascii() else "a name"
            found = _unstorable(key, f"{where}.{name}" if where else name) or _unstorable(
                item, f"{where}.{name}" if where else name
            )
            if found:
                return found
    elif isinstance(value, list | tuple):
        for number, item in enumerate(value):
            found = _unstorable(item, f"{where}[{number}]")
            if found:
                return found
    return None


def _cannot_be_stored(where: str) -> str:
    return (
        f"{where} holds half of a character pair (a lone surrogate), which cannot be stored "
        "in a file. That is usually what is left of an emoji that was cut in two: remove "
        "it and send the request again. Nothing was changed."
    )


def _is_no_number(where: str, value: float) -> str:
    return (
        f"{where} is {value} (not a finite number), which cannot be stored in a file. Send "
        "a number, or leave the field out to keep what is there. Nothing was changed."
    )


def _written_forms(secrets: Sequence[str]) -> list[bytes]:
    """Each of ``secrets`` as it stands in a file, as bytes, the longest first: as it
    is, and escaped the way JSON writes it (with and without its escapes for letters
    beyond ASCII).  A record of a run is a JSON file and a job may print JSON into
    its log: a password with a quote, a backslash or an umlaut stands there escaped."""
    forms: set[bytes] = set()
    for secret in secrets:
        if not secret:
            continue
        for text in (
            secret,
            json.dumps(secret)[1:-1],
            json.dumps(secret, ensure_ascii=False)[1:-1],
        ):
            try:
                # As the environment gave it: bytes that are no UTF-8 come back as they were.
                forms.add(text.encode("utf-8", "surrogateescape"))
            except UnicodeEncodeError:
                # Half a surrogate pair from a file written by hand: no file holds it as
                # UTF-8, and it must not cost the answer.
                forms.add(text.encode("utf-8", "surrogatepass"))
    return sorted(forms, key=lambda form: (-len(form), form))


def _url_secrets(url: Any) -> set[str]:
    """The token or password in the URL of a repository, as written and as decoded (git
    prints either form)."""
    secret = userinfo(url) if isinstance(url, str) else None
    found = {secret, unquote(secret)} if secret else set()
    return {text for text in found if len(text) >= _MIN_URL_SECRET_CHARS}


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

    def _later(self, name: str) -> Callable[..., Any]:
        """A function the experiments module gained after the API was first written.  An
        installation whose module has none is told so, like one without the module."""
        function = getattr(self._experiments(), name, None)
        if not callable(function):
            raise ApiError(
                400,
                "not_available",
                f"This installation's experiments module has no '{name}', which this "
                "needs. Reinstall the package.",
            )
        return function

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

    def will_not_run(self, project: Project, experiment_id: str) -> str | None:
        """Why a run of the experiment would not start although its page reads approved
        (``experiments.will_not_run``): it is no longer what was approved, or the run it
        was approved for is over.  The rest of a sentence about the experiment; ``None``
        when a run would start, and where that cannot be told here (the run says it)."""
        try:
            reason = self._later("will_not_run")(project, experiment_id)
        except (ApiError, ProjectError, OSError):
            return None
        return str(reason) if reason else None

    def last_success(self, project: Project, experiment_id: str) -> dict[str, Any] | None:
        """The latest run of the experiment that succeeded, ``{"number", "result_page"}``
        (``experiments.last_success``); ``None`` when none did or it cannot be told."""
        try:
            found = self._later("last_success")(project, experiment_id)
        except (ApiError, ProjectError, OSError):
            return None
        return dict(found) if found else None

    def secrets(
        self,
        project: Project | None = None,
        *,
        server: Sequence[str] | None = None,
        pages: bool = True,
    ) -> list[str]:
        """The strings no answer may carry, longest first: the values of the server's
        credentials (the rule of ``paper-orchestra doctor``; workers inherit them, and
        a library may quote one in an error) and the token or password in the URL of
        the project's repository, which a job that clones it is handed.  An
        experiment may name a repository of its own on its page: the token of that
        address counts as well.

        ``server`` is what a call without a project returned, for an answer about
        many projects: the environment is then gone through once, not once for each.
        Without ``pages`` no page is read (the library reads none), and only the
        repository of the project's settings is looked at.
        """
        from paper_orchestra import doctor

        found = set(doctor.secret_values(os.environ) if server is None else server)
        repo = project.settings.repo if project is not None else None
        found |= _url_secrets(repo.get("path") if isinstance(repo, Mapping) else repo)
        if project is not None and pages:
            with contextlib.suppress(ProjectError, OSError):
                for page in project.pages(kind="experiment"):
                    found |= _url_secrets(page.meta.get("repo"))
        return sorted(found, key=len, reverse=True)

    def masked(self, project: Project, text: str) -> str:
        """``text`` without the credentials it may hold (see :meth:`secrets`)."""
        return str(_presentable(text, self.secrets(project)))

    def request_cancel(self, project: Project, ref: str) -> None:
        self._experiments().request_cancel(project, ref)

    def withdraw(self, project: Project, ref: str) -> ResearchPage:
        """Take back the approval of a run that has not started (``experiments.withdraw``):
        the page, ``planned`` again.  ``ProjectError`` when there is none to take back:
        a process is running the experiment, or the run it was approved for is over
        (the page is then moved to how that run ended)."""
        return self._later("withdraw")(project, ref)

    def _engine(self) -> Any:
        """The engine module, for what only it knows: how a state is read, what an
        ``advance`` would do, how a phase is named.  ``ImportError`` where it cannot be
        loaded; the callers then answer without it."""
        return importlib.import_module("paper_orchestra.research.engine")

    def engine_state(self, project: Project) -> dict[str, Any]:
        """Where the project stands as the engine reads it (``engine.read_state``): what
        ``state.json`` says, or round, phases and what comes next rebuilt from the
        project's events when the file is lost or holds no phase.  Nothing is written."""
        return dict(self._engine().read_state(project))

    def state_problem(self, project: Project) -> str | None:
        """What is wrong with ``state.json`` in a sentence (``engine.state_problem``):
        that the state is read from the events instead, or that nothing on file says
        where the project stands.  ``None`` when the file is the engine's account, or
        no phase has run."""
        problem = self._engine().state_problem(project)
        return str(problem) if problem else None

    def next_step(self, project: Project, state: Mapping[str, Any]) -> dict[str, Any]:
        """What an ``advance`` without a phase would do from ``state`` (the engine's
        reading), as ``engine.next_step`` says it: ``{"will_run", "then", "experiments",
        "not_running", "results", "checkpoint_of", "needs_phase", "blocked"}``.  Reads
        only.  Whether the models can be used is part of it (``blocked``): the engine
        asks :meth:`client_for`, as a worker of this server would find its clients."""
        return dict(self._engine().next_step(project, state, client_for=self.client_for))

    def client_for(self, model: str) -> Any:
        """The client of ``model`` as a worker this server starts would get it
        (``llm.router.get_client``), for asking whether the model can be used.
        ``LLMConfigError`` when none can be made: no credential, most of the time.

        Nothing is contacted.  A client that can be made is kept by the router; one
        that cannot is tried anew at every call, at the cost of making one, and a
        project is looked at often.  So a refusal is remembered here for
        ``_REFUSAL_SECONDS``, and what the server reads from its environment when it
        starts (a credential set since needs a restart, as for the Setup check)
        stays true for that long at least.
        """
        from paper_orchestra.llm.base import LLMConfigError
        from paper_orchestra.llm.router import get_client

        refused: dict[str, tuple[float, LLMConfigError]] = self.__dict__.setdefault(
            "_refused_clients", {}
        )
        # One look at a time: whoever asks while another look is under way gets its answer.
        with self.__dict__.setdefault("_client_lock", threading.Lock()):
            known = refused.get(model)
            if known is not None and time.monotonic() < known[0]:
                # One exception for many looks: raised without the frames of the looks
                # before, which it would otherwise gather, with all they held.
                raise known[1].with_traceback(None)
            try:
                client = get_client(model)
            except LLMConfigError as exc:
                refused[model] = (time.monotonic() + _REFUSAL_SECONDS, exc)
                raise
            refused.pop(model, None)
            return client

    def phase_words(self) -> dict[str, str]:
        """How a sentence for the person names each phase (``the hypothesis phase``), in
        the engine's words; empty where the engine cannot be asked."""
        try:
            name = self._engine().phase_in_words
        except (ImportError, AttributeError):
            return {}
        return {phase: str(name(phase)) for phase in PHASES}

    def runs(self, project: Project, experiment_id: str) -> list[dict[str, Any]]:
        return list(self._experiments().runs(project, experiment_id))

    def run_held(self, project: Project, experiment_id: str) -> bool | None:
        """Whether a process is running the experiment right now: whoever runs it holds
        its run lock, and the kernel drops that lock with the process, however it
        died.  ``None`` where that cannot be told (no such locks here)."""
        name = getattr(self._experiments(), "RUN_LOCK", ".run.lock")
        return lock_held(project.experiment_dir(experiment_id) / str(name))

    def dependents(self, project: Project, ref: str) -> Any:
        """What rests on the experiment: the findings based on its runs and the verdicts
        on hypotheses they carry, as ``experiments.dependents`` gives them."""
        return self._later("dependents")(project, ref)

    def number_report(self, project: Project, page: ResearchPage | str) -> dict[str, Any]:
        """The number check of a finding (a page, or text to check as it is) as
        ``experiments.number_report`` gives it: ``{"checked", "skipped", "elsewhere",
        "unverified", "summary"}``."""
        return dict(self._later("number_report")(project, page))

    def edited_record(self, project: Project, page: ResearchPage) -> bool:
        """Whether a record (a result or a source page) no longer reads as code wrote it.

        The experiments module decides (``edited_record``: a person edited it, or the
        text of a result differs from what its run wrote).  Where it cannot, the
        page's own stamp does: a record whose current version a person wrote.
        """
        if not page.is_record:
            return False
        try:
            return bool(self._later("edited_record")(project, page))
        except (ApiError, ProjectError, OSError):
            return normalize_author(page.meta.get("updated_by") or page.author) == HUMAN

    def backend_names(self, project: Project) -> list[str]:
        """The compute backends the approval form asks about, the project's own first.

        A form offers the first one that can run as its choice.  That is then what the
        person set for the project, never their own machine in place of a sandbox
        because of the order of a list.
        """
        from paper_orchestra.compute import BACKEND_NAMES

        own = str(project.settings.compute)
        return sorted(BACKEND_NAMES, key=lambda name: name != own)

    def options(self, project: Project, ref: str) -> list[dict[str, Any]]:
        # By name: the experiments module builds each backend itself, and one that cannot
        # be built here (Modal is not installed) comes back with that as its reason.
        return list(self._experiments().options(project, ref, self.backend_names(project)))

    def number_check(self, project: Project, page: ResearchPage) -> list[dict[str, Any]]:
        return list(self._experiments().number_check(project, page))

    def default_settings(self) -> dict[str, Any]:
        """The settings a project starts with when nothing else is said."""
        return ProjectSettings.from_dict(settings_for_new_project(None)).to_dict()

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

    def paper_left_out(self, project: Project) -> list[dict[str, Any]]:
        """The accepted findings a paper would not report, each with why: ``[{"id",
        "path", "title", "reason"}]`` as ``writeup.left_out`` gives them (every record
        such a finding is based on belongs to an experiment that was rejected or is
        gone).  Empty where the write-up cannot be asked."""
        try:
            writeup = importlib.import_module("paper_orchestra.research.writeup")
        except ImportError:
            return []
        left_out = getattr(writeup, "left_out", None)
        return [dict(item) for item in left_out(project)] if callable(left_out) else []


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


#: What a project keeps at the top of its directory as its own.  ``Project.open``
#: refuses a project in which one of these is a symbolic link that leads out of the
#: project directory (what is there would be shown, and written to, as the
#: project's); the same names here say when to ask it again (:func:`_linked_out`).
_OWN_PLACES = (
    PROJECT_FILE,
    WIKI_DIR,
    PROPOSALS_DIR,
    REQUESTS_DIR,
    INPUTS_DIR,
    NOTES_DIR,
    TRANSCRIPTS_DIR,
)


def _linked_out(root: str | Path) -> bool:
    """Whether one of the project's own places in ``root`` is a symbolic link that leads
    out of the project directory.  A handful of ``lstat`` calls where none is a link."""
    base = os.path.realpath(root)
    inside = base if base.endswith(os.sep) else base + os.sep
    for name in _OWN_PLACES:
        place = os.path.join(base, name)
        if os.path.islink(place) and not os.path.realpath(place).startswith(inside):
            return True
    return False


class Workspace:
    """Finds projects by the id in their ``project.json``.  Thread-safe.

    One :class:`Project` object is kept per directory (it caches parsed pages) and
    re-reads ``project.json`` when another process changed it.

    **An id keeps naming the directory it named.**  A project directory that is
    copied inside the workspace keeps the id in its ``project.json``, so two
    directories then claim one id.  The one that was addressed by it goes on
    being addressed by it, and the newcomer gets a derived id
    (``<id>-<directory>``), whatever the two are called: a tab that is open on a
    project, a bookmark or a request in flight never lands in its copy.  A server
    that sees both for the first time gives the id to the directory the project
    was created as (see :meth:`_seniority`).  A renamed directory keeps its id,
    also when a copy of it was put under its old name meanwhile: the directory is
    known by what the file system knows it by (:meth:`_identity`), not by what it
    is called.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._guard = threading.RLock()
        self._opened: dict[Path, tuple[tuple[int, int, int], Project]] = {}
        #: The id each directory is addressed by, and the reverse.  Both hold every
        #: project directory of the last scan, also one that cannot be opened now.
        self._keys: dict[Path, str] = {}
        self._by_id: dict[str, Path] = {}
        #: Which directory each id was given to (:meth:`_identity`): another directory
        #: under the same name is not it.
        self._held: dict[str, tuple[int, int] | None] = {}
        #: Why a directory with a ``project.json`` could not be opened.
        self._unreadable: dict[Path, str] = {}

    def _open(self, directory: Path) -> Project | None:
        """The project in ``directory``, from the cache while its file is unchanged;
        ``None`` when it holds none or it cannot be opened (:meth:`unreadable` says why)."""
        signature = _signature(directory / PROJECT_FILE)
        if signature is None:
            self._opened.pop(directory, None)
            self._unreadable.pop(directory, None)
            return None
        cached = self._opened.get(directory)
        if cached is not None and _linked_out(directory):
            # Looked at each time: a link can be put there without project.json changing.
            # The project is then opened anew, which the store refuses, saying why: it
            # is not read or written through such a link, here as on the command line.
            cached = None
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            if cached is not None:
                project = cached[1]
                project.reload()
            else:
                project = Project.open(directory)
        except (ProjectError, OSError) as exc:
            if self._unreadable.get(directory) != str(exc):  # said once, not at every look
                log.warning("%s cannot be opened: %s", directory, exc)
            self._opened.pop(directory, None)
            self._unreadable[directory] = str(exc)
            return None
        self._unreadable.pop(directory, None)
        self._opened[directory] = (signature, project)
        return project

    @staticmethod
    def _identity(directory: Path) -> tuple[int, int] | None:
        """What tells a directory from another one put under its name: its device and
        inode, which a rename keeps and a copy does not get.  ``None`` where that
        cannot be read."""
        try:
            found = os.lstat(directory)
        except OSError:
            return None
        return found.st_dev, found.st_ino

    @staticmethod
    def _seniority(project: Project) -> tuple[int, int, str]:
        """Orders the directories that claim one id, the original first.

        By what the project itself recorded when it was created
        (``Project.at_home``, which the command line goes by as well): the
        directory it was created as comes first, a known copy last.  Where that
        does not tell (an older project, a directory moved to another file
        system), by when the file a project is created with was put where it is:
        the change time of the brief (or, without one, of the manifest), which
        nothing rewrites and no copying tool can carry over, so a copy is
        younger.  Equal times fall back to the name.
        """
        home = project.at_home()
        rank = 0 if home else 1 if home is None else 2
        for name in (f"{INPUTS_DIR}/{BRIEF_FILE}", f"{INPUTS_DIR}/{MANIFEST_FILE}", PROJECT_FILE):
            with contextlib.suppress(OSError):
                return (rank, os.lstat(project.root / name).st_ctime_ns, project.root.name)
        return (rank, 0, project.root.name)

    @staticmethod
    def _fits(key: str, project: Project) -> bool:
        """Whether ``key`` is an id this project can be addressed by."""
        return key == project.id or key.startswith(f"{project.id}-")

    @staticmethod
    def _free_key(project: Project, taken: set[str]) -> str:
        """The project's own id, or one derived from it and the directory's name when
        another directory is addressed by that id already."""
        key, copy = project.id, 1
        while key in taken:
            copy += 1
            key = f"{project.id}-{slugify(project.root.name)}" + (f"-{copy}" if copy > 2 else "")
        return key

    def scan(self) -> dict[str, Project]:
        """Every project in the workspace that can be opened, by the id it is addressed
        by.  Costs a handful of ``stat`` calls per project that has not changed: no
        page is read."""
        with self._guard:
            try:
                entries = sorted(os.scandir(self.root), key=lambda entry: entry.name)
            except OSError:
                entries = []
            present: list[tuple[Path, Project | None]] = []
            identity: dict[Path, tuple[int, int] | None] = {}
            for entry in entries:
                if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
                    continue
                directory = Path(entry.path)
                project = self._open(directory)
                if project is not None or directory in self._unreadable:
                    present.append((directory, project))
                    identity[directory] = self._identity(directory)

            def keeps(held: str | None, project: Project | None) -> bool:
                # A project.json that was given another id by hand makes it a newcomer.
                return held is not None and (project is None or self._fits(held, project))

            keys: dict[Path, str] = {}
            given = {where: key for key, where in self._held.items() if where is not None}
            for directory, project in present:
                # The very directory an id was given to keeps it, whatever it is called now.
                held = given.get(identity[directory])
                if keeps(held, project):
                    keys[directory] = str(held)
            taken = set(keys.values())
            newcomers: list[Project] = []
            for directory, project in present:
                if directory in keys:
                    continue
                # Else by its name (a file system that does not keep inodes), unless the
                # directory that held the id is still there under another name: what
                # stands under the old name is then a copy of it, and a newcomer.
                held = self._keys.get(directory)
                if keeps(held, project) and held not in taken:
                    keys[directory] = str(held)
                    taken.add(str(held))
                elif project is not None:
                    newcomers.append(project)
            for project in sorted(newcomers, key=self._seniority):
                keys[project.root] = self._free_key(project, taken)
                taken.add(keys[project.root])
            self._keys = keys
            self._by_id = {key: directory for directory, key in keys.items()}
            self._held = {key: identity[directory] for directory, key in keys.items()}
            listed = {directory for directory, _ in present}
            for directory in set(self._opened) - listed:
                del self._opened[directory]
            for directory in set(self._unreadable) - listed:
                del self._unreadable[directory]
            return {keys[directory]: project for directory, project in present if project}

    def unreadable(self) -> list[dict[str, Any]]:
        """The directories of the last scan that hold a project which cannot be opened:
        ``{"dir", "id", "message"}``, by name.  ``id`` is what the project was addressed
        by while it could be opened (``None`` if this server never saw it whole);
        ``message`` is what the command line says about it."""
        with self._guard:
            return [
                {"dir": directory.name, "id": self._keys.get(directory), "message": message}
                for directory, message in sorted(self._unreadable.items())
            ]

    def project(self, pid: str) -> Project:
        """The project addressed by that id.  ``ApiError`` 404 when the workspace has
        none, 422 ``project_unreadable`` with the reason when it has one that cannot
        be opened."""
        with self._guard:
            directory = self._by_id.get(pid)
            if directory is not None and self._identity(directory) != self._held.get(pid):
                directory = None  # another directory stands under that name: look again
            project = self._open(directory) if directory is not None else None
            if project is None or self._keys.get(project.root) != pid:
                project = self.scan().get(pid)
            if project is not None:
                return project
            directory = self._by_id.get(pid)
            reason = self._unreadable.get(directory) if directory is not None else None
            if reason is not None:
                raise ApiError(422, "project_unreadable", reason)
            raise ApiError(404, "not_found", f"There is no project {pid!r} in this workspace.")

    def key_of(self, project: Project) -> str:
        """The id the project is addressed by here (its own, unless a copy shares it)."""
        with self._guard:
            return self._keys.get(project.root, project.id)

    def restore(self, project: Project) -> None:
        """Make the kept :class:`Project` say what is on disk again, after a change to
        it that could not be saved.  If even that fails it is dropped, and the next
        request opens the directory anew."""
        with self._guard:
            try:
                project.reload()
            except (ProjectError, OSError):
                self._opened.pop(project.root, None)

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


class _Counts:
    """What waits for the person in each project, for the library.  Thread-safe.

    Counting reads every page of a project, which the listing of hundreds of
    projects cannot do at every look.  So the numbers of a project are kept until
    something changes in it: its events (every change made through the project
    adds one) or a directory that holds pages or edit proposals (a file put
    there or replaced by hand); that costs a dozen ``stat`` calls and reads
    nothing.  One listing spends a bounded time on counting; a project it does
    not get to is counted by the next one.  A project is opened apart for
    counting, so that the pages of every project in the library are not kept in
    memory.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._known: dict[Path, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    @staticmethod
    def _signature(root: Path) -> tuple[Any, ...]:
        # Plain strings: this runs for every project at every look at the library.
        wiki = os.path.join(root, WIKI_DIR)
        places = [os.path.join(root, EVENTS_FILE), os.path.join(root, PROPOSALS_DIR), wiki]
        places += [os.path.join(wiki, section) for section in SECTIONS.values() if section]
        found = []
        for place in places:
            try:
                status = os.stat(place)
            except OSError:
                found.append(None)
            else:
                found.append((status.st_ino, status.st_mtime_ns, status.st_size))
        return tuple(found)

    def of(self, root: Path, deadline: float) -> dict[str, Any] | None:
        """``Project.counts()`` of the project in ``root`` as it is now, or ``None`` when
        it changed since it was last counted and ``deadline`` (``time.monotonic()``)
        has passed."""
        signature = self._signature(root)
        with self._guard:
            known = self._known.get(root)
        if known is not None and known[0] == signature:
            return known[1]
        if time.monotonic() >= deadline:
            return None
        counts = Project.open(root).counts()
        with self._guard:
            self._known[root] = (signature, counts)
        return counts

    def keep(self, roots: set[Path]) -> None:
        """Forget the projects that are no longer in the workspace."""
        with self._guard:
            for root in set(self._known) - roots:
                del self._known[root]


@dataclass(frozen=True)
class _Standing:
    """Where a project stands, as an answer says it."""

    #: The engine's state.  Shared between answers: read, never changed.
    state: dict[str, Any]
    #: What is wrong with ``state.json``, in a sentence; ``None`` when nothing is.
    problem: str | None
    #: ``file`` (state.json is the engine's account), ``events`` (it is not, and the
    #: state was rebuilt from events.jsonl), ``not_started`` (no phase has run) or
    #: ``unknown`` (nothing that can be read says where the project stands).
    source: str
    #: False where the engine could not be asked: what an ``advance`` would do is then
    #: not foretold.
    engine: bool = True
    #: The file that kept the engine from being asked: something that is not an
    #: ordinary file stands in its place (``problem`` says so).
    odd: str | None = None


class _Standings:
    """Where each project stands, as the engine last read it.  Thread-safe.

    A project whose ``state.json`` holds its phases is read from that file at
    every look, as it always was; what is kept is the engine's reading of it,
    for as long as the file holds the same bytes (:meth:`of_file`), so that a
    look costs one reading of the file and no second one.

    Where the file does not say, the engine reads the project's events from the
    first to the last (and, where they do not tell either, looks at who wrote
    the pages), which neither the listing of hundreds of projects nor every look
    at one project can do.  That answer is kept until something changes that it
    rests on (:meth:`of`): the state file, the events (every change made through
    the project adds one), the usage ledger, a directory that holds pages, or
    whether one of those three files is the project's own (a link in its place is
    not read through).  Finding that out costs some fifteen ``stat`` calls and
    reads nothing.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._known: dict[Path, tuple[tuple[Any, ...], _Standing]] = {}
        self._files: dict[Path, tuple[bytes, _Standing]] = {}
        #: What the log was told about a project already (see :meth:`first_complaint`).
        self._complained: set[tuple[Path, str]] = set()

    def first_complaint(self, root: Path, about: str) -> bool:
        """Whether this is the first time ``about`` is to be logged for the project: a
        failure that every look at a project meets is said once."""
        with self._guard:
            known = (root, about) in self._complained
            self._complained.add((root, about))
        return not known

    def of_file(self, root: Path, content: bytes, read: Callable[[], _Standing]) -> _Standing:
        """What ``read`` made of the project's ``state.json`` when the file held
        ``content``; ``read`` is called when it holds something else now."""
        with self._guard:
            known = self._files.get(root)
        if known is not None and known[0] == content:
            return known[1]
        standing = read()
        with self._guard:
            self._files[root] = (content, standing)
        return standing

    @staticmethod
    def _signature(root: Path) -> tuple[Any, ...]:
        files = (_signature(root / STATE_FILE), _signature(root / _USAGE_FILE))
        # A link and the file it leads to look alike to ``stat``; the reading differs
        # (the engine is not asked through a link), so which of the two is there counts.
        own = tuple(_not_own_file(root, name) for name in _READ_BY_THE_ENGINE)
        return (*files, *own, *_Counts._signature(root))

    def of(self, root: Path, read: Callable[[], _Standing]) -> _Standing:
        """What ``read`` returned when the project in ``root`` last changed; ``read`` is
        called (outside the lock: it may take a while) when it changed since."""
        signature = self._signature(root)  # before the reading: a change during it is seen
        with self._guard:
            known = self._known.get(root)
        if known is not None and known[0] == signature:
            return known[1]
        standing = read()
        with self._guard:
            self._known[root] = (signature, standing)
        return standing

    def keep(self, roots: set[Path]) -> None:
        """Forget the projects that are no longer in the workspace."""
        with self._guard:
            for kept in (self._known, self._files):
                for root in set(kept) - roots:
                    del kept[root]
            self._complained = {said for said in self._complained if said[0] in roots}


@dataclass
class _Trail:
    """How far a project's events were read for its write-ups, and what they said."""

    guard: threading.Lock = field(default_factory=threading.Lock)
    inode: int | None = None
    offset: int = 0
    started: dict[str, Any] | None = None
    finished: dict[str, Any] | None = None


class _PaperEvents:
    """The last write-up of each project as its ``paper.`` events tell it.  Thread-safe.

    ``events.jsonl`` only grows, so it is read once from its beginning and then
    only as far as it grew: a look at the paper costs one ``stat`` when nothing
    happened.  What is kept per project is the last ``paper.started`` and the
    ``paper.finished`` that belongs to it, cut down to what an answer shows.  A
    stream that was replaced or cut short is read anew; one that is not the
    project's own file (a link) is not read at all.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._trails: dict[Path, _Trail] = {}

    def last(self, root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """``(started, finished)`` of the project's last write-up: the events that say
        how it began and how it ended, either of them ``None`` when there is none."""
        with self._guard:
            trail = self._trails.setdefault(root, _Trail())
        with trail.guard:
            self._catch_up(trail, root / EVENTS_FILE)
            return trail.started, trail.finished

    def keep(self, roots: set[Path]) -> None:
        """Forget the projects that are no longer in the workspace."""
        with self._guard:
            for root in set(self._trails) - roots:
                del self._trails[root]

    @staticmethod
    def _catch_up(trail: _Trail, path: Path, piece: int = 1 << 20) -> None:
        try:
            found = os.lstat(path)
        except OSError:
            found = None
        if found is None or found.st_ino != trail.inode or found.st_size < trail.offset:
            # Another file under that name, a shorter one, or none: what was read counts
            # for nothing.
            trail.inode = found.st_ino if found else None
            trail.offset, trail.started, trail.finished = 0, None, None
        if found is None or found.st_size == trail.offset:
            return
        try:
            handle = _open_plain(path)
        except OSError:
            return  # a link, a pipe: not the project's events
        with handle:
            handle.seek(trail.offset)
            rest = b""
            while data := handle.read(piece):
                *lines, rest = (rest + data).split(b"\n")
                for raw in lines:
                    trail.offset += len(raw) + 1
                    if b'"paper.' in raw:
                        _PaperEvents._take(trail, raw)
            # (``rest`` is a line still being written: it is read when it is whole.)

    @staticmethod
    def _take(trail: _Trail, raw: bytes) -> None:
        try:
            event = json.loads(raw)
        except (ValueError, RecursionError):
            return
        kind = event.get("type") if isinstance(event, dict) else None
        if kind not in ("paper.started", "paper.finished"):
            return
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        said = {"at": event.get("ts"), "summary": event.get("summary"), "data": data}
        if kind == "paper.started":
            said["data"] = {"template": data.get("template")}  # the rest can be long
            trail.started, trail.finished = said, None
        elif trail.started is not None and trail.finished is None:
            trail.finished = said
        else:
            # An end without a beginning of its own: the write-up was refused, or its
            # worker died, before it said that it started.
            trail.started, trail.finished = None, said


@dataclass
class Context:
    """Everything an endpoint works with; one per app, at ``app.state.context``."""

    workspace: Workspace
    workers: Workers
    services: Services = field(default_factory=Services)
    live: LiveSettings = field(default_factory=LiveSettings)
    uploads: UploadLimits = field(default_factory=UploadLimits)
    counts: _Counts = field(default_factory=_Counts)
    standings: _Standings = field(default_factory=_Standings)
    paper: _PaperEvents = field(default_factory=_PaperEvents)


def _context(request: Request) -> Context:
    return request.app.state.context


Ctx = Annotated[Context, Depends(_context)]


def _project(pid: str, request: Request, context: Ctx) -> Project:
    project = context.workspace.project(pid)
    # A refusal is blanked of this project's credentials like an answer (_hidden_from).
    request.state.project = project
    return project


Proj = Annotated[Project, Depends(_project)]


def _presenting(endpoint: Callable[..., Any]) -> Callable[..., Any]:
    """``endpoint`` with its answer passed through :func:`_presentable`.

    The credentials to blank are those of the server and, where the endpoint
    works on a project (its ``project`` argument), of that project's repository.
    An answer that is a response of its own (a file, the event stream) is left
    alone: those are made presentable where they are built.
    """
    if "context" not in inspect.signature(endpoint).parameters:
        raise TypeError(
            f"{endpoint.__name__} must take the context (context: Ctx): what it answers is "
            "blanked of credentials with it."
        )

    def present(answer: Any, arguments: Mapping[str, Any]) -> Any:
        if isinstance(answer, Response):
            return answer
        project = arguments.get("project")
        services: Services = arguments["context"].services
        hidden = services.secrets(project if isinstance(project, Project) else None)
        return _presentable(answer, hidden)

    if inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        async def presenting(*args: Any, **arguments: Any) -> Any:
            return present(await endpoint(*args, **arguments), arguments)

    else:

        @functools.wraps(endpoint)
        def presenting(*args: Any, **arguments: Any) -> Any:
            return present(endpoint(*args, **arguments), arguments)

    return presenting


class _Router(APIRouter):
    """A router whose every endpoint answers through :func:`_presenting`: an endpoint
    added later cannot forget to blank credentials."""

    def add_api_route(self, path: str, endpoint: Callable[..., Any], **options: Any) -> None:
        super().add_api_route(path, _presenting(endpoint), **options)


router = _Router(prefix="/api")


class _Body(BaseModel):
    """A JSON request body.  Unknown fields are ignored; what cannot be stored (see
    :func:`_unstorable`) is refused before any endpoint sees it."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _storable(cls, data: Any) -> Any:
        problem = _unstorable(data)
        if problem is not None:
            raise PydanticCustomError("unstorable", "{problem}", {"problem": problem})
        return data


def _open_plain(path: Path) -> BinaryIO:
    """Open a file that lies directly in a project directory for reading, without
    following a link put in its place.  ``OSError`` when there is none, or when what
    is there is not an ordinary file (a pipe would be waited for without end)."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags | getattr(os, "O_BINARY", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"{path} is not an ordinary file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_json(root: Path, relative: str) -> Any:
    """What the JSON file ``relative`` of the project in ``root`` holds.  ``None`` when it
    is absent, not JSON, larger than any record of a project gets, or not an ordinary
    file inside the project: a link that leads out of it is not followed, since what
    the file holds is shown in the app."""
    try:
        with open_confined(root, relative) as handle:
            if os.fstat(handle.fileno()).st_size > _MAX_JSON_BYTES:
                return None
            return json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError, RecursionError):  # PathEscape is a ValueError
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
    """That the server is there, and what it is.  ``stops_workers_on_exit`` says what
    its end does to the agents and experiments it runs: ``false`` (the default) leaves
    them running, to be found again by the next server; ``true`` stops them
    (``paper-orchestra ui --stop-workers-on-exit``)."""
    return {
        "ok": True,
        "name": "PaperOrchestra",
        "version": __version__,
        "workspace": str(context.workspace.root),
        "stops_workers_on_exit": bool(context.workers.stop_on_exit),
    }


@router.get("/doctor")
def doctor(context: Ctx, model: str | None = None) -> dict[str, Any]:
    """What ``paper-orchestra doctor --json`` reports: where credentials come from and
    which backends are usable, never a value."""
    return context.services.doctor(model)


@router.get("/templates")
def templates(context: Ctx) -> dict[str, Any]:
    return {"templates": [{"name": name} for name in context.services.templates()]}


@router.get("/defaults")
def defaults(context: Ctx) -> dict[str, Any]:
    """What a new project starts with when the composer says nothing else:
    ``settings`` (the object ``POST /api/projects`` merges its ``settings`` into),
    the engine's ``phases`` in order, and ``request_for``: whom a request can be
    addressed to (``POST .../requests``, field ``for``).

    Also the values a form may offer, which are the ones the settings accept:
    ``efforts`` (``settings.models.effort``) and ``compute_backends``
    (``settings.compute``).  ``phase_words`` is how a sentence for the person names
    each phase (``{"hypothesize": "the hypothesis phase"}``), as the engine's own
    sentences and the server's refusals name it, and ``phase_names`` what a phase
    is called by itself (``{"hypothesize": "hypotheses"}``: "checkpoint after
    hypotheses"), as the command line says it too.

    ``creatable_kinds`` are the kinds of page the person can write (``kind`` of
    ``POST .../pages``), and ``kind_writers`` says for each kind which agents write
    it as well: none for a lesson, which is the person's alone."""
    return {
        "settings": context.services.default_settings(),
        "phases": list(PHASES),
        "request_for": list(REQUEST_FOR),
        "efforts": list(EFFORTS),
        "compute_backends": list(COMPUTE_BACKENDS),
        "phase_words": context.services.phase_words(),
        "phase_names": {phase: phase_name(phase) for phase in PHASES},
        "creatable_kinds": list(_PERSON_KINDS),
        "kind_writers": {
            kind: [role for role in ROLES if kind in ROLE_MAY_CREATE.get(role, ())]
            for kind in _PERSON_KINDS
        },
    }


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
    text = _blank(body.markdown, context.services.secrets(project))
    return {"html": render_markdown(text, project=project, pid=pid, page_path=body.path)}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

_NOTHING_SPENT = {"model_usd": 0.0, "experiment_seconds": 0.0, "experiment_usd": 0.0}


def _engine_state(root: Path) -> dict[str, Any]:
    """``state.json`` as the server itself reads it (never through a link), with every
    key of spec 5.2 present.  What an answer carries is :func:`_standing`: the engine's
    reading, which this is the first step of."""
    return _state_on_file(root)[0]


def _state_on_file(root: Path) -> tuple[dict[str, Any], bytes | None]:
    """:func:`_engine_state`, and the bytes of the file it was read from (``None`` when
    the project has no such file of its own)."""
    written, content = read_state_file(root)
    spent = written.get("spent")
    state = {
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
    return state, content


def _phase_records(state: Mapping[str, Any]) -> bool:
    """Whether the state holds a record of a phase, by the engine's own measure (an
    object that names a phase and a round).  ``state.json`` is then the engine's
    account, and nothing else is read to say where the project stands."""
    phases = state.get("phases")
    for record in phases if isinstance(phases, list) else ():
        round = record.get("round") if isinstance(record, dict) else None
        whole = isinstance(round, int) and not isinstance(round, bool) and round >= 0
        if whole and record.get("phase") in PHASES:
            return True
    return False


def _not_own_file(root: Path, name: str) -> bool:
    """Whether something stands under ``name`` in the project directory that is not an
    ordinary file: a link, a pipe, a directory.  One ``lstat``."""
    try:
        found = os.lstat(root / name)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return not stat.S_ISREG(found.st_mode)


_NOT_A_FILE = (
    "{name} of this project is not an ordinary file (a link, or something else in its "
    "place), so where the project stands is not read here: the server opens a project's "
    "records only as files of the project itself. Remove or rename it and look again; "
    "the agents write their state anew when they next work."
)


def _from_file(services: Services, project: Project, written: dict[str, Any]) -> _Standing:
    """Where a project stands whose ``state.json`` holds its phases (``written``, as the
    server read it): the file as the engine reads it, and nothing more."""
    try:
        return _Standing(services.engine_state(project), None, "file")
    except ImportError:  # an installation without the engine shows the file as it is
        return _Standing(written, None, "file", engine=False)


def _from_elsewhere(services: Services, project: Project, written: dict[str, Any]) -> _Standing:
    """Where a project stands whose ``state.json`` holds no phase (``written``, as the
    server read it): rebuilt from its events by the engine, not started, or unknown.

    The engine opens the files it reads by their names.  The server opens nothing
    of a project through a link that leads out of it and waits for no pipe, so
    the engine is asked only where those files are the project's own.
    """
    root = project.root
    odd = next((name for name in _READ_BY_THE_ENGINE if _not_own_file(root, name)), None)
    if odd is not None:
        problem = _NOT_A_FILE.format(name=odd)
        return _Standing(written, problem, "unknown", engine=False, odd=odd)
    try:
        state, problem = services.engine_state(project), services.state_problem(project)
    except ImportError:
        return _Standing(written, None, "not_started", engine=False)
    if state["phases"]:
        return _Standing(state, problem, "events" if problem else "file")
    return _Standing(state, problem, "unknown" if problem else "not_started")


def _standing(context: Context, project: Project, *, apart: bool = False) -> _Standing:
    """Where the project stands, as the engine reads it (see :class:`_Standing`).

    Costs one reading of ``state.json`` for a project whose file holds its phases
    (the engine reads it a second time only when it changed), and a few ``stat``
    calls for any other once it has been read (:class:`_Standings`).  With
    ``apart`` the project is opened anew where its pages have to be looked at, so
    that the listing of a library keeps no pages in memory.  Nothing is written
    and no lock is taken.
    """
    root = project.root
    written, content = _state_on_file(root)
    if content is not None and _phase_records(written):
        services = context.services
        return context.standings.of_file(
            root, content, lambda: _from_file(services, project, written)
        )

    def read() -> _Standing:
        opened = Project.open(root) if apart else project
        return _from_elsewhere(context.services, opened, written)

    return context.standings.of(root, read)


def _still_recommends(project: Project, checkpoint: Mapping[str, Any]) -> bool:
    """Whether the recommendation a checkpoint carries stands as the pages are now: its
    journal page is there, was not rejected since, and still says ``next: research``."""
    path = checkpoint.get("recommends_page")
    if checkpoint.get("recommends") != "research" or not isinstance(path, str) or not path:
        return False
    try:
        page = project.page(path)
    except (ProjectError, OSError, ValueError):
        return False
    asks = str(page.meta.get("next") or "").strip().lower() == "research"
    return page.kind == "journal" and page.review != "rejected" and asks


def _as_it_stands(project: Project, state: dict[str, Any]) -> dict[str, Any]:
    """``state`` with the recommendation of its checkpoint as it stands now: ``recommends``
    and ``recommends_page`` are ``None`` once the journal page no longer recommends
    anything (the person rejected or changed it), so that a client need not look the
    page up to know.  The summary of the checkpoint is what was said then, and stays."""
    checkpoint = state.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("recommends"):
        return state
    if _still_recommends(project, checkpoint):
        return state
    return {**state, "checkpoint": {**checkpoint, "recommends": None, "recommends_page": None}}


def _shown_standing(context: Context, project: Project) -> dict[str, Any]:
    """Where the project stands, as the project view and the event stream carry it:
    ``state``, ``state_problem`` and ``state_source``."""
    standing = _standing(context, project)
    return {
        "state": _as_it_stands(project, standing.state),
        "state_problem": standing.problem,
        "state_source": standing.source,
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


def _standing_and_worker(context: Context, project: Project) -> tuple[_Standing, dict[str, Any]]:
    root = project.root
    standing = _standing(context, project)
    worker = _worker(context, root, standing.state) or context.workers.status(root)
    if standing.state["status"] == "running" and worker["state"] == "idle":
        standing = _standing(context, project)  # the look has just recorded that it is gone
    return standing, worker


def _modified(root: Path) -> str:
    """When the project last moved, as far as two ``stat`` calls can tell."""
    times = []
    for name in (STATE_FILE, EVENTS_FILE, PROJECT_FILE):
        with contextlib.suppress(OSError):
            times.append((root / name).stat().st_mtime)
    return format_stamp(datetime.fromtimestamp(max(times, default=0.0), UTC))


def _summary(context: Context, key: str, project: Project) -> dict[str, Any]:
    """One row of the library, from ``project.json`` and ``state.json`` alone where that
    file says where the project stands (:func:`_standing`, which is what the project
    view goes by as well: a row never says "not started" of a project whose page says
    otherwise).  ``pending`` is what the last checkpoint counted; :func:`_counted` puts
    the live number there."""
    standing = _standing(context, project, apart=True)
    state = standing.state
    worker = _worker(context, project.root, state)
    if worker is not None and state["status"] == "running" and worker["state"] == "idle":
        standing = _standing(context, project, apart=True)
        state = standing.state
    checkpoint = state.get("checkpoint") if isinstance(state.get("checkpoint"), dict) else None
    waiting = checkpoint.get("pending") if checkpoint else None
    if checkpoint:
        # Whether its recommendation still stands is told by a page, which a row does not
        # read: the row says nothing about it (the project view does).
        unread = ("recommends", "recommends_page")
        checkpoint = {name: value for name, value in checkpoint.items() if name not in unread}
    return {
        "id": key,
        "title": project.title,
        "created": project.created,
        "dir": project.root.name,
        # Built to be looked at: what it "spent" on models is its story's, not a bill.
        "demonstration": project.demonstration,
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
        "state_problem": standing.problem,
        "state_source": standing.source,
        "updated": _modified(project.root),
        "worker": worker["state"] if worker else "idle",
        # False: the project's files name a worker this server cannot verify (no Stop).
        "worker_verified": worker.get("verified") if worker else None,
        "counts": None,
    }


def _counted(
    context: Context, project: Project, row: dict[str, Any], deadline: float
) -> dict[str, Any]:
    """``row`` with the project's live ``counts`` and, from them, how much is ``pending``:
    proposed pages and edit proposals as they are now, not as the last checkpoint
    found them.  Left as it is when the counting has to wait for the next listing."""
    try:
        counts = context.counts.of(project.root, deadline)
    except (ProjectError, OSError) as exc:
        log.warning("What waits in %s could not be counted: %s", project.root, exc)
        counts = None
    if counts is None:
        return row
    return {**row, "counts": counts, "pending": counts["pending"]["total"]}


def _not_listed(key: str, project: Project, exc: Exception) -> dict[str, Any]:
    """The entry for a project whose row could not be built: the listing of all the
    others must not fail over it.  The type only: the text may quote anything."""
    log.error("The library row of %s could not be built.", project.root, exc_info=exc)
    return {
        "dir": project.root.name,
        "id": key,
        "message": (
            f"The project in {project.root.name}/ could not be listed "
            f"({type(exc).__name__}). The details are in the terminal where the server runs."
        ),
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


def _next_step(
    context: Context, project: Project, standing: _Standing, worker: Mapping[str, Any]
) -> dict[str, Any] | None:
    """What a press of Continue (an ``advance`` without a phase) would do, as the engine
    says it; ``None`` while a worker has the project (it decides as it goes) and where
    the engine cannot be asked.  A forecast that fails must not cost the project view."""
    if not standing.engine or worker.get("state") != "idle":
        return None
    try:
        return context.services.next_step(project, standing.state)
    except Exception as exc:
        if context.standings.first_complaint(project.root, type(exc).__name__):
            # Said once for each kind of failure, not at every look at the project.
            log.exception("What an advance of %s would do could not be told.", project.root)
        return None


def _problems(project: Project, standing: _Standing) -> tuple[list[dict[str, str]], int]:
    """What the project's files had to be read around (``Project.problems()``): the first
    ``_MAX_PROBLEMS`` of them, and how many there are.

    Each is listed under its file, so a message does not begin with that file's
    name a second time.  That ``state.json`` is lost is left out where
    ``state_problem`` says it: one thing is said once, by whoever knows it best.

    The store opens ``state.json`` and the events by their names to say whether the
    state is lost, and would wait for ever on a pipe put in their place.  Where
    one of them is no ordinary file (``state_problem`` then names it), nothing is
    listed until that is put right.
    """
    if standing.odd is not None:
        return [], 0
    state_problem = standing.problem
    found = []
    for item in project.problems():
        where, message = str(item.get("path") or ""), str(item.get("message") or "")
        if where == STATE_FILE and state_problem:
            continue
        if where and message.startswith(f"{where} "):
            message = message[len(where) + 1 :]
        found.append({"path": where, "message": message})
    return found[:_MAX_PROBLEMS], len(found)


def _project_view(context: Context, project: Project) -> dict[str, Any]:
    standing, worker = _standing_and_worker(context, project)
    state = _as_it_stands(project, standing.state)
    usage = _read_json(project.root, _USAGE_FILE)
    problems, total = _problems(project, standing)
    return {
        "schema": SCHEMA,
        "id": context.workspace.key_of(project),
        "title": project.title,
        "created": project.created,
        "settings": _shown_settings(project),
        "dir": project.root.name,
        "demonstration": project.demonstration,
        "state": state,
        "state_problem": standing.problem,
        "state_source": standing.source,
        "next_step": _next_step(context, project, standing, worker),
        "counts": project.counts(),
        "checkpoint": state.get("checkpoint"),
        "worker": worker,
        "attachments": project.attachments(),
        "usage": usage if isinstance(usage, dict) else None,
        "problems": problems,
        "problems_total": total,
    }


@router.get("/projects")
def list_projects(context: Ctx) -> dict[str, Any]:
    """Every project of the workspace, the most recently active first.

    A row is ``{"id", "title", "created", "dir", "demonstration", "status",
    "round", "phase", "next_phase", "checkpoint", "pending", "spent", "error",
    "state_problem", "state_source", "updated", "worker", "worker_verified",
    "counts"}``.  ``demonstration`` is true for a project that ``research demo``
    built: the model cost in ``spent`` is part of its story, and no model was
    called for it.
    ``counts`` is ``Project.counts()`` as the project view gives it and
    ``pending`` its total (proposed pages and edit proposals as they are now).
    Counting is bounded per listing: where a project's pages changed and the
    listing did not get to it, ``counts`` is ``null`` and ``pending`` is what the
    last checkpoint counted, until the next listing.

    ``status``, ``round``, ``phase``, ``next_phase``, ``checkpoint``, ``spent`` and
    ``error`` are the engine's reading of where the project stands, with
    ``state_problem`` and ``state_source`` as the project view has them: a project
    whose ``state.json`` is lost is listed where its events say it stands, not as
    one that has not started.  That reading replays the events once for such a
    project and is kept until its files change; a project whose ``state.json``
    reads fine costs nothing more than that file.

    ``unreadable`` lists the directories that hold a project which cannot be
    opened or listed, each with ``dir``, ``id`` (``null`` if this server never
    saw it whole) and a ``message`` that says why and what to do.
    """
    found = context.workspace.scan()
    rows: list[tuple[str, Project, dict[str, Any]]] = []
    failed: list[dict[str, Any]] = []
    for key, project in found.items():
        try:
            rows.append((key, project, _summary(context, key, project)))
        except Exception as exc:
            failed.append(_not_listed(key, project, exc))
    rows.sort(key=lambda row: (str(row[2]["updated"]), str(row[2]["created"])), reverse=True)
    deadline = time.monotonic() + _COUNT_SECONDS
    server = context.services.secrets()
    listed = []
    for key, project, row in rows:  # the most recently active are counted first
        try:
            counted = _counted(context, project, row, deadline)
            # Each row with the credentials of its own project blanked.
            hidden = context.services.secrets(project, server=server, pages=False)
            listed.append(_presentable(counted, hidden))
        except Exception as exc:
            failed.append(_not_listed(key, project, exc))
    roots = {project.root for project in found.values()}
    context.counts.keep(roots)
    context.standings.keep(roots)
    context.paper.keep(roots)
    return {"projects": listed, "unreadable": context.workspace.unreadable() + failed}


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


def _free_path(name: str, files: set[str], folders: dict[str, str]) -> str:
    """``name`` (a relative path), changed where needed so that it collides with nothing
    stored so far: not with a file of the same name, and not with a file where it needs
    a folder or a folder where it is a file.  Compared without case: the workspace may
    be on a file system that ignores it, and the project keeps no two inputs that
    differ by case alone.  So a folder that came before under another spelling
    (``Data``, then ``data``) is the same folder, under the spelling it came with
    first (``folders``: that spelling, by the path in lower case)."""
    *parents, leaf = name.split("/")
    prefix = ""
    for part in parents:
        candidate, copy = part, 1
        while f"{prefix}{candidate}".lower() in files:
            copy += 1
            candidate = f"{part}-{copy}"
        prefix = folders.setdefault(f"{prefix}{candidate}".lower(), f"{prefix}{candidate}") + "/"
    stem, suffix = os.path.splitext(leaf)
    candidate, copy = leaf, 1
    while f"{prefix}{candidate}".lower() in files or f"{prefix}{candidate}".lower() in folders:
        copy += 1
        candidate = f"{stem}-{copy}{suffix}"
    files.add(f"{prefix}{candidate}".lower())
    return f"{prefix}{candidate}"


def _stage_uploads(
    uploads: list[UploadFile], staging: Path, limits: UploadLimits
) -> list[dict[str, str]]:
    """Write the uploaded files into ``staging`` under safe names; returns what each was
    called (``sent``) and what it is called in the project's ``inputs/`` (``stored``).
    Nothing is written anywhere else."""
    stored: list[dict[str, str]] = []
    # The project keeps these two names in inputs/ for the brief typed into the form
    # and for its own list of the files; an attachment called the same would be
    # renamed there, and the answer would name another file than the one attached.
    files: set[str] = {BRIEF_FILE.lower(), MANIFEST_FILE.lower()}
    folders: dict[str, str] = {}
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
                # A new project gets the runtime its form named, else the one the
                # environment of the app asks for (PAPER_ORCHESTRA_AGENT_RUNTIME).
                project = Project.create(
                    directory,
                    title=title,
                    brief=brief,
                    files=tops,
                    settings=settings_for_new_project(settings),
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
            unstorable = _unstorable(settings, "settings")
            if unstorable is not None:
                raise ApiError(422, "invalid", unstorable)
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
    manifest = _read_json(project.root, f"{INPUTS_DIR}/{MANIFEST_FILE}")
    skipped = manifest.get("skipped", []) if isinstance(manifest, dict) else []
    return {**view, "uploads": stored, "skipped": skipped, **started}


@router.get("/projects/{pid}")
def get_project(project: Proj, context: Ctx) -> dict[str, Any]:
    """``project.json``, the engine's ``state``, ``counts``, the ``checkpoint`` and the
    ``worker``.  ``demonstration`` is true for a project built by ``research demo``:
    what it says it spent on models is its story's cost.

    ``state`` is where the project stands as the engine reads it: ``state.json``,
    or, when that file is lost or holds no phase, round, phases, checkpoint and
    what comes next rebuilt from the project's events.  ``state_problem`` is the
    engine's sentence about that (``null`` when the file is its account) and
    ``state_source`` says where the reading came from: ``"file"``, ``"events"``,
    ``"not_started"`` (no phase has run, nothing is wrong) or ``"unknown"``
    (nothing on file says where the project stands: Continue has to name a phase).

    ``next_step`` is what a press of Continue (``POST .../advance`` without a
    phase) would do, by the engine's own rule: ``{"will_run": [phases, in order],
    "then": the phase its checkpoint names as next, "experiments": [ids of the
    experiments a run phase would start], "not_running": [{"id", "reason"} for
    each experiment that reads approved and would not be started: it is no longer
    what was approved, or that approval was used], "results": how many result
    records have no analysis yet (a run the person cancelled is none),
    "checkpoint_of": a phase whose owed checkpoint is written first (or ``null``),
    "needs_phase": the call would be refused until a phase is named, "blocked":
    ``null`` or where the press stops for want of a model}``.  It is ``null`` while
    a worker has the project.

    ``next_step.blocked`` is ``{"phase", "needs": "model", "settings", "models",
    "message"}`` where the project's models cannot be used by this server's
    workers (no credential is configured, or a model offers no tools): ``phase``
    is the first phase that is not started, ``settings`` the entries of
    ``settings.models`` concerned, and ``message`` the one sentence that says what
    is missing and what to set up.  ``will_run`` then lists only what happens
    without a model (the run of what is approved) and ``then`` is that phase; with
    ``will_run`` empty a press starts nothing.  The server finds this out from its
    own environment, without contacting anything, and asks again after a short
    while; a credential set since the server started needs a restart.

    ``worker.last_exit.reason`` is ``"needs_model"`` when the last worker started
    nothing for that reason (``POST .../advance`` still answers 202: the worker
    finds it out for itself and ends).  It did not die, no error is among the
    events, and ``state`` is as it was.

    ``checkpoint.did`` is the account of what the phases did before the pause,
    without the count of what waited, without what was next and without the
    sentence about a model the next phase could not use; ``summary`` has all of
    it as the checkpoint said it.  (A checkpoint of an earlier version has no
    ``did``.)

    ``checkpoint`` (also ``state.checkpoint``) carries ``recommends`` and
    ``recommends_page``: ``"research"`` and the path of the journal page when the
    analyst recommended another look at the literature and that page still
    stands; both ``null`` otherwise.

    ``problems`` lists at most 50 of what the project's files had to be read
    around, each ``{"path", "message"}`` with the message not repeating the
    path; ``problems_total`` says how many there are.  A lost ``state.json`` is
    not among them when ``state_problem`` says it.
    """
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
    except BaseException:
        # Whatever went wrong (a refused value, a full disk): what could not be saved is
        # not kept in memory either, or it would be served as if it had been saved.
        context.workspace.restore(project)
        raise
    return _project_view(context, project)


# ---------------------------------------------------------------------------
# The agents: start, stop
# ---------------------------------------------------------------------------


class AdvanceBody(_Body):
    phase: str | None = None
    instruction: str | None = None


def _refuse_unread_state(context: Context, project: Project) -> None:
    """Refuse to start the agents where the view says that the project's state is not
    read (:func:`_from_elsewhere`: a link, a pipe or a directory stands where the state
    file, the events or the usage ledger should be).  The view foretells no press then,
    and what it does not offer must not happen behind it: the engine would read through
    the link, or wait for the pipe, by the file's name."""
    standing = _standing(context, project)
    if standing.odd is not None:
        raise ApiError(409, "conflict", f"Nothing was started. {standing.problem}")


@router.post("/projects/{pid}/advance", status_code=202)
def advance(project: Proj, context: Ctx, body: AdvanceBody | None = None) -> dict[str, Any]:
    """Start the engine from its next phase (or ``phase``) up to the next checkpoint.

    Returns at once; the worker reports through the event stream.  409 with the
    running ``worker`` and ``phase`` when the project already has one.

    Where the phase that would run needs a model and none can be used, the worker
    finds that out, starts nothing and ends: the answer here is 202 all the same.
    The project view said so beforehand (``next_step.blocked``, with the sentence
    to show) and says how that worker ended (``worker.last_exit.reason`` is
    ``"needs_model"``); no error is written to the project's events.  What needs
    no model, the run of an approved experiment, happens first.

    409 ``conflict`` as well, with nothing started, where the project view says that
    the state is not read because ``state.json``, the events or the usage ledger is not
    an ordinary file of the project (``state_source`` is ``"unknown"`` and ``next_step``
    is ``null``): the engine opens those files by their names, and the server starts
    none that would read through a link what it does not read itself.  The message is
    ``state_problem`` of the view, which says how to put it right.
    """
    _refuse_unread_state(context, project)
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


@router.post("/projects/{pid}/worker/forget")
def forget_worker(project: Proj, context: Ctx) -> dict[str, Any]:
    """Clear the name of a worker this server cannot verify (``worker.verified`` is
    ``false``: the project's files name one, and it is not a process this server
    started or can show to be this project's).  Nothing is signalled; a phase that
    read as running is recorded as interrupted and can be continued.  400
    ``worker_failed`` for a worker that is this server's to stop; with nothing to
    forget the status is returned as it is."""
    return {"worker": context.workers.forget(project.root)}


@router.get("/projects/{pid}/worker/log")
def worker_log(project: Proj, context: Ctx, offset: int = -_LOG_TAIL_BYTES) -> dict[str, Any]:
    """What the project's workers wrote (``worker.log``): why one could not start, what
    it said before it died.  The same shape as a run's log (see ``.../runs/{n}/log``);
    by default its end."""
    return _log_piece(context, project, WORKER_LOG, offset, whole_lines=True)


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
    """Hands out the complete lines added to a file since the last look.  The file is the
    project's own: a link put in its place is not followed, and reads as nothing."""

    def __init__(self, path: Path, chunk: int = 1 << 20) -> None:
        self.path = path
        self.chunk = chunk
        self.offset = 0
        self._inode: int | None = None

    def grown(self) -> bool:
        """Whether there may be something to read: one ``stat``, nothing is opened."""
        try:
            found = os.lstat(self.path)
        except OSError:
            return False
        return found.st_ino != self._inode or found.st_size != self.offset

    def read(self) -> list[bytes]:
        try:
            found = os.lstat(self.path)
        except OSError:
            return []
        if found.st_ino != self._inode or found.st_size < self.offset:
            # Another file under the same name, or one cut short: start over.  The
            # caller skips what it has already sent by its seq.
            self._inode, self.offset = found.st_ino, 0
        if found.st_size == self.offset:
            return []
        try:
            handle = _open_plain(self.path)
        except OSError:
            # A link, a pipe, a directory: not the project's events.  Nothing is read
            # until something else is there.
            self.offset = found.st_size
            return []
        with handle:
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


class _LineMask:
    """Blanks credentials in the lines of ``events.jsonl`` on their way to a client.

    A line without one is passed on as it was written, byte for byte, which is
    nearly every line; one that holds a credential (as it is, or escaped the way
    JSON writes it) is parsed, blanked and written anew.
    """

    def __init__(self, secrets: Sequence[str]) -> None:
        self.secrets = list(secrets)
        self._forms = _written_forms(self.secrets)

    def line(self, raw: bytes) -> bytes:
        if not any(form in raw for form in self._forms):
            return raw
        try:
            event = json.loads(raw)
        except ValueError:
            return raw  # not an event: the caller does not send it
        return json.dumps(_presentable(event, self.secrets)).encode()


async def event_stream(
    root: Path,
    after: int,
    live: LiveSettings,
    workers: Workers | None = None,
    *,
    secrets: Callable[[], Sequence[str]] | None = None,
    standing: Callable[[], Mapping[str, Any]] | None = None,
) -> AsyncIterator[bytes]:
    """The project's activity as server-sent events, for as long as the client listens.

    * every event of ``events.jsonl`` with ``seq`` greater than ``after``, in order:
      first those already there, then each new one as it is appended.  The frame
      is ``id: <seq>`` and ``data: <the event>`` with no ``event:`` line, so one
      ``onmessage`` handler receives them all and the browser resumes with
      ``Last-Event-ID``;
    * ``event: state`` with ``{"state", "state_problem", "state_source", "worker"}``
      at the start and whenever ``state.json`` or the worker changes: where the
      project stands as the project view says it, and the worker's status;
    * a comment line after ``heartbeat_seconds`` of silence.

    ``standing`` gives ``state``, ``state_problem`` and ``state_source`` (the engine's
    reading, see :func:`_shown_standing`).  Without it the frame carries ``state``
    alone, as ``state.json`` holds it.

    Following costs three ``stat`` calls per look; files are read off the event loop
    and in bounded pieces, so one slow client holds back only itself.

    ``secrets`` gives the strings that must not be sent (asked again whenever there
    is something to send: the project's settings may change while a client
    listens); each is replaced by ``[redacted]`` in events and in the state.
    """
    hidden = secrets or (lambda: ())
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
        # (Whether the state file is the project's own counts too: a link put in its place
        # leads to a file that ``stat`` cannot tell from the one that stood there.)
        current = (_signature(root / STATE_FILE), _not_own_file(root, STATE_FILE), glance)
        if current != seen:
            seen = current
            shown = await asyncio.to_thread(_stream_standing, root, standing)
            worker = await asyncio.to_thread(workers.status, root) if workers else None
            status = shown["state"].get("status")
            busy = status == "running" or bool(worker and worker["state"] != "idle")
            yield _sse("state", _presentable({**shown, "worker": worker}, hidden()))
            sent = True
        while tail.grown() and (lines := await asyncio.to_thread(tail.read)):
            mask = _LineMask(hidden())
            for raw in lines:
                seq = _seq(raw)
                if seq is None or seq <= last or not _is_event(raw):
                    continue
                last = seq
                yield b"id: %d\ndata: %s\n\n" % (seq, mask.line(raw.strip()))
                sent = True
        if sent:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= live.heartbeat_seconds:
            quiet_since = time.monotonic()
            yield b": keep-alive\n\n"
        await asyncio.sleep(live.poll_seconds)


def _stream_standing(
    root: Path, standing: Callable[[], Mapping[str, Any]] | None
) -> dict[str, Any]:
    """What a ``state`` frame of the stream says about where the project stands.  A
    reading that fails (the project's files changed under it) is not the end of the
    stream: the frame then carries the state file as it is."""
    if standing is not None:
        try:
            return dict(standing())
        except (ProjectError, OSError) as exc:
            log.warning("Where %s stands could not be read for its stream: %s", root, exc)
    return {"state": _engine_state(root)}


@router.get("/projects/{pid}/events")
async def events(pid: str, request: Request, context: Ctx, after: int = 0) -> StreamingResponse:
    """``text/event-stream`` of the project's activity (see :func:`event_stream`).

    Starts after ``after``, or after ``Last-Event-ID`` when the browser reconnects.
    """
    project = await run_in_threadpool(context.workspace.project, pid)
    resumed = request.headers.get("last-event-id", "").strip()
    # What this server gave out as an id: a few ASCII digits.  Anything else that passes
    # for a number somewhere (a superscript, a digit of another script) is not one.
    known = resumed.isascii() and resumed.isdigit() and len(resumed) <= 18
    start = max(after, int(resumed)) if known else after
    return StreamingResponse(
        event_stream(
            project.root,
            max(0, start),
            context.live,
            context.workers,
            secrets=lambda: context.services.secrets(project),
            standing=lambda: _shown_standing(context, project),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _lines_backwards(path: Path, chunk: int = 1 << 16) -> Iterator[bytes]:
    """The lines of a file of the project, last first, reading only as far back as the
    caller goes.  Nothing for a file that is absent or not the project's own (a link)."""
    try:
        handle = _open_plain(path)
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
def activity(
    project: Proj, context: Ctx, before: int | None = None, limit: int = 100
) -> dict[str, Any]:
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


def _by(author: str) -> str:
    """An author as a sentence names them: ``you``, ``the analyst``, ``the engine``."""
    try:
        author = normalize_author(author)
    except ProjectError:
        return "its author"  # a proposal file written by hand may name anybody
    if author == HUMAN:
        return "you"
    return f"the {role_of(author)}" if is_agent(author) else "the engine"


def _as_proposed(page: ResearchPage, proposal: EditProposal) -> ResearchPage:
    """``page`` as it would read with ``proposal`` accepted as it stands: its fields set,
    its relations added, its text in place of the page's.  Nothing is written."""
    meta = dict(page.meta)
    for name, value in proposal.fields.items():
        if value is None or value == []:
            meta.pop(name, None)
        else:
            meta[name] = value
    for name, targets in proposal.relations_add.items():
        meta[name] = [*page.relations.get(name, []), *targets]
    return page_from_parts(page.path, meta, page.body if proposal.body is None else proposal.body)


def _as_stored(page: ResearchPage, fields: dict[str, Any], hidden: Sequence[str]) -> dict:
    """``fields`` of an edit to ``page``, with the address of a repository that came back
    as it was shown (its token blanked) replaced by the stored one: showing a page
    must not cost the token when what was shown is sent back with an edit."""
    stored, sent = page.meta.get("repo"), fields.get("repo")
    if not (isinstance(stored, str) and isinstance(sent, str)) or sent == stored:
        return fields
    return {**fields, "repo": stored} if sent.strip() == _presentable(stored, hidden) else fields


def _blank(text: str, hidden: Sequence[str]) -> str:
    """``text`` with the credentials ``hidden`` blanked, for what is rendered from it.
    Rendering escapes and marks up what it is given: a credential that holds such
    characters would stand in the HTML in a form that blanking the finished answer
    does not find."""
    return str(_presentable(text, hidden))


def _numbers(context: Context, project: Project, page: ResearchPage) -> dict[str, Any]:
    """The number report of a finding as ``page`` reads, under ``number_report``; where it
    cannot be made, ``None`` and the reason under ``number_report_error``."""
    try:
        return {"number_report": context.services.number_report(project, page)}
    except (ApiError, ProjectError) as exc:
        return {"number_report": None, "number_report_error": str(exc)}


def _proposal_view(
    context: Context,
    project: Project,
    proposal: EditProposal,
    page: ResearchPage,
    hidden: Sequence[str],
) -> dict[str, Any]:
    """An edit proposal with what a person needs to judge it: the unified diff against
    the page as it is, the page as it would read with the changes marked, and, for a
    finding, the number report of the text as it would read with the edit applied:
    the numbers an edit brings are checked before the decision, not after it."""
    current = _blank(page.body, hidden)
    proposed = current if proposal.body is None else _blank(proposal.body, hidden)
    finding = page.kind == "finding"
    # Headed as the app says everything else: whose suggestion it is, not "agent:analyst".
    diff = unified_diff(
        current,
        proposed,
        old_label="the page as it is",
        new_label=f"as {_by(proposal.author)} {'suggest' if proposal.author == HUMAN else 'suggests'} it",
    )
    marked = render_marked(
        current,
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
        **(_numbers(context, project, _as_proposed(page, proposal)) if finding else {}),
    }


def _decided_edits(project: Project, page: ResearchPage) -> list[dict[str, Any]]:
    """The edit proposals for ``page`` that the person has decided, the latest decision
    first (at most ``_MAX_DECIDED_EDITS``): each as the proposal is on file, with its
    ``review``, ``decided_at`` and ``decision_reason``, but without the text it proposed
    (``changes_text`` says whether it proposed one).  So that a suggestion that was
    turned down, and why, can be found on the page it was about."""
    decided = [
        proposal
        for proposal in project.proposals(review=None)
        if proposal.target == page.path and proposal.review != "proposed"
    ]
    # Decisions of one second are told apart by when the edits were suggested.
    decided.sort(
        key=lambda proposal: (proposal.decided_at or "", proposal.created, proposal.id),
        reverse=True,
    )
    rows = []
    for proposal in decided[:_MAX_DECIDED_EDITS]:
        row = proposal.to_dict()
        text = row.pop("body")
        rows.append({**row, "changes_text": text is not None})
    return rows


def _page_view(context: Context, project: Project, page: ResearchPage) -> dict[str, Any]:
    """A page as the wiki view shows it."""
    pid = context.workspace.key_of(project)
    graph = project.graph(include_rejected=True)
    nodes = {node["id"]: node for node in graph["nodes"]}
    hidden = context.services.secrets(project)
    view: dict[str, Any] = {
        "page": page.to_dict(),
        "html": render_markdown(
            _blank(page.body, hidden), project=project, pid=pid, page_path=page.path
        ),
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
            _proposal_view(context, project, proposal, page, hidden)
            for proposal in project.proposals(review="proposed")
            if proposal.target == page.path
        ],
        # What was suggested for this page before, and what the person decided.
        "decided_proposals": _decided_edits(project, page),
        # A record (a result, a source) that no longer reads as code wrote it.
        "edited_record": context.services.edited_record(project, page),
    }
    if page.kind == "finding":
        try:
            view["number_check"] = context.services.number_check(project, page)
        except (ApiError, ProjectError) as exc:
            # The page is still worth reading without the check; say that it is missing.
            view["number_check"] = None
            view["number_check_error"] = str(exc)
        view.update(_numbers(context, project, page))
    return view


@router.get("/projects/{pid}/pages")
def list_pages(
    project: Proj,
    context: Ctx,
    kind: str | None = None,
    review: str | None = None,
    state: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    """Pages without their text.  ``kind``, ``review`` and ``state`` take one value or
    several separated by commas; ``q`` keeps the pages that match a search, best first,
    each with its ``snippet``.

    The row of an experiment also carries ``interrupted``, as the experiments'
    own listing does: true where the page says ``running`` and nothing is running
    it any more, so that whatever is drawn from this list (a glyph, a pill) can
    say what the experiment's view says in words.  It carries ``will_not_run`` and
    ``last_success`` as that listing does, too: an experiment that reads approved
    and will not run as it stands, and one that reads cancelled or failed although
    an earlier run of it succeeded, are told wherever experiments are drawn."""
    pages = project.pages(kind=_many(kind), review=_many(review), state=_many(state))
    interrupted = _interrupted(context, project, [p for p in pages if p.kind == "experiment"])

    def row(page: ResearchPage) -> dict[str, Any]:
        listed = page.to_dict(body=False)
        if page.kind == "experiment":
            listed["interrupted"] = page.id in interrupted
            listed.update(_run_standing(context, project, page))
        return listed

    if q and q.strip():
        hits = {hit.path: hit for hit in project.search(q, k=len(pages) or 1, review=None)}
        order = {path: position for position, path in enumerate(hits)}
        pages = sorted((p for p in pages if p.path in hits), key=lambda p: order[p.path])
        return {"pages": [{**row(p), "snippet": _snippet(hits[p.path].snippet)} for p in pages]}
    return {"pages": [row(page) for page in pages]}


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
    hidden = context.services.secrets(project)
    current = _blank(project.page(target).body, hidden) if target else None
    body = _blank(body, hidden)
    diff = (
        unified_diff(
            body, current, old_label=f"the version of {stamp}", new_label="the page as it is"
        )
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
    with the relation ``type``), ``history`` (earlier versions, newest first) and the
    pending edit ``proposals`` for it (each with ``diff``, ``diff_html`` and
    ``marked_html``).  ``path`` may also be an id (``H2``) or a link target.

    ``decided_proposals`` are the edits suggested for the page that the person has
    decided, the latest decision first: each with ``review`` (``accepted`` or
    ``rejected``), ``decided_at``, ``decision_reason``, ``author``, ``summary``,
    ``fields`` and ``changes_text`` (whether it proposed a new text), without that
    text.  ``GET .../proposals?review=all`` lists the decided edits of every page.

    ``page.created`` is when the page was first written (for a proposal: when it
    was proposed) and stays so; ``page.updated`` and ``page.meta.updated_by`` say
    when and by whom it was last changed, which an edit of the person's moves.

    For a finding, ``number_check`` lists where each of its numbers was found
    (``{"number", "found_in", "in_basis", "kinds"}``) and ``number_report`` is the
    whole check: ``{"checked", "skipped", "elsewhere", "unverified", "summary"}``,
    with the numbers that were not looked at because they are short or round
    (``skipped``), those found only outside the records the finding is based on
    (``elsewhere``) and a sentence that says all of it.  Each pending edit proposal
    of a finding carries the ``number_report`` of the text as it would read with
    the edit applied.  What the check looks at is the metrics of runs that
    succeeded (of experiments that are not rejected), the accepted sources and the
    person's files; it says that a number occurs there, not that it is the number
    the sentence is about, and it does not look at short and round numbers at all.

    ``edited_record`` is true for a result or source record that no longer reads
    as code wrote it (a person edited it)."""
    return _page_view(context, project, project.page(path))


class PageEdit(_Body):
    title: str | None = None
    kind: str | None = None
    body: str | None = None
    fields: dict[str, Any] | None = None
    relations: dict[str, Any] | None = None
    base_updated: str | None = None
    #: "Save and accept": the edit is also the person's decision to accept the page.
    accept: bool = False


@router.put("/projects/{pid}/pages/{path:path}")
def put_page(path: str, body: PageEdit, project: Proj, context: Ctx) -> dict[str, Any]:
    """The person's edit.  ``base_updated`` is the ``updated`` stamp of the version they
    started from: when the page changed since, the answer is 409 with the current
    ``page`` and nothing is written.

    Editing is not deciding: a proposed page stays proposed and a rejected one
    stays rejected, with the person's corrections in it.  ``"accept": true`` is
    "save and accept": the same as accepting the page in a review with these
    changes as its edits."""
    page = project.page(path)
    if body.kind and body.kind != page.kind:
        raise ApiError(
            422,
            "invalid",
            f"This page is {page.kind} and stays one: the kind follows from where a page "
            "is kept. Create a new page of the other kind instead.",
        )
    fields = _as_stored(page, dict(body.fields or {}), context.services.secrets(project))
    if body.title is not None:
        fields["title"] = body.title
    updated = project.update_page(
        page.path,
        author="human",
        body=body.body,
        fields=fields or None,
        relations=body.relations,
        base_updated=body.base_updated,
        accept=body.accept,
    )
    return _page_view(context, project, updated)


@router.delete("/projects/{pid}/pages/{path:path}")
def delete_page(path: str, project: Proj, context: Ctx) -> dict[str, Any]:
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


def _edited_page(project: Project, ref: str) -> ResearchPage:
    """The page the edits of a decision apply to: the page ``ref`` names, or the one the
    edit proposal of that id would change."""
    try:
        return project.page(ref)
    except NotFound:
        return project.page(project.proposal(ref).target)


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
        edits = item.edits
        if edits and isinstance(edits.get("fields"), dict):
            with contextlib.suppress(ProjectError):  # the review says what is wrong with it
                edited = _edited_page(project, item.ref)
                hidden = context.services.secrets(project)
                edits = {**edits, "fields": _as_stored(edited, edits["fields"], hidden)}
        decided = project.review(
            item.ref,
            item.decision,
            reason=item.reason,
            edits=edits,
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
        if decided.kind == "experiment" and decided.review == "rejected":
            # Said with the decision: findings and verdicts that rested on its runs do
            # not go away by themselves, and the person may want to look at them.
            with contextlib.suppress(ApiError, ProjectError):
                outcome["dependents"] = context.services.dependents(project, decided.path)
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

    Rejecting an experiment also returns ``dependents``: what rests on it,
    ``{"experiment", "results", "findings": [{"id", "path", "title", "review",
    "only"}], "hypotheses": [{"id", "path", "title", "state", "through"}]}``.
    Nothing of that is changed by the rejection: the findings based on its runs
    (``only``: on nothing else) and the verdicts they carry are the person's to
    revisit.  The records of a rejected experiment are no evidence for the number
    check, and a paper does not report a finding that rests on nothing else.
    """
    results = [_decide(context, project, item) for item in body.items]
    return {"ok": all(result["ok"] for result in results), "results": results}


@router.get("/projects/{pid}/proposals")
def list_proposals(project: Proj, context: Ctx, review: str = "proposed") -> dict[str, Any]:
    """Edit proposals, oldest first: the pending ones, or ``review=accepted``,
    ``rejected`` or ``all``.  Each carries the ``page`` it would change (without text)
    and how many lines it adds and removes.

    A pending edit to a finding also carries ``number_report``: the number check of
    the finding as it would read with the edit applied, as the page view gives it
    for the same proposal, so that an edit can be judged, and an edit that brings
    a number found nowhere can be marked, where it is listed.  ``number_report`` is
    ``null`` where the report could not be made (``number_report_error`` says why)
    and where the listing did not get to it: one listing spends a bounded time on
    these checks.  A client that decides in the list treats ``null`` as "not
    checked here" and shows the edit's own view first."""
    wanted = None if review == "all" else review
    pages = {page.path: page for page in project.pages()}
    deadline = time.monotonic() + _ROW_CHECK_SECONDS
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
        if page is not None and page.kind == "finding" and proposal.review == "proposed":
            row.update(number_report=None, number_report_error=None)
            if time.monotonic() < deadline:
                row.update(_numbers(context, project, _as_proposed(page, proposal)))
        rows.append(row)
    return {"proposals": rows}


# ---------------------------------------------------------------------------
# Graph, requests, search
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/graph")
def graph(project: Proj, context: Ctx, include_rejected: str | None = None) -> dict[str, Any]:
    """``{"nodes", "edges", "dangling"}`` as ``Project.graph`` builds them."""
    return project.graph(include_rejected=_flag(include_rejected))


@router.get("/projects/{pid}/requests")
def list_requests(project: Proj, context: Ctx, state: str = "all") -> dict[str, Any]:
    """The person's instructions to the agents, oldest first (``state=open`` or ``done``
    to narrow)."""
    wanted = None if state == "all" else state
    return {"requests": [request.to_dict() for request in project.requests(state=wanted)]}


class RequestBody(_Body):
    text: str
    target: str | None = None
    #: Whom the request is for: ``"all"`` (the default) or one role.
    role: str | None = Field(default=None, alias="for")


@router.post("/projects/{pid}/requests", status_code=201)
def add_request(body: RequestBody, project: Proj, context: Ctx) -> dict[str, Any]:
    """Leave an instruction for the agents: ``{"text", "target", "for"}``.

    ``target`` is the page it is about, if any.  ``for`` says who gets it:
    ``"all"`` (the default: every phase that runs until the round's analysis has
    completed) or one role (``scout``, ``theorist``, ``designer``, ``analyst``:
    that role's next phase); anything else is refused (422).  The answer is the
    ``request`` as it was stored, with ``for`` and ``handled_in`` (the phases that
    were given it so far).
    """
    request = project.add_request(body.text, target=body.target or None, role=body.role)
    return {"request": request.to_dict()}


@router.delete("/projects/{pid}/requests/{rid}")
def delete_request(rid: str, project: Proj, context: Ctx) -> dict[str, Any]:
    project.delete_request(rid)
    return {"deleted": rid}


_MARKDOWN_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")
_EMPHASIS = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_LINE_MARK = re.compile(r"^\s*(?:#{1,6}|>+|[-*+]|\d+[.)])\s+")


def _snippet(line: Any) -> str:
    """The line of a page a search matched, as text: without the marks a page is written
    with.  A row of a table reads ``a · b``, a link is its label, and ``**``, backticks
    and the mark that begins a heading, a quotation or a list item are left out.  (The
    snippet is shown where nothing is rendered: a list of hits, the palette.)"""
    text = " ".join(str(line or "").split())
    if text.startswith("|"):
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        text = " · ".join(cell for cell in cells if cell.strip("-: "))
    text = _MARKDOWN_LINK.sub(r"\1", plain_links(_LINE_MARK.sub("", text)))
    return _EMPHASIS.sub(r"\1", text.replace("**", "").replace("`", "")).strip()


@router.get("/projects/{pid}/search")
def search(project: Proj, context: Ctx, q: str = "", limit: int = 20) -> dict[str, Any]:
    """Keyword search over every page, whatever its review; best first.  The ``snippet``
    of a hit is the matching line as plain text (no table pipes, link brackets or
    emphasis marks)."""
    if not q.strip():
        return {"query": q, "hits": []}
    pages = {page.path: page for page in project.pages()}
    hits = []
    for hit in project.search(q, k=max(1, min(limit, 100)), review=None):
        page = pages.get(hit.path)
        if page is not None:
            snippet = _snippet(hit.snippet)
            hits.append({**page.to_dict(body=False), "score": hit.score, "snippet": snippet})
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


def _interrupted(context: Context, project: Project, pages: Sequence[ResearchPage]) -> set[str]:
    """The ids of the experiments among ``pages`` whose run was interrupted: the page
    says ``running``, the project has no worker and no process holds the experiment's
    run lock.  Whoever ran it died without settling the run (a killed worker whose
    job outlived it, a run on a remote backend, a machine that went down); running
    the experiment again settles what is left and starts it."""
    running = [page for page in pages if page.state == "running" and page.id]
    if not running or context.workers.status(project.root)["state"] != "idle":
        return set()
    return {
        str(page.id)
        for page in running
        if context.services.run_held(project, str(page.id)) is not True
    }


def _cancel_does(context: Context, project: Project, page: ResearchPage) -> str | None:
    """What ``POST .../cancel`` would do to the experiment as it stands: ``"withdraw"``
    (it is approved and has not started: the approval is taken back and it is a plan
    again), ``"cancel"`` (it waits in the queue or is being run: the run ends as
    cancelled) or ``None`` (there is nothing to cancel).

    An experiment that reads ``approved`` while a process holds its run lock is
    being run (the page was saved from an editor meanwhile): its approval is in
    use, and what can be done is to cancel the run.
    """
    if page.state in ("queued", "running"):
        return "cancel"
    if page.state != "approved":
        return None
    return "cancel" if context.services.run_held(project, str(page.id)) is True else "withdraw"


def _run_standing(context: Context, project: Project, page: ResearchPage) -> dict[str, Any]:
    """What a listing says of an experiment beyond the state on its page, so that the
    state is not taken for more than it says:

    ``will_not_run``: for an accepted experiment whose page reads ``approved`` (or
    ``queued``, ``running``), why a run of it would not start as it stands: it is no
    longer what was approved (its code or plan changed), or that approval was used.
    The rest of a sentence (``it is not what was approved at ... (main.py: ...)``);
    ``None`` when it would run.  Such an experiment needs the person's approval again,
    and a press of Continue does not run it (``next_step.not_running`` names it too).

    ``last_success``: for an experiment whose page reads ``cancelled`` or ``failed``,
    the latest run of it that succeeded, ``{"number", "result_page"}``: its state is
    how the last approved run ended, not that it has no result.  ``None`` otherwise.

    Only the experiments in those states cost a look at their code or their runs.
    """
    waits = page.review == "accepted" and page.state in ("approved", "queued", "running")
    over = page.state in ("cancelled", "failed")
    name = str(page.id or "")
    return {
        "will_not_run": context.services.will_not_run(project, name) if waits and name else None,
        "last_success": context.services.last_success(project, name) if over and name else None,
    }


def _dependents(context: Context, project: Project, page: ResearchPage) -> Any:
    """What rests on a rejected experiment, as the answer to its rejection said it;
    ``None`` for an experiment that is not rejected, and where it cannot be told."""
    if page.review != "rejected":
        return None
    try:
        return context.services.dependents(project, page.path)
    except (ApiError, ProjectError, OSError) as exc:
        log.warning("What rests on %s could not be listed: %s", page.id, exc)
        return None


@router.get("/projects/{pid}/experiments")
def list_experiments(project: Proj, context: Ctx) -> dict[str, Any]:
    """Every experiment page (without text) with a summary of its runs: how many, and
    the latest one's record.  ``interrupted`` is true where the page says ``running``
    and nothing is running it any more, and ``cancel_does`` says what a cancel would
    do (see the experiment's own view for both).

    ``will_not_run`` is ``null``, or why an experiment that reads approved would not
    run as it stands (it changed since the approval: it needs approval again).
    ``last_success`` is ``null``, or, for an experiment that reads ``cancelled`` or
    ``failed``, its latest run that succeeded: ``{"number", "result_page"}``."""
    pages = project.pages(kind="experiment")
    interrupted = _interrupted(context, project, pages)
    rows = []
    for page in pages:
        runs = context.services.runs(project, page.id) if page.id else []
        latest = max(runs, key=lambda run: run.get("number") or 0, default=None)
        rows.append(
            {
                **page.to_dict(body=False),
                "runs": len(runs),
                "last_run": _run_view(context, project, page.id, latest) if latest else None,
                "interrupted": page.id in interrupted,
                "cancel_does": _cancel_does(context, project, page),
                **_run_standing(context, project, page),
            }
        )
    return {"experiments": rows}


_INTERRUPTED = (
    "The run of {id} was interrupted: the worker that ran it is gone and nothing is "
    "running it now. Run it again to go on (what is left of the unfinished run is "
    "settled first), or cancel it."
)


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
    interrupted = bool(_interrupted(context, project, [page]))
    return {
        "page": page.to_dict(),
        # The page says "running" and nothing runs it: Run starts it again.
        "interrupted": interrupted,
        "interrupted_note": _INTERRUPTED.format(id=experiment_id) if interrupted else None,
        # What the cancel endpoint would do now: "withdraw", "cancel" or nothing.
        "cancel_does": _cancel_does(context, project, page),
        # Why it would not run although it reads approved; an earlier run that succeeded.
        **_run_standing(context, project, page),
        # For a rejected experiment: the findings and verdicts that rest on its runs.
        "dependents": _dependents(context, project, page),
        "html": render_markdown(
            _blank(page.body, context.services.secrets(project)),
            project=project,
            pid=context.workspace.key_of(project),
            page_path=page.path,
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
    plan's ``repo`` is shown without the token of its URL.

    ``interrupted`` is true when the page says ``running`` although nothing is
    running the experiment any more (the project has no worker and no process
    holds the experiment's run lock); ``interrupted_note`` then says so in a
    sentence, and ``POST .../run`` settles the unfinished run and starts the
    experiment again.

    ``cancel_does`` is what ``POST .../cancel`` would do as things stand:
    ``"withdraw"`` (the experiment is approved and has not started: its approval
    is taken back), ``"cancel"`` (it is queued or being run: the run ends as
    cancelled) or ``null`` (nothing to cancel).

    ``will_not_run`` is ``null``, or, for an experiment whose page reads approved
    (or queued, running), why a run would not start as it stands, as the rest of
    a sentence: it is not what was approved any more, or the run it was approved
    for is over.  ``last_success`` is ``null``, or, for an experiment that reads
    ``cancelled`` or ``failed``, the latest run of it that succeeded
    (``{"number", "result_page"}``).

    ``dependents`` is ``null`` unless the experiment is rejected; then it is what
    rests on it, in the shape the review answer gives with the rejection
    (``{"experiment", "results", "findings", "hypotheses"}``), for as long as the
    rejection stands."""
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
    unless its code has to be repaired).  409 when the project already has a worker.

    An experiment whose run was interrupted (its page says ``running``, the
    project has no worker and nothing holds the experiment's run lock) is run
    again the same way: the worker settles the unfinished run first, as ``research
    run`` does from a terminal.  409 when another process is running it.
    """
    page = _experiment(project, eid)
    idle = True
    if page.state == "running":
        # A look is what notices a worker that died, and the runs it left are settled a
        # moment later (the settling holds the experiment meanwhile).  What is decided
        # here is decided on the page as it is after that.
        idle = context.workers.status(project.root)["state"] == "idle"
        context.workers.settled(project.root, timeout=_SETTLE_SECONDS)
        page = _experiment(project, eid)
    if page.state == "running":
        if idle and context.services.run_held(project, str(page.id)) is True:
            raise ApiError(
                409,
                "conflict",
                f"{page.id} is being run by a process this server did not start (it holds "
                "the experiment's run lock). Wait for that run, or cancel it.",
            )
        # With a worker at work the start below is refused, as any second worker is.
    elif page.state not in ("approved", "queued"):
        raise ApiError(
            422,
            "invalid",
            f"{page.id} is {page.state}; only an approved experiment can be run. Approve it first.",
        )
    return {"worker": _start_run(context, project, str(page.id))}


class CancelBody(_Body):
    #: What the person was told the press does (``cancel_does`` of the experiment as they
    #: saw it): ``"withdraw"`` or ``"cancel"``.  When the experiment has moved on and the
    #: press would do the other, nothing is done (409).
    expect: str | None = None


_CANCEL_DOES = ("withdraw", "cancel")
_NOT_AS_TOLD = {
    "withdraw": (
        "{id} is {state} and has been taken up by a run since you looked: its approval is "
        "in use and cannot be taken back. Nothing was changed. Cancel its run if you want "
        "it stopped."
    ),
    "cancel": (
        "{id} is approved and has not started: there is no run to cancel. Nothing was "
        "changed. Take the approval back if you do not want it run."
    ),
}
_CANCEL_SAID = {
    "withdrawn": (
        "{id} is a plan again: its approval was taken back and nothing ran. Approve it "
        "again when you want it run."
    ),
    "cancelled": "{id} is cancelled. Approve it again when you want it run.",
    "cancelling": (
        "{id} was told to stop. The worker that runs it ends the job and records the run "
        "as cancelled within about a second."
    ),
    "nothing": "{id} is {state}: there is nothing to cancel.",
}


def _take_back(context: Context, project: Project, page: ResearchPage) -> str:
    """Take back the approval of ``page``, which has not started; returns the outcome:
    ``withdrawn``, ``ended`` (the run it was approved for was over: the page now says
    how) or ``nothing`` (it is a plan already: somebody else took the approval back).

    409 when a process took the experiment up in this very moment, whether its page
    still reads ``approved`` or has moved on to ``queued`` or ``running``: an approval
    that is in use is not taken back, and nothing of the run in flight is touched.
    """
    try:
        context.services.withdraw(project, page.path)
    except ProjectError as exc:
        now = _experiment(project, str(page.id))
        if now.state == "approved":
            in_use = str(exc)  # the experiments module's own words: it is being run
        elif now.state in ("queued", "running"):
            in_use = _NOT_AS_TOLD["withdraw"].format(id=page.id, state=now.state)
        else:
            return "nothing" if now.state == "planned" else "ended"
        raise ApiError(
            409,
            "conflict",
            in_use,
            page=now.to_dict(body=False),
            cancel_does=_cancel_does(context, project, now),
        ) from exc
    return "withdrawn"


def _cancel_run(context: Context, project: Project, page: ResearchPage) -> str:
    """Cancel the run of ``page``, which is queued or being run; returns the outcome:
    ``cancelled`` (nobody was running it: done here and now), ``cancelling`` (whoever
    runs it was told to stop) or ``ended`` (its run was over already)."""
    context.services.request_cancel(project, page.path)
    state = _experiment(project, str(page.id)).state
    if state == "cancelled":
        return "cancelled"
    return "cancelling" if state in ("approved", "queued", "running") else "ended"


@router.post("/projects/{pid}/experiments/{eid}/cancel")
def cancel(
    eid: str, project: Proj, context: Ctx, response: Response, body: CancelBody | None = None
) -> dict[str, Any]:
    """Take back the approval of an experiment that has not started, or cancel its run:
    what ``paper-orchestra research cancel`` does.

    An approved experiment that no process is running becomes a plan again
    (``planned``): nothing is cancelled and nothing is recorded as a run.  One
    that waits in the queue is cancelled at once; one that is being run is told
    to stop, and its worker ends the job and records the run as cancelled.
    ``cancel_does`` of the experiment says beforehand which of these a press does.

    The answer says what happened: ``{"experiment", "outcome", "before", "state",
    "message", "page", "cancel_does"}``.  ``outcome`` is ``"withdrawn"``,
    ``"cancelled"``, ``"cancelling"`` (status 202: the run ends a moment later),
    ``"nothing"`` (there was nothing to cancel, or no longer: another window took
    the approval back first) or ``"ended"`` (the run was over before anything could
    be done: the page now says how it ended).  ``before`` and
    ``state`` are the experiment's state before and after, ``message`` says it in a
    sentence, ``page`` is the page as it is now and ``cancel_does`` what a further
    press would do.

    ``{"expect": "withdraw" | "cancel"}`` is what the person was told the press
    does.  When the experiment moved on meanwhile and the press would do the
    other, nothing is done: 409 with the ``page`` and ``cancel_does`` as they are
    now.  Taking an approval back never cancels, requeues or starts a run: when a
    process took the experiment up in that very moment (its page may still read
    ``approved``, or have moved on to ``queued`` or ``running``) the answer is 409
    as well, and the run in flight is not touched.
    """
    expect = ((body.expect if body else None) or "").strip().lower() or None
    if expect is not None and expect not in _CANCEL_DOES:
        raise ApiError(
            422, "invalid", f"expect is 'withdraw' or 'cancel', not {str(expect)[:40]!r}."
        )
    page = _experiment(project, eid)
    before, does = str(page.state), _cancel_does(context, project, page)
    if expect is not None and does is not None and expect != does:
        raise ApiError(
            409,
            "conflict",
            _NOT_AS_TOLD[expect].format(id=page.id, state=before),
            page=page.to_dict(body=False),
            cancel_does=does,
        )
    if does == "withdraw":
        outcome = _take_back(context, project, page)
    elif does == "cancel":
        outcome = _cancel_run(context, project, page)
    else:
        outcome = "nothing"
    now = _experiment(project, str(page.id))
    if outcome == "ended":
        what = "the approval could be taken back" if does == "withdraw" else "it could be cancelled"
        message = f"{page.id} is {now.state}: its run had ended before {what}."
    else:
        message = _CANCEL_SAID[outcome].format(id=page.id, state=now.state)
    response.status_code = 202 if outcome == "cancelling" else 200
    return {
        "experiment": page.id,
        "outcome": outcome,
        "before": before,
        "state": now.state,
        "message": message,
        "page": now.to_dict(body=False),
        "cancel_does": _cancel_does(context, project, now),
    }


def _find_run(
    context: Context, project: Project, experiment_id: str, number: int
) -> dict[str, Any]:
    for run in context.services.runs(project, experiment_id):
        if run.get("number") == number:
            return run
    raise ApiError(404, "not_found", f"{experiment_id} has no run {number}.")


_NO_LOG: dict[str, Any] = {
    "offset": 0, "next": 0, "size": 0, "text": "", "eof": True, "reset": False,
}  # fmt: skip


def _spans(window: bytes, needles: Sequence[bytes]) -> list[tuple[int, int]]:
    """Where any of ``needles`` occurs in ``window``: ``(start, end)`` pairs in order,
    occurrences that touch or overlap joined into one."""
    found = []
    for needle in needles:
        at = window.find(needle)
        while at >= 0:
            found.append((at, at + len(needle)))
            at = window.find(needle, at + 1)
    joined: list[tuple[int, int]] = []
    for start, end in sorted(found):
        if joined and start <= joined[-1][1]:
            joined[-1] = (joined[-1][0], max(joined[-1][1], end))
        else:
            joined.append((start, end))
    return joined


def _begun(data: bytes, needles: Sequence[bytes]) -> int:
    """How many bytes at the end of ``data`` are the beginning of one of ``needles`` and
    not yet all of it (``0`` for none, or for fewer than a credential's first few)."""
    longest = 0
    for needle in needles:
        for size in range(
            min(len(needle) - 1, len(data)), max(longest, _MIN_HELD_BACK_BYTES - 1), -1
        ):
            if data.endswith(needle[:size]):
                longest = size
                break
    return longest


def _read_log(
    handle: BinaryIO,
    offset: int,
    limit: int = _LOG_CHUNK_BYTES,
    *,
    whole_lines: bool = False,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """A piece of an open log file from byte ``offset`` (negative: that many bytes from
    the end), cut at a character boundary so that pieces can be joined.  With
    ``whole_lines`` a piece that is not the end of the file ends at a line break.

    ``eof`` says that nothing more can be read at the moment: ask again later, from
    ``next``.  That is also so when the file ends in the middle of a character (it is
    being written, or its writer was killed): ``next`` then stays before that
    character, and a client that reads on while ``eof`` is false does not spin.

    Each of ``secrets`` is replaced by ``[redacted]`` in ``text`` (as it is, and as
    JSON writes it: see :func:`_written_forms`), also where the piece begins or
    ends inside one (the file is read a little beyond both ends to see that):
    then the part inside the piece is what is replaced.  Where the file itself
    ends in what may be the first part of a secret that is still being written,
    the piece ends before it, like before half a character, and ``eof`` is true:
    the rest of it has to be waited for.  ``offset``, ``next`` and ``size`` are
    always those of the file.
    """
    needles = _written_forms(secrets)
    margin = max((len(needle) for needle in needles), default=1) - 1
    size = os.fstat(handle.fileno()).st_size
    from_end = offset < 0
    start = max(0, size + offset) if from_end else offset
    reset = start > size  # the file was written anew since the client last read it
    if reset:
        start = 0
    lead = min(margin, start)
    handle.seek(start - lead)
    window = handle.read(lead + limit + margin)
    begin, end = lead, min(lead + limit, len(window))
    if from_end and start > 0:
        # Begin at a line, not in the middle of one (or of a character).
        begin = window.find(b"\n", begin, end) + 1 or begin
    at_end = start - lead + end >= size
    if whole_lines and not at_end:
        end = window.rfind(b"\n", begin, end) + 1 or end
    spans = _spans(window, needles)
    waits = at_end
    if start - lead + len(window) >= size:
        # What was read holds the end of the file.  What follows the last secret that
        # is all there may be the start of the next: the piece ends before it, also
        # where its size would have cut it inside that start.
        whole = max((last for _, last in spans), default=0)
        begun = len(window) - _begun(window[whole:], needles)
        if begun < end:
            end, waits = max(begin, begun), True
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    decoder.decode(window[begin:end], final=False)
    end -= len(decoder.getstate()[0])  # half a character at the end waits for its rest
    shown, cursor = [], begin
    for first, last in spans if end > begin else ():  # an empty piece shows nothing
        if last <= begin or first >= end:
            continue
        shown += [window[cursor : max(first, cursor)], _REDACTED.encode()]
        cursor = min(last, end)
    shown.append(window[cursor:end])
    return {
        "offset": start - lead + begin,
        "next": start - lead + end,
        "size": size,
        "text": b"".join(shown).decode("utf-8", errors="replace"),
        "eof": waits,
        "reset": reset,
    }


def _log_piece(
    context: Context,
    project: Project,
    relative: str,
    offset: int,
    *,
    limit: int = _LOG_CHUNK_BYTES,
    whole_lines: bool = False,
) -> dict[str, Any]:
    """A piece of the project's log file ``relative`` (see :func:`_read_log`), with the
    credentials of the server and of the project's repository blanked out.

    The file is opened inside the project or not at all: a project directory may
    come from somebody else, with its log linked to a file of the person's.  A
    log that is absent reads as empty; one that is a link out of the project, or
    not a file, reads as empty too and ``problem`` says why.
    """
    try:
        handle = open_confined(project.root, relative)
    except (FileNotFoundError, NotADirectoryError):
        return dict(_NO_LOG)
    except (PathEscape, OSError) as exc:
        reason = str(exc) if isinstance(exc, PathEscape) else type(exc).__name__
        problem = (
            f"{relative} of this project is not read: {reason}. The server opens a log "
            "only as an ordinary file inside the project directory; remove the link (or "
            "whatever is there) and the next worker writes a new log."
        )
        return {**_NO_LOG, "problem": problem}
    with handle:
        return _read_log(
            handle,
            offset,
            limit,
            whole_lines=whole_lines,
            secrets=context.services.secrets(project),
        )


def _run_log(experiment_id: str, number: int) -> str:
    """Where the log of a run is, relative to the project."""
    return f"experiments/{experiment_id}/runs/{number}/log.txt"


@router.get("/projects/{pid}/experiments/{eid}/runs/{number}")
def get_run(eid: str, number: int, project: Proj, context: Ctx) -> dict[str, Any]:
    """``run`` (the record of ``run.json`` with the addresses of its artifacts) and
    ``log``: the end of the log, with the offset to continue from."""
    page = _experiment(project, eid)
    run = _find_run(context, project, str(page.id), number)
    tail = _log_piece(context, project, _run_log(str(page.id), number), -_LOG_TAIL_BYTES)
    return {"run": _run_view(context, project, str(page.id), run), "log": tail}


@router.get("/projects/{pid}/experiments/{eid}/runs/{number}/log")
def get_log(eid: str, number: int, project: Proj, context: Ctx, offset: int = 0) -> dict[str, Any]:
    """The run's log from byte ``offset``: ``{"offset", "next", "size", "text", "eof",
    "reset"}``.  Ask again with ``offset=next`` to follow it; ``reset`` says the log
    was started anew and ``text`` is from its beginning.  A negative ``offset`` reads
    that many bytes from the end.

    Credentials a job printed are blanked out in ``text``, also where a piece
    begins or ends inside one; the offsets are those of the file.  A log that
    ends in the first part of a credential still being written is given up to
    there (``next`` stays before it, ``eof`` is true).  The log is read only as a
    file inside the project: for one that is a link leading out of it, ``text``
    is empty and ``problem`` says why."""
    page = _experiment(project, eid)
    if number < 1:
        raise ApiError(404, "not_found", f"{page.id} has no run {number}.")
    return _log_piece(context, project, _run_log(str(page.id), number), offset)


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


def _holds_any(path: Path, needles: Sequence[bytes], piece: int = 1 << 20) -> bool:
    """Whether the file holds one of ``needles``, read in pieces that overlap by what a
    needle may span."""
    margin = max((len(needle) for needle in needles), default=1) - 1
    carried = b""
    with path.open("rb") as handle:
        while data := handle.read(piece):
            window = carried + data
            if any(needle in window for needle in needles):
                return True
            carried = window[max(0, len(window) - margin) :] if margin else b""
    return False


def _blanked(path: Path, needles: Sequence[bytes], piece: int = 1 << 20) -> Iterator[bytes]:
    """The file's content with every one of ``needles`` (longest first) replaced by
    ``[redacted]``, in pieces; the end of each piece is kept back until the next one
    shows that no needle continues across it."""
    margin = max((len(needle) for needle in needles), default=1) - 1
    carried = b""
    with path.open("rb") as handle:
        while data := handle.read(piece):
            window = carried + data
            for needle in needles:
                window = window.replace(needle, _REDACTED.encode())
            keep = min(margin, len(window))
            carried = window[len(window) - keep :]
            yield window[: len(window) - keep]
    yield carried


def _disposition(name: str, attach: bool) -> str:
    """The ``Content-Disposition`` of a file called ``name``, as the framework writes it
    for a file it serves itself."""
    kind = "attachment" if attach else "inline"
    quoted = quote(name)
    return f'{kind}; filename="{name}"' if quoted == name else f"{kind}; filename*=utf-8''{quoted}"


@router.get("/projects/{pid}/files/{path:path}")
def get_file(path: str, project: Proj, context: Ctx, download: str | None = None) -> Response:
    """A file of the project under ``inputs/``, ``experiments/`` or ``paper/``.

    The path is resolved inside the project, symbolic links included.  Images and
    PDFs are shown; text of any kind is served as plain text; everything else is a
    download (as is anything with ``?download=1``).

    A text file that holds a credential of the server or the token of the
    project's repository (the whole log of a run, its record, something a job
    wrote into an artifact) is served with it replaced by ``[redacted]``, like
    every other answer, also where it stands escaped as JSON writes it; such an
    answer has no length and cannot be resumed.  An SVG drawing is text in this
    sense.  What is not
    text (an archive, a model file) is the job's own data and is handed over as
    it is.
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
    if media_type.startswith("text/plain") or suffix == ".svg":  # a drawing is text too
        needles = _written_forms(context.services.secrets(project))
        if needles and _holds_any(real, needles):
            headers["Content-Disposition"] = _disposition(real.name, attach)
            return StreamingResponse(
                _blanked(real, needles), media_type=media_type, headers=headers
            )
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


def _last_write_up(context: Context, project: Project, running: bool) -> dict[str, Any] | None:
    """How the project's last write-up went, from its ``paper.`` events; ``None`` when
    none is on record.

    ``outcome`` is ``written``, ``failed`` (it ran and no paper came of it:
    ``paper/state.json`` says where), ``stopped`` (it was interrupted: starting it
    again continues it), ``refused`` (it ended before it began: nothing to write a
    paper from, a setting that cannot be read; ``summary`` and ``error`` say why),
    ``running``, or ``unfinished`` (it began and nothing says how it ended: its
    worker is gone).  ``exit_code`` is that of the worker where the server recorded
    the end for a worker that left no account of its own (:mod:`workers`), ``None``
    where the write-up told its own end.
    """
    started, finished = context.paper.last(project.root)
    if started is None and finished is None:
        return None
    ended = finished["data"] if finished else {}
    if finished is None:
        outcome = "running" if running else "unfinished"
    elif ended.get("success") is True:
        outcome = "written"
    elif ended.get("interrupted") is True:
        outcome = "stopped"
    else:
        # An end without a beginning of its own is a write-up that never got that far.
        outcome = "refused" if started is None or ended.get("started") is False else "failed"
    return {
        "outcome": outcome,
        "started_at": started["at"] if started else None,
        "started_summary": started["summary"] if started else None,
        "template": started["data"].get("template") if started else None,
        "ended_at": finished["at"] if finished else None,
        "summary": finished["summary"] if finished else None,
        "error": ended.get("error"),
        "exit_code": ended.get("exit_code"),
        "pdf": ended.get("pdf") if ended.get("success") is True else None,
    }


def _paper_view(context: Context, project: Project) -> dict[str, Any]:
    paper = project.root / PAPER_DIR
    state = _read_json(project.root, f"{PAPER_DIR}/{STATE_FILE}")
    worker = context.workers.status(project.root)
    final = paper / "final_paper.pdf"
    missing = context.services.paper_missing(project)
    running = worker["state"] != "idle" and worker["kind"] == "write"
    return {
        "exists": paper.is_dir(),
        "state": state if isinstance(state, dict) else None,
        "running": running,
        "final_pdf": f"{PAPER_DIR}/final_paper.pdf" if final.is_file() else None,
        # What a write-up would refuse for ("results", "finding"), or None: it can start.
        "missing": missing,
        # Which accepted findings were not counted for that, and why, in a sentence.
        "missing_detail": str(getattr(missing, "detail", "") or ""),
        # The accepted findings a paper would not report, whether or not it can start.
        "left_out": context.services.paper_left_out(project),
        # How the last write-up went, as the project's events tell it.
        "last": _last_write_up(context, project, running),
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
    ``"finding"``) or ``null`` when it can be written.

    ``left_out`` lists the accepted findings a paper would not report, each as
    ``{"id", "path", "title", "reason"}``: every record such a finding is based on
    belongs to an experiment that was rejected or is gone.  ``missing_detail`` says
    the same in a sentence where that is why the paper cannot be written yet (empty
    otherwise).

    ``last`` says how the last write-up went, read from the project's ``paper.``
    events (so that it is the same after a restart of the server), or ``null``
    when none is on record: ``{"outcome", "started_at", "started_summary",
    "template", "ended_at", "summary", "error", "exit_code", "pdf"}``.  ``outcome``
    is ``"written"``, ``"failed"``, ``"stopped"``, ``"refused"`` (it ended before it
    began, for the reason ``summary`` and ``error`` give), ``"running"`` or
    ``"unfinished"`` (it began, and nothing says how it ended).  ``started_*`` are
    those of its ``paper.started`` event (``null`` for a write-up that was refused
    before it began), ``ended_at`` and ``summary`` those of its ``paper.finished``.
    ``exit_code`` is ``null`` unless the account of the end is the server's: the
    worker ended without one of its own (it refused before it began, or was
    killed: a negative code), and ``summary`` then says what happened and what to
    do.  ``paper/state.json`` of a write-up that was killed still reads as it did
    in that moment; ``last`` is what says that it is over.
    """
    return _paper_view(context, project)
