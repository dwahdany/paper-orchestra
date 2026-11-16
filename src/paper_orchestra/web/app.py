"""The web app: the API, the static frontend, and the server that runs them.

``create_app`` builds the ASGI application for a workspace (a directory of
project directories); ``serve`` runs it with uvicorn, which is what
``paper-orchestra ui`` calls.

The server ``serve`` runs always asks for an access token, on loopback too:
the link it prints once at start carries the token, and a browser it opens is
handed that link through a file only the person can read (see
``_BrowserOpener``), never as an argument of a process.
"""

from __future__ import annotations

import atexit
import contextlib
import errno
import html
import inspect
import logging
import os
import re
import shutil
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from pathlib import Path
from types import FrameType
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.routing import Match

from paper_orchestra.web import api as api_module
from paper_orchestra.web.api import (
    Context,
    LiveSettings,
    Services,
    UploadLimits,
    Workspace,
    install_error_handlers,
    router,
)
from paper_orchestra.web.security import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_BODY_BYTES,
    MIN_TOKEN_CHARS,
    TOKEN_ENV,
    TOKEN_PARAM,
    PathEscape,
    SecurityMiddleware,
    confined,
    fixed_token,
    is_loopback,
    new_token,
    offers_token,
)
from paper_orchestra.web.workers import GRACE_SECONDS, Launcher, Workers

__all__ = ["OPENER_SECONDS", "STAGING_DIR", "STATIC_DIR", "create_app", "serve", "sweep_staging"]

log = logging.getLogger(__name__)

#: The frontend: static files shipped in the wheel (HTML, CSS, ES modules; no build step).
STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Where the API keeps the files of a project that is being created, inside the workspace.
STAGING_DIR: str = getattr(api_module, "_STAGING_DIR", ".incoming")
#: A staged upload that nothing has touched for this long belongs to a server that was
#: killed while it took the upload in.  Far longer than the largest upload takes.
STAGING_STALE_SECONDS = 3600.0
#: What the API calls the directory of one upload: sixteen hexadecimal digits.
_STAGED_UPLOAD = re.compile(r"[0-9a-f]{16}")

#: How long the file that hands the link to a browser exists, in seconds: long enough
#: for a browser that is only just starting to read it, and no longer.
OPENER_SECONDS = 30.0
#: That file.  It sends the browser on to the link at once (and replaces itself in the
#: history of the tab), and tells the app's address nothing about where it came from.
_OPENER_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="referrer" content="no-referrer">
<meta http-equiv="refresh" content="0; url={link}">
<title>Opening PaperOrchestra</title>
</head>
<body>
<p>Opening PaperOrchestra. If nothing happens,
<a rel="noreferrer" href="{link}">open it here</a>.</p>
</body>
</html>
"""

_FRONTEND_MISSING = (
    "PaperOrchestra is running, but the files of its web interface are not installed "
    "(paper_orchestra/web/static/index.html is missing).\n"
    "Reinstall the package, or use the API under /api/ and the command line.\n"
)

#: Stated here rather than guessed from the system's tables, which differ between
#: machines: a module script is only run when it is served as JavaScript.
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".webmanifest": "application/manifest+json",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def _static_file(request: Request, file: Path, *, cache: str) -> Response:
    """``file`` with its content type and a validator, or 304 when the browser has it."""
    stat = file.stat()
    etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    headers = {"Cache-Control": cache, "ETag": etag}
    if cache != "no-store" and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    media_type = _STATIC_TYPES.get(file.suffix.lower(), "application/octet-stream")
    return FileResponse(file, media_type=media_type, headers=headers, stat_result=stat)


def _frontend(static: Path) -> Callable[..., Response]:
    """The handler for everything that is not the API: the frontend's files, and its
    ``index.html`` for any other address, so that a reload of a client-side route works."""

    def serve_frontend(request: Request, path: str = "") -> Response:
        if path == "api" or path.startswith("api/"):
            # This route takes every method, so that an unknown address is "not found"
            # whatever was tried on it; an endpoint that exists but does not take the
            # method still says so.
            for route in request.app.router.routes:
                if route.matches(request.scope)[0] is Match.PARTIAL:
                    raise HTTPException(405, f"/{path} does not take {request.method}.")
            raise HTTPException(404, f"There is no API endpoint /{path}.")
        if request.method not in ("GET", "HEAD"):
            raise HTTPException(405, "The pages of the app can only be read.")
        index = static / "index.html"
        if not index.is_file():
            return PlainTextResponse(_FRONTEND_MISSING, status_code=503)
        if path:
            try:
                file = confined(static, path)
            except PathEscape:
                file = None
            if file is not None and file.is_file():
                # Always revalidated: after an upgrade the browser must not run old code.
                return _static_file(request, file, cache="no-cache")
            if "." in path.rsplit("/", 1)[-1]:
                # A missing script or image must not be answered with a page of HTML.
                raise HTTPException(404, f"There is no file /{path}.")
        return _static_file(request, index, cache="no-store")

    return serve_frontend


def _name_too_long(request: Request, exc: OSError) -> Response:
    """An address or a name longer than the file system has names for.

    Nothing is stored under such a name, so reading it finds nothing, and storing
    it is refused; every other error of the operating system stays what it was,
    an unexpected error.
    """
    if exc.errno != errno.ENAMETOOLONG:
        raise exc
    if request.method in ("GET", "HEAD"):
        status, code = 404, "not_found"
        message = (
            "There is nothing at this address: a part of it is longer than any name in a project."
        )
    else:
        status, code = 422, "invalid"
        message = "A name in this request is longer than a file name can be. Use a shorter one."
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _last_touched(directory: str) -> float:
    """When anything in ``directory`` was last written, the directory itself included."""
    newest = os.lstat(directory).st_mtime
    for folder, folders, files in os.walk(directory):
        for name in (*folders, *files):
            with contextlib.suppress(OSError):
                newest = max(newest, os.lstat(os.path.join(folder, name)).st_mtime)
    return newest


def sweep_staging(
    workspace: str | Path, *, older_than: float = STAGING_STALE_SECONDS
) -> list[Path]:
    """Remove what a server that was killed while it took in an upload left behind.

    A new project's files are staged in ``<workspace>/.incoming/<random name>`` and
    moved from there; the directory is removed when the request ends, which a
    killed server never reaches.  Only directories with the API's own kind of
    name in which nothing was written for ``older_than`` seconds are removed
    (another server on this workspace may be staging an upload right now);
    links are not followed.  Returns what was removed.
    """
    staging = Path(workspace) / STAGING_DIR
    removed: list[Path] = []
    try:
        if staging.is_symlink() or not staging.is_dir():
            return removed
        entries = list(os.scandir(staging))
    except OSError:
        return removed
    now = time.time()
    for entry in entries:
        try:
            ours = _STAGED_UPLOAD.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False)
            if not ours or now - _last_touched(entry.path) < older_than:
                continue
        except OSError:
            continue
        shutil.rmtree(entry.path, ignore_errors=True)
        if not os.path.lexists(entry.path):
            removed.append(Path(entry.path))
    with contextlib.suppress(OSError):
        staging.rmdir()  # only when nothing is left in it
    if removed:
        log.info("Removed %d upload(s) an earlier server left half staged.", len(removed))
    return removed


def _work(worker: Mapping[str, Any]) -> str:
    """What a running worker is doing, for a line a person reads."""
    if worker.get("kind") == "advance":
        return f"the agents ({worker['phase']} phase)" if worker.get("phase") else "the agents"
    return {"run": "an experiment", "write": "the write-up of the paper"}.get(
        str(worker.get("kind")), "work"
    )


def _left_running(workers: list[dict[str, Any]]) -> str:
    """What the server says when it ends while workers are still at work."""
    if len(workers) == 1:
        first = (
            "The server has stopped; 1 worker keeps running. It belongs to its project, "
            "and the next `paper-orchestra ui` finds it again:"
        )
    else:
        first = (
            f"The server has stopped; {len(workers)} workers keep running. They belong to "
            "their projects, and the next `paper-orchestra ui` finds them again:"
        )
    lines = [first]
    lines += [
        f"  {Path(str(worker['dir'])).name}: {_work(worker)}, process {worker['pid']}"
        for worker in workers
    ]
    lines.append(
        "To stop one, start the app again and press Stop, or run `kill -INT <process>` "
        "with its number. With `paper-orchestra ui --stop-workers-on-exit` the server "
        "stops them when it ends."
    )
    return "\n".join(lines)


def _address(host: str, port: int, token: str | None = None) -> str:
    """The address a person opens: a wildcard address is reachable as ``localhost``."""
    name = "localhost" if host in ("", "0.0.0.0", "::") else host
    name = f"[{name}]" if ":" in name else name
    return f"http://{name}:{port}/" + (f"?{TOKEN_PARAM}={token}" if token else "")


def _free_port(host: str) -> int:
    """A port nothing listens on at the moment, for ``--port 0``."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


class _BrowserOpener:
    """Opens the app in the person's browser, a moment after the server has started.

    With an access token the browser is not given the link: an argument of a
    process can be read by every account of the machine for as long as the
    process lives.  It is given a file that leads to the link.  The file is the
    person's alone (mode 0600, in a directory of its own with mode 0700) and is
    removed ``OPENER_SECONDS`` after it was written or when the app ends,
    whichever comes first.  If it cannot be written no browser is opened, and
    ``say`` is told so; the link printed at start is then the way in.

    Without a token (an app built for an in-process test) the plain address is
    opened.
    """

    def __init__(self, address: str, link: str | None, say: Callable[[str], None]) -> None:
        self._address = address
        self._link = link
        self._say = say
        self._lock = threading.Lock()
        self._timers: list[threading.Timer] = []
        self._directory: Path | None = None
        self._closed = False

    def start(self) -> None:
        # A moment later, so that the server answers when the page asks.
        self._later(0.5, self._open)

    def close(self) -> None:
        """The app has ended: nothing more is opened, and the file is gone."""
        with self._lock:
            self._closed = True
            timers, self._timers = self._timers, []
        for timer in timers:
            timer.cancel()
        self._remove()

    def _later(self, seconds: float, call: Callable[[], None]) -> None:
        timer = threading.Timer(seconds, call)
        timer.daemon = True
        with self._lock:
            if self._closed:
                return
            self._timers.append(timer)
        timer.start()

    def _open(self) -> None:
        if self._link is None:
            webbrowser.open(self._address)
            return
        try:
            page = self._write()
        except OSError as exc:
            self._remove()
            self._say(
                "The browser was not opened: the file that hands it the link could not "
                f"be written ({exc}). Open the link that was printed at start yourself."
            )
            return
        if page is None:
            return
        # Counted from now, not from when the browser returns: a browser started by this
        # call may only return when the person closes it.
        self._later(OPENER_SECONDS, self._remove)
        began = time.monotonic()
        try:
            opened = webbrowser.open(page.as_uri())
        except Exception:  # whatever a browser's launcher may raise
            log.debug("The browser could not be opened.", exc_info=True)
            opened = False
        if not opened and time.monotonic() - began < OPENER_SECONDS:
            self._remove()
            self._say(
                "No browser could be started here. Open the link that was printed at "
                "start yourself."
            )

    def _write(self) -> Path | None:
        """Write the file and return where it is; ``None`` when the app has ended."""
        directory = Path(tempfile.mkdtemp(prefix="paper-orchestra-ui-"))  # mode 0700
        with self._lock:
            closed = self._closed
            if not closed:
                self._directory = directory
        if closed:
            shutil.rmtree(directory, ignore_errors=True)
            return None
        # Should the process end without the app's own ending (an error at start).
        atexit.register(self._remove)
        page = directory / "open.html"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(page, flags, 0o600), "w", encoding="utf-8") as handle:
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o600)  # whatever the umask says
            handle.write(_OPENER_PAGE.format(link=html.escape(str(self._link), quote=True)))
        return page

    def _remove(self) -> None:
        with self._lock:
            directory, self._directory = self._directory, None
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
            atexit.unregister(self._remove)


def _output_is_shared() -> bool:
    """Whether what the server prints goes to a file that other users of the machine can
    read (`paper-orchestra ui > ui.log` under the usual umask).  A terminal, a pipe and a
    file of the person's alone are not; where there are no such permissions, nothing is."""
    if os.name != "posix":
        return False
    try:
        status = os.fstat(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return False
    return stat.S_ISREG(status.st_mode) and bool(status.st_mode & 0o044)


def _how_to_open(
    address: str, link: str, *, opens: bool, fixed: bool, shared: bool, beyond: bool
) -> str:
    """What the server says at start about the way into the app.

    ``link`` is the address with the access token.  It is said once, here, unless
    the output is ``shared`` (other users can read it): then the token is left
    out and the message says how to get in without it being written down.
    ``opens``: a browser is being opened; ``fixed``: the token is the one the
    person set in the environment; ``beyond``: the address is not loopback.
    """
    own = f"set {TOKEN_ENV} to a token of your own ({MIN_TOKEN_CHARS} characters or more)"
    if shared:
        lines = [
            f"PaperOrchestra is running at {address}",
            "The link that opens it carries the access token and is not written here: "
            "this output goes to a file that other users of this machine can read.",
        ]
        if fixed:
            lines.append(f"Open {address}?{TOKEN_PARAM}= followed by the token in {TOKEN_ENV}.")
        else:
            start = "It opens in your browser. For a link in writing" if opens else "To open it"
            lines.append(
                f"{start}, start the server in a terminal, or {own} and open "
                f"{address}?{TOKEN_PARAM}=<that token>."
            )
    else:
        first = (
            "PaperOrchestra is running and opens in your browser. If it does not, open "
            "this link, which carries the access token:"
            if opens
            else "PaperOrchestra is running. Open it with this link, which carries the "
            "access token:"
        )
        kept = (
            f"The token is the one in {TOKEN_ENV}, so the link stays good across restarts."
            if fixed
            else f"Every start makes a new token; {own} to keep one across restarts."
        )
        lines = [
            first,
            f"  {link}",
            "The link is shown only here, once. Whoever has it uses the app as you, with "
            "your API credentials: keep it to yourself. Nobody else can open the app "
            f"without it, on this machine or from another. {kept}",
        ]
    if beyond:
        lines.append(
            "This address can be reached from other machines: the token is all that keeps them out."
        )
    return "\n".join(lines)


def _not_started(address: str) -> str:
    """What the server says when it ends before it could listen (the reason is in the
    line the web server wrote before): no link is given, because none leads to this app."""
    return (
        f"PaperOrchestra did not start: it could not listen at {address} (the line above "
        "says why). If that address is in use, another PaperOrchestra may be running "
        "there: open that one with its own link, or stop it. Otherwise give this server "
        "another port with --port."
    )


def _untraced_link() -> dict[str, Any]:
    """The argument that keeps the link out of the framework's own tracing.

    A FastAPI that traces requests by itself (where OpenTelemetry is set up for
    the process) does so from outside the security layer and records the address
    of each request in full, the access token of the link included; what it
    records goes where the person's set-up sends it, by default to a port of
    this machine.  Requests whose address names a token are kept from it.  A
    FastAPI without such tracing does not know the argument and is not given it.
    """
    if "telemetry" not in inspect.signature(FastAPI.__init__).parameters:
        return {}
    return {"telemetry": {"exclude": offers_token}}


def create_app(
    workspace: str | Path,
    *,
    launcher: Launcher | None = None,
    open_browser: bool = False,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: str | None = None,
    allowed_hosts: Iterable[str] = (),
    services: Services | None = None,
    uploads: UploadLimits | None = None,
    live: LiveSettings | None = None,
    static_dir: str | Path | None = None,
    grace: float = GRACE_SECONDS,
    stop_workers_on_exit: bool = False,
    max_body_bytes: int = MAX_BODY_BYTES,
    report: Callable[[str], None] | None = None,
) -> FastAPI:
    """The web app for the projects in ``workspace``.

    ``launcher`` builds the command line of a worker process (see
    :mod:`paper_orchestra.web.workers`); ``open_browser`` opens the app in the
    person's browser when the app starts (see ``_BrowserOpener``; ``serve`` does
    that itself instead, once it listens at its address).
    ``host`` and ``port`` are where the server will listen: requests whose
    ``Host`` is not that address (or loopback) are refused.

    ``token`` is the access token every request must then carry (see
    :class:`~paper_orchestra.web.security.SecurityMiddleware`).  ``serve``
    always gives one.  ``None`` means that no token is asked for: that is for
    in-process tests, which reach the app without a network, and for nothing
    else.

    **How the app ends.**  Its end never stops a worker: agents and experiments
    belong to their project, go on without the server, and are found again by
    the next one (a stop the person asked for before is carried out first).
    ``report`` is called with what a person should know then (which workers keep
    running, and how to stop one); without it that is logged.  With
    ``stop_workers_on_exit`` the workers that are this server's are interrupted
    (and killed after ``grace`` seconds) before the app has ended.

    ``max_body_bytes`` bounds every request body but the upload of a new
    project, which is bounded by ``uploads``.  The remaining arguments replace
    parts for tests: the calls into other modules, the event stream's timing,
    and the frontend directory.

    A test client must name a host the app accepts:
    ``TestClient(app, base_url="http://127.0.0.1:8600")``.
    """
    root = Path(workspace).expanduser()
    limits = uploads or UploadLimits()
    context = Context(
        workspace=Workspace(root),
        workers=Workers(launcher=launcher, grace=grace, stop_on_exit=stop_workers_on_exit),
        services=services or Services(),
        live=live or LiveSettings(),
        uploads=limits,
    )

    def say(message: str) -> None:
        if report is None:
            log.info("%s", message)
            return
        # The terminal may be the very thing that went away.
        with contextlib.suppress(OSError, ValueError):
            report(message)

    def end_workers() -> None:
        if not stop_workers_on_exit:
            asked = context.workers.stopping()
            if asked:
                say(
                    f"Waiting for {asked} worker{'s' if asked != 1 else ''} you asked to "
                    f"stop; this can take up to {grace:.0f} seconds."
                )
            left = context.workers.close()
            if left:
                say(_left_running(left))
            return
        count = context.workers.live()
        if count:
            say(
                f"Stopping {count} running worker{'s' if count != 1 else ''} with the "
                f"server; this can take up to {grace:.0f} seconds."
            )
        context.workers.shutdown()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        context.live.closing.clear()
        context.workers.reopen()
        # Off the event loop and not waited for: a large upload left behind takes a
        # moment to remove, and the server answers meanwhile.
        threading.Thread(
            target=sweep_staging, args=(root,), name="sweep-staging", daemon=True
        ).start()
        opener = None
        if open_browser:
            link = _address(host, port, token) if token else None
            opener = _BrowserOpener(_address(host, port), link, say)
            opener.start()
        try:
            yield
        finally:
            if opener is not None:
                opener.close()
            context.live.closing.set()  # streams end at their next look
            await run_in_threadpool(end_workers)

    app = FastAPI(
        title="PaperOrchestra",
        lifespan=lifespan,
        # The interactive documentation loads its scripts from other sites.
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        **_untraced_link(),
    )
    app.state.context = context
    install_error_handlers(app)
    app.add_exception_handler(OSError, _name_too_long)
    app.include_router(router)
    app.add_api_route(
        "/{path:path}",
        _frontend(Path(static_dir) if static_dir is not None else STATIC_DIR),
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    # The form around the files (the brief, the part headers) is allowed for on top, and
    # a little more than the API's own count allows, so that its refusal, which names
    # the limit for a project, is the one a person reads.
    upload = limits.total_bytes + limits.text_bytes + 2 * 1024 * 1024
    app.add_middleware(
        SecurityMiddleware,
        host=host,
        token=token,
        allowed_hosts=allowed_hosts,
        max_body_bytes=max_body_bytes,
        body_limits={("POST", "/api/projects"): max(upload, max_body_bytes)},
    )
    return app


def serve(
    workspace: str | Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    stop_workers_on_exit: bool = False,
) -> None:
    """Run the web app until it is told to stop (``paper-orchestra ui``).

    Listens on loopback unless ``host`` says otherwise.  On every address,
    loopback included, the app asks for an access token: a fresh random one per
    start, or the value of ``PAPER_ORCHESTRA_UI_TOKEN`` when that is set (an
    empty value is none; one that cannot serve as a token raises
    :class:`~paper_orchestra.web.security.TokenError` before anything is
    started).  The variable is taken out of this process's environment, so that
    no worker, job or browser inherits it.  The link that carries the token is
    printed once, and with ``open_browser`` the browser is handed it through a
    file only the person can read.  Both happen only when the server listens at
    its address: if it cannot (another program holds the port), it says so, gives
    no link and ends.  Where the output goes to a file that other users can
    read, the token is not printed and the message says what to do.

    However the server ends (Ctrl-C, a termination signal, the terminal or the
    SSH session going away), the agents and experiments that are running go on:
    they belong to their projects, and the next server finds them.  The server
    prints which ones those are and how to stop one.  With
    ``stop_workers_on_exit`` (``--stop-workers-on-exit``) they are stopped with
    the server instead, in all three cases.
    """
    import uvicorn

    fixed = fixed_token()  # first: a token that cannot be one stops the start
    token = fixed or new_token()
    workspace = Path(workspace).expanduser()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"The workspace {workspace} cannot be created ({exc}). Choose a directory you "
            "can write to with --workspace."
        ) from exc
    if port == 0:
        port = _free_port(host)  # chosen here, so that the address printed is the real one

    def say(message: str) -> None:
        # At once: a termination signal ends the process before a buffer is written.
        print(message, flush=True)

    app = create_app(
        workspace,
        host=host,
        port=port,
        token=token,
        open_browser=False,  # opened from here, once the server listens (see ``announce``)
        stop_workers_on_exit=stop_workers_on_exit,
        report=say,
    )
    closing = app.state.context.live.closing
    opener = (
        _BrowserOpener(_address(host, port), _address(host, port, token), say)
        if open_browser
        else None
    )
    if stop_workers_on_exit:
        ending = "Press Ctrl-C to stop it, and with it the agents and experiments it runs."
    else:
        ending = (
            "Press Ctrl-C to stop the server. Agents and experiments that are running "
            "then go on without it, and the next start finds them."
        )

    def announce() -> None:
        """Say how to get into the app, and send the browser there."""
        say(
            _how_to_open(
                _address(host, port),
                _address(host, port, token),
                opens=open_browser,
                fixed=fixed is not None,
                shared=_output_is_shared(),
                beyond=not is_loopback(host),
            )
        )
        say(f"Projects are kept in {workspace.resolve()}. {ending}")
        if opener is not None:
            opener.start()

    class Server(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            # Nothing is said about the address, and no browser is sent to it, before the
            # server listens there.  Where another program holds the port (on a shared
            # machine: another account's), the link would hand that program the token.
            await super().startup(sockets=sockets)
            if self.started:
                announce()

        async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
            if opener is not None:
                opener.close()  # first: ending the workers may take a while
            await super().shutdown(sockets=sockets)

        def handle_exit(self, sig: int, frame: FrameType | None) -> None:
            # uvicorn waits for open connections before it shuts the app down, and an
            # event stream never ends by itself: tell the streams first.
            closing.set()
            super().handle_exit(sig, frame)

        def hang_up(self, sig: int, frame: FrameType | None) -> None:
            """The terminal went away: end as on a termination signal, not at once and
            without a word, so that there is one rule for what happens to the workers."""
            closing.set()
            self.should_exit = True

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        server_header=False,
        timeout_graceful_shutdown=5,
    )
    server = Server(config)
    try:
        with _hang_up_handled(server.hang_up), contextlib.suppress(KeyboardInterrupt):
            server.run()
    except SystemExit:
        if not server.started:
            say(_not_started(_address(host, port)))
        raise
    finally:
        if opener is not None:
            opener.close()


@contextlib.contextmanager
def _hang_up_handled(handler: Callable[[int, FrameType | None], None]) -> Iterator[None]:
    """Call ``handler`` when the terminal hangs up, for as long as the block runs.

    Not where that signal is ignored (the server was started with ``nohup``: it is
    meant to outlive its terminal), where there is no such signal (Windows), or
    off the main thread, where signals cannot be handled.
    """
    hang_up = getattr(signal, "SIGHUP", None)
    usable = hang_up is not None and threading.current_thread() is threading.main_thread()
    if not usable or signal.getsignal(hang_up) is signal.SIG_IGN:
        yield
        return
    previous = signal.signal(hang_up, handler)
    try:
        yield
    finally:
        signal.signal(hang_up, previous)
