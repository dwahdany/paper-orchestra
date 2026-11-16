"""The web app: the API, the static frontend, and the server that runs them.

``create_app`` builds the ASGI application for a workspace (a directory of
project directories); ``serve`` runs it with uvicorn, which is what
``paper-orchestra ui`` calls.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import re
import shutil
import signal
import socket
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
    TOKEN_PARAM,
    PathEscape,
    SecurityMiddleware,
    confined,
    is_loopback,
    new_token,
)
from paper_orchestra.web.workers import GRACE_SECONDS, Launcher, Workers

__all__ = ["STAGING_DIR", "STATIC_DIR", "create_app", "serve", "sweep_staging"]

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
    :mod:`paper_orchestra.web.workers`); ``open_browser`` opens the app's address
    when the server has started.  ``host`` and ``port`` are where the server will
    listen: requests whose ``Host`` is not that address (or loopback) are
    refused, and ``token`` is required when given (``serve`` generates one for
    any address but loopback).

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
        if open_browser:
            # A moment later, so that the server answers when the page asks.
            timer = threading.Timer(0.5, webbrowser.open, args=(_address(host, port, token),))
            timer.daemon = True
            timer.start()
        try:
            yield
        finally:
            context.live.closing.set()  # streams end at their next look
            await run_in_threadpool(end_workers)

    app = FastAPI(
        title="PaperOrchestra",
        lifespan=lifespan,
        # The interactive documentation loads its scripts from other sites.
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
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

    Listens on loopback unless ``host`` says otherwise.  On any other address an
    access token is generated and the link that carries it is printed once.

    However the server ends (Ctrl-C, a termination signal, the terminal or the
    SSH session going away), the agents and experiments that are running go on:
    they belong to their projects, and the next server finds them.  The server
    prints which ones those are and how to stop one.  With
    ``stop_workers_on_exit`` (``--stop-workers-on-exit``) they are stopped with
    the server instead, in all three cases.
    """
    import uvicorn

    workspace = Path(workspace).expanduser()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"The workspace {workspace} cannot be created ({exc}). Choose a directory you "
            "can write to with --workspace."
        ) from exc
    token = None if is_loopback(host) else new_token()
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
        open_browser=open_browser,
        stop_workers_on_exit=stop_workers_on_exit,
        report=say,
    )
    closing = app.state.context.live.closing

    class Server(uvicorn.Server):
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

    say(f"PaperOrchestra is running at {_address(host, port)}")
    if stop_workers_on_exit:
        ending = "Press Ctrl-C to stop it, and with it the agents and experiments it runs."
    else:
        ending = (
            "Press Ctrl-C to stop the server. Agents and experiments that are running "
            "then go on without it, and the next start finds them."
        )
    say(f"Projects are kept in {workspace.resolve()}. {ending}")
    if token is not None:
        say(
            "This address can be reached from other machines, and whoever opens it uses "
            "your API credentials through the app. Open it with this link, which carries "
            f"the access token:\n  {_address(host, port, token)}"
        )
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
    with _hang_up_handled(server.hang_up), contextlib.suppress(KeyboardInterrupt):
        server.run()


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
