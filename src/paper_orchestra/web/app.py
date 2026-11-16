"""The web app: the API, the static frontend, and the server that runs them.

``create_app`` builds the ASGI application for a workspace (a directory of
project directories); ``serve`` runs it with uvicorn, which is what
``paper-orchestra ui`` calls.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from types import FrameType

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.routing import Match

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
    TOKEN_PARAM,
    PathEscape,
    SecurityMiddleware,
    confined,
    is_loopback,
    new_token,
)
from paper_orchestra.web.workers import GRACE_SECONDS, Launcher, Workers

__all__ = ["STATIC_DIR", "create_app", "serve"]

log = logging.getLogger(__name__)

#: The frontend: static files shipped in the wheel (HTML, CSS, ES modules; no build step).
STATIC_DIR = Path(__file__).resolve().parent / "static"

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
) -> FastAPI:
    """The web app for the projects in ``workspace``.

    ``launcher`` builds the command line of a worker process (see
    :mod:`paper_orchestra.web.workers`); ``open_browser`` opens the app's address
    when the server has started.  ``host`` and ``port`` are where the server will
    listen: requests whose ``Host`` is not that address (or loopback) are
    refused, and ``token`` is required when given (``serve`` generates one for
    any address but loopback).  The remaining arguments replace parts for tests:
    the calls into other modules, the upload limits, the event stream's timing,
    the frontend directory, and how long a worker gets to stop.

    A test client must name a host the app accepts:
    ``TestClient(app, base_url="http://127.0.0.1:8600")``.
    """
    context = Context(
        workspace=Workspace(Path(workspace).expanduser()),
        workers=Workers(launcher=launcher, grace=grace),
        services=services or Services(),
        live=live or LiveSettings(),
        uploads=uploads or UploadLimits(),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        context.live.closing.clear()
        context.workers.reopen()
        if open_browser:
            # A moment later, so that the server answers when the page asks.
            timer = threading.Timer(0.5, webbrowser.open, args=(_address(host, port, token),))
            timer.daemon = True
            timer.start()
        try:
            yield
        finally:
            # Streams end at their next look; workers are interrupted, then killed.
            context.live.closing.set()
            if context.workers.live():
                log.info(
                    "Stopping %d running agent process(es); this can take up to %.0f seconds.",
                    context.workers.live(),
                    grace,
                )
            await run_in_threadpool(context.workers.shutdown)

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
    app.include_router(router)
    app.add_api_route(
        "/{path:path}",
        _frontend(Path(static_dir) if static_dir is not None else STATIC_DIR),
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    app.add_middleware(SecurityMiddleware, host=host, token=token, allowed_hosts=allowed_hosts)
    return app


def serve(
    workspace: str | Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """Run the web app until it is interrupted (``paper-orchestra ui``).

    Listens on loopback unless ``host`` says otherwise.  On any other address an
    access token is generated and the link that carries it is printed once.
    Stopping the server (Ctrl-C) stops the agents it runs.
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
    app = create_app(workspace, host=host, port=port, token=token, open_browser=open_browser)
    closing = app.state.context.live.closing

    class Server(uvicorn.Server):
        def handle_exit(self, sig: int, frame: FrameType | None) -> None:
            # uvicorn waits for open connections before it shuts the app down, and an
            # event stream never ends by itself: tell the streams first.
            closing.set()
            super().handle_exit(sig, frame)

    print(f"PaperOrchestra is running at {_address(host, port)}")
    print(f"Projects are kept in {workspace.resolve()}. Press Ctrl-C to stop.")
    if token is not None:
        print(
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
    with contextlib.suppress(KeyboardInterrupt):
        Server(config).run()
