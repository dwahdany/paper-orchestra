"""What keeps a local web app local.

The server holds the person's API credentials (its workers spend them) and
serves pages whose text was written by models and copied from the web.  Four
things follow, all enforced here:

* **Who may talk to it.**  It binds to loopback unless told otherwise.  A request
  whose ``Host`` header is not the server's own address is refused: that is how
  a web page on another site reaches a local server through DNS rebinding.  A
  request that changes something and comes from another origin is refused too.
  When the server listens on anything but loopback it also wants a token, which
  the browser gets once from the link printed at start and keeps as a cookie.
* **What a response may do in the browser.**  Every response carries a
  Content-Security-Policy that allows the app's own scripts, styles and images
  and nothing else, and tells the browser not to guess content types.
* **Which files a URL can reach.**  A path from a URL is resolved inside the
  project directory, symbolic links included, or refused.
* **What an upload may be called.**  Names from the browser are reduced to plain
  relative paths.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, MutableMapping, Sequence
from http.cookies import CookieError, SimpleCookie
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

__all__ = [
    "CONTENT_SECURITY_POLICY",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "FILE_AREAS",
    "FILE_CONTENT_SECURITY_POLICY",
    "PDF_CONTENT_SECURITY_POLICY",
    "TOKEN_COOKIE",
    "PathEscape",
    "SecurityMiddleware",
    "confined",
    "is_loopback",
    "mask_secrets",
    "new_token",
    "safe_upload_path",
    "security_headers",
]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8600
TOKEN_COOKIE = "po_token"
TOKEN_PARAM = "token"

#: The parts of a project directory the files endpoint serves, and the only places a
#: rendered page may take an image from.
FILE_AREAS: tuple[str, ...] = ("inputs", "experiments", "paper")

#: For the app's own pages and API.  Scripts and styles come from the server's own files
#: only: no inline script, no inline event handler, no ``style`` attribute in markup (set
#: sizes through the CSSOM, ``element.style.width = ...``, which the policy allows).
#: Frames are allowed from the server itself so a PDF can be shown in the page.
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data: blob:",
        "font-src 'self'",
        "connect-src 'self'",
        "frame-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    )
)
#: For files of a project (inputs, artifacts): they were written by models, runs and
#: whoever made the attachments, so a file opened as a document must not be able to run
#: script with the app's origin.  ``sandbox`` gives it an origin of its own.
FILE_CONTENT_SECURITY_POLICY = "default-src 'none'; style-src 'unsafe-inline'; sandbox"
#: A PDF is shown by the browser's viewer, which a sandbox would switch off.
PDF_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; object-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'self'"
)

_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::", "*"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def is_loopback(host: str) -> bool:
    """Whether ``host`` (a bind address or a host name) reaches this machine only."""
    name = _bare_host(host)
    if name in _LOOPBACK_NAMES:
        return True
    try:
        # Only a literal address: ``127.0.0.1.example.com`` is somebody's host name.
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def new_token() -> str:
    """An access token for a server that listens beyond loopback."""
    return secrets.token_urlsafe(24)


def _bare_host(value: str) -> str:
    return str(value or "").strip().lower().strip("[]").rstrip(".")


def _split_host(value: str) -> tuple[str, int | None]:
    """``host`` and ``port`` of a ``Host`` header or a URL's network location."""
    value = str(value or "").strip().lower()
    port: int | None = None
    if value.startswith("["):
        name, _, rest = value[1:].partition("]")
        if rest.startswith(":") and rest[1:].isdigit():
            port = int(rest[1:])
        return name.rstrip("."), port
    name, colon, digits = value.rpartition(":")
    if colon and digits.isdigit() and ":" not in name:
        return name.rstrip("."), int(digits)
    return value.rstrip("."), None


# ---------------------------------------------------------------------------
# Paths from URLs
# ---------------------------------------------------------------------------


class PathEscape(ValueError):
    """A path from a URL does not name a place inside the directory it is confined to."""


def confined(root: Path, relative: str, *, areas: Sequence[str] | None = None) -> Path:
    """The real location of ``relative`` under ``root``; ``PathEscape`` when it has none.

    ``relative`` is a posix path from a URL.  It may not be absolute, climb with
    ``..`` or name anything hidden, and after every symbolic link on the way is
    followed the result must still lie inside ``root`` (and, with ``areas``,
    inside one of those top-level directories).  The file need not exist.
    """
    if not isinstance(relative, str) or not relative or "\x00" in relative or "\\" in relative:
        raise PathEscape("not a path inside the project")
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/"):
        raise PathEscape("not a path inside the project")
    if any(part in ("..", ".") or part.startswith(".") or ":" in part for part in parts):
        raise PathEscape("the path leaves the project or names a hidden file")
    base = Path(os.path.realpath(root))
    real = Path(os.path.realpath(base.joinpath(*parts)))
    if not real.is_relative_to(base) or real == base:
        raise PathEscape("the path leads outside the project")
    inside = real.relative_to(base).parts
    if any(part.startswith(".") for part in inside):
        raise PathEscape("the path names a hidden file")
    if areas is not None and (parts[0] not in areas or inside[0] not in areas):
        raise PathEscape(f"only files under {', '.join(areas)} are served")
    return real


# ---------------------------------------------------------------------------
# Upload names
# ---------------------------------------------------------------------------

_UNSAFE_NAME_CHARS = re.compile(r"[^\w.\-+()]+")
_WINDOWS_DEVICES = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)$", re.IGNORECASE)
_MAX_NAME_CHARS = 120
_MAX_UPLOAD_DEPTH = 8


def _safe_name(part: str) -> str | None:
    """One path segment as a plain file name, or ``None`` when nothing usable is left."""
    cleaned = _UNSAFE_NAME_CHARS.sub("_", part).strip("._ ")
    stem, suffix = os.path.splitext(cleaned)
    stem = stem.rstrip("._ ")
    if not any(character.isalnum() for character in stem):
        return None
    if _WINDOWS_DEVICES.match(stem):
        stem = f"_{stem}"
    return stem[: _MAX_NAME_CHARS - len(suffix[:20])] + suffix[:20]


def safe_upload_path(name: str) -> str | None:
    """A file name sent by a browser as a relative posix path that is safe to create.

    Directories in the name are kept (a folder dropped on the page arrives as
    ``folder/sub/file.csv``) but every segment is reduced to letters, digits and
    ``._-+()``: no ``..``, no absolute path, no hidden file, no control character,
    whichever slash the browser used.  ``None`` when no segment survives.
    """
    text = unicodedata.normalize("NFC", str(name or ""))
    parts = [_safe_name(part) for part in re.split(r"[\\/]+", text)]
    kept = [part for part in parts if part]
    return "/".join(kept[-_MAX_UPLOAD_DEPTH:]) or None


# ---------------------------------------------------------------------------
# Credentials in text that is shown
# ---------------------------------------------------------------------------


def mask_secrets(text: str) -> str:
    """``text`` with the values of this process's credentials blanked out.

    For output of worker processes before it is shown in the browser: they inherit
    the server's environment, and a library may echo a key in an error message.
    The rule for what counts as a credential is the one ``paper-orchestra doctor``
    applies to its own report.
    """
    from paper_orchestra.doctor import scrub, secret_values

    return str(scrub(text, secret_values(os.environ)))


# ---------------------------------------------------------------------------
# The middleware
# ---------------------------------------------------------------------------


def _plain(status: int, code: str, message: str, *, api: bool) -> tuple[int, bytes, bytes]:
    """Status, content type and body of a refusal: JSON for the API, text for a person."""
    if api:
        body = json.dumps({"error": {"code": code, "message": message}}).encode()
        return status, b"application/json", body
    return status, b"text/plain; charset=utf-8", (message + "\n").encode()


class SecurityMiddleware:
    """Checks ``Host``, ``Origin`` and the access token, and adds the security headers.

    Plain ASGI on purpose: nothing here buffers a response, so event streams pass
    through untouched.

    ``host`` is the address the server listens on.  For a loopback address the
    accepted ``Host`` names are the loopback ones; for a specific other address,
    that address as well; for a wildcard address any name is accepted, because the
    machine's names cannot be listed, and the token (required off loopback) is
    what stands between a stranger and the API.  ``allowed_hosts`` adds names, for
    a reverse proxy or a test client.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        host: str = DEFAULT_HOST,
        token: str | None = None,
        allowed_hosts: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.token = token or None
        bound = _bare_host(host)
        self.any_host = bound in _WILDCARD_HOSTS
        names = set(_LOOPBACK_NAMES) | {_bare_host(name) for name in allowed_hosts}
        if not self.any_host:
            names.add(bound)
        self.hosts = frozenset(names)

    # -- checks ------------------------------------------------------------------

    def _host_allowed(self, header: str) -> bool:
        name, _ = _split_host(header)
        if not name:
            return False
        return self.any_host or name in self.hosts or is_loopback(name)

    @staticmethod
    def _same_origin(headers: dict[str, str]) -> bool:
        """Whether a request that changes something comes from the app's own pages.

        A browser names the page's origin in ``Origin`` on every such request, and
        newer ones add ``Sec-Fetch-Site``.  A client that sends neither is not a
        browser acting for another site, and is let through.
        """
        origin = headers.get("origin")
        if origin is not None:
            parsed = urlsplit(origin.strip())
            if parsed.scheme not in _DEFAULT_PORTS or not parsed.hostname:
                return False  # "null" (a sandboxed or local page) and anything odd
            default = _DEFAULT_PORTS[parsed.scheme]
            try:
                theirs = (_bare_host(parsed.hostname), parsed.port or default)
            except ValueError:
                return False
            name, port = _split_host(headers.get("host", ""))
            return theirs == (name, port or default)
        site = headers.get("sec-fetch-site")
        return site is None or site.strip().lower() in ("same-origin", "none")

    def _token_given(self, headers: dict[str, str]) -> str | None:
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get("cookie", ""))
        except CookieError:
            cookie = SimpleCookie()
        if TOKEN_COOKIE in cookie:
            return cookie[TOKEN_COOKIE].value
        scheme, _, value = headers.get("authorization", "").partition(" ")
        return value.strip() if scheme.lower() == "bearer" and value.strip() else None

    def _token_ok(self, given: str | None) -> bool:
        return bool(given) and secrets.compare_digest(str(given).encode(), self.token.encode())

    # -- responses ---------------------------------------------------------------

    async def _refuse(self, send: Send, status: int, code: str, message: str, *, api: bool) -> None:
        status, media_type, body = _plain(status, code, message, api=api)
        headers = [
            (b"content-type", media_type),
            (b"content-length", str(len(body)).encode()),
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def _accept_token(self, scope: Scope, send: Send, query: list[tuple[str, str]]) -> None:
        """Keep the token as a cookie and send the browser on without it in the address."""
        rest = [(key, value) for key, value in query if key != TOKEN_PARAM]
        # The path as it was asked for, but never more than a path of this server: a
        # browser reads "//host" and "/\\host" as another site, and the path arrives
        # percent-decoded, so it may hold a line break.
        path = "/" + str(scope.get("path") or "/").lstrip("/\\")
        location = quote(path) + (f"?{urlencode(rest)}" if rest else "")
        cookie = f"{TOKEN_COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Strict"
        headers = [
            (b"location", location.encode()),
            (b"set-cookie", cookie.encode()),
            (b"content-length", b"0"),
        ]
        await send({"type": "http.response.start", "status": 303, "headers": headers})
        await send({"type": "http.response.body", "body": b""})

    # -- ASGI --------------------------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            if scope["type"] == "lifespan":
                await self.app(scope, receive, send)
            return  # nothing here speaks WebSocket; such a connection is simply not accepted
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        path = scope.get("path", "/")
        api = path.startswith("/api/")
        send = _with_security_headers(send, api=api)

        if not self._host_allowed(headers.get("host", "")):
            await self._refuse(
                send,
                403,
                "bad_host",
                "This server answers only to its own address. Open it at the address it "
                "printed when it started.",
                api=api,
            )
            return
        if scope.get("method", "GET").upper() not in _SAFE_METHODS and not self._same_origin(
            headers
        ):
            await self._refuse(
                send,
                403,
                "cross_origin",
                "The request came from another site and was not carried out.",
                api=api,
            )
            return
        if self.token is not None:
            query = parse_qsl(scope.get("query_string", b"").decode("latin-1"))
            offered = dict(query).get(TOKEN_PARAM)
            if offered is not None and self._token_ok(offered) and not api:
                await self._accept_token(scope, send, query)
                return
            if not self._token_ok(self._token_given(headers)):
                await self._refuse(
                    send,
                    401,
                    "token_required",
                    "This server is reachable from other machines, so it asks for its "
                    "access link: open the address with ?token=... that was printed in "
                    "the terminal where it was started.",
                    api=api,
                )
                return
        await self.app(scope, receive, send)


def security_headers(*, api: bool) -> list[tuple[bytes, bytes]]:
    """The headers every response of the app carries unless it sets its own.

    The middleware adds them.  An answer that is written outside the middleware
    (the framework's last resort for an unexpected error) has to add them itself.
    """
    headers = [
        (b"content-security-policy", CONTENT_SECURITY_POLICY.encode()),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
        (b"cross-origin-opener-policy", b"same-origin"),
        (b"cross-origin-resource-policy", b"same-origin"),
    ]
    if api:
        # What the API says changes from one second to the next.
        headers.append((b"cache-control", b"no-store"))
    return headers


def _with_security_headers(send: Send, *, api: bool) -> Send:
    """``send``, adding to every response the headers it does not already set."""
    defaults = security_headers(api=api)

    async def send_with_headers(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            present = {key.lower() for key, _ in headers}
            headers += [(key, value) for key, value in defaults if key not in present]
            message = {**message, "headers": headers}
        await send(message)

    return send_with_headers
