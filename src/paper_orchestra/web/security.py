"""What keeps a local web app local.

The server holds the person's API credentials (its workers spend them) and
serves pages whose text was written by models and copied from the web.  Four
things follow, all enforced here:

* **Who may talk to it.**  It binds to loopback unless told otherwise.  A request
  whose ``Host`` header is not the server's own address is refused: that is how
  a web page on another site reaches a local server through DNS rebinding.  A
  request that changes something and comes from another origin is refused too.
  And every request has to prove that it is the person's, on loopback as well:
  loopback is shared by every account and every process of the machine.  The
  proof is an access token, which the browser gets once from the link printed
  at start and keeps as a cookie, and which other clients send as
  ``Authorization: Bearer``.
* **What a response may do in the browser.**  Every response carries a
  Content-Security-Policy that allows the app's own scripts, styles and images
  and nothing else, and tells the browser not to guess content types.
* **Which files a URL can reach.**  A path from a URL is resolved inside the
  project directory, symbolic links included, or refused; a file the server
  opens for a project is opened without following a link out of it.
* **What an upload may be called.**  Names from the browser are reduced to plain
  relative paths.
* **How much a request may carry.**  Every request body is counted while it
  arrives and refused once it is larger than its limit, whatever length the
  client announced and whatever the endpoint would have made of it.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import hmac
import html
import ipaddress
import json
import os
import re
import secrets
import stat
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

__all__ = [
    "CONTENT_SECURITY_POLICY",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "FILE_AREAS",
    "FILE_CONTENT_SECURITY_POLICY",
    "MAX_BODY_BYTES",
    "MIN_TOKEN_CHARS",
    "PDF_CONTENT_SECURITY_POLICY",
    "TOKEN_COOKIE",
    "TOKEN_ENV",
    "PathEscape",
    "SecurityMiddleware",
    "TokenError",
    "confined",
    "fixed_token",
    "is_loopback",
    "mask_secrets",
    "new_token",
    "offers_token",
    "open_confined",
    "open_for_append",
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
#: The cookie that holds the token; its full name also says the port (see ``_cookie_name``).
TOKEN_COOKIE = "po_token"
TOKEN_PARAM = "token"
#: The environment variable that fixes the token's value, for a test harness or a
#: bookmark that should survive restarts.
TOKEN_ENV = "PAPER_ORCHESTRA_UI_TOKEN"
#: A token the person chose must be at least this long to be worth the name.
MIN_TOKEN_CHARS = 16
#: What a token may consist of: it travels in a link, a cookie and a header, as it is.
_TOKEN_CHARS = re.compile(r"[A-Za-z0-9._~-]+")
#: The largest body a request may carry unless its endpoint is given a limit of its own
#: (the upload of a new project is): room for the longest page, code file or preview the
#: API accepts, written as JSON, and no more.
MAX_BODY_BYTES = 8 * 1024 * 1024

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
    """A fresh access token: 192 random bits, in characters a link can carry."""
    return secrets.token_urlsafe(24)


class TokenError(ValueError):
    """The access token named in the environment cannot be used; the message says why."""


def fixed_token(environ: MutableMapping[str, str] | None = None) -> str | None:
    """The access token the person fixed in ``PAPER_ORCHESTRA_UI_TOKEN``, or ``None``.

    The variable is taken out of ``environ`` (the process's own by default)
    whatever it held, so that nothing the server starts inherits the token:
    workers, the jobs they run, a browser.  An empty value counts as none.  A
    value shorter than ``MIN_TOKEN_CHARS`` characters, or one with characters
    that a link, a cookie and a header cannot all carry, raises ``TokenError``
    with a message that does not repeat the value.
    """
    environ = os.environ if environ is None else environ
    value = str(environ.pop(TOKEN_ENV, "") or "").strip()
    if not value:
        return None
    if len(value) < MIN_TOKEN_CHARS:
        raise TokenError(
            f"The access token in {TOKEN_ENV} is {len(value)} characters long; it must be "
            f"at least {MIN_TOKEN_CHARS} characters to keep anybody out. Choose a longer "
            f"one, or unset {TOKEN_ENV} and the server makes a token at every start."
        )
    if not _TOKEN_CHARS.fullmatch(value):
        raise TokenError(
            f"The access token in {TOKEN_ENV} must be at least {MIN_TOKEN_CHARS} characters "
            "of letters, digits, '-', '.', '_' and '~' only, because it is carried by a "
            f"link and a cookie. Choose another one, or unset {TOKEN_ENV} and the server "
            "makes a token at every start."
        )
    return value


def offers_token(scope: Mapping[str, Any]) -> bool:
    """Whether the address of a request names an access token.

    That is the link a person opens, or somebody's attempt at it.  Such an
    address is not for anything that keeps the addresses of requests (a trace,
    a log): ``create_app`` keeps it from the framework's own tracing.  Never
    raises, whatever the query holds.
    """
    query = bytes(scope.get("query_string") or b"").decode("latin-1")
    return any(key == TOKEN_PARAM for key, _ in parse_qsl(query, keep_blank_values=True))


def _bare_host(value: str) -> str:
    return str(value or "").strip().lower().strip("[]").rstrip(".")


def _port(digits: str) -> int | None:
    """The port that ``digits`` name, or ``None`` when they name none.

    ``str.isdigit`` is true of characters that ``int`` does not read (a superscript
    two), and ``int`` refuses a number of thousands of digits: both arrive in a
    header anybody can send, before the server has looked at who is asking.
    """
    if digits.isascii() and digits.isdigit() and len(digits) <= 5:
        return int(digits)
    return None


def _split_host(value: str) -> tuple[str, int | None]:
    """``host`` and ``port`` of a ``Host`` header or a URL's network location.

    Never raises.  Where the port is written in digits that name no port, the
    value is no address: the host comes back empty, which nothing accepts.
    """
    value = str(value or "").strip().lower()
    port: int | None = None
    if value.startswith("["):
        name, _, rest = value[1:].partition("]")
        if rest.startswith(":") and rest[1:].isdigit():
            port = _port(rest[1:])
            if port is None:
                return "", None
        return name.rstrip("."), port
    name, colon, digits = value.rpartition(":")
    if colon and digits.isdigit() and ":" not in name:
        port = _port(digits)
        return (name.rstrip("."), port) if port is not None else ("", None)
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


def _plain_file(descriptor: int, what: str) -> None:
    """``PathEscape`` unless the open file is an ordinary one (not a directory, a pipe or
    a device, which would make the server wait or read without end)."""
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise PathEscape(f"{what} is not a file")


def open_confined(root: Path, relative: str, *, areas: Sequence[str] | None = None) -> BinaryIO:
    """Open the file ``relative`` names inside ``root`` for reading, as bytes.

    The path is resolved as :func:`confined` does (``PathEscape`` when it leaves
    ``root`` through ``..`` or a symbolic link, or names something hidden), and the
    resolved file is then opened without following a link, so that a link put
    there between the check and the opening is not followed either.  Raises
    ``FileNotFoundError`` when there is no such file and ``PathEscape`` when it is
    not an ordinary file.  This is how a project's logs are read: a project
    directory may come from somebody else, with ``worker.log`` or a run's
    ``log.txt`` linked to a file of the person's.
    """
    real = confined(root, relative, areas=areas)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(real, flags | getattr(os, "O_BINARY", 0))
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise PathEscape("the path is a link") from exc
        raise
    try:
        _plain_file(descriptor, relative)
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def open_for_append(root: Path, name: str) -> BinaryIO:
    """Open the file ``name`` directly in ``root`` for appending, creating it if needed.

    ``name`` is a plain file name.  The file is opened without following a
    symbolic link, and it must be an ordinary file with no other name anywhere:
    what the server appends (a worker's command line with the person's
    instruction, and everything the worker prints) must not land in a file
    outside the project that a link or a second name stands for.  Raises
    ``PathEscape`` otherwise, ``OSError`` when the file cannot be written.
    """
    if (
        not isinstance(name, str)
        or not name
        or name.startswith(".")
        or name != os.path.basename(name)
        or "\x00" in name
    ):
        raise PathEscape("not the name of a file in the project directory")
    path = Path(root) / name
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    # Without waiting: a named pipe put in the file's place would hold the server for ever.
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if not hasattr(os, "O_NOFOLLOW") and path.is_symlink():  # pragma: no cover - Windows
        raise PathEscape(f"{path} is a link; the server does not write through it")
    try:
        descriptor = os.open(path, flags | getattr(os, "O_BINARY", 0), 0o644)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENXIO, errno.EISDIR):
            raise PathEscape(
                f"{path} is a link or not a file; the server does not write through it"
            ) from exc
        raise
    try:
        _plain_file(descriptor, str(path))
        if os.fstat(descriptor).st_nlink > 1:
            raise PathEscape(
                f"{path} is another name for a file that exists elsewhere; the server does "
                "not write to it"
            )
        with contextlib.suppress(ImportError, OSError):
            import fcntl

            # The worker inherits this file as its output: it is to write as to any file.
            mode = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            fcntl.fcntl(descriptor, fcntl.F_SETFL, mode & ~os.O_NONBLOCK)
        return os.fdopen(descriptor, "ab")
    except BaseException:
        os.close(descriptor)
        raise


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


#: What a request without the token is told.  The same for a missing and a wrong token,
#: and nothing in it depends on the request.
_TOKEN_REQUIRED = (
    "This app is opened with the link printed in the terminal where `paper-orchestra ui` "
    "runs: the link carries the access token (it ends in ?token=...). It is shown once, "
    "when the server starts; if it is gone, restart the server to get a new one."
)
#: Added for the API, whose callers are programs.
_TOKEN_FOR_PROGRAMS = "A program sends the token as `Authorization: Bearer <token>`."
#: The same for a person who opened an address of the app in a browser.  No style and no
#: script: the policy allows neither inline, and every file of the app needs the token.
_TOKEN_REQUIRED_PAGE = b"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex">
<title>PaperOrchestra: open it with its link</title>
</head>
<body>
<h1>This app is opened with its link</h1>
<p>PaperOrchestra needs the link printed in the terminal where
<code>paper-orchestra ui</code> runs: that link carries the access token
(it ends in <code>?token=...</code>).</p>
<p>The link is shown once, when the server starts. If it is gone, restart the
server to get a new one.</p>
</body>
</html>
"""
#: What the link is answered with when a page of another site led to it (see
#: ``SecurityMiddleware._accept_token``): the cookie is set, and this page goes on to the
#: address without the token by itself.
_ENTERING_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light dark">
<meta name="referrer" content="no-referrer">
<meta http-equiv="refresh" content="0; url={location}">
<title>Opening PaperOrchestra</title>
</head>
<body>
<p>Opening PaperOrchestra. If nothing happens, <a href="{location}">continue</a>.</p>
</body>
</html>
"""
_NO_STORE = (b"cache-control", b"no-store")


def _cookie_pairs(header: str) -> list[tuple[str, str]]:
    """The ``name=value`` pairs of a ``Cookie`` header, whatever else is in it.

    Read by hand: every app on this host name shares the cookie jar, whichever
    port it listens on, and one pair that is not well formed must not hide the
    others (the standard library's parser gives up on the whole header).
    """
    pairs = []
    for part in header.split(";"):
        name, equals, value = part.partition("=")
        if equals and name.strip():
            pairs.append((name.strip(), value.strip()))
    return pairs


class _BodyTooLarge(Exception):
    """A request body passed its limit while it arrived."""


#: What the browser sends back as ``Last-Event-ID``: an event's ``seq``, as the stream
#: wrote it.
_EVENT_ID = re.compile(rb"[0-9]{1,18}")


def _megabytes(count: int) -> str:
    size = count / (1024 * 1024)
    return f"{size:.0f} MB" if size >= 10 else f"{size:.1f} MB"


def _announced_length(value: str | None) -> int | None:
    """The body length a request announces, or ``None`` when it announces none that can
    be read.  A number too long to be a length is one that is too large."""
    digits = (value or "").strip()
    if not digits or not digits.isascii() or not digits.isdigit():
        return None
    return int(digits) if len(digits) <= 18 else 10**18


class _BoundedBody:
    """The ``receive`` and ``send`` of one request whose body may be at most ``limit``
    bytes.

    The body is counted as it arrives.  Past the limit the application is not
    given another byte (its ``receive`` raises), and whatever it then answers,
    or fails with, the client is told once that the body was too large.
    """

    def __init__(
        self, receive: Receive, send: Send, limit: int, refuse: Callable[[], Awaitable[None]]
    ) -> None:
        self._receive = receive
        self._send = send
        self._refuse = refuse
        self.limit = limit
        self.seen = 0
        self.over = False
        self.started = False
        self.refused = False

    async def receive(self) -> Message:
        if self.over:
            raise _BodyTooLarge
        message = await self._receive()
        if message.get("type") == "http.request":
            self.seen += len(message.get("body") or b"")
            if self.seen > self.limit:
                self.over = True
                raise _BodyTooLarge
        return message

    async def refuse(self) -> None:
        """Answer 413, unless an answer has gone out already."""
        if self.started or self.refused:
            return
        self.refused = True
        await self._refuse()

    async def send(self, message: Message) -> None:
        if self.over and not self.started:
            # The application's own account of a body that was cut off (a parse error,
            # an unexpected error) is not what happened.
            await self.refuse()
            return
        if message.get("type") == "http.response.start":
            self.started = True
        await self._send(message)


class SecurityMiddleware:
    """Checks ``Host``, ``Origin`` and the access token, bounds the request body, and
    adds the security headers.

    Plain ASGI on purpose: nothing here buffers a response, so event streams pass
    through untouched.

    ``host`` is the address the server listens on.  For a loopback address the
    accepted ``Host`` names are the loopback ones; for a specific other address,
    that address as well; for a wildcard address any name is accepted, because the
    machine's names cannot be listed, and the token is what stands between a
    stranger and the API.  ``allowed_hosts`` adds names, for a reverse proxy or a
    test client.

    ``token`` is the access token (``serve()`` always gives one; ``None`` means
    that none is asked for, which is for in-process tests only).  With it, every
    request must carry the token: as the cookie that opening the link sets, or
    as ``Authorization: Bearer <token>``.  ``GET /any/page?token=<token>`` is
    that link: it sets the cookie (``HttpOnly``, ``SameSite=Strict``) and sends
    the browser on to the same address without the token (a redirect; where a
    page of another site led to the link, a page that goes on by itself, see
    ``_accept_token``).  The token in the query of an API call, or of anything
    but a ``GET``, counts for nothing.  A
    request without a valid token is answered 401: the API in its JSON error
    shape, a page a browser asked for with a small page that says where the
    link is, anything else in a sentence.  A wrong token is answered exactly
    like a missing one.

    ``max_body_bytes`` is the most a request body may hold; ``body_limits`` gives
    single endpoints another limit, as ``{(method, path): bytes}`` (the upload of
    a new project).  A request that announces more is refused before a byte of it
    is read, and one that sends more than it announced is cut off where the limit
    is passed: 413 with the code ``too_large``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        host: str = DEFAULT_HOST,
        token: str | None = None,
        allowed_hosts: Iterable[str] = (),
        max_body_bytes: int = MAX_BODY_BYTES,
        body_limits: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        self.app = app
        self.token = token or None
        bound = _bare_host(host)
        self.any_host = bound in _WILDCARD_HOSTS
        names = set(_LOOPBACK_NAMES) | {_bare_host(name) for name in allowed_hosts}
        if not self.any_host:
            names.add(bound)
        self.hosts = frozenset(names)
        self.max_body_bytes = max_body_bytes
        self.body_limits = {
            (method.upper(), path): limit for (method, path), limit in (body_limits or {}).items()
        }

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
            try:
                # (Taking it apart fails on an address with half a bracket, reading its
                # port on one that is no number.)
                parsed = urlsplit(origin.strip())
                if parsed.scheme not in _DEFAULT_PORTS or not parsed.hostname:
                    return False  # "null" (a sandboxed or local page) and anything odd
                default = _DEFAULT_PORTS[parsed.scheme]
                theirs = (_bare_host(parsed.hostname), parsed.port or default)
            except ValueError:
                return False
            name, port = _split_host(headers.get("host", ""))
            return theirs == (name, port or default)
        site = headers.get("sec-fetch-site")
        return site is None or site.strip().lower() in ("same-origin", "none")

    @staticmethod
    def _cookie_name(scope: Scope, headers: dict[str, str]) -> str:
        """The name of the token's cookie for the address this request was sent to.

        A browser keeps cookies by host name and not by port, so the port is part
        of the name: two servers on one machine (two workspaces, two forwarded
        ports) each keep their own, and opening the one does not close the other.

        The name keeps the servers apart, not the cookie in: a browser sends it
        to every port of the host name.  A program that listens on another port
        of the machine the browser runs on is given it when the browser opens
        that program's address (``SameSite`` holds only for the first request
        another site leads to, not for what the page then asks by itself).
        That is a limit of cookies on a host name that programs share; a name
        cannot lift it.
        """
        _, port = _split_host(headers.get("host", ""))
        if port is None:
            port = _DEFAULT_PORTS.get(str(scope.get("scheme", "http")), 80)
        return f"port{port}_{TOKEN_COOKIE}"

    def _tokens_given(self, scope: Scope, headers: dict[str, str]) -> list[str]:
        """What the request offers as the token: its cookie for this address (a stale
        one of that name may stand beside the current one) and the Bearer header."""
        name = self._cookie_name(scope, headers)
        # Every Cookie line: a client or a proxy may send the pairs on several.
        lines = [
            value.decode("latin-1")
            for key, value in scope.get("headers", [])
            if key.lower() == b"cookie"
        ]
        given = [value for key, value in _cookie_pairs("; ".join(lines)) if key == name]
        scheme, _, value = headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer":
            given.append(value.strip())
        return given

    def _token_ok(self, given: str | None) -> bool:
        """Whether ``given`` is the token, in time that does not depend on how much of
        it is right, nor on its length (both sides are hashed first)."""
        if not given or self.token is None:
            return False
        ours = hashlib.sha256(self.token.encode()).digest()
        theirs = hashlib.sha256(str(given).encode("utf-8", "surrogatepass")).digest()
        return hmac.compare_digest(ours, theirs)

    def _authorised(self, scope: Scope, headers: dict[str, str]) -> bool:
        # Every candidate is compared: no early exit that would tell which one matched.
        results = [self._token_ok(given) for given in self._tokens_given(scope, headers)]
        return any(results)

    # -- responses ---------------------------------------------------------------

    async def _refuse(self, send: Send, status: int, code: str, message: str, *, api: bool) -> None:
        status, media_type, body = _plain(status, code, message, api=api)
        await self._answer(send, status, media_type, body)

    @staticmethod
    async def _answer(
        send: Send,
        status: int,
        media_type: bytes,
        body: bytes,
        extra: Sequence[tuple[bytes, bytes]] = (),
    ) -> None:
        headers = [
            (b"content-type", media_type),
            (b"content-length", str(len(body)).encode()),
            *extra,
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def _ask_for_token(
        self, scope: Scope, send: Send, headers: dict[str, str], *, api: bool
    ) -> None:
        """Answer 401 to a request without the token (or with a wrong one: the same).

        A person who opened an address of the app in a browser gets a small page
        that says where the link is; the API answers in its error shape; anything
        else (a script, an image, a command-line client) gets the sentence.
        """
        page = (
            not api
            and str(scope.get("method", "GET")).upper() in ("GET", "HEAD")
            and "text/html" in headers.get("accept", "").lower()
        )
        if page:
            media_type, body = b"text/html; charset=utf-8", _TOKEN_REQUIRED_PAGE
        else:
            message = f"{_TOKEN_REQUIRED} {_TOKEN_FOR_PROGRAMS}" if api else _TOKEN_REQUIRED
            _, media_type, body = _plain(401, "token_required", message, api=api)
        await self._answer(send, 401, media_type, body, [_NO_STORE])

    async def _accept_token(
        self, scope: Scope, send: Send, headers: dict[str, str], query: list[tuple[str, str]]
    ) -> None:
        """Keep the token as a cookie and send the browser on without it in the address.

        A redirect does that, except where a page of another site led to the link
        (the file a browser is opened with, a terminal that runs in a browser, a
        link in a mail): a browser does not send a ``SameSite=Strict`` cookie
        along with a navigation another site started, the redirect included, so
        the person would arrive without the cookie they were just given.  There
        the answer is a page of this server that goes on to the address by
        itself, which makes the next request the server's own.
        """
        rest = [(key, value) for key, value in query if key != TOKEN_PARAM]
        # The path as it was asked for, but never more than a path of this server: a
        # browser reads "//host" and "/\\host" as another site, and the path arrives
        # percent-decoded, so it may hold a line break.
        path = "/" + str(scope.get("path") or "/").lstrip("/\\")
        location = quote(path) + (f"?{urlencode(rest)}" if rest else "")
        # For this session of the browser, to scripts invisible, and never sent along
        # with a request that another site started.
        name = self._cookie_name(scope, headers)
        cookie = f"{name}={self.token}; Path=/; HttpOnly; SameSite=Strict"
        kept = [(b"set-cookie", cookie.encode()), _NO_STORE]
        if headers.get("sec-fetch-site", "").strip().lower() == "cross-site":
            page = _ENTERING_PAGE.format(location=html.escape(location, quote=True))
            await self._answer(send, 200, b"text/html; charset=utf-8", page.encode(), kept)
            return
        answer = [(b"location", location.encode()), (b"content-length", b"0"), *kept]
        await send({"type": "http.response.start", "status": 303, "headers": answer})
        await send({"type": "http.response.body", "body": b""})

    # -- ASGI --------------------------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            # Nothing here speaks WebSocket.  The connection is refused in so many words:
            # left unanswered, the server answers 500 and writes an error into the person's
            # terminal, for anybody who asks and as often as they ask.
            if scope["type"] == "websocket":
                await receive()  # the request to connect
                await send({"type": "websocket.close", "code": 1008})
            return
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
            if self._is_link(scope, api=api):
                query = parse_qsl(
                    scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True
                )
                offered = [value for key, value in query if key == TOKEN_PARAM]
                if offered and all([self._token_ok(value) for value in offered]):
                    await self._accept_token(scope, send, headers, query)
                    return
            if not self._authorised(scope, headers):
                await self._ask_for_token(scope, send, headers, api=api)
                return
        await self._bounded(_known_event_id(scope), receive, send, api=api)

    @staticmethod
    def _is_link(scope: Scope, *, api: bool) -> bool:
        """Whether this is a request a person makes by opening a link: the only kind
        whose address may carry the token.  Never an API call (such an address ends
        up in logs and histories) and never anything that changes something."""
        return not api and str(scope.get("method", "GET")).upper() == "GET"

    async def _bounded(self, scope: Scope, receive: Receive, send: Send, *, api: bool) -> None:
        """Hand the request to the application with a body that cannot pass its limit."""
        method = str(scope.get("method", "GET")).upper()
        limit = self.body_limits.get((method, str(scope.get("path", "/"))), self.max_body_bytes)
        message = (
            f"The request carries more than the {_megabytes(limit)} this server takes at "
            "once here. Send less: a shorter text, or fewer or smaller files."
        )

        async def refuse() -> None:
            await self._refuse(send, 413, "too_large", message, api=api)

        announced = None
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                announced = _announced_length(value.decode("latin-1"))
        if announced is not None and announced > limit:
            await refuse()  # before a byte of it is read
            return
        body = _BoundedBody(receive, send, limit, refuse)
        try:
            await self.app(scope, body.receive, body.send)
        except _BodyTooLarge:
            pass
        except Exception:
            if not body.over:
                raise
            # What the application made of the body that was cut off is not an error
            # of the server's.
        if body.over:
            await body.refuse()


def _known_event_id(scope: Scope) -> Scope:
    """``scope`` without a ``Last-Event-ID`` that is not one this server gave out.

    The stream numbers its events; a browser sends the last number back when it
    reconnects.  Anything else under that name (a digit of another script, a
    number of thousands of places) is no place to resume from.
    """
    headers = scope.get("headers", [])
    kept = [
        (key, value)
        for key, value in headers
        if key.lower() != b"last-event-id" or _EVENT_ID.fullmatch(value.strip())
    ]
    return scope if len(kept) == len(headers) else {**scope, "headers": kept}


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
