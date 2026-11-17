"""What keeps an agent with a shell from deciding what is the person's to decide.

An SDK agent can run any command and read and write files.  The person keeps
the gate all the same: an agent must not be able to accept a proposal, approve
a run or change the files the project is steered by, and it must not get at
the credentials of the machine.  Four things see to that, and each stands on
its own:

- **The shell's sandbox** (the agent's process applies it to every command):
  the shell can write only inside the agent's own folder, and cannot read the
  person's home directory, the project's directory, the place the identity
  token lies or the directory for temporary files that this process and the
  person's other programs share, except for what is named as readable (its own
  folder, the Python it runs code with).  Its temporary files lie in a
  directory made for this agent alone (:func:`make_temp`), which no other agent
  reads and which is removed when the agent ends.  The sandbox is required, not wished for: where it
  cannot start (on Linux it needs ``bwrap`` and ``socat``), no agent starts
  (:func:`sandbox_fault`), and the agent's process is told to refuse every
  command besides (``failIfUnavailable``), so that no command ever runs
  outside it.  A file of settings with secrets (``.env``, ``.envrc``,
  ``*.env``) is named to it as well, which on Linux only holds in the agent's
  folder: one that lies outside every
  hidden place (``/srv/app/.env``) can be read by the shell, though not by the
  file tools.  A write outside the folder does not always fail with an
  error: the sandbox may take it and throw it away, so that the command seems
  to succeed and nothing is changed.  The shell has the web, but not this
  machine's own services: the sandbox refuses link-local addresses (where a
  cloud's metadata server hands out the machine's credentials) and the
  loopback, and the settings name those hosts as denied besides
  (:data:`CLOSED_HOSTS`).  Whether the sandbox holds is checked against a real
  agent by ``tests/e2e/agent_guardrails.py``.
- **Permission rules** for the built-in file tools: reading the places where
  credentials lie is denied by the agent's process itself.
- **The check before every tool call** (:meth:`Guard.check`, run in this
  process as the agent's ``PreToolUse`` hook, also for helpers the agent
  starts): built-in tools that write stay inside the agent's folder, the ones
  that read stay out of what the sandbox hides and out of the trees of the
  system that show processes and devices, and the ones that search (Grep, Glob)
  do not start at a directory that has any of that beneath it (``/``,
  ``/home``), the tool that opens a web page does not open this machine's own
  services under any spelling of their address (:func:`closed_host` resolves
  the name as the connection would), and a shell command that runs the
  project's own command line is refused, with the way that is open instead:
  propose it on the board.  That last refusal reads the text of the command:
  it catches the command line and an import of the package where they are
  written out, not a script that assembles the name.  It tells an agent that
  means no harm what to do instead; what keeps the project's files from an
  agent that does is the sandbox.
- **The environment** the agent's process starts with (:func:`agent_env`): only a
  short list of this process's variables reaches it (:data:`KEPT_ENV`) and the ones
  its own sign-in uses; every other one is taken away, whatever its name.

``settings.agents.sandbox: false`` takes the first away and the file checks of
the third with it: the agent then works as the person's own Claude Code would,
and its shell reaches whatever the person's does: the project's files, this
machine's services, and on a cloud machine the metadata server and so the
machine's own account.  The refusal of the project's command line stays, and
so do the rules that keep the file tools from the places where credentials lie
(the identity token among them); the shell is under none of them then, and
reads whatever the person's own shell reads.
"""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import socket
import sys
import tempfile
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_orchestra.research.sdk.driver import ToolUse

__all__ = [
    "CLOSED_HOSTS",
    "FEDERATION_ENV",
    "JOB_ENV_PREFIX",
    "KEPT_ENV",
    "LAUNCHER",
    "PASS_ENV",
    "TEMP_ENV",
    "Guard",
    "SandboxUnavailable",
    "agent_env",
    "claude_config_dir",
    "closed_host",
    "launcher_text",
    "make_temp",
    "runs_the_project_cli",
    "sandbox_fault",
]

#: What Workload Identity Federation needs (four ids and the path of the identity token).
FEDERATION_ENV: tuple[str, ...] = (
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)
#: Under these names an experiment agent's shell holds the federation settings of a run
#: that was approved with network access.  Not under their own names: those would make
#: the agent's own process sign in with them, whatever credential the project chose.
JOB_ENV_PREFIX = "PAPER_ORCHESTRA_JOB_"
#: The file in an experiment agent's folder that starts a command with those settings.
LAUNCHER = "with-models"
#: Names where the directories for the agents' temporary files are made; without it
#: they are made in ``/var/tmp`` (on a disk, which ``/tmp`` often is not), else in
#: ``/tmp``.  It must be short (about 14 bytes at most): see :data:`_TEMP_BYTES`.
TEMP_ENV = "PAPER_ORCHESTRA_AGENT_TMPDIR"
#: The most bytes the path of an agent's directory for temporary files may have.  The
#: agent's process keeps sockets there, whose paths are short by the system's rule: it
#: makes ``claude-<uid>`` in the directory it is named (``CLAUDE_CODE_TMPDIR``), and
#: when that path is longer than 44 bytes it quietly uses the shared directory instead
#: (seen with a directory inside an experiment's run folder).
_TEMP_BYTES = 44
#: What the sandbox of the agent's shell is made of on Linux.
SANDBOX_PROGRAMS: tuple[str, ...] = ("bwrap", "socat")

_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
#: A name that says where something lies is no credential.  (``_ID`` is not among these:
#: ``AWS_ACCESS_KEY_ID`` is half of a credential, and a name is only asked about here
#: when it has one of the words above in it.)
_NOT_A_SECRET_SUFFIX = ("_FILE", "_PATH", "_DIR")
#: The variables of this process that an agent's process and its shell keep; every other
#: one is taken away.  A list of what is let through, not of what is kept back: a name
#: says little about what its value opens (``SSH_AUTH_SOCK`` is the person's ssh keys,
#: ``DOCKER_HOST`` and ``KUBECONFIG`` are machines, ``DATABASE_URL`` and ``SENTRY_DSN``
#: hold a password in the value), so no pattern of names finds them all.  Here are: who
#: and where (the shell and its programs need them), the language, where the agent's
#: process keeps its own files, the certificates and proxies of this network, the
#: interpreter's environment, which devices compute, and the person's choices about
#: what the agent's process reports home.
KEPT_ENV: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TERM",
        "COLORTERM",
        "NO_COLOR",
        "TZ",
        "CLAUDE_CONFIG_DIR",
        # The agent's process is told by its starter what started it.
        "CLAUDE_CODE_ENTRYPOINT",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "DISABLE_TELEMETRY",
        "DISABLE_AUTOUPDATER",
        "DISABLE_ERROR_REPORTING",
        "DO_NOT_TRACK",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    }
)
_KEPT_ENV_PREFIXES = ("LC_",)
#: Names more variables of this process for the agents to keep, separated by commas or
#: spaces: the person's own choice, for what their experiments need (``HF_HOME``, say).
#: Whatever is named here reaches every agent's shell, which has the web.  A variable
#: that changes how the agent's process signs in (``ANTHROPIC_*``, ``CLAUDE*``) is not
#: passed on by it.
PASS_ENV = "PAPER_ORCHESTRA_AGENT_ENV"
_NOT_PASSED_ON = ("ANTHROPIC_", "CLAUDE")
#: An address with a name and a password before the host (a proxy's, a database's).
_PROXY_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_USERINFO = re.compile(r"[a-z][a-z0-9+.-]*://[^/?#\s]*@", re.IGNORECASE)
#: What separates the words of a shell command, for finding the programs it names.
_WORD_BREAKS = re.compile(r"[\s;|&()<>`$\"'=,{}\[\]]+")
_CLI_NAMES = ("paper-orchestra", "paper_orchestra")

_BOARD_INSTEAD = (
    "The project's own commands (review, approve, decide, advance, run, cancel and the "
    "rest) are the researcher's, and an agent does not run them. Put what you want on the "
    "board instead: a new page with the memory tool, a change to a page with propose_edit, "
    "a choice that is the researcher's with ask_decision. The researcher decides and "
    "answers, and an experiment they approve is carried out by another agent."
)
#: Hosts no agent's shell or web tool reaches: the metadata servers of the clouds, which
#: hand the machine's own credentials to whoever asks from the machine, and the machine
#: itself, where the app that decides proposals may be listening.  The unspecified
#: address is among them: a connection to ``0.0.0.0`` arrives at the loopback.
CLOSED_HOSTS: tuple[str, ...] = (
    "metadata.google.internal",
    "metadata.goog",
    "metadata",
    "instance-data",
    "instance-data.ec2.internal",
    "169.254.169.254",
    "100.100.100.200",
    "168.63.129.16",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "0",
    "::",
    "::1",
)
#: Trees of the system that show processes and devices, not files: closed to the file
#: tools whatever else is readable (the shell's sandbox has its own view of them).
_SYSTEM_TREES = (Path("/proc"), Path("/sys"))
_NOT_A_PAGE = (
    "That address is this machine's own (or a cloud's metadata server), not a page on the "
    "web: it is closed to agents."
)
_HTTP_SCHEME = re.compile(r"https?:", re.IGNORECASE)
_HTTP_START = re.compile(r"https?://[^/]", re.IGNORECASE)
_WRITING_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
_READING_TOOLS = ("Read", "Glob", "Grep", "LS", "NotebookRead")
_PATH_KEYS = ("file_path", "notebook_path", "path")


def runs_the_project_cli(command: str) -> bool:
    """Whether a shell command starts this package's command line, or its code.

    A word of the command that is the program (``paper-orchestra``, also by a
    path), the module (``python -m paper_orchestra``) or an import of it
    (``python -c "from paper_orchestra..."``).  A directory that merely has the
    name in it (``paper-orchestra-notes/``) is not one.

    This reads the command's text and no more: ``python run.py`` with the import
    inside the file, or a name put together from pieces, is not seen.  It is a
    signpost for an agent that means no harm, not the boundary: under the
    sandbox such code cannot change the project's files whatever it imports,
    and without the sandbox nothing here keeps a determined agent out.
    """
    for word in _WORD_BREAKS.split(str(command or "")):
        last = word.rstrip("/").rsplit("/", 1)[-1]
        if last in _CLI_NAMES or last.startswith("paper_orchestra."):
            return True
    return False


def _addresses(host: str) -> list[str]:
    """The addresses the name ``host`` resolves to, as a connection to it would find them."""
    found = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(entry[4][0]) for entry in found))


def _closed_address(text: str) -> bool:
    """Whether the address ``text`` is one of :data:`CLOSED_HOSTS`, the loopback, a
    link-local address or the unspecified one; also when it is written as an IPv4
    address inside an IPv6 one.  What is no address is closed."""
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return (
        str(address) in CLOSED_HOSTS
        or address.is_loopback
        or address.is_link_local
        or address.is_unspecified
    )


def closed_host(url: str) -> bool:
    """Whether ``url`` names a host that is closed to agents (:data:`CLOSED_HOSTS`), or
    any other address of the loopback or the link-local range, however it is spelled.

    An address can be written in more ways than one (``127.1``, ``2130706433``,
    ``0x7f000001``, ``0177.0.0.1``), and any name can resolve to one
    (``127.0.0.1.nip.io``): the host is read the way a connection reads it, and a
    name is resolved, and closed when one of its addresses is.  An address that
    does not parse and a name that does not resolve are closed: nothing is
    opened that cannot be read.  (A name whose answer changes between this look
    and the connection is not caught here.)

    Two readers of one address can find two hosts in it: this function reads
    ``http://127.0.0.1\\@example.com/`` as ``example.com``, and the program that
    connects reads a backslash as a slash and goes to ``127.0.0.1``.  So an
    address is closed wherever readers are known to differ, before any host is
    looked for: with a backslash, a control character or a space in it, with
    ``http:`` not followed by exactly two slashes, with a name or password
    before the host (``user@``), and with a ``%`` in the host.  A host in other
    letters than ASCII is first written the way the connection writes it
    (``１２７.０.０.１`` is ``127.0.0.1``).
    """
    url = str(url).strip(" ")
    if "\\" in url or any(ord(char) <= 0x20 or ord(char) == 0x7F for char in url):
        return True
    if _HTTP_SCHEME.match(url) and not _HTTP_START.match(url):
        return True
    try:
        parts = urllib.parse.urlsplit(url if "//" in url else f"//{url}")
        host = parts.hostname
    except ValueError:
        return True
    if not host or "@" in parts.netloc:
        return True
    host = host.rstrip(".").lower()
    if "%" in host and ":" not in host:
        return True
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii").rstrip(".").lower()
        except UnicodeError:
            return True
    if not host or host in CLOSED_HOSTS or host.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        pass
    else:
        return _closed_address(host)
    try:
        # The short, the decimal, the octal and the hexadecimal ways to write an address.
        return _closed_address(socket.inet_ntoa(socket.inet_aton(host)))
    except (OSError, UnicodeError):
        pass
    try:
        addresses = _addresses(host)
    except (OSError, UnicodeError):
        return True
    return not addresses or any(_closed_address(address) for address in addresses)


def sandbox_fault(path: str | None = None, platform: str | None = None) -> tuple[str, str] | None:
    """Why the sandbox of an agent's shell cannot start on this machine, as ``(why, what
    to do)``; ``None`` when nothing is known to be missing.

    On Linux the sandbox is made of two programs (:data:`SANDBOX_PROGRAMS`), which
    must be on the path the agent's process is started with (``path``; this
    process's without).  Elsewhere the system brings what is needed.  Without them
    the agent's process would run every command of the agent outside any sandbox,
    so no agent is started.
    """
    if not (platform or sys.platform).startswith("linux"):
        return None
    where = os.environ.get("PATH", os.defpath) if path is None else path
    missing = [name for name in SANDBOX_PROGRAMS if shutil.which(name, path=where) is None]
    if not missing:
        return None
    return (
        "the sandbox that an agent's shell runs in cannot start on this machine: "
        f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not installed",
        "install bubblewrap and socat (`apt install bubblewrap socat`), or set "
        "settings.agents.sandbox to false if you accept agents whose shell reaches "
        "everything yours does",
    )


class SandboxUnavailable(RuntimeError):
    """The project asks for agents in a sandbox, and the sandbox cannot start.

    No agent works then: a shell outside the sandbox is what the setting forbids.
    Its message is a whole sentence to the person, and trying again changes nothing.
    """

    plain = True
    whole_sentence = True
    #: Read by ``errors.stops_the_run``: nobody tries this again by itself.
    unusable = True


def claude_config_dir(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """Claude Code's configuration directory: ``CLAUDE_CONFIG_DIR`` where it is set,
    else ``.claude`` in the home directory.  The person's login lies there."""
    environ = os.environ if environ is None else environ
    named = str(environ.get("CLAUDE_CONFIG_DIR") or "").strip()
    return Path(os.path.expanduser(named)) if named else (home or Path.home()) / ".claude"


def _real(path: str | os.PathLike[str], base: Path) -> Path:
    """Where ``path`` leads, links followed, taken from ``base`` when it is relative."""
    text = os.path.expanduser(os.fspath(path))
    joined = text if os.path.isabs(text) else os.path.join(base, text)
    return Path(os.path.realpath(joined))


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _package_caches(home: Path) -> list[Path]:
    """What of uv's cache is opened for reading: the packages it has unpacked, not the
    checkouts of git repositories it keeps there (``git-v0``), which can be private."""
    try:
        found = sorted((home / ".cache" / "uv").iterdir())
    except OSError:
        return []
    return [place for place in found if place.is_dir() and not place.name.startswith("git-")]


#: The names such a file usually has, as patterns for the agent's process and its
#: sandbox.  (The check before a tool call knows more of them: :func:`_is_env_file`.)
_ENV_NAMES = (".env", ".env.local", ".envrc", "*.env")
#: Endings of a file that shows which settings an ``.env`` file takes, without values.
_ENV_SAMPLES = (".example", ".sample", ".template", ".dist")


def _is_env_file(path: Path) -> bool:
    """Whether ``path`` is named like a file that holds settings with their secrets:
    ``.env``, ``.env.local``, ``.envrc``, ``prod.env``; not ``.env.example``."""
    name = path.name
    if name in (".env", ".envrc") or name.endswith(".env"):
        return True
    return name.startswith(".env.") and not name.endswith(_ENV_SAMPLES)


def _holds(place: Path, closed: Sequence[Path]) -> bool:
    """Whether one of ``closed`` lies beneath ``place``: a search that starts at
    ``place`` would walk into it."""
    return any(place in other.parents for other in closed)


@dataclass(frozen=True)
class Guard:
    """The guardrails of one agent: ``folder`` is its own, inside ``project``."""

    folder: Path
    project: Path
    sandbox: bool = True
    home: Path | None = None
    #: Places outside the agent's folder its shell and its file tools may read although
    #: they lie where reading is denied (the Python it runs code with).
    readable: Sequence[Path] = ()
    #: The identity token file, for an experiment agent whose run the person approved
    #: with network access while model access for jobs is on; ``None`` for everyone else.
    token_file: Path | None = None
    #: The environment of this process (``os.environ`` without): where it names its
    #: identity token and its directory for temporary files.
    environ: Mapping[str, str] | None = None
    #: The shared directories for temporary files, where they are not to be asked of
    #: this process (:meth:`shared_temp`).
    shared: Sequence[Path] | None = None
    #: The agent's own directory for temporary files (:func:`make_temp`), which its
    #: shell and its file tools read and write besides its folder.
    temp: Path | None = None

    # -- the places ----------------------------------------------------------------

    def _home(self) -> Path:
        return Path(os.path.realpath(self.home or Path.home()))

    def _folder(self) -> Path:
        return Path(os.path.realpath(self.folder))

    def _environ(self) -> Mapping[str, str]:
        return os.environ if self.environ is None else self.environ

    def _temp(self) -> Path | None:
        return None if self.temp is None else Path(os.path.realpath(self.temp))

    def _temp_parent(self) -> list[Path]:
        """The directory the agents' own directories for temporary files are made in:
        closed, so that no agent reads another's."""
        own = self._temp()
        return [] if own is None else [own.parent]

    def shared_temp(self) -> list[Path]:
        """Where this process, and so every agent's process before it is told otherwise,
        keeps temporary files: the person's other programs and the other agents keep
        theirs there too, so it is closed to the agent, which has a directory of its
        own (``temp``)."""
        if self.shared is not None:
            return list(dict.fromkeys(Path(os.path.realpath(place)) for place in self.shared))
        environ = self._environ()
        named = [environ.get(name) for name in ("TMPDIR", "CLAUDE_CODE_TMPDIR", "CLAUDE_TMPDIR")]
        uid = os.getuid() if hasattr(os, "getuid") else 0
        # (The last is where a Claude Code process keeps them when nothing is named.)
        places = [tempfile.gettempdir(), *named, f"/tmp/claude-{uid}"]
        return list(dict.fromkeys(Path(os.path.realpath(place)) for place in places if place))

    def _token_places(self) -> list[Path]:
        """The identity token file this process signs in with, as it is named and where
        that leads: a credential wherever it lies, unless it is this agent's to read."""
        named = str(self._environ().get("ANTHROPIC_IDENTITY_TOKEN_FILE") or "").strip()
        if not named:
            return []
        places = [Path(os.path.abspath(os.path.expanduser(named))), Path(os.path.realpath(named))]
        granted = self.token_file is not None and Path(os.path.realpath(self.token_file))
        return [place for place in dict.fromkeys(places) if place != granted]

    def credential_places(self) -> list[Path]:
        """Where credentials lie on this machine, whatever else is readable."""
        home = self._home()
        # Where Claude Code keeps the login, also when its directory is named elsewhere.
        login = Path(os.path.realpath(claude_config_dir(home=home))) / ".credentials.json"
        return [
            home / ".anthropic",
            home / ".config" / "anthropic",
            home / ".config" / "gh",
            home / ".config" / "op",
            home / ".ssh",
            home / ".claude" / ".credentials.json",
            *([login] if login != home / ".claude" / ".credentials.json" else []),
            home / ".claude.json",
            home / ".netrc",
            home / ".aws",
            home / ".config" / "gcloud",
            # Where `uv auth login` keeps what it is given, in plain text.
            home / ".local" / "share" / "uv" / "credentials",
            *self._token_places(),
        ]

    def hidden(self) -> list[Path]:
        """What the agent may not read: the person's home directory, the project, the
        place of the identity token and the shared directory for temporary files.
        Its own folder lies inside the project and is the exception
        (:meth:`open_places`)."""
        places = [self._home(), Path(os.path.realpath(self.project)), Path("/run/user")]
        return list(dict.fromkeys([*places, *self.shared_temp(), *self._temp_parent()]))

    def open_places(self) -> list[Path]:
        """What may be read although it lies in a hidden place."""
        home = self._home()
        places = [
            self._folder(),
            # The Python this process runs with, for the agent's own code: the packages
            # an experiment imports are there.
            Path(os.path.realpath(sys.prefix)),
            Path(os.path.realpath(sys.base_prefix)),
            # Programs the person installed for themselves (`uv` is usually one).
            home / ".local" / "bin",
            # The interpreters and tools `uv` installed: a virtual environment's Python
            # is a link into them.  Not all of uv's directory: its credentials lie there.
            home / ".local" / "share" / "uv" / "python",
            home / ".local" / "share" / "uv" / "tools",
            *_package_caches(home),
            home / ".cache" / "pip",
            *(Path(os.path.realpath(path)) for path in self.readable),
            *([self._temp()] if self.temp is not None else []),
        ]
        if self.token_file is not None:
            places.append(Path(os.path.realpath(self.token_file)))
        credentials = self.credential_places()
        return [
            place
            for place in dict.fromkeys(places)
            if not any(_within(place, secret) for secret in credentials)
        ]

    def may_read(self, path: Path) -> bool:
        """Whether the real path ``path`` is the agent's to read.

        A directory is read by searching it (Grep, Glob), and a search walks down:
        a directory that has a closed place anywhere beneath it is closed as well,
        whatever else lies in it.  ``/`` and ``/home`` are such directories.
        """
        if self.token_file is not None and path == Path(os.path.realpath(self.token_file)):
            return True
        credentials = self.credential_places()
        if _is_env_file(path) or any(_within(path, place) for place in credentials):
            return False
        if _holds(path, credentials):
            return False
        if not self.sandbox:
            return True
        if any(_within(path, tree) for tree in _SYSTEM_TREES):
            # Processes (their environment, their memory) and devices.  The agent's
            # process has refusals of its own for these; this one does not rest on them.
            return False
        if _holds(path, [*_SYSTEM_TREES, *self.hidden()]):
            # No exception for a place that is open: a search walks down into
            # whatever lies beneath.
            return False
        # The closest place decides: the agent's folder is open inside the hidden
        # project, the project is hidden wherever it lies, and what is named as
        # readable is open inside a hidden home directory.
        hidden = max((len(p.parts) for p in self.hidden() if _within(path, p)), default=0)
        opened = max((len(p.parts) for p in self.open_places() if _within(path, p)), default=0)
        return hidden == 0 or opened >= hidden

    def may_write(self, path: Path) -> bool:
        """Whether the real path ``path`` is the agent's to write: only its own folder,
        and its own directory for temporary files."""
        own = self._temp()
        return (
            not self.sandbox
            or _within(path, self._folder())
            or (own is not None and _within(path, own))
        )

    # -- the settings of the agent's process ------------------------------------------

    def settings(self) -> dict[str, Any]:
        """The permission rules and the sandbox, as the agent's process takes them."""
        credentials = self.credential_places()
        denied = [f"Read(/{place}/**)" for place in credentials if place.suffix == ""]
        denied += [f"Read(/{place})" for place in credentials]
        # (Two spellings: beneath the agent's folder, and anywhere on the machine.)
        for name in _ENV_NAMES:
            denied += [f"Read(**/{name})", f"Read(//**/{name})"]
        settings: dict[str, Any] = {"permissions": {"deny": list(dict.fromkeys(denied))}}
        if not self.sandbox:
            return settings
        settings["sandbox"] = {
            "enabled": True,
            # Where the sandbox cannot start, the agent's process refuses to work.
            # Without this it warns and runs every command outside any sandbox.
            "failIfUnavailable": True,
            # A command that runs in the sandbox needs nobody's leave; none runs outside it.
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                # What is hidden, and a credential that lies outside of it (a login
                # whose directory ``CLAUDE_CONFIG_DIR`` names somewhere else).
                "denyRead": [
                    *(str(place) for place in self.hidden()),
                    *(
                        str(place)
                        for place in credentials
                        if not any(_within(place, hidden) for hidden in self.hidden())
                    ),
                    # A file of settings with secrets outside of that.  On Linux the
                    # sandbox applies a pattern only where it looks for matches when a
                    # command starts, which is the agent's folder (checked live: `cat`
                    # of one there is refused, of one in /var/tmp it is not).  Elsewhere the shell can read such a
                    # file; the file tools cannot, wherever it lies.
                    *(f"{above}**/{name}" for name in _ENV_NAMES for above in ("", "/")),
                ],
                "allowRead": [str(place) for place in self.open_places()],
                # (Its folder is the shell's to write by itself; this is besides.)
                **({"allowWrite": [str(self._temp())]} if self.temp is not None else {}),
                # (The shared directory for temporary files besides: the agent's
                # process opens its own for writing, and that is the agent's now.)
                "denyWrite": [
                    *(str(place) for place in credentials),
                    *(
                        str(place)
                        for place in self.shared_temp()
                        if not _within(self._folder(), place)
                        and not (self.temp is not None and _within(self._temp(), place))
                    ),
                ],
            },
            # The shell has the web.  It does not have this machine's own services.
            # The sandbox refused link-local addresses before these names were here
            # (checked live); with them, the live check shows the metadata server and
            # the loopback both closed, and a later version cannot quietly open them.
            "network": {"deniedDomains": list(CLOSED_HOSTS)},
        }
        return settings

    # -- the check before every tool call ---------------------------------------------

    def check(self, use: ToolUse) -> str | None:
        """Why the tool call ``use`` is refused, as the agent is told; ``None`` to let it
        through.  Also asked for the calls of helpers the agent started."""
        given = use.input if isinstance(use.input, Mapping) else {}
        command = given.get("command")
        if isinstance(command, str) and runs_the_project_cli(command):
            return _BOARD_INSTEAD
        if use.name == "WebFetch" and isinstance(given.get("url"), str):
            # This tool runs in the agent's process, outside the shell's sandbox.
            return _NOT_A_PAGE if closed_host(given["url"]) else None
        folder = self._folder()
        if use.name in _WRITING_TOOLS:
            target = next((given[key] for key in _PATH_KEYS if isinstance(given.get(key), str)), "")
            if not target or not self.may_write(_real(target, folder)):
                return (
                    "You write files only in your own folder "
                    f"({folder}); nothing outside it is yours to change. What should "
                    "change in the wiki is proposed on the board."
                )
            return None
        if use.name in _READING_TOOLS:
            targets = [given[key] for key in _PATH_KEYS if isinstance(given.get(key), str)]
            pattern = given.get("pattern")
            if use.name == "Glob" and isinstance(pattern, str):
                # A pattern can name a place itself: what stands before its first wildcard.
                fixed = re.split(r"[*?\[{]", pattern, maxsplit=1)[0]
                if fixed.startswith(("/", "~")) or ".." in fixed.split("/"):
                    targets.append(fixed if fixed.endswith("/") else os.path.dirname(fixed) or "/")
                named = os.path.basename(pattern)
                if named.startswith(".env") or named.endswith(".env"):
                    targets.append(os.path.join(fixed or ".", ".env"))
            for target in targets or ["."]:
                if not self.may_read(_real(target, folder)):
                    return (
                        "That place is not yours to read: credentials, the researcher's "
                        "own files and the project's directory are closed to agents. The "
                        "wiki and the attached files are read with the memory tool; your "
                        "own files are in your folder."
                    )
        return None


def make_temp(environ: Mapping[str, str] | None = None) -> Path:
    """A new directory for one agent's temporary files, for it alone.

    Not the directory this process keeps its own in, which the person's other
    programs and the other agents share.  Not the agent's folder either: the
    path must be short (:data:`_TEMP_BYTES`).  It is made in a directory of this
    user's (``po-<uid>``, closed to others) under ``/var/tmp``, ``/tmp`` or what
    :data:`TEMP_ENV` names; the guard hides that directory from every agent and
    opens to each only its own.  Whoever made it removes it when the agent ends.
    """
    environ = os.environ if environ is None else environ
    uid = os.getuid() if hasattr(os, "getuid") else 0
    named = str(environ.get(TEMP_ENV) or "").strip()
    failed: OSError | None = None
    for base in [named] if named else ["/var/tmp", "/tmp"]:
        parent = Path(os.path.realpath(os.path.expanduser(base))) / f"po-{uid}"
        try:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if hasattr(os, "getuid") and parent.stat().st_uid != uid:
                raise PermissionError(f"{parent} belongs to another user")
            made = Path(tempfile.mkdtemp(prefix="", dir=parent))
        except OSError as exc:
            failed = exc
            continue
        most = _TEMP_BYTES - len(f"/claude-{uid}")
        if len(os.fsencode(made)) > most:
            shutil.rmtree(made, ignore_errors=True)
            failed = OSError(
                f"{made} is too long a path for an agent's temporary files (at most {most} bytes)"
            )
            continue
        return made
    raise OSError(
        "No directory for an agent's temporary files could be made"
        + (f" under {named} ({TEMP_ENV})" if named else " under /var/tmp or /tmp")
        + f": {failed}. Name a short directory of your own in {TEMP_ENV}."
    )


def _is_secret(name: str) -> bool:
    return bool(_SECRET_NAME.search(name)) and not name.upper().endswith(_NOT_A_SECRET_SUFFIX)


def agent_env(
    credential: str,
    environ: Mapping[str, str] | None = None,
    *,
    job_models: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment variables to set for an agent's process: an empty value takes a
    variable of this process away from it.

    Only what is on a short list reaches the agent (:data:`KEPT_ENV`, the ``LC_*``
    variables, and what the person names in :data:`PASS_ENV`); every other
    variable of this process is taken away, whatever it is called.  The agent's
    shell has the web, and a name does not say what its value opens: the socket
    of the person's ssh agent, a database's address with its password in it.
    Of the list, a variable whose value holds a password (a proxy's address
    with one) is taken away too.

    With the ``login`` credential no ``ANTHROPIC_*`` variable reaches it either,
    because the agent's process signs in with whichever of them it finds before
    it looks at the person's Claude Code login.  With ``federation`` the five
    federation settings stay, which are what it signs in with.  ``job_models``
    are the federation settings an experiment agent's code may use
    (``compute.local.job_model_env()``); they are passed on under other names
    (:data:`JOB_ENV_PREFIX`), for the launcher in its folder to put back.
    """
    environ = os.environ if environ is None else environ
    signs_in = set(FEDERATION_ENV) if credential == "federation" else set()
    named = {
        name
        for name in re.split(r"[,\s]+", str(environ.get(PASS_ENV) or ""))
        if name and not name.startswith(_NOT_PASSED_ON)
    }

    def stays(name: str, value: str) -> bool:
        if name in signs_in or name in named:
            return True
        if name not in KEPT_ENV and not name.startswith(_KEPT_ENV_PREFIXES):
            return False
        if name.upper() in _PROXY_NAMES and "@" in str(value):
            return False  # also without a scheme: `me:word@proxy:3128`
        return not _is_secret(name) and not _USERINFO.search(str(value))

    env = {name: "" for name, value in environ.items() if not stays(name, value)}
    for name, value in (job_models or {}).items():
        env[f"{JOB_ENV_PREFIX}{name}"] = value
    return env


def launcher_text() -> str:
    """The launcher an experiment agent starts its code with when the run may call the
    Anthropic API.  It holds no value: it gives the command the settings under the
    names the ``anthropic`` package reads."""
    names = " ".join(name.removeprefix("ANTHROPIC_") for name in FEDERATION_ENV)
    return (
        "#!/bin/sh\n"
        "# Starts a command with the settings for the Anthropic API that the researcher\n"
        "# granted this run: ./with-models python main.py\n"
        f"for name in {names}; do\n"
        f'  eval "value=\\${{{JOB_ENV_PREFIX}ANTHROPIC_$name:-}}"\n'
        '  if [ -n "$value" ]; then export "ANTHROPIC_$name=$value"; fi\n'
        "done\n"
        'exec "$@"\n'
    )
