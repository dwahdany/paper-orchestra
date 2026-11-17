"""What keeps an agent with a shell from deciding what is the person's to decide.

An SDK agent can run any command and read and write files.  The person keeps
the gate all the same: an agent must not be able to accept a proposal, approve
a run or change the files the project is steered by, and it must not get at
the credentials of the machine.  Four things see to that, and each stands on
its own:

- **The shell's sandbox** (the agent's process applies it to every command):
  the shell can write only inside the agent's own folder, and cannot read the
  person's home directory, the project's directory or the place the identity
  token lies, except for what is named as readable (its own folder, the Python
  it runs code with).  A file of settings with secrets (``.env``, ``.envrc``,
  ``*.env``) is named to it as well, which on Linux only holds in the agent's
  folder and its directory for temporary files: one that lies outside every
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
  do not start at a directory that has any of that beneath it (``/``, ``/home``), the tool that opens a web page does
  not open this machine's own services, and a shell command that runs the
  project's own command line is refused, with the way that is open instead:
  propose it on the board.  That last refusal reads the text of the command:
  it catches the command line and an import of the package where they are
  written out, not a script that assembles the name.  It tells an agent that
  means no harm what to do instead; what keeps the project's files from an
  agent that does is the sandbox.
- **The environment** the agent's process starts with (:func:`agent_env`): no
  credential of this process reaches it, except the ones its own sign-in uses.

``settings.agents.sandbox: false`` takes the first away and the file checks of
the third with it: the agent then works as the person's own Claude Code would,
and its shell reaches whatever the person's does: the project's files, this
machine's services, and on a cloud machine the metadata server and so the
machine's own account.  The refusal of the project's command line and the rules
about credentials stay, and without the sandbox they can be got around.
"""

from __future__ import annotations

import ipaddress
import os
import re
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
    "LAUNCHER",
    "Guard",
    "agent_env",
    "claude_config_dir",
    "closed_host",
    "launcher_text",
    "runs_the_project_cli",
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

_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
#: A name that says where something lies is no credential.  (``_ID`` is not among these:
#: ``AWS_ACCESS_KEY_ID`` is half of a credential, and a name is only asked about here
#: when it has one of the words above in it.)
_NOT_A_SECRET_SUFFIX = ("_FILE", "_PATH", "_DIR")
#: What separates the words of a shell command, for finding the programs it names.
_WORD_BREAKS = re.compile(r"[\s;|&()<>`$\"'=,{}\[\]]+")
_CLI_NAMES = ("paper-orchestra", "paper_orchestra")

_BOARD_INSTEAD = (
    "The project's own commands (review, approve, advance, run, cancel and the rest) are "
    "the researcher's, and an agent does not run them. Put what you want on the board "
    "instead: a new page with the memory tool, a change to a page with propose_edit. The "
    "researcher decides, and an experiment they approve is carried out by another agent."
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


def closed_host(url: str) -> bool:
    """Whether ``url`` names a host that is closed to agents (:data:`CLOSED_HOSTS`), or
    any other address of the loopback or the link-local range.  An address that does
    not parse is closed: nothing is opened that cannot be read."""
    try:
        host = urllib.parse.urlsplit(url if "//" in url else f"//{url}").hostname
    except ValueError:
        return True
    if not host:
        return True
    host = host.rstrip(".").lower()
    if host in CLOSED_HOSTS or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_link_local or address.is_unspecified


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


def _temporary_places() -> list[str]:
    """Where this process, and so the agent's, keeps temporary files."""
    named = [os.environ.get(name) for name in ("TMPDIR", "CLAUDE_CODE_TMPDIR")]
    return list(dict.fromkeys([tempfile.gettempdir(), *(path for path in named if path)]))


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

    # -- the places ----------------------------------------------------------------

    def _home(self) -> Path:
        return Path(os.path.realpath(self.home or Path.home()))

    def _folder(self) -> Path:
        return Path(os.path.realpath(self.folder))

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
        ]

    def hidden(self) -> list[Path]:
        """What the agent may not read: the person's home directory, the project and
        the place of the identity token.  Its own folder lies inside the project and
        is the exception (:meth:`open_places`)."""
        places = [self._home(), Path(os.path.realpath(self.project)), Path("/run/user")]
        return list(dict.fromkeys(places))

    def open_places(self) -> list[Path]:
        """What may be read although it lies in a hidden place."""
        home = self._home()
        places = [
            self._folder(),
            # The Python this process runs with, for the agent's own code: the packages
            # an experiment imports are there.
            Path(os.path.realpath(sys.prefix)),
            Path(os.path.realpath(sys.base_prefix)),
            home / ".local" / "bin",
            home / ".local" / "share" / "uv",
            home / ".cache" / "uv",
            home / ".cache" / "pip",
            *(Path(os.path.realpath(path)) for path in self.readable),
            # The directory for temporary files, where it lies in a hidden place: the
            # sandbox keeps there what connects a command to the network.
            *(Path(os.path.realpath(path)) for path in _temporary_places()),
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
            # No exception for a place that is open: the directory for temporary files
            # is, and a project that lies beneath it is hidden all the same.
            return False
        # The closest place decides: the agent's folder is open inside the hidden
        # project, the project is hidden wherever it lies, and what is named as
        # readable is open inside a hidden home directory.
        hidden = max((len(p.parts) for p in self.hidden() if _within(path, p)), default=0)
        opened = max((len(p.parts) for p in self.open_places() if _within(path, p)), default=0)
        return hidden == 0 or opened >= hidden

    def may_write(self, path: Path) -> bool:
        """Whether the real path ``path`` is the agent's to write: only its own folder."""
        return not self.sandbox or _within(path, self._folder())

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
                    # command starts, which is the agent's folder and its directory for
                    # temporary files (checked live: `cat` of one there is refused, of
                    # one in /var/tmp it is not).  Elsewhere the shell can read such a
                    # file; the file tools cannot, wherever it lies.
                    *(f"{above}**/{name}" for name in _ENV_NAMES for above in ("", "/")),
                ],
                "allowRead": [str(place) for place in self.open_places()],
                "denyWrite": [str(place) for place in credentials],
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

    Nothing that looks like a credential reaches the agent, whose shell has the web.
    With the ``login`` credential no ``ANTHROPIC_*`` variable reaches it either,
    because the agent's process signs in with whichever of them it finds before
    it looks at the person's Claude Code login.  With ``federation`` the five
    federation settings stay, which are what it signs in with.  ``job_models``
    are the federation settings an experiment agent's code may use
    (``compute.local.job_model_env()``); they are passed on under other names
    (:data:`JOB_ENV_PREFIX`), for the launcher in its folder to put back.
    """
    environ = os.environ if environ is None else environ
    kept = set(FEDERATION_ENV) if credential == "federation" else set()
    env = {
        name: ""
        for name in environ
        if name not in kept and (_is_secret(name) or name.startswith("ANTHROPIC_"))
    }
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
