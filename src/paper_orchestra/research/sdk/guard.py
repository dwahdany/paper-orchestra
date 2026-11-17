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
  it runs code with).  Whether the sandbox holds is checked against a real
  agent by ``tests/e2e/agent_guardrails.py``.
- **Permission rules** for the built-in file tools: reading the places where
  credentials lie is denied by the agent's process itself.
- **The check before every tool call** (:meth:`Guard.check`, run in this
  process as the agent's ``PreToolUse`` hook, also for helpers the agent
  starts): built-in tools that write stay inside the agent's folder, the ones
  that read stay out of what the sandbox hides, and a shell command that runs
  the project's own command line is refused, with the way that is open
  instead: propose it on the board.
- **The environment** the agent's process starts with (:func:`agent_env`): no
  credential of this process reaches it, except the ones its own sign-in uses.

``settings.agents.sandbox: false`` takes the first away and the file checks of
the third with it: the agent then works as the person's own Claude Code would.
The refusal of the project's command line and the rules about credentials stay.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_orchestra.research.sdk.driver import ToolUse

__all__ = [
    "FEDERATION_ENV",
    "JOB_ENV_PREFIX",
    "LAUNCHER",
    "Guard",
    "agent_env",
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
_NOT_A_SECRET_SUFFIX = ("_FILE", "_PATH", "_DIR", "_ID")
#: What separates the words of a shell command, for finding the programs it names.
_WORD_BREAKS = re.compile(r"[\s;|&()<>`$\"'=,{}\[\]]+")
_CLI_NAMES = ("paper-orchestra", "paper_orchestra")

_BOARD_INSTEAD = (
    "The project's own commands (review, approve, advance, run, cancel and the rest) are "
    "the researcher's, and an agent does not run them. Put what you want on the board "
    "instead: a new page with the memory tool, a change to a page with propose_edit. The "
    "researcher decides, and an experiment they approve is carried out by another agent."
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
    """
    for word in _WORD_BREAKS.split(str(command or "")):
        last = word.rstrip("/").rsplit("/", 1)[-1]
        if last in _CLI_NAMES or last.startswith("paper_orchestra."):
            return True
    return False


def _real(path: str | os.PathLike[str], base: Path) -> Path:
    """Where ``path`` leads, links followed, taken from ``base`` when it is relative."""
    text = os.path.expanduser(os.fspath(path))
    joined = text if os.path.isabs(text) else os.path.join(base, text)
    return Path(os.path.realpath(joined))


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_env_file(path: Path) -> bool:
    name = path.name
    return name == ".env" or (name.startswith(".env.") and not name.endswith(".example"))


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
        return [
            home / ".anthropic",
            home / ".config" / "anthropic",
            home / ".config" / "gh",
            home / ".config" / "op",
            home / ".ssh",
            home / ".claude" / ".credentials.json",
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
        """Whether the real path ``path`` is the agent's to read."""
        if self.token_file is not None and path == Path(os.path.realpath(self.token_file)):
            return True
        if _is_env_file(path) or any(_within(path, place) for place in self.credential_places()):
            return False
        if not self.sandbox:
            return True
        if any(_within(path, place) for place in self.open_places()):
            return True
        return not any(_within(path, place) for place in self.hidden())

    def may_write(self, path: Path) -> bool:
        """Whether the real path ``path`` is the agent's to write: only its own folder."""
        return not self.sandbox or _within(path, self._folder())

    # -- the settings of the agent's process ------------------------------------------

    def settings(self) -> dict[str, Any]:
        """The permission rules and the sandbox, as the agent's process takes them."""
        denied = [f"Read(/{place}/**)" for place in self.credential_places() if place.suffix == ""]
        denied += [f"Read(/{place})" for place in self.credential_places()]
        denied += ["Read(**/.env)", "Read(**/.env.local)"]
        settings: dict[str, Any] = {"permissions": {"deny": list(dict.fromkeys(denied))}}
        if not self.sandbox:
            return settings
        settings["sandbox"] = {
            "enabled": True,
            # A command that runs in the sandbox needs nobody's leave; none runs outside it.
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "denyRead": [str(place) for place in self.hidden()],
                "allowRead": [str(place) for place in self.open_places()],
                "denyWrite": [str(place) for place in self.credential_places()],
            },
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
                if os.path.basename(pattern).startswith(".env"):
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
