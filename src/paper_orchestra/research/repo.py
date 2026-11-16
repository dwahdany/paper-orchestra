"""A copy of the person's repository for one run of an experiment.

An experiment in ``repo`` mode runs a command inside the person's own code.  It
never runs in their working tree: what runs is a copy made for the run, and
what the designer wants changed is a patch applied to that copy.  This module
makes the copy and applies the patch; it knows nothing about projects or runs.

* A local directory is copied as git sees it where git is at hand (tracked
  files and untracked ones that are not ignored), and walked otherwise.  Either
  way version-control data, caches, environments, ``.env`` files and files
  above a size limit stay behind, and what stayed behind is reported.  Reading
  the repository runs git and nothing else: no program that the repository's
  configuration names (a file-system monitor, a hook, a filter) is started, and
  git itself gets an environment without credentials.
* A repository behind a URL is not fetched here at all.  :func:`clone_source`
  writes a small directory that makes the *job* clone it, so that the network
  is used only where, and only if, the person approved it.  A password or
  token in the URL is used for that fetch and is gone from the job's working
  directory before the experiment's command starts.
* :func:`apply_patch` uses ``git apply`` and falls back to a small applier of
  its own, which also reads the patches a model writes: right in what they
  change, often wrong in their line counts.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import re
import selectors
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from paper_orchestra.compute.base import (
    CHECK_SECONDS,
    SOURCE_EXCLUDES,
    JobStopped,
    SourceReport,
    human_size,
    walk_source,
)
from paper_orchestra.research.model import ProjectError

#: The file in an experiment's code directory that holds the changes to the repository.
PATCH_NAME = "patch.diff"
#: A file larger than this is not what an experiment's code needs; it stays behind.
MAX_FILE_BYTES = 50 * 1024 * 1024
#: A copy larger than this is refused: it is made once per run.
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
#: How many of the files left behind are listed by name.
MAX_SKIPPED_LISTED = 200

#: What a run brings back.  These paths hold what the experiment's command wrote, so a
#: patch may not put them there, and a clone job removes them before the command starts.
RESULT_PATHS = ("metrics.json", "artifacts")
#: Seconds ``git archive`` may take to write the tree of a commit: a copy is at most
#: ``MAX_TOTAL_BYTES``, read from a local disk.
ARCHIVE_SECONDS = 1800.0

#: Where the files of a clone job live inside its working directory.
SEED_DIR = ".paper-orchestra"
#: Exit statuses of the clone job's own steps (see :func:`clone_failure`).
EXIT_NO_GIT = 97
EXIT_CLONE_FAILED = 96
EXIT_PATCH_FAILED = 95
EXIT_FILE_EXISTS = 94
#: Starts every line the clone job prints itself.
CLONE_PREFIX = "[repo] "

_URL = re.compile(r"^(?:https?|ssh|git)://(?:[^\s@/]+@)?[A-Za-z0-9][^\s]*$", re.IGNORECASE)
_SCP_URL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*@[A-Za-z0-9][A-Za-z0-9_.\-]*:[^\s]+$")
_REF = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./\-~^]*$")
# The user information of a URL: all that stands before the *last* ``@`` of the part
# that names the host, as git and curl read it.  A password with an ``@`` in it is
# written that way more often than it is escaped.
_USERINFO = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*://)([^/?#\s]*)@")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_SECONDS = 120.0
# The name of a filter driver, from a key of git's configuration such as
# ``filter.lfs.smudge``.  The name may itself hold dots.
_FILTER_KEY = re.compile(r"filter\.(.+)\.(?:clean|smudge|process|required)", re.DOTALL)
#: What git gets of this process's environment: enough to find itself, the person's own
#: configuration (their ignore file, the directories they declared safe) and a place for
#: temporary files.  No credentials, and nothing that redirects git (``GIT_DIR``, ...).
_GIT_ENV = (
    "PATH",
    "HOME",
    "XDG_CONFIG_HOME",
    "TMPDIR",
    "SYSTEMROOT",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    # Which of the person's own configuration files git reads is theirs to say.
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
)
#: Options for every call of git: nothing a repository's configuration names is started
#: on its behalf.  A file-system monitor would be run by any command that reads the
#: index, a hook by some that write it, a pager by none here.
_GIT_INERT = (
    "--no-pager",
    "-c",
    "core.fsmonitor=false",
    "-c",
    f"core.hooksPath={os.devnull}",
)
#: The one "protocol" git may use to reach another repository, which is none: the name
#: is no protocol of git's, and nothing can stand in for one under it (no alias and no
#: helper program is found by a name with ``_`` in it).  See :func:`_git_env`.
_NO_TRANSPORT = "no_transport"
#: Keys of a configuration that make a repository a partial clone: it holds part of
#: the files of its history and names the remote git fetches the others from.
_PROMISOR_KEYS = r"^(extensions\.partialclone|remote\..*\.promisor)$"

_BY_NAME = "left out by name (version-control data, caches, environments and .env files)"
_OWN_FILES = "the research project's own files, which are not part of the repository"
_VENV = "virtual environment"
_OUTSIDE = "a link that leads out of the repository, or a special file"
_BEYOND = "listed by git under a path that leads out of the repository (up, or through a link)"
_THROUGH_A_LINK = "named where the copy already holds a link or a file, or below a link"


class RepoError(ProjectError):
    """The copy of the repository could not be made; the message says what to do."""


class PatchError(RepoError):
    """A patch that cannot be applied.  ``hunk`` is the rejected hunk, when one was."""

    def __init__(self, message: str, *, hunk: str = "") -> None:
        super().__init__(message)
        self.hunk = hunk


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def is_url(location: str) -> bool:
    """Whether ``location`` names a remote repository rather than a directory."""
    text = str(location or "").strip()
    return "://" in text or bool(_SCP_URL.match(text))


def url_problem(url: str) -> str | None:
    """Why ``url`` cannot be cloned, or ``None``.

    Only network transports are accepted.  ``file://`` would read this machine
    and ``ext::`` runs a command; a host that starts with a dash is an option
    in disguise.
    """
    text = str(url or "").strip()
    if _URL.match(text) or _SCP_URL.match(text):
        return None
    return (
        f"{display_url(text)!r} is not a repository URL that can be cloned: use an https://, "
        "ssh:// or git:// URL (or git@host:owner/name.git), or the path of a local directory."
    )


def userinfo(location: str) -> str | None:
    """The secret in the address of a repository, for masking; ``None`` if it has none.

    That is the token of ``https://token@host/...`` and the password of
    ``scheme://user:password@host/...``, exactly as written in the URL.  A path has
    none, and the user name of an ssh address (``ssh://git@host``, ``git@host:...``)
    is not one.  Where the secret is printed, :func:`display_url` shows ``***``.
    """
    text = str(location or "").strip()
    match = _USERINFO.match(text)
    if match is None:
        return None
    user, colon, password = match.group(2).partition(":")
    if colon:
        return password or None
    return user or None if match.group(1).lower().startswith("http") else None


def _without_secret(location: str) -> str:
    """A URL as a tool may be told it: without its password, and over http(s) without
    any user information, where a lone user name is how a token is passed."""
    text = str(location or "").strip()
    match = _USERINFO.match(text)
    if match is None or userinfo(text) is None:
        return text
    rest = text[match.end() :]
    if match.group(1).lower().startswith("http"):
        return match.group(1) + rest
    return f"{match.group(1)}{match.group(2).partition(':')[0]}@{rest}"


def display_url(location: str) -> str:
    """``location`` as it may be written down: without the password or token of a URL.

    A run's record, its result page and its log are read by people and by agents
    and travel with the project; what lets a job clone a private repository
    stays in the project's settings.  ``https://token@host`` and
    ``scheme://user:password@host`` lose the secret; a path, and the user name
    of an ssh URL, are left as they are.
    """
    text = str(location or "").strip()
    match = _USERINFO.match(text)
    if match is None:
        return text
    scheme, (user, colon, _) = match.group(1), match.group(2).partition(":")
    # Over https a lone user name is how an access token is passed.
    token = not colon and scheme.lower().startswith("http")
    shown = "***" if token else f"{user}:***" if colon else user
    return f"{scheme}{shown}@{text[match.end() :]}"


def ref_problem(ref: str) -> str | None:
    """Why ``ref`` cannot name a branch, tag or commit, or ``None``."""
    text = str(ref or "")
    if _REF.match(text) and ".." not in text:
        return None
    return (
        f"{text!r} is not a branch, tag or commit: use letters, digits and . _ / - only, "
        "starting with a letter or digit."
    )


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def git_available() -> bool:
    return shutil.which("git") is not None


def _git_env(ceiling: Path | None = None) -> dict[str, str]:
    """The environment git runs in: see ``_GIT_ENV``.

    ``GIT_OPTIONAL_LOCKS=0`` keeps read-only commands (``status``) from
    refreshing the index: the person's repository is looked at, never written.
    ``ceiling`` stops git from finding a repository above that directory.

    Nor does git fetch anything.  A partial clone names a remote from which git
    gets, by itself, a file that a command needs and the repository lacks, through
    whatever program the configuration names for the way there (``core.sshCommand``,
    ``remote.<name>.uploadpack``, an ``ext::`` address).  ``GIT_NO_LAZY_FETCH`` tells
    git not to (2.45 and later); for a git that does not know it, no protocol is
    allowed, which overrides what the repository's configuration allows.
    """
    env = {key: os.environ[key] for key in _GIT_ENV if key in os.environ}
    env.setdefault("PATH", os.defpath)
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C", GIT_PAGER="cat")
    env.update(GIT_NO_LAZY_FETCH="1", GIT_ALLOW_PROTOCOL=_NO_TRANSPORT)
    if ceiling is not None:
        env["GIT_CEILING_DIRECTORIES"] = str(ceiling)
    return env


def _git(
    args: list[str], *, cwd: Path, ceiling: Path | None = None, quiet: Sequence[str] = ()
) -> subprocess.CompletedProcess[bytes]:
    """Run git without ever asking for input, taking a lock in the repository or
    starting a program the repository names (``_GIT_INERT``), in an environment
    without credentials.  ``quiet`` are further options that switch such programs off
    (see :func:`_quiet_filters`)."""
    command = ["git", *_GIT_INERT, *quiet, *args]
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=_git_env(ceiling),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_GIT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(command, 1, b"", str(exc).encode())


def _said(done: subprocess.CompletedProcess[bytes]) -> str:
    return (done.stderr or done.stdout).decode("utf-8", errors="replace").strip()


def _in_work_tree(directory: Path) -> bool:
    done = _git(["rev-parse", "--is-inside-work-tree"], cwd=directory)
    return done.returncode == 0 and done.stdout.strip() == b"true"


def _quiet_filters(source: Path) -> list[str] | None:
    """Options that switch off every filter the configuration in force in ``source``
    defines; ``None`` when one cannot be switched off this way.

    A filter is a program that git runs on a file's content: when it compares the
    working tree with the index (``status``) and when it writes files out
    (``archive``).  Which files go through which filter is said by ``.gitattributes``
    in the tree; what the program is, by the configuration, that of the repository
    included.  Copying a repository must not start programs it names, so each driver's
    commands are replaced by nothing for the one call.  The option that does this
    (``-c name=value``) cannot carry a name with ``=`` in it.
    """
    done = _git(["config", "--includes", "--list", "-z"], cwd=source)
    if done.returncode != 0:
        return None
    names: set[str] = set()
    for entry in done.stdout.split(b"\0"):
        key = entry.split(b"\n", 1)[0].decode("utf-8", errors="surrogateescape")
        match = _FILTER_KEY.fullmatch(key)
        if match:
            names.add(match.group(1))
    if any("=" in name for name in names):
        return None
    options: list[str] = []
    for name in sorted(names):
        for command in ("clean", "smudge", "process"):
            options += ["-c", f"filter.{name}.{command}="]
        # A filter that "must succeed" would otherwise fail every file it is not run on.
        options += ["-c", f"filter.{name}.required=false"]
    return options


def _is_partial_clone(source: Path) -> bool:
    """Whether the repository in ``source`` holds only part of the files of its history
    and names a remote for the rest."""
    done = _git(["config", "--includes", "--get-regexp", _PROMISOR_KEYS], cwd=source)
    return done.returncode == 0 and bool(done.stdout.strip())


def _is_dirty(source: Path, quiet: Sequence[str] | None) -> bool | None:
    """Whether the working tree differs from its last commit; ``None`` when not known.

    git compares content where a file's time stamp differs from the index.  That is
    done with the filters switched off, and not at all when they cannot be: the flag
    is a note on the run's record and not worth running a repository's programs for.
    """
    if quiet is None:
        return None
    status = _git(["status", "--porcelain", "--ignore-submodules=all"], cwd=source, quiet=quiet)
    return bool(status.stdout.strip()) if status.returncode == 0 else None


# ---------------------------------------------------------------------------
# Copying a local repository
# ---------------------------------------------------------------------------


@dataclass
class RepoCopy:
    """What a copy was made from, and what it leaves out."""

    #: The directory or URL the copy comes from.
    source: str
    #: ``git`` (as git lists the working tree), ``archive`` (the tree at ``ref``),
    #: ``walk`` (every file, without git) or ``clone`` (made by the job itself).
    method: str
    ref: str | None = None
    #: The commit the copy is of (``archive``) or was made on top of (``git``).
    commit: str | None = None
    #: Whether the working tree differed from that commit; ``None`` when not known.
    dirty: bool | None = None
    files: int = 0
    bytes: int = 0
    #: ``{"path", "reason"}`` for what was left behind, at most ``MAX_SKIPPED_LISTED``.
    skipped: list[dict[str, str]] = field(default_factory=list)
    #: How many more were left behind than are listed.
    skipped_more: int = 0

    def skip(self, path: str, reason: str) -> None:
        if len(self.skipped) < MAX_SKIPPED_LISTED:
            self.skipped.append({"path": path, "reason": reason})
        else:
            self.skipped_more += 1

    def to_dict(self) -> dict[str, Any]:
        """What a run record keeps about the copy (the skipped files are listed apart)."""
        data: dict[str, Any] = {"source": self.source, "ref": self.ref, "method": self.method}
        if self.method != "clone":
            data.update(commit=self.commit, dirty=self.dirty, files=self.files, bytes=self.bytes)
            data["skipped_more"] = self.skipped_more
        return data


class _Stopper:
    """Asks ``should_stop`` now and then while many files are copied."""

    def __init__(self, should_stop: Callable[[], bool] | None) -> None:
        self._should_stop = should_stop
        self._next = 0.0

    def check(self) -> None:
        if self._should_stop is None or time.monotonic() < self._next:
            return
        if self._should_stop():
            raise JobStopped
        self._next = time.monotonic() + CHECK_SECONDS


class _Budget:
    """The size limits of one copy."""

    def __init__(self, copy: RepoCopy, max_file_bytes: int, max_total_bytes: int) -> None:
        self._copy = copy
        self._max_file = max_file_bytes
        self._max_total = max_total_bytes

    def admits(self, relative: str, size: int) -> bool:
        """Whether a file of ``size`` bytes is copied; counts it if so."""
        if size > self._max_file:
            self._copy.skip(
                relative, f"larger than {human_size(self._max_file)} ({human_size(size)})"
            )
            return False
        if self._copy.bytes + size > self._max_total:
            raise RepoError(
                f"The copy of {self._copy.source} would be larger than "
                f"{human_size(self._max_total)}, which is the most that is copied for a run. "
                "Keep data and build output out of it (list them in .gitignore), or give "
                "the experiment a smaller directory to run in."
            )
        self._copy.files += 1
        self._copy.bytes += size
        return True


def _excluded(relative: str) -> bool:
    """Whether a part of the path is one of the names that never go into a working directory."""
    return any(
        fnmatch.fnmatchcase(part, pattern)
        for part in PurePosixPath(relative).parts
        for pattern in SOURCE_EXCLUDES
    )


class _Environments:
    """Finds the virtual environments in a tree: directories that hold ``pyvenv.cfg``."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._known: dict[str, bool] = {}

    def holds(self, relative: str) -> bool:
        """Whether ``relative`` lies in a virtual environment."""
        folder = PurePosixPath(relative).parent
        while folder.parts:
            key = folder.as_posix()
            if key not in self._known:
                self._known[key] = (self._root / key / "pyvenv.cfg").is_file()
            if self._known[key]:
                return True
            folder = folder.parent
        return False


class _Tree:
    """Tells which of the paths git lists lie in the tree itself.

    What git lists comes from the repository's index, which is a file like any other
    and may have come with the directory.  It can name a path that leads up and out
    (``../``), an absolute one, or a file below a directory that is a link by now.
    Reading any of them would bring files from outside the repository into the copy,
    and placing them could write outside the copy.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._links: dict[str, bool] = {}

    def holds(self, relative: str) -> bool:
        """Whether ``relative`` names a place inside the tree that no link leads to."""
        path = PurePosixPath(relative)
        if path.is_absolute() or os.path.isabs(relative) or ".." in path.parts:
            return False
        folder = path.parent
        while folder.parts:
            key = folder.as_posix()
            if key not in self._links:
                self._links[key] = (self._root / key).is_symlink()
            if self._links[key]:
                return False
            folder = folder.parent
        return True


def _link_stays_inside(path: Path, relative: str) -> bool:
    """Whether the link is relative and still points into the tree once the tree is copied."""
    target = os.readlink(path)
    if os.path.isabs(target):
        return False
    resolved = os.path.normpath(os.path.join(os.path.dirname(relative), target))
    return resolved != ".." and not resolved.startswith(".." + os.sep)


class _Landing:
    """Tells whether a file, link or directory would be written inside the copy.

    ``git archive`` writes out what a commit says, and a commit that came with the
    directory can say what no working tree holds: a link and, under the same name, a
    directory or a file.  Written in that order the file would go where the link
    leads, and a link can lead out of the copy although its target is spelled as a
    path inside it (it passes through another link).  A file system that takes
    ``Data`` and ``data`` for one name makes the same of an ordinary tree.  So
    nothing is written through a link, and nothing where something already is.
    """

    def __init__(self, dest: Path) -> None:
        self._dest = dest
        #: Directories of the copy that are known to be directories, not links.  They
        #: stay that: nothing is ever placed where something is.
        self._real: set[str] = set()

    def admits(self, relative: str, *, directory: bool) -> bool:
        parts = PurePosixPath(relative).parts
        for depth in range(1, len(parts)):
            key = "/".join(parts[:depth])
            if key in self._real:
                continue
            path = self._dest / key
            if path.is_symlink() or (os.path.lexists(path) and not path.is_dir()):
                return False
            if path.is_dir():
                self._real.add(key)
        target = self._dest / relative
        if not os.path.lexists(target):
            return True
        return directory and target.is_dir() and not target.is_symlink()


def _place(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        os.symlink(os.readlink(source), target)
    else:
        shutil.copy2(source, target, follow_symlinks=False)


def _listed_by_git(source: Path) -> list[str]:
    """The files of the working tree as git sees them: tracked, and untracked but not ignored."""
    done = _git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=source)
    if done.returncode != 0:
        raise RepoError(f"git could not list the files of {source}: {_said(done)}")
    names = (os.fsdecode(name) for name in done.stdout.split(b"\0") if name)
    return list(dict.fromkeys(names))


def _own_prefixes(source: Path, own: Iterable[Path], copy: RepoCopy) -> tuple[str, ...]:
    """The directories in ``own`` that lie inside ``source``, as relative prefixes.

    A copy is made inside the project, and a project may live inside the repository
    it experiments on.  Its files are not the repository's: copied along, the copy
    would hold the whole research workspace, and without git it would hold itself.
    """
    base = source.resolve()
    prefixes: list[str] = []
    for directory in own:
        try:
            relative = directory.resolve().relative_to(base).as_posix()
        except (ValueError, OSError):
            continue  # not inside the repository: nothing to leave out
        if relative == ".":
            raise RepoError(
                f"The repository {source} is the research project's own directory. Point "
                "the project's repository setting at the code the experiments should run in."
            )
        if f"{relative}/" not in prefixes and (source / relative).is_dir():
            prefixes.append(f"{relative}/")
            copy.skip(relative, _OWN_FILES)
    return tuple(prefixes)


def _copy_listed(
    source: Path,
    dest: Path,
    copy: RepoCopy,
    budget: _Budget,
    stop: _Stopper,
    own: tuple[str, ...],
) -> None:
    environments = _Environments(source)
    tree = _Tree(source)
    landing = _Landing(dest)
    for relative in _listed_by_git(source):
        stop.check()
        if not tree.holds(relative):
            copy.skip(relative, _BEYOND)
            continue
        if relative.startswith(own):
            continue
        path = source / relative
        try:
            mode = path.lstat().st_mode
        except OSError:
            continue  # tracked, but deleted in the working tree
        if _excluded(relative):
            copy.skip(relative, _BY_NAME)
        elif environments.holds(relative):
            copy.skip(relative, _VENV)
        elif stat.S_ISDIR(mode):
            copy.skip(relative, "a repository inside the repository (a submodule)")
        elif not landing.admits(relative, directory=False):
            copy.skip(relative, _THROUGH_A_LINK)
        elif stat.S_ISLNK(mode):
            if _link_stays_inside(path, relative):
                _place(path, dest / relative)
            else:
                copy.skip(relative, _OUTSIDE)
        elif not stat.S_ISREG(mode):
            copy.skip(relative, _OUTSIDE)
        elif budget.admits(relative, path.lstat().st_size):
            _place(path, dest / relative)


def _copy_walked(
    source: Path,
    dest: Path,
    copy: RepoCopy,
    budget: _Budget,
    should_stop: Callable[[], bool] | None,
    own: tuple[str, ...],
) -> None:
    report = SourceReport()
    environments: list[str] = []
    landing = _Landing(dest)
    for relative, path, kind in walk_source(source, report, should_stop):
        if f"{relative}/".startswith((*own, *environments)):
            continue
        if kind == "dir":
            # Directories are made when a file goes into them: as with git, one that
            # holds nothing that is copied is not part of the copy.
            if (path / "pyvenv.cfg").is_file():
                environments.append(relative + "/")
                copy.skip(relative, _VENV)
        elif not landing.admits(relative, directory=False):
            copy.skip(relative, _THROUGH_A_LINK)
        elif kind == "link" or budget.admits(relative, path.lstat().st_size):
            _place(path, dest / relative)
    for relative in report.excluded:
        copy.skip(relative, _BY_NAME)
    for relative in report.skipped:
        copy.skip(relative, _OUTSIDE)


def _resolve_ref(source: Path, ref: str) -> str:
    problem = ref_problem(ref)
    if problem:
        raise RepoError(problem)
    done = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=source)
    commit = done.stdout.decode("ascii", errors="replace").strip()
    if done.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise RepoError(
            f"{source} has no branch, tag or commit named {ref!r}. Check the ref in the "
            "project's repository settings."
        )
    return commit


def _member_problem(member: tarfile.TarInfo) -> str | None:
    """Why a member of an archive is not extracted, or ``None``."""
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts:
        return _OUTSIDE
    if _excluded(member.name):
        return _BY_NAME
    if member.issym():
        target = os.path.normpath(os.path.join(os.path.dirname(member.name), member.linkname))
        inside = not os.path.isabs(member.linkname) and not target.startswith("..")
        return None if inside else _OUTSIDE
    return None if member.isreg() or member.isdir() else _OUTSIDE


def _extract(
    archive: tarfile.TarFile, dest: Path, copy: RepoCopy, budget: _Budget, stop: _Stopper
) -> None:
    """Write the members of ``archive`` that belong in a copy under ``dest``."""
    landing = _Landing(dest)
    for member in archive:
        stop.check()
        relative = member.name.rstrip("/")
        problem = _member_problem(member)
        if problem is None and not landing.admits(relative, directory=member.isdir()):
            problem = _THROUGH_A_LINK
        if problem:
            copy.skip(relative, problem)
        elif member.isdir():
            (dest / relative).mkdir(parents=True, exist_ok=True)
        elif member.issym():
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            os.symlink(member.linkname, dest / relative)
        elif budget.admits(relative, member.size):
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            reader = archive.extractfile(member)
            assert reader is not None
            with reader, open(dest / relative, "wb") as writer:
                shutil.copyfileobj(reader, writer)
            os.chmod(dest / relative, member.mode & 0o777 or 0o644)


class _Watched:
    """The output of a git process, read with a deadline and an ear for a stop request.

    Reading a pipe blocks while the other side is silent.  This reader waits in short
    turns instead and, between them, asks ``stop`` (which raises ``JobStopped``) and
    looks at the clock (:class:`RepoError` once ``seconds`` have passed), so that a
    git that hangs cannot hold a run for ever.  It runs on the caller's thread.
    """

    def __init__(self, stream: Any, stop: _Stopper, seconds: float, what: str) -> None:
        self._stream = stream
        self._stop = stop
        self._seconds = seconds
        self._deadline = time.monotonic() + seconds
        self._what = what
        self._selector: selectors.BaseSelector | None = None
        if os.name == "posix":  # elsewhere a pipe cannot be waited for; reads then block
            self._selector = selectors.DefaultSelector()
            self._selector.register(stream.fileno(), selectors.EVENT_READ)

    def read(self, size: int = -1) -> bytes:
        if self._selector is None:
            return self._stream.read(size)
        while not self._selector.select(max(CHECK_SECONDS, 0.05)):
            self._stop.check()
            if time.monotonic() >= self._deadline:
                raise RepoError(
                    f"{self._what} took longer than {self._seconds:g} s and was stopped. "
                    "Check that the repository's disk responds, or copy the working tree "
                    "as it is by removing the ref from the project's repository settings."
                )
        self._stop.check()
        return os.read(self._stream.fileno(), size if size > 0 else 1 << 16)

    def close(self) -> None:
        if self._selector is not None:
            self._selector.close()


def _copy_archived(
    source: Path,
    dest: Path,
    commit: str,
    copy: RepoCopy,
    budget: _Budget,
    stop: _Stopper,
    quiet: Sequence[str],
) -> None:
    """Extract the tree of ``commit`` as ``git archive`` writes it, member by member.

    The files are taken as they are stored: with ``quiet`` no filter is run on them.
    """
    # What git says goes to a file: a pipe nobody reads while the archive is read would
    # fill up and stop git.
    with tempfile.TemporaryFile() as said:
        process = subprocess.Popen(
            ["git", *_GIT_INERT, *quiet, "archive", "--format=tar", commit],
            cwd=source,
            env=_git_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=said,
        )
        assert process.stdout is not None
        what = f"Exporting {commit[:12]} of {source} with git"
        output = _Watched(process.stdout, stop, ARCHIVE_SECONDS, what)
        failed: Exception | None = None
        finished = False
        try:
            with tarfile.open(fileobj=output, mode="r|", bufsize=1 << 16) as archive:
                _extract(archive, dest, copy, budget, stop)
            # git pads the archive after its last member; let it finish writing.
            while output.read(1 << 16):
                pass
            finished = True
        except tarfile.TarError as exc:
            failed = exc
        finally:
            # A copy that ends early (stopped, too large, a broken archive) leaves git
            # writing into a pipe nobody reads any more.  Waiting for it would never end.
            if not finished:
                process.kill()
            output.close()
            process.stdout.close()
            status = process.wait()
        said.seek(0)
        words = said.read().decode("utf-8", errors="replace").strip()
    if failed is not None or status != 0:
        why = f"git could not export {commit[:12]} of {source}: {words or failed}"
        if _is_partial_clone(source):
            why += (
                f"\n{source} is a partial clone that does not hold every file of that "
                "commit, and nothing is fetched for a copy: git would contact the remote, "
                "with the programs the repository's configuration names, from this "
                f"machine. Let git fetch the files once yourself (`git -C {source} archive "
                f"-o {os.devnull} {commit[:12]}` does), or remove the ref from the project's "
                "repository settings to copy the working tree as it is."
            )
        raise RepoError(why)


def copy_local(
    source: str | Path,
    dest: str | Path,
    *,
    ref: str | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    use_git: bool | None = None,
    exclude: Iterable[str | Path] = (),
) -> RepoCopy:
    """Copy the repository in ``source`` into the new directory ``dest``.

    ``source`` is only read.  Without ``ref`` the copy is the working tree as it
    is now, uncommitted changes included: with git (``use_git``; by default
    whenever git is installed and ``source`` is in a work tree) the files git
    tracks plus the untracked ones ``.gitignore`` does not exclude, otherwise
    every file.  With ``ref`` it is the committed tree of that branch, tag or
    commit, which needs git.  ``.git``, caches, virtual environments, ``.env``
    files, links that lead out of the tree and files above ``max_file_bytes``
    are never copied; :attr:`RepoCopy.skipped` lists them.  Neither are ``dest``
    itself and the directories in ``exclude`` where they lie inside ``source``
    (the research project, when it lives in the repository it experiments on).

    The copy holds what is in the repository and nothing else, whatever the
    repository's own data says: a path git lists that leads out of the tree (up,
    or through a directory that is a link) is not read, nothing is written
    through a link or where the copy already holds something, and git fetches
    nothing (a partial clone is copied as far as its files are on this disk).

    Raises :class:`RepoError` when ``source`` is not a directory, the ref does
    not exist, a file cannot be read or written, or the copy would exceed
    ``max_total_bytes``, and ``JobStopped`` when ``should_stop`` turns true on
    the way.  ``dest`` may then hold part of the copy; the caller removes it.
    """
    source, dest = Path(source), Path(dest)
    if not source.is_dir():
        raise RepoError(
            f"The repository {source} does not exist or is not a directory. Set the "
            "repository in the project's settings to a directory on this machine."
        )
    with_git = git_available() and _in_work_tree(source) if use_git is None else use_git
    if ref and not with_git:
        raise RepoError(
            f"Copying {source} at {ref!r} needs git and a git repository; without them only "
            "the files as they are now can be copied. Install git, or remove the ref."
        )
    copy = RepoCopy(source=str(source), method="walk", ref=ref or None)
    budget = _Budget(copy, max_file_bytes, max_total_bytes)
    stop = _Stopper(should_stop)
    stop.check()
    dest.mkdir(parents=True, exist_ok=True)
    own = _own_prefixes(source, [*(Path(directory) for directory in exclude), dest], copy)
    try:
        if ref:
            copy.method = "archive"
            copy.commit = _resolve_ref(source, ref)
            quiet = _quiet_filters(source)
            if quiet is None:
                raise RepoError(
                    f"The git configuration of {source} defines a filter that cannot be "
                    "switched off for a copy (its name holds '=', or the configuration "
                    "cannot be read), and a copy never runs a repository's programs. Remove "
                    "the ref from the project's repository settings to copy the working "
                    "tree as it is, or rename the filter."
                )
            _copy_archived(source, dest, copy.commit, copy, budget, stop, quiet)
        elif with_git:
            copy.method = "git"
            head = _git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=source)
            copy.commit = head.stdout.decode("ascii", errors="replace").strip() or None
            copy.dirty = _is_dirty(source, _quiet_filters(source))
            _copy_listed(source, dest, copy, budget, stop, own)
        else:
            _copy_walked(source, dest, copy, budget, should_stop, own)
    except OSError as exc:
        # A file nobody may read, a disk that is full: the run cannot have its copy, and
        # a copy with holes in it would fail later in ways nobody could explain.
        raise RepoError(
            f"The repository {source} could not be copied for the run: {exc}. Make the file "
            "readable, or keep it out of the copy by listing it in .gitignore; if the "
            "disk is full, make room."
        ) from exc
    return copy


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


@dataclass
class Hunk:
    """One ``@@`` block of a unified diff."""

    header: str
    #: 1-based line of the old file where the hunk says it starts, and how many lines of
    #: the old file it says it covers (0: it only adds lines, after that line).
    old_start: int
    old_count: int = 1
    #: The lines of the hunk without their newline, each with its ``' '``, ``'-'`` or ``'+'``.
    lines: list[str] = field(default_factory=list)
    #: Whether the old, respectively new, side ends without a final newline.
    old_open: bool = False
    new_open: bool = False

    @property
    def old(self) -> list[str]:
        return [line[1:] for line in self.lines if line[0] in " -"]

    @property
    def new(self) -> list[str]:
        return [line[1:] for line in self.lines if line[0] in " +"]

    @property
    def text(self) -> str:
        return "\n".join([self.header, *self.lines])


@dataclass
class FilePatch:
    """The changes of a patch to one file.  ``old`` is ``None`` for a file the patch
    creates, ``new`` for one it deletes."""

    old: str | None
    new: str | None
    hunks: list[Hunk] = field(default_factory=list)
    #: Whether the paths were written with git's ``a/`` and ``b/`` in front (now removed).
    prefixed: bool = False

    @property
    def path(self) -> str:
        return self.new or self.old or ""


def _header_path(line: str) -> str | None:
    """The path of a ``---`` or ``+++`` line as written; ``None`` for ``/dev/null``."""
    text = line[4:].split("\t", 1)[0].strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    return None if text == "/dev/null" else text


def _is_file_header(lines: list[str], index: int) -> bool:
    return (
        lines[index].startswith("--- ")
        and index + 1 < len(lines)
        and lines[index + 1].startswith("+++ ")
    )


def _read_hunk(lines: list[str], index: int) -> tuple[Hunk, int]:
    """The hunk whose ``@@`` line is ``lines[index]``, and the index after it.

    The counts in the header say where the hunk ends, as long as lines that can
    belong to it follow.  Where the counts are wrong (too small), the lines up to
    the next header are taken as well: what a hunk changes is in its text.
    """
    match = _HUNK.match(lines[index])
    assert match is not None
    old_left = int(match.group(2)) if match.group(2) is not None else 1
    new_left = int(match.group(4)) if match.group(4) is not None else 1
    hunk = Hunk(header=lines[index], old_start=int(match.group(1)), old_count=old_left)
    index += 1
    while index < len(lines):
        line = lines[index]
        counted = old_left > 0 or new_left > 0
        if line.startswith("\\"):
            last = hunk.lines[-1][0] if hunk.lines else " "
            hunk.old_open = hunk.old_open or last in " -"
            hunk.new_open = hunk.new_open or last in " +"
        elif line == "" and counted:
            hunk.lines.append(" ")  # an editor stripped the space of an empty context line
            old_left, new_left = old_left - 1, new_left - 1
        elif line[:1] in (" ", "-", "+") and not _starts_file(lines, index, counted):
            hunk.lines.append(line)
            old_left -= line[0] in " -"
            new_left -= line[0] in " +"
        else:
            break
        index += 1
    return hunk, index


def _starts_file(lines: list[str], index: int, counted: bool) -> bool:
    """Whether the line begins the next file of a patch rather than continuing a hunk.

    While the hunk's counts are not used up, a line ``--- x`` may be a removed line
    that reads ``-- x``; only a complete header (``---``, ``+++``, ``@@``) ends the hunk.
    """
    if not _is_file_header(lines, index):
        return False
    return not counted or (index + 2 < len(lines) and bool(_HUNK.match(lines[index + 2])))


def _strip_prefixes(patch: FilePatch) -> None:
    """Drop the ``a/`` and ``b/`` that git puts before the two paths."""
    old_prefixed = patch.old is None or patch.old.startswith("a/")
    new_prefixed = patch.new is None or patch.new.startswith("b/")
    if old_prefixed and new_prefixed:
        patch.old = patch.old[2:] if patch.old else None
        patch.new = patch.new[2:] if patch.new else None
        patch.prefixed = True


def _check_patch_path(path: str) -> None:
    parts = PurePosixPath(path).parts
    outside = not parts or PurePosixPath(path).is_absolute() or ".." in parts or "\\" in path
    if outside or parts[0] == ".git" or "\0" in path:
        raise PatchError(
            f"{PATCH_NAME} names {path!r}, which is outside the repository or not allowed. "
            "A patch changes files of the repository by their relative paths."
        )


def _is_result(path: str) -> bool:
    """Whether ``path`` is one of ``RESULT_PATHS`` or lies in one."""
    parts = PurePosixPath(path).parts
    return any(parts[: len(wanted)] == wanted for wanted in map(_result_parts, RESULT_PATHS))


def _result_parts(name: str) -> tuple[str, ...]:
    return PurePosixPath(name).parts


def _check_not_a_result(patch: FilePatch) -> None:
    """Refuse a patch that touches where a run's results go.

    Creating or changing such a file would put numbers and figures on record that no
    run produced.  Deleting one has no effect, since they are removed before the
    command starts anyway; it is refused all the same, so that the rule is one a
    person can remember: a patch does not name these paths.
    """
    for path in filter(None, (patch.old, patch.new)):
        if _is_result(path):
            raise PatchError(
                f"{PATCH_NAME} changes {path!r}. {', '.join(RESULT_PATHS[:-1])} and "
                f"{RESULT_PATHS[-1]}/ hold what the experiment's command produces when it "
                "runs, and are removed before it starts: results come from running the "
                "experiment, not from a file put there beforehand. Take that change out "
                "of the patch."
            )


def parse_patch(text: str) -> list[FilePatch]:
    """The file changes in a unified diff.

    Understands what ``git diff`` and ``diff -u`` write for text files.
    Raises :class:`PatchError` for a patch without changes, one that renames
    files or carries binary data (which only git can apply, and which an
    experiment has no need for), one that names a path outside the tree, and one
    that touches a result of the run (``RESULT_PATHS``).
    """
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()  # the end of the text is not an empty line of context
    patches: list[FilePatch] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(("rename from ", "rename to ", "copy from ", "GIT binary patch")):
            raise PatchError(
                f"{PATCH_NAME} has a rename, a copy or binary data ({line.strip()!r}). Write "
                "the change as lines removed from one file and added to another."
            )
        if _is_file_header(lines, index):
            patch = FilePatch(_header_path(line), _header_path(lines[index + 1]))
            _strip_prefixes(patch)
            patches.append(patch)
            index += 2
        elif _HUNK.match(line) and patches:
            hunk, index = _read_hunk(lines, index)
            patches[-1].hunks.append(hunk)
        else:
            index += 1
    patches = [patch for patch in patches if patch.hunks and patch.path]
    if not patches:
        raise PatchError(
            f"{PATCH_NAME} holds no changes: it has to be a unified diff (--- a/file, "
            "+++ b/file, then @@ hunks), as `git diff` writes it."
        )
    for patch in patches:
        for path in filter(None, (patch.old, patch.new)):
            _check_patch_path(path)
        _check_not_a_result(patch)
    return patches


def _find(lines: list[str], block: list[str], start: int) -> int | None:
    """Where ``block`` occurs in ``lines``: at ``start``, else as near to it as it does."""
    if not block:
        return min(max(start, 0), len(lines))
    last = len(lines) - len(block)
    for distance in range(max(len(lines), 1)):
        for position in {start - distance, start + distance}:
            if 0 <= position <= last and lines[position : position + len(block)] == block:
                return position
    return None


def _rejected(patch: FilePatch, hunk: Hunk, why: str) -> PatchError:
    return PatchError(
        f"{PATCH_NAME} does not apply to {patch.path}: {why}. The rejected hunk:\n{hunk.text}",
        hunk=hunk.text,
    )


def _patched(patch: FilePatch, original: bytes | None) -> bytes | None:
    """The file's new content, or ``None`` when the patch deletes it."""
    first = patch.hunks[0]
    if patch.old is None:
        if original:
            raise _rejected(patch, first, "the patch creates this file, but it already exists")
        original = b""
    elif original is None:
        raise _rejected(patch, first, "there is no such file in the repository")
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        raise _rejected(patch, first, "the file is not text") from None
    ending = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(ending)
    closed = lines[-1] == ""  # the file ends with a newline
    if closed:
        lines.pop()
    offset = 0
    for hunk in patch.hunks:
        # A hunk that only adds lines names the line it adds them after.
        stated = hunk.old_start - 1 if hunk.old_count else hunk.old_start
        position = _find(lines, hunk.old, stated + offset)
        if position is None:
            raise _rejected(patch, hunk, "the lines it changes were not found in the file")
        lines[position : position + len(hunk.old)] = hunk.new
        offset = position + len(hunk.new) - len(hunk.old) - stated
        if hunk.new_open and position + len(hunk.new) == len(lines):
            closed = False
        elif hunk.old_open and position + len(hunk.new) == len(lines):
            closed = True
    if patch.new is None:
        return None
    return (ending.join(lines) + (ending if closed and lines else "")).encode("utf-8")


def _inside(root: Path, relative: str) -> Path:
    """``root/relative``, refused when a link on the way leads out of ``root``."""
    target = root / relative
    if not target.parent.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise PatchError(
            f"{PATCH_NAME} would write {relative!r} through a link that leads outside the "
            "repository; that is not allowed."
        )
    return target


def _apply_parsed(root: Path, patches: list[FilePatch]) -> list[str]:
    """Apply ``patches`` under ``root``, all or nothing; returns the paths touched."""
    planned: list[tuple[Path, bytes | None]] = []
    touched: list[str] = []
    for patch in patches:
        source = _inside(root, patch.old) if patch.old else None
        target = _inside(root, patch.new) if patch.new else None
        original = source.read_bytes() if source is not None and source.is_file() else None
        content = _patched(patch, original)
        if source is not None and source != target:
            planned.append((source, None))
        if target is not None:
            planned.append((target, content))
        touched.append(patch.path)
    for path, content in planned:
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return touched


_MIXED_PREFIXES = "the patch writes some paths with a/ and b/ in front and others without"


def _git_apply_options(patches: list[FilePatch]) -> list[str] | None:
    """How ``git apply`` has to be called for ``patches`` to land where the patch says;
    ``None`` when it cannot be told to (the paths are written in two ways)."""
    # The strip level is said outright: guessed wrong, `src/util.py` would become `util.py`
    # and the change could land in another file of that name.
    prefixed = {patch.prefixed for patch in patches}
    if len(prefixed) != 1:
        return None
    options = ["--whitespace=nowarn", "-p1" if prefixed.pop() else "-p0"]
    # A hunk that begins or ends with a changed line is taken by git to sit at the very
    # start or end of the file, and is applied there when its old lines also occur
    # there (always, for a hunk that only adds lines): the wrong place, without a word.
    # With this option git goes by the line numbers and the old lines instead.
    return [*options, "--unidiff-zero"] if any(map(_is_bare, _changes(patches))) else options


def _changes(patches: list[FilePatch]) -> list[Hunk]:
    """The hunks that change a file which exists before and after the patch."""
    return [
        hunk
        for patch in patches
        if patch.old is not None and patch.new is not None
        for hunk in patch.hunks
    ]


def _is_bare(hunk: Hunk) -> bool:
    """Whether the hunk has no line of context before or after what it changes."""
    return not hunk.lines or hunk.lines[0][:1] != " " or hunk.lines[-1][:1] != " "


def _git_apply(root: Path, text: str, patches: list[FilePatch]) -> str | None:
    """Apply the patch with ``git apply``; returns what git said when it refused."""
    options = _git_apply_options(patches)
    if options is None:
        return _MIXED_PREFIXES
    if any(_is_bare(hunk) and (hunk.old_open or hunk.new_open) for hunk in _changes(patches)):
        # Going by line numbers, git takes a last line without a newline for the same
        # line further up that has one, and changes that.  This module's applier does not.
        return "git cannot place a hunk without context at an end of file without a newline"
    with tempfile.TemporaryDirectory(prefix="paper-orchestra-patch-") as scratch:
        file = Path(scratch) / PATCH_NAME
        file.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        done = _git(
            ["apply", *options, str(file)],
            cwd=root,
            # The copy may lie inside some other repository (the project's own); the patch
            # is about the files of the copy, not about that one.
            ceiling=root.resolve().parent,
        )
    return None if done.returncode == 0 else _said(done) or "git apply failed"


def apply_patch(root: str | Path, patch: str, *, use_git: bool | None = None) -> list[str]:
    """Apply the unified diff ``patch`` to the tree under ``root``; returns the paths changed.

    ``git apply`` is used when git is installed (or ``use_git`` says so).  What
    git refuses is tried once more by this module's own applier, which finds a
    hunk by its text when the line numbers or counts in its header are off, as
    they often are in a patch a model wrote.  A hunk without a line of context
    around what it changes has only its line numbers to go by, and is applied
    where they say.  All or nothing: when a hunk is rejected,
    :class:`PatchError` carries it and no file has been changed.
    """
    root = Path(root)
    patches = parse_patch(patch)
    for parsed in patches:
        for path in filter(None, (parsed.old, parsed.new)):
            _inside(root, path)
    if git_available() if use_git is None else use_git:
        refusal = _git_apply(root, patch, patches)
        if refusal is None:
            return [parsed.path for parsed in patches]
        try:
            return _apply_parsed(root, patches)
        except PatchError as exc:
            raise PatchError(f"{exc}\n\ngit apply said: {refusal}", hunk=exc.hunk) from None
    return _apply_parsed(root, patches)


# ---------------------------------------------------------------------------
# The experiment's own files
# ---------------------------------------------------------------------------


def _exists_message(name: str) -> str:
    return (
        f"{name} already exists in the repository. A file of the experiment is added next "
        f"to the repository's files and never replaces one: change the repository's file "
        f"through {PATCH_NAME}, or give the experiment's file another name."
    )


def add_files(root: str | Path, files: Mapping[str, bytes]) -> list[str]:
    """Add the experiment's own files to the copy under ``root``; returns their names.

    A name that exists in the repository is refused (:class:`RepoError`): a
    change to the person's code is a patch they can read, not a silent replacement.
    Nothing is written unless every file can be.
    """
    root = Path(root)
    targets: list[tuple[Path, bytes]] = []
    for name, data in files.items():
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
            raise RepoError(f"{name!r} is not a relative path inside the repository.")
        target = root / name
        if not target.parent.resolve().is_relative_to(root.resolve()):
            raise RepoError(f"{name!r} would be written outside the copy of the repository.")
        if os.path.lexists(target):
            raise RepoError(_exists_message(name))
        targets.append((target, data))
    for target, data in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return list(files)


# ---------------------------------------------------------------------------
# A repository behind a URL: cloned by the job
# ---------------------------------------------------------------------------

_BOOTSTRAP = '''\
"""Clone the repository of an experiment inside its job, then run the command in it.

Written by paper-orchestra into the job's working directory.  It runs where the
job runs, so the clone uses the network of the job and of nothing else.

The address of the repository may carry a password or a token.  It is read from
job.json, taken out of that file at once and handed to git for the one fetch.
No remote is set up with it, and what git wrote down of the fetch is removed, so
that the experiment's command, which runs next in this directory, finds it nowhere.
"""

import json
import os
import shutil
import subprocess
import sys
from urllib.parse import unquote

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = @PREFIX@


def stop(status, message):
    print(PREFIX + message, flush=True)
    sys.exit(status)


def git(*args):
    # A repository that wants a password is a failure to report, not a question to ask.
    # And no helper is asked for one or told one: a helper that the system configures
    # would keep the token of the address where the command could ask for it later.
    return subprocess.run(
        ["git", "-c", "credential.helper=", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
    )


def take_address():
    """The job as job.json describes it, and the address to fetch from, which the file
    no longer holds once this returns."""
    path = os.path.join(HERE, "job.json")
    with open(path, encoding="utf-8") as handle:
        job = json.load(handle)
    url, secret = job["url"], job.pop("secret", None)
    job["url"] = job["remote"]  # what a tool may be told: the address without its secret
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2)
    return job, url, secret


def spellings(url):
    """The address as it is, and as git names it: without what stands before its first
    "@", which is not all of a password that has an "@" in it."""
    scheme, separator, rest = url.partition("://")
    found = [url]
    while separator and "@" in rest.split("/", 1)[0]:
        rest = rest.split("@", 1)[1]
        found.append(scheme + separator + rest)
    # The last one is the address without any user information: nothing to hide in it.
    return found[:-1] or found


def forget(secrets):
    """Remove what git kept of the fetch, and the secret from whatever else it wrote."""
    try:
        os.remove(os.path.join(".git", "FETCH_HEAD"))
    except OSError:
        pass
    wanted = [secret.encode("utf-8") for secret in secrets]
    for folder, folders, names in os.walk(".git") if wanted else ():
        if folder == ".git" and "objects" in folders:
            folders.remove("objects")  # what was fetched, not what was said about it
        for name in names:
            path = os.path.join(folder, name)
            try:
                if os.path.islink(path) or os.path.getsize(path) > 1024 * 1024:
                    continue
                with open(path, "rb") as handle:
                    data = handle.read()
                if any(secret in data for secret in wanted):
                    for secret in wanted:
                        data = data.replace(secret, b"***")
                    with open(path, "wb") as handle:
                        handle.write(data)
            except OSError:
                continue


def clear_results(names):
    """A result is what the command writes: what the repository, the patch or the
    experiment's files brought under those names is removed before it starts.  Not
    through a link, though: what lies behind one is not this directory's to remove,
    and is never collected as a result either."""
    removed = []
    for name in names:
        parts = name.split("/")
        above = (os.path.join(*parts[:depth]) for depth in range(1, len(parts)))
        if any(os.path.islink(folder) for folder in above) or not os.path.lexists(name):
            continue
        if os.path.isdir(name) and not os.path.islink(name):
            shutil.rmtree(name)
        else:
            os.remove(name)
        removed.append(name)
    if removed:
        said = "Removed " + ", ".join(removed) + " that was there before the run (it came "
        said += "with the repository): only what the command itself writes counts as a result."
        print(PREFIX + said, flush=True)


def main():
    job, url, secret = take_address()
    if shutil.which("git") is None:
        stop(@NO_GIT@, "git is not installed where this job runs; the repository cannot be cloned.")
    # git quotes a password as it was sent, which is not always how the URL spells it.
    spelled = {text for text in (secret, unquote(secret or "")) if text}
    secrets = sorted(spelled, key=len, reverse=True)

    def shown(text):
        # The log is kept and read; the password or token of the URL is not for it.
        for spelling in spellings(url):
            text = text.replace(spelling, job["shown"])
        for text_secret in secrets:
            text = text.replace(text_secret, "***")
        return text

    steps = (
        ("init", "-q", "."),
        # The address goes to git for this one command, and into no file.
        ("fetch", "-q", "--depth", "1", url, job["ref"] or "HEAD"),
        ("checkout", "-q", "FETCH_HEAD"),
    )
    for step in steps:
        done = git(*step)
        if done.returncode != 0:
            forget(secrets)
            said = "(git " + step[0] + "): " + shown(done.stdout.strip())
            stop(@CLONE_FAILED@, "The repository could not be cloned " + said)
    head = git("rev-parse", "HEAD").stdout.strip()
    # For tools that note where the code came from: the address without its secret.
    git("remote", "add", "origin", job["remote"])
    forget(secrets)
    print(PREFIX + "cloned " + job["shown"] + " at " + head, flush=True)
    patch = os.path.join(HERE, @PATCH@)
    if os.path.exists(patch):
        done = git("apply", *job["apply"], patch)
        if done.returncode != 0:
            said = done.stdout.strip()
            stop(@PATCH_FAILED@, @PATCH@ + " does not apply to the repository: " + said)
    files = os.path.join(HERE, "files")
    for folder, _, names in os.walk(files):
        for name in sorted(names):
            source = os.path.join(folder, name)
            relative = os.path.relpath(source, files)
            if os.path.lexists(relative):
                stop(@FILE_EXISTS@, @EXISTS@.replace("{name}", relative))
            os.makedirs(os.path.dirname(relative) or ".", exist_ok=True)
            shutil.copyfile(source, relative)
    clear_results(job["outputs"])
    command = list(job["command"])
    if command[0] in ("python", "python3"):
        command[0] = sys.executable
    sys.stdout.flush()
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        stop(127, "The command could not be started: " + str(exc))


main()
'''

_CLONE_FAILURES: dict[int, tuple[str, str]] = {
    EXIT_NO_GIT: (
        "setup",
        "The repository could not be cloned: git is not installed where the job runs.",
    ),
    EXIT_CLONE_FAILED: ("setup", "The repository could not be cloned."),
    EXIT_PATCH_FAILED: ("patch", f"{PATCH_NAME} does not apply to the repository."),
    EXIT_FILE_EXISTS: (
        "setup",
        "A file of the experiment has the name of a file in the repository.",
    ),
}


def _bootstrap() -> str:
    values = {
        "@PREFIX@": repr(CLONE_PREFIX),
        "@NO_GIT@": str(EXIT_NO_GIT),
        "@CLONE_FAILED@": str(EXIT_CLONE_FAILED),
        "@PATCH_FAILED@": str(EXIT_PATCH_FAILED),
        "@FILE_EXISTS@": str(EXIT_FILE_EXISTS),
        "@PATCH@": repr(PATCH_NAME),
        "@EXISTS@": repr(_exists_message("{name}")),
    }
    script = _BOOTSTRAP
    for marker, value in values.items():
        script = script.replace(marker, value)
    return script


def clone_source(
    seed: str | Path,
    *,
    url: str,
    ref: str | None = None,
    patch: str | None = None,
    files: Mapping[str, bytes] | None = None,
    command: Iterable[str],
    outputs: Sequence[str] = RESULT_PATHS,
) -> list[str]:
    """Write the source directory of a job that clones ``url`` itself, and return the
    command of that job.

    Nothing is fetched here.  The job clones the repository at ``ref`` (depth 1)
    into its working directory, applies ``patch``, adds ``files`` (refusing a name
    the repository already has), removes the paths in ``outputs`` that all this
    brought along (a result is what the command writes) and then runs ``command``
    there.  When one of its own steps fails it prints why on a line that starts
    with ``CLONE_PREFIX`` and exits with a status :func:`clone_failure` knows; it
    says on such a line, too, which outputs it removed.  The job needs git and
    network access; ``url`` and ``ref`` must have passed :func:`url_problem` and
    :func:`ref_problem`.

    A password or token in ``url`` reaches the job in ``job.json`` in the seed, which
    is the only place it is written to.  The job takes it out of that file before it
    does anything else, gives it to git for the fetch alone and leaves no trace of it
    in the working directory for ``command`` to find; what the job prints names the
    repository as :func:`display_url` does.  The seed directory itself still holds
    the URL until its owner removes it.

    Raises :class:`PatchError` for a ``patch`` that is not one, or that git cannot
    be told how to apply: inside the job there is only ``git apply``.
    """
    options = _git_apply_options(parse_patch(patch)) if patch is not None else []
    if options is None:
        raise PatchError(
            f"{PATCH_NAME} cannot be applied in a job that clones the repository: "
            f"{_MIXED_PREFIXES}. Write every path the same way, as `git diff` does."
        )
    for name in outputs:
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
            raise RepoError(f"{name!r} is not a relative path inside the repository.")
    home = Path(seed) / SEED_DIR
    home.mkdir(parents=True, exist_ok=True)
    (home / "bootstrap.py").write_text(_bootstrap(), encoding="utf-8")
    job = {
        "url": url,
        "secret": userinfo(url),
        "shown": display_url(url),
        "remote": _without_secret(url),
        "ref": ref or None,
        "apply": options,
        "command": list(command),
        "outputs": [PurePosixPath(name).as_posix() for name in outputs],
    }
    description = home / "job.json"
    description.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    # It may hold a token until the job has taken it out: for nobody else on this machine.
    description.chmod(0o600)
    if patch is not None:
        (home / PATCH_NAME).write_text(patch, encoding="utf-8")
    for name, data in (files or {}).items():
        target = home / "files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return ["python", f"{SEED_DIR}/bootstrap.py"]


def clone_failure(returncode: int | None) -> tuple[str, str] | None:
    """``(kind, message)`` when a clone job ended in one of its own steps, else ``None``.

    ``kind`` is ``"patch"`` for a patch that did not apply and ``"setup"`` for
    everything else that kept the command from starting.
    """
    return _CLONE_FAILURES.get(returncode) if returncode is not None else None


def clone_lines(log: Iterable[str]) -> list[str]:
    """What the clone job said itself, from the lines of its log."""
    said = []
    for line in log:
        with contextlib.suppress(ValueError):
            said.append(line[line.index(CLONE_PREFIX) + len(CLONE_PREFIX) :].strip())
    return said
