"""The project store: every reader and writer of a research project goes through here.

A project is a directory of plain files whose root is also the root of a
:class:`~paper_orchestra.memory.wiki.Wiki`.  :class:`Project` adds what the wiki
layer does not know about: who may change what, review, edit proposals, the
person's requests, the history of every page and the event stream.

Three rules hold for every method that changes something:

* it runs under the project's lock, so a web server and an engine worker can
  use the same project at once (ids never collide, no event is lost);
* the version of a page it replaces or removes is first copied to ``history/``;
* it appends one event that says, in a sentence, what happened.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

from paper_orchestra import doctor
from paper_orchestra.memory.lock import WikiLock
from paper_orchestra.memory.wiki import (
    INDEX_FILE,
    NOTES_DIR,
    TRANSCRIPTS_DIR,
    WIKI_DIR,
    Page,
    SearchHit,
    Wiki,
    WikiError,
    slugify,
    split_front_matter,
)
from paper_orchestra.research import repo as repositories
from paper_orchestra.research.model import (
    APPROVAL_FIELDS,
    EDITED_BY_PERSON,
    EVENT_TYPES,
    FOR_ALL,
    HUMAN,
    ID_PREFIXES,
    IDEA_PATH,
    INITIAL_STATE,
    KINDS,
    MANAGED_FIELDS,
    MENTIONS,
    RECORD_KINDS,
    RELATIONS,
    REVIEWS,
    SCHEMA,
    SECTIONS,
    STATES,
    SYSTEM,
    Conflict,
    EditProposal,
    NotAllowed,
    NotFound,
    PageVersion,
    ProjectError,
    ProjectSettings,
    Request,
    ResearchPage,
    check_transition,
    clean_relations,
    clean_target,
    compact_stamp,
    creatable_kinds,
    format_id,
    format_stamp,
    id_in_name,
    initial_review,
    is_agent,
    moved_on,
    normalize_author,
    ordered_meta,
    page_from_parts,
    page_path,
    page_problems,
    parse_id,
    parse_stamp,
    quotable,
    request_role,
    role_author,
    role_of,
    section_path,
    stamp_after,
    with_article,
)
from paper_orchestra.utils.common import write_json, write_text

__all__ = [
    "Conflict",
    "NotAllowed",
    "NotFound",
    "Project",
    "ProjectError",
    "ResearchPage",
]

PROJECT_FILE = "project.json"
#: Key of project.json that says the title is only the brief's first line.
TITLE_FROM_BRIEF = "title_from_brief"
TITLE_LIMIT = 120
STATE_FILE = "state.json"
EVENTS_FILE = "events.jsonl"
INPUTS_DIR = "inputs"
BRIEF_FILE = "brief.md"
MANIFEST_FILE = "manifest.json"
PROPOSALS_DIR = "proposals"
HISTORY_DIR = "history"
REQUESTS_DIR = "requests"
EXPERIMENTS_DIR = "experiments"
PAPER_DIR = "paper"
LOCK_FILE = ".project.lock"

#: Sections only pipeline code writes.  The wiki layer refuses a model session's write
#: to them whatever backend the session came through.
RECORD_SECTIONS: tuple[str, ...] = tuple(f"{section_path(kind)}/" for kind in RECORD_KINDS)
#: Size of the index injected into agent sessions; it is read in full by every session.
INDEX_MAX_CHARS = 20_000
#: How long a writer waits for another one to finish before giving up.
LOCK_WAIT_SECONDS = 30.0

_FIELD_NAME = re.compile(r"^[A-Za-z_][\w\-]*$")
_RECORD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
_HISTORY_NAME = re.compile(r"^(\d{8}T\d{6}Z)-(.+)\.md$")
_DECISIONS = {"accept": "accept", "accepted": "accept", "reject": "reject", "rejected": "reject"}
#: States an experiment reaches only through its approval and its run.
_RUN_STATES = ("approved", "queued", "running")
#: Of the record of an approval, what nobody sets by editing a page.  ``backend`` is the
#: one field of it the person may also write by hand (an approval decides it anew).
_APPROVAL_RECORD = APPROVAL_FIELDS - {"backend"}
#: What ``record_approval`` writes: the options of the approved run and its record.
_APPROVAL_WRITES = APPROVAL_FIELDS | {"gpu", "timeout_seconds", "network"}
#: Fields of an experiment that say which repository it copies or clones and runs.  The
#: person names it, on the page or in the project's settings; no edit proposal does.
_REPOSITORY_FIELDS = frozenset({"repo", "repo_ref"})
#: A file a writer of this package leaves when it dies between writing and renaming:
#: ``.<name>.<pid>.<8 hex digits>.tmp`` (``utils.common.write_text``, the wiki layer).
_TEMPORARY_NAME = re.compile(r"^\..+\.(\d{1,10})\.[0-9a-f]{8}\.tmp$")
#: How old such a file has to be before it is taken for litter, in seconds: one being
#: written lives for milliseconds.
TEMPORARY_FILE_AGE = 60.0
#: Where a project is built before it is moved into place: a hidden directory next to
#: it, named with the process that builds it.
_BUILD_PREFIX = ".paper-orchestra-new-"
#: The top-level names a new project consists of, besides project.json and the lock.
_PROJECT_NAMES = frozenset(
    {
        WIKI_DIR,
        NOTES_DIR,
        TRANSCRIPTS_DIR,
        INPUTS_DIR,
        PROPOSALS_DIR,
        HISTORY_DIR,
        REQUESTS_DIR,
        EXPERIMENTS_DIR,
        EVENTS_FILE,
        INDEX_FILE,
    }
)
#: Key of project.json: the inode of the directory the project was created as.  A copy
#: of the directory carries the key and not the inode, which tells it from the original.
HOME_KEY = "home"
_HOW_TO_APPROVE = (
    "A run is approved with `paper-orchestra research approve`, or with Approve in the app: "
    "that records what may run, and nothing else moves an experiment to approved, queued or "
    "running."
)
_EXTRA_MEDIA_TYPES = {
    ".md": "text/markdown",
    ".tex": "text/x-tex",
    ".bib": "text/x-bibtex",
    ".py": "text/x-python",
    ".jsonl": "application/jsonl",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".ipynb": "application/x-ipynb+json",
}


# ---------------------------------------------------------------------------
# Locking and time
# ---------------------------------------------------------------------------


class _FileLock(WikiLock):
    """The cross-process lock of ``memory.lock``, on a file of the project's own.

    The wiki's consolidation holds ``WikiLock`` on the same directory for minutes;
    the project's writes take milliseconds and must not wait for it.
    """

    def __init__(self, path: Path, *, wait_seconds: float, poll: float) -> None:
        super().__init__(path.parent, wait_seconds=wait_seconds, poll=poll)
        self.path = path


class _ProjectLock:
    """Mutual exclusion for one project directory: reentrant, across threads and processes.

    Methods that change the project call each other (a review updates a page and
    emits an event), so the lock is taken once by the outermost and counted inside.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._threads = threading.RLock()
        self._depth = 0
        self._file: _FileLock | None = None

    def __enter__(self) -> _ProjectLock:
        self._threads.acquire()
        try:
            if self._depth == 0:
                self._file = self._lock_file()
            self._depth += 1
        except BaseException:
            # Also on Ctrl-C while waiting: a thread lock left held would stop every other
            # thread of this process at its next write, for good.
            self._threads.release()
            raise
        return self

    def _lock_file(self) -> _FileLock:
        lock = _FileLock(self._root / LOCK_FILE, wait_seconds=LOCK_WAIT_SECONDS, poll=0.005)
        try:
            lock.__enter__()
        except BaseException:
            lock.__exit__(None, None, None)
            raise
        if not lock.acquired:
            why = lock.error or "another process is writing to it"
            raise ProjectError(
                f"The project at {self._root} could not be locked within "
                f"{LOCK_WAIT_SECONDS:.0f} seconds ({why}). Nothing was changed; try again."
            )
        return lock

    def __exit__(self, *exc: object) -> None:
        self._depth -= 1
        if self._depth == 0 and self._file is not None:
            self._file.__exit__(None, None, None)
            self._file = None
        self._threads.release()


_LOCKS: dict[Path, _ProjectLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> _ProjectLock:
    """One lock object per project directory in this process, however often it is opened."""
    key = root.resolve()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, _ProjectLock(key))


class _Clock:
    """The project's clock, which one thread can hold still for the length of a write.

    The wiki stamps ``updated`` from its clock.  The project decides that stamp
    itself (it must grow with every version of a page), so it hands the wiki this
    clock and pins it while it writes.
    """

    def __init__(self, base: Callable[[], datetime]) -> None:
        self._base = base
        self._local = threading.local()

    def __call__(self) -> datetime:
        pinned = getattr(self._local, "moment", None)
        return pinned if pinned is not None else self._base()

    def now(self) -> str:
        return format_stamp(self._base())

    @contextlib.contextmanager
    def at(self, stamp: str) -> Iterator[None]:
        self._local.moment = parse_stamp(stamp)
        try:
            yield
        finally:
            self._local.moment = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    """The JSON in ``path``, or ``None`` when it is missing or not JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def _read_text(path: Path) -> str | None:
    """The text in ``path``, or ``None`` when it cannot be read."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


#: Text a page shows as code: fenced blocks (``` or ~~~) and inline spans.
_CODE = re.compile(
    r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ ]{0,3}(?P=fence)[`~]*[ \t]*$|\Z)"
    r"|(?P<ticks>`+)[^\n]*?(?P=ticks)",
    re.DOTALL | re.MULTILINE,
)


def _body_links(page: ResearchPage) -> list[str]:
    """Wiki link targets in a page's prose, in order.

    What a page shows as code is quoted material (a job's log, a diff, a matrix
    that numpy prints as ``[[0.1 0.2]]``), not the page referring to another one.
    """
    return Page(page.path, {}, _CODE.sub(" ", page.body)).links()


def _split(text: str) -> tuple[dict[str, Any], str]:
    """Front matter and body of a page file as it was read.

    Editors on some systems put a byte order mark before the first character; it is
    not content, and with it in place the front matter would be taken for body text.
    """
    return split_front_matter(text.removeprefix("\ufeff"))


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str, limit: int) -> str:
    text = _one_line(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _sentence(text: str) -> str:
    text = _one_line(text)
    return text if text.endswith((".", "!", "?")) else text + "."


def _with_newline(body: str) -> str:
    return body if body.endswith("\n") else body + "\n"


def _author_slug(author: str) -> str:
    """``agent:theorist`` -> ``agent-theorist``: no colon in a file name."""
    return re.sub(r"[^A-Za-z0-9_\-]", "-", author) or "unknown"


def _author_from_slug(slug: str) -> str:
    return "agent:" + slug[len("agent-") :] if slug.startswith("agent-") else slug


def _who(author: str) -> str:
    if author == HUMAN:
        return "You"
    return f"The {role_of(author)}" if is_agent(author) else "The pipeline"


def _whose(author: str) -> str:
    if author == HUMAN:
        return "your"
    return f"the {role_of(author)}'s" if is_agent(author) else "the pipeline's"


def _what(page: ResearchPage) -> str:
    """``hypothesis H2 "Warmup reduces variance"``.

    The title is whoever wrote the page's: it stands as one quotation, cut, with
    nothing in it that could read as the quotation's end.
    """
    title = quotable(page.title, 80).rstrip(".")
    return f'{page.kind} {page.id} "{title}"' if page.id else f'{page.kind} "{title}"'


def _said(text: Any, limit: int = 200) -> str:
    """An agent's own words (the summary of an edit) as a quotation in an event's sentence."""
    return f'"{quotable(text, limit).rstrip(".")}"'


def _name(page: ResearchPage) -> str:
    """What a message calls the page: its id where it has one, else its path."""
    return page.id or page.path


def _approval_sentence(by: str, page: ResearchPage, fields: Mapping[str, Any]) -> str:
    """What a change to the record of an approval means, for the activity feed."""
    what = _what(page)
    if fields.get("approved_digest"):
        limit = fields.get("timeout_seconds")
        how = [
            f"on {quotable(fields['backend'], 40)}" if fields.get("backend") else "",
            f"with a {quotable(fields['gpu'], 40)} GPU" if fields.get("gpu") else "",
            f"for up to {limit:g} s" if isinstance(limit, int | float) else "",
            ("with network access" if fields["network"] else "without network access")
            if "network" in fields
            else "",
        ]
        said = ", ".join(part for part in how if part)
        return f"{_who(by)} set how {what} may run{': ' + said if said else ''}."
    if fields.get("repaired_digest"):
        return f"The run of {what} goes on with the code as the runner repaired it."
    return f"{_who(by)} changed the record of the approval of {what}."


def _media_type(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0] or _EXTRA_MEDIA_TYPES.get(path.suffix.lower())
    if guessed:
        return guessed
    try:
        with path.open("rb") as handle:
            head = handle.read(8192)
        if b"\x00" in head:
            return "application/octet-stream"
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut by the 8 KB window is still text.
        return "text/plain" if exc.start >= len(head) - 4 else "application/octet-stream"
    except OSError:
        return "application/octet-stream"
    return "text/plain"


def _free_name(name: str, taken: set[str]) -> str:
    """``name``, or ``name-2`` and so on, so that no two inputs share a place."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, number = name, 1
    while candidate.lower() in taken:
        number += 1
        candidate = f"{stem}-{number}{suffix}"
    taken.add(candidate.lower())
    return candidate


def _gather_inputs(files: Iterable[str | Path]) -> tuple[list[tuple[Path, str]], list[dict]]:
    """``(file, name under inputs/)`` for everything to copy, and what is left out and why.

    Directories are walked.  Hidden files and directories are left out (they hold
    tool state and credentials more often than research material), and so is every
    symbolic link: what it points at was not what the person attached.
    """
    chosen: list[tuple[Path, str]] = []
    skipped: list[dict] = []
    taken = {BRIEF_FILE, MANIFEST_FILE}
    for given in files:
        source = Path(given).expanduser()
        if source.is_symlink():
            skipped.append({"path": str(given), "reason": "symbolic link"})
        elif not source.exists():
            raise ProjectError(f"The attached path {given} does not exist.")
        elif source.name.startswith("."):
            skipped.append({"path": str(given), "reason": "hidden"})
        elif source.is_file():
            chosen.append((source, _free_name(source.name, taken)))
        elif source.is_dir():
            base = _free_name(source.name, taken)

            def unreadable(exc: OSError, source: Path = source, base: str = base) -> None:
                # What cannot be listed is said to be left out, not dropped in silence.
                where = Path(exc.filename or source)
                with contextlib.suppress(ValueError):
                    where = Path(base) / where.relative_to(source)
                skipped.append({"path": where.as_posix(), "reason": "cannot be read"})

            for folder, dirs, names in os.walk(source, onerror=unreadable):
                here = Path(folder)
                for name in [d for d in dirs if (here / d).is_symlink()]:
                    relative = (here / name).relative_to(source).as_posix()
                    skipped.append({"path": f"{base}/{relative}", "reason": "symbolic link"})
                dirs[:] = sorted(
                    d for d in dirs if not d.startswith(".") and not (here / d).is_symlink()
                )
                for name in sorted(n for n in names if not n.startswith(".")):
                    file = here / name
                    relative = file.relative_to(source).as_posix()
                    if file.is_symlink():
                        skipped.append({"path": f"{base}/{relative}", "reason": "symbolic link"})
                    elif file.is_file():
                        chosen.append((file, f"{base}/{relative}"))
    return chosen, skipped


def _copy_with_digest(source: Path, target: Path) -> tuple[int, str]:
    """Copy ``source`` to ``target`` and return its size and SHA-256."""
    digest = hashlib.sha256()
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    try:
        with source.open("rb") as reader, partial.open("wb") as writer:
            while chunk := reader.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
                writer.write(chunk)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return size, digest.hexdigest()


def _entries(root: Path) -> list[str]:
    """What is in ``root`` besides the project's own lock file."""
    return [name for name in os.listdir(root) if name != LOCK_FILE]


def _builder(name: str) -> int | None:
    """The process that builds a project in the hidden directory ``name``, or ``None``
    when ``name`` is not the name of such a directory."""
    pid = name[len(_BUILD_PREFIX) :].split("-", 1)[0] if name.startswith(_BUILD_PREFIX) else ""
    return int(pid) if pid.isdigit() else None


def _sweep_unfinished(parent: Path) -> None:
    """Remove what a creation that died left next to where its project was to be: the
    hidden directory it was building in, once the process that made it is gone."""
    try:
        entries = [entry for entry in parent.iterdir() if _builder(entry.name) is not None]
    except OSError:
        return
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            continue
        if _process_is_gone(_builder(entry.name) or 0) and _older_than(entry, TEMPORARY_FILE_AGE):
            shutil.rmtree(entry, ignore_errors=True)


def _unfinished_creation(root: Path, names: Sequence[str]) -> bool:
    """Whether what ``root`` holds is only what a creation that died there left behind:
    the hidden directory it built in and, of the project's own top-level names, the ones
    it had already moved into place.  ``ProjectError`` when the process that builds
    there is still at work."""
    builds = [name for name in names if _builder(name) is not None]
    for name in builds:
        pid = _builder(name) or 0
        if not _process_is_gone(pid):
            raise ProjectError(
                f"A project is being created in {root} right now (by process {pid}). Wait "
                "for it, or choose another directory."
            )
    return bool(builds) and all(name in _PROJECT_NAMES for name in names if name not in builds)


def _clear_unfinished(root: Path) -> None:
    """Remove from ``root`` what a creation that died there left (see
    :func:`_unfinished_creation`), so that the directory is empty again."""
    names = _entries(root)
    if not names or (root / PROJECT_FILE).exists() or not _unfinished_creation(root, names):
        return
    for name in names:
        entry = root / name
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                entry.unlink()


def _already_there(root: Path) -> ProjectError:
    if (root / PROJECT_FILE).exists():
        return ProjectError(
            f"{root} already holds a research project. Open it, or choose another directory."
        )
    return ProjectError(
        f"{root} is not an empty directory. A project needs a directory of its own."
    )


def _move_into_place(built: Path, root: Path) -> None:
    """Make the finished project in ``built`` the new directory ``root``, in one step: a
    rename.  If something came to be there meanwhile (a project another process
    finished), it is left as it is and ``ProjectError`` says so."""
    try:
        if os.path.lexists(root):
            raise FileExistsError(errno.EEXIST, "something is there", str(root))
        os.rename(built, root)
    except OSError as exc:
        if os.path.lexists(root):
            raise _already_there(root) from None
        raise ProjectError(
            f"The project could not be put in place at {root} ({exc}). Nothing was created; "
            "choose a place you can write to."
        ) from exc


def _move_contents(built: Path, root: Path, own: str) -> None:
    """Move the finished project in ``built`` into the existing, empty directory ``root``.

    The directory itself stays the one it was (a shell may be standing in it), so its
    entries are moved, ``project.json`` last: until that one is there, what ``root``
    holds is an unfinished creation, which the next attempt recognises by the build
    directory ``own`` and clears.  Done under the lock of ``root``, so that two
    creations never move into each other.
    """
    with _lock_for(root):
        others = [name for name in _entries(root) if name != own]
        if others:
            raise _already_there(root)
        names = sorted(os.listdir(built), key=lambda name: name == PROJECT_FILE)
        moved: list[Path] = []
        try:
            for name in names:
                if name == LOCK_FILE:
                    continue  # ``root`` has its own, and it is held
                os.rename(built / name, root / name)
                moved.append(root / name)
        except OSError as exc:
            for entry in moved:  # the directory is left as empty as it was found
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    with contextlib.suppress(OSError):
                        entry.unlink()
            raise ProjectError(
                f"The project could not be put in place at {root} ({exc}). Nothing was "
                "created; choose a place you can write to."
            ) from exc


def _texts(value: Any) -> Iterator[str]:
    """Every text in a value that is about to be written: itself, or what a list or an
    object holds, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _texts(key)
            yield from _texts(item)
    elif isinstance(value, list | tuple | set):
        for item in value:
            yield from _texts(item)


def _check_storable(what: str, *values: Any) -> None:
    """Raise ``ProjectError`` when a text among ``values`` cannot be written to a file:
    it holds half of a surrogate pair, which is no character.  Checked before anything
    is written, so that a refusal leaves nothing behind."""
    for value in values:
        for text in _texts(value):
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ProjectError(
                    f"{what} cannot be stored: the text holds half of a surrogate pair at "
                    f"position {exc.start}, which is not a character. Remove it and try again."
                ) from None


def _open_for_append(path: Path) -> Any:
    """``path`` opened for reading and appending, created if needed: the file itself,
    never what a symbolic link there points at, and only a regular file.

    A project directory can come from anywhere.  An event stream that is a link would
    have every event of the project written into a file of the person's elsewhere.
    """
    refused = ProjectError(
        f"{path} is a symbolic link or not a regular file, so no event is written to it. "
        "Remove it (or put the project's own events.jsonl back); the project then starts "
        "a new one."
    )
    if path.is_symlink():  # where the system has no O_NOFOLLOW this is the whole check
        raise refused
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK, errno.EISDIR) or path.is_dir():
            raise refused from None
        raise
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise refused
    return os.fdopen(descriptor, "ab+")


def _process_is_gone(pid: int) -> bool:
    """Whether no process of this machine has the id ``pid``.  Where that cannot be asked
    without a side effect (on Windows signal 0 is Ctrl-C), the answer is no: what a
    process may still be writing is left alone."""
    if pid <= 0 or os.name != "posix":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # it exists and is someone else's
    return False


def _older_than(path: Path, seconds: float) -> bool:
    try:
        return time.time() - path.lstat().st_mtime > seconds
    except OSError:
        return False


def _own_copy(value: Any) -> Any:
    """A front-matter value the caller may change without changing anyone else's."""
    return copy.deepcopy(value) if isinstance(value, (list, dict)) else value


def _as_filter(value: str | Iterable[str] | None) -> frozenset[str] | None:
    """The values a listing is limited to; ``None`` or an empty string for no limit."""
    if value is None or value == "":
        return None
    return frozenset([value]) if isinstance(value, str) else frozenset(value)


def _page_order(page: ResearchPage) -> tuple[int, int, str]:
    """Kinds in the order research proceeds, numbered pages by number (H2 before H10)."""
    parsed = parse_id(page.id) if page.kind in ID_PREFIXES else None
    return (KINDS.index(page.kind), parsed[1] if parsed else 0, page.path)


class _Resolver:
    """Finds the page a reference means, over one reading of the wiki."""

    def __init__(self, wiki: Wiki, pages: Sequence[ResearchPage]) -> None:
        self._wiki = wiki
        self.pages = {page.path: page for page in pages}
        self._by_id: dict[str, str] = {}
        self._by_stem: dict[str, list[str]] = {}
        # Numbered pages first: ``H2`` is the hypothesis even if a source has that key.
        for page in sorted(pages, key=lambda page: page.kind not in ID_PREFIXES):
            if page.id:
                self._by_id.setdefault(page.id.lower(), page.path)
            self._by_stem.setdefault(PurePosixPath(page.path).stem.lower(), []).append(page.path)

    def resolve(self, ref: Any) -> str | None:
        """Path of the page ``ref`` names: a path, an id, a link target or a file name."""
        if not isinstance(ref, str):
            return None
        target = clean_target(ref)
        if not target:
            return None
        path = self._wiki.resolve_link(target)
        if path in self.pages:
            return path
        section, _, name = target.lower().rpartition("/")
        stems = self._by_stem.get(name, [])
        # ``H2-some-other-slug``: the id is what identifies a numbered page.
        leading = re.match(r"^([qhef]\d+)-", name)
        guesses = (
            self._by_id.get(name),
            stems[0] if len(stems) == 1 else None,
            self._by_id.get(leading.group(1)) if leading else None,
        )
        for found in guesses:
            # A reference that names a section means a page there or nothing: the stale
            # path ``topics/e1-setup`` must not turn into experiment E1.
            if found and (not section or self.pages[found].link.lower().startswith(section + "/")):
                return found
        return None

    def canonical(self, target: str) -> str:
        """How a relation to that page is written: its id, else its link target."""
        path = self.resolve(target)
        if path is None:
            return target
        page = self.pages[path]
        return page.id or page.link


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class Project:
    """A research project on disk.  Thread-safe; safe to open from several processes.

    All paths taken and returned are project-relative posix strings
    (``wiki/hypotheses/H2-warmup.md``).  A ``ref`` is such a path, an id (``H2``,
    in any case), a wiki link target (``hypotheses/H2-warmup``) or the path as an
    agent sees it (``/memories/wiki/...``).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Use :meth:`create` or :meth:`open`."""
        self.root = Path(root)
        self._clock = _Clock(clock or (lambda: datetime.now(UTC)))
        self._lock = _lock_for(self.root)
        self.wiki = Wiki(
            self.root,
            name="research",
            description="The research wiki of this project.",
            read_only=RECORD_SECTIONS,
            clock=self._clock,
            # Nothing is touched by opening a project; ``create`` lays out the directories
            # under the project's lock.  MEMORY.md is written here too, from what agents may
            # see: the wiki's own catalog would list other sessions' unreviewed proposals.
            create=False,
            seed_index=False,
        )
        self.id = ""
        self.title = ""
        self.created = ""
        self.settings = ProjectSettings()
        self._extra: dict[str, Any] = {}
        #: The title as project.json holds it, to tell a renaming from a settings change.
        self._title_on_file = ""
        self._file_problems: list[str] = []
        #: Size of events.jsonl when this process last wrote it, and the last seq in it.
        self._tail: tuple[int, int] = (0, 0)
        self._state_cache: tuple[tuple[int, int], dict[str, Any]] | None = None
        #: path -> (inode, mtime, size) of the file when it was parsed, and the page.
        self._page_cache: dict[str, tuple[tuple[int, int, int], ResearchPage | None]] = {}

    def __repr__(self) -> str:
        return f"Project({str(self.root)!r}, id={self.id!r}, title={self.title!r})"

    # -- creating and opening --------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        title: str,
        brief: str = "",
        files: Iterable[str | Path] = (),
        settings: ProjectSettings | Mapping[str, Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> Project:
        """Start a project in ``root`` (a new or empty directory) from the person's idea.

        Copies ``files`` (paths; directories are walked, hidden files and symbolic
        links are left out) to ``inputs/``, writes ``inputs/brief.md`` and
        ``inputs/manifest.json`` (name, size, media type and SHA-256 of each file),
        and creates ``wiki/idea.md`` from the brief as the person's own page.
        ``settings`` may be partial; what it leaves out gets the defaults.  An
        empty ``title`` is taken from the first line of the brief.

        Creating is all or nothing: ``root`` holds the whole project when this
        returns, and is as it was (absent, or the empty directory it was) when it
        raises.  When the process dies in the middle, a new ``root`` is still
        absent, and an existing one holds at most what the next attempt recognises
        as an unfinished creation and clears; a hidden build directory next to (or
        in) it is removed by the next creation there.
        """
        root = Path(root)
        cls._check_free(root)
        typed = cls._typed_settings(settings)
        brief = str(brief or "")
        given = _one_line(title)
        title = given or _clip(brief.strip().split("\n", 1)[0], 80) or "Untitled project"
        chosen, skipped = _gather_inputs(files)
        # All or nothing: the project is built in a hidden directory and put in place
        # when it is whole.  A new directory appears in one step (a rename).  An empty
        # directory that exists stays the directory it is and gets the entries moved in,
        # project.json last; what a creation that died there left is recognised by the
        # next one and cleared.  Either way nothing is ever at ``root`` that a command
        # refuses as "not empty" or lists as a project without its idea page.
        existed = root.is_dir()
        home = root if existed else root.parent
        try:
            home.mkdir(parents=True, exist_ok=True)
            if existed:
                _clear_unfinished(root)
            _sweep_unfinished(home)
            workshop = Path(tempfile.mkdtemp(prefix=f"{_BUILD_PREFIX}{os.getpid()}-", dir=home))
        except OSError as exc:
            raise ProjectError(
                f"The directory {root} could not be created ({exc}). Choose a place you can "
                "write to."
            ) from exc
        built = workshop / "project"
        try:
            built.mkdir()
            project = cls(built, clock=clock)
            project.id = f"p-{secrets.token_hex(6)}"
            project.title = title
            project.settings = typed
            if not given:
                project._extra[TITLE_FROM_BRIEF] = True
            # The directory the project will be: a rename keeps the inode.
            project._extra[HOME_KEY] = {"inode": (root if existed else built).stat().st_ino}
            project._title_on_file = title
            project.created = project._clock.now()
            project._start(brief, chosen, skipped)
            if existed:
                _move_contents(built, root, workshop.name)
            else:
                _move_into_place(built, root)
        finally:
            # After the move it is empty; after a failure it holds what was built so far.
            shutil.rmtree(workshop, ignore_errors=True)
            with _LOCKS_GUARD:
                _LOCKS.pop(built.resolve(), None)
        return cls.open(root, clock=clock)

    @staticmethod
    def _check_free(root: Path) -> None:
        """Raise unless ``root`` is free for a new project: absent, an empty directory,
        or a directory that holds nothing but what a creation that died there left."""
        if (root / PROJECT_FILE).exists() or (root.exists() and not root.is_dir()):
            raise _already_there(root)
        left = _entries(root) if root.is_dir() else []
        if left and not _unfinished_creation(root, left):
            raise _already_there(root)

    @staticmethod
    def _typed_settings(settings: ProjectSettings | Mapping[str, Any] | None) -> ProjectSettings:
        typed = ProjectSettings()
        if isinstance(settings, ProjectSettings):
            settings.validate()
            typed = ProjectSettings.from_dict(settings.to_dict(), strict=True)
        elif settings is not None:
            typed.update(settings)
        return typed

    def _start(self, brief: str, chosen: list[tuple[Path, str]], skipped: list[dict]) -> None:
        for sub in (
            WIKI_DIR,
            NOTES_DIR,
            TRANSCRIPTS_DIR,
            INPUTS_DIR,
            PROPOSALS_DIR,
            HISTORY_DIR,
            REQUESTS_DIR,
            EXPERIMENTS_DIR,
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        entries = []
        for source, name in chosen:
            target = self.inputs_dir / name
            try:
                size, digest = _copy_with_digest(source, target)
            except OSError as exc:
                raise ProjectError(
                    f"The attached file {source} could not be copied ({exc}). No project was "
                    "created; check that the file can be read and that the disk has room."
                ) from exc
            entries.append(
                {"name": name, "size": size, "media_type": _media_type(target), "sha256": digest}
            )
        write_text(self.inputs_dir / BRIEF_FILE, brief)
        write_json(
            self.inputs_dir / MANIFEST_FILE,
            {"brief": BRIEF_FILE, "files": entries, "skipped": skipped},
        )
        self._write_project_file()
        self.wiki.name = self.title
        meta = {"title": self.title, "kind": "idea", "review": "accepted", "author": HUMAN}
        meta["round"] = 0
        self._write(IDEA_PATH, meta, brief, editor=HUMAN, stamp=self.created)
        self._changed()
        attached = f" with {len(entries)} attached file{'s' if len(entries) != 1 else ''}"
        self.emit(
            "project.created",
            f'Project "{_clip(self.title, 80)}" was created{attached if entries else ""}.',
            refs=[IDEA_PATH, f"{INPUTS_DIR}/{BRIEF_FILE}"],
            data={"id": self.id, "files": len(entries), "skipped": len(skipped)},
            round=0,
        )

    @classmethod
    def open(cls, root: str | Path, *, clock: Callable[[], datetime] | None = None) -> Project:
        """Open the project in ``root``.

        Pages written by hand, an older ``project.json`` and unknown settings are
        all accepted; :meth:`problems` says what was read leniently.  Nothing on
        disk is touched, with one exception: a project whose idea page is missing
        (its creation by an earlier version died half way, or the file was removed)
        gets it written again from ``inputs/brief.md``, because nothing can be done
        with a project that has no idea page.
        """
        root = Path(root)
        if not (root / PROJECT_FILE).is_file():
            raise ProjectError(
                f"There is no research project at {root}: {PROJECT_FILE} is missing. "
                "Create one with `paper-orchestra research new`."
            )
        project = cls(root, clock=clock)
        project.reload()
        project._restore_idea()
        return project

    def _restore_idea(self) -> None:
        """Write the idea page again from the brief on file, where the page is missing."""
        brief = self.inputs_dir / BRIEF_FILE
        if (self.root / IDEA_PATH).exists() or brief.is_symlink() or not brief.is_file():
            return
        # A project that cannot be written to (or locked) just now is opened all the same;
        # ``problems`` then says that the idea page is missing.
        with contextlib.suppress(ProjectError, OSError, UnicodeDecodeError), self._lock:
            if (self.root / IDEA_PATH).exists():
                return  # another process was quicker
            text = brief.read_text(encoding="utf-8")
            meta = {"title": self.title, "kind": "idea", "review": "accepted", "author": HUMAN}
            meta["round"] = 0
            page = self._write(
                IDEA_PATH, meta, text, editor=HUMAN, stamp=self._next_stamp(IDEA_PATH)
            )
            self._changed()
            self._page_event(
                "page.created",
                f"The idea page was missing and was written again from {INPUTS_DIR}/{BRIEF_FILE}.",
                page,
                SYSTEM,
                restored=True,
            )

    def at_home(self) -> bool | None:
        """Whether this directory is the one the project was created as: ``False`` for a
        copy of it (and for one moved to another file system), ``None`` when the project
        does not say where it was created (it is older than this record).

        Two directories with one id are a project and its copy; this tells which is
        which, so that the id goes on naming the project and not the copy.
        """
        home = self._extra.get(HOME_KEY)
        inode = home.get("inode") if isinstance(home, dict) else None
        if not isinstance(inode, int) or isinstance(inode, bool):
            return None
        try:
            return self.root.stat().st_ino == inode
        except OSError:
            return None

    def reload(self) -> None:
        """Read ``project.json`` again: another process may have changed title or settings."""
        path = self.root / PROJECT_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as exc:
            said = "it is nested too deeply" if isinstance(exc, RecursionError) else exc
            raise ProjectError(
                f"{path} cannot be read as JSON ({said}). Repair the file or restore it "
                "from a copy; the wiki pages themselves are untouched."
            ) from exc
        if not isinstance(data, dict):
            raise ProjectError(f"{path} must hold a JSON object, not {type(data).__name__}.")
        problems: list[str] = []
        project_id = data.get("id")
        if not isinstance(project_id, str) or not project_id.strip():
            digest = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:12]
            project_id = f"p-{digest}"
            problems.append(f"no id; {project_id} (derived from the location) is used")
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            title = self.root.name
            problems.append(f"no title; the directory name '{title}' is used")
        self.id = project_id
        self.title = _one_line(title)
        self.created = str(data.get("created") or "")
        self.settings = ProjectSettings.from_dict(data.get("settings"))
        self._extra = {
            key: value
            for key, value in data.items()
            if key not in ("schema", "id", "title", "created", "settings")
        }
        self._file_problems = problems + self.settings.load_problems
        self._title_on_file = self.title
        self.wiki.name = self.title

    def _write_project_file(self) -> None:
        write_json(
            self.root / PROJECT_FILE,
            {
                "schema": SCHEMA,
                "id": self.id,
                "title": self.title,
                "created": self.created,
                "settings": self.settings.to_dict(),
                **self._extra,
            },
        )

    def save_settings(self) -> None:
        """Write ``title`` and ``settings`` to ``project.json``.

        Refuses invalid settings with every problem named; the file is left as it was,
        and so is this object: when the save is refused or fails, ``title`` and
        ``settings`` are read back from the file.
        """
        try:
            self.settings.validate()
            title = _one_line(self.title)
            if not title:
                raise ProjectError("A project needs a title.")
            _check_storable("The project's title or settings", title, self.settings.to_dict())
            with self._lock:
                self.title = title
                if title != self._title_on_file:
                    self._extra.pop(TITLE_FROM_BRIEF, None)  # someone chose this one
                self._write_project_file()
                self._title_on_file = title
                self._file_problems = []
                self.wiki.name = title
                self._changed()
        except BaseException:
            # What could not be saved is not kept in memory: whoever holds this object
            # would go on showing, and working with, settings the project does not have.
            with contextlib.suppress(ProjectError):
                self.reload()
            raise

    @property
    def title_is_provisional(self) -> bool:
        """Whether the title is only the first line of the brief, because the person gave none."""
        return bool(self._extra.get(TITLE_FROM_BRIEF))

    def rename(self, title: str, *, by: str) -> None:
        """Give the project the title ``title``.

        The person may always.  An agent may only where the title is still the
        provisional one taken from the brief (``NotAllowed`` otherwise): a name
        the person chose, or one already given, is not an agent's to change.
        The idea page carried the provisional title and follows it; its text is
        the person's and is not touched.
        """
        by = normalize_author(by)
        title = _one_line(title)
        if not title or len(title) > TITLE_LIMIT:
            raise ProjectError(
                f"A project's title is one line of at most {TITLE_LIMIT} characters."
            )
        _check_storable("The project's title", title)
        with self._lock:
            self.reload()
            if by != HUMAN and not self.title_is_provisional:
                raise NotAllowed(
                    f'The project already has a title ("{self.title}"). Only the person renames it.'
                )
            previous = self.title
            self.title = title
            self._extra.pop(TITLE_FROM_BRIEF, None)
            self._write_project_file()
            self._title_on_file = title
            self.wiki.name = title
            if self.wiki.exists(IDEA_PATH) and self.page(IDEA_PATH).title == previous:
                self.update_page(IDEA_PATH, author=SYSTEM, fields={"title": title})
            self._changed()

    # -- places ------------------------------------------------------------------

    @property
    def inputs_dir(self) -> Path:
        return self.root / INPUTS_DIR

    @property
    def paper_dir(self) -> Path:
        return self.root / PAPER_DIR

    def experiment_dir(self, experiment_id: str) -> Path:
        """``experiments/E<n>``: the working files of an experiment (not created here)."""
        parsed = parse_id(experiment_id)
        if parsed is None or parsed[0] != ID_PREFIXES["experiment"]:
            resolver = self._resolver()
            page = resolver.pages.get(resolver.resolve(experiment_id) or "")
            parsed = parse_id(page.id) if page and page.kind == "experiment" else None
        if parsed is None:
            raise ProjectError(f"{experiment_id!r} is not an experiment id such as E1.")
        return self.root / EXPERIMENTS_DIR / f"{parsed[0]}{parsed[1]}"

    def attachments(self) -> list[dict[str, Any]]:
        """The files the person attached: name (under ``inputs/``), size, media type, sha256."""
        manifest = _read_json(self.inputs_dir / MANIFEST_FILE)
        files = manifest.get("files") if isinstance(manifest, dict) else None
        return [dict(entry) for entry in files or [] if isinstance(entry, dict)]

    def sweep_temporary_files(self) -> list[str]:
        """Remove the temporary files that writers which died left in the project, and
        return their project-relative paths.

        Every file of the project is written under a temporary name and renamed into
        place (``.<name>.<pid>.<random>.tmp``); a writer killed in between leaves the
        temporary file for good.  One is removed when the process named in it is gone
        and it is older than ``TEMPORARY_FILE_AGE`` seconds.  The person's ``inputs/``
        and anything behind a symbolic link are not looked at.  Called when a worker
        starts on the project.
        """
        removed: list[str] = []
        for folder, dirs, names in os.walk(self.root):
            here = Path(folder)
            dirs[:] = [
                name
                for name in dirs
                if not (here / name).is_symlink() and (here != self.root or name != INPUTS_DIR)
            ]
            for name in names:
                match = _TEMPORARY_NAME.match(name)
                file = here / name
                if match is None or file.is_symlink():
                    continue
                if _process_is_gone(int(match.group(1))) and _older_than(file, TEMPORARY_FILE_AGE):
                    with contextlib.suppress(OSError):
                        file.unlink()
                        removed.append(file.relative_to(self.root).as_posix())
        return sorted(removed)

    # -- time and engine state ---------------------------------------------------

    def _engine_state(self) -> dict[str, Any]:
        """``state.json`` as the engine last wrote it (empty before the first phase)."""
        path = self.root / STATE_FILE
        try:
            stat = path.stat()
        except OSError:
            return {}
        key = (stat.st_mtime_ns, stat.st_size)
        if self._state_cache is None or self._state_cache[0] != key:
            data = _read_json(path)
            self._state_cache = (key, data if isinstance(data, dict) else {})
        return self._state_cache[1]

    def current_round(self) -> int:
        """The round the engine is in, 0 before it has started."""
        value = self._engine_state().get("round")
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def _running_phase(self) -> str:
        state = self._engine_state()
        return str(state.get("phase") or "") if state.get("status") == "running" else ""

    # -- reading pages -----------------------------------------------------------

    def _file_stamp(self, path: str) -> str:
        try:
            moment = datetime.fromtimestamp((self.root / path).stat().st_mtime, UTC)
        except OSError:
            return ""
        return format_stamp(moment)

    def _scan(self) -> list[tuple[str, tuple[int, int, int]]]:
        """Every page file under ``wiki/``, with what tells one version of it from the next.

        The rules are the wiki layer's own (``Wiki.list``): Markdown files, nothing
        hidden, and no symbolic link is followed, so nothing outside ``wiki/`` is ever
        read as a page.  It is done here because every operation of the store starts
        by listing the pages, and this way that costs one ``stat`` per page.
        """
        found: list[tuple[str, tuple[int, int, int]]] = []
        folders = [WIKI_DIR]
        while folders:
            folder = folders.pop()
            try:
                entries = list(os.scandir(self.root / folder))
            except OSError:
                continue
            for entry in entries:
                if entry.name.startswith(".") or entry.is_symlink():
                    continue
                path = f"{folder}/{entry.name}"
                try:
                    if entry.is_dir(follow_symlinks=False):
                        folders.append(path)
                    elif entry.name.endswith(".md") and entry.is_file(follow_symlinks=False):
                        stat = entry.stat(follow_symlinks=False)
                        found.append((path, (stat.st_ino, stat.st_mtime_ns, stat.st_size)))
                except OSError:
                    continue
        return found

    def _load_page(self, path: str) -> ResearchPage | None:
        """Parse one page file; ``None`` for what is not a page (not text, or generated)."""
        try:
            text = (self.root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        meta, body = _split(text)
        # A listing the wiki layer generated for a crowded section is not a page.
        if meta.get("auto") and meta.get("kind") == "index":
            return None
        stamp = "" if parse_stamp(meta.get("updated")) else self._file_stamp(path)
        return page_from_parts(path, meta, body, file_stamp=stamp)

    def _all_pages(self) -> list[ResearchPage]:
        """Every page, in reading order.  A file is parsed again only when it changed.

        The pages returned are shared with later calls: internal code reads them and
        never changes them (:meth:`pages` hands out copies).
        """
        known = self._page_cache
        current: dict[str, tuple[tuple[int, int, int], ResearchPage | None]] = {}
        for path, signature in self._scan():
            cached = known.get(path)
            if cached is None or cached[0] != signature:
                cached = (signature, self._load_page(path))
            current[path] = cached
        self._page_cache = current
        return sorted((page for _, page in current.values() if page is not None), key=_page_order)

    def _resolver(self) -> _Resolver:
        return _Resolver(self.wiki, self._all_pages())

    def pages(
        self,
        *,
        kind: str | Iterable[str] | None = None,
        review: str | Iterable[str] | None = None,
        state: str | Iterable[str] | None = None,
    ) -> list[ResearchPage]:
        """Every page, optionally only those of a kind, review or state (one or several).

        Ordered by kind as research proceeds, then by number, then by path.
        """
        kinds, reviews, states = _as_filter(kind), _as_filter(review), _as_filter(state)
        return [
            replace(
                page,
                meta={key: _own_copy(value) for key, value in page.meta.items()},
                relations={name: list(targets) for name, targets in page.relations.items()},
            )
            for page in self._all_pages()
            if (kinds is None or page.kind in kinds)
            and (reviews is None or page.review in reviews)
            and (states is None or page.state in states)
        ]

    def resolve(self, ref: str) -> str | None:
        """The path of the page ``ref`` names, or ``None``."""
        return self._resolver().resolve(ref)

    def page(self, ref: str) -> ResearchPage:
        """The page ``ref`` names, as it is on disk now.  ``NotFound`` when there is none."""
        return self._fresh(ref)[0]

    def _fresh(self, ref: str) -> tuple[ResearchPage, str]:
        """The page and its file's text, read from disk and not from any cache."""
        path = self.resolve(ref)
        text = None
        if path is not None:
            with contextlib.suppress(WikiError, OSError, UnicodeDecodeError):
                text = self.wiki.read_text(path)
        if path is None or text is None:
            raise NotFound(f"No page matches {ref!r}. Use a path, an id such as H2, or a link.")
        meta, body = _split(text)
        stamp = "" if parse_stamp(meta.get("updated")) else self._file_stamp(path)
        return page_from_parts(path, meta, body, file_stamp=stamp), text

    # -- writing pages (internals) -----------------------------------------------

    def _history_dir(self, path: str) -> Path:
        return self.root / HISTORY_DIR / path

    def _versions(self, path: str) -> list[PageVersion]:
        folder = self._history_dir(path)
        versions = []
        if folder.is_dir():
            for file in folder.iterdir():
                match = _HISTORY_NAME.match(file.name)
                moment = parse_stamp(match.group(1)) if match else None
                if match is None or moment is None or not file.is_file():
                    continue
                versions.append(
                    PageVersion(
                        stamp=format_stamp(moment),
                        author=_author_from_slug(match.group(2)),
                        size=file.stat().st_size,
                        path=file.relative_to(self.root).as_posix(),
                    )
                )
        return sorted(versions, key=lambda version: version.stamp, reverse=True)

    def _next_stamp(self, path: str) -> str:
        """A stamp later than every version kept of this page, also within one second.

        The version being replaced must be in the history already (:meth:`_supersede`).
        """
        versions = self._versions(path)
        return stamp_after(self._clock.now(), versions[0].stamp if versions else "")

    def _archive(self, page: ResearchPage, text: str) -> str:
        """Keep ``text``, the version of ``page`` about to be replaced or removed, under
        ``history/``, and return the stamp it is filed under.

        That is the page's own ``updated`` stamp.  A page edited or restored by hand may
        carry no usable stamp, or the stamp of an earlier version with another text; it
        is then filed by the time the file was written, or after everything kept so far,
        so that no version ever replaces another.
        """
        stamp = page.updated
        if parse_stamp(stamp) is None:
            stamp = self._file_stamp(page.path) or self._clock.now()
        versions = self._versions(page.path)
        taken = [self.root / version.path for version in versions if version.stamp == stamp]
        if any(_read_text(file) != text for file in taken):
            stamp = stamp_after(stamp, versions[0].stamp)
        writer = str(page.meta.get("updated_by") or page.author)
        name = f"{compact_stamp(stamp)}-{_author_slug(writer)}.md"
        write_text(self._history_dir(page.path) / name, text)
        return stamp

    def _supersede(self, page: ResearchPage, text: str) -> str:
        """Archive the version about to be replaced; return the stamp of the one replacing it."""
        self._archive(page, text)
        return self._next_stamp(page.path)

    def _write(
        self,
        path: str,
        meta: Mapping[str, Any],
        body: str,
        *,
        editor: str,
        stamp: str,
        previous: ResearchPage | None = None,
    ) -> ResearchPage:
        """Write one version of a page, stamped.  ``previous`` is the version it replaces,
        which the caller has archived (``stamp`` comes from :meth:`_supersede`)."""
        stamped = {**meta, "updated": stamp, "updated_by": editor}
        stamped.setdefault("created", (previous.created if previous else "") or stamp)
        _check_storable(path, stamped, body)
        try:
            with self._clock.at(stamp):
                written = self.wiki.write(path, body, ordered_meta(stamped), force=True)
        except WikiError as exc:
            raise ProjectError(f"{path} could not be written: {exc}") from exc
        self._page_cache.pop(path, None)
        return page_from_parts(path, written.meta, written.body)

    def _changed(self) -> None:
        """Keep ``MEMORY.md`` in step: the index of what agents may build on."""
        try:
            self.wiki.write_text(INDEX_FILE, self.index_text())
        except (WikiError, OSError):
            # The index is derived; a page change must not fail over it.
            return

    def _split_fields(
        self,
        kind: str,
        fields: Mapping[str, Any] | None,
        relations: Mapping[str, Any] | None,
        resolver: _Resolver,
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """Separate plain front-matter changes from relation changes, both validated.

        Managed fields are left out (the project sets them); a relation given among
        the fields is treated as a relation.  ``None`` as a value, or an empty
        relation, means "remove".
        """
        changes: dict[str, Any] = {}
        given_relations: dict[str, Any] = {}
        for key, value in (fields or {}).items():
            key = str(key)
            if key in MANAGED_FIELDS:
                continue
            if key in RELATIONS:
                given_relations[key] = value
            elif not _FIELD_NAME.match(key):
                raise ProjectError(
                    f"{key!r} cannot be a front matter field: a field name starts with a "
                    "letter and has only letters, digits, '_' and '-'."
                )
            elif key in ("title", "summary") and value is not None:
                changes[key] = _one_line(value)
            else:
                changes[key] = value
        given_relations.update(relations or {})
        cleaned = clean_relations(kind, given_relations)
        relation_changes = {
            name: [resolver.canonical(target) for target in cleaned.get(name, [])]
            for name in given_relations
        }
        if "title" in changes and not changes["title"]:
            raise ProjectError("A page needs a title.")
        return changes, relation_changes

    def _check_state(self, page: ResearchPage, state: Any, by: str) -> None:
        """Raise unless ``by`` may move ``page`` to ``state`` (messages name the page)."""
        try:
            check_transition(page.kind, page.state, str(state), by)
        except ProjectError as exc:
            raise type(exc)(f"{_name(page)}: {exc}") from None
        unaccepted = page.review != "accepted"
        if page.kind == "experiment" and state == "approved" and unaccepted:
            raise ProjectError(
                f"{_name(page)} is {page.review}, not accepted. Accept the experiment's page "
                "before approving its run."
            )

    def _apply_changes(
        self,
        page: ResearchPage,
        *,
        by: str,
        body: str | None,
        fields: Mapping[str, Any] | None,
        relations: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], str]:
        """The page's front matter and body with the changes made.  Nothing is written.

        This is every edit of a page: a direct one, the person's edits while
        accepting, an accepted edit proposal.  None of them approves an experiment
        or writes the record of an approval; ``set_state`` and ``record_approval``
        do, for ``research.experiments.approve`` and the run.
        """
        meta = dict(page.meta)
        changes, relation_changes = self._split_fields(
            page.kind, fields, relations, self._resolver()
        )
        state = changes.pop("state", None)
        # A state cannot be removed, so "no state" in a form means "leave it".
        if state is not None and state != page.state:
            if page.kind == "experiment" and state in _RUN_STATES:
                if by != HUMAN:
                    self._check_state(page, state, by)  # says whose move it is
                raise ProjectError(
                    f"{_name(page)} cannot be made {state} by editing its page. " + _HOW_TO_APPROVE
                )
            self._check_state(page, state, by)
            meta["state"] = state
        for key in sorted(_APPROVAL_RECORD & changes.keys()):
            if changes[key] != page.meta.get(key):
                raise ProjectError(
                    f"{_name(page)}: '{key}' is the record of an approval and is written by "
                    "the approval itself, not by an edit. " + _HOW_TO_APPROVE
                )
        for key, value in {**changes, **relation_changes}.items():
            if value is None or value == []:
                meta.pop(key, None)
            else:
                meta[key] = value
        return meta, page.body if body is None else _with_newline(str(body))

    @staticmethod
    def _check_base(page: ResearchPage, base_updated: str | None) -> None:
        """Raise ``Conflict`` when ``page`` is no longer the version someone acted on."""
        if base_updated is None or base_updated == page.updated:
            return
        last = page.meta.get("updated_by") or page.author
        raise Conflict(
            f"{_name(page)} changed after the version being edited: it was written at "
            f"{base_updated}, and {last} changed it at {page.updated}. Reload the page and "
            "make the change again.",
            page=page,
        )

    def _check_may_modify(self, page: ResearchPage, author: str, *, text: bool) -> None:
        """Raise ``NotAllowed`` unless ``author`` may change ``page`` directly.

        The person may change anything.  An agent may change only a page of its
        own that no person has decided or edited yet and that is still in its first
        state: a plan the person approved, or a hypothesis under test, stays what it
        was when that happened.  Pipeline code writes records, may set fields anywhere,
        and rewrites or removes the text of a page only while no person wrote it
        or decided on it (``text`` says whether the body is touched).
        """
        if author == HUMAN:
            return
        name = _name(page)
        decided = bool(page.decided_by) or page.review == "rejected"
        if is_agent(author):
            if page.is_record:
                raise NotAllowed(
                    f"{name} is a record the pipeline wrote from verified data; an agent "
                    "cannot change it. Say what should be different in a page of your own "
                    "that refers to it."
                )
            if page.author == HUMAN:
                raise NotAllowed(
                    f"{name} was written by the person; propose an edit with propose_edit."
                )
            if page.review == "rejected":
                reason = f" ({_clip(page.decision_reason, 200)})" if page.decision_reason else ""
                raise NotAllowed(
                    f"{name} was rejected by the person{reason} and is kept as it was. "
                    "Write a new page that addresses the reason instead."
                )
            if decided:
                raise NotAllowed(
                    f"{name} was accepted by the person; propose an edit with propose_edit."
                )
            if page.author != author:
                raise NotAllowed(
                    f"{name} was written by {page.author}; propose an edit with propose_edit."
                )
            if moved_on(page):
                # Where the kind is accepted without review, this is all that stands
                # between an agent and the plan the person approved for a run.
                raise NotAllowed(
                    f"{name} is {page.state}, so it stays as it was when that was decided; "
                    "propose an edit with propose_edit."
                )
            if self.edited_by_person(page):
                raise NotAllowed(
                    f"{name} was edited by the person while it waits for their decision, so "
                    "it stays as they left it; propose an edit with propose_edit."
                )
            return
        if page.is_record or not text:
            return
        if page.author == HUMAN or decided:
            raise NotAllowed(
                f"{name} is the person's {'page' if page.author == HUMAN else 'decision'}: "
                "pipeline code may set fields on it but does not rewrite its text. "
                "Propose the change with propose_edit."
            )

    @staticmethod
    def edited_by_person(page: ResearchPage) -> bool:
        """Whether the person edited ``page`` themselves, a page someone else wrote
        (``update_page`` notes it on the page).  Such a page is no longer its author's
        to rewrite, also while the decision on it is still open.  A decision, an
        approval or a state the person set is not an edit."""
        return bool(page.meta.get(EDITED_BY_PERSON))

    def _created_after(self, earlier: Sequence[EditProposal | Request]) -> str:
        """A creation stamp later than those of ``earlier``, so "oldest first" is the order
        things were written in even when several are written within a second."""
        latest = max((item.created for item in earlier), default="")
        return stamp_after(self._clock.now(), latest)

    def _allocate_id(self, kind: str) -> str:
        """The next id of ``kind``: one more than any id ever used, deleted pages included."""
        highest = 0
        section = section_path(kind)
        names = [page.id or "" for page in self._all_pages() if page.kind == kind]
        for folder in (self.root / section, self.root / HISTORY_DIR / section):
            if folder.is_dir():
                names += [id_in_name(kind, Path(name).stem) or "" for name in os.listdir(folder)]
        if kind == "experiment" and (self.root / EXPERIMENTS_DIR).is_dir():
            names += os.listdir(self.root / EXPERIMENTS_DIR)
        for name in names:
            parsed = parse_id(name)
            if parsed and parsed[0] == ID_PREFIXES[kind]:
                highest = max(highest, parsed[1])
        return format_id(kind, highest + 1)

    def _new_path(self, kind: str, page_id: str | None, name: str) -> str:
        slug = slugify(name, 48)
        if kind == "idea" or page_id:
            return page_path(kind, id=page_id, slug=slug)
        candidate, number = slug, 1
        while (self.root / page_path(kind, slug=candidate)).exists():
            number += 1
            candidate = f"{slug}-{number}"
        return page_path(kind, slug=candidate)

    def _page_event(
        self, type: str, summary: str, page: ResearchPage, by: str, **data: Any
    ) -> None:
        self.emit(
            type,
            summary,
            agent=role_of(by) or "",
            refs=[page.path],
            data={
                "path": page.path,
                "kind": page.kind,
                "id": page.id,
                "review": page.review,
                "state": page.state,
                "by": by,
                **data,
            },
            round=page.round if type == "page.created" else None,
        )

    def _state_event(
        self, before: ResearchPage, after: ResearchPage, by: str, reason: str | None = None
    ) -> None:
        """Announce that an experiment moved, whichever method moved it: the board and
        the run phase follow ``experiment.state`` events, not page edits."""
        if after.kind != "experiment" or after.state == before.state:
            return
        what = _what(after)
        self._page_event(
            "experiment.state",
            f"{what[0].upper()}{what[1:]} is now {after.state}.",
            after,
            by,
            previous=before.state,
            reason=reason,
        )

    # -- pages -------------------------------------------------------------------

    def create_page(
        self,
        kind: str,
        title: str,
        body: str,
        *,
        author: str,
        summary: str = "",
        fields: Mapping[str, Any] | None = None,
        relations: Mapping[str, Any] | None = None,
        round: int | None = None,
        slug: str | None = None,
    ) -> ResearchPage:
        """Create a page of ``kind`` and return it.

        The id (for questions, hypotheses, experiments and findings) and the file
        name are allocated here.  ``review`` follows from the author and the
        project's policy: a person's page is accepted, an agent's is proposed
        unless the policy for the kind is ``auto``.  An agent's or the pipeline's
        page starts in the kind's first state whatever ``fields`` says, and nobody's
        experiment starts ``approved``, ``queued`` or ``running``: a run is approved
        through ``research.experiments.approve``.  Managed fields in ``fields`` are
        ignored.  Records are written with :meth:`write_record`.
        """
        author = normalize_author(author)
        if kind not in KINDS:
            raise ProjectError(
                f"'{kind}' is not a kind of page; the kinds are: {', '.join(KINDS)}."
            )
        if kind in RECORD_KINDS:
            raise ProjectError(
                f"A {kind} is a record written by pipeline code from verified data: "
                "use write_record, not create_page."
            )
        if kind == "idea" and author != HUMAN:
            raise NotAllowed("The idea page is the person's; only the person writes it.")
        title = _one_line(title)
        if not title:
            raise ProjectError("A page needs a title.")
        with self._lock:
            if kind == "idea" and self.wiki.exists(IDEA_PATH):
                raise ProjectError(
                    f"The project already has its idea page ({IDEA_PATH}); edit that page."
                )
            changes, relation_changes = self._split_fields(
                kind, fields, relations, self._resolver()
            )
            page_id = self._allocate_id(kind) if kind in ID_PREFIXES else None
            path = self._new_path(kind, page_id, slug or title)
            meta: dict[str, Any] = {"title": title, "kind": kind}
            if page_id:
                meta["id"] = page_id
            summary = _one_line(summary) or changes.get("summary") or ""
            changes.pop("summary", None)
            if summary:
                meta["summary"] = summary
            meta["review"] = initial_review(kind, author, {kind: self.settings.policy(kind)})
            wanted = changes.pop("state", None)
            if kind in INITIAL_STATE:
                meta["state"] = INITIAL_STATE[kind]
                if author == HUMAN and wanted is not None:
                    if wanted not in STATES[kind]:
                        raise ProjectError(
                            f"'{wanted}' is not a state of {with_article(kind)}; the states "
                            f"are: {', '.join(STATES[kind])}."
                        )
                    if kind == "experiment" and wanted in _RUN_STATES:
                        raise ProjectError(
                            f"A new experiment cannot start as {wanted}. " + _HOW_TO_APPROVE
                        )
                    meta["state"] = wanted
            meta["author"] = author
            meta["round"] = self.current_round() if round is None else int(round)
            changes.pop("title", None)
            meta.update({key: value for key, value in changes.items() if value is not None})
            meta.update({name: targets for name, targets in relation_changes.items() if targets})
            page = self._write(
                path, meta, str(body or ""), editor=author, stamp=self._next_stamp(path)
            )
            self._changed()
            verb = "proposed" if page.review == "proposed" else "added"
            self._page_event("page.created", f"{_who(author)} {verb} {_what(page)}.", page, author)
        return page

    def update_page(
        self,
        ref: str,
        *,
        author: str,
        body: str | None = None,
        fields: Mapping[str, Any] | None = None,
        relations: Mapping[str, Any] | None = None,
        base_updated: str | None = None,
        accept: bool = False,
    ) -> ResearchPage:
        """Change a page directly and return the new version.

        ``body`` replaces the text; ``fields`` sets front-matter fields (``None``
        removes one; ``title`` and ``summary`` are fields; a ``state`` goes through
        the lifecycle rules); ``relations`` replaces the named relation lists.
        With ``base_updated`` the change is refused as a ``Conflict`` when the page
        is no longer the version the editor started from.

        No edit approves an experiment: ``state`` ``approved``, ``queued`` or
        ``running`` and the record of an approval (``approved_digest``,
        ``approved_at``, ``repaired_digest``) are refused for everyone, with a
        message that says where a run is approved.  The person may still take an
        approval back (``state: planned``).

        Editing is not deciding.  A proposed page the person edits stays proposed
        and a rejected one stays rejected, with their corrections in it; ``accept``
        (the person's "save and accept") is the explicit decision, the same as
        :meth:`review` with these changes as its ``edits``, and only the person can
        make it.  A page that was accepted without review becomes the person's
        decision once they edit it.  An agent may change only its own page that no
        person has decided or edited; anything else is ``NotAllowed`` and goes
        through :meth:`propose_edit`.
        """
        author = normalize_author(author)
        if accept and author != HUMAN:
            raise NotAllowed(
                f"Only the person accepts a page; {author} can edit its own proposal and "
                "waits for the decision."
            )
        with self._lock:
            page, text = self._fresh(ref)
            self._check_base(page, base_updated)
            settled = page.review == "accepted" and (bool(page.decided_by) or page.author == HUMAN)
            if accept and not settled:
                given = {"body": body, "fields": fields, "relations": relations}
                edits = {name: value for name, value in given.items() if value is not None}
                return self._decide_page(page.path, "accept", None, edits, base_updated)
            self._check_may_modify(page, author, text=body is not None)
            meta, new_body = self._apply_changes(
                page, by=author, body=body, fields=fields, relations=relations
            )
            if meta == page.meta and new_body == page.body:
                return page
            _check_storable(page.path, meta, new_body)
            stamp = self._supersede(page, text)
            if author == HUMAN and page.review == "accepted" and not page.decided_by:
                # It was accepted without a look; now it holds what the person wrote, and
                # from here on its author proposes changes like everyone else.  (Nothing
                # is decided by this: the page was accepted already.)
                meta.update({"decided_by": HUMAN, "decided_at": stamp})
            if author == HUMAN and page.author != HUMAN:
                meta[EDITED_BY_PERSON] = stamp
            new = self._write(page.path, meta, new_body, editor=author, stamp=stamp, previous=page)
            self._changed()
            verb = "edited" if author == HUMAN else "revised" if is_agent(author) else "updated"
            self._page_event("page.updated", f"{_who(author)} {verb} {_what(new)}.", new, author)
            self._state_event(page, new, author)
        return new

    def delete_page(self, ref: str, *, author: str) -> None:
        """Remove a page.  Its last version is kept under ``history/``.

        The person may delete any page but the idea.  An agent may delete only its
        own page that no person has decided; pipeline code only records and pages
        no person wrote or decided on.
        """
        author = normalize_author(author)
        with self._lock:
            page, text = self._fresh(ref)
            self._check_may_modify(page, author, text=True)
            if page.path == IDEA_PATH:
                raise ProjectError(
                    "The idea page is the root of the project and cannot be deleted; edit it."
                )
            if page.kind == "experiment" and page.state in ("queued", "running"):
                raise ProjectError(
                    f"{_name(page)} is {page.state}; cancel the run before deleting."
                )
            self._archive(page, text)
            try:
                self.wiki.delete(page.path, force=True)
            except WikiError as exc:
                raise ProjectError(f"{page.path} could not be deleted: {exc}") from exc
            self._page_cache.pop(page.path, None)
            self._changed()
            self._page_event("page.deleted", f"{_who(author)} deleted {_what(page)}.", page, author)

    def history(self, ref: str) -> list[PageVersion]:
        """Earlier versions of a page, newest first (the current one is the page itself).

        Also answers for a deleted page, by its path.
        """
        path = self._history_path(ref)
        versions = self._versions(path)
        with contextlib.suppress(NotFound):
            page, text = self._fresh(path)
            # A copy of the current version can be there too (it was filed before a write
            # that then failed, or the person restored that version by hand): it is not an
            # earlier one.
            versions = [
                version
                for version in versions
                if version.stamp != page.updated or _read_text(self.root / version.path) != text
            ]
        return versions

    def version(self, ref: str, stamp: str) -> str:
        """The full text (front matter and body) of the version written at ``stamp``."""
        path = self._history_path(ref)
        wanted = compact_stamp(stamp)
        for version in self._versions(path):
            if compact_stamp(version.stamp) == wanted:
                return (self.root / version.path).read_text(encoding="utf-8")
        with contextlib.suppress(NotFound):
            current, text = self._fresh(path)
            if compact_stamp(current.updated) == wanted:
                return text
        raise NotFound(f"{path} has no version written at {stamp}.")

    def _history_path(self, ref: str) -> str:
        path = self.resolve(ref)
        if path is not None:
            return path
        try:
            candidate = self.wiki.normalize(str(ref).strip().removeprefix("/memories/"))
        except WikiError:
            candidate = ""
        if candidate and self._history_dir(candidate).is_dir():
            return candidate
        raise NotFound(f"No page, present or deleted, matches {ref!r}.")

    # -- review ------------------------------------------------------------------

    def review(
        self,
        ref_or_proposal_id: str,
        decision: str,
        *,
        by: str = "human",
        reason: str | None = None,
        edits: Mapping[str, Any] | None = None,
        base_updated: str | None = None,
    ) -> ResearchPage | EditProposal:
        """Decide a proposed page or an edit proposal; returns what was decided.

        ``decision`` is ``accept`` or ``reject``.  Only the person decides: any
        other ``by`` is ``NotAllowed``.  ``edits`` (``{"body", "fields",
        "relations"}``) are the person's own changes, applied on accepting.
        ``base_updated`` is the ``updated`` stamp of the page as the person saw it:
        when its author has revised it since, the decision is refused as a
        ``Conflict`` instead of landing on text they have not read.

        Accepting a page leaves its lifecycle state alone (an accepted experiment
        is still only ``planned``, and ``edits`` cannot approve it: that is
        ``research.experiments.approve``).  Rejecting keeps the page as it was, with
        the reason.  Accepting an edit proposal applies its fields and added relations
        and replaces the body; when the page changed since the edit was proposed,
        the fields still go in but the body does not, and ``Conflict`` is raised
        to say so (``.applied`` tells whether anything was changed).
        """
        choice = _DECISIONS.get(str(decision).strip().lower())
        if choice is None:
            raise ProjectError(f"A review decision is 'accept' or 'reject', not {decision!r}.")
        if normalize_author(by) != HUMAN:
            raise NotAllowed(
                f"Only the person accepts or rejects; {by} cannot decide a review. "
                "Agents propose, and wait."
            )
        self._check_edits(edits)
        reason = _one_line(reason) or None
        with self._lock:
            proposal = self._proposal_or_none(ref_or_proposal_id)
            if proposal is not None:
                return self._decide_proposal(proposal, choice, reason, edits or {})
            return self._decide_page(ref_or_proposal_id, choice, reason, edits or {}, base_updated)

    @staticmethod
    def _check_edits(edits: Mapping[str, Any] | None) -> None:
        if edits is None:
            return
        unknown = sorted(set(edits) - {"body", "fields", "relations"}) if edits else []
        if not isinstance(edits, Mapping) or unknown:
            raise ProjectError(
                "The edits made while accepting are an object with 'body', 'fields' and/or "
                f"'relations'; got {unknown or edits!r}."
            )

    def _decide_page(
        self,
        ref: str,
        choice: str,
        reason: str | None,
        edits: Mapping[str, Any],
        base_updated: str | None,
    ) -> ResearchPage:
        page, text = self._fresh(ref)
        self._check_base(page, base_updated)
        if page.author == HUMAN and choice == "reject":
            raise ProjectError(
                f"{_name(page)} is the person's own page; edit or delete it instead."
            )
        meta, body = dict(page.meta), page.body
        if choice == "accept":
            meta, body = self._apply_changes(
                page,
                by=HUMAN,
                body=edits.get("body"),
                fields=edits.get("fields"),
                relations=edits.get("relations"),
            )
            edited = meta != page.meta or body != page.body
            # Nothing to decide: already accepted by the person, or their own page.  (A page
            # of theirs that a hand-written file marks ``proposed`` is accepted below.)
            settled = bool(page.decided_by) or page.author == HUMAN
            if page.review == "accepted" and settled and not edited:
                return page
        else:
            edited = False
            if page.kind == "experiment" and page.state in ("queued", "running"):
                raise ProjectError(
                    f"{_name(page)} is {page.state}; cancel the run before rejecting its plan."
                )
            if page.kind == "experiment" and page.state == "approved":
                # A rejected plan must not be picked up by the next run phase.
                meta["state"] = "planned"
        stamp = self._supersede(page, text)
        if edited and page.author != HUMAN:
            meta[EDITED_BY_PERSON] = stamp
        meta.update(
            {
                "review": "accepted" if choice == "accept" else "rejected",
                "decided_by": HUMAN,
                "decided_at": stamp,
            }
        )
        meta.pop("decision_reason", None)
        if reason:
            meta["decision_reason"] = reason
        new = self._write(page.path, meta, body, editor=HUMAN, stamp=stamp, previous=page)
        self._changed()
        verb = "accepted" if choice == "accept" else "rejected"
        tail = f": {reason}" if reason and choice == "reject" else ""
        self._page_event(
            "review.decided",
            _sentence(f"You {verb} {_what(new)}{tail}"),
            new,
            HUMAN,
            decision=choice,
            reason=reason,
            edited=edited,
            target="page",
        )
        self._state_event(page, new, HUMAN, reason)
        return new

    # -- edit proposals ----------------------------------------------------------

    def _proposal_path(self, proposal_id: str) -> Path:
        return self.root / PROPOSALS_DIR / f"{proposal_id}.json"

    def _proposal_or_none(self, proposal_id: Any) -> EditProposal | None:
        if not isinstance(proposal_id, str) or not re.fullmatch(r"ep-[0-9a-f]+", proposal_id):
            return None
        data = _read_json(self._proposal_path(proposal_id))
        return EditProposal.from_dict(data) if isinstance(data, dict) else None

    def proposal(self, proposal_id: str) -> EditProposal:
        """The edit proposal with that id.  ``NotFound`` when there is none."""
        found = self._proposal_or_none(proposal_id)
        if found is None:
            raise NotFound(f"There is no edit proposal {proposal_id!r}.")
        return found

    def proposals(self, *, review: str | None = "proposed") -> list[EditProposal]:
        """Edit proposals with that review (``None`` for all), oldest first."""
        folder = self.root / PROPOSALS_DIR
        found = []
        for file in sorted(folder.glob("ep-*.json")) if folder.is_dir() else []:
            data = _read_json(file)
            if isinstance(data, dict) and data.get("id") and data.get("target"):
                proposal = EditProposal.from_dict(data)
                if review is None or proposal.review == review:
                    found.append(proposal)
        return sorted(found, key=lambda proposal: (proposal.created, proposal.id))

    def pending(self) -> list[ResearchPage | EditProposal]:
        """Everything waiting for the person: proposed pages and edit proposals, oldest first."""
        waiting: list[ResearchPage | EditProposal] = [*self.pages(review="proposed")]
        waiting += self.proposals(review="proposed")
        return sorted(waiting, key=lambda item: item.created)

    def _check_proposed_fields(self, page: ResearchPage, fields: Mapping[str, Any]) -> None:
        """What an edit proposal may ask for; raises on anything else."""
        for key, value in fields.items():
            if key in MANAGED_FIELDS or key in ("session", "state_reason"):
                raise NotAllowed(
                    f"'{key}' is set by the project and cannot be proposed. An edit can "
                    "change the text, the title, the summary, relations and the page's "
                    "own fields."
                )
            if key in APPROVAL_FIELDS:
                raise NotAllowed(
                    f"'{key}' is part of the record of an approval: where a run goes and "
                    "what was approved are the person's to decide when they approve the "
                    "experiment, and cannot be proposed."
                )
            if key in _REPOSITORY_FIELDS and page.kind == "experiment":
                raise NotAllowed(
                    f"'{key}' names the repository an experiment copies or clones and runs. "
                    "That is the person's choice, made on the page or in the project's "
                    "settings (settings.repo), and cannot be proposed. Say in the text of "
                    "your edit which repository you think it should be."
                )
            if key != "state":
                continue
            if page.kind == "experiment":
                raise NotAllowed(
                    f"{_name(page)}: the state of an experiment changes only when the person "
                    "approves it and when it runs; an edit cannot propose it."
                )
            states = STATES.get(page.kind, ())
            if value not in states:
                raise ProjectError(
                    f"{_name(page)}: '{value}' is not a state of {with_article(page.kind)}"
                    + (f"; the states are: {', '.join(states)}." if states else " (it has none).")
                )

    def propose_edit(
        self,
        target: str,
        *,
        author: str,
        summary: str,
        body: str | None = None,
        fields: Mapping[str, Any] | None = None,
        relations_add: Mapping[str, Any] | None = None,
        round: int | None = None,
    ) -> EditProposal:
        """Suggest a change to a page; nothing changes until the person accepts it.

        ``body`` is the full replacement text or ``None``; ``fields`` are
        front-matter fields to set (relations given there are added, like
        ``relations_add``).  Refused for records (their content comes from data,
        not from suggestions), for managed fields, for the state of an
        experiment, which only the person's approval and the run itself change,
        for the record of an approval (``backend``, ``approved_digest``,
        ``approved_at``, ``repaired_digest``) and for the repository an experiment
        runs in (``repo``, ``repo_ref``), which only the person names.  Other
        fields of an experiment's plan may be proposed; accepting such an edit
        after the run was approved means the experiment needs a new approval
        before it runs.
        """
        author = normalize_author(author)
        summary = _one_line(summary)
        if not summary:
            raise ProjectError("An edit proposal needs a one-line summary of what it changes.")
        with self._lock:
            page, _ = self._fresh(target)
            if page.is_record:
                raise NotAllowed(
                    f"{_name(page)} is a record the pipeline wrote from verified data; its "
                    "content cannot be edited. Say what should be different in a page that "
                    "refers to it."
                )
            plain = {str(k): v for k, v in (fields or {}).items() if k not in RELATIONS}
            added = {str(k): v for k, v in (fields or {}).items() if k in RELATIONS}
            added.update(relations_add or {})
            self._check_proposed_fields(page, plain)
            changes, relation_changes = self._split_fields(
                page.kind, plain, added, self._resolver()
            )
            relation_changes = {name: items for name, items in relation_changes.items() if items}
            if body is None and not changes and not relation_changes:
                raise ProjectError(
                    "An edit proposal changes something: give a body, fields or relations."
                )
            _check_storable("The edit proposal", summary, body, changes, relation_changes)
            proposal = EditProposal(
                id=f"ep-{secrets.token_hex(4)}",
                target=page.path,
                author=author,
                round=self.current_round() if round is None else int(round),
                created=self._created_after(self.proposals(review=None)),
                summary=summary,
                body=None if body is None else _with_newline(str(body)),
                fields=changes,
                relations_add=relation_changes,
                base_updated=page.updated,
            )
            write_json(self._proposal_path(proposal.id), proposal.to_dict())
            self.emit(
                "proposal.created",
                _sentence(f"{_who(author)} proposed an edit to {_what(page)}: {_said(summary)}"),
                agent=role_of(author) or "",
                refs=[page.path, f"{PROPOSALS_DIR}/{proposal.id}.json"],
                data={"proposal": proposal.id, "path": page.path, "id": page.id, "by": author},
                round=proposal.round,
            )
        return proposal

    def _decide_proposal(
        self, proposal: EditProposal, choice: str, reason: str | None, edits: Mapping[str, Any]
    ) -> EditProposal:
        if proposal.review != "proposed":
            raise ProjectError(
                f"{proposal.id} was already {proposal.review} at {proposal.decided_at}."
            )
        decided = {"decided_by": HUMAN, "decided_at": self._clock.now(), "decision_reason": reason}
        if choice == "reject":
            return self._close_proposal(replace(proposal, review="rejected", **decided), None)
        try:
            page, text = self._fresh(proposal.target)
        except NotFound:
            raise NotFound(
                f"{proposal.target}, the page {proposal.id} would change, no longer exists. "
                "Reject the proposal."
            ) from None
        fields = dict(edits["fields"]) if "fields" in edits else dict(proposal.fields)
        if page.kind == "experiment" and "state" in proposal.fields and "fields" not in edits:
            raise NotAllowed(
                f"{proposal.id} asks to change the state of {_name(page)}. Only the person "
                "does that, by approving the experiment itself; reject this proposal."
            )
        carried = sorted(APPROVAL_FIELDS & proposal.fields.keys())
        if carried and "fields" not in edits:
            # A file written around the store: ``propose_edit`` refuses these fields.
            raise NotAllowed(
                f"{proposal.id} asks to set {', '.join(carried)} of {_name(page)}, which "
                "belong to the record of an approval. Only the person's approval of the "
                "experiment writes them; reject this proposal."
            )
        named = sorted(_REPOSITORY_FIELDS & proposal.fields.keys())
        if named and page.kind == "experiment" and "fields" not in edits:
            raise NotAllowed(
                f"{proposal.id} asks to set {', '.join(named)} of {_name(page)}: the "
                "repository an experiment runs in. Only the person names it, on the page "
                "or in the project's settings; reject this proposal."
            )
        # The person's own text was written looking at the page as it is; the agent's was
        # written against the text of an earlier version.  A page that already reads as
        # proposed has nothing to conflict with (an accept that was cut off after writing
        # the page and is now repeated).
        stale = (
            "body" not in edits
            and proposal.body is not None
            and proposal.body != page.body
            and self._text_changed_since(page, proposal.base_updated)
        )
        body = edits.get("body", None if stale else proposal.body)
        relations = {
            name: [*page.relations.get(name, []), *targets]
            for name, targets in proposal.relations_add.items()
        }
        relations.update(edits.get("relations") or {})
        meta, new_body = self._apply_changes(
            page, by=HUMAN, body=body, fields=fields, relations=relations
        )
        new = page
        applied = meta != page.meta or new_body != page.body
        if applied:
            stamp = self._supersede(page, text)
            if page.review == "accepted" and not page.decided_by:
                # A page that was accepted without review now holds something the person
                # approved: from here on its author proposes changes like everyone else.
                meta.update({"decided_by": HUMAN, "decided_at": stamp})
            new = self._write(
                page.path,
                meta,
                new_body,
                editor=proposal.author,
                stamp=stamp,
                previous=page,
            )
            self._changed()
            self._state_event(page, new, HUMAN)
        if not stale:
            return self._close_proposal(replace(proposal, review="accepted", **decided), new)
        conflict = (
            f"The text of {_name(page)} changed after this edit was proposed (it was written "
            f"against the version of {proposal.base_updated}; the page is now that of "
            f"{page.updated}), so its replacement text was not applied."
        )
        left = replace(proposal, conflict=conflict)
        if applied:
            left = self._close_proposal(replace(left, review="accepted", **decided), new)
        else:
            write_json(self._proposal_path(left.id), left.to_dict())
        raise Conflict(
            conflict
            + (
                " Its field changes were applied."
                if applied
                else " Nothing was changed and the proposal is still open."
            )
            + " Compare the proposed text with the page and edit the page by hand"
            + ("." if applied else ", then reject the proposal."),
            page=new,
            proposal=left,
            applied=applied,
        )

    def _text_changed_since(self, page: ResearchPage, base_updated: str) -> bool:
        """Whether the page's text differs from the version written at ``base_updated``.

        A replacement text is out of date only when the text it replaces is not the
        one its author read.  A decision or a field set since then (accepting the
        page, another edit's fields, a state) leaves that text as it was.
        """
        if page.updated == base_updated:
            return False
        try:
            base = self.version(page.path, base_updated)
        except NotFound:
            return True
        return _split(base)[1] != page.body

    def _close_proposal(self, proposal: EditProposal, page: ResearchPage | None) -> EditProposal:
        write_json(self._proposal_path(proposal.id), proposal.to_dict())
        verb = "accepted" if proposal.review == "accepted" else "rejected"
        what = _what(page) if page is not None else quotable(proposal.target, 120)
        # The reason is the person's own; the summary is what an agent called its edit.
        tail = proposal.decision_reason if verb == "rejected" and proposal.decision_reason else ""
        self.emit(
            "review.decided",
            _sentence(
                f"You {verb} {_whose(proposal.author)} edit to {what}: "
                f"{tail or _said(proposal.summary)}"
            ),
            refs=[proposal.target, f"{PROPOSALS_DIR}/{proposal.id}.json"],
            data={
                "proposal": proposal.id,
                "path": proposal.target,
                "decision": "accept" if verb == "accepted" else "reject",
                "reason": proposal.decision_reason,
                "conflict": bool(proposal.conflict),
                "target": "proposal",
                "by": HUMAN,
            },
        )
        return proposal

    # -- lifecycle ---------------------------------------------------------------

    def set_state(
        self,
        ref: str,
        state: str,
        *,
        by: str,
        reason: str | None = None,
        base_updated: str | None = None,
    ) -> ResearchPage:
        """Move a question, hypothesis or experiment along its lifecycle.

        Enforces which moves exist and who makes them (``model.check_transition``):
        an experiment becomes ``approved`` only by the person, and only once its
        page is accepted (``research.experiments.approve`` is the one caller that
        does this: it records what was approved, and a run starts only against
        that record); the run's own steps are made by pipeline code; whether a
        hypothesis held is the person's verdict.  Setting the state a page already
        has changes nothing.  With ``base_updated`` (the page as the person saw
        it) a page that changed since is a ``Conflict``: what gets approved is what
        was read.
        """
        by = normalize_author(by)
        with self._lock:
            page, text = self._fresh(ref)
            self._check_base(page, base_updated)
            self._check_state(page, state, by)
            if page.state == state:
                # Nothing moves, so nothing is written: not even by an author who could
                # not have made a real move.
                return page
            meta = dict(page.meta)
            meta["state"] = state
            meta.pop("state_reason", None)
            if _one_line(reason):
                meta["state_reason"] = _one_line(reason)
            new = self._write(
                page.path,
                meta,
                page.body,
                editor=by,
                stamp=self._supersede(page, text),
                previous=page,
            )
            self._changed()
            if page.kind == "experiment":
                self._state_event(page, new, by, _one_line(reason) or None)
            else:
                what = _what(new)
                self._page_event(
                    "page.updated",
                    f"{what[0].upper()}{what[1:]} is now {state}.",
                    new,
                    by,
                    previous=page.state,
                    reason=_one_line(reason) or None,
                )
        return new

    def locked(self) -> _ProjectLock:
        """The project's lock, for ``with project.locked():`` around several steps that
        must be one: a check and the write that relies on it.

        It is the lock every changing method takes (reentrant, and it excludes other
        threads and other processes), so the methods can be called inside it.  Hold it
        for the steps only, never while something slow runs.
        """
        return self._lock

    def record_approval(
        self,
        ref: str,
        fields: Mapping[str, Any],
        *,
        by: str,
        base_updated: str | None = None,
    ) -> ResearchPage:
        """Write the record of an approval onto an experiment's page; returns the new version.

        ``fields`` holds the options of the approved run (``backend``, ``gpu``,
        ``timeout_seconds``, ``network``) and the record itself (``approved_digest``,
        ``approved_at``, ``repaired_digest``); ``None`` removes a field.  This is how
        ``research.experiments.approve`` (as the person) and the run (as pipeline
        code, after a repair) write them: no edit, review or proposal can, and no
        agent.  It records and does not approve: the state moves with
        :meth:`set_state`.  With ``base_updated`` a page that changed since is a
        ``Conflict``.
        """
        by = normalize_author(by)
        if is_agent(by) and by != role_author("runner"):
            raise NotAllowed(
                f"The record of an approval is written when the person approves a run; {by} "
                "cannot write it."
            )
        unknown = sorted(set(fields) - _APPROVAL_WRITES)
        if unknown:
            raise ProjectError(
                f"record_approval writes {', '.join(sorted(_APPROVAL_WRITES))}; "
                f"got {', '.join(unknown)}. Change other fields with update_page."
            )
        with self._lock:
            page, text = self._fresh(ref)
            if page.kind != "experiment":
                raise ProjectError(
                    f"{_name(page)} is {with_article(page.kind)}, not an experiment: only an "
                    "experiment has a run to approve."
                )
            self._check_base(page, base_updated)
            meta = dict(page.meta)
            for key, value in fields.items():
                if value is None:
                    meta.pop(key, None)
                else:
                    meta[key] = value
            if meta == page.meta:
                return page
            new = self._write(
                page.path,
                meta,
                page.body,
                editor=by,
                stamp=self._supersede(page, text),
                previous=page,
            )
            self._changed()
            self._page_event(
                "page.updated",
                _approval_sentence(by, new, fields),
                new,
                by,
                approval=sorted(fields),
            )
        return new

    # -- records -----------------------------------------------------------------

    def write_record(
        self,
        kind: str,
        name: str,
        title: str,
        body: str,
        *,
        summary: str = "",
        fields: Mapping[str, Any] | None = None,
        relations: Mapping[str, Any] | None = None,
        round: int | None = None,
    ) -> ResearchPage:
        """Write a source or result record as pipeline code (author ``system``).

        ``name`` is what the record is known by: the citation key of a source
        (also its id), ``E1-run1`` for a result.  Writing an existing record
        replaces it (the old version goes to history) and keeps the person's
        decision on it.  A result is accepted on creation; a source too, unless
        the project's review policy for sources says ``review``.
        """
        if kind not in RECORD_KINDS:
            raise ProjectError(
                f"write_record writes records ({', '.join(RECORD_KINDS)}); "
                f"use create_page for a {kind}."
            )
        name = str(name or "").strip().removesuffix(".md")
        if not _RECORD_NAME.match(name):
            raise ProjectError(
                f"{name!r} cannot name a {kind}: use letters, digits, '.', '_' and '-'."
            )
        title = _one_line(title)
        if not title:
            raise ProjectError("A record needs a title.")
        path = page_path(kind, name=name)
        with self._lock:
            previous = self._fresh(path) if self.wiki.exists(path) else None
            changes, relation_changes = self._split_fields(
                kind, fields, relations, self._resolver()
            )
            state = changes.pop("state", None)
            if state is not None and state not in STATES.get(kind, ()):
                raise ProjectError(
                    f"'{state}' is not a state of a {kind}"
                    + (f"; the states are: {', '.join(STATES[kind])}." if kind in STATES else ".")
                )
            meta: dict[str, Any] = {"title": title, "kind": kind}
            if kind == "source":
                meta["id"] = name
            summary = _one_line(summary) or changes.get("summary") or ""
            changes.pop("summary", None)
            if summary:
                meta["summary"] = summary
            # A result is what a run produced; a source is held back only where the person
            # asked to vet sources themselves.
            meta["review"] = "accepted" if self.settings.policy(kind) == "auto" else "proposed"
            if state is not None:
                meta["state"] = state
            meta["author"] = SYSTEM
            meta["round"] = self.current_round() if round is None else int(round)
            if previous is not None:
                kept = ("review", "round", "created", "decided_by", "decided_at", "decision_reason")
                meta.update({key: previous[0].meta[key] for key in kept if key in previous[0].meta})
            changes.pop("title", None)
            meta.update({key: value for key, value in changes.items() if value is not None})
            meta.update({rel: targets for rel, targets in relation_changes.items() if targets})
            page = self._write(
                path,
                meta,
                str(body or ""),
                editor=SYSTEM,
                stamp=self._supersede(*previous) if previous else self._next_stamp(path),
                previous=previous[0] if previous else None,
            )
            self._changed()
            if previous is None:
                self._page_event(
                    "page.created", f"The pipeline recorded {_what(page)}.", page, SYSTEM
                )
            else:
                self._page_event(
                    "page.updated", f"The pipeline updated {_what(page)}.", page, SYSTEM
                )
        return page

    # -- requests ----------------------------------------------------------------

    def _request_path(self, request_id: str) -> Path:
        return self.root / REQUESTS_DIR / f"{request_id}.json"

    def _request(self, request_id: Any) -> Request:
        data = None
        if isinstance(request_id, str) and re.fullmatch(r"rq-[0-9a-f]+", request_id):
            data = _read_json(self._request_path(request_id))
        if not isinstance(data, dict):
            raise NotFound(f"There is no request {request_id!r}.")
        return Request.from_dict(data)

    def add_request(
        self, text: str, *, target: str | None = None, role: str | None = None
    ) -> Request:
        """Leave an instruction for the agents, optionally about one page.

        ``role`` says who it is for: one of ``model.REQUEST_ROLES`` (``designer``),
        whose next phase is then given it and closes it, or nothing (``all``) for
        every agent: it is then shown to every phase that runs until the round's
        analysis has completed.  Who was given it is kept on it (``handled_in``).
        """
        text = str(text or "").strip()
        if not text:
            raise ProjectError("A request needs some text: what should the agents do?")
        addressee = request_role(role)
        _check_storable("The request", text)
        with self._lock:
            path = None
            if target:
                path = self.resolve(target)
                if path is None:
                    raise NotFound(f"No page matches {target!r}, the page the request is about.")
            request = Request(
                id=f"rq-{secrets.token_hex(4)}",
                text=text,
                target=path,
                created=self._created_after(self.requests(state=None)),
                role=addressee,
            )
            write_json(self._request_path(request.id), request.to_dict())
            whom = "the agents" if addressee == FOR_ALL else f"the {addressee}"
            self.emit(
                "request.created",
                _sentence(f"You asked {whom}: {_clip(text, 160)}"),
                refs=[f"{REQUESTS_DIR}/{request.id}.json", *([path] if path else [])],
                data={"request": request.id, "target": path, "for": addressee},
            )
        return request

    def requests(self, *, state: str | None = "open") -> list[Request]:
        """Requests in that state (``None`` for all), oldest first."""
        folder = self.root / REQUESTS_DIR
        found = []
        for file in sorted(folder.glob("rq-*.json")) if folder.is_dir() else []:
            data = _read_json(file)
            if isinstance(data, dict) and data.get("id"):
                request = Request.from_dict(data)
                if state is None or request.state == state:
                    found.append(request)
        return sorted(found, key=lambda request: (request.created, request.id))

    def note_request(
        self, request_id: str, *, round: int, phase: str, close: bool = False
    ) -> Request:
        """Note on a request that ``phase`` of ``round`` was given it, and return it.

        ``handled_in`` lists every phase that was, in order and each once.  With
        ``close`` the request is done as well: nobody else is to be shown it.  What
        else its file holds is kept.
        """
        with self._lock:
            request = self._request(request_id)
            entry = {"round": int(round), "phase": str(phase)}
            given = [*request.handled_in, *([] if entry in request.handled_in else [entry])]
            noted = replace(request, handled_in=given, state="done" if close else request.state)
            if noted != request:
                on_file = _read_json(self._request_path(request.id))
                extra = on_file if isinstance(on_file, dict) else {}
                write_json(self._request_path(request.id), {**extra, **noted.to_dict()})
        return noted

    def close_request(self, request_id: str, *, round: int, phase: str) -> None:
        """Mark a request as done with the phase that was given it last."""
        self.note_request(request_id, round=round, phase=phase, close=True)

    def delete_request(self, request_id: str) -> None:
        """Withdraw a request the person no longer wants handled."""
        with self._lock:
            request = self._request(request_id)
            self._request_path(request.id).unlink(missing_ok=True)

    # -- events ------------------------------------------------------------------

    def _last_seq(self, handle: Any, size: int) -> int:
        """The ``seq`` of the last complete event in the open stream of ``size`` bytes
        (0 when there is none)."""
        if size == self._tail[0]:
            return self._tail[1]
        window = 1 << 16
        while True:
            handle.seek(max(0, size - window))
            lines = handle.read().split(b"\n")
            # The first piece of a window may be the tail of a longer line.
            for raw in reversed(lines[1:] if window < size else lines):
                try:
                    seq = json.loads(raw)["seq"]
                except (ValueError, KeyError, TypeError, RecursionError):
                    continue
                if isinstance(seq, int):
                    return seq
            if window >= size:
                return 0
            window *= 8

    def emit(
        self,
        type: str,
        summary: str,
        *,
        phase: str = "",
        agent: str = "",
        refs: Iterable[str] = (),
        data: Mapping[str, Any] | None = None,
        round: int | None = None,
    ) -> dict[str, Any]:
        """Append one event to ``events.jsonl`` and return it.

        ``summary`` is one plain sentence for the activity feed.  ``seq`` is
        allocated under the project's lock: it grows by exactly one per event
        whichever process writes.  ``round`` and ``phase`` default to what the
        engine's ``state.json`` says is running.

        The values of this process's credentials (the rule of ``paper-orchestra
        doctor``) and the password or token in the address of the project's
        repository are blanked out of ``summary`` and of every text in ``data``
        before the event is written: an error that quotes a key is kept without
        it.  The stream is appended to as the regular file it is; an
        ``events.jsonl`` that is a symbolic link (or anything else) is refused
        with ``ProjectError``.
        """
        if type not in EVENT_TYPES:
            raise ProjectError(
                f"{type!r} is not an event type; the types are: {', '.join(EVENT_TYPES)}."
            )
        path = self.root / EVENTS_FILE
        secrets_known = self._secrets()
        with self._lock, _open_for_append(path) as handle:
            size = handle.seek(0, os.SEEK_END)
            event = {
                "seq": self._last_seq(handle, size) + 1,
                "ts": self._clock.now(),
                "type": type,
                "phase": phase or self._running_phase(),
                "round": self.current_round() if round is None else int(round),
                "agent": agent,
                # Blanked here, at the source: the stream is read by the app, by later
                # agent sessions and by whoever is given the project directory.
                "summary": doctor.scrub(_one_line(summary), secrets_known),
                "refs": [str(ref) for ref in refs],
                "data": doctor.scrub(dict(data or {}), secrets_known),
            }
            line = self._event_line(event)
            for secret in secrets_known:
                # Also where a value was not a string until it was written (a path, an
                # exception): nothing of the line that is kept holds a credential.
                if secret.encode("utf-8", errors="replace") in line:
                    event = doctor.scrub(json.loads(line), secrets_known)
                    line = self._event_line(event)
                    break
            if size:
                # A writer that died mid-line must not have the next event glued on.
                handle.seek(size - 1)
                if handle.read(1) != b"\n":
                    line = b"\n" + line
            handle.write(line)
            handle.flush()
            self._tail = (handle.tell(), event["seq"])
        return event

    @staticmethod
    def _event_line(event: Mapping[str, Any]) -> bytes:
        """One event as a line of the stream.  What cannot be written as it is (half of a
        surrogate pair in something a model or a job said) is written as an escape: an
        event that cannot be recorded would hide what happened."""
        try:
            text = json.dumps(event, ensure_ascii=False, default=str)
            return text.encode("utf-8") + b"\n"
        except UnicodeEncodeError:
            return json.dumps(event, ensure_ascii=True, default=str).encode("ascii") + b"\n"

    def _secrets(self) -> list[str]:
        """What must never be written into the event stream, longest first: the
        credentials in this process's environment and what the address of the
        project's repository carries."""
        found = set(doctor.secret_values(os.environ))
        given = self.settings.repo
        carried = repositories.userinfo(given.path) if given else None
        if carried:
            # git prints either form of a password that needed escaping in the URL.
            found.update(text for text in (carried, unquote(carried)) if len(text) >= 4)
        return sorted(found, key=len, reverse=True)

    def events(self, *, after: int = 0) -> list[dict[str, Any]]:
        """Events with ``seq`` greater than ``after``, in order."""
        path = self.root / EVENTS_FILE
        found: list[dict[str, Any]] = []
        if path.is_symlink():
            return []  # what a link points at is not this project's stream
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    if not raw.endswith(b"\n"):
                        break  # still being written
                    try:
                        event = json.loads(raw)
                    except (ValueError, RecursionError):
                        continue
                    seq = event.get("seq") if isinstance(event, dict) else None
                    if isinstance(seq, int) and seq > after:
                        found.append(event)
        except OSError:
            return []
        return found

    # -- graph -------------------------------------------------------------------

    def graph(self, *, include_rejected: bool = False) -> dict[str, Any]:
        """The wiki as nodes and edges.

        ``{"nodes": [{"id": path, "kind", "label", "title", "summary", "review",
        "state", "author", "round"}], "edges": [{"source": path, "target": path,
        "type": relation or "mentions"}], "dangling": [{"source", "target", "type"}]}``.
        Relation fields give typed edges, wiki links in a body give ``mentions``
        (left out where a relation already joins the two pages).  A reference to a
        page that does not exist is no edge; it is listed under ``dangling``.
        """
        everything = self._all_pages()
        resolver = _Resolver(self.wiki, everything)
        shown = [page for page in everything if include_rejected or page.review != "rejected"]
        visible = {page.path for page in shown}
        edges: list[dict[str, str]] = []
        dangling: list[dict[str, str]] = []
        for page in shown:
            joined: set[str] = set()
            links = [(name, t) for name, targets in page.relations.items() for t in targets]
            links += [(MENTIONS, target) for target in _body_links(page)]
            seen: set[tuple[str, str]] = set()
            for type, target in links:
                path = resolver.resolve(target)
                if path is None:
                    dangling.append({"source": page.path, "target": target, "type": type})
                    continue
                if path == page.path or path not in visible or (type, path) in seen:
                    continue
                if type == MENTIONS and path in joined:
                    continue  # a relation already says more about this pair
                seen.add((type, path))
                joined.add(path)
                edges.append({"source": page.path, "target": path, "type": type})
        nodes = [
            {
                "id": page.path,
                "kind": page.kind,
                "label": page.label,
                "title": page.title,
                "summary": page.summary,
                "review": page.review,
                "state": page.state,
                "author": page.author,
                "round": page.round,
            }
            for page in shown
        ]
        return {"nodes": nodes, "edges": edges, "dangling": dangling}

    # -- what agents are told, summaries, search ---------------------------------

    def index_text(
        self,
        for_role: str | None = None,
        *,
        session: str | None = None,
        max_chars: int = INDEX_MAX_CHARS,
    ) -> str:
        """The index injected into agent sessions, at most ``max_chars`` long.

        Lists every accepted page with its path, id, review, state, title and
        summary, grouped by kind.  Rejected pages appear by title and reason only.
        An unreviewed proposal is listed for nobody but the session that wrote it:
        pass that session's id as ``session``.  ``for_role`` adds a line saying
        where that role may create pages.  When the wiki outgrows the budget,
        summaries are shortened first, then the longest groups are cut to their
        latest pages, with a line saying how to find the rest.
        """
        role = role_of(role_author(for_role)) if for_role else None
        pages = self._all_pages()
        listed = [
            page
            for page in pages
            if page.review == "accepted"
            or (page.review == "proposed" and session and page.session == session)
        ]
        rejected = [page for page in pages if page.review == "rejected"]
        text = ""
        for summary_chars, per_group in ((140, 0), (60, 0), (0, 0), (0, 40), (0, 15), (0, 5)):
            text = self._render_index(listed, rejected, role, summary_chars, per_group)
            if len(text) <= max_chars:
                return text
        cut = text[:max_chars].rsplit("\n", 1)[0]
        return cut + "\n\n[index cut short: search the wiki for what is not listed]\n"

    def _render_index(
        self,
        listed: Sequence[ResearchPage],
        rejected: Sequence[ResearchPage],
        role: str | None,
        summary_chars: int,
        per_group: int,
    ) -> str:
        lines = [
            f"# Research wiki: {self.title}",
            "",
            "Pages are under /memories/wiki/. Each line reads: path [id, review, state] "
            "title: summary.",
        ]
        places = [f"{SECTIONS[kind]}/" for kind in creatable_kinds(role or "")]
        if places:
            lines.append(
                f"As the {role} you can create pages under: {', '.join(places)}. "
                "To change any other page, propose an edit."
            )
        if not listed:
            lines += ["", "_No pages yet._"]
        for kind in KINDS:
            group = [page for page in listed if page.kind == kind]
            if not group:
                continue
            lines += ["", f"## {SECTIONS[kind] or kind} ({len(group)})"]
            if per_group and len(group) > per_group:
                hidden = len(group) - per_group
                newest = sorted(group, key=lambda page: (page.created, _page_order(page)))
                kept = {page.path for page in newest[-per_group:]}
                group = [page for page in group if page.path in kept]
                lines.append(
                    f"- ... {hidden} earlier pages are not listed: view "
                    f"/memories/{section_path(kind)} or use wiki_search"
                )
            for page in group:
                review = "proposed in this session" if page.review == "proposed" else page.review
                tags = ", ".join(tag for tag in (page.id, review, page.state) if tag)
                summary = _clip(page.summary, summary_chars) if summary_chars else ""
                lines.append(
                    f"- {page.path.removeprefix(WIKI_DIR + '/')} [{tags}] {_clip(page.title, 100)}"
                    + (f": {summary}" if summary and summary != page.title else "")
                )
        if rejected:
            lines += ["", f"## rejected by the person ({len(rejected)}): title and reason only"]
            shown = rejected[-per_group:] if per_group else rejected
            for page in shown:
                reason = _clip(page.decision_reason, 200) or "no reason given"
                label = f"{page.kind} {page.id}" if page.id and page.kind != "source" else page.kind
                lines.append(f'- {label} "{_clip(page.title, 100)}": {reason}')
        return "\n".join(lines) + "\n"

    def counts(self) -> dict[str, Any]:
        """Numbers for summaries.

        ``{"pages": total, "by_kind": {kind: n}, "by_review": {review: n},
        "pending": {"total", "pages", "proposals", "by_kind": {kind: n}},
        "experiments": {state: n}, "requests": {"open": n}}``.  ``by_kind``,
        ``by_review`` and ``experiments`` name every kind, review and state, with
        zeros; rejected experiments are not counted on the board.
        """
        pages = self._all_pages()
        waiting = [page for page in pages if page.review == "proposed"]
        proposals = self.proposals(review="proposed")
        pending_kinds: dict[str, int] = {}
        for page in waiting:
            pending_kinds[page.kind] = pending_kinds.get(page.kind, 0) + 1
        experiments = [p for p in pages if p.kind == "experiment" and p.review != "rejected"]
        return {
            "pages": len(pages),
            "by_kind": {kind: sum(page.kind == kind for page in pages) for kind in KINDS},
            "by_review": {name: sum(page.review == name for page in pages) for name in REVIEWS},
            "pending": {
                "total": len(waiting) + len(proposals),
                "pages": len(waiting),
                "proposals": len(proposals),
                "by_kind": pending_kinds,
            },
            "experiments": {
                state: sum(page.state == state for page in experiments)
                for state in STATES["experiment"]
            },
            "requests": {"open": len(self.requests(state="open"))},
        }

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        review: str | Iterable[str] | None = "accepted",
        kind: str | Iterable[str] | None = None,
    ) -> list[SearchHit]:
        """Keyword search over the wiki's pages, best first.

        By default only accepted pages, which is what agents may build on; pass
        ``review=None`` to search everything (the person's view).
        """
        reviews, kinds = _as_filter(review), _as_filter(kind)
        hits = []
        pages = {page.path: page for page in self._all_pages()}
        for hit in self.wiki.search(query, k=100_000, prefix=WIKI_DIR):
            page = pages.get(hit.path)
            if page is None:
                continue
            if (reviews is None or page.review in reviews) and (
                kinds is None or page.kind in kinds
            ):
                hits.append(hit)
        return hits[: max(1, k)]

    # -- diagnosis ---------------------------------------------------------------

    def state_lost(self) -> str | None:
        """What to say when the engine's ``state.json`` is missing, cannot be read or
        holds no record of a phase although phases have run (their events are on file);
        ``None`` otherwise.  Such a project is not "not started"."""
        state = _read_json(self.root / STATE_FILE)
        if isinstance(state, dict) and isinstance(state.get("phases"), list) and state["phases"]:
            return None
        try:
            with (self.root / EVENTS_FILE).open("rb") as handle:
                ran = any(
                    b'"type": "phase.' in line or b'"type": "checkpoint"' in line for line in handle
                )
        except OSError:
            ran = False
        if not ran:
            return None
        return (
            "state.json is missing or cannot be read, although phases of this project have "
            "run. The project is not started over: where it stands is read from "
            "events.jsonl, and the engine writes the file again when it next works on it."
        )

    def problems(self) -> list[dict[str, str]]:
        """What was read leniently: ``[{"path": ..., "message": ...}]``, empty when clean.

        A project edited by hand opens whatever state it is in; this says what a
        person may want to repair: settings replaced by defaults (and a research
        cutoff that cannot be read, which stops the research until it is corrected),
        an engine state that was lost, pages with missing or unknown front matter,
        ids used twice, relations that point at nothing or at the wrong kind of
        page, proposal and request files that are not valid JSON.
        """
        found = [{"path": PROJECT_FILE, "message": message} for message in self._file_problems]
        lost = self.state_lost()
        if lost:
            found.append({"path": STATE_FILE, "message": lost})
        pages = self._all_pages()
        resolver = _Resolver(self.wiki, pages)
        if IDEA_PATH not in resolver.pages:
            found.append({"path": IDEA_PATH, "message": "the idea page is missing"})
        owners: dict[str, str] = {}
        for page in pages:
            messages = page_problems(page)
            if page.id and page.kind in ID_PREFIXES:
                first = owners.setdefault(page.id, page.path)
                if first != page.path:
                    messages.append(f"id {page.id} is also used by {first}")
            for name, targets in page.relations.items():
                relation = RELATIONS[name]
                for target in targets:
                    path = resolver.resolve(target)
                    if path is None:
                        messages.append(f"{name}: '{target}' does not match any page")
                    elif resolver.pages[path].kind not in relation.targets:
                        messages.append(
                            f"{name}: '{target}' is {with_article(resolver.pages[path].kind)}; "
                            f"{name} points at: {', '.join(relation.targets)}"
                        )
            found += [{"path": page.path, "message": message} for message in messages]
        for folder, prefix in ((PROPOSALS_DIR, "ep-"), (REQUESTS_DIR, "rq-")):
            directory = self.root / folder
            for file in sorted(directory.glob(f"{prefix}*.json")) if directory.is_dir() else []:
                if not isinstance(_read_json(file), dict):
                    found.append(
                        {
                            "path": file.relative_to(self.root).as_posix(),
                            "message": "not a JSON object; the file is ignored",
                        }
                    )
        return found
