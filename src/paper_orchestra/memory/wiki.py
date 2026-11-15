"""The wiki: PaperOrchestra's memory abstraction layer.

Everything an agent remembers between sessions lives in one directory, and
every read or write of that memory goes through :class:`Wiki`.  The layout
follows the four layers of agent memory:

``transcripts/``
    Append-only event logs of each session (owned by ``TranscriptLog``).
``notes/``
    Salient observations written during a session.  One file per note, so
    concurrent writers never overwrite each other.
``wiki/``
    The representation layer: linked Markdown pages, one per entity (a paper,
    an experiment, a venue rule, a lesson).  ``wiki/sources/`` holds verbatim
    ground truth and is read-only to agents.
``MEMORY.md``
    The injected context: a small index loaded into every session.  It has a
    short, curated *standing notes* block and an auto-generated page catalog.

Pages are Markdown with a small ``key: value`` front matter block and
``[[section/page]]`` wikilinks.  Nothing here calls a model; consolidation
("dreaming") is layered on top in ``paper_orchestra.memory.dreaming``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

log = logging.getLogger(__name__)

INDEX_FILE = "MEMORY.md"
WIKI_DIR = "wiki"
NOTES_DIR = "notes"
CONSOLIDATED_DIR = "notes/consolidated"
TRANSCRIPTS_DIR = "transcripts"

#: Injected-context budget.  The index is loaded into every session, so it has
#: to stay small: wrong or bloated notes there propagate everywhere.
INDEX_MAX_LINES = 200
INDEX_MAX_BYTES = 25_000
#: Budget for the curated standing notes, which every session reads in full.
STANDING_MAX_LINES = 30
STANDING_MAX_CHARS = 3_000

STANDING_START = "<!-- standing:start -->"
STANDING_END = "<!-- standing:end -->"
CATALOG_START = "<!-- catalog:start -->"
CATALOG_END = "<!-- catalog:end -->"
_NO_STANDING = "_None yet._"

_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_KEY_VALUE = re.compile(r"^([A-Za-z_][\w\-]*)[ \t]*:[ \t]*(.*)$")
_LIST_ITEM = re.compile(r"^[ \t]*-[ \t]+(.*)$")
_WIKILINK = re.compile(r"\[\[([^\[\]|#\n]+)(#[^\[\]|\n]*)?(\|[^\[\]\n]*)?\]\]")
_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_PLAIN_SCALAR = re.compile(r"^[A-Za-z0-9_./()+\- ]*$")

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in",
        "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were",
        "which", "with", "we", "our", "their", "they", "can", "not", "no",
    }
)  # fmt: skip


class WikiError(ValueError):
    """Invalid path, read-only violation, missing page, or oversize write."""


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in '[{"' or value in ("true", "false", "null") or _NUMBER.match(value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def _dump_scalar(value: Any) -> str:
    if isinstance(value, str):
        plain = (
            value
            and value == value.strip()
            and _PLAIN_SCALAR.match(value)
            and value not in ("true", "false", "null")
            and not _NUMBER.match(value)
            and not value.startswith("-")
        )
        return value if plain else json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, default=str)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``text`` into ``(meta, body)``.

    Understands scalars, inline JSON lists/objects, and block lists of scalars
    (``key:`` followed by ``- item`` lines), which covers what the pipeline
    writes and what models usually produce.  Unparseable lines are ignored.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, Any] = {}
    pending_list: str | None = None
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = _LIST_ITEM.match(line)
        if item and pending_list is not None:
            meta[pending_list].append(_parse_scalar(item.group(1)))
            continue
        pair = _KEY_VALUE.match(line)
        if not pair:
            pending_list = None
            continue
        key, raw = pair.group(1), pair.group(2)
        if raw.strip() == "":
            meta[key] = []
            pending_list = key
        else:
            meta[key] = _parse_scalar(raw)
            pending_list = None
    for key, value in list(meta.items()):
        if value == []:
            meta[key] = ""
    return meta, text[match.end() :]


def join_front_matter(meta: dict[str, Any], body: str) -> str:
    if not meta:
        return body
    lines = ["---"]
    lines.extend(f"{key}: {_dump_scalar(value)}" for key, value in meta.items())
    lines.append("---")
    return "\n".join(lines) + "\n" + body.lstrip("\n")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def slugify(text: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "page"


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def link_for(path: str) -> str:
    """``wiki/literature/x.md`` -> ``literature/x`` (the wikilink target)."""
    target = path[:-3] if path.endswith(".md") else path
    if target.startswith(WIKI_DIR + "/"):
        target = target[len(WIKI_DIR) + 1 :]
    return target


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip() or None
    return None


def _first_sentence(body: str, limit: int = 140) -> str:
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "<!--", "---", "```", "|", ">")):
            return line[:limit]
    return ""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Page:
    path: str
    meta: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def title(self) -> str:
        title = self.meta.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
        return _first_heading(self.body) or PurePosixPath(self.path).stem

    @property
    def summary(self) -> str:
        summary = self.meta.get("summary")
        if isinstance(summary, str) and summary.strip():
            return " ".join(summary.split())
        return _first_sentence(self.body)

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind") or "")

    def links(self) -> list[str]:
        """Wikilink targets in the body, in order, without anchors or labels."""
        return [m.group(1).strip() for m in _WIKILINK.finditer(self.body)]

    def render(self) -> str:
        return join_front_matter(self.meta, self.body)


@dataclass(frozen=True)
class PageInfo:
    path: str
    title: str
    summary: str
    kind: str
    size: int
    updated: str


@dataclass(frozen=True)
class SearchHit:
    path: str
    title: str
    score: float
    snippet: str


@dataclass
class LintReport:
    broken_links: list[tuple[str, str]] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    oversize: list[tuple[str, int]] = field(default_factory=list)
    missing_summary: list[str] = field(default_factory=list)
    index_lines: int = 0
    index_bytes: int = 0
    index_over_budget: bool = False
    standing_lines: int = 0
    standing_over_budget: bool = False
    pending_notes: int = 0

    @property
    def ok(self) -> bool:
        return not (self.broken_links or self.oversize or self.index_over_budget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "broken_links": [list(pair) for pair in self.broken_links],
            "orphans": self.orphans,
            "oversize": [list(pair) for pair in self.oversize],
            "missing_summary": self.missing_summary,
            "index_lines": self.index_lines,
            "index_bytes": self.index_bytes,
            "index_over_budget": self.index_over_budget,
            "standing_lines": self.standing_lines,
            "standing_over_budget": self.standing_over_budget,
            "pending_notes": self.pending_notes,
        }


@dataclass
class _Cached:
    mtime_ns: int
    size: int
    page: Page
    tokens: Counter[str]
    length: int


# ---------------------------------------------------------------------------
# Wiki
# ---------------------------------------------------------------------------


class Wiki:
    """A directory of linked Markdown pages plus notes and an injected index."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "project",
        description: str = "",
        read_only: Sequence[str] = ("wiki/sources/",),
        max_page_bytes: int = 200_000,
        soft_page_bytes: int = 40_000,
        index_max_lines: int = INDEX_MAX_LINES,
        index_max_bytes: int = INDEX_MAX_BYTES,
        standing_max_lines: int = STANDING_MAX_LINES,
        standing_max_chars: int = STANDING_MAX_CHARS,
        collapse_over: int = 40,
        always_collapse: Sequence[str] = (),
        clock: Callable[[], datetime] | None = None,
        create: bool = True,
    ) -> None:
        """``create=False`` opens an existing memory directory without touching it (viewers)."""
        self.root = Path(root)
        self.name = name
        self.description = description
        self.read_only = tuple(read_only)
        self.max_page_bytes = max_page_bytes
        self.soft_page_bytes = soft_page_bytes
        self.index_max_lines = index_max_lines
        self.index_max_bytes = index_max_bytes
        self.standing_max_lines = standing_max_lines
        self.standing_max_chars = standing_max_chars
        #: Sections with more pages than this are listed on their own index page instead of
        #: in MEMORY.md: the index should say what exists and where, not enumerate it.
        self.collapse_over = collapse_over
        self.always_collapse = tuple(always_collapse)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._cache: dict[str, _Cached] = {}
        if create:
            for sub in (WIKI_DIR, NOTES_DIR, TRANSCRIPTS_DIR):
                (self.root / sub).mkdir(parents=True, exist_ok=True)
            if not (self.root / INDEX_FILE).exists():
                self.rebuild_index()

    # -- paths ---------------------------------------------------------------

    def normalize(self, rel: str) -> str:
        """Validate a memory-relative path and return its canonical posix form.

        Only ``MEMORY.md`` and Markdown files under ``wiki/`` and ``notes/`` are
        addressable.  Rejects absolute paths, ``..``, hidden files and anything
        that would resolve outside the memory root (including via symlinks).
        """
        if not isinstance(rel, str) or not rel.strip():
            raise WikiError("empty path")
        if "\x00" in rel or "\\" in rel:
            raise WikiError(f"invalid characters in path: {rel!r}")
        rel = rel.strip()
        if rel.startswith("/"):
            raise WikiError(f"path must be relative to the memory root: {rel!r}")
        parts = PurePosixPath(rel).parts
        if not parts or any(p in ("..", ".") or p.startswith(".") for p in parts):
            raise WikiError(f"path may not contain '..' or hidden segments: {rel!r}")
        canonical = "/".join(parts)
        if canonical != INDEX_FILE:
            if parts[0] not in (WIKI_DIR, NOTES_DIR):
                raise WikiError(
                    f"only {INDEX_FILE}, {WIKI_DIR}/ and {NOTES_DIR}/ are addressable: {rel!r}"
                )
            if len(parts) < 2:
                raise WikiError(f"{rel!r} is a directory, not a page")
            if not canonical.endswith(".md"):
                raise WikiError(f"pages are Markdown files ending in .md: {rel!r}")
        target = (self.root / canonical).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise WikiError(f"path escapes the memory root: {rel!r}")
        return canonical

    def is_read_only(self, rel: str) -> bool:
        return any(rel.startswith(prefix) for prefix in self.read_only)

    def _abs(self, rel: str) -> Path:
        return self.root / rel

    def _now(self) -> str:
        return self._clock().strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- reading -------------------------------------------------------------

    def exists(self, rel: str) -> bool:
        try:
            return self._abs(self.normalize(rel)).is_file()
        except WikiError:
            return False

    def read_text(self, rel: str) -> str:
        rel = self.normalize(rel)
        path = self._abs(rel)
        if not path.is_file():
            raise WikiError(f"no such page: {rel}")
        return path.read_text(encoding="utf-8")

    def _load(self, rel: str) -> _Cached:
        path = self._abs(rel)
        stat = path.stat()
        with self._lock:
            cached = self._cache.get(rel)
            if cached and cached.mtime_ns == stat.st_mtime_ns and cached.size == stat.st_size:
                return cached
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        page = Page(path=rel, meta=meta, body=body)
        tokens: Counter[str] = Counter()
        for token in tokenize(page.title) + tokenize(link_for(rel)):
            tokens[token] += 3
        explicit_summary = page.meta.get("summary")
        if isinstance(explicit_summary, str):
            for token in tokenize(explicit_summary):
                tokens[token] += 2
        tokens.update(tokenize(body))
        cached = _Cached(stat.st_mtime_ns, stat.st_size, page, tokens, sum(tokens.values()))
        with self._lock:
            self._cache[rel] = cached
        return cached

    def read(self, rel: str) -> Page:
        rel = self.normalize(rel)
        if not self._abs(rel).is_file():
            raise WikiError(f"no such page: {rel}")
        cached = self._load(rel)
        return Page(path=rel, meta=dict(cached.page.meta), body=cached.page.body)

    def _iter_paths(self, prefix: str = "") -> Iterable[str]:
        prefix = prefix.strip("/")
        root = self.root.resolve()
        tops = (WIKI_DIR, NOTES_DIR) if not prefix else (prefix.split("/")[0],)
        for top in tops:
            base = self.root / top
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.md")):
                rel = path.relative_to(self.root).as_posix()
                if any(part.startswith(".") for part in PurePosixPath(rel).parts):
                    continue
                if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
                    continue
                # A symlink could point outside the memory root; never read through one.
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    continue
                try:
                    self._load(rel)
                except (OSError, UnicodeDecodeError) as exc:
                    # Deleted since the scan, or not text: skip it rather than fail every
                    # listing, search and index rebuild.
                    log.warning("skipping unreadable memory page %s: %s", rel, exc)
                    continue
                yield rel

    def list(self, prefix: str = WIKI_DIR) -> list[PageInfo]:
        """Pages under ``prefix`` (default: the wiki layer), sorted by path."""
        infos = []
        for rel in self._iter_paths(prefix):
            cached = self._load(rel)
            page = cached.page
            infos.append(
                PageInfo(
                    path=rel,
                    title=page.title,
                    summary=page.summary,
                    kind=page.kind,
                    size=cached.size,
                    updated=str(page.meta.get("updated") or ""),
                )
            )
        return infos

    # -- writing -------------------------------------------------------------

    def _check_writable(self, rel: str, force: bool) -> None:
        if not force and self.is_read_only(rel):
            raise WikiError(f"{rel} is read-only ground truth and cannot be changed")

    def _atomic_write(self, rel: str, text: str) -> None:
        data = text.encode("utf-8")
        # The index has its own budget (enforced when it is injected), not the page limit.
        if rel != INDEX_FILE and len(data) > self.max_page_bytes:
            raise WikiError(
                f"{rel} would be {len(data)} bytes; the limit is {self.max_page_bytes}. "
                "Split it into several linked pages."
            )
        path = self._abs(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        with self._lock:
            self._cache.pop(rel, None)

    def write_text(self, rel: str, text: str, *, author: str = "", force: bool = False) -> Page:
        """Write a page from raw text (front matter included, if any)."""
        rel = self.normalize(rel)
        self._check_writable(rel, force)
        if rel == INDEX_FILE:
            self._atomic_write(rel, text)
            return Page(path=rel, body=text)
        meta, body = split_front_matter(text)
        return self.write(rel, body, meta, author=author, force=force)

    def write(
        self,
        rel: str,
        body: str,
        meta: dict[str, Any] | None = None,
        *,
        author: str = "",
        force: bool = False,
    ) -> Page:
        """Create or replace a page.  ``updated`` (and ``author``) are stamped."""
        rel = self.normalize(rel)
        if rel == INDEX_FILE:
            raise WikiError(f"use write_text() or rebuild_index() for {INDEX_FILE}")
        self._check_writable(rel, force)
        meta = dict(meta or {})
        meta["updated"] = self._now()
        if author:
            meta["author"] = author
        page = Page(path=rel, meta=meta, body=body if body.endswith("\n") else body + "\n")
        with self._lock:
            self._atomic_write(rel, page.render())
        return page

    def append(self, rel: str, text: str, *, author: str = "", force: bool = False) -> Page:
        rel = self.normalize(rel)
        with self._lock:
            if self._abs(rel).is_file():
                page = self.read(rel)
                body = page.body.rstrip("\n") + "\n\n" + text.strip("\n") + "\n"
                return self.write(rel, body, page.meta, author=author, force=force)
            return self.write(rel, text, author=author, force=force)

    def delete(self, rel: str, *, force: bool = False) -> None:
        rel = self.normalize(rel)
        if rel == INDEX_FILE:
            raise WikiError(f"{INDEX_FILE} cannot be deleted")
        self._check_writable(rel, force)
        path = self._abs(rel)
        if not path.is_file():
            raise WikiError(f"no such page: {rel}")
        with self._lock:
            path.unlink()
            self._cache.pop(rel, None)

    def rename(self, old: str, new: str, *, force: bool = False) -> None:
        old, new = self.normalize(old), self.normalize(new)
        if INDEX_FILE in (old, new):
            raise WikiError(f"{INDEX_FILE} cannot be renamed")
        self._check_writable(old, force)
        self._check_writable(new, force)
        src, dst = self._abs(old), self._abs(new)
        if not src.is_file():
            raise WikiError(f"no such page: {old}")
        if dst.exists():
            raise WikiError(f"{new} already exists")
        with self._lock:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            self._cache.pop(old, None)
            self._cache.pop(new, None)

    # -- notes ---------------------------------------------------------------

    def add_note(
        self,
        text: str,
        *,
        stage: str = "",
        agent: str = "",
        kind: str = "observation",
        title: str | None = None,
    ) -> str:
        """Record a salient observation as its own file and return its path.

        Notes are append-only and uniquely named, so parallel agents and
        parallel runs sharing a wiki can all write without losing each other's
        facts.  A later dream verifies them and folds them into pages.
        """
        stamp = self._clock().strftime("%Y%m%dT%H%M%SZ")
        first_line = text.strip().splitlines()[0] if text.strip() else "note"
        slug = slugify(title or first_line, 40)
        rel = f"{NOTES_DIR}/{stamp}-{slugify(stage or 'run', 24)}-{slug}-{secrets.token_hex(2)}.md"
        meta = {"kind": kind, "stage": stage, "agent": agent}
        if title:
            meta["title"] = title
        self.write(rel, text, {k: v for k, v in meta.items() if v})
        return rel

    def pending_notes(self) -> list[PageInfo]:
        """Notes not yet consolidated by a dream, oldest first."""
        return [
            info
            for info in self.list(NOTES_DIR)
            if not info.path.startswith(CONSOLIDATED_DIR + "/")
        ]

    def archive_notes(self, paths: Iterable[str], dream_id: str) -> list[str]:
        """Move consolidated notes out of the pending set; returns new paths."""
        moved = []
        for rel in paths:
            rel = self.normalize(rel)
            if not rel.startswith(NOTES_DIR + "/") or rel.startswith(CONSOLIDATED_DIR + "/"):
                continue
            if not self._abs(rel).is_file():
                continue
            new = f"{CONSOLIDATED_DIR}/{slugify(dream_id, 40)}/{PurePosixPath(rel).name}"
            if self._abs(new).exists():
                new = new[:-3] + f"-{secrets.token_hex(2)}.md"
            self.rename(rel, new, force=True)
            moved.append(new)
        return moved

    # -- links ---------------------------------------------------------------

    def resolve_link(self, target: str) -> str | None:
        """Map a wikilink target to an existing page path, or ``None``."""
        target = target.strip().strip("/")
        if not target:
            return None
        if target in ("MEMORY", INDEX_FILE):
            return INDEX_FILE
        if target.endswith(".md"):
            target = target[:-3]
        candidates = [f"{target}.md"]
        if not target.startswith((WIKI_DIR + "/", NOTES_DIR + "/")):
            candidates.insert(0, f"{WIKI_DIR}/{target}.md")
        for candidate in candidates:
            try:
                if self._abs(self.normalize(candidate)).is_file():
                    return candidate
            except WikiError:
                continue
        return None

    def _guess_link(self, target: str, paths: Sequence[str]) -> str | None:
        """Find the unique page a broken link most plausibly meant."""
        wanted = slugify(PurePosixPath(target.strip().strip("/")).name)
        by_stem = [p for p in paths if slugify(PurePosixPath(p).stem) == wanted]
        if len(by_stem) == 1:
            return by_stem[0]
        by_title = [p for p in paths if slugify(self._load(p).page.title) == wanted]
        return by_title[0] if len(by_title) == 1 else None

    def repair_links(self) -> list[tuple[str, str, str]]:
        """Repoint broken links that have exactly one plausible target.

        Returns ``(page, old_target, new_target)`` for every rewrite.  Read-only
        pages are left alone.
        """
        repairs: list[tuple[str, str, str]] = []
        with self._lock:
            paths = list(self._iter_paths(WIKI_DIR))
            for rel in list(self._iter_paths()):
                if self.is_read_only(rel) or rel.startswith(CONSOLIDATED_DIR + "/"):
                    continue
                page = self.read(rel)
                changed = False

                def fix(match: re.Match[str], rel: str = rel) -> str:
                    nonlocal changed
                    target = match.group(1).strip()
                    if self.resolve_link(target):
                        return match.group(0)
                    guess = self._guess_link(target, paths)
                    if not guess:
                        return match.group(0)
                    changed = True
                    new_target = link_for(guess)
                    repairs.append((rel, target, new_target))
                    return f"[[{new_target}{match.group(2) or ''}{match.group(3) or ''}]]"

                body = _WIKILINK.sub(fix, page.body)
                if changed:
                    self.write(rel, body, page.meta, author=str(page.meta.get("author") or ""))
        return repairs

    # -- search --------------------------------------------------------------

    def search(self, query: str, *, k: int = 8, prefix: str = "") -> list[SearchHit]:
        """BM25 search over pages and pending notes.

        Title and link-path terms are weighted above the summary, which is
        weighted above the body.  ``prefix`` narrows the search (for example
        ``wiki/literature``).
        """
        terms = tokenize(query)
        if not terms:
            return []
        docs = [
            self._load(rel)
            for rel in self._iter_paths(prefix)
            if not rel.startswith(CONSOLIDATED_DIR + "/")
        ]
        if not docs:
            return []
        n_docs = len(docs)
        avg_len = sum(d.length for d in docs) / n_docs or 1.0
        doc_freq = Counter(term for d in docs for term in set(terms) if term in d.tokens)
        k1, b = 1.5, 0.75
        scored: list[tuple[float, _Cached]] = []
        for doc in docs:
            score = 0.0
            for term in set(terms):
                tf = doc.tokens.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc.length / avg_len))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1].page.path))
        return [
            SearchHit(
                path=doc.page.path,
                title=doc.page.title,
                score=round(score, 4),
                snippet=_snippet(doc.page, terms),
            )
            for score, doc in scored[: max(1, k)]
        ]

    def grep(self, needle: str, *, k: int = 20, prefix: str = "") -> list[tuple[str, int, str]]:
        """Exact (case-insensitive, whitespace-normalised) substring search.

        Returns ``(path, line_number, line)`` for matching lines.  Keyword
        search splits ``87.3`` into tokens, so checking a number or an exact
        phrase needs this instead.
        """
        wanted = " ".join(needle.lower().split())
        if not wanted:
            return []
        matches: list[tuple[str, int, str]] = []
        for rel in self._iter_paths(prefix):
            if rel.startswith(CONSOLIDATED_DIR + "/"):
                continue
            for number, line in enumerate(self._load(rel).page.render().split("\n"), 1):
                if wanted in " ".join(line.lower().split()):
                    shown = line.strip()
                    matches.append((rel, number, shown if len(shown) <= 300 else shown[:299] + "…"))
                    if len(matches) >= k:
                        return matches
        return matches

    # -- index ---------------------------------------------------------------

    def standing_notes(self) -> str:
        """The curated block of ``MEMORY.md`` (durable rules, kept short)."""
        path = self.root / INDEX_FILE
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
        if STANDING_START in text and STANDING_END in text:
            text = text.split(STANDING_START, 1)[1].split(STANDING_END, 1)[0]
        elif CATALOG_START in text:
            # Markers were edited away: keep whatever is not the generated catalog.
            text = text.split(CATALOG_START, 1)[0]
        boilerplate = ("# Memory index", "## Standing notes", _NO_STANDING, STANDING_START)
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith(boilerplate)]
        if self.description:
            lines = [ln for ln in lines if ln.strip() != self.description.strip()]
        return "\n".join(lines).strip()

    def set_standing_notes(self, text: str) -> None:
        self._write_index(text.strip())

    def rebuild_index(self) -> str:
        """Regenerate the page catalog in ``MEMORY.md``, keeping standing notes.

        The catalog lists every wiki page with a one-line summary.  When that
        would blow the injected-context budget, the largest sections collapse
        to a single line that points at a generated ``<section>/index`` page.
        """
        return self._write_index(self.standing_notes())

    def _write_index(self, standing: str) -> str:
        with self._lock:
            groups: dict[str, list[PageInfo]] = {}
            for info in self.list(WIKI_DIR):
                parts = PurePosixPath(info.path).parts
                if parts[-1] == "index.md" and self._load(info.path).page.meta.get("auto"):
                    continue
                group = parts[1] if len(parts) > 2 else "(top level)"
                groups.setdefault(group, []).append(info)

            header = [f"# Memory index: {self.name}", ""]
            if self.description:
                header += [self.description, ""]
            header += [STANDING_START, "## Standing notes", ""]
            header += [standing or _NO_STANDING, STANDING_END, ""]
            fixed_lines = len(header) + 4
            fixed_bytes = len("\n".join(header).encode()) + 64

            collapsed: set[str] = {
                group
                for group, infos in groups.items()
                if group != "(top level)"
                and (group in self.always_collapse or len(infos) > self.collapse_over)
            }

            def render_catalog() -> list[str]:
                lines = [CATALOG_START, "## Pages", ""]
                if not groups:
                    lines.append("_No pages yet._")
                for group in sorted(groups):
                    infos = groups[group]
                    if group in collapsed:
                        lines.append(
                            f"- **{group}/**: {len(infos)} pages, listed in [[{group}/index]]"
                        )
                        continue
                    lines.append(f"### {group} ({len(infos)})")
                    for info in infos:
                        summary = (info.summary or info.title)[:140]
                        lines.append(f"- [[{link_for(info.path)}]]: {summary}")
                lines.append(CATALOG_END)
                return lines

            catalog = render_catalog()
            while (
                len(catalog) + fixed_lines > self.index_max_lines
                or len("\n".join(catalog).encode()) + fixed_bytes > self.index_max_bytes
            ):
                open_groups = [g for g in groups if g not in collapsed and g != "(top level)"]
                if not open_groups:
                    break
                collapsed.add(max(open_groups, key=lambda g: (len(groups[g]), g)))
                catalog = render_catalog()

            for group, infos in groups.items():
                index_rel = f"{WIKI_DIR}/{group}/index.md"
                if group in collapsed:
                    self.write(
                        index_rel,
                        self._section_listing(group, infos),
                        {
                            "title": f"{group} index",
                            "summary": f"Every page under {group}/",
                            "kind": "index",
                            "auto": True,
                        },
                        force=True,
                    )
                elif (
                    group != "(top level)"
                    and self._abs(index_rel).is_file()
                    and self._load(index_rel).page.meta.get("auto")
                ):
                    self.delete(index_rel, force=True)

            text = "\n".join(header + catalog) + "\n"
            self._atomic_write(INDEX_FILE, text)
            return text

    def _section_listing(self, group: str, infos: Sequence[PageInfo]) -> str:
        """Body of a generated ``<section>/index`` page, kept within the page size limit."""
        budget = self.max_page_bytes - 2_000
        for summary_chars in (200, 80, 0):
            lines = [f"# {group}: all pages", ""]
            for info in infos:
                summary = (info.summary or info.title)[:summary_chars]
                lines.append(f"- [[{link_for(info.path)}]]" + (f": {summary}" if summary else ""))
            body = "\n".join(lines)
            if len(body.encode("utf-8")) <= budget:
                return body
        kept: list[str] = []
        size = 0
        for line in lines:
            size += len(line.encode("utf-8")) + 1
            if size > budget:
                break
            kept.append(line)
        kept.append(f"- ... and {len(lines) - len(kept)} more: search the wiki to find them")
        return "\n".join(kept)

    def injected_context(self) -> str:
        """The text loaded into every session: ``MEMORY.md``, hard-capped."""
        path = self.root / INDEX_FILE
        text = path.read_text(encoding="utf-8") if path.is_file() else self.rebuild_index()
        lines = text.splitlines()
        truncated = len(lines) > self.index_max_lines
        lines = lines[: self.index_max_lines]
        out = "\n".join(lines)
        encoded = out.encode("utf-8")
        if len(encoded) > self.index_max_bytes:
            out = encoded[: self.index_max_bytes].decode("utf-8", errors="ignore")
            out = out.rsplit("\n", 1)[0]
            truncated = True
        if truncated:
            out += "\n\n[index truncated: search the wiki for anything not listed]"
        return out + "\n"

    # -- maintenance ---------------------------------------------------------

    def lint(self) -> LintReport:
        report = LintReport()
        wiki_paths = list(self._iter_paths(WIKI_DIR))
        inbound: Counter[str] = Counter()
        for rel in wiki_paths:
            cached = self._load(rel)
            page = cached.page
            if cached.size > self.soft_page_bytes:
                report.oversize.append((rel, cached.size))
            auto = bool(page.meta.get("auto"))
            if not auto and not str(page.meta.get("summary") or "").strip():
                report.missing_summary.append(rel)
            for target in page.links():
                resolved = self.resolve_link(target)
                if resolved is None:
                    report.broken_links.append((rel, target))
                elif resolved != rel and not auto:
                    inbound[resolved] += 1
        report.orphans = [
            rel
            for rel in wiki_paths
            if not inbound[rel]
            and not self._load(rel).page.meta.get("auto")
            and not self.is_read_only(rel)
        ]
        index = self.root / INDEX_FILE
        if index.is_file():
            text = index.read_text(encoding="utf-8")
            report.index_lines = len(text.splitlines())
            report.index_bytes = len(text.encode("utf-8"))
            standing = self.standing_notes()
            report.standing_lines = len([ln for ln in standing.splitlines() if ln.strip()])
            report.standing_over_budget = (
                report.standing_lines > self.standing_max_lines
                or len(standing) > self.standing_max_chars
            )
            report.index_over_budget = (
                report.index_lines > self.index_max_lines
                or report.index_bytes > self.index_max_bytes
            )
        report.pending_notes = len(self.pending_notes())
        return report

    def stats(self) -> dict[str, Any]:
        pages = self.list(WIKI_DIR)
        return {
            "name": self.name,
            "root": str(self.root),
            "pages": len(pages),
            "bytes": sum(p.size for p in pages),
            "pending_notes": len(self.pending_notes()),
        }


def _snippet(page: Page, terms: Sequence[str], width: int = 240) -> str:
    wanted = set(terms)
    for line in page.body.splitlines():
        if wanted.intersection(tokenize(line)):
            line = " ".join(line.split())
            return line if len(line) <= width else line[: width - 1] + "…"
    return page.summary[:width]
