"""What an agent session may see and change of a research project.

``ProjectMemoryBackend`` sits behind the provider's native memory tool, like
``memory.tools.WikiMemoryBackend`` does for the paper pipeline: the model sees a
``/memories`` directory and reads or edits files in it.  Here every operation
goes through :class:`~paper_orchestra.research.project.Project`, and the backend
decides what this one session, working in one role, is shown and allowed:

* it reads accepted pages, the proposals it wrote itself, and the person's
  attached files where they are text; a rejected page is shown as its title and
  the reason only; another session's unreviewed proposal does not exist for it;
* it creates pages of the kinds its role may create, as proposals: whatever the
  text says about review, author, round, id or state is replaced by what the
  project decides;
* it revises only pages it created in this session that no person has decided;
* everything else is a suggestion made with the ``propose_edit`` tool.

Errors are written for the model: they say what to do instead.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from paper_orchestra.llm.base import ToolError, ToolSpec

# Shared on purpose: a path must be vetted, and a file shown, exactly as the wiki's own
# memory backend does it.
from paper_orchestra.memory.tools import (
    FRONT_MATTER_LINE,
    MAX_VIEW_CHARS,
    VIRTUAL_ROOT,
    WikiMemoryBackend,
    front_matter_line_count,
    human_size,
    virtual_path,
)
from paper_orchestra.memory.wiki import INDEX_FILE, WIKI_DIR, WikiError, split_front_matter
from paper_orchestra.research.model import (
    AGENT_LOCKED_FIELDS,
    HUMAN,
    ID_PREFIXES,
    IDEA_PATH,
    KINDS,
    RECORD_KINDS,
    RELATIONS,
    SECTIONS,
    NotFound,
    ProjectError,
    ResearchPage,
    clean_relations,
    creatable_kinds,
    kind_for_path,
    moved_on,
    role_author,
    role_of,
    section_path,
    with_article,
)
from paper_orchestra.research.project import INPUTS_DIR, Project

__all__ = ["ProjectMemoryBackend", "propose_edit_tool", "search_tool"]

#: Largest attached file shown as text; anything bigger is data to analyse, not to read.
MAX_INPUT_BYTES = 2_000_000
#: Longest directory listing returned in one call.
MAX_LISTING_LINES = 300


def _own_number(kind: str) -> re.Pattern[str]:
    """The id a model put at the start of a file name: ``H7-`` in ``H7-warmup``."""
    return re.compile(rf"^{ID_PREFIXES[kind]}\d+(?:[-_ ]+|$)", re.IGNORECASE)


def _missing(path: Any) -> ToolError:
    return ToolError(f"The path {path} does not exist. Please provide a valid path.")


def _reason(page: ResearchPage) -> str:
    return f" ({page.decision_reason})" if page.decision_reason else ""


def _sections_for(role: str) -> str:
    """``/memories/wiki/hypotheses/, /memories/wiki/questions/`` for what a role may create."""
    places = [f"{virtual_path(section_path(kind))}/" for kind in creatable_kinds(role)]
    return ", ".join(places)


def _title_from(stem: str) -> str:
    words = re.sub(r"[-_]+", " ", stem).strip()
    return words[:1].upper() + words[1:] if words else "Untitled"


def _vet_targets(
    project: Project, relations: Mapping[str, list[str]], seen: Mapping[str, ResearchPage]
) -> None:
    """Raise ``ToolError`` unless every relation target is a page an agent can build on.

    ``seen`` holds the pages the agent can see, by path.  A target must be one of
    them, of a kind the relation points at, and not rejected by the person: a
    relation is a claim about the wiki, and one to a page that does not exist is
    an invented one.
    """
    for name, targets in relations.items():
        wanted = RELATIONS[name].targets
        for target in targets:
            found = project.resolve(target)
            page = seen.get(found or "")
            if page is None and found:
                # A record awaiting review exists as a fact, though it cannot be read yet.
                try:
                    candidate = project.page(found)
                except NotFound:
                    candidate = None
                page = candidate if candidate is not None and candidate.is_record else None
            if page is None:
                hint = (
                    "Sources are recorded by the verify_paper tool; cite only keys listed "
                    "under sources in the index."
                    if name == "cites"
                    else "Use an id such as H2 or a path listed in the index."
                )
                raise ToolError(f"{name}: '{target}' is not a page in this wiki. {hint}")
            if page.review == "rejected":
                raise ToolError(
                    f"{name}: {target} was rejected by the person{_reason(page)}; "
                    "do not build on it."
                )
            if page.kind not in wanted:
                raise ToolError(
                    f"{name}: {target} is {with_article(page.kind)}; {name} points at: "
                    f"{', '.join(wanted)}."
                )


def _numbered(path: str, text: str, view_range: tuple[int, int] | None) -> str:
    """``text`` as the memory tool shows a file: numbered lines, a range, a cut with a hint."""
    lines = text.split("\n")
    start, end = 1, len(lines)
    if view_range is not None:
        try:
            first, last = int(view_range[0]), int(view_range[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ToolError("view_range must be two integers: [start_line, end_line]") from exc
        start = max(1, first)
        end = len(lines) if last == -1 else min(last, len(lines))
        if start > len(lines) or end < start:
            raise ToolError(f"Invalid view_range {list(view_range)}: {path} has {len(lines)} lines")
    shown: list[str] = []
    used = 0
    cut_at: int | None = None
    for number in range(start, end + 1):
        line = f"{number:>6}\t{lines[number - 1]}"
        if used + len(line) + 1 > MAX_VIEW_CHARS and shown:
            cut_at = number
            break
        shown.append(line)
        used += len(line) + 1
    out = f"Here's the content of {path} with line numbers:\n" + "\n".join(shown)
    if cut_at is not None:
        out += (
            f"\n[Truncated at line {cut_at} of {len(lines)}. "
            f"Use view_range [{cut_at}, -1] to read on.]"
        )
    return out


class ProjectMemoryBackend:
    """``llm.base.MemoryBackend`` over a research project, for one agent session.

    ``created`` and ``changed`` list the project-relative paths of the pages this
    session created and created-or-revised, in order, for whoever runs the session.
    """

    def __init__(self, project: Project, role: str, round: int, session_id: str) -> None:
        """``role`` is the agent's role (``theorist`` or ``agent:theorist``); ``round`` is
        stamped on what it creates; ``session_id`` marks its pages as revisable by this
        session only (a resumed session passes the same id)."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ProjectError("An agent session needs a session id.")
        self.project = project
        self.author = role_author(role)
        self.role = role_of(self.author) or ""
        self.round = int(round)
        self.session_id = session_id.strip()
        self.created: list[str] = []
        self.changed: list[str] = []
        # Never written through: it only lends its vetting of the paths a model sends.
        self._paths = WikiMemoryBackend(project.wiki, write="none")

    # -- what this session sees ------------------------------------------------

    def _mode(self, page: ResearchPage) -> str:
        """``full``, ``stub`` (title and reason only) or ``none`` (does not exist for it)."""
        if page.review == "accepted":
            return "full"
        if page.review == "rejected":
            return "stub"
        # An unreviewed proposal is context for the session that wrote it, and for no other.
        mine = page.author == self.author and page.session == self.session_id
        return "full" if mine else "none"

    def _seen(self) -> dict[str, ResearchPage]:
        return {page.path: page for page in self.project.pages() if self._mode(page) != "none"}

    def _revisable(self, page: ResearchPage) -> bool:
        """Its own page, from this session, that no person has decided or acted on."""
        return (
            not page.is_record
            and page.author == self.author
            and page.session == self.session_id
            and page.review != "rejected"
            and not page.decided_by
            and not moved_on(page)
        )

    def _refuse_change(self, page: ResearchPage, path: str) -> None:
        """Raise the reason this session may not change ``page``, and what to do instead."""
        if self._revisable(page):
            return
        name = page.id or path
        if page.is_record:
            raise ToolError(
                f"{name} is a record the pipeline wrote from verified data; it cannot be "
                "edited. Say what should be different in a page of your own that refers to it."
            )
        if page.author == HUMAN:
            raise ToolError(f"{name} was written by the person; propose an edit with propose_edit.")
        if page.review == "rejected":
            raise ToolError(
                f"{name} was rejected by the person{_reason(page)}. It is kept as it was; "
                "write a new page that addresses the reason instead."
            )
        if page.decided_by:
            raise ToolError(
                f"{name} was accepted by the person; propose an edit with propose_edit."
            )
        if page.author != self.author:
            raise ToolError(
                f"{name} was written by {page.author}; propose an edit with propose_edit."
            )
        if moved_on(page):
            raise ToolError(
                f"{name} is {page.state}, so it stays as it was when that was decided; "
                "propose an edit with propose_edit."
            )
        raise ToolError(
            f"{name} is your page from an earlier session and can no longer be edited "
            "directly. Propose an edit with propose_edit, or write a new page."
        )

    # -- paths -----------------------------------------------------------------

    def _rel(self, path: Any) -> str:
        """Virtual path -> project-relative path (``""`` for the root), vetted."""
        return self._paths.split(path)[1]

    def _input_file(self, rel: str, path: str) -> Path:
        """The attached file ``rel`` names; never through a symbolic link."""
        base = self.project.inputs_dir.resolve()
        target = base.joinpath(*PurePosixPath(rel).parts[1:])
        if target.resolve() != target or not target.is_file():
            raise _missing(path)
        return target

    def _input_files(self) -> dict[str, int]:
        """Project-relative path -> size of every attached file (no hidden files, no links)."""
        base = self.project.inputs_dir
        files: dict[str, int] = {}
        for folder, dirs, names in os.walk(base):
            here = Path(folder)
            dirs[:] = sorted(
                d for d in dirs if not d.startswith(".") and not (here / d).is_symlink()
            )
            for name in sorted(names):
                file = here / name
                if name.startswith(".") or file.is_symlink() or not file.is_file():
                    continue
                files[f"{INPUTS_DIR}/{file.relative_to(base).as_posix()}"] = file.stat().st_size
        return files

    # -- view ------------------------------------------------------------------

    def _files(self) -> dict[str, int]:
        """Everything this session can open: path -> size."""
        files: dict[str, int] = {INDEX_FILE: len(self._index().encode("utf-8"))}
        for rel, page in self._seen().items():
            if self._mode(page) == "stub":
                files[rel] = len(self._stub(page).encode("utf-8"))
                continue
            try:
                files[rel] = (self.project.root / rel).stat().st_size
            except OSError:
                continue
        files.update(self._input_files())
        return files

    def _index(self) -> str:
        return self.project.index_text(for_role=self.role, session=self.session_id)

    def _listing(self, rel: str, path: str) -> str:
        base = PurePosixPath(rel).parts if rel else ()
        # The root, wiki/, inputs/ and the wiki's sections exist even while they are empty.
        known = len(base) <= 1 or (
            len(base) == 2 and base[0] == WIKI_DIR and base[1] in SECTIONS.values()
        )
        entries: dict[str, str] = {}
        for file_rel, size in self._files().items():
            parts = PurePosixPath(file_rel).parts
            if parts[: len(base)] != base:
                continue
            below = parts[len(base) :]
            for level in range(1, min(len(below), 2) + 1):
                entry = "/".join(base + below[:level])
                entries.setdefault(entry, human_size(size) if level == len(below) else "4.0K")
        if not base:
            for top in (WIKI_DIR, INPUTS_DIR):
                entries.setdefault(top, "4.0K")
        if not entries and not known:
            raise _missing(path)
        shown = virtual_path(rel)
        lines = [f"{size}\t{virtual_path(entry)}" for entry, size in sorted(entries.items())]
        if len(lines) > MAX_LISTING_LINES:
            more = len(lines) - MAX_LISTING_LINES
            lines = [*lines[:MAX_LISTING_LINES], f"[{more} more entries: list a subdirectory]"]
        header = (
            f"Here're the files and directories up to 2 levels deep in {shown}, "
            "excluding hidden items and node_modules:"
        )
        return "\n".join([header, f"4.0K\t{shown}", *lines])

    def _stub(self, page: ResearchPage) -> str:
        """All a session is told of a page the person turned down."""
        lines = [
            "This page was rejected by the person. Only its title and the reason are shown.",
            f"title: {page.title}",
            f"kind: {page.kind}",
        ]
        if page.id:
            lines.append(f"id: {page.id}")
        lines.append(f"reason: {page.decision_reason or 'no reason given'}")
        lines.append(
            "Do not build on it. If the thought is still worth pursuing, write a new page "
            "that addresses the reason."
        )
        return "\n".join(lines)

    def _input_text(self, rel: str, path: str) -> str:
        target = self._input_file(rel, path)
        size = target.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise ToolError(
                f"{path} is {human_size(size)}, too large to read here. It is data to "
                "analyse in an experiment rather than text to read; the wiki says what it holds."
            )
        data = target.read_bytes()
        try:
            if b"\x00" in data:
                raise UnicodeDecodeError("utf-8", data, 0, 1, "binary data")
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(
                f"{path} is not a text file and cannot be read here. Its content was given "
                "to the intake agent; read what the wiki says about it."
            ) from None

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str:
        rel = self._rel(path)
        top = PurePosixPath(rel).parts[:1]
        if rel == INDEX_FILE:
            return _numbered(path, self._index(), view_range)
        if top == (INPUTS_DIR,) and (self.project.root / rel).is_file():
            return _numbered(path, self._input_text(rel, path), view_range)
        if top == (WIKI_DIR,) and rel.endswith(".md"):
            seen = self._seen()
            page = seen.get(rel)
            if page is None:
                # A model often builds the path from the id alone.
                meant = self.project.resolve(PurePosixPath(rel).stem)
                if meant in seen:
                    raise ToolError(
                        f"The path {path} does not exist. The page you mean is at "
                        f"{virtual_path(meant)}."
                    )
                raise _missing(path)
            if self._mode(page) == "stub":
                return _numbered(path, self._stub(page), view_range)
            try:
                return _numbered(path, self.project.wiki.read_text(rel), view_range)
            except WikiError:
                raise _missing(path) from None
        if top not in ((), (WIKI_DIR,), (INPUTS_DIR,)):
            # Transcripts, notes, proposals, history and the project's own files are not
            # part of what an agent works from.
            raise _missing(path)
        return self._listing(rel, path)

    # -- create ----------------------------------------------------------------

    def _kind_to_create(self, rel: str, path: str) -> str:
        """The kind a new page at ``rel`` would have; raises when this role may not put one there."""
        mine = _sections_for(self.role) or "nowhere: this role only proposes edits"
        if rel == INDEX_FILE:
            raise ToolError(
                "MEMORY.md is the index, generated from the wiki; it cannot be written."
            )
        parts = PurePosixPath(rel).parts
        if parts[:1] == (INPUTS_DIR,):
            raise ToolError(
                f"{virtual_path(INPUTS_DIR)}/ holds the person's own files and is read-only. "
                f"Write what you learned from them in a page under: {mine}."
            )
        if rel == IDEA_PATH:
            raise ToolError("The idea page is the person's; propose an edit with propose_edit.")
        kind = kind_for_path(rel)
        if kind is None or len(parts) != 3 or not rel.endswith(".md"):
            raise ToolError(
                f"{path} is not a place for a page. As the {self.role} you can create "
                f"Markdown pages (.md) directly under: {mine}."
            )
        if kind in RECORD_KINDS:
            raise ToolError(
                f"{virtual_path(section_path(kind))}/ holds records that the pipeline writes "
                "from verified data (a source from the verify_paper tool, a result from a "
                f"run); they cannot be written by hand. You can create pages under: {mine}."
            )
        if kind not in creatable_kinds(self.role):
            raise ToolError(
                f"The {self.role} does not create {kind} pages. You can create pages under: "
                f"{mine}. To change another page, call propose_edit."
            )
        return kind

    def _relations_in(self, kind: str, meta: Mapping[str, Any]) -> dict[str, list[str]]:
        """The relation fields of ``meta``, each target checked against the wiki."""
        try:
            relations = clean_relations(kind, {n: meta[n] for n in RELATIONS if n in meta})
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        _vet_targets(self.project, relations, self._seen())
        return relations

    def _new_page(self, kind: str, rel: str, file_text: str) -> ResearchPage:
        meta, body = split_front_matter(file_text)
        stem = PurePosixPath(rel).stem
        # The number is the project's to give; of the name the model chose, the words are kept.
        words = _own_number(kind).sub("", stem) if kind in ID_PREFIXES else stem
        title = str(meta.get("title") or "").strip()
        if not title:
            heading = next((ln for ln in body.splitlines() if ln.startswith("#")), "")
            title = heading.lstrip("#").strip() or _title_from(words or stem)
        return self._propose(kind, title, body, meta, words or None)

    def _propose(
        self, kind: str, title: str, body: str, meta: Mapping[str, Any], slug: str | None
    ) -> ResearchPage:
        """Create this session's page from front matter as written; the project sets the rest."""
        if not str(body or "").strip():
            raise ToolError("The page has no text. Write its content below the front matter.")
        fields = {
            key: value
            for key, value in meta.items()
            if key not in AGENT_LOCKED_FIELDS and key not in RELATIONS and key != "title"
        }
        fields["session"] = self.session_id
        try:
            page = self.project.create_page(
                kind,
                title,
                body,
                author=self.author,
                fields=fields,
                relations=self._relations_in(kind, meta),
                round=self.round,
                slug=slug,
            )
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        self.created.append(page.path)
        self.changed.append(page.path)
        return page

    def create_page(
        self,
        kind: str,
        title: str,
        body: str,
        *,
        summary: str = "",
        fields: Mapping[str, Any] | None = None,
        relations: Mapping[str, Any] | None = None,
        slug: str | None = None,
    ) -> ResearchPage:
        """Create a page on this session's behalf, for a tool that writes pages in code.

        The rules are those of a page written through the memory tool: only the
        kinds the role may create, relations checked against what the session can
        build on, and review, author, round and state set by the project.  The page
        belongs to the session, which can then read and revise it.  Raises
        ``ToolError`` with what to do instead, as it is called from tool handlers.
        """
        if kind not in creatable_kinds(self.role):
            mine = _sections_for(self.role) or "nowhere: this role only proposes edits"
            raise ToolError(
                f"The {self.role} does not create {kind} pages. You can create pages under: "
                f"{mine}. To change another page, call propose_edit."
            )
        meta = {**(fields or {}), **(relations or {})}
        if summary:
            meta["summary"] = summary
        return self._propose(kind, title, body, meta, slug)

    def _ignored(self, meta: Mapping[str, Any], page: ResearchPage) -> str:
        """A note naming the project-set fields the model wrote something else into."""
        names = sorted(
            key for key in AGENT_LOCKED_FIELDS if key in meta and meta[key] != page.meta.get(key)
        )
        if not names:
            return ""
        return (
            f" These fields are set by the project and the person, not by the text of a page, "
            f"and were left as they are: {', '.join(names)}."
        )

    def create(self, path: str, file_text: str) -> str:
        if not isinstance(file_text, str):
            raise ToolError("file_text must be a string")
        rel = self._rel(path)
        if not rel:
            raise ToolError(f"{path} is a directory, not a file")
        existing = self._seen().get(rel)
        if existing is not None:
            self._refuse_change(existing, path)
            return self._revise(existing, file_text, f"File created successfully at: {path}")
        kind = self._kind_to_create(rel, path)
        page = self._new_page(kind, rel, file_text)
        where = virtual_path(page.path)
        status = (
            "It is a proposal: the person will accept, edit or reject it, and until they "
            "decide you can still revise it in this session."
            if page.review == "proposed"
            else f"It was accepted on creation, as this project does for {kind} pages."
        )
        known_as = f"It is {page.id}; refer to it by that id. " if page.id else ""
        return f"File created successfully at: {where}\n{known_as}{status}" + self._ignored(
            split_front_matter(file_text)[0], page
        )

    # -- revise ----------------------------------------------------------------

    def _own(self, path: str) -> tuple[ResearchPage, str]:
        """The page at ``path`` and its text, when this session may change it."""
        rel = self._rel(path)
        if not rel:
            raise ToolError(f"{path} is a directory, not a file")
        page = self._seen().get(rel)
        if page is None:
            if rel == INDEX_FILE or PurePosixPath(rel).parts[:1] == (INPUTS_DIR,):
                self._kind_to_create(rel, path)
            raise _missing(path)
        self._refuse_change(page, path)
        try:
            return page, self.project.wiki.read_text(rel)
        except WikiError:
            raise _missing(path) from None

    def _revise(self, page: ResearchPage, new_text: str, done: str) -> str:
        """Replace the text of this session's own page, keeping what the project sets."""
        meta, body = split_front_matter(new_text)
        if not body.strip():
            raise ToolError(
                "The page would have no text left. Keep its content below the front matter."
            )
        fields: dict[str, Any] = {
            key: value
            for key, value in meta.items()
            if key not in AGENT_LOCKED_FIELDS and key not in RELATIONS
        }
        # A field the new text no longer has was removed; the title never is.
        for key in page.meta:
            if key not in meta and key not in AGENT_LOCKED_FIELDS and key != "title":
                fields[key] = None
        relations = {name: [] for name in RELATIONS if name in page.meta}
        relations.update(self._relations_in(page.kind, meta))
        fields = {key: value for key, value in fields.items() if key not in RELATIONS}
        try:
            new = self.project.update_page(
                page.path, author=self.author, body=body, fields=fields, relations=relations
            )
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        ignored = self._ignored(meta, page)
        if new == page and ignored:
            raise ToolError(
                "No change was made." + ignored + " An experiment is approved, and a page "
                "accepted, only by the person."
            )
        if page.path not in self.changed:
            self.changed.append(page.path)
        return done + ignored

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        if not isinstance(old_str, str) or not old_str:
            raise ToolError("old_str must be a non-empty string")
        if not isinstance(new_str, str):
            new_str = ""
        page, text = self._own(path)
        count = text.count(old_str)
        if count == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim "
                f"in {path}."
            )
        if count > 1:
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in {path}. Please ensure it is unique."
            )
        return self._revise(
            page, text.replace(old_str, new_str, 1), "The memory file has been edited."
        )

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        if not isinstance(insert_text, str):
            raise ToolError("insert_text must be a string")
        if isinstance(insert_line, bool) or not isinstance(insert_line, int):
            raise ToolError("insert_line must be an integer")
        page, text = self._own(path)
        lines = text.split("\n")
        if not 0 <= insert_line <= len(lines):
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the "
                f"range of lines of the file: [0, {len(lines)}]"
            )
        new_lines = insert_text.rstrip("\n").split("\n")
        header = front_matter_line_count(lines)
        if 0 < insert_line < header and not all(
            FRONT_MATTER_LINE.match(line) for line in new_lines
        ):
            # Free text inside the front matter block would be dropped when the page is saved.
            raise ToolError(
                f"Line {insert_line} is inside the page's front matter (lines 1-{header}). "
                f"Insert body text at line {header} or later."
            )
        lines[insert_line:insert_line] = new_lines
        return self._revise(page, "\n".join(lines), f"The file {path} has been edited.")

    def delete(self, path: str) -> str:
        if not self._rel(path):
            raise ToolError(f"Cannot delete the {VIRTUAL_ROOT} directory itself")
        page, _ = self._own(path)
        try:
            self.project.delete_page(page.path, author=self.author)
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        if page.path in self.created:
            self.created.remove(page.path)
        return f"Successfully deleted {path}"

    def rename(self, old_path: str, new_path: str) -> str:
        self._rel(old_path)
        self._rel(new_path)
        raise ToolError(
            "Pages are not renamed: other pages refer to them by id and path. To change "
            "what a page is called, edit its `title:` line."
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _same_proposal(
    project: Project, target: str, author: str, body: Any, fields: Any
) -> str | None:
    """Id of an open proposal that already asks for exactly this, if there is one."""
    wanted = None if body is None else body.rstrip("\n")
    for proposal in project.proposals():
        theirs = None if proposal.body is None else proposal.body.rstrip("\n")
        asked = {**proposal.fields, **proposal.relations_add}
        if (proposal.target, proposal.author, theirs) == (
            target,
            author,
            wanted,
        ) and asked == fields:
            return proposal.id
    return None


def propose_edit_tool(project: Project, role: str, round: int) -> ToolSpec:
    """The tool an agent uses to suggest a change to a page it may not change itself.

    Inputs: ``target`` (an id such as H2, or a path), ``summary`` (one line),
    ``body`` (the full replacement text, optional) and ``fields`` (front-matter
    fields to set, optional; relation fields given there are added to the page's).
    """
    author = role_author(role)

    def handler(tool_input: dict[str, Any]) -> str:
        target, summary = tool_input.get("target"), tool_input.get("summary")
        body, fields = tool_input.get("body"), tool_input.get("fields")
        if not isinstance(target, str) or not target.strip():
            raise ToolError("target must name a page: an id such as H2, or its path")
        if not isinstance(summary, str) or not summary.strip():
            raise ToolError("summary must say in one line what the edit changes and why")
        if body is not None and (not isinstance(body, str) or not body.strip()):
            raise ToolError("body, when given, is the full new text of the page")
        if fields is not None and not isinstance(fields, dict):
            raise ToolError('fields, when given, is an object such as {"state": "supported"}')
        path = project.resolve(target)
        page = project.page(path) if path else None
        if page is not None and page.review == "proposed" and page.author == author:
            raise ToolError(
                f"{page.id or target} is your own proposal and still awaits the person's "
                "review. If you wrote it in this session, edit it with the memory tool; "
                "otherwise leave it as it is until they have decided."
            )
        if page is None or page.review == "proposed":
            # Another agent's unreviewed proposal does not exist for this one.
            raise ToolError(
                f"No page matches '{target}'. Use an id such as H2 or a path listed in the index."
            )
        if page.review == "rejected":
            raise ToolError(
                f"{page.id or target} was rejected by the person{_reason(page)}. It is kept as "
                "it was; write a new page that addresses the reason instead."
            )
        note = ""
        if body is not None:
            front, text = split_front_matter(body)
            if front:
                body = text
                note = (
                    " The front matter at the top of body was left out: changes to fields "
                    "go in `fields`."
                )
        fields = dict(fields or {})
        try:
            added = clean_relations(page.kind, {n: fields[n] for n in RELATIONS if n in fields})
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        # What this agent can build on: what is accepted or turned down, and its own proposals.
        usable = {
            other.path: other
            for other in project.pages()
            if other.review != "proposed" or other.author == author
        }
        _vet_targets(project, added, usable)
        known = _same_proposal(project, page.path, author, body, fields)
        if known:
            return f"This edit is already proposed as {known} and awaits the person's decision."
        try:
            proposal = project.propose_edit(
                page.path, author=author, summary=summary, body=body, fields=fields, round=round
            )
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        return (
            f"Proposed as {proposal.id}: {proposal.summary} "
            f"({page.id or virtual_path(page.path)} is unchanged until the person accepts "
            f"the edit; do not propose it again.){note}"
        )

    return ToolSpec(
        name="propose_edit",
        description=(
            "Suggest a change to a wiki page that you cannot edit yourself: a page the person "
            "wrote or accepted, or another agent's page. Nothing changes until the person "
            "accepts the suggestion, and they may edit or reject it, so say in the summary "
            "what changes and why. Give `body` to replace the page's whole text, `fields` to "
            "set front-matter fields (for example the state of a hypothesis that a finding "
            "now supports, or relation fields such as cites, which are added to the ones the "
            "page has), or both. Records (sources and results) cannot be edited, and the "
            "state of an experiment is not something to propose: only the person approves a "
            "run. To change a page you created in this session, edit it with the memory tool "
            "instead."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The page to change: its id (H2) or its /memories path.",
                },
                "summary": {
                    "type": "string",
                    "description": "One line: what the edit changes and why.",
                },
                "body": {
                    "type": "string",
                    "description": "The full new text of the page, without front matter.",
                },
                "fields": {
                    "type": "object",
                    "description": (
                        "Front-matter fields to set, for example "
                        '{"state": "supported", "cites": ["Smith2020Deep"]}.'
                    ),
                },
            },
            "required": ["target", "summary"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def search_tool(project: Project) -> ToolSpec:
    """``wiki_search`` for agents: keyword or verbatim search over the accepted pages."""

    def handler(tool_input: dict[str, Any]) -> str:
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        kind = tool_input.get("kind") or None
        if kind is not None and kind not in KINDS:
            raise ToolError(f"kind must be one of: {', '.join(KINDS)}")
        try:
            k = max(1, min(int(tool_input.get("k") or 8), 25))
        except (TypeError, ValueError) as exc:
            raise ToolError("k must be an integer") from exc
        if tool_input.get("exact"):
            accepted = {page.path for page in project.pages(review="accepted", kind=kind)}
            matches = [
                f"{virtual_path(rel)}:{number}: {line}"
                for rel, number, line in project.wiki.grep(query, k=10_000, prefix=WIKI_DIR)
                if rel in accepted
            ]
            return "\n".join(matches[:k]) if matches else "No line contains that text verbatim."
        hits = project.search(query, k=k, kind=kind)
        if not hits:
            return "No matching pages."
        return "\n".join(
            f"{number}. {virtual_path(hit.path)}: {hit.title}\n   {hit.snippet}"
            for number, hit in enumerate(hits, 1)
        )

    return ToolSpec(
        name="wiki_search",
        description=(
            "Search the research wiki by keyword. Returns the best-matching accepted pages "
            "with their /memories paths, titles and a matching line. Use it to find a page "
            "the index does not show in full, for example a source by topic or an earlier "
            "finding about a metric. It returns snippets only: open a page with the memory "
            "tool to read it. Keyword search treats 87.3 as the words 87 and 3, so set exact "
            "to true when you are checking a number or a quotation; that returns the "
            "matching lines with their line numbers. Pages awaiting the person's review and "
            "pages they rejected are not searched."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Words to look for."},
                "kind": {
                    "type": "string",
                    "enum": list(KINDS),
                    "description": "Only pages of this kind, for example 'source'.",
                },
                "k": {"type": "integer", "description": "Maximum number of results (default 8)."},
                "exact": {
                    "type": "boolean",
                    "description": "Match the query verbatim (for numbers and exact phrases).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )
