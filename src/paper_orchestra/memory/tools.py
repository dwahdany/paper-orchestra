"""Tools that expose the wiki to a model.

``WikiMemoryBackend`` sits behind the provider's native memory tool: the model
sees a ``/memories`` directory and reads or edits files in it, and every one of
those operations is mapped onto :class:`~paper_orchestra.memory.wiki.Wiki`
(which owns path validation, read-only sources and size limits).  Search over
pages and transcripts is offered as two ordinary tools.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import PurePosixPath
from typing import Any, Literal

from paper_orchestra.llm.base import ToolError, ToolSpec
from paper_orchestra.memory.transcript import TranscriptLog
from paper_orchestra.memory.wiki import (
    CONSOLIDATED_DIR,
    INDEX_FILE,
    NOTES_DIR,
    WIKI_DIR,
    Wiki,
    WikiError,
)

VIRTUAL_ROOT = "/memories"
LAB_PREFIX = "lab"
#: Longest file view returned in one call; the model pages with ``view_range``.
MAX_VIEW_CHARS = 16_000

WriteScope = Literal["none", "notes", "all"]
LabWriteScope = Literal["none", "all"]


_FRONT_MATTER_LINE = re.compile(r"^([A-Za-z_][\w\-]*[ \t]*:.*|[ \t]*-[ \t]+.*|[ \t]*#.*|[ \t]*)$")


def _front_matter_line_count(lines: list[str]) -> int:
    """Number of leading lines taken by a ``---`` front matter block (0 if there is none)."""
    if not lines or lines[0].strip() != "---":
        return 0
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return index + 1
    return 0


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}K"
    return f"{size / (1024 * 1024):.1f}M"


def virtual_path(rel: str, *, lab: bool = False) -> str:
    """Memory-relative path -> the path the model sees."""
    prefix = f"{VIRTUAL_ROOT}/{LAB_PREFIX}" if lab else VIRTUAL_ROOT
    return f"{prefix}/{rel}" if rel else prefix


class WikiMemoryBackend:
    """Implements ``llm.base.MemoryBackend`` over a project wiki and an optional lab wiki."""

    def __init__(
        self,
        project: Wiki,
        lab: Wiki | None = None,
        *,
        write: WriteScope = "notes",
        lab_write: LabWriteScope = "none",
        author: str = "agent",
        on_write: Callable[[str, str], None] | None = None,
        on_read: Callable[[str, str, bool], None] | None = None,
        guard: Callable[[str, str], str | None] | None = None,
    ) -> None:
        """``guard(scope, text)`` may veto a write by returning the reason (leak checks).
        ``on_read(scope, path, complete)`` is told of every file view, and whether the
        view reached the end of the file.

        Of a lab wiki mounted beside the project only the pages are visible.  Its
        notes are lessons that no consolidation has checked yet, and they are read
        by the lab's own consolidation session alone (where the lab is ``project``).
        """
        self.project = project
        self.lab = lab
        self.write = write
        self.lab_write = lab_write
        self.author = author
        self._on_write = on_write
        self._on_read = on_read
        self._guard = guard
        #: Notes this session created.  With the ``notes`` scope a session may only revise
        #: its own notes, never another session's.
        self._created: set[str] = set()

    # -- path handling ---------------------------------------------------------

    def _split(self, path: Any) -> tuple[Wiki, str, bool]:
        """Virtual path -> (wiki, memory-relative path or "" for the root, is_lab)."""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path must be a non-empty string starting with /memories")
        path = path.strip()
        if "\\" in path or "\x00" in path or "%2e" in path.lower():
            raise ToolError(f"Invalid characters in path: {path!r}")
        parts = PurePosixPath(path).parts
        if len(parts) < 2 or parts[0] != "/" or parts[1] != VIRTUAL_ROOT.strip("/"):
            raise ToolError(f"Path must start with {VIRTUAL_ROOT}, got: {path}")
        rest = parts[2:]
        if any(part in ("..", ".") or part.startswith(".") for part in rest):
            raise ToolError(f"Path {path} would escape the {VIRTUAL_ROOT} directory")
        if any(part != part.strip() for part in rest):
            # The wiki would strip the whitespace and resolve the path after all.
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
        if rest and rest[0] == LAB_PREFIX:
            if self.lab is None:
                raise ToolError(f"The path {path} does not exist. There is no lab memory here.")
            if len(rest) > 1 and rest[1] == NOTES_DIR:
                raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
            return self.lab, "/".join(rest[1:]), True
        return self.project, "/".join(rest), False

    def _page(self, path: Any) -> tuple[Wiki, str, bool]:
        """Like ``_split`` but the path must name a file the wiki accepts."""
        wiki, rel, is_lab = self._split(path)
        if not rel:
            raise ToolError(f"{path} is a directory, not a file")
        try:
            return wiki, wiki.normalize(rel), is_lab
        except WikiError as exc:
            raise ToolError(
                f"{exc}. Memory files are Markdown: use a path ending in .md under "
                f"{VIRTUAL_ROOT}/{NOTES_DIR}/ or {VIRTUAL_ROOT}/{WIKI_DIR}/."
            ) from exc

    def _check_write(
        self, wiki: Wiki, rel: str, is_lab: bool, path: str, *, creating: bool = False
    ) -> None:
        if is_lab:
            if self.lab_write != "all":
                raise ToolError("The lab memory is read-only in this session.")
        elif self.write == "none":
            raise ToolError("Memory is read-only in this session.")
        elif self.write == "notes":
            in_notes = rel.startswith(NOTES_DIR + "/") and not rel.startswith(
                CONSOLIDATED_DIR + "/"
            )
            if not in_notes:
                raise ToolError(
                    f"In this session you can only add notes under {VIRTUAL_ROOT}/{NOTES_DIR}/. "
                    "Wiki pages are maintained by the consolidation pass: write a note that "
                    "says what should change and why."
                )
            if not creating and rel not in self._created:
                raise ToolError(
                    f"{path} was written by another session. You can add a new note, or edit "
                    "notes you created in this session."
                )
        if wiki.is_read_only(rel):
            raise ToolError(f"{path} is read-only ground truth and cannot be changed.")
        if rel.startswith(CONSOLIDATED_DIR + "/"):
            raise ToolError(f"{path} is an archived note and cannot be changed.")

    def _wrote(self, is_lab: bool, rel: str) -> None:
        if self._on_write is not None:
            self._on_write("lab" if is_lab else "project", rel)

    def _vet(self, is_lab: bool, text: str, rel: str | None = None) -> None:
        """Ask the guard about ``text`` and, with ``rel``, about the name of the file.

        A file name becomes a link in the index that every session reads, so it is
        vetted like the text.
        """
        if self._guard is None:
            return
        subject = text if rel is None else f"{PurePosixPath(rel).stem}\n{text}"
        reason = self._guard("lab" if is_lab else "project", subject)
        if reason:
            raise ToolError(f"This text cannot be written to memory: {reason}")

    def _added_text(self, wiki: Wiki, rel: str, text: str) -> str:
        """What ``text`` adds to the page at ``rel``: its lines that are not already there.

        Rewriting a page to merge something in must not put its existing lines on
        trial again; they were vetted when they were written.
        """
        if self._guard is None or not wiki.exists(rel):
            return text
        try:
            existing = {line.strip() for line in wiki.read_text(rel).splitlines()}
        except WikiError:
            return text
        return "\n".join(line for line in text.splitlines() if line.strip() not in existing)

    # -- view ------------------------------------------------------------------

    def _listing(self, wiki: Wiki, rel: str, is_lab: bool, depth: int = 2) -> list[str]:
        """``size<TAB>path`` lines for files and directories up to ``depth`` levels below."""
        base_parts = PurePosixPath(rel).parts if rel else ()
        files: dict[str, int] = {}
        if not rel and (wiki.root / INDEX_FILE).is_file():
            files[INDEX_FILE] = (wiki.root / INDEX_FILE).stat().st_size
        prefixes = [rel] if rel else ([WIKI_DIR] if is_lab else [WIKI_DIR, NOTES_DIR])
        for prefix in prefixes:
            for info in wiki.list(prefix):
                if info.path.startswith(CONSOLIDATED_DIR + "/") and not rel.startswith(
                    CONSOLIDATED_DIR
                ):
                    continue
                files[info.path] = info.size
        entries: dict[str, str] = {}
        for file_rel, size in files.items():
            parts = PurePosixPath(file_rel).parts
            below = parts[len(base_parts) :]
            for level in range(1, min(len(below), depth) + 1):
                entry = "/".join(base_parts + below[:level])
                is_file = level == len(below)
                entries.setdefault(entry, _human_size(size) if is_file else "4.0K")
        return [
            f"{size}\t{virtual_path(entry, lab=is_lab)}" for entry, size in sorted(entries.items())
        ]

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str:
        wiki, rel, is_lab = self._split(path)
        target = wiki.root / rel if rel else wiki.root
        is_file = bool(rel) and rel.endswith(".md")
        if not is_file:
            if rel and PurePosixPath(rel).parts[0] not in (WIKI_DIR, NOTES_DIR):
                raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
            lines = self._listing(wiki, rel, is_lab)
            if rel and not lines and not target.is_dir():
                raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
            if not rel and not is_lab and self.lab is not None:
                lines.append(f"4.0K\t{virtual_path('', lab=True)}")
                lines.extend(self._listing(self.lab, "", True, depth=1))
            shown = virtual_path(rel, lab=is_lab)
            header = (
                f"Here're the files and directories up to 2 levels deep in {shown}, "
                "excluding hidden items and node_modules:"
            )
            return "\n".join([header, f"4.0K\t{shown}", *lines])

        wiki, rel, is_lab = self._page(path)
        try:
            text = wiki.read_text(rel)
        except WikiError as exc:
            raise ToolError(
                f"The path {path} does not exist. Please provide a valid path."
            ) from exc
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
                raise ToolError(
                    f"Invalid view_range {list(view_range)}: {path} has {len(lines)} lines"
                )
        numbered: list[str] = []
        used = 0
        truncated_at: int | None = None
        for number in range(start, end + 1):
            line = f"{number:>6}\t{lines[number - 1]}"
            if used + len(line) + 1 > MAX_VIEW_CHARS and numbered:
                truncated_at = number
                break
            numbered.append(line)
            used += len(line) + 1
        if self._on_read is not None:
            # A file counts as read only when the view reached its end: what a session saw
            # of a cut or partial view is not the whole note.
            complete = truncated_at is None and end == len(lines)
            self._on_read("lab" if is_lab else "project", rel, complete)
        body = "\n".join(numbered)
        out = f"Here's the content of {path} with line numbers:\n{body}"
        if truncated_at is not None:
            out += (
                f"\n[Truncated at line {truncated_at} of {len(lines)}. "
                f"Use view_range [{truncated_at}, -1] to read on.]"
            )
        return out

    # -- mutations -------------------------------------------------------------

    def create(self, path: str, file_text: str) -> str:
        if not isinstance(file_text, str):
            raise ToolError("file_text must be a string")
        wiki, rel, is_lab = self._page(path)
        self._check_write(wiki, rel, is_lab, path, creating=True)
        if not is_lab and self.write == "notes" and wiki.exists(rel) and rel not in self._created:
            # Notes are append-only across sessions: overwriting would silently lose the
            # other session's observation.
            raise ToolError(f"{path} already exists. Choose a different file name for your note.")
        self._vet(is_lab, self._added_text(wiki, rel, file_text), rel)
        try:
            wiki.write_text(rel, file_text, author=self.author)
        except WikiError as exc:
            raise ToolError(str(exc)) from exc
        if not is_lab:
            self._created.add(rel)
        self._wrote(is_lab, rel)
        return f"File created successfully at: {path}"

    def _read_for_edit(self, path: str) -> tuple[Wiki, str, bool, str]:
        wiki, rel, is_lab = self._page(path)
        self._check_write(wiki, rel, is_lab, path)
        try:
            return wiki, rel, is_lab, wiki.read_text(rel)
        except WikiError as exc:
            raise ToolError(
                f"The path {path} does not exist. Please provide a valid path."
            ) from exc

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        if not isinstance(old_str, str) or not old_str:
            raise ToolError("old_str must be a non-empty string")
        if not isinstance(new_str, str):
            new_str = ""
        wiki, rel, is_lab, text = self._read_for_edit(path)
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
        self._vet(is_lab, new_str)
        try:
            wiki.write_text(rel, text.replace(old_str, new_str, 1), author=self.author)
        except WikiError as exc:
            raise ToolError(str(exc)) from exc
        self._wrote(is_lab, rel)
        return "The memory file has been edited."

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        if not isinstance(insert_text, str):
            raise ToolError("insert_text must be a string")
        if isinstance(insert_line, bool) or not isinstance(insert_line, int):
            raise ToolError("insert_line must be an integer")
        wiki, rel, is_lab, text = self._read_for_edit(path)
        lines = text.split("\n")
        if not 0 <= insert_line <= len(lines):
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the "
                f"range of lines of the file: [0, {len(lines)}]"
            )
        new_lines = insert_text.rstrip("\n").split("\n")
        header = _front_matter_line_count(lines)
        if 0 < insert_line < header and not all(
            _FRONT_MATTER_LINE.match(line) for line in new_lines
        ):
            # Free text inside the front matter block would be dropped when the page is saved.
            raise ToolError(
                f"Line {insert_line} is inside the page's front matter (lines 1-{header}). "
                f"Insert body text at line {header} or later."
            )
        lines[insert_line:insert_line] = new_lines
        self._vet(is_lab, insert_text)
        try:
            wiki.write_text(rel, "\n".join(lines), author=self.author)
        except WikiError as exc:
            raise ToolError(str(exc)) from exc
        self._wrote(is_lab, rel)
        return f"The file {path} has been edited."

    def delete(self, path: str) -> str:
        wiki, rel, is_lab = self._split(path)
        if not rel:
            raise ToolError(f"Cannot delete the {VIRTUAL_ROOT} directory itself")
        wiki, rel, is_lab = self._page(path)
        self._check_write(wiki, rel, is_lab, path)
        if rel == INDEX_FILE:
            raise ToolError("MEMORY.md is the index and cannot be deleted")
        try:
            wiki.delete(rel)
        except WikiError as exc:
            raise ToolError(str(exc)) from exc
        self._created.discard(rel)
        self._wrote(is_lab, rel)
        return f"Successfully deleted {path}"

    def rename(self, old_path: str, new_path: str) -> str:
        wiki, old_rel, is_lab = self._page(old_path)
        new_wiki, new_rel, new_is_lab = self._page(new_path)
        if wiki is not new_wiki:
            raise ToolError("Cannot move a file between the project memory and the lab memory")
        self._check_write(wiki, old_rel, is_lab, old_path)
        self._check_write(new_wiki, new_rel, new_is_lab, new_path, creating=True)
        self._vet(new_is_lab, "", new_rel)
        try:
            wiki.rename(old_rel, new_rel)
        except WikiError as exc:
            raise ToolError(str(exc)) from exc
        if old_rel in self._created:
            self._created.discard(old_rel)
            self._created.add(new_rel)
        self._wrote(is_lab, old_rel)
        self._wrote(is_lab, new_rel)
        return f"Successfully renamed {old_path} to {new_path}"


def wiki_search_tool(project: Wiki, lab: Wiki | None = None) -> ToolSpec:
    """Full-text search over wiki pages and unconsolidated notes.

    In a lab wiki mounted beside the project only pages are searched: its pending
    notes are unchecked lessons from other runs.
    """
    scopes = ["project", "lab", "all"] if lab is not None else ["project"]

    def handler(tool_input: dict[str, Any]) -> str:
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        scope = tool_input.get("scope") or ("all" if lab is not None else "project")
        if scope not in scopes:
            raise ToolError(f"scope must be one of {scopes}")
        section = str(tool_input.get("section") or "").strip().strip("/")
        try:
            k = max(1, min(int(tool_input.get("k") or 8), 25))
        except (TypeError, ValueError) as exc:
            raise ToolError("k must be an integer") from exc
        prefix = f"{WIKI_DIR}/{section}" if section else ""
        lab_prefix = prefix or WIKI_DIR
        if tool_input.get("exact"):
            lines: list[str] = []
            if scope in ("project", "all"):
                lines += [
                    f"{virtual_path(rel)}:{number}: {line}"
                    for rel, number, line in project.grep(query, k=k, prefix=prefix)
                ]
            if lab is not None and scope in ("lab", "all"):
                lines += [
                    f"{virtual_path(rel, lab=True)}:{number}: {line}"
                    for rel, number, line in lab.grep(query, k=k, prefix=lab_prefix)
                ]
            return "\n".join(lines[:k]) if lines else "No line contains that text verbatim."
        hits: list[tuple[float, str]] = []
        if scope in ("project", "all"):
            for hit in project.search(query, k=k, prefix=prefix):
                hits.append((hit.score, f"{virtual_path(hit.path)}: {hit.title}\n   {hit.snippet}"))
        if lab is not None and scope in ("lab", "all"):
            for hit in lab.search(query, k=k, prefix=lab_prefix):
                hits.append(
                    (
                        hit.score,
                        f"{virtual_path(hit.path, lab=True)}: {hit.title}\n   {hit.snippet}",
                    )
                )
        if not hits:
            return "No matching pages."
        hits.sort(key=lambda pair: -pair[0])
        return "\n".join(f"{i}. {text}" for i, (_, text) in enumerate(hits[:k], 1))

    properties: dict[str, Any] = {
        "query": {"type": "string", "description": "Words to look for."},
        "section": {
            "type": "string",
            "description": "Limit the search to one wiki section, for example 'literature'.",
        },
        "k": {"type": "integer", "description": "Maximum number of results (default 8)."},
        "exact": {
            "type": "boolean",
            "description": (
                "Match the query verbatim instead of by keyword, and return the matching lines "
                "with line numbers. Use this to check a number or an exact phrase."
            ),
        },
    }
    if lab is not None:
        properties["scope"] = {
            "type": "string",
            "enum": scopes,
            "description": "Which memory to search (default: all).",
        }
    return ToolSpec(
        name="wiki_search",
        description=(
            "Search the memory wiki by keyword. Returns the best-matching pages and "
            "unconsolidated notes with their /memories paths, titles and a matching line. Use "
            "it to find a page when the index does not name it, for example a paper in this "
            "run's verified bibliography by topic or a lesson about an error message. It "
            "searches page text, not the "
            "transcripts, and returns snippets only: open a page with the memory tool to read "
            "it in full. Keyword search treats 87.3 as the words 87 and 3, so set exact to "
            "true when you are checking a number or a quotation."
        ),
        input_schema={
            "type": "object",
            "properties": properties,
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def transcript_search_tool(
    transcripts: TranscriptLog, *, exclude_agents: Sequence[str] = ()
) -> ToolSpec:
    """Search the raw event log, to check a note or page against what happened.

    ``exclude_agents`` names the agents whose events are not evidence: the
    session that is doing the checking.  Its own prompts quote the notes under
    review and its own searches contain the query, so without this every search
    would find itself.
    """
    skipped = tuple(exclude_agents)

    def handler(tool_input: dict[str, Any]) -> str:
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        stage = tool_input.get("stage") or None
        try:
            k = max(1, min(int(tool_input.get("k") or 8), 25))
        except (TypeError, ValueError) as exc:
            raise ToolError("k must be an integer") from exc
        # Only what the run recorded: memory content that passed through the transcript
        # (pushed pages, memory-tool results) would confirm any claim taken from memory.
        find = transcripts.grep if tool_input.get("exact") else transcripts.search
        hits = find(query, k=k, stage=stage, exclude_agents=skipped, evidence_only=True)
        if not hits:
            return "No matching transcript events."
        return "\n".join(
            f"{i}. [{hit.ts} {hit.stage or '-'}/{hit.agent or '-'} {hit.type} "
            f"session={hit.session} seq={hit.seq}]\n   {hit.snippet}"
            for i, hit in enumerate(hits, 1)
        )

    return ToolSpec(
        name="transcript_search",
        description=(
            "Search the session transcripts: the raw record of model requests and replies, "
            "tool calls, compile results and review scores from every session of this run. Use "
            "it to verify a claim before recording it as fact, for example whether a compile "
            "error really occurred and what fixed it, or what scores a revision received. "
            "Returns matching events with their stage, time and a snippet; it does not return "
            "whole transcripts. Consolidation sessions such as this one are left out, so a "
            "result is evidence from the run itself. An event of type 'note' is a claim some "
            "session made, not a record of what happened."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Words to look for."},
                "stage": {
                    "type": "string",
                    "description": "Only events from this pipeline stage, e.g. 'refinement'.",
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
