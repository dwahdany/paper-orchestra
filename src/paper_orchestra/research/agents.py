"""The roles of a research project: intake, scout, theorist, designer, runner, analyst.

Each role is one function that runs its session through the context harness.
A session sees the wiki through :class:`~paper_orchestra.research.access.ProjectMemoryBackend`,
so what it may read and write is decided by the project, not by its prompt: it
proposes pages of the kinds its role may create, and suggests every other
change with ``propose_edit``.  What a role adds here is its tools (the scout's
``verify_paper``, the designer's ``write_code`` and read-only view of the
person's repository), the checks that code runs on what the model produced
(an experiment must parse; a finding's numbers must be on record), and the
bounded conversation in which the model is asked to put right what a check found.

The engine calls these functions; they never decide what phase comes next.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from paper_orchestra.compute.base import (
    SOURCE_EXCLUDES,
    ComputeBackend,
    ComputeError,
    SourceReport,
    human_size,
    walk_source,
)
from paper_orchestra.config import DEFAULT_MODEL, default_lab_memory_dir
from paper_orchestra.llm.base import (
    Capabilities,
    Image,
    LLMConfigError,
    Part,
    Pdf,
    Text,
    ToolError,
    ToolSpec,
    UsageTracker,
    WebSearch,
)
from paper_orchestra.memory.cache import KeyValueCache
from paper_orchestra.memory.harness import ContextHarness, MemoryPolicy
from paper_orchestra.memory.tools import virtual_path
from paper_orchestra.memory.transcript import TranscriptLog
from paper_orchestra.memory.wiki import TRANSCRIPTS_DIR
from paper_orchestra.research import experiments, prompts
from paper_orchestra.research.access import (
    MAX_INPUT_BYTES,
    ProjectMemoryBackend,
    propose_edit_tool,
    search_tool,
)
from paper_orchestra.research.model import (
    IDEA_PATH,
    SECTIONS,
    NotFound,
    ProjectError,
    Request,
    ResearchPage,
    agent_author,
    creatable_kinds,
    kind_for_path,
    moved_on,
    role_author,
)
from paper_orchestra.research.project import INPUTS_DIR, Project
from paper_orchestra.research.sources import SourceVerifier, normalize_cutoff, verify_paper_tool
from paper_orchestra.utils.pdf import load_paper, page_count
from paper_orchestra.utils.scholar import SemanticScholar

__all__ = [
    "MODEL_ROLES",
    "AgentContext",
    "AgentReport",
    "Inputs",
    "RepairAnswer",
    "SessionStopped",
    "build_harness",
    "count_of",
    "describe_event",
    "experiment_problems",
    "make_repair",
    "model_for",
    "model_problems",
    "read_inputs",
    "repo_tools",
    "run_analyst",
    "run_designer",
    "run_intake",
    "run_scout",
    "run_theorist",
    "write_code_tool",
]

#: Which of the project's models (``settings.models``) each role works with.
MODEL_ROLES: dict[str, str] = {
    "intake": "research",
    "scout": "research",
    "theorist": "reasoning",
    "designer": "coding",
    "runner": "coding",
    "analyst": "reasoning",
}

#: Requests a session may make before its turn is cut off.  A research session reads,
#: searches and writes many pages in one turn.
MAX_TURNS = 200
WEB_SEARCHES = 40
#: How often the designer is asked to put right what the check of its experiments found,
#: and the analyst what the check of its numbers found.
DESIGN_REPAIRS = 2
NUMBER_REPAIRS = 2

#: Attached files, as the intake session is shown them.
MAX_FILE_CHARS = 40_000
MAX_TOTAL_CHARS = 240_000
MAX_IMAGE_BYTES = 5_000_000
MAX_IMAGES = 20
MAX_PDF_PAGES = 100
#: Images and PDFs travel in one request, base64-encoded (a third larger), and a request
#: is refused above 32 MB: what is sent as itself stays well below that in total.  A PDF
#: that does not fit is given as its extracted text instead.
MAX_ATTACHED_BYTES = 20_000_000
#: PDF pages sent as pages, over all PDFs of the request.
MAX_TOTAL_PDF_PAGES = 300
#: An image up to this size, in pixels along its longer edge and in bytes, is sent as it
#: is.  A larger one is scaled down and encoded again by the model's client, to at most
#: ``RESIZED_IMAGE_BYTES`` (the most a provider takes for one image).
PLAIN_IMAGE_EDGE = 2000
PLAIN_IMAGE_BYTES = 3_000_000
RESIZED_IMAGE_BYTES = 4_000_000

#: The person's repository, as the designer's read-only tools show it.
REPO_MAX_FILE_BYTES = 1_000_000
REPO_MAX_VIEW_CHARS = 40_000
REPO_MAX_ENTRIES = 400
REPO_MAX_MATCHES = 50

_ID_IN_TEXT = re.compile(r"\b[Hh]\d+\b")


# ---------------------------------------------------------------------------
# What a session is given, and what it leaves behind
# ---------------------------------------------------------------------------


def _no_guard() -> None:
    return None


def _never() -> bool:
    return False


class SessionStopped(BaseException):
    """A stop request reached a session between two model calls of one turn.

    Not an ``Exception`` on purpose.  A model client answers an exception raised by
    a tool with an error message to the model and carries on with the turn; a stop
    has to end it, the way an interrupt does.  Whoever runs the phase records it as
    interrupted; what the session wrote until then stays, and a resumed session
    finds it.
    """


@dataclass
class AgentContext:
    """What a role's session is given by whoever runs the phase.

    ``session`` is the stem of the session id: the same for every attempt at one
    run of a phase, so that a resumed session sees and revises what it proposed
    before it was interrupted, and different for any other run.  ``guard`` is
    called before every session and every turn and raises when the work must
    stop (a stop request, a spent budget).  ``stopping`` says whether a stop was
    requested; it is asked before every tool call, so that a long turn ends at
    its next model call instead of running to its end.  ``compute`` is a backend,
    or a callable choosing one for a plan, as ``research.experiments.run`` takes
    it.  ``produced`` fills with the paths of the pages this run of the phase
    created, as they are created, so that an interrupted phase can still say what
    it left behind.
    """

    project: Project
    harness: ContextHarness
    round: int
    phase: str
    session: str
    requests: Sequence[Request] = ()
    instruction: str | None = None
    guard: Callable[[], None] = _no_guard
    stopping: Callable[[], bool] = _never
    scholar: SemanticScholar | None = None
    compute: ComputeBackend | Callable[[Any], ComputeBackend] | None = None
    produced: list[str] = field(default_factory=list)

    def session_id(self, role: str) -> str:
        return f"{self.session}-{role}"

    def heed_stop(self) -> None:
        """Raise :class:`SessionStopped` when a stop was requested."""
        if self.stopping():
            raise SessionStopped("the phase was asked to stop")

    def note(self, paths: Iterable[str]) -> None:
        """Remember pages this run of the phase created."""
        self.produced += [path for path in paths if path not in self.produced]

    def activity(self, summary: str, *, agent: str, refs: Iterable[str] = ()) -> None:
        """One line for the activity feed about what an agent is doing."""
        self.project.emit(
            "agent.activity", summary, phase=self.phase, agent=agent, refs=refs, round=self.round
        )

    def backend(self, plan: Any = None) -> ComputeBackend | None:
        """The compute backend that would run ``plan`` (the default one without a plan)."""
        compute = self.compute
        if compute is None or isinstance(compute, ComputeBackend):
            return compute
        try:
            return compute(plan)
        except (ComputeError, AttributeError, TypeError):
            return None


@dataclass
class AgentReport:
    """What a role's session left behind, for the phase record and the checkpoint."""

    #: Project-relative paths of the pages this run of the phase created, in every attempt.
    produced: list[str] = field(default_factory=list)
    #: One sentence saying what was done.
    summary: str = ""
    #: What the person should be told at the checkpoint beyond the counts.
    notes: list[str] = field(default_factory=list)


def build_harness(
    project: Project, *, client_for: Callable[[str], Any] | None = None
) -> ContextHarness:
    """The harness of a research project: transcripts and usage, no wiki of its own.

    The project's wiki reaches a session through the memory backend the session is
    opened with, never through the harness, which would consolidate and index it by
    rules that are not the project's.
    """
    cache = (
        KeyValueCache(default_lab_memory_dir() / "cache") if project.settings.lab_memory else None
    )
    return ContextHarness(
        transcripts=TranscriptLog(project.root / TRANSCRIPTS_DIR),
        client_for=client_for,
        usage=UsageTracker(),
        default_effort=project.settings.models.effort,  # type: ignore[arg-type]
        dream_mode="off",
        cache=cache,
    )


def model_for(project: Project, role: str) -> str:
    """The model a role works with."""
    return str(project.settings.models[MODEL_ROLES[role]])


def model_problems(project: Project, harness: ContextHarness) -> list[str]:
    """Why the research agents cannot work with the project's models; empty when they can.

    The agents read and write the wiki through tools and the memory tool.  A model
    without them is refused here, before anything is spent, with the setting to change.
    """
    problems: list[str] = []
    for setting in dict.fromkeys(MODEL_ROLES.values()):
        model = str(project.settings.models[setting])
        fix = (
            f"Set settings.models.{setting} to a model that has both (for example "
            f"{DEFAULT_MODEL}) and run the command again. Nothing was spent."
        )
        try:
            capabilities: Capabilities = harness.client_for(model).capabilities
        except LLMConfigError as exc:
            problems.append(
                f"The model set for `{setting}` ({model}) cannot be used: {exc} "
                "Put this right, or change settings.models."
                f"{setting}, and run the command again. Nothing was spent."
            )
            continue
        missing = [
            name
            for name, has in (
                ("tools", capabilities.tools),
                ("the memory tool", capabilities.memory_tool),
            )
            if not has
        ]
        if missing:
            problems.append(
                f"The model set for `{setting}` ({model}) offers no {' and no '.join(missing)}, "
                "which the research agents need to read and write the wiki. " + fix
            )
    return problems


# ---------------------------------------------------------------------------
# A session
# ---------------------------------------------------------------------------


class _SessionBackend(ProjectMemoryBackend):
    """The project's memory backend, with the phase's limit on new pages of a kind."""

    def __init__(
        self,
        project: Project,
        role: str,
        round: int,
        session_id: str,
        *,
        caps: Mapping[str, int] | None = None,
        heed_stop: Callable[[], None] = _no_guard,
    ) -> None:
        super().__init__(project, role, round, session_id)
        self._caps = dict(caps or {})
        #: Called before every command of the memory tool, which is where a turn that
        #: reads and writes many pages can be ended between two model calls.
        self._heed_stop = heed_stop

    def proposed(self, kind: str | None = None) -> list[ResearchPage]:
        """Pages (of ``kind``) this session created, in this attempt or one it resumes."""
        return [
            page
            for page in self.project.pages(kind=kind)
            if page.author == self.author and page.session == self.session_id
        ]

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str:
        self._heed_stop()
        return super().view(path, view_range)

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        self._heed_stop()
        return super().str_replace(path, old_str, new_str)

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        self._heed_stop()
        return super().insert(path, insert_line, insert_text)

    def delete(self, path: str) -> str:
        self._heed_stop()
        return super().delete(path)

    def create(self, path: str, file_text: str) -> str:
        self._heed_stop()
        rel = self._rel(path)
        kind = kind_for_path(rel) if rel else None
        cap = self._caps.get(kind or "")
        # Writing to a path that exists revises that page; only a new page counts.
        if cap is not None and not (self.project.root / rel).is_file():
            have = len(self.proposed(kind or ""))
            if have >= cap:
                raise ToolError(
                    f"This phase proposes at most {count_of(cap, kind or '')}, and {have} are "
                    "proposed. Revise or delete one of yours instead of adding another; the "
                    "researcher can raise the limit in the project's settings."
                )
        return super().create(path, file_text)


def _stoppable(tool: ToolSpec, heed_stop: Callable[[], None]) -> ToolSpec:
    """``tool`` with a stop request heeded before each call."""

    def handler(tool_input: dict[str, Any]) -> str:
        heed_stop()
        return tool.handler(tool_input)

    return replace(tool, handler=handler)


def _short(text: Any, limit: int = 120) -> str:
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _shown(path: Any) -> str:
    """A ``/memories`` path as a person reads it in the feed."""
    return str(path or "").removeprefix("/memories").strip("/") or "the wiki"


def describe_event(role: str, event: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    """One plain sentence about a session event, with the paths it concerns; or ``None``.

    Only what a person following the work would want to see: what was read, looked
    up and checked.  Pages written already announce themselves through the project.
    """
    who = f"The {role}"
    if event.get("type") == "usage":
        searches = event.get("web_searches") or 0
        if searches:
            plural = "es" if searches != 1 else ""
            return f"{who} searched the web ({searches} search{plural}).", []
        return None
    if event.get("type") != "tool_call":
        return None
    name = event.get("name")
    given = event.get("input")
    given = given if isinstance(given, Mapping) else {}
    if name == "memory":
        if given.get("command") != "view":
            return None
        path = _shown(given.get("path"))
        ref = [path] if path.endswith(".md") or path.startswith(f"{INPUTS_DIR}/") else []
        return (f"{who} read {path}." if ref else f"{who} looked through {path}."), ref
    if name == "wiki_search":
        return f"{who} searched the wiki for “{_short(given.get('query'), 80)}”.", []
    if name == "verify_paper":
        title = _short(given.get("title"), 100)
        return f"{who} checked “{title}” against Semantic Scholar.", []
    if name == "write_code":
        return f"{who} wrote {_short(given.get('name'), 60)} for {given.get('experiment')}.", []
    if name in ("repo_list", "repo_read", "repo_search"):
        what = _short(given.get("path") or given.get("query") or "the top level", 80)
        verb = {"repo_list": "listed", "repo_read": "read", "repo_search": "searched for"}[name]
        return f"{who} {verb} {what} in your repository.", []
    if name == "propose_edit":
        return None
    return f"{who} used {name}.", []


class _Session:
    """One role's conversation: its memory backend, its tools, what it wrote."""

    def __init__(
        self,
        ctx: AgentContext,
        role: str,
        *,
        tools: Sequence[ToolSpec] | Callable[[_SessionBackend], Sequence[ToolSpec]] = (),
        web_search: WebSearch | None = None,
        caps: Mapping[str, int] | None = None,
    ) -> None:
        """``tools`` are the role's own; a callable builds them from the session's backend
        (a tool that writes on the session's behalf needs to know whose session it is)."""
        ctx.guard()
        self.ctx = ctx
        self.role = role
        project = ctx.project
        self.backend = _SessionBackend(
            project, role, ctx.round, ctx.session_id(role), caps=caps, heed_stop=ctx.heed_stop
        )
        if callable(tools):
            tools = tools(self.backend)
        offered = [search_tool(project), propose_edit_tool(project, role, ctx.round), *tools]
        index = project.index_text(role, session=self.backend.session_id)
        self._last = ""
        self.conversation = ctx.harness.open(
            agent=role,
            stage=ctx.phase,
            model=model_for(project, role),
            system=prompts.system_prompt(role, index=index),
            memory=MemoryPolicy.none(),
            memory_backend=self.backend,
            tools=[_stoppable(tool, ctx.heed_stop) for tool in offered],
            web_search=web_search,
            effort=project.settings.models.effort,  # type: ignore[arg-type]
            max_turns=MAX_TURNS,
            on_event=self._watch,
        )

    def _watch(self, event: dict[str, Any]) -> None:
        described = describe_event(self.role, event)
        # The same page opened twice in a row is one thing to tell.
        if described is not None and described[0] != self._last:
            self._last = described[0]
            self.ctx.activity(described[0], agent=self.role, refs=described[1])

    def send(self, parts: Part | Sequence[Part]) -> str:
        self.ctx.guard()
        try:
            return self.conversation.send(parts).text
        finally:
            # Also when the turn was cut short: what it wrote by then is in the wiki.
            self.ctx.note(self.created())

    def proposed(self, kind: str | None = None) -> list[ResearchPage]:
        return self.backend.proposed(kind)

    def created(self) -> list[str]:
        """Paths of the pages this session created and that are still there."""
        return [page.path for page in self.backend.proposed()]


def _idea(project: Project) -> str:
    return project.page(IDEA_PATH).body


def _request_items(ctx: AgentContext) -> list[dict[str, Any]]:
    return [{"text": request.text, "target": request.target} for request in ctx.requests]


def _item(page: ResearchPage) -> dict[str, Any]:
    return {
        "kind": page.kind,
        "id": page.id,
        "title": page.title,
        "summary": page.summary,
        "reason": page.decision_reason,
    }


def _earlier(ctx: AgentContext, role: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """What the role must not propose twice.

    The undecided proposals of its earlier sessions, which this session cannot read
    because they belong to another one, and the rejected pages of the kinds it writes.
    """
    author, own = role_author(role), ctx.session_id(role)
    kinds = creatable_kinds(role)
    pending = [
        _item(page)
        for page in ctx.project.pages(kind=kinds, review="proposed")
        if page.author == author and page.session != own
    ]
    rejected = [_item(page) for page in ctx.project.pages(kind=kinds, review="rejected")]
    return pending, rejected


def _plural(count: int, noun: str) -> str:
    if count == 1:
        return f"1 {noun}"
    return f"{count} {noun[:-1] + 'ies' if noun.endswith('y') else noun + 's'}"


def count_of(count: int, kind: str) -> str:
    """``2 hypotheses``, ``1 journal page``: a number of pages of a kind, as one says it."""
    if kind == "journal":
        return _plural(count, "journal page")
    return f"{count} {kind if count == 1 else SECTIONS.get(kind) or kind + 's'}"


def _listed(parts: Sequence[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _counted(paths: Iterable[str]) -> str:
    """``3 topics and 4 questions`` for the pages at ``paths``; ``nothing`` for none."""
    counts: dict[str, int] = {}
    for path in paths:
        kind = kind_for_path(path) or "page"
        counts[kind] = counts.get(kind, 0) + 1
    return _listed([count_of(count, kind) for kind, count in counts.items()]) or "nothing"


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    """The attached files as the intake session receives them."""

    parts: list[Part] = field(default_factory=list)
    unread: list[tuple[str, str]] = field(default_factory=list)
    attached: int = 0
    _chars: int = 0
    _images: int = 0
    #: What the images and PDFs given as themselves weigh, and the pages of those PDFs.
    _bytes: int = 0
    _pages: int = 0

    def has_room(self, size: int, pages: int = 0) -> bool:
        """Whether an image or a PDF of ``size`` bytes (and ``pages``) still fits into the
        one request the files are sent in."""
        return (
            self._bytes + size <= MAX_ATTACHED_BYTES and self._pages + pages <= MAX_TOTAL_PDF_PAGES
        )

    def text(
        self, name: str, text: str, size: int, *, whole: bool = True, extracted: bool = False
    ) -> None:
        """Add a file's text.  ``whole`` says whether ``text`` is all of the file."""
        where = virtual_path(f"{INPUTS_DIR}/{name}")
        room = min(MAX_FILE_CHARS, MAX_TOTAL_CHARS - self._chars)
        if room <= 0:
            self.unread.append(
                (
                    name,
                    f"not included for lack of room; it can be read with the memory tool at {where}",
                )
            )
            return
        notes = ["text extracted from the PDF; figures and layout are not included"] * extracted
        if len(text) > room or not whole:
            shown = min(len(text), room)
            total = f"{len(text)} characters" if whole else f"a file of {human_size(size)}"
            rest = (
                "the rest is too large to be read here"
                if extracted or size > MAX_INPUT_BYTES
                else f"read the rest with the memory tool at {where}"
            )
            notes.append(f"the first {shown} characters of {total} are shown; {rest}")
            text = text[:shown]
        self._chars += len(text)
        self.parts.append(
            Text(prompts.file_block(f"{INPUTS_DIR}/{name}", text, note="; ".join(notes)))
        )

    def image(self, name: str, image: Image) -> None:
        if self._images >= MAX_IMAGES:
            self.unread.append((name, f"more than {MAX_IMAGES} images were attached"))
            return
        weight = _image_weight(image.data)
        if not self.has_room(weight):
            self.unread.append(
                (name, "an image not shown for lack of room: the files before it fill the request")
            )
            return
        self._images += 1
        self._bytes += weight
        self.parts += [Text(f"The image {INPUTS_DIR}/{name} follows."), image]

    def pdf(self, name: str, data: bytes, pages: int) -> None:
        """Add a PDF as itself; the caller has asked :meth:`has_room`."""
        self._bytes += len(data)
        self._pages += pages
        self.parts += [Text(f"The PDF {INPUTS_DIR}/{name} follows."), Pdf(data, title=name)]


def _image_weight(data: bytes) -> int:
    """At most how many bytes the image is once the model's client has made it fit.

    A file of few bytes can hold many pixels; scaled down and encoded again it may
    weigh more than it did, so its own size is no bound.
    """
    edge: int | None = None
    try:
        from PIL import Image as Picture

        with Picture.open(io.BytesIO(data)) as picture:
            edge = max(picture.size)
    except Exception:  # not readable here: assume the client has to work on it
        edge = None
    plain = edge is not None and edge <= PLAIN_IMAGE_EDGE and len(data) <= PLAIN_IMAGE_BYTES
    return len(data) if plain else RESIZED_IMAGE_BYTES


def _decoded(data: bytes, *, partial: bool) -> str | None:
    """``data`` as text, or ``None`` when it is not UTF-8 text."""
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A read that stopped inside a character is still text.
        if partial and exc.start >= len(data) - 4:
            return data[: exc.start].decode("utf-8", errors="ignore")
        return None


def _pdf_pages(path: Path) -> int | None:
    try:
        return page_count(path)
    except Exception:  # a damaged file, or no PDF library: the page count is unknown
        return None


def _add_pdf(inputs: Inputs, name: str, path: Path, size: int, capabilities: Capabilities) -> None:
    pages = _pdf_pages(path)
    fits = pages is not None and pages <= MAX_PDF_PAGES and inputs.has_room(size, pages)
    if capabilities.pdf_input and fits:
        inputs.pdf(name, path.read_bytes(), pages or 0)
        return
    try:
        inputs.text(name, load_paper(path), size, extracted=True)
    except Exception as exc:  # no text layer, a damaged file, no PDF library
        inputs.unread.append((name, f"a PDF whose text could not be extracted ({_short(exc)})"))


def _input_path(base: Path, name: str) -> Path | None:
    """The attached file the manifest calls ``name``; ``None`` unless it is a file in ``base``.

    The manifest is a file of the project like any other and may have been edited
    by hand.  Whatever it names, only what lies inside ``inputs/`` is read: no
    ``..``, no absolute path, and nothing reached through a symbolic link.
    """
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or ".." in parts or "\\" in name or "\0" in name:
        return None
    path = base.joinpath(*parts)
    try:
        root = base.resolve()
        real = path.resolve()
    except OSError:
        return None
    inside = real == root.joinpath(*parts) and real.is_relative_to(root)
    return path if inside and path.is_file() else None


def _add_file(inputs: Inputs, entry: Mapping[str, Any], base: Path, caps: Capabilities) -> None:
    name = str(entry.get("name") or "")
    media_type = str(entry.get("media_type") or "")
    path = _input_path(base, name)
    if path is None:
        inputs.unread.append((name or "(unnamed)", "the file is not among the attached files"))
        return
    size = path.stat().st_size
    if media_type == "application/pdf" or path.suffix.lower() == ".pdf":
        _add_pdf(inputs, name, path, size, caps)
        return
    with path.open("rb") as handle:
        head = handle.read(4 * MAX_FILE_CHARS)
    if media_type.startswith("image/") and media_type != "image/svg+xml":
        try:
            image = Image.from_bytes(path.read_bytes() if size <= MAX_IMAGE_BYTES else head)
        except ValueError:
            inputs.unread.append((name, f"an image format that cannot be shown ({media_type})"))
            return
        if not caps.image_input:
            inputs.unread.append((name, "an image, and this model does not read images"))
        elif size > MAX_IMAGE_BYTES:
            inputs.unread.append((name, f"an image of {human_size(size)}, too large to show"))
        else:
            inputs.image(name, image)
        return
    text = _decoded(head, partial=size > len(head))
    if text is None:
        kind = media_type or "unknown type"
        inputs.unread.append((name, f"not a text file ({kind}, {human_size(size)})"))
        return
    inputs.text(name, text, size, whole=size <= len(head))


def read_inputs(project: Project, capabilities: Capabilities) -> Inputs:
    """The person's attached files as parts for a model with ``capabilities``.

    Text is given as text, cut with a note that says so; images and PDFs as
    themselves where the model reads them; a PDF otherwise as its extracted text.
    Whatever could not be given is listed in ``unread`` with the reason, so that
    nothing is described that was not read.
    """
    inputs = Inputs()
    entries = project.attachments()
    inputs.attached = len(entries)
    for entry in entries:
        _add_file(inputs, entry, project.inputs_dir, capabilities)
    return inputs


def name_project_tool(ctx: AgentContext) -> ToolSpec:
    """Intake's tool for naming a project that only has the brief's first line as its title."""
    project = ctx.project

    def handler(tool_input: dict[str, Any]) -> str:
        title = " ".join(str(tool_input.get("title") or "").split()).rstrip(".")
        if not 3 <= len(title) <= 80:
            raise ToolError(
                "title must be one short line of 3 to 80 characters: a few words that say "
                "what the research is about"
            )
        try:
            project.rename(title, by=agent_author("intake"))
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        ctx.activity(f'The intake named the project "{title}".', agent="intake")
        return f'The project is now called "{title}". The researcher can rename it.'

    return ToolSpec(
        name="name_project",
        description=(
            "Give the project a title. Offered only while the project has none of its own "
            "(it shows the first line of the idea). One short line, at most 80 characters, "
            "that says what the research is about."
        ),
        input_schema={
            "type": "object",
            "properties": {"title": {"type": "string", "description": "The title, one line."}},
            "required": ["title"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def run_intake(ctx: AgentContext) -> AgentReport:
    """Put into the wiki what the person's attached files contribute."""
    project = ctx.project
    capabilities = ctx.harness.client_for(model_for(project, "intake")).capabilities
    inputs = read_inputs(project, capabilities)
    untitled = project.title_is_provisional
    session = _Session(ctx, "intake", tools=[name_project_tool(ctx)] if untitled else ())
    pending, rejected = _earlier(ctx, "intake")
    message = prompts.intake_message(
        idea=_idea(project),
        attached=inputs.attached,
        unread=inputs.unread,
        untitled=untitled,
        pending=pending,
        rejected=rejected,
        requests=_request_items(ctx),
        instruction=ctx.instruction,
    )
    session.send([Text(message), *inputs.parts])
    produced = session.created()
    read = inputs.attached - len(inputs.unread)
    summary = (
        f"Intake read {read} of {_plural(inputs.attached, 'attached file')} and proposed "
        f"{_counted(produced)}."
        if inputs.attached
        else f"Intake proposed {_counted(produced)} from the idea."
    )
    notes = []
    if inputs.unread:
        names = _listed([f"{name} ({why})" for name, why in inputs.unread[:5]])
        more = len(inputs.unread) - 5
        notes.append(
            f"{_plural(len(inputs.unread), 'attached file')} could not be read: {names}"
            + (f" and {more} more." if more > 0 else ".")
        )
    return AgentReport(produced=produced, summary=summary, notes=notes)


# ---------------------------------------------------------------------------
# Scout
# ---------------------------------------------------------------------------


def run_scout(ctx: AgentContext) -> AgentReport:
    """Find prior work, record what can be verified, synthesise it, ask what is open."""
    project = ctx.project
    verifier = SourceVerifier(project, scholar=ctx.scholar, harness=ctx.harness, round=ctx.round)
    capabilities = ctx.harness.client_for(model_for(project, "scout")).capabilities
    session = _Session(
        ctx,
        "scout",
        tools=[verify_paper_tool(verifier)],
        web_search=WebSearch(max_uses=WEB_SEARCHES),
    )
    pending, rejected = _earlier(ctx, "scout")
    message = prompts.scout_message(
        idea=_idea(project),
        cutoff=normalize_cutoff(project.settings.research_cutoff),
        web_search=capabilities.web_search,
        sources=len(project.pages(kind="source", review=["accepted", "proposed"])),
        max_sources=int(project.settings.limits.max_sources),
        pending=pending,
        rejected=rejected,
        requests=_request_items(ctx),
        instruction=ctx.instruction,
    )
    try:
        session.send(message)
    finally:
        ctx.note(verifier.recorded)
    confirmed = verifier.candidates - len(verifier.rejected)
    if verifier.candidates:
        ctx.activity(
            f"Verified {confirmed} of {_plural(verifier.candidates, 'candidate paper')}.",
            agent="scout",
            refs=verifier.recorded,
        )
    pages = session.created()
    notes = []
    if verifier.down:
        notes.append(
            "Semantic Scholar stopped answering during the research, so some papers could "
            "not be verified; run the research phase again later to add them."
        )
    return AgentReport(
        produced=[*verifier.recorded, *pages],
        summary=f"The scout recorded {_plural(len(verifier.recorded), 'source')} and proposed "
        f"{_counted(pages)}.",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Theorist
# ---------------------------------------------------------------------------


def run_theorist(ctx: AgentContext) -> AgentReport:
    """Propose hypotheses an experiment could contradict."""
    project = ctx.project
    limit = int(project.settings.limits.max_hypotheses)
    session = _Session(ctx, "theorist", caps={"hypothesis": limit})
    pending, rejected = _earlier(ctx, "theorist")
    session.send(
        prompts.theorist_message(
            idea=_idea(project),
            max_hypotheses=limit,
            pending=pending,
            rejected=rejected,
            requests=_request_items(ctx),
            instruction=ctx.instruction,
        )
    )
    produced = session.created()
    return AgentReport(produced=produced, summary=f"The theorist proposed {_counted(produced)}.")


# ---------------------------------------------------------------------------
# Designer
# ---------------------------------------------------------------------------


def _repo_root(project: Project) -> Path | None:
    """The person's repository when it is a directory on this machine."""
    repo = project.settings.repo
    if repo is None or not repo.path or "://" in repo.path or repo.path.startswith("git@"):
        return None
    root = Path(repo.path).expanduser()
    return root.resolve() if root.is_dir() else None


def _excluded(parts: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(part, pattern) for part in parts for pattern in SOURCE_EXCLUDES)


def _repo_path(root: Path, given: Any) -> Path:
    """The file or directory ``given`` names inside ``root``; ``ToolError`` for anything else.

    Version-control data, environments and ``.env`` files are left out, as they are
    left out of the copy an experiment runs in: a model has no use for them, and the
    last hold credentials.
    """
    text = str(given or "").strip().strip("/")
    parts = PurePosixPath(text).parts if text else ()
    if "\0" in text or any(part in ("..", "~") for part in parts) or _excluded(parts):
        raise ToolError(f"{given!r} is not a path inside the repository that can be read.")
    target = root.joinpath(*parts).resolve()
    if target != root and root not in target.parents:
        raise ToolError(f"{given!r} leads outside the repository.")
    # A link inside the repository must not be a way round what is left out.
    if _excluded(target.relative_to(root).parts):
        raise ToolError(f"{given!r} is not a path inside the repository that can be read.")
    if not target.exists():
        raise ToolError(f"There is no {text or 'such path'} in the repository.")
    return target


def _repo_text(path: Path) -> str:
    size = path.stat().st_size
    if size > REPO_MAX_FILE_BYTES:
        raise ToolError(f"{path.name} is {human_size(size)}, too large to read here.")
    text = _decoded(path.read_bytes(), partial=False)
    if text is None:
        raise ToolError(f"{path.name} is not a text file.")
    return text


def _repo_list(root: Path, tool_input: Mapping[str, Any]) -> str:
    folder = _repo_path(root, tool_input.get("path"))
    if not folder.is_dir():
        raise ToolError(f"{tool_input.get('path')} is a file; read it with repo_read.")
    lines = []
    for entry in sorted(folder.iterdir(), key=lambda item: (item.is_file(), item.name)):
        if _excluded([entry.name]) or entry.is_symlink():
            continue
        rel = entry.relative_to(root).as_posix()
        lines.append(
            f"{rel}/" if entry.is_dir() else f"{rel}  ({human_size(entry.stat().st_size)})"
        )
    if not lines:
        return "The directory is empty."
    more = len(lines) - REPO_MAX_ENTRIES
    shown = lines[:REPO_MAX_ENTRIES]
    return "\n".join(shown + ([f"[{more} more entries: list a subdirectory]"] if more > 0 else []))


def _repo_read(root: Path, tool_input: Mapping[str, Any]) -> str:
    path = _repo_path(root, tool_input.get("path"))
    if not path.is_file():
        raise ToolError(f"{tool_input.get('path')} is a directory; list it with repo_list.")
    lines = _repo_text(path).split("\n")
    try:
        start = max(1, int(tool_input.get("start_line") or 1))
    except (TypeError, ValueError) as exc:
        raise ToolError("start_line must be a line number") from exc
    shown: list[str] = []
    used = 0
    for number in range(start, len(lines) + 1):
        line = f"{number:>6}\t{lines[number - 1]}"
        if used + len(line) > REPO_MAX_VIEW_CHARS and shown:
            shown.append(f"[cut at line {number} of {len(lines)}: read on with start_line]")
            break
        shown.append(line)
        used += len(line) + 1
    return "\n".join(shown) or f"The file has {len(lines)} lines; nothing at line {start}."


def _repo_search(root: Path, tool_input: Mapping[str, Any]) -> str:
    query = str(tool_input.get("query") or "")
    if not query.strip():
        raise ToolError("query must be the text to look for")
    base = _repo_path(root, tool_input.get("path"))
    needle = query.lower()
    matches: list[str] = []
    for relative, path, kind in walk_source(base if base.is_dir() else base.parent, SourceReport()):
        if kind != "file" or (base.is_file() and path != base):
            continue
        try:
            text = _repo_text(path)
        except (ToolError, OSError):
            continue
        shown = path.relative_to(root).as_posix() if root in path.parents else relative
        for number, line in enumerate(text.split("\n"), 1):
            if needle in line.lower():
                matches.append(f"{shown}:{number}: {_short(line, 200)}")
                if len(matches) >= REPO_MAX_MATCHES:
                    return "\n".join([*matches, f"[stopped at {REPO_MAX_MATCHES} matches]"])
    return "\n".join(matches) or "No line contains that text."


def repo_tools(root: Path) -> list[ToolSpec]:
    """Read-only tools over the person's repository: list, read, search.

    Confined to ``root``; nothing is ever written, and what an experiment's copy
    of the repository leaves out cannot be read here either.
    """
    root = root.resolve()
    path = {"type": "string", "description": "A path inside the repository; empty for its top."}

    def tool(
        name: str, description: str, handler: Any, properties: dict, required: list
    ) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            handler=lambda tool_input: handler(root, tool_input),
        )

    return [
        tool(
            "repo_list",
            "List the files and directories in one directory of the researcher's repository "
            "(read-only). Start with an empty path to see the top level.",
            _repo_list,
            {"path": path},
            [],
        ),
        tool(
            "repo_read",
            "Read a text file of the researcher's repository with line numbers (read-only). "
            "A long file is cut; read on from the line the cut names with start_line.",
            _repo_read,
            {"path": path, "start_line": {"type": "integer", "description": "First line shown."}},
            ["path"],
        ),
        tool(
            "repo_search",
            "Find the lines of the researcher's repository that contain a text, ignoring "
            "case (read-only). Give a path to search one directory or file only.",
            _repo_search,
            {"query": {"type": "string", "description": "The text to look for."}, "path": path},
            ["query"],
        ),
    ]


def _own_proposal(page: ResearchPage, backend: ProjectMemoryBackend) -> bool:
    """A page this session proposed and that is still its to work on."""
    return (
        page.author == backend.author
        and page.session == backend.session_id
        and page.review != "rejected"
        and not page.decided_by
        and not moved_on(page)
    )


def _lacks_code(project: Project, page: ResearchPage) -> bool:
    """An accepted plan that nobody has written code for, so that it cannot be approved."""
    return (
        page.review == "accepted"
        and page.state == "planned"
        and bool(page.id)
        and not experiments.code_files(project, str(page.id))
    )


def write_code_tool(ctx: AgentContext, backend: ProjectMemoryBackend) -> ToolSpec:
    """The designer's tool for the code of an experiment: one its session proposed, or
    an accepted plan that came without code."""
    project = ctx.project

    def target(ref: Any) -> ResearchPage:
        path = project.resolve(str(ref or "")) if isinstance(ref, str) else None
        page = project.page(path) if path else None
        if page is None or page.kind != "experiment" or not page.id:
            raise ToolError(
                f"No experiment matches {ref!r}. Create the experiment's page first; the "
                "tool result gives its id (E1, E2, ...)."
            )
        if _own_proposal(page, backend):
            return page
        if experiments.supplies_missing_code(project, page, backend.author):
            return page  # an accepted plan that has no code of anyone else's yet
        raise ToolError(
            f"The code of {page.id} is not yours to write: you write code for an "
            "experiment you proposed in this session, while the researcher has not decided "
            "on it, and for an accepted plan that has no code yet. Propose an experiment of "
            "your own, or describe the change you have in mind with propose_edit."
        )

    def handler(tool_input: dict[str, Any]) -> str:
        name, text = tool_input.get("name"), tool_input.get("text")
        if not isinstance(name, str) or not name.strip():
            raise ToolError("name must be the file's name, for example main.py")
        if not isinstance(text, str) or not text.strip():
            raise ToolError("text must be the whole content of the file")
        page = target(tool_input.get("experiment"))
        try:
            problems = experiments.write_code(
                project, str(page.id), {name.strip(): text}, author=backend.author
            )
        except ProjectError as exc:
            raise ToolError(str(exc)) from exc
        if not problems:
            return f"Wrote {name} for {page.id}. Its code parses and the entry point is there."
        found = "\n".join(f"- {problem}" for problem in problems)
        return (
            f"Wrote {name} for {page.id}. The check of {page.id}'s code as it now stands "
            f"found:\n{found}\nFix these before you finish (a file you have not written yet "
            "stops being a problem once it is there)."
        )

    return ToolSpec(
        name="write_code",
        description=(
            "Write one file of an experiment's code. Give the experiment's id (as the tool "
            "result said when you created its page), the file's name relative to the "
            "experiment's code directory (the entry point is main.py unless the page says "
            "otherwise; a change to the researcher's repository is the file patch.diff) and "
            "the whole text of the file. The code is parsed, never run, and the problems "
            "found are returned. Writing a file again replaces it; earlier versions are kept."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "experiment": {"type": "string", "description": "The experiment's id, such as E1."},
                "name": {"type": "string", "description": "The file's name, such as main.py."},
                "text": {"type": "string", "description": "The whole content of the file."},
            },
            "required": ["experiment", "name", "text"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _each_problem(message: str) -> list[str]:
    """The problems of a message that lists several (``...:\\n- one\\n- two``), one by one,
    each on one line: they are kept as a list on the page and shown as one."""
    _, *items = message.split("\n- ")
    return [" ".join(item.split()) for item in items if item.strip()] or [" ".join(message.split())]


def experiment_problems(ctx: AgentContext, page: ResearchPage) -> list[str]:
    """What would stop the experiment ``page`` from being approved or run; empty when sound.

    Checked by code, never by running anything: the page names what it tests, its
    plan is valid, its code parses and has its entry point, and the backend the
    project would use can run it.  A warning from the code check (something the
    person weighs at approval) is not a problem here.
    """
    problems: list[str] = []
    if not page.relations.get("tests"):
        problems.append(
            "The page names no hypothesis in `tests`; an experiment says which hypotheses "
            "it tests, for example `tests: [H1]`."
        )
    plan = None
    try:
        plan = experiments.load_plan(ctx.project, page.path)
    except ProjectError as exc:
        problems += _each_problem(str(exc))
    try:
        problems += experiments.split_problems(experiments.check_code(ctx.project, str(page.id)))[0]
    except ProjectError as exc:
        problems.append(str(exc))
    backend = ctx.backend(plan) if plan is not None else None
    if backend is not None and plan is not None:
        problems += backend.supports(experiments.job_spec(plan))
    return list(dict.fromkeys(problems))


def _compute_text(ctx: AgentContext) -> str:
    """What the designer is told about where experiments run."""
    backend = ctx.backend()
    name = ctx.project.settings.compute
    if backend is None:
        return (
            f"Runs go to the `{name}` backend unless the researcher chooses another when "
            "approving a run."
        )
    jobs = backend.describe().get("jobs") or {}
    features = [
        f"GPU: {'yes' if jobs.get('gpu') else 'no'}",
        "packages: installed on request"
        if jobs.get("packages")
        else "packages: only what is already installed where it runs",
        "network: blocked unless approved"
        if jobs.get("network_isolation")
        else "network: cannot be blocked here, so the code itself must not use it",
    ]
    summary = str(jobs.get("summary") or "").strip()
    lines = [f"Backend: {backend.name or name}. {summary}".strip(), *[f"- {f}" for f in features]]
    if not jobs.get("supported", True):
        lines.append("- This backend does not run experiment jobs at all.")
    return "\n".join(lines)


def _named_in_requests(ctx: AgentContext) -> set[str]:
    """Ids of the hypotheses the person's requests name, by target or in their text."""
    named: set[str] = set()
    for request in ctx.requests:
        named.update(match.upper() for match in _ID_IN_TEXT.findall(request.text))
        if request.target:
            page_id = (
                ctx.project.page(request.target).id if ctx.project.resolve(request.target) else None
            )
            if page_id:
                named.add(page_id)
    return named


def _to_design(ctx: AgentContext) -> list[dict[str, Any]]:
    """The accepted hypotheses that want an experiment: the open ones, and those a request names."""
    project = ctx.project
    named = _named_in_requests(ctx)
    testing: dict[str, list[str]] = {}
    for experiment in project.pages(kind="experiment", review="accepted"):
        for target in experiment.relations.get("tests", []):
            path = project.resolve(target) or target
            testing.setdefault(path, []).append(f"{experiment.id} ({experiment.state})")
    return [
        {"id": page.id, "title": page.title, "experiments": testing.get(page.path, [])}
        for page in project.pages(kind="hypothesis", review="accepted")
        if page.state == "open" or page.id in named
    ]


def _design_limits(project: Project, spent: Mapping[str, float]) -> dict[str, Any]:
    """The limits a design has to fit: the longest one run may take (a cap on each run,
    not a total) and what is left of the project's total for compute."""
    limits = project.settings.limits
    return {
        "max_experiments": int(limits.max_experiments),
        "experiment_seconds": float(limits.experiment_seconds),
        "experiment_usd": float(limits.experiment_usd),
        "usd_left": max(0.0, limits.experiment_usd - spent.get("experiment_usd", 0.0)),
    }


def _checked(ctx: AgentContext, session: _Session) -> dict[str, list[str]]:
    """Problems of every experiment this session proposed, by id; sound ones are left out."""
    found = {
        str(page.id): experiment_problems(ctx, page)
        for page in session.proposed("experiment")
        if page.id
    }
    return {name: problems for name, problems in found.items() if problems}


def _record_problems(
    ctx: AgentContext, session: _Session, problems: Mapping[str, list[str]]
) -> list[str]:
    """Write what is still wrong onto each experiment's page; return notes for the checkpoint."""
    notes: list[str] = []
    for page in session.proposed("experiment"):
        found = problems.get(str(page.id))
        if found or page.meta.get("problems"):
            # (A page the person deleted while the session ran has nothing to flag.)
            with contextlib.suppress(NotFound):
                ctx.project.update_page(
                    page.path, author="system", fields={"problems": found or None}
                )
        if found:
            notes.append(
                f"{page.id} cannot be approved as it stands; its page lists why: "
                f"{_short(found[0], 200)}" + (f" (and {len(found) - 1} more)" if found[1:] else "")
            )
    return notes


def run_designer(ctx: AgentContext, *, spent: Mapping[str, float] | None = None) -> AgentReport:
    """Propose experiments, with code, for the hypotheses that want one.

    Every experiment the session proposed is then checked by code, and the same
    conversation is asked to put right what the check found, a bounded number of
    times.  What still fails stays a proposal with the problems listed on its page
    (the ``problems`` field).  ``spent`` is what the project has spent on
    experiments so far, so that the designer is told what is left of the cost limit.
    """
    project = ctx.project
    root = _repo_root(project)

    def tools(backend: _SessionBackend) -> list[ToolSpec]:
        return [write_code_tool(ctx, backend), *(repo_tools(root) if root else [])]

    limit = int(project.settings.limits.max_experiments)
    session = _Session(ctx, "designer", tools=tools, caps={"experiment": limit})
    waiting = [
        {"id": page.id, "title": page.title}
        for page in project.pages(kind="experiment", review="accepted", state="planned")
        if _lacks_code(project, page)
    ]
    pending, rejected = _earlier(ctx, "designer")
    session.send(
        prompts.designer_message(
            idea=_idea(project),
            hypotheses=_to_design(ctx),
            plans_without_code=waiting,
            compute=_compute_text(ctx),
            limits=_design_limits(project, spent or {}),
            repo=str(root) if root else None,
            pending=pending,
            rejected=rejected,
            requests=_request_items(ctx),
            instruction=ctx.instruction,
        )
    )
    problems = _checked(ctx, session)
    for attempt in range(DESIGN_REPAIRS):
        if not problems:
            break
        session.send(prompts.designer_repair_message(problems, last=attempt == DESIGN_REPAIRS - 1))
        problems = _checked(ctx, session)
    notes = _record_problems(ctx, session, problems)
    produced = session.created()
    return AgentReport(
        produced=produced, summary=f"The designer proposed {_counted(produced)}.", notes=notes
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

#: Code shown to the repair session; an experiment is a few small files.
REPAIR_MAX_CODE_CHARS = 200_000
REPAIR_MAX_LOG_CHARS = 20_000


class RepairFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="The file's name, as it is in the experiment's code.")
    text: str = Field(description="The whole corrected content of the file.")


class RepairAnswer(BaseModel):
    """What the runner's repair session returns."""

    model_config = ConfigDict(extra="ignore")

    explanation: str = Field(
        description="One or two plain sentences: what was wrong and what was changed, or why "
        "this is not a programming error and nothing was changed."
    )
    files: list[RepairFile] = Field(
        default_factory=list,
        description="Every changed file in full. Empty when nothing should be changed.",
    )


def _safe_name(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return bool(parts) and not name.startswith("/") and ".." not in parts and "\0" not in name


def _corrected(answer: RepairAnswer, current: Mapping[str, str]) -> dict[str, str]:
    """The files of ``answer`` that really differ from the code as it was run."""
    return {
        item.name.strip(): item.text
        for item in answer.files
        if _safe_name(item.name.strip())
        and item.text.strip()
        and item.text != current.get(item.name.strip())
    }


def make_repair(
    ctx: AgentContext,
) -> Callable[[Any, Mapping[str, Any], str], dict[str, str] | None]:
    """The repair callback of ``research.experiments.run``.

    ``repair(plan, run_record, log_tail)`` asks the coding model for corrected
    files after a run failed for a programming error.  It returns the files that
    change, whole, or ``None`` when the model changes nothing or could not be
    asked; what it says about the repair goes to the activity feed.  It never
    raises for a model failure: the run has already failed, and stays failed.
    """
    project = ctx.project

    def ask(plan: Any, record: Mapping[str, Any], log_tail: str, files: Mapping[str, str]) -> Any:
        shown: dict[str, str] = {}
        used = 0
        for name, text in files.items():
            if used + len(text) > REPAIR_MAX_CODE_CHARS:
                shown[name] = "[not shown: the experiment's code is too large to show in full]"
                continue
            shown[name] = text
            used += len(text)
        message = prompts.repair_message(
            experiment=str(plan.id),
            plan=project.page(plan.path).body,
            files=shown,
            entrypoint=str(plan.entrypoint),
            record=record,
            log_tail=str(log_tail or "")[-REPAIR_MAX_LOG_CHARS:],
        )
        return ctx.harness.generate(
            message,
            agent="runner",
            stage=ctx.phase,
            model=model_for(project, "runner"),
            system=prompts.RUNNER_SYSTEM,
            schema=RepairAnswer,
            memory=MemoryPolicy.none(),
            effort=project.settings.models.effort,  # type: ignore[arg-type]
        ).parsed

    def repair(plan: Any, run_record: Mapping[str, Any], log_tail: str) -> dict[str, str] | None:
        experiment = str(plan.id)
        try:
            ctx.guard()
            current = experiments.code_files(project, experiment)
            answer = ask(plan, run_record, log_tail, current)
            if not isinstance(answer, RepairAnswer):
                answer = RepairAnswer.model_validate(answer)
        except Exception as exc:  # the run has failed already; a failed repair must not hide that
            ctx.activity(
                f"No repair of {experiment} was attempted: {_short(exc, 200)}", agent="runner"
            )
            return None
        files = _corrected(answer, current)
        said = _short(answer.explanation, 300) or "no explanation was given"
        if not files:
            ctx.activity(f"The runner left {experiment} as it is: {said}", agent="runner")
            return None
        ctx.activity(
            f"The runner corrected {_listed(sorted(files))} of {experiment}: {said}",
            agent="runner",
            refs=[str(plan.path)],
        )
        return files

    return repair


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------


def _result_items(project: Project, results: Sequence[ResearchPage]) -> list[dict[str, Any]]:
    items = []
    for result in results:
        experiment = next(iter(result.relations.get("run_of", [])), "")
        tests: list[str] = []
        if experiment and project.resolve(experiment):
            tests = project.page(experiment).relations.get("tests", [])
        items.append(
            {
                "path": virtual_path(result.path),
                "experiment": experiment or "an experiment that is not named",
                "state": result.state or "state unknown",
                "tests": tests,
            }
        )
    return items


def _unsupported(ctx: AgentContext, session: _Session) -> dict[str, list[str]]:
    """Numbers in the session's findings that no record, source or input holds, by finding."""
    flagged: dict[str, list[str]] = {}
    for finding in session.proposed("finding"):
        numbers = experiments.unverified_numbers(ctx.project, finding)
        if numbers:
            flagged[str(finding.id)] = list(numbers)
    return flagged


def _record_unsupported(
    ctx: AgentContext, session: _Session, flagged: Mapping[str, list[str]]
) -> list[str]:
    """Flag what is still unsupported on each finding (``unverified``); notes for the checkpoint."""
    notes: list[str] = []
    for finding in session.proposed("finding"):
        numbers = flagged.get(str(finding.id))
        if numbers or finding.meta.get("unverified"):
            with contextlib.suppress(NotFound):
                ctx.project.update_page(
                    finding.path, author="system", fields={"unverified": numbers or None}
                )
        if numbers:
            notes.append(
                f"{finding.id} states {_plural(len(numbers), 'number')} that no result record, "
                f"source or input supports: {', '.join(numbers[:8])}"
                + (" and more." if len(numbers) > 8 else ".")
            )
    return notes


def run_analyst(ctx: AgentContext, *, results: Sequence[ResearchPage], journal: str) -> AgentReport:
    """Interpret the new result records, propose verdicts and next steps, write the journal.

    ``results`` are the records not analysed yet; ``journal`` is the name of the
    round's journal page (``round-2``).  Afterwards the numbers in each new finding
    are checked against the records, and the same conversation is asked to correct
    or remove what nothing supports, a bounded number of times; what remains is
    written to the finding's ``unverified`` field and reported.
    """
    project = ctx.project
    limits = project.settings.limits
    caps = {"hypothesis": int(limits.max_hypotheses), "experiment": int(limits.max_experiments)}
    session = _Session(ctx, "analyst", caps=caps)
    pending, rejected = _earlier(ctx, "analyst")
    journal_path = f"{SECTIONS['journal']}/{journal}.md"
    session.send(
        prompts.analyst_message(
            idea=_idea(project),
            results=_result_items(project, results),
            journal_path=virtual_path(f"wiki/{journal_path}"),
            round=ctx.round,
            max_hypotheses=caps["hypothesis"],
            max_experiments=caps["experiment"],
            pending=pending,
            rejected=rejected,
            requests=_request_items(ctx),
            instruction=ctx.instruction,
        )
    )
    flagged = _unsupported(ctx, session)
    for attempt in range(NUMBER_REPAIRS):
        if not flagged:
            break
        session.send(prompts.numbers_message(flagged, last=attempt == NUMBER_REPAIRS - 1))
        flagged = _unsupported(ctx, session)
    notes = _record_unsupported(ctx, session, flagged)
    if not project.resolve(f"wiki/{journal_path}"):
        notes.append(f"The analyst wrote no journal page for round {ctx.round}.")
    produced = session.created()
    return AgentReport(
        produced=produced,
        summary=f"The analyst read {_plural(len(results), 'new result')} and proposed "
        f"{_counted(produced)}.",
        notes=notes,
    )
