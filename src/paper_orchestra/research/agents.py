"""The roles of a research project: intake, scout, theorist, designer, runner, analyst.

Each role is one function that runs its session through the context harness.
A session sees the wiki through :class:`~paper_orchestra.research.access.ProjectMemoryBackend`,
so what it may read and write is decided by the project, not by its prompt: it
proposes pages of the kinds its role may create, and suggests every other
change with ``propose_edit``.  What a role adds here is its tools (the scout's
``verify_paper``, the designer's ``write_code`` and read-only view of the
person's repository), the checks that code runs on what the model produced
(an experiment must parse; a finding's numbers are looked for on record), and
the bounded conversation in which the model is asked to put right what a check
found.

What a session is told beyond the index of the wiki: the person's requests that
are addressed to its role and the ones for every agent, the undecided proposals
of other sessions of the kinds it writes (by title and summary, so that nothing
is proposed twice), what the person rejected (pages, and suggested edits with
their reasons), and for the scout that follows intake the pages intake wrote,
which it may read before the person has reviewed them.

The engine calls these functions; they never decide what phase comes next.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import os
import re
import unicodedata
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
from paper_orchestra.compute.local import job_model_access
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
    Usage,
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
    FOR_ALL,
    IDEA_PATH,
    REQUEST_ROLES,
    SECTIONS,
    EditProposal,
    NotFound,
    ProjectError,
    Request,
    ResearchPage,
    agent_author,
    creatable_kinds,
    kind_for_path,
    moved_on,
    role_author,
    role_of,
)
from paper_orchestra.research.project import INPUTS_DIR, Project
from paper_orchestra.research.sources import SourceVerifier, project_cutoff, verify_paper_tool
from paper_orchestra.utils.pdf import load_paper, page_count
from paper_orchestra.utils.scholar import SemanticScholar

__all__ = [
    "MODEL_ROLES",
    "AgentContext",
    "AgentReport",
    "Inputs",
    "RepairAnswer",
    "SessionStopped",
    "addressee",
    "build_harness",
    "count_of",
    "describe_event",
    "experiment_problems",
    "gpu_has_price",
    "make_repair",
    "model_for",
    "model_need",
    "named",
    "need_sentence",
    "priced_gpus",
    "read_inputs",
    "repo_tools",
    "run_analyst",
    "run_designer",
    "run_intake",
    "run_scout",
    "run_theorist",
    "said",
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
#: How many pages, and how many suggested edits, of others a session is told about (the
#: latest ones): enough to keep it from proposing something twice, without crowding out
#: its task.
MAX_TOLD = 60

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
#: A name or a path that can stand in a sentence as it is: it holds no space and no
#: punctuation a sentence could end or begin with.
_PLAIN_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_/-]*(?:\.[A-Za-z0-9_/-]+)*")
#: What a reader can take for a quotation mark besides the marks themselves (the Unicode
#: categories Pi and Pf): straight marks and the accents typed for them, low marks, primes,
#: modifier letters, Hebrew and CJK marks, ornaments.  Any of them could seem to end the
#: quotation a model's words are given in.
_LIKE_QUOTATION_MARKS = frozenset(
    "\"'`\u00b4\u201a\u201e"
    "\u2032\u2033\u2034\u2035\u2036\u2037\u2057"
    "\u02b9\u02ba\u02bb\u02bc\u02bd\u02c8\u02ca\u02cb\u02dd\u02ee"
    "\u05f3\u05f4\u3003\u301d\u301e\u301f"
    "\u275b\u275c\u275d\u275e\u275f\u2760\u2e42\uff02\uff07"
    "\U0001f676\U0001f677\U0001f678"
)
_APOSTROPHES = re.compile("'{2,}")
#: How a model request ended when the client goes on to the next request of the turn.
#: After ``tool_use`` it runs the tools that were called first; the other two lead
#: straight to the next request (a paused server tool loop is continued, an answer cut
#: off at the default length is asked for again).
_TURN_GOES_ON = ("tool_use", "pause_turn", "max_tokens")


# ---------------------------------------------------------------------------
# What a session is given, and what it leaves behind
# ---------------------------------------------------------------------------


def _no_guard() -> None:
    return None


def _never() -> bool:
    return False


def _within_budget(model: str, usage: Usage, **more: Any) -> BaseException | None:
    return None


class SessionStopped(BaseException):
    """A session was ended between two model calls of one turn.

    Not an ``Exception`` on purpose.  A model client answers an exception raised by
    a tool with an error message to the model and carries on with the turn; a stop
    has to end it, the way an interrupt does.  Whoever runs the phase records it as
    interrupted; what the session wrote until then stays, and a resumed session
    finds it.

    ``reason`` is ``None`` for a stop request.  When the turn was ended because it
    took the project past a budget, it is what ``AgentContext.over_budget``
    returned, so that whoever runs the phase can say so instead of "stopped".
    """

    def __init__(self, message: str = "", *, reason: BaseException | None = None) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class AgentContext:
    """What a role's session is given by whoever runs the phase.

    ``session`` is the stem of the session id: the same for every attempt at one
    run of a phase, so that a resumed session sees and revises what it proposed
    before it was interrupted, and different for any other run.  ``guard`` is
    called before every session and every turn and raises when the work must
    stop (a stop request, a spent budget).  ``stopping`` says whether a stop was
    requested; it is asked before every tool call and after every model request
    that leads to another, so that a long turn ends before its next model request
    instead of running to its end.  ``over_budget`` is asked after every model
    request of a turn with the model and what the turn has used so far (which the
    harness's ledger learns only when the turn ends); it returns why the work must
    stop, or ``None``, and a turn it stopped ends where a stop request ends one
    (it is also given ``more_usd=`` when helpers of the turn worked on other models).
    ``compute`` is a backend, or a
    callable choosing one for a plan, as ``research.experiments.run`` takes it.
    ``handover`` names the roles whose undecided pages of this round the session
    may read although the person has not reviewed them (the scout reads what
    intake wrote right before it).  ``produced`` fills with the paths of the
    pages this run of the phase created, as they are created, so that an
    interrupted phase can still say what it left behind.
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
    over_budget: Callable[..., BaseException | None] = _within_budget
    scholar: SemanticScholar | None = None
    compute: ComputeBackend | Callable[[Any], ComputeBackend] | None = None
    handover: Sequence[str] = ()
    produced: list[str] = field(default_factory=list)
    #: The ``sdk`` runtime (``research.sdk.session.AgentRuntime``) when the project's
    #: roles are Claude Agent SDK agents; ``None`` for the classic tool loop.
    runtime: Any = None
    #: What is left of the model budget, in dollars, for a session that can be told a
    #: sum it must not pass; ``None`` when nobody says.
    left: Callable[[], float | None] = _no_guard
    #: The sessions this run of the phase opened that hold something to close (an SDK
    #: agent's process); whoever runs the phase closes them when it ends.
    sessions: list[Any] = field(default_factory=list, repr=False)
    #: Why ``over_budget`` said the work must stop, once it has.
    _spent: BaseException | None = field(default=None, init=False, repr=False)

    def session_id(self, role: str) -> str:
        return f"{self.session}-{role}"

    def used(self, model: str, usage: Usage, *, more_usd: float = 0.0) -> None:
        """Take note of what the turn in flight has used after one more model request.

        ``more_usd`` is what the turn has cost besides, already priced: the requests
        of helpers that work on another model than ``model``.

        Nothing is raised here.  A turn that is past the budget ends at
        :meth:`heed_stop`, which the session asks before each of its tool calls and
        after each model request that leads to another.
        """
        if self._spent is None:
            self._spent = (
                self.over_budget(model, usage, more_usd=more_usd)
                if more_usd
                else self.over_budget(model, usage)
            )

    def ending(self) -> bool:
        """Whether the turn in flight ends the next time :meth:`heed_stop` is asked."""
        return self._spent is not None or self.stopping()

    def heed_stop(self) -> None:
        """Raise :class:`SessionStopped` when a stop was requested or a budget is spent."""
        if self.stopping():
            raise SessionStopped("the phase was asked to stop")
        if self._spent is not None:
            raise SessionStopped(str(self._spent), reason=self._spent)

    def budget_left(self) -> float | None:
        """What is left of the model budget in dollars (never below zero), or ``None``."""
        left = self.left()
        return None if left is None else max(0.0, float(left))

    def close_sessions(self) -> None:
        """End what the sessions of this run of the phase still hold (agent processes)."""
        sessions, self.sessions[:] = list(self.sessions), []
        for session in sessions:
            with contextlib.suppress(Exception):
                session.close()

    def note(self, paths: Iterable[str]) -> None:
        """Remember pages this run of the phase created."""
        self.produced += [path for path in paths if path not in self.produced]

    def activity(
        self,
        summary: str,
        *,
        agent: str,
        refs: Iterable[str] = (),
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """One line for the activity feed about what an agent is doing.  ``data`` says
        the same in parts, for a reader that sets a command apart from the sentence."""
        self.project.emit(
            "agent.activity",
            summary,
            phase=self.phase,
            agent=agent,
            refs=refs,
            data=dict(data) if data else None,
            round=self.round,
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


#: How each provider's credentials are set, for the one sentence that says what to do.
_CREDENTIALS = {
    "anthropic": (
        "Anthropic",
        "set ANTHROPIC_API_KEY or the Workload Identity Federation variables",
    ),
    "gemini": ("Gemini", "set GEMINI_API_KEY (or VERTEX_AI_PROJECT and VERTEX_AI_LOCATION)"),
    "openai": ("OpenAI", "set OPENAI_API_KEY"),
}
#: Where the person finds out what is configured, said once at the end of that sentence.
_WHERE_TO_LOOK = "Setup in the app and `paper-orchestra doctor` show what is configured"


@dataclass(frozen=True)
class _Fault:
    """Why the model of one setting (``settings.models.<setting>``) cannot serve the agents."""

    setting: str
    model: str
    #: What is wrong with it, as the rest of "the model set for ... (<model>) ...".
    says: str
    #: What the person does about it, as the rest of a sentence.
    fix: str


def _first_sentence(text: Any) -> str:
    """The first sentence of a provider's error, as a clause: on one line, without its
    full stop, and beginning in lower case unless it begins with a name."""
    line = str(text).strip().split("\n", 1)[0].split(". ", 1)[0].rstrip(".: ")
    line = _short(line, 200)
    return line[:1].lower() + line[1:] if line[1:2].islower() else line


def _unbuilt(project: Project, setting: str, model: str, exc: LLMConfigError) -> _Fault:
    """The fault of a model whose client could not be made.

    No credential is by far the most common reason, and the provider explains it
    in several lines that suit a terminal.  Here it is one clause and one thing to
    do; what else can be wrong (a package that is not installed, a model no
    provider serves) is said in the first sentence of the provider's own error.
    """
    from paper_orchestra import doctor
    from paper_orchestra.llm.router import provider_for_model

    change = f"put that right or change settings.models.{setting}"
    try:
        provider = provider_for_model(model)
    except LLMConfigError:
        provider = ""
    missing_package = isinstance(exc.__cause__, ImportError)
    no_credential = provider in _CREDENTIALS and not missing_package
    if no_credential and doctor.environment_credential(provider, os.environ) is None:
        name, fix = _CREDENTIALS[provider]
        return _Fault(
            setting, model, f"cannot be used because no {name} credentials were found", fix
        )
    # A library may quote what it was configured with: no credential gets into a sentence.
    why = experiments.mask_credentials(project, _first_sentence(exc))
    return _Fault(setting, model, f"cannot be used because {why}", change)


def _fault(project: Project, setting: str, client_for: Callable[[str], Any]) -> _Fault | None:
    model = str(project.settings.models[setting])
    try:
        capabilities: Capabilities = client_for(model).capabilities
    except LLMConfigError as exc:
        return _unbuilt(project, setting, model, exc)
    missing = [
        name
        for name, has in (("tools", capabilities.tools), ("memory tool", capabilities.memory_tool))
        if not has
    ]
    if not missing:
        return None
    # The agents read and write the wiki through tools and the memory tool.
    return _Fault(
        setting,
        model,
        f"offers no {' and no '.join(missing)}",
        f"set settings.models.{setting} to a model that has "
        f"{'both' if len(missing) > 1 else 'it'} (for example {DEFAULT_MODEL})",
    )


def model_need(
    project: Project,
    client_for: Callable[[str], Any],
    *,
    settings: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """What keeps the research agents from working with the project's models, for
    whoever has to say so; ``None`` when they can work::

        {"needs": "model", "settings": ["research", "reasoning", "coding"],
         "models": ["claude-fable-5-2"],
         "why": "the model set for research, reasoning and coding (claude-fable-5-2) "
                "cannot be used because no Anthropic credentials were found",
         "fix": "set ANTHROPIC_API_KEY or the Workload Identity Federation variables"}

    ``client_for`` is how a model's client is found (``ContextHarness.client_for``,
    ``llm.router.get_client``).  A model cannot be used when no client can be made
    for it (no credential, most of the time) or when it offers no tools or no
    memory tool, through which the agents read and write the wiki.  ``settings``
    are the entries of ``settings.models`` that are asked about: all that an
    agent works with, unless a caller needs one role's alone.

    Finding this out contacts nothing and spends nothing: a client is made from
    the environment and local files, and asked what it offers.
    :func:`need_sentence` says the answer in one sentence.
    """
    asked = list(dict.fromkeys(settings or MODEL_ROLES.values()))
    found: dict[str, Any] = {}

    def once(model: str) -> Any:
        """The client of ``model``, or the refusal, looked for once however many roles
        share the model: a client that cannot be made costs as much at every try."""
        if model not in found:
            try:
                found[model] = client_for(model)
            except LLMConfigError as exc:
                found[model] = exc
        if isinstance(found[model], LLMConfigError):
            raise found[model]
        return found[model]

    faults = [fault for name in asked if (fault := _fault(project, name, once)) is not None]
    if not faults:
        return None
    groups: dict[tuple[str, str], list[str]] = {}
    for fault in faults:
        groups.setdefault((fault.model, fault.says), []).append(fault.setting)
    why = [
        f"the model set for {_listed(names)} ({model}) {says}"
        for (model, says), names in groups.items()
    ]
    return {
        "needs": "model",
        "settings": [fault.setting for fault in faults],
        "models": list(dict.fromkeys(fault.model for fault in faults)),
        "why": ", and ".join(why),
        "fix": ", and ".join(dict.fromkeys(fault.fix for fault in faults)),
    }


def need_sentence(need: Mapping[str, Any], what: str) -> str:
    """The one sentence about models that cannot be used (``need``, from
    :func:`model_need`): ``what`` (``The analysis phase``, ``A repair``) needs a model,
    which model set for which roles cannot be used and why, and what to do.

    It names every role concerned once and says nothing about a command or a
    button: whoever shows it (the command line, the app, a checkpoint) adds how
    to go on from where the person is.
    """
    return f"{what} needs a model, and {need['why']}: {need['fix']} ({_WHERE_TO_LOOK})."


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
        handover: Sequence[str] = (),
    ) -> None:
        super().__init__(project, role, round, session_id)
        self._caps = dict(caps or {})
        #: Called before every command of the memory tool, which is where a turn that
        #: reads and writes many pages can be ended between two model calls.
        self._heed_stop = heed_stop
        #: Authors whose undecided pages of this round the session may read.
        self._handover = frozenset(role_author(name) for name in handover)

    def proposed(self, kind: str | None = None) -> list[ResearchPage]:
        """Pages (of ``kind``) this session created, in this attempt or one it resumes."""
        return [
            page
            for page in self.project.pages(kind=kind)
            if page.author == self.author and page.session == self.session_id
        ]

    def _handed_over(self, page: ResearchPage) -> bool:
        return (
            page.review == "proposed" and page.author in self._handover and page.round == self.round
        )

    def handed_over(self) -> list[ResearchPage]:
        """The undecided pages this session may read without their being its own."""
        return [page for page in self.project.pages(review="proposed") if self._handed_over(page)]

    def _mode(self, page: ResearchPage) -> str:
        # Readable, and no more: what was handed over is not the session's to change.
        return "full" if self._handed_over(page) else super()._mode(page)

    def _refuse_change(self, page: ResearchPage, path: str) -> None:
        if self._handed_over(page):
            raise ToolError(
                f"{page.id or path} was written by the {role_of(page.author)} agent and has "
                "not been reviewed by the researcher yet. You can read it, but it is not "
                "yours to change, and an edit cannot be suggested to a page that is still "
                "undecided: write what you have to add in a page of your own."
            )
        super()._refuse_change(page, path)

    def _index(self) -> str:
        """The wiki's index, with what was handed over listed apart: it is no part of
        what the person accepted."""
        index = super()._index()
        pages = self.handed_over()
        if not pages:
            return index
        lines = [
            "",
            f"## not yet reviewed by the researcher ({len(pages)})",
            "Written just before your session. You can open these pages; the researcher "
            "may still edit or reject them.",
        ]
        for page in pages:
            tags = ", ".join(tag for tag in (page.id, "not yet reviewed") if tag)
            # (Another agent's words that nobody has reviewed: they stay a line of the index.)
            summary = f": {prompts.inert_line(page.summary, 140)}" if page.summary else ""
            title = prompts.inert_line(page.title, 100)
            lines.append(f"- {page.path.removeprefix('wiki/')} [{tags}] {title}{summary}")
        return index.rstrip("\n") + "\n" + "\n".join(lines) + "\n"

    def index(self) -> str:
        """The index a session of this backend is opened with."""
        return self._index()

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

    def rename(self, old_path: str, new_path: str) -> str:
        # Refused by the project whatever is asked; a turn that must end ends here too.
        self._heed_stop()
        return super().rename(old_path, new_path)

    def _refuse_beyond_limit(self, kind: str | None) -> None:
        """Raise when one more new page of ``kind`` would pass the phase's limit."""
        cap = self._caps.get(kind or "")
        if cap is None:
            return
        have = len(self.proposed(kind or ""))
        if have >= cap:
            raise ToolError(
                f"This phase proposes at most {count_of(cap, kind or '')}, and {have} are "
                "proposed. Revise or delete one of yours instead of adding another; the "
                "researcher can raise the limit in the project's settings."
            )

    def create(self, path: str, file_text: str) -> str:
        self._heed_stop()
        rel = self._rel(path)
        # Writing to a page this session can see revises it (or is refused, when it is
        # not the session's own); anything else becomes a new page and counts.  Whether a
        # file lies at the path decides nothing: one the session cannot see, such as
        # another session's undecided proposal, is not replaced but joined by a new
        # page, and the answer must not tell whether it is there.
        if rel and rel not in self._seen():
            self._refuse_beyond_limit(kind_for_path(rel))
        return super().create(path, file_text)

    def create_page(self, kind: str, title: str, body: str, **options: Any) -> ResearchPage:
        """A page written in code on the session's behalf is a new page like any other."""
        self._heed_stop()
        self._refuse_beyond_limit(kind)
        return super().create_page(kind, title, body, **options)


def _stoppable(tool: ToolSpec, heed_stop: Callable[[], None]) -> ToolSpec:
    """``tool`` with a stop request heeded before each call."""

    def handler(tool_input: dict[str, Any]) -> str:
        heed_stop()
        return tool.handler(tool_input)

    return replace(tool, handler=handler)


def _whole(value: Any) -> int:
    """A count as a provider's client reported it; nothing for what is not a count."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _short(text: Any, limit: int = 120) -> str:
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _plain_marks(text: str) -> str:
    """``text`` with whatever reads as a quotation mark, alone or in a row, made one
    apostrophe: inside a quotation nothing may look like its end."""
    plain = "".join(
        "'" if ch in _LIKE_QUOTATION_MARKS or unicodedata.category(ch) in ("Pi", "Pf") else ch
        for ch in text
    )
    return _APOSTROPHES.sub("'", plain)


def said(text: Any, limit: int = 120) -> str:
    """Words a model chose, as they stand in a sentence the project says to the person.

    In quotation marks, on one line and cut to ``limit`` characters, with quotation
    marks of their own (and what looks like one) made plain apostrophes and
    characters that are not shown (or that change the order of what is shown) left
    out.  Whatever the words say, they read as a quotation, never as a further
    sentence of the feed or of a checkpoint.
    """
    visible = "".join(
        " " if ch.isspace() else ch
        for ch in str(text if text is not None else "")
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co")
    )
    return f"“{_short(_plain_marks(visible), limit)}”"


def named(value: Any, limit: int = 80) -> str:
    """A name or a path a model gave, as it stands in such a sentence.

    A plain one (letters, digits, ``_``, ``-``, ``/`` and dots inside it) is shown as
    it is, the way a person writes ``main.py`` or ``wiki/idea.md``; anything else is
    quoted with :func:`said`.
    """
    text = str(value if value is not None else "")
    if len(text) <= limit and _PLAIN_NAME.fullmatch(text):
        return text
    return said(text, limit)


def _shown(path: Any) -> str:
    """A ``/memories`` path as a person reads it in the feed."""
    return str(path or "").removeprefix("/memories").strip("/") or "the wiki"


def _described_view(who: str, given: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The sentence about a ``view`` of the memory tool, and the path it concerns."""
    path = _shown(given.get("path"))
    if path == "the wiki":
        return f"{who} looked through the wiki.", []
    shown = named(path, 120)
    # A page or an attached file is a reference of the sentence, where what was given
    # can be a path at all: one line, and of a length a path has.
    is_file = path.endswith(".md") or path.startswith(f"{INPUTS_DIR}/")
    if is_file and len(path) <= 200 and path.isprintable():
        return f"{who} read {shown}.", [path]
    return f"{who} looked through {shown}.", []


def describe_event(role: str, event: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    """One plain sentence about a session event, with the paths it concerns; or ``None``.

    Only what a person following the work would want to see: what was read, looked
    up and checked.  Pages written already announce themselves through the project.
    What the model chose (a path, a query, a title, the name of a tool) stands in
    the sentence as a plain name or as a quotation, cut to a length, so that it can
    never read as a further sentence of the feed.
    """
    who = f"The {role}"
    if event.get("type") == "usage":
        searches = _whole(event.get("web_searches"))
        if searches:
            plural = "es" if searches != 1 else ""
            return f"{who} searched the web ({searches} search{plural}).", []
        return None
    if event.get("type") != "tool_call":
        return None
    name = event.get("name")
    given = event.get("input")
    given = given if isinstance(given, Mapping) else {}
    # Every argument is the model's: it is shown as a name or as a quotation, never as
    # words of the sentence it stands in.
    if name == "memory":
        return _described_view(who, given) if given.get("command") == "view" else None
    if name == "wiki_search":
        return f"{who} searched the wiki for {said(given.get('query'), 80)}.", []
    if name == "verify_paper":
        return f"{who} checked {said(given.get('title'), 100)} against Semantic Scholar.", []
    if name == "repo_search":
        return f"{who} searched for {said(given.get('query'), 80)} in your repository.", []
    if name in ("repo_list", "repo_read"):
        what = named(given["path"], 80) if given.get("path") else "the top level"
        verb = "listed" if name == "repo_list" else "read"
        return f"{who} {verb} {what} in your repository.", []
    if name in ("propose_edit", "name_project", "write_code"):
        # What came of these is told where it happens (the project's own events, the
        # tool once the file is written): a call that was refused did nothing.
        return None
    return f"{who} used {named(name, 40)}.", []


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
            project,
            role,
            ctx.round,
            ctx.session_id(role),
            caps=caps,
            heed_stop=ctx.heed_stop,
            handover=ctx.handover,
        )
        if callable(tools):
            tools = tools(self.backend)
        #: Edit proposals that were on file before this attempt: what comes after is its own.
        self._known_edits = {proposal.id for proposal in project.proposals(review=None)}
        offered = [search_tool(project), propose_edit_tool(project, role, ctx.round), *tools]
        index = self.backend.index()
        self._last = ""
        self._model = model_for(project, role)
        #: What the turn in flight has used, request by request.  The harness adds a
        #: turn's usage to its ledger when the turn ends; a budget has to see it before.
        self._turn = Usage()
        #: Whether the model request that was reported last leads to another one.
        self._goes_on = False
        self.conversation = ctx.harness.open(
            agent=role,
            stage=ctx.phase,
            model=self._model,
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
        """Follow the session: count what it uses, tell the feed what it does, and end a
        turn that must end before its next model request.

        A client reports how each model request ended and then what it used.  Where
        the turn goes on from there, a stop request or a spent budget ends it here,
        by :class:`SessionStopped`, which no client and no harness catches.  Waiting
        for the next tool call of the session would not do: after a paused server
        tool loop and after an answer cut off at the default length the client sends
        the next request with no tool call in between, and it answers a call of an
        unknown tool, or one whose input is not valid, without running anything.
        """
        kind = event.get("type")
        if kind == "assistant":
            self._goes_on = event.get("stop_reason") in _TURN_GOES_ON
        if kind == "usage":
            self._count(event)
        if kind == "tool_call" and self.ctx.ending():
            return  # the call is announced, but the turn ends before it is made
        described = describe_event(self.role, event)
        # The same page opened twice in a row is one thing to tell.
        if described is not None and described[0] != self._last:
            self._last = described[0]
            self.ctx.activity(described[0], agent=self.role, refs=described[1])
        if kind == "usage" and self._goes_on:
            self._goes_on = False
            self.ctx.heed_stop()

    def _count(self, event: Mapping[str, Any]) -> None:
        """Add the usage of one model request to the turn's, and let the budget see it."""
        self._turn.add(Usage(**{name: _whole(event.get(name)) for name in Usage().to_dict()}))
        self.ctx.used(self._model, self._turn)

    def send(self, parts: Part | Sequence[Part]) -> str:
        self.ctx.guard()
        # What earlier turns used is in the harness's ledger by now.
        self._turn = Usage()
        self._goes_on = False
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

    def suggested(self) -> list[EditProposal]:
        """The edits this attempt of the session suggested and the person has not decided."""
        return [
            proposal
            for proposal in self.ctx.project.proposals()
            if proposal.author == self.backend.author and proposal.id not in self._known_edits
        ]


def _open_session(
    ctx: AgentContext,
    role: str,
    *,
    tools: Sequence[ToolSpec] | Callable[[_SessionBackend], Sequence[ToolSpec]] = (),
    web_search: WebSearch | None = None,
    caps: Mapping[str, int] | None = None,
) -> Any:
    """The session of ``role`` in the project's runtime: the classic tool loop
    (:class:`_Session`), or a Claude Agent SDK agent with the same face
    (``research.sdk.session.AgentSession``) when the engine handed over the ``sdk``
    runtime.  Everything a role function does with its session is the same for both."""
    if ctx.runtime is None:
        return _Session(ctx, role, tools=tools, web_search=web_search, caps=caps)
    from paper_orchestra.research.sdk.session import AgentSession

    return AgentSession(ctx, role, tools=tools, web_search=web_search, caps=caps)


def _capabilities(ctx: AgentContext, role: str) -> Capabilities:
    """What the model of ``role`` takes in a message.  An SDK agent is not asked through
    a client of this package, so none is built for it (there may be no API key at all)."""
    if ctx.runtime is not None:
        return ctx.runtime.capabilities
    return ctx.harness.client_for(model_for(ctx.project, role)).capabilities


def _idea(project: Project) -> str:
    return project.page(IDEA_PATH).body


def addressee(request: Request) -> str:
    """The role a request of the person is addressed to, or ``all`` for every agent."""
    return request.role if request.role in REQUEST_ROLES else FOR_ALL


def _request_items(ctx: AgentContext, role: str) -> list[dict[str, Any]]:
    """The requests the session is shown, each marked as addressed to its role or to all."""
    return [
        {"text": request.text, "target": request.target, "mine": addressee(request) == role}
        for request in ctx.requests
    ]


def _item(page: ResearchPage) -> dict[str, Any]:
    return {
        "kind": page.kind,
        "id": page.id,
        "title": page.title,
        "summary": page.summary,
        "reason": page.decision_reason,
        "by": role_of(page.author),
        "path": virtual_path(page.path),
    }


def _rejected_edits(project: Project, role: str) -> list[dict[str, Any]]:
    """The suggested edits the person turned down that this role should know of: its own,
    and anyone's to a hypothesis or a finding (a verdict or a reading the person did
    not accept concerns every role that builds on it)."""
    author = role_author(role)
    items = []
    for proposal in project.proposals(review="rejected"):
        path = project.resolve(proposal.target)
        page = project.page(path) if path else None
        kind = page.kind if page is not None else kind_for_path(proposal.target)
        if proposal.author != author and kind not in ("hypothesis", "finding"):
            continue
        changes = [f"{name}: {_short(value, 80)}" for name, value in proposal.fields.items()]
        items.append(
            {
                "kind": kind or "page",
                "id": page.id if page is not None else None,
                "title": page.title if page is not None else proposal.target,
                "summary": proposal.summary,
                "changes": changes,
                "reason": proposal.decision_reason or "",
                "by": role_of(proposal.author),
            }
        )
    return items[-MAX_TOLD:]


def _earlier(
    ctx: AgentContext, role: str, readable: Sequence[ResearchPage] = ()
) -> dict[str, list[dict[str, Any]]]:
    """What the role must not propose twice, as the keyword arguments of its message.

    ``pending``: the undecided proposals of other sessions of the kinds this role
    writes, whoever proposed them (the analyst's next steps are hypotheses and
    experiments too).  The session cannot read them, because nothing should be
    built on a page that may still be turned down, but it is told that they are
    on the researcher's desk.  ``rejected``: the rejected pages of those kinds.
    ``rejected_edits``: suggested edits the person turned down, with the reason.
    ``readable`` are undecided pages the session may open; they are told apart.
    """
    author, own = role_author(role), ctx.session_id(role)
    kinds = creatable_kinds(role)
    opened = {page.path for page in readable}
    pending = [
        _item(page)
        for page in ctx.project.pages(kind=kinds, review="proposed")
        if not (page.author == author and page.session == own) and page.path not in opened
    ]
    rejected = [_item(page) for page in ctx.project.pages(kind=kinds, review="rejected")]
    return {
        "pending": pending[-MAX_TOLD:],
        "rejected": rejected[-MAX_TOLD:],
        "rejected_edits": _rejected_edits(ctx.project, role),
    }


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
        ctx.activity(f"The intake named the project {said(title, 80)}.", agent="intake")
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
    inputs = read_inputs(project, _capabilities(ctx, "intake"))
    untitled = project.title_is_provisional
    session = _open_session(ctx, "intake", tools=[name_project_tool(ctx)] if untitled else ())
    if ctx.runtime is not None:
        # An agent is sent text; the images and PDFs lie in its folder for its file tools.
        session.give_inputs()
    message = prompts.intake_message(
        idea=_idea(project),
        attached=inputs.attached,
        unread=inputs.unread,
        untitled=untitled,
        **_earlier(ctx, "intake"),
        requests=_request_items(ctx, "intake"),
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
    """Find prior work, record what can be verified, synthesise it, ask what is open.

    Raises ``sources.CutoffError`` before anything is asked of a model when the
    project's research cutoff cannot be read: searching without the cutoff the
    person set would look at exactly the work they meant to leave out.
    """
    project = ctx.project
    cutoff = project_cutoff(project)
    verifier = SourceVerifier(project, scholar=ctx.scholar, harness=ctx.harness, round=ctx.round)
    capabilities = _capabilities(ctx, "scout")
    session = _open_session(
        ctx,
        "scout",
        tools=[verify_paper_tool(verifier)],
        web_search=WebSearch(max_uses=WEB_SEARCHES),
    )
    handed_over = session.backend.handed_over()
    message = prompts.scout_message(
        idea=_idea(project),
        cutoff=cutoff,
        web_search=capabilities.web_search,
        sources=len(project.pages(kind="source", review=["accepted", "proposed"])),
        max_sources=int(project.settings.limits.max_sources),
        handed_over=[_item(page) for page in handed_over],
        **_earlier(ctx, "scout", handed_over),
        requests=_request_items(ctx, "scout"),
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
    session = _open_session(ctx, "theorist", caps={"hypothesis": limit})
    session.send(
        prompts.theorist_message(
            idea=_idea(project),
            max_hypotheses=limit,
            **_earlier(ctx, "theorist"),
            requests=_request_items(ctx, "theorist"),
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
        # Told here and not when the call was made: only now is it true.
        role = role_of(backend.author) or "designer"
        ctx.activity(
            f"The {role} wrote {named(name.strip(), 60)} for {page.id}.",
            agent=role,
            refs=[page.path],
        )
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
            "found are returned. Writing a file again replaces it; earlier versions are kept. "
            "A file the researcher wrote or changed is theirs and is not replaced: say what "
            "should change with propose_edit instead."
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


def priced_gpus() -> list[str]:
    """The GPU types a run's cost can be estimated for, cheapest first as the table has them."""
    from paper_orchestra.compute.modal_backend import MODAL_PRICES_USD_PER_SECOND

    return [str(name) for name in MODAL_PRICES_USD_PER_SECOND["gpu"]]


def gpu_has_price(gpu: str) -> bool:
    """Whether what a run on ``gpu`` (a type, or ``type:count``) costs can be estimated.

    Without a price nothing bounds the cost of a run, so the project's limit for
    experiments (``settings.limits.experiment_usd``) cannot hold it.
    """
    from paper_orchestra.compute.modal_backend import estimate_cost

    try:
        return estimate_cost(1.0, gpu=str(gpu), cpu=0.0, memory_mb=0) is not None
    except ValueError:  # not a GPU request the table can read: there is nothing to price
        return False


def _no_price(gpu: str) -> str:
    return (
        f"Its GPU type {named(gpu, 40)} has no price estimate, so what a run costs could not "
        "be held to the project's limit for experiments (settings.limits.experiment_usd). "
        f"Name a GPU type that has one: {', '.join(priced_gpus())}."
    )


def experiment_problems(ctx: AgentContext, page: ResearchPage, *, code: bool = True) -> list[str]:
    """What would stop the experiment ``page`` from being approved or run; empty when sound.

    Checked by code, never by running anything: the page names what it tests, its
    plan is valid, its code parses and has its entry point, the backend the
    project would use can run it, and a GPU it asks for has a price (a run whose
    cost cannot be estimated cannot be held to the project's cost limit).  A
    warning from the code check (something the person weighs at approval) is not
    a problem here.  ``code=False`` leaves the code out of it: for a plan that is
    proposed without code, which the designer writes once the plan is accepted.
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
        found = experiments.check_code(ctx.project, str(page.id)) if code else []
        problems += experiments.split_problems(found)[0]
    except ProjectError as exc:
        problems.append(str(exc))
    backend = ctx.backend(plan) if plan is not None else None
    if backend is not None and plan is not None:
        problems += backend.supports(experiments.job_spec(plan))
    # GPUs are Modal's, which bills by the second: also where that backend cannot be
    # asked here (it is not installed, or none was given), that is where a GPU plan runs.
    billed = backend is None or getattr(backend, "name", "") == "modal"
    if plan is not None and plan.gpu and billed and not gpu_has_price(plan.gpu):
        problems.append(_no_price(plan.gpu))
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
        "network: blocked unless the researcher grants it when approving the run"
        if jobs.get("network_isolation")
        else "network: cannot be blocked here, so the code itself must not use it",
    ]
    models = (backend.name or name) == "local" and job_model_access()
    if models:
        features[-1] = (
            "network: allowed for a run the researcher approves with network access "
            "(set `network: true` on the page and say why under `## Resources`)"
        )
        features.append(
            "model inference: a run approved with network access can call the Anthropic "
            "API. The `anthropic` Python package is installed; create the client with "
            "`anthropic.Anthropic()` and no arguments, because the credentials are already "
            "in the job's environment (never print or store environment variables). Models "
            "include claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5 and "
            "claude-fable-5-2. So a hypothesis about how models behave is tested on real "
            "model calls, not on simulated data. Keep the number of calls small enough for "
            "the time limit (use a thread pool, retry rate limits with backoff), write "
            "every raw response to a file under `artifacts/`, and report how many calls "
            "were made as a metric. There are no local model weights and no GPU"
        )
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
    not a total), what is left of the project's total for compute, and the GPU types
    whose cost can be estimated and so be held to that total."""
    limits = project.settings.limits
    return {
        "max_experiments": int(limits.max_experiments),
        "experiment_seconds": float(limits.experiment_seconds),
        "experiment_usd": float(limits.experiment_usd),
        "usd_left": max(0.0, limits.experiment_usd - spent.get("experiment_usd", 0.0)),
        "priced_gpus": priced_gpus(),
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


def _network_notes(session: _Session) -> list[str]:
    """Said at the checkpoint for each proposed experiment whose page asks for the network.

    ``network: true`` on a page is a request and grants nothing: the person decides it
    when they approve the run, and should know before then that they will be asked.
    """
    return [
        f"{page.id} asks for network access. Its page can only ask: you grant or refuse it "
        "when you approve the run."
        for page in session.proposed("experiment")
        if page.id and str(page.meta.get("network")).strip().lower() in ("true", "yes", "on", "1")
    ]


def run_designer(ctx: AgentContext, *, spent: Mapping[str, float] | None = None) -> AgentReport:
    """Propose experiments, with code, for the hypotheses that want one.

    Every experiment the session proposed is then checked by code, and the same
    conversation is asked to put right what the check found, a bounded number of
    times.  What still fails stays a proposal with the problems listed on its page
    (the ``problems`` field).  ``spent`` is what the project has spent on
    experiments so far, so that the designer is told what is left of the cost limit.
    The report's notes name what is not sound and what asks for the network.
    """
    project = ctx.project
    root = _repo_root(project)

    def tools(backend: _SessionBackend) -> list[ToolSpec]:
        return [write_code_tool(ctx, backend), *(repo_tools(root) if root else [])]

    limit = int(project.settings.limits.max_experiments)
    session = _open_session(ctx, "designer", tools=tools, caps={"experiment": limit})
    waiting = [
        {"id": page.id, "title": page.title}
        for page in project.pages(kind="experiment", review="accepted", state="planned")
        if _lacks_code(project, page)
    ]
    session.send(
        prompts.designer_message(
            idea=_idea(project),
            hypotheses=_to_design(ctx),
            plans_without_code=waiting,
            compute=_compute_text(ctx),
            limits=_design_limits(project, spent or {}),
            repo=str(root) if root else None,
            **_earlier(ctx, "designer"),
            requests=_request_items(ctx, "designer"),
            instruction=ctx.instruction,
        )
    )
    problems = _checked(ctx, session)
    for attempt in range(DESIGN_REPAIRS):
        if not problems:
            break
        session.send(prompts.designer_repair_message(problems, last=attempt == DESIGN_REPAIRS - 1))
        problems = _checked(ctx, session)
    notes = _record_problems(ctx, session, problems) + _network_notes(session)
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
            # (An error can carry what the model answered: it is given as a quotation too.
            # It can also quote a credential of the request that failed: that is blanked.)
            why = experiments.mask_credentials(project, str(exc))
            ctx.activity(
                f"No repair of {experiment} was attempted: {said(why, 200)}", agent="runner"
            )
            return None
        files = _corrected(answer, current)
        # The explanation and the file names are the model's: given as what it said.
        why = said(answer.explanation, 300) if answer.explanation.strip() else "it gave no reason."
        if not files:
            ctx.activity(f"The runner left {experiment} as it is: {why}", agent="runner")
            return None
        names = _listed([named(name, 60) for name in sorted(files)])
        ctx.activity(
            f"The runner corrected {names} of {experiment}: {why}",
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


@dataclass
class _Numbers:
    """What the number check made of one text the analyst wrote."""

    #: How the text is named to the analyst and to the person: ``F1``, or
    #: ``The edit suggested to F2 (ep-1a2b3c4d)``.
    label: str
    #: How many numbers were looked for.
    checked: int
    #: The ones found nowhere on record.
    missing: list[str]
    #: The ones found, but not in the records the finding names in ``based_on``.
    elsewhere: list[str]
    #: The text is a finding as a suggested edit would make it read, not a page.
    edit: bool = False


def _numbers_of(
    project: Project, label: str, page: ResearchPage, *, edit: bool = False
) -> _Numbers:
    """The number check of ``page`` (a finding, or one as an edit would make it read)."""
    report = experiments.number_report(project, page)
    return _Numbers(
        label, len(report["checked"]), list(report["unverified"]), list(report["elsewhere"]), edit
    )


def _as_edited(page: ResearchPage, proposal: EditProposal) -> ResearchPage:
    """``page`` as it would read with the suggested edit applied."""
    relations = {name: list(targets) for name, targets in page.relations.items()}
    for name, added in proposal.relations_add.items():
        have = relations.setdefault(name, [])
        have += [target for target in added if target not in have]
    fields = proposal.fields
    return replace(
        page,
        title=str(fields.get("title") or page.title),
        summary=str(fields.get("summary") or page.summary),
        body=page.body if proposal.body is None else proposal.body,
        relations=relations,
        meta={**page.meta, **fields},
    )


def _edits_to_findings(
    ctx: AgentContext, proposals: Iterable[EditProposal]
) -> list[tuple[EditProposal, ResearchPage]]:
    """Of ``proposals``, those that change what a finding says, each with the finding."""
    found = []
    for proposal in proposals:
        path = ctx.project.resolve(proposal.target)
        page = ctx.project.page(path) if path else None
        says = proposal.body is not None or {"title", "summary"} & set(proposal.fields)
        if page is not None and page.kind == "finding" and says:
            found.append((proposal, page))
    return found


def _checked_numbers(
    ctx: AgentContext, session: _Session, edits: Iterable[EditProposal]
) -> list[_Numbers]:
    """The number check of every finding the session proposed and of every finding as
    one of ``edits`` would make it read; only those where something was not found
    where it should be."""
    project = ctx.project
    checks = [
        _numbers_of(project, str(finding.id), finding) for finding in session.proposed("finding")
    ]
    checks += [
        _numbers_of(
            project,
            f"The edit suggested to {page.id or page.title} ({proposal.id})",
            _as_edited(page, proposal),
            edit=True,
        )
        for proposal, page in _edits_to_findings(ctx, edits)
    ]
    return [check for check in checks if check.missing or check.elsewhere]


def _listed_numbers(numbers: Sequence[str]) -> str:
    return ", ".join(numbers[:8]) + (" and more" if len(numbers) > 8 else "")


def _were(count: int) -> str:
    return "was" if count == 1 else "were"


def _number_note(check: _Numbers) -> str:
    """One sentence for the checkpoint that says what the check is and what it found."""
    parts = []
    if check.missing:
        count = len(check.missing)
        parts.append(f"{count} {_were(count)} not found ({_listed_numbers(check.missing)})")
    if check.elsewhere:
        count = len(check.elsewhere)
        parts.append(
            f"{count} {_were(count)} found only outside the records this finding is based on "
            f"({_listed_numbers(check.elsewhere)})"
        )
    return (
        f"{check.label}: {_plural(check.checked, 'number')} {_were(check.checked)} looked for "
        f"in the run records, sources and your files: {_listed(parts)}; short and round "
        "numbers are not checked."
    )


def _analyst_edits(ctx: AgentContext) -> list[EditProposal]:
    """The undecided edits the analyst suggested in this round, in any of its sessions."""
    author = role_author("analyst")
    return [
        proposal
        for proposal in ctx.project.proposals()
        if proposal.author == author and proposal.round == ctx.round
    ]


def _next_step_notes(ctx: AgentContext, session: _Session) -> list[str]:
    """What the checkpoint says about the experiments the analyst proposed as next steps.

    They are plans without code: the analyst has no tool to write any.  The person
    should know that before they find an experiment they cannot approve, and what
    is wrong with a plan (it names no hypothesis, its time limit is beyond the
    project's) is listed on its page as it is for the designer's.
    """
    notes: list[str] = []
    for page in session.proposed("experiment"):
        if not page.id:
            continue
        found = experiment_problems(ctx, page, code=False)
        if found or page.meta.get("problems"):
            with contextlib.suppress(NotFound):
                ctx.project.update_page(
                    page.path, author="system", fields={"problems": found or None}
                )
        after = (
            "Once you accept it and approve its run, the agent that carries it out writes "
            "the code from the plan."
            if ctx.runtime is not None
            else "Once you accept it, the designer writes its code in the next design "
            "phase; its run can be approved after that."
        )
        notes.append(
            f"{page.id} is a next step the analyst proposed as a plan, without code. {after}"
            + (
                f" Its page lists what else stands in the way: {_short(found[0], 200)}"
                + (f" (and {len(found) - 1} more)" if found[1:] else "")
                if found
                else ""
            )
        )
    return notes


def run_analyst(
    ctx: AgentContext,
    *,
    results: Sequence[ResearchPage],
    journal: str,
    spent: Mapping[str, float] | None = None,
) -> AgentReport:
    """Interpret the new result records, propose verdicts and next steps, write the journal.

    ``results`` are the records not analysed yet; ``journal`` is the name of the
    round's journal page (``round-2``); ``spent`` is what the project has spent on
    experiments, so that a next step is planned within what is left.

    Afterwards the numbers in each new finding, and in each finding as an edit
    the session suggested would make it read, are looked for on record
    (``research.experiments.number_report``), and the same conversation is asked
    to correct or remove what was not found, a bounded number of times.  What
    remains is said in the report's notes, computed here from the pages as they
    stand: nothing about it is stored on a page, where it would go stale with
    the next edit.
    """
    project = ctx.project
    limits = project.settings.limits
    caps = {"hypothesis": int(limits.max_hypotheses), "experiment": int(limits.max_experiments)}
    session = _open_session(ctx, "analyst", caps=caps)
    journal_path = f"{SECTIONS['journal']}/{journal}.md"
    session.send(
        prompts.analyst_message(
            idea=_idea(project),
            results=_result_items(project, results),
            journal_path=virtual_path(f"wiki/{journal_path}"),
            round=ctx.round,
            max_hypotheses=caps["hypothesis"],
            max_experiments=caps["experiment"],
            compute=_compute_text(ctx),
            limits=_design_limits(project, spent or {}),
            **_earlier(ctx, "analyst"),
            requests=_request_items(ctx, "analyst"),
            instruction=ctx.instruction,
        )
    )
    flagged = _checked_numbers(ctx, session, session.suggested())
    #: The suggested edits the session has been asked about.  An edit cannot be changed
    #: once it is made, only replaced by another: asking about it again would be asking
    #: for what cannot be done, at the price of a turn.  It is reported to the person.
    told_of: set[str] = set()
    for attempt in range(NUMBER_REPAIRS):
        # A number found on record, though not in the finding's own records, may well be
        # right (a value from a source): it is asked about once, not insisted on.
        asked = [
            check
            for check in flagged
            if (check.missing or attempt == 0) and check.label not in told_of
        ]
        if not asked:
            break
        told_of |= {check.label for check in asked if check.edit}
        told = [
            {
                "label": check.label,
                "missing": check.missing,
                "elsewhere": check.elsewhere if attempt == 0 else [],
            }
            for check in asked
        ]
        session.send(prompts.numbers_message(told, last=attempt == NUMBER_REPAIRS - 1))
        flagged = _checked_numbers(ctx, session, session.suggested())
    # For the person: every edit of the analyst's that awaits them, also one that an
    # attempt before this one suggested.
    notes = [_number_note(check) for check in _checked_numbers(ctx, session, _analyst_edits(ctx))]
    notes += _next_step_notes(ctx, session)
    if not project.resolve(f"wiki/{journal_path}"):
        notes.append(f"The analyst wrote no journal page for round {ctx.round}.")
    produced = session.created()
    return AgentReport(
        produced=produced,
        summary=f"The analyst read {_plural(len(results), 'new result')} and proposed "
        f"{_counted(produced)}.",
        notes=notes,
    )
