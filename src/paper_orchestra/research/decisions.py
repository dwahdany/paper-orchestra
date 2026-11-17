"""Decisions: what an agent asks the researcher to choose, and what they answered.

Some questions no evidence settles: what the idea means, which of two readings is
the object of study, whether a wider scope is worth its cost.  They are the
researcher's to answer.  An agent that meets one raises it with ``ask_decision``:
one sentence, the options, the one it would take and what it does meanwhile.  The
project does not wait for the answer: the agents go on with the stated default,
and the person answers when they next look (the Decisions panel of the app,
``paper-orchestra research decide``).

A decision is one small file, ``decisions/d-<8 hex digits>.json``, written and
read through ``Project`` (``ask_decision``, ``decisions``, ``answer_decision``,
``dismiss_decision``).  Agents can only raise one.  Nothing an agent is given
answers, withdraws or changes one, and the folder is not part of what a session's
memory tool reaches.

Every session of every role is shown the decisions in its user message
(:func:`told`): the open ones, so that nothing is asked twice and the default
holds, and every answer, for as long as the project lives.  An instruction of
the person is forgotten once its phases have run; what they decided is not.

Options are numbered from 1 wherever a number stands for one: in the file, in the
tool, on the command line and in the app.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from paper_orchestra.llm.base import ToolError, ToolSpec
from paper_orchestra.research.model import NotFound, ProjectError

if TYPE_CHECKING:
    from paper_orchestra.research.project import Project

__all__ = [
    "ASKING_ROLES",
    "DECISIONS_DIR",
    "DECISION_STATES",
    "MAX_OPEN",
    "MAX_PER_SESSION",
    "Answer",
    "Decision",
    "Option",
    "ask_decision_tool",
    "told",
    "waiting",
]

DECISIONS_DIR = "decisions"
DECISION_STATES: tuple[str, ...] = ("open", "answered", "withdrawn")
#: The events of a decision: asked by an agent, answered or left to the agents by the person.
DECISION_ASKED = "decision.asked"
DECISION_ANSWERED = "decision.answered"
DECISION_DISMISSED = "decision.dismissed"
#: The roles that are given ``ask_decision``: those whose sessions write pages.  (The
#: runner executes what was approved and writes none.)
ASKING_ROLES: tuple[str, ...] = ("intake", "scout", "theorist", "designer", "analyst")
#: How many decisions one run of a phase may raise.  A resumed attempt is the same session
#: and shares the count: three questions are what a person answers at a checkpoint.
MAX_PER_SESSION = 3
#: How many may wait at once.  Past it the agents go on with their defaults instead of
#: adding to a list nobody has answered.
MAX_OPEN = 12
MIN_OPTIONS, MAX_OPTIONS = 2, 5

QUESTION_CHARS = 300
CONTEXT_CHARS = 1200
LABEL_CHARS = 120
MEANING_CHARS = 500
LINE_CHARS = 500
#: The person's own words: long enough for a paragraph, short enough to be shown to
#: every later session.
SAY_CHARS = 4000


def _one_line(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def _whole(value: Any) -> int | None:
    """A whole number as a file or a tool gives it (``2``, ``"2"``), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


@dataclass(frozen=True)
class Option:
    """One thing the researcher can choose: a short label and what choosing it means."""

    label: str
    meaning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "meaning": self.meaning}


@dataclass(frozen=True)
class Answer:
    """What the person said to a decision at one moment.

    ``option`` is the number of the option they chose (from 1), or ``None`` when
    they answered in their own words only; ``text`` is those words.  ``dismissed``
    marks the other thing a person can say: not theirs to decide, use the default.
    """

    at: str
    option: int | None = None
    text: str = ""
    dismissed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "text": self.text,
            "at": self.at,
            "dismissed": self.dismissed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Answer:
        return cls(
            at=str(data.get("at") or ""),
            option=_whole(data.get("option")),
            text=str(data.get("text") or ""),
            dismissed=bool(data.get("dismissed")),
        )


@dataclass(frozen=True)
class Decision:
    """A question to the researcher, with its options and what they answered."""

    id: str
    question: str
    options: tuple[Option, ...] = ()
    #: Short Markdown: why the choice matters.
    context: str = ""
    #: The number of the option the agent would take (from 1), and why; or ``None``.
    recommended: int | None = None
    reason: str = ""
    #: What the agents do for as long as nobody has answered.
    default: str = ""
    #: The wiki page the question belongs to, and the pages that wait on the answer.
    page: str | None = None
    blocks: tuple[str, ...] = ()
    #: Who raised it: ``{"role", "round", "phase", "session"}``.
    raised_by: Mapping[str, Any] = field(default_factory=dict)
    created: str = ""
    state: str = "open"
    #: Everything the person said to it, oldest first.  The last entry is the answer.
    answers: tuple[Answer, ...] = ()

    @property
    def answer(self) -> Answer | None:
        """What holds now: the person's latest word, or ``None`` while it is open."""
        return self.answers[-1] if self.answers and self.state != "open" else None

    @property
    def chosen(self) -> Option | None:
        """The option the answer names, when it names one that exists."""
        answer = self.answer
        if answer is None or answer.dismissed or answer.option is None:
            return None
        return self.options[answer.option - 1] if 1 <= answer.option <= len(self.options) else None

    @property
    def role(self) -> str:
        return str(self.raised_by.get("role") or "")

    def to_dict(self) -> dict[str, Any]:
        answer = self.answer
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "options": [option.to_dict() for option in self.options],
            "recommended": self.recommended,
            "reason": self.reason,
            "default": self.default,
            "page": self.page,
            "blocks": list(self.blocks),
            "raised_by": dict(self.raised_by),
            "created": self.created,
            "state": self.state,
            "answer": answer.to_dict() if answer else None,
            "answers": [given.to_dict() for given in self.answers],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Decision:
        """Tolerant of what a hand-written file holds: an unknown state is read from the
        answers that are there, and a recommendation that names no option is none."""
        options = tuple(
            Option(label=str(item.get("label") or ""), meaning=str(item.get("meaning") or ""))
            for item in (data.get("options") if isinstance(data.get("options"), list) else [])
            if isinstance(item, Mapping) and item.get("label")
        )
        listed = data.get("answers")
        answers = [
            Answer.from_dict(item)
            for item in (listed if isinstance(listed, list) else [])
            if isinstance(item, Mapping)
        ]
        if not answers and isinstance(data.get("answer"), Mapping):
            answers = [Answer.from_dict(data["answer"])]
        state = data.get("state")
        if state not in DECISION_STATES:
            state = "open" if not answers else "withdrawn" if answers[-1].dismissed else "answered"
        recommended = _whole(data.get("recommended"))
        raised = data.get("raised_by")
        blocks = data.get("blocks")
        return cls(
            id=str(data.get("id") or ""),
            question=str(data.get("question") or ""),
            options=options,
            context=str(data.get("context") or ""),
            recommended=recommended if recommended and 1 <= recommended <= len(options) else None,
            reason=str(data.get("reason") or ""),
            default=str(data.get("default") or ""),
            page=str(data["page"]) if data.get("page") else None,
            blocks=tuple(str(item) for item in blocks) if isinstance(blocks, list) else (),
            raised_by=dict(raised) if isinstance(raised, Mapping) else {},
            created=str(data.get("created") or ""),
            state=state,
            answers=tuple(answers),
        )


# ---------------------------------------------------------------------------
# What a decision may hold
# ---------------------------------------------------------------------------


def _limited(what: str, value: Any, limit: int, *, needed: bool = False) -> str:
    text = str(value if value is not None else "").strip()
    if needed and not text:
        raise ProjectError(f"A decision needs {what}.")
    if len(text) > limit:
        raise ProjectError(
            f"{what[0].upper()}{what[1:]} is {len(text)} characters long; keep it within "
            f"{limit}. The researcher reads a decision on a phone: what does not fit belongs "
            "on the page it is about."
        )
    return text


def clean_options(value: Any) -> tuple[Option, ...]:
    """The options of a new decision, from a list of ``{"label", "meaning"}`` (a bare
    text is a label).  ``ProjectError`` unless there are two to five different ones."""
    items = list(value) if isinstance(value, list | tuple) else []
    options = []
    for item in items:
        given = item if isinstance(item, Mapping) else {"label": item}
        label = _one_line(_limited("the label of an option", given.get("label"), LABEL_CHARS))
        if not label:
            raise ProjectError("Every option needs a label: a few words that name the choice.")
        meaning = given.get("meaning", given.get("means"))
        options.append(
            Option(label=label, meaning=_limited("what an option means", meaning, MEANING_CHARS))
        )
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise ProjectError(
            f"A decision has {MIN_OPTIONS} to {MAX_OPTIONS} options, not {len(options)}: with "
            "one there is nothing to choose, and more than five are not read."
        )
    if len({option.label.lower() for option in options}) != len(options):
        raise ProjectError("Two options have the same label; each one names a different choice.")
    return tuple(options)


def new_decision(
    *,
    id: str,
    question: Any,
    options: Any,
    context: Any = "",
    recommended: Any = None,
    reason: Any = "",
    default: Any = "",
    page: str | None = None,
    blocks: Sequence[str] = (),
    raised_by: Mapping[str, Any] | None = None,
    created: str = "",
) -> Decision:
    """A decision as it is first written, checked.  ``ProjectError`` says what is wrong."""
    cleaned = clean_options(options)
    number = None
    if recommended not in (None, "", 0):
        number = _whole(recommended)
        if number is None or not 1 <= number <= len(cleaned):
            raise ProjectError(
                f"recommended is the number of an option, 1 to {len(cleaned)}, or nothing; "
                f"not {recommended!r}."
            )
    return Decision(
        id=id,
        question=_one_line(_limited("a question", question, QUESTION_CHARS, needed=True)),
        options=cleaned,
        context=_limited("the context", context, CONTEXT_CHARS),
        recommended=number,
        reason=_one_line(_limited("the reason for the recommendation", reason, LINE_CHARS)),
        default=_one_line(_limited("the default", default, LINE_CHARS)),
        page=page,
        blocks=tuple(blocks),
        raised_by=dict(raised_by or {}),
        created=created,
    )


def with_answer(decision: Decision, *, at: str, option: Any = None, text: Any = "") -> Decision:
    """``decision`` answered: an option, the person's own words, or both.

    A second answer is a new answer with its own time; the earlier ones are kept
    and the latest one holds.
    """
    said = _limited("what you say", text, SAY_CHARS)
    number = None
    if option not in (None, ""):
        number = _whole(option)
        if number is None or not 1 <= number <= len(decision.options):
            raise ProjectError(
                f"{decision.id} has options 1 to {len(decision.options)}; {option!r} is not "
                "one of them."
            )
    if number is None and not said:
        raise ProjectError(
            "An answer names an option, says it in your own words, or both; this one is empty."
        )
    answer = Answer(at=at, option=number, text=said)
    return replace(decision, state="answered", answers=(*decision.answers, answer))


def with_dismissal(decision: Decision, *, at: str) -> Decision:
    """``decision`` left to the agents: they go on with the default they stated."""
    answer = Answer(at=at, dismissed=True)
    return replace(decision, state="withdrawn", answers=(*decision.answers, answer))


def same_question(first: str, second: str) -> bool:
    """Whether two questions are the same words, whatever their case and spacing."""
    return _one_line(first).lower().rstrip("?.! ") == _one_line(second).lower().rstrip("?.! ")


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------


def answer_words(decision: Decision) -> str:
    """What the person answered, in a few words: ``option 2, "A note to itself"``, with
    their own words after it where they gave some.  Empty while the decision is open."""
    answer = decision.answer
    if answer is None:
        return ""
    if answer.dismissed:
        return "left to the agents: the default holds"
    parts = []
    chosen = decision.chosen
    if chosen is not None:
        parts.append(f'option {answer.option}, "{chosen.label}"')
    if answer.text:
        parts.append(
            f'"{_one_line(answer.text)}"' if chosen is not None else _one_line(answer.text)
        )
    return "; ".join(parts)


def waiting(count: int) -> str:
    """``2 decisions wait for your answer.``, or nothing when none does."""
    if count <= 0:
        return ""
    return (
        "1 decision waits for your answer."
        if count == 1
        else f"{count} decisions wait for your answer."
    )


def told(project: Project) -> list[dict[str, Any]]:
    """What every session is told about the decisions, oldest first: for each one its
    state, the question, the page it is about, the labels of its options, the default
    and, once the person has spoken, the answer (``{"option", "label", "text"}``)."""
    items = []
    for decision in project.decisions():
        answer = decision.answer
        chosen = decision.chosen
        items.append(
            {
                "id": decision.id,
                "state": decision.state,
                "question": decision.question,
                "page": decision.page,
                "options": [option.label for option in decision.options],
                "default": decision.default,
                "by": decision.role,
                "answer": None
                if answer is None or answer.dismissed
                else {
                    "option": answer.option if chosen is not None else None,
                    "label": chosen.label if chosen is not None else "",
                    "meaning": chosen.meaning if chosen is not None else "",
                    "text": answer.text,
                },
            }
        )
    return items


# ---------------------------------------------------------------------------
# The agents' tool
# ---------------------------------------------------------------------------


def ask_decision_tool(
    project: Project,
    role: str,
    *,
    round: int,
    phase: str,
    session: str,
    limit: int = MAX_PER_SESSION,
) -> ToolSpec:
    """The tool with which a role puts a choice in front of the researcher.

    ``session`` is the id of the role's session: what it has asked is counted from
    the files, so an attempt that resumes the session does not start at zero.
    """

    def page_named(ref: Any, what: str) -> str:
        path = project.resolve(ref) if isinstance(ref, str) and ref.strip() else None
        if path is None:
            raise ToolError(
                f"{what}: no page matches {ref!r}. Use an id such as Q3 or a path listed in "
                "the index, or leave it out."
            )
        return path

    def handler(tool_input: dict[str, Any]) -> str:
        question = tool_input.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolError("question must be one sentence the researcher can answer")
        default = tool_input.get("default")
        if not isinstance(default, str) or not default.strip():
            raise ToolError(
                "default must say what you go on with until the researcher answers: the "
                "project does not wait for them"
            )
        recommended, reason = tool_input.get("recommended"), tool_input.get("reason")
        if recommended not in (None, "") and not str(reason or "").strip():
            raise ToolError("reason must say in a sentence why you would take that option")
        page = tool_input.get("page")
        path = page_named(page, "page") if page not in (None, "") else None
        waits = tool_input.get("blocks") or []
        if not isinstance(waits, list):
            raise ToolError("blocks, when given, is a list of pages that wait on the answer")
        blocked = list(dict.fromkeys(page_named(ref, "blocks") for ref in waits))
        known = project.decisions()
        for earlier in known:
            if same_question(earlier.question, question):
                said = answer_words(earlier)
                return (
                    f"This was asked before, as {earlier.id}"
                    + (f", and the researcher answered: {said}." if said else "; no answer yet.")
                    + " Nothing new was asked."
                )
            if path and earlier.page == path and earlier.state == "open":
                return (
                    f'{earlier.id} already asks the researcher about that page: "'
                    f'{earlier.question}" It waits for their answer; nothing new was asked. '
                    f"Go on with its default: {earlier.default}"
                )
        asked = sum(1 for earlier in known if earlier.raised_by.get("session") == session)
        if asked >= limit:
            raise ToolError(
                f"This session has asked the researcher {asked} decisions, which is the "
                "limit. Go on with the default you would have stated, and say on the page "
                "that the choice is theirs."
            )
        waiting_now = sum(1 for earlier in known if earlier.state == "open")
        if waiting_now >= MAX_OPEN:
            raise ToolError(
                f"{waiting_now} decisions already wait for the researcher. Go on with the "
                "default you would have stated, and say on the page that the choice is theirs."
            )
        try:
            decision = project.ask_decision(
                question,
                options=tool_input.get("options"),
                context=tool_input.get("context") or "",
                recommended=recommended,
                reason=reason or "",
                default=default,
                page=path,
                blocks=blocked,
                role=role,
                round=round,
                phase=phase,
                session=session,
            )
        except (ProjectError, NotFound) as exc:
            raise ToolError(str(exc)) from exc
        left = limit - asked - 1
        more = (
            "This session can ask no further decision."
            if left <= 0
            else f"This session can ask {left} more."
        )
        return (
            f"Asked as {decision.id}. The researcher answers when they next look; the project "
            f"does not wait. Go on with your default: {decision.default} {more}"
        )

    return ToolSpec(
        name="ask_decision",
        description=(
            "Put a choice in front of the researcher that is theirs to make: what is to be "
            "studied, which reading of their idea is meant, a trade-off of cost or scope. Not "
            "for what a source, a run or more work could settle: find that out, or write a "
            "question page. Give two to five options, say which one you would take and why, "
            "and state the default you go on with: the project does not wait for the answer. "
            "They see the question, the options and your recommendation at the top of their "
            "inbox, and every later session of every agent is shown what they answered. Do "
            "not ask what the user message lists as already asked or answered. At most "
            f"{MAX_PER_SESSION} per session, so ask only what changes the work."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The choice in one sentence, as a question to the researcher.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Why it matters, in a few sentences of Markdown: what changes with "
                        "the answer. They read it on a phone."
                    ),
                },
                "options": {
                    "type": "array",
                    "minItems": MIN_OPTIONS,
                    "maxItems": MAX_OPTIONS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "A few words that name the choice.",
                            },
                            "meaning": {
                                "type": "string",
                                "description": "What choosing it means for the work.",
                            },
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                    "description": "The choices, two to five.",
                },
                "recommended": {
                    "type": "integer",
                    "description": "The number of the option you would take, counted from 1.",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence: why you would take that option.",
                },
                "default": {
                    "type": "string",
                    "description": (
                        "What the agents do until the researcher answers, in one sentence."
                    ),
                },
                "page": {
                    "type": "string",
                    "description": (
                        "The wiki page the decision belongs to, if any: its id (Q3) or its "
                        "/memories path."
                    ),
                },
                "blocks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pages that wait on the answer (ids or paths), if any.",
                },
            },
            "required": ["question", "options", "default"],
            "additionalProperties": False,
        },
        handler=handler,
    )
