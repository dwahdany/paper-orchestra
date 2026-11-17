"""The vocabulary of a research project, as code.

A project is a directory of plain files: wiki pages (Markdown with a flat front
matter), edit proposals, requests and an event stream.  This module says what
may be written in them and by whom: the kinds of page and where each lives, how
ids are formed, the review values, the lifecycle of the things pages describe
and who may move them along it, the relation fields that make the wiki a graph,
and what each agent role may create.  It reads and writes nothing; the store
that enforces these rules on disk is :mod:`paper_orchestra.research.project`.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, get_args

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.llm.base import Effort
from paper_orchestra.memory.wiki import (
    WIKI_DIR,
    Page,
    join_front_matter,
    link_for,
    split_front_matter,
)

SCHEMA = 1

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProjectError(RuntimeError):
    """The project, or what was asked of it, is not right; the message says what to do."""


class Conflict(ProjectError):
    """An edit was based on a version of a page that has changed since.

    ``page`` is the page as it now is.  When the conflict arose while accepting an
    edit proposal, ``proposal`` is that proposal as it was left and ``applied`` says
    whether its field changes went in although its replacement text did not.
    """

    def __init__(
        self,
        message: str,
        *,
        page: ResearchPage | None = None,
        proposal: EditProposal | None = None,
        applied: bool = False,
    ) -> None:
        super().__init__(message)
        self.page = page
        self.proposal = proposal
        self.applied = applied


class NotAllowed(ProjectError):
    """The change exists but is not this author's to make (an agent touching a record)."""


class NotFound(ProjectError):
    """No page, proposal or request answers to the reference given."""


# ---------------------------------------------------------------------------
# Kinds, sections, ids
# ---------------------------------------------------------------------------

KINDS: tuple[str, ...] = (
    "idea",
    "question",
    "source",
    "topic",
    "hypothesis",
    "experiment",
    "result",
    "finding",
    "journal",
    "lesson",
)

#: Directory of each kind under ``wiki/``.  The idea is the single page ``wiki/idea.md``.
SECTIONS: dict[str, str] = {
    "idea": "",
    "question": "questions",
    "source": "sources",
    "topic": "topics",
    "hypothesis": "hypotheses",
    "experiment": "experiments",
    "result": "results",
    "finding": "findings",
    "journal": "journal",
    "lesson": "lessons",
}
KIND_OF_SECTION: dict[str, str] = {section: kind for kind, section in SECTIONS.items() if section}
IDEA_PATH = f"{WIKI_DIR}/idea.md"

#: Kinds whose pages are numbered: ``H2`` is the second hypothesis ever created.
ID_PREFIXES: dict[str, str] = {
    "question": "Q",
    "hypothesis": "H",
    "experiment": "E",
    "finding": "F",
}
_KIND_OF_PREFIX = {prefix: kind for kind, prefix in ID_PREFIXES.items()}
_ID = re.compile(r"^([QHEF])(\d+)$", re.IGNORECASE)

#: Written by pipeline code from verified data or from what a run produced, never by a model.
RECORD_KINDS: tuple[str, ...] = ("source", "result")

REVIEWS: tuple[str, ...] = ("proposed", "accepted", "rejected")
POLICIES: tuple[str, ...] = ("auto", "review")
#: Kinds the project's review policy covers.  The idea is the person's and a result is what
#: a run produced: neither is ever held for review.
REVIEWED_KINDS: tuple[str, ...] = (
    "source",
    "topic",
    "question",
    "hypothesis",
    "experiment",
    "finding",
    "journal",
    "lesson",
)


def is_record(kind: str) -> bool:
    return kind in RECORD_KINDS


def kind_for_path(path: str) -> str | None:
    """The kind the location of ``path`` implies, or ``None`` outside the known sections."""
    if path == IDEA_PATH:
        return "idea"
    parts = PurePosixPath(path).parts
    if len(parts) >= 3 and parts[0] == WIKI_DIR:
        return KIND_OF_SECTION.get(parts[1])
    return None


def section_path(kind: str) -> str:
    """``hypothesis`` -> ``wiki/hypotheses`` (``wiki`` for the idea)."""
    section = SECTIONS[kind]
    return f"{WIKI_DIR}/{section}" if section else WIKI_DIR


def page_path(kind: str, *, id: str | None = None, slug: str = "", name: str = "") -> str:
    """Where a page of ``kind`` lives.

    Numbered kinds take ``id`` and ``slug`` (``H2-warmup.md``), records take the
    ``name`` they are known by (a citation key, ``E1-run1``), the rest take ``slug``.
    """
    if kind not in KINDS:
        raise ProjectError(f"'{kind}' is not a kind of page; the kinds are: {', '.join(KINDS)}")
    if kind == "idea":
        return IDEA_PATH
    if kind in ID_PREFIXES:
        if not id:
            raise ProjectError(f"a {kind} page needs an id")
        stem = f"{id}-{slug}" if slug else id
    else:
        stem = name or slug
    if not stem:
        raise ProjectError(f"a {kind} page needs a name")
    return f"{section_path(kind)}/{stem}.md"


def parse_id(text: Any) -> tuple[str, int] | None:
    """``"h2"`` -> ``("H", 2)``; ``None`` for anything that is not exactly an id."""
    match = _ID.match(str(text).strip()) if isinstance(text, (str, int)) else None
    return (match.group(1).upper(), int(match.group(2))) if match else None


def format_id(kind: str, number: int) -> str:
    return f"{ID_PREFIXES[kind]}{number}"


def id_in_name(kind: str, stem: str) -> str | None:
    """The id a file name of a numbered kind starts with: ``H2-warmup`` -> ``H2``."""
    prefix = ID_PREFIXES.get(kind)
    if prefix is None:
        return None
    match = re.match(rf"^{prefix}(\d+)(?:-|$)", stem, re.IGNORECASE)
    return f"{prefix}{int(match.group(1))}" if match else None


def kind_of_id(page_id: str) -> str | None:
    """``H2`` -> ``hypothesis``."""
    parsed = parse_id(page_id)
    return _KIND_OF_PREFIX[parsed[0]] if parsed else None


# ---------------------------------------------------------------------------
# Authors and roles
# ---------------------------------------------------------------------------

HUMAN = "human"
SYSTEM = "system"
AGENT_PREFIX = "agent:"
_AGENT_NAME = re.compile(r"^[a-z][a-z0-9_\-]*$")

ROLES: tuple[str, ...] = (
    "intake",
    "scout",
    "theorist",
    "designer",
    "runner",
    "analyst",
    "experimenter",
)

#: What each role may create through its memory tool.  Sources and results are records,
#: written by code; the runner executes experiments and writes no page.
ROLE_MAY_CREATE: dict[str, tuple[str, ...]] = {
    "intake": ("topic", "question"),
    "scout": ("topic", "question"),
    "theorist": ("hypothesis", "question"),
    "designer": ("experiment",),
    "analyst": ("finding", "question", "hypothesis", "experiment", "journal"),
    # The agent that carries out an approved experiment (the ``sdk`` runtime).  Unlike
    # the runner it is a model with a will of its own, so it is an agent like the others:
    # it moves no experiment and writes no record, and what it wants next (a follow-up
    # experiment, a question) it proposes.
    "experimenter": ("experiment", "question"),
}


def agent_author(role: str) -> str:
    """``scout`` -> ``agent:scout``."""
    return f"{AGENT_PREFIX}{role}"


def role_author(role: str) -> str:
    """The author a role writes as, from ``scout`` or ``agent:scout``."""
    text = str(role).strip().lower()
    return normalize_author(text if text.startswith(AGENT_PREFIX) else agent_author(text))


def normalize_author(value: Any) -> str:
    """Canonical author: ``human``, ``system`` or ``agent:<name>``.

    A bare role name is taken as that agent.  Anything else is refused, because the
    author decides what a page's writer may do next.
    """
    text = str(value).strip().lower() if isinstance(value, str) else ""
    if text in (HUMAN, SYSTEM):
        return text
    if text in ROLES:
        return agent_author(text)
    if text.startswith(AGENT_PREFIX) and _AGENT_NAME.match(text[len(AGENT_PREFIX) :]):
        return text
    raise ProjectError(
        f"{value!r} is not an author; use 'human', 'agent:<name>' (for example agent:scout) "
        "or 'system'"
    )


def is_agent(author: str) -> bool:
    return author.startswith(AGENT_PREFIX)


def role_of(author: str) -> str | None:
    """``agent:scout`` -> ``scout``; ``None`` for the person and for pipeline code."""
    return author[len(AGENT_PREFIX) :] if is_agent(author) else None


def creatable_kinds(role: str) -> tuple[str, ...]:
    return ROLE_MAY_CREATE.get(role, ())


def may_create(role: str, kind: str) -> bool:
    return kind in creatable_kinds(role)


def initial_review(kind: str, author: str, policy: Mapping[str, str]) -> str:
    """The review a new page starts with.

    What a person writes is accepted, and so is what pipeline code writes.  An
    agent's page is accepted only where the project's policy for the kind is
    ``auto``; a kind the policy does not name is held for review.
    """
    if not is_agent(author):
        return "accepted"
    return "accepted" if policy.get(kind, "review") == "auto" else "proposed"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

STATES: dict[str, tuple[str, ...]] = {
    "question": ("open", "answered", "dropped"),
    "hypothesis": ("open", "testing", "supported", "refuted", "inconclusive", "dropped"),
    "experiment": (
        "planned",
        "approved",
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    ),
    "result": ("succeeded", "failed", "cancelled"),
}
INITIAL_STATE: dict[str, str] = {"question": "open", "hypothesis": "open", "experiment": "planned"}

_PERSON = frozenset({"person"})
_CODE = frozenset({"code"})
_EITHER = _PERSON | _CODE

#: Every legal move of an experiment and who makes it.  Approval is the person's alone, from
#: any state: it is what spends their money.  Code only carries an approved experiment
#: through its run, puts an interrupted one back in the queue, and takes an approval back
#: when the experiment is no longer what was approved (never the other way round).  The
#: end of a run is recorded from ``approved`` and ``queued`` as well as from ``running``:
#: a page saved from an editor while its run was under way must not read as waiting to run
#: once that run is over.
_EXPERIMENT_MOVES: dict[tuple[str, str], frozenset[str]] = {
    ("planned", "approved"): _PERSON,
    ("planned", "cancelled"): _PERSON,
    ("approved", "planned"): _EITHER,
    ("approved", "queued"): _CODE,
    ("approved", "running"): _CODE,
    ("approved", "succeeded"): _CODE,
    ("approved", "failed"): _CODE,
    ("approved", "cancelled"): _EITHER,
    ("queued", "planned"): _EITHER,
    ("queued", "running"): _CODE,
    ("queued", "succeeded"): _CODE,
    ("queued", "failed"): _CODE,
    ("queued", "cancelled"): _EITHER,
    ("running", "queued"): _CODE,
    ("running", "succeeded"): _CODE,
    ("running", "failed"): _CODE,
    ("running", "cancelled"): _EITHER,
    ("succeeded", "approved"): _PERSON,
    ("succeeded", "planned"): _PERSON,
    ("failed", "approved"): _PERSON,
    ("failed", "planned"): _PERSON,
    ("cancelled", "approved"): _PERSON,
    ("cancelled", "planned"): _PERSON,
}


def actor_class(by: str) -> str:
    """``person``, ``code`` (the pipeline and the runner, which only executes) or ``agent``."""
    author = normalize_author(by)
    if author == HUMAN:
        return "person"
    if author in (SYSTEM, agent_author("runner")):
        return "code"
    return "agent"


#: What a reader can take for a quotation mark besides the marks themselves (the Unicode
#: categories Pi and Pf): straight marks and the accents typed for them, low marks, primes,
#: modifier letters, Hebrew and CJK marks, ornaments.
_LIKE_QUOTATION_MARKS = frozenset(
    "\"'`\u00b4\u201a\u201e"
    "\u2032\u2033\u2034\u2035\u2036\u2037\u2057"
    "\u02b9\u02ba\u02bb\u02bc\u02bd\u02c8\u02ca\u02cb\u02dd\u02ee"
    "\u05f3\u05f4\u3003\u301d\u301e\u301f"
    "\u275b\u275c\u275d\u275e\u275f\u2760\u2e42\uff02\uff07"
    "\U0001f676\U0001f677\U0001f678"
)
_APOSTROPHES = re.compile("'{2,}")


def quotable(text: Any, limit: int = 120) -> str:
    """Words someone else chose (an agent's title, a summary, a field value), made safe
    to stand between quotation marks in a sentence the project says: on one line, cut
    to ``limit`` characters, whatever reads as a quotation mark made a plain apostrophe,
    and characters that are not shown (or that change the order of what is shown) left
    out.  The caller adds the marks; inside them nothing can look like their end, so
    the words read as a quotation and never as a further sentence."""
    visible = "".join(
        " " if ch.isspace() else ch
        for ch in str(text if text is not None else "")
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co")
    )
    plain = "".join(
        "'" if ch in _LIKE_QUOTATION_MARKS or unicodedata.category(ch) in ("Pi", "Pf") else ch
        for ch in visible
    )
    line = " ".join(_APOSTROPHES.sub("'", plain).split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def with_article(kind: str) -> str:
    """``a hypothesis``, ``an idea``."""
    return f"an {kind}" if kind[0] in "aeiou" else f"a {kind}"


def _movers(kind: str, current: str, new: str) -> frozenset[str] | None:
    """Who may make the move, or ``None`` when the move does not exist."""
    if kind == "experiment":
        return _EXPERIMENT_MOVES.get((current, new))
    if kind == "hypothesis" and {current, new} == {"open", "testing"}:
        # Bookkeeping that follows from an experiment being run, not a judgement.
        return _EITHER
    # Whether a hypothesis held or a question is answered is a verdict: the person's.
    return _PERSON


def check_transition(kind: str, current: str | None, new: str, by: str) -> None:
    """Raise unless ``by`` may move a page of ``kind`` from state ``current`` to ``new``.

    ``ProjectError`` when the kind has no lifecycle, ``new`` is not one of its states
    or the move does not exist; ``NotAllowed`` when it exists but is not ``by``'s to
    make.  A missing ``current`` (a page written by hand) counts as the first state.
    """
    if kind in RECORD_KINDS:
        raise ProjectError(
            f"The state of {with_article(kind)} is part of the record: it is copied from the run "
            "when the record is written, and is not changed afterwards."
        )
    states = STATES.get(kind)
    if states is None:
        raise ProjectError(f"{with_article(kind).capitalize()} page has no lifecycle states.")
    if new not in states:
        raise ProjectError(
            f"'{new}' is not a state of {with_article(kind)}; the states are: {', '.join(states)}."
        )
    current = current if current in states else INITIAL_STATE[kind]
    if current == new:
        return
    who = actor_class(by)
    movers = _movers(kind, current, new)
    if movers is None and who != "agent":
        options = [target for (start, target) in _EXPERIMENT_MOVES if start == current]
        raise ProjectError(
            f"{with_article(kind).capitalize()} that is {current} cannot go straight to {new}; "
            f"from {current} it can become: {', '.join(options)}."
        )
    if movers is not None and who in movers:
        return
    # From here on the answer is no; an agent hears why whether or not the move exists.
    if kind != "experiment":
        raise NotAllowed(
            f"Whether {with_article(kind)} is {new} is the person's decision; {by} cannot set it. "
            "An agent proposes the change with propose_edit and the person accepts or rejects it."
        )
    if new == "approved" or (movers is not None and "code" not in movers):
        raise NotAllowed(
            f"Only the person can move an experiment from {current} to {new}: running it "
            "spends their time and money, so nothing an agent or the pipeline writes approves "
            "one. Ask the person to approve it."
        )
    raise NotAllowed(
        f"Moving an experiment from {current} to {new} is done by the run itself, not by {by}."
    )


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Relation:
    """A list field in the front matter of the page the arrow starts from."""

    name: str
    #: Kinds the arrow may start from; empty means any kind.
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    meaning: str


RELATIONS: dict[str, Relation] = {
    relation.name: relation
    for relation in (
        Relation("about", ("question",), ("idea", "topic"), "the question is about it"),
        Relation("cites", (), ("source",), "relies on that source"),
        Relation("answers", ("topic", "finding"), ("question",), "answers the question"),
        Relation(
            "motivated_by",
            ("hypothesis",),
            ("source", "topic", "question", "finding"),
            "why it was proposed",
        ),
        Relation("tests", ("experiment",), ("hypothesis",), "the hypothesis it tests"),
        Relation("run_of", ("result",), ("experiment",), "the experiment that was run"),
        Relation("supports", ("finding",), ("hypothesis",), "the evidence speaks for it"),
        Relation("refutes", ("finding",), ("hypothesis",), "the evidence speaks against it"),
        Relation("based_on", ("finding",), ("result",), "the records it interprets"),
        Relation(
            "follows",
            ("hypothesis", "experiment", "question"),
            ("finding",),
            "a next step derived from it",
        ),
    )
}
#: The body link ``[[...]]``: an edge without a claim attached.
MENTIONS = "mentions"


def relations_from(kind: str) -> tuple[str, ...]:
    """Relation fields a page of ``kind`` may carry."""
    return tuple(
        name
        for name, relation in RELATIONS.items()
        if not relation.sources or kind in relation.sources
    )


def clean_target(value: Any) -> str:
    """One relation target as written -> an id (``H2``) or a link target (``topics/warmup``)."""
    text = str(value).strip().strip("\"'").strip()
    text = text.removeprefix("[[").removesuffix("]]")
    text = text.split("|", 1)[0].split("#", 1)[0].strip().strip("/")
    text = text.removeprefix("memories/")
    if text.endswith(".md"):
        text = text[:-3]
    text = text.removeprefix(f"{WIKI_DIR}/")
    parsed = parse_id(text)
    return f"{parsed[0]}{parsed[1]}" if parsed else text


def as_targets(value: Any) -> list[str]:
    """The targets of a relation field, however the list was written.

    Accepts a list, a single value, or one line holding several (``H1, H2`` or
    ``[Smith2020Deep, Lee2021Wide]``), with or without wiki-link brackets.  Order is
    kept and repeats are dropped.
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]") and not text.startswith("[["):
            text = text[1:-1]
        items: Iterable[Any] = text.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    targets: list[str] = []
    for item in items:
        target = clean_target(item)
        if target and target not in targets:
            targets.append(target)
    return targets


def clean_relations(kind: str, relations: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """Validated relation fields for a page of ``kind``; empty ones are left out.

    Refuses a name that is not a relation, and a relation that does not start from
    this kind.  Whether the targets exist is the store's question, not this one's.
    """
    cleaned: dict[str, list[str]] = {}
    for name, value in (relations or {}).items():
        relation = RELATIONS.get(name)
        allowed = relations_from(kind)
        if relation is None:
            raise ProjectError(
                f"'{name}' is not a relation; {with_article(kind)} page can use: {', '.join(allowed)}."
            )
        targets = as_targets(value)
        if name not in allowed and not targets:
            # A form that sends every relation, most of them empty, asks for nothing here.
            continue
        if name not in allowed:
            raise ProjectError(
                f"'{name}' is not a field of {with_article(kind)} page (it starts from: "
                f"{', '.join(relation.sources)}). {with_article(kind).capitalize()} can use: "
                f"{', '.join(allowed)}."
            )
        if targets:
            cleaned[name] = targets
    return cleaned


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------

#: Front-matter field of a page someone else wrote: when the person last edited it
#: themselves (a stamp).  From then on the page is not its author's to rewrite, also
#: while the decision on it is open; on a record it says that the page is no longer
#: only what code wrote.
EDITED_BY_PERSON = "edited_by_person"
#: Fields the project sets itself.  They are never taken from the ``fields`` a caller
#: passes, and an edit proposal may not carry them.
MANAGED_FIELDS: frozenset[str] = frozenset(
    {
        "kind",
        "id",
        "review",
        "author",
        "round",
        "created",
        "updated",
        "updated_by",
        "decided_by",
        "decided_at",
        "decision_reason",
        EDITED_BY_PERSON,
    }
)
#: The fields of an experiment page that decide what a run executes, where and for how
#: long.  An approval pins them (with the code) in ``approved_digest``; a run starts only
#: while they are what was approved.
RUN_FIELDS: tuple[str, ...] = (
    "mode",
    "entrypoint",
    "command",
    "packages",
    "repo",
    "repo_ref",
    "gpu",
    "network",
    "backend",
    "timeout_seconds",
)
#: The record of an approval on an experiment page: where the run goes, the digest of
#: what was approved and when, and the digest after the runner's own repair.  Only the
#: approval (``research.experiments.approve``) and the run write them; no agent writes or
#: proposes them, and of these only ``backend`` is a field the person may edit by hand.
APPROVAL_FIELDS: frozenset[str] = frozenset(
    {"backend", "approved_digest", "approved_at", "repaired_digest"}
)
#: What a model session cannot set by writing it into a page: the managed fields, the
#: lifecycle state (approval is the person's), the session the page belongs to and the
#: record of an approval.
AGENT_LOCKED_FIELDS: frozenset[str] = (
    MANAGED_FIELDS | {"state", "state_reason", "session"} | APPROVAL_FIELDS
)

_FIELD_ORDER: tuple[str, ...] = (
    "title",
    "kind",
    "id",
    "summary",
    "review",
    "state",
    "state_reason",
    "author",
    "round",
    "created",
    "updated",
    "updated_by",
    "session",
    "decided_by",
    "decided_at",
    "decision_reason",
    EDITED_BY_PERSON,
    "approved_at",
    "approved_digest",
    "repaired_digest",
    *RELATIONS,
)

STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_COMPACT_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def format_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(STAMP_FORMAT)


def parse_stamp(value: Any) -> datetime | None:
    """A stamp as written in a page (or as it appears in a history file name) -> UTC time."""
    if not isinstance(value, str):
        return None
    for layout in (STAMP_FORMAT, _COMPACT_STAMP_FORMAT):
        try:
            return datetime.strptime(value.strip(), layout).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def compact_stamp(stamp: str) -> str:
    """``2026-11-15T12:00:05Z`` -> ``20261115T120005Z``, safe in a file name on any system."""
    moment = parse_stamp(stamp)
    return moment.strftime(_COMPACT_STAMP_FORMAT) if moment else re.sub(r"[^0-9A-Za-z]", "", stamp)


def stamp_after(now: str, floor: str) -> str:
    """``now``, or one second past ``floor`` when ``now`` is not later than it.

    Stamps have one-second resolution and a page's ``updated`` identifies its
    version, so two changes within a second must still get different stamps.
    """
    current, lowest = parse_stamp(now), parse_stamp(floor)
    if current is None or lowest is None or current > lowest:
        return now
    return format_stamp(lowest + timedelta(seconds=1))


def ordered_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """``meta`` with the fields a reader looks for first in a fixed order, the rest after."""
    ordered = {key: meta[key] for key in _FIELD_ORDER if key in meta}
    ordered.update({key: value for key, value in meta.items() if key not in ordered})
    return ordered


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchPage:
    """One wiki page: the typed fields every view needs, the whole front matter, the body."""

    path: str
    kind: str
    id: str | None
    title: str
    summary: str
    review: str
    state: str | None
    author: str
    round: int | None
    created: str
    updated: str
    relations: dict[str, list[str]]
    meta: dict[str, Any]
    body: str

    @property
    def label(self) -> str:
        """What the page is called for short: its id, or its title."""
        return self.id or self.title

    @property
    def link(self) -> str:
        """The wiki link target: ``hypotheses/H2-warmup``."""
        return link_for(self.path)

    @property
    def is_record(self) -> bool:
        return is_record(self.kind)

    @property
    def decided_by(self) -> str | None:
        """Who decided the page's review, when a person did."""
        value = self.meta.get("decided_by")
        return str(value) if value else None

    @property
    def decision_reason(self) -> str:
        return str(self.meta.get("decision_reason") or "")

    @property
    def session(self) -> str | None:
        """The agent session that created the page, where one did."""
        value = self.meta.get("session")
        return str(value) if value else None

    def to_dict(self, *, body: bool = True) -> dict[str, Any]:
        """Plain JSON.  ``body=False`` leaves the text out (lists and graphs)."""
        data: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "id": self.id,
            "label": self.label,
            "title": self.title,
            "summary": self.summary,
            "review": self.review,
            "state": self.state,
            "author": self.author,
            "round": self.round,
            "created": self.created,
            "updated": self.updated,
            "relations": {name: list(targets) for name, targets in self.relations.items()},
            "meta": dict(self.meta),
        }
        if body:
            data["body"] = self.body
        return data


def _kind_of(path: str, meta: Mapping[str, Any]) -> str:
    """Where a page lies says what it is; elsewhere its own word counts, then ``topic``."""
    located = kind_for_path(path)
    if located is not None:
        return located
    stated = str(meta.get("kind") or "")
    return stated if stated in KINDS and stated != "idea" else "topic"


def _author_of(meta: Mapping[str, Any]) -> str:
    try:
        return normalize_author(meta.get("author"))
    except ProjectError:
        # A page nobody signed was put there by hand.
        return HUMAN


def _id_of(kind: str, path: str, meta: Mapping[str, Any]) -> str | None:
    stem = PurePosixPath(path).stem
    if kind == "source":
        return stem
    prefix = ID_PREFIXES.get(kind)
    if prefix is None:
        return None
    parsed = parse_id(meta.get("id"))
    if parsed and parsed[0] == prefix:
        return f"{prefix}{parsed[1]}"
    return id_in_name(kind, stem)


def _round_of(meta: Mapping[str, Any]) -> int | None:
    value = meta.get("round")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, str) and value.strip().isdigit() else None


def page_from_parts(
    path: str, meta: Mapping[str, Any], body: str, *, file_stamp: str = ""
) -> ResearchPage:
    """The typed view of a page, whatever state its front matter is in.

    Never raises: a page written by hand may lack fields or carry values this
    version does not know, and must still be listed.  What is missing is filled
    with the safest reading (:func:`page_problems` says what was wrong); ``meta``
    keeps the front matter exactly as it was read.  ``file_stamp`` stands in for
    missing ``created``/``updated`` stamps, and for an ``updated`` that is not a
    stamp: it names the version in ``history/``, so it has to be one.
    """
    kind = _kind_of(path, meta)
    author = _author_of(meta)
    wiki_page = Page(path=path, meta=dict(meta), body=body)
    review = str(meta.get("review") or "")
    if review not in REVIEWS:
        # Unsigned or the person's: theirs to keep.  An agent's: not context until decided.
        review = "proposed" if is_agent(author) else "accepted"
    states = STATES.get(kind)
    state: str | None = None
    if states:
        stated = str(meta.get("state") or "")
        state = stated if stated in states else INITIAL_STATE.get(kind)
    relations = {name: targets for name in RELATIONS if (targets := as_targets(meta.get(name)))}
    summary = wiki_page.summary
    if not str(meta.get("summary") or "").strip():
        # Taken from the first sentence of the text, where a link is markup, not words.
        summary = plain_links(_summary_from(body))
    updated = str(meta.get("updated") or "")
    if file_stamp and parse_stamp(updated) is None:
        updated = file_stamp
    return ResearchPage(
        path=path,
        kind=kind,
        id=_id_of(kind, path, meta),
        title=wiki_page.title,
        summary=summary,
        review=review,
        state=state,
        author=author,
        round=_round_of(meta),
        created=str(meta.get("created") or "") or updated,
        updated=updated,
        relations=relations,
        meta=dict(meta),
        body=body,
    )


#: How long a summary taken from a page's text may be (what the wiki takes of a first line).
_SUMMARY_CHARS = 140
#: What begins a line that is no text to summarise (as ``memory.wiki`` has it), and what
#: begins the item of a list, which is not run together with the next.
_NOT_TEXT = ("#", "<!--", "---", "```", "|", ">")
_LIST_ITEM = re.compile(r"(?:[-*+]|\d+[.)])\s")


def _summary_from(body: str) -> str:
    """The summary of a page that states none: the first paragraph of its text, whole
    where it is short, else cut where a word ends and marked as cut.

    The wiki takes the first line of the text.  Where a paragraph's lines were
    broken by hand, or the line is long, that is the start of a sentence cut
    anywhere, shown as if the author had stopped there.
    """
    paragraph: list[str] = []
    for line in (raw.strip() for raw in body.splitlines()):
        ends = not line or line.startswith(_NOT_TEXT)
        if not paragraph:
            if ends:
                continue  # before the text begins
            paragraph.append(line)
            if _LIST_ITEM.match(line):
                break
        elif ends or _LIST_ITEM.match(line):
            break
        else:
            paragraph.append(line)
    text = " ".join(paragraph)
    if len(text) <= _SUMMARY_CHARS:
        return text
    cut = text[: _SUMMARY_CHARS - 1]
    whole = cut.rsplit(None, 1)[0] if " " in cut else cut
    return whole.rstrip(" ,;:") + "…"


#: ``[[target#anchor|label]]`` as a page's text writes a link to another page.
_WIKI_LINK = re.compile(r"\[\[([^\[\]|#\n]*)(?:#[^\[\]|\n]*)?(?:\|([^\[\]\n]*))?\]\]")


def plain_links(text: str) -> str:
    """``text`` with every wiki link replaced by what it is read as: its label, or else
    the name of the page it leads to (``[[topics/warmup]]`` -> ``warmup``, ``[[H2]]`` ->
    ``H2``).  For one-line summaries, which are shown where links are not rendered."""

    def name(match: re.Match[str]) -> str:
        label = (match.group(2) or "").strip()
        return label or PurePosixPath(match.group(1).strip()).name.removesuffix(".md")

    return " ".join(_WIKI_LINK.sub(name, text).split())


def moved_on(page: ResearchPage) -> bool:
    """Whether what the page describes has left its first state: an experiment the person
    approved (or that ran), a hypothesis under test or judged, a question answered.

    From then on its author no longer rewrites it: what was approved or tested must stay
    what the person read.
    """
    first = INITIAL_STATE.get(page.kind)
    return first is not None and page.state != first


def parse_page(path: str, text: str, *, file_stamp: str = "") -> ResearchPage:
    """A page file's text -> :class:`ResearchPage` (see :func:`page_from_parts`)."""
    meta, body = split_front_matter(text)
    return page_from_parts(path, meta, body, file_stamp=file_stamp)


def render_page(page: ResearchPage) -> str:
    """The text of the page's file: its front matter in reading order, then the body."""
    return join_front_matter(ordered_meta(page.meta), page.body)


def page_problems(page: ResearchPage) -> list[str]:
    """What is wrong with the page's own front matter, in words a person can act on.

    Checks the page alone; whether its relations point at pages that exist needs
    the whole wiki and is the store's to report.
    """
    meta = page.meta
    problems: list[str] = []
    stated_kind = meta.get("kind")
    if stated_kind in (None, ""):
        problems.append(f"no kind in the front matter; read as '{page.kind}' from its location")
    elif stated_kind != page.kind:
        problems.append(
            f"kind '{stated_kind}' does not match its location; read as '{page.kind}'"
            if stated_kind in KINDS
            else f"kind '{stated_kind}' is not a kind of page ({', '.join(KINDS)}); "
            f"read as '{page.kind}'"
        )
    if not str(meta.get("title") or "").strip():
        problems.append(f"no title in the front matter; shown as '{page.title}'")
    if meta.get("review") not in REVIEWS:
        problems.append(
            f"review is {meta.get('review')!r}, not one of {', '.join(REVIEWS)}; "
            f"read as '{page.review}'"
        )
    if "author" in meta and _author_of(meta) != meta.get("author"):
        problems.append(f"author {meta.get('author')!r} is not human, agent:<name> or system")
    if meta.get("updated") not in (None, "") and parse_stamp(meta.get("updated")) is None:
        problems.append(
            f"updated is {meta.get('updated')!r}, not a UTC stamp such as 2026-11-15T12:00:00Z; "
            "the time the file was last written is used"
        )
    states = STATES.get(page.kind)
    if states and meta.get("state") not in states:
        problems.append(
            f"'{meta.get('state')}' is not a state of {with_article(page.kind)} "
            f"({', '.join(states)}); read as '{page.state}'"
            if meta.get("state")
            else f"no state; read as '{page.state}'"
        )
    if page.kind in ID_PREFIXES:
        named = id_in_name(page.kind, PurePosixPath(page.path).stem)
        if page.id is None:
            problems.append(
                f"no id: {with_article(page.kind)} page is named {ID_PREFIXES[page.kind]}<n>-<slug>.md "
                "and carries the id in its front matter"
            )
        elif named and named != page.id:
            problems.append(f"id {page.id} does not match the file name ({named})")
    allowed = relations_from(page.kind)
    for name in page.relations:
        if name not in allowed:
            problems.append(
                f"'{name}' is not a relation of {with_article(page.kind)} page "
                f"(it can use: {', '.join(allowed)})"
            )
    return problems


@dataclass(frozen=True)
class PageVersion:
    """An earlier version of a page, kept under ``history/``."""

    #: When that version was written: the ``updated`` stamp it carried.
    stamp: str
    #: Who wrote that version.
    author: str
    size: int
    #: Project-relative path of the file holding it.
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stamp": self.stamp, "author": self.author, "size": self.size, "path": self.path}


# ---------------------------------------------------------------------------
# Edit proposals and requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditProposal:
    """A change an agent suggests to a page that is not its own proposal."""

    id: str
    target: str
    author: str = SYSTEM
    round: int | None = None
    created: str = ""
    summary: str = ""
    #: The full replacement body, or ``None`` to leave the text alone.
    body: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    relations_add: dict[str, list[str]] = field(default_factory=dict)
    #: The target's ``updated`` stamp when this was proposed.
    base_updated: str = ""
    review: str = "proposed"
    decided_by: str | None = None
    decided_at: str | None = None
    decision_reason: str | None = None
    #: Why the replacement text could not be applied, once an accept found the page changed.
    conflict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "target": self.target,
            "author": self.author,
            "round": self.round,
            "created": self.created,
            "summary": self.summary,
            "body": self.body,
            "fields": dict(self.fields),
            "relations_add": {name: list(items) for name, items in self.relations_add.items()},
            "base_updated": self.base_updated,
            "review": self.review,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "decision_reason": self.decision_reason,
        }
        if self.conflict:
            data["conflict"] = self.conflict
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EditProposal:
        """Tolerant of missing keys: a proposal file may have been written by hand."""
        review = data.get("review")
        fields = data.get("fields")
        relations = data.get("relations_add")
        body = data.get("body")
        return cls(
            id=str(data.get("id") or ""),
            target=str(data.get("target") or ""),
            author=str(data.get("author") or SYSTEM),
            round=_round_of(data),
            created=str(data.get("created") or ""),
            summary=str(data.get("summary") or ""),
            body=body if isinstance(body, str) else None,
            fields=dict(fields) if isinstance(fields, Mapping) else {},
            relations_add=(
                {str(name): as_targets(items) for name, items in relations.items()}
                if isinstance(relations, Mapping)
                else {}
            ),
            base_updated=str(data.get("base_updated") or ""),
            review=review if review in REVIEWS else "proposed",
            decided_by=_optional_text(data.get("decided_by")),
            decided_at=_optional_text(data.get("decided_at")),
            decision_reason=_optional_text(data.get("decision_reason")),
            conflict=_optional_text(data.get("conflict")),
        )


REQUEST_STATES: tuple[str, ...] = ("open", "done")
#: A request that is for every agent, as opposed to one for a single role.
FOR_ALL = "all"
#: The roles a request can be addressed to: those whose phases act on instructions.
REQUEST_ROLES: tuple[str, ...] = ("scout", "theorist", "designer", "analyst")


def request_role(value: Any) -> str:
    """Who a request is for, from how it was said: ``all`` (also for nothing given) or one
    of ``REQUEST_ROLES`` (``designer`` or ``agent:designer``).  ``ProjectError`` for
    anything else, naming the choices."""
    text = str(value or "").strip().lower().removeprefix(AGENT_PREFIX)
    if text in ("", FOR_ALL, "everyone"):
        return FOR_ALL
    if text not in REQUEST_ROLES:
        raise ProjectError(
            f"A request is for {FOR_ALL} (every agent) or for one of: "
            f"{', '.join(REQUEST_ROLES)}; not for {value!r}."
        )
    return text


def _handled(value: Any) -> list[dict[str, Any]]:
    """The phases a request was given to, however the file lists them (one object, as
    earlier versions wrote it, or a list of them)."""
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, list | tuple):
        return []
    return [dict(item) for item in items if isinstance(item, Mapping) and item.get("phase")]


@dataclass(frozen=True)
class Request:
    """An instruction from the person to the agents."""

    id: str
    text: str
    target: str | None = None
    created: str = ""
    state: str = "open"
    #: Who it is for: ``all`` (every phase that runs until the round's analysis) or one
    #: of ``REQUEST_ROLES`` (that role's next phase).  Written as ``for`` in the file.
    role: str = FOR_ALL
    #: Every phase that was given it, in order: ``[{"round": 2, "phase": "hypothesize"}]``.
    handled_in: list[dict[str, Any]] = field(default_factory=list)

    @property
    def for_all(self) -> bool:
        return self.role == FOR_ALL

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "target": self.target,
            "created": self.created,
            "state": self.state,
            "for": self.role,
            "handled_in": [dict(item) for item in self.handled_in],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Request:
        """Tolerant of what a hand-written or older file holds: a request for a role this
        version does not know is for everyone, so that it is shown rather than lost."""
        state = data.get("state")
        try:
            role = request_role(data.get("for", data.get("role")))
        except ProjectError:
            role = FOR_ALL
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            target=_optional_text(data.get("target")),
            created=str(data.get("created") or ""),
            state=state if state in REQUEST_STATES else "open",
            role=role,
            handled_in=_handled(data.get("handled_in")),
        )


def _optional_text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


# ---------------------------------------------------------------------------
# What a phase is called
# ---------------------------------------------------------------------------

#: What each phase of the engine is called where a person reads its name by itself (a
#: heading, "checkpoint after ...", "next: ..."): the words of the app, and of the command
#: line.  The names a file, an option or a program uses are the keys.
PHASE_NAMES: dict[str, str] = {
    "intake": "intake",
    "research": "research",
    "hypothesize": "hypotheses",
    "design": "experiment design",
    "run": "runs",
    "analyze": "analysis",
}


def phase_name(phase: Any) -> str:
    """``hypotheses`` for ``hypothesize``, ``analysis`` for ``analyze``: a phase as a
    person reads it.  What is no phase of the engine keeps the name it came with."""
    return PHASE_NAMES.get(phase, str(phase)) if isinstance(phase, str) else str(phase)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

EVENT_TYPES: tuple[str, ...] = (
    "project.created",
    "phase.started",
    "phase.finished",
    "phase.failed",
    "checkpoint",
    "page.created",
    "page.updated",
    "page.deleted",
    "review.decided",
    "proposal.created",
    "request.created",
    "request.withdrawn",
    "decision.asked",
    "decision.answered",
    "decision.dismissed",
    "agent.activity",
    "experiment.state",
    "run.started",
    "run.log",
    "run.finished",
    "budget.warning",
    "paper.started",
    "paper.finished",
    "error",
)

# ---------------------------------------------------------------------------
# Settings (project.json["settings"])
# ---------------------------------------------------------------------------

EFFORTS: tuple[str, ...] = get_args(Effort)
COMPUTE_BACKENDS: tuple[str, ...] = ("local", "modal")
_CUTOFF = re.compile(r"^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?$")
#: A cutoff as people type it into a file: the month or the day with one digit.
_CUTOFF_AS_TYPED = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")
#: Said of a cutoff that cannot be read, wherever the settings were loaded leniently.
CUTOFF_STOPS_RESEARCH = (
    "it is kept as written and is not read as 'no cutoff': no paper is looked up or "
    "recorded and no paper is written until it is corrected"
)


def as_cutoff(value: Any) -> Any:
    """A research cutoff in its usual form where what was meant is plain: a year given
    as a number (``2024``), a month or day written with one digit (``2024-6``), and an
    empty text, which is no cutoff (a settings field that was emptied; it is also how
    whoever looks up papers reads it, ``sources.normalize_cutoff``).  Anything else is
    returned as it is, for the check to judge."""
    text = value
    if isinstance(text, str) and not text.strip():
        return None
    if isinstance(text, float) and text.is_integer():
        text = int(text)
    if isinstance(text, int) and not isinstance(text, bool):
        text = str(text)
    match = _CUTOFF_AS_TYPED.match(text.strip()) if isinstance(text, str) else None
    if match is None:
        return value
    year, month, day = match.groups()
    typed = year + (f"-{int(month):02d}" if month else "") + (f"-{int(day):02d}" if day else "")
    return typed if _CUTOFF.match(typed) else value


#: The end of a loading problem that says what was put in the place of a bad value.
_DEFAULT_USED = re.compile(r"; the defaults? (?:\(.*\) )?(?:is|are) used$")


def default_review_policy() -> dict[str, str]:
    """Agents' syntheses, hypotheses, plans and findings wait for the person; verified
    sources and the agents' own process pages do not."""
    return {
        "source": "auto",
        "topic": "review",
        "question": "review",
        "hypothesis": "review",
        "experiment": "review",
        "finding": "review",
        "journal": "auto",
        "lesson": "auto",
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Section:
    """A group of settings: attributes, with a mapping's read interface on top.

    ``limits.model_usd`` and ``limits["model_usd"]`` are the same value, so code
    written against the JSON shape and code written against the typed view agree.
    Keys this version does not know are kept in ``extra`` and written back.
    """

    extra: dict[str, Any]

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in dataclass_fields(cls) if f.name != "extra")  # type: ignore[arg-type]

    def __getitem__(self, key: str) -> Any:
        if key in self.names():
            return getattr(self, key)
        return self.extra[key]

    def __contains__(self, key: object) -> bool:
        return key in self.names() or key in self.extra

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return [*self.names(), *self.extra]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def values(self) -> list[Any]:
        return [self[key] for key in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(key, self[key]) for key in self.keys()]

    def to_dict(self) -> dict[str, Any]:
        return {**{name: getattr(self, name) for name in self.names()}, **self.extra}

    def invalid(self) -> dict[str, str]:
        """Field name -> what is wrong with its value."""
        raise NotImplementedError

    def normalise(self) -> None:
        """Bring values that are right but oddly written into their usual form."""

    @classmethod
    def load(cls, raw: Any, where: str, problems: list[str]) -> Any:
        """Build the section from ``raw``, replacing each invalid value by its default."""
        section = cls()
        if raw is None:
            return section
        if not isinstance(raw, Mapping):
            problems.append(f"{where} is {raw!r}, not an object; the defaults are used")
            return section
        for key, value in raw.items():
            if key in cls.names():
                setattr(section, key, value)
            else:
                section.extra[str(key)] = value
        section.normalise()
        defaults = cls()
        for name, message in section.invalid().items():
            default = getattr(defaults, name)
            problems.append(f"{where}.{name} {message}; the default ({default!r}) is used")
            setattr(section, name, default)
        return section


@dataclass
class ModelSettings(_Section):
    """Which model does which kind of work, and how hard it thinks."""

    research: str = DEFAULT_MODEL
    reasoning: str = DEFAULT_MODEL
    coding: str = DEFAULT_MODEL
    fast: str = DEFAULT_MODEL
    effort: str = "high"
    extra: dict[str, Any] = field(default_factory=dict)

    def invalid(self) -> dict[str, str]:
        wrong: dict[str, str] = {}
        for name in ("research", "reasoning", "coding", "fast"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                wrong[name] = f"is {value!r}; it must be the name of a model"
        if self.effort not in EFFORTS:
            wrong["effort"] = f"is {self.effort!r}; it must be one of: {', '.join(EFFORTS)}"
        return wrong


@dataclass
class Limits(_Section):
    """What the project may spend before it stops at a checkpoint and says so."""

    model_usd: float = 25000.0
    experiment_seconds: float = 900
    experiment_usd: float = 5.0
    repair_rounds: int = 3
    max_sources: int = 40
    max_hypotheses: int = 6
    max_experiments: int = 6
    extra: dict[str, Any] = field(default_factory=dict)

    _COUNTS = ("repair_rounds", "max_sources", "max_hypotheses", "max_experiments")

    def invalid(self) -> dict[str, str]:
        wrong: dict[str, str] = {}
        for name in ("model_usd", "experiment_seconds", "experiment_usd"):
            value = getattr(self, name)
            # NaN is below no limit and above none, so it would limit nothing; neither
            # it nor infinity is JSON that another program reads.
            if not _is_number(value) or not math.isfinite(value) or value < 0:
                wrong[name] = f"is {value!r}; it must be a number, zero or more"
        for name in self._COUNTS:
            value = getattr(self, name)
            whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            if not whole or isinstance(value, bool) or value < 0:
                wrong[name] = f"is {value!r}; it must be a whole number, zero or more"
        return wrong

    def normalise(self) -> None:
        # JSON written by other tools often has 3.0 for 3.
        for name in self._COUNTS:
            value = getattr(self, name)
            if isinstance(value, float) and value.is_integer():
                setattr(self, name, int(value))


#: How the roles that use a model are run: ``classic`` is the tool loop of this package
#: over a provider's API; ``sdk`` makes each of them a Claude Agent SDK agent with a
#: shell, the web and the board tools.
AGENT_RUNTIMES: tuple[str, ...] = ("classic", "sdk")
#: Whose account an SDK agent works on: the person's Claude Code login, or the worker's
#: Workload Identity Federation settings.
AGENT_CREDENTIALS: tuple[str, ...] = ("login", "federation")
#: Makes ``sdk`` the runtime of projects that are created while it is set to ``sdk``.
AGENT_RUNTIME_ENV = "PAPER_ORCHESTRA_AGENT_RUNTIME"


#: What the name of a model looks like: letters, digits and the few marks model names
#: and their regional or versioned forms use.
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/\[\]-]{0,127}\Z")


@dataclass
class AgentSettings(_Section):
    """How the project's agents are run (``settings.agents``).

    ``runtime`` chooses between the two ways (:data:`AGENT_RUNTIMES`).  The other
    values concern the ``sdk`` runtime only: the ``model`` every agent works with,
    whether it is asked for in ``fast`` mode and with ``ultracode`` (sub-agents and
    workflows), which ``credential`` the agents use (:data:`AGENT_CREDENTIALS`), and
    whether they work inside the ``sandbox`` (their own folder, no credentials, no
    control files of the project).
    """

    runtime: str = "classic"
    model: str = DEFAULT_MODEL
    fast: bool = True
    ultracode: bool = True
    credential: str = "login"
    sandbox: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def invalid(self) -> dict[str, str]:
        wrong: dict[str, str] = {}
        if self.runtime not in AGENT_RUNTIMES:
            wrong["runtime"] = (
                f"is {self.runtime!r}; it must be one of: {', '.join(AGENT_RUNTIMES)}"
            )
        if not isinstance(self.model, str) or not _MODEL_NAME.match(self.model):
            # The name goes to the agents' command line as it stands: a name, nothing
            # with spaces or punctuation of a shell in it.
            wrong["model"] = (
                f"is {self.model!r}; it must be the name of a Claude model, such as {DEFAULT_MODEL}"
            )
        for name in ("fast", "ultracode", "sandbox"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                wrong[name] = f"is {value!r}; it must be true or false"
        if self.credential not in AGENT_CREDENTIALS:
            wrong["credential"] = (
                f"is {self.credential!r}; it must be one of: {', '.join(AGENT_CREDENTIALS)}"
            )
        return wrong

    @property
    def sdk(self) -> bool:
        """Whether the project's agents are Claude Agent SDK agents."""
        return self.runtime == "sdk"


def default_agent_runtime(environ: Mapping[str, str] | None = None) -> str:
    """The runtime a project gets when it is created and nothing names one:
    ``classic``, or ``sdk`` where :data:`AGENT_RUNTIME_ENV` says so.  Only asked when
    a project is created: a project that exists keeps what its file says, whatever
    the environment of the process that opens it."""
    environ = os.environ if environ is None else environ
    asked = str(environ.get(AGENT_RUNTIME_ENV) or "").strip().lower()
    return asked if asked in AGENT_RUNTIMES else "classic"


def settings_for_new_project(
    settings: Mapping[str, Any] | None, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """``settings`` as given for a project about to be created, with the runtime of its
    agents named: the one given, else :func:`default_agent_runtime`."""
    chosen = dict(settings or {})
    agents = chosen.get("agents")
    if agents is None or (isinstance(agents, Mapping) and "runtime" not in agents):
        chosen["agents"] = {**(agents or {}), "runtime": default_agent_runtime(environ)}
    return chosen


class RepoSettings(dict):
    """The person's repository for experiments in ``repo`` mode.

    It is the object written in ``project.json`` (``{"path": ..., "ref": ...}``) and
    stays a plain dict, so it compares and serialises as one; ``path`` and ``ref``
    also read as attributes, like the other groups of settings.
    """

    @property
    def path(self) -> str:
        """A local directory or a git URL."""
        return str(self.get("path") or "")

    @property
    def ref(self) -> str | None:
        """The branch, tag or commit to check out; ``None`` for the default."""
        value = self.get("ref")
        return str(value) if value not in (None, "") else None


def _repo_problem(value: Any) -> str | None:
    """What is wrong with a ``repo`` setting, or ``None`` when it is valid."""
    expected = 'it must be null or an object such as {"path": "/path/or/git-url", "ref": null}'
    if value is None:
        return None
    if isinstance(value, str):
        return None if value.strip() else f"is empty; {expected}"
    if not isinstance(value, Mapping):
        return f"is {value!r}; {expected}"
    path, ref = value.get("path"), value.get("ref")
    if not isinstance(path, str) or not path.strip():
        return f"has the path {path!r}; {expected}"
    if ref is not None and (not isinstance(ref, str) or not ref.strip()):
        return f"has the ref {ref!r}; it must be a branch, tag or commit, or null"
    return None


def _as_repo(value: Any) -> RepoSettings | None:
    """A valid ``repo`` setting in its usual form (a bare path is taken as the path)."""
    if value is None:
        return None
    if isinstance(value, str):
        return RepoSettings(path=value.strip(), ref=None)
    ref = value.get("ref")
    known = {"path": str(value["path"]).strip(), "ref": ref.strip() if ref else None}
    return RepoSettings({**value, **known})


@dataclass
class ProjectSettings:
    """Typed view of ``project.json["settings"]``.

    ``from_dict`` loads whatever is there (an older or hand-edited file): missing
    values get their defaults, invalid ones too and are listed in
    ``load_problems``, unknown keys are kept.  One value is not replaced by its
    default: a ``research_cutoff`` that cannot be read stays as it was written
    (:meth:`cutoff_problem` says what is wrong), because its default is "no
    cutoff".  ``validate`` is the strict check made before settings are saved.
    """

    models: ModelSettings = field(default_factory=ModelSettings)
    #: Ignore work published after this month (``YYYY-MM``); ``None`` for no cutoff.  After
    #: a lenient load it can hold a value that is neither: see :meth:`cutoff_problem`.
    research_cutoff: str | None = None
    compute: str = "local"
    #: kind -> ``auto`` (an agent's page is accepted on creation) or ``review``.
    review: dict[str, str] = field(default_factory=default_review_policy)
    limits: Limits = field(default_factory=Limits)
    #: How the agents are run: the classic tool loop, or Claude Agent SDK agents.
    agents: AgentSettings = field(default_factory=AgentSettings)
    lab_memory: bool = True
    #: The person's repository for experiments in ``repo`` mode; ``None`` when there is none.
    repo: RepoSettings | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: What ``from_dict`` had to replace by a default.  Not saved.
    load_problems: list[str] = field(default_factory=list, compare=False, repr=False)

    # -- reading -----------------------------------------------------------------

    def policy(self, kind: str) -> str:
        """``auto`` or ``review`` for an agent's page of ``kind``."""
        if kind not in REVIEWED_KINDS:
            return "auto"
        value = self.review.get(kind)
        return value if value in POLICIES else default_review_policy()[kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models.to_dict(),
            "research_cutoff": self.research_cutoff,
            "compute": self.compute,
            "review": dict(self.review),
            "limits": self.limits.to_dict(),
            "agents": self.agents.to_dict(),
            "lab_memory": self.lab_memory,
            # Written in its usual form whatever was assigned; an invalid value is written
            # as it is so that ``validate`` and a reader can see it.
            "repo": self.repo if _repo_problem(self.repo) else _as_repo(self.repo),
            **self.extra,
        }

    # -- checking ----------------------------------------------------------------

    def cutoff_problem(self) -> str | None:
        """What is wrong with ``research_cutoff``, as a sentence, or ``None`` when it can
        be applied.  A cutoff that cannot be read is never "no cutoff": whoever looks
        up literature for the project stops with this message instead."""
        cutoff = self.research_cutoff
        if cutoff is None or (isinstance(cutoff, str) and _CUTOFF.match(cutoff)):
            return None
        return (
            f"settings.research_cutoff is {cutoff!r}; it must be a date such as 2024-06 "
            "(YYYY, YYYY-MM or YYYY-MM-DD), or null for no cutoff"
        )

    def _own_problems(self) -> dict[str, str]:
        wrong: dict[str, str] = {}
        cutoff = self.cutoff_problem()
        if cutoff:
            wrong["research_cutoff"] = cutoff.removeprefix("settings.research_cutoff ")
        if self.compute not in COMPUTE_BACKENDS:
            wrong["compute"] = (
                f"is {self.compute!r}; it must be one of: {', '.join(COMPUTE_BACKENDS)}"
            )
        if not isinstance(self.lab_memory, bool):
            wrong["lab_memory"] = f"is {self.lab_memory!r}; it must be true or false"
        repo = _repo_problem(self.repo)
        if repo:
            wrong["repo"] = repo
        return wrong

    def _review_problems(self) -> dict[str, str]:
        if not isinstance(self.review, dict):
            return {"review": f"is {self.review!r}; it must map each kind to auto or review"}
        return {
            f"review.{kind}": f"is {value!r}; it must be one of: {', '.join(POLICIES)}"
            for kind, value in self.review.items()
            if kind in REVIEWED_KINDS and value not in POLICIES
        }

    def problems(self) -> list[str]:
        """Everything wrong with the current values, one message each; empty when valid."""
        found = [f"settings.models.{name} {text}" for name, text in self.models.invalid().items()]
        found += [f"settings.{name} {text}" for name, text in self._own_problems().items()]
        found += [f"settings.{name} {text}" for name, text in self._review_problems().items()]
        found += [f"settings.limits.{name} {text}" for name, text in self.limits.invalid().items()]
        found += [f"settings.agents.{name} {text}" for name, text in self.agents.invalid().items()]
        return found

    def validate(self) -> None:
        """Raise ``ProjectError`` naming every invalid value."""
        found = self.problems()
        if found:
            raise ProjectError("The project settings are not valid:\n" + "\n".join(found))

    # -- loading and changing ----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Any, *, strict: bool = False) -> ProjectSettings:
        """Settings from the JSON object ``data`` (see the class docstring).

        With ``strict`` an invalid value raises ``ProjectError`` instead of being
        replaced by its default.
        """
        problems: list[str] = []
        if data is not None and not isinstance(data, Mapping):
            problems.append(f"settings is {data!r}, not an object; the defaults are used")
        raw: Mapping[str, Any] = data if isinstance(data, Mapping) else {}
        known = {
            "models",
            "research_cutoff",
            "compute",
            "review",
            "limits",
            "agents",
            "lab_memory",
            "repo",
        }
        settings = cls(
            models=ModelSettings.load(raw.get("models"), "settings.models", problems),
            limits=Limits.load(raw.get("limits"), "settings.limits", problems),
            agents=AgentSettings.load(raw.get("agents"), "settings.agents", problems),
            extra={str(key): value for key, value in raw.items() if key not in known},
        )
        for name in ("research_cutoff", "compute", "lab_memory", "repo"):
            if name in raw:
                setattr(settings, name, raw[name])
        settings.research_cutoff = as_cutoff(settings.research_cutoff)
        defaults = cls()
        for name, message in settings._own_problems().items():
            if name == "research_cutoff":
                # Fail closed: the default would be "no cutoff", which lets in exactly
                # the work the person meant to leave out.
                kept = "" if strict else f"; {CUTOFF_STOPS_RESEARCH}"
                problems.append(f"settings.{name} {message}{kept}")
                continue
            default = getattr(defaults, name)
            problems.append(f"settings.{name} {message}; the default ({default!r}) is used")
            setattr(settings, name, default)
        settings.repo = _as_repo(settings.repo)
        review = raw.get("review")
        if isinstance(review, Mapping):
            settings.review.update({str(kind): value for kind, value in review.items()})
            for name, message in settings._review_problems().items():
                kind = name.split(".", 1)[1]
                default = defaults.review[kind]
                problems.append(f"settings.{name} {message}; the default ({default!r}) is used")
                settings.review[kind] = default
        elif review is not None:
            problems.append(f"settings.review is {review!r}, not an object; the defaults are used")
        if strict and problems:
            # A refusal uses nothing: it says what is wrong and stops there.
            said = [_DEFAULT_USED.sub("", problem) for problem in problems]
            raise ProjectError("The project settings are not valid:\n" + "\n".join(said))
        settings.load_problems = problems
        return settings

    def update(self, changes: Mapping[str, Any]) -> None:
        """Merge a partial settings object (as the web app sends one) into these.

        Nested objects are merged key by key.  All or nothing: when the result
        would be invalid, ``ProjectError`` names every problem and nothing changes.
        """
        if not isinstance(changes, Mapping):
            raise ProjectError(f"settings must be an object, not {changes!r}")
        agents = changes.get("agents")
        if isinstance(agents, Mapping):
            # A misspelt name here would be stored and silently do nothing, and one of
            # these values is the sandbox.  (A file that holds a name this version does
            # not know is still read: that is a newer version's value, kept as it is.)
            unknown = sorted(str(key) for key in agents if key not in AgentSettings.names())
            if unknown:
                raise ProjectError(
                    f"settings.agents has no value named {', '.join(unknown)}; it has: "
                    f"{', '.join(AgentSettings.names())}."
                )
        merged = self.to_dict()
        for key, value in changes.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        updated = ProjectSettings.from_dict(merged, strict=True)
        for name in (
            "models",
            "research_cutoff",
            "compute",
            "review",
            "limits",
            "agents",
            "lab_memory",
            "repo",
        ):
            setattr(self, name, getattr(updated, name))
        self.extra = updated.extra
