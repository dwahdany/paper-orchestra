# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The literature agent: search, verify, build the bibliography, write Intro + Related Work.

0. *Known papers*: when the materials hold ``known_papers.json`` (a paper written
   from a research project: the sources its researcher accepted and rejected), the
   bibliography starts from the papers listed there, under the citation keys they
   came with and without looking them up again, and a paper listed as excluded is
   never cited, whatever a search proposes.  Without the file nothing is known and
   the stage is the plain search below.
1. *Discovery*: one small web-search call per search task of the outline
   proposes candidate papers (title, year, reason).
2. *Verification*: every candidate has to be found in Semantic Scholar under
   (nearly) its title and be published before the research cutoff.  This is
   the gate that keeps hallucinated references out of the paper.  The verified
   papers are saved at once (``collected_papers.json``), so that a later
   attempt of the stage whose writing step failed does not search again.
3. *Assets*: ``references.bib``, ``citation_map.json`` and the outline with
   ``citation_candidates`` per section, all using the same citation keys.
4. *Writing*: the writer model fills the Introduction and Related Work of the
   template; a static check drives repair turns in the same conversation.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import re
import string
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from paper_orchestra.agents.outline import write_outline_page
from paper_orchestra.errors import stops_the_run
from paper_orchestra.llm.base import LLMParseError, WebSearch
from paper_orchestra.llm.parsing import parse_latex
from paper_orchestra.memory.harness import MemoryPolicy
from paper_orchestra.memory.wiki import WikiError
from paper_orchestra.prompts.literature import (
    DISCOVERY_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    exploration_discovery_prompt,
    min_cited,
    targeted_discovery_prompt,
    writer_repair_prompt,
    writer_user_prompt,
)
from paper_orchestra.utils.common import read_text, write_json, write_text
from paper_orchestra.utils.latex import cite_keys, static_check, strip_comments
from paper_orchestra.utils.scholar import (
    SemanticScholar,
    SemanticScholarError,
    is_before_cutoff,
    normalize_title,
    parse_cutoff,
    select_match,
)

if TYPE_CHECKING:
    from paper_orchestra.context import RunContext
    from paper_orchestra.memory.harness import ContextHarness, HarnessConversation

log = logging.getLogger(__name__)

AGENT = "LiteratureAgent"
STAGE = "literature"
DISCOVERY_LABEL = "literature.discovery"
LITERATURE_DIR = "wiki/literature"
#: Namespace of the cross-run Semantic Scholar cache (see ``memory.cache``).
CACHE_NAMESPACE = "semantic-scholar"
#: File in the stage's output directory that holds the verified papers of this run's search.
#: It is the one file there that an attempt of the stage reads back, so whoever clears the
#: directory between attempts has to leave it in place for the search not to be repeated.
CHECKPOINT_NAME = "collected_papers.json"
_CHECKPOINT_VERSION = 1
#: File in the materials directory that names the papers the bibliography starts from and
#: the papers that must not be cited: ``{"version": 1, "papers": [<the fields of PaperData>],
#: "excluded": [{"citation_key", "title", "reason"}]}``.  ``research/writeup.py`` writes it
#: for a paper written from a research project (the sources the researcher accepted and
#: rejected); a paper written from raw materials alone has none.
KNOWN_PAPERS_NAME = "known_papers.json"
_KNOWN_PAPERS_VERSION = 1
#: Added to the reason of a known paper that has no abstract.  A search drops such a paper,
#: because nothing says what it found; this one the person accepted, so it stays, and
#: whoever cites it is told what is not on record.
_NO_ABSTRACT_NOTE = (
    "No abstract is on record for it: cite it for what its title says and nothing more."
)
#: What a citation key taken from that file may look like.  It becomes a BibTeX key, an
#: argument of ``\cite`` and the name of a wiki page, and is used as it stands; the page is
#: a file, so the key has to leave room in a file name for what is written around it.
_MAX_KEY_CHARS = 200
_CITABLE_KEY = re.compile(rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,{_MAX_KEY_CHARS - 1}}}")
#: How many of the problems with that file the note in the run's memory spells out.  The
#: log and the stage's result have every one; a note is something a later session reads.
_MAX_PROBLEMS_IN_NOTE = 5

INTRO_SECTION = "Introduction"
DEFAULT_VENUE = "arXiv"
ABSTRACT_LIMIT = 1500
_ABSTRACT_MARK = "... [Truncated]"

_TARGETED_SEARCHES = 3
_EXPLORATION_SEARCHES = 6
_MAX_CANDIDATES_PER_TASK = 20
#: After this many lookups in a row have failed, Semantic Scholar is treated as down for the
#: rest of the run instead of waiting out the retries of every remaining candidate.
_MAX_CONSECUTIVE_LOOKUP_FAILURES = 5

# The cache is a fixed set of shard pages, each kept under the wiki's soft page size, so it
# stays bounded however many papers pass through the lab memory.

#: Words that are skipped when a citation key is built from a title.
_KEY_STOPWORDS = frozenset(
    {
        "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are", "with", "by",
        "study",
    }
)  # fmt: skip
_PROCEEDINGS = re.compile(r"conference|proceedings|workshop|symposium|meeting", re.IGNORECASE)
_SECTION_PLACEHOLDERS = (
    ("Introduction", "Introduction here."),
    ("Related Work", "Related work here."),
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class CandidatePaper(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(description="Exact official title of the paper.")
    year: int | None = Field(description="Year of publication, or null if unknown.")
    reason: str = Field(
        description="One sentence on what the paper contributes and why it fits the task."
    )

    @field_validator("year", mode="before")
    @classmethod
    def _coerce_year(cls, value: Any) -> int | None:
        # "2021", "c. 2021" or a missing year must not invalidate an otherwise good answer.
        if isinstance(value, str):
            match = re.search(r"\d{4}", value)
            return int(match.group()) if match else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return int(value)
            except (ValueError, OverflowError):  # NaN, infinity
                return None
        return None


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: list[CandidatePaper]

    @field_validator("candidates", mode="before")
    @classmethod
    def _keep_usable(cls, value: Any) -> Any:
        """Skip entries without a title instead of rejecting the whole answer.

        Only the title is indispensable (it is what gets verified), so a
        missing year or reason is filled in rather than treated as an error.
        """
        if value is None:  # "no candidate" spelled as null
            return []
        if not isinstance(value, list):
            return value
        usable: list[Any] = []
        for item in value:
            if isinstance(item, str):
                item = {"title": item}
            if isinstance(item, CandidatePaper):
                usable.append(item)
            elif (
                isinstance(item, Mapping)
                and isinstance(item.get("title"), str)
                and item["title"].strip()
            ):
                reason = item.get("reason")
                usable.append(
                    {
                        **item,
                        "year": item.get("year"),
                        "reason": "" if reason is None else str(reason),
                    }
                )
        if len(usable) < len(value):
            log.warning(
                "Ignored %d candidate(s) without a title in a discovery answer",
                len(value) - len(usable),
            )
        return usable


class PaperData(BaseModel):
    """A verified paper.  Bibliographic fields come from Semantic Scholar only: from this
    stage's own lookup or, for a known paper (``known_papers.json``), from the record it
    was put on when it was verified."""

    citation_key: str
    title: str
    authors: list[str]
    venue: str
    year: int | None
    abstract: str
    citation_count: int
    #: The outline section whose search found the paper first (empty for a known paper
    #: that no search returned).
    found_in_section: str
    #: Every outline section whose searches returned it, ``found_in_section`` first.
    sections: list[str]
    reason: str
    journal: str | None = None
    volume: str | None = None
    pages: str | None = None
    publication_date: str | None = None


@dataclass(frozen=True)
class SearchTask:
    section: str
    focus: str
    context: str
    search_type: Literal["exploration", "targeted"]


@dataclass
class LiteratureStats:
    tasks: int = 0
    failed_tasks: int = 0
    empty_tasks: int = 0
    candidates: int = 0
    duplicates: int = 0
    cache_hits: int = 0
    lookups: int = 0
    lookup_errors: int = 0
    skipped_lookups: int = 0
    unmatched: int = 0
    after_cutoff: int = 0
    no_abstract: int = 0
    verified: int = 0
    #: Papers the collection started from (``known_papers.json``); not among ``verified``.
    known: int = 0
    #: Candidates left out because the materials say the paper must not be cited.
    excluded: int = 0

    def describe(self) -> str:
        text = (
            f"{self.tasks} search task(s) ({self.failed_tasks} failed, {self.empty_tasks} "
            f"without candidates), {self.candidates} candidate(s), {self.verified} verified "
            f"paper(s); {self.duplicates} duplicate(s), {self.unmatched} not matched in "
            f"Semantic Scholar before the cutoff, {self.after_cutoff} past the cutoff, "
            f"{self.no_abstract} without an abstract, {self.lookup_errors} failed lookup(s), "
            f"{self.skipped_lookups} skipped lookup(s), {self.lookups} live lookup(s), "
            f"{self.cache_hits} cache hit(s)"
        )
        if self.known or self.excluded:
            # Only where there is something to say: a paper written from raw materials
            # alone is summed up in the words it always was.
            text += (
                f"; {self.known} paper(s) known before the search, {self.excluded} "
                "candidate(s) left out because the researcher rejected the paper as a source"
            )
        return text


@dataclass
class LiteratureResult:
    outline: dict[str, Any]
    citation_map: dict[str, dict[str, Any]]
    bibtex: str
    latex: str
    papers: int
    #: How many of ``papers`` were known before the search (``known_papers.json``).
    known: int = 0
    #: Candidates of the search that were left out because the materials say the paper
    #: must not be cited (a source the researcher rejected).
    excluded: int = 0
    #: What the person has to be told about the known papers: what in the file could not
    #: be used, and which of its papers are not in the bibliography and why.
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnownPapers:
    """What the materials say about the literature before any search.

    ``papers`` are cited without a lookup (they were verified when they were put on
    record), under the citation keys they came with.  A paper whose normalised title
    is in ``excluded`` is never cited, and ``reserved_keys`` (case-folded) are the
    citation keys of those papers (as far as they could be citation keys), which no
    other paper is given: a key the person saw on a paper they rejected does not come
    back on another one.  ``problems`` says, for the person, what in the file could
    not be used and which of its papers were left out.
    """

    papers: tuple[PaperData, ...] = ()
    excluded: frozenset[str] = frozenset()
    reserved_keys: frozenset[str] = frozenset()
    problems: tuple[str, ...] = ()


@dataclass
class _Collection:
    """What one run has gathered so far.  Only the calling thread touches it."""

    #: Normalised Semantic Scholar title -> paper, in the order the papers were verified.
    papers: dict[str, PaperData] = field(default_factory=dict)
    #: Normalised candidate title -> normalised title of the paper it resolved to
    #: (``None``: looked up and rejected), so no title is looked up twice.
    resolved: dict[str, str | None] = field(default_factory=dict)
    #: Case-folded citation keys already handed out.
    keys: set[str] = field(default_factory=set)
    #: Normalised titles of the papers that must not be collected (``KnownPapers.excluded``),
    #: and the other wordings under which a search proposed one of them.
    excluded: set[str] = field(default_factory=set)
    stats: LiteratureStats = field(default_factory=LiteratureStats)
    consecutive_lookup_failures: int = 0
    scholar_down: bool = False


@dataclass
class _Draft:
    latex: str
    blocking: list[str]
    advisory: list[str]

    def rank(self) -> tuple[bool, int, int]:
        """Sort key, lower is better.

        A complete document always beats a truncated one: only a complete one can be kept
        when the repair turns run out, however many other problems it has.
        """
        return not _is_complete_document(self.latex), len(self.blocking), len(self.advisory)


# ---------------------------------------------------------------------------
# Known papers
# ---------------------------------------------------------------------------


def load_known_papers(materials_dir: Path, *, cutoff: str | None = None) -> KnownPapers:
    """What ``known_papers.json`` in ``materials_dir`` says, as far as it can be used.

    A missing file means that nothing is known.  Nothing the file holds makes this
    raise: what cannot be read, or is not what the format says, becomes an entry of
    ``problems`` and the rest is used, so that one bad entry costs neither the other
    papers nor the paper run.  A listed paper is left out, with a problem that names
    it, when it has no title, when its citation key cannot be used as it stands, when
    it is also listed as excluded, when it repeats the title or the citation key of
    an earlier one, or when it was not published before ``cutoff`` (the rule every
    other paper is held to; ``cutoff`` must be one that ``parse_cutoff`` reads).
    The papers that are kept come in the shape the papers of a search are kept in
    (see :func:`_as_kept`), so that whatever the file holds can be written to the
    files of the run.  A paper that is left out is not thereby kept out (only an
    excluded one is): a search that proposes it has it verified like any other.
    """
    path = Path(materials_dir) / KNOWN_PAPERS_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return KnownPapers()
    except (OSError, ValueError) as exc:  # a directory, no permission, not UTF-8
        return KnownPapers(problems=(_unusable_file(path, f"it could not be read ({exc})"),))
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        return KnownPapers(problems=(_unusable_file(path, f"it is not valid JSON ({exc})"),))
    if not isinstance(data, dict):
        why = "it does not hold an object with 'papers' and 'excluded'"
        return KnownPapers(problems=(_unusable_file(path, why),))
    if data.get("version") != _KNOWN_PAPERS_VERSION:
        why = (
            f"its version is {_quoted(data.get('version'), 40)}, and this version of the "
            f"program reads version {_KNOWN_PAPERS_VERSION}"
        )
        return KnownPapers(problems=(_unusable_file(path, why),))
    problems: list[str] = []
    excluded, reserved = _read_excluded(path, data.get("excluded"), problems)
    papers = _read_known(path, data.get("papers"), excluded, cutoff, problems)
    return KnownPapers(tuple(papers), frozenset(excluded), frozenset(reserved), tuple(problems))


def _quoted(text: Any, limit: int = 120) -> str:
    """Words from the file (a title, a key) as a quotation in a sentence this stage says.

    On one line, cut to ``limit`` characters and with nothing inside that reads as the
    end of the quotation: the sentence goes into the log, the stage's record and the
    run's memory, where a title must not read as a further sentence of ours.
    """
    line = clean_text(text) or ""
    line = "".join(
        "'" if char == '"' or unicodedata.category(char) in ("Pi", "Pf") else char for char in line
    )
    if len(line) > limit:
        line = line[: limit - 1].rstrip() + "…"
    return f'"{line}"'


def _unusable_file(path: Path, why: str) -> str:
    return (
        f"The list of papers this paper starts from ({path}) cannot be used: {why}. The "
        "literature is searched without it: the sources accepted in the research wiki are "
        "cited only if the search finds them, and the rejected ones are not kept out. "
        "Correct or remove the file (starting the write-up of a research project again "
        "writes it anew), then run the same command again."
    )


def _read_excluded(path: Path, listed: Any, problems: list[str]) -> tuple[set[str], set[str]]:
    """The papers not to cite: their normalised titles, and their citation keys
    (case-folded) where those are keys a paper could be given."""
    titles: set[str] = set()
    keys: set[str] = set()
    if listed is None:
        return titles, keys
    if not isinstance(listed, list):
        problems.append(
            f"{path}: 'excluded' is not a list, so it keeps no paper out: a source rejected "
            "in the research wiki is cited if the search finds it. Correct the file, then "
            "run the same command again."
        )
        return titles, keys
    for index, entry in enumerate(listed):
        title = entry.get("title") if isinstance(entry, Mapping) else entry
        wanted = normalize_title(title) if isinstance(title, str) else ""
        if not wanted:
            problems.append(
                f"{path}: excluded[{index}] names no title, so it keeps no paper out. "
                "Correct the file, then run the same command again."
            )
            continue
        titles.add(wanted)
        key = entry.get("citation_key") if isinstance(entry, Mapping) else None
        # Only a key that a paper could be given needs keeping free; any other text is not
        # held on to (it would go into the record of the search, which is a file).
        if isinstance(key, str) and _CITABLE_KEY.fullmatch(key.strip()):
            keys.add(key.strip().casefold())
    return titles, keys


def _read_known(
    path: Path, listed: Any, excluded: set[str], cutoff: str | None, problems: list[str]
) -> list[PaperData]:
    """The listed papers that can be cited, in the order of the list."""
    if listed is None:
        return []
    if not isinstance(listed, list):
        problems.append(
            f"{path}: 'papers' is not a list, so no paper is taken from it: the sources "
            "accepted in the research wiki are cited only if the search finds them. Correct "
            "the file, then run the same command again."
        )
        return []
    by_title: dict[str, PaperData] = {}
    by_key: dict[str, PaperData] = {}
    for index, entry in enumerate(listed):
        try:
            paper = _as_kept(PaperData.model_validate(entry))
        except ValidationError as exc:
            problems.append(
                f"{path}: papers[{index}] is not a paper record ({_validation_text(exc)}), "
                "so it is not in the bibliography. Correct the file, then run the same "
                "command again."
            )
            continue
        reason = _why_not_cited(paper, by_title, by_key, excluded, cutoff)
        if reason is not None:
            key = paper.citation_key
            name = _quoted(paper.title) + (f" ({key})" if _CITABLE_KEY.fullmatch(key) else "")
            # "Left out of the papers taken from the wiki", not "not cited": the search
            # may still propose the paper, and then it is verified like any other.
            problems.append(
                f"The accepted source {name} is left out of the papers taken from the "
                f"wiki: {reason}"
            )
            continue
        if not paper.abstract:
            said = f"{paper.reason} {_NO_ABSTRACT_NOTE}".strip()
            paper = paper.model_copy(update={"reason": said})
        by_title[normalize_title(paper.title)] = paper
        by_key[paper.citation_key.casefold()] = paper
    return list(by_title.values())


def _as_kept(paper: PaperData) -> PaperData:
    """``paper`` in the shape the papers of a search are kept in.

    A search keeps what ``public_record`` made of a Semantic Scholar record: text
    without control characters or unpaired surrogates (which no file of the run could
    hold), on one line except for the abstract, a venue to print and an abstract of
    bounded length.  The list is a file somebody wrote, so its papers are brought
    into that shape here; nothing else about them is changed.  The citation key is
    left as it is: it is checked (:func:`_why_not_cited`), not corrected.
    """
    return paper.model_copy(
        update={
            "title": clean_text(paper.title) or "",
            "authors": [name for author in paper.authors if (name := clean_text(author))],
            "venue": clean_text(paper.venue) or DEFAULT_VENUE,
            "abstract": _clip_abstract(_strip_controls(paper.abstract).strip()),
            "reason": clean_text(paper.reason) or "",
            "found_in_section": clean_text(paper.found_in_section) or "",
            "sections": [name for section in paper.sections if (name := clean_text(section))],
            "journal": clean_text(paper.journal),
            "volume": clean_text(paper.volume),
            "pages": clean_text(paper.pages),
            "publication_date": clean_text(paper.publication_date),
        }
    )


def _why_not_cited(
    paper: PaperData,
    by_title: Mapping[str, PaperData],
    by_key: Mapping[str, PaperData],
    excluded: set[str],
    cutoff: str | None,
) -> str | None:
    """Why a listed paper is not taken into the bibliography from the list, as the end of
    a sentence; or ``None``."""
    title, key = normalize_title(paper.title), paper.citation_key
    if not title:
        return "it has no title."
    if not _CITABLE_KEY.fullmatch(key):
        return (
            f"its citation key {_quoted(key, 60)} cannot be used as one (letters, digits, "
            f"'.', '_' and '-' only, starting with a letter or a digit, at most "
            f"{_MAX_KEY_CHARS} characters)."
        )
    if title in excluded:
        return (
            "it is also among the papers not to be cited (the rejected sources), and the "
            "rejection stands."
        )
    if title in by_title:
        return f"it has the same title as {by_title[title].citation_key}, which is cited."
    if key.casefold() in by_key:
        other = by_key[key.casefold()]
        return f"its citation key is also that of {_quoted(other.title)}, which is cited under it."
    if not is_before_cutoff(paper.publication_date, paper.year, cutoff):
        return _past_the_cutoff(paper, cutoff)
    return None


def _past_the_cutoff(paper: PaperData, cutoff: str | None) -> str:
    if paper.publication_date:
        said = (
            f"its publication date, {paper.publication_date}, is not before the research "
            f"cutoff {cutoff}"
        )
    elif paper.year:
        said = (
            f"only its year, {paper.year}, is on record, and a paper known by its year alone "
            f"is cited only when that year is before the year of the research cutoff {cutoff}"
        )
    else:
        said = (
            "neither a publication date nor a year is on record for it, so it cannot be "
            f"shown to precede the research cutoff {cutoff}"
        )
    return f"{said}. Move the cutoff (--research-cutoff) if the paper should be cited."


def _in_short(problems: Sequence[str]) -> str:
    """The first of ``problems`` and how many more there are, for the note in the run's memory.

    A list of sources can leave out as many papers as it has (a cutoff before all of
    them), and a note that repeats a sentence for each is no observation; one longer
    than a page of the memory could not be written at all.
    """
    shown = " ".join(problems[:_MAX_PROBLEMS_IN_NOTE])
    more = len(problems) - _MAX_PROBLEMS_IN_NOTE
    if more <= 0:
        return shown
    return f"{shown} And {more} more of the kind: the log of the run names every one."


def _validation_text(exc: ValidationError) -> str:
    """The first complaints of a failed validation, on one line."""
    errors = exc.errors(include_url=False, include_input=False)
    said = [
        f"{'.'.join(str(part) for part in error['loc']) or 'the entry'}: {error['msg']}"
        for error in errors[:3]
    ]
    more = len(errors) - len(said)
    return "; ".join(said) + (f"; and {more} more" if more > 0 else "")


# ---------------------------------------------------------------------------
# Search tasks and the outline
# ---------------------------------------------------------------------------


def _related_work_section(subsection: Mapping[str, Any]) -> str:
    return f"Related Work: {subsection.get('subsection_title') or 'General'}"


def _section_name(section: Mapping[str, Any]) -> str:
    return str(section.get("section_title") or "Unknown Section")


def _subsection_name(section: Mapping[str, Any], subsection: Mapping[str, Any]) -> str:
    return f"{_section_name(section)} - {subsection.get('subsection_title') or 'General'}"


def collect_search_tasks(outline: Mapping[str, Any]) -> list[SearchTask]:
    """The search tasks an outline asks for, in outline order and without repeats.

    Three families: the Introduction's search directions (exploration), the
    limitation queries of each Related Work cluster (exploration), and the
    citation hints of the section plan (targeted: one specific paper each).
    """
    tasks: list[SearchTask] = []
    seen: set[tuple[str, str]] = set()

    def add(section: str, focus: Any, context: str, search_type: Any) -> None:
        focus = " ".join(str(focus or "").split())
        if focus and (section, focus) not in seen:
            seen.add((section, focus))
            tasks.append(SearchTask(section, focus, context, search_type))

    plan = outline.get("intro_related_work_plan") or {}
    intro = plan.get("introduction_strategy") or {}
    intro_context = (
        f"Hook: {intro.get('hook_hypothesis', '')}. Gap: {intro.get('problem_gap_hypothesis', '')}"
    )
    for direction in intro.get("search_directions") or []:
        add(INTRO_SECTION, direction, intro_context, "exploration")

    for cluster in (plan.get("related_work_strategy") or {}).get("subsections") or []:
        queries = cluster.get("limitation_search_queries") or [
            f"{cluster.get('subsection_title', '')} {cluster.get('methodology_cluster', '')}"
        ]
        context = (
            f"Mission: {cluster.get('sota_investigation_mission', '')}. "
            f"Hypothesis: {cluster.get('limitation_hypothesis', '')}"
        )
        for query in queries:
            add(_related_work_section(cluster), query, context, "exploration")

    for section in outline.get("section_plan") or []:
        # Hints attached to a section without subsections (hand-written outlines).
        for hint in section.get("citation_hints") or []:
            context = f"Must-have citation for the section: {_section_name(section)}"
            add(_section_name(section), hint, context, "targeted")
        for subsection in section.get("subsections") or []:
            covering = " ".join(str(b) for b in subsection.get("content_bullets") or [])
            for hint in subsection.get("citation_hints") or []:
                add(
                    _subsection_name(section, subsection),
                    hint,
                    f"Must-have citation for section covering: {covering}",
                    "targeted",
                )
    return tasks


def _core_problem(outline: Mapping[str, Any]) -> str:
    """The problem the paper addresses, which every exploration search is told about."""
    intro = (outline.get("intro_related_work_plan") or {}).get("introduction_strategy") or {}
    return str(intro.get("problem_gap_hypothesis") or "N/A")


def _search_fingerprint(
    tasks: Sequence[SearchTask],
    core_problem: str,
    cutoff: str | None,
    model: str,
    known: KnownPapers | None = None,
) -> str:
    """Digest of what a literature search is asked: the tasks, the cutoff, who searches,
    and the papers it starts from and must leave out.

    A saved collection answers exactly this question.  When the outline is regenerated, the
    cutoff moves, another model is put in charge of discovery or the known papers change
    (the researcher accepted or rejected a source), the digest differs and the search
    runs again.
    """
    asked: dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "tasks": [[task.search_type, task.section, task.focus, task.context] for task in tasks],
        "core_problem": core_problem,
        "cutoff": cutoff,
        "model": model,
    }
    if known is not None and (known.papers or known.excluded or known.reserved_keys):
        # Left out when nothing is known, so that a collection saved for a paper written
        # from raw materials alone (also by an earlier version) still answers the question.
        asked["known"] = {
            "papers": [paper.model_dump() for paper in known.papers],
            "excluded": sorted(known.excluded),
            "reserved_keys": sorted(known.reserved_keys),
        }
    text = json.dumps(asked, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inject_citations(outline: dict[str, Any], papers: Sequence[PaperData]) -> dict[str, Any]:
    """Add ``citation_candidates`` to every part of the outline (in place).

    Each part lists the keys of the papers its own searches returned.  The
    ``citation_hints`` of the section plan are removed: they have been resolved
    into candidates, and the writers must cite keys, not hints.
    """
    by_section: dict[str, list[str]] = {}
    for paper in papers:
        for section in paper.sections:
            by_section.setdefault(section, []).append(paper.citation_key)

    plan = outline.setdefault("intro_related_work_plan", {})
    intro = plan.setdefault("introduction_strategy", {})
    intro["citation_candidates"] = by_section.get(INTRO_SECTION, [])
    for cluster in plan.setdefault("related_work_strategy", {}).get("subsections") or []:
        cluster["citation_candidates"] = by_section.get(_related_work_section(cluster), [])

    for section in outline.get("section_plan") or []:
        if "citation_hints" in section:
            del section["citation_hints"]
            section["citation_candidates"] = by_section.get(_section_name(section), [])
        for subsection in section.get("subsections") or []:
            subsection["citation_candidates"] = by_section.get(
                _subsection_name(section, subsection), []
            )
            subsection.pop("citation_hints", None)
    return outline


# ---------------------------------------------------------------------------
# Citation keys and BibTeX
# ---------------------------------------------------------------------------


# Letters that Unicode does not decompose into a base letter plus a mark.
_UNDECOMPOSED_LETTERS = str.maketrans(
    {
        "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "Th", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ı": "i",
    }
)  # fmt: skip


def _ascii(text: str) -> str:
    """Closest ASCII rendering of ``text``; characters without one are dropped."""
    text = text.translate(_UNDECOMPOSED_LETTERS)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def make_citation_key(authors: Sequence[str], year: int | None, title: str) -> str:
    """``<Surname><year><First two title words>``, ASCII letters and digits only."""
    surname = authors[0].split()[-1] if authors and authors[0].split() else ""
    author = re.sub(r"[^A-Za-z]", "", _ascii(surname)).capitalize() or "Unknown"
    words = re.sub(r"[^a-z0-9\s]", "", _ascii(title).lower()).split()
    meaningful = [word.capitalize() for word in words if word not in _KEY_STOPWORDS]
    return f"{author}{year or 'nd'}{''.join(meaningful[:2]) or 'Paper'}"


def _suffixes() -> Iterator[str]:
    """a, b, ..., z, aa, ab, ..."""
    for length in itertools.count(1):
        for letters in itertools.product(string.ascii_lowercase, repeat=length):
            yield "".join(letters)


def unique_citation_key(base: str, taken: set[str]) -> str:
    """``base``, or ``base`` plus a letter suffix, that no earlier paper uses.

    ``taken`` holds case-folded keys and is updated.  Keys are compared without
    regard to case because BibTeX treats ``Smith2020Deep`` and ``smith2020deep``
    as the same entry (and so do case-insensitive file systems, for the wiki
    page named after the key).
    """
    key = base
    if key.casefold() in taken:
        key = next(
            candidate
            for candidate in (base + suffix for suffix in _suffixes())
            if candidate.casefold() not in taken
        )
    taken.add(key.casefold())
    return key


# BibTeX counts braces without regard to backslashes, so ``\{`` would still unbalance a field:
# literal braces are spelled with macros whose own braces pair up.
_BIBTEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "{": r"\textbraceleft{}",
    "}": r"\textbraceright{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # In the default (OT1) text fonts these three print as other glyphs: ¡, ¿ and a dash.
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}
# The two tables below hold only spellings that the default OT1 font encoding has, because
# the venue templates do not load T1.  The ogonek (``\k``), eth, thorn and d-with-stroke
# raise "Command unavailable in encoding OT1" there, so those letters are reduced to their
# closest plain ones instead.

#: Unicode combining mark -> the LaTeX accent command that draws it.
_COMBINING_ACCENTS = {
    "\u0300": "`",  # grave
    "\u0301": "'",  # acute
    "\u0302": "^",  # circumflex
    "\u0303": "~",  # tilde
    "\u0304": "=",  # macron
    "\u0306": "u",  # breve
    "\u0307": ".",  # dot above
    "\u0308": '"',  # diaeresis
    "\u030a": "r",  # ring above
    "\u030b": "H",  # double acute
    "\u030c": "v",  # caron
    "\u0327": "c",  # cedilla
}
_LATEX_SPELLINGS = {
    "ß": r"{\ss}",
    "ø": r"{\o}",
    "Ø": r"{\O}",
    "æ": r"{\ae}",
    "Æ": r"{\AE}",
    "œ": r"{\oe}",
    "Œ": r"{\OE}",
    "ł": r"{\l}",
    "Ł": r"{\L}",
    "ı": r"{\i}",
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "--",  # figure dash
    "–": "--",
    "—": "---",
    "\u2212": "-",  # minus sign
    "×": r"{$\times$}",
    "µ": r"{$\mu$}",  # micro sign
    "‘": "`",
    "’": "'",
    "\u02bc": "'",  # modifier letter apostrophe
    "“": "``",
    "”": "''",
    "…": r"\ldots{}",
    "\u00a0": "~",  # no-break space
}
# Greek letters that LaTeX has a math macro for (\alpha, \Gamma); any other Greek letter
# would be an undefined control sequence, so it is replaced by a Latin letter.
_GREEK_SMALL_WITH_MACRO = frozenset(
    {
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
        "omega",
    }
)  # fmt: skip
_GREEK_CAPITALS_WITH_MACRO = frozenset(
    {"gamma", "delta", "theta", "lambda", "xi", "pi", "sigma", "upsilon", "phi", "psi", "omega"}
)


def _latex_for(char: str) -> str:
    """LaTeX spelling of one non-ASCII character; its closest ASCII form, or nothing, if none."""
    if char in _LATEX_SPELLINGS:
        return _LATEX_SPELLINGS[char]
    base, *marks = unicodedata.normalize("NFD", char)
    if base.isascii() and base.isalpha() and len(marks) == 1 and marks[0] in _COMBINING_ACCENTS:
        accent = _COMBINING_ACCENTS[marks[0]]
        if base in "ij" and accent != "c":
            base = f"\\{base}"  # an accent above replaces the dot
        return f"{{\\{accent}{{{base}}}}}"
    name = unicodedata.name(char, "")
    if name.startswith(("GREEK SMALL LETTER ", "GREEK CAPITAL LETTER ")) and not marks:
        # Unicode names the eleventh letter "lamda"; the macro is \lambda.
        letter = name.rsplit(" ", 1)[-1].lower().replace("lamda", "lambda")
        if name.startswith("GREEK CAPITAL"):
            if letter in _GREEK_CAPITALS_WITH_MACRO:
                return f"{{$\\{letter.capitalize()}$}}"
            return letter[0].upper()
        if letter in _GREEK_SMALL_WITH_MACRO:
            return f"{{$\\{letter}$}}"
        return letter[0]
    # A compatibility character can reduce to a LaTeX special (a full-width ampersand to "&").
    return "".join(_BIBTEX_SPECIALS.get(reduced, reduced) for reduced in _ascii(char))


def escape_bibtex(text: str) -> str:
    """Make ``text`` safe as a BibTeX field value.

    LaTeX special characters are escaped in a way that keeps the braces of
    the field balanced, accented letters are spelled with accent macros so
    that BibTeX and pdflatex agree on them, and characters with no LaTeX
    spelling are reduced to ASCII rather than left to break the ``.bbl``.
    """
    out: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        if char in _BIBTEX_SPECIALS:
            out.append(_BIBTEX_SPECIALS[char])
        elif char.isascii():
            if char.isprintable() or char.isspace():  # TeX rejects control characters
                out.append(char)
        else:
            out.append(_latex_for(char))
    return " ".join("".join(out).split())


def _bibtex_author(name: str) -> str:
    """One name of a BibTeX ``author`` field (empty when nothing printable is left).

    BibTeX reports an error for a name with more than two commas (in practice an
    affiliation that was extracted as an author), and a square bracket in a name ends the
    optional argument of natbib's ``\\bibitem[...]`` early.  Braces make both be taken
    literally.
    """
    escaped = escape_bibtex(name).strip(", ")
    if escaped.count(",") > 2 or "[" in escaped or "]" in escaped:
        return "{" + escaped + "}"
    return escaped


def bibtex_entry(paper: PaperData) -> str:
    fields: list[tuple[str, str]] = [
        # The inner braces keep the capitalisation of the title under every bibliography style.
        ("title", "{" + escape_bibtex(paper.title) + "}"),
        ("author", " and ".join(filter(None, map(_bibtex_author, paper.authors))) or "Unknown"),
    ]
    # Semantic Scholar files preprints under a journal called "ArXiv" (volume "abs/<id>") and
    # proceedings under a "journal" named after the conference.
    source = paper.journal or paper.venue
    volume = paper.volume or ""
    is_arxiv = source.lower().startswith("arxiv")
    is_journal = bool(paper.journal) and not is_arxiv and not _PROCEEDINGS.search(source)
    if is_arxiv:
        identifier = volume.removeprefix("abs/") if volume.startswith("abs/") else ""
        suffix = f" arXiv:{escape_bibtex(identifier)}" if identifier else ""
        fields.append(("journal", f"arXiv preprint{suffix}"))
    elif is_journal:
        fields.append(("journal", escape_bibtex(source)))
    else:
        fields.append(("booktitle", escape_bibtex(source)))
    if paper.year:
        fields.append(("year", str(paper.year)))
    if is_journal and volume:
        fields.append(("volume", escape_bibtex(volume)))
    if paper.pages and not is_arxiv:
        fields.append(("pages", escape_bibtex(re.sub(r"\s*-+\s*", "--", paper.pages))))
    entry_type = "article" if is_arxiv or is_journal else "inproceedings"
    body = ",\n".join(f"  {name}={{{value}}}" for name, value in fields)
    return f"@{entry_type}{{{paper.citation_key},\n{body}\n}}"


def generate_bibtex(papers: Sequence[PaperData]) -> str:
    """``references.bib``: one entry per paper (an empty string when there are none)."""
    return "\n\n".join(bibtex_entry(paper) for paper in papers) + ("\n" if papers else "")


def generate_citation_map(papers: Sequence[PaperData]) -> dict[str, dict[str, Any]]:
    return {
        paper.citation_key: {
            "citation_key": paper.citation_key,
            "title": paper.title,
            "authors": paper.authors,
            "venue": paper.venue,
            "year": paper.year,
            "abstract": paper.abstract.strip(),
        }
        for paper in papers
    }


# ---------------------------------------------------------------------------
# Semantic Scholar records
# ---------------------------------------------------------------------------


def _strip_controls(text: str) -> str:
    """``text`` without control, format and unpaired surrogate characters (white space stays).

    Scraped metadata carries them now and then (a NUL, a soft hyphen inside a word, half
    of an emoji).  TeX rejects the first as an invalid character and the last cannot be
    written to a UTF-8 file at all.
    """
    return "".join(c for c in text if c.isspace() or unicodedata.category(c)[0] != "C")


def clean_text(value: Any) -> str | None:
    """``value`` as one line of text, or ``None`` when nothing is left of it."""
    text = " ".join(_strip_controls(str(value)).split()) if value is not None else ""
    return text or None


def _clip_abstract(abstract: str) -> str:
    """Bound the abstract (it is repeated in several prompts).  Applying it twice is a no-op."""
    if len(abstract) <= ABSTRACT_LIMIT:
        return abstract
    return abstract[:ABSTRACT_LIMIT] + _ABSTRACT_MARK


def public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The bibliographic fields of a Semantic Scholar result that the pipeline uses.

    Total on purpose: the input is a service's answer or a cache page, and a field of an
    unexpected type becomes "unknown" instead of an exception that would fail the stage.
    """
    journal = record.get("journal") if isinstance(record.get("journal"), Mapping) else {}
    authors = record.get("authors") if isinstance(record.get("authors"), list) else []
    year, abstract, citations = (record.get(k) for k in ("year", "abstract", "citationCount"))
    return {
        "title": clean_text(record.get("title")) or "",
        "authors": [
            {"name": name}
            for author in authors
            if isinstance(author, Mapping) and (name := clean_text(author.get("name")))
        ],
        "venue": clean_text(record.get("venue")),
        "year": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "abstract": (
            _clip_abstract(_strip_controls(abstract).strip()) if isinstance(abstract, str) else ""
        ),
        "citationCount": (
            citations if isinstance(citations, int) and not isinstance(citations, bool) else 0
        ),
        "journal": {
            "name": clean_text(journal.get("name")),
            "volume": clean_text(journal.get("volume")),
            "pages": clean_text(journal.get("pages")),
        },
        "publicationDate": clean_text(record.get("publicationDate")),
    }


def record_year(record: Mapping[str, Any]) -> int | None:
    """The year a record was published, from its year or its publication date."""
    if record.get("year"):
        return int(record["year"])
    match = re.match(r"\d{4}", record.get("publicationDate") or "")
    return int(match.group()) if match else None


class ScholarCache:
    """Semantic Scholar matches kept across runs, beside the lab memory.

    Only public bibliographic metadata is stored, and only for papers that
    passed the cutoff of the run that fetched them.  An entry remembers that
    cutoff: a later run with a later cutoff might be owed a newer paper of the
    same name, so it looks the title up again.  Every hit is re-checked
    against the current cutoff by the caller.

    The entries live in the harness's cache, outside the wiki, so no model
    session can read them: a paper may only use records that its own
    literature stage verified against its own cutoff.
    """

    def __init__(self, harness: ContextHarness, cutoff: str | None) -> None:
        self._store = harness.cache
        self._cutoff = cutoff or None

    def _covers(self, cached_cutoff: Any) -> bool:
        """Whether a match chosen under ``cached_cutoff`` is still the best choice now."""
        try:
            cached = parse_cutoff(cached_cutoff if isinstance(cached_cutoff, str) else None)
        except ValueError:
            return False
        if cached is None:
            return True
        wanted = parse_cutoff(self._cutoff)
        return wanted is not None and wanted <= cached

    def get(self, title: str) -> dict[str, Any] | None:
        if self._store is None:
            return None
        entry = self._store.get(CACHE_NAMESPACE, normalize_title(title))
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("paper"), dict)
            and self._covers(entry.get("cutoff"))
        ):
            return entry["paper"]
        return None

    def put(self, title: str, paper: Mapping[str, Any]) -> None:
        if self._store is not None:
            self._store.put(
                CACHE_NAMESPACE,
                normalize_title(title),
                {"cutoff": self._cutoff, "paper": dict(paper)},
            )


# ---------------------------------------------------------------------------
# Checks on the written document
# ---------------------------------------------------------------------------


def _preamble(tex: str) -> str | None:
    """Everything before ``\\begin{document}``, without comments and layout whitespace."""
    stripped = strip_comments(tex)
    if "\\begin{document}" not in stripped:
        return None
    return " ".join(stripped.split("\\begin{document}", 1)[0].split())


def _is_complete_document(tex: str) -> bool:
    stripped = strip_comments(tex)
    return all(
        marker in stripped for marker in ("\\documentclass", "\\begin{document}", "\\end{document}")
    )


def check_document(
    latex: str, *, template: str, citation_keys: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Problems of an updated template: ``(blocking, advisory)``.

    Blocking problems make the document unusable or wrong (truncated, lost
    preamble, citations that are not in the bibliography, a section left
    unwritten).  The advisory one is citing too few of the collected papers.
    """
    allowed = set(citation_keys)
    blocking = static_check(latex, allowed_citations=allowed, check_placeholders=False)

    original, returned = _preamble(template), _preamble(latex)
    if original is not None and returned is not None and original != returned:
        blocking.append(
            "the preamble (everything before \\begin{document}) differs from the original "
            "template.tex; restore it exactly"
        )
    body = strip_comments(latex)
    for name, placeholder in _SECTION_PLACEHOLDERS:
        if placeholder in template and placeholder in body:
            blocking.append(
                f"the {name} section still contains the template placeholder '{placeholder}'"
            )

    advisory: list[str] = []
    required = min_cited(len(citation_keys))
    cited = cite_keys(latex) & allowed
    if len(cited) < required:
        missing = [key for key in citation_keys if key not in cited]
        shown = ", ".join(missing[:40]) + (" ..." if len(missing) > 40 else "")
        advisory.append(
            f"only {len(cited)} of the {len(citation_keys)} collected papers are cited; the "
            f"two sections should cite at least {required}. Not cited yet: {shown}"
        )
    return blocking, advisory


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LiteratureAgent:
    def __init__(self, ctx: RunContext, *, scholar: SemanticScholar | None = None) -> None:
        self.ctx = ctx
        self._scholar = scholar or SemanticScholar()
        self._cutoff = ctx.cfg.research_cutoff.strip() or None
        self._cache = ScholarCache(ctx.harness, self._cutoff)

    def run(self, outline: dict[str, Any]) -> LiteratureResult:
        """Collect verified literature for ``outline`` and write Introduction + Related Work.

        The bibliography starts from the papers ``known_papers.json`` in the materials
        lists, when there is such a file, and holds none of the papers it excludes;
        what was wrong with the file, or kept one of its papers out, is in the result's
        ``warnings`` (and in the log), and never fails the stage.
        """
        cfg, paths, harness = self.ctx.cfg, self.ctx.paths, self.ctx.harness
        parse_cutoff(self._cutoff)  # a malformed cutoff should fail before any model call
        idea = read_text(paths.idea)
        experimental_log = read_text(paths.experimental_log)
        template = read_text(paths.template_tex)
        guidelines = read_text(paths.guidelines)
        outline = copy.deepcopy(outline)

        tasks = collect_search_tasks(outline)
        log.info(
            "Literature search: %d task(s), cutoff %s, %d worker(s)",
            len(tasks),
            self._cutoff or "none",
            cfg.literature_workers,
        )
        known = load_known_papers(paths.materials_dir, cutoff=self._cutoff)
        for problem in known.problems:
            log.warning("%s", problem)
        if known.papers or known.excluded:
            log.info(
                "The bibliography starts from %d known paper(s); %d paper(s) are not to be cited",
                len(known.papers),
                len(known.excluded),
            )
        fingerprint = _search_fingerprint(
            tasks, _core_problem(outline), self._cutoff, cfg.models.fast, known
        )
        found = self._load_checkpoint(fingerprint)
        searched = found is None
        if found is None:
            found = self._collect(tasks, outline, known)
            log.info("Literature search finished: %s", found.stats.describe())
            # Saved before the writing turn: that turn can fail (a refusal, a truncated or
            # incomplete document, an API error), and the search is the expensive part.
            self._save_checkpoint(fingerprint, found)
        stats = found.stats
        papers = list(found.papers.values())
        if stats.failed_tasks or stats.lookup_errors or stats.skipped_lookups:
            harness.note(
                "The literature search was incomplete: "
                f"{stats.failed_tasks} of {stats.tasks} search task(s) failed, "
                f"{stats.lookup_errors} Semantic Scholar lookup(s) failed and "
                f"{stats.skipped_lookups} were skipped. The bibliography may miss papers "
                "that the outline asked for.",
                stage=STAGE,
                agent=AGENT,
                title="Literature search incomplete",
            )
        if not papers:
            log.warning(
                "No verified papers were collected (%s); continuing with an empty bibliography",
                stats.describe(),
            )
            if searched:  # otherwise the attempt that searched has left this note already
                harness.note(
                    "No paper could be verified in Semantic Scholar, so references.bib is "
                    "empty and the Introduction and Related Work were written without "
                    f"citations. Search summary: {stats.describe()}.",
                    stage=STAGE,
                    agent=AGENT,
                    title="Empty bibliography",
                )
        if known.problems and searched:  # as above: said once, by the attempt that searched
            harness.note(
                "Not everything the materials say about the papers to start from and the "
                f"papers not to cite could be used. {_in_short(known.problems)}",
                stage=STAGE,
                agent=AGENT,
                title="Known papers not used in full",
            )

        bibtex = generate_bibtex(papers)
        citation_map = generate_citation_map(papers)
        outline = inject_citations(outline, papers)
        latex = self._write_sections(
            template=template,
            outline=outline,
            idea=idea,
            experimental_log=experimental_log,
            guidelines=guidelines,
            papers=papers,
        )

        out = paths.literature_dir
        write_json(out / "outline_v1.json", outline)
        write_text(out / "updated_template.tex", latex)
        write_text(out / "references.bib", bibtex)
        write_json(paths.citation_map, citation_map)
        write_json(paths.outline, outline)
        write_text(paths.template_tex, latex)
        write_text(paths.references_bib, bibtex)

        self._write_wiki(outline, papers)
        cited = len(cite_keys(latex) & set(citation_map))
        harness.record(
            "literature", stage=STAGE, agent=AGENT, papers=len(papers), cited=cited, **asdict(stats)
        )
        log.info(
            "Literature outputs written to %s (%d papers, %d cited in Introduction and Related Work)",
            out,
            len(papers),
            cited,
        )
        return LiteratureResult(
            outline=outline,
            citation_map=citation_map,
            bibtex=bibtex,
            latex=latex,
            papers=len(papers),
            known=stats.known,
            excluded=stats.excluded,
            warnings=list(known.problems),
        )

    # -- checkpoint ------------------------------------------------------------

    def _save_checkpoint(self, fingerprint: str, found: _Collection) -> None:
        """Keep the verified papers for a later attempt of this stage.

        Only a complete search is kept.  One that lost search tasks or lookups (usually to
        the kind of outage that also fails the writing turn) is worth repeating, because
        the next attempt may find what this one missed.
        """
        stats = found.stats
        if stats.failed_tasks or stats.lookup_errors or stats.skipped_lookups:
            return
        path = self.ctx.paths.literature_dir / CHECKPOINT_NAME
        try:
            write_json(
                path,
                {
                    "version": _CHECKPOINT_VERSION,
                    "fingerprint": fingerprint,
                    "stats": asdict(stats),
                    "papers": [paper.model_dump() for paper in found.papers.values()],
                },
            )
        except OSError as exc:
            # The checkpoint saves a repeated search; the stage does not depend on it.
            log.warning("Could not save the verified papers to %s: %s", path, exc)

    def _load_checkpoint(self, fingerprint: str) -> _Collection | None:
        """The collection an earlier attempt saved for this very search, if there is one.

        Anything else (no file, another search, a file that does not hold what this
        module wrote) means the search runs: the papers in the file skip verification,
        so it is trusted only as a whole.
        """
        path = self.ctx.paths.literature_dir / CHECKPOINT_NAME
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("Ignoring the unreadable literature checkpoint %s: %s", path, exc)
            return None
        if not isinstance(saved, dict) or saved.get("version") != _CHECKPOINT_VERSION:
            log.warning("Ignoring the literature checkpoint %s: unknown format", path)
            return None
        if saved.get("fingerprint") != fingerprint:
            log.info(
                "The verified papers in %s were collected for other search tasks, another "
                "cutoff, other known papers or by another model; searching again",
                path,
            )
            return None
        try:
            papers = [PaperData.model_validate(entry) for entry in saved["papers"]]
            stats = LiteratureStats(**saved["stats"])
            if not all(type(count) is int for count in asdict(stats).values()):
                raise TypeError("a search counter is not a number")
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Ignoring the damaged literature checkpoint %s: %s", path, exc)
            return None
        found = _Collection(stats=stats)
        for paper in papers:
            found.papers[normalize_title(paper.title)] = paper
            found.keys.add(paper.citation_key.casefold())
        if len(found.papers) != len(papers) or len(found.keys) != len(papers):
            log.warning("Ignoring the literature checkpoint %s: it repeats a paper", path)
            return None
        log.info(
            "Reusing the %d verified paper(s) that an earlier attempt of this stage "
            "collected (%s); no search is run",
            len(papers),
            stats.describe(),
        )
        return found

    # -- discovery and verification --------------------------------------------

    @staticmethod
    def _start_from(known: KnownPapers) -> _Collection:
        """A collection that holds the known papers, before any search has added to it.

        They keep their citation keys (a paper found later that would get the same key
        gets a suffix), their titles count as resolved, so no search looks them up
        again, and the keys of the excluded papers are taken, so nothing else gets one.
        """
        found = _Collection(excluded=set(known.excluded), keys=set(known.reserved_keys))
        for paper in known.papers:
            canonical = normalize_title(paper.title)
            # A copy: the searches add the sections that return the paper.
            found.papers[canonical] = paper.model_copy(deep=True)
            found.resolved[canonical] = canonical
            found.keys.add(paper.citation_key.casefold())
        found.stats.known = len(found.papers)
        return found

    def _collect(
        self, tasks: Sequence[SearchTask], outline: Mapping[str, Any], known: KnownPapers
    ) -> _Collection:
        """Start from the ``known`` papers, run every search task and verify what the
        tasks propose.

        Discovery runs in a thread pool.  Results are consumed in outline
        order by this thread, which alone talks to Semantic Scholar and owns
        the registry, so attribution and citation keys do not depend on which
        call happens to finish first.
        """
        found = self._start_from(known)
        found.stats.tasks = len(tasks)
        if not tasks:
            return found
        cfg = self.ctx.cfg
        web_search = self.ctx.client(cfg.models.fast).capabilities.web_search
        if not web_search:
            log.warning(
                "Model '%s' has no web search; candidate papers come from its memory and "
                "depend entirely on the Semantic Scholar check",
                cfg.models.fast,
            )
        core_problem = _core_problem(outline)

        last_error: Exception | None = None
        pool = ThreadPoolExecutor(
            max_workers=max(1, cfg.literature_workers), thread_name_prefix="discovery"
        )
        interrupted = False
        try:
            futures = [
                pool.submit(self._discover, task, core_problem, web_search) for task in tasks
            ]
            for number, (task, future) in enumerate(zip(tasks, futures, strict=True), start=1):
                try:
                    candidates = future.result()
                except Exception as exc:
                    if stops_the_run(exc):
                        raise
                    # One search task failing costs some references, not the stage.
                    found.stats.failed_tasks += 1
                    last_error = exc
                    log.warning(
                        "Search task %d/%d failed [%s] '%s': %s",
                        number,
                        len(tasks),
                        task.section,
                        task.focus,
                        exc,
                    )
                    continue
                log.info(
                    "Search task %d/%d [%s] '%s': %d candidate(s)",
                    number,
                    len(tasks),
                    task.section,
                    task.focus,
                    len(candidates),
                )
                if not candidates:
                    found.stats.empty_tasks += 1
                for candidate in candidates:
                    self._verify(candidate, task, found)
        except KeyboardInterrupt:
            # Searches in flight are not followed by further calls.
            interrupted = True
            self.ctx.harness.cancel()
            raise
        finally:
            # On the normal path everything has finished; after an error this drops the calls
            # that have not started.  An interrupt is not held up by the calls in flight:
            # they end on their own and, the run being cancelled, start nothing new.
            pool.shutdown(wait=not interrupted, cancel_futures=True)
        if last_error is not None and found.stats.failed_tasks == len(tasks):
            raise RuntimeError(
                f"all {len(tasks)} literature search tasks failed; the last error was: {last_error}"
            ) from last_error
        return found

    def _discover(
        self, task: SearchTask, core_problem: str, web_search: bool
    ) -> list[CandidatePaper]:
        """Ask the fast model for candidate papers for one task (runs in a worker thread)."""
        cfg = self.ctx.cfg
        targeted = task.search_type == "targeted"
        if targeted:
            prompt = targeted_discovery_prompt(
                hint=task.focus,
                context=task.context,
                cutoff=self._cutoff or "",
                web_search=web_search,
            )
        else:
            prompt = exploration_discovery_prompt(
                focus=task.focus,
                context=task.context,
                core_problem=core_problem,
                cutoff=self._cutoff or "",
                web_search=web_search,
            )
        response = self.ctx.harness.generate(
            prompt,
            agent=AGENT,
            stage=STAGE,
            model=cfg.models.fast,
            system=DISCOVERY_SYSTEM_PROMPT,
            schema=DiscoveryResult,
            memory=MemoryPolicy.none(),
            web_search=WebSearch(
                max_uses=_TARGETED_SEARCHES if targeted else _EXPLORATION_SEARCHES
            ),
            effort=cfg.models.fast_effort,
            temperature=0.1 if targeted else 0.4,
            label=DISCOVERY_LABEL,
        )
        result = response.parsed
        if not isinstance(result, DiscoveryResult):
            result = DiscoveryResult.model_validate(result)
        candidates = result.candidates
        if len(candidates) > _MAX_CANDIDATES_PER_TASK:
            log.info(
                "Search task '%s' returned %d candidates; keeping the first %d",
                task.focus,
                len(candidates),
                _MAX_CANDIDATES_PER_TASK,
            )
            candidates = candidates[:_MAX_CANDIDATES_PER_TASK]
        return candidates

    def _from_cache(self, title: str, year_hint: int | None) -> dict[str, Any] | None:
        """A record of the lab cache that is the paper called ``title`` under today's cutoff."""
        cached = self._cache.get(title)
        if cached is None:
            return None
        # The same rule as for live results, so the cutoff is applied again on every hit.
        return select_match(
            [public_record(cached)], title, year_hint=year_hint, cutoff=self._cutoff
        )

    def _from_scholar(self, title: str, year_hint: int | None) -> dict[str, Any] | None:
        """Ask Semantic Scholar for ``title`` and remember an answer the cutoff allows."""
        match = self._scholar.find_paper(title, year_hint=year_hint, cutoff=self._cutoff)
        if match is None:
            return None
        record = public_record(match)
        if is_before_cutoff(record["publicationDate"], record["year"], self._cutoff):
            self._cache.put(title, record)
        return record

    def _verify(self, candidate: CandidatePaper, task: SearchTask, found: _Collection) -> None:
        """Register ``candidate`` if Semantic Scholar confirms it, the cutoff allows it and
        it is not one of the papers that must not be cited."""
        stats = found.stats
        stats.candidates += 1
        title = " ".join(candidate.title.split())
        wanted = normalize_title(title)
        if not wanted:
            stats.unmatched += 1
            return
        if wanted in found.excluded:
            self._leave_out(title, found)
            return
        if wanted in found.resolved:
            stats.duplicates += 1
            known = found.resolved[wanted]
            if known is not None:
                self._add_section(found.papers[known], task.section)
            return

        record = self._from_cache(title, candidate.year)
        if record is not None:
            stats.cache_hits += 1
        elif found.scholar_down:
            stats.skipped_lookups += 1
            return
        else:
            stats.lookups += 1
            try:
                record = self._from_scholar(title, candidate.year)
            except SemanticScholarError as exc:
                stats.lookup_errors += 1
                found.consecutive_lookup_failures += 1
                log.warning("Semantic Scholar lookup failed for '%s': %s", title, exc)
                if found.consecutive_lookup_failures >= _MAX_CONSECUTIVE_LOOKUP_FAILURES:
                    found.scholar_down = True
                    log.error(
                        "Semantic Scholar failed %d times in a row; the remaining candidates "
                        "of this run are only looked up in the cache, and those not found "
                        "there cannot be cited",
                        found.consecutive_lookup_failures,
                    )
                return
            found.consecutive_lookup_failures = 0
        found.resolved[wanted] = None

        if record is None:
            stats.unmatched += 1
            log.debug("No Semantic Scholar match before the cutoff for '%s'", title)
            return
        canonical = normalize_title(record["title"])
        if canonical in found.excluded:
            # Proposed under another wording of its title; the lookup tells which paper it
            # is, and the wording is remembered so that it is not looked up a second time.
            found.excluded.add(wanted)
            self._leave_out(record["title"], found)
            return
        if not is_before_cutoff(record.get("publicationDate"), record.get("year"), self._cutoff):
            stats.after_cutoff += 1
            log.info("Dropped '%s': published at or after the cutoff", record.get("title"))
            return
        if not record.get("abstract"):
            stats.no_abstract += 1
            level = logging.WARNING if task.search_type == "targeted" else logging.INFO
            log.log(
                level,
                "Dropped '%s' (%s): Semantic Scholar has no abstract for it",
                record.get("title"),
                "a must-have citation" if task.search_type == "targeted" else task.section,
            )
            return

        if canonical in found.papers:
            stats.duplicates += 1
            self._add_section(found.papers[canonical], task.section)
        else:
            authors = [author["name"] for author in record["authors"]]
            year = record_year(record)
            journal = record["journal"]
            found.papers[canonical] = PaperData(
                citation_key=unique_citation_key(
                    make_citation_key(authors, year, record["title"]), found.keys
                ),
                title=record["title"],
                authors=authors,
                venue=record.get("venue") or DEFAULT_VENUE,
                year=year,
                abstract=_clip_abstract(record["abstract"]),
                citation_count=int(record.get("citationCount") or 0),
                found_in_section=task.section,
                sections=[task.section],
                reason=" ".join(candidate.reason.split()),
                journal=journal.get("name"),
                volume=journal.get("volume"),
                pages=journal.get("pages"),
                publication_date=record.get("publicationDate"),
            )
            stats.verified += 1
            log.debug("Verified '%s' for %s", record["title"], task.section)
        found.resolved[wanted] = canonical
        found.resolved[canonical] = canonical

    @staticmethod
    def _leave_out(title: str, found: _Collection) -> None:
        """Count a candidate that is one of the papers not to cite."""
        found.stats.excluded += 1
        log.info("Left out '%s': the researcher rejected it as a source", title)

    @staticmethod
    def _add_section(paper: PaperData, section: str) -> None:
        if section not in paper.sections:
            paper.sections.append(section)
        if not paper.found_in_section:
            # A known paper: this is the first search that returned it.
            paper.found_in_section = section

    # -- writing -----------------------------------------------------------------

    def _write_sections(
        self,
        *,
        template: str,
        outline: Mapping[str, Any],
        idea: str,
        experimental_log: str,
        guidelines: str,
        papers: Sequence[PaperData],
    ) -> str:
        """Have the writer model fill in Introduction and Related Work; repair until it checks."""
        cfg, harness = self.ctx.cfg, self.ctx.harness
        keys = [paper.citation_key for paper in papers]
        conversation = harness.open(
            agent=AGENT, stage=STAGE, model=cfg.models.writer, system=WRITER_SYSTEM_PROMPT
        )
        message = writer_user_prompt(
            template=template,
            plan=outline.get("intro_related_work_plan") or {},
            idea=idea,
            experimental_log=experimental_log,
            guidelines=guidelines,
            papers=[paper.model_dump() for paper in papers],
            citation_keys=keys,
            cutoff=self._cutoff or "",
        )
        rounds = max(0, cfg.max_repair_rounds)
        best: _Draft | None = None
        first_problems: list[str] = []
        repairs = 0
        asked_for_more_citations = False
        for attempt in range(rounds + 1):
            latex = self._ask(conversation, message)
            if latex is None:
                blocking = [
                    "the reply did not contain the LaTeX document in a ```latex fenced block"
                ]
                advisory: list[str] = []
            else:
                blocking, advisory = check_document(latex, template=template, citation_keys=keys)
                draft = _Draft(latex, blocking, advisory)
                if best is None or draft.rank() <= best.rank():
                    best = draft
            if attempt == 0:
                first_problems = blocking + advisory
            # Citing too few papers earns one request for more, not the whole repair budget.
            if not blocking and (not advisory or asked_for_more_citations):
                break
            if attempt == rounds:
                break
            log.warning(
                "Introduction/Related Work draft %d needs a repair turn: %s",
                attempt + 1,
                "; ".join(blocking + advisory),
            )
            asked_for_more_citations = asked_for_more_citations or bool(advisory)
            message = writer_repair_prompt(blocking + advisory)
            repairs += 1

        if best is None:
            raise RuntimeError(
                f"the writer model returned no LaTeX document for the Introduction and Related "
                f"Work in {repairs + 1} attempt(s)"
            )
        if not _is_complete_document(best.latex):
            raise RuntimeError(
                "the Introduction and Related Work document is still incomplete after "
                f"{repairs} repair turn(s): {'; '.join(best.blocking)}"
            )
        remaining = best.blocking + best.advisory
        if remaining:
            log.warning(
                "Keeping the best Introduction/Related Work draft with unresolved problems: %s",
                "; ".join(remaining),
            )
        if first_problems:
            outcome = (
                f"Still unresolved after {repairs} repair turn(s): {'; '.join(remaining)}."
                if remaining
                else f"Resolved after {repairs} repair turn(s) in the same conversation."
            )
            harness.note(
                "The first Introduction/Related Work draft failed the static check: "
                f"{'; '.join(first_problems)}. {outcome}",
                stage=STAGE,
                agent=AGENT,
                kind="lesson",
                title="Introduction and Related Work draft needed repair",
            )
        return best.latex

    @staticmethod
    def _ask(conversation: HarnessConversation, message: str) -> str | None:
        """Send one turn; ``None`` when the reply holds no LaTeX document."""
        try:
            return conversation.send(message, parser=parse_latex).parsed
        except LLMParseError as exc:
            log.warning("The writer's reply contained no LaTeX document: %s", exc)
            return None

    # -- memory ------------------------------------------------------------------

    def _write_wiki(self, outline: Mapping[str, Any], papers: Sequence[PaperData]) -> None:
        """One page per verified paper, and the outline page with its citation candidates."""
        harness = self.ctx.harness
        keys = {paper.citation_key for paper in papers}
        try:
            # Pages of papers that an earlier run of this stage verified and this one did not
            # would advertise citation keys that references.bib no longer has.
            if harness.project is not None:
                for info in harness.project.list(LITERATURE_DIR):
                    if info.kind == "literature" and PurePosixPath(info.path).stem not in keys:
                        harness.project.delete(info.path, force=True)
            for paper in papers:
                harness.write_page(
                    f"{LITERATURE_DIR}/{paper.citation_key}.md",
                    _paper_page(paper),
                    _paper_meta(paper),
                )
        except WikiError as exc:
            # Memory is an aid; the files written above are what the pipeline depends on.
            log.warning("Could not update the literature pages of the wiki: %s", exc)
        library = "\n".join(
            f"- [[literature/{paper.citation_key}]]: {paper.title} ({paper.year or 'n.d.'})"
            for paper in papers
        )
        write_outline_page(
            harness,
            outline,
            summary=(
                f"Outline of the paper with citation candidates: {len(papers)} verified "
                "paper(s) from the literature stage."
            ),
            notes=f"## Citation library\n\n{library}" if papers else "",
        )


def _authors_line(authors: Sequence[str]) -> str:
    if not authors:
        return "Unknown authors"
    return ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")


def _paper_meta(paper: PaperData) -> dict[str, Any]:
    summary = f"{_authors_line(paper.authors)} ({paper.year or 'n.d.'}), {paper.venue}: "
    return {
        "title": paper.title,
        "citation_key": paper.citation_key,
        "authors": paper.authors,
        "year": paper.year,
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "kind": "literature",
        "summary": (summary + paper.title)[:200],
    }


def _paper_page(paper: PaperData) -> str:
    quoted = "\n".join(f"> {line}" for line in paper.abstract.splitlines())
    sections = "\n".join(f"- {section}" for section in paper.sections) or (
        "No search of the outline returned it: it was known before the search and is in "
        "the bibliography for that reason."
    )
    return (
        f"# {paper.title}\n\n"
        f"Cite as `\\cite{{{paper.citation_key}}}`. {_authors_line(paper.authors)}. "
        f"{paper.venue}, {paper.year or 'n.d.'}. {paper.citation_count} citation(s) "
        "according to Semantic Scholar.\n\n"
        "## Abstract\n\n"
        f"{quoted}\n\n"
        "## Why it was collected\n\n"
        f"{paper.reason or 'The search that proposed it gave no reason.'}\n\n"
        "## Outline sections that list it\n\n"
        f"{sections}\n\n"
        "The plan these sections belong to is on [[paper/outline]]."
    )
