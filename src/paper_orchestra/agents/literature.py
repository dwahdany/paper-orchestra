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

1. *Discovery*: one small web-search call per search task of the outline
   proposes candidate papers (title, year, reason).
2. *Verification*: every candidate has to be found in Semantic Scholar under
   (nearly) its title and be published before the research cutoff.  This is
   the gate that keeps hallucinated references out of the paper.
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
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_orchestra.agents.outline import write_outline_page
from paper_orchestra.llm.base import LLMParseError, WebSearch
from paper_orchestra.llm.parsing import parse_json_object, parse_latex
from paper_orchestra.memory.harness import MemoryPolicy
from paper_orchestra.memory.wiki import WikiError
from paper_orchestra.prompts.literature import (
    DISCOVERY_SYSTEM_PROMPT,
    MIN_CITED_FRACTION,
    WRITER_SYSTEM_PROMPT,
    exploration_discovery_prompt,
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
CACHE_DIR = "wiki/literature-cache"

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
_CACHE_SHARDS = 64
_CACHE_SHARD_CHARS = 30_000

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
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"\d{4}", value)
            return int(match.group()) if match else None
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
        if not isinstance(value, list):
            return value
        usable: list[Any] = []
        for item in value:
            if isinstance(item, str):
                item = {"title": item}
            if isinstance(item, CandidatePaper):
                usable.append(item)
            elif isinstance(item, Mapping) and str(item.get("title") or "").strip():
                usable.append({"year": None, "reason": "", **item})
        if len(usable) < len(value):
            log.warning(
                "Ignored %d candidate(s) without a title in a discovery answer",
                len(value) - len(usable),
            )
        return usable


class PaperData(BaseModel):
    """A verified paper.  Bibliographic fields come from Semantic Scholar only."""

    citation_key: str
    title: str
    authors: list[str]
    venue: str
    year: int | None
    abstract: str
    citation_count: int
    #: The outline section whose search found the paper first.
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

    def describe(self) -> str:
        return (
            f"{self.tasks} search task(s) ({self.failed_tasks} failed, {self.empty_tasks} "
            f"without candidates), {self.candidates} candidate(s), {self.verified} verified "
            f"paper(s); {self.duplicates} duplicate(s), {self.unmatched} not matched in "
            f"Semantic Scholar before the cutoff, {self.after_cutoff} past the cutoff, "
            f"{self.no_abstract} without an abstract, {self.lookup_errors} failed lookup(s), "
            f"{self.skipped_lookups} skipped lookup(s), {self.lookups} live lookup(s), "
            f"{self.cache_hits} cache hit(s)"
        )


@dataclass
class LiteratureResult:
    outline: dict[str, Any]
    citation_map: dict[str, dict[str, Any]]
    bibtex: str
    latex: str
    papers: int


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
    stats: LiteratureStats = field(default_factory=LiteratureStats)
    consecutive_lookup_failures: int = 0
    scholar_down: bool = False


@dataclass
class _Draft:
    latex: str
    blocking: list[str]
    advisory: list[str]

    def rank(self) -> tuple[int, int]:
        return len(self.blocking), len(self.advisory)


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


def _ascii(text: str) -> str:
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
}
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
    "\u0328": "k",  # ogonek
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
    "–": "--",
    "—": "---",
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "…": r"\ldots{}",
    "\u00a0": "~",  # no-break space
}
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
        if base in "ij" and accent not in "ck":
            base = f"\\{base}"  # an accent above replaces the dot
        return f"{{\\{accent}{{{base}}}}}"
    name = unicodedata.name(char, "")
    if name.startswith("GREEK SMALL LETTER ") and not marks:
        letter = name.rsplit(" ", 1)[-1].lower()
        return "o" if letter == "omicron" else f"{{$\\{letter}$}}"
    if name.startswith("GREEK CAPITAL LETTER ") and not marks:
        letter = name.rsplit(" ", 1)[-1].lower()
        if letter in _GREEK_CAPITALS_WITH_MACRO:
            return f"{{$\\{letter.capitalize()}$}}"
        return letter[0].upper()
    return _ascii(char)


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
            out.append(char)
        else:
            out.append(_latex_for(char))
    return " ".join("".join(out).split())


def bibtex_entry(paper: PaperData) -> str:
    fields: list[tuple[str, str]] = [
        # The inner braces keep the capitalisation of the title under every bibliography style.
        ("title", "{" + escape_bibtex(paper.title) + "}"),
        ("author", " and ".join(filter(None, map(escape_bibtex, paper.authors))) or "Unknown"),
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


def _clean(value: Any) -> str | None:
    text = " ".join(str(value).split()) if value is not None else ""
    return text or None


def _clip_abstract(abstract: str) -> str:
    """Bound the abstract (it is repeated in several prompts).  Applying it twice is a no-op."""
    if len(abstract) <= ABSTRACT_LIMIT:
        return abstract
    return abstract[:ABSTRACT_LIMIT] + _ABSTRACT_MARK


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The bibliographic fields of a Semantic Scholar result that the pipeline uses."""
    journal = record.get("journal") if isinstance(record.get("journal"), Mapping) else {}
    year = record.get("year")
    return {
        "title": _clean(record.get("title")) or "",
        "authors": [
            {"name": name}
            for author in record.get("authors") or []
            if isinstance(author, Mapping) and (name := _clean(author.get("name")))
        ],
        "venue": _clean(record.get("venue")),
        "year": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "abstract": _clip_abstract((record.get("abstract") or "").strip()),
        "citationCount": record.get("citationCount") or 0,
        "journal": {
            "name": _clean(journal.get("name")),
            "volume": _clean(journal.get("volume")),
            "pages": _clean(journal.get("pages")),
        },
        "publicationDate": _clean(record.get("publicationDate")),
    }


def _record_year(record: Mapping[str, Any]) -> int | None:
    if record.get("year"):
        return int(record["year"])
    match = re.match(r"\d{4}", record.get("publicationDate") or "")
    return int(match.group()) if match else None


class _ScholarCache:
    """Semantic Scholar matches kept in the lab wiki across runs.

    Only public bibliographic metadata is stored, and only for papers that
    passed the cutoff of the run that fetched them.  An entry remembers that
    cutoff: a later run with a later cutoff might be owed a newer paper of the
    same name, so it looks the title up again.  Every hit is re-checked
    against the current cutoff by the caller.
    """

    def __init__(self, harness: ContextHarness, cutoff: str | None) -> None:
        self._harness = harness
        self._cutoff = cutoff or None

    @staticmethod
    def _page(key: str) -> str:
        shard = hashlib.sha256(key.encode("utf-8")).digest()[0] % _CACHE_SHARDS
        return f"{CACHE_DIR}/shard-{shard:02d}.md"

    def _entries(self, rel: str) -> list[dict[str, Any]]:
        body = self._harness.read_page(rel, scope="lab")
        if body is None:
            return []
        entries = (parse_json_object(body) or {}).get("entries")
        if not isinstance(entries, list):
            log.warning("Ignoring the unreadable literature cache page %s", rel)
            return []
        return [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("query"), str)
            and isinstance(entry.get("paper"), dict)
        ]

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
        key = normalize_title(title)
        for entry in self._entries(self._page(key)):
            if entry["query"] == key and self._covers(entry.get("cutoff")):
                return entry["paper"]
        return None

    def put(self, title: str, paper: Mapping[str, Any]) -> None:
        key = normalize_title(title)
        rel = self._page(key)
        entries = [entry for entry in self._entries(rel) if entry["query"] != key]
        entries.append({"query": key, "cutoff": self._cutoff, "paper": dict(paper)})

        def render() -> str:
            text = json.dumps({"entries": entries}, indent=1, ensure_ascii=False)
            # A backtick run inside an abstract must not end the Markdown fence early.
            return text.replace("`", "\\u0060")

        while len(entries) > 1 and len(render()) > _CACHE_SHARD_CHARS:
            entries.pop(0)  # oldest first
        body = (
            f"# Semantic Scholar cache, {PurePosixPath(rel).stem}\n\n"
            "Public bibliographic metadata returned by Semantic Scholar title lookups. "
            "Pipeline code reads and rewrites the JSON below.\n\n"
            f"```json\n{render()}\n```"
        )
        meta = {
            "title": f"Semantic Scholar cache {PurePosixPath(rel).stem}",
            "summary": "Machine-readable cache of Semantic Scholar title lookups (public metadata).",
            "kind": "cache",
        }
        try:
            self._harness.write_page(rel, body, meta, scope="lab")
        except WikiError as exc:
            log.warning("Could not update the literature cache page %s: %s", rel, exc)


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
    required = int(len(citation_keys) * MIN_CITED_FRACTION)
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
        self._cache = _ScholarCache(ctx.harness, self._cutoff)

    def run(self, outline: dict[str, Any]) -> LiteratureResult:
        """Collect verified literature for ``outline`` and write Introduction + Related Work."""
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
        found = self._collect(tasks, outline)
        stats = found.stats
        papers = list(found.papers.values())
        log.info("Literature search finished: %s", stats.describe())
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
            harness.note(
                "No paper could be verified in Semantic Scholar, so references.bib is empty "
                "and the Introduction and Related Work were written without citations. "
                f"Search summary: {stats.describe()}.",
                stage=STAGE,
                agent=AGENT,
                title="Empty bibliography",
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
        )

    # -- discovery and verification --------------------------------------------

    def _collect(self, tasks: Sequence[SearchTask], outline: Mapping[str, Any]) -> _Collection:
        """Run every search task and verify what the tasks propose.

        Discovery runs in a thread pool.  Results are consumed in outline
        order by this thread, which alone talks to Semantic Scholar and owns
        the registry, so attribution and citation keys do not depend on which
        call happens to finish first.
        """
        found = _Collection()
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
        intro = (outline.get("intro_related_work_plan") or {}).get("introduction_strategy") or {}
        core_problem = str(intro.get("problem_gap_hypothesis") or "N/A")

        last_error: Exception | None = None
        pool = ThreadPoolExecutor(
            max_workers=max(1, cfg.literature_workers), thread_name_prefix="discovery"
        )
        try:
            futures = [
                pool.submit(self._discover, task, core_problem, web_search) for task in tasks
            ]
            for number, (task, future) in enumerate(zip(tasks, futures, strict=True), start=1):
                try:
                    candidates = future.result()
                except Exception as exc:
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
        finally:
            # On the normal path everything has finished; after an error or an interrupt this
            # drops the calls that have not started instead of waiting for them.
            pool.shutdown(wait=True, cancel_futures=True)
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

    def _lookup(
        self, title: str, year_hint: int | None, found: _Collection
    ) -> dict[str, Any] | None:
        """The Semantic Scholar record for ``title``: from the lab cache, else from the API."""
        cached = self._cache.get(title)
        if cached is not None:
            # The same rule as for live results, so the cutoff is applied again on every hit.
            match = select_match(
                [_public_record(cached)], title, year_hint=year_hint, cutoff=self._cutoff
            )
            if match is not None:
                found.stats.cache_hits += 1
                return match
        found.stats.lookups += 1
        match = self._scholar.find_paper(title, year_hint=year_hint, cutoff=self._cutoff)
        if match is None:
            return None
        record = _public_record(match)
        self._cache.put(title, record)
        return record

    def _verify(self, candidate: CandidatePaper, task: SearchTask, found: _Collection) -> None:
        """Register ``candidate`` if Semantic Scholar confirms it and the cutoff allows it."""
        stats = found.stats
        stats.candidates += 1
        title = " ".join(candidate.title.split())
        wanted = normalize_title(title)
        if not wanted:
            stats.unmatched += 1
            return
        if wanted in found.resolved:
            stats.duplicates += 1
            known = found.resolved[wanted]
            if known is not None:
                self._add_section(found.papers[known], task.section)
            return
        if found.scholar_down:
            stats.skipped_lookups += 1
            return

        try:
            record = self._lookup(title, candidate.year, found)
        except SemanticScholarError as exc:
            stats.lookup_errors += 1
            found.consecutive_lookup_failures += 1
            log.warning("Semantic Scholar lookup failed for '%s': %s", title, exc)
            if found.consecutive_lookup_failures >= _MAX_CONSECUTIVE_LOOKUP_FAILURES:
                found.scholar_down = True
                log.error(
                    "Semantic Scholar failed %d times in a row; the remaining candidates of "
                    "this run will not be verified and cannot be cited",
                    found.consecutive_lookup_failures,
                )
            return
        found.consecutive_lookup_failures = 0
        found.resolved[wanted] = None

        if record is None:
            stats.unmatched += 1
            log.debug("No Semantic Scholar match before the cutoff for '%s'", title)
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

        canonical = normalize_title(record["title"])
        if canonical in found.papers:
            stats.duplicates += 1
            self._add_section(found.papers[canonical], task.section)
        else:
            authors = [author["name"] for author in record["authors"]]
            year = _record_year(record)
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
    def _add_section(paper: PaperData, section: str) -> None:
        if section not in paper.sections:
            paper.sections.append(section)

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
    sections = "\n".join(f"- {section}" for section in paper.sections)
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
