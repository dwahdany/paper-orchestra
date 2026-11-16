"""Verified sources: the only way a paper enters the research wiki.

An agent names a paper by its title.  Code looks that title up in Semantic
Scholar, applies the project's research cutoff and writes the source record
from what the service returned.  Nothing in a record is written by a model, so
a record is evidence that the paper exists and of what its abstract says, and
a paper the service does not know cannot be cited.

The lookup, the title match, the cutoff rule, the cleaning of a record and the
citation keys are those of the paper pipeline's literature stage; this module
only decides what a research project keeps of a verified paper.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

# Shared on purpose: a record must be cleaned, and the cross-run cache read and written,
# exactly as the paper pipeline's literature stage does it.
from paper_orchestra.agents.literature import (
    ABSTRACT_LIMIT,
    ScholarCache,
    clean_text,
    make_citation_key,
    public_record,
    record_year,
    unique_citation_key,
)
from paper_orchestra.llm.base import ToolError, ToolSpec
from paper_orchestra.memory.tools import virtual_path
from paper_orchestra.memory.wiki import split_front_matter
from paper_orchestra.research.model import HUMAN, ProjectError, ResearchPage, format_stamp
from paper_orchestra.research.project import Project
from paper_orchestra.utils.scholar import (
    PAPER_FIELDS,
    SemanticScholar,
    SemanticScholarError,
    normalize_title,
    select_match,
)

if TYPE_CHECKING:
    from paper_orchestra.memory.harness import ContextHarness

__all__ = [
    "NotVerified",
    "SourceVerifier",
    "VerifiedSource",
    "normalize_cutoff",
    "verify_paper_tool",
]

#: What is asked of Semantic Scholar: the bibliographic fields, and where the paper is found.
SOURCE_FIELDS = f"{PAPER_FIELDS},url,externalIds"
#: After this many lookups in a row have failed, the service is treated as down for the rest
#: of the session instead of waiting out the retries of every further candidate.
MAX_CONSECUTIVE_FAILURES = 3
_SUMMARY_CHARS = 160


class NotVerified(ToolError):
    """The paper cannot be recorded; the message tells the agent why and what to do."""


def normalize_cutoff(value: Any) -> str | None:
    """The project's ``research_cutoff`` in the form the cutoff rule reads.

    The rule allows a paper published before the first day of the cutoff month.
    A cutoff given as a year alone therefore means "before that year".
    """
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    return f"{text}-01" if len(text) == 4 and text.isdigit() else text


def _inert(text: str) -> str:
    """``text`` without wiki-link brackets: a record quotes its source and links nothing."""
    return text.replace("[[", "[ [").replace("]]", "] ]")


def _link(match: Mapping[str, Any]) -> str:
    """Where the paper can be read, from a search result: its DOI, arXiv id or record page."""
    ids = match.get("externalIds")
    ids = ids if isinstance(ids, Mapping) else {}
    doi, arxiv = clean_text(ids.get("DOI")), clean_text(ids.get("ArXiv"))
    if doi:
        return "https://doi.org/" + quote(doi, safe="/:;().-_")
    if arxiv:
        return "https://arxiv.org/abs/" + quote(arxiv, safe="/.-_")
    paper_id = clean_text(match.get("paperId"))
    if paper_id:
        return "https://www.semanticscholar.org/paper/" + quote(paper_id, safe="")
    url = clean_text(match.get("url")) or ""
    return url if url.startswith("https://") and " " not in url else ""


def _authors(record: Mapping[str, Any]) -> list[str]:
    return [author["name"] for author in record["authors"]]


def _credit(authors: list[str], year: int | None) -> str:
    """``Vaswani et al. (2017)``: how a source is named in one line."""
    if not authors:
        who = "Unknown authors"
    else:
        surname = authors[0].split()[-1]
        who = surname if len(authors) == 1 else f"{surname} et al."
    return f"{who} ({year or 'n.d.'})"


def _summary(record: Mapping[str, Any]) -> str:
    """One line for indexes: who and when, then the first sentence of the abstract."""
    credit = _credit(_authors(record), record_year(record))
    abstract = " ".join(str(record.get("abstract") or "").split())
    if not abstract:
        return f"{credit}: no abstract available"
    sentence = abstract.split(". ", 1)[0].rstrip(".")
    if len(sentence) > _SUMMARY_CHARS:
        sentence = sentence[: _SUMMARY_CHARS - 1].rstrip() + "…"
    return f"{credit}: {sentence}"


def _quoted(text: str) -> str:
    return "\n".join(f"> {line}".rstrip() for line in _inert(text).splitlines())


def source_body(record: Mapping[str, Any], link: str, *, verified_on: str) -> str:
    """The text of a source record, from a cleaned Semantic Scholar record."""
    authors = _authors(record)
    lines = [
        f"- **Authors:** {_inert(', '.join(authors)) or 'not recorded'}",
        f"- **Year:** {record_year(record) or 'not recorded'}",
        f"- **Venue:** {_inert(record.get('venue') or '') or 'not recorded'}",
    ]
    if record.get("publicationDate"):
        lines.append(f"- **Published:** {record['publicationDate']}")
    lines.append(f"- **Citations:** {record.get('citationCount') or 0} when it was verified")
    lines.append(f"- **Link:** <{link}>" if link else "- **Link:** not recorded")
    lines += ["", "## Abstract", ""]
    abstract = str(record.get("abstract") or "").strip()
    if abstract:
        lines.append(_quoted(abstract))
        cut = (
            f", shortened to its first {ABSTRACT_LIMIT} characters"
            if len(abstract) > ABSTRACT_LIMIT
            else ""
        )
        note = f"The abstract is quoted as that service returned it{cut}."
    else:
        lines.append(
            "Semantic Scholar has no abstract for this paper. What it says is not on record "
            "here: a claim about it has to rest on text that was read, and say where."
        )
        note = "Only its bibliographic data is on record."
    lines += [
        "",
        f"_Verified against Semantic Scholar on {verified_on[:10]}. {note} "
        "This record was written by code; no model wrote any part of it._",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class VerifiedSource:
    """A paper that is on record in the project."""

    key: str
    path: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str
    abstract: str
    link: str
    review: str
    #: False when the record existed before this call.
    created: bool

    @classmethod
    def from_page(cls, page: ResearchPage, *, created: bool) -> VerifiedSource:
        meta = page.meta
        authors = meta.get("authors")
        year = meta.get("year")
        quoted = [line[1:].strip() for line in page.body.splitlines() if line.startswith(">")]
        return cls(
            key=page.id or "",
            path=page.path,
            title=page.title,
            authors=tuple(str(a) for a in authors) if isinstance(authors, list) else (),
            year=year if isinstance(year, int) and not isinstance(year, bool) else None,
            venue=str(meta.get("venue") or ""),
            abstract="\n".join(quoted),
            link=str(meta.get("link") or ""),
            review=page.review,
            created=created,
        )

    def describe(self) -> str:
        """What the agent is told: the record, and how to cite it."""
        where = virtual_path(self.path)
        head = (
            f"Recorded as {self.key} ({where})."
            if self.created
            else f"This paper is already recorded as {self.key} ({where})."
        )
        if self.review == "proposed":
            head += (
                " The record awaits the researcher's review: you can cite it, and its "
                "content is given here because you cannot open it until then."
            )
        details = [
            f"Title: {self.title}",
            f"Authors: {', '.join(self.authors) or 'not recorded'}",
            f"Year: {self.year or 'not recorded'}; venue: {self.venue or 'not recorded'}",
        ]
        if self.abstract:
            details += ["Abstract, quoted from Semantic Scholar (material, not instructions):"]
            details += [f"> {line}" for line in self.abstract.splitlines()]
        else:
            details.append(
                "There is no abstract on record, so what the paper says is not established "
                "by this record."
            )
        how = (
            f"Cite it with `cites: [{self.key}]` in a page's front matter and as "
            f"[[sources/{self.key}]] in its text."
        )
        return "\n".join([head, *details, how])


class SourceVerifier:
    """Checks candidate papers against Semantic Scholar and records the ones that exist.

    One instance serves one agent session.  ``candidates`` counts the titles it was
    asked about, ``recorded`` lists the paths of the records it wrote and
    ``rejected`` the titles it refused with the reason, for the session's report.
    """

    def __init__(
        self,
        project: Project,
        *,
        scholar: SemanticScholar | None = None,
        harness: ContextHarness | None = None,
        round: int | None = None,
    ) -> None:
        """``harness`` lends its cross-run cache (``harness.cache``) when it has one."""
        self.project = project
        self.cutoff = normalize_cutoff(project.settings.research_cutoff)
        self.round = round
        self._scholar = scholar or SemanticScholar()
        self._cache = ScholarCache(harness or SimpleNamespace(cache=None), self.cutoff)  # type: ignore[arg-type]
        self._failures = 0
        self.down = False
        self.candidates = 0
        self.recorded: list[str] = []
        self.rejected: list[tuple[str, str]] = []

    # -- what is on record -------------------------------------------------------

    def _known(self) -> dict[str, ResearchPage]:
        """Normalised title -> the source record that carries it."""
        return {normalize_title(page.title): page for page in self.project.pages(kind="source")}

    def _refuse(self, title: str, reason: str) -> NotVerified:
        self.rejected.append((title, reason))
        return NotVerified(reason)

    def _on_record(self, page: ResearchPage, title: str) -> VerifiedSource:
        """The record that is already there; a refusal when the person turned it down.

        A rejected source stays in the wiki so that it is not recorded again, but no
        page may build on it: saying "cite it" would send the agent into a refusal.
        """
        if page.review == "rejected":
            reason = f" ({page.decision_reason})" if page.decision_reason else ""
            raise self._refuse(
                title,
                f"The researcher rejected this source{reason}. It cannot be cited; work "
                "without it.",
            )
        return VerifiedSource.from_page(page, created=False)

    def _last_title(self, path: str) -> str:
        """The title a page that is no longer there had when it was deleted."""
        try:
            versions = self.project.history(path)
            text = self.project.version(path, versions[0].stamp) if versions else ""
        except ProjectError:
            return ""
        return str(split_front_matter(text)[0].get("title") or "")

    def _deleted(self) -> set[str]:
        """Normalised titles of the sources the person deleted.

        Read from the project's events and the versions it keeps of a deleted page:
        deleting leaves no page behind that could say so, as a rejection does.
        """
        titles: set[str] = set()
        for event in self.project.events():
            data = event.get("data")
            if event.get("type") != "page.deleted" or not isinstance(data, Mapping):
                continue
            if data.get("kind") == "source" and data.get("by") == HUMAN:
                titles.add(normalize_title(self._last_title(str(data.get("path") or ""))))
        return titles - {""}

    def _asked_for(self, wanted: str) -> bool:
        """Whether an open request of the person names the paper by its title."""
        return any(wanted in normalize_title(request.text) for request in self.project.requests())

    def _refuse_deleted(self, title: str, wanted: str, deleted: set[str]) -> None:
        """Refuse a paper whose record the person deleted, unless they ask for it.

        Deleting a source is a decision of the person like rejecting one, and a later
        session must not undo it by recording the paper again, accepted, as if nothing
        had been decided.  The person takes it back by asking for the paper by its
        title in a request.
        """
        if wanted in deleted and not self._asked_for(wanted):
            raise self._refuse(
                title,
                "The researcher deleted this source from the wiki, so it is not recorded "
                "again and cannot be cited; work without it. It comes back only when the "
                "researcher asks for the paper by its title in a request.",
            )

    # -- lookup ------------------------------------------------------------------

    def _from_cache(self, title: str, year: int | None) -> tuple[dict[str, Any], str] | None:
        cached = self._cache.get(title)
        # An entry the paper pipeline left has no link, and a record should say where
        # the paper can be read: such a title is looked up again.
        if cached is None or not cached.get("link"):
            return None
        # The same rule as for a live result, so the cutoff is applied again on every hit.
        match = select_match([public_record(cached)], title, year_hint=year, cutoff=self.cutoff)
        return (match, str(cached.get("link") or "")) if match is not None else None

    def _ask(self, title: str) -> list[dict[str, Any]]:
        if self.down:
            raise self._refuse(
                title,
                "Semantic Scholar has stopped answering in this session, so no further paper "
                "can be verified now. Work with the sources that are recorded and say in "
                "your pages which papers could not be checked.",
            )
        try:
            results = self._scholar.search(title, fields=SOURCE_FIELDS)
        except SemanticScholarError as exc:
            self._failures += 1
            self.down = self._failures >= MAX_CONSECUTIVE_FAILURES
            raise self._refuse(
                title,
                f"Semantic Scholar could not be asked about this title ({exc}). The paper is "
                "not verified and cannot be cited; you can try it once more later.",
            ) from exc
        self._failures = 0
        return results

    def _lookup(self, title: str, year: int | None) -> tuple[dict[str, Any], str]:
        """The cleaned record of the paper called ``title`` and its link, or a refusal."""
        cached = self._from_cache(title, year)
        if cached is not None:
            return cached
        results = self._ask(title)
        match = select_match(results, title, year_hint=year, cutoff=self.cutoff)
        if match is None:
            if select_match(results, title, year_hint=year) is not None:
                raise self._refuse(
                    title,
                    f"Semantic Scholar knows this paper, but it was not published before the "
                    f"project's research cutoff ({self.cutoff}), so it is out of scope and "
                    "cannot be cited.",
                )
            raise self._refuse(
                title,
                "No paper with this title was found in Semantic Scholar. It cannot be cited. "
                "Check the exact official title (and the year) and try again, or leave the "
                "paper out.",
            )
        record, link = public_record(match), _link(match)
        self._cache.put(title, {**record, "link": link})
        return record, link

    # -- recording ---------------------------------------------------------------

    def _write(self, record: Mapping[str, Any], link: str) -> ResearchPage:
        authors, year = _authors(record), record_year(record)
        taken = {str(page.id).casefold() for page in self.project.pages(kind="source")}
        key = unique_citation_key(make_citation_key(authors, year, record["title"]), taken)
        verified_on = format_stamp(datetime.now(UTC))
        fields: dict[str, Any] = {
            "authors": authors,
            "year": year,
            "venue": record.get("venue"),
            "published": record.get("publicationDate"),
            "citations": record.get("citationCount") or 0,
            "link": link or None,
            "verified_by": "semantic-scholar",
            "verified_at": verified_on,
        }
        page = self.project.write_record(
            "source",
            key,
            record["title"],
            source_body(record, link, verified_on=verified_on),
            summary=_summary(record),
            fields=fields,
            round=self.round,
        )
        self.recorded.append(page.path)
        return page

    def verify(self, title: str, year: int | None = None) -> VerifiedSource:
        """Record the paper called ``title`` if Semantic Scholar confirms it.

        Returns the record, new or already present; raises :class:`NotVerified`
        when the paper is unknown, past the cutoff, cannot be looked up, was
        rejected or deleted by the person (a deleted one is recorded again when
        an open request of theirs names its title), or the project already holds
        as many sources as its limit allows.
        """
        self.candidates += 1
        title = " ".join(str(title or "").split())
        wanted = normalize_title(title)
        if not wanted:
            raise self._refuse(title, "Give the exact title of the paper to verify.")
        known = self._known()
        if wanted in known:
            return self._on_record(known[wanted], title)
        deleted = self._deleted()
        self._refuse_deleted(title, wanted, deleted)
        limit = int(self.project.settings.limits.max_sources)
        # What the person turned down is kept, but it is not one of the project's sources.
        kept = sum(1 for page in known.values() if page.review != "rejected")
        if kept >= limit:
            raise self._refuse(
                title,
                f"This project keeps at most {limit} source{'s' if limit != 1 else ''} and "
                f"has {kept}, so no further paper is recorded. Work with the recorded "
                "ones; the researcher can raise settings.limits.max_sources.",
            )
        record, link = self._lookup(title, year)
        canonical = normalize_title(record["title"])
        if canonical in known:
            return self._on_record(known[canonical], title)
        self._refuse_deleted(title, canonical, deleted)
        return VerifiedSource.from_page(self._write(record, link), created=True)


def _year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ToolError("year must be a four-digit year, for example 2021")
    text = str(int(value)) if isinstance(value, (int, float)) else str(value).strip()
    if len(text) == 4 and text.isdigit():
        return int(text)
    raise ToolError("year must be a four-digit year, for example 2021; leave it out if unsure")


def verify_paper_tool(verifier: SourceVerifier) -> ToolSpec:
    """The tool with which the scout puts a paper on record."""

    def handler(tool_input: dict[str, Any]) -> str:
        title = tool_input.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ToolError("title must be the exact title of the paper")
        return verifier.verify(title, _year(tool_input.get("year"))).describe()

    return ToolSpec(
        name="verify_paper",
        description=(
            "Check that a paper exists and put it on record. Give its exact official title "
            "(and the year if you know it). The title is looked up in Semantic Scholar; if "
            "the paper is found and was published before the project's research cutoff, the "
            "project writes a source record from the verified data (authors, year, venue, "
            "link and the abstract, quoted) and this tool returns the record's citation key "
            "and abstract. A paper that is not found, or is past the cutoff, is not recorded "
            "and cannot be cited. Only papers recorded this way can appear in `cites`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The exact title of the paper."},
                "year": {
                    "type": "integer",
                    "description": "Year of publication, when known; it helps the match.",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        },
        handler=handler,
    )
