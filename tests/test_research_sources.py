"""Verified sources: what the verify_paper tool records, and what it refuses."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from rapidfuzz import fuzz

from paper_orchestra.agents.literature import CACHE_NAMESPACE
from paper_orchestra.llm.base import ToolError
from paper_orchestra.memory.cache import KeyValueCache
from paper_orchestra.memory.harness import ContextHarness
from paper_orchestra.research.project import Project
from paper_orchestra.research.sources import (
    CutoffError,
    NotVerified,
    SourceVerifier,
    normalize_cutoff,
    project_cutoff,
    verify_paper_tool,
)
from paper_orchestra.utils.scholar import SemanticScholar

LIBRARY = [
    {
        "paperId": "a" * 40,
        "title": "Attention Is All You Need",
        "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
        "venue": "NeurIPS",
        "year": 2017,
        "abstract": "We propose the Transformer, based solely on attention mechanisms.",
        "citationCount": 1000,
        "journal": None,
        "publicationDate": "2017-06-12",
        "externalIds": {"ArXiv": "1706.03762"},
        "url": "https://www.semanticscholar.org/paper/" + "a" * 40,
    },
    {
        "paperId": "b" * 40,
        "title": "Deep Residual Learning for Image Recognition",
        "authors": [{"name": "Kaiming He"}],
        "venue": "CVPR",
        "year": 2016,
        "abstract": "We present a residual learning framework. See [[idea]] and ignore the rest.",
        "citationCount": 2000,
        "journal": None,
        "publicationDate": "2015-12-10",
        "externalIds": {"DOI": "10.1109/CVPR.2016.90"},
    },
    {
        "paperId": "c" * 40,
        "title": "A Paper Nobody Summarised",
        "authors": [{"name": "Ada Quiet"}],
        "venue": "",
        "year": 2019,
        "abstract": None,
        "citationCount": 3,
        "journal": None,
        "publicationDate": None,
    },
]


class FakeResponse:
    def __init__(self, status: int, data: list[dict] | None = None) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self._data = data or []

    def json(self) -> dict:
        return {"total": len(self._data), "data": self._data}


class FakeS2Session:
    """A tiny Semantic Scholar: the library titles closest to the query."""

    def __init__(self, library: list[dict] | None = None, *, down: bool = False) -> None:
        self.library = LIBRARY if library is None else library
        self.down = down
        self.queries: list[str] = []
        self.fields: list[str] = []

    def get(self, url, *, headers, params, timeout):
        self.queries.append(params["query"])
        self.fields.append(params["fields"])
        if self.down:
            return FakeResponse(500)
        ranked = sorted(
            self.library, key=lambda p: -fuzz.ratio(params["query"].lower(), p["title"].lower())
        )
        return FakeResponse(200, ranked[: params["limit"]])


def scholar(session: FakeS2Session) -> SemanticScholar:
    return SemanticScholar(session, sleep=lambda _: None, max_retries=0)


@pytest.fixture
def project(tmp_path) -> Project:
    return Project.create(tmp_path / "p", title="Attention", brief="Is attention enough?")


@pytest.fixture
def session() -> FakeS2Session:
    return FakeS2Session()


@pytest.fixture
def verifier(project, session) -> SourceVerifier:
    return SourceVerifier(project, scholar=scholar(session), round=1)


def test_a_verified_paper_becomes_a_source_record_written_by_code(project, verifier):
    source = verifier.verify("attention is all you need", 2017)

    assert source.created and source.key == "Vaswani2017AttentionAll"
    page = project.page(source.key)
    assert page.path == "wiki/sources/Vaswani2017AttentionAll.md" == source.path
    assert (page.kind, page.id, page.author, page.review) == (
        "source",
        "Vaswani2017AttentionAll",
        "system",
        "accepted",
    )
    assert page.title == "Attention Is All You Need"
    assert page.round == 1
    assert page.relations == {}
    assert page.meta["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert page.meta["year"] == 2017
    assert page.meta["venue"] == "NeurIPS"
    assert page.meta["link"] == "https://arxiv.org/abs/1706.03762"
    assert "Vaswani" in page.summary and "2017" in page.summary
    assert "> We propose the Transformer, based solely on attention mechanisms." in page.body
    assert "<https://arxiv.org/abs/1706.03762>" in page.body
    today = datetime.now(UTC).date().isoformat()
    assert f"Semantic Scholar record fetched on {today}." in page.body
    assert page.meta["verified_by"] == "semantic-scholar"
    assert page.meta["verified_at"].startswith(today) and "verified_from" not in page.meta
    assert verifier.recorded == [page.path]


def test_the_link_prefers_a_doi_and_text_from_the_service_cannot_link_pages(project, verifier):
    source = verifier.verify("Deep Residual Learning for Image Recognition")

    page = project.page(source.key)
    assert page.meta["link"] == "https://doi.org/10.1109/CVPR.2016.90"
    # An abstract is quoted, not trusted: a wiki link in it must not become an edge.
    assert "[[idea]]" not in page.body
    assert project.graph()["edges"] == []


def test_a_title_the_service_does_not_know_is_refused_and_nothing_is_written(project, verifier):
    with pytest.raises(NotVerified, match="No paper with this title"):
        verifier.verify("Quantum Sourdough Dynamics in Large Language Models", 2024)

    assert project.pages(kind="source") == []
    assert verifier.candidates == 1 and verifier.recorded == []
    assert len(verifier.rejected) == 1


def test_a_paper_published_after_the_cutoff_is_refused(project, session):
    project.settings.research_cutoff = "2017-06"
    verifier = SourceVerifier(project, scholar=scholar(session))

    with pytest.raises(NotVerified, match="cutoff"):
        verifier.verify("Attention Is All You Need", 2017)
    assert verifier.verify("Deep Residual Learning for Image Recognition").created
    assert [page.id for page in project.pages(kind="source")] == ["He2016DeepResidual"]


def test_a_cutoff_given_as_a_year_means_before_that_year():
    assert normalize_cutoff(None) is None
    assert normalize_cutoff(" ") is None
    assert normalize_cutoff("2024") == "2024-01"
    assert normalize_cutoff("2024-06") == "2024-06"
    assert normalize_cutoff("2024-06-15") == "2024-06-15"


def test_a_cutoff_is_read_as_it_was_meant_or_not_at_all():
    # A year typed as a number, a month typed with one digit: what is meant is plain.
    assert normalize_cutoff(2024) == "2024-01"
    assert normalize_cutoff("2024-6") == "2024-06"
    assert normalize_cutoff(" 2024-06 ") == "2024-06"
    # Anything else is not guessed at, and above all is not read as "no cutoff".
    for unreadable in ("June 2024", "2024/06", "2024-13", "2024-00", "24-06", True, 20245, 2024.5):
        with pytest.raises(ValueError, match="research_cutoff"):
            normalize_cutoff(unreadable)


def edited_by_hand(tmp_path, cutoff) -> Project:
    """A project whose ``project.json`` got its cutoff from an editor, not from the app."""
    project = Project.create(tmp_path / "edited", title="Attention", brief="Is attention enough?")
    path = project.root / "project.json"
    data = json.loads(path.read_text())
    data["settings"]["research_cutoff"] = cutoff
    path.write_text(json.dumps(data))
    return Project.open(project.root)


@pytest.mark.parametrize("cutoff", ["2017/06", "June 2017", "2017-13", ["2017-06"]])
def test_a_cutoff_that_cannot_be_read_stops_every_lookup(tmp_path, session, cutoff):
    project = edited_by_hand(tmp_path, cutoff)

    with pytest.raises(CutoffError, match="research cutoff cannot be read"):
        project_cutoff(project)
    verifier = SourceVerifier(project, scholar=scholar(session))
    for title in ("Attention Is All You Need", "Deep Residual Learning for Image Recognition"):
        with pytest.raises(NotVerified, match="research cutoff cannot be read") as refusal:
            verifier.verify(title)
        assert "settings.research_cutoff" in str(refusal.value)
    # Not read as "no cutoff": nothing was asked for and nothing is on record.
    assert session.queries == [] and project.pages(kind="source") == []


@pytest.mark.parametrize("cutoff", [2017, "2017-1", "2017"])
def test_a_cutoff_typed_into_the_file_in_a_plain_form_is_applied(tmp_path, session, cutoff):
    project = edited_by_hand(tmp_path, cutoff)

    assert project_cutoff(project) == "2017-01"
    verifier = SourceVerifier(project, scholar=scholar(session))
    with pytest.raises(NotVerified, match="not published before"):
        verifier.verify("Attention Is All You Need", 2017)
    assert verifier.verify("Deep Residual Learning for Image Recognition").created


def test_the_same_paper_is_recorded_once_whatever_title_it_is_asked_by(project, verifier, session):
    first = verifier.verify("Attention Is All You Need")
    again = verifier.verify("Attention is all you need!")
    variant = verifier.verify("Attention Is All You Need (Transformer)")

    assert again.key == variant.key == first.key
    assert not again.created and not variant.created
    assert len(project.pages(kind="source")) == 1
    assert "already" in again.describe().lower()
    # The exact title needed no second lookup.
    assert len(session.queries) == 2


def test_a_resumed_session_finds_what_an_earlier_one_recorded(project, session):
    SourceVerifier(project, scholar=scholar(session)).verify("Attention Is All You Need")
    resumed = SourceVerifier(project, scholar=scholar(session))

    source = resumed.verify("Attention Is All You Need")

    assert not source.created and resumed.recorded == []
    assert len(project.pages(kind="source")) == 1


def test_two_papers_never_share_a_key(project, session):
    session.library = [
        dict(LIBRARY[0]),
        {**LIBRARY[0], "paperId": "d" * 40, "title": "Attention All: A Second Look"},
    ]
    verifier = SourceVerifier(project, scholar=scholar(session))

    first = verifier.verify("Attention Is All You Need")
    second = verifier.verify("Attention All: A Second Look")

    assert first.key == "Vaswani2017AttentionAll"
    assert second.key == "Vaswani2017AttentionAlla"


def test_the_project_keeps_at_most_max_sources(project, session):
    project.settings.limits.max_sources = 1
    verifier = SourceVerifier(project, scholar=scholar(session))
    verifier.verify("Attention Is All You Need")

    with pytest.raises(NotVerified, match="at most 1 source"):
        verifier.verify("Deep Residual Learning for Image Recognition")
    # What is already recorded can still be asked for.
    assert not verifier.verify("Attention Is All You Need").created


def test_a_paper_without_an_abstract_is_recorded_and_says_so(project, verifier):
    source = verifier.verify("A Paper Nobody Summarised")

    page = project.page(source.key)
    assert "no abstract" in page.body.lower()
    assert "no abstract" in source.describe().lower()
    assert page.meta["link"] == "https://www.semanticscholar.org/paper/" + "c" * 40


def test_a_service_that_is_down_is_not_asked_again_and_again(project):
    session = FakeS2Session(down=True)
    verifier = SourceVerifier(project, scholar=scholar(session))

    for _ in range(5):
        with pytest.raises(NotVerified, match="Semantic Scholar"):
            verifier.verify("Attention Is All You Need")

    assert verifier.down
    assert len(session.queries) == 3
    assert project.pages(kind="source") == []


def test_the_cross_run_cache_answers_without_a_lookup(project, tmp_path, session):
    harness = ContextHarness(cache=KeyValueCache(tmp_path / "cache"))
    SourceVerifier(project, scholar=scholar(session), harness=harness).verify(
        "Attention Is All You Need"
    )
    other = Project.create(tmp_path / "q", title="Other", brief="Another idea")
    offline = FakeS2Session(down=True)

    source = SourceVerifier(other, scholar=scholar(offline), harness=harness).verify(
        "Attention Is All You Need"
    )

    assert source.created and offline.queries == []
    assert other.page(source.key).meta["link"] == "https://arxiv.org/abs/1706.03762"


def at(day: str):
    """A clock that stands at noon of ``day``."""
    return lambda: datetime.fromisoformat(f"{day}T12:00:00").replace(tzinfo=UTC)


def cached_by_another_project(tmp_path, session, day: str) -> ContextHarness:
    """A harness whose cache holds the paper as another project fetched it on ``day``."""
    harness = ContextHarness(cache=KeyValueCache(tmp_path / "cache"))
    first = Project.create(tmp_path / "first", title="First", brief="An earlier idea")
    SourceVerifier(first, scholar=scholar(session), harness=harness, now=at(day)).verify(
        "Attention Is All You Need"
    )
    assert "fetched on " + day in first.page("Vaswani2017AttentionAll").body
    return harness


def test_a_record_from_the_cache_says_when_the_service_returned_it(project, tmp_path, session):
    harness = cached_by_another_project(tmp_path, session, "2026-10-20")
    asked = len(session.queries)

    source = SourceVerifier(
        project, scholar=scholar(session), harness=harness, now=at("2026-11-10")
    ).verify("Attention Is All You Need")

    page = project.page(source.key)
    assert len(session.queries) == asked  # the cache answered
    # The record is dated by the day the service returned it, not by the day it was copied.
    assert page.meta["verified_at"].startswith("2026-10-20")
    assert "Semantic Scholar record fetched on 2026-10-20" in page.body
    assert "cache" in page.body and "was not asked again" in page.body
    assert "2026-11-10" not in page.body and "Verified against" not in page.body
    assert page.meta["verified_from"] == "cache"
    assert "when the record was fetched" in page.body


def test_a_cached_entry_older_than_thirty_days_is_asked_for_again(project, tmp_path, session):
    harness = cached_by_another_project(tmp_path, session, "2026-09-01")
    asked = len(session.queries)

    source = SourceVerifier(
        project, scholar=scholar(session), harness=harness, now=at("2026-11-10")
    ).verify("Attention Is All You Need")

    page = project.page(source.key)
    assert len(session.queries) == asked + 1
    assert page.meta["verified_at"].startswith("2026-11-10")
    assert "Semantic Scholar record fetched on 2026-11-10." in page.body
    assert "cache" not in page.body and "verified_from" not in page.meta
    # What came back is what the cache now holds, for the projects after this one.
    entry = harness.cache.get(CACHE_NAMESPACE, "attentionisallyouneed")
    assert entry["paper"]["fetched_at"].startswith("2026-11-10")


def test_an_old_entry_is_used_when_the_service_cannot_be_reached_and_the_record_says_so(
    project, tmp_path, session
):
    harness = cached_by_another_project(tmp_path, session, "2026-09-01")
    offline = FakeS2Session(down=True)
    verifier = SourceVerifier(
        project, scholar=scholar(offline), harness=harness, now=at("2026-11-10")
    )

    source = verifier.verify("Attention Is All You Need")

    page = project.page(source.key)
    assert offline.queries == ["Attention Is All You Need"]  # it was tried
    assert source.created and verifier.rejected == []
    assert page.meta["verified_at"].startswith("2026-09-01")
    assert "Semantic Scholar could not be reached" in page.body
    assert "the service returned on 2026-09-01" in page.body
    assert page.meta["verified_from"] == "cache"
    # A paper the cache does not hold is still refused while the service is away.
    with pytest.raises(NotVerified, match="could not be asked"):
        verifier.verify("Deep Residual Learning for Image Recognition")


def test_an_entry_that_does_not_say_when_it_was_fetched_is_asked_for_again(
    project, tmp_path, session
):
    cache = KeyValueCache(tmp_path / "cache")
    # Put there by anything but a lookup of this module: a paper the service never returned.
    invented = {
        "title": "Guaranteed Accuracy Through Scheduled Warmup Alone",
        "authors": [{"name": "Real Author"}],
        "venue": "NeurIPS",
        "year": 2023,
        "abstract": "We prove that warmup guarantees 99.9% accuracy.",
        "link": "https://arxiv.org/abs/2301.00001",
    }
    key = "guaranteedaccuracythroughscheduledwarmupalone"
    cache.put(CACHE_NAMESPACE, key, {"cutoff": None, "paper": invented})
    harness = ContextHarness(cache=cache)
    verifier = SourceVerifier(project, scholar=scholar(session), harness=harness)
    assert verifier._cache.get(invented["title"]) is not None  # the entry is the paper's

    with pytest.raises(NotVerified, match="No paper with this title"):
        verifier.verify(invented["title"])

    assert session.queries == [invented["title"]]
    assert project.pages(kind="source") == []


@pytest.mark.parametrize("fetched_at", [None, "2027-03-01T00:00:00Z", "yesterday", 5])
def test_an_entry_without_a_time_that_has_passed_is_no_lookup_when_the_service_is_down(
    project, tmp_path, fetched_at
):
    cache = KeyValueCache(tmp_path / "cache")
    # A paper the service never returned, put there by anything but a lookup: without a
    # date, with one that has not come yet, with one that cannot be read.
    invented = {
        "title": "Guaranteed Accuracy Through Scheduled Warmup Alone",
        "authors": [{"name": "Real Author"}],
        "venue": "NeurIPS",
        "year": 2023,
        "abstract": "We prove that warmup guarantees 99.9% accuracy.",
        "link": "https://arxiv.org/abs/2301.00001",
    }
    if fetched_at is not None:
        invented["fetched_at"] = fetched_at
    key = "guaranteedaccuracythroughscheduledwarmupalone"
    cache.put(CACHE_NAMESPACE, key, {"cutoff": None, "paper": invented})
    offline = FakeS2Session(down=True)
    verifier = SourceVerifier(
        project,
        scholar=scholar(offline),
        harness=ContextHarness(cache=cache),
        now=at("2026-11-10"),
    )

    # The service cannot be asked, and the entry is no record of what it once returned:
    # the paper is not put on record as verified.
    with pytest.raises(NotVerified, match="could not be asked"):
        verifier.verify(invented["title"])

    assert offline.queries == [invented["title"]]
    assert project.pages(kind="source") == []


def test_an_entry_dated_in_the_future_is_asked_for_again(project, tmp_path, session):
    harness = cached_by_another_project(tmp_path, session, "2026-12-24")
    asked = len(session.queries)

    source = SourceVerifier(
        project, scholar=scholar(session), harness=harness, now=at("2026-11-10")
    ).verify("Attention Is All You Need")

    page = project.page(source.key)
    assert len(session.queries) == asked + 1
    assert page.meta["verified_at"].startswith("2026-11-10")
    assert "cache" not in page.body and "verified_from" not in page.meta


def test_the_cache_does_not_let_a_paper_past_a_later_cutoff(project, tmp_path, session):
    harness = ContextHarness(cache=KeyValueCache(tmp_path / "cache"))
    SourceVerifier(project, scholar=scholar(session), harness=harness).verify(
        "Attention Is All You Need"
    )
    other = Project.create(
        tmp_path / "q", title="Other", brief="x", settings={"research_cutoff": "2016-01"}
    )

    with pytest.raises(NotVerified):
        SourceVerifier(other, scholar=scholar(session), harness=harness).verify(
            "Attention Is All You Need"
        )
    assert other.pages(kind="source") == []


def test_the_tool_reports_the_record_and_how_to_cite_it(project, verifier):
    tool = verify_paper_tool(verifier)

    text = tool.handler({"title": "Attention Is All You Need", "year": "2017"})

    assert tool.name == "verify_paper"
    assert "Vaswani2017AttentionAll" in text
    assert "cites: [Vaswani2017AttentionAll]" in text
    assert "We propose the Transformer" in text
    with pytest.raises(ToolError, match="title"):
        tool.handler({"title": "  "})
    with pytest.raises(ToolError, match="year"):
        tool.handler({"title": "Attention Is All You Need", "year": "last spring"})


def test_a_source_held_for_review_is_still_recorded_and_citable(tmp_path, session):
    project = Project.create(
        tmp_path / "r", title="Vetting", brief="x", settings={"review": {"source": "review"}}
    )

    source = SourceVerifier(project, scholar=scholar(session)).verify("Attention Is All You Need")

    assert project.page(source.key).review == "proposed"
    assert "awaits the researcher's review" in source.describe()


def test_extra_fields_are_asked_for_so_the_record_has_a_link(verifier, session):
    verifier.verify("Attention Is All You Need")

    assert "externalIds" in session.fields[0] and "url" in session.fields[0]


def test_a_source_the_person_rejected_is_neither_offered_for_citation_nor_recorded_again(
    tmp_path, session
):
    project = Project.create(
        tmp_path / "r",
        title="Vetting",
        brief="x",
        settings={"review": {"source": "review"}, "limits": {"max_sources": 1}},
    )
    first = SourceVerifier(project, scholar=scholar(session))
    project.review(first.verify("Attention Is All You Need").path, "reject", reason="Off topic")
    verifier = SourceVerifier(project, scholar=scholar(session))

    for title in ("Attention Is All You Need", "Attention Is All You Need (Transformer)"):
        with pytest.raises(NotVerified, match=r"rejected this source \(Off topic\)") as refusal:
            verifier.verify(title)
        assert "cites:" not in str(refusal.value)
    assert len(project.pages(kind="source")) == 1 and verifier.recorded == []
    assert len(verifier.rejected) == 2
    # What was turned down does not take the place of a source the project may keep.
    assert verifier.verify("Deep Residual Learning for Image Recognition").created


def test_a_source_the_person_deleted_comes_back_only_when_they_ask_for_it(project, session):
    first = SourceVerifier(project, scholar=scholar(session))
    source = first.verify("Attention Is All You Need")
    assert project.page(source.key).review == "accepted"
    project.delete_page(source.path, author="human")
    # A later session of the scout comes across the same paper.
    later = SourceVerifier(project, scholar=scholar(session))
    asked = len(session.queries)

    with pytest.raises(NotVerified, match="deleted this source") as refusal:
        later.verify("Attention Is All You Need")
    assert "cites:" not in str(refusal.value)
    # Known by its title: the service was not asked again.
    assert len(session.queries) == asked
    # Under another wording of the title it is the same paper.
    with pytest.raises(NotVerified, match="deleted this source"):
        later.verify("Attention Is All You Need (Transformer)")
    assert project.pages(kind="source") == [] and later.recorded == []
    assert len(later.rejected) == 2
    # Other papers are recorded as before.
    assert later.verify("Deep Residual Learning for Image Recognition").created

    # Deleting is the person's decision, and so is taking it back: they ask for the paper.
    project.add_request("Please put Attention Is All You Need back among the sources.")
    back = later.verify("Attention Is All You Need")
    assert back.created and project.page(back.key).title == "Attention Is All You Need"
    assert not later.verify("Attention Is All You Need").created


def test_a_cache_entry_without_a_link_is_looked_up_again(project, tmp_path, session):
    cache = KeyValueCache(tmp_path / "cache")
    harness = ContextHarness(cache=cache)
    # As the paper pipeline leaves an entry: the bibliographic fields, no link.
    record = {key: LIBRARY[0][key] for key in ("title", "authors", "venue", "year", "abstract")}
    cache.put(CACHE_NAMESPACE, "attentionisallyouneed", {"cutoff": None, "paper": record})

    source = SourceVerifier(project, scholar=scholar(session), harness=harness).verify(
        "Attention Is All You Need"
    )

    assert session.queries == ["Attention Is All You Need"]
    assert project.page(source.key).meta["link"] == "https://arxiv.org/abs/1706.03762"
    # The entry now has the link, so the next project needs no lookup.
    assert cache.get(CACHE_NAMESPACE, "attentionisallyouneed")["paper"]["link"]
