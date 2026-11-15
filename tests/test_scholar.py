from __future__ import annotations

import logging
import threading

import pytest
import requests

from paper_orchestra.utils import scholar
from paper_orchestra.utils.scholar import (
    KEYED_INTERVAL_S,
    KEYLESS_INTERVAL_S,
    PAPER_FIELDS,
    SEARCH_URL,
    RateLimiter,
    SemanticScholar,
    SemanticScholarError,
    is_before_cutoff,
    normalize_title,
    parse_cutoff,
    select_match,
    title_similarity,
)

SECRET = "s2-SECRET-123"


class FakeTime:
    """A clock that only moves when somebody sleeps."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def clock(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


class FakeResponse:
    def __init__(self, status: int = 200, payload=None, headers=None, bad_json=False) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"total": 0, "data": []}
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("Expecting value")
        return self._payload


class FakeSession:
    """Answers from a queue (responses or exceptions), then with empty results."""

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, *, headers, params, timeout):
        self.calls.append(
            {"url": url, "headers": dict(headers), "params": dict(params), "timeout": timeout}
        )
        item = self._responses.pop(0) if self._responses else FakeResponse()
        if isinstance(item, Exception):
            raise item
        return item


def paper(title: str, year: int | None = 2020, date: str | None = "2020-06-01", **extra) -> dict:
    return {"title": title, "year": year, "publicationDate": date, **extra}


def results(*papers: dict) -> FakeResponse:
    return FakeResponse(payload={"total": len(papers), "data": list(papers)})


def make_client(session: FakeSession, **kwargs) -> tuple[SemanticScholar, FakeTime]:
    fake_time = FakeTime()
    client = SemanticScholar(session, sleep=fake_time.sleep, clock=fake_time.clock, **kwargs)
    return client, fake_time


# -- cutoff ------------------------------------------------------------------


def test_parse_cutoff_accepts_a_month_and_ignores_the_day():
    assert parse_cutoff("2024-11") == (2024, 11)
    assert parse_cutoff(" 2024-3 ") == (2024, 3)
    assert parse_cutoff("2024-11-15") == (2024, 11)
    assert parse_cutoff(None) is None
    assert parse_cutoff("  ") is None


@pytest.mark.parametrize("bad", ["2024", "2024/11", "2024-13", "2024-00", "November 2024", "24-11"])
def test_parse_cutoff_rejects_anything_else(bad):
    with pytest.raises(ValueError, match="YYYY-MM"):
        parse_cutoff(bad)


@pytest.mark.parametrize(
    ("date", "year", "cutoff", "allowed"),
    [
        # The boundary is the first day of the cutoff month, exclusive.
        ("2024-10-31", 2024, "2024-11", True),
        ("2024-11-01", 2024, "2024-11", False),
        ("2024-11-20", 2024, "2024-11", False),
        ("2025-03-01", 2025, "2024-11", False),
        ("2023-12-31", 2023, "2024-01", True),
        ("2024-01-01", 2024, "2024-01", False),
        # Year only: the paper may have appeared in December of that year.
        (None, 2023, "2024-11", True),
        (None, 2024, "2024-11", False),
        (None, 2024, "2024-01", False),
        (None, 2025, "2024-11", False),
        (None, "2023", "2024-11", True),
        # Year and month only.
        ("2024-10", None, "2024-11", True),
        ("2024-11", None, "2024-11", False),
        # Undated papers cannot be shown to precede the cutoff.
        (None, None, "2024-11", False),
        ("not a date", None, "2024-11", False),
        # An unreadable date falls back to the year; a readable one beats the year.
        ("not a date", 2023, "2024-11", True),
        ("2023-06-01", 2025, "2024-11", True),
        ("2024-12-01", 2023, "2024-11", False),
        # A day in the cutoff is ignored.
        ("2024-11-10", 2024, "2024-11-15", False),
        ("2024-10-30", 2024, "2024-11-15", True),
        # No cutoff, no restriction.
        ("2030-01-01", 2030, None, True),
        (None, None, "", True),
    ],
)
def test_the_cutoff_rule(date, year, cutoff, allowed):
    assert is_before_cutoff(date, year, cutoff) is allowed


def test_a_malformed_cutoff_is_an_error_not_an_open_gate():
    with pytest.raises(ValueError):
        is_before_cutoff("2030-01-01", 2030, "2024/11")


# -- title matching ----------------------------------------------------------


def test_normalize_title_ignores_case_punctuation_and_keeps_other_scripts():
    assert normalize_title("Attention Is All You Need!") == "attentionisallyouneed"
    assert normalize_title("  attention is  all you need ") == "attentionisallyouneed"
    assert normalize_title("深度学习") == "深度学习"
    assert normalize_title("?!") == ""


def test_select_match_needs_more_than_the_threshold():
    exact = paper("Deep Residual Learning for Image Recognition")
    assert select_match([exact], "deep residual learning for image recognition") == exact
    assert select_match([exact], "Graph Attention Networks") is None
    assert select_match([], "anything") is None
    assert select_match([{"title": None}, {"year": 2020}], "anything") is None


def test_select_match_year_hint_lifts_a_borderline_title():
    title = "Deep Residual Learning for Image Recognition"
    query = "Deep Residual Learning"
    assert 60 < title_similarity(query, title) <= 70  # not enough on its own
    candidate = paper(title, year=2016)
    assert select_match([candidate], query) is None
    assert select_match([candidate], query, year_hint=2015) is None
    assert select_match([candidate], query, year_hint=2016) == candidate


def test_select_match_prefers_the_best_title_and_then_the_hinted_year():
    preprint = paper("Segment Anything", year=2023, date="2023-04-05")
    sequel = paper("Segment Anything 2", year=2024, date="2024-08-01")
    assert select_match([preprint, sequel], "Segment Anything 2") == sequel
    assert select_match([sequel, preprint], "Segment Anything", year_hint=2023) == preprint


def test_select_match_filters_by_cutoff_before_matching():
    preprint = paper("Segment Anything", year=2023, date="2023-04-05")
    sequel = paper("Segment Anything 2", year=2024, date="2024-08-01")
    # The exact match is past the cutoff, so the older paper of nearly the same name wins.
    assert select_match([preprint, sequel], "Segment Anything 2", cutoff="2024-01") == preprint
    assert select_match([sequel], "Segment Anything 2", cutoff="2024-08") is None
    assert select_match([sequel], "Segment Anything 2", cutoff="2024-09") == sequel


# -- client: requests --------------------------------------------------------


def test_search_sends_the_documented_request_without_a_key():
    session = FakeSession(results(paper("Attention Is All You Need")))
    client, _ = make_client(session)
    found = client.search("Attention Is All You Need")
    assert [p["title"] for p in found] == ["Attention Is All You Need"]
    call = session.calls[0]
    assert call["url"] == SEARCH_URL
    assert call["headers"] == {}
    assert call["params"] == {
        "query": "Attention Is All You Need",
        "limit": 3,
        "fields": PAPER_FIELDS,
    }
    assert call["timeout"] == client.timeout
    assert not client.has_api_key


def test_search_replaces_hyphens_and_skips_empty_queries():
    session = FakeSession()
    client, _ = make_client(session)
    client.search("BERT: Pre-training of  Deep Bidirectional Transformers", limit=1, fields="title")
    assert session.calls[0]["params"] == {
        "query": "BERT: Pre training of Deep Bidirectional Transformers",
        "limit": 1,
        "fields": "title",
    }
    assert client.search(" - ") == []
    assert len(session.calls) == 1


def test_the_key_is_read_when_a_request_is_made(monkeypatch):
    session = FakeSession()
    client, _ = make_client(session)  # constructed before the key exists
    client.search("one")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", f" {SECRET}\n")
    assert client.has_api_key
    client.search("two")
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY")
    client.search("three")
    assert [call["headers"] for call in session.calls] == [{}, {"X-API-KEY": SECRET}, {}]


def test_the_key_never_reaches_logs_errors_or_repr(monkeypatch, caplog):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", SECRET)
    caplog.set_level(logging.DEBUG)
    session = FakeSession(
        FakeResponse(429),
        FakeResponse(403),
        requests.exceptions.InvalidHeader(f"Invalid header value {SECRET!r}"),
    )
    client, _ = make_client(session)
    with pytest.raises(SemanticScholarError, match="HTTP 403") as forbidden:
        client.search("first")
    assert "SEMANTIC_SCHOLAR_API_KEY" in str(forbidden.value)  # says where to look, not what
    with pytest.raises(SemanticScholarError, match="InvalidHeader") as invalid:
        client.search("second")
    assert invalid.value.__cause__ is None and invalid.value.__suppress_context__
    for text in (caplog.text, str(forbidden.value), str(invalid.value), repr(client)):
        assert SECRET not in text
    assert "retrying" in caplog.text


# -- client: rate limiting ---------------------------------------------------


def test_requests_are_spaced_about_a_second_apart_without_a_key():
    session = FakeSession()
    client, fake_time = make_client(session)
    for query in ("a", "b", "c"):
        client.search(query)
    assert fake_time.sleeps == pytest.approx([KEYLESS_INTERVAL_S, KEYLESS_INTERVAL_S])
    assert 1.0 <= KEYLESS_INTERVAL_S <= 1.5


def test_requests_are_faster_with_a_key_and_the_pace_can_be_set(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", SECRET)
    client, fake_time = make_client(FakeSession())
    for query in ("a", "b", "c"):
        client.search(query)
    assert fake_time.sleeps == pytest.approx([KEYED_INTERVAL_S, KEYED_INTERVAL_S])
    assert KEYED_INTERVAL_S < KEYLESS_INTERVAL_S

    fast, fast_time = make_client(FakeSession(), min_interval=0.1)
    fast.search("a")
    fast.search("b")
    assert fast_time.sleeps == pytest.approx([0.1])


def test_no_wait_when_enough_time_has_passed():
    client, fake_time = make_client(FakeSession())
    client.search("a")
    fake_time.now += 5.0
    client.search("b")
    assert fake_time.sleeps == []


def test_clients_share_one_process_wide_limiter_by_default(monkeypatch):
    fake_time = FakeTime()
    monkeypatch.setattr(
        scholar, "_PROCESS_LIMITER", RateLimiter(clock=fake_time.clock, sleep=fake_time.sleep)
    )
    first, second = SemanticScholar(FakeSession()), SemanticScholar(FakeSession())
    first.search("a")
    second.search("b")  # a different client still has to wait its turn
    first.search("c")
    assert fake_time.sleeps == pytest.approx([KEYLESS_INTERVAL_S, KEYLESS_INTERVAL_S])


def test_the_limiter_gives_concurrent_callers_distinct_slots():
    sleeps: list[float] = []
    lock = threading.Lock()

    def record(seconds: float) -> None:
        with lock:
            sleeps.append(seconds)

    # The clock stands still, so each caller's wait is exactly the distance to its slot.
    limiter = RateLimiter(clock=lambda: 100.0, sleep=record)
    threads = [threading.Thread(target=limiter.wait, args=(1.0,)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(sleeps) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


# -- client: retries ---------------------------------------------------------


def test_rate_limit_and_server_errors_are_retried_with_backoff():
    session = FakeSession(
        FakeResponse(429),
        FakeResponse(503),
        requests.ConnectionError("connection reset"),
        requests.Timeout("timed out"),
        results(paper("Found At Last")),
    )
    client, fake_time = make_client(session, min_interval=0.0)
    assert [p["title"] for p in client.search("found at last")] == ["Found At Last"]
    assert len(session.calls) == 5
    assert fake_time.sleeps == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.parametrize(
    "fault",
    [
        requests.exceptions.ChunkedEncodingError("connection broken: IncompleteRead"),
        requests.exceptions.ContentDecodingError("incorrect header check"),
    ],
)
def test_a_response_cut_off_or_garbled_in_transit_is_retried(fault):
    session = FakeSession(fault, results(paper("Found At Last")))
    client, fake_time = make_client(session, min_interval=0.0)
    assert [p["title"] for p in client.search("found at last")] == ["Found At Last"]
    assert len(session.calls) == 2 and fake_time.sleeps == [1.0]


def test_retry_after_is_honoured_and_capped():
    session = FakeSession(
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(429, headers={"Retry-After": "600"}),
        FakeResponse(429, headers={"Retry-After": "soon"}),
        results(),
    )
    client, fake_time = make_client(session, min_interval=0.0, max_backoff=30.0)
    assert client.search("x") == []
    assert fake_time.sleeps == [7.0, 30.0, 4.0]


def test_gives_up_after_max_retries():
    session = FakeSession(*[FakeResponse(500) for _ in range(10)])
    client, fake_time = make_client(session, min_interval=0.0, max_retries=2)
    with pytest.raises(SemanticScholarError, match=r"after 3 attempts \(HTTP 500\)"):
        client.search("x")
    assert len(session.calls) == 3
    assert fake_time.sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_errors_are_not_retried(status):
    session = FakeSession(FakeResponse(status))
    client, fake_time = make_client(session)
    with pytest.raises(SemanticScholarError, match=f"HTTP {status}"):
        client.search("x")
    assert len(session.calls) == 1 and fake_time.sleeps == []


def test_a_backoff_holds_back_every_client_on_the_same_limiter():
    fake_time = FakeTime()
    limiter = RateLimiter(clock=fake_time.clock, sleep=fake_time.sleep)
    throttled = SemanticScholar(
        FakeSession(FakeResponse(429), FakeResponse(500)),
        limiter=limiter,
        min_interval=0.0,
        max_retries=1,
        backoff=5.0,
    )
    bystander = SemanticScholar(FakeSession(), limiter=limiter, min_interval=0.0)
    with pytest.raises(SemanticScholarError):
        throttled.search("x")
    assert fake_time.sleeps == [5.0]
    limiter.defer(3.0)  # what a retry in another thread would have done
    bystander.search("y")
    assert fake_time.sleeps == [5.0, 3.0]


def test_unusable_bodies_are_errors_and_odd_shapes_are_tolerated():
    client, _ = make_client(FakeSession(FakeResponse(bad_json=True)), min_interval=0.0)
    with pytest.raises(SemanticScholarError, match="not JSON"):
        client.search("x")
    odd = FakeSession(
        FakeResponse(payload={"total": 0}),
        FakeResponse(payload={"data": None}),
        FakeResponse(payload=["unexpected"]),
        FakeResponse(payload={"data": ["junk", paper("Kept")]}),
    )
    client, _ = make_client(odd, min_interval=0.0)
    assert [client.search("x") for _ in range(3)] == [[], [], []]
    assert [p["title"] for p in client.search("x")] == ["Kept"]


# -- find_paper --------------------------------------------------------------


def test_find_paper_combines_search_matching_and_cutoff():
    hits = results(
        paper("Unrelated Work on Protein Folding", year=2019, date="2019-01-01"),
        paper("LoRA: Low-Rank Adaptation of Large Language Models", year=2021, date="2021-06-17"),
    )
    client, _ = make_client(FakeSession(hits, hits, hits), min_interval=0.0)
    title = "LoRA: Low-Rank Adaptation of Large Language Models"
    found = client.find_paper(title, year_hint=2021, cutoff="2024-10")
    assert found is not None and found["year"] == 2021
    assert client.find_paper(title, cutoff="2021-06") is None  # published in the cutoff month
    assert client.find_paper("A Paper That Does Not Exist Anywhere", cutoff="2024-10") is None


def test_find_paper_propagates_service_failures():
    client, _ = make_client(FakeSession(FakeResponse(404)))
    with pytest.raises(SemanticScholarError):
        client.find_paper("anything")
