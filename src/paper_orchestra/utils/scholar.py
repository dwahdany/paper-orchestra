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

"""Semantic Scholar lookups: the anti-hallucination gate of the literature agent.

A paper proposed by a model is only cited when Semantic Scholar knows a paper
with (nearly) that title.  This module holds the HTTP client, the title
matching rule and the one research-cutoff rule of the pipeline.

The API key is read from ``SEMANTIC_SCHOLAR_API_KEY`` when a request is made,
never at import time, and it is never logged or put into an error message.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

import requests
from rapidfuzz import fuzz

log = logging.getLogger(__name__)

API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PAPER_FIELDS = "title,authors,venue,year,abstract,citationCount,journal,publicationDate"

#: A search result is accepted when its similarity to the requested title is above this.
MATCH_THRESHOLD = 70.0
#: Added to the similarity when the result's year equals the year the caller expected.
YEAR_BONUS = 10.0

# Keyless requests share one public quota with every other anonymous client, so they are
# spaced a little wider than the one request per second a personal key is granted.  Keyed
# requests keep a margin under that second too: sent exactly one second apart, two of them
# can reach the service within the same second, and it answers the second with HTTP 429.
KEYLESS_INTERVAL_S = 1.2
KEYED_INTERVAL_S = 1.1

_CUTOFF = re.compile(r"^\s*(\d{4})-(\d{1,2})(?:-\d{1,2})?\s*$")
_DATE = re.compile(r"^\s*(\d{4})(?:-(\d{1,2}))?")


class SemanticScholarError(RuntimeError):
    """A lookup could not be completed (as opposed to finding nothing)."""


# ---------------------------------------------------------------------------
# Cutoff rule
# ---------------------------------------------------------------------------


def parse_cutoff(cutoff: str | None) -> tuple[int, int] | None:
    """``"2024-10"`` -> ``(2024, 10)``; ``None`` or an empty string -> no cutoff.

    ``YYYY-MM-DD`` is accepted and the day is ignored, because the cutoff is
    always the first day of the month.  Anything else is a configuration error.
    """
    if cutoff is None or not str(cutoff).strip():
        return None
    match = _CUTOFF.match(str(cutoff))
    if not match or not 1 <= int(match.group(2)) <= 12:
        raise ValueError(
            f"research cutoff must look like YYYY-MM (for example 2024-10), got {cutoff!r}"
        )
    return int(match.group(1)), int(match.group(2))


def _year_month(publication_date: str | None, year: Any) -> tuple[int, int | None] | None:
    """What is known about when a paper appeared: ``(year, month or None)``."""
    match = _DATE.match(publication_date) if isinstance(publication_date, str) else None
    if match:
        month = int(match.group(2)) if match.group(2) else None
        return int(match.group(1)), month if month and 1 <= month <= 12 else None
    try:
        return (int(year), None) if year else None
    except (TypeError, ValueError):
        return None


def is_before_cutoff(publication_date: str | None, year: Any, cutoff: str | None) -> bool:
    """The research-cutoff rule.  Every filter in the pipeline calls this function.

    A paper is allowed iff its publication date, or its year when no date is
    known, is strictly before the first day of the cutoff month:

    * a dated paper is allowed when its (year, month) is earlier than the
      cutoff's (the day never matters, the boundary being the first of a month);
    * a paper known only by its year may have appeared at any time in that
      year, so it is allowed only when the year is earlier than the cutoff year;
    * a paper with neither date nor year cannot be shown to precede the cutoff
      and is not allowed;
    * without a cutoff everything is allowed.

    ``publication_date`` is ``YYYY-MM-DD`` as Semantic Scholar reports it; when
    it is present it takes precedence over ``year``.
    """
    limit = parse_cutoff(cutoff)
    if limit is None:
        return True
    known = _year_month(publication_date, year)
    if known is None:
        return False
    paper_year, paper_month = known
    # An unknown month counts as December: the latest the paper can have appeared.
    return (paper_year, paper_month or 12) < limit


# ---------------------------------------------------------------------------
# Title matching
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """Case-, punctuation- and spacing-insensitive form of a title, for de-duplication."""
    return "".join(char for char in title.casefold() if char.isalnum())


def title_similarity(query: str, title: str) -> float:
    """Similarity of two titles on a 0-100 scale (case-insensitive edit ratio)."""
    return fuzz.ratio(query.lower(), title.lower())


def select_match(
    results: Iterable[Mapping[str, Any]],
    title: str,
    *,
    year_hint: int | None = None,
    cutoff: str | None = None,
) -> dict[str, Any] | None:
    """The search result that is the paper called ``title``, or ``None``.

    Results past the cutoff are discarded first, so an older paper of the same
    name can still match.  The best remaining result wins when its similarity
    (plus :data:`YEAR_BONUS` when its year equals ``year_hint``) is above
    :data:`MATCH_THRESHOLD`.
    """
    best: Mapping[str, Any] | None = None
    best_score = 0.0
    for result in results:
        candidate_title = result.get("title")
        if not candidate_title or not isinstance(candidate_title, str):
            continue
        if not is_before_cutoff(result.get("publicationDate"), result.get("year"), cutoff):
            continue
        score = title_similarity(title, candidate_title)
        if year_hint and result.get("year") == year_hint:
            score += YEAR_BONUS
        if score > best_score:
            best, best_score = result, score
    return dict(best) if best is not None and best_score > MATCH_THRESHOLD else None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Spaces requests in time.  Shared by every thread that holds the same instance."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self, interval: float) -> None:
        """Block until this caller's turn, then reserve ``interval`` seconds after it."""
        with self._lock:
            now = self._clock()
            start = max(now, self._next_start)
            self._next_start = start + interval
        if start > now:
            self._sleep(start - now)

    def defer(self, seconds: float) -> None:
        """Push every caller's next turn at least ``seconds`` into the future (backoff)."""
        with self._lock:
            self._next_start = max(self._next_start, self._clock() + seconds)


#: One limiter for the whole process: the quota belongs to the API key (or the
#: machine's address), not to whichever agent happens to be making the request.
_PROCESS_LIMITER = RateLimiter()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class HttpSession(Protocol):
    """The part of ``requests.Session`` the client uses."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any],
        timeout: float,
    ) -> Any: ...


# Network faults worth another attempt.  The last two are a response that was cut off or
# garbled on the way: requests reads the body inside ``get`` and reports those separately.
_TRANSIENT_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)


def _retry_after(response: Any) -> float | None:
    """Seconds the server asked us to wait, when it said so in a usable form."""
    value = (getattr(response, "headers", None) or {}).get("Retry-After")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


class SemanticScholar:
    """Paper search with rate limiting and retries.

    ``session``, ``sleep`` and ``clock`` exist for tests.  By default all
    clients of a process share one rate limiter; passing ``sleep`` or ``clock``
    (or ``limiter``) gives the client its own.
    """

    def __init__(
        self,
        session: HttpSession | None = None,
        *,
        api_key: str | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        min_interval: float | None = None,
        max_retries: int = 4,
        backoff: float = 1.0,
        max_backoff: float = 30.0,
        timeout: float = 15.0,
    ) -> None:
        self._session = session
        self._explicit_key = api_key
        if limiter is None:
            if sleep is None and clock is None:
                limiter = _PROCESS_LIMITER
            else:
                limiter = RateLimiter(clock=clock or time.monotonic, sleep=sleep or time.sleep)
        self._limiter = limiter
        self._min_interval = min_interval
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.timeout = timeout

    def _api_key(self) -> str | None:
        return (self._explicit_key or os.environ.get(API_KEY_ENV) or "").strip() or None

    @property
    def has_api_key(self) -> bool:
        """Whether requests will be authenticated (for diagnostics; never the key itself)."""
        return self._api_key() is not None

    def _http(self) -> HttpSession:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _get(self, params: Mapping[str, Any]) -> Any:
        """GET the search endpoint, retrying rate limits, server errors and network faults."""
        failure = ""
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            key = self._api_key()
            interval = self._min_interval
            if interval is None:
                interval = KEYED_INTERVAL_S if key else KEYLESS_INTERVAL_S
            self._limiter.wait(interval)
            server_delay: float | None = None
            try:
                response = self._http().get(
                    SEARCH_URL,
                    headers={"X-API-KEY": key} if key else {},
                    params=params,
                    timeout=self.timeout,
                )
            except _TRANSIENT_ERRORS as exc:
                failure = type(exc).__name__
            except requests.RequestException as exc:
                # Not retryable.  The original message can quote request headers (an API key
                # with a stray newline is reported verbatim), so only the type is passed on.
                raise SemanticScholarError(
                    f"Semantic Scholar request could not be sent ({type(exc).__name__})"
                ) from None
            else:
                status = response.status_code
                if status == 200:
                    return response
                if status != 429 and status < 500:
                    hint = f" (check {API_KEY_ENV})" if status in (401, 403) else ""
                    raise SemanticScholarError(
                        f"Semantic Scholar rejected the request with HTTP {status}{hint}"
                    )
                failure = f"HTTP {status}"
                server_delay = _retry_after(response)
            if attempt + 1 < attempts:
                delay = min(self.max_backoff, self.backoff * 2**attempt)
                if server_delay is not None:
                    delay = min(self.max_backoff, max(delay, server_delay))
                log.info(
                    "Semantic Scholar request failed (%s); retrying in %.1f s (attempt %d of %d)",
                    failure,
                    delay,
                    attempt + 1,
                    attempts,
                )
                self._limiter.defer(delay)
        raise SemanticScholarError(
            f"Semantic Scholar request failed after {attempts} attempts ({failure})"
        )

    def search(
        self, query: str, *, limit: int = 3, fields: str = PAPER_FIELDS
    ) -> list[dict[str, Any]]:
        """Relevance search.  Returns the raw result objects (possibly none).

        Raises :class:`SemanticScholarError` when the service cannot be reached
        or keeps refusing; an empty list means it answered and found nothing.
        """
        # The search endpoint returns nothing for hyphenated terms, so "Pre-training" has to
        # be sent as "Pre training".
        cleaned = " ".join(query.replace("-", " ").split())
        if not cleaned:
            return []
        response = self._get({"query": cleaned, "limit": limit, "fields": fields})
        try:
            payload = response.json()
        except ValueError as exc:
            raise SemanticScholarError("Semantic Scholar returned a body that is not JSON") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def find_paper(
        self,
        title: str,
        *,
        year_hint: int | None = None,
        cutoff: str | None = None,
    ) -> dict[str, Any] | None:
        """The paper called ``title`` published before ``cutoff``, or ``None`` if unknown."""
        return select_match(self.search(title), title, year_hint=year_hint, cutoff=cutoff)
