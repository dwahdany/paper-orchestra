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

"""Autoraters: citation F1, literature-review quality and side-by-side comparisons.

Each autorater lives in its own module (``citation_f1``, ``lit_review_quality``,
``sxs_lit_review_quality``, ``sxs_paper_quality``) and reaches a model only
through the provider-neutral LLM layer.  This module holds what they share:
bounded retries, schema-validated generation, a lock-protected results file,
the batch thread pool and the side-by-side aggregation.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, BeforeValidator, ValidationError

from paper_orchestra.llm.base import (
    Effort,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMParseError,
    LLMRefusal,
    Part,
    Text,
    normalize_parts,
)
from paper_orchestra.llm.parsing import parse_json
from paper_orchestra.utils.common import write_json

log = logging.getLogger(__name__)

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)

SUMMARY_KEY = "_summary_stats"


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------


def resolve_client(model_name: str, client: LLMClient | None = None) -> LLMClient:
    """Return ``client`` when one is injected, else the shared client serving ``model_name``."""
    if client is not None:
        return client
    from paper_orchestra.llm.router import get_client

    return get_client(model_name)


@dataclass(frozen=True)
class RetryPolicy:
    """How often a failed model call is repeated and how long to wait in between."""

    attempts: int = 4
    base_delay: float = 2.0
    max_delay: float = 60.0
    sleep: Callable[[float], None] = time.sleep

    def delay(self, failed_attempts: int) -> float:
        """Exponential backoff (base, 2x, 4x ... capped) with jitter so workers spread out."""
        ceiling = min(self.max_delay, self.base_delay * 2 ** (failed_attempts - 1))
        return ceiling * random.uniform(0.75, 1.0)


def call_with_retries(fn: Callable[[], T], *, what: str, retry: RetryPolicy | None = None) -> T:
    """Run ``fn``, repeating it after an :class:`LLMError` with backoff.

    The last error is re-raised once the attempts are used up.  Configuration
    errors and refusals are raised at once: asking again cannot change them.
    """
    policy = retry or RetryPolicy()
    attempts = max(1, policy.attempts)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (LLMConfigError, LLMRefusal):
            raise
        except LLMError as exc:
            if attempt == attempts:
                exc.add_note(f"{what} failed after {attempts} attempt(s).")
                raise
            delay = policy.delay(attempt)
            log.warning(
                "%s failed (attempt %d of %d): %s. Retrying in %.1f s.",
                what,
                attempt,
                attempts,
                exc,
                delay,
            )
            policy.sleep(delay)
    raise AssertionError("unreachable: the loop returns or raises")


def _schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Return your answer as a single JSON object inside a ```json fenced block. It must "
        "conform to this JSON Schema:\n```json\n"
        + json.dumps(schema.model_json_schema(), indent=1, sort_keys=True)
        + "\n```"
    )


def _validated(schema: type[ModelT], parsed: Any, raw_text: str) -> ModelT:
    if isinstance(parsed, schema):
        return parsed
    try:
        return schema.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(
            f"response does not match {schema.__name__}: {exc}", raw_text=raw_text
        ) from exc


def generate_structured(
    client: LLMClient,
    parts: Part | Sequence[Part],
    *,
    model: str,
    schema: type[ModelT],
    system: str | None = None,
    effort: Effort | None = None,
    temperature: float | None = None,
    label: str = "",
) -> ModelT:
    """One request whose answer is validated against ``schema``.

    Uses the provider's native structured output when it has one; otherwise
    (or when the provider rejects the schema) the JSON Schema is appended to
    the prompt and the reply is parsed and validated here.  A reply that does
    not fit raises :class:`LLMParseError`; retrying is the caller's decision.
    """
    options: dict[str, Any] = {
        "model": model,
        "system": system,
        "effort": effort,
        "temperature": temperature,
        "label": label,
        "parse_retries": 0,
    }
    if client.capabilities.structured_output:
        try:
            response = client.generate(parts, schema=schema, **options)
        except LLMConfigError as exc:
            log.info("%s: native structured output unavailable (%s); parsing text", label, exc)
        else:
            return _validated(schema, response.parsed, response.text)

    def parse(text: str) -> ModelT | None:
        try:
            return schema.model_validate(parse_json(text))
        except ValidationError:
            return None

    response = client.generate(
        [*normalize_parts(parts), Text(_schema_instructions(schema))], parser=parse, **options
    )
    return _validated(schema, response.parsed, response.text)


# ---------------------------------------------------------------------------
# Shared result files and batch execution
# ---------------------------------------------------------------------------


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Read a results file: ``{}`` when it is missing or empty, an error when it is damaged.

    A damaged file is never treated as empty, because the next write would
    replace whatever results it still holds.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{p} is not valid JSON ({exc}). Fix it or move it aside to start over."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object, found {type(data).__name__}.")
    return data


class JsonResultsFile:
    """A JSON object on disk that worker threads add entries to.

    Every update happens under one lock and replaces the file atomically, so
    neither a crash nor a concurrent writer leaves a half-written file behind.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data = load_json_object(self.path)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            write_json(self.path, self._data)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


def run_batch(
    items: Mapping[str, T], worker: Callable[[str, T], None], *, max_workers: int, what: str
) -> dict[str, str]:
    """Run ``worker(name, item)`` for every item in a thread pool.

    One item failing is logged and reported in the returned ``{name: error}``
    map; the others still run.  A configuration error stops the whole batch,
    since every remaining item would fail the same way.
    """
    failures: dict[str, str] = {}
    total = len(items)
    if not total:
        return failures
    executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
    try:
        futures = {executor.submit(worker, name, item): name for name, item in items.items()}
        for done, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                future.result()
            except LLMConfigError:
                raise
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                log.error("[%d/%d] %s failed for %s: %s", done, total, what, name, failures[name])
            else:
                log.info("[%d/%d] %s finished for %s.", done, total, what, name)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        log.error(
            "%s failed for %d of %d item(s): %s", what, len(failures), total, ", ".join(failures)
        )
    return failures


# ---------------------------------------------------------------------------
# Side-by-side (SxS) comparisons
# ---------------------------------------------------------------------------

_WINNERS = ("paper_1", "paper_2", "tie")
_TIE_WORD = re.compile(r"(?<![a-z])tie(?![a-z])")


def normalize_winner(value: Any) -> Any:
    """Map spellings such as ``"Paper 1"`` onto ``paper_1`` / ``paper_2`` / ``tie``.

    Anything else is returned unchanged, so validation rejects it instead of
    a verdict being guessed.
    """
    if not isinstance(value, str):
        return value
    text = "_".join(value.strip().lower().split())
    for winner in ("paper_1", "paper_2"):
        if winner in text:
            return winner
    return "tie" if _TIE_WORD.search(text) else value


Winner = Annotated[Literal["paper_1", "paper_2", "tie"], BeforeValidator(normalize_winner)]

_FIVE_POINT = {2: "win", 1: "leaning_win", 0: "tie", -1: "leaning_loss", -2: "loss"}
_SWAPPED = {"p1": "p2", "p2": "p1", "tie": "tie"}


def _presented_winner(rating: Mapping[str, Any]) -> str:
    winner = normalize_winner(rating.get("winner"))
    if winner not in _WINNERS:
        raise ValueError(f"SxS rating has no usable 'winner' field: {rating.get('winner')!r}")
    return {"paper_1": "p1", "paper_2": "p2", "tie": "tie"}[winner]


def aggregate_sxs_runs(
    rating1: Mapping[str, Any],
    rating2: Mapping[str, Any],
    *,
    paper1_path: str,
    paper2_path: str,
    exp1_name: str,
    exp2_name: str,
) -> dict[str, Any]:
    """Combine the two presentation orders of one pair into a result entry.

    ``rating1`` presented paper 1 first; ``rating2`` presented the papers
    swapped, so its winner is mapped back before scoring.  Scores are from
    paper 1's point of view: +1 per run it wins, -1 per run it loses.
    """
    r1_result = _presented_winner(rating1)
    r2_result = _SWAPPED[_presented_winner(rating2)]
    score = {"p1": 1, "p2": -1, "tie": 0}
    total_score = score[r1_result] + score[r2_result]
    return {
        "run1": dict(rating1),
        "run2": dict(rating2),
        "p1_result_run1": r1_result,
        "p1_result_run2": r2_result,
        "total_score": total_score,
        "category_5pt": _FIVE_POINT[total_score],
        "category_3pt": "win" if total_score > 0 else "loss" if total_score < 0 else "tie",
        "paper_1_path": paper1_path,
        "paper_2_path": paper2_path,
        "exp1_name": exp1_name,
        "exp2_name": exp2_name,
    }


def summarize_sxs_results(
    results: Mapping[str, Any], *, exp1_name: str, exp2_name: str
) -> dict[str, Any]:
    """Win/tie/loss counts and rates over every rated pair in ``results``."""
    five = dict.fromkeys(_FIVE_POINT.values(), 0)
    three = {"win": 0, "tie": 0, "loss": 0}
    valid = 0
    for paper_id, entry in results.items():
        if paper_id == SUMMARY_KEY or not isinstance(entry, dict) or "category_5pt" not in entry:
            continue
        if entry["category_5pt"] in five:
            five[entry["category_5pt"]] += 1
        if entry.get("category_3pt") in three:
            three[entry["category_3pt"]] += 1
        valid += 1

    def rate(count: int) -> float:
        return count / valid if valid else 0

    return {
        "exp1_name": exp1_name,
        "exp2_name": exp2_name,
        "total_valid_evals": valid,
        "five_point": {
            "wins": five["win"],
            "leaning_wins": five["leaning_win"],
            "ties": five["tie"],
            "leaning_losses": five["leaning_loss"],
            "losses": five["loss"],
            "win_rate": rate(five["win"]),
            "leaning_win_rate": rate(five["leaning_win"]),
            "tie_rate": rate(five["tie"]),
        },
        "three_point": {
            "wins": three["win"],
            "ties": three["tie"],
            "losses": three["loss"],
            "win_rate": rate(three["win"]),
            "tie_rate": rate(three["tie"]),
        },
    }


def conference_tag(project_folder: str | Path) -> str:
    """The venue tag used in SxS result file names, read off the project folder name."""
    name = str(project_folder).lower()
    if "iclr" in name:
        return "iclr"
    if "cvpr" in name:
        return "cvpr"
    return "unknown"


def _subfolders(folder: str | Path) -> list[str]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"project folder not found: {root}")
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def run_sxs_batch(
    rate_both_orders: Callable[[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    project_folder_1: str,
    project_folder_2: str,
    rel_path_1: str,
    rel_path_2: str,
    exp1_name: str,
    exp2_name: str,
    output_json_path: str | Path,
    max_workers: int,
    what: str,
) -> dict[str, Any]:
    """Rate every paper present in both project folders and write the results file.

    ``rate_both_orders(paper1_path, paper2_path)`` returns the ratings for the
    two presentation orders.  Pairs already in the results file are skipped,
    each finished pair is saved immediately, and the summary is stored under
    ``_summary_stats``.  Returns the content of the results file.
    """
    results = JsonResultsFile(output_json_path)
    second = set(_subfolders(project_folder_2))
    pairs: dict[str, tuple[str, str]] = {}
    for paper_id in _subfolders(project_folder_1):
        paper1 = Path(project_folder_1) / paper_id / rel_path_1
        paper2 = Path(project_folder_2) / paper_id / rel_path_2
        if paper_id not in second or not (paper1.is_file() and paper2.is_file()):
            continue
        if paper_id in results:
            log.info("[%s] Already rated; skipping.", paper_id)
            continue
        pairs[paper_id] = (str(paper1), str(paper2))
    log.info("%s: comparing [%s] with [%s] on %d paper(s).", what, exp1_name, exp2_name, len(pairs))

    def worker(paper_id: str, paths: tuple[str, str]) -> None:
        rating1, rating2 = rate_both_orders(*paths)
        entry = aggregate_sxs_runs(
            rating1,
            rating2,
            paper1_path=paths[0],
            paper2_path=paths[1],
            exp1_name=exp1_name,
            exp2_name=exp2_name,
        )
        results.set(paper_id, entry)
        log.info(
            "[%s] %s rated in both orders: 5pt %s, 3pt %s.",
            paper_id,
            what,
            entry["category_5pt"],
            entry["category_3pt"],
        )

    run_batch(pairs, worker, max_workers=max_workers, what=what)

    stats = summarize_sxs_results(results.snapshot(), exp1_name=exp1_name, exp2_name=exp2_name)
    results.set(SUMMARY_KEY, stats)
    five, three = stats["five_point"], stats["three_point"]
    log.info(
        "%s summary over %d pair(s). 5-point: [%s] wins %d (%.1f%%), leaning wins %d (%.1f%%), "
        "ties %d (%.1f%%), [%s] leaning wins %d, wins %d. 3-point: [%s] wins %d (%.1f%%), "
        "ties %d (%.1f%%), [%s] wins %d. Results saved to %s.",
        what,
        stats["total_valid_evals"],
        exp1_name,
        five["wins"],
        100 * five["win_rate"],
        five["leaning_wins"],
        100 * five["leaning_win_rate"],
        five["ties"],
        100 * five["tie_rate"],
        exp2_name,
        five["leaning_losses"],
        five["losses"],
        exp1_name,
        three["wins"],
        100 * three["win_rate"],
        three["ties"],
        100 * three["tie_rate"],
        exp2_name,
        three["losses"],
        results.path,
    )
    return results.snapshot()
