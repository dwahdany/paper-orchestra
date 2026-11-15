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

"""Citation F1: how well a generated paper's reference list matches the original's.

Both reference lists are extracted with a model, resolved to Semantic Scholar
paper ids, and compared.  The original's references are also split into
must-have (P0) and good-to-have (P1) so recall can be reported per priority.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import requests

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.eval import (
    JsonResultsFile,
    RetryPolicy,
    call_with_retries,
    load_json_object,
    resolve_client,
    run_batch,
)
from paper_orchestra.eval.references import (
    extract_paper_title_from_citation,
    get_paper_references_from_pdf,
)
from paper_orchestra.llm.base import LLMClient
from paper_orchestra.llm.parsing import parse_json_object
from paper_orchestra.utils.common import write_json
from paper_orchestra.utils.pdf import load_paper

log = logging.getLogger(__name__)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"

#: Where the benchmark datasets live when a run folder has no copy of the original paper.
DEFAULT_DATASET_DIR = Path("datasets")
_DATASET_VENUES = ("cvpr2025", "iclr2025")

OVERALL_KEY = "overall_metrics"
SCORE_KEYS = ("overall_precision", "overall_recall", "overall_f1", "p0_recall", "p1_recall")
COUNT_KEYS = (
    "ground_truth_ids_all",
    "ground_truth_ids_p0",
    "ground_truth_ids_p1",
    "generated_ids_all",
    "matches_overall",
    "matches_p0",
    "matches_p1",
)

PRIORITY_SYSTEM_PROMPT = """You are an expert academic reviewer. You will be given the text of a paper inside <paper_text> tags and its numbered reference list inside <references> tags.
Your goal is to categorize the provided references into two priorities:
- P0 (Must-Have): Core citations strictly necessary for the paper. These include:
    * Baselines directly compared against in experiments
    * Datasets the paper utilizes or evaluates on
    * Core methods the paper is directly building upon or modifying
    * Metrics or standard numbers heavily relied upon and cited from another paper
- P1 (Good-To-Have): Supplemental citations. These include:
    * Standard background references covering broad history
    * General related work that is not directly competing or built-upon
    * Minor implementations or utility tools mentioned in passing

Return only a JSON dictionary where the keys are the exact reference numbers (e.g., "1", "2") and the values are either "P0" or "P1". Include every reference number.
Example output:

```json
{
    "1": "P0",
    "2": "P1",
    "3": "P0"
}
```
"""

_REFERENCE_MARKER = re.compile(r"\[\d+\]")


def split_reference_list(raw_text: str) -> list[str]:
    """Split ``"[1] ... [2] ..."`` into citation strings with normalised whitespace.

    Text before the first ``[n]`` marker is not a reference (a heading or a
    stray sentence from the model) and would shift every reference number.
    """
    first = _REFERENCE_MARKER.search(raw_text)
    if first is None:
        return []
    segments = _REFERENCE_MARKER.split(raw_text[first.start() :])
    return [" ".join(segment.split()) for segment in segments if segment.strip()]


def parse_priorities(text: str) -> dict[str, str] | None:
    """Parse ``{"1": "P0", ...}``; ``None`` when no valid entry is present."""
    data = parse_json_object(text)
    if not data:
        return None
    priorities: dict[str, str] = {}
    for key, value in data.items():
        number = str(key).strip().strip("[]")
        label = str(value).strip().upper()
        if number.isdigit() and label in ("P0", "P1"):
            priorities[str(int(number))] = label
    return priorities or None


def _split_by_priority(pairs: Iterable[tuple[str, str]]) -> tuple[set[str], set[str]]:
    """``(paper id, priority)`` pairs -> ``(P0 ids, P1 ids)``; P0 wins for a repeated id."""
    pairs = list(pairs)
    p0 = {pid for pid, priority in pairs if priority == "P0"}
    p1 = {pid for pid, priority in pairs if priority != "P0"} - p0
    return p0, p1


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


class ScholarLookupError(RuntimeError):
    """Semantic Scholar rejected the request or kept failing after every retry."""


class HttpSession(Protocol):
    def get(
        self, url: str, *, headers: Mapping[str, str], params: Mapping[str, Any], timeout: float
    ) -> Any: ...


class SemanticScholarLookup:
    """Title -> paper id lookups against the Semantic Scholar Graph API.

    Share one instance between threads: requests are spaced ``min_interval``
    seconds apart across all of them, and rate-limit or server errors are
    retried with backoff a bounded number of times.  A lookup that still
    fails raises instead of reporting "not found", so an outage is never
    mistaken for a missing paper.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: HttpSession | None = None,
        timeout: float = 20.0,
        max_attempts: int = 6,
        min_interval: float = 1.0,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = api_key
        self._session: HttpSession = session or requests
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.min_interval = min_interval
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _headers(self) -> dict[str, str]:
        # Read at call time so the key can come from a .env loaded after construction.
        key = self._api_key or os.environ.get(S2_API_KEY_ENV)
        return {"X-API-KEY": key} if key else {}

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
        if slot > now:
            self._sleep(slot - now)

    def _backoff(self, failed_attempts: int, retry_after: str | None) -> float:
        delay = min(self.max_delay, self.base_delay * 2 ** (failed_attempts - 1))
        if retry_after and retry_after.strip().isdigit():
            delay = min(self.max_delay, max(delay, float(retry_after)))
        return delay

    def find(self, title: str) -> dict[str, str] | None:
        """Return ``{"fetched_paper_id", "fetched_paper_title"}`` of the best match, or ``None``."""
        if not title or not title.strip():
            return None
        params = {"query": title, "limit": 1, "fields": "paperId,title"}
        problem = ""
        for attempt in range(1, self.max_attempts + 1):
            self._wait_for_slot()
            retry_after: str | None = None
            try:
                response = self._session.get(
                    S2_SEARCH_URL, headers=self._headers(), params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return _first_match(response.json())
                    except ValueError:
                        problem = "the response was not valid JSON"
                elif status in (401, 403):
                    raise ScholarLookupError(
                        f"Semantic Scholar rejected the request (HTTP {status}). "
                        f"Check the key in {S2_API_KEY_ENV}."
                    )
                elif status == 429 or status >= 500:
                    problem = f"HTTP {status}"
                    retry_after = response.headers.get("Retry-After")
                else:
                    log.warning(
                        "Semantic Scholar answered HTTP %d for the title %r; "
                        "treating the reference as not found.",
                        status,
                        title,
                    )
                    return None
            if attempt < self.max_attempts:
                delay = self._backoff(attempt, retry_after)
                log.info(
                    "Semantic Scholar lookup failed (%s); attempt %d of %d, waiting %.1f s.",
                    problem,
                    attempt,
                    self.max_attempts,
                    delay,
                )
                self._sleep(delay)
        raise ScholarLookupError(
            f"Semantic Scholar title search failed {self.max_attempts} times in a row "
            f"(last problem: {problem})."
        )


def _first_match(payload: Any) -> dict[str, str] | None:
    items = payload.get("data") if isinstance(payload, dict) else None
    if not items or not isinstance(items[0], dict) or not items[0].get("paperId"):
        return None
    return {
        "fetched_paper_id": items[0]["paperId"],
        "fetched_paper_title": items[0].get("title") or "",
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def create_log_folder(
    prefix: str = "", date_str: str = "", log_dir: str | Path = "./logs/paper_writing"
) -> str:
    """Create and return ``<log_dir>/<date>/<prefix>_log_<timestamp>`` (date: today by default)."""
    now = datetime.now()
    name = f"log_{now:%Y_%m_%d_%H_%M_%S}"
    if prefix:
        name = f"{prefix}_{name}"
    path = Path(log_dir) / (date_str or f"{now:%Y%m%d}") / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class ReferenceF1V1Evaluator:
    """Turns reference lists into Semantic Scholar ids and scores one list against another."""

    def __init__(
        self,
        api_key: str | None = None,
        log_folder: str | None = None,
        model_name: str = DEFAULT_MODEL,
        *,
        client: LLMClient | None = None,
        scholar: SemanticScholarLookup | None = None,
        title_model_name: str | None = None,
        title_workers: int = 10,
        retry: RetryPolicy | None = None,
    ) -> None:
        """``log_folder`` (created on first use when omitted) receives the per-run reports.

        ``title_model_name`` runs the many small title-extraction calls and
        defaults to ``model_name``.  Pass a shared ``scholar`` when several
        evaluators run in parallel so they respect one rate limit.
        """
        self.model_name = model_name
        self.title_model_name = title_model_name or model_name
        self.log_folder = log_folder
        self._client = client
        self._scholar = scholar or SemanticScholarLookup(api_key)
        self._title_workers = max(1, title_workers)
        self._retry = retry

    def _extract_title(self, citation: str) -> str:
        return extract_paper_title_from_citation(
            citation, self.title_model_name, client=self._client, retry=self._retry
        )

    def references_from_pdf(self, pdf_path: str | Path) -> str:
        """The reference list of a paper as ``"[1] ... [2] ..."`` (``""`` when it has none)."""
        return get_paper_references_from_pdf(
            pdf_path, model_name=self.model_name, client=self._client, retry=self._retry
        )

    def ground_truth_from_pdf(self, pdf_path: str | Path) -> dict[str, Any]:
        """Extract, resolve and prioritise the references of an original paper."""
        references_text = self.references_from_pdf(pdf_path)
        if not split_reference_list(references_text):
            raise ValueError(
                f"No numbered references could be extracted from the original paper {pdf_path}; "
                "refusing to score against (and cache) an empty ground truth."
            )
        # The first 20 pages cover the body of a conference paper and keep the prompt bounded.
        paper_text = load_paper(pdf_path, num_pages=20)
        return self.process_text_to_data(references_text, tag="gt", paper_text=paper_text)

    def _clean_text_to_list(self, raw_text: str) -> tuple[list[str], list[str]]:
        """Parse ``"[1] ... [2] ..."`` into citation strings and their extracted titles."""
        cleaned_refs = split_reference_list(raw_text)
        if not cleaned_refs:
            return [], []
        with ThreadPoolExecutor(max_workers=self._title_workers) as pool:
            cleaned_titles = list(pool.map(self._extract_title, cleaned_refs))
        return cleaned_refs, cleaned_titles

    def _fetch_paper_id(self, paper_title: str) -> dict[str, str] | None:
        """Look a title up on Semantic Scholar; ``None`` when it has no match."""
        return self._scholar.find(paper_title)

    def _get_ids_for_ref_list(
        self, ref_list: list[str], filler: str = "-1"
    ) -> tuple[set[str], list[str], list[str]]:
        """Resolve titles to ids: ``(unique ids, id per title, matched title per title)``.

        Titles without a match get ``filler`` in both lists.
        """
        ids_set: set[str] = set()
        ids_list: list[str] = []
        titles_list: list[str] = []
        log.info("Resolving %d reference(s) on Semantic Scholar.", len(ref_list))
        for title in ref_list:
            fetched = self._fetch_paper_id(title)
            if fetched:
                ids_set.add(fetched["fetched_paper_id"])
                ids_list.append(fetched["fetched_paper_id"])
                titles_list.append(fetched["fetched_paper_title"])
            else:
                ids_list.append(filler)
                titles_list.append(filler)
        return ids_set, ids_list, titles_list

    def _get_citation_priorities(self, paper_text: str, cleaned_refs: list[str]) -> dict[str, str]:
        """Ask the model to label each reference ``P0`` or ``P1``, keyed by 1-based number."""
        references = "\n".join(f"[{i}] {ref}" for i, ref in enumerate(cleaned_refs, start=1))
        prompt = (
            f"<paper_text>\n{paper_text}\n</paper_text>\n\n"
            f"<references>\n{references}\n</references>\n\n"
            "Categorize every reference above as P0 or P1."
        )
        client = resolve_client(self.model_name, self._client)
        response = call_with_retries(
            lambda: client.generate(
                prompt,
                model=self.model_name,
                system=PRIORITY_SYSTEM_PROMPT,
                parser=parse_priorities,
                temperature=0.0,
                label="eval.citation_priorities",
                parse_retries=0,
            ),
            what="citation priority classification",
            retry=self._retry,
        )
        priorities: dict[str, str] = response.parsed
        unlabelled = sum(str(i) not in priorities for i in range(1, len(cleaned_refs) + 1))
        if unlabelled:
            log.warning(
                "The model left %d of %d reference(s) without a priority; they count as P1.",
                unlabelled,
                len(cleaned_refs),
            )
        return priorities

    def process_text_to_data(
        self, raw_text: str, tag: str = "na", paper_text: str | None = None
    ) -> dict[str, Any]:
        """Turn a ``"[1] ... [2] ..."`` reference string into a serialisable record.

        With ``paper_text`` the references are also labelled P0/P1 by the
        model; without it every reference counts as P1.
        """
        ref_list, ref_titles = self._clean_text_to_list(raw_text)
        priorities: dict[str, str] = {}
        if paper_text is not None and ref_list:
            priorities = self._get_citation_priorities(paper_text, ref_list)

        filler = f"{tag}_NA"
        ids_set, ids_list, fetched_titles = self._get_ids_for_ref_list(ref_titles, filler=filler)

        citation_info = [
            {
                "citation_text": ref_list[index],
                "extracted_paper_title": ref_titles[index],
                "fetched_paper_id": ids_list[index],
                "fetched_paper_title": fetched_titles[index],
                "priority": priorities.get(str(index + 1), "P1"),
            }
            for index in range(len(ref_list))
        ]
        p0_ids, p1_ids = _split_by_priority(
            (item["fetched_paper_id"], item["priority"])
            for item in citation_info
            if item["fetched_paper_id"] != filler
        )
        return {
            "citation_info": citation_info,
            "all_ids": sorted(ids_set),
            "all_ids_count": len(ids_set),
            "p0_ids": sorted(p0_ids),
            "p0_ids_count": len(p0_ids),
            "p1_ids": sorted(p1_ids),
            "p1_ids_count": len(p1_ids),
        }

    def evaluate_from_data(
        self, gt_data: Mapping[str, Any], gen_data: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Compute overall precision/recall/F1 and P0/P1 recall; write the reports."""
        if self.log_folder is None:
            self.log_folder = create_log_folder(prefix="citation_autorater")
        log_folder = Path(self.log_folder)
        write_json(log_folder / "gt_ref_info.json", gt_data)
        write_json(log_folder / "gen_ref_info.json", gen_data)

        gt_ids_all, gt_ids_p0, gt_ids_p1 = _ground_truth_ids(gt_data)
        gen_ids_all = {pid for pid in gen_data.get("all_ids", []) if pid and pid != "gen_NA"}

        tp_overall = len(gen_ids_all & gt_ids_all)
        tp_p0 = len(gen_ids_all & gt_ids_p0)
        tp_p1 = len(gen_ids_all & gt_ids_p1)
        precision = tp_overall / len(gen_ids_all) if gen_ids_all else 0.0
        recall = tp_overall / len(gt_ids_all) if gt_ids_all else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        metrics = {
            "overall_precision": precision,
            "overall_recall": recall,
            "overall_f1": f1,
            "p0_recall": tp_p0 / len(gt_ids_p0) if gt_ids_p0 else 0.0,
            "p1_recall": tp_p1 / len(gt_ids_p1) if gt_ids_p1 else 0.0,
            "counts": {
                "ground_truth_ids_all": len(gt_ids_all),
                "ground_truth_ids_p0": len(gt_ids_p0),
                "ground_truth_ids_p1": len(gt_ids_p1),
                "generated_ids_all": len(gen_ids_all),
                "matches_overall": tp_overall,
                "matches_p0": tp_p0,
                "matches_p1": tp_p1,
            },
        }
        metrics_path = write_json(log_folder / "metrics.json", metrics)
        log.info(
            "Citation metrics written to %s: precision %.2f, recall %.2f, F1 %.2f, "
            "P0 recall %.2f, P1 recall %.2f.",
            metrics_path,
            precision,
            recall,
            f1,
            metrics["p0_recall"],
            metrics["p1_recall"],
        )
        return metrics


def _ground_truth_ids(gt_data: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    """``(all ids, P0 ids, P1 ids)`` of a ground-truth record, old cache formats included."""
    all_ids = set(gt_data.get("all_ids", []))
    p0_ids = set(gt_data.get("p0_ids", []))
    p1_ids = set(gt_data.get("p1_ids", []))
    if p0_ids or p1_ids:
        return all_ids, p0_ids, p1_ids
    # Caches written by earlier versions only carry the per-citation list.
    for legacy_key in ("citation_info", "pairwise"):
        if legacy_key in gt_data:
            pairs = [
                (item["fetched_paper_id"], item.get("priority", "P1"))
                for item in gt_data[legacy_key]
                if item.get("fetched_paper_id") and item["fetched_paper_id"] != "gt_NA"
            ]
            p0_ids, p1_ids = _split_by_priority(pairs)
            return {pid for pid, _ in pairs}, p0_ids, p1_ids
    return all_ids, p0_ids, p1_ids


def _priority_counts(gt_data: Mapping[str, Any]) -> tuple[int, int]:
    """Number of P0 and P1 citations of a ground-truth record, old cache formats included."""
    if "p0_ids_count" not in gt_data and "citation_info" in gt_data:
        entries = gt_data["citation_info"]
    elif "pairwise" in gt_data:
        entries = gt_data["pairwise"]
    else:
        return gt_data.get("p0_ids_count", 0), gt_data.get("p1_ids_count", 0)
    p0 = sum(1 for item in entries if item.get("priority") == "P0")
    return p0, len(entries) - p0


def get_citation_f1_metrics(
    original_pdf: str,
    generated_pdf: str,
    log_folder: str | None = None,
    gt_cache_path: str | None = None,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    scholar: SemanticScholarLookup | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Compute priority-aware citation F1 of ``generated_pdf`` against ``original_pdf``.

    The processed ground truth is cached at ``gt_cache_path`` (default: next
    to the original PDF) so repeated evaluations skip its model and API calls.
    """
    for path in (original_pdf, generated_pdf):
        if not Path(path).is_file():
            raise FileNotFoundError(f"paper PDF not found: {path}")
    evaluator = ReferenceF1V1Evaluator(
        log_folder=log_folder, model_name=model_name, client=client, scholar=scholar, retry=retry
    )
    cache = Path(
        gt_cache_path or f"{os.path.splitext(original_pdf)[0]}_gt_citations_{model_name}.json"
    )
    gt_data: dict[str, Any] | None = None
    if cache.is_file():
        try:
            gt_data = load_json_object(cache) or None
        except ValueError as exc:
            log.warning("Ignoring the ground-truth cache: %s", exc)
    if gt_data is None:
        log.info("Processing the original paper %s.", original_pdf)
        gt_data = evaluator.ground_truth_from_pdf(original_pdf)
        try:
            write_json(cache, gt_data)
        except OSError as exc:
            # The cache only saves time; an unwritable dataset folder must not fail the run.
            log.warning("Could not write the ground-truth cache %s: %s", cache, exc)
    else:
        log.info("Using cached ground-truth references from %s.", cache)

    log.info("Processing the generated paper %s.", generated_pdf)
    gen_data = evaluator.process_text_to_data(
        evaluator.references_from_pdf(generated_pdf), tag="gen"
    )
    if not gen_data["citation_info"]:
        log.warning("%s has no extractable references; its citation scores are 0.", generated_pdf)
    return evaluator.evaluate_from_data(gt_data, gen_data)


def generate_gt_citations_batch(
    base_folder: str,
    sub_folder_name: str,
    progress_json_path: str | None = None,
    max_workers: int = 5,
    evaluator_log_folder: str | None = None,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    scholar: SemanticScholarLookup | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Build the ground-truth citation file of every paper folder under ``base_folder``.

    For ``<base_folder>/<paper>/`` holding a PDF and a ``sub_folder_name``
    directory, writes ``<sub_folder_name>/original_paper_gt_citations_<model>.json``
    (skipped when present), records P0/P1 counts in ``<base_folder>/info.json``
    when that file exists, and tracks finished papers in ``progress_json_path``.
    Returns the progress record.
    """
    base = Path(base_folder)
    if not base.is_dir():
        raise FileNotFoundError(f"base folder not found: {base}")
    resolve_client(model_name, client)  # fail before any work when the provider is unusable
    evaluator = ReferenceF1V1Evaluator(
        log_folder=evaluator_log_folder,
        model_name=model_name,
        client=client,
        scholar=scholar or SemanticScholarLookup(),
        retry=retry,
    )

    progress: dict[str, Any] = {"processed_count": 0, "successful_papers": {}}
    if progress_json_path:
        progress.update(load_json_object(progress_json_path))
    counts: dict[str, tuple[int, int]] = {}
    lock = threading.Lock()

    def mark_done(paper: str, output_path: Path, gt_data: Mapping[str, Any]) -> None:
        with lock:
            counts[paper] = _priority_counts(gt_data)
            progress["successful_papers"][paper] = str(output_path)
            progress["processed_count"] = len(progress["successful_papers"])
            if progress_json_path:
                write_json(progress_json_path, progress)

    def process(paper: str, paper_path: Path) -> None:
        output_path = (
            paper_path / sub_folder_name / f"original_paper_gt_citations_{model_name}.json"
        )
        gt_data = load_json_object(output_path)
        if gt_data:
            log.info("[%s] %s already exists; skipping.", paper, output_path)
            mark_done(paper, output_path, gt_data)
            return
        gt_data = evaluator.ground_truth_from_pdf(_paper_pdf(paper_path))
        write_json(output_path, gt_data)
        mark_done(paper, output_path, gt_data)

    papers = {
        entry.name: entry
        for entry in sorted(base.iterdir())
        if entry.is_dir() and (entry / sub_folder_name).is_dir()
    }
    log.info(
        "Found %d paper folder(s) with a '%s' subfolder in %s.", len(papers), sub_folder_name, base
    )
    run_batch(papers, process, max_workers=max_workers, what="ground-truth citation extraction")

    _update_info_json(base / "info.json", counts)
    return progress


def _paper_pdf(paper_path: Path) -> Path:
    """The original PDF of a dataset paper folder: ``<name>.pdf``, else the first PDF found."""
    exact = paper_path / f"{paper_path.name}.pdf"
    if exact.is_file():
        return exact
    pdfs = sorted(paper_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDF found in {paper_path}")
    log.warning("%s not found; using %s instead.", exact.name, pdfs[0].name)
    return pdfs[0]


def _update_info_json(info_path: Path, counts: Mapping[str, tuple[int, int]]) -> None:
    """Record P0/P1 citation counts per ``paper_id`` in a dataset's ``info.json`` list."""
    if not info_path.is_file() or not counts:
        return
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("Leaving %s untouched: it is not valid JSON (%s).", info_path, exc)
        return
    if not isinstance(info, list):
        log.warning("Leaving %s untouched: it is not a list of papers.", info_path)
        return
    for item in info:
        if isinstance(item, dict) and item.get("paper_id") in counts:
            item["num_gt_citations_p0"], item["num_gt_citations_p1"] = counts[item["paper_id"]]
    write_json(info_path, info)
    log.info("Updated %s with P0/P1 citation counts.", info_path)


def _original_pdf(paper_folder: Path, project_folder: str, dataset_dir: Path) -> Path:
    """The original paper of a run folder: its own copy, else the benchmark dataset's."""
    local = paper_folder / "raw_materials" / "original_paper.pdf"
    if local.is_file():
        return local
    paper_id = paper_folder.name
    for venue in _DATASET_VENUES:
        if venue in project_folder.lower():
            candidate = dataset_dir / venue / "papers" / paper_id / f"{paper_id}.pdf"
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"original paper found neither at {local} nor at {candidate}")
    raise FileNotFoundError(
        f"original paper not found at {local}, and the project folder name does not name a "
        f"dataset ({', '.join(_DATASET_VENUES)}) to look it up in"
    )


def review_citation_f1_batch(
    project_folder: str,
    output_folder: str | None = None,
    gen_paper_rel_path: str = "final_paper.pdf",
    max_workers: int = 5,
    model_name: str = DEFAULT_MODEL,
    *,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    client: LLMClient | None = None,
    scholar: SemanticScholarLookup | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Score every generated paper under ``project_folder`` and aggregate the metrics.

    Results go to ``citation_f1_results_<model>.json`` in ``output_folder``
    (default: ``project_folder``); papers already in that file are skipped.
    Returns the content of the results file, including ``overall_metrics``.
    """
    project = Path(project_folder)
    if not project.is_dir():
        raise FileNotFoundError(f"project folder not found: {project}")
    resolve_client(model_name, client)  # fail before any work when the provider is unusable
    scholar = scholar or SemanticScholarLookup()
    results_path = Path(output_folder or project) / f"citation_f1_results_{model_name}.json"
    results = JsonResultsFile(results_path)

    papers = {
        entry.name: entry
        for entry in sorted(project.iterdir())
        if entry.is_dir() and (entry / gen_paper_rel_path).is_file() and entry.name not in results
    }
    if not papers:
        log.warning("No unscored paper with %s found under %s.", gen_paper_rel_path, project)

    def process(paper_id: str, paper_folder: Path) -> None:
        original_pdf = _original_pdf(paper_folder, project_folder, Path(dataset_dir))
        generated_pdf = paper_folder / gen_paper_rel_path
        log_folder = paper_folder / "citation_f1_metrics"
        log_folder.mkdir(parents=True, exist_ok=True)
        gt_cache = paper_folder / "raw_materials" / f"original_paper_gt_citations_{model_name}.json"
        metrics = get_citation_f1_metrics(
            original_pdf=str(original_pdf),
            generated_pdf=str(generated_pdf),
            log_folder=str(log_folder),
            gt_cache_path=str(gt_cache),
            model_name=model_name,
            client=client,
            scholar=scholar,
            retry=retry,
        )
        results.set(
            paper_id,
            {
                "paper_id": paper_id,
                "real_pdf_path": str(original_pdf),
                "generated_pdf_path": str(generated_pdf),
                "metrics": metrics,
            },
        )

    run_batch(papers, process, max_workers=max_workers, what="citation F1")
    analyze_citation_metrics(str(results_path))
    return load_json_object(results_path)


def analyze_citation_metrics(json_file_path: str) -> dict[str, Any] | None:
    """Average the per-paper metrics of a results file into its ``overall_metrics`` entry.

    Returns the averages, or ``None`` when the file holds no scored paper.
    """
    path = Path(json_file_path)
    data = load_json_object(path)
    scored = [
        entry["metrics"]
        for paper_id, entry in data.items()
        if paper_id != OVERALL_KEY and isinstance(entry, dict) and entry.get("metrics")
    ]
    if not scored:
        log.warning("No paper with citation metrics in %s.", path)
        return None

    overall: dict[str, Any] = {
        key: sum(metrics.get(key, 0.0) for metrics in scored) / len(scored) for key in SCORE_KEYS
    }
    overall["counts"] = {
        key: sum(metrics.get("counts", {}).get(key, 0) for metrics in scored) / len(scored)
        for key in COUNT_KEYS
    }
    data[OVERALL_KEY] = overall
    write_json(path, data)

    log.info("Citation metrics averaged over %d paper(s):", len(scored))
    for key in SCORE_KEYS:
        log.info("  %-24s %.4f", key, overall[key])
    for key in COUNT_KEYS:
        log.info("  %-24s %.2f", key, overall["counts"][key])
    return overall
