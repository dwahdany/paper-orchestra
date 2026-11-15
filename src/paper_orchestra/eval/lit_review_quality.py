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

"""Literature-review quality: a rubric-based 0-100 rating of a paper's Introduction and
Related Work sections."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.eval import (
    JsonResultsFile,
    RetryPolicy,
    call_with_retries,
    generate_structured,
    load_json_object,
    resolve_client,
    run_batch,
)
from paper_orchestra.eval.prompts.lit_review_quality_prompts import (
    lit_review_quality_system_prompt,
    lit_review_quality_user_prompt,
)
from paper_orchestra.llm.base import LLMClient, Part, Pdf
from paper_orchestra.utils.common import read_json, write_json
from paper_orchestra.utils.pdf import load_paper

log = logging.getLogger(__name__)

OVERALL_KEY = "overall_metrics"
#: Top-level axis keys of ratings written by the first version of this autorater.
_LEGACY_AXES = ("clarity", "technical_depth", "connectivity", "presentation")


class AxisScore(BaseModel):
    score: float = Field(description="Score from 0 to 100 on the rubric.")
    justification: str = Field(description="Evidence-based justification, 2-5 sentences.")


class CitationStatistics(BaseModel):
    estimated_unique_citations: int
    citation_density_assessment: Literal["low", "appropriate", "high"]
    breadth_across_subareas: Literal["narrow", "moderate", "broad"]
    comparison_to_baseline: str
    notes: str

    @field_validator("citation_density_assessment", "breadth_across_subareas", mode="before")
    @classmethod
    def _lowercase(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class AxisScores(BaseModel):
    coverage_and_completeness: AxisScore
    relevance_and_focus: AxisScore
    critical_analysis_and_synthesis: AxisScore
    positioning_and_novelty: AxisScore
    organization_and_writing: AxisScore
    citation_practices_and_rigor: AxisScore


class Penalty(BaseModel):
    reason: str
    points_deducted: float


class ReviewSummary(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    top_improvements: list[str]


class LitReviewRating(BaseModel):
    """The rating the rubric in ``lit_review_quality_system_prompt`` asks for."""

    paper_title: str | None
    citation_statistics: CitationStatistics
    axis_scores: AxisScores
    penalties: list[Penalty]
    summary: ReviewSummary
    overall_score: float = Field(description="Final score from 0 to 100 after penalties.")


def _paper_parts(client: LLMClient, pdf_path: Path) -> list[Part]:
    if client.capabilities.pdf_input:
        # A neutral title: the file name could tell the rater which system wrote the paper.
        return [
            Pdf(data=pdf_path.read_bytes(), title="Paper under review"),
            lit_review_quality_user_prompt(),
        ]
    return [lit_review_quality_user_prompt(load_paper(pdf_path, min_size=100))]


def rate_single_paper(
    paper_pdf_path: str,
    model_name: str = DEFAULT_MODEL,
    avg_citation_count: int = 58,
    *,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Rate the literature review of one paper; returns the rating as a plain dict.

    Raises ``FileNotFoundError`` for a missing PDF and the last ``LLMError``
    when the model call still fails after the retries.
    """
    path = Path(paper_pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"paper PDF not found: {path}")
    llm = resolve_client(model_name, client)
    parts = _paper_parts(llm, path)
    system = lit_review_quality_system_prompt.format(avg_citation_count=avg_citation_count)
    rating = call_with_retries(
        lambda: generate_structured(
            llm,
            parts,
            model=model_name,
            schema=LitReviewRating,
            system=system,
            temperature=0.0,
            label="eval.lit_review_quality",
        ),
        what=f"literature-review rating of {path}",
        retry=retry,
    )
    return rating.model_dump()


def perform_batch_rating(
    paper_path_dict: dict[str, str],
    output_json_path: str,
    max_workers: int = 10,
    avg_citation_count: int = 58,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Rate ``{paper_id: pdf_path}`` in parallel, saving each rating as soon as it is ready.

    Papers already present in ``output_json_path`` are skipped.  A paper whose
    rating fails is logged and left out, so a later run tries it again.
    Returns the content of the results file.
    """
    results = JsonResultsFile(output_json_path)
    pending = {pid: path for pid, path in paper_path_dict.items() if pid not in results}
    if not pending:
        log.info("All %d paper(s) are already rated in %s.", len(paper_path_dict), output_json_path)
        return results.snapshot()
    llm = resolve_client(model_name, client)

    def rate(paper_id: str, paper_path: str) -> None:
        rating = rate_single_paper(
            paper_path,
            model_name=model_name,
            avg_citation_count=avg_citation_count,
            client=llm,
            retry=retry,
        )
        rating["pdf_path"] = paper_path
        results.set(paper_id, rating)

    log.info("Rating %d paper(s) with %d worker(s).", len(pending), max_workers)
    run_batch(pending, rate, max_workers=max_workers, what="literature-review rating")
    return results.snapshot()


def perform_batch_rating_folder(
    root_folder: str,
    paper_rel_path: str = "final_paper.pdf",
    avg_citation_count: int = 58,
    model_name: str = DEFAULT_MODEL,
    *,
    max_workers: int = 10,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Rate ``<root_folder>/<paper_id>/<paper_rel_path>`` for every paper folder.

    Results are written to
    ``<root_folder>/lit_review_quality_ratings_<model>_<pdf name>.json``.
    """
    root = Path(root_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"root folder not found: {root}")
    pdf_name = paper_rel_path.split("/")[-1].replace(".pdf", "")
    output_json_path = root / f"lit_review_quality_ratings_{model_name}_{pdf_name}.json"

    paper_path_dict = {
        entry.name.strip(): str(entry / paper_rel_path)
        for entry in sorted(root.iterdir())
        if entry.is_dir() and (entry / paper_rel_path).is_file()
    }
    if not paper_path_dict:
        log.warning("No paper folder under %s contains %s.", root, paper_rel_path)

    ratings = perform_batch_rating(
        paper_path_dict=paper_path_dict,
        output_json_path=str(output_json_path),
        max_workers=max_workers,
        avg_citation_count=avg_citation_count,
        model_name=model_name,
        client=client,
        retry=retry,
    )
    if ratings:
        print_literature_review_stats(str(output_json_path))
        ratings = load_json_object(output_json_path)
    return ratings


def perform_batch_rating_json(
    paper_info_json_path: str,
    avg_citation_count: int = 58,
    model_name: str = DEFAULT_MODEL,
    *,
    max_workers: int = 10,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Rate the papers listed in a JSON file of ``{"paper_id", "pdf_path"}`` records.

    Results are written next to it as ``<name>_lit_review_quality_ratings.json``.
    """
    info_path = Path(paper_info_json_path)
    paper_path_dict: dict[str, str] = {}
    for paper_info in read_json(info_path):
        if Path(paper_info["pdf_path"]).is_file():
            paper_path_dict[paper_info["paper_id"]] = paper_info["pdf_path"]
        else:
            log.warning(
                "Skipping %s: %s does not exist.", paper_info["paper_id"], paper_info["pdf_path"]
            )

    output_json_path = info_path.with_name(f"{info_path.stem}_lit_review_quality_ratings.json")
    ratings = perform_batch_rating(
        paper_path_dict=paper_path_dict,
        output_json_path=str(output_json_path),
        max_workers=max_workers,
        avg_citation_count=avg_citation_count,
        model_name=model_name,
        client=client,
        retry=retry,
    )
    if ratings:
        print_literature_review_stats(str(output_json_path))
        ratings = load_json_object(output_json_path)
    return ratings


def print_literature_review_stats(file_path: str) -> dict[str, float] | None:
    """Average the overall and per-axis scores of a ratings file.

    The averages are logged and stored in the file under ``overall_metrics``;
    they are also returned (``None`` when the file holds no rating).  Both the
    current rating format (``overall_score`` + ``axis_scores``) and the older
    one (``overall_quality_score`` + top-level axes) are understood.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"ratings file not found: {path}")
    data = load_json_object(path)

    scores: dict[str, list[float]] = {"overall_score": []}
    for paper_id, rating in data.items():
        if paper_id == OVERALL_KEY or not isinstance(rating, dict):
            continue
        overall = rating.get("overall_score", rating.get("overall_quality_score"))
        if isinstance(overall, (int, float)):
            scores["overall_score"].append(overall)
        axes = rating.get("axis_scores") or {name: rating.get(name) for name in _LEGACY_AXES}
        for name, details in axes.items():
            if isinstance(details, dict) and isinstance(details.get("score"), (int, float)):
                scores.setdefault(name, []).append(details["score"])

    papers = len(scores["overall_score"])
    if not papers:
        log.warning("No rated paper in %s.", path)
        return None

    axes_sorted = sorted(name for name in scores if name != "overall_score")
    overall_metrics = {
        name: sum(scores[name]) / len(scores[name]) for name in ("overall_score", *axes_sorted)
    }
    data[OVERALL_KEY] = overall_metrics
    write_json(path, data)

    log.info("Literature review evaluation over %d paper(s):", papers)
    for name, value in overall_metrics.items():
        log.info("  %-36s %.2f", name.replace("_", " ").title(), value)
    return overall_metrics
