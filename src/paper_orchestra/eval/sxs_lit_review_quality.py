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

"""Side-by-side comparison of the literature reviews of two papers.

Each pair is judged twice, once per presentation order, so a preference for
the first or second position cancels out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.eval import (
    RetryPolicy,
    Winner,
    call_with_retries,
    conference_tag,
    generate_structured,
    resolve_client,
    run_sxs_batch,
)
from paper_orchestra.eval.prompts.sxs_quality_prompts import (
    sxs_lit_review_quality_system_prompt,
    sxs_lit_review_user_prompt,
)
from paper_orchestra.llm.base import LLMClient
from paper_orchestra.utils.pdf import load_paper


class SxSLitReviewVerdict(BaseModel):
    paper_1_analysis: str = Field(description="Analysis of paper 1.")
    paper_2_analysis: str = Field(description="Analysis of paper 2.")
    comparison_justification: str = Field(description="Comparison reasoning.")
    winner: Winner


def rate_sxs_lit_review(
    paper1_text: str,
    paper2_text: str,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Compare two papers' literature reviews; returns the verdict as a plain dict.

    ``winner`` is ``"paper_1"``, ``"paper_2"`` or ``"tie"``.  Raises the last
    ``LLMError`` when the model call still fails after the retries.
    """
    llm = resolve_client(model_name, client)
    prompt = sxs_lit_review_user_prompt(paper1_text, paper2_text)
    verdict = call_with_retries(
        lambda: generate_structured(
            llm,
            prompt,
            model=model_name,
            schema=SxSLitReviewVerdict,
            system=sxs_lit_review_quality_system_prompt,
            temperature=0.0,
            label="eval.sxs_lit_review",
        ),
        what="SxS literature-review comparison",
        retry=retry,
    )
    return verdict.model_dump()


def perform_sxs_lit_review_batch(
    project_folder_1: str,
    project_folder_2: str,
    rel_path_1: str,
    rel_path_2: str,
    exp1_name: str,
    exp2_name: str,
    output_folder: str = "./logs/sxs_lit_review_quality",
    model_name: str = DEFAULT_MODEL,
    *,
    max_workers: int = 3,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Compare ``<project_folder_1>/<id>/<rel_path_1>`` with its counterpart for every shared id.

    Results (one entry per paper plus ``_summary_stats``) are written to
    ``<output_folder>/sxs_lit_review_enhanced_<model>_<exp1>_vs_<exp2>_<venue>.json``
    and returned.  Pairs already in that file are skipped; a pair that fails is
    logged and left out, so a later run tries it again.
    """
    llm = resolve_client(model_name, client)

    def rate_both_orders(paper1_path: str, paper2_path: str) -> tuple[dict, dict]:
        paper1_text = load_paper(paper1_path, min_size=100)
        paper2_text = load_paper(paper2_path, min_size=100)
        return (
            rate_sxs_lit_review(paper1_text, paper2_text, model_name, client=llm, retry=retry),
            rate_sxs_lit_review(paper2_text, paper1_text, model_name, client=llm, retry=retry),
        )

    output_json_path = Path(output_folder) / (
        f"sxs_lit_review_enhanced_{model_name}_{exp1_name}_vs_{exp2_name}"
        f"_{conference_tag(project_folder_1)}.json"
    )
    return run_sxs_batch(
        rate_both_orders,
        project_folder_1=project_folder_1,
        project_folder_2=project_folder_2,
        rel_path_1=rel_path_1,
        rel_path_2=rel_path_2,
        exp1_name=exp1_name,
        exp2_name=exp2_name,
        output_json_path=output_json_path,
        max_workers=max_workers,
        what="SxS literature review",
    )
