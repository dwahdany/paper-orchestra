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

"""Side-by-side holistic comparison of two papers (science, writing and presentation).

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
    SXS_PAPER_QUALITY_TASK,
    sxs_paper_quality_system_prompt,
)
from paper_orchestra.llm.base import Image, LLMClient, Part, Pdf, Text
from paper_orchestra.utils.pdf import load_paper, pdf_page_images

#: Page images sent per paper to providers that cannot read a PDF directly.
MAX_PAGE_IMAGES = 15


class SxSPaperQualityVerdict(BaseModel):
    paper_1_holistic_analysis: str = Field(
        description="Analysis of paper_1 writing, presentation, and scientific execution."
    )
    paper_2_holistic_analysis: str = Field(
        description="Analysis of paper_2 writing, presentation, and scientific execution."
    )
    comparison_justification: str = Field(description="Comparison reasoning.")
    winner: Winner


def _paper_parts(client: LLMClient, pdf_path: Path, number: int) -> list[Part]:
    """One paper as the client can best take it: the PDF itself, else its text and page images."""
    if client.capabilities.pdf_input:
        # The title labels the document itself (a provider may move documents ahead of
        # the text) and keeps the file name, which can reveal the paper's origin, out.
        return [
            Text(f"Paper {number}:"),
            Pdf(data=pdf_path.read_bytes(), title=f"Paper {number}"),
        ]
    text = load_paper(pdf_path, min_size=100)
    parts: list[Part] = [Text(f"<paper_{number}>\n{text}\n</paper_{number}>")]
    if client.capabilities.image_input:
        pages = pdf_page_images(pdf_path, scale=1.0, max_pages=MAX_PAGE_IMAGES, image_format="jpeg")
        parts.append(Text(f"Paper {number} visual pages:"))
        parts.extend(Image(data=page, media_type="image/jpeg") for page in pages)
    return parts


def rate_sxs_paper_quality(
    paper1_path: str,
    paper2_path: str,
    model_name: str = DEFAULT_MODEL,
    *,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Compare two paper PDFs holistically; returns the verdict as a plain dict.

    ``winner`` is ``"paper_1"``, ``"paper_2"`` or ``"tie"``.  Raises
    ``FileNotFoundError`` for a missing PDF and the last ``LLMError`` when the
    model call still fails after the retries.
    """
    paths = [Path(paper1_path), Path(paper2_path)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"paper PDF not found: {path}")
    llm = resolve_client(model_name, client)
    parts: list[Part] = [
        *_paper_parts(llm, paths[0], 1),
        *_paper_parts(llm, paths[1], 2),
        Text(SXS_PAPER_QUALITY_TASK),
    ]
    verdict = call_with_retries(
        lambda: generate_structured(
            llm,
            parts,
            model=model_name,
            schema=SxSPaperQualityVerdict,
            system=sxs_paper_quality_system_prompt,
            temperature=0.0,
            label="eval.sxs_paper_quality",
        ),
        what=f"SxS paper-quality comparison of {paths[0]} and {paths[1]}",
        retry=retry,
    )
    return verdict.model_dump()


def perform_sxs_paper_quality_batch(
    project_folder_1: str,
    project_folder_2: str,
    rel_path_1: str,
    rel_path_2: str,
    exp1_name: str,
    exp2_name: str,
    output_folder: str = "./logs/sxs_paper_quality",
    model_name: str = DEFAULT_MODEL,
    *,
    max_workers: int = 3,
    client: LLMClient | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Compare ``<project_folder_1>/<id>/<rel_path_1>`` with its counterpart for every shared id.

    Results (one entry per paper plus ``_summary_stats``) are written to
    ``<output_folder>/sxs_paper_quality_enhanced_<model>_<exp1>_vs_<exp2>_<venue>.json``
    and returned.  Pairs already in that file are skipped; a pair that fails is
    logged and left out, so a later run tries it again.
    """
    llm = resolve_client(model_name, client)

    def rate_both_orders(paper1_path: str, paper2_path: str) -> tuple[dict, dict]:
        return (
            rate_sxs_paper_quality(paper1_path, paper2_path, model_name, client=llm, retry=retry),
            rate_sxs_paper_quality(paper2_path, paper1_path, model_name, client=llm, retry=retry),
        )

    output_json_path = Path(output_folder) / (
        f"sxs_paper_quality_enhanced_{model_name}_{exp1_name}_vs_{exp2_name}"
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
        what="SxS paper quality",
    )
