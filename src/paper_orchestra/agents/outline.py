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

"""The outline agent: idea + experimental log + template -> ``outline.json``.

The outline is the plan every later stage follows: which figures to draw,
what literature to search for, and what each section has to say.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from paper_orchestra.llm.base import LLMParseError
from paper_orchestra.memory.wiki import WikiError
from paper_orchestra.prompts.outline import (
    ASPECT_RATIOS,
    OUTLINE_SYSTEM_PROMPT,
    outline_user_prompt,
)
from paper_orchestra.utils.common import read_text, write_json
from paper_orchestra.utils.scholar import parse_cutoff

if TYPE_CHECKING:
    from paper_orchestra.context import RunContext
    from paper_orchestra.memory.harness import ContextHarness

log = logging.getLogger(__name__)

AGENT = "OutlineAgent"
STAGE = "outline"
OUTLINE_PAGE = "wiki/paper/outline.md"

DEFAULT_ASPECT_RATIO = "16:9"
DATA_SOURCES = ("idea.md", "experimental_log.md", "both")

_RATIO = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[:/xX×]\s*(\d+(?:\.\d+)?)\s*$")
_PLOT_WORDS = ("plot", "chart", "graph", "curve", "histogram", "heatmap")


# ---------------------------------------------------------------------------
# Schema of outline.json
# ---------------------------------------------------------------------------


class _Part(BaseModel):
    # Keys the schema does not name are dropped instead of failing the whole outline.
    model_config = ConfigDict(extra="ignore")


class PlotPlan(_Part):
    figure_id: str = Field(
        description="Identifier summarizing the figure, for example fig_framework_overview."
    )
    title: str = ""
    plot_type: str = Field(description='Exactly "plot" or "diagram".')
    data_source: str = Field(
        default="both", description='Exactly "idea.md", "experimental_log.md" or "both".'
    )
    objective: str = Field(
        description="What the figure has to show. For a plot, name the chart type here."
    )
    aspect_ratio: str = Field(
        default=DEFAULT_ASPECT_RATIO, description="One of: " + ", ".join(ASPECT_RATIOS) + "."
    )


class IntroductionStrategy(_Part):
    hook_hypothesis: str
    problem_gap_hypothesis: str
    search_directions: list[str] = Field(default_factory=list)


class RelatedWorkSubsection(_Part):
    subsection_title: str
    methodology_cluster: str
    sota_investigation_mission: str
    limitation_hypothesis: str
    limitation_search_queries: list[str] = Field(default_factory=list)
    bridge_to_our_method: str


class RelatedWorkStrategy(_Part):
    overview: str = ""
    subsections: list[RelatedWorkSubsection] = Field(default_factory=list)


class IntroRelatedWorkPlan(_Part):
    introduction_strategy: IntroductionStrategy
    related_work_strategy: RelatedWorkStrategy


class SubsectionPlan(_Part):
    subsection_title: str
    content_bullets: list[str] = Field(default_factory=list)
    citation_hints: list[str] = Field(default_factory=list)


class SectionPlan(_Part):
    section_title: str
    subsections: list[SubsectionPlan] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _fold_section_level_content(cls, data: Any) -> Any:
        """Keep bullets and hints a model attached to the section itself.

        Later stages only read subsections, so content given at section level
        becomes a leading subsection named after the section instead of being
        dropped as an unknown key.
        """
        if not isinstance(data, Mapping):
            return data
        bullets = data.get("content_bullets") or []
        hints = data.get("citation_hints") or []
        subsections = data.get("subsections") or []
        # Anything but a list of subsections is left for validation to reject.
        if (not bullets and not hints) or not isinstance(subsections, list):
            return data
        own = {
            "subsection_title": data.get("section_title", ""),
            "content_bullets": bullets,
            "citation_hints": hints,
        }
        return {**data, "subsections": [own, *subsections]}


class Outline(_Part):
    """``outline.json``.  The key names are read by every later stage and by the frontend."""

    plotting_plan: list[PlotPlan] = Field(default_factory=list)
    intro_related_work_plan: IntroRelatedWorkPlan
    section_plan: list[SectionPlan]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_plot_type(value: str) -> str:
    """Map a free-form figure type onto ``plot`` (data chart) or ``diagram``."""
    kind = value.strip().lower()
    if kind in ("plot", "diagram"):
        return kind
    if "diagram" not in kind and any(word in kind for word in _PLOT_WORDS):
        return "plot"
    return "diagram"


def _ratio_value(ratio: str) -> float:
    width, height = ratio.split(":")
    return int(width) / int(height)


def normalize_aspect_ratio(value: str) -> str:
    """The allowed aspect ratio closest to ``value`` (16:9 when it is not a ratio)."""
    match = _RATIO.match(value)
    if not match:
        return DEFAULT_ASPECT_RATIO
    width, height = float(match.group(1)), float(match.group(2))
    if width <= 0 or height <= 0:
        return DEFAULT_ASPECT_RATIO
    wanted = math.log(width / height)
    return min(ASPECT_RATIOS, key=lambda ratio: abs(math.log(_ratio_value(ratio)) - wanted))


def normalize_data_source(value: str) -> str:
    source = value.strip().lower()
    from_idea = "idea" in source
    from_log = "experiment" in source or "log" in source
    if from_idea and not from_log:
        return "idea.md"
    if from_log and not from_idea:
        return "experimental_log.md"
    return "both"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_outline(outline: dict[str, Any]) -> dict[str, Any]:
    """Bring the plotting plan into the shape the plotting agent relies on (in place).

    ``figure_id`` becomes a unique file-name-safe slug, and ``plot_type``,
    ``data_source`` and ``aspect_ratio`` are mapped onto their allowed values.
    """
    used: set[str] = set()
    for index, figure in enumerate(outline.get("plotting_plan", []), start=1):
        requested_type = str(figure.get("plot_type", "")).strip()
        figure["plot_type"] = normalize_plot_type(requested_type)
        objective = str(figure.get("objective", ""))
        if requested_type.lower() not in ("", "plot", "diagram") and (
            requested_type.lower() not in objective.lower()
        ):
            # The chart type belongs in the objective; keep it when the model put it here.
            figure["objective"] = f"{objective} (requested as: {requested_type})".strip()
        figure["data_source"] = normalize_data_source(str(figure.get("data_source", "")))
        figure["aspect_ratio"] = normalize_aspect_ratio(str(figure.get("aspect_ratio", "")))

        base = _slug(str(figure.get("figure_id", ""))) or _slug(str(figure.get("title", "")))
        base = base or f"fig_{index}"
        figure_id, suffix = base, 2
        while figure_id in used:
            figure_id, suffix = f"{base}_{suffix}", suffix + 1
        used.add(figure_id)
        figure["figure_id"] = figure_id
    return outline


# ---------------------------------------------------------------------------
# Wiki page
# ---------------------------------------------------------------------------


def write_outline_page(
    harness: ContextHarness, outline: Mapping[str, Any], *, summary: str, notes: str = ""
) -> None:
    """Write ``wiki/paper/outline.md``: the outline as fenced JSON, preceded by ``notes``."""
    parts = ["# Paper outline", "", summary, ""]
    if notes:
        parts += [notes.strip(), ""]
    parts += [
        "The plan below is what every later stage follows; `outline.json` holds the same data.",
        "",
        "```json",
        json.dumps(outline, indent=2, ensure_ascii=False),
        "```",
    ]
    meta = {"title": "Paper outline", "summary": summary, "kind": "outline"}
    try:
        harness.write_page(OUTLINE_PAGE, "\n".join(parts), meta)
    except WikiError as exc:
        # Memory is an aid; the outline on disk is what the pipeline depends on.
        log.warning("Could not write %s to the wiki: %s", OUTLINE_PAGE, exc)


def count_citation_hints(outline: Mapping[str, Any]) -> int:
    return sum(
        len(subsection.get("citation_hints", []))
        for section in outline.get("section_plan", [])
        for subsection in section.get("subsections", [])
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class OutlineAgent:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def run(self) -> dict[str, Any]:
        """Generate, validate and normalize the outline; write ``outline.json``."""
        cfg, paths = self.ctx.cfg, self.ctx.paths
        parse_cutoff(cfg.research_cutoff)  # a malformed cutoff should fail before any model call
        prompt = outline_user_prompt(
            idea=read_text(paths.idea),
            experimental_log=read_text(paths.experimental_log),
            template=read_text(paths.template_tex),
            guidelines=read_text(paths.guidelines),
            cutoff=cfg.research_cutoff,
        )
        response = self.ctx.harness.generate(
            prompt,
            agent=AGENT,
            stage=STAGE,
            model=cfg.models.writer,
            system=OUTLINE_SYSTEM_PROMPT,
            schema=Outline,
        )
        parsed = response.parsed
        if not isinstance(parsed, Outline):
            try:
                parsed = Outline.model_validate(parsed)
            except ValidationError as exc:
                raise LLMParseError(f"the outline does not match its schema: {exc}") from exc
        outline = normalize_outline(parsed.model_dump())
        if not outline["section_plan"]:
            raise RuntimeError("the outline has an empty section_plan; nothing could be written")

        write_json(paths.outline, outline)
        figures, sections = len(outline["plotting_plan"]), len(outline["section_plan"])
        hints = count_citation_hints(outline)
        write_outline_page(
            self.ctx.harness,
            outline,
            summary=(
                f"Outline of the paper: {figures} planned figure(s), {sections} section(s), "
                f"{hints} citation hint(s) still to be resolved by the literature stage."
            ),
        )
        self.ctx.harness.record(
            "outline", stage=STAGE, agent=AGENT, figures=figures, sections=sections, hints=hints
        )
        log.info(
            "Outline written to %s (%d figures, %d sections, %d citation hints)",
            paths.outline,
            figures,
            sections,
            hints,
        )
        return outline
