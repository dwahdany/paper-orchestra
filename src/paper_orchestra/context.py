"""Per-run context shared by every agent.

Agents receive one :class:`RunContext` instead of a dozen path arguments.  It
carries the configuration, the on-disk layout of a run, the memory harness,
the compute backend and the usage ledger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from paper_orchestra.config import PipelineConfig
from paper_orchestra.llm.base import LLMClient, UsageTracker

if TYPE_CHECKING:
    from paper_orchestra.compute.base import ComputeBackend
    from paper_orchestra.memory.harness import ContextHarness

log = logging.getLogger("paper_orchestra")


@dataclass(frozen=True)
class RunPaths:
    """Where a run reads its inputs and writes its outputs.

    The output names are a public contract: the Streamlit frontend, the
    autoraters and existing datasets all rely on them.
    """

    base_dir: Path
    idea_filename: str = "idea_sparse.md"
    experimental_log_filename: str = "experimental_log.md"

    # -- inputs --
    @property
    def materials_dir(self) -> Path:
        return self.base_dir / "raw_materials"

    @property
    def idea(self) -> Path:
        return self.materials_dir / self.idea_filename

    @property
    def experimental_log(self) -> Path:
        return self.materials_dir / self.experimental_log_filename

    @property
    def provided_figures_dir(self) -> Path:
        return self.materials_dir / "figures"

    # -- LaTeX workspace --
    @property
    def latex_dir(self) -> Path:
        return self.base_dir / "latex_writeup"

    @property
    def template_tex(self) -> Path:
        return self.latex_dir / "template.tex"

    @property
    def guidelines(self) -> Path:
        return self.latex_dir / "guidelines.md"

    @property
    def references_bib(self) -> Path:
        return self.latex_dir / "references.bib"

    @property
    def figures_dir(self) -> Path:
        return self.latex_dir / "figures"

    @property
    def figures_info(self) -> Path:
        return self.figures_dir / "info.json"

    @property
    def raw_draft_tex(self) -> Path:
        return self.latex_dir / "raw_draft_paper.tex"

    # -- stage outputs --
    @property
    def outline(self) -> Path:
        return self.base_dir / "outline.json"

    @property
    def literature_dir(self) -> Path:
        return self.base_dir / "literature_agent_output"

    @property
    def citation_map(self) -> Path:
        return self.literature_dir / "citation_map.json"

    @property
    def plotting_results(self) -> Path:
        return self.base_dir / "plotting_results.json"

    @property
    def generated_figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @property
    def refinement_dir(self) -> Path:
        return self.base_dir / "content_refinement_workdir"

    @property
    def final_pdf(self) -> Path:
        return self.base_dir / "final_paper.pdf"

    @property
    def final_tex(self) -> Path:
        return self.base_dir / "final_paper.tex"

    # -- bookkeeping --
    @property
    def state(self) -> Path:
        return self.base_dir / "state.json"

    @property
    def usage(self) -> Path:
        return self.base_dir / "usage.json"

    @property
    def time_breakdown(self) -> Path:
        return self.base_dir / "time_breakdown.json"

    @property
    def memory_dir(self) -> Path:
        return self.base_dir / "memory"


@dataclass
class RunContext:
    cfg: PipelineConfig
    paths: RunPaths
    harness: ContextHarness
    compute: ComputeBackend
    usage: UsageTracker = field(default_factory=UsageTracker)

    def client(self, model: str) -> LLMClient:
        """The provider client for ``model`` (shared, cached by the harness)."""
        return self.harness.client_for(model)
