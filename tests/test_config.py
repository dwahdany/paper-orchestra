from __future__ import annotations

from pathlib import Path

import pytest

from paper_orchestra.config import (
    DEFAULT_MODEL,
    MemoryConfig,
    ModelConfig,
    PipelineConfig,
    default_lab_memory_dir,
    resolve_template_dir,
)
from paper_orchestra.context import RunPaths


def test_defaults_use_fable_everywhere():
    models = ModelConfig()
    assert DEFAULT_MODEL == "claude-fable-5-2"
    assert {models.writer, models.reflection, models.plotting, models.fast} == {DEFAULT_MODEL}
    assert models.image is None and models.effort == "high" and models.fast_effort == "low"


def test_config_round_trips_through_plain_dict(tmp_path):
    cfg = PipelineConfig(
        base_dir=tmp_path / "run",
        template_dir=tmp_path / "tpl",
        use_plotting=True,
        models=ModelConfig(writer="gemini-x", image="gemini-img"),
        memory=MemoryConfig(lab=False, project_dir=tmp_path / "mem", dream="end"),
        research_cutoff="2024-11",
    )
    data = cfg.to_dict()
    assert isinstance(data["base_dir"], str) and data["memory"]["project_dir"] == str(
        tmp_path / "mem"
    )
    restored = PipelineConfig.from_dict(data)
    assert restored == cfg
    assert restored.project_memory_dir == tmp_path / "mem"


def test_memory_dirs(tmp_path, monkeypatch):
    cfg = PipelineConfig(base_dir=tmp_path, template_dir=tmp_path)
    assert cfg.project_memory_dir == tmp_path / "memory"
    assert cfg.memory.resolved_lab_dir() == Path.home() / ".paper_orchestra" / "memory"
    monkeypatch.setenv("PAPER_ORCHESTRA_MEMORY_DIR", str(tmp_path / "lab"))
    assert default_lab_memory_dir() == tmp_path / "lab"
    assert MemoryConfig(lab=False).resolved_lab_dir() is None
    assert MemoryConfig(enabled=False).resolved_lab_dir() is None


def test_resolve_template_dir(tmp_path, monkeypatch, repo_root):
    assert resolve_template_dir("cvpr2025") == repo_root / "templates" / "cvpr2025"
    custom = tmp_path / "mine"
    custom.mkdir()
    assert resolve_template_dir(custom) == custom
    monkeypatch.setenv("PAPER_ORCHESTRA_TEMPLATES", str(tmp_path))
    assert resolve_template_dir("mine") == custom
    with pytest.raises(FileNotFoundError, match="nope"):
        resolve_template_dir("nope")


def test_run_paths_keep_the_public_output_names(tmp_path):
    paths = RunPaths(tmp_path, idea_filename="idea_dense.md")
    rel = lambda p: p.relative_to(tmp_path).as_posix()  # noqa: E731
    assert rel(paths.idea) == "raw_materials/idea_dense.md"
    assert rel(paths.experimental_log) == "raw_materials/experimental_log.md"
    assert rel(paths.outline) == "outline.json"
    assert rel(paths.citation_map) == "literature_agent_output/citation_map.json"
    assert rel(paths.plotting_results) == "plotting_results.json"
    assert rel(paths.raw_draft_tex) == "latex_writeup/raw_draft_paper.tex"
    assert rel(paths.figures_info) == "latex_writeup/figures/info.json"
    assert rel(paths.refinement_dir) == "content_refinement_workdir"
    assert rel(paths.final_pdf) == "final_paper.pdf"
    assert rel(paths.time_breakdown) == "time_breakdown.json"
