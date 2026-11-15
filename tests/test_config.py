from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from paper_orchestra.config import (
    DEFAULT_MODEL,
    DEPENDENT_RESULT_SETTINGS,
    NEUTRAL_SETTINGS,
    RESULT_SETTINGS,
    MemoryConfig,
    ModelConfig,
    PipelineConfig,
    default_lab_memory_dir,
    resolve_template_dir,
    result_settings,
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


def test_provided_figures_are_used_without_plotting_or_when_kept(tmp_path):
    cfg = PipelineConfig(base_dir=tmp_path, template_dir=tmp_path)
    assert cfg.keep_provided_figures is False
    assert cfg.uses_provided_figures  # no plotting: the provided figures are the figures

    cfg.use_plotting = True
    assert not cfg.uses_provided_figures  # as in the original method
    cfg.keep_provided_figures = True
    assert cfg.uses_provided_figures

    data = cfg.to_dict()
    assert data["keep_provided_figures"] is True
    assert PipelineConfig.from_dict(data) == cfg
    # A configuration written before the option existed still loads.
    del data["keep_provided_figures"]
    assert PipelineConfig.from_dict(data).keep_provided_figures is False


def test_every_option_is_classified_as_shaping_the_paper_or_not():
    # A new option must be put in one list or the other: an option that changes the paper
    # and is missing from the result settings would let a resumed run reuse stale stages.
    options = {field.name for field in fields(PipelineConfig)} - {"models", "memory"}
    options |= {f"models.{field.name}" for field in fields(ModelConfig)}
    options |= {f"memory.{field.name}" for field in fields(MemoryConfig)}
    dependent = [name for names in DEPENDENT_RESULT_SETTINGS.values() for name in names]
    classified = [*RESULT_SETTINGS, *dependent, *NEUTRAL_SETTINGS]

    assert sorted(classified) == sorted(options)
    assert set(DEPENDENT_RESULT_SETTINGS) <= set(RESULT_SETTINGS)


def test_result_settings_are_what_shapes_the_paper(tmp_path):
    cfg = PipelineConfig(base_dir=tmp_path / "a", template_dir=tmp_path / "t")
    settings = cfg.result_settings()
    assert settings["models.writer"] == DEFAULT_MODEL and settings["models.effort"] == "high"
    assert settings["use_plotting"] is False and settings["memory.dream"] == "stages"
    # They survive the trip through state.json unchanged.
    assert result_settings(json.loads(json.dumps(cfg.to_dict()))) == settings

    # Where and how fast the work is done does not count.
    elsewhere = PipelineConfig(
        base_dir=tmp_path / "b",
        template_dir=tmp_path / "t2",
        compute="modal",
        resume=False,
        max_stage_attempts=5,
        literature_workers=1,
        figure_workers=1,
        figure_code_timeout=5.0,
        latex_timeout=5.0,
        research_cutoff="2020-01",
        memory=MemoryConfig(
            project_dir=tmp_path / "m", lab_dir=tmp_path / "l", lab_consolidate=False
        ),
    )
    assert elsewhere.result_settings() == settings

    # The options of a part that is switched off are not in play.
    cfg.models.plotting = "another-model"
    cfg.plotting_max_critic_rounds = 7
    cfg.keep_provided_figures = True
    assert cfg.result_settings() == settings
    cfg.use_plotting = True
    plotting = cfg.result_settings()
    assert plotting["models.plotting"] == "another-model"
    assert plotting["plotting_max_critic_rounds"] == 7 and plotting["keep_provided_figures"]
    assert "models.image" in plotting

    cfg.memory.enabled = False
    without_memory = cfg.result_settings()
    assert without_memory["memory.enabled"] is False
    assert "memory.lab" not in without_memory and "memory.dream" not in without_memory

    # What a stored configuration does not record is absent, not guessed.
    assert result_settings({"use_plotting": True, "models": {"writer": "m"}}) == {
        "use_plotting": True,
        "models.writer": "m",
    }
    assert result_settings({"models": "not a mapping"}) == {}


def test_memory_dirs(tmp_path, monkeypatch):
    cfg = PipelineConfig(base_dir=tmp_path, template_dir=tmp_path)
    assert cfg.project_memory_dir == tmp_path / "memory"
    assert cfg.memory.resolved_lab_dir() == Path.home() / ".paper_orchestra" / "memory"
    monkeypatch.setenv("PAPER_ORCHESTRA_MEMORY_DIR", str(tmp_path / "lab"))
    assert default_lab_memory_dir() == tmp_path / "lab"
    assert MemoryConfig(lab=False).resolved_lab_dir() is None
    assert MemoryConfig(enabled=False).resolved_lab_dir() is None


def test_resolve_template_dir(tmp_path, monkeypatch, repo_root):
    assert (
        resolve_template_dir("cvpr2025")
        == repo_root / "src" / "paper_orchestra" / "templates" / "cvpr2025"
    )
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
