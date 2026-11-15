from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from paper_orchestra import pipeline
from paper_orchestra.compute.base import CompileResult, ComputeBackend, ExecResult
from paper_orchestra.config import MemoryConfig, PipelineConfig
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import LLMConfigError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, TranscriptLog, Wiki
from paper_orchestra.pipeline import PaperResult, prepare_run_dir, run_paper
from paper_orchestra.utils.common import read_json, write_json

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
TEMPLATE = "\\documentclass{article}\n\\begin{document}\n% INTRO\n\\end{document}\n"
FILLED_TEMPLATE = TEMPLATE.replace("% INTRO", "Prior work \\cite{Smith2020}.")
IDEA = "# Idea\n\nWe propose FooNet, a sparse attention variant.\n"
LOG = "# Experimental log\n\nFooNet reaches 91.2 accuracy on BarBench.\n"
OLD_TIME_KEYS = {
    "Start_Time_Absolute",
    "OutlineAgent",
    "Parallel_Lit_and_Plot",
    "SectionWritingAgent",
    "ContentRefinementAgent",
    "Total_Time_Span",
}


class FakeCompute(ComputeBackend):
    name = "fake"

    def __init__(self) -> None:
        self.closed = 0

    def run_python(self, code, *, inputs=None, outputs=(), timeout=120.0) -> ExecResult:
        return ExecResult(returncode=0)

    def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0) -> CompileResult:
        return CompileResult(ok=True)

    def close(self) -> None:
        self.closed += 1


class StubAgents:
    """Stand-ins for the five agents: they write the files the real ones would."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        #: Exceptions raised by successive calls of a stage, consumed front to back.
        self.failures: dict[str, list[BaseException]] = {}
        self.seen: dict[str, Any] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.figure_id = "fig_arch"
        #: When set, literature and plotting must both be inside ``run`` to get past it.
        self.barrier: threading.Barrier | None = None
        self.plotting_delay = 0.0
        self._lock = threading.Lock()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> StubAgents:
        monkeypatch.setattr(pipeline, "_outline_agent", lambda ctx: _Outline(self, ctx))
        monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: _Literature(self, ctx))
        monkeypatch.setattr(pipeline, "_plotting_agent", lambda ctx: _Plotting(self, ctx))
        monkeypatch.setattr(pipeline, "_section_writing_agent", lambda ctx: _Section(self, ctx))
        monkeypatch.setattr(pipeline, "_refinement_agent", lambda ctx: _Refinement(self, ctx))
        return self

    def enter(self, stage: str, ctx: RunContext) -> None:
        with self._lock:
            self.calls.append(stage)
            self.states[stage] = read_json(ctx.paths.state)
            queue = self.failures.get(stage)
            error = queue.pop(0) if queue else None
        if error is not None:
            raise error

    def count(self) -> Counter[str]:
        return Counter(self.calls)


class _Agent:
    def __init__(self, stubs: StubAgents, ctx: RunContext) -> None:
        self.stubs = stubs
        self.ctx = ctx


class _Outline(_Agent):
    def run(self) -> dict[str, Any]:
        ctx = self.ctx
        self.stubs.enter("outline", ctx)
        ctx.harness.generate(
            "Plan the paper.", agent="OutlineAgent", stage="outline", model=ctx.cfg.models.writer
        )
        outline = {
            "plotting_plan": [{"figure_id": "fig_arch", "plot_type": "diagram"}],
            "intro_related_work_plan": {"introduction_strategy": {}},
            "section_plan": [{"section_title": "Method"}],
        }
        write_json(ctx.paths.outline, outline)
        return outline


class _Literature(_Agent):
    def run(self, outline: dict[str, Any]) -> None:
        ctx = self.ctx
        self.stubs.seen.setdefault("literature_outlines", []).append(outline)
        self.stubs.seen.setdefault("literature_templates", []).append(
            ctx.paths.template_tex.read_text()
        )
        if self.stubs.barrier is not None:
            self.stubs.barrier.wait(timeout=5)
        self.stubs.enter("literature", ctx)
        out = ctx.paths.literature_dir
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "outline_v1.json", {**outline, "citations": ["Smith2020"]})
        (out / "updated_template.tex").write_text(FILLED_TEMPLATE)
        (out / "references.bib").write_text("@article{Smith2020, title={Deep Things}}\n")
        write_json(out / "citation_map.json", {"Smith2020": {"title": "Deep Things"}})
        ctx.harness.write_page(
            "wiki/literature/Smith2020.md",
            "> Deep things are deep.",
            {"summary": "Deep Things (2020)", "kind": "literature"},
        )


class _Plotting(_Agent):
    def run(self, outline: dict[str, Any]) -> list[dict[str, Any]]:
        ctx = self.ctx
        self.stubs.seen.setdefault("plotting_outlines", []).append(outline)
        self.stubs.seen.setdefault("plotting_figure_lists", []).append(
            read_json(ctx.paths.figures_info)
        )
        if self.stubs.barrier is not None:
            self.stubs.barrier.wait(timeout=5)
        time.sleep(self.stubs.plotting_delay)
        self.stubs.enter("plotting", ctx)
        name = f"{self.stubs.figure_id}.png"
        for directory in (ctx.paths.generated_figures_dir, ctx.paths.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_bytes(PNG)
        results = [
            {
                "figure_id": self.stubs.figure_id,
                "image_path": f"figures/{name}",
                "caption": "Overview.",
            }
        ]
        write_json(ctx.paths.plotting_results, results)
        listed = read_json(ctx.paths.figures_info)
        write_json(ctx.paths.figures_info, [*listed, {"name": name, "caption": "Overview."}])
        return results


class _Section(_Agent):
    def run(self) -> Path:
        ctx = self.ctx
        self.stubs.enter("section_writing", ctx)
        seen = self.stubs.seen
        seen["figures"] = read_json(ctx.paths.figures_info)
        seen["writer_template"] = ctx.paths.template_tex.read_text()
        seen["writer_outline"] = read_json(ctx.paths.outline)
        seen["writer_bib"] = ctx.paths.references_bib.read_text()
        if ctx.harness.project is not None:
            wiki = ctx.harness.project
            seen["pending_notes"] = [wiki.read(info.path) for info in wiki.pending_notes()]
        ctx.paths.raw_draft_tex.write_text(seen["writer_template"] + "% draft\n")
        return ctx.paths.raw_draft_tex


class _Refinement(_Agent):
    def run(self, tex_path: Path) -> Path:
        ctx = self.ctx
        self.stubs.enter("refinement", ctx)
        assert tex_path == ctx.paths.raw_draft_tex
        work = ctx.paths.refinement_dir
        work.mkdir(exist_ok=True)
        (work / "final_refined_paper.tex").write_text(tex_path.read_text() + "% refined\n")
        (work / "final_refined_paper.pdf").write_bytes(b"%PDF-1.4 refined paper")
        return work / "final_refined_paper.pdf"


def responder(request: FakeRequest) -> str:
    if request.label.startswith("dream"):
        if "wiki/synthesis" in request.text:
            request.memory.create(
                "/memories/wiki/synthesis/themes.md",
                "---\nsummary: Themes of the collected literature\nkind: synthesis\n---\n"
                "# Themes\n\nSparse attention builds on [[literature/Smith2020]].\n",
            )
        return "Consolidated."
    return "ok"


def make_harness(cfg: PipelineConfig, client: FakeLLM | None = None) -> ContextHarness:
    client = client or FakeLLM(responder)
    project = Wiki(cfg.project_memory_dir)
    return ContextHarness(
        project=project,
        transcripts=TranscriptLog(project.root / "transcripts"),
        client_for=lambda model: client,
        dream_model="fake-dreamer",
    )


def events(harness: ContextHarness, *types: str) -> list[dict[str, Any]]:
    assert harness.transcripts is not None
    return [e for e in harness.transcripts.events() if e["type"] in types]


@pytest.fixture
def template_dir(tmp_path: Path) -> Path:
    template = tmp_path / "templates" / "demo"
    template.mkdir(parents=True)
    (template / "template.tex").write_text(TEMPLATE)
    (template / "guidelines.md").write_text("Eight pages. Anonymous.\n")
    (template / "references.bib").write_text("@article{Sample, title={Shipped with template}}\n")
    (template / "venue.sty").write_text("% style\n")
    return template


@pytest.fixture
def cfg(tmp_path: Path, template_dir: Path) -> PipelineConfig:
    base = tmp_path / "run"
    materials = base / "raw_materials"
    materials.mkdir(parents=True)
    (materials / "idea_sparse.md").write_text(IDEA)
    (materials / "experimental_log.md").write_text(LOG)
    return PipelineConfig(base_dir=base, template_dir=template_dir, memory=MemoryConfig(lab=False))


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> StubAgents:
    return StubAgents().install(monkeypatch)


def run(
    cfg: PipelineConfig, harness: ContextHarness | None = None
) -> tuple[PaperResult, ContextHarness, FakeCompute]:
    harness = harness or make_harness(cfg)
    compute = FakeCompute()
    return run_paper(cfg, harness=harness, compute=compute), harness, compute


def state_of(cfg: PipelineConfig) -> dict[str, Any]:
    return read_json(cfg.base_dir / "state.json")


# -- happy paths -----------------------------------------------------------------


def test_run_without_plotting_honours_the_file_contract(cfg, stubs, caplog):
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, harness, compute = run(cfg)
    paths = RunPaths(cfg.base_dir)

    assert result.success and result.error is None
    assert result.pdf_path == paths.final_pdf
    assert result.base_dir == cfg.base_dir
    assert paths.final_pdf.read_bytes() == b"%PDF-1.4 refined paper"
    assert paths.final_tex.read_text().endswith("% draft\n% refined\n")
    assert stubs.calls == ["outline", "literature", "section_writing", "refinement"]

    for produced in (
        paths.template_tex,
        paths.guidelines,
        paths.references_bib,
        paths.figures_info,
        paths.raw_draft_tex,
        paths.outline,
        paths.citation_map,
        paths.literature_dir / "outline_v1.json",
        paths.literature_dir / "updated_template.tex",
        paths.literature_dir / "references.bib",
        paths.refinement_dir / "final_refined_paper.pdf",
        paths.final_pdf,
        paths.final_tex,
        paths.state,
        paths.usage,
        paths.time_breakdown,
        paths.memory_dir / "MEMORY.md",
        cfg.base_dir / "memory_report.json",
        cfg.base_dir / "outline_v0.json",
        paths.latex_dir / "venue.sty",
    ):
        assert produced.is_file(), produced
    assert not paths.plotting_results.exists()

    # The pipeline wires the literature output into the workspace the writer reads.
    assert stubs.seen["writer_template"] == FILLED_TEMPLATE
    assert stubs.seen["writer_outline"]["citations"] == ["Smith2020"]
    assert "Smith2020" in stubs.seen["writer_bib"] and "Sample" not in stubs.seen["writer_bib"]

    # No figures and no plotting: an empty list, and the user is told.
    assert stubs.seen["figures"] == []
    assert read_json(paths.figures_info) == []
    assert any("will have no figures" in r.getMessage() for r in caplog.records)
    # A compute backend handed in by the caller stays open.
    assert compute.closed == 0
    assert harness.usage.total().requests > 0


def test_ingest_mirrors_the_inputs_into_the_wiki(cfg, stubs):
    _, harness, _ = run(cfg)
    wiki = harness.project

    idea = wiki.read("wiki/sources/idea.md")
    assert idea.body == IDEA
    assert idea.kind == "source" and idea.meta["summary"]
    assert idea.meta["source_file"] == "raw_materials/idea_sparse.md"
    assert wiki.read("wiki/sources/experimental_log.md").body == LOG
    assert wiki.read("wiki/sources/guidelines.md").body == "Eight pages. Anonymous.\n"
    # The template page is the template as shipped, not the one literature filled in.
    template = wiki.read("wiki/sources/template.md").body
    assert template == f"```latex\n{TEMPLATE.rstrip()}\n```\n"
    # The index was refreshed after the stages, so every session sees the pages.
    index = (wiki.root / "MEMORY.md").read_text()
    assert "[[sources/idea]]" in index and "[[literature/Smith2020]]" in index


def test_run_with_plotting_runs_both_stages_concurrently_and_merges_figures(cfg, stubs, caplog):
    figures = cfg.base_dir / "raw_materials" / "figures"
    figures.mkdir()
    (figures / "teaser.png").write_bytes(PNG)
    (figures / "extra.jpg").write_bytes(b"\xff\xd8\xff jpeg")
    write_json(
        figures / "info.json",
        [
            {"name": "teaser", "caption": "The teaser.", "width": 0.5},
            {"name": "ghost.png", "caption": "Not on disk."},
        ],
    )
    cfg.use_plotting = True
    # Both stubs block until the other one is running too: sequential stages would time out.
    stubs.barrier = threading.Barrier(2)

    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, harness, _ = run(cfg)
    paths = RunPaths(cfg.base_dir)

    assert result.success, result.error
    assert stubs.calls[0] == "outline"
    assert set(stubs.calls[1:3]) == {"literature", "plotting"}
    assert stubs.calls[3:] == ["section_writing", "refinement"]
    assert list(result.stages) == list(pipeline.STAGES)
    assert all(record["status"] == "completed" for record in result.stages.values())

    # Provided figures are listed with their real file names; a file without a caption is
    # still offered to the writer, and an entry without a file is dropped.
    provided = [
        {"name": "teaser.png", "caption": "The teaser.", "width": 0.5},
        {"name": "extra.jpg", "caption": ""},
    ]
    assert stubs.seen["plotting_figure_lists"] == [provided]
    assert stubs.seen["figures"] == [*provided, {"name": "fig_arch.png", "caption": "Overview."}]
    for name in ("teaser.png", "extra.jpg", "fig_arch.png"):
        assert (paths.figures_dir / name).is_file()
    assert read_json(paths.plotting_results)[0]["figure_id"] == "fig_arch"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "ghost.png" in messages and "no caption is given for extra.jpg" in messages

    # Both parallel stages get the outline as the outline stage wrote it.
    assert "citations" not in stubs.seen["literature_outlines"][0]
    assert stubs.seen["plotting_outlines"][0] == stubs.seen["literature_outlines"][0]

    page = harness.project.read("wiki/figures/teaser.md")
    assert page.kind == "figure" and page.meta["origin"] == "provided"
    assert "The teaser." in page.body


def test_a_repeated_plotting_stage_leaves_no_figures_of_the_earlier_one(cfg, stubs, monkeypatch):
    figures = cfg.base_dir / "raw_materials" / "figures"
    figures.mkdir()
    (figures / "teaser.png").write_bytes(PNG)
    cfg.use_plotting = True
    run(cfg)
    paths = RunPaths(cfg.base_dir)
    assert (paths.figures_dir / "fig_arch.png").is_file()
    paths.plotting_results.unlink()

    again = StubAgents().install(monkeypatch)
    again.figure_id = "fig_results"
    result, _, _ = run(cfg)

    assert result.success
    assert again.calls == ["plotting", "section_writing", "refinement"]
    assert again.seen["figures"] == [
        {"name": "teaser.png", "caption": ""},
        {"name": "fig_results.png", "caption": "Overview."},
    ]
    assert sorted(p.name for p in paths.figures_dir.iterdir()) == [
        "fig_results.png",
        "info.json",
        "teaser.png",
    ]
    assert [p.name for p in paths.generated_figures_dir.iterdir()] == ["fig_results.png"]


def test_literature_stage_is_followed_by_a_synthesis_dream(cfg, stubs):
    client = FakeLLM(responder)
    result, harness, _ = run(cfg, harness=make_harness(cfg, client))

    assert result.success
    dreams = [r for r in client.requests if r.label == "dream.project"]
    assert pipeline.LITERATURE_SYNTHESIS_FOCUS in dreams[0].text
    synthesis = harness.project.read("wiki/synthesis/themes.md")
    assert "[[literature/Smith2020]]" in synthesis.body
    stages = [e["stage"] for e in events(harness, "dream")]
    assert stages == ["literature", "finalize"]


# -- bookkeeping -----------------------------------------------------------------


def test_state_json_follows_every_transition(cfg, stubs):
    result, harness, _ = run(cfg)

    during_outline = stubs.states["outline"]
    assert during_outline["status"] == "running"
    assert during_outline["stages"]["ingest"]["status"] == "completed"
    assert during_outline["stages"]["outline"] == {
        "status": "running",
        "attempts": 1,
        "seconds": 0.0,
        "error": None,
        "outputs": [],
        "reused": False,
    }
    assert during_outline["stages"]["literature"]["status"] == "pending"
    during_refinement = stubs.states["refinement"]["stages"]
    assert during_refinement["section_writing"]["outputs"] == ["latex_writeup/raw_draft_paper.tex"]
    assert during_refinement["finalize"]["status"] == "pending"

    state = state_of(cfg)
    order = ["ingest", "outline", "literature", "section_writing", "refinement", "finalize"]
    assert state["status"] == "completed" and state["error"] is None
    assert list(state["stages"]) == order
    assert state["stages"] == result.stages
    assert state["stages"]["refinement"]["outputs"] == [
        "content_refinement_workdir/final_refined_paper.pdf",
        "content_refinement_workdir/final_refined_paper.tex",
    ]
    assert state["config"]["models"]["writer"] == "claude-fable-5-2"

    boundaries = [(e["type"], e["stage"]) for e in events(harness, "stage_start", "stage_end")]
    assert boundaries == [(kind, name) for name in order for kind in ("stage_start", "stage_end")]
    assert [e["type"] for e in events(harness, "run_start", "run_end")] == ["run_start", "run_end"]


def test_time_breakdown_usage_and_memory_report(cfg, stubs):
    before = time.time()
    result, harness, _ = run(cfg)
    paths = RunPaths(cfg.base_dir)

    breakdown = read_json(paths.time_breakdown)
    assert set(breakdown) == OLD_TIME_KEYS
    assert before <= breakdown["Start_Time_Absolute"] <= time.time()
    assert breakdown["OutlineAgent"] == result.stages["outline"]["seconds"]
    assert breakdown["Parallel_Lit_and_Plot"] == result.stages["literature"]["seconds"]
    assert breakdown["Total_Time_Span"] >= breakdown["ContentRefinementAgent"] >= 0

    usage = read_json(paths.usage)
    assert usage == result.usage
    rows = {(row["label"], row["model"]): row for row in usage["by_label"]}
    assert rows[("outline.OutlineAgent", "claude-fable-5-2")]["requests"] == 1
    assert ("dream.project", "fake-dreamer") in rows
    assert usage["total"]["requests"] == harness.usage.total().requests
    # The writer model is priced; the made-up dream model is reported as not priced.
    assert usage["estimated_cost_usd"] > 0
    assert usage["unpriced_models"] == ["fake-dreamer"]

    report = read_json(cfg.base_dir / "memory_report.json")
    assert report["enabled"] is True and report["lab"] is None
    assert report["project"]["pages"] >= 5
    assert [(r["scope"], r["stage"]) for r in report["reports"]] == [("project", "finalize")]


def test_run_with_memory_disabled(cfg, stubs):
    harness = ContextHarness(client_for=lambda model: FakeLLM(responder))
    result, _, _ = run(cfg, harness=harness)

    assert result.success
    assert read_json(cfg.base_dir / "memory_report.json") == {
        "enabled": False,
        "project": None,
        "lab": None,
        "reports": [],
    }
    assert not (cfg.base_dir / "memory").exists()


def test_harness_and_compute_are_built_from_the_config_and_compute_is_closed(
    cfg, stubs, monkeypatch
):
    built: list[FakeCompute] = []

    def build_compute(config: PipelineConfig) -> FakeCompute:
        assert config.compute == "local"
        built.append(FakeCompute())
        return built[-1]

    monkeypatch.setattr(pipeline, "_build_harness", make_harness)
    monkeypatch.setattr(pipeline, "_build_compute", build_compute)

    assert run_paper(cfg).success
    assert built[0].closed == 1

    # The backend is released on failure too.
    cfg.resume = False
    stubs.failures["outline"] = [RuntimeError("down"), RuntimeError("down")]
    assert not run_paper(cfg).success
    assert built[1].closed == 1


def test_the_default_factories_build_the_real_agents_harness_and_backend(cfg):
    ctx = RunContext(
        cfg=cfg,
        paths=RunPaths(cfg.base_dir),
        harness=ContextHarness(client_for=lambda model: FakeLLM(["unused"])),
        compute=FakeCompute(),
    )
    factories = (
        pipeline._outline_agent,
        pipeline._literature_agent,
        pipeline._plotting_agent,
        pipeline._section_writing_agent,
        pipeline._refinement_agent,
    )
    assert [type(factory(ctx)).__name__ for factory in factories] == [
        "OutlineAgent",
        "LiteratureAgent",
        "PlottingAgent",
        "SectionWritingAgent",
        "ContentRefinementAgent",
    ]

    backend = pipeline._build_compute(cfg)
    assert backend.name == "local"
    harness = pipeline._build_harness(cfg)
    assert harness.project.root == cfg.base_dir / "memory"
    assert harness.lab is None and harness.dream_mode == "stages"


# -- failures --------------------------------------------------------------------


def test_a_failed_stage_is_retried_alone_with_a_note_about_the_failure(cfg, stubs):
    stubs.failures["section_writing"] = [RuntimeError("compile exploded on cleveref")]
    result, harness, _ = run(cfg)

    assert result.success
    assert stubs.count() == {"outline": 1, "literature": 1, "section_writing": 2, "refinement": 1}
    record = result.stages["section_writing"]
    assert record["status"] == "completed" and record["attempts"] == 2 and record["error"] is None

    # The retry could read what went wrong.
    notes = [n for n in stubs.seen["pending_notes"] if n.meta.get("stage") == "section_writing"]
    assert len(notes) == 1
    assert notes[0].kind == "lesson"
    assert "RuntimeError: compile exploded on cleveref" in notes[0].body
    assert "attempt 1 of 2" in notes[0].body and "retried" in notes[0].body

    ends = [(e["status"], e["attempt"]) for e in events(harness, "stage_end") if
            e["stage"] == "section_writing"]  # fmt: skip
    assert ends == [("failed", 1), ("completed", 2)]
    failed = next(e for e in events(harness, "stage_end") if e["status"] == "failed")
    assert failed["error"] == "RuntimeError: compile exploded on cleveref"


def test_a_stage_that_keeps_failing_fails_the_run_without_raising(cfg, stubs):
    stubs.failures["section_writing"] = [RuntimeError("first"), RuntimeError("second")]
    result, harness, _ = run(cfg)
    paths = RunPaths(cfg.base_dir)

    assert not result.success and result.pdf_path is None
    assert result.error == (
        "stage 'section_writing' failed after 2 attempt(s): RuntimeError: second"
    )
    assert "refinement" not in stubs.calls
    assert not paths.final_pdf.exists() and not paths.time_breakdown.exists()

    state = state_of(cfg)
    assert state["status"] == "failed" and state["error"] == result.error
    assert state["stages"]["literature"]["status"] == "completed"
    assert state["stages"]["section_writing"] == {
        "status": "failed",
        "attempts": 2,
        "seconds": state["stages"]["section_writing"]["seconds"],
        "error": "RuntimeError: second",
        "outputs": [],
        "reused": False,
    }
    assert state["stages"]["refinement"]["status"] == "pending"
    assert read_json(paths.usage)["total"]["requests"] > 0
    assert events(harness, "run_end")[0]["status"] == "failed"
    # Both failures were noted for whoever picks the run up again.
    kinds = [harness.project.read(n.path).kind for n in harness.project.pending_notes()]
    assert kinds == ["lesson", "lesson"]


def test_a_stage_without_its_output_counts_as_failed(cfg, stubs, monkeypatch):
    class Lazy(_Section):
        def run(self) -> Path:
            self.stubs.enter("section_writing", self.ctx)
            return self.ctx.paths.raw_draft_tex

    monkeypatch.setattr(pipeline, "_section_writing_agent", lambda ctx: Lazy(stubs, ctx))
    result, _, _ = run(cfg)

    assert not result.success
    assert "StageError" in result.error and "raw_draft_paper.tex" in result.error


def test_literature_failure_is_fatal_and_plotting_is_still_awaited(cfg, stubs):
    cfg.use_plotting = True
    stubs.plotting_delay = 0.15
    stubs.failures["literature"] = [RuntimeError("scholar down"), RuntimeError("scholar down")]
    result, _, _ = run(cfg)

    assert not result.success
    assert result.error.startswith("stage 'literature' failed after 2 attempt(s)")
    # Plotting was slower than both literature attempts, yet it had finished on return.
    assert result.stages["plotting"]["status"] == "completed"
    assert RunPaths(cfg.base_dir).plotting_results.is_file()
    assert "section_writing" not in stubs.calls


def test_plotting_failure_is_tolerated(cfg, stubs, caplog, monkeypatch):
    figures = cfg.base_dir / "raw_materials" / "figures"
    figures.mkdir()
    (figures / "teaser.png").write_bytes(PNG)
    cfg.use_plotting = True
    stubs.failures["plotting"] = [RuntimeError("matplotlib boom"), RuntimeError("still boom")]
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, harness, _ = run(cfg)

    assert result.success, result.error
    assert stubs.count()["plotting"] == 2
    record = state_of(cfg)["stages"]["plotting"]
    assert record["status"] == "failed" and record["tolerated"] is True
    assert record["error"] == "RuntimeError: still boom"
    assert state_of(cfg)["status"] == "completed"
    # The paper is written with the provided figure only.
    assert stubs.seen["figures"] == [{"name": "teaser.png", "caption": ""}]
    assert any("continuing without its output" in r.getMessage() for r in caplog.records)
    # Both failures were noted (a dream may already have archived them).
    wiki = harness.project
    notes = [wiki.read(info.path) for info in wiki.list("notes")]
    bodies = [note.body for note in notes if note.meta.get("stage") == "plotting"]
    assert len(bodies) == 2
    assert any("still boom" in body and "continues without" in body for body in bodies)
    assert read_json(RunPaths(cfg.base_dir).time_breakdown)["Parallel_Lit_and_Plot"] >= 0

    # The run accepted that failure, so resuming it has nothing left to do.
    again = StubAgents().install(monkeypatch)
    resumed, _, _ = run(cfg)
    assert resumed.success and again.calls == []
    assert resumed.stages["plotting"]["status"] == "failed" and resumed.stages["plotting"]["reused"]


def test_long_error_messages_are_cut_in_state_and_notes(cfg, stubs):
    stubs.failures["outline"] = [RuntimeError("x" * 50_000), RuntimeError("short")]
    result, harness, _ = run(cfg)

    assert not result.success and result.stages["outline"]["error"] == "RuntimeError: short"
    first = events(harness, "stage_end")[1]["error"]
    assert len(first) < 2_100 and first.endswith("(cut here; the log has the full error)")
    assert all(info.size < 3_000 for info in harness.project.pending_notes())


def test_configuration_errors_inside_a_stage_are_not_retried(cfg, stubs):
    stubs.failures["outline"] = [LLMConfigError("structured outputs are disabled")]
    result, _, _ = run(cfg)

    assert not result.success
    assert result.stages["outline"]["attempts"] == 1
    assert stubs.calls == ["outline"]


def test_an_interrupt_is_recorded_and_propagates(cfg, stubs):
    stubs.failures["section_writing"] = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        run(cfg)

    state = state_of(cfg)
    assert state["status"] == "interrupted"
    assert state["stages"]["section_writing"]["status"] == "interrupted"
    assert state["stages"]["literature"]["status"] == "completed"


# -- resume ----------------------------------------------------------------------


def test_resume_runs_only_what_is_left(cfg, stubs, monkeypatch):
    stubs.failures["refinement"] = [RuntimeError("reviewer down"), RuntimeError("reviewer down")]
    first, _, _ = run(cfg)
    assert not first.success

    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)

    assert result.success
    assert again.calls == ["refinement"]
    for name in ("ingest", "outline", "literature", "section_writing"):
        assert result.stages[name]["status"] == "completed"
        assert result.stages[name]["reused"] is True
        assert result.stages[name]["seconds"] == first.stages[name]["seconds"]
    assert result.stages["refinement"]["reused"] is False
    assert result.stages["refinement"]["attempts"] == 1

    # Usage and time cover both invocations.
    rows = {row["label"]: row for row in result.usage["by_label"]}
    assert rows["outline.OutlineAgent"]["requests"] == 1
    breakdown = read_json(RunPaths(cfg.base_dir).time_breakdown)
    assert breakdown["Start_Time_Absolute"] == state_of(cfg)["start_time"]
    assert breakdown["OutlineAgent"] == first.stages["outline"]["seconds"]


def test_resuming_a_finished_run_does_nothing(cfg, stubs, monkeypatch):
    first, _, _ = run(cfg)
    again = StubAgents().install(monkeypatch)
    client = FakeLLM(responder)
    result, _, _ = run(cfg, harness=make_harness(cfg, client))

    assert result.success and result.pdf_path.is_file()
    assert again.calls == [] and client.requests == []
    assert all(record["reused"] for record in result.stages.values())
    assert result.usage == first.usage


def test_a_missing_output_reruns_that_stage_and_everything_after_it(cfg, stubs, monkeypatch):
    run(cfg)
    paths = RunPaths(cfg.base_dir)
    paths.citation_map.unlink()

    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)

    assert result.success
    assert again.calls == ["literature", "section_writing", "refinement"]
    assert result.stages["outline"]["reused"] and not result.stages["finalize"]["reused"]
    # The repeated literature stage starts from the untouched outline and template, not
    # from what its first pass left in the workspace.
    assert "citations" not in again.seen["literature_outlines"][0]
    assert again.seen["literature_templates"] == [TEMPLATE]
    assert again.seen["writer_template"] == FILLED_TEMPLATE


@pytest.mark.parametrize("change", ["idea", "template", "no_resume"])
def test_changed_inputs_or_no_resume_start_over(cfg, stubs, monkeypatch, change):
    run(cfg)
    if change == "idea":
        (cfg.base_dir / "raw_materials" / "idea_sparse.md").write_text(IDEA + "\nNow with more.\n")
    elif change == "template":
        (cfg.template_dir / "guidelines.md").write_text("Nine pages.\n")
    else:
        cfg.resume = False

    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)

    assert result.success
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]
    assert not any(record["reused"] for record in result.stages.values())
    # Usage of the abandoned run is not carried into a fresh start.
    rows = {row["label"]: row for row in result.usage["by_label"]}
    assert rows["outline.OutlineAgent"]["requests"] == 1


def test_enabling_plotting_on_resume_reruns_from_the_parallel_stage(cfg, stubs, monkeypatch):
    run(cfg)
    cfg.use_plotting = True
    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)

    assert result.success
    assert again.calls == ["plotting", "section_writing", "refinement"]
    assert result.stages["literature"]["reused"] is True
    assert again.seen["figures"] == [{"name": "fig_arch.png", "caption": "Overview."}]


# -- validation ------------------------------------------------------------------


def test_missing_inputs_raise_before_anything_runs(cfg, stubs):
    paths = RunPaths(cfg.base_dir)
    paths.idea.unlink()
    with pytest.raises(FileNotFoundError, match=r"idea file was not found: .*idea_sparse\.md"):
        run(cfg)
    assert stubs.calls == [] and not paths.state.exists()

    paths.idea.write_text(IDEA)
    paths.experimental_log.write_text("  \n")
    with pytest.raises(ValueError, match="experimental log file is empty"):
        run(cfg)

    paths.experimental_log.write_text(LOG)
    (cfg.template_dir / "guidelines.md").unlink()
    with pytest.raises(FileNotFoundError, match=r"has no guidelines\.md"):
        run(cfg)
    assert stubs.calls == [] and not paths.state.exists()


def test_invalid_settings_raise_value_error(cfg, stubs):
    cfg.max_stage_attempts = 0
    with pytest.raises(ValueError, match="max_stage_attempts must be at least 1"):
        run(cfg)
    cfg.max_stage_attempts = 2
    cfg.idea_filename = "../idea_sparse.md"
    with pytest.raises(ValueError, match="idea_filename must be a file name"):
        run(cfg)
    cfg.idea_filename = "idea_sparse.md"
    cfg.research_cutoff = "last spring"
    with pytest.raises(ValueError, match="research cutoff must look like YYYY-MM"):
        run(cfg)
    assert stubs.calls == []


def test_an_unusable_model_fails_the_run_before_any_stage(cfg, stubs):
    def client_for(model: str) -> FakeLLM:
        if model == "gpt-missing-key":
            raise LLMConfigError("OPENAI_API_KEY is not set")
        return FakeLLM(responder)

    cfg.models.reflection = "gpt-missing-key"
    result = run_paper(cfg, harness=ContextHarness(client_for=client_for), compute=FakeCompute())

    assert not result.success
    assert result.error == (
        "the reflection model 'gpt-missing-key' cannot be used: OPENAI_API_KEY is not set"
    )
    assert stubs.calls == []
    state = state_of(cfg)
    assert state["status"] == "failed" and state["error"] == result.error
    assert {record["status"] for record in state["stages"].values()} == {"pending"}


def test_an_unusable_image_model_only_warns(cfg, stubs, caplog):
    def client_for(model: str) -> FakeLLM:
        if model == "imagen-missing-key":
            raise LLMConfigError("GEMINI_API_KEY is not set")
        return FakeLLM(responder)

    cfg.use_plotting = True
    cfg.models.image = "imagen-missing-key"
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result = run_paper(
            cfg, harness=ContextHarness(client_for=client_for), compute=FakeCompute()
        )

    assert result.success
    assert any("image model 'imagen-missing-key'" in r.getMessage() for r in caplog.records)


def test_missing_tex_fails_the_run_before_any_stage(cfg, stubs, monkeypatch):
    from paper_orchestra.compute.local import LocalBackend

    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: False))
    result = run_paper(cfg, harness=make_harness(cfg), compute=LocalBackend())

    assert not result.success
    assert "pdflatex/bibtex were not found" in result.error
    assert stubs.calls == []


def test_prepare_run_dir_copies_and_validates_raw_materials(tmp_path, template_dir):
    source = tmp_path / "materials"
    (source / "figures").mkdir(parents=True)
    (source / "idea_sparse.md").write_text(IDEA)
    (source / "experimental_log.md").write_text(LOG)
    (source / "figures" / "teaser.png").write_bytes(PNG)
    config = PipelineConfig(base_dir=tmp_path / "out" / "run", template_dir=template_dir)

    paths = prepare_run_dir(config, source)
    assert paths.idea.read_text() == IDEA
    assert (paths.provided_figures_dir / "teaser.png").is_file()

    # A second call replaces the copy, so files removed from the source do not linger.
    (paths.materials_dir / "stale.md").write_text("old")
    (source / "figures" / "teaser.png").unlink()
    prepare_run_dir(config, source)
    assert not (paths.materials_dir / "stale.md").exists()
    assert not (paths.provided_figures_dir / "teaser.png").exists()

    # Pointing at the run's own copy is a no-op, and so is passing nothing.
    (paths.materials_dir / "kept.md").write_text("mine")
    assert prepare_run_dir(config, paths.materials_dir) == paths
    assert prepare_run_dir(config, None) == paths
    assert (paths.materials_dir / "kept.md").is_file()


def test_prepare_run_dir_rejects_bad_locations(tmp_path, template_dir):
    source = tmp_path / "materials"
    source.mkdir()
    (source / "idea_sparse.md").write_text(IDEA)
    (source / "experimental_log.md").write_text(LOG)

    with pytest.raises(FileNotFoundError, match="Raw materials directory not found"):
        prepare_run_dir(PipelineConfig(tmp_path / "a", template_dir), tmp_path / "nowhere")
    assert not (tmp_path / "a").exists()

    with pytest.raises(FileNotFoundError, match="LaTeX template directory not found"):
        prepare_run_dir(PipelineConfig(tmp_path / "a", tmp_path / "no-template"), source)

    with pytest.raises(ValueError, match="inside the raw materials directory"):
        prepare_run_dir(PipelineConfig(source / "out", template_dir), source)
    assert sorted(p.name for p in source.iterdir()) == ["experimental_log.md", "idea_sparse.md"]

    nested = PipelineConfig(tmp_path / "b", template_dir)
    inner = nested.base_dir / "raw_materials" / "inner"
    inner.mkdir(parents=True)
    (inner / "idea_sparse.md").write_text(IDEA)
    (inner / "experimental_log.md").write_text(LOG)
    with pytest.raises(ValueError, match="replaced on every run"):
        prepare_run_dir(nested, inner)
    assert (inner / "idea_sparse.md").is_file()

    with pytest.raises(FileNotFoundError, match="Raw materials directory not found"):
        prepare_run_dir(PipelineConfig(tmp_path / "empty", template_dir), None)


def test_state_file_is_valid_json_at_every_write(cfg, stubs, monkeypatch):
    # Every save goes through a temporary file and an atomic rename.
    seen: list[str] = []
    real = pipeline.write_json

    def spy(path, data):
        result = real(path, data)
        if Path(path).name == "state.json":
            seen.append(json.loads(Path(path).read_text())["status"])
            assert not list(Path(path).parent.glob(".state.json.*.tmp"))
        return result

    monkeypatch.setattr(pipeline, "write_json", spy)
    run(cfg)
    assert seen[0] == "running" and seen[-1] == "completed" and len(seen) > 12
