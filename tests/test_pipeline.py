from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import signal
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from paper_orchestra import pipeline
from paper_orchestra.compute.base import CompileResult, ComputeBackend, ComputeError, ExecResult
from paper_orchestra.config import MemoryConfig, PipelineConfig, result_settings
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import LLMConfigError, Usage
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, DreamReport, RunCancelled, TranscriptLog, Wiki
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
        #: What ``check`` reports; a test sets it to make the backend unusable.
        self.problems: list[str] = []
        self.checks: list[dict[str, bool]] = []

    def run_python(self, code, *, inputs=None, outputs=(), timeout=120.0) -> ExecResult:
        return ExecResult(returncode=0)

    def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0) -> CompileResult:
        return CompileResult(ok=True)

    def check(self, *, python: bool = True, latex: bool = True) -> list[str]:
        self.checks.append({"python": python, "latex": latex})
        return list(self.problems)

    def close(self) -> None:
        self.closed += 1


class StubAgents:
    """Stand-ins for the five agents: they write the files the real ones would."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        #: Exceptions raised by successive calls of a stage, consumed front to back.
        self.failures: dict[str, list[BaseException]] = {}
        #: Called with the run context each time a stage is entered, before it may fail.
        self.hooks: dict[str, Any] = {}
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
        if stage in self.hooks:
            self.hooks[stage](ctx)
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


def provide_figure(cfg: PipelineConfig, caption: str = "The teaser.") -> None:
    figures = cfg.base_dir / "raw_materials" / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "teaser.png").write_bytes(PNG)
    write_json(figures / "info.json", [{"name": "teaser", "caption": caption}])


def stage_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name.startswith("stage-")]


def wait_for_stage_threads(timeout: float = 10.0) -> None:
    for thread in stage_threads():
        thread.join(timeout)
    assert stage_threads() == []


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


def test_run_with_plotting_runs_both_stages_concurrently_and_merges_kept_figures(
    cfg, stubs, caplog
):
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
    cfg.keep_provided_figures = True
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


def test_plotting_generates_every_figure_and_leaves_the_provided_ones_alone(cfg, stubs, caplog):
    # As in the original method: the figures of a paper that ships them (a benchmark folder
    # serves both settings) are not shown to anyone when the run is asked to make its own.
    provide_figure(cfg, caption="Our method reaches 93.1 on the original benchmark.")
    cfg.use_plotting = True
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, harness, _ = run(cfg)
    paths = RunPaths(cfg.base_dir)

    assert result.success, result.error
    generated = [{"name": "fig_arch.png", "caption": "Overview."}]
    assert stubs.seen["plotting_figure_lists"] == [[]]
    assert stubs.seen["figures"] == generated
    assert read_json(paths.figures_info) == generated
    assert sorted(p.name for p in paths.figures_dir.iterdir()) == ["fig_arch.png", "info.json"]
    wiki = harness.project
    assert not wiki.exists("wiki/figures/teaser.md")
    assert "93.1" not in wiki.injected_context()
    # Leaving them out is said, with the way to get them in.
    ignored = [r.getMessage() for r in caplog.records if "--keep-provided-figures" in r.message]
    assert len(ignored) == 1 and "1 figure" in ignored[0]


def test_a_repeated_plotting_stage_leaves_no_figures_of_the_earlier_one(cfg, stubs, monkeypatch):
    figures = cfg.base_dir / "raw_materials" / "figures"
    figures.mkdir()
    (figures / "teaser.png").write_bytes(PNG)
    cfg.use_plotting = True
    cfg.keep_provided_figures = True
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
    # The record says the follow-up ran, so a resumed run neither repeats nor misses it.
    assert state_of(cfg)["stages"]["literature"]["after_done"] is True
    assert "after_done" not in state_of(cfg)["stages"]["outline"]


# -- bookkeeping -----------------------------------------------------------------


def test_state_json_follows_every_transition(cfg, stubs):
    result, harness, _ = run(cfg)

    during_outline = stubs.states["outline"]
    assert during_outline["status"] == "running"
    assert during_outline["stages"]["ingest"]["status"] == "completed"
    assert during_outline["stages"]["outline"] == {
        "status": "running",
        "attempts": 1,
        "started_at": during_outline["stages"]["outline"]["started_at"],
        "finished_at": None,
        "seconds": 0.0,
        "error": None,
        "outputs": [],
        "reused": False,
    }
    assert during_outline["stages"]["literature"] == {
        "status": "pending",
        "attempts": 0,
        "started_at": None,
        "finished_at": None,
        "seconds": 0.0,
        "error": None,
        "outputs": [],
        "reused": False,
    }
    during_refinement = stubs.states["refinement"]["stages"]
    assert during_refinement["section_writing"]["outputs"] == ["latex_writeup/raw_draft_paper.tex"]
    assert during_refinement["finalize"]["status"] == "pending"

    state = state_of(cfg)
    order = ["ingest", "outline", "literature", "section_writing", "refinement", "finalize"]
    assert state["status"] == "succeeded" and state["error"] is None
    assert state["version"] == 1
    # Timestamps are UTC, to the second, and ordered.
    stamp = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    assert stamp.fullmatch(state["started_at"]) and stamp.fullmatch(state["updated_at"])
    assert state["started_at"] <= state["updated_at"]
    for record in state["stages"].values():
        assert stamp.fullmatch(record["started_at"]) and stamp.fullmatch(record["finished_at"])
        assert state["started_at"] <= record["started_at"] <= record["finished_at"]
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


def test_on_stage_reports_each_stage_once_it_has_settled(cfg, stubs, monkeypatch, caplog):
    cfg.use_plotting = True
    stubs.plotting_delay = 0.1
    stubs.failures["plotting"] = [RuntimeError("boom"), RuntimeError("still boom")]
    seen: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def on_stage(name: str, record: dict[str, Any]) -> None:
        on_disk = {n: r["status"] for n, r in state_of(cfg)["stages"].items()}
        seen.append((name, dict(record), on_disk))
        if name == "outline":
            raise RuntimeError("callback bug")

    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result = run_paper(cfg, harness=make_harness(cfg), compute=FakeCompute(), on_stage=on_stage)

    # A callback that raises is logged and does not fail the run.
    assert result.success, result.error
    assert "on_stage callback for the outline stage failed: callback bug" in caplog.text
    assert [(name, record["status"]) for name, record, _ in seen] == [
        ("ingest", "completed"),
        ("outline", "completed"),
        ("literature", "completed"),
        ("plotting", "failed"),
        ("section_writing", "completed"),
        ("refinement", "completed"),
        ("finalize", "completed"),
    ]
    # Literature finished long before plotting gave up, yet it was only reported once
    # nothing was running any more: the callback may checkpoint the run directory.
    _, literature, on_disk = seen[2]
    assert on_disk["plotting"] == "failed" and on_disk["section_writing"] == "pending"
    assert literature == state_of(cfg)["stages"]["literature"]
    assert seen[3][1]["tolerated"] is True and seen[3][1]["attempts"] == 2

    # Stages reused from the earlier run are reported as well.
    StubAgents().install(monkeypatch)
    reused: list[tuple[str, bool]] = []
    resumed = run_paper(
        cfg,
        harness=make_harness(cfg),
        compute=FakeCompute(),
        on_stage=lambda name, record: reused.append((name, record["reused"])),
    )
    assert resumed.success
    assert reused == [(name, True) for name in pipeline.STAGES]


def test_on_stage_reports_the_stage_that_failed_the_run(cfg, stubs):
    stubs.failures["outline"] = [RuntimeError("first"), RuntimeError("second")]
    seen: list[tuple[str, str, int]] = []
    result = run_paper(
        cfg,
        harness=make_harness(cfg),
        compute=FakeCompute(),
        on_stage=lambda name, record: seen.append((name, record["status"], record["attempts"])),
    )

    assert not result.success
    # Once per stage, not once per attempt, and nothing for stages that never started.
    assert seen == [("ingest", "completed", 1), ("outline", "failed", 2)]


def test_usage_is_not_counted_twice_when_one_harness_serves_a_resumed_run(cfg, stubs, monkeypatch):
    stubs.failures["refinement"] = [RuntimeError("reviewer down"), RuntimeError("reviewer down")]
    harness = make_harness(cfg)
    # Spent before the run started: not this run's usage.
    harness.generate("warm up", agent="Probe", stage="setup", model="claude-fable-5-2")
    first, _, _ = run(cfg, harness=harness)
    assert not first.success

    StubAgents().install(monkeypatch)
    second, _, _ = run(cfg, harness=harness)

    assert second.success
    rows = {row["label"]: row for row in second.usage["by_label"]}
    assert "setup.Probe" not in rows
    # The outline call was made once, in the first invocation.
    assert rows["outline.OutlineAgent"]["requests"] == 1
    assert second.usage["total"]["requests"] == harness.usage.total().requests - 1
    assert read_json(RunPaths(cfg.base_dir).usage) == second.usage


def test_a_transcript_that_cannot_be_written_does_not_fail_the_run(cfg, stubs, caplog):
    harness = make_harness(cfg)
    real_record = harness.transcripts.record

    def record(event: dict[str, Any]) -> dict[str, Any]:
        if event.get("type") in ("run_start", "stage_start", "stage_end", "run_end"):
            raise OSError("No space left on device")
        return real_record(event)

    harness.transcripts.record = record
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, _, _ = run(cfg, harness=harness)

    assert result.success, result.error
    assert state_of(cfg)["status"] == "succeeded"
    assert "transcript entry 'stage_end' failed: No space left on device" in caplog.text


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
        "started_at": state["stages"]["section_writing"]["started_at"],
        "finished_at": state["stages"]["section_writing"]["finished_at"],
        "seconds": state["stages"]["section_writing"]["seconds"],
        "error": "RuntimeError: second",
        "outputs": [],
        "reused": False,
    }
    assert state["stages"]["section_writing"]["finished_at"] is not None
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
    cfg.keep_provided_figures = True
    stubs.failures["plotting"] = [RuntimeError("matplotlib boom"), RuntimeError("still boom")]
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, harness, _ = run(cfg)

    assert result.success, result.error
    assert stubs.count()["plotting"] == 2
    record = state_of(cfg)["stages"]["plotting"]
    assert record["status"] == "failed" and record["tolerated"] is True
    assert record["error"] == "RuntimeError: still boom"
    assert state_of(cfg)["status"] == "succeeded"
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


def unusable(compute: FakeCompute) -> Any:
    """A hook that makes ``compute`` report itself unusable, as after a token expired."""

    def hook(ctx: RunContext) -> None:
        compute.problems = ["Modal is not authenticated: run `modal setup`."]

    return hook


def test_a_backend_failure_is_retried_while_the_backend_is_usable(cfg, stubs):
    stubs.failures["section_writing"] = [ComputeError("the sandbox stopped")]
    result, _, _ = run(cfg)

    assert result.success
    assert result.stages["section_writing"]["attempts"] == 2


def test_an_unusable_backend_fails_the_run_instead_of_being_retried(cfg, stubs):
    compute = FakeCompute()
    stubs.hooks["section_writing"] = unusable(compute)
    stubs.failures["section_writing"] = [
        ComputeError("Modal is not authenticated."),
        ComputeError("Modal is not authenticated."),
    ]
    result = run_paper(cfg, harness=make_harness(cfg), compute=compute)

    assert not result.success
    # A second attempt would have paid for the whole draft again, to fail the same way.
    assert stubs.count()["section_writing"] == 1
    assert result.stages["section_writing"]["attempts"] == 1
    assert result.error.startswith("stage 'section_writing' failed after 1 attempt(s)")
    assert "Modal is not authenticated: run `modal setup`." in result.error
    assert state_of(cfg)["error"] == result.error


def test_a_backend_error_that_says_the_backend_is_unusable_is_not_retried(cfg, stubs):
    class Unauthenticated(ComputeError):
        """What only the failing call can know: the token is there, but Modal rejects it."""

        unusable = True

    # The backend's own check finds nothing wrong.
    stubs.failures["section_writing"] = [Unauthenticated("Modal rejected the token.")] * 2
    result, _, _ = run(cfg)

    assert not result.success
    assert stubs.count()["section_writing"] == 1
    assert result.error == (
        "stage 'section_writing' failed after 1 attempt(s): "
        "Unauthenticated: Modal rejected the token."
    )


def test_an_unusable_backend_is_not_tolerated_in_the_plotting_stage(cfg, stubs, monkeypatch):
    cfg.use_plotting = True
    compute = FakeCompute()
    stubs.hooks["plotting"] = unusable(compute)
    # The plotting agent reports the backend's error in its own words.
    lost = "none of the 1 planned figure(s) could be produced (fig_arch: not authenticated)"
    stubs.failures["plotting"] = [RuntimeError(lost), RuntimeError(lost)]
    result = run_paper(cfg, harness=make_harness(cfg), compute=compute)

    assert not result.success
    assert stubs.count()["plotting"] == 1
    assert "section_writing" not in stubs.calls
    record = state_of(cfg)["stages"]["plotting"]
    assert record["status"] == "failed" and "tolerated" not in record
    assert result.error.startswith("stage 'plotting' failed after 1 attempt(s)")
    assert "Modal is not authenticated: run `modal setup`." in result.error

    # Once the backend works again the same command makes the figures.
    again = StubAgents().install(monkeypatch)
    resumed, _, _ = run(cfg)
    assert resumed.success
    assert again.calls == ["plotting", "section_writing", "refinement"]
    assert again.seen["figures"] == [{"name": "fig_arch.png", "caption": "Overview."}]


def test_a_configuration_error_is_not_tolerated_in_the_plotting_stage(cfg, stubs):
    cfg.use_plotting = True
    wrapped = RuntimeError("no figure could be planned")
    wrapped.__cause__ = LLMConfigError("the plotting model cannot use tools")
    stubs.failures["plotting"] = [wrapped]
    result, _, _ = run(cfg)

    assert not result.success
    assert stubs.count()["plotting"] == 1
    assert "section_writing" not in stubs.calls
    assert "tolerated" not in result.stages["plotting"]
    assert result.error == (
        "stage 'plotting' failed after 1 attempt(s): RuntimeError: no figure could be planned "
        "(LLMConfigError: the plotting model cannot use tools)"
    )


def test_an_interrupt_is_recorded_and_propagates(cfg, stubs):
    stubs.failures["section_writing"] = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        run(cfg)

    state = state_of(cfg)
    assert state["status"] == "interrupted"
    assert state["stages"]["section_writing"]["status"] == "interrupted"
    assert state["stages"]["literature"]["status"] == "completed"


def test_an_interrupted_stage_leaves_what_it_spent_and_no_model_call_follows(cfg, stubs):
    def spend_then_interrupt(ctx: RunContext) -> None:
        ctx.harness.generate(
            "Write.", agent="SectionWritingAgent", stage="section_writing", model="fake"
        )
        raise KeyboardInterrupt

    stubs.hooks["section_writing"] = spend_then_interrupt
    harness = make_harness(cfg)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, harness)

    # The stage never settled, and what it spent is on the books all the same.
    rows = {row["label"]: row for row in read_json(RunPaths(cfg.base_dir).usage)["by_label"]}
    assert rows["section_writing.SectionWritingAgent"]["requests"] == 1
    # Whatever the stage left working on threads of its own gets no further model call.
    assert harness.cancelled


def test_a_closed_state_takes_no_further_updates(tmp_path):
    path = tmp_path / "state.json"
    stages = {"plotting": {"status": "running", "attempts": 1, "outputs": []}}
    state = pipeline._State(path, {"version": pipeline.STATE_VERSION}, stages)
    state.close("interrupted", "the run was interrupted")
    closed = path.read_text()
    assert read_json(path)["stages"]["plotting"]["status"] == "interrupted"

    # A stage that was still working when the run was closed reports in afterwards.
    state.update("plotting", status="completed", outputs=["plotting_results.json"])
    state.interrupt("stopped again")
    state.close("succeeded")
    assert path.read_text() == closed
    assert state.record("plotting")["status"] == "interrupted"
    assert state.closed


def test_the_interruption_is_on_record_before_the_stages_are_waited_for(cfg, stubs, monkeypatch):
    # Whoever stops the run may not wait as long as the run waits for its stages: the
    # frontend, a batch and a scheduler all kill the process after a grace period.
    cfg.use_plotting = True
    monkeypatch.setattr(pipeline, "STOP_WAIT_SECONDS", 4.0)
    plotting_started = threading.Event()
    seen: dict[str, Any] = {}

    class Interrupted(_Literature):
        def run(self, outline: dict[str, Any]) -> None:
            assert plotting_started.wait(timeout=5)
            self.ctx.harness.generate("Search.", agent="Agent", stage="literature", model="fake")
            raise KeyboardInterrupt

    class Busy(_Plotting):
        def run(self, outline: dict[str, Any]) -> list[dict[str, Any]]:
            # In the middle of a long model call when the run is stopped.
            paths = self.ctx.paths
            plotting_started.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                seen["state"] = read_json(paths.state)
                seen["usage"] = read_json(paths.usage)
                labels = {row["label"] for row in seen["usage"]["by_label"]}
                if seen["state"]["status"] == "interrupted" and "literature.Agent" in labels:
                    break
                time.sleep(0.01)
            seen["run_over"] = run_over.is_set()
            raise RuntimeError("the call came back")

    run_over = threading.Event()
    monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: Interrupted(stubs, ctx))
    monkeypatch.setattr(pipeline, "_plotting_agent", lambda ctx: Busy(stubs, ctx))
    try:
        with pytest.raises(KeyboardInterrupt):
            run(cfg)
        run_over.set()
    finally:
        wait_for_stage_threads()

    # Plotting looked while the run was still waiting for it.
    assert not seen["run_over"]
    assert seen["state"]["status"] == "interrupted"
    assert seen["state"]["error"] == "the run was interrupted"
    assert seen["state"]["stages"]["literature"]["status"] == "interrupted"
    assert seen["state"]["stages"]["plotting"]["status"] == "interrupted"
    rows = {row["label"]: row for row in seen["usage"]["by_label"]}
    assert rows["literature.Agent"]["requests"] == 1
    # A stage that stops within the wait still says how it ended.
    record = state_of(cfg)["stages"]["plotting"]
    assert record["status"] == "interrupted" and record["finished_at"] is not None


def test_an_interrupt_does_not_wait_for_stages_running_side_by_side(cfg, stubs, monkeypatch):
    cfg.use_plotting = True
    monkeypatch.setattr(pipeline, "STOP_WAIT_SECONDS", 0.2)
    plotting_started = threading.Event()
    release = threading.Event()
    plotting_attempts: list[str] = []

    class Interrupted(_Literature):
        def run(self, outline: dict[str, Any]) -> None:
            assert plotting_started.wait(timeout=5)
            raise KeyboardInterrupt

    class Stuck(_Plotting):
        def run(self, outline: dict[str, Any]) -> list[dict[str, Any]]:
            plotting_attempts.append("attempt")
            plotting_started.set()
            assert release.wait(timeout=10)
            raise RuntimeError("the sandbox went away")

    monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: Interrupted(stubs, ctx))
    monkeypatch.setattr(pipeline, "_plotting_agent", lambda ctx: Stuck(stubs, ctx))
    harness = make_harness(cfg)
    try:
        with pytest.raises(KeyboardInterrupt):
            run(cfg, harness)
        # The interrupt came through while plotting was still in its first attempt, and the
        # thread it runs on cannot keep the process alive.
        assert not release.is_set()
        assert [thread.name for thread in stage_threads()] == ["stage-plotting"]
        assert all(thread.daemon for thread in stage_threads())
        # Whatever is still running stops at its next model call.
        assert harness.cancelled
        state = state_of(cfg)
        assert state["status"] == "interrupted"
        assert state["stages"]["literature"]["status"] == "interrupted"
        assert state["stages"]["plotting"]["status"] == "interrupted"
    finally:
        release.set()
        wait_for_stage_threads()

    # Plotting ended the attempt it was in and started no retry.  What it did after the run
    # was closed is not recorded: the stage stays interrupted, and its end is no lesson.
    assert plotting_attempts == ["attempt"]
    state = state_of(cfg)
    assert state["status"] == "interrupted"
    assert state["stages"]["plotting"]["status"] == "interrupted"
    assert state["stages"]["plotting"]["attempts"] == 1
    assert "section_writing" not in stubs.calls
    assert harness.project.pending_notes() == []


def test_a_stage_that_ends_after_the_interrupt_is_not_recorded_as_completed(
    cfg, stubs, monkeypatch
):
    cfg.use_plotting = True
    monkeypatch.setattr(pipeline, "STOP_WAIT_SECONDS", 0.2)
    plotting_started = threading.Event()
    release = threading.Event()

    class Interrupted(_Literature):
        def run(self, outline: dict[str, Any]) -> None:
            super().run(outline)
            assert plotting_started.wait(timeout=5)
            raise KeyboardInterrupt

    class Slow(_Plotting):
        def run(self, outline: dict[str, Any]) -> list[dict[str, Any]]:
            plotting_started.set()
            assert release.wait(timeout=10)
            # The model call it was in when the run was stopped comes back.
            self.ctx.harness.usage.record("plotting.Agent", "fake", Usage(requests=1))
            return super().run(outline)

    monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: Interrupted(stubs, ctx))
    monkeypatch.setattr(pipeline, "_plotting_agent", lambda ctx: Slow(stubs, ctx))
    harness = make_harness(cfg)
    refreshes: list[str] = []
    refresh = harness.refresh
    harness.refresh = lambda: (refreshes.append("refresh"), refresh())[1]
    paths = RunPaths(cfg.base_dir)
    try:
        with pytest.raises(KeyboardInterrupt):
            run(cfg, harness)
        closed = {path: path.read_text() for path in (paths.state, paths.usage)}
        refreshed = len(refreshes)
    finally:
        release.set()
        wait_for_stage_threads()

    # Plotting wrote all of its files after the run had been closed as interrupted.  What
    # the run left when it was closed is final: by now a resumed run may own these files.
    assert paths.plotting_results.is_file()
    assert {path: path.read_text() for path in closed} == closed
    assert len(refreshes) == refreshed
    state = state_of(cfg)
    assert state["status"] == "interrupted"
    assert state["stages"]["plotting"]["status"] == "interrupted"
    assert state["stages"]["literature"]["status"] == "interrupted"

    # So the resumed run does both stages again, and literature gets its synthesis.
    again = StubAgents().install(monkeypatch)
    result, harness, _ = run(cfg)
    assert result.success, result.error
    assert again.calls[:2] in (["literature", "plotting"], ["plotting", "literature"])
    assert again.calls[2:] == ["section_writing", "refinement"]
    assert result.stages["outline"]["reused"] and not result.stages["literature"]["reused"]
    assert harness.project.exists("wiki/synthesis/themes.md")


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs POSIX signals")
def test_ctrl_c_stops_both_stages_at_their_next_model_call(cfg, stubs, monkeypatch):
    cfg.use_plotting = True
    made: Counter[str] = Counter()
    both_running = threading.Barrier(2)
    sent = threading.Event()

    def busy(stage: str) -> Any:
        def hook(ctx: RunContext) -> None:
            both_running.wait(timeout=5)
            # Four seconds of model calls, unless something stops them.
            for _ in range(400):
                ctx.harness.generate("Work.", agent="Agent", stage=stage, model="fake")
                made[stage] += 1
                if stage == "plotting" and made["literature"] and not sent.is_set():
                    # What the terminal does on Ctrl-C: the main thread gets the signal.
                    sent.set()
                    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
                time.sleep(0.01)

        return hook

    stubs.hooks["literature"] = busy("literature")
    stubs.hooks["plotting"] = busy("plotting")
    # A process started in the background ignores the signal unless it asks for it.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            run(cfg)
    finally:
        signal.signal(signal.SIGINT, previous or signal.SIG_DFL)

    # The interrupt was passed on once both stages had stopped, a few calls in.
    assert stage_threads() == []
    counts = dict(made)
    assert 0 < counts["plotting"] < 400 and 0 < counts["literature"] < 400
    time.sleep(0.05)
    assert dict(made) == counts
    state = state_of(cfg)
    assert state["status"] == "interrupted"
    for name in ("literature", "plotting"):
        record = state["stages"][name]
        assert record["status"] == "interrupted" and record["attempts"] == 1
        assert record["finished_at"] is not None and record["error"] is None
    assert "section_writing" not in stubs.calls
    # What the stages spent until they stopped is on the books.
    rows = {row["label"]: row for row in read_json(RunPaths(cfg.base_dir).usage)["by_label"]}
    assert rows["plotting.Agent"]["requests"] == counts["plotting"]
    assert rows["literature.Agent"]["requests"] == counts["literature"]

    # The same command picks the run up at the stages that were stopped.
    again = StubAgents().install(monkeypatch)
    result, harness, _ = run(cfg)
    assert result.success, result.error
    assert sorted(again.calls[:2]) == ["literature", "plotting"]
    assert again.calls[2:] == ["section_writing", "refinement"]
    assert result.stages["outline"]["reused"] is True
    assert harness.project.exists("wiki/synthesis/themes.md")
    # Usage of the interrupted invocation is carried.
    rows = {row["label"]: row for row in result.usage["by_label"]}
    assert rows["plotting.Agent"]["requests"] == counts["plotting"]


class ExitHooks:
    """Stands in for ``atexit``: what the process calls on its way out."""

    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def register(self, hook: Any) -> Any:
        self.hooks.append(hook)
        return hook

    def unregister(self, hook: Any) -> None:
        if hook in self.hooks:
            self.hooks.remove(hook)


@pytest.mark.parametrize("ends_first", ["stage", "process"])
def test_the_backend_is_closed_once_the_interrupted_stage_or_the_process_has_ended(
    cfg, stubs, monkeypatch, ends_first
):
    cfg.use_plotting = True
    monkeypatch.setattr(pipeline, "STOP_WAIT_SECONDS", 0.2)
    exits = ExitHooks()
    monkeypatch.setattr(pipeline, "atexit", exits)
    plotting_started = threading.Event()
    release = threading.Event()
    compute = FakeCompute()
    closed_while_running: list[int] = []

    class Interrupted(_Literature):
        def run(self, outline: dict[str, Any]) -> None:
            assert plotting_started.wait(timeout=5)
            raise KeyboardInterrupt

    class Stuck(_Plotting):
        def run(self, outline: dict[str, Any]) -> list[dict[str, Any]]:
            plotting_started.set()
            assert release.wait(timeout=10)
            # The stage is still using the backend here.
            closed_while_running.append(compute.closed)
            raise RuntimeError("stopped")

    monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: Interrupted(stubs, ctx))
    monkeypatch.setattr(pipeline, "_plotting_agent", lambda ctx: Stuck(stubs, ctx))
    monkeypatch.setattr(pipeline, "_build_harness", make_harness)
    monkeypatch.setattr(pipeline, "_build_compute", lambda config: compute)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_paper(cfg)
        assert compute.closed == 0
        if ends_first == "process":
            # Plotting is still in its job when the process ends.  What the backend holds
            # (sandboxes on a remote service, say) must not stay behind.
            assert len(exits.hooks) == 1
            exits.hooks[0]()
            assert compute.closed == 1
    finally:
        release.set()
        wait_for_stage_threads()
        for thread in threading.enumerate():
            if thread.name == "compute-close":
                thread.join(5)

    if ends_first == "stage":
        assert closed_while_running == [0]
    assert compute.closed == 1
    # Nothing is left for the end of the process to do, or to keep the backend alive for.
    assert exits.hooks == []


def test_a_cancelled_run_is_an_interruption_not_a_failed_stage(cfg, stubs):
    harness = make_harness(cfg)

    def cancel(ctx: RunContext) -> None:
        ctx.harness.cancel()
        ctx.harness.generate(
            "Write.", agent="SectionWritingAgent", stage="section_writing", model="fake"
        )

    stubs.hooks["section_writing"] = cancel
    with pytest.raises(RunCancelled):
        run(cfg, harness)

    assert stubs.count()["section_writing"] == 1
    state = state_of(cfg)
    assert state["status"] == "interrupted"
    record = state["stages"]["section_writing"]
    assert record["status"] == "interrupted" and record["attempts"] == 1
    assert record["error"] is None
    assert state["stages"]["literature"]["status"] == "completed"
    # Being stopped is not something a later run should learn from.
    assert harness.project.pending_notes() == []
    ends = [e["status"] for e in events(harness, "stage_end") if e["stage"] == "section_writing"]
    assert ends == ["interrupted"]


def test_a_stage_that_returns_after_the_run_was_cancelled_is_not_trusted(cfg, stubs):
    # Agents skip over a failed model call in places, so a stage can return normally with
    # less than its full work after the run was stopped.
    stubs.hooks["section_writing"] = lambda ctx: ctx.harness.cancel()
    with pytest.raises(RunCancelled):
        run(cfg)

    assert RunPaths(cfg.base_dir).raw_draft_tex.is_file()
    state = state_of(cfg)
    assert state["status"] == "interrupted"
    assert state["stages"]["section_writing"]["status"] == "interrupted"
    assert "refinement" not in stubs.calls


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


def test_an_earlier_research_cutoff_invalidates_what_was_collected(cfg, stubs, monkeypatch, caplog):
    cfg.research_cutoff = "2025-06"
    run(cfg)

    # A later cutoff (the default moves on every month) keeps the collected literature valid.
    cfg.research_cutoff = "2025-09"
    later = StubAgents().install(monkeypatch)
    assert run(cfg)[0].success and later.calls == []

    # An earlier one does not: papers from the months in between may have been cited.
    cfg.research_cutoff = "2025-03"
    earlier = StubAgents().install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, _, _ = run(cfg)
    assert result.success
    assert earlier.calls == ["outline", "literature", "section_writing", "refinement"]
    assert not any(record["reused"] for record in result.stages.values())
    assert "research cutoff is now 2025-03" in caplog.text
    assert state_of(cfg)["config"]["research_cutoff"] == "2025-03"


def test_switching_plotting_on_or_off_starts_over(cfg, stubs, monkeypatch, caplog):
    provide_figure(cfg)
    _, harness, _ = run(cfg)
    assert stubs.seen["figures"] == [{"name": "teaser.png", "caption": "The teaser."}]
    assert harness.project.exists("wiki/figures/teaser.md")

    # With plotting the figures are generated, so the workspace and the memory that were
    # built around the provided ones do not stand.
    cfg.use_plotting = True
    plotted = StubAgents().install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, harness, _ = run(cfg)
    assert result.success
    assert sorted(plotted.calls[1:3]) == ["literature", "plotting"]
    assert plotted.count() == {
        "outline": 1,
        "literature": 1,
        "plotting": 1,
        "section_writing": 1,
        "refinement": 1,
    }
    assert not any(record["reused"] for record in result.stages.values())
    assert "use_plotting: False -> True" in caplog.text
    assert plotted.seen["figures"] == [{"name": "fig_arch.png", "caption": "Overview."}]
    assert not harness.project.exists("wiki/figures/teaser.md")

    # And back: the generated figure must not stay in the paper.
    cfg.use_plotting = False
    plain = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success
    assert plain.calls == ["outline", "literature", "section_writing", "refinement"]
    assert plain.seen["figures"] == [{"name": "teaser.png", "caption": "The teaser."}]
    assert "plotting" not in state_of(cfg)["stages"]
    paths = RunPaths(cfg.base_dir)
    assert not paths.plotting_results.exists() and not paths.generated_figures_dir.exists()
    assert sorted(p.name for p in paths.figures_dir.iterdir()) == ["info.json", "teaser.png"]


def use_another_idea_file(cfg: PipelineConfig) -> None:
    # The same text under another name: the inputs' digest does not change.
    (cfg.base_dir / "raw_materials" / "idea_dense.md").write_text(IDEA)
    cfg.idea_filename = "idea_dense.md"


SETTING_CHANGES = {
    "idea_filename": use_another_idea_file,
    "models.writer": lambda cfg: setattr(cfg.models, "writer", "claude-other"),
    "models.reflection": lambda cfg: setattr(cfg.models, "reflection", "claude-other"),
    "models.fast": lambda cfg: setattr(cfg.models, "fast", "claude-small"),
    "models.effort": lambda cfg: setattr(cfg.models, "effort", "max"),
    "models.fast_effort": lambda cfg: setattr(cfg.models, "fast_effort", "medium"),
    "max_reflections": lambda cfg: setattr(cfg, "max_reflections", 5),
    "retry_after_rejection": lambda cfg: setattr(cfg, "retry_after_rejection", False),
    "max_formatting_rounds": lambda cfg: setattr(cfg, "max_formatting_rounds", 2),
    "review_ensemble": lambda cfg: setattr(cfg, "review_ensemble", 1),
    "max_repair_rounds": lambda cfg: setattr(cfg, "max_repair_rounds", 1),
    "memory.enabled": lambda cfg: setattr(cfg.memory, "enabled", False),
    "memory.lab": lambda cfg: setattr(cfg.memory, "lab", True),
    "memory.dream": lambda cfg: setattr(cfg.memory, "dream", "off"),
}


@pytest.mark.parametrize("setting", sorted(SETTING_CHANGES))
def test_a_changed_result_setting_starts_over(cfg, stubs, monkeypatch, caplog, setting):
    first, _, _ = run(cfg)
    assert first.success
    before = cfg.result_settings()[setting]

    SETTING_CHANGES[setting](cfg)
    again = StubAgents().install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, _, _ = run(cfg)

    assert result.success
    # The paper on disk was written under the earlier setting: nothing of it is passed off
    # as the work of this configuration.
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]
    assert not any(record["reused"] for record in result.stages.values())
    warning = next(r.getMessage() for r in caplog.records if "every stage runs again" in r.message)
    assert f"{setting}: {before!r} -> {cfg.result_settings()[setting]!r}" in warning
    assert result_settings(state_of(cfg)["config"]) == cfg.result_settings()


def test_a_stage_left_over_from_a_failed_run_is_not_finished_by_another_model(
    cfg, stubs, monkeypatch
):
    stubs.failures["refinement"] = [RuntimeError("reviewer down"), RuntimeError("reviewer down")]
    assert not run(cfg)[0].success

    cfg.models.reflection = "claude-other"
    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]


def test_settings_that_do_not_shape_the_paper_keep_the_resume(cfg, stubs, monkeypatch, tmp_path):
    cfg.research_cutoff = "2025-06"
    assert run(cfg)[0].success

    cfg.compute = "modal"
    cfg.max_stage_attempts = 5
    cfg.literature_workers = 1
    cfg.figure_workers = 1
    cfg.figure_code_timeout = 10.0
    cfg.latex_timeout = 999.0
    cfg.memory.lab_dir = tmp_path / "another-lab"
    cfg.memory.lab_consolidate = False
    cfg.research_cutoff = "2025-09"
    # Plotting is off, so its options are not in play.
    cfg.models.plotting = "claude-other"
    cfg.models.image = "some-image-model"
    cfg.plotting_max_critic_rounds = 9
    cfg.keep_provided_figures = True

    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success and again.calls == []
    assert all(record["reused"] for record in result.stages.values())


def test_a_state_file_without_the_settings_of_its_run_is_not_resumed(cfg, stubs, monkeypatch):
    run(cfg)
    state = state_of(cfg)
    del state["config"]["models"]
    write_json(cfg.base_dir / "state.json", state)

    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]


def test_keeping_the_provided_figures_is_a_setting_of_the_run(cfg, stubs, monkeypatch):
    provide_figure(cfg)
    cfg.use_plotting = True
    run(cfg)
    assert stubs.seen["figures"] == [{"name": "fig_arch.png", "caption": "Overview."}]

    cfg.keep_provided_figures = True
    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success
    assert again.count()["outline"] == 1 and again.count()["plotting"] == 1
    assert again.seen["figures"] == [
        {"name": "teaser.png", "caption": "The teaser."},
        {"name": "fig_arch.png", "caption": "Overview."},
    ]


def test_figures_that_are_not_used_do_not_invalidate_a_resume(cfg, stubs, monkeypatch):
    provide_figure(cfg)
    cfg.use_plotting = True
    run(cfg)
    (cfg.base_dir / "raw_materials" / "figures" / "later.png").write_bytes(PNG)
    again = StubAgents().install(monkeypatch)
    assert run(cfg)[0].success and again.calls == []

    # Once they are part of the run, a change to them is a change of the inputs.
    cfg.use_plotting = False
    run(cfg)
    (cfg.base_dir / "raw_materials" / "figures" / "more.png").write_bytes(PNG)
    changed = StubAgents().install(monkeypatch)
    assert run(cfg)[0].success
    assert changed.calls == ["outline", "literature", "section_writing", "refinement"]


def test_a_tolerated_plotting_failure_is_retried_when_the_run_did_not_finish(
    cfg, stubs, monkeypatch
):
    cfg.use_plotting = True
    stubs.failures["plotting"] = [RuntimeError("no display"), RuntimeError("no display")]
    stubs.failures["section_writing"] = [RuntimeError("compile"), RuntimeError("compile")]
    first, _, _ = run(cfg)
    assert not first.success
    assert first.stages["plotting"]["tolerated"] is True

    # The run never delivered a paper without figures, so going on without them was not a
    # result anyone accepted: plotting gets another chance.
    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success
    assert again.calls == ["plotting", "section_writing", "refinement"]
    assert result.stages["literature"]["reused"] is True
    plotting = result.stages["plotting"]
    assert plotting["status"] == "completed" and "tolerated" not in plotting
    assert again.seen["figures"] == [{"name": "fig_arch.png", "caption": "Overview."}]


def test_the_literature_synthesis_is_made_up_for_when_the_stage_is_reused(cfg, stubs, monkeypatch):
    harness = make_harness(cfg)
    real_dream = harness.dream

    def killed_during_the_synthesis(**kwargs: Any) -> Any:
        if kwargs["stage"] == "literature":
            raise KeyboardInterrupt
        return real_dream(**kwargs)

    harness.dream = killed_during_the_synthesis
    with pytest.raises(KeyboardInterrupt):
        run(cfg, harness)
    record = state_of(cfg)["stages"]["literature"]
    assert record["status"] == "completed" and "after_done" not in record

    again = StubAgents().install(monkeypatch)
    result, harness, _ = run(cfg)
    assert result.success
    assert again.calls == ["section_writing", "refinement"]
    assert result.stages["literature"]["reused"] is True
    assert result.stages["literature"]["after_done"] is True
    assert harness.project.exists("wiki/synthesis/themes.md")
    assert [e["stage"] for e in events(harness, "dream")] == ["literature", "finalize"]


def test_a_synthesis_that_did_not_complete_is_tried_again_on_resume(cfg, stubs, monkeypatch):
    harness = make_harness(cfg)
    real_dream = harness.dream

    def busy(**kwargs: Any) -> Any:
        if kwargs["stage"] == "literature":
            return DreamReport(
                dream_id="d1",
                scope="project",
                stage="literature",
                error="another consolidation of this wiki is in progress",
            )
        return real_dream(**kwargs)

    harness.dream = busy
    stubs.failures["refinement"] = [RuntimeError("reviewer down"), RuntimeError("reviewer down")]
    first, _, _ = run(cfg, harness)
    assert not first.success
    assert "after_done" not in first.stages["literature"]

    again = StubAgents().install(monkeypatch)
    result, harness, _ = run(cfg)
    assert result.success and again.calls == ["refinement"]
    assert result.stages["literature"]["after_done"] is True
    assert harness.project.exists("wiki/synthesis/themes.md")

    # A finished run is left alone, also when the synthesis never came about.
    state = state_of(cfg)
    del state["stages"]["literature"]["after_done"]
    write_json(cfg.base_dir / "state.json", state)
    client = FakeLLM(responder)
    assert run(cfg, make_harness(cfg, client))[0].success
    assert client.requests == []


def test_reused_ingest_writes_its_source_pages_again_when_the_wiki_lacks_them(
    cfg, stubs, monkeypatch
):
    provide_figure(cfg)
    stubs.failures["refinement"] = [RuntimeError("reviewer down"), RuntimeError("reviewer down")]
    first, _, _ = run(cfg)
    assert not first.success
    shutil.rmtree(cfg.project_memory_dir)

    again = StubAgents().install(monkeypatch)
    client = FakeLLM(responder)
    result, harness, _ = run(cfg, make_harness(cfg, client))

    assert result.success and again.calls == ["refinement"]
    assert result.stages["ingest"]["reused"] is True
    wiki = harness.project
    assert wiki.read("wiki/sources/idea.md").body == IDEA
    assert wiki.read("wiki/sources/experimental_log.md").body == LOG
    assert wiki.exists("wiki/sources/guidelines.md") and wiki.exists("wiki/sources/template.md")
    assert wiki.read("wiki/figures/teaser.md").meta["origin"] == "provided"
    # Writing them back is plain file work.
    assert client.requests == []

    # Pages that are in place are left as they are.
    marker = wiki.read("wiki/sources/idea.md").meta["updated"]
    assert run(cfg, make_harness(cfg))[0].success
    assert Wiki(cfg.project_memory_dir).read("wiki/sources/idea.md").meta["updated"] == marker


def test_resume_counts_stages_that_ran_side_by_side_once(cfg, stubs, monkeypatch):
    cfg.use_plotting = True
    stubs.failures["refinement"] = [RuntimeError("reviewer down"), RuntimeError("reviewer down")]
    assert not run(cfg)[0].success
    # As if literature had taken 100 s and plotting, next to it, 60 s.
    state = state_of(cfg)
    state["stages"]["literature"]["seconds"] = 100.0
    state["stages"]["plotting"]["seconds"] = 60.0
    write_json(cfg.base_dir / "state.json", state)

    StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success
    breakdown = read_json(RunPaths(cfg.base_dir).time_breakdown)
    assert breakdown["Parallel_Lit_and_Plot"] == 100.0
    # The pair took 100 s of the run's time, not 160 s.
    assert 100.0 <= breakdown["Total_Time_Span"] < 150.0


def test_usage_is_written_while_a_stage_is_still_running(cfg, stubs, monkeypatch):
    monkeypatch.setattr(pipeline, "USAGE_FLUSH_SECONDS", 0.02)
    usage_path = RunPaths(cfg.base_dir).usage
    seen: dict[str, Any] = {}

    def spend(ctx: RunContext) -> None:
        ctx.harness.generate(
            "Write.", agent="SectionWritingAgent", stage="section_writing", model="fake"
        )
        # The stage has not settled, and would lose this if the process were killed now.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            labels = {row["label"] for row in read_json(usage_path)["by_label"]}
            if "section_writing.SectionWritingAgent" in labels:
                break
            time.sleep(0.01)
        seen["labels"] = labels
        seen["stage"] = read_json(ctx.paths.state)["stages"]["section_writing"]["status"]

    stubs.hooks["section_writing"] = spend
    result, _, _ = run(cfg)

    assert result.success
    assert seen["stage"] == "running"
    assert "section_writing.SectionWritingAgent" in seen["labels"]
    assert read_json(usage_path) == result.usage


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


def test_the_backend_is_asked_whether_it_can_do_the_work_before_any_stage(cfg, stubs):
    compute = FakeCompute()
    compute.problems = ["Modal is not authenticated: run `modal setup`."]
    result = run_paper(cfg, harness=make_harness(cfg), compute=compute)

    assert not result.success
    assert result.error == "Modal is not authenticated: run `modal setup`."
    assert stubs.calls == []
    # Figure code only runs with plotting.
    assert compute.checks == [{"python": False, "latex": True}]

    cfg.use_plotting = True
    run_paper(cfg, harness=make_harness(cfg), compute=compute)
    assert compute.checks[-1] == {"python": True, "latex": True}


def test_a_backend_check_that_raises_fails_the_run_before_any_stage(cfg, stubs):
    class Broken(FakeCompute):
        def check(self, *, python: bool = True, latex: bool = True) -> list[str]:
            raise ComputeError("the Modal SDK is not installed")

    result = run_paper(cfg, harness=make_harness(cfg), compute=Broken())

    assert not result.success
    assert result.error == "the Modal SDK is not installed"
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
    assert seen[0] == "running" and seen[-1] == "succeeded" and len(seen) > 12


def test_a_retried_literature_stage_keeps_the_record_of_verified_papers(cfg, stubs):
    from paper_orchestra.agents.literature import CHECKPOINT_NAME

    seen: list[str | None] = []

    def literature(ctx: RunContext) -> None:
        checkpoint = ctx.paths.literature_dir / CHECKPOINT_NAME
        seen.append(checkpoint.read_text() if checkpoint.is_file() else None)
        # What an earlier attempt leaves besides the record must still be cleared.
        assert not (ctx.paths.literature_dir / "citation_map.json").exists()
        ctx.paths.literature_dir.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"verified in attempt {len(seen)}")
        (ctx.paths.literature_dir / "citation_map.json").write_text("{}")

    stubs.hooks["literature"] = literature
    stubs.failures["literature"] = [RuntimeError("the writer was refused")]
    result, _, _ = run(cfg)
    assert result.success and result.stages["literature"]["attempts"] == 2
    assert seen == [None, "verified in attempt 1"]

    # Starting over plans a new outline, and with it a new search.
    cfg.resume = False
    seen.clear()
    assert run(cfg)[0].success
    assert seen == [None]


# -- what invalidates a resume, and what a failed re-run leaves behind ---------------------


@pytest.mark.parametrize("change", ["experimental_log", "figure", "figure_caption"])
def test_changed_results_or_figures_start_over(cfg, stubs, monkeypatch, change):
    figures = cfg.base_dir / "raw_materials" / "figures"
    figures.mkdir()
    (figures / "teaser.png").write_bytes(PNG)
    (figures / "info.json").write_text(json.dumps([{"name": "teaser.png", "caption": "old"}]))
    run(cfg)
    if change == "experimental_log":
        (cfg.base_dir / "raw_materials" / "experimental_log.md").write_text(LOG + "Now 93.4.\n")
    elif change == "figure":
        (figures / "teaser.png").write_bytes(PNG + b"changed")
    else:
        (figures / "info.json").write_text(json.dumps([{"name": "teaser.png", "caption": "new"}]))
    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    # New results or a replaced figure must never be answered with the old paper.
    assert result.success
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]
    assert not any(record["reused"] for record in result.stages.values())


def test_a_failed_rerun_does_not_leave_the_earlier_paper_as_its_result(cfg, stubs, monkeypatch):
    run(cfg)
    paths = RunPaths(cfg.base_dir)
    assert paths.final_pdf.is_file() and paths.raw_draft_tex.is_file()
    (paths.latex_dir / "leftover_from_the_first_run.tex").write_text("old")
    (paths.literature_dir / "leftover.json").write_text("{}")
    (cfg.base_dir / "raw_materials" / "idea_sparse.md").write_text("# A different idea\n")

    again = StubAgents().install(monkeypatch)
    again.failures["section_writing"] = [RuntimeError("down"), RuntimeError("down")]
    result, _, _ = run(cfg)
    assert not result.success
    # Anyone looking at the output directory must not take the old paper for the result.
    for stale in (paths.final_pdf, paths.final_tex, paths.time_breakdown):
        assert not stale.exists(), stale
    assert not (cfg.base_dir / "memory_report.json").exists()
    assert not paths.raw_draft_tex.exists()  # the first run's draft is not this run's
    assert not (paths.latex_dir / "leftover_from_the_first_run.tex").exists()
    assert not (paths.literature_dir / "leftover.json").exists()


def test_an_interrupted_or_killed_run_resumes_where_it_stopped(cfg, stubs, monkeypatch):
    stubs.failures["section_writing"] = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    assert state_of(cfg)["status"] == "interrupted"

    again = StubAgents().install(monkeypatch)
    # A kill (SIGKILL, preemption) cannot even record that the run was stopped.
    again.failures["refinement"] = [KeyboardInterrupt()]
    with monkeypatch.context() as patch:
        patch.setattr(pipeline._State, "interrupt", lambda self, error: None)
        patch.setattr(pipeline._State, "close", lambda self, status, error=None: None)
        with pytest.raises(KeyboardInterrupt):
            run(cfg)
    assert again.calls == ["section_writing", "refinement"]
    assert state_of(cfg)["status"] == "running"

    last = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)
    assert result.success and last.calls == ["refinement"]
    assert state_of(cfg)["status"] == "succeeded"


# -- a paper written from a research project: the papers its wiki already knows ---------------

#: The file ``research/writeup.py`` exports next to the idea and the experimental log.
KNOWN_PAPERS = "known_papers.json"
WARMUP = {
    "citation_key": "Aaa2020Warmup",
    "title": "Warmup, Measured",
    "authors": ["A. Aaa", "B. Bbb"],
    "venue": "ICML",
    "year": 2020,
    "abstract": "We measure warmup at small batch sizes.",
    "citation_count": 12,
    "found_in_section": "",
    "sections": [],
    "reason": "A source the researcher accepted in the project's research wiki.",
    "journal": None,
    "volume": None,
    "pages": None,
    "publication_date": None,
    "link": "https://example.org/aaa2020",
    "record": "wiki/sources/Aaa2020Warmup.md",
}
SHALLOW = WARMUP | {
    "citation_key": "Jones2019Shallow",
    "title": "Shallow Things",
    "authors": ["Jo Jones"],
    "year": 2019,
    "record": "wiki/sources/Jones2019Shallow.md",
}
NOT_SHALLOW = {
    "citation_key": "Jones2019Shallow",
    "title": "Shallow Things",
    "reason": "Not about attention.",
}


def give_known(cfg: PipelineConfig, papers=(), excluded=()) -> Path:
    path = cfg.base_dir / "raw_materials" / KNOWN_PAPERS
    write_json(path, {"version": 1, "papers": list(papers), "excluded": list(excluded)})
    return path


@pytest.mark.parametrize("change", ["accepted another", "rejected another", "gone", "appeared"])
def test_a_changed_list_of_known_papers_starts_over(cfg, stubs, monkeypatch, change):
    if change != "appeared":
        give_known(cfg, [WARMUP])
    run(cfg)
    if change == "accepted another":
        give_known(cfg, [WARMUP, SHALLOW])
    elif change == "rejected another":
        give_known(cfg, [WARMUP], [NOT_SHALLOW])
    elif change == "gone":
        (cfg.base_dir / "raw_materials" / KNOWN_PAPERS).unlink()
    else:
        give_known(cfg, [WARMUP])

    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)

    # The paper on disk was written from other sources: it is not this list's paper.
    assert result.success
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]
    assert not any(record["reused"] for record in result.stages.values())


def test_an_unchanged_list_of_known_papers_keeps_the_resume(cfg, stubs, monkeypatch):
    give_known(cfg, [WARMUP], [NOT_SHALLOW])
    run(cfg)
    give_known(cfg, [WARMUP], [NOT_SHALLOW])  # exported again by the next write-up, the same

    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)

    assert result.success and again.calls == []
    assert all(record["reused"] for record in result.stages.values())


#: An accepted source of June 2021: too recent for a paper whose cutoff is January 2021.
MIDYEAR = WARMUP | {"year": 2021, "publication_date": "2021-06-01"}


def test_a_later_cutoff_that_lets_in_a_known_paper_starts_over(cfg, stubs, monkeypatch, caplog):
    give_known(cfg, [SHALLOW, MIDYEAR])
    cfg.research_cutoff = "2021-01"
    run(cfg)

    # Later, but not late enough for the paper that was left out: the paper stands.
    cfg.research_cutoff = "2021-06"
    same = StubAgents().install(monkeypatch)
    assert run(cfg)[0].success and same.calls == []

    # The person moves the cutoff as the warning about the source told them to.  What was
    # collected is still valid, but the paper on disk is not the one this list gives now.
    cfg.research_cutoff = "2022-01"
    again = StubAgents().install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        result, _, _ = run(cfg)
    assert result.success
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]
    assert not any(record["reused"] for record in result.stages.values())
    assert "research cutoff is now 2022-01" in caplog.text and "Aaa2020Warmup" in caplog.text

    # And with no cutoff at all it is the same question.
    cfg.research_cutoff = "2021-01"
    run(cfg)
    cfg.research_cutoff = ""
    unbounded = StubAgents().install(monkeypatch)
    assert run(cfg)[0].success
    assert unbounded.calls == ["outline", "literature", "section_writing", "refinement"]


def test_the_source_a_moved_cutoff_lets_in_is_cited(cfg, stubs, monkeypatch):
    from paper_orchestra.agents.literature import LiteratureAgent
    from paper_orchestra.utils.latex import bib_keys

    scholar = _TinyScholar()
    monkeypatch.setattr(pipeline, "_outline_agent", lambda ctx: _SearchingOutline(stubs, ctx))
    monkeypatch.setattr(
        pipeline, "_literature_agent", lambda ctx: LiteratureAgent(ctx, scholar=scholar)
    )
    client = FakeLLM(literature_responder)
    give_known(cfg, [MIDYEAR])

    cfg.research_cutoff = "2021-01"
    first, _, _ = run(cfg, harness=make_harness(cfg, client))
    assert first.success and "Aaa2020Warmup" not in bib_keys(stubs.seen["writer_bib"])
    (warning,) = first.stages["literature"]["warnings"]
    assert "2021-06-01" in warning and "Move the cutoff" in warning

    cfg.research_cutoff = "2022-01"
    moved, _, _ = run(cfg, harness=make_harness(cfg, client))
    assert moved.success and "Aaa2020Warmup" in bib_keys(stubs.seen["writer_bib"])
    # Nothing is said any more about a cutoff that is no longer the run's.
    assert "warnings" not in moved.stages["literature"]
    assert moved.stages["literature"]["known_papers"] == 1


def test_a_list_of_known_papers_that_cannot_be_opened_does_not_stop_the_run(
    cfg, stubs, monkeypatch
):
    listed = give_known(cfg, [WARMUP])
    run(cfg)
    readable = state_of(cfg)["inputs"]
    read_bytes = Path.read_bytes

    def refuse(self: Path) -> bytes:
        if self == listed:
            raise PermissionError(13, "Permission denied", str(self))
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", refuse)
    again = StubAgents().install(monkeypatch)
    result, _, _ = run(cfg)

    # The run goes on (the literature stage says what it could not read), and it is not
    # the run of the list that could be read: every stage runs, now and when it can again.
    assert result.success
    assert again.calls == ["outline", "literature", "section_writing", "refinement"]
    assert state_of(cfg)["inputs"] != readable
    listed.unlink()
    removed = StubAgents().install(monkeypatch)
    assert run(cfg)[0].success and removed.calls == again.calls


def test_the_inputs_of_a_run_without_known_papers_are_fingerprinted_as_they_were(cfg, stubs):
    import hashlib

    run(cfg)
    materials = cfg.base_dir / "raw_materials"
    files = [
        ("idea", materials / "idea_sparse.md"),
        ("experimental_log", materials / "experimental_log.md"),
    ]
    files += [
        (f"template/{path.relative_to(cfg.template_dir).as_posix()}", path)
        for path in sorted(p for p in cfg.template_dir.rglob("*") if p.is_file())
    ]
    digest = hashlib.sha256()
    for label, path in files:
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    # A run that is under way when the package is upgraded resumes where it stopped.
    assert state_of(cfg)["inputs"] == digest.hexdigest()


def test_the_literature_record_says_what_the_agent_reported(cfg, stubs, monkeypatch):
    from types import SimpleNamespace

    problem = f"{KNOWN_PAPERS} is not valid JSON; the paper's literature is searched without it"
    reports = [SimpleNamespace(warnings=[problem], known=2, excluded=1)]

    class Reporting(_Literature):
        def run(self, outline: dict[str, Any]) -> Any:
            super().run(outline)
            return reports.pop(0)

    monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: Reporting(stubs, ctx))
    result, _, _ = run(cfg)

    record = result.stages["literature"]
    assert record["warnings"] == [problem]
    assert (record["known_papers"], record["excluded_candidates"]) == (2, 1)
    assert state_of(cfg)["stages"]["literature"] == record
    # Nothing of the kind is said about a stage that had nothing to report.
    assert set(result.stages["outline"]) == set(pipeline._blank_record())

    # A resumed run still says it: the stage's result is the earlier one's.
    again = StubAgents().install(monkeypatch)
    resumed, _, _ = run(cfg)
    assert again.calls == []
    assert resumed.stages["literature"] == record | {"reused": True}

    # And a stage that runs again says what it found this time, not what it found before.
    cfg.resume = False
    reports.append(SimpleNamespace(warnings=[], known=0, excluded=0))
    monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: Reporting(again, ctx))
    fresh, _, _ = run(cfg)
    assert fresh.success
    assert set(fresh.stages["literature"]) == set(pipeline._blank_record()) | {"after_done"}


def test_what_a_failed_attempt_reported_is_not_said_of_the_attempt_that_succeeded(
    cfg, stubs, monkeypatch
):
    from types import SimpleNamespace

    reports = [SimpleNamespace(warnings=["only true of the first attempt"], known=5, excluded=5)]

    class Reporting(_Literature):
        def run(self, outline: dict[str, Any]) -> Any:
            super().run(outline)
            return reports.pop(0) if reports else None

    monkeypatch.setattr(pipeline, "_literature_agent", lambda ctx: Reporting(stubs, ctx))
    # The first attempt gets as far as reporting, then turns out to lack one of its files.
    real_copy, copies = pipeline._atomic_copy, []

    def copy(source: Path, target: Path) -> None:
        copies.append(source.name)
        if source.name == "references.bib" and copies.count("references.bib") == 1:
            source.unlink()
            raise FileNotFoundError(str(source))
        real_copy(source, target)

    monkeypatch.setattr(pipeline, "_atomic_copy", copy)
    result, _, _ = run(cfg)

    record = result.stages["literature"]
    assert result.success and record["attempts"] == 2
    assert not {"warnings", "known_papers", "excluded_candidates"} & set(record)


class _SearchingOutline(_Outline):
    """An outline that asks for one literature search."""

    def run(self) -> dict[str, Any]:
        outline = super().run()
        outline["intro_related_work_plan"]["introduction_strategy"] = {
            "problem_gap_hypothesis": "Attention is dense.",
            "search_directions": ["sparse attention"],
        }
        write_json(self.ctx.paths.outline, outline)
        return outline


#: Title -> first author and year of the papers :class:`_TinyScholar` knows.
TINY_LIBRARY = {"Deep Things": ("Sam Smith", 2020), "Shallow Things": ("Jo Jones", 2019)}


class _TinyScholar:
    """Semantic Scholar with two papers, found under exactly their titles."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def find_paper(self, title: str, *, year_hint: int | None = None, cutoff: str | None = None):
        self.asked.append(title)
        if title not in TINY_LIBRARY:
            return None
        author, year = TINY_LIBRARY[title]
        return {
            "title": title,
            "authors": [{"name": author}],
            "venue": "NeurIPS",
            "year": year,
            "abstract": f"On {title.lower()}.",
            "citationCount": 3,
            "journal": None,
            "publicationDate": f"{year}-05-01",
        }


def literature_responder(request: FakeRequest) -> str:
    """The models of the real literature agent; every other call as :func:`responder`."""
    if request.label == "literature.discovery":
        titles = ["Deep Things", "Shallow Things", "Warmup, Measured"]
        return json.dumps(
            {"candidates": [{"title": title, "year": None, "reason": "r"} for title in titles]}
        )
    if request.label == "literature.LiteratureAgent":
        block = re.search(r"<citation_checklist>\n(.*?)\n</citation_checklist>", request.text, re.S)
        keys = ", ".join(block.group(1).split())
        return "```latex\n" + TEMPLATE.replace("% INTRO", f"Prior work \\cite{{{keys}}}.") + "```\n"
    return responder(request)


def test_a_paper_written_from_a_project_starts_from_its_sources_and_follows_them(
    cfg, stubs, monkeypatch
):
    from paper_orchestra.agents.literature import LiteratureAgent
    from paper_orchestra.utils.latex import bib_keys

    scholar = _TinyScholar()
    monkeypatch.setattr(pipeline, "_outline_agent", lambda ctx: _SearchingOutline(stubs, ctx))
    monkeypatch.setattr(
        pipeline, "_literature_agent", lambda ctx: LiteratureAgent(ctx, scholar=scholar)
    )
    client = FakeLLM(literature_responder)

    def write_up() -> PaperResult:
        return run(cfg, harness=make_harness(cfg, client))[0]

    def searches() -> int:
        return sum(request.label == "literature.discovery" for request in client.requests)

    # The person accepted one source and rejected another; the search proposes both.
    give_known(cfg, [WARMUP], [NOT_SHALLOW])
    first = write_up()
    assert first.success and searches() == 1
    assert bib_keys(stubs.seen["writer_bib"]) == {"Aaa2020Warmup", "Smith2020DeepThings"}
    assert scholar.asked == ["Deep Things"]  # neither the known nor the rejected paper
    record = first.stages["literature"]
    assert (record["known_papers"], record["excluded_candidates"]) == (1, 1)
    assert "warnings" not in record

    # Nothing changed in the wiki: the paper stands, and so does what was said about it.
    again = write_up()
    assert again.success and searches() == 1
    assert again.stages["literature"] == record | {"reused": True}

    # The person accepts the rejected paper after all: the literature is searched and the
    # paper written again, now citing it under the key the wiki has for it.
    give_known(cfg, [WARMUP, SHALLOW])
    changed = write_up()
    assert changed.success and searches() == 2
    assert bib_keys(stubs.seen["writer_bib"]) == {
        "Aaa2020Warmup",
        "Jones2019Shallow",
        "Smith2020DeepThings",
    }
    assert changed.stages["literature"]["known_papers"] == 2
    assert "excluded_candidates" not in changed.stages["literature"]
    assert not any(stage["reused"] for stage in changed.stages.values())

    # A list that cannot be read does not cost the paper; the stage's record says so.
    (cfg.base_dir / "raw_materials" / KNOWN_PAPERS).write_text("{ cut off")
    broken = write_up()
    assert broken.success and searches() == 3
    assert bib_keys(stubs.seen["writer_bib"]) == {"Jones2019ShallowThings", "Smith2020DeepThings"}
    (warning,) = broken.stages["literature"]["warnings"]
    assert KNOWN_PAPERS in warning and "is not valid JSON" in warning
    assert state_of(cfg)["stages"]["literature"]["warnings"] == [warning]


def test_the_write_up_of_a_research_project_cites_its_sources_as_the_person_decided(
    tmp_path, template_dir, stubs, monkeypatch
):
    """The seam with ``research/writeup.py``: what it exports is what the literature stage
    reads, from the wiki's decisions to the bibliography the section writer is given."""
    from paper_orchestra.agents.literature import LiteratureAgent
    from paper_orchestra.research import Project, writeup
    from paper_orchestra.research.demo import build_demo_project
    from paper_orchestra.utils.latex import bib_keys

    build_demo_project(tmp_path / "project")
    project = Project.open(tmp_path / "project")
    accepted = project.pages(kind="source", review="accepted")
    turned_down = accepted.pop()
    project.review(turned_down.path, "reject", reason="Not about warmup at all.")
    proposed = [turned_down.title, accepted[0].title, "Deep Things"]

    def models(request: FakeRequest) -> str:
        if request.label == "literature.discovery":
            found = [{"title": title, "year": None, "reason": "r"} for title in proposed]
            return json.dumps({"candidates": found})
        return literature_responder(request)

    client, scholar = FakeLLM(models), _TinyScholar()
    monkeypatch.setattr(pipeline, "_build_harness", lambda cfg: make_harness(cfg, client))
    monkeypatch.setattr(pipeline, "_build_compute", lambda cfg: FakeCompute())
    monkeypatch.setattr(pipeline, "_outline_agent", lambda ctx: _SearchingOutline(stubs, ctx))
    monkeypatch.setattr(
        pipeline, "_literature_agent", lambda ctx: LiteratureAgent(ctx, scholar=scholar)
    )

    result = writeup.write_paper(project, template=str(template_dir))

    assert result.success, result.error
    # Every source the person accepted, under the key the wiki has for it; what the search
    # added; and not the one they rejected, although the search proposed it.
    assert bib_keys(stubs.seen["writer_bib"]) == {page.id for page in accepted} | {
        "Smith2020DeepThings"
    }
    assert turned_down.id not in stubs.seen["writer_bib"]
    assert scholar.asked == ["Deep Things"]  # nothing the wiki decided on is looked up again
    record = result.stages["literature"]
    assert (record["known_papers"], record["excluded_candidates"]) == (len(accepted), 1)
    assert "warnings" not in record

    # The person turns another source down: the next write-up searches and writes again,
    # and that paper is gone from the bibliography.
    project.review(accepted[0].path, "reject", reason="Superseded.")
    again = writeup.write_paper(project, template=str(template_dir))

    assert again.success and not again.stages["literature"]["reused"]
    assert accepted[0].id not in stubs.seen["writer_bib"]
    assert bib_keys(stubs.seen["writer_bib"]) == {page.id for page in accepted[1:]} | {
        "Smith2020DeepThings"
    }
    record = again.stages["literature"]
    assert (record["known_papers"], record["excluded_candidates"]) == (len(accepted) - 1, 2)

    # Started once more with nothing decided in between, the write-up exports the same list
    # and keeps the paper it has: no search is paid for twice.
    searches = sum(request.label == "literature.discovery" for request in client.requests)
    resumed = writeup.write_paper(project, template=str(template_dir))
    assert resumed.success and resumed.stages["literature"] == record | {"reused": True}
    assert searches == 2 == sum(r.label == "literature.discovery" for r in client.requests)
