from __future__ import annotations

import io
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image as PILImage

from paper_orchestra.agents.section_writing import (
    DRAFT_PAGE,
    SectionWritingAgent,
    SectionWritingError,
    check_latex,
    format_citation_library,
    new_problems,
    repair_lesson,
    source_diff,
)
from paper_orchestra.compute.base import CompileResult, ComputeBackend, ComputeError
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.config import PipelineConfig
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import LLMRefusal
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, TranscriptLog, Wiki
from paper_orchestra.prompts.section_writing import (
    NO_LATEX_IN_REPLY,
    SECTION_WRITING_SYSTEM_PROMPT,
    repair_prompt,
    section_writing_user_prompt,
)
from paper_orchestra.utils.latex import static_check

LABEL = "section_writing.SectionWritingAgent"

BIB = """@inproceedings{Vaswani2017Attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
"""

CITATION_MAP = {
    "Vaswani2017Attention": {
        "citation_key": "Vaswani2017Attention",
        "title": "Attention Is All You Need",
        "authors": ["A. Vaswani", "N. Shazeer", "N. Parmar", "J. Uszkoreit"],
        "venue": "NeurIPS",
        "year": 2017,
        "abstract": "We propose the Transformer, based solely on attention mechanisms.",
    }
}

FIGURE = r"""
\begin{figure}
  \centering
  \includegraphics[width=1.0\linewidth]{figures/accuracy.png}
  \caption{Accuracy against routing sparsity.}
  \label{fig:accuracy}
\end{figure}
"""


def png(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (24, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def finished_paper(repo_root: Path, *, intro_extra: str = "", method_extra: str = "") -> str:
    """The cvpr2025 template with every placeholder section written."""
    template = (
        repo_root / "src" / "paper_orchestra" / "templates" / "cvpr2025" / "template.tex"
    ).read_text(encoding="utf-8")
    written = {
        "My Title": "Sparse Routing for Tiny Models",
        "Abstract here.": "We study sparse routing for small models.",
        "Introduction here.": "Attention is widely used~\\cite{Vaswani2017Attention}. "
        "Our method is described in \\cref{sec:method}." + intro_extra,
        "Related work here.": "Routing has a long history.",
        "Methodology here.": "We route each token to few experts; see \\cref{fig:accuracy}."
        + FIGURE
        + method_extra,
        "Experimental setup, results, and analysis here.": "Accuracy improves by 2.1 points.",
        "Conclusion here.": "Sparse routing works.",
    }
    for placeholder, text in written.items():
        assert placeholder in template
        template = template.replace(placeholder, text)
    return template.strip() + "\n"


def fenced(tex: str) -> str:
    return f"Here is the completed paper.\n```latex\n{tex}```\n"


class ScriptedCompute(ComputeBackend):
    """Builds nothing: ``decide(tex)`` says how the build of each document ends."""

    name = "scripted"

    def __init__(self, decide: Callable[[str], CompileResult] | None = None) -> None:
        self.decide = decide or (lambda tex: CompileResult(ok=True))
        #: ``(workdir, tex_name, tex, files present in the workdir)`` of every build.
        self.builds: list[tuple[Path, str, str, set[str]]] = []

    def run_python(self, code, *, inputs=None, outputs=(), timeout=120.0):
        raise AssertionError("the writing agents never run Python")

    def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0):
        work = Path(workdir)
        tex = (work / f"{tex_name}.tex").read_text(encoding="utf-8")
        present = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
        self.builds.append((work, tex_name, tex, present))
        result = self.decide(tex)
        if result.ok:
            result.pdf_path = Path(output_pdf) if output_pdf else work / f"{tex_name}.pdf"
            result.pdf_path.write_bytes(b"%PDF-1.4 scripted")
        return result


def by_marker(outcomes: dict[str, CompileResult]) -> Callable[[str], CompileResult]:
    """Decide a build by the first marker string found in the source."""

    def decide(tex: str) -> CompileResult:
        for marker, result in outcomes.items():
            if marker in tex:
                return result
        return CompileResult(ok=True)

    return decide


def make_ctx(
    tmp_path: Path,
    repo_root: Path,
    client: FakeLLM,
    compute: ComputeBackend,
    *,
    memory: bool = True,
    **overrides,
) -> RunContext:
    base = tmp_path / "run"
    template_dir = repo_root / "src" / "paper_orchestra" / "templates" / "cvpr2025"
    cfg = PipelineConfig(base_dir=base, template_dir=template_dir, **overrides)
    paths = RunPaths(base)
    paths.materials_dir.mkdir(parents=True)
    paths.idea.write_text("# Idea\nRoute each token to a few experts.", encoding="utf-8")
    paths.experimental_log.write_text(
        "# Log\nBaseline 71.3, ours 73.4 (+2.1 points).", encoding="utf-8"
    )
    shutil.copytree(template_dir, paths.latex_dir)
    paths.references_bib.write_text(BIB, encoding="utf-8")
    paths.figures_dir.mkdir()
    (paths.figures_dir / "accuracy.png").write_bytes(png())
    paths.figures_info.write_text(
        json.dumps([{"name": "accuracy.png", "caption": "Accuracy against sparsity."}]),
        encoding="utf-8",
    )
    paths.outline.write_text(
        json.dumps({"section_plan": [{"section_title": "Methodology"}]}), encoding="utf-8"
    )
    paths.literature_dir.mkdir()
    paths.citation_map.write_text(json.dumps(CITATION_MAP), encoding="utf-8")
    if memory:
        harness = ContextHarness(
            project=Wiki(base / "memory"),
            transcripts=TranscriptLog(base / "memory" / "transcripts"),
            client_for=lambda model: client,
        )
    else:
        harness = ContextHarness(client_for=lambda model: client)
    return RunContext(cfg=cfg, paths=paths, harness=harness, compute=compute)


def events(ctx: RunContext, kind: str) -> list[dict]:
    return [e for e in ctx.harness.transcripts.events() if e["type"] == kind]


def notes(ctx: RunContext) -> list[str]:
    wiki = ctx.harness.project
    return [wiki.read(info.path).body for info in wiki.pending_notes()]


def workspace_files(ctx: RunContext) -> set[str]:
    root = ctx.paths.latex_dir
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def drafts_in_turn_order(*drafts: str) -> Callable[[FakeRequest], str]:
    return lambda request: drafts[request.turn]


# -- the happy path ------------------------------------------------------------


def test_clean_first_draft_is_written_without_touching_the_workspace(tmp_path, repo_root):
    paper = finished_paper(repo_root)
    client = FakeLLM({LABEL: fenced(paper)})
    compute = ScriptedCompute()
    ctx = make_ctx(tmp_path, repo_root, client, compute)
    before = workspace_files(ctx)

    out = SectionWritingAgent(ctx).run()

    assert out == ctx.paths.raw_draft_tex
    assert out.read_text(encoding="utf-8") == paper
    assert workspace_files(ctx) == before | {"raw_draft_paper.tex"}

    # One model call, on the writer model, with memory attached and every input in its tag.
    [request] = client.requests
    assert request.model == ctx.cfg.models.writer
    assert request.system.startswith("Role: Senior AI Researcher.")
    assert "Strict Knowledge Isolation" in request.system
    assert "# Memory" in request.system and request.memory is not None
    text = request.text
    for tag in (
        "outline",
        "citation_library",
        "idea",
        "experimental_log",
        "figures",
        "guidelines",
        "template",
    ):
        assert f"<{tag}>" in text and f"</{tag}>" in text
    assert '"section_title": "Methodology"' in text
    assert "--- Key: Vaswani2017Attention ---" in text
    assert "Title: Attention Is All You Need" in text
    assert "Authors: A. Vaswani, N. Shazeer, N. Parmar et al. (2017)" in text
    assert "Abstract: We propose the Transformer, based solely on attention mechanisms." in text
    assert "Route each token to a few experts." in text
    assert "Baseline 71.3, ours 73.4 (+2.1 points)." in text
    assert "File: figures/accuracy.png" in text and "Caption: Accuracy against sparsity." in text
    assert "# CVPR 2025 Paper Writing Guidelines" in text
    assert "Methodology here." in text
    # The figure itself is attached, introduced by the path used to include it.
    assert [image.media_type for image in request.images] == ["image/png"]
    assert request.images[0].data == png()
    assert "Image of figures/accuracy.png:" in text

    # The build ran once, in a scratch copy of the workspace that is gone afterwards.
    [(workdir, tex_name, tex, present)] = compute.builds
    assert tex_name == "raw_draft_paper" and tex == paper
    assert workdir != ctx.paths.latex_dir and not workdir.exists()
    assert {"references.bib", "cvpr.sty", "figures/accuracy.png", "raw_draft_paper.tex"} <= present

    [compiled] = events(ctx, "compile")
    assert compiled["stage"] == "section_writing" and compiled["agent"] == "SectionWritingAgent"
    assert (compiled["round"], compiled["ok"], compiled["clean"]) == (0, True, True)
    assert notes(ctx) == []
    page = ctx.harness.project.read(DRAFT_PAGE)
    assert page.meta["kind"] == "draft"
    assert "latex_writeup/raw_draft_paper.tex" in page.body and "builds cleanly" in page.body
    assert "\\documentclass" not in page.body


def test_runs_with_memory_disabled(tmp_path, repo_root):
    paper = finished_paper(repo_root)
    client = FakeLLM({LABEL: fenced(paper)})
    ctx = make_ctx(tmp_path, repo_root, client, ScriptedCompute(), memory=False)
    assert SectionWritingAgent(ctx).run().read_text(encoding="utf-8") == paper
    assert client.requests[0].memory is None and "# Memory" not in client.requests[0].system


# -- the repair loop -----------------------------------------------------------


def test_repair_loop_converges_in_the_same_conversation(tmp_path, repo_root):
    good = finished_paper(repo_root)
    broken = finished_paper(repo_root, intro_extra=" Also \\cite{Ghost2099} and \\badmacro.")
    client = FakeLLM({LABEL: drafts_in_turn_order(fenced(broken), fenced(good))})
    compute = ScriptedCompute(
        by_marker(
            {
                "\\badmacro": CompileResult(
                    ok=True,
                    errors=["Undefined control sequence. [l.38 \\badmacro]"],
                    warnings=["undefined citation 'Ghost2099'", "overfull hbox up to 40pt"],
                )
            }
        )
    )
    ctx = make_ctx(tmp_path, repo_root, client, compute)

    out = SectionWritingAgent(ctx).run()

    assert out.read_text(encoding="utf-8") == good
    first, second = client.requests
    assert (first.turn, second.turn) == (0, 1)
    # Same conversation: the repair turn sees the first exchange and carries only the feedback.
    assert second.history == [(first.text, fenced(broken))]
    assert not second.images and "<template>" not in second.text
    assert "1 citation key(s) are not in the bibliography: Ghost2099" in second.text
    assert "Undefined control sequence. [l.38 \\badmacro]" in second.text
    assert "undefined citation 'Ghost2099'" in second.text
    assert "complete corrected file" in second.text

    first_build, second_build = events(ctx, "compile")
    assert (first_build["round"], first_build["ok"], first_build["clean"]) == (0, True, False)
    assert first_build["static_problems"] == [
        "1 citation key(s) are not in the bibliography: Ghost2099"
    ]
    assert (second_build["round"], second_build["clean"]) == (1, True)

    # The failure and what fixed it are kept as a lesson.
    [lesson] = notes(ctx)
    [info] = ctx.harness.project.pending_notes()
    assert ctx.harness.project.read(info.path).meta["kind"] == "lesson"
    assert "3 of 3 reported problems were fixed and the document now builds cleanly" in lesson
    assert "- Undefined control sequence. [l.38 \\badmacro]" in lesson
    assert "- undefined citation 'Ghost2099'" in lesson
    assert "overfull hbox" not in lesson  # not a reason to repair
    assert "```diff" in lesson and "Also \\cite{Ghost2099} and \\badmacro." in lesson
    assert "Still wrong" not in lesson
    assert workspace_files(ctx) >= {"raw_draft_paper.tex"}
    assert not any(name.endswith((".aux", ".pdf", ".log")) for name in workspace_files(ctx))


@pytest.mark.parametrize(
    ("problems_per_draft", "kept"),
    [
        # Fewest problems wins; the last draft does not compile at all.
        ((3, 1, None), 1),
        # A tie goes to the later repair.
        ((2, 1, 1), 2),
        # An earlier draft with fewer problems beats later ones.
        ((1, 2, 3), 0),
    ],
)
def test_best_compiling_draft_is_kept_when_none_is_clean(
    tmp_path, repo_root, problems_per_draft, kept
):
    drafts = [
        finished_paper(repo_root, intro_extra=f" Draft marker {index}.")
        for index in range(len(problems_per_draft))
    ]
    outcomes = {
        f"Draft marker {index}.": CompileResult(ok=False, errors=["Emergency stop."])
        if count is None
        else CompileResult(ok=True, errors=[f"error {n} of draft {index}" for n in range(count)])
        for index, count in enumerate(problems_per_draft)
    }
    client = FakeLLM({LABEL: drafts_in_turn_order(*map(fenced, drafts))})
    ctx = make_ctx(
        tmp_path, repo_root, client, ScriptedCompute(by_marker(outcomes)), max_repair_rounds=2
    )

    out = SectionWritingAgent(ctx).run()

    assert len(client.requests) == 3
    assert out.read_text(encoding="utf-8") == drafts[kept]
    page = ctx.harness.project.read(DRAFT_PAGE).body
    assert f"the one kept is draft {kept}" in page
    assert f"error 0 of draft {kept}" in page and "open problem(s)" in page
    assert len(notes(ctx)) == 2  # one lesson per repair turn


def test_a_clean_draft_ends_the_loop_early(tmp_path, repo_root):
    drafts = [finished_paper(repo_root, intro_extra=" \\badmacro"), finished_paper(repo_root)]
    client = FakeLLM({LABEL: drafts_in_turn_order(*map(fenced, drafts))})
    compute = ScriptedCompute(
        by_marker({"\\badmacro": CompileResult(ok=True, errors=["Undefined control sequence."])})
    )
    ctx = make_ctx(tmp_path, repo_root, client, compute, max_repair_rounds=5)
    SectionWritingAgent(ctx).run()
    assert len(client.requests) == 2 and len(compute.builds) == 2


def test_no_compiling_draft_raises_and_writes_nothing(tmp_path, repo_root):
    truncated = finished_paper(repo_root).replace("\\end{document}", "")
    client = FakeLLM({LABEL: fenced(truncated)})
    compute = ScriptedCompute(
        lambda tex: CompileResult(
            ok=False, errors=["Emergency stop."], warnings=["undefined citation 'X'"]
        )
    )
    ctx = make_ctx(tmp_path, repo_root, client, compute, max_repair_rounds=1)

    with pytest.raises(SectionWritingError) as info:
        SectionWritingAgent(ctx).run()

    message = str(info.value)
    assert "none of the 2 drafts" in message and ctx.cfg.models.writer in message
    assert "missing \\end{document}" in message and "Emergency stop." in message
    assert not ctx.paths.raw_draft_tex.exists()
    assert not ctx.harness.project.exists(DRAFT_PAGE)
    assert len(client.requests) == 2
    # A failed build's unresolved references are noise next to its errors.
    assert "undefined citation" not in client.requests[1].text
    assert "LaTeX did not produce a PDF." in client.requests[1].text
    assert any("none of 2 drafts compiled" in note for note in notes(ctx))


def test_compute_error_propagates(tmp_path, repo_root):
    def no_tex(tex: str) -> CompileResult:
        raise ComputeError("pdflatex/bibtex not found on PATH")

    client = FakeLLM({LABEL: fenced(finished_paper(repo_root))})
    ctx = make_ctx(tmp_path, repo_root, client, ScriptedCompute(no_tex))
    with pytest.raises(ComputeError, match="pdflatex/bibtex not found"):
        SectionWritingAgent(ctx).run()
    assert len(client.requests) == 1 and not ctx.paths.raw_draft_tex.exists()


def test_failed_repair_request_keeps_the_draft_that_compiles(tmp_path, repo_root, caplog):
    flawed = finished_paper(repo_root, intro_extra=" \\badmacro")

    def responder(request: FakeRequest) -> str:
        if request.turn:
            raise LLMRefusal("the model declined the request", category="other")
        return fenced(flawed)

    compute = ScriptedCompute(
        by_marker({"\\badmacro": CompileResult(ok=True, errors=["Undefined control sequence."])})
    )
    ctx = make_ctx(tmp_path, repo_root, FakeLLM({LABEL: responder}), compute)
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.agents.section_writing"):
        out = SectionWritingAgent(ctx).run()
    assert out.read_text(encoding="utf-8") == flawed
    assert "repair round 1 failed (LLMRefusal" in caplog.text


def test_failed_first_request_propagates(tmp_path, repo_root):
    def responder(request: FakeRequest) -> str:
        raise LLMRefusal("the model declined the request", category="other")

    ctx = make_ctx(tmp_path, repo_root, FakeLLM({LABEL: responder}), ScriptedCompute())
    with pytest.raises(LLMRefusal):
        SectionWritingAgent(ctx).run()
    assert not ctx.paths.raw_draft_tex.exists()


def test_reply_without_latex_is_asked_for_again(tmp_path, repo_root):
    paper = finished_paper(repo_root)
    client = FakeLLM(
        {LABEL: drafts_in_turn_order("I need more details before I can write.", fenced(paper))}
    )
    compute = ScriptedCompute()
    ctx = make_ctx(tmp_path, repo_root, client, compute)

    assert SectionWritingAgent(ctx).run().read_text(encoding="utf-8") == paper
    assert client.requests[1].text == NO_LATEX_IN_REPLY
    assert len(compute.builds) == 1


# -- inputs --------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["idea", "experimental_log", "outline", "citation_map"])
def test_missing_input_raises_before_any_model_call(tmp_path, repo_root, missing):
    client = FakeLLM({LABEL: fenced(finished_paper(repo_root))})
    ctx = make_ctx(tmp_path, repo_root, client, ScriptedCompute())
    getattr(ctx.paths, missing).unlink()
    with pytest.raises(FileNotFoundError, match="required file not found"):
        SectionWritingAgent(ctx).run()
    assert client.requests == []


def test_figures_are_listed_by_their_real_files_and_shown_to_the_writer(
    tmp_path, repo_root, caplog
):
    import pymupdf

    client = FakeLLM({LABEL: fenced(finished_paper(repo_root))})
    ctx = make_ctx(tmp_path, repo_root, client, ScriptedCompute())
    figures = ctx.paths.figures_dir
    PILImage.new("RGB", (24, 16), "red").save(figures / "curve.JPG", format="JPEG")
    document = pymupdf.open()
    document.new_page(width=120, height=80).insert_text((10, 40), "diagram")
    document.save(figures / "pipeline.pdf")
    document.close()
    (figures / "anim.gif").write_bytes(b"GIF89a")
    elsewhere = tmp_path / "elsewhere.png"
    elsewhere.write_bytes(png())
    ctx.paths.figures_info.write_text(
        json.dumps(
            [
                {"name": "accuracy", "caption": "Accuracy."},
                {"name": "curve.JPG", "caption": "Loss curve."},
                {"name": "pipeline.pdf"},
                {"name": "ghost.png", "caption": "Never rendered."},
                {"name": "anim.gif", "caption": "Animation."},
                {"name": str(elsewhere), "caption": "Not part of the workspace."},
                {"name": "../../../elsewhere.png", "caption": "Not part of it either."},
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="paper_orchestra.agents.section_writing"):
        SectionWritingAgent(ctx).run()

    [request] = client.requests
    listing = request.text.split("<figures>")[1].split("</figures>")[0]
    assert "File: figures/accuracy.png" in listing  # the extension was resolved
    assert "File: figures/curve.JPG" in listing and "Caption: Loss curve." in listing
    assert "File: figures/pipeline.pdf" in listing and "Caption: No caption provided." in listing
    assert "ghost" not in listing and "anim" not in listing and "elsewhere" not in listing
    # A PDF figure is shown as a rendering of its first page.
    assert [image.media_type for image in request.images] == [
        "image/png",
        "image/jpeg",
        "image/png",
    ]
    assert "Image of figures/pipeline.pdf:" in request.text
    assert "'ghost.png') is left out of the paper: no such file" in caplog.text
    assert "'anim.gif') is left out of the paper: pdflatex cannot include .gif files" in caplog.text
    # Only the workspace is copied for a build, so a figure outside it could never be included.
    assert caplog.text.count("is left out of the paper: it is outside the LaTeX workspace") == 2


def test_figures_without_info_json_are_taken_from_the_directory(tmp_path, repo_root, caplog):
    client = FakeLLM({LABEL: fenced(finished_paper(repo_root))})
    ctx = make_ctx(tmp_path, repo_root, client, ScriptedCompute())
    ctx.paths.figures_info.unlink()
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.agents.section_writing"):
        SectionWritingAgent(ctx).run()
    [request] = client.requests
    assert "File: figures/accuracy.png" in request.text and len(request.images) == 1
    assert "info.json does not exist; using the 1 figure file(s)" in caplog.text


def test_paper_without_figures(tmp_path, repo_root):
    paper = finished_paper(repo_root).replace(FIGURE, "").replace("; see \\cref{fig:accuracy}", "")
    client = FakeLLM({LABEL: fenced(paper)})
    ctx = make_ctx(tmp_path, repo_root, client, ScriptedCompute())
    shutil.rmtree(ctx.paths.figures_dir)
    assert SectionWritingAgent(ctx).run().read_text(encoding="utf-8") == paper
    [request] = client.requests
    assert "<figures>\nNo figures provided.\n</figures>" in request.text and not request.images


@pytest.mark.parametrize("content", ['{"name": "accuracy.png"}', '["accuracy.png"]', "not json"])
def test_malformed_figure_list_is_an_error(tmp_path, repo_root, content):
    client = FakeLLM({LABEL: fenced(finished_paper(repo_root))})
    ctx = make_ctx(tmp_path, repo_root, client, ScriptedCompute())
    ctx.paths.figures_info.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=r"info\.json"):
        SectionWritingAgent(ctx).run()
    assert client.requests == []


def test_citation_library_formatting():
    library = format_citation_library(
        {
            "Solo2020Paper": {"title": "Solo", "authors": ["A. Solo"], "year": 2020},
            "Bare2021Entry": "unexpected",
        }
    )
    assert "--- Key: Solo2020Paper ---\nTitle: Solo\nAuthors: A. Solo (2020)" in library
    assert "Abstract: No abstract available." in library
    assert "--- Key: Bare2021Entry ---\nTitle: Unknown Title\nAuthors: Unknown (N/A)" in library
    assert format_citation_library({}) == "No citation data provided."


# -- checking a document ---------------------------------------------------------


class LoggingCompute(ScriptedCompute):
    """Like a real backend, leaves the log of the final pdflatex pass in the workdir."""

    def __init__(self, result: CompileResult, final_log: str) -> None:
        super().__init__(lambda tex: result)
        self.final_log = final_log

    def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0):
        (Path(workdir) / f"{tex_name}.log").write_text(self.final_log, encoding="utf-8")
        return super().compile_latex(workdir, tex_name, output_pdf=output_pdf, timeout=timeout)


def workdir_with(tmp_path: Path, repo_root: Path) -> Path:
    work = tmp_path / "work"
    (work / "figures").mkdir(parents=True)
    (work / "figures" / "accuracy.png").write_bytes(png())
    (work / "references.bib").write_text(BIB, encoding="utf-8")
    return work


def test_first_pass_warnings_are_dropped_and_remaining_ones_kept(tmp_path, repo_root):
    work = workdir_with(tmp_path, repo_root)
    tex = finished_paper(repo_root)
    all_passes = CompileResult(
        ok=True,
        warnings=[
            "undefined citation 'Vaswani2017Attention'",
            "undefined reference 'sec:method'",
            "undefined reference 'fig:gone'",
            "overfull hbox up to 31pt (content spills into the margin)",
        ],
    )
    final_log = "LaTeX Warning: Reference `fig:gone' on page 2 undefined on input line 40.\n"
    check = check_latex(LoggingCompute(all_passes, final_log), work, "paper", tex, timeout=10)

    assert check.compile.warnings == [
        "undefined reference 'fig:gone'",
        "overfull hbox up to 31pt (content spills into the margin)",
    ]
    assert check.problems == ["undefined reference 'fig:gone'"] and not check.clean
    assert "undefined citation" not in check.feedback()
    assert check.pdf == work / "paper.pdf"

    resolved = check_latex(LoggingCompute(all_passes, ""), work, "paper", tex, timeout=10)
    assert resolved.clean and resolved.compile.warnings == [
        "overfull hbox up to 31pt (content spills into the margin)"
    ]
    # Without a log to consult, the backend's word is taken as it is.
    trusted = check_latex(ScriptedCompute(lambda tex: all_passes), work, "paper", tex, timeout=10)
    assert len(trusted.problems) == 3


def test_bibtex_complaint_about_a_paper_without_citations_is_not_a_problem(tmp_path, repo_root):
    work = workdir_with(tmp_path, repo_root)
    complaint = "bibtex: I found no \\citation commands---while reading file paper.aux"
    result = CompileResult(ok=True, errors=[complaint])
    citing = finished_paper(repo_root)
    uncited = citing.replace("~\\cite{Vaswani2017Attention}", "")

    assert check_latex(ScriptedCompute(lambda t: result), work, "paper", uncited, timeout=10).clean
    kept = check_latex(ScriptedCompute(lambda t: result), work, "paper", citing, timeout=10)
    assert kept.problems == [complaint]


def test_stale_build_products_never_pass_for_a_new_build(tmp_path, repo_root):
    work = workdir_with(tmp_path, repo_root)
    for suffix in (".aux", ".bbl", ".log", ".pdf"):
        (work / f"paper{suffix}").write_text("left over from an earlier build", encoding="utf-8")
    compute = ScriptedCompute(lambda tex: CompileResult(ok=False, errors=["Emergency stop."]))

    check = check_latex(compute, work, "paper", finished_paper(repo_root), timeout=10)

    assert not check.compiles and check.pdf is None
    assert compute.builds[0][3] == {"paper.tex", "references.bib", "figures/accuracy.png"}
    assert check.problems == ["LaTeX did not produce a PDF", "Emergency stop."]


def test_static_problems_are_reported_next_to_build_errors(tmp_path, repo_root):
    work = workdir_with(tmp_path, repo_root)
    tex = finished_paper(repo_root).replace("figures/accuracy.png", "figures/missing.png")
    tex = tex.replace("Sparse Routing for Tiny Models", "My Title")
    result = CompileResult(ok=True, warnings=["file not found: figures/missing.png"])
    check = check_latex(ScriptedCompute(lambda t: result), work, "paper", tex, timeout=10)
    assert check.problems == [
        "figure file(s) not found: figures/missing.png",
        "template placeholder still present: 'My Title'",
        "file not found: figures/missing.png",
    ]
    feedback = check.feedback()
    assert feedback.startswith("Problems found by checking the source:\n- figure file(s)")
    assert "Warnings:\n- file not found: figures/missing.png" in feedback


def test_a_defect_is_recognised_wherever_it_is_reported(tmp_path, repo_root):
    error = "Undefined control sequence. [{line}We route each token. \\badmacro]"
    before = [
        "./initial_draft.tex:50: " + error.format(line="l.50 "),
        "undefined reference 'fig:gone'",
        "bibtex: I found no \\bibstyle command---while reading file initial_draft.aux",
        "bibtex: Repeated entry---line 12 of file references.bib",
    ]
    after = [
        # The same four defects in a later version: other file name, other lines.
        "./refined_paper_v1.tex:57: " + error.format(line="l.57 "),
        "undefined reference 'fig:gone'",
        "bibtex: I found no \\bibstyle command---while reading file refined_paper_v1.aux",
        "bibtex: Repeated entry---line 19 of file references.bib",
        # Two new ones.
        "./refined_paper_v1.tex:57: Undefined control sequence. [l.57 Another line. \\badmacro]",
        "undefined reference 'fig:other'",
    ]
    assert new_problems(after, before) == after[4:]
    assert new_problems(before, after) == [] and new_problems(after, []) == after

    # The static checks put every unknown key (and every missing figure) on one line.
    work = workdir_with(tmp_path, repo_root)

    def static(extra: str) -> list[str]:
        return static_check(finished_paper(repo_root, intro_extra=extra), workdir=work)

    two = static(" \\cite{Ghost2099,Phantom2098} \\includegraphics{figures/a.png}")
    one = static(" \\cite{Phantom2098} \\includegraphics{figures/a.png}")
    other = static(" \\cite{Phantom2098,Wraith2097} \\includegraphics{figures/b.png}")
    assert len(two) == len(one) == len(other) == 2 and two != one
    assert new_problems(one, two) == []  # dropping one of two unknown keys adds no defect
    assert new_problems(two, one) == [two[0]]
    assert new_problems(other, two) == other


def test_lesson_does_not_count_a_moved_error_as_fixed(tmp_path, repo_root):
    work = workdir_with(tmp_path, repo_root)

    def check_of(tex: str, line: int):
        error = f"./paper.tex:{line}: Undefined control sequence. [l.{line} experts. \\badmacro]"
        compute = ScriptedCompute(lambda _: CompileResult(ok=True, errors=[error, "Other."]))
        return check_latex(compute, work, "paper", tex, timeout=10)

    before = check_of(finished_paper(repo_root), 50)
    after = check_of(finished_paper(repo_root, intro_extra="\n\nA new paragraph."), 52)
    assert before.problems[0] != after.problems[0]

    lesson = repair_lesson("the draft", before.problems, before.tex, after)

    assert "0 of 2 reported problems were fixed." in lesson
    assert "Still wrong after the fix:\n- ./paper.tex:52: Undefined control sequence." in lesson


def test_source_diff_shows_changed_lines_and_is_clipped():
    assert source_diff("a\nb\nc\n", "a\nB\nc\n") == "-b\n+B"
    long = source_diff("\n".join(map(str, range(200))), "", max_lines=5)
    assert long.count("\n") == 5 and long.endswith("(195 more changed lines)")


# -- prompts -------------------------------------------------------------------


def test_prompts_render_literal_latex():
    assert "{{" not in SECTION_WRITING_SYSTEM_PROMPT and "}}" not in SECTION_WRITING_SYSTEM_PROMPT
    assert (
        "Do not change `\\usepackage[capitalize]{cleveref}` into "
        "`\\usepackage[capitalize]{cleverref}`" in SECTION_WRITING_SYSTEM_PROMPT
    )
    assert "`\\cite{Hu2021LoraLowrank}`" in SECTION_WRITING_SYSTEM_PROMPT
    assert "(\\toprule, \\midrule, \\bottomrule)" in SECTION_WRITING_SYSTEM_PROMPT
    assert "author names if missing" not in SECTION_WRITING_SYSTEM_PROMPT
    # Not every template loads the packages the instructions rely on (iclr2025 has neither).
    assert "\\includegraphics without `graphicx`" in SECTION_WRITING_SYSTEM_PROMPT
    assert "\\toprule without `booktabs`" in SECTION_WRITING_SYSTEM_PROMPT

    latex = "\\title{{A} {B}}\n\\begin{document}{x}\\end{document}"
    prompt = section_writing_user_prompt(
        outline='{"a": {"b": 1}}',
        citation_library="lib",
        idea="idea",
        experimental_log="log",
        figures="figs",
        guidelines="rules",
        template=latex,
    )
    assert f"<template>\n{latex}\n</template>" in prompt
    assert '<outline>\n{"a": {"b": 1}}\n</outline>' in prompt
    assert "<problems>\n- \\begin{figure*} ended by \\end{figure}\n</problems>" in repair_prompt(
        "- \\begin{figure*} ended by \\end{figure}"
    )


# -- the real thing ------------------------------------------------------------


@pytest.mark.latex
def test_real_compile_of_a_valid_draft_is_clean(tmp_path, repo_root):
    paper = finished_paper(repo_root)
    client = FakeLLM({LABEL: fenced(paper)})
    ctx = make_ctx(tmp_path, repo_root, client, LocalBackend())
    before = workspace_files(ctx)

    out = SectionWritingAgent(ctx).run()

    assert out.read_text(encoding="utf-8") == paper
    # pdflatex reports the citation and both cross-references as undefined on its first
    # pass; a document that resolves them all must not be sent back for repair.
    assert len(client.requests) == 1
    [compiled] = events(ctx, "compile")
    assert compiled["ok"] is True and compiled["clean"] is True
    assert compiled["errors"] == [] and compiled["static_problems"] == []
    assert not any(w.startswith("undefined") for w in compiled["warnings"])
    assert workspace_files(ctx) == before | {"raw_draft_paper.tex"}
