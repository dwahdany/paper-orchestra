from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import smtplib
import sys
import textwrap
import time
import zipfile
from datetime import datetime
from email import message_from_bytes
from pathlib import Path
from typing import ClassVar

import pytest

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.context import RunPaths
from paper_orchestra.memory import Wiki

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def _load_frontend_utils():
    """``frontend/`` is a script directory, not a package: load the module by path.

    It is registered under the name the Streamlit script imports, so the app
    smoke test below shares this module object.
    """
    spec = importlib.util.spec_from_file_location(
        "frontend_utils", FRONTEND_DIR / "frontend_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["frontend_utils"] = module
    spec.loader.exec_module(module)
    return module


fu = _load_frontend_utils()

posix_only = pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX-specific")


def flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_default_command_runs_the_installed_package_with_fable_everywhere(tmp_path):
    options = fu.RunOptions(run_dir=tmp_path / "run")
    command = fu.build_write_command(options, python="/venv/bin/python", templates_dir=tmp_path)
    assert command == [
        "/venv/bin/python",
        "-m",
        "paper_orchestra",
        "write",
        "--raw-materials-dir",
        str(tmp_path / "run" / "raw_materials"),
        "--output-dir",
        str(tmp_path / "run"),
        "--template",
        str(tmp_path / "cvpr2025"),
        "--idea-filename",
        "idea_sparse.md",
        "--experimental-log-filename",
        "experimental_log.md",
        "--model",
        "claude-fable-5-2",
        "--writer-model",
        "claude-fable-5-2",
        "--reflection-model",
        "claude-fable-5-2",
        "--compute",
        "local",
        "--use-plotting",
        "--plotting-model",
        "claude-fable-5-2",
        "--memory",
        "--dream",
        "stages",
    ]
    # No image model unless one is chosen: diagrams are then drawn from code.
    assert "--image-model" not in command
    assert fu.build_write_command(options)[0] == sys.executable


def test_model_list_leads_with_fable_then_other_claude_models_then_gemini():
    assert fu.MODEL_CHOICES[0] == DEFAULT_MODEL == "claude-fable-5-2"
    assert fu.MODEL_CHOICES[1:3] == ("claude-opus-5-5", "claude-sonnet-5-5")
    assert all(model.startswith("gemini-") for model in fu.MODEL_CHOICES[3:])
    assert all(model.startswith("gemini-") for model in fu.IMAGE_MODEL_CHOICES)


@pytest.mark.parametrize("model", fu.MODEL_CHOICES)
def test_selected_model_takes_every_role(tmp_path, model):
    command = fu.build_write_command(fu.RunOptions(run_dir=tmp_path, model=model))
    for flag in ("--model", "--writer-model", "--reflection-model", "--plotting-model"):
        assert flag_value(command, flag) == model
    # No role is left on another provider's default model.
    assert "--fast-model" not in command


def test_plotting_off_drops_every_plotting_flag(tmp_path):
    options = fu.RunOptions(
        run_dir=tmp_path, use_plotting=False, image_model="gemini-3-pro-image-preview"
    )
    command = fu.build_write_command(options)
    assert "--no-use-plotting" in command
    assert not {"--use-plotting", "--plotting-model", "--image-model"} & set(command)


def test_image_model_is_passed_only_when_chosen(tmp_path):
    options = fu.RunOptions(run_dir=tmp_path, image_model="gemini-3-pro-image-preview")
    command = fu.build_write_command(options)
    assert flag_value(command, "--image-model") == "gemini-3-pro-image-preview"
    assert flag_value(command, "--plotting-model") == DEFAULT_MODEL


@pytest.mark.parametrize("template", fu.TEMPLATE_CHOICES)
def test_template_is_passed_as_a_directory_of_the_repository(tmp_path, template):
    command = fu.build_write_command(fu.RunOptions(run_dir=tmp_path, template=template))
    template_dir = Path(flag_value(command, "--template"))
    assert template_dir == fu.TEMPLATES_DIR / template
    assert (template_dir / "template.tex").is_file()


@pytest.mark.parametrize("compute", fu.COMPUTE_CHOICES)
@pytest.mark.parametrize("dream", fu.DREAM_CHOICES)
def test_compute_and_dream_modes_are_forwarded(tmp_path, compute, dream):
    command = fu.build_write_command(fu.RunOptions(run_dir=tmp_path, compute=compute, dream=dream))
    assert flag_value(command, "--compute") == compute
    assert flag_value(command, "--dream") == dream
    assert "--memory" in command and "--no-memory" not in command


def test_memory_off_disables_memory_and_sends_no_dream_mode(tmp_path):
    command = fu.build_write_command(fu.RunOptions(run_dir=tmp_path, memory=False, dream="end"))
    assert "--no-memory" in command
    assert "--memory" not in command and "--dream" not in command


def test_unknown_compute_or_dream_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="compute backend"):
        fu.RunOptions(run_dir=tmp_path, compute="kubernetes")
    with pytest.raises(ValueError, match="dream mode"):
        fu.RunOptions(run_dir=tmp_path, dream="always")


def test_every_command_is_accepted_by_the_real_cli(tmp_path):
    """Parse (without running) the commands against the actual ``write`` command."""
    cli = pytest.importorskip("paper_orchestra.cli")
    typer_main = pytest.importorskip("typer.main")
    write = typer_main.get_command(cli.app).commands["write"]
    variants = [
        fu.RunOptions(run_dir=tmp_path),
        fu.RunOptions(run_dir=tmp_path, use_plotting=False, memory=False),
        fu.RunOptions(
            run_dir=tmp_path,
            model="gemini-3.1-pro-preview",
            image_model="gemini-3-pro-image-preview",
            compute="modal",
            dream="off",
            template="iclr2025",
        ),
    ]
    for options in variants:
        arguments = fu.build_write_command(options)[4:]
        context = write.make_context("write", list(arguments))
        assert context.args == []


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


def test_run_directory_is_timestamped_unique_and_relocatable(tmp_path, monkeypatch):
    moment = datetime(2026, 3, 4, 5, 6, 7)
    monkeypatch.delenv(fu.DEMO_DIR_ENV, raising=False)
    default = fu.new_run_directory(moment)
    assert default.parent == fu.REPO_ROOT.parent / "demo_logs"
    assert default.name.startswith("paper_orchestra_run_20260304_050607_")
    # Two sessions starting in the same second used to share (and overwrite) one directory.
    assert len({fu.new_run_directory(moment) for _ in range(20)}) == 20
    monkeypatch.setenv(fu.DEMO_DIR_ENV, str(tmp_path / "demo"))
    assert fu.new_run_directory(moment).parent == tmp_path / "demo"
    assert not (tmp_path / "demo").exists()  # nothing is created until a run starts


class Upload:
    """Stands in for a Streamlit upload."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_raw_materials_are_written_where_the_pipeline_reads_them(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG path figure")
    figures = [
        {"file": str(source), "name": "overview.png", "caption": "System overview."},
        {"file": Upload("plot.jpg", b"jpeg bytes"), "name": "../../plot.jpg", "caption": "Results"},
        {"file": Upload("plot.jpg", b"newer jpeg"), "name": "plot.jpg", "caption": "Results, v2"},
    ]
    run_dir = tmp_path / "run"
    materials = fu.write_raw_materials(run_dir, "the idea", "the log", figures)

    paths = RunPaths(run_dir, fu.IDEA_FILENAME, fu.EXPERIMENTAL_LOG_FILENAME)
    assert materials == paths.materials_dir
    assert paths.idea.read_text() == "the idea"
    assert paths.experimental_log.read_text() == "the log"
    figure_dir = paths.provided_figures_dir
    assert (figure_dir / "overview.png").read_bytes() == b"\x89PNG path figure"
    # The browser-supplied name cannot leave the figures folder ...
    assert (figure_dir / "plot.jpg").read_bytes() == b"jpeg bytes"
    assert not (tmp_path / "plot.jpg").exists()
    # ... and a second figure with the same name is kept next to the first, not over it.
    assert (figure_dir / "plot_2.jpg").read_bytes() == b"newer jpeg"
    # Captions reach the writer through info.json (the old frontend never wrote it).
    assert json.loads((figure_dir / "info.json").read_text()) == [
        {"name": "overview.png", "caption": "System overview."},
        {"name": "plot.jpg", "caption": "Results"},
        {"name": "plot_2.jpg", "caption": "Results, v2"},
    ]
    assert sorted(path.name for path in figure_dir.iterdir()) == [
        "info.json",
        "overview.png",
        "plot.jpg",
        "plot_2.jpg",
    ]


def test_figure_names_are_made_safe_for_the_file_system_and_latex():
    assert fu.safe_figure_name("overview.png") == "overview.png"
    assert fu.safe_figure_name("C:\\Users\\me\\my plot #1 (50%).png") == "my_plot_1_50_.png"
    assert fu.safe_figure_name("../../etc/passwd.png") == "passwd.png"
    assert fu.safe_figure_name(".hidden.png") == "hidden.png"
    # Numbered until free, without regard to case (the run may sit on such a file system).
    assert fu.safe_figure_name("Fig.PNG", taken=["fig.png", "Fig_2.PNG"]) == "Fig_3.PNG"
    for unusable in ("", ".", "..", "a/..", "/", "info.json", "INFO.JSON", "%%%", "__.png"):
        with pytest.raises(ValueError, match="figure file name"):
            fu.safe_figure_name(unusable)


def test_raw_materials_without_figures_have_no_info_json_and_bad_names_fail(tmp_path):
    fu.write_raw_materials(tmp_path / "run", "idea", "log")
    figure_dir = tmp_path / "run" / "raw_materials" / "figures"
    assert figure_dir.is_dir() and list(figure_dir.iterdir()) == []
    with pytest.raises(ValueError, match="figure file name"):
        fu.write_raw_materials(
            tmp_path / "run", "idea", "log", [{"file": Upload("x", b""), "name": "info.json"}]
        )


def test_bundled_example_loads_with_its_figure():
    example = fu.load_example()
    assert len(example["idea"]) > 100 and len(example["log"]) > 100
    (figure,) = example["figures"]
    assert Path(figure["file"]).is_file() and figure["caption"]
    assert figure["name"] == Path(figure["file"]).name


# ---------------------------------------------------------------------------
# Progress from state.json
# ---------------------------------------------------------------------------


def stage_record(status: str = "pending", **fields) -> dict:
    record = {
        "status": status,
        "attempts": 0,
        "seconds": 0.0,
        "error": None,
        "outputs": [],
        "reused": False,
    }
    record.update(fields)
    return record


def write_state(run_dir: Path, status: str, stages: dict, error: str | None = None) -> None:
    """A ``state.json`` in the shape the pipeline writes (version 1)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "start_time": 1767225600.0,
        "inputs": "fingerprint",
        "config": {},
        "status": status,
        "error": error,
        "stages": stages,
        "updated_at": "2026-01-01T00:10:00Z",
    }
    (run_dir / "state.json").write_text(json.dumps(state))


def test_progress_before_the_pipeline_wrote_any_state(tmp_path):
    progress = fu.read_progress(tmp_path, use_plotting=True)
    assert progress.status == "starting"
    assert [stage.name for stage in progress.stages] == [
        "ingest",
        "outline",
        "literature",
        "plotting",
        "section_writing",
        "refinement",
        "finalize",
    ]
    assert {stage.status for stage in progress.stages} == {"pending"}
    assert progress.fraction == 0.0 and progress.active == ()

    without_plots = fu.read_progress(tmp_path, use_plotting=False)
    assert "plotting" not in [stage.name for stage in without_plots.stages]
    # The process died before writing anything: that is not "starting".
    assert fu.read_progress(tmp_path, process_running=False).status == "interrupted"


def test_progress_mid_run_reports_parallel_stages_attempts_and_failures(tmp_path):
    write_state(
        tmp_path,
        "running",
        {
            "ingest": stage_record("completed", attempts=1, seconds=0.4, outputs=["a"]),
            "outline": stage_record("completed", attempts=1, seconds=75.25, reused=True),
            "literature": stage_record("running", attempts=2, error=None),
            "plotting": stage_record(
                "failed", attempts=2, seconds=31.0, error="ComputeError: sandbox", tolerated=True
            ),
            "section_writing": stage_record(),
            "refinement": stage_record(),
            "finalize": stage_record(),
        },
    )
    progress = fu.read_progress(tmp_path, use_plotting=True)
    assert progress.status == "running" and progress.error is None
    by_name = {stage.name: stage for stage in progress.stages}
    assert list(by_name) == list(fu.STAGE_LABELS)
    assert by_name["outline"] == fu.StageProgress(
        "outline", "Outline Generation", "completed", attempts=1, seconds=75.25, reused=True
    )
    assert by_name["literature"].status == "running" and by_name["literature"].attempts == 2
    assert by_name["plotting"].status == "failed" and by_name["plotting"].tolerated
    assert by_name["plotting"].error == "ComputeError: sandbox"
    assert by_name["refinement"].status == "pending"
    assert progress.active == ("Literature Review",)
    assert progress.fraction == pytest.approx(2 / 7)


def test_progress_of_a_finished_and_of_a_failed_run(tmp_path):
    names = ["ingest", "outline", "literature", "section_writing", "refinement", "finalize"]
    done = tmp_path / "done"
    write_state(done, "completed", {name: stage_record("completed", attempts=1) for name in names})
    progress = fu.read_progress(done, use_plotting=False, process_running=False)
    assert progress.status == "completed" and progress.fraction == 1.0
    assert "plotting" not in [stage.name for stage in progress.stages]

    failed = tmp_path / "failed"
    stages = {name: stage_record() for name in names}
    stages["ingest"] = stage_record("completed", attempts=1)
    stages["outline"] = stage_record("failed", attempts=2, error="LLMConfigError: no credentials")
    message = "stage 'outline' failed after 2 attempt(s): LLMConfigError: no credentials"
    write_state(failed, "failed", stages, error=message)
    progress = fu.read_progress(failed, use_plotting=False, process_running=False)
    assert progress.status == "failed" and progress.error == message
    assert progress.stages[1].error == "LLMConfigError: no credentials"


def test_a_killed_run_is_not_reported_as_still_running(tmp_path):
    write_state(
        tmp_path,
        "running",
        {"ingest": stage_record("completed"), "outline": stage_record("running", attempts=1)},
    )
    alive = fu.read_progress(tmp_path, process_running=True)
    assert alive.status == "running" and alive.stages[1].status == "running"
    dead = fu.read_progress(tmp_path, process_running=False)
    assert dead.status == "interrupted"
    assert [stage.status for stage in dead.stages] == ["completed", "interrupted"]
    assert dead.active == ()


def test_unknown_stages_and_statuses_are_shown_rather_than_dropped(tmp_path):
    write_state(
        tmp_path,
        "running",
        {
            "review_panel": stage_record("warming_up"),
            "outline": stage_record("completed"),
            "broken": "not a record",
        },
    )
    progress = fu.read_progress(tmp_path)
    assert [(s.name, s.label, s.status) for s in progress.stages] == [
        ("outline", "Outline Generation", "completed"),
        ("review_panel", "Review Panel", "pending"),
        ("broken", "Broken", "pending"),
    ]


def test_the_status_words_of_the_pipeline_spec_are_understood(tmp_path):
    """The spec calls a finished run 'succeeded' and lists a 'skipped' stage status."""
    write_state(
        tmp_path,
        "succeeded",
        {
            "ingest": stage_record("completed", attempts=1),
            "plotting": stage_record("skipped"),
            "finalize": stage_record("completed", attempts=1),
        },
    )
    progress = fu.read_progress(tmp_path, process_running=False)
    assert progress.status == "completed"
    assert [stage.status for stage in progress.stages] == ["completed", "skipped", "completed"]
    assert progress.done == 3 and progress.fraction == 1.0


def test_odd_values_in_the_state_file_do_not_break_the_page(tmp_path):
    write_state(
        tmp_path,
        ["running"],
        {
            "ingest": stage_record("completed", attempts="two", seconds="soon"),
            "outline": stage_record("failed", attempts=None, seconds=float("nan"), error=["boom"]),
            "literature": stage_record(["running"], attempts=[1], seconds={"s": 1}),
        },
        error={"stage": "outline"},
    )
    progress = fu.read_progress(tmp_path, process_running=False)
    assert progress.status == "interrupted"
    assert progress.error == "{'stage': 'outline'}"
    ingest, outline, literature = progress.stages
    assert (ingest.attempts, ingest.seconds) == (0, 0.0)
    assert (outline.attempts, outline.seconds, outline.error) == (0, 0.0, "['boom']")
    assert (literature.status, literature.attempts, literature.seconds) == ("pending", 0, 0.0)


def test_unreadable_state_counts_as_no_state(tmp_path, caplog):
    (tmp_path / "state.json").write_text('{"stages": {"outline"')
    with caplog.at_level(logging.WARNING):
        assert fu.read_run_state(tmp_path) is None
    assert "state.json" in caplog.text
    (tmp_path / "state.json").write_text('{"status": "running"}')
    assert fu.read_run_state(tmp_path) is None
    assert fu.read_progress(tmp_path).status == "starting"


# ---------------------------------------------------------------------------
# Wiki memory
# ---------------------------------------------------------------------------


@pytest.fixture
def run_with_memory(tmp_path):
    run_dir = tmp_path / "run"
    wiki = Wiki(run_dir / "memory")
    wiki.write(
        "wiki/sources/idea.md",
        "We propose X.",
        {"title": "Idea", "summary": "The idea, verbatim.", "kind": "source"},
        force=True,
    )
    wiki.write(
        "wiki/literature/He2016.md",
        "> Deep residual learning.\n\nSee [[sources/idea]].",
        {"title": "Deep Residual Learning", "summary": "ResNets.", "kind": "literature"},
    )
    wiki.write("wiki/overview.md", "# Overview\n\nTop-level page.", {"summary": "Overview."})
    wiki.add_note("pdflatex failed on an unescaped %.", stage="section_writing", kind="lesson")
    wiki.set_standing_notes("- Escape percent signs in captions.")
    return run_dir


def test_opening_the_wiki_of_a_run_without_memory_creates_nothing(tmp_path):
    assert fu.open_run_wiki(tmp_path / "run") is None
    assert not (tmp_path / "run").exists()

    # A memory directory that only has its index so far is viewed as it is.
    memory_dir = tmp_path / "early" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "MEMORY.md").write_text("# Memory\n")
    wiki = fu.open_run_wiki(tmp_path / "early")
    assert fu.list_wiki_pages(wiki) == []
    assert fu.memory_index_markdown(wiki) == "# Memory"
    assert sorted(path.name for path in memory_dir.iterdir()) == ["MEMORY.md"]


def test_wiki_listing_survives_pages_that_vanish_are_unreadable_or_point_outside(
    run_with_memory, tmp_path, monkeypatch
):
    """The run moves and archives pages while the page is polling: one bad page is not fatal."""
    wiki = fu.open_run_wiki(run_with_memory)
    wiki_dir = run_with_memory / "memory" / "wiki"
    (wiki_dir / "broken.md").write_bytes(b"\xff\xfe not utf-8")
    secret = tmp_path / "secret.md"
    secret.write_text("# Outside\n\nNot part of the memory.")
    (wiki_dir / "escape.md").symlink_to(secret)
    (wiki_dir / ".hidden.md").write_text("hidden")

    # A page that is listed by the directory scan but gone when it is read.
    real_read = wiki.read

    def read(rel: str):
        if rel == "wiki/overview.md":
            raise FileNotFoundError(rel)
        return real_read(rel)

    monkeypatch.setattr(wiki, "read", read)
    paths = [entry.path for entry in fu.list_wiki_pages(wiki, include_notes=False)]
    assert paths == ["wiki/literature/He2016.md", "wiki/sources/idea.md"]

    monkeypatch.undo()
    assert fu.read_wiki_page(wiki, "wiki/broken.md") is None
    assert fu.read_wiki_page(wiki, "wiki/escape.md") is None
    assert fu.read_wiki_page(wiki, "wiki/.hidden.md") is None
    assert fu.read_wiki_page(wiki, "transcripts/session.jsonl") is None
    assert fu.read_wiki_page(wiki, "/etc/passwd") is None

    (run_with_memory / "memory" / "MEMORY.md").unlink()
    assert fu.memory_index_markdown(wiki) == ""


def test_memory_pages_do_not_make_the_browser_fetch_images(run_with_memory):
    wiki = Wiki(run_with_memory / "memory")
    wiki.write(
        "wiki/synthesis/themes.md",
        "Intro ![chart](https://tracker.example/pixel.png?run=1) and ![ref-style][logo].\n\n"
        "A [normal link](https://example.org) and an exclamation! [kept](https://example.org).",
        {"summary": "Themes."},
    )
    _, body = fu.read_wiki_page(fu.open_run_wiki(run_with_memory), "wiki/synthesis/themes.md")
    assert "![" not in body
    assert "[chart](https://tracker.example/pixel.png?run=1)" in body
    assert "[ref-style][logo]" in body
    assert "A [normal link](https://example.org) and an exclamation! [kept]" in body
    assert fu.without_embedded_images("![a](b) ![c](d)") == "[a](b) [c](d)"


def test_wiki_pages_are_listed_by_section_with_notes_last(run_with_memory):
    wiki = fu.open_run_wiki(run_with_memory)
    entries = fu.list_wiki_pages(wiki)
    assert [(entry.section, entry.title) for entry in entries[:3]] == [
        ("literature", "Deep Residual Learning"),
        ("wiki", "Overview"),
        ("sources", "Idea"),
    ]
    assert entries[0] == fu.WikiEntry(
        "wiki/literature/He2016.md",
        "Deep Residual Learning",
        "ResNets.",
        "literature",
        "literature",
    )
    assert entries[0].label == "literature / Deep Residual Learning"
    note = entries[3]
    assert note.section == "notes" and note.kind == "lesson"
    assert note.path.startswith("notes/") and len(entries) == 4
    assert [e.section for e in fu.list_wiki_pages(wiki, include_notes=False)] == [
        "literature",
        "wiki",
        "sources",
    ]


def test_memory_index_and_pages_are_readable_without_changing_the_wiki(run_with_memory):
    memory_dir = run_with_memory / "memory"
    before = {
        path.relative_to(memory_dir).as_posix(): path.read_bytes()
        for path in sorted(memory_dir.rglob("*"))
        if path.is_file()
    }
    wiki = fu.open_run_wiki(run_with_memory)

    index = fu.memory_index_markdown(wiki)
    assert "<!--" not in index
    assert "- Escape percent signs in captions." in index
    assert "[[literature/He2016]]: ResNets." in index

    meta, body = fu.read_wiki_page(wiki, "wiki/literature/He2016.md")
    assert meta["title"] == "Deep Residual Learning" and meta["kind"] == "literature"
    assert body.startswith("> Deep residual learning.")
    assert fu.read_wiki_page(wiki, "wiki/literature/Gone.md") is None
    assert fu.read_wiki_page(wiki, "../../etc/passwd") is None

    after = {
        path.relative_to(memory_dir).as_posix(): path.read_bytes()
        for path in sorted(memory_dir.rglob("*"))
        if path.is_file()
    }
    assert after == before


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def make_latex_workspace(run_dir: Path) -> Path:
    latex = run_dir / "latex_writeup"
    (latex / "figures").mkdir(parents=True)
    (latex / "template.tex").write_text("intro draft")
    (latex / "raw_draft_paper.tex").write_text("raw draft")
    (latex / "references.bib").write_text("@article{a}")
    (latex / "figures" / "fig1.png").write_bytes(b"png")
    return latex


def zip_names(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_latex_zip_of_a_finished_run_is_ready_to_compile(tmp_path):
    assert fu.latex_source_zip(tmp_path) is None and fu.final_tex_path(tmp_path) is None
    make_latex_workspace(tmp_path)
    refined = tmp_path / "content_refinement_workdir" / "final_refined_paper.tex"
    refined.parent.mkdir()
    refined.write_text("refined paper")
    assert fu.final_tex_path(tmp_path) == refined
    assert zip_names(fu.latex_source_zip(tmp_path)) == {
        "figures/fig1.png": b"png",
        "references.bib": b"@article{a}",
        "main.tex": b"refined paper",
    }
    # The copy the pipeline places at the top of the run wins when it exists.
    (tmp_path / "final_paper.tex").write_text("final paper")
    assert zip_names(fu.latex_source_zip(tmp_path))["main.tex"] == b"final paper"


def test_latex_zip_of_an_unfinished_run_keeps_the_drafts(tmp_path):
    make_latex_workspace(tmp_path)
    assert sorted(zip_names(fu.latex_source_zip(tmp_path))) == [
        "figures/fig1.png",
        "raw_draft_paper.tex",
        "references.bib",
        "template.tex",
    ]
    assert fu.result_attachments(tmp_path) == []


def test_result_attachments_are_the_pdf_and_the_zipped_source(tmp_path):
    make_latex_workspace(tmp_path)
    (tmp_path / "final_paper.tex").write_text("final paper")
    (tmp_path / "final_paper.pdf").write_bytes(b"%PDF-1.7")
    attachments = fu.result_attachments(tmp_path)
    assert attachments == [tmp_path / "final_paper.pdf", tmp_path / "latex_source.zip"]
    assert "main.tex" in zip_names(attachments[1].read_bytes())


def test_last_accepted_version_follows_the_refinement_loop():
    worklog = {
        "v1": {"round": 1, "outcome": "ACCEPTED_SCORE_INCREASE"},
        "v2": {"round": 2, "outcome": "ACCEPTED_NEUTRAL_IMPROVEMENT"},
        "v3": {"round": 3, "outcome": "REJECTED_SCORE_DECREASE"},
    }
    # The old page reported the first accepted round, although each round builds on the last.
    assert fu.last_accepted_version(worklog) == "v2"
    assert fu.last_accepted_version({"v1": {"outcome": "REJECTED_DEGRADATION"}}) is None
    assert fu.last_accepted_version({}) is None
    formatting = {
        "v1": {"outcome": "ACCEPTED_COMPILE_SUCCESS"},
        "v2": {"outcome": "REJECTED_COMPILE_FAILURE"},
        "v10": {"outcome": "ACCEPTED_COMPILE_SUCCESS"},
    }
    assert fu.last_accepted_version(formatting) == "v10"
    assert fu.version_number("v10") == 10 and fu.version_number("draft") == 0


# ---------------------------------------------------------------------------
# E-mail
# ---------------------------------------------------------------------------


class FakeSMTP:
    instances: ClassVar[list[FakeSMTP]] = []
    fail_login = False

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.logins: list[tuple[str, str]] = []
        self.messages: list = []
        FakeSMTP.instances.append(self)

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def login(self, user: str, password: str) -> None:
        if FakeSMTP.fail_login:
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
        self.logins.append((user, password))

    def send_message(self, message) -> None:
        self.messages.append(message)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    FakeSMTP.fail_login = False
    monkeypatch.setattr(fu.smtplib, "SMTP_SSL", FakeSMTP)
    for name in ("SMTP_EMAIL", "SMTP_PASSWORD", "SMTP_HOST", "SMTP_PORT"):
        monkeypatch.delenv(name, raising=False)
    return FakeSMTP


def test_email_is_not_attempted_without_smtp_settings(smtp, monkeypatch):
    sent, message = fu.send_notification_email("reader@example.org", "Generated Paper")
    assert not sent and "SMTP_EMAIL" in message
    monkeypatch.setenv("SMTP_EMAIL", "bot@example.org")
    assert fu.send_notification_email("reader@example.org", "Generated Paper")[0] is False
    assert smtp.instances == []


def test_email_reads_settings_at_call_time_and_attaches_the_results(
    smtp, monkeypatch, tmp_path, caplog
):
    pdf = tmp_path / "final_paper.pdf"
    pdf.write_bytes(b"%PDF-1.7 paper")
    monkeypatch.setenv("SMTP_EMAIL", "bot@example.org")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2-secret")
    with caplog.at_level(logging.DEBUG):
        sent, message = fu.send_notification_email(
            "reader@example.org", "Generated Paper", [pdf, tmp_path / "missing.zip"]
        )
    assert sent and "missing.zip" in message
    (server,) = smtp.instances
    assert (server.host, server.port) == ("smtp.gmail.com", 465)
    assert server.timeout == fu.SMTP_TIMEOUT_SECONDS
    assert server.logins == [("bot@example.org", "hunter2-secret")]
    mail = message_from_bytes(server.messages[0].as_bytes())
    assert mail["To"] == "reader@example.org" and mail["From"] == "bot@example.org"
    assert mail["Subject"] == "Paper Orchestra: Generated Paper is Complete!"
    attached = {part.get_filename(): part.get_payload(decode=True) for part in mail.walk()}
    assert attached["final_paper.pdf"] == b"%PDF-1.7 paper"
    assert "hunter2-secret" not in caplog.text and "hunter2-secret" not in message

    monkeypatch.setenv("SMTP_HOST", "mail.example.org")
    monkeypatch.setenv("SMTP_PORT", "2465")
    assert fu.send_notification_email("reader@example.org", "Generated Paper", [pdf]) == (
        True,
        "Email sent successfully!",
    )
    assert (smtp.instances[1].host, smtp.instances[1].port) == ("mail.example.org", 2465)


@pytest.mark.parametrize(
    "address",
    [
        "reader@example.org\nBcc: victim@example.org",
        "reader@example.org, other@example.org",
        "Reader <reader@example.org>",
        "not an address",
        "reader@localhost",
        "",
    ],
)
def test_email_is_only_sent_to_one_plain_address(smtp, monkeypatch, address):
    """The address is typed into a public page: it must not add recipients or headers."""
    monkeypatch.setenv("SMTP_EMAIL", "bot@example.org")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    assert not fu.valid_email_address(address)
    sent, message = fu.send_notification_email(address, "Generated Paper")
    assert not sent and "not a valid e-mail address" in message
    assert smtp.instances == []
    assert fu.valid_email_address("first.last+tag@mail.example.org")


def test_email_failures_are_reported_without_the_password(smtp, monkeypatch):
    monkeypatch.setenv("SMTP_EMAIL", "bot@example.org")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2-secret")
    smtp.fail_login = True
    sent, message = fu.send_notification_email("reader@example.org", "Generated Paper")
    assert not sent and "SMTPAuthenticationError" in message
    assert "hunter2-secret" not in message

    monkeypatch.setenv("SMTP_PORT", "not-a-port")
    assert fu.send_notification_email("reader@example.org", "Generated Paper") == (
        False,
        "SMTP_PORT must be a number.",
    )


# ---------------------------------------------------------------------------
# The pipeline process
# ---------------------------------------------------------------------------


def python_command(source: str, *arguments: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(source), *arguments]


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        # A zombie still has a pid but is no longer running.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def wait_until(condition, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_pipeline_run_captures_output_and_reports_completion(tmp_path):
    finished: list[fu.PipelineRun] = []
    command = python_command(
        """
        import os, sys
        print("unbuffered:", os.environ.get("PYTHONUNBUFFERED"))
        print("cwd:", os.getcwd())
        for n in range(5):
            print("line", n)
        print("to stderr", file=sys.stderr)
        """
    )
    run = fu.PipelineRun(
        command,
        cwd=tmp_path,
        log_path=tmp_path / "logs" / "pipeline.log",
        on_finish=finished.append,
    )
    run.start()
    assert run.wait(timeout=30)
    assert not run.running and run.returncode == 0
    assert run.launch_error is None and not run.stopped
    assert finished == [run]
    output = run.log_tail()
    assert "unbuffered: 1" in output and f"cwd: {tmp_path}" in output
    assert "to stderr" in output
    assert run.log_tail(lines=2).count("\n") == 2
    assert (tmp_path / "logs" / "pipeline.log").read_text() == output


def test_credentials_echoed_by_the_pipeline_are_masked_in_the_tail_and_on_disk(
    tmp_path, monkeypatch
):
    """The log tail is shown in the browser; a library may print a key in an error message."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-0123456789")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2-secret")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-secret-key")
    command = python_command(
        """
        import os
        print("request failed: key=" + os.environ["ANTHROPIC_API_KEY"] + " rejected")
        print("smtp", os.environ["SMTP_PASSWORD"], "s2", os.environ["SEMANTIC_SCHOLAR_API_KEY"])
        print("an ordinary line")
        """
    )
    run = fu.PipelineRun(command, cwd=tmp_path, log_path=tmp_path / "pipeline.log")
    run.start()
    assert run.wait(timeout=30) and run.returncode == 0
    for output in (run.log_tail(), (tmp_path / "pipeline.log").read_text()):
        assert "request failed: key=*** rejected" in output
        assert "smtp *** s2 ***" in output and "an ordinary line" in output
        assert "secret" not in output


def test_secret_values_are_chosen_by_variable_name():
    environ = {
        "ANTHROPIC_API_KEY": "sk-ant-0123456789",
        "ANTHROPIC_AUTH_TOKEN": "token-0123456789-long",
        "MODAL_TOKEN_SECRET": "as-0123456789",
        "SMTP_PASSWORD": "correct horse!",
        "ANTHROPIC_IDENTITY_TOKEN_FILE": "/var/run/secrets/token",
        "OPENAI_API_KEY": "short",
        "PATH": "/usr/local/bin:/usr/bin",
        "KEYBOARD_LAYOUT": "us-international",
    }
    assert fu.secret_values(environ) == (
        "token-0123456789-long",
        "sk-ant-0123456789",
        "correct horse!",
        "as-0123456789",
    )
    assert fu.mask_secrets("a correct horse b", ["correct horse"]) == "a *** b"


def test_pipeline_run_keeps_only_the_tail_in_memory_but_everything_on_disk(tmp_path):
    command = python_command("for n in range(50): print('line', n)")
    run = fu.PipelineRun(
        command, cwd=tmp_path, log_path=tmp_path / "pipeline.log", max_log_lines=10
    )
    run.start()
    assert run.wait(timeout=30)
    assert run.log_tail(lines=100).splitlines() == [f"line {n}" for n in range(40, 50)]
    assert len((tmp_path / "pipeline.log").read_text().splitlines()) == 50


def test_pipeline_run_reports_a_failing_exit_code_and_a_failed_launch(tmp_path):
    run = fu.PipelineRun(python_command("raise SystemExit(3)"), cwd=tmp_path)
    run.start()
    assert run.wait(timeout=30)
    assert run.returncode == 3
    assert not fu.run_succeeded(run, tmp_path)

    finished: list[fu.PipelineRun] = []
    missing = fu.PipelineRun(
        [str(tmp_path / "no-such-interpreter"), "-m", "paper_orchestra"],
        cwd=tmp_path,
        on_finish=finished.append,
    )
    missing.start()
    assert missing.wait(timeout=30)
    assert "could not be started" in missing.launch_error
    assert missing.returncode is None and not missing.running
    assert finished == [missing]
    missing.stop()  # nothing to stop: must not raise


def test_success_needs_both_a_clean_exit_and_the_final_pdf(tmp_path):
    """The old page trusted exit code 0 alone and announced papers that did not exist."""
    run = fu.PipelineRun(python_command("pass"), cwd=tmp_path)
    run.start()
    assert run.wait(timeout=30) and run.returncode == 0
    assert not fu.run_succeeded(run, tmp_path)
    (tmp_path / "final_paper.pdf").write_bytes(b"%PDF-1.7")
    assert fu.run_succeeded(run, tmp_path)


@posix_only
def test_stop_ends_the_whole_process_group_even_children_that_ignore_the_interrupt(tmp_path):
    command = python_command(
        """
        import os, signal, subprocess, sys, time
        stubborn = "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(120)"
        child = subprocess.Popen([sys.executable, "-c", stubborn])
        print("leader", os.getpid(), "session", os.getsid(0), "child", child.pid, flush=True)
        try:
            time.sleep(120)
        except KeyboardInterrupt:
            print("interrupted", flush=True)
            raise SystemExit(130)
        """
    )
    run = fu.PipelineRun(command, cwd=tmp_path)
    run.start()
    assert wait_until(lambda: "leader" in run.log_tail())
    words = run.log_tail().split()
    leader, session, child = int(words[1]), int(words[3]), int(words[5])
    # Its own session and process group: a Ctrl+C in the server's terminal does not reach it.
    assert session == leader and os.getpgid(leader) == leader != os.getpgid(0)
    assert process_alive(leader) and process_alive(child)

    started = time.monotonic()
    run.stop(grace=10.0)
    assert time.monotonic() - started < 10.0  # the interrupt was honoured, no need to wait it out
    assert run.stopped and not run.running
    assert run.returncode == 130 and "interrupted" in run.log_tail()
    assert wait_until(lambda: not process_alive(leader) and not process_alive(child))


@posix_only
def test_stop_reaches_children_the_pipeline_started_in_sessions_of_their_own(tmp_path):
    """The local compute backend runs LaTeX and figure code that way; they used to be orphaned."""
    command = python_command(
        """
        import subprocess, sys, time
        detached = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
        )
        print("detached", detached.pid, flush=True)
        try:
            detached.communicate(timeout=120)
        except KeyboardInterrupt:
            raise SystemExit(130)
        """
    )
    run = fu.PipelineRun(command, cwd=tmp_path)
    run.start()
    assert wait_until(lambda: "detached" in run.log_tail())
    detached = int(run.log_tail().split()[1])
    assert process_alive(detached) and os.getpgid(detached) != os.getpgid(run._process.pid)
    assert fu._descendant_groups(run._process.pid) == {os.getpgid(detached)}
    assert os.getpgid(0) not in fu._descendant_groups(os.getpid())

    run.stop(grace=10.0)
    assert run.stopped and not run.running
    assert wait_until(lambda: not process_alive(detached))


@posix_only
def test_the_process_table_can_also_be_read_with_ps(tmp_path):
    """Systems without /proc (macOS) list processes with ps; both must find this process."""
    if fu.shutil.which("ps") is None:
        pytest.skip("ps is not installed")
    me = (os.getpid(), os.getppid(), os.getpgid(0))
    assert me in fu._process_table(tmp_path / "no-proc-here")
    if Path("/proc/self/stat").is_file():
        assert me in fu._process_table()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-specific")
def test_the_kernel_is_asked_to_kill_the_pipeline_when_its_parent_dies(tmp_path):
    command = python_command(
        """
        import ctypes, signal
        current = ctypes.c_int(0)
        ctypes.CDLL(None).prctl(2, ctypes.byref(current))  # PR_GET_PDEATHSIG
        print("death signal:", signal.Signals(current.value).name)
        """
    )
    run = fu.PipelineRun(command, cwd=tmp_path)
    run.start()
    assert run.wait(timeout=30)
    assert "death signal: SIGKILL" in run.log_tail()


def test_process_isolation_and_stop_signals_on_macos_and_windows(monkeypatch):
    """Only Linux gets the prctl hook, and Windows gets neither sessions nor killpg."""

    class Process:
        pid = 4242

        def __init__(self) -> None:
            self.signals: list[int] = []

        def send_signal(self, number: int) -> None:
            self.signals.append(number)

    with monkeypatch.context() as patched:
        patched.setattr(fu.sys, "platform", "darwin")
        assert fu._isolation_options() == {"start_new_session": True}

    process = Process()
    commands: list[tuple[list[str], dict]] = []
    with monkeypatch.context() as patched:
        patched.setattr(fu.os, "name", "nt")
        patched.setattr(fu.signal, "CTRL_BREAK_EVENT", 21, raising=False)
        patched.setattr(fu.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
        patched.setattr(
            fu.subprocess, "run", lambda command, **options: commands.append((command, options))
        )
        patched.setattr(fu.os, "killpg", None, raising=False)  # must not be touched on Windows
        windows_options = fu._isolation_options()
        fu._signal_tree(process, force=False)
        fu._signal_tree(process, force=True)

    assert windows_options == {"creationflags": 0x200}
    assert process.signals == [21]
    assert commands == [
        (["taskkill", "/PID", "4242", "/T", "/F"], {"capture_output": True, "check": False})
    ]


@posix_only
def test_stop_kills_a_pipeline_that_ignores_the_interrupt(tmp_path):
    command = python_command(
        """
        import signal, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("ready", flush=True)
        time.sleep(120)
        """
    )
    run = fu.PipelineRun(command, cwd=tmp_path)
    run.start()
    assert wait_until(lambda: "ready" in run.log_tail())
    run.stop(grace=0.3)
    assert not run.running and run.stopped
    assert run.returncode is not None and run.returncode < 0


# ---------------------------------------------------------------------------
# Completion handler
# ---------------------------------------------------------------------------


def finished_run(tmp_path: Path, source: str = "pass") -> fu.PipelineRun:
    run = fu.PipelineRun(python_command(source), cwd=tmp_path)
    run.start()
    assert run.wait(timeout=30)
    return run


def test_completion_emails_the_results_of_a_successful_run(smtp, monkeypatch, tmp_path):
    monkeypatch.setenv("SMTP_EMAIL", "bot@example.org")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    make_latex_workspace(tmp_path)
    (tmp_path / "final_paper.tex").write_text("final paper")
    (tmp_path / "final_paper.pdf").write_bytes(b"%PDF-1.7")
    run = finished_run(tmp_path)

    fu.notify_completion(run, tmp_path, "reader@example.org")
    assert run.notification == "Email sent successfully!"
    mail = message_from_bytes(smtp.instances[0].messages[0].as_bytes())
    names = {part.get_filename() for part in mail.walk()} - {None}
    assert names == {"final_paper.pdf", "latex_source.zip"}


def test_completion_sends_nothing_for_failed_stopped_or_unaddressed_runs(
    smtp, monkeypatch, tmp_path
):
    monkeypatch.setenv("SMTP_EMAIL", "bot@example.org")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    failed = finished_run(tmp_path, "raise SystemExit(1)")
    fu.notify_completion(failed, tmp_path, "reader@example.org")
    assert "did not produce a paper" in failed.notification

    (tmp_path / "final_paper.pdf").write_bytes(b"%PDF-1.7")
    succeeded = finished_run(tmp_path)
    fu.notify_completion(succeeded, tmp_path, None)
    assert succeeded.notification is None
    succeeded.stopped = True
    fu.notify_completion(succeeded, tmp_path, "reader@example.org")
    assert succeeded.notification is None
    assert smtp.instances == []


# ---------------------------------------------------------------------------
# The Streamlit page
# ---------------------------------------------------------------------------

STUB_PIPELINE = """
import json, sys
from pathlib import Path
from paper_orchestra.memory import Wiki

run_dir = Path(sys.argv[1])
arguments = sys.argv[2:]
materials = run_dir / "raw_materials"
print("stub pipeline started with", " ".join(arguments))
names = ["ingest", "outline", "literature", "plotting", "section_writing", "refinement", "finalize"]
record = {"status": "completed", "attempts": 1, "seconds": 1.5, "error": None, "outputs": [], "reused": False}
state = {"version": 1, "status": "completed", "error": None, "stages": {name: dict(record) for name in names}}
(run_dir / "state.json").write_text(json.dumps(state))
(run_dir / "outline.json").write_text(json.dumps({"idea_seen": (materials / "idea_sparse.md").read_text()}))
(run_dir / "final_paper.pdf").write_bytes(b"%PDF-1.7 stub")
wiki = Wiki(run_dir / "memory")
wiki.write("wiki/lessons/escape-percent.md", "Escape percent signs.", {"title": "Escape percent", "summary": "A LaTeX pitfall."})
wiki.rebuild_index()
print("stub pipeline finished")
"""


@pytest.fixture
def app(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    run_dir = tmp_path / "demo" / "paper_orchestra_run_test"
    commands: list[list[str]] = []

    def stub_command(options: fu.RunOptions) -> list[str]:
        commands.append(build_write_command(options))
        return [sys.executable, "-c", STUB_PIPELINE, str(options.run_dir), *commands[-1][4:]]

    build_write_command = fu.build_write_command
    monkeypatch.setattr(fu, "build_write_command", stub_command)
    monkeypatch.setattr(fu, "new_run_directory", lambda: run_dir)
    monkeypatch.setattr(fu, "POLL_SECONDS", 0.05)
    # A developer's .env must not leak credentials into the test process.
    monkeypatch.setattr("paper_orchestra.config.load_env", lambda *args, **kwargs: None)
    at = testing.AppTest.from_file(str(FRONTEND_DIR / "app.py"), default_timeout=60)
    at.commands = commands
    at.run_dir = run_dir
    return at


def test_page_offers_the_new_controls_and_rejects_an_empty_form(app):
    app.run()
    assert not app.exception
    labels = {box.label: box for box in app.sidebar.selectbox}
    assert next(iter(labels["Select Model"].options)) == "claude-fable-5-2"
    assert labels["Select Model"].value == "claude-fable-5-2"
    assert list(labels["Compute Backend"].options) == ["local", "modal"]
    assert labels["Diagram Image Model"].value.startswith("None")
    assert len(labels["Memory Consolidation (Dreaming)"].options) == 3
    assert app.sidebar.checkbox[0].label == "Wiki Memory" and app.sidebar.checkbox[0].value is True

    next(button for button in app.button if "Begin Writing" in button.label).click().run()
    assert not app.exception
    assert any("required fields" in error.value for error in app.error)
    assert app.commands == [] and not app.run_dir.exists()


def test_page_runs_the_pipeline_and_shows_progress_results_and_memory(app):
    app.run()
    app.text_area(key="idea_text_main").set_value("A new idea.")
    app.text_area(key="log_text_main").set_value("Accuracy 91.2%.")
    selectboxes = {box.label: box for box in app.sidebar.selectbox}
    selectboxes["Compute Backend"].set_value("modal")
    selectboxes["Memory Consolidation (Dreaming)"].set_value("end")
    app.run()
    next(button for button in app.button if "Begin Writing" in button.label).click().run()
    assert not app.exception

    # The page polled until the child process was over, then rendered the final state.
    run = app.session_state.run
    assert not run.running and run.returncode == 0
    (command,) = app.commands
    assert command[:4] == [sys.executable, "-m", "paper_orchestra", "write"]
    assert flag_value(command, "--compute") == "modal" and flag_value(command, "--dream") == "end"
    assert flag_value(command, "--writer-model") == "claude-fable-5-2"
    assert (app.run_dir / "raw_materials" / "idea_sparse.md").read_text() == "A new idea."
    assert "stub pipeline finished" in (app.run_dir / "pipeline.log").read_text()

    assert any("Paper writing complete" in box.value for box in app.success)
    page = "\n".join(element.value for element in app.markdown)
    for label in ("Outline Generation", "Literature Review", "Figure Generation", "Finalizing"):
        assert label in page
    assert "Completed in 1s." in page
    assert any("stub pipeline finished" in block.value for block in app.code)
    assert json.loads(app.json[0].value) == {"idea_seen": "A new idea."}
    # Memory tab: the injected index and a selectable wiki page.
    assert "[[lessons/escape-percent]]: A LaTeX pitfall." in page
    assert "Escape percent signs." in page
    assert app.selectbox(key="memory_page").value == "wiki/lessons/escape-percent.md"
    assert {button.label for button in app.button} >= {"Start a New Paper"}


def test_a_tab_that_fails_does_not_stop_the_page_from_following_the_run(app, monkeypatch):
    """An exception in one tab used to end the script before its refresh: a frozen page."""

    def exploding_wiki(run_dir):
        raise RuntimeError("wiki exploded")

    monkeypatch.setattr(fu, "open_run_wiki", exploding_wiki)
    app.run()
    app.text_area(key="idea_text_main").set_value("A new idea.")
    app.text_area(key="log_text_main").set_value("Accuracy 91.2%.")
    app.run()
    next(button for button in app.button if "Begin Writing" in button.label).click().run()

    assert not app.exception
    assert any("RuntimeError: wiki exploded" in error.value for error in app.error)
    # The page kept polling to the end of the run and shows its outcome and the other tabs.
    assert not app.session_state.run.running
    assert any("Paper writing complete" in box.value for box in app.success)
    assert json.loads(app.json[0].value) == {"idea_seen": "A new idea."}


def test_page_refuses_to_start_with_an_invalid_notification_address(app):
    app.run()
    app.text_area(key="idea_text_main").set_value("A new idea.")
    app.text_area(key="log_text_main").set_value("Accuracy 91.2%.")
    app.sidebar.text_input(key="email_input").set_value("reader@example.org, other@example.org")
    app.run()
    next(button for button in app.button if "Begin Writing" in button.label).click().run()
    assert not app.exception
    assert any("email address is not valid" in error.value for error in app.error)
    assert app.commands == [] and not app.run_dir.exists()


def test_an_upload_that_is_not_an_image_stays_deletable_and_does_not_break_the_form(app):
    """Its preview used to raise above the delete and start buttons, leaving the form stuck."""
    app.run()
    app.session_state.figures = [
        {"id": 1, "file": io.BytesIO(b"not an image"), "name": "broken.png", "caption": "c"}
    ]
    app.run()
    assert not app.exception
    assert any("cannot be shown as an image" in warning.value for warning in app.warning)
    assert any("Begin Writing" in button.label for button in app.button)
    next(button for button in app.button if "Delete Image" in button.label).click().run()
    assert not app.exception and app.session_state.figures == []
