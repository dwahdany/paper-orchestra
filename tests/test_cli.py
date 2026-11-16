from __future__ import annotations

import fnmatch
import inspect
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from paper_orchestra import __version__, cli
from paper_orchestra import doctor as doctor_module
from paper_orchestra.compute.base import ComputeBackend, ComputeError
from paper_orchestra.config import DEFAULT_MODEL, PipelineConfig
from paper_orchestra.llm.base import LLMError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.llm.router import register_client, reset_clients
from paper_orchestra.memory import Wiki
from paper_orchestra.pipeline import PaperResult
from paper_orchestra.utils.common import read_json, write_json

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    """No test leaves a registered client, a changed log level or a signal handler behind."""
    logger = logging.getLogger("paper_orchestra")
    level = logger.level
    on_sigterm = signal.getsignal(signal.SIGTERM)
    yield
    # ``cli.main`` installs a SIGTERM handler for the life of its process.
    if on_sigterm is not None:
        signal.signal(signal.SIGTERM, on_sigterm)
    logger.setLevel(level)
    reset_clients()


@pytest.fixture
def materials(tmp_path: Path) -> Path:
    raw = tmp_path / "materials"
    raw.mkdir()
    (raw / "idea_sparse.md").write_text("# Idea\n\nSparse attention.\n")
    (raw / "experimental_log.md").write_text("# Log\n\n91.2 accuracy.\n")
    (raw / "notes.md").write_text("custom idea\n")
    (raw / "runs.md").write_text("custom log\n")
    return raw


@pytest.fixture
def template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "templates"
    venue = root / "demo2026"
    venue.mkdir(parents=True)
    (venue / "template.tex").write_text("\\documentclass{article}\n")
    (venue / "guidelines.md").write_text("Eight pages.\n")
    monkeypatch.setenv("PAPER_ORCHESTRA_TEMPLATES", str(root))
    return venue


class Recorder:
    """Stands in for ``run_paper``: records the config and returns a canned result."""

    def __init__(self, *, success: bool = True, raises: Exception | None = None) -> None:
        self.success = success
        self.raises = raises
        self.configs: list[PipelineConfig] = []

    def __call__(self, cfg: PipelineConfig, **kwargs: Any) -> PaperResult:
        assert kwargs == {}
        self.configs.append(cfg)
        if self.raises is not None:
            raise self.raises
        usage = {
            "total": {"requests": 7, "input_tokens": 1200, "output_tokens": 340},
            "estimated_cost_usd": 1.5,
        }
        if self.success:
            return PaperResult(True, cfg.base_dir / "final_paper.pdf", cfg.base_dir, {}, usage)
        return PaperResult(
            False, None, cfg.base_dir, {}, usage, error="stage 'outline' failed after 2 attempt(s)"
        )

    @property
    def cfg(self) -> PipelineConfig:
        assert len(self.configs) == 1
        return self.configs[0]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(cli, "run_paper", rec)
    return rec


# -- legacy flag translation -------------------------------------------------------


@pytest.mark.parametrize(("old", "new"), sorted(cli.LEGACY_OPTIONS.items()))
def test_every_old_flag_is_translated_in_both_forms(old, new):
    assert cli.translate_legacy_args([old, "value"]) == [new, "value"]
    assert cli.translate_legacy_args([f"{old}=a=b"]) == [f"{new}=a=b"]


def test_the_old_flag_set_is_complete():
    assert sorted([*cli.LEGACY_OPTIONS, cli.LEGACY_USE_PLOTTING]) == sorted(
        [
            "--raw_materials_dir",
            "--latex_template_dir",
            "--output_dir",
            "--idea_filename",
            "--experimental_log_filename",
            "--writer_model_name",
            "--reflection_model_name",
            "--research_cutoff",
            "--use_plotting",
            "--plotting_model_name",
            "--image_model_name",
            "--plotting_max_critic_rounds",
        ]
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--use_plotting", "true"], ["--use-plotting"]),
        (["--use_plotting", "True"], ["--use-plotting"]),
        (["--use_plotting", "1"], ["--use-plotting"]),
        (["--use_plotting=yes"], ["--use-plotting"]),
        (["--use_plotting", "false"], ["--no-use-plotting"]),
        (["--use_plotting", "0"], ["--no-use-plotting"]),
        (["--use_plotting=No"], ["--no-use-plotting"]),
        (["--use_plotting"], ["--use-plotting"]),
        (["--use_plotting", "--output_dir", "x"], ["--use-plotting", "--output-dir", "x"]),
    ],
)
def test_use_plotting_takes_a_value_like_the_original_cli(args, expected):
    assert cli.translate_legacy_args(args) == expected


def test_translation_leaves_everything_else_alone():
    args = ["--output-dir", "o", "--", "--output_dir", "--use_plotting", "true"]
    assert cli.translate_legacy_args(args) == args
    assert cli.translate_legacy_args(["--template", "--output_dirs"]) == [
        "--template",
        "--output_dirs",
    ]
    with pytest.raises(typer.BadParameter, match="expected true or false, got 'maybe'"):
        cli.translate_legacy_args(["--use_plotting=maybe"])


# -- write -------------------------------------------------------------------------


def test_write_accepts_the_original_command_line(materials, template, tmp_path, recorder):
    out = tmp_path / "out"
    result = runner.invoke(
        cli.app,
        [
            "write",
            "--raw_materials_dir", str(materials),
            "--latex_template_dir", str(template),
            "--output_dir", str(out),
            "--idea_filename", "notes.md",
            "--experimental_log_filename", "runs.md",
            "--writer_model_name", "gpt-writer",
            "--reflection_model_name", "gpt-reviewer",
            "--research_cutoff", "2024-11",
            "--use_plotting", "True",
            "--plotting_model_name", "gpt-plotter",
            "--image_model_name", "imagen-x",
            "--plotting_max_critic_rounds", "5",
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    cfg = recorder.cfg
    assert cfg.base_dir == out
    assert cfg.template_dir == template
    assert (cfg.idea_filename, cfg.experimental_log_filename) == ("notes.md", "runs.md")
    assert cfg.models.writer == "gpt-writer"
    assert cfg.models.reflection == "gpt-reviewer"
    assert cfg.models.plotting == "gpt-plotter"
    assert cfg.models.image == "imagen-x"
    # The original command line had no flag for the discovery model: it follows the writer.
    assert cfg.models.fast == "gpt-writer"
    assert cfg.research_cutoff == "2024-11"
    assert cfg.use_plotting is True
    assert cfg.plotting_max_critic_rounds == 5
    # The raw materials were brought into the run directory before the pipeline started.
    assert (out / "raw_materials" / "notes.md").read_text() == "custom idea\n"
    assert f"Paper written: {out / 'final_paper.pdf'}" in result.output
    assert "7 requests, 1200 input and 340 output tokens, about $1.50" in result.output


def test_write_new_flags(materials, template, tmp_path, recorder):
    result = runner.invoke(
        cli.app,
        [
            "write",
            "--raw-materials-dir", str(materials),
            "--template", "demo2026",
            "--output-dir", str(tmp_path / "out"),
            "--model", "claude-opus-5-5",
            "--fast-model", "claude-haiku-5-5",
            "--effort", "max",
            "--compute", "modal",
            "--no-memory",
            "--memory-dir", str(tmp_path / "mem"),
            "--no-lab-memory",
            "--lab-memory-dir", str(tmp_path / "lab"),
            "--dream", "end",
            "--no-resume",
            "--no-plotting",
            "--keep-provided-figures",
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    cfg = recorder.cfg
    # A template name is resolved to its directory.
    assert cfg.template_dir == template
    assert cfg.models.writer == cfg.models.reflection == cfg.models.plotting == "claude-opus-5-5"
    assert cfg.models.fast == "claude-haiku-5-5"
    assert cfg.models.effort == "max"
    assert cfg.compute == "modal"
    assert cfg.memory.enabled is False
    assert cfg.memory.project_dir == tmp_path / "mem"
    assert cfg.memory.lab is False
    assert cfg.memory.lab_dir == tmp_path / "lab"
    assert cfg.memory.dream == "end"
    assert cfg.resume is False
    assert cfg.use_plotting is False
    assert cfg.keep_provided_figures is True


@pytest.mark.parametrize(
    ("flags", "fast", "reflection"),
    [
        ([], DEFAULT_MODEL, DEFAULT_MODEL),
        (["--writer-model", "gemini-writer"], "gemini-writer", DEFAULT_MODEL),
        (["--writer_model_name=gemini-writer"], "gemini-writer", DEFAULT_MODEL),
        # What the original frontend sent for a run on one provider: every model it could name.
        (
            ["--writer_model_name", "gemini-pro", "--reflection_model_name", "gemini-pro",
             "--use_plotting", "true", "--plotting_model_name", "gemini-pro",
             "--image_model_name", "gemini-image"],
            "gemini-pro",
            "gemini-pro",
        ),
        (["--writer-model", "gemini-writer", "--fast-model", "gemini-fast"], "gemini-fast",
         DEFAULT_MODEL),
        # An explicit --model still sets every role that has no flag of its own.
        (["--writer-model", "gemini-writer", "--model", "claude-opus-5-5"], "claude-opus-5-5",
         "claude-opus-5-5"),
        (["--model", "claude-opus-5-5"], "claude-opus-5-5", "claude-opus-5-5"),
        (["--reflection-model", "gemini-reviewer"], DEFAULT_MODEL, "gemini-reviewer"),
    ],
)  # fmt: skip
def test_the_fast_model_follows_the_writer_unless_a_model_flag_says_otherwise(
    flags, fast, reflection, materials, template, tmp_path, recorder
):
    result = runner.invoke(
        cli.app,
        ["write", "--raw-materials-dir", str(materials), "--template", "demo2026",
         "--output-dir", str(tmp_path / "out"), *flags],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    models = recorder.cfg.models
    assert (models.fast, models.reflection) == (fast, reflection)


def test_write_defaults(materials, template, tmp_path, recorder, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        cli.app,
        ["write", "--raw-materials-dir", str(materials), "--latex-template-dir", "demo2026"],
    )

    assert result.exit_code == 0, result.output
    cfg = recorder.cfg
    assert cfg.base_dir.name.startswith(f"paper_output_{datetime.now():%Y}")
    assert (tmp_path / cfg.base_dir / "raw_materials" / "idea_sparse.md").is_file()
    models = cfg.models
    assert {models.writer, models.reflection, models.plotting, models.fast} == {DEFAULT_MODEL}
    assert models.image is None and models.effort == "high"
    assert cfg.research_cutoff == datetime.now().strftime("%Y-%m")
    assert cfg.use_plotting is False and cfg.resume is True and cfg.compute == "local"
    assert cfg.keep_provided_figures is False
    assert cfg.memory.enabled and cfg.memory.lab and cfg.memory.dream == "stages"
    assert cfg.memory.project_dir is None and cfg.memory.lab_dir is None


def test_write_reports_a_pipeline_failure_with_exit_code_1(
    materials, template, tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "run_paper", Recorder(success=False))
    out = tmp_path / "out"
    result = runner.invoke(
        cli.app,
        ["write", "--raw-materials-dir", str(materials), "--template", "demo2026",
         "--output-dir", str(out)],
    )  # fmt: skip

    assert result.exit_code == 1
    assert (
        "Error: the paper was not written: stage 'outline' failed after 2 attempt(s)"
        in result.stderr
    )
    assert str(out / "state.json") in result.stderr
    assert "Paper written" not in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--raw-materials-dir", "{tmp}/nowhere", "--template", "demo2026"],
         "Raw materials directory not found"),
        (["--raw-materials-dir", "{raw}", "--template", "no-such-venue"],
         "LaTeX template 'no-such-venue' not found"),
        (["--raw-materials-dir", "{raw}", "--template", "demo2026", "--research-cutoff", "soon"],
         "research cutoff must look like YYYY-MM"),
        (["--raw-materials-dir", "{raw}", "--template", "demo2026", "--idea-filename", "x.md"],
         "The idea file was not found"),
        (["--template", "demo2026"], "Missing option '--raw-materials-dir'"),
        (["--raw-materials-dir", "{raw}", "--template", "demo2026", "--compute", "cloud"],
         "Invalid value for '--compute'"),
        (["--raw-materials-dir", "{raw}", "--template", "demo2026", "--use_plotting=maybe"],
         "expected true or false"),
        (["--raw-materials-dir", "{raw}", "--template", "demo2026", "--dream", "always"],
         "Invalid value for '--dream'"),
    ],
)  # fmt: skip
def test_write_usage_and_validation_errors_exit_with_2(
    args, message, materials, template, tmp_path, recorder
):
    args = [a.format(tmp=tmp_path, raw=materials) for a in args]
    result = runner.invoke(cli.app, ["write", "--output-dir", str(tmp_path / "out"), *args])

    assert result.exit_code == 2, result.output
    assert message in " ".join(result.output.split())
    assert recorder.configs == []


def test_write_maps_an_unavailable_compute_backend_to_exit_code_2(
    materials, template, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cli, "run_paper", Recorder(raises=ComputeError("The Modal SDK is not installed."))
    )
    result = runner.invoke(
        cli.app,
        ["write", "--raw-materials-dir", str(materials), "--template", "demo2026",
         "--output-dir", str(tmp_path / "out"), "--compute", "modal"],
    )  # fmt: skip

    assert result.exit_code == 2
    assert "Error: The Modal SDK is not installed." in result.stderr


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--plotting", True),
        ("--use-plotting", True),
        ("--no-plotting", False),
        ("--no-use-plotting", False),
    ],
)
def test_plotting_can_be_switched_with_either_spelling(
    flag, expected, materials, template, tmp_path, recorder
):
    result = runner.invoke(
        cli.app,
        ["write", "--raw-materials-dir", str(materials), "--template", "demo2026",
         "--output-dir", str(tmp_path / "out"), flag],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert recorder.cfg.use_plotting is expected


def test_help_shows_the_new_flags_only():
    result = runner.invoke(cli.app, ["write", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    for flag in ("--raw-materials-dir", "--template", "--lab-memory-dir", "--dream", "--resume",
                 "--keep-provided-figures"):  # fmt: skip
        assert flag in result.output
    assert f"[default: ({DEFAULT_MODEL})]" in result.output
    assert "_" not in "".join(w for w in result.output.split() if w.startswith("--"))


def test_the_top_of_the_help_says_what_the_tool_is_now():
    result = runner.invoke(cli.app, ["--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    said = " ".join(result.output.split())
    assert (
        "From an idea to a research wiki: agents propose sources, hypotheses, experiments "
        "and findings, you decide, and a paper can be written from it."
    ) in said
    assert "Turn an idea and an experimental log into a LaTeX paper." not in said
    # The paper pipeline is still a command of its own, and says what it does.
    assert "write" in said and "research" in said and "ui" in said


@pytest.mark.parametrize("command", [["ui"], ["research", "approve"], ["research", "advance"]])
def test_a_long_description_is_wrapped_once_whatever_the_width(command):
    """A description is written in paragraphs; where its lines break in the source is not
    where they break on the screen."""
    narrow = runner.invoke(cli.app, [*command, "--help"], env={"COLUMNS": "80"})
    wide = runner.invoke(cli.app, [*command, "--help"], env={"COLUMNS": "200"})

    assert narrow.exit_code == wide.exit_code == 0
    for output in (narrow.output, wide.output):
        described = output.split("Usage:", 1)[1].split("╭", 1)[0].splitlines()[1:]
        paragraphs = "\n".join(line.strip() for line in described).strip().split("\n\n")
        assert len(paragraphs) >= 2
        for paragraph in paragraphs:
            lines = paragraph.splitlines()
            # Every line but the last of a paragraph is filled: none ends after a word
            # or two because the source broke there.
            width = max(len(line) for line in lines)
            assert all(len(line) > width - 30 for line in lines[:-1]), paragraph
    # The same words either way.
    assert " ".join(narrow.output.split("╭", 1)[0].split()) == " ".join(
        wide.output.split("╭", 1)[0].split()
    )


def test_verbose_and_version(recorder):
    assert runner.invoke(cli.app, ["--version"]).output == f"paper-orchestra {__version__}\n"

    logger = logging.getLogger("paper_orchestra")
    logger.setLevel(logging.INFO)
    runner.invoke(cli.app, ["-v", "memory", "show", "--memory-dir", "/nonexistent"])
    assert logger.level == logging.DEBUG


# -- batch -------------------------------------------------------------------------


def test_batch_passes_its_options_on_and_reports_failures(tmp_path, template, monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_run_batch(cfg: PipelineConfig, **kwargs: Any) -> Path:
        calls.append({"cfg": cfg, **kwargs})
        log = Path(kwargs["output_dir"]) / "paper_writing_log.json"
        write_json(log, outcome)
        return log

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    for paper_id in ("a", "b", "c"):
        (tmp_path / "papers" / paper_id / "raw_materials").mkdir(parents=True)
    args = [
        "batch",
        "--root-input-folder", str(tmp_path / "papers"),
        "--output-dir", str(tmp_path / "out"),
        "--template", "demo2026",
        "--n-papers", "3",
        "--max-workers", "2",
        "--plotting",
        "--keep-provided-figures",
        "--writer-model", "claude-opus-5-5",
    ]  # fmt: skip

    outcome = {"a": {"status": "success", "paper_path": "a.pdf"}}
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "1 written, 0 failed, 0 in error" in result.output
    call = calls[0]
    assert call["root_input_folder"] == tmp_path / "papers" and call["paper_info_path"] is None
    assert call["output_dir"] == tmp_path / "out"
    assert (call["n_papers"], call["max_workers"]) == (3, 2)
    assert call["cfg"].template_dir == template
    assert call["cfg"].use_plotting and call["cfg"].keep_provided_figures
    models = call["cfg"].models
    # Without --model the fast role follows the writer; the others keep the default.
    assert (models.writer, models.fast) == ("claude-opus-5-5", "claude-opus-5-5")
    assert (models.reflection, models.plotting) == (DEFAULT_MODEL, DEFAULT_MODEL)

    outcome = {
        "a": {"status": "success", "paper_path": "a.pdf"},
        "b": {"status": "failed", "paper_path": "", "error": "stage failed"},
        "c": {"status": "error", "paper_path": "", "error": "worker died"},
    }
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "1 written, 1 failed, 1 in error" in result.output


def test_batch_summary_and_exit_status_cover_this_batch_only(tmp_path, template, monkeypatch):
    # The log of the output directory still holds a paper of an earlier, larger batch
    # that failed.  It is not part of this batch, so it is neither counted nor an error.
    def fake_run_batch(cfg: PipelineConfig, **kwargs: Any) -> Path:
        log = Path(kwargs["output_dir"]) / "paper_writing_log.json"
        write_json(
            log,
            {
                "a": {"status": "success", "paper_path": "a.pdf"},
                "b": {"status": "success", "paper_path": "b.pdf"},
                "z": {"status": "failed", "paper_path": "", "error": "an earlier batch"},
            },
        )
        return log

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    for paper_id in ("a", "b", "z"):
        (tmp_path / "papers" / paper_id / "raw_materials").mkdir(parents=True)
    result = runner.invoke(
        cli.app,
        ["batch", "--root-input-folder", str(tmp_path / "papers"), "--n-papers", "2",
         "--output-dir", str(tmp_path / "out"), "--template", "demo2026"],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "2 written, 0 failed, 0 in error" in result.output


def test_batch_validation_errors_exit_with_2(tmp_path, template):
    result = runner.invoke(
        cli.app, ["batch", "--output-dir", str(tmp_path / "out"), "--template", "demo2026"]
    )
    assert result.exit_code == 2
    assert "exactly one of root_input_folder and paper_info_path" in result.stderr

    result = runner.invoke(
        cli.app,
        ["batch", "--output-dir", str(tmp_path / "out"), "--template", "demo2026",
         "--root-input-folder", str(tmp_path / "missing")],
    )  # fmt: skip
    assert result.exit_code == 2
    assert "Batch input folder not found" in result.stderr


# -- doctor ------------------------------------------------------------------------


class DescribedClient(FakeLLM):
    provider = "anthropic"

    def __init__(self, description: dict[str, Any]) -> None:
        super().__init__(lambda request: "ok")
        self._description = description

    def describe(self) -> dict[str, Any]:
        return dict(self._description)


def test_doctor_reports_the_credential_source_and_warnings(monkeypatch, tmp_path):
    secret = "sk-ant-api03-very-secret-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-key-that-must-not-be-shown")
    monkeypatch.setenv("PAPER_ORCHESTRA_MEMORY_DIR", str(tmp_path / "lab"))
    register_client(
        "anthropic",
        DescribedClient(
            {
                "provider": "anthropic",
                "credential_source": "api_key",
                "base_url": "https://api.anthropic.com",
                # A client that leaks its key into the description must not get it printed.
                "debug": f"key={secret}",
                "warnings": [
                    "ANTHROPIC_API_KEY is set and takes precedence over the Workload Identity "
                    "Federation variables"
                ],
            }
        ),
    )

    result = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "240"})
    assert result.exit_code == 0, result.output
    assert f"anthropic ({DEFAULT_MODEL})" in result.output
    assert "credentials: api_key" in result.output
    assert "Warning: ANTHROPIC_API_KEY is set and takes precedence" in result.output
    assert "SEMANTIC_SCHOLAR_API_KEY is set" in result.output
    assert str(tmp_path / "lab") in result.output
    assert "pdflatex" in result.output or "compute: local" in result.output
    assert secret not in result.output and "s2-key" not in result.output
    assert "key=[redacted]" in result.output

    result = runner.invoke(cli.app, ["doctor", "--json"])
    report = json.loads(result.output)
    assert result.exit_code == 0 and report["ok"] is True
    assert report["provider"]["credential_source"] == "api_key"
    assert report["provider"]["details"] == {
        "base_url": "https://api.anthropic.com",
        "debug": "key=[redacted]",
    }
    assert report["semantic_scholar_api_key"] is True
    assert report["memory"]["lab_dir_source"] == "PAPER_ORCHESTRA_MEMORY_DIR"
    assert report["memory"]["lab_exists"] is False
    assert set(report["compute"]) == {"local", "modal"}
    assert report["packages"]["anthropic"]
    assert secret not in result.output and "s2-key" not in result.output
    # Looking at the lab memory must not create it.
    assert not (tmp_path / "lab").exists()


def test_doctor_exits_with_1_when_the_default_provider_is_unusable(monkeypatch):
    register_client("anthropic", DescribedClient({"credential_source": "none", "warnings": []}))
    result = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "240"})
    assert result.exit_code == 1
    assert "no usable credentials" in result.output
    assert "No credentials are configured for this provider." in result.output
    # Whoever runs this first, as the README has it, is told what works all the same.
    said = " ".join(result.output.split())
    assert (
        "The demonstration and the app work without credentials: `paper-orchestra research "
        "demo` builds a project to look at and `paper-orchestra ui` opens it."
    ) in said
    report = json.loads(runner.invoke(cli.app, ["doctor", "--json"]).output)
    assert report["ok"] is False
    assert report["without_credentials"].startswith("The demonstration and the app work")


def test_doctor_does_not_talk_about_missing_credentials_when_there_are_some(monkeypatch):
    register_client("anthropic", DescribedClient({"credential_source": "api_key"}))
    result = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    assert "work without credentials" not in result.output

    # A provider that cannot even be constructed is a row of the report, not a traceback.
    reset_clients()
    result = runner.invoke(cli.app, ["doctor", "--model", "fake-model", "--json"])
    report = json.loads(result.output)
    assert result.exit_code == 1 and report["ok"] is False
    assert report["provider"]["provider"] == "fake"
    assert "No fake client is registered" in report["provider"]["error"]

    result = runner.invoke(cli.app, ["doctor", "--model", "mystery-model", "--json"])
    report = json.loads(result.output)
    assert result.exit_code == 1
    assert report["provider"]["provider"] is None
    assert "Cannot tell which provider serves model 'mystery-model'" in report["provider"]["error"]


def test_doctor_reports_backends_that_fail_to_load_as_rows(monkeypatch):
    import paper_orchestra.compute as compute

    def broken(name: str, **options: Any):
        if name == "modal":
            raise ComputeError("The Modal compute backend is not available. uv sync --extra modal")
        raise RuntimeError("no subprocess support here")

    monkeypatch.setattr(compute, "get_backend", broken)
    register_client("anthropic", DescribedClient({"credential_source": "workload_identity"}))

    report = doctor_module.collect()
    assert report["ok"] is True
    assert report["compute"]["modal"] == {
        "backend": "modal",
        "available": False,
        "error": "The Modal compute backend is not available. uv sync --extra modal",
    }
    assert report["compute"]["local"]["error"] == "RuntimeError: no subprocess support here"

    result = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    assert "uv sync --extra modal" in result.output
    assert "credentials: workload_identity" in result.output
    assert "Traceback" not in result.output


def test_doctor_reports_what_the_compute_backends_say_is_missing(monkeypatch):
    import paper_orchestra.compute as compute

    problems: dict[str, list[str]] = {
        "local": ["pdflatex is too old to compile the paper."],
        "modal": ["Modal is not authenticated on this machine."],
    }
    closed: list[str] = []

    class Backend(ComputeBackend):
        def __init__(self, name: str) -> None:
            self.name = name

        def run_python(self, code: str, **kwargs: Any):
            raise NotImplementedError

        def compile_latex(self, workdir: Any, tex_name: str, **kwargs: Any):
            raise NotImplementedError

        def check(self, *, python: bool = True, latex: bool = True) -> list[str]:
            return list(problems[self.name])

        def describe(self) -> dict[str, Any]:
            # Nothing in the description hints at a problem: only the backend's own
            # readiness check, the one a run is refused on, knows.
            return {
                "backend": self.name,
                "pdflatex": "/usr/bin/pdflatex",
                "bibtex": "/usr/bin/bibtex",
                "token_configured": True,
            }

        def close(self) -> None:
            closed.append(self.name)

    monkeypatch.setattr(compute, "get_backend", lambda name, **options: Backend(name))
    register_client("anthropic", DescribedClient({"credential_source": "api_key"}))

    report = doctor_module.collect()
    assert report["ok"] is True
    for name in ("local", "modal"):
        assert report["compute"][name]["available"] is False
        assert report["compute"][name]["problems"] == problems[name]
        assert report["compute"][name]["error"] is None
    assert closed == ["local", "modal"]
    result = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "240"})
    assert result.exit_code == 0, result.output
    assert "pdflatex is too old to compile the paper." in result.output
    assert "Modal is not authenticated on this machine." in result.output

    problems = {"local": [], "modal": []}
    report = doctor_module.collect()
    assert report["compute"]["local"]["available"] and report["compute"]["modal"]["available"]
    assert report["compute"]["modal"]["problems"] == []
    result = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "240"})
    assert "pdflatex: /usr/bin/pdflatex" in result.output and "token configured" in result.output


def test_doctor_hints_at_optional_providers_from_the_environment(monkeypatch):
    register_client("anthropic", DescribedClient({"credential_source": "api_key"}))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")
    monkeypatch.setenv("VERTEX_AI_PROJECT", "my-project")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "us-central1")

    others = {row["provider"]: row for row in doctor_module.collect()["other_providers"]}
    assert set(others) == {"gemini", "openai"}
    assert others["openai"]["credential_source"] == "api_key"
    assert others["gemini"]["credential_source"] == "vertex_ai"


# -- memory ------------------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    base = tmp_path / "run"
    wiki = Wiki(base / "memory", name="project")
    wiki.write(
        "wiki/literature/Smith2020.md",
        "> Sparse attention scales linearly.\n\nUsed by [[paper/outline]].",
        {"title": "Deep Things", "summary": "Sparse attention paper (2020)", "kind": "literature"},
    )
    wiki.write(
        "wiki/paper/outline.md",
        "The outline cites [[literature/Smith2020]].",
        {"summary": "Outline of the paper", "kind": "outline"},
    )
    wiki.rebuild_index()
    return base


def test_memory_show_prints_the_index_or_a_page(run_dir):
    result = runner.invoke(cli.app, ["memory", "show", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert result.output == (run_dir / "memory" / "MEMORY.md").read_text()
    assert "[[literature/Smith2020]]: Sparse attention paper (2020)" in result.output

    memory_dir = str(run_dir / "memory")
    result = runner.invoke(
        cli.app, ["memory", "show", "literature/Smith2020", "--memory-dir", memory_dir]
    )
    assert result.exit_code == 0
    assert "title: Deep Things" in result.output
    assert "> Sparse attention scales linearly." in result.output

    result = runner.invoke(
        cli.app, ["memory", "show", "literature/Nobody", "--memory-dir", memory_dir]
    )
    assert result.exit_code == 2
    assert "There is no page 'literature/Nobody'" in result.stderr


def test_memory_commands_need_exactly_one_existing_location(run_dir, tmp_path):
    for command in ("show", "lint", "dream"):
        result = runner.invoke(cli.app, ["memory", command])
        assert result.exit_code == 2
        assert "exactly one of --memory-dir and --run-dir" in result.stderr

    result = runner.invoke(
        cli.app,
        [
            "memory",
            "search",
            "x",
            "--run-dir",
            str(run_dir),
            "--memory-dir",
            str(run_dir / "memory"),
        ],
    )
    assert result.exit_code == 2

    missing = tmp_path / "elsewhere"
    result = runner.invoke(cli.app, ["memory", "lint", "--run-dir", str(missing)])
    assert result.exit_code == 2
    assert f"No memory found at {missing / 'memory'}" in " ".join(result.stderr.split())
    # Looking for a memory must not create one.
    assert not missing.exists()


def test_memory_viewers_leave_the_memory_directory_as_it_is(tmp_path):
    memory = tmp_path / "copied" / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("# Memory index: project\n\nNothing yet.\n")

    for command in (["show"], ["search", "anything"], ["lint"]):
        result = runner.invoke(cli.app, ["memory", *command, "--memory-dir", str(memory)])
        assert result.exit_code == 0, result.output
    assert (
        "Nothing yet."
        in runner.invoke(cli.app, ["memory", "show", "--memory-dir", str(memory)]).output
    )
    # Opening a wiki normally creates its folders; looking at one must not.
    assert sorted(p.name for p in memory.iterdir()) == ["MEMORY.md"]


def test_memory_search(run_dir):
    result = runner.invoke(
        cli.app, ["memory", "search", "sparse attention", "--run-dir", str(run_dir)]
    )
    assert result.exit_code == 0
    first = result.output.splitlines()[0]
    assert first.startswith("wiki/literature/Smith2020.md") and first.endswith("Deep Things")
    assert "Sparse attention scales linearly." in result.output

    result = runner.invoke(
        cli.app,
        ["memory", "search", "sparse attention", "--run-dir", str(run_dir), "--section", "paper"],
    )
    assert "Smith2020.md" not in result.output

    result = runner.invoke(cli.app, ["memory", "search", "zebra", "--run-dir", str(run_dir)])
    assert result.exit_code == 0 and result.output == "No matches.\n"


def test_memory_lint_exit_code_follows_the_report(run_dir):
    result = runner.invoke(cli.app, ["memory", "lint", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "healthy" in result.output

    wiki = Wiki(run_dir / "memory")
    wiki.write("wiki/lessons/latex.md", "See [[lessons/gone]].", {"summary": "LaTeX pitfalls"})
    wiki.add_note("cleveref must load last", stage="section_writing", kind="lesson")
    result = runner.invoke(cli.app, ["memory", "lint", "--run-dir", str(run_dir)])
    assert result.exit_code == 1
    assert "needs attention" in result.output
    assert "broken link: wiki/lessons/latex.md -> [[lessons/gone]]" in result.output
    assert "1 note(s) waiting to be consolidated" in result.output

    result = runner.invoke(cli.app, ["memory", "lint", "--run-dir", str(run_dir), "--json"])
    report = json.loads(result.output)
    assert result.exit_code == 1 and report["ok"] is False
    assert report["broken_links"] == [["wiki/lessons/latex.md", "lessons/gone"]]


def test_memory_dream_runs_a_forced_consolidation(run_dir):
    wiki = Wiki(run_dir / "memory")
    wiki.add_note("cleveref must load after hyperref", stage="section_writing", kind="lesson")
    seen: list[FakeRequest] = []

    def responder(request: FakeRequest) -> str:
        seen.append(request)
        request.memory.create(
            "/memories/wiki/lessons/latex.md",
            "---\nsummary: LaTeX pitfalls\nkind: lesson\n---\nLoad cleveref after hyperref.\n",
        )
        return "Folded one note into lessons/latex."

    register_client("fake", FakeLLM(responder))
    result = runner.invoke(
        cli.app,
        ["memory", "dream", "--run-dir", str(run_dir), "--model", "fake-dreamer",
         "--focus", "Write down the LaTeX lessons."],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "Notes: 1 seen, 1 consolidated. Pages changed: 1." in result.output
    assert "wiki/lessons/latex.md" in result.output
    assert "Folded one note into lessons/latex." in result.output
    assert seen[0].model == "fake-dreamer"
    assert "Write down the LaTeX lessons." in seen[0].text
    assert wiki.pending_notes() == []
    assert "[[lessons/latex]]" in (run_dir / "memory" / "MEMORY.md").read_text()


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    "state",
    [
        None,
        "{ not json",
        "[]",
        '{"status": "running"}',
        '{"config": {"models": {"reflection": ""}}}',
    ],
)
def test_memory_dream_without_credentials_fails_cleanly_before_it_touches_the_memory(
    state, run_dir
):
    Wiki(run_dir / "memory").add_note("something to consolidate", stage="outline")
    if state is not None:
        (run_dir / "state.json").write_text(state)
    before = _files(run_dir)

    # The tests have no credentials, and nothing here names another model than the default.
    result = runner.invoke(cli.app, ["memory", "dream", "--run-dir", str(run_dir)])

    assert result.exit_code == 1
    error = " ".join(result.stderr.split())
    assert error.startswith(
        f"Error: The consolidation was not started: model '{DEFAULT_MODEL}' cannot be used: "
        "No Anthropic credentials were found."
    )
    assert error.endswith(f"(details: paper-orchestra doctor --model {DEFAULT_MODEL})")
    # One line, as for every other command; `doctor` has the setup instructions.
    assert len(result.stderr.strip().splitlines()) == 1 and "Traceback" not in result.output
    # No failed session was added to the memory's records, and the note is still waiting.
    assert _files(run_dir) == before


def test_memory_dream_defaults_to_the_reflection_model_of_the_run(run_dir):
    wiki = Wiki(run_dir / "memory")
    wiki.add_note("cleveref must load after hyperref", stage="section_writing", kind="lesson")
    write_json(
        run_dir / "state.json",
        {
            "status": "succeeded",
            "config": {"models": {"writer": "fake-writer", "reflection": "fake-reviewer"}},
        },
    )
    models: list[str] = []

    def responder(request: FakeRequest) -> str:
        models.append(request.model)
        return "Nothing to fold in."

    register_client("fake", FakeLLM(responder))

    # No Anthropic credentials exist here: the run's own model is what makes this work.
    result = runner.invoke(cli.app, ["memory", "dream", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert set(models) == {"fake-reviewer"}

    # A memory directory leads to the run it belongs to; --focus asks for a session
    # although no note is pending any more.
    del models[:]
    memory_dir = str(run_dir / "memory")
    result = runner.invoke(
        cli.app, ["memory", "dream", "--memory-dir", memory_dir, "--focus", "Check the links."]
    )
    assert result.exit_code == 0, result.output
    assert set(models) == {"fake-reviewer"}

    del models[:]
    result = runner.invoke(
        cli.app,
        ["memory", "dream", "--run-dir", str(run_dir), "--model", "fake-dreamer",
         "--focus", "Check the links."],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert set(models) == {"fake-dreamer"}


def test_memory_dream_reports_failures_and_refuses_the_lab_memory(run_dir, tmp_path):
    Wiki(run_dir / "memory").add_note("something to consolidate", stage="outline")
    # No client is registered for the model: the consolidation is not even started.
    result = runner.invoke(
        cli.app, ["memory", "dream", "--run-dir", str(run_dir), "--model", "fake-dreamer"]
    )
    assert result.exit_code == 1
    assert "The consolidation was not started: model 'fake-dreamer' cannot be used" in " ".join(
        result.stderr.split()
    )
    assert "Traceback" not in result.output

    # A session that fails once it is under way is reported after the fact.
    def overloaded(request: FakeRequest) -> str:
        raise LLMError("the model is overloaded")

    register_client("fake", FakeLLM(overloaded))
    result = runner.invoke(
        cli.app, ["memory", "dream", "--run-dir", str(run_dir), "--model", "fake-dreamer"]
    )
    assert result.exit_code == 1
    assert "The consolidation did not complete: LLMError: the model is overloaded" in " ".join(
        result.stderr.split()
    )
    assert "Traceback" not in result.output

    lab = Wiki(tmp_path / "lab", name="lab")
    result = runner.invoke(cli.app, ["memory", "dream", "--memory-dir", str(lab.root)])
    assert result.exit_code == 2
    assert "is the lab memory" in " ".join(result.stderr.split())


# -- modal -------------------------------------------------------------------------


def test_modal_run_passes_the_remaining_arguments_to_modal(monkeypatch):
    commands: list[list[str]] = []

    def fake_call(command: list[str]) -> int:
        commands.append(command)
        return 3

    monkeypatch.setattr(subprocess, "call", fake_call)
    # Only the command line handed to Modal is looked at, so the SDK need not be installed.
    real = cli.importlib.util.find_spec
    monkeypatch.setattr(
        cli.importlib.util,
        "find_spec",
        lambda name, *args: object() if name == "modal" else real(name, *args),
    )
    result = runner.invoke(
        cli.app,
        ["modal", "run", "--detach", "--raw-materials-dir", "raw", "--template", "cvpr2025",
         "--plotting", "--model", "claude-opus-5-5"],
    )  # fmt: skip

    assert result.exit_code == 3
    assert commands == [
        [
            sys.executable, "-m", "modal", "run", "--detach",
            "-m", "paper_orchestra.modal_app::main",
            "--raw-materials-dir", "raw", "--template", "cvpr2025",
            "--plotting", "--model", "claude-opus-5-5",
        ]
    ]  # fmt: skip


def test_modal_deploy_and_print_command(monkeypatch):
    def unexpected(command: list[str]) -> int:
        raise AssertionError("nothing may be executed with --print-command")

    monkeypatch.setattr(subprocess, "call", unexpected)
    result = runner.invoke(cli.app, ["modal", "deploy", "--print-command", "--name", "papers"])
    assert result.exit_code == 0
    assert result.output.strip().endswith(
        "-m modal deploy -m paper_orchestra.modal_app --name papers"
    )

    result = runner.invoke(cli.app, ["modal", "run", "--print-command", "--run-id", "r1"])
    assert result.output.strip().endswith(
        "-m modal run -m paper_orchestra.modal_app::main --run-id r1"
    )


def test_modal_batch_submits_the_batch_to_the_deployed_app(tmp_path, template, monkeypatch):
    modal = pytest.importorskip("modal")
    from paper_orchestra import modal_app

    calls: list[dict[str, Any]] = []
    outcome: dict[str, Any] | Exception = {"a": {"status": "success", "paper_path": "a.pdf"}}

    def fake_run(options: dict[str, Any], **kwargs: Any) -> Path:
        calls.append({"options": options, **kwargs})
        if isinstance(outcome, Exception):
            raise outcome
        log = Path(kwargs["output_dir"]) / "paper_writing_log.json"
        write_json(log, outcome)
        return log

    real_run_batch_on_modal = cli.run_batch_on_modal
    monkeypatch.setattr(cli, "run_batch_on_modal", fake_run)
    # The deployed function and the way to find a call again are only handed on here.
    write_paper = object()
    looked_up: list[tuple[str, str]] = []

    def from_name(app_name: str, name: str) -> Any:
        looked_up.append((app_name, name))
        return write_paper

    def from_id(call_id: str) -> Any:
        raise AssertionError("no call is looked up by the command itself")

    monkeypatch.setattr(modal.Function, "from_name", from_name)
    monkeypatch.setattr(modal.FunctionCall, "from_id", from_id)
    (tmp_path / "papers" / "a" / "raw_materials").mkdir(parents=True)
    args = [
        "modal", "batch",
        "--root-input-folder", str(tmp_path / "papers"),
        "--output-dir", str(tmp_path / "out"),
        "--template", "demo2026",
        "--plotting",
        "--model", "claude-opus-5-5",
        "--fast-model", "claude-haiku-5-5",
        "--effort", "max",
        "--dream", "end",
        "--no-lab-memory",
    ]  # fmt: skip

    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "1 written, 0 failed, 0 in error" in result.output
    call = calls[0]
    assert call["template_dir"] == template and call["root_input_folder"] == tmp_path / "papers"
    assert call["volume"] is modal_app.volume
    # One call per paper, and a way to find a call again after the local side lost it.
    assert call["write_paper"] is write_paper and "write_batch" not in call
    assert looked_up == [(modal_app.SETTINGS.app_name, "write_paper")]
    assert call["lookup_call"] is from_id
    assert call["cancel"] is False
    # The batch client is replaced above; this is the call it would really have received.
    arguments = {name: value for name, value in call.items() if name != "options"}
    inspect.signature(real_run_batch_on_modal).bind(call["options"], **arguments)
    options = call["options"]
    assert options["use_plotting"] is True and options["effort"] == "max"
    assert options["keep_provided_figures"] is False
    assert options["writer_model"] == options["plotting_model"] == "claude-opus-5-5"
    assert options["fast_model"] == "claude-haiku-5-5" and options["image_model"] is None
    assert (options["memory"], options["lab_memory"], options["dream"]) == (True, False, "end")
    # Exactly the options a job is built from.
    job = modal_app.build_job(run_id="a", **options)
    assert job["models"]["reflection"] == "claude-opus-5-5" and job["memory"]["dream"] == "end"

    # Without --model the fast role follows the writer, as it does for write and batch.
    result = runner.invoke(
        cli.app, [*args[:8], "--writer-model", "claude-opus-5-5", "--keep-provided-figures"]
    )
    assert result.exit_code == 0, result.output
    options = calls[-1]["options"]
    assert (options["writer_model"], options["fast_model"]) == ("claude-opus-5-5",) * 2
    assert (options["reflection_model"], options["plotting_model"]) == (DEFAULT_MODEL,) * 2
    assert options["keep_provided_figures"] is True
    assert modal_app.build_job(run_id="a", **options)["keep_provided_figures"] is True

    outcome = {"a": {"status": "cancelled", "paper_path": "", "call_id": "fc-1"}}
    result = runner.invoke(cli.app, [*args, "--cancel"])
    assert result.exit_code == 0, result.output
    assert calls[-1]["cancel"] is True and calls[-1]["options"] == calls[0]["options"]
    assert "0 written, 0 failed, 0 in error, 1 cancelled" in result.output

    outcome = {"a": {"status": "failed", "paper_path": "", "error": "stage failed"}}
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "0 written, 1 failed, 0 in error" in result.output

    outcome = modal.exception.NotFoundError("App 'paper-orchestra' not found.")
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert "App 'paper-orchestra' not found." in result.stderr
    assert "paper-orchestra modal deploy" in " ".join(result.stderr.split())

    # Any other Modal failure is one line, not a traceback.
    outcome = modal.exception.FunctionTimeoutError("write_paper ran out of time")
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "Modal reported FunctionTimeoutError: write_paper ran out of time" in result.stderr


def test_modal_help_says_what_the_commands_take():
    result = runner.invoke(cli.app, ["modal", "run", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    lines = [line.strip() for line in result.output.splitlines()]
    # Every option of the Modal entrypoint is listed as it is laid out, one per line.
    for option, text in cli.MODAL_RUN_OPTIONS:
        assert any(line.startswith(option) and line.endswith(text) for line in lines), option
    assert "--run-id ID" in result.output and "resume" in result.output
    assert "--model MODEL" in result.output
    assert "--detach" in result.output and "--print-command" in result.output

    result = runner.invoke(cli.app, ["modal", "batch", "--help"], env={"COLUMNS": "200"})
    help_text = " ".join(result.output.split())
    assert "run id derived from its paper id and the options" in help_text
    assert "run the command again to re-attach" in help_text
    assert "--cancel" in help_text and "--keep-provided-figures" in help_text


def test_the_modal_run_help_lists_exactly_the_options_of_the_modal_entrypoint():
    pytest.importorskip("modal")
    from paper_orchestra import modal_app

    expected = []
    for name, parameter in inspect.signature(modal_app.main.info.raw_f).parameters.items():
        flag = name.replace("_", "-")
        # Modal gives every switch a --no- form; the help shows the one that changes something.
        expected.append(f"--no-{flag}" if parameter.default is True else f"--{flag}")
    documented = [option.split()[0] for option, _ in cli.MODAL_RUN_OPTIONS]
    assert sorted(documented) == sorted(expected)
    assert {"--run-id", "--model", "--raw-materials-dir", "--template"} <= set(documented)


def test_modal_commands_need_the_modal_sdk(monkeypatch):
    real = cli.importlib.util.find_spec
    monkeypatch.setattr(
        cli.importlib.util, "find_spec", lambda name: None if name == "modal" else real(name)
    )
    monkeypatch.setattr(
        subprocess, "call", lambda command: pytest.fail("modal must not be started")
    )
    result = runner.invoke(cli.app, ["modal", "deploy"])
    assert result.exit_code == 2
    assert "uv sync --extra modal" in result.stderr

    result = runner.invoke(
        cli.app, ["modal", "batch", "--output-dir", "out", "--template", "cvpr2025"]
    )
    assert result.exit_code == 2
    assert "uv sync --extra modal" in result.stderr


# -- entry points ------------------------------------------------------------------


def test_main_loads_the_environment_and_logging_before_dispatch(monkeypatch, capsys):
    order: list[str] = []
    monkeypatch.setattr(cli, "load_env", lambda: order.append("load_env"))
    monkeypatch.setattr(cli, "setup_logging", lambda: order.append("setup_logging"))

    def dispatch(args: list[str] | None = None) -> None:
        order.append(f"dispatch {args}")

    monkeypatch.setattr(cli, "app", dispatch)
    cli.main(["doctor"])
    assert order == ["load_env", "setup_logging", "dispatch ['doctor']"]


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX signals")
def test_main_turns_sigterm_into_the_interrupt_path(monkeypatch):
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    # Left at its default, SIGTERM would end the test process; with this handler in place
    # a command line that does nothing about SIGTERM is a failing test instead.
    unhandled: list[int] = []
    signal.signal(signal.SIGTERM, lambda signum, frame: unhandled.append(signum))

    def command(args: list[str] | None = None) -> None:
        os.kill(os.getpid(), signal.SIGTERM)  # what kill, timeout and job schedulers send
        time.sleep(2)
        raise AssertionError("the command was not interrupted")

    monkeypatch.setattr(cli, "app", command)
    # KeyboardInterrupt is what `write` and `batch` close their state and workers on.
    with pytest.raises(KeyboardInterrupt):
        cli.main(["write"])
    assert unhandled == []


#: The real command line and pipeline, with a model whose first answer never comes.
STUCK_MODEL_DRIVER = """
import sys
import time
from pathlib import Path

from paper_orchestra import cli
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.llm.fake import FakeLLM
from paper_orchestra.llm.router import register_client


def stuck(request):
    Path(sys.argv[1]).write_text("")
    time.sleep(120)
    return "{}"


register_client("fake", FakeLLM(stuck))
# The run is stopped long before anything is compiled, so TeX need not be installed.
LocalBackend.check = lambda self, **kinds: []
cli.main(sys.argv[2:])
"""


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX signals")
def test_sigterm_on_write_closes_the_run_as_interrupted(materials, template, tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(STUCK_MODEL_DRIVER)
    model_called, out = tmp_path / "model-called", tmp_path / "out"
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(model_called), "write",
         "--raw-materials-dir", str(materials), "--template", str(template),
         "--output-dir", str(out), "--model", "fake-model", "--no-lab-memory"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 60
        while not model_called.exists():
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        assert read_json(out / "state.json")["status"] == "running"
        proc.terminate()  # SIGTERM: kill, timeout, a job scheduler
        assert proc.wait(timeout=60) == 130
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # The run was closed the way Ctrl-C closes it, not left looking as if it still ran.
    state = read_json(out / "state.json")
    assert state["status"] == "interrupted"
    assert "interrupted" in {record["status"] for record in state["stages"].values()}
    assert "running" not in {record["status"] for record in state["stages"].values()}


def test_logs_go_to_stderr_so_that_stdout_stays_machine_readable(monkeypatch, capsys):
    class ChattyClient(DescribedClient):
        def describe(self) -> dict[str, Any]:
            logging.getLogger("paper_orchestra.llm").info("Client ready (credential source: x).")
            return super().describe()

    register_client("anthropic", ChattyClient({"credential_source": "api_key"}))
    monkeypatch.setattr(cli, "load_env", lambda: None)
    logger = logging.getLogger("paper_orchestra")
    handlers, propagate = list(logger.handlers), logger.propagate
    try:
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["doctor", "--json"])
    finally:
        logger.handlers[:] = handlers
        logger.propagate = propagate

    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert json.loads(captured.out)["provider"]["credential_source"] == "api_key"
    assert "Client ready (credential source: x)." in captured.err


def test_main_exits_with_the_command_status(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"paper-orchestra {__version__}\n"

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["memory", "show"])
    assert exit_info.value.code == 2


def test_python_dash_m_runs_the_cli(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "paper_orchestra", "--version"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == f"paper-orchestra {__version__}\n"


def test_importing_the_cli_loads_no_agent_and_no_provider_sdk(tmp_path):
    heavy = (
        "paper_orchestra.agents.",
        "paper_orchestra.research",
        "paper_orchestra.web",
        "anthropic",
        "modal",
        "openai",
        "google.genai",
        "fastapi",
        "uvicorn",
    )
    code = (
        "import sys, paper_orchestra.cli\n"
        f"print(sorted(m for m in sys.modules if m.startswith({heavy!r})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
        check=True,
    )
    assert completed.stdout.strip() == "[]"


# -- the repository around the command line ----------------------------------------


def test_ci_tests_the_python_version_of_each_matrix_job_with_all_extras(repo_root):
    workflow = repo_root / ".github" / "workflows" / "ci.yml"
    if not workflow.is_file():
        pytest.skip("not running from a checkout")
    text = workflow.read_text()

    # Without this, `uv run` takes its interpreter from .python-version, whatever the job
    # is called, and rebuilds the environment on it.
    assert "UV_PYTHON: ${{ matrix.python-version }}" in text
    assert "uv sync --all-extras" in text
    commands = [
        line.strip()
        for line in text.splitlines()
        if "uv run " in line and not line.lstrip().startswith("#")
    ]
    # Every step uses the environment as `uv sync --all-extras` left it.  A plain `uv run`
    # may rebuild it, and then without the extras: the tests of the optional providers,
    # Modal and the frontend would be skipped without a word.
    assert commands and all("uv run --no-sync " in command for command in commands)
    for tool in ("ruff check .", "ruff format --check .", "pytest"):
        assert any(command.endswith(f"uv run --no-sync {tool}") for command in commands), tool


def test_the_output_directories_of_the_documented_commands_are_ignored_by_git(repo_root):
    readme, gitignore = repo_root / "README.md", repo_root / ".gitignore"
    if not (readme.is_file() and gitignore.is_file()):
        pytest.skip("not running from a checkout")
    patterns = [
        line.strip().rstrip("/")
        for line in gitignore.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    documented = set(re.findall(r"--output-dir[ =]([A-Za-z0-9_.-]+)", readme.read_text()))

    # Generated papers and copies of the raw materials must not show up in `git status`.
    assert {"paper_output", "batch_output"} <= documented
    for directory in sorted(documented):
        assert any(fnmatch.fnmatch(directory, pattern) for pattern in patterns), directory


#: The console script with a command that is interrupted while a worker thread is still
#: inside a long call, the way a stage's thread pool is during a model call.
BUSY_THREAD_DRIVER = """
import atexit, sys, threading, time
from pathlib import Path
from paper_orchestra import cli

marker = Path(sys.argv[1])
sys.argv = ["paper-orchestra", "write"]

def command(args=None):
    threading.Thread(target=time.sleep, args=(120,)).start()  # not a daemon thread
    atexit.register(lambda: marker.write_text("exit handlers ran"))
    raise SystemExit(130)  # what the command line turns Ctrl-C into

cli.load_env = lambda: None
cli.app = command
cli.main()
"""


def test_an_interrupted_command_does_not_wait_for_calls_in_flight(tmp_path):
    marker = tmp_path / "marker"
    started = time.monotonic()
    finished = subprocess.run(
        [sys.executable, "-c", BUSY_THREAD_DRIVER, str(marker)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == 130, finished.stderr
    assert time.monotonic() - started < 30  # not the two minutes the busy thread would take
    assert marker.read_text() == "exit handlers ran"  # backends are closed, logs flushed


def test_the_command_line_tests_pass_where_output_is_styled_for_a_terminal():
    """On GitHub Actions Typer colours help and error text.  The escape codes cut option
    names and messages in pieces, so tests that look for them failed there and nowhere
    else.  A fresh interpreter is needed: Typer takes the decision when it is imported."""
    env = {**os.environ, "GITHUB_ACTIONS": "true", "CI": "true", "FORCE_COLOR": "1"}
    selected = (
        "test_write_usage_and_validation_errors_exit_with_2 or test_help_shows_the_new_flags_only "
        "or test_modal_help_says_what_the_commands_take or test_doctor"
    )
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider", "-k", selected],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # 0 means the selected tests ran and passed; selecting none would be exit code 5.
    assert finished.returncode == 0, finished.stdout[-4000:]
