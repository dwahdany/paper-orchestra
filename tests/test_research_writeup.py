"""The paper bridge: what the paper pipeline is given from a wiki, and how it is run.

The materials are exported from the demonstration project, which holds every case
that matters: accepted, proposed and rejected pages, a run that succeeded with a
figure, runs that failed, experiments that never ran.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from paper_orchestra import cli
from paper_orchestra.config import PipelineConfig
from paper_orchestra.pipeline import PaperResult
from paper_orchestra.research import Project, ProjectError, experiments, writeup
from paper_orchestra.research.demo import build_demo_project
from paper_orchestra.utils.common import read_json, write_json


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("writeup") / "demo"
    build_demo_project(root)
    return root


@pytest.fixture
def project(built: Path, tmp_path: Path) -> Project:
    """A copy of the demonstration project; the write-up puts files into it."""
    shutil.copytree(built, tmp_path / "project")
    return Project.open(tmp_path / "project")


@pytest.fixture
def exported(project: Project, tmp_path: Path) -> Path:
    return writeup.export_materials(project, tmp_path / "materials")


def text_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    return " ".join(text.split())


def metric_rows(result_body: str, metrics: dict[str, Any]) -> str:
    """The rows of a result record's table that hold the run's metrics, as they are written."""
    rows = [
        line
        for line in result_body.splitlines()
        if line.startswith("|") and any(f" {name} " in line for name in metrics)
    ]
    assert len(rows) == len(metrics)
    return "\n".join(rows)


# -- export ----------------------------------------------------------------------------


def test_export_writes_the_three_things_the_paper_pipeline_reads(project, exported, tmp_path):
    assert exported == tmp_path / "materials"
    assert sorted(p.relative_to(exported).as_posix() for p in exported.rglob("*")) == [
        "experimental_log.md",
        "figures",
        "figures/E1-run1-loss_curve.png",
        "figures/info.json",
        "figures/provided-loss_curves.png",
        "idea.md",
    ]
    # Exporting reads the project; it changes nothing in it.
    assert not project.paper_dir.exists()
    assert not [e for e in project.events() if e["type"].startswith("paper.")]


def test_the_idea_file_is_the_idea_then_accepted_hypotheses_then_what_was_run(project, exported):
    idea = text_of(exported / "idea.md")
    plain = flat(idea)

    assert idea.startswith(f"# {project.title}\n")
    assert "it is small batches that need it, because early second-moment estimates" in plain
    # The accepted hypotheses, each with where it stands.
    for ref, state in (("H1", "supported"), ("H2", "testing"), ("H3", "refuted")):
        page = project.page(ref)
        assert f"### {ref}: {page.title} ({state})" in idea
        assert flat(page.body.split("## Rationale")[0].split("## Statement")[1]) in plain
    # The method, as the plan of the experiment that ran describes it.
    ran = project.page("E1")
    assert f"### E1: {ran.title}" in idea
    assert "Tests: H1, H2, H3." in idea
    assert "logistic regression on 4000 examples with 1000 sparse binary features" in plain
    assert idea.index("## Hypotheses") < idea.index("## Method")
    # Page headings nest under the file's own.
    assert "\n## Statement" not in idea and "#### Statement" in idea


def test_nothing_the_person_did_not_accept_reaches_the_materials(project, exported):
    everything = text_of(exported / "idea.md") + text_of(exported / "experimental_log.md")
    plain = flat(everything)
    rejected, proposed = project.page("H4"), project.page("H5")
    assert (rejected.review, proposed.review) == ("rejected", "proposed")

    for page in (rejected, proposed, project.page("F2"), project.page("E3"), project.page("Q6")):
        assert page.title not in plain, page.path
        assert flat(page.body)[:80] not in plain, page.path
    for word in ("RAdam", "Transformer", "1840", "gives back what it gained"):
        assert word not in everything, word
    # Experiments that did not produce a result are not described as done.
    for ref in ("E2", "E4"):
        assert project.page(ref).title not in plain
    assert "4096" not in everything


def test_the_log_copies_the_metrics_table_from_the_result_record_verbatim(project, exported):
    log = text_of(exported / "experimental_log.md")
    result = project.page("results/E1-run1")
    record = experiments.runs(project, "E1")[0]

    assert metric_rows(result.body, record["metrics"]) in log
    for name, value in record["metrics"].items():
        assert f"| {name} | {value} |" in log
    assert "## E1 run 1: Warmup against no warmup at batch sizes 16 and 256" in log
    # The setup: how the run was made.
    for fact in (
        "- mode: script",
        "- entrypoint: main.py",
        "- packages: numpy, matplotlib",
        "- timeout seconds: 300",
        "- backend: local",
        f"- code digest: {record['code_digest']}",
        "- result record: wiki/results/E1-run1.md",
    ):
        assert fact in log, fact
    assert "E2" not in log.split("## Findings")[0]  # no failed run is reported


def test_the_log_ends_with_the_accepted_findings(project, exported):
    log = text_of(exported / "experimental_log.md")
    finding = project.page("F1")
    findings = log.split("## Findings")[1]

    assert f"### F1: {finding.title}" in findings
    assert "\nSupports: H1, H2. Refutes: H3. Based on: E1 run 1.\n" in findings
    metrics = experiments.runs(project, "E1")[0]["metrics"]
    for name in ("spike_b16_no_warmup", "final_loss_b16_no_warmup", "final_loss_b16_warmup"):
        assert str(metrics[name]) in findings
    assert "\n#### What was observed\n" in findings and "F2" not in findings
    assert "as H1 and H2 predicted" in flat(findings)
    assert "not to be reported" not in findings  # every number of F1 is on record


def test_wiki_links_become_what_they_are_called(exported):
    for name in ("idea.md", "experimental_log.md"):
        assert "[[" not in text_of(exported / name), name
    idea = flat(text_of(exported / "idea.md"))
    assert "(Ma and Yarats, 2021)" in idea  # a label is kept
    # A bare link to a page the materials list becomes its id; to any other, its title.
    assert "Test H1, H2 and H3 in one small run" in idea
    assert "(Q3: Does warmup" not in idea
    assert "(Does warmup under Adam change where training ends, or only how it starts?)" in idea


def test_a_link_to_a_page_that_is_not_accepted_keeps_only_the_name_it_was_written_with(
    project, tmp_path
):
    """An accepted page may point at one that is still proposed or was rejected.  The
    title of that page is its author's text, not the person's: it stays out."""
    waiting, rejected, answered = project.page("Q6"), project.page("H4"), project.page("Q1")
    assert (waiting.review, rejected.review, answered.review) == (
        "proposed",
        "rejected",
        "accepted",
    )
    project.create_page(
        "finding",
        "What to look at next",
        f"Next is [[Q6]], not [[{rejected.link}]]; [[Q1]] is settled.\n",
        author="human",
    )

    log = flat(text_of(writeup.export_materials(project, tmp_path / "out") / "experimental_log.md"))

    written = rejected.link.rsplit("/", 1)[-1]
    assert f"Next is Q6, not {written}; {answered.title} is settled." in log
    assert waiting.title not in log and rejected.title not in log


def test_a_finding_with_unsupported_numbers_carries_a_warning_once_it_is_accepted(
    project, tmp_path
):
    project.review("F2", "accept")

    log = text_of(writeup.export_materials(project, tmp_path / "out") / "experimental_log.md")

    assert "### F2: After the warmup ends" in log
    assert (
        "Numbers in this finding that no result record, source or input file supports, and "
        "that are not to be reported: 1840." in flat(log)
    )


def test_a_page_rejected_later_leaves_the_materials(project, tmp_path):
    project.review("F1", "reject", reason="Three seeds are too few.")
    project.review("H3", "reject", reason="Not what I meant.")

    out = writeup.export_materials(project, tmp_path / "out")

    assert "## Findings" not in text_of(out / "experimental_log.md")
    idea = text_of(out / "idea.md")
    assert "### H3" not in idea and "Tests: H1, H2." in idea


def test_figures_are_the_image_artifacts_of_succeeded_runs_with_captions(project, exported):
    source = project.experiment_dir("E1") / "runs" / "1" / "artifacts" / "loss_curve.png"
    copied = exported / "figures" / "E1-run1-loss_curve.png"
    assert copied.read_bytes() == source.read_bytes()
    assert read_json(exported / "figures" / "info.json") == [
        {
            "name": "E1-run1-loss_curve.png",
            "caption": "Warmup against no warmup at batch sizes 16 and 256 (E1 run 1): loss curve.",
        },
        # The image the person attached to the project is a figure too, and says whose it is.
        {
            "name": "provided-loss_curves.png",
            "caption": "Provided by the researcher: loss curves.",
        },
    ]
    assert "- E1-run1-loss_curve.png" in text_of(exported / "experimental_log.md")


def test_only_real_image_files_inside_the_run_directory_become_figures(project, tmp_path):
    directory = project.experiment_dir("E1") / "runs" / "1"
    (directory / "artifacts" / "table.csv").write_text("a,b\n1,2\n")
    (directory / "artifacts" / "linked.png").symlink_to(project.root / "inputs" / "loss_curves.png")
    record = read_json(directory / "run.json")
    record["artifacts"] += [
        "artifacts/table.csv",
        "artifacts/linked.png",
        "artifacts/missing.png",
        "../../../../inputs/loss_curves.png",
        str(project.root / "inputs" / "loss_curves.png"),
        17,
    ]
    write_json(directory / "run.json", record)

    out = writeup.export_materials(project, tmp_path / "out")

    assert sorted(p.name for p in (out / "figures").iterdir()) == [
        "E1-run1-loss_curve.png",
        "info.json",
        "provided-loss_curves.png",  # attached by the person, not smuggled in by the run
    ]


def test_figures_get_names_that_are_unique_and_safe_in_a_latex_path(project, tmp_path):
    """The experiment's code names its artifacts.  Two of one run may share a name in
    different folders, and a name may hold what LaTeX reads as markup."""
    directory = project.experiment_dir("E1") / "runs" / "1"
    curve = (directory / "artifacts" / "loss_curve.png").read_bytes()
    for name, data in (
        ("small/plot.png", b"small"),
        ("large/plot.png", b"large"),
        ("50% drop & more #1.PNG", curve),
    ):
        (directory / "artifacts" / name).parent.mkdir(exist_ok=True)
        (directory / "artifacts" / name).write_bytes(data)
    record = read_json(directory / "run.json")
    record["artifacts"] += [
        "artifacts/small/plot.png",
        "artifacts/large/plot.png",
        "artifacts/50% drop & more #1.PNG",
        "artifacts/loss_curve.png",  # listed twice: still one figure
    ]
    write_json(directory / "run.json", record)

    figures = writeup.export_materials(project, tmp_path / "out") / "figures"

    listed = read_json(figures / "info.json")
    names = [entry["name"] for entry in listed]
    assert names == [
        "E1-run1-loss_curve.png",
        "E1-run1-small-plot.png",
        "E1-run1-large-plot.png",
        "E1-run1-50-drop-more-1.PNG",
        "provided-loss_curves.png",
    ]
    assert sorted(p.name for p in figures.iterdir()) == sorted([*names, "info.json"])
    assert (figures / "E1-run1-small-plot.png").read_bytes() == b"small"
    assert (figures / "E1-run1-large-plot.png").read_bytes() == b"large"
    # The caption still says what the run called it.
    assert listed[3]["caption"].endswith("(E1 run 1): 50% drop & more #1.")


def test_exporting_again_replaces_what_was_exported_before(project, exported):
    (exported / "figures" / "stale.png").write_bytes(b"old")
    (exported / "notes-of-mine.txt").write_text("kept")
    project.review("F1", "reject", reason="Three seeds are too few.")

    writeup.export_materials(project, exported)

    assert not (exported / "figures" / "stale.png").exists()
    assert "## Findings" not in text_of(exported / "experimental_log.md")
    assert text_of(exported / "notes-of-mine.txt") == "kept"  # only its own files are replaced


def test_a_project_without_runs_exports_its_idea_and_an_empty_log(tmp_path):
    project = Project.create(tmp_path / "fresh", title="A fresh idea", brief="Nothing yet.")

    out = writeup.export_materials(project, tmp_path / "out")

    assert text_of(out / "idea.md") == "# A fresh idea\n\nNothing yet.\n"
    assert "## " not in text_of(out / "experimental_log.md")
    assert not (out / "figures").exists()


def test_only_the_table_under_the_metrics_heading_is_taken_for_the_metrics(project, tmp_path):
    """The record is written by another module; what it calls its metrics is what counts."""
    record = "Run 1 succeeded.\n\n| | |\n|---|---|\n| Status | succeeded |\n\n{metrics}## Log\n"
    table = "| metric | value |\n|---|---|\n| accuracy | 0.9130 |\n| n | 1,024 |"
    with_table = record.format(metrics=f"## Metrics of the run\n\n{table}\n\ntext\n\n")

    assert writeup._metrics_table(with_table) == table
    assert writeup._metrics_table(record.format(metrics="")) == ""

    project.write_record(
        "result", "E1-run1", "E1 run 1", record.format(metrics=""),
        fields={"state": "succeeded"}, relations={"run_of": ["E1"]},
    )  # fmt: skip
    log = text_of(writeup.export_materials(project, tmp_path / "out") / "experimental_log.md")
    assert "_The result record has no metrics table; see wiki/results/E1-run1.md._" in log
    assert "| Status | succeeded |" not in log


# -- write_paper -----------------------------------------------------------------------


class Pipeline:
    """Stands in for ``pipeline.run_paper``: records the configuration it was given."""

    def __init__(self, *, success: bool = True, raises: BaseException | None = None) -> None:
        self.success, self.raises = success, raises
        self.configs: list[PipelineConfig] = []
        self.materials: dict[str, str] = {}

    def __call__(self, cfg: PipelineConfig, **kwargs: Any) -> PaperResult:
        assert kwargs == {}
        self.configs.append(cfg)
        raw = cfg.base_dir / "raw_materials"
        self.materials = {
            p.relative_to(raw).as_posix(): p.name for p in raw.rglob("*") if p.is_file()
        }
        if self.raises is not None:
            raise self.raises
        usage = {"total": {"requests": 7}, "estimated_cost_usd": 1.5}
        if self.success:
            return PaperResult(True, cfg.base_dir / "final_paper.pdf", cfg.base_dir, {}, usage)
        return PaperResult(
            False, None, cfg.base_dir, {}, usage, error="stage 'outline' failed\nand more"
        )


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    stub = Pipeline()
    monkeypatch.setattr(writeup, "run_paper", stub)
    return stub


def paper_events(project: Project) -> list[dict[str, Any]]:
    return [event for event in project.events() if event["type"].startswith("paper.")]


def test_write_paper_runs_the_pipeline_in_the_projects_paper_directory(project, pipeline):
    result = writeup.write_paper(project, template="cvpr2025")

    assert result.success and result.pdf_path == project.paper_dir / "final_paper.pdf"
    (cfg,) = pipeline.configs
    assert cfg.base_dir == project.paper_dir
    assert cfg.template_dir.name == "cvpr2025"
    assert (cfg.idea_filename, cfg.experimental_log_filename) == ("idea.md", "experimental_log.md")
    # The pipeline found the materials in place when it started.
    assert sorted(pipeline.materials) == [
        "experimental_log.md",
        "figures/E1-run1-loss_curve.png",
        "figures/info.json",
        "figures/provided-loss_curves.png",
        "idea.md",
    ]
    # The figures are what the runs produced: they are used, and kept if more are generated.
    assert cfg.use_plotting is False and cfg.keep_provided_figures is True
    assert cfg.uses_provided_figures


def test_write_paper_follows_the_projects_settings_unless_told_otherwise(project, pipeline):
    project.settings.update(
        {
            "models": {
                "research": "research-model",
                "reasoning": "reasoning-model",
                "coding": "coding-model",
                "fast": "fast-model",
                "effort": "medium",
            },
            "research_cutoff": "2025-06",
            "compute": "modal",
            "lab_memory": False,
        }
    )
    project.save_settings()

    writeup.write_paper(project, template="cvpr2025")
    writeup.write_paper(
        project,
        template="iclr2025",
        model="one-model",
        plotting_model="plots-model",
        effort="low",
        research_cutoff="2024-01",
        compute="local",
        lab_memory=True,
        use_plotting=True,
        resume=False,
        dream=None,  # an option that is not given follows the default
    )

    settings, chosen = pipeline.configs
    assert (
        settings.models.writer,
        settings.models.reflection,
        settings.models.plotting,
        settings.models.fast,
        settings.models.effort,
    ) == ("reasoning-model", "reasoning-model", "coding-model", "fast-model", "medium")
    assert (settings.research_cutoff, settings.compute, settings.memory.lab) == (
        "2025-06",
        "modal",
        False,
    )
    assert settings.resume is True and settings.memory.dream == "stages"
    assert (
        chosen.models.writer,
        chosen.models.reflection,
        chosen.models.plotting,
        chosen.models.fast,
        chosen.models.effort,
    ) == ("one-model", "one-model", "plots-model", "one-model", "low")
    assert (chosen.research_cutoff, chosen.compute, chosen.memory.lab) == (
        "2024-01",
        "local",
        True,
    )
    assert chosen.use_plotting is True and chosen.keep_provided_figures is True
    assert chosen.resume is False and chosen.memory.dream == "stages"
    assert chosen.template_dir.name == "iclr2025"


def test_write_paper_tells_the_project_when_it_starts_and_how_it_ended(project, pipeline):
    writeup.write_paper(project, template="cvpr2025")

    started, finished = paper_events(project)
    assert started["type"] == "paper.started"
    assert started["summary"] == (
        "Writing a paper from the wiki with the cvpr2025 template: 1 succeeded run, "
        "1 accepted finding, 3 attached files."
    )
    assert started["refs"] == [
        "paper/raw_materials/idea.md",
        "paper/raw_materials/experimental_log.md",
    ]
    assert started["data"] == {
        "template": "cvpr2025",
        "dir": "paper",
        "results": ["wiki/results/E1-run1.md"],
        "findings": [project.page("F1").path],
        "figures": ["E1-run1-loss_curve.png", "provided-loss_curves.png"],
        "provided": ["notes.md", "prior_run.csv", "loss_curves.png"],
    }
    assert finished["type"] == "paper.finished"
    assert finished["summary"] == "The paper is written: paper/final_paper.pdf."
    assert finished["refs"] == ["paper/final_paper.pdf"]
    assert finished["data"] == {
        "success": True,
        "pdf": "paper/final_paper.pdf",
        "error": None,
        "cost_usd": 1.5,
    }


def test_a_paper_that_was_not_written_is_reported_as_such(project, monkeypatch):
    monkeypatch.setattr(writeup, "run_paper", Pipeline(success=False))

    result = writeup.write_paper(project, template="cvpr2025")

    assert not result.success
    finished = paper_events(project)[-1]
    assert finished["summary"] == "No paper was written: stage 'outline' failed"
    assert finished["refs"] == ["paper/state.json"]
    assert finished["data"]["success"] is False and finished["data"]["pdf"] is None


def test_an_interrupted_write_up_is_on_record_and_the_interrupt_goes_on(project, monkeypatch):
    monkeypatch.setattr(writeup, "run_paper", Pipeline(raises=KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        writeup.write_paper(project, template="cvpr2025")

    finished = paper_events(project)[-1]
    assert finished["type"] == "paper.finished"
    assert "was stopped" in finished["summary"] and "continues it" in finished["summary"]
    assert finished["data"]["interrupted"] is True and finished["data"]["success"] is False


def refused(project: Project, pipeline: Pipeline, message: str, error: type = ProjectError) -> str:
    with pytest.raises(error, match=message) as caught:
        writeup.write_paper(project, template="cvpr2025")
    # A refusal starts nothing and leaves nothing behind.
    assert pipeline.configs == [] and paper_events(project) == []
    assert not project.paper_dir.exists()
    return str(caught.value)


def test_write_paper_refuses_a_project_without_a_succeeded_run(tmp_path, pipeline):
    fresh = Project.create(tmp_path / "fresh", title="A fresh idea", brief="Nothing yet.")
    fresh.create_page("finding", "A finding of my own", "Seen it.\n", author="human")

    message = refused(fresh, pipeline, "no accepted experiment has a run that succeeded")

    assert "no results were attached" in message
    assert "paper-orchestra research approve" in message


def test_write_paper_refuses_a_project_without_an_accepted_finding(project, pipeline):
    project.review("F1", "reject", reason="Three seeds are too few.")

    message = refused(project, pipeline, "no finding is accepted")

    assert "accept a finding" in message
    # The finding that still waits does not count: it is not the person's yet.
    assert project.page("F2").review == "proposed"


def test_a_run_whose_plan_was_rejected_is_not_a_run_to_report(project, tmp_path):
    project.review("E1", "reject", reason="Wrong toy problem.")

    log = text_of(writeup.export_materials(project, tmp_path / "out") / "experimental_log.md")

    assert "E1 run 1" not in log.split("## Findings")[0]
    assert "No experiment of this project has a run that succeeded." in log


def test_results_the_person_brought_are_what_a_paper_reports_when_nothing_was_run(
    tmp_path, pipeline
):
    # Someone who already has results attaches them, writes down what they mean, and
    # wants the paper.  No experiment of the project needs to have run for that.
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "results.csv").write_text("batch,warmup,peak\n16,no,3.41\n16,yes,2.31\n")
    (mine / "notes.md").write_text("# Notes\n\nThree seeds each.\n```\nraw\n```\n")
    (mine / "loss curve.png").write_bytes(b"\x89PNG fake")
    (mine / "kingma2015.pdf").write_bytes(b"%PDF-1.4 someone else's paper")
    project = Project.create(
        tmp_path / "own",
        title="My own results",
        brief="Warmup matters at small batches.",
        files=sorted(mine.iterdir()),
    )
    project.create_page(
        "finding", "Warmup lowers the peak", "The peak falls from 3.41 to 2.31.\n", author="human"
    )

    assert writeup.write_paper(project, template="cvpr2025").success

    raw = project.paper_dir / "raw_materials"
    log = text_of(raw / "experimental_log.md")
    assert "## Material the researcher provided" in log
    assert "### inputs/results.csv" in log and "16,no,3.41\n16,yes,2.31" in log
    # A fence in the file cannot end the block it is quoted in.
    assert "````text\n# Notes" in log and "```\nraw\n```\n````" in log
    assert "%PDF" not in log and "kingma2015" not in log
    assert log.index("## Material the researcher provided") < log.index("## Findings")
    assert "### F1: Warmup lowers the peak" in log
    assert read_json(raw / "figures" / "info.json") == [
        {"name": "provided-loss-curve.png", "caption": "Provided by the researcher: loss curve."}
    ]
    (started, _finished) = paper_events(project)
    assert started["summary"].endswith("0 succeeded runs, 1 accepted finding, 3 attached files.")
    assert started["data"]["provided"] == ["loss curve.png", "notes.md", "results.csv"]


def test_attached_text_is_given_whole_up_to_a_limit_and_says_where_it_was_cut(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(writeup, "MAX_PROVIDED_FILE_CHARS", 40)
    monkeypatch.setattr(writeup, "MAX_PROVIDED_CHARS", 60)
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "a.txt").write_text("a" * 100)
    (mine / "b.txt").write_text("b" * 100)
    (mine / "c.txt").write_text("c" * 100)
    project = Project.create(tmp_path / "own", title="T", brief="b", files=sorted(mine.iterdir()))
    (project.inputs_dir / "link.txt").symlink_to(tmp_path / "mine" / "a.txt")
    manifest = read_json(project.inputs_dir / "manifest.json")
    manifest["files"] += [{"name": "link.txt"}, {"name": "../../mine/a.txt"}, {"name": 7}]
    write_json(project.inputs_dir / "manifest.json", manifest)

    log = text_of(writeup.export_materials(project, tmp_path / "out") / "experimental_log.md")

    assert "a" * 40 in log and "a" * 41 not in log
    assert "_Cut after 40 of 100 characters" in log
    assert "b" * 20 in log and "b" * 21 not in log
    assert "### inputs/c.txt\n\n_Left out for lack of room" in log
    # A manifest is data: only regular files inside inputs/ are read.
    assert "link.txt" not in log and "mine/a.txt" not in log


def test_an_unknown_template_or_option_is_refused_before_anything_is_written(project, pipeline):
    with pytest.raises(FileNotFoundError, match="nosuchvenue"):
        writeup.write_paper(project, template="nosuchvenue")
    with pytest.raises(TypeError, match="colour") as caught:
        writeup.write_paper(project, template="cvpr2025", colour="blue", base_dir=Path("x"))

    assert "base_dir" in str(caught.value) and "writer_model" in str(caught.value)
    assert pipeline.configs == [] and paper_events(project) == []
    assert not project.paper_dir.exists()


def test_a_write_up_is_the_projects_one_worker_while_it_runs(project, monkeypatch):
    """One worker per project (spec 10.4).  The engine and the write-up share the lock:
    while a paper is being written nothing else works on the project, and a second
    write-up cannot run a second pipeline in the same paper directory."""
    from paper_orchestra.research.engine import Engine, WorkerBusy

    refusals: list[Exception] = []

    named: list[dict[str, Any]] = []

    class Busy(Pipeline):
        """A pipeline during which someone tries to start more work on the project."""

        def __call__(self, cfg: PipelineConfig, **kwargs: Any) -> PaperResult:
            if not self.configs:  # only the write-up that got to run tries
                named.append(read_json(project.root / "state.json")["worker"])
                for start in (
                    lambda: writeup.write_paper(project, template="cvpr2025"),
                    lambda: Engine(project).run_experiment("E4"),
                ):
                    with pytest.raises(ProjectError) as caught:
                        start()
                    refusals.append(caught.value)
            return super().__call__(cfg, **kwargs)

    stub = Busy()
    monkeypatch.setattr(writeup, "run_paper", stub)

    assert writeup.write_paper(project, template="cvpr2025").success

    second, engine = refusals
    assert "Another worker is at work on this project" in str(second)
    assert "one worker at a time" in str(second)
    assert isinstance(engine, WorkerBusy)
    assert len(stub.configs) == 1 and len(paper_events(project)) == 2
    assert project.page("E4").state == "approved" and experiments.runs(project, "E4") == []
    # While it ran, the write-up was on file as the project's worker, the way an engine
    # is, so that the web server shows it and starts nothing beside it.
    (worker,) = named
    assert worker["pid"] == os.getpid() and worker["kind"] == "write"
    assert worker["host"] and worker["started"] and worker["heartbeat"]
    after = read_json(project.root / "state.json")
    assert after["worker"] is None
    # Being the worker changes nothing else of the engine's state.
    assert (after["status"], after["round"], after["next_phase"]) == ("checkpoint", 2, "run")
    # The lock goes with the write-up: afterwards the project is free again.
    assert writeup.write_paper(project, template="cvpr2025").success


def test_writing_again_exports_the_wiki_as_it_is_now(project, pipeline):
    writeup.write_paper(project, template="cvpr2025")
    project.review("F2", "accept")

    writeup.write_paper(project, template="cvpr2025")

    log = text_of(project.paper_dir / "raw_materials" / "experimental_log.md")
    assert "### F2:" in log
    assert json.loads(text_of(project.paper_dir / "raw_materials" / "figures" / "info.json"))
    assert len(pipeline.configs) == 2 and len(paper_events(project)) == 4


# -- the command ---------------------------------------------------------------------


def test_the_write_command_writes_the_paper_of_a_project(project, pipeline):
    """`research write` is what the web app starts as a worker: the command line, the
    export and the pipeline's configuration together, with only the pipeline stubbed."""
    result = CliRunner().invoke(
        cli.app,
        ["research", "write", str(project.root), "--template=cvpr2025", "--model=paper-model"],
    )

    assert result.exit_code == 0, result.output
    (cfg,) = pipeline.configs
    assert cfg.base_dir == project.paper_dir
    assert {cfg.models.writer, cfg.models.reflection, cfg.models.plotting, cfg.models.fast} == {
        "paper-model"
    }
    assert f"Paper written: {project.paper_dir / 'final_paper.pdf'}" in result.stdout
    assert f"(details: {project.paper_dir / 'usage.json'})" in result.stdout
    events = [line for line in result.stderr.splitlines() if line.strip()]
    assert events[0].endswith(
        "paper: Writing a paper from the wiki with the cvpr2025 template: 1 succeeded run, "
        "1 accepted finding, 3 attached files."
    )
    assert events[-1].endswith("paper: The paper is written: paper/final_paper.pdf.")


def test_the_write_command_says_what_a_project_still_lacks(project, pipeline):
    project.review("F1", "reject", reason="Three seeds are too few.")

    result = CliRunner().invoke(
        cli.app, ["research", "write", str(project.root), "--template", "cvpr2025"]
    )

    assert result.exit_code == 2
    assert "Error: There is nothing to write a paper from yet: no finding is accepted." in (
        flat(result.stderr)
    )
    assert pipeline.configs == []
