from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest

from paper_orchestra import batch
from paper_orchestra.batch import load_papers, run_batch, summarize
from paper_orchestra.config import MemoryConfig, ModelConfig, PipelineConfig
from paper_orchestra.pipeline import PaperResult
from paper_orchestra.utils.common import read_json, write_json


@pytest.fixture
def template_dir(tmp_path: Path) -> Path:
    template = tmp_path / "template"
    template.mkdir()
    (template / "template.tex").write_text("\\documentclass{article}\n")
    (template / "guidelines.md").write_text("Eight pages.\n")
    return template


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Three papers, one of them without the original PDF, plus a folder that is not a paper."""
    root = tmp_path / "papers"
    for paper_id in ("p1", "p2", "p3"):
        materials = root / paper_id / "raw_materials"
        materials.mkdir(parents=True)
        (materials / "idea_sparse.md").write_text(f"Idea of {paper_id}.\n")
        (materials / "experimental_log.md").write_text(f"Log of {paper_id}.\n")
    (root / "p1" / "p1.pdf").write_bytes(b"%PDF original p1")
    (root / "p2" / "p2.pdf").write_bytes(b"%PDF original p2")
    (root / "notes").mkdir()
    (root / "README.md").write_text("not a paper\n")
    return root


@pytest.fixture
def cfg(tmp_path: Path, template_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        base_dir=tmp_path / "ignored",
        template_dir=template_dir,
        models=ModelConfig(writer="claude-opus-5-5"),
        memory=MemoryConfig(project_dir=tmp_path / "shared-memory", lab=False),
        use_plotting=True,
    )


class Workers:
    """Replaces the per-paper worker; runs in threads so the test can observe it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, Any]) -> None:
        self.outcomes = outcomes
        self.jobs: list[dict[str, Any]] = []
        #: Papers that must already be in the log on disk when a paper's worker runs.
        self.must_see: dict[str, set[str]] = {}
        self.logs_seen: dict[str, dict[str, Any]] = {}
        self.pool_sizes: list[int] = []
        monkeypatch.setattr(batch, "_write_paper", self.write)
        monkeypatch.setattr(batch, "_process_pool", self.pool)

    def pool(self, max_workers: int) -> ThreadPoolExecutor:
        self.pool_sizes.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    def write(self, job: dict[str, Any]) -> dict[str, Any]:
        self.jobs.append(job)
        paper_dir = Path(job["config"]["base_dir"])
        log_path = paper_dir.parent / batch.LOG_NAME
        # Wait (briefly) for the papers this one must find in the log already.
        expected = self.must_see.get(job["paper_id"], set())
        deadline = time.monotonic() + 5
        seen: dict[str, Any] = {}
        while True:
            seen = read_json(log_path) if log_path.is_file() else {}
            if expected <= set(seen) or time.monotonic() > deadline:
                break
            time.sleep(0.01)
        self.logs_seen[job["paper_id"]] = seen
        outcome = self.outcomes[job["paper_id"]]
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "success":
            paper_dir.mkdir(parents=True, exist_ok=True)
            pdf = paper_dir / "final_paper.pdf"
            pdf.write_bytes(b"%PDF")
            return {"paper_id": job["paper_id"], "status": "success", "paper_path": str(pdf)}
        return {
            "paper_id": job["paper_id"],
            "status": "failed",
            "paper_path": "",
            "error": outcome,
        }


# -- which papers ------------------------------------------------------------------


def test_papers_are_found_in_a_folder_in_a_stable_order(root, caplog):
    with caplog.at_level(logging.WARNING, logger="paper_orchestra"):
        papers = load_papers(root_input_folder=root)

    assert papers == [
        {
            "paper_id": "p1",
            "content_folder": str(root / "p1" / "raw_materials"),
            "pdf_path": str(root / "p1" / "p1.pdf"),
        },
        {
            "paper_id": "p2",
            "content_folder": str(root / "p2" / "raw_materials"),
            "pdf_path": str(root / "p2" / "p2.pdf"),
        },
        # A paper without the original PDF is still written.
        {"paper_id": "p3", "content_folder": str(root / "p3" / "raw_materials"), "pdf_path": None},
    ]
    assert "has no raw_materials directory" in caplog.text and "notes" in caplog.text
    assert [p["paper_id"] for p in load_papers(root_input_folder=root, n_papers=2)] == ["p1", "p2"]
    assert load_papers(root_input_folder=root, n_papers=0) == []


def test_papers_from_a_list_resolve_paths_against_the_list(root, tmp_path):
    info = tmp_path / "lists" / "papers.json"
    write_json(
        info,
        [
            {"paper_id": "a", "content_folder": "../papers/p1/raw_materials", "pdf_path": "a.pdf"},
            {"paper_id": "b", "content_folder": str(root / "p2" / "raw_materials")},
        ],
    )

    assert load_papers(paper_info_path=info) == [
        {
            "paper_id": "a",
            "content_folder": str(info.parent / "../papers/p1/raw_materials"),
            "pdf_path": str(info.parent / "a.pdf"),
        },
        {"paper_id": "b", "content_folder": str(root / "p2" / "raw_materials"), "pdf_path": None},
    ]


def test_invalid_batches_are_rejected(root, tmp_path):
    with pytest.raises(ValueError, match="exactly one of"):
        load_papers()
    with pytest.raises(ValueError, match="exactly one of"):
        load_papers(root_input_folder=root, paper_info_path=tmp_path / "x.json")
    with pytest.raises(FileNotFoundError, match="Batch input folder not found"):
        load_papers(root_input_folder=tmp_path / "missing")

    info = tmp_path / "papers.json"
    for entries, message in (
        ({"paper_id": "a"}, "must contain a JSON list"),
        ([{"content_folder": "x"}], "entry 0 has no paper_id"),
        ([{"paper_id": "a"}], "paper a has no content_folder"),
        ([{"paper_id": "a", "content_folder": "x"}] * 2, "appears more than once"),
        ([{"paper_id": "../a", "content_folder": "x"}], "cannot be used as a directory name"),
        ([{"paper_id": "..", "content_folder": "x"}], "cannot be used as a directory name"),
    ):
        write_json(info, entries)
        with pytest.raises(ValueError, match=message):
            load_papers(paper_info_path=info)


# -- running the batch -------------------------------------------------------------


def test_batch_writes_the_log_as_papers_finish(root, cfg, tmp_path, monkeypatch, caplog):
    workers = Workers(
        monkeypatch,
        {
            "p1": "success",
            "p2": "stage 'literature' failed after 2 attempt(s)",
            "p3": BrokenProcessPool("A process in the process pool was terminated abruptly"),
        },
    )
    workers.must_see = {"p2": {"p1"}, "p3": {"p1", "p2"}}
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger="paper_orchestra"):
        log_path = run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=1)

    assert log_path == out / "paper_writing_log.json"
    assert read_json(log_path) == {
        "p1": {"status": "success", "paper_path": str(out / "p1" / "final_paper.pdf")},
        "p2": {
            "status": "failed",
            "paper_path": "",
            "error": "stage 'literature' failed after 2 attempt(s)",
        },
        "p3": {
            "status": "error",
            "paper_path": "",
            "error": "BrokenProcessPool: A process in the process pool was terminated abruptly",
        },
    }
    # The log on disk grew while the batch was still running: each paper found the
    # results of the ones that finished before it.
    assert workers.logs_seen["p2"]["p1"]["status"] == "success"
    assert sorted(workers.logs_seen["p3"]) == ["p1", "p2"]
    assert workers.pool_sizes == [1]
    assert "Batch finished: 3 paper(s), 1 written, 1 failed, 1 in error." in caplog.text
    assert "Paper p3 error (3 of 3): BrokenProcessPool" in caplog.text
    assert summarize(read_json(log_path)) == {"success": 1, "failed": 1, "error": 1}


def test_every_paper_gets_its_own_config(root, cfg, tmp_path, monkeypatch):
    workers = Workers(monkeypatch, {"p1": "success", "p2": "success", "p3": "success"})
    out = tmp_path / "out"
    run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=8)

    assert workers.pool_sizes == [3]
    by_id = {job["paper_id"]: job for job in workers.jobs}
    assert set(by_id) == {"p1", "p2", "p3"}
    for paper_id, job in by_id.items():
        paper_cfg = PipelineConfig.from_dict(job["config"])
        assert paper_cfg.base_dir == out / paper_id
        # The shared project memory of the template is not inherited: one wiki per paper.
        assert paper_cfg.memory.project_dir is None
        assert paper_cfg.project_memory_dir == out / paper_id / "memory"
        assert paper_cfg.template_dir == cfg.template_dir
        assert paper_cfg.models.writer == "claude-opus-5-5"
        assert paper_cfg.use_plotting is True and paper_cfg.memory.lab is False
        assert job["content_folder"] == str(root / paper_id / "raw_materials")
    assert by_id["p3"]["pdf_path"] is None
    # The template itself is left as it was.
    assert cfg.base_dir == tmp_path / "ignored"
    assert cfg.memory.project_dir == tmp_path / "shared-memory"


def test_batch_resumes_and_skips_papers_that_are_already_written(root, cfg, tmp_path, monkeypatch):
    out = tmp_path / "out"
    first = Workers(monkeypatch, {"p1": "success", "p2": "boom", "p3": "success"})
    log_path = run_batch(cfg, root_input_folder=root, output_dir=out)
    assert len(first.jobs) == 3
    # p3 "succeeded" but its PDF is gone, so the log entry is not trusted.
    (out / "p3" / "final_paper.pdf").unlink()
    earlier = read_json(log_path)
    earlier["other-batch"] = {"status": "success", "paper_path": "/elsewhere/final_paper.pdf"}
    write_json(log_path, earlier)

    second = Workers(monkeypatch, {"p2": "success", "p3": "success"})
    run_batch(cfg, root_input_folder=root, output_dir=out)

    assert sorted(job["paper_id"] for job in second.jobs) == ["p2", "p3"]
    log = read_json(log_path)
    assert {paper_id: entry["status"] for paper_id, entry in log.items()} == {
        "p1": "success",
        "p2": "success",
        "p3": "success",
        "other-batch": "success",
    }
    assert "error" not in log["p2"]

    third = Workers(monkeypatch, {})
    assert run_batch(cfg, root_input_folder=root, output_dir=out) == log_path
    assert third.jobs == [] and third.pool_sizes == []
    assert read_json(log_path) == log


def test_batch_rejects_bad_requests(root, cfg, tmp_path, monkeypatch):
    workers = Workers(monkeypatch, {})
    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        run_batch(cfg, root_input_folder=root, output_dir=tmp_path / "out", max_workers=0)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no papers"):
        run_batch(cfg, root_input_folder=empty, output_dir=tmp_path / "out")

    out = tmp_path / "broken"
    out.mkdir()
    (out / batch.LOG_NAME).write_text("{ not json")
    with pytest.raises(ValueError, match="is not valid JSON"):
        run_batch(cfg, root_input_folder=root, output_dir=out)
    assert workers.jobs == []


# -- the worker --------------------------------------------------------------------


def job_for(root: Path, cfg: PipelineConfig, out: Path, paper_id: str) -> dict[str, Any]:
    paper = next(p for p in load_papers(root_input_folder=root) if p["paper_id"] == paper_id)
    config = cfg.to_dict()
    config["base_dir"] = str(out / paper_id)
    config["memory"]["project_dir"] = None
    return {**paper, "config": config}


def test_worker_prepares_the_run_directory_and_reports_the_result(root, cfg, tmp_path, monkeypatch):
    seen: list[PipelineConfig] = []

    def fake_run_paper(config: PipelineConfig) -> PaperResult:
        seen.append(config)
        logging.getLogger("paper_orchestra.pipeline").warning("Stage outline started.")
        assert (config.base_dir / "raw_materials" / "idea_sparse.md").read_text() == "Idea of p1.\n"
        pdf = config.base_dir / "final_paper.pdf"
        pdf.write_bytes(b"%PDF")
        return PaperResult(True, pdf, config.base_dir, {}, {})

    monkeypatch.setattr(batch, "run_paper", fake_run_paper)
    handlers = list(logging.getLogger("paper_orchestra").handlers)
    out = tmp_path / "out"
    outcome = batch._write_paper(job_for(root, cfg, out, "p1"))

    assert outcome == {
        "paper_id": "p1",
        "status": "success",
        "paper_path": str(out / "p1" / "final_paper.pdf"),
    }
    assert seen[0].base_dir == out / "p1" and seen[0].models.writer == "claude-opus-5-5"
    # The ground-truth paper travels with the raw materials for the autoraters.
    original = out / "p1" / "raw_materials" / "original_paper.pdf"
    assert original.read_bytes() == b"%PDF original p1"
    # The pipeline's log of this paper is kept next to its outputs.
    assert "Stage outline started." in (out / "p1" / "paper_writing.log").read_text()
    assert logging.getLogger("paper_orchestra").handlers == handlers


def test_worker_returns_failures_instead_of_raising(root, cfg, tmp_path, monkeypatch):
    out = tmp_path / "out"

    def failing(config: PipelineConfig) -> PaperResult:
        return PaperResult(False, None, config.base_dir, {}, {}, error="stage 'outline' failed")

    monkeypatch.setattr(batch, "run_paper", failing)
    assert batch._write_paper(job_for(root, cfg, out, "p3")) == {
        "paper_id": "p3",
        "status": "failed",
        "paper_path": "",
        "error": "stage 'outline' failed",
    }
    assert not (out / "p3" / "raw_materials" / "original_paper.pdf").exists()

    def crashing(config: PipelineConfig) -> PaperResult:
        raise RuntimeError("out of disk")

    monkeypatch.setattr(batch, "run_paper", crashing)
    outcome = batch._write_paper(job_for(root, cfg, out, "p2"))
    assert outcome["status"] == "error" and outcome["error"] == "RuntimeError: out of disk"
    assert "out of disk" in (out / "p2" / "paper_writing.log").read_text()

    # Invalid inputs are an error of that paper, not of the batch.
    (root / "p1" / "raw_materials" / "idea_sparse.md").unlink()
    outcome = batch._write_paper(job_for(root, cfg, out, "p1"))
    assert outcome["status"] == "error"
    assert outcome["error"].startswith("FileNotFoundError: The idea file was not found")


def test_batch_runs_papers_in_real_worker_processes(root, cfg, tmp_path):
    # No model can be reached offline: an unroutable model id makes every worker stop at
    # the pipeline's first check, which is enough to exercise the process pool end to end.
    cfg.models = ModelConfig(
        writer="unroutable-model",
        reflection="unroutable-model",
        plotting="unroutable-model",
        fast="unroutable-model",
    )
    out = tmp_path / "out"
    log_path = run_batch(cfg, root_input_folder=root, output_dir=out, n_papers=2, max_workers=2)

    log = read_json(log_path)
    assert sorted(log) == ["p1", "p2"]
    for paper_id, entry in log.items():
        assert entry["status"] == "failed" and entry["paper_path"] == ""
        assert "the writer model 'unroutable-model' cannot be used" in entry["error"]
        state = read_json(out / paper_id / "state.json")
        assert state["status"] == "failed"
        assert state["config"]["base_dir"] == str(out / paper_id)
        assert (out / paper_id / "raw_materials" / "original_paper.pdf").is_file()
        assert "The run failed" in (out / paper_id / "paper_writing.log").read_text()
