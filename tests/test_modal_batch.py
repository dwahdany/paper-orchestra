from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from paper_orchestra.batch import summarize
from paper_orchestra.modal_batch import run_batch_on_modal
from paper_orchestra.utils.common import read_json

pytest.importorskip("modal")


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


# -- the batch on Modal ------------------------------------------------------------


class FakeVolume:
    """A ``modal.Volume`` whose contents are a local directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.uploads: list[bool] = []

    @contextmanager
    def batch_upload(self, force: bool = False):
        self.uploads.append(force)
        yield self

    def put_file(self, local: Path, remote: str) -> None:
        target = self.root / remote.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local).read_bytes())

    def read_file(self, path: str):
        target = self.root / path
        if not target.is_file():
            raise FileNotFoundError(path)
        yield target.read_bytes()


class FakeCall:
    object_id = "fc-batch"

    def __init__(self, summaries: list[dict[str, Any]]) -> None:
        self._summaries = summaries

    def get(self, timeout: float | None = None) -> list[dict[str, Any]]:
        return self._summaries


class FakeWriteBatch:
    """The deployed ``write_batch``: writes each paper from what was uploaded for it."""

    def __init__(self, volume: FakeVolume, failing: tuple[str, ...] = ()) -> None:
        self.volume = volume
        self.failing = failing
        self.batches: list[list[dict[str, Any]]] = []

    def spawn(self, jobs: list[dict[str, Any]]) -> FakeCall:
        self.batches.append(jobs)
        return FakeCall([self._write(job) for job in jobs])

    def _write(self, job: dict[str, Any]) -> dict[str, Any]:
        run_id = job["run_id"]
        if run_id in self.failing:
            error = "stage 'outline' failed after 2 attempt(s)"
            return {"success": False, "run_id": run_id, "error": error, "files": {}}
        root = self.volume.root
        idea = (root / job["raw_materials_dir"] / job["idea_filename"]).read_text()
        template = (root / job["template_dir"] / "template.tex").read_text()
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final_paper.pdf").write_text(f"%PDF {idea}")
        (run_dir / "final_paper.tex").write_text(template)
        files = {"pdf": f"runs/{run_id}/final_paper.pdf", "tex": f"runs/{run_id}/final_paper.tex"}
        return {"success": True, "run_id": run_id, "error": None, "files": files}


@pytest.fixture
def modal_volume(tmp_path: Path) -> FakeVolume:
    pytest.importorskip("modal")
    return FakeVolume(tmp_path / "volume")


def test_modal_batch_uploads_submits_and_downloads(root, template_dir, tmp_path, modal_volume):
    (root / "p3" / "raw_materials" / "idea_sparse.md").unlink()
    (root / "not a run id" / "raw_materials").mkdir(parents=True)
    write_batch = FakeWriteBatch(modal_volume, failing=("p2",))
    out = tmp_path / "out"
    options = {"use_plotting": True, "writer_model": "claude-opus-5-5", "research_cutoff": None}

    log_path = run_batch_on_modal(
        options,
        template_dir=template_dir,
        volume=modal_volume,
        write_batch=write_batch,
        root_input_folder=root,
        output_dir=out,
    )

    log = read_json(log_path)
    assert log_path == out / "paper_writing_log.json"
    assert log["p1"] == {"status": "success", "paper_path": str(out / "p1" / "final_paper.pdf")}
    assert log["p2"] == {
        "status": "failed",
        "paper_path": "",
        "error": "stage 'outline' failed after 2 attempt(s)",
    }
    # Papers that cannot be submitted are errors of their own; the others still run.
    assert log["p3"]["status"] == "error" and "idea_sparse.md" in log["p3"]["error"]
    assert (
        log["not a run id"]["status"] == "error"
        and "run_id must be" in log["not a run id"]["error"]
    )
    assert (out / "p1" / "final_paper.pdf").read_text() == "%PDF Idea of p1.\n"
    assert (out / "p1" / "final_paper.tex").read_text() == "\\documentclass{article}\n"

    [jobs] = write_batch.batches
    assert [job["run_id"] for job in jobs] == ["p1", "p2"]
    assert jobs[0]["raw_materials_dir"] == "inputs/p1/raw_materials"
    assert jobs[0]["template_dir"] == "inputs/p1/template" and "template" not in jobs[0]
    assert jobs[0]["use_plotting"] is True and jobs[0]["models"] == {"writer": "claude-opus-5-5"}
    assert modal_volume.uploads == [True, True]

    # Submitting again skips the paper that was downloaded and resumes the others.
    again = FakeWriteBatch(modal_volume)
    run_batch_on_modal(
        options,
        template_dir=template_dir,
        volume=modal_volume,
        write_batch=again,
        root_input_folder=root,
        output_dir=out,
    )
    assert [[job["run_id"] for job in batch] for batch in again.batches] == [["p2"]]
    log = read_json(log_path)
    assert log["p2"] == {"status": "success", "paper_path": str(out / "p2" / "final_paper.pdf")}
    assert summarize(log) == {"success": 2, "failed": 0, "error": 2}


def test_modal_batch_rejects_what_is_wrong_for_every_paper_before_uploading(
    root, template_dir, tmp_path, modal_volume
):
    write_batch = FakeWriteBatch(modal_volume)
    common = {
        "volume": modal_volume,
        "write_batch": write_batch,
        "root_input_folder": root,
        "output_dir": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="effort must be one of"):
        run_batch_on_modal({"effort": "ludicrous"}, template_dir=template_dir, **common)
    (template_dir / "guidelines.md").unlink()
    with pytest.raises(FileNotFoundError, match=r"has no guidelines\.md"):
        run_batch_on_modal({}, template_dir=template_dir, **common)
    with pytest.raises(ValueError, match="exactly one of"):
        run_batch_on_modal(
            {}, template_dir=template_dir, volume=modal_volume, write_batch=write_batch,
            output_dir=tmp_path / "out",
        )  # fmt: skip
    assert modal_volume.uploads == [] and write_batch.batches == []
