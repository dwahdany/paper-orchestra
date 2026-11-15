from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from paper_orchestra import batch
from paper_orchestra.batch import load_papers, run_batch, summarize
from paper_orchestra.config import MemoryConfig, ModelConfig, PipelineConfig
from paper_orchestra.pipeline import PaperResult
from paper_orchestra.utils.common import read_json, write_json

posix_only = pytest.mark.skipif(os.name != "posix", reason="needs POSIX signals")


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


SCRIPT_ENV = "PAPER_ORCHESTRA_TEST_BATCH_SCRIPT"
#: Long enough that a paper still "being written" can only end by being stopped.
FOREVER = 120.0


def scripted_paper(job: dict[str, Any]) -> dict[str, Any]:
    """Stands in for ``batch._write_paper`` inside the real worker processes.

    A worker process sees none of the test's objects: its script arrives through the
    environment and what it observed is left as files in the paper's run directory.
    """
    paper_id = job["paper_id"]
    paper_dir = Path(job["config"]["base_dir"])
    out = paper_dir.parent
    step = json.loads(os.environ[SCRIPT_ENV])[paper_id]
    write_json(paper_dir / "job.json", job)
    write_json(paper_dir / "log_seen.json", read_json(out / batch.LOG_NAME, default={}))
    (paper_dir / "started").write_text(str(os.getpid()))
    # Papers that must be in flight together with this one before it goes on.
    deadline = time.monotonic() + 30
    for other in step.get("alongside", ()):
        while not (out / other / "started").exists():
            if time.monotonic() > deadline:
                return {"paper_id": paper_id, "status": "failed", "error": f"{other} never ran"}
            time.sleep(0.01)
    time.sleep(step.get("linger", 0.0))

    action = step["do"]
    if action == "fail":
        return {"paper_id": paper_id, "status": "failed", "paper_path": "", "error": step["error"]}
    if action == "raise":
        raise RuntimeError(step["error"])
    if action == "exit":
        os._exit(step["code"])
    if action == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    if action == "deaf at first":
        # What happens to a worker with more than one thread now and then: the batch's
        # stop request is taken by another thread and never interrupts the main one.
        interrupt = signal.getsignal(signal.SIGTERM)
        missed: list[int] = []

        def miss_the_first(signum: int, frame: object) -> None:
            if not missed:
                missed.append(signum)
                (paper_dir / "missed").write_text("")
            else:
                interrupt(signum, frame)

        signal.signal(signal.SIGTERM, miss_the_first)
    if action in ("sleep", "stubborn", "deaf at first"):
        try:
            time.sleep(FOREVER)
        except KeyboardInterrupt:
            (paper_dir / "interrupted").write_text("")
            if action != "stubborn":
                raise
            time.sleep(FOREVER)
    pdf = paper_dir / "final_paper.pdf"
    pdf.write_bytes(b"%PDF")
    return {"paper_id": paper_id, "status": "success", "paper_path": str(pdf)}


class Script:
    """Tells the stand-in workers what to do and reads back what they left behind."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, out: Path, steps: dict[str, dict[str, Any]]
    ) -> None:
        self.out = out
        for name in ("job.json", "log_seen.json", "started", "interrupted"):
            for leftover in out.glob(f"*/{name}"):
                leftover.unlink()
        monkeypatch.setattr(batch, "_write_paper", scripted_paper)
        monkeypatch.setenv(SCRIPT_ENV, json.dumps(steps))

    def jobs(self) -> dict[str, dict[str, Any]]:
        return {path.parent.name: read_json(path) for path in self.out.glob("*/job.json")}

    def log_seen(self, paper_id: str) -> dict[str, Any]:
        """The batch log as it was on disk when the paper's worker started."""
        return read_json(self.out / paper_id / "log_seen.json")

    def marked(self, marker: str) -> list[str]:
        return sorted(path.parent.name for path in self.out.glob(f"*/{marker}"))


SUCCESS = {"do": "success"}


def interrupt_after_the_first_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C arrives in the batch process right after it logged its first finished paper."""
    real_write_json = batch.write_json
    fired: list[bool] = []

    def write(path: Path, data: Any) -> Path:
        result = real_write_json(path, data)
        if not fired:
            fired.append(True)
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(batch, "write_json", write)


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
    out = tmp_path / "out"
    script = Script(
        monkeypatch,
        out,
        {
            "p1": SUCCESS,
            "p2": {"do": "fail", "error": "stage 'literature' failed after 2 attempt(s)"},
            "p3": {"do": "raise", "error": "out of disk"},
        },
    )
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
        "p3": {"status": "error", "paper_path": "", "error": "RuntimeError: out of disk"},
    }
    # One paper at a time, and the log on disk grew while the batch was still running:
    # each paper started with the results of the ones before it.
    assert script.log_seen("p1") == {}
    assert script.log_seen("p2")["p1"]["status"] == "success"
    assert sorted(script.log_seen("p3")) == ["p1", "p2"]
    assert "Batch finished: 3 paper(s), 1 written, 1 failed, 1 in error." in caplog.text
    assert "Paper p3 error (3 of 3): RuntimeError: out of disk" in caplog.text
    assert summarize(read_json(log_path)) == {"success": 1, "failed": 1, "error": 1}
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize(
    ("death", "reported"),
    [
        ({"do": "exit", "code": 3}, "worker exited with code 3"),
        pytest.param({"do": "kill"}, "worker exited with code -9 (SIGKILL)", marks=posix_only),
    ],
)
def test_a_worker_that_dies_takes_only_its_own_paper_with_it(
    death, reported, root, cfg, tmp_path, monkeypatch, caplog
):
    out = tmp_path / "out"
    Script(
        monkeypatch,
        out,
        {
            # Still being written when p2's worker dies next to it.
            "p1": {"do": "success", "alongside": ["p2"], "linger": 1.0},
            "p2": {**death, "alongside": ["p1"]},
            # Still queued at that moment.
            "p3": SUCCESS,
        },
    )
    with caplog.at_level(logging.INFO, logger="paper_orchestra"):
        log = read_json(run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=2))

    assert log == {
        "p1": {"status": "success", "paper_path": str(out / "p1" / "final_paper.pdf")},
        "p2": {"status": "error", "paper_path": "", "error": reported},
        "p3": {"status": "success", "paper_path": str(out / "p3" / "final_paper.pdf")},
    }
    assert (out / "p1" / "final_paper.pdf").is_file() and (out / "p3" / "final_paper.pdf").is_file()
    assert "Batch finished: 3 paper(s), 2 written, 0 failed, 1 in error." in caplog.text
    assert multiprocessing.active_children() == []


def test_every_paper_gets_its_own_config(root, cfg, tmp_path, monkeypatch):
    out = tmp_path / "out"
    everyone = ["p1", "p2", "p3"]
    script = Script(
        monkeypatch, out, {paper_id: {**SUCCESS, "alongside": everyone} for paper_id in everyone}
    )
    log = read_json(run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=8))

    # All three were in flight together: each waited for the other two before it finished.
    assert {paper_id: entry["status"] for paper_id, entry in log.items()} == dict.fromkeys(
        everyone, "success"
    )
    by_id = script.jobs()
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


def test_no_more_than_max_workers_papers_are_in_flight(root, cfg, tmp_path, monkeypatch):
    out = tmp_path / "out"
    script = Script(
        monkeypatch,
        out,
        {
            "p1": {**SUCCESS, "alongside": ["p2"]},
            "p2": {**SUCCESS, "alongside": ["p1"]},
            "p3": SUCCESS,
        },
    )
    run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=2)

    assert script.log_seen("p1") == script.log_seen("p2") == {}
    # The third paper had to wait for a free slot: one of the others was already logged.
    assert set(script.log_seen("p3")) & {"p1", "p2"}


def test_batch_resumes_and_skips_papers_that_are_already_written(root, cfg, tmp_path, monkeypatch):
    out = tmp_path / "out"
    first = Script(
        monkeypatch, out, {"p1": SUCCESS, "p2": {"do": "fail", "error": "boom"}, "p3": SUCCESS}
    )
    log_path = run_batch(cfg, root_input_folder=root, output_dir=out)
    assert sorted(first.jobs()) == ["p1", "p2", "p3"]
    # p3 "succeeded" but its PDF is gone, so the log entry is not trusted.
    (out / "p3" / "final_paper.pdf").unlink()
    earlier = read_json(log_path)
    earlier["other-batch"] = {"status": "success", "paper_path": "/elsewhere/final_paper.pdf"}
    write_json(log_path, earlier)

    second = Script(monkeypatch, out, {"p2": SUCCESS, "p3": SUCCESS})
    run_batch(cfg, root_input_folder=root, output_dir=out)

    assert sorted(second.jobs()) == ["p2", "p3"]
    log = read_json(log_path)
    assert {paper_id: entry["status"] for paper_id, entry in log.items()} == {
        "p1": "success",
        "p2": "success",
        "p3": "success",
        "other-batch": "success",
    }
    assert "error" not in log["p2"]

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no worker may be started when every paper is written")

    monkeypatch.setattr(batch, "_Worker", unexpected)
    assert run_batch(cfg, root_input_folder=root, output_dir=out) == log_path
    assert read_json(log_path) == log


def test_only_a_success_with_its_pdf_counts_as_already_written(tmp_path):
    pdf = tmp_path / "final_paper.pdf"
    pdf.write_bytes(b"%PDF")
    assert batch._already_written({"status": "success", "paper_path": str(pdf)})
    # "submitted" and "cancelled" are written by the Modal batch client for papers in flight.
    for status in ("failed", "error", "submitted", "cancelled"):
        assert not batch._already_written({"status": status, "paper_path": str(pdf)})
    assert not batch._already_written({"status": "success", "paper_path": str(tmp_path / "gone")})
    assert not batch._already_written({"status": "success", "paper_path": ""})
    assert not batch._already_written(None)
    assert summarize(
        {"a": {"status": "submitted"}, "b": {"status": "cancelled"}, "c": {"status": "success"}}
    ) == {"success": 1, "failed": 0, "error": 0}


@posix_only
def test_an_interrupted_batch_stops_its_workers_and_starts_no_further_papers(
    root, cfg, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    script = Script(
        monkeypatch,
        out,
        {
            "p1": {**SUCCESS, "alongside": ["p2"]},
            # Still being written when the interrupt arrives.
            "p2": {"do": "sleep"},
            # Still queued.
            "p3": SUCCESS,
        },
    )
    interrupt_after_the_first_paper(monkeypatch)
    began = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=2)

    # The paper in flight was stopped, not waited for, and nothing outlives the batch.
    assert time.monotonic() - began < FOREVER / 2
    assert multiprocessing.active_children() == []
    # Its worker was interrupted the way Ctrl-C would have, so the run could close its state.
    assert script.marked("interrupted") == ["p2"]
    assert script.marked("started") == ["p1", "p2"]
    assert read_json(out / "paper_writing_log.json") == {
        "p1": {"status": "success", "paper_path": str(out / "p1" / "final_paper.pdf")},
        "p2": {"status": "error", "paper_path": "", "error": batch.INTERRUPTED},
    }


@posix_only
def test_a_worker_that_ignores_the_stop_is_killed_after_the_grace_period(
    root, cfg, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    script = Script(
        monkeypatch,
        out,
        {"p1": {**SUCCESS, "alongside": ["p2"]}, "p2": {"do": "stubborn"}, "p3": SUCCESS},
    )
    monkeypatch.setattr(batch, "STOP_GRACE_SECONDS", 1.0)
    interrupt_after_the_first_paper(monkeypatch)
    began = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=2)

    assert time.monotonic() - began < FOREVER / 2
    assert multiprocessing.active_children() == []
    # It was asked first and carried on regardless.
    assert script.marked("interrupted") == ["p2"]
    assert not (out / "p2" / "final_paper.pdf").exists()
    assert read_json(out / "paper_writing_log.json")["p2"] == {
        "status": "error",
        "paper_path": "",
        "error": batch.INTERRUPTED,
    }


@posix_only
def test_a_worker_that_missed_the_stop_request_is_asked_again(root, cfg, tmp_path, monkeypatch):
    out = tmp_path / "out"
    script = Script(
        monkeypatch,
        out,
        {"p1": {**SUCCESS, "alongside": ["p2"]}, "p2": {"do": "deaf at first"}, "p3": SUCCESS},
    )
    # A grace period that is short for a failing test and many requests long.
    monkeypatch.setattr(batch, "STOP_GRACE_SECONDS", 6.0)
    monkeypatch.setattr(batch, "STOP_REPEAT_SECONDS", 0.2)
    interrupt_after_the_first_paper(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=2)

    assert multiprocessing.active_children() == []
    assert script.marked("missed") == ["p2"]
    # The paper was interrupted, so its run could close itself: it was not left to be
    # killed when the grace period ran out.
    assert script.marked("interrupted") == ["p2"]
    assert read_json(out / "paper_writing_log.json")["p2"] == {
        "status": "error",
        "paper_path": "",
        "error": batch.INTERRUPTED,
    }


@posix_only
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["ctrl_c", "sigterm"])
def test_an_interrupt_while_a_worker_is_started_still_stops_that_worker(
    signum, root, cfg, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    script = Script(monkeypatch, out, {"p1": {"do": "sleep"}, "p2": SUCCESS, "p3": SUCCESS})

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    context = multiprocessing.get_context("spawn")
    start_process = context.Process.start

    def start(process: Any) -> None:
        start_process(process)
        # The worker's process exists and the batch has not taken note of it yet.
        os.kill(os.getpid(), signum)

    monkeypatch.setattr(context.Process, "start", start)
    # Ctrl-C as Python takes it and SIGTERM as the command line does.
    before = signal.signal(signum, interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_batch(cfg, root_input_folder=root, output_dir=out, max_workers=1)
        left = multiprocessing.active_children()
        handler = signal.getsignal(signum)
    finally:
        signal.signal(signum, signal.SIG_DFL if before is None else before)
        # A worker the batch lost track of would keep this test session from ending.
        for process in multiprocessing.active_children():
            process.kill()
            process.join()

    assert left == [], "a worker outlived the batch that started it"
    # The interrupt was acted on once the worker was known: it was stopped and its paper
    # recorded, and no further paper was started.
    assert script.marked("started") in ([], ["p1"])
    assert read_json(out / "paper_writing_log.json") == {
        "p1": {"status": "error", "paper_path": "", "error": batch.INTERRUPTED}
    }
    # The signal is handled as before from then on.
    assert handler is interrupt


#: The real command line with a slow stand-in for the pipeline.  The stand-in is installed
#: at module level on purpose: the worker processes import this file again and need it too.
SIGNAL_DRIVER = """
import os
import sys
import threading
import time
from pathlib import Path

from paper_orchestra import batch, cli
from paper_orchestra.pipeline import PaperResult


def slow_run_paper(cfg, **kwargs):
    base = Path(cfg.base_dir)
    # A run has more threads than its main one (it keeps usage.json current from another),
    # and a process with several threads does not always hand a signal to the main one.
    threading.Thread(target=threading.Event().wait, daemon=True).start()
    try:
        # Inside the try: the test sends its signal as soon as this file exists.
        (base / "started").write_text(str(os.getpid()))
        time.sleep(120)
    except KeyboardInterrupt:
        (base / "interrupted").write_text("")
        if not os.environ.get("BATCH_TEST_STUBBORN"):
            raise
        time.sleep(120)
    pdf = base / "final_paper.pdf"
    pdf.write_bytes(b"%PDF")
    return PaperResult(True, pdf, base, {}, {})


batch.run_paper = slow_run_paper

if __name__ == "__main__":
    cli.main(sys.argv[1:])
"""


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _stopping(stderr: Path) -> bool:
    """Whether the batch has said (in its log, on stderr) that it is stopping its papers."""
    return "Stopping the batch" in " ".join(stderr.read_text().split())


@posix_only
@pytest.mark.parametrize("how", ["ctrl_c", "sigterm", "ctrl_c_twice"])
def test_stopping_the_batch_command_stops_the_papers_in_flight(how, root, template_dir, tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(SIGNAL_DRIVER)
    out = tmp_path / "out"
    # In the last case the papers carry on after their interrupt, as a run does while a
    # stage finishes its current model call.
    env = {**os.environ, "BATCH_TEST_STUBBORN": "1" if how == "ctrl_c_twice" else ""}
    said = tmp_path / "stderr.txt"
    with said.open("w") as stderr:
        proc = subprocess.Popen(
            [sys.executable, str(driver), "batch", "--root-input-folder", str(root),
             "--template", str(template_dir), "--output-dir", str(out), "--max-workers", "2",
             "--no-lab-memory"],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            # Its own process group, so the signal of a terminal's Ctrl-C can be reproduced.
            start_new_session=True,
        )  # fmt: skip
    try:
        deadline = time.monotonic() + 60
        while len(list(out.glob("*/started"))) < 2:
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        if how == "sigterm":
            os.kill(proc.pid, signal.SIGTERM)  # kill, timeout, a scheduler: the batch only
        else:
            os.killpg(proc.pid, signal.SIGINT)  # Ctrl-C: the batch and its workers at once
        if how == "ctrl_c_twice":
            # A second Ctrl-C ends the grace period the batch gives the papers.  It has to
            # come once that period has begun: an interrupt that arrives while the batch is
            # still acting on the first one is the same interrupt.
            while len(list(out.glob("*/interrupted"))) < 2 or not _stopping(said):
                assert proc.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            os.killpg(proc.pid, signal.SIGINT)

        # Long before a paper would have finished, and before the grace period is over.
        assert proc.wait(timeout=batch.STOP_GRACE_SECONDS / 2) == 130
        deadline = time.monotonic() + 10
        while _group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _group_alive(proc.pid), "worker processes outlived the batch"
        # Both papers in flight took their interrupt path; the third was never started.
        assert sorted(p.parent.name for p in out.glob("*/started")) == ["p1", "p2"]
        assert sorted(p.parent.name for p in out.glob("*/interrupted")) == ["p1", "p2"]
        assert not list(out.glob("*/final_paper.pdf"))
        interrupted = {"status": "error", "paper_path": "", "error": batch.INTERRUPTED}
        assert read_json(out / "paper_writing_log.json") == {"p1": interrupted, "p2": interrupted}
    finally:
        if _group_alive(proc.pid):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_batch_rejects_bad_requests(root, cfg, tmp_path, monkeypatch):
    script = Script(monkeypatch, tmp_path / "out", {})
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
    assert script.jobs() == {} and multiprocessing.active_children() == []


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
