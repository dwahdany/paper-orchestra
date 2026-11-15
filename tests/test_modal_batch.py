from __future__ import annotations

import inspect
import logging
import re
import shutil
import types
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from paper_orchestra.batch import _already_written, summarize
from paper_orchestra.utils.common import read_json

modal = pytest.importorskip("modal")

from paper_orchestra import modal_app  # noqa: E402
from paper_orchestra.modal_batch import (  # noqa: E402
    batch_run_id,
    options_digest,
    run_batch_on_modal,
)


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
def logs(caplog, monkeypatch):
    """Records of `paper_orchestra.*`, even if an earlier test stopped the logger propagating."""
    monkeypatch.setattr(logging.getLogger("paper_orchestra"), "propagate", True)
    caplog.set_level(logging.INFO, logger="paper_orchestra")
    return caplog


# -- the Modal side, faked ---------------------------------------------------------


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

    def remove_file(self, path: str, recursive: bool = False) -> None:
        target = self.root / path
        if not target.exists():
            raise FileNotFoundError(path)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    def listdir(self, path: str, *, recursive: bool = False):
        kinds = modal.volume.FileEntryType
        return [
            types.SimpleNamespace(
                path=found.relative_to(self.root).as_posix(),
                type=kinds.DIRECTORY if found.is_dir() else kinds.FILE,
            )
            for found in sorted((self.root / path).rglob("*"))
        ]


class FakeCall:
    """A ``modal.FunctionCall``: says "still running" ``pending`` times, then ends."""

    def __init__(self, call_id: str, run: Callable[[], Any], pending: int = 0) -> None:
        self.object_id = call_id
        self.pending = pending
        self.polls = 0
        self.cancelled_with: bool | None = None
        #: Raised by the next ``get`` instead of answering (an interrupt, a lost connection).
        self.interruption: BaseException | None = None
        #: Called at the start of every ``get``.
        self.on_poll: Callable[[], None] = lambda: None
        self._run = run
        self._outcome: tuple[str, Any] | None = None

    def get(self, timeout: float | None = None) -> Any:
        assert timeout == 0, "the batch only ever asks, it never blocks on one call"
        self.polls += 1
        self.on_poll()
        if self.interruption is not None:
            interruption, self.interruption = self.interruption, None
            raise interruption
        if self.cancelled_with is not None:
            raise modal.exception.RemoteError("the function call was cancelled")
        if self.pending:
            self.pending -= 1
            raise TimeoutError
        if self._outcome is None:
            try:
                self._outcome = ("returned", self._run())
            except Exception as exc:
                self._outcome = ("raised", exc)
        kind, value = self._outcome
        if kind == "raised":
            raise value
        return value

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled_with = terminate_containers


class FakeWritePaper:
    """The deployed ``write_paper``: writes a paper from what was uploaded for it.

    Like the real one it resumes: a run directory that already holds a paper
    is returned untouched, with every stage reused, unless the job says not to.
    """

    def __init__(self, volume: FakeVolume, failing: tuple[str, ...] = ()) -> None:
        self.volume = volume
        self.failing = failing
        self.jobs: list[dict[str, Any]] = []
        self.calls: dict[str, FakeCall] = {}
        #: How many polls each new call stays running, by paper id.
        self.pending: dict[str, int] = {}
        self.crashing: tuple[str, ...] = ()
        self.written: list[str] = []

    def spawn(self, job: dict[str, Any]) -> FakeCall:
        self.jobs.append(job)
        paper_id = job["run_id"].rsplit("-", 1)[0]
        call = FakeCall(
            f"fc-{len(self.calls) + 1}", lambda: self._write(job), self.pending.get(paper_id, 0)
        )
        self.calls[call.object_id] = call
        return call

    def lookup(self, call_id: str) -> FakeCall:
        """``modal.FunctionCall.from_id``: lazy, so an unknown id only fails when asked."""

        def gone() -> Any:
            raise modal.exception.NotFoundError(f"function call {call_id} not found")

        return self.calls.get(call_id) or FakeCall(call_id, gone)

    def call_of(self, paper_id: str) -> FakeCall:
        """The latest call spawned for a paper."""
        indexes = [i for i, job in enumerate(self.jobs) if job["run_id"].startswith(paper_id)]
        return self.calls[f"fc-{indexes[-1] + 1}"]

    def _write(self, job: dict[str, Any]) -> dict[str, Any]:
        run_id = job["run_id"]
        paper_id = run_id.rsplit("-", 1)[0]
        if paper_id in self.crashing:
            raise RuntimeError("container crashed")
        if paper_id in self.failing:
            error = "stage 'outline' failed after 2 attempt(s)"
            return {"success": False, "run_id": run_id, "error": error, "files": {}}
        root = self.volume.root
        run_dir = root / "runs" / run_id
        files = {"pdf": f"runs/{run_id}/final_paper.pdf", "tex": f"runs/{run_id}/final_paper.tex"}
        reused = (run_dir / "final_paper.pdf").is_file() and job.get("resume", True)
        if not reused:
            idea = (root / job["raw_materials_dir"] / job["idea_filename"]).read_text()
            template = (root / job["template_dir"] / "template.tex").read_text()
            writer = job["models"].get("writer", "default-model")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "final_paper.pdf").write_text(f"%PDF {idea}written by {writer}")
            (run_dir / "final_paper.tex").write_text(template)
            self.written.append(run_id)
        return {
            "success": True,
            "run_id": run_id,
            "error": None,
            "files": files,
            "stages": {"outline": {"reused": reused}, "finalize": {"reused": reused}},
        }


@pytest.fixture
def volume(tmp_path: Path) -> FakeVolume:
    return FakeVolume(tmp_path / "volume")


@pytest.fixture
def write_paper(volume: FakeVolume) -> FakeWritePaper:
    return FakeWritePaper(volume)


@pytest.fixture
def submit(root, template_dir, volume, write_paper, tmp_path):
    """Run the batch against the fakes; keyword arguments override the defaults."""

    def run(options: dict[str, Any] | None = None, **overrides: Any) -> Path:
        arguments: dict[str, Any] = {
            "template_dir": template_dir,
            "volume": volume,
            "write_paper": write_paper,
            "lookup_call": write_paper.lookup,
            "root_input_folder": root,
            "output_dir": tmp_path / "out",
            "poll_seconds": 0,
        }
        arguments.update(overrides)
        return run_batch_on_modal(options or {}, **arguments)

    return run


def run_ids(function: FakeWritePaper) -> list[str]:
    return [job["run_id"] for job in function.jobs]


# -- the batch on Modal ------------------------------------------------------------


def test_modal_batch_uploads_submits_and_downloads(root, tmp_path, volume, write_paper, submit):
    (root / "p3" / "raw_materials" / "idea_sparse.md").unlink()
    (root / "not a run id" / "raw_materials").mkdir(parents=True)
    write_paper.failing = ("p2",)
    out = tmp_path / "out"
    options = {"use_plotting": True, "writer_model": "claude-opus-5-5", "research_cutoff": None}

    log_path = submit(options)

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
    assert (out / "p1" / "final_paper.pdf").read_text().startswith("%PDF Idea of p1.\n")
    assert (out / "p1" / "final_paper.tex").read_text() == "\\documentclass{article}\n"

    # One call per paper, under the paper id plus the digest of the options and template.
    first, second = write_paper.jobs
    assert re.fullmatch(r"p1-[0-9a-f]{8}", first["run_id"])
    assert re.fullmatch(r"p2-[0-9a-f]{8}", second["run_id"])
    assert first["run_id"].split("-")[1] == second["run_id"].split("-")[1]
    assert first["raw_materials_dir"] == f"inputs/{first['run_id']}/raw_materials"
    assert first["template_dir"] == f"inputs/{first['run_id']}/template" and "template" not in first
    assert first["use_plotting"] is True and first["models"] == {"writer": "claude-opus-5-5"}
    assert volume.uploads == [True, True]

    # Submitting again skips the paper that was downloaded and resumes the failed one.
    write_paper.failing = ()
    submit(options)
    assert run_ids(write_paper)[2:] == [second["run_id"]]
    log = read_json(log_path)
    assert log["p2"] == {"status": "success", "paper_path": str(out / "p2" / "final_paper.pdf")}
    assert summarize(log) == {"success": 2, "failed": 0, "error": 2}


def test_modal_batch_rejects_what_is_wrong_for_every_paper_before_uploading(
    root, template_dir, tmp_path, volume, write_paper, submit
):
    with pytest.raises(ValueError, match="effort must be one of"):
        submit({"effort": "ludicrous"})
    (template_dir / "guidelines.md").unlink()
    with pytest.raises(FileNotFoundError, match=r"has no guidelines\.md"):
        submit()
    with pytest.raises(ValueError, match="exactly one of"):
        submit(root_input_folder=None)
    assert volume.uploads == [] and write_paper.jobs == []


def test_calls_are_logged_before_waiting_and_each_paper_is_collected_as_it_finishes(
    root, tmp_path, write_paper, submit
):
    """A batch must not depend on one call that returns everything at the very end."""
    (root / "p3" / "raw_materials" / "idea_sparse.md").unlink()
    write_paper.pending = {"p2": 3}
    log_path = tmp_path / "out" / "paper_writing_log.json"
    seen: dict[str, Any] = {}

    def at_first_poll() -> None:
        seen.setdefault("before waiting", read_json(log_path))

    def while_p2_runs() -> None:
        seen.setdefault("while p2 runs", read_json(log_path))

    spawn = write_paper.spawn

    def spawn_and_watch(job: dict[str, Any]) -> FakeCall:
        if job["run_id"].startswith("p2"):
            seen["while submitting"] = read_json(log_path, default={})
        call = spawn(job)
        call.on_poll = at_first_poll if job["run_id"].startswith("p1") else while_p2_runs
        return call

    write_paper.spawn = spawn_and_watch
    submit()

    p1, p2 = run_ids(write_paper)
    # A call is on disk as soon as it exists, not once every paper has been submitted: a
    # process that is killed while it is still uploading leaves what it started findable.
    assert seen["while submitting"] == {
        "p1": {"status": "submitted", "paper_path": "", "run_id": p1, "call_id": "fc-1"}
    }
    before = seen["before waiting"]
    assert before["p1"] == {
        "status": "submitted",
        "paper_path": "",
        "run_id": p1,
        "call_id": "fc-1",
    }
    assert before["p2"] == {
        "status": "submitted",
        "paper_path": "",
        "run_id": p2,
        "call_id": "fc-2",
    }
    assert before["p3"]["status"] == "error"
    # Neither status counts as a written paper.
    assert summarize(before) == {"success": 0, "failed": 0, "error": 1}
    assert not _already_written(before["p1"])
    # p1 was downloaded and logged while p2 was still being written.
    during = seen["while p2 runs"]
    assert during["p1"]["status"] == "success" and during["p2"]["status"] == "submitted"
    assert Path(during["p1"]["paper_path"]).is_file()
    assert write_paper.call_of("p2").polls == 4
    assert summarize(read_json(log_path)) == {"success": 2, "failed": 0, "error": 1}


def test_an_interrupted_batch_keeps_running_and_the_next_invocation_reattaches(
    tmp_path, volume, write_paper, submit, logs
):
    write_paper.pending = {"p1": 1, "p2": 1, "p3": 1}
    spawn = write_paper.spawn

    def spawn_then_interrupt(job: dict[str, Any]) -> FakeCall:
        call = spawn(job)
        if job["run_id"].startswith("p2"):
            call.interruption = KeyboardInterrupt()
        return call

    write_paper.spawn = spawn_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        submit()

    log_path = tmp_path / "out" / "paper_writing_log.json"
    log = read_json(log_path)
    assert [log[p]["status"] for p in ("p1", "p2", "p3")] == ["submitted"] * 3
    assert [log[p]["call_id"] for p in ("p1", "p2", "p3")] == ["fc-1", "fc-2", "fc-3"]
    # Nothing was cancelled, and the user is told what is going on and what to do.
    assert all(call.cancelled_with is None for call in write_paper.calls.values())
    assert "3 paper(s) are still being written on Modal (p1, p2, p3)" in logs.text
    assert "Run the same command again" in logs.text and "--cancel" in logs.text

    # The same command again waits for the same calls: no upload, no second run of a paper.
    write_paper.spawn = spawn
    uploads = list(volume.uploads)
    submit()
    assert len(write_paper.jobs) == 3 and volume.uploads == uploads
    assert sorted(write_paper.written) == sorted(run_ids(write_paper))
    log = read_json(log_path)
    assert summarize(log) == {"success": 3, "failed": 0, "error": 0}
    assert log["p2"] == {
        "status": "success",
        "paper_path": str(tmp_path / "out" / "p2" / "final_paper.pdf"),
    }
    assert "waiting for that call instead of starting the paper a second time" in logs.text


def test_losing_the_connection_fails_no_paper_and_starts_none_again(
    tmp_path, write_paper, submit, logs
):
    spawn = write_paper.spawn

    def spawn_then_disconnect(job: dict[str, Any]) -> FakeCall:
        call = spawn(job)
        call.interruption = modal.exception.ConnectionError("Deadline exceeded")
        return call

    write_paper.spawn = spawn_then_disconnect
    with pytest.raises(modal.exception.ConnectionError):
        submit()
    log = read_json(tmp_path / "out" / "paper_writing_log.json")
    assert {entry["status"] for entry in log.values()} == {"submitted"}
    assert "Stopped waiting. 3 paper(s) are still being written on Modal" in logs.text

    # Still unreachable when asked again: the papers stay submitted and nothing is spawned.
    for call in write_paper.calls.values():
        call.interruption = modal.exception.ConnectionError("Deadline exceeded")
    with pytest.raises(modal.exception.ConnectionError):
        submit()
    assert len(write_paper.jobs) == 3

    for call in write_paper.calls.values():
        call.interruption = None
    write_paper.spawn = spawn
    submit()
    assert len(write_paper.jobs) == 3
    log = read_json(tmp_path / "out" / "paper_writing_log.json")
    assert summarize(log) == {"success": 3, "failed": 0, "error": 0}


def test_a_download_that_fails_leaves_the_paper_to_the_next_invocation(
    tmp_path, volume, write_paper, submit
):
    read_file = volume.read_file

    def unreachable(path: str):
        raise modal.exception.ConnectionError("Deadline exceeded")
        yield b""

    volume.read_file = unreachable
    with pytest.raises(modal.exception.ConnectionError):
        submit(n_papers=1)
    log_path = tmp_path / "out" / "paper_writing_log.json"
    assert read_json(log_path)["p1"]["status"] == "submitted"

    # The call is over and Modal still has its result: it is collected, not run again.
    volume.read_file = read_file
    submit(n_papers=1)
    assert len(write_paper.jobs) == 1
    assert read_json(log_path)["p1"] == {
        "status": "success",
        "paper_path": str(tmp_path / "out" / "p1" / "final_paper.pdf"),
    }


def test_a_paper_is_submitted_again_only_when_its_earlier_call_left_nothing(
    tmp_path, write_paper, submit, logs
):
    write_paper.pending = {"p1": 1, "p2": 1, "p3": 1}
    spawn = write_paper.spawn

    def spawn_then_interrupt(job: dict[str, Any]) -> FakeCall:
        call = spawn(job)
        if job["run_id"].startswith("p1"):
            call.interruption = KeyboardInterrupt()
        return call

    write_paper.spawn = spawn_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        submit()
    write_paper.spawn = spawn
    write_paper.pending = {}
    first = run_ids(write_paper)

    # Meanwhile: p1's result expired, p2's call is unknown to Modal, p3 failed as a paper.
    def expired() -> Any:
        raise modal.exception.OutputExpiredError()

    write_paper.calls["fc-1"] = FakeCall("fc-1", expired)
    del write_paper.calls["fc-2"]
    write_paper.failing = ("p3",)
    log_path = submit()

    # p1 and p2 are started again under the same run ids, which resumes them.  p3 has a
    # result, a failed one: it is reported, and started again by the next invocation.
    assert run_ids(write_paper)[3:] == first[:2]
    log = read_json(log_path)
    assert log["p1"]["status"] == log["p2"]["status"] == "success"
    assert log["p3"] == {
        "status": "failed",
        "paper_path": "",
        "error": "stage 'outline' failed after 2 attempt(s)",
    }
    assert "the earlier call for run " + first[0] + " is over and left no result" in logs.text
    assert "OutputExpiredError" in logs.text and "NotFoundError" in logs.text

    # A call that raises while this invocation waits for it is that paper's failure.
    write_paper.failing = ()
    write_paper.crashing = ("p3",)
    submit()
    assert run_ids(write_paper)[5:] == [first[2]]
    assert read_json(log_path)["p3"] == {
        "status": "failed",
        "paper_path": "",
        "error": "RuntimeError: container crashed",
    }


def test_cancel_stops_the_calls_that_are_still_running(tmp_path, write_paper, submit, logs):
    write_paper.pending = {"p1": 1, "p2": 5, "p3": 5}
    spawn = write_paper.spawn

    def spawn_then_interrupt(job: dict[str, Any]) -> FakeCall:
        call = spawn(job)
        if job["run_id"].startswith("p1"):
            call.interruption = KeyboardInterrupt()
        return call

    write_paper.spawn = spawn_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        submit()
    write_paper.spawn = spawn
    write_paper.call_of("p1").pending = 0  # p1 finished before the user cancelled

    log_path = submit(cancel=True)

    _, p2, p3 = run_ids(write_paper)
    assert len(write_paper.jobs) == 3  # cancelling starts nothing
    cancelled = {name: call.cancelled_with for name, call in write_paper.calls.items()}
    assert cancelled == {"fc-1": None, "fc-2": True, "fc-3": True}  # containers are stopped too
    log = read_json(log_path)
    assert log["p1"]["status"] == "success" and Path(log["p1"]["paper_path"]).is_file()
    assert log["p2"] == {"status": "cancelled", "paper_path": "", "run_id": p2, "call_id": "fc-2"}
    assert log["p3"] == {"status": "cancelled", "paper_path": "", "run_id": p3, "call_id": "fc-3"}
    assert summarize(log) == {"success": 1, "failed": 0, "error": 0}
    assert not _already_written(log["p2"])
    assert "Cancelled 2 paper(s) on Modal" in logs.text

    # Cancelling again finds nothing to stop; submitting again starts the cancelled papers.
    submit(cancel=True)
    assert len(write_paper.jobs) == 3
    assert "No paper of this batch is being written on Modal" in logs.text
    write_paper.pending = {}
    submit()
    assert run_ids(write_paper)[3:] == [p2, p3]
    assert summarize(read_json(log_path)) == {"success": 3, "failed": 0, "error": 0}


def test_cancelling_a_batch_that_was_never_submitted_leaves_a_log_to_report(
    tmp_path, volume, write_paper, submit, logs
):
    """The caller reads the log this returns; `--cancel` in a new directory must have one."""
    log_path = submit(cancel=True)
    assert log_path == tmp_path / "out" / "paper_writing_log.json"
    assert read_json(log_path, default="missing") == {}
    assert write_paper.jobs == [] and volume.uploads == []
    assert "No paper of this batch is being written on Modal" in logs.text
    assert "Cancelled" not in logs.text


# -- run ids -----------------------------------------------------------------------


def test_a_batch_with_other_options_does_not_get_the_papers_of_an_earlier_batch(
    tmp_path, write_paper, submit, template_dir, logs
):
    """The run directory on the volume belongs to a paper *and* an option set."""
    out_a, out_b, out_c = (tmp_path / name for name in ("out_a", "out_b", "out_c"))
    submit({"writer_model": "model-a"}, output_dir=out_a, n_papers=1)
    submit({"writer_model": "model-b", "effort": "low"}, output_dir=out_b, n_papers=1)
    run_a, run_b = run_ids(write_paper)
    assert run_a != run_b and run_a.startswith("p1-") and run_b.startswith("p1-")
    assert (out_a / "p1" / "final_paper.pdf").read_text().endswith("written by model-a")
    assert (out_b / "p1" / "final_paper.pdf").read_text().endswith("written by model-b")
    assert "nothing was written this time" not in logs.text

    # The same batch again, into another directory: same run, resumed, and the user is told
    # that nothing was written this time.
    submit({"writer_model": "model-b", "effort": "low"}, output_dir=out_c, n_papers=1)
    assert run_ids(write_paper)[2] == run_b and write_paper.written == [run_a, run_b]
    assert f"Run {run_b} had already finished on the volume" in logs.text
    assert "--no-resume" in logs.text

    # --no-resume writes that same run again instead of starting an unrelated one.
    submit(
        {"writer_model": "model-b", "effort": "low", "resume": False},
        output_dir=tmp_path / "out_d",
        n_papers=1,
    )
    assert run_ids(write_paper)[3] == run_b and write_paper.written == [run_a, run_b, run_b]


def test_the_digest_covers_what_shapes_a_paper_and_nothing_else(template_dir, monkeypatch):
    base = options_digest({"writer_model": "m"}, template_dir)
    assert re.fullmatch(r"[0-9a-f]{8}", base)
    same = [
        {"writer_model": "m", "resume": False},
        {"writer_model": "m", "research_cutoff": None, "use_plotting": False},
        {"writer_model": "m", "keep_provided_figures": False},
        {"writer_model": "m", "reflection_model": None, "dream": "stages", "memory": True},
    ]
    assert {options_digest(options, template_dir) for options in same} == {base}
    different = [
        {"writer_model": "other"},
        {"writer_model": "m", "effort": "low"},
        {"writer_model": "m", "use_plotting": True},
        {"writer_model": "m", "use_plotting": True, "keep_provided_figures": True},
        {"writer_model": "m", "lab_memory": False},
        {"writer_model": "m", "research_cutoff": "2024-11"},
        {"writer_model": "m", "idea_filename": "idea.md"},
    ]
    digests = {options_digest(options, template_dir) for options in different}
    assert len(digests) == len(different) and base not in digests

    # An unset cutoff means "now": the run must not move when the month changes.
    monkeypatch.setattr(modal_app, "default_research_cutoff", lambda: "2031-01")
    assert options_digest({"writer_model": "m"}, template_dir) == base

    # The template: its contents and its file names, hidden files aside.
    (template_dir / ".DS_Store").write_text("junk")
    assert options_digest({"writer_model": "m"}, template_dir) == base
    (template_dir / "guidelines.md").write_text("Nine pages.\n")
    edited = options_digest({"writer_model": "m"}, template_dir)
    (template_dir / "guidelines.md").rename(template_dir / "rules.md")
    renamed = options_digest({"writer_model": "m"}, template_dir)
    assert len({base, edited, renamed}) == 3


def test_run_ids_of_long_paper_ids_stay_valid_and_distinct():
    assert batch_run_id("p1", "0123abcd") == "p1-0123abcd"
    assert batch_run_id("x" * 71, "0123abcd") == "x" * 71 + "-0123abcd"
    common = "a-very-long-paper-identifier-" * 3
    long_ids = [common + "first", common + "second", "y" * 72]
    shortened = [batch_run_id(paper_id, "0123abcd") for paper_id in long_ids]
    assert len(set(shortened)) == 3
    for run_id in shortened:
        assert len(run_id) == modal_app.RUN_ID_MAX and run_id.endswith("-0123abcd")
        assert modal_app.validate_run_id(run_id) == run_id
    assert shortened[0].startswith(common[:50])
    assert batch_run_id(long_ids[0], "0123abcd") == shortened[0]  # stable, so it resumes
    with pytest.raises(modal_app.JobError, match="run_id must be"):
        batch_run_id("has a space", "0123abcd")


def test_long_paper_ids_are_submitted_under_their_shortened_run_ids(
    root, tmp_path, write_paper, submit
):
    long_id = "paper-" + "0123456789" * 9
    (root / "p1").rename(root / long_id)
    log_path = submit()
    [run_id] = [run_id for run_id in run_ids(write_paper) if run_id.startswith("paper-")]
    assert len(run_id) == modal_app.RUN_ID_MAX and run_id.startswith("paper-0123456789")
    # The log and the download directory keep the paper's own id.
    assert read_json(log_path)[long_id]["status"] == "success"
    assert (tmp_path / "out" / long_id / "final_paper.pdf").is_file()


def test_fakes_match_the_modal_sdk_calls_the_batch_makes():
    calls = [
        (modal.Function.spawn, (None, {"run_id": "r"}), {}),
        (modal.FunctionCall.from_id, (None, "fc-123"), {}),
        (modal.FunctionCall.get, (None,), {"timeout": 0}),
        (modal.FunctionCall.cancel, (None,), {"terminate_containers": True}),
    ]
    for function, args, kwargs in calls:
        inspect.signature(function).bind(*args, **kwargs)
    assert hasattr(modal.FunctionCall, "object_id")
    # What FunctionCall.get raises: the builtin while the call runs, these when it is over
    # or when Modal cannot be asked.
    for name in (
        "OutputExpiredError",
        "NotFoundError",
        "RemoteError",
        "ConnectionError",
        "AuthError",
        "ServiceError",
        "ResourceExhaustedError",
    ):
        assert issubclass(getattr(modal.exception, name), modal.exception.Error)
    assert not issubclass(modal.exception.OutputExpiredError, TimeoutError)
