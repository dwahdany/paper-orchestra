# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write many papers, each in a worker process of its own.

The batch is resumable at two levels: papers that already succeeded are
skipped (``paper_writing_log.json``), and a paper that failed part-way resumes
from its last completed stage (the pipeline's own ``state.json``).

Every paper has its own process, so one that dies (an out-of-memory kill, a
crash in a native library) costs that paper only, and stopping the batch stops
the papers in flight instead of leaving them to finish on their own.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import shutil
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

from paper_orchestra.config import PipelineConfig
from paper_orchestra.log import LOGGER_NAME
from paper_orchestra.pipeline import prepare_run_dir, run_paper
from paper_orchestra.utils.common import read_json, write_json

log = logging.getLogger(__name__)

LOG_NAME = "paper_writing_log.json"
#: Per-paper log file written by the worker, next to the paper's outputs.
PAPER_LOG_NAME = "paper_writing.log"
#: The ground-truth paper, kept with the raw materials for the autoraters.
ORIGINAL_PAPER_NAME = "original_paper.pdf"
#: How long a stopped batch waits for the papers in flight to close their runs as
#: interrupted before it kills their workers.
STOP_GRACE_SECONDS = 30.0
#: How often a stopped batch repeats its request to the papers that are still in flight.
STOP_REPEAT_SECONDS = 1.0
#: The logged error of a paper that was being written when the batch was stopped.
INTERRUPTED = "interrupted before the paper was written"


def _papers_in_folder(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Batch input folder not found: {root}")
    papers: list[dict[str, Any]] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        content = folder / "raw_materials"
        if not content.is_dir():
            log.warning("Skipping %s: it has no raw_materials directory.", folder)
            continue
        pdf = folder / f"{folder.name}.pdf"
        papers.append(
            {
                "paper_id": folder.name,
                "content_folder": str(content),
                "pdf_path": str(pdf) if pdf.is_file() else None,
            }
        )
    return papers


def _papers_in_list(info: Path) -> list[dict[str, Any]]:
    entries = read_json(info)
    if not isinstance(entries, list):
        raise ValueError(f"{info} must contain a JSON list of papers.")
    papers: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("paper_id"):
            raise ValueError(f"{info}: entry {index} has no paper_id.")
        if not entry.get("content_folder"):
            raise ValueError(f"{info}: paper {entry['paper_id']} has no content_folder.")
        pdf_path = entry.get("pdf_path")
        papers.append(
            {
                "paper_id": str(entry["paper_id"]),
                # Relative paths are relative to the list itself, wherever the batch is run.
                "content_folder": str(info.parent / entry["content_folder"]),
                "pdf_path": str(info.parent / pdf_path) if pdf_path else None,
            }
        )
    return papers


def load_papers(
    *,
    root_input_folder: str | Path | None = None,
    paper_info_path: str | Path | None = None,
    n_papers: int = -1,
) -> list[dict[str, Any]]:
    """The papers of a batch as ``{"paper_id", "content_folder", "pdf_path"}`` entries.

    ``root_input_folder`` holds one directory per paper with a
    ``raw_materials`` folder and, optionally, the original paper as
    ``<paper_id>.pdf``.  ``paper_info_path`` is a JSON list of such entries.
    ``n_papers`` keeps the first ``n`` (all of them when negative).
    """
    if root_input_folder is not None and paper_info_path is None:
        papers = _papers_in_folder(Path(root_input_folder))
    elif paper_info_path is not None and root_input_folder is None:
        papers = _papers_in_list(Path(paper_info_path))
    else:
        raise ValueError("Give exactly one of root_input_folder and paper_info_path.")

    seen: set[str] = set()
    for paper in papers:
        paper_id = paper["paper_id"]
        if Path(paper_id).name != paper_id or paper_id == "..":
            raise ValueError(f"paper_id {paper_id!r} cannot be used as a directory name.")
        if paper_id in seen:
            raise ValueError(f"paper_id {paper_id!r} appears more than once in the batch.")
        seen.add(paper_id)
    return papers if n_papers < 0 else papers[:n_papers]


def summarize(write_log: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Number of papers per status (``success``, ``failed``, ``error``)."""
    counts = {"success": 0, "failed": 0, "error": 0}
    for entry in write_log.values():
        status = entry.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def _load_log(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(
            f"{path} is not valid JSON ({exc}); fix or remove it to continue."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by paper id.")
    return data


def _already_written(entry: dict[str, Any] | None) -> bool:
    return bool(
        entry
        and entry.get("status") == "success"
        and entry.get("paper_path")
        and Path(entry["paper_path"]).is_file()
    )


@contextmanager
def _log_to_file(path: Path) -> Iterator[None]:
    """Send the pipeline's log to ``path`` while one paper is being written."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def _write_paper(job: dict[str, Any]) -> dict[str, Any]:
    """Write one paper (runs in a worker process).  Failures are returned, not raised."""
    paper_id = job["paper_id"]
    try:
        cfg = PipelineConfig.from_dict(job["config"])
        paths = prepare_run_dir(cfg, Path(job["content_folder"]))
        if job.get("pdf_path"):
            shutil.copyfile(job["pdf_path"], paths.materials_dir / ORIGINAL_PAPER_NAME)
        with _log_to_file(cfg.base_dir / PAPER_LOG_NAME):
            try:
                result = run_paper(cfg)
            except Exception:
                log.exception("Paper %s could not be written.", paper_id)
                raise
    except Exception as exc:
        return {
            "paper_id": paper_id,
            "status": "error",
            "paper_path": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if result.success:
        return {"paper_id": paper_id, "status": "success", "paper_path": str(result.pdf_path)}
    return {"paper_id": paper_id, "status": "failed", "paper_path": "", "error": result.error}


def _disregard(signum: int, frame: object) -> None:
    """Signal handler that does nothing."""


def _disregard_stop_signals() -> None:
    # A handler rather than SIG_IGN: an ignored signal stays ignored in every program the
    # worker starts afterwards, a handled one does not.
    signal.signal(signal.SIGINT, _disregard)
    signal.signal(signal.SIGTERM, _disregard)


def _interrupt_once(signum: int, frame: object) -> None:
    # Ctrl-C at a terminal reaches the workers as well as the batch, whose own stop
    # request follows it: a second interrupt would cut short the run's handling of the
    # first one (closing its state as interrupted).
    _disregard_stop_signals()
    raise KeyboardInterrupt


def _worker_main(
    sender: Connection,
    level: int,
    write: Callable[[dict[str, Any]], dict[str, Any]],
    job: dict[str, Any],
) -> None:
    """Body of a worker process: write one paper and send what came of it to the batch."""
    outcome: dict[str, Any] = {"status": "error", "error": INTERRUPTED}
    try:
        try:
            # The batch stops a worker with SIGTERM; taking it as an interrupt lets the run
            # close its state the way it does for Ctrl-C.
            signal.signal(signal.SIGINT, _interrupt_once)
            signal.signal(signal.SIGTERM, _interrupt_once)
            # A spawned process starts with unconfigured logging; without a level the
            # per-paper log file would only receive warnings.
            logging.getLogger(LOGGER_NAME).setLevel(level)
            outcome = write(job)
        except Exception as exc:
            outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            # From here on a stop request must not cut the report short.
            _disregard_stop_signals()
    except KeyboardInterrupt:
        # Raised at most once (see _interrupt_once); the outcome is whatever was reached.
        pass
    # The batch may be gone already; there is then nobody to tell.
    with suppress(OSError):
        sender.send(outcome)
    sender.close()


class _Worker:
    """The process that writes one paper, and the pipe its outcome comes back on."""

    def __init__(self, job: dict[str, Any], level: int) -> None:
        # "spawn" rather than "fork": a forked child inherits the parent's threads' locks and
        # the provider SDKs' connection state in whatever condition they happen to be in.
        context = multiprocessing.get_context("spawn")
        self.paper_id: str = job["paper_id"]
        self._outcome: dict[str, Any] | None = None
        self._receiver: Connection | None
        self._receiver, sender = context.Pipe(duplex=False)
        # Not a daemon: a daemonic process may not start processes of its own.
        self.process = context.Process(
            target=_worker_main,
            args=(sender, level, _write_paper, job),
            name=f"paper-{self.paper_id}",
        )
        self.process.start()
        # The worker now holds the only writing end, so the pipe ends when the worker does.
        sender.close()

    def waitables(self) -> list[Any]:
        """What becomes ready when there is news: the worker exited, or sent its outcome."""
        waitables: list[Any] = [self.process.sentinel]
        if self._receiver is not None:
            waitables.append(self._receiver)
        return waitables

    def read(self) -> None:
        """Take the outcome off the pipe, if it has arrived."""
        if self._receiver is None:
            return
        try:
            if not self._receiver.poll():
                return
            self._outcome = self._receiver.recv()
        except (EOFError, OSError):
            # The pipe ended without an outcome: the exit code will have to tell.
            pass
        self._receiver.close()
        self._receiver = None

    def outcome(self, *, stopped: bool = False) -> dict[str, Any]:
        """What became of the paper, once the worker has exited.

        A worker that exited without a word was stopped by the batch (``stopped``) or
        died; a death is reported with its exit code (negative: the signal that killed it).
        """
        self.read()
        self.process.join()
        if self._outcome is not None:
            return self._outcome
        if stopped:
            return {"status": "error", "error": INTERRUPTED}
        code = self.process.exitcode
        error = f"worker exited with code {code}"
        if code is not None and code < 0:
            with suppress(ValueError):
                error += f" ({signal.Signals(-code).name})"
        return {"status": "error", "error": error}


def _exited(workers: Sequence[_Worker], timeout: float | None = None) -> list[_Worker]:
    """Wait for news from ``workers``, read what they sent, and return those that exited."""
    ready = wait([item for worker in workers for item in worker.waitables()], timeout)
    exited = []
    for worker in workers:
        # Read before the exit is acted on: a large outcome fills the pipe, and its
        # sender cannot exit until it has been taken off.
        worker.read()
        if worker.process.sentinel in ready:
            exited.append(worker)
    return exited


def _ask_to_stop(workers: Sequence[_Worker]) -> None:
    """Ask the papers in flight to stop and give them ``STOP_GRACE_SECONDS`` to do so."""
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    alive = list(workers)
    asked = float("-inf")
    while alive and (now := time.monotonic()) < deadline:
        if now - asked >= STOP_REPEAT_SECONDS:
            # Asked again and again, not once.  A signal interrupts a worker only when its
            # main thread is the one that takes it.  After Ctrl-C at a terminal the worker
            # has that interrupt on its hands when this request arrives, and another of
            # its threads can then take both: the main thread stays in the call it is
            # blocked in, and the run goes on.  A request that arrives later, on its own,
            # reaches the main thread.  The worker acts on the first one it notices and
            # disregards the rest (see _interrupt_once).
            for worker in alive:
                worker.process.terminate()
            asked = now
        gone = _exited(alive, min(deadline, asked + STOP_REPEAT_SECONDS) - now)
        alive = [worker for worker in alive if worker not in gone]


def _kill(workers: Sequence[_Worker]) -> None:
    """End the workers that are still alive, and wait until every worker is gone."""
    for worker in workers:
        if worker.process.is_alive():
            worker.process.kill()
    for worker in workers:
        worker.process.join()


@contextmanager
def _interrupts_held() -> Iterator[None]:
    """Hold back what SIGINT and SIGTERM raise in this thread until the block is left.

    A worker is started inside: an interrupt raised between the start of its process and
    the batch taking note of it would leave a worker that is neither stopped nor recorded,
    and that keeps this process from exiting until its paper is written.
    """
    held: dict[int, Any] = {}
    received: list[int] = []

    def later(signum: int, frame: object) -> None:
        received.append(signum)

    try:
        # Handlers run in the main thread only: in any other, nothing is raised here.
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handler = signal.getsignal(signum)
                # Only a handler written in Python raises anything.  An ignored signal and
                # a default action stay as they are, and with them what the worker inherits.
                if callable(handler):
                    held[signum] = handler
                    signal.signal(signum, later)
        yield
    finally:
        for signum, handler in held.items():
            signal.signal(signum, handler)
        for signum in received:
            signal.raise_signal(signum)


def _run_jobs(
    jobs: Sequence[dict[str, Any]],
    max_workers: int,
    record: Callable[[str, dict[str, Any]], None],
) -> None:
    """Write each job's paper in a process of its own, at most ``max_workers`` at a time.

    ``record`` is called with every paper's outcome as it finishes.  When the batch is
    interrupted (or fails itself), no further paper is started and the ones in flight are
    stopped and recorded before the exception continues.
    """
    level = logging.getLogger(LOGGER_NAME).getEffectiveLevel()
    queued = deque(jobs)
    running: list[_Worker] = []
    try:
        while queued or running:
            while queued and len(running) < max_workers:
                with _interrupts_held():
                    running.append(_Worker(queued.popleft(), level))
            for worker in _exited(running):
                running.remove(worker)
                record(worker.paper_id, worker.outcome())
    except BaseException:
        # Left alone, the workers would go on writing (and spending) after the batch was
        # told to stop, and this process could not exit before they were done.
        try:
            log.warning(
                "Stopping the batch: %d paper(s) not started; %d in flight get up to %.0f s to "
                "close their runs (interrupt again to end them at once).",
                len(queued),
                len(running),
                STOP_GRACE_SECONDS,
            )
            _ask_to_stop(running)
        finally:
            # Reached at once when another interrupt (a second Ctrl-C) arrives meanwhile,
            # wherever that is: whoever asks twice does not want to wait.
            _kill(running)
            try:
                for worker in running:
                    record(worker.paper_id, worker.outcome(stopped=True))
            except Exception:
                log.exception("The batch log could not be updated after the batch was stopped.")
        raise


def run_batch(
    cfg_template: PipelineConfig,
    *,
    root_input_folder: str | Path | None = None,
    paper_info_path: str | Path | None = None,
    output_dir: str | Path,
    n_papers: int = -1,
    max_workers: int = 4,
) -> Path:
    """Write every paper of a batch and return the path of ``paper_writing_log.json``.

    Each paper gets ``<output_dir>/<paper_id>`` as its run directory and a copy
    of ``cfg_template`` pointing there.  The log maps each paper id to
    ``{"status": "success" | "failed" | "error", "paper_path": ...}`` (plus
    ``error`` when it did not succeed) and is rewritten as each paper finishes,
    so an interrupted batch picks up where it stopped.

    An interrupt (Ctrl-C) starts no further paper, stops the papers in flight,
    logs them as interrupted and is raised again.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")
    papers = load_papers(
        root_input_folder=root_input_folder, paper_info_path=paper_info_path, n_papers=n_papers
    )
    if not papers:
        raise ValueError("The batch contains no papers.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / LOG_NAME
    write_log = _load_log(log_path)

    template = cfg_template.to_dict()
    # Every paper keeps its own project memory inside its run directory.
    template["memory"]["project_dir"] = None
    jobs = [
        {**paper, "config": {**template, "base_dir": str(output / paper["paper_id"])}}
        for paper in papers
        if not _already_written(write_log.get(paper["paper_id"]))
    ]
    log.info(
        "Writing %d paper(s) with %d worker(s); %d already written.",
        len(jobs),
        min(max_workers, max(1, len(jobs))),
        len(papers) - len(jobs),
    )

    done = 0

    def record(paper_id: str, outcome: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        entry = {"status": outcome["status"], "paper_path": outcome.get("paper_path", "")}
        if outcome.get("error"):
            entry["error"] = outcome["error"]
        write_log[paper_id] = entry
        write_json(log_path, write_log)
        if entry["status"] == "success":
            log.info("Paper %s written (%d of %d).", paper_id, done, len(jobs))
        else:
            log.error(
                "Paper %s %s (%d of %d): %s",
                paper_id,
                entry["status"],
                done,
                len(jobs),
                entry.get("error", "no details"),
            )

    if jobs:
        _run_jobs(jobs, max_workers, record)
    else:
        write_json(log_path, write_log)

    _log_summary(papers, write_log, log_path)
    return log_path


def _log_summary(
    papers: list[dict[str, Any]], write_log: dict[str, dict[str, Any]], log_path: Path
) -> None:
    counts = summarize({paper["paper_id"]: write_log[paper["paper_id"]] for paper in papers})
    log.info(
        "Batch finished: %d paper(s), %d written, %d failed, %d in error. Results: %s",
        len(papers),
        counts["success"],
        counts["failed"],
        counts["error"],
        log_path,
    )
