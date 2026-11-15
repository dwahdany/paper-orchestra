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

"""Write many papers: one worker process per paper, or one Modal container per paper.

The batch is resumable at two levels: papers that already succeeded are
skipped (``paper_writing_log.json``), and a paper that failed part-way resumes
from its last completed stage (the pipeline's own ``state.json``).
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import shutil
from collections.abc import Iterator, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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


def _init_worker(level: int) -> None:
    # A spawned worker starts with unconfigured logging; without a level the per-paper
    # log file would only receive warnings.
    logging.getLogger(LOGGER_NAME).setLevel(level)


def _process_pool(max_workers: int) -> Executor:
    # "spawn" rather than "fork": a forked child inherits the parent's threads' locks and
    # the provider SDKs' connection state in whatever condition they happen to be in.
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(logging.getLogger(LOGGER_NAME).getEffectiveLevel(),),
    )


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

    if jobs:
        pool = _process_pool(min(max_workers, len(jobs)))
        try:
            futures = {pool.submit(_write_paper, job): job["paper_id"] for job in jobs}
            for done, future in enumerate(as_completed(futures), start=1):
                paper_id = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    # The worker process died, or its result could not be sent back.
                    outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                entry = {
                    "status": outcome["status"],
                    "paper_path": outcome.get("paper_path", ""),
                }
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
        except BaseException:
            # An interrupt has to stop the batch.  Leaving the pool the usual way would
            # start every paper that is still queued and wait for all of them.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
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


def run_batch_on_modal(
    options: Mapping[str, Any],
    *,
    template_dir: str | Path,
    volume: Any,
    write_batch: Any,
    root_input_folder: str | Path | None = None,
    paper_info_path: str | Path | None = None,
    output_dir: str | Path,
    n_papers: int = -1,
) -> Path:
    """Write every paper of a batch with the deployed Modal app; same log as :func:`run_batch`.

    ``options`` are the ``modal_app.build_job`` arguments shared by all papers,
    ``volume`` is the app's Volume and ``write_batch`` its ``write_batch``
    Function.  Each paper runs under the run id ``<paper_id>``; its PDF, LaTeX
    source and usage report are downloaded to ``<output_dir>/<paper_id>``.
    Submitting the batch again skips the papers already downloaded and resumes
    the others from their state on the volume.
    """
    from paper_orchestra import modal_app

    papers = load_papers(
        root_input_folder=root_input_folder, paper_info_path=paper_info_path, n_papers=n_papers
    )
    if not papers:
        raise ValueError("The batch contains no papers.")
    # What is wrong for every paper is reported here, before anything is uploaded.
    modal_app.build_job(run_id="options-check", **options)
    template = Path(template_dir)
    for name in ("template.tex", "guidelines.md"):
        if not (template / name).is_file():
            raise FileNotFoundError(f"The LaTeX template directory {template} has no {name}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / LOG_NAME
    write_log = _load_log(log_path)

    jobs: list[dict[str, Any]] = []
    for paper in papers:
        paper_id = paper["paper_id"]
        if _already_written(write_log.get(paper_id)):
            continue
        try:
            job = modal_app.build_job(run_id=paper_id, **options)
            raw = Path(paper["content_folder"])
            for name in (job["idea_filename"], job["experimental_log_filename"]):
                if not (raw / name).is_file():
                    raise FileNotFoundError(f"required file not found: {raw / name}")
            job["raw_materials_dir"], job["template_dir"] = modal_app.upload_inputs(
                volume, paper_id, raw, template
            )
        except (FileNotFoundError, ValueError) as exc:
            # A problem with one paper's inputs is that paper's error, not the batch's.
            error = f"{type(exc).__name__}: {exc}"
            write_log[paper_id] = {"status": "error", "paper_path": "", "error": error}
            log.error("Paper %s cannot be submitted: %s", paper_id, error)
            continue
        jobs.append(job)
    write_json(log_path, write_log)

    if jobs:
        call = write_batch.spawn(jobs)
        log.info(
            "Submitted %d paper(s) to Modal (function call %s); %d not submitted.",
            len(jobs),
            getattr(call, "object_id", "unknown"),
            len(papers) - len(jobs),
        )
        summaries = modal_app.wait_for_result(call)
        for job, summary in zip(jobs, summaries, strict=True):
            paper_id = job["run_id"]
            saved = modal_app.download_outputs(volume, summary, output / paper_id)
            if summary.get("success") and "pdf" in saved:
                write_log[paper_id] = {"status": "success", "paper_path": str(saved["pdf"])}
                log.info("Paper %s written.", paper_id)
            else:
                error = str(summary.get("error") or "final_paper.pdf is missing from the volume")
                write_log[paper_id] = {"status": "failed", "paper_path": "", "error": error}
                log.error("Paper %s failed: %s", paper_id, error)
            write_json(log_path, write_log)

    _log_summary(papers, write_log, log_path)
    return log_path
