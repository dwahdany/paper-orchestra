"""Write a batch of papers with the deployed Modal app.

The local side of ``paper-orchestra modal batch``: it uploads each paper's
inputs to the app's Volume, submits the papers, and downloads what they wrote.
It keeps the same ``paper_writing_log.json`` as a local batch
(:mod:`paper_orchestra.batch`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from paper_orchestra.batch import LOG_NAME, _already_written, _load_log, _log_summary, load_papers
from paper_orchestra.utils.common import write_json

log = logging.getLogger(__name__)


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
