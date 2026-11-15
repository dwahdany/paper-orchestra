"""Write a batch of papers with the deployed Modal app.

The local side of ``paper-orchestra modal batch``.  It uploads each paper's
inputs to the app's Volume, starts one ``write_paper`` call per paper and
collects each paper as its call finishes.  The calls run on Modal whether or
not this process is still there, so the batch can be left and picked up again.

It keeps the same ``paper_writing_log.json`` as a local batch
(:mod:`paper_orchestra.batch`), with two more statuses for papers that are not
written (yet):

* ``submitted``: a call is writing the paper.  The entry holds the ``run_id``
  and the ``call_id`` of that call, and a later invocation re-attaches to it
  instead of starting the paper a second time.
* ``cancelled``: the call was stopped with ``cancel=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_orchestra.batch import LOG_NAME, _already_written, _load_log, _log_summary, load_papers
from paper_orchestra.utils.common import write_json

log = logging.getLogger(__name__)

SUBMITTED = "submitted"
CANCELLED = "cancelled"

_DIGEST_CHARS = 8
#: Seconds between two lines that say how many papers are still being written.
_PROGRESS_SECONDS = 300.0


def options_digest(options: Mapping[str, Any], template_dir: Path) -> str:
    """Eight hex digits that stand for the options and the template shared by a batch.

    Part of every run id of the batch: the same batch submitted again finds its
    runs on the Volume and resumes them, and a batch with another model, effort
    or template gets run directories of its own instead of the papers an earlier
    batch wrote.
    """
    from paper_orchestra import modal_app

    # Built the way the container will see them, so that leaving an option out and giving
    # its default are the same batch.
    identity = modal_app.build_job(run_id="digest", **options)
    del identity["run_id"]
    # Whether to resume says how to treat the run, not which run it is.
    del identity["resume"]
    if not options.get("research_cutoff"):
        # The default is the current month: it would move every paper to a new run when
        # the month changes.
        del identity["research_cutoff"]
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for path in modal_app.visible_files(template_dir):
        name = path.relative_to(template_dir).as_posix()
        digest.update(b"\0" + name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:_DIGEST_CHARS]


def batch_run_id(paper_id: str, digest: str) -> str:
    """The run id of a paper in a batch: ``<paper id>-<digest>``.

    A paper id too long for a run id is cut and given a digest of the whole id,
    so that two long ids with the same beginning still get different runs.
    Raises ``modal_app.JobError`` for an id that cannot be part of a run id.
    """
    from paper_orchestra import modal_app

    room = modal_app.RUN_ID_MAX - len(digest) - 1
    name = paper_id
    if len(name) > room:
        whole = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
        name = f"{paper_id[: room - len(whole) - 1]}-{whole}"
    return modal_app.validate_run_id(f"{name}-{digest}")


def _state_unknown(exc: BaseException) -> bool:
    """Whether ``exc`` says that Modal could not be asked, rather than how a call ended."""
    from modal import exception

    return isinstance(
        exc,
        exception.ConnectionError
        | exception.AuthError
        | exception.ServiceError
        | exception.ResourceExhaustedError,
    )


def _listed(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


@dataclass
class _Call:
    """A ``write_paper`` call this batch is waiting for."""

    paper: dict[str, Any]
    run_id: str
    call_id: Any
    call: Any
    #: Taken from the log of an earlier invocation and not seen running yet.  If it turns
    #: out to be gone, the paper is submitted again.
    unconfirmed: bool = False


class _Batch:
    def __init__(
        self,
        options: Mapping[str, Any],
        *,
        template: Path,
        volume: Any,
        write_paper: Any,
        lookup_call: Callable[[str], Any],
        output: Path,
    ) -> None:
        from paper_orchestra import modal_app

        self.app = modal_app
        self.options = options
        self.template = template
        self.volume = volume
        self.write_paper = write_paper
        self.lookup_call = lookup_call
        self.output = output
        self.log_path = output / LOG_NAME
        self.write_log = _load_log(self.log_path)
        #: The calls that are running on Modal as far as this process knows, by paper id.
        self.waiting: dict[str, _Call] = {}
        self._digest: str | None = None
        self._papers_by_run: dict[str, str] = {}

    def save(self) -> None:
        write_json(self.log_path, self.write_log)

    def run_id(self, paper_id: str) -> str:
        if self._digest is None:
            self._digest = options_digest(self.options, self.template)
        return batch_run_id(paper_id, self._digest)

    # -- starting and finding calls ------------------------------------------------

    def submit(self, paper: dict[str, Any]) -> None:
        """Upload a paper's inputs and start its call; a paper that cannot be is an error."""
        paper_id = paper["paper_id"]
        try:
            run_id = self.run_id(paper_id)
            if self._papers_by_run.setdefault(run_id, paper_id) != paper_id:
                raise ValueError(
                    f"run id {run_id} is already the run id of paper {self._papers_by_run[run_id]}"
                )
            job = self.app.build_job(run_id=run_id, **self.options)
            raw = Path(paper["content_folder"])
            for name in (job["idea_filename"], job["experimental_log_filename"]):
                if not (raw / name).is_file():
                    raise FileNotFoundError(f"required file not found: {raw / name}")
            job["raw_materials_dir"], job["template_dir"] = self.app.upload_inputs(
                self.volume, run_id, raw, self.template
            )
        except (FileNotFoundError, ValueError) as exc:
            # A problem with one paper's inputs is that paper's error, not the batch's.
            error = f"{type(exc).__name__}: {exc}"
            self.write_log[paper_id] = {"status": "error", "paper_path": "", "error": error}
            self.save()
            log.error("Paper %s cannot be submitted: %s", paper_id, error)
            return
        call = self.write_paper.spawn(job)
        call_id = getattr(call, "object_id", None)
        # On disk before anything else happens: from here on the paper is being written on
        # Modal, and only this entry says how to find the call again.
        self.write_log[paper_id] = {
            "status": SUBMITTED,
            "paper_path": "",
            "run_id": run_id,
            "call_id": call_id,
        }
        self.save()
        self.waiting[paper_id] = _Call(paper, run_id, call_id, call)
        log.info("Paper %s submitted as run %s (function call %s).", paper_id, run_id, call_id)

    def earlier_call(self, paper: dict[str, Any]) -> _Call | None:
        """The call an earlier invocation started for this paper and did not see finish."""
        entry = self.write_log.get(paper["paper_id"]) or {}
        call_id, run_id = entry.get("call_id"), entry.get("run_id")
        if entry.get("status") != SUBMITTED or not call_id or not isinstance(run_id, str):
            return None
        try:
            call = self.lookup_call(call_id)
        except Exception as exc:
            if _state_unknown(exc):
                raise
            log.warning(
                "Paper %s: the function call %s that was writing it cannot be found (%s: %s).",
                paper["paper_id"],
                call_id,
                type(exc).__name__,
                exc,
            )
            return None
        return _Call(paper, run_id, call_id, call, unconfirmed=True)

    def attach(self, paper: dict[str, Any]) -> bool:
        """Wait for the call that is already writing this paper, if there is one."""
        earlier = self.earlier_call(paper)
        if earlier is None:
            return False
        paper_id = paper["paper_id"]
        self.waiting[paper_id] = earlier
        log.info(
            "Paper %s was submitted earlier as run %s (function call %s); waiting for that "
            "call instead of starting the paper a second time.",
            paper_id,
            earlier.run_id,
            earlier.call_id,
        )
        try:
            same_batch = self.run_id(paper_id) == earlier.run_id
        except ValueError:
            same_batch = False
        if not same_batch:
            log.warning(
                "Paper %s: that call was started with other options or another template. "
                "Those given now do not apply to it; to use them, run this command with "
                "--cancel and then once more.",
                paper_id,
            )
        return True

    # -- collecting results --------------------------------------------------------

    @staticmethod
    def ask(waiting: _Call) -> tuple[str, Any]:
        """How a call is doing: ``running``, ``returned`` (its value) or ``raised`` (its error).

        An error that only says Modal could not be asked is raised from here.
        """
        try:
            return "returned", waiting.call.get(timeout=0)
        except TimeoutError:  # the builtin: FunctionCall.get raises it while the call runs
            return "running", None
        except Exception as exc:
            if _state_unknown(exc):
                raise
            return "raised", exc

    def poll(self, paper_id: str) -> None:
        """Ask once about the call of a paper; when the call is over, record the paper."""
        waiting = self.waiting[paper_id]
        state, value = self.ask(waiting)
        if state == "running":
            waiting.unconfirmed = False
            return
        if state == "returned":
            # Still waited for until its files are here: a download that fails leaves the
            # paper to the next invocation, which finds the same result.
            self.record(waiting, value)
            del self.waiting[paper_id]
            return
        del self.waiting[paper_id]
        if waiting.unconfirmed:
            log.warning(
                "Paper %s: the earlier call for run %s is over and left no result (%s: %s); "
                "submitting the paper again.",
                paper_id,
                waiting.run_id,
                type(value).__name__,
                value,
            )
            self.submit(waiting.paper)
        else:
            log.error("Run %s failed: %r", waiting.run_id, value)
            self.failed(paper_id, f"{type(value).__name__}: {value}")

    def record(self, waiting: _Call, summary: Any) -> None:
        """Download what a finished call wrote and log how its paper fared."""
        paper_id = waiting.paper["paper_id"]
        if not isinstance(summary, Mapping):
            self.failed(paper_id, f"write_paper returned {summary!r} instead of a summary")
            return
        saved = self.app.download_outputs(self.volume, summary, self.output / paper_id)
        if not summary.get("success") or "pdf" not in saved:
            error = summary.get("error") or "final_paper.pdf is missing from the volume"
            self.failed(paper_id, str(error))
            return
        self.write_log[paper_id] = {"status": "success", "paper_path": str(saved["pdf"])}
        self.save()
        log.info("Paper %s written.", paper_id)
        stages = summary.get("stages")
        if (
            isinstance(stages, Mapping)
            and stages
            and all(isinstance(stage, Mapping) and stage.get("reused") for stage in stages.values())
        ):
            log.warning(
                "Paper %s: nothing was written this time. Run %s had already finished on the "
                "volume with the same inputs, options and template, and its paper was "
                "downloaded. Pass --no-resume to write it again.",
                paper_id,
                waiting.run_id,
            )

    def failed(self, paper_id: str, error: str) -> None:
        self.write_log[paper_id] = {"status": "failed", "paper_path": "", "error": error}
        self.save()
        log.error("Paper %s failed: %s", paper_id, error)

    def collect(self, poll_seconds: float) -> None:
        """Poll the calls until all are over, recording each paper as its call finishes."""
        started = reported = time.monotonic()
        while True:
            for paper_id in list(self.waiting):
                self.poll(paper_id)
            if not self.waiting:
                return
            now = time.monotonic()
            if now - reported >= _PROGRESS_SECONDS:
                reported = now
                log.info(
                    "%d paper(s) still being written on Modal after %.0f min.",
                    len(self.waiting),
                    (now - started) / 60,
                )
            time.sleep(poll_seconds)

    def cancel(self, papers: list[dict[str, Any]]) -> int:
        """Stop the calls of this batch that are still running; return how many were."""
        cancelled = 0
        for paper in papers:
            paper_id = paper["paper_id"]
            earlier = self.earlier_call(paper)
            if earlier is None:
                continue
            state, value = self.ask(earlier)
            if state == "returned":
                # It finished in the meantime: there is a paper to collect, nothing to stop.
                self.record(earlier, value)
            elif state == "raised":
                self.failed(paper_id, f"{type(value).__name__}: {value}")
            else:
                earlier.call.cancel(terminate_containers=True)
                self.write_log[paper_id] = {
                    "status": CANCELLED,
                    "paper_path": "",
                    "run_id": earlier.run_id,
                    "call_id": earlier.call_id,
                }
                self.save()
                cancelled += 1
                log.info(
                    "Paper %s: cancelled function call %s (run %s).",
                    paper_id,
                    earlier.call_id,
                    earlier.run_id,
                )
        return cancelled


def run_batch_on_modal(
    options: Mapping[str, Any],
    *,
    template_dir: str | Path,
    volume: Any,
    write_paper: Any,
    lookup_call: Callable[[str], Any],
    root_input_folder: str | Path | None = None,
    paper_info_path: str | Path | None = None,
    output_dir: str | Path,
    n_papers: int = -1,
    cancel: bool = False,
    poll_seconds: float = 30.0,
) -> Path:
    """Write every paper of a batch with the deployed Modal app; same log as :func:`run_batch`.

    ``options`` are the ``modal_app.build_job`` arguments shared by all papers,
    ``volume`` is the app's Volume, ``write_paper`` its deployed ``write_paper``
    Function and ``lookup_call`` turns the id of a function call back into the
    call (``modal.FunctionCall.from_id``).

    Each paper runs under the run id ``<paper_id>-<digest>``, where the digest
    stands for the options and the template (:func:`options_digest`), and its
    PDF, LaTeX source and usage report are downloaded to
    ``<output_dir>/<paper_id>`` as soon as its call has finished.  The calls
    are recorded in the log before this function starts waiting, so stopping
    it (Ctrl-C, a lost connection) stops nothing on Modal: running it again
    with the same arguments skips the papers already downloaded, waits for the
    calls that are still running, collects the ones that finished meanwhile
    (Modal keeps a result for seven days) and starts the papers that failed
    again, which resumes them from their state on the Volume.

    ``cancel=True`` starts nothing: it stops the calls of this batch that are
    still running, records them as ``cancelled`` and collects the papers whose
    calls had already finished.
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
    batch = _Batch(
        options,
        template=template,
        volume=volume,
        write_paper=write_paper,
        lookup_call=lookup_call,
        output=output,
    )
    if cancel:
        cancelled = batch.cancel(papers)
        # Whoever asked reads the log next, also when there was nothing to stop or to record.
        batch.save()
        if cancelled:
            log.info(
                "Cancelled %d paper(s) on Modal; their containers are being stopped. Running "
                "the batch again resumes them: a paper whose container was stopped before it "
                "could give up its run directory starts a few minutes later, once that hold "
                "has expired. Results: %s",
                cancelled,
                batch.log_path,
            )
        else:
            log.info(
                "No paper of this batch is being written on Modal; nothing was cancelled. "
                "Results: %s",
                batch.log_path,
            )
        return batch.log_path

    try:
        for paper in papers:
            if _already_written(batch.write_log.get(paper["paper_id"])):
                continue
            if not batch.attach(paper):
                batch.submit(paper)
        batch.save()
        batch.collect(poll_seconds)
    except BaseException as exc:
        # Whatever stopped this process, the calls it was waiting for are still running.
        batch.save()
        if batch.waiting:
            log.warning(
                "%s. %d paper(s) are still being written on Modal (%s) and are recorded as "
                "'%s' in %s. Run the same command again to wait for them and collect their "
                "papers, or run it with --cancel to stop them.",
                "Interrupted" if isinstance(exc, KeyboardInterrupt) else "Stopped waiting",
                len(batch.waiting),
                _listed(sorted(batch.waiting)),
                SUBMITTED,
                batch.log_path,
            )
        raise

    _log_summary(papers, batch.write_log, batch.log_path)
    return batch.log_path
