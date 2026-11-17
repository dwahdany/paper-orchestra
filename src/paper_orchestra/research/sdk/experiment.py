"""The agent that carries out an approved experiment (``sdk`` runtime).

``research.experiments.run`` does everything around a run that has nothing to do
with who executes it: the run lock, the check that what runs is what the person
approved, the run directory and its record, the page's states, the person's
cancel, the verdict from the files the run left, the result record.  It asks a
compute backend for one thing, ``run_job``.  :class:`AgentBackend` is that one
thing done by an agent instead of by ``python main.py``: it stands in for the
backend the person approved (and reports that backend's name), gives an
experiment agent a folder with the approved code in it, and collects
``metrics.json`` and ``artifacts/`` from that folder when the agent has finished.
The run record and the ``E<n>-run<k>`` result page are then written by the same
code as for any run, so the analyst and the number check work unchanged.

The agent's folder is ``experiments/E<n>/runs/<k>/work``, not the run directory
itself: the run's record, its log and the snapshot of the approved code lie
beside it, out of the agent's reach.

What the record says of such a run is what was observed and no more.  The agent
may run the code many times, change it, or never run it: the record therefore
names no exit status, calls the plan's command the plan's, and names the code as
the agent left it (``runs/<k>/code-as-run``, with the files it changed) beside
the code that was approved.

The plan's ``timeout_seconds`` bounds the agent's whole work (writing, running,
repairing), not one execution of the code.  The seconds of the run are the
agent's wall time, accounted as for any run.  What the agent's model costs goes
to the model budget, not to the run's ``cost_usd``.  When the time is up and a
valid ``metrics.json`` is in place that no change of the code followed, the run
is recorded with those metrics: the agent was told to keep one in place, and
its results do not depend on whether it had also said that it was done.

Three ways an agent can end are not the experiment's failure, and the run then
goes back to the queue with its approval: a stop request, a spent model budget,
and an agent whose process fails (an overloaded API, a login that ran out),
which is tried once more first.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from paper_orchestra.compute.base import (
    BackendUnusable,
    ComputeError,
    JobLog,
    JobResult,
    JobSpec,
    clear_outputs,
    open_run,
)
from paper_orchestra.compute.local import collect_outputs, copy_source, job_model_env
from paper_orchestra.errors import stops_the_run
from paper_orchestra.research import agents, experiments, prompts
from paper_orchestra.research.sdk.guard import LAUNCHER, Guard, launcher_text
from paper_orchestra.research.sdk.session import AgentSession

__all__ = ["ROLE", "WORK_DIR", "AgentBackend", "AgentUnusable"]

#: The role an experiment agent works as, and writes its proposals as.
ROLE = "experimenter"
#: The agent's folder inside the run directory.
WORK_DIR = "work"
#: Seconds that must be left of a run's time for a second agent to take it up after
#: the first one's process failed.
RETRY_NEEDS_SECONDS = 30.0


class AgentUnusable(BackendUnusable):
    """The agent that was to carry a run out could not work: its process failed, twice.

    Not the experiment's failure.  As with a backend that cannot be used, the run
    goes back to the queue and keeps its approval, and the work stops with the
    reason for the person to act on.
    """


class AgentBackend:
    """Stands in for the compute backend of one run: an agent does the job.

    ``inner`` is the backend the person approved.  What an approval asks a
    backend (whether it can run the plan at all) is still asked of it; only the
    job itself is the agent's.  After the run, ``stopped`` holds the reason a
    stop request or a spent budget ended the agent, for whoever started the run
    to act on (the run itself is then recorded as cancelled and queued again).
    """

    #: Read by ``experiments.run`` for the run's record and its ``run.started`` event:
    #: the run says of itself that an agent carried it out.
    carried_out_by_agent = True

    def __init__(self, ctx: agents.AgentContext, inner: Any, page: Any) -> None:
        self._ctx = ctx
        self._inner = inner
        self._page = page
        self.name = str(getattr(inner, "name", "") or "local")
        self.stopped: BaseException | None = None

    # -- what is still the approved backend's ---------------------------------------

    def supports(self, spec: JobSpec) -> list[str]:
        return list(self._inner.supports(spec))

    def check(self, **options: Any) -> list[str]:
        return list(self._inner.check(**options))

    def describe(self) -> dict[str, Any]:
        return dict(self._inner.describe())

    def end_job(self, record: Any) -> list[str]:
        end = getattr(self._inner, "end_job", None)
        return list(end(record)) if end is not None else []

    def close(self) -> None:
        return None  # the engine closes the backend it made

    # -- the job ---------------------------------------------------------------------

    def _stop_of(self, exc: BaseException) -> BaseException | None:
        """Why the work must stop, when ``exc`` ended the agent because it must: a stop
        request, or the project's guard saying so (the model budget).  The guard is
        asked again here, because it answers with an exception of whoever runs the
        phase; ``None`` when the agent's process simply failed."""
        ctx = self._ctx
        if ctx.stopping():
            return exc
        try:
            ctx.guard()
        except Exception as why:
            if ctx.stopping():
                return why
            return agents.SessionStopped(str(why), reason=why)
        return None

    @staticmethod
    def _results_stand(work: Path, run_dir: Path) -> bool:
        """Whether the agent's folder holds results an agent whose time ran out may be
        taken to have finished: a ``metrics.json`` that reads as metrics and that no
        change of the code followed.  (The code is what was kept as the code of the
        run; a file changed after the metrics were written may not be what produced
        them.)"""
        metrics = work / "metrics.json"
        try:
            experiments.read_metrics(work)
            written = metrics.stat().st_mtime
            kept = run_dir / experiments.CODE_AS_RUN_DIR
            if not kept.is_dir():
                return False
            for path in kept.rglob("*"):
                source = work / path.relative_to(kept)
                if path.is_file() and source.is_file() and source.stat().st_mtime > written:
                    return False
        except (OSError, ValueError):
            return False
        return True

    def _folder(
        self, source_dir: Path, run_dir: Path, outputs: Sequence[str]
    ) -> tuple[Path, list[str]]:
        """A fresh folder for the agent with the approved code in it, and none of the
        outputs: a result is what this run wrote."""
        work = run_dir / WORK_DIR
        if work.exists():
            shutil.rmtree(work)
        notes = copy_source(source_dir, work) if source_dir.is_dir() else []
        work.mkdir(parents=True, exist_ok=True)
        clear_outputs(work, outputs)
        return work, notes

    def run_job(
        self,
        spec: JobSpec,
        *,
        source_dir: Path,
        run_dir: Path,
        on_log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        mask: Sequence[str] = (),
    ) -> JobResult:
        ctx, project = self._ctx, self._ctx.project
        began = time.monotonic()
        log_path = open_run(run_dir, spec.outputs)

        def result(**how: Any) -> JobResult:
            return JobResult(
                returncode=how.pop("returncode", None),
                seconds=time.monotonic() - began,
                log_path=log_path,
                outputs_dir=run_dir,
                backend=self.name,
                **how,
            )

        with JobLog(log_path, on_log, mask=mask) as log:
            try:
                work, notes = self._folder(Path(source_dir), Path(run_dir), spec.outputs)
            except (OSError, ComputeError) as exc:
                return result(error=f"The agent's folder could not be prepared: {exc}")
            models = job_model_env() if spec.network else {}
            token = models.get("ANTHROPIC_IDENTITY_TOKEN_FILE")
            if models:
                launcher = work / LAUNCHER
                launcher.write_text(launcher_text(), encoding="utf-8")
                launcher.chmod(0o755)
            files = sorted(
                str(path.relative_to(work))
                for path in work.rglob("*")
                if path.is_file() and path.name != LAUNCHER
            )
            page = project.page(self._page.path)
            limits = project.settings.limits
            entry = next((word for word in spec.command[1:] if word.endswith(".py")), "main.py")
            command = shlex.join(spec.command)
            log.note(f"An agent carries out {page.id}; it has {round(spec.timeout)} seconds.")
            message = prompts.experiment_message(
                experiment=str(page.id),
                title=page.title,
                plan=page.body,
                files=files,
                entrypoint=entry,
                command=command,
                timeout_seconds=float(spec.timeout),
                network=bool(spec.network),
                models=bool(models),
                launcher=LAUNCHER,
                packages=spec.packages,
                max_experiments=int(limits.max_experiments),
            )
            deadline = began + float(spec.timeout)
            # A command of the agent's shell may take as long as the run may: without
            # this the agent's process moves a command to the background after two
            # minutes, and the agent is left to wait for it with `sleep`.
            command_ms = str(max(1, round(float(spec.timeout) * 1000)))
            for attempt in (1, 2):
                try:
                    session = AgentSession(
                        ctx,
                        ROLE,
                        caps={"experiment": int(limits.max_experiments)},
                        folder=work,
                        guard=Guard(
                            folder=work,
                            project=project.root,
                            sandbox=bool(project.settings.agents.sandbox),
                            token_file=Path(token) if token else None,
                        ),
                        job_models=models,
                        env={
                            "BASH_DEFAULT_TIMEOUT_MS": command_ms,
                            "BASH_MAX_TIMEOUT_MS": command_ms,
                        },
                        should_stop=should_stop,
                        deadline=deadline,
                        on_line=log.note,
                        on_output=lambda text: log.feed(text.encode("utf-8", "replace")),
                        max_turns=None,
                        # One agent for one run: every run of a phase has the same
                        # session id, and a run starts in a fresh folder, so nothing is
                        # carried on.
                        record=f"{ctx.session_id(ROLE)}-{page.id}-run{Path(run_dir).name}",
                        resume=False,
                    )
                    session.send(message if attempt == 1 else prompts.TAKEN_UP + message)
                    break
                except agents.SessionStopped as exc:
                    # A stop request, or the model budget: the run was not carried out,
                    # and goes back to the queue.  Whoever started it is told why.
                    self.stopped = exc
                    log.note("The agent was stopped before the run was finished.")
                    return result(cancelled=True, notes=notes)
                except Exception as exc:
                    stop = self._stop_of(exc)
                    if stop is not None:
                        self.stopped = stop
                        log.note("The agent was stopped before the run was finished.")
                        return result(cancelled=True, notes=notes)
                    # The agent's process failed.  That says nothing about the
                    # experiment: a second agent takes the run up, and when that one
                    # fails too the run goes back to the queue with its approval.
                    why = experiments.mask_credentials(project, f"{type(exc).__name__}: {exc}")
                    left = deadline - time.monotonic()
                    if attempt == 1 and not stops_the_run(exc) and left > RETRY_NEEDS_SECONDS:
                        log.note(
                            f"The agent's process failed ({why}); another agent takes the "
                            "run up once more."
                        )
                        continue
                    log.note(f"The agent could not carry out the run: {why}")
                    raise AgentUnusable(
                        f"The agent that was to carry out {page.id} could not work ({why}). "
                        f"{page.id} keeps its approval and is in the queue again; it is run "
                        "when the work goes on."
                    ) from exc
                finally:
                    ctx.close_sessions()
            # The agent has ended.  What it made of the approved code is kept beside
            # it, for the record to name the code that was there with the results.
            with contextlib.suppress(OSError):
                experiments.keep_code_as_run(
                    work,
                    Path(run_dir),
                    whole=Path(source_dir) == Path(run_dir) / experiments.CODE_DIR,
                    skip=(LAUNCHER,),
                )
            # (Asked before the outputs are moved: of the files as the agent left them.)
            results_stand = session.timed_out and self._results_stand(work, Path(run_dir))
            notes += collect_outputs(work, spec.outputs, Path(run_dir))
            if session.cancelled:
                log.note("The run was cancelled while the agent was at work.")
                return result(cancelled=True, notes=notes)
            if session.timed_out and results_stand:
                said = (
                    f"The agent's time ({round(spec.timeout)} seconds) was up before it had "
                    "said that it was done. metrics.json was in place by then, and no "
                    "file of the code was changed after it was written: the run is recorded "
                    "with those metrics. What the agent would have said about the run is "
                    "missing; its log shows what it did."
                )
                log.note(said)
                return result(returncode=0, notes=[*notes, said])
            if session.timed_out:
                log.note(
                    f"The agent's time ({round(spec.timeout)} seconds) was up before it "
                    "had finished."
                )
                return result(timed_out=True, notes=notes)
            if not os.path.isfile(os.path.join(run_dir, "metrics.json")):
                log.note("The agent finished without leaving metrics.json in its folder.")
            # Zero says to whoever judges the job that the agent ended its work by
            # itself.  It is no exit status of the experiment's code, which nobody
            # measured: the run's record names none (``experiments._finish``).
            return result(returncode=0, notes=notes)
