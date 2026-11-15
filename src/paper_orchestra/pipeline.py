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

"""The paper-writing pipeline: one code path, with or without plotting.

Stages run in this order::

    ingest -> outline -> literature (+ plotting, in parallel) -> section_writing
           -> refinement -> finalize

Each stage is retried on its own, its outcome is persisted in ``state.json``
after every transition, and a later invocation in the same directory resumes
after the last stage that completed (as long as the inputs did not change).
Agents are created through the module-level factory functions below, which
import lazily, so tests can swap in stubs and importing this module stays cheap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paper_orchestra.config import PipelineConfig
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import LLMConfigError, Usage, UsageTracker
from paper_orchestra.utils.common import read_json, read_text, write_json

if TYPE_CHECKING:
    from paper_orchestra.compute.base import ComputeBackend
    from paper_orchestra.memory.harness import ContextHarness

log = logging.getLogger(__name__)

#: Stage names in execution order (``plotting`` only runs with ``cfg.use_plotting``).
STAGES = (
    "ingest",
    "outline",
    "literature",
    "plotting",
    "section_writing",
    "refinement",
    "finalize",
)
STATE_VERSION = 1

#: The outline as the outline stage wrote it.  The literature stage overwrites
#: ``outline.json`` with citation candidates, so a retried or resumed literature
#: or plotting stage reads this copy instead of its own earlier output.
PRISTINE_OUTLINE_NAME = "outline_v0.json"
MEMORY_REPORT_NAME = "memory_report.json"

FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf")

LITERATURE_SYNTHESIS_FOCUS = (
    "The literature stage just finished collecting and verifying papers (one page each under "
    "wiki/literature/). Build a short synthesis of that literature under wiki/synthesis/: the "
    "main themes, how each cluster of papers relates to our method, and the gaps our work "
    "addresses. Write one page per theme that links the paper pages it draws on, and quote "
    "facts from those pages instead of restating them from memory."
)

_USAGE_FIELDS = tuple(Usage().to_dict())
#: A compile log in an exception message should not swamp ``state.json`` or a memory note.
_MAX_ERROR_CHARS = 2_000


class StageError(RuntimeError):
    """A stage finished without producing the files the next stage needs."""


@dataclass
class PaperResult:
    success: bool
    pdf_path: Path | None
    base_dir: Path
    #: Per stage: ``status``, ``seconds``, ``attempts``, ``error`` (plus ``outputs``, ``reused``).
    stages: dict[str, dict[str, Any]]
    usage: dict[str, Any]
    error: str | None = None


# ---------------------------------------------------------------------------
# Factories (replaced by tests; keep every agent import lazy)
# ---------------------------------------------------------------------------


def _outline_agent(ctx: RunContext) -> Any:
    from paper_orchestra.agents.outline import OutlineAgent

    return OutlineAgent(ctx)


def _literature_agent(ctx: RunContext) -> Any:
    from paper_orchestra.agents.literature import LiteratureAgent

    return LiteratureAgent(ctx)


def _plotting_agent(ctx: RunContext) -> Any:
    from paper_orchestra.agents.plotting import PlottingAgent

    return PlottingAgent(ctx)


def _section_writing_agent(ctx: RunContext) -> Any:
    from paper_orchestra.agents.section_writing import SectionWritingAgent

    return SectionWritingAgent(ctx)


def _refinement_agent(ctx: RunContext) -> Any:
    from paper_orchestra.agents.refinement import ContentRefinementAgent

    return ContentRefinementAgent(ctx)


def _build_harness(cfg: PipelineConfig) -> ContextHarness:
    from paper_orchestra.memory.harness import ContextHarness

    return ContextHarness.create(cfg)


def _build_compute(cfg: PipelineConfig) -> ComputeBackend:
    from paper_orchestra.compute import get_backend

    return get_backend(cfg.compute)


# ---------------------------------------------------------------------------
# Run directory and input validation
# ---------------------------------------------------------------------------


def prepare_run_dir(cfg: PipelineConfig, raw_materials_dir: Path | None) -> RunPaths:
    """Create the run directory, bring the raw materials in, and validate the inputs.

    ``raw_materials_dir`` is copied to ``<base_dir>/raw_materials`` (replacing
    what was there) unless it already is that directory; ``None`` means the
    materials are expected to be in place.  Raises ``FileNotFoundError`` or
    ``ValueError`` with a message that says what to fix.
    """
    paths = RunPaths(cfg.base_dir, cfg.idea_filename, cfg.experimental_log_filename)
    _validate_settings(cfg)
    _validate_template(cfg.template_dir)

    source = paths.materials_dir if raw_materials_dir is None else Path(raw_materials_dir)
    if not source.is_dir():
        raise FileNotFoundError(
            f"Raw materials directory not found: {source}. It must hold the idea file "
            f"({cfg.idea_filename}) and the experimental log ({cfg.experimental_log_filename})."
        )
    for label, name in (
        ("idea", cfg.idea_filename),
        ("experimental log", cfg.experimental_log_filename),
    ):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"The {label} file was not found: {path}")
        if not path.read_bytes().strip():
            raise ValueError(f"The {label} file is empty: {path}")

    copy = _needs_copy(source, paths.materials_dir)
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    if copy:
        if paths.materials_dir.exists():
            shutil.rmtree(paths.materials_dir)
        shutil.copytree(source, paths.materials_dir)
    return paths


def _validate_settings(cfg: PipelineConfig) -> None:
    if cfg.max_stage_attempts < 1:
        raise ValueError(f"max_stage_attempts must be at least 1, got {cfg.max_stage_attempts}")
    for label, name in (
        ("idea_filename", cfg.idea_filename),
        ("experimental_log_filename", cfg.experimental_log_filename),
    ):
        if not name or Path(name).name != name:
            raise ValueError(
                f"{label} must be a file name inside the raw materials directory, got {name!r}"
            )
    from paper_orchestra.utils.scholar import parse_cutoff

    parse_cutoff(cfg.research_cutoff)


def _validate_template(template_dir: Path) -> None:
    if not template_dir.is_dir():
        raise FileNotFoundError(f"LaTeX template directory not found: {template_dir}")
    for name, purpose in (
        ("template.tex", "the LaTeX skeleton the paper is written into"),
        ("guidelines.md", "the venue's writing and formatting guidelines"),
    ):
        if not (template_dir / name).is_file():
            raise FileNotFoundError(
                f"The LaTeX template directory {template_dir} has no {name} ({purpose})."
            )


def _needs_copy(source: Path, target: Path) -> bool:
    """Whether ``source`` has to be copied to ``target``; rejects locations that would clash."""
    src, dst = source.resolve(), target.resolve()
    if src == dst:
        return False
    if dst.is_relative_to(src):
        raise ValueError(
            f"The output directory ({target.parent}) is inside the raw materials directory "
            f"({source}); choose an output directory outside it."
        )
    if src.is_relative_to(dst):
        raise ValueError(
            f"The raw materials directory ({source}) is inside the run's own copy of the raw "
            f"materials ({target}), which is replaced on every run; pass the original directory."
        )
    return True


def _preflight(cfg: PipelineConfig, harness: ContextHarness, compute: ComputeBackend) -> str | None:
    """Why the run cannot succeed as configured, found before any model call is paid for."""
    from paper_orchestra.compute.local import LocalBackend

    roles = {
        "writer": cfg.models.writer,
        "reflection": cfg.models.reflection,
        "fast": cfg.models.fast,
    }
    if cfg.use_plotting:
        roles["plotting"] = cfg.models.plotting
    checked: set[str] = set()
    for role, model in roles.items():
        if model in checked:
            continue
        checked.add(model)
        try:
            harness.client_for(model)
        except LLMConfigError as exc:
            return f"the {role} model '{model}' cannot be used: {exc}"
    image = cfg.models.image
    if cfg.use_plotting and image and image not in checked:
        try:
            harness.client_for(image)
        except LLMConfigError as exc:
            # Diagrams fall back to model-written plotting code, so this is not fatal.
            log.warning("The image model '%s' cannot be used: %s", image, exc)
    if isinstance(compute, LocalBackend) and not compute.latex_available():
        return (
            "pdflatex/bibtex were not found on PATH, so the paper cannot be compiled. "
            "Install TeX Live, or run with --compute modal to compile in a Modal sandbox."
        )
    return None


# ---------------------------------------------------------------------------
# Small file helpers
# ---------------------------------------------------------------------------


def _atomic_copy(source: Path, target: Path) -> None:
    """Copy ``source`` over ``target`` without a reader ever seeing a partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _describe(exc: BaseException) -> str:
    """``Type: message`` for state, notes and results; the log has the full traceback."""
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= _MAX_ERROR_CHARS:
        return text
    return text[:_MAX_ERROR_CHARS] + " ... (cut here; the log has the full error)"


def _fingerprint(cfg: PipelineConfig, paths: RunPaths) -> str:
    """Digest of everything the authors supplied: idea, log, figures and template."""
    digest = hashlib.sha256()

    def add(label: str, path: Path) -> None:
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())

    add("idea", paths.idea)
    add("experimental_log", paths.experimental_log)
    for label, root in (("figures", paths.provided_figures_dir), ("template", cfg.template_dir)):
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            add(f"{label}/{path.relative_to(root).as_posix()}", path)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Figures supplied with the raw materials
# ---------------------------------------------------------------------------


def _read_figure_list(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Entries of a figures ``info.json`` plus a problem description, if it is unusable."""
    if not path.is_file():
        return [], None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], f"{path} is not valid JSON ({exc})"
    if not isinstance(data, list):
        return [], f"{path} must contain a JSON list of {{'name', 'caption'}} objects"
    return [entry for entry in data if isinstance(entry, dict) and entry.get("name")], None


def _resolve_figure_name(name: str, files: Sequence[str]) -> str | None:
    """Match a listed figure name to a file; names may omit the extension."""
    if name in files:
        return name
    if Path(name).suffix.lower() in FIGURE_SUFFIXES:
        return None
    for suffix in FIGURE_SUFFIXES:
        for candidate in files:
            if Path(candidate).stem == name and Path(candidate).suffix.lower() == suffix:
                return candidate
    return None


def _provided_figures(paths: RunPaths) -> tuple[list[dict[str, Any]], list[str]]:
    """Figure entries (``name``, ``caption``) for ``raw_materials/figures`` and what is off.

    Listed figures keep their order and any extra keys.  Image files that
    ``info.json`` does not mention are appended without a caption, so a figure
    is never silently dropped because its description is missing.
    """
    source = paths.provided_figures_dir
    if not source.is_dir():
        return [], []
    files = sorted(
        p.name for p in source.iterdir() if p.is_file() and p.suffix.lower() in FIGURE_SUFFIXES
    )
    listed, problem = _read_figure_list(source / "info.json")
    problems = [problem] if problem else []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in listed:
        name = _resolve_figure_name(str(raw["name"]), files)
        if name is None:
            problems.append(
                f"figure '{raw['name']}' is listed in info.json but there is no such file in "
                f"{source}; it is ignored"
            )
            continue
        if name in seen:
            continue
        seen.add(name)
        entries.append({**raw, "name": name, "caption": str(raw.get("caption") or "")})
    uncaptioned = [name for name in files if name not in seen]
    entries.extend({"name": name, "caption": ""} for name in uncaptioned)
    if uncaptioned:
        problems.append(
            f"no caption is given for {', '.join(uncaptioned)}; add them to "
            f"{source / 'info.json'} as [{{'name': ..., 'caption': ...}}] so the writer knows "
            "what they show"
        )
    return entries, problems


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def _usage_of(row: dict[str, Any]) -> Usage:
    return Usage(**{name: int(row.get(name) or 0) for name in _USAGE_FIELDS})


def _carried_usage(path: Path) -> list[dict[str, Any]]:
    """Usage rows recorded by the earlier invocations of a resumed run."""
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["by_label"]
        return [
            {"label": str(row["label"]), "model": str(row["model"]), **_usage_of(row).to_dict()}
            for row in rows
        ]
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        log.warning("Ignoring unreadable %s (%r); usage restarts from zero.", path, exc)
        return []


def _usage_report(tracker: UsageTracker, carried: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The run's usage ledger with an estimated cost, including earlier invocations."""
    merged = UsageTracker()
    for row in (*carried, *tracker.to_dict()["by_label"]):
        merged.record(row["label"], row["model"], _usage_of(row))
    report = merged.to_dict()
    try:
        from paper_orchestra.llm.pricing import estimate_cost_usd
    except ImportError:
        return report
    costs = [
        (row["model"], estimate_cost_usd(row["model"], _usage_of(row)))
        for row in report["by_label"]
    ]
    priced = [cost for _, cost in costs if cost is not None]
    report["estimated_cost_usd"] = round(sum(priced), 4) if priced or not costs else None
    # Models missing from the price table: their tokens are not part of the estimate.
    report["unpriced_models"] = sorted({model for model, cost in costs if cost is None})
    return report


# ---------------------------------------------------------------------------
# state.json
# ---------------------------------------------------------------------------


def _blank_record() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "seconds": 0.0,
        "error": None,
        "outputs": [],
        "reused": False,
    }


class _State:
    """``state.json``: what every stage did, rewritten atomically on each transition."""

    def __init__(
        self, path: Path, header: dict[str, Any], stages: dict[str, dict[str, Any]]
    ) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {**header, "status": "running", "error": None}
        self._data["stages"] = stages
        with self._lock:
            self._save()

    def _save(self) -> None:
        self._data["updated_at"] = _now()
        write_json(self._path, self._data)

    def update(self, stage: str, **fields: Any) -> None:
        with self._lock:
            self._data["stages"][stage].update(fields)
            self._save()

    def record(self, stage: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._data["stages"][stage])

    def stages(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: dict(record) for name, record in self._data["stages"].items()}

    def close(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._data["status"] = status
            self._data["error"] = error
            if status == "interrupted":
                for record in self._data["stages"].values():
                    if record["status"] == "running":
                        record["status"] = "interrupted"
            self._save()


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Stage:
    name: str
    #: Does the work and returns the files the stage produced.
    run: Callable[[], Sequence[Path]]
    #: A failure is tolerated: the run continues without this stage's output.
    optional: bool = False
    #: Runs once after the stage completed (not when its result is reused).
    after: Callable[[], None] | None = None


class _Runner:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        middle = [_Stage("literature", self._literature, after=self._synthesize_literature)]
        if ctx.cfg.use_plotting:
            middle.append(_Stage("plotting", self._plotting, optional=True))
        self.groups: tuple[tuple[_Stage, ...], ...] = (
            (_Stage("ingest", self._ingest),),
            (_Stage("outline", self._outline),),
            tuple(middle),
            (_Stage("section_writing", self._section_writing),),
            (_Stage("refinement", self._refinement),),
            (_Stage("finalize", self._finalize),),
        )
        self.pristine_outline = ctx.paths.base_dir / PRISTINE_OUTLINE_NAME
        self.memory_report = ctx.paths.base_dir / MEMORY_REPORT_NAME
        self.reuse: set[str] = set()
        self.carried_usage: list[dict[str, Any]] = []
        self.carried_seconds = 0.0
        self.invocation_start = 0.0
        self.start_time = 0.0
        self.state: _State

    # -- orchestration ---------------------------------------------------------

    def run(self) -> PaperResult:
        ctx = self.ctx
        cfg, paths, harness = ctx.cfg, ctx.paths, ctx.harness
        self.invocation_start = time.time()
        fingerprint = _fingerprint(cfg, paths)
        previous = self._previous_state(fingerprint)
        self.reuse = self._plan_reuse(previous)
        names = [stage.name for group in self.groups for stage in group]
        to_run = [name for name in names if name not in self.reuse]

        failure: str | None = None
        if to_run:
            failure = _preflight(cfg, harness, ctx.compute)
            # A deliverable left by an earlier run must not pass for this one's.
            for stale in (paths.final_pdf, paths.final_tex, paths.time_breakdown):
                stale.unlink(missing_ok=True)
            self.memory_report.unlink(missing_ok=True)

        stages = {name: _blank_record() for name in names}
        self.start_time = self.invocation_start
        if previous is not None:
            self.carried_usage = _carried_usage(paths.usage)
            for name in self.reuse:
                stages[name] = {**_blank_record(), **previous["stages"][name], "reused": True}
            self.carried_seconds = sum(float(stages[n]["seconds"] or 0.0) for n in self.reuse)
            if self.reuse:
                self.start_time = float(previous.get("start_time") or self.invocation_start)
                log.info(
                    "Resuming the run in %s: reusing %s; %s.",
                    paths.base_dir,
                    ", ".join(n for n in names if n in self.reuse),
                    f"running {', '.join(to_run)}" if to_run else "nothing is left to run",
                )
        self.state = _State(
            paths.state,
            {
                "version": STATE_VERSION,
                "start_time": self.start_time,
                "inputs": fingerprint,
                "config": cfg.to_dict(),
            },
            stages,
        )
        harness.record(
            "run_start",
            models=cfg.to_dict()["models"],
            use_plotting=cfg.use_plotting,
            reused=sorted(self.reuse),
        )

        if failure is None:
            try:
                failure = self._run_groups()
            except BaseException:
                self.state.close("interrupted", "the run was interrupted")
                self._guarded(self._write_usage, "usage report")
                raise

        self.state.close("failed" if failure else "completed", failure)
        harness.record("run_end", status="failed" if failure else "completed", error=failure)
        usage = self._write_usage()
        if failure:
            log.error("The run failed: %s", failure)
        else:
            log.info("The paper is at %s.", paths.final_pdf)
        return PaperResult(
            success=failure is None,
            pdf_path=None if failure else paths.final_pdf,
            base_dir=paths.base_dir,
            stages=self.state.stages(),
            usage=usage,
            error=failure,
        )

    def _previous_state(self, fingerprint: str) -> dict[str, Any] | None:
        """The earlier run's state, when this invocation may build on it."""
        cfg, paths = self.ctx.cfg, self.ctx.paths
        if not paths.state.is_file():
            return None
        if not cfg.resume:
            log.info("Resume is off: every stage runs again.")
            return None
        try:
            previous = json.loads(paths.state.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable %s (%s): every stage runs again.", paths.state, exc)
            return None
        if (
            not isinstance(previous, dict)
            or previous.get("version") != STATE_VERSION
            or not isinstance(previous.get("stages"), dict)
        ):
            log.warning("Ignoring %s (unknown format): every stage runs again.", paths.state)
            return None
        if previous.get("inputs") != fingerprint:
            log.warning(
                "The idea, experimental log, figures or template changed since the previous run "
                "in %s, so every stage runs again. The project memory there is kept; use a new "
                "output directory for an unrelated paper.",
                paths.base_dir,
            )
            return None
        return previous

    def _plan_reuse(self, previous: dict[str, Any] | None) -> set[str]:
        """Stages whose earlier result stands.

        A stage is reused when it completed, its recorded outputs still exist
        and every stage before it is reused too.  An optional stage whose
        failure the earlier run accepted is settled as well: resuming continues
        a run, it does not reopen decisions the run already made.
        """
        reuse: set[str] = set()
        if previous is None:
            return reuse
        base = self.ctx.paths.base_dir
        for group in self.groups:
            done = []
            for stage in group:
                record = previous["stages"].get(stage.name)
                if not isinstance(record, dict):
                    continue
                outputs = record.get("outputs") or []
                if record.get("status") == "completed":
                    if outputs and all((base / rel).is_file() for rel in outputs):
                        done.append(stage.name)
                elif stage.optional and record.get("tolerated"):
                    done.append(stage.name)
            reuse.update(done)
            if len(done) < len(group):
                break
        return reuse

    def _run_groups(self) -> str | None:
        """Run every stage; return a description of the failure that stopped the run, if any."""
        for group in self.groups:
            if len(group) == 1:
                outcomes = {group[0].name: self._run_stage(group[0])}
            else:
                with ThreadPoolExecutor(len(group), thread_name_prefix="stage") as pool:
                    futures = {s.name: pool.submit(self._run_stage, s) for s in group}
                    # Every result is awaited, so a failing stage never leaves its
                    # sibling running behind the pipeline's back.
                    outcomes = {name: future.result() for name, future in futures.items()}
            for stage in group:
                if outcomes[stage.name]:
                    continue
                record = self.state.record(stage.name)
                if stage.optional:
                    self.state.update(stage.name, tolerated=True)
                    log.warning(
                        "Stage %s failed after %d attempt(s); continuing without its output. "
                        "Last error: %s",
                        stage.name,
                        record["attempts"],
                        record["error"],
                    )
                    continue
                return (
                    f"stage '{stage.name}' failed after {record['attempts']} attempt(s): "
                    f"{record['error']}"
                )
        return None

    def _run_stage(self, stage: _Stage) -> bool:
        name, harness = stage.name, self.ctx.harness
        if name in self.reuse:
            log.info("Stage %s: reusing the result of the earlier run.", name)
            return True

        attempts = self.ctx.cfg.max_stage_attempts
        base = self.ctx.paths.base_dir
        seconds = 0.0
        for attempt in range(1, attempts + 1):
            self.state.update(name, status="running", attempts=attempt, error=None)
            harness.record("stage_start", stage=name, attempt=attempt)
            log.info("Stage %s started (attempt %d of %d).", name, attempt, attempts)
            started = time.monotonic()
            try:
                outputs = [Path(path) for path in stage.run()]
                missing = [str(path) for path in outputs if not path.is_file()]
                if missing:
                    raise StageError(
                        f"the {name} stage finished without producing {', '.join(missing)}"
                    )
            except Exception as exc:
                seconds += time.monotonic() - started
                error = _describe(exc)
                # A configuration problem (credentials, SDK, unsupported option) is the
                # same on every attempt, so it is not retried.
                retry = attempt < attempts and not isinstance(exc, LLMConfigError)
                log.error(
                    "Stage %s failed (attempt %d of %d): %s",
                    name,
                    attempt,
                    attempts,
                    error,
                    exc_info=True,
                )
                self.state.update(name, status="failed", seconds=round(seconds, 3), error=error)
                harness.record(
                    "stage_end",
                    stage=name,
                    status="failed",
                    attempt=attempt,
                    seconds=round(seconds, 3),
                    error=error,
                )
                self._note_failure(stage, attempt, attempts, exc, retry)
                if not retry:
                    break
                continue

            seconds += time.monotonic() - started
            self.state.update(
                name,
                status="completed",
                seconds=round(seconds, 3),
                error=None,
                outputs=[_relative(path, base) for path in outputs],
            )
            harness.record(
                "stage_end",
                stage=name,
                status="completed",
                attempt=attempt,
                seconds=round(seconds, 3),
            )
            log.info("Stage %s completed in %.1f s.", name, seconds)
            if stage.after is not None:
                self._guarded(stage.after, f"consolidation after the {name} stage")
            self._settle(name)
            return True

        self._settle(name)
        return False

    def _note_failure(
        self, stage: _Stage, attempt: int, attempts: int, exc: Exception, retry: bool
    ) -> None:
        """Leave the failure in memory so the retry and later stages can see what went wrong."""
        frames = traceback.extract_tb(exc.__traceback__)
        where = (
            f" It was raised in {frames[-1].name} ({Path(frames[-1].filename).name})."
            if frames
            else ""
        )
        if retry:
            outcome = "The stage is being retried from its start."
        elif stage.optional:
            outcome = "The run continues without this stage's output."
        else:
            outcome = "The run stopped here; a resumed run starts again from this stage."
        text = (
            f"The {stage.name} stage failed on attempt {attempt} of {attempts} with "
            f"{_describe(exc)}{where} {outcome}"
        )
        self._guarded(
            lambda: self.ctx.harness.note(
                text,
                stage=stage.name,
                kind="lesson",
                title=f"{stage.name} stage failed (attempt {attempt})",
            ),
            f"note about the {stage.name} failure",
        )

    def _guarded(self, action: Callable[[], Any], what: str) -> None:
        """Memory upkeep and bookkeeping must never take the run down with them."""
        try:
            action()
        except Exception as exc:
            log.warning("The %s failed: %s", what, exc, exc_info=True)

    def _settle(self, name: str) -> None:
        """After a stage ran: refresh the memory index and persist what the run has spent."""
        self._guarded(self.ctx.harness.refresh, f"memory refresh after the {name} stage")
        self._guarded(self._write_usage, "usage report")

    def _write_usage(self) -> dict[str, Any]:
        report = _usage_report(self.ctx.harness.usage, self.carried_usage)
        write_json(self.ctx.paths.usage, report)
        return report

    # -- stages ----------------------------------------------------------------

    def _ingest(self) -> list[Path]:
        cfg, paths = self.ctx.cfg, self.ctx.paths
        # Start from the template as shipped: drafts and figures of an earlier run must not
        # leak into this one.
        if paths.latex_dir.exists():
            shutil.rmtree(paths.latex_dir)
        shutil.copytree(cfg.template_dir, paths.latex_dir)
        paths.figures_dir.mkdir(exist_ok=True)
        if paths.provided_figures_dir.is_dir():
            shutil.copytree(paths.provided_figures_dir, paths.figures_dir, dirs_exist_ok=True)

        figures, problems = _provided_figures(paths)
        for problem in problems:
            log.warning("Provided figures: %s.", problem)
        write_json(paths.figures_info, figures)
        if not figures:
            if cfg.use_plotting:
                log.info(
                    "No figures were provided in %s; the plotting stage generates them.",
                    paths.provided_figures_dir,
                )
            else:
                log.warning(
                    "No figures (%s) were found in %s and plotting is off, so the paper will "
                    "have no figures. Put the figures there, with an info.json listing each "
                    "one's name and caption, or enable plotting (--plotting).",
                    ", ".join(FIGURE_SUFFIXES),
                    paths.provided_figures_dir,
                )
        self._write_source_pages(figures)
        return [paths.template_tex, paths.guidelines, paths.figures_info]

    def _write_source_pages(self, figures: Sequence[dict[str, Any]]) -> None:
        """Mirror the inputs into the wiki as read-only ground truth."""
        cfg, paths, harness = self.ctx.cfg, self.ctx.paths, self.ctx.harness
        sources = (
            ("idea", "Idea", "The research idea, verbatim as the authors provided it.", paths.idea),
            (
                "experimental_log",
                "Experimental log",
                "Experimental setup, raw results and observations, verbatim as provided.",
                paths.experimental_log,
            ),
            (
                "guidelines",
                "Venue guidelines",
                "The venue's writing and formatting guidelines shipped with the template.",
                paths.guidelines,
            ),
            (
                "template",
                "LaTeX template",
                "The venue's LaTeX template as shipped, before any section was written.",
                cfg.template_dir / "template.tex",
            ),
        )
        for name, title, summary, path in sources:
            text = read_text(path)
            harness.write_page(
                f"wiki/sources/{name}.md",
                _fenced(text, "latex") if path.suffix == ".tex" else text,
                {
                    "title": title,
                    "summary": summary,
                    "kind": "source",
                    "source_file": _relative(path, paths.base_dir),
                },
            )

        from paper_orchestra.memory.wiki import slugify

        taken: set[str] = set()
        for entry in figures:
            name = str(entry["name"])
            figure_id = slugify(Path(name).stem)
            if figure_id in taken:
                figure_id = slugify(name)
            taken.add(figure_id)
            caption = " ".join(str(entry.get("caption") or "").split())
            harness.write_page(
                f"wiki/figures/{figure_id}.md",
                "\n".join(
                    [
                        f"# Figure {name}",
                        "",
                        f"- File: `figures/{name}` in the LaTeX workspace.",
                        "- Origin: provided by the authors with the raw materials, not generated.",
                        "",
                        "## Caption",
                        "",
                        caption or "No caption was provided.",
                    ]
                ),
                {
                    "title": f"Figure {name}",
                    "summary": caption[:200] or f"Provided figure {name} (no caption given).",
                    "kind": "figure",
                    "file": name,
                    "origin": "provided",
                },
            )

    def _outline(self) -> list[Path]:
        cfg, paths = self.ctx.cfg, self.ctx.paths
        for stale in (paths.outline, self.pristine_outline):
            stale.unlink(missing_ok=True)
        # The outline is planned against the template as shipped, also when an earlier run
        # already filled sections in.
        _atomic_copy(cfg.template_dir / "template.tex", paths.template_tex)
        _outline_agent(self.ctx).run()
        if not paths.outline.is_file():
            raise StageError(f"the outline agent did not write {paths.outline}")
        _atomic_copy(paths.outline, self.pristine_outline)
        return [paths.outline, self.pristine_outline]

    def _outline_for_agents(self) -> dict[str, Any]:
        outline = read_json(self.pristine_outline)
        if not isinstance(outline, dict):
            raise StageError(f"{self.pristine_outline} does not hold an outline object")
        return outline

    def _literature(self) -> list[Path]:
        cfg, paths = self.ctx.cfg, self.ctx.paths
        # A retried or resumed stage must see the inputs as they were before its own earlier
        # attempt filled in the template and injected citations into the outline.
        _reset_dir(paths.literature_dir)
        _atomic_copy(cfg.template_dir / "template.tex", paths.template_tex)
        _atomic_copy(self.pristine_outline, paths.outline)
        _literature_agent(self.ctx).run(self._outline_for_agents())

        in_place = {
            paths.literature_dir / "outline_v1.json": paths.outline,
            paths.literature_dir / "updated_template.tex": paths.template_tex,
            paths.literature_dir / "references.bib": paths.references_bib,
        }
        produced = [*in_place, paths.citation_map]
        missing = [str(path) for path in produced if not path.is_file()]
        if missing:
            raise StageError(f"the literature agent did not write {', '.join(missing)}")
        # The agent puts these in the workspace itself; copying them again keeps the handover
        # to the section writer independent of that.
        for source, target in in_place.items():
            _atomic_copy(source, target)
        return [*produced, *in_place.values()]

    def _synthesize_literature(self) -> None:
        report = self.ctx.harness.dream(
            stage="literature", focus=LITERATURE_SYNTHESIS_FOCUS, build=True
        )
        if report is not None and report.error:
            log.warning("The literature synthesis did not complete: %s", report.error)

    def _plotting(self) -> list[Path]:
        paths = self.ctx.paths
        self._reset_figure_workspace()
        _plotting_agent(self.ctx).run(self._outline_for_agents())
        return [paths.plotting_results, paths.figures_info]

    def _reset_figure_workspace(self) -> None:
        """Undo what an earlier plotting attempt left: only the provided figures remain listed."""
        paths = self.ctx.paths
        provided, _ = _provided_figures(paths)
        keep = {entry["name"] for entry in provided}
        if paths.generated_figures_dir.is_dir():
            for stale in paths.generated_figures_dir.iterdir():
                if stale.name not in keep:
                    (paths.figures_dir / stale.name).unlink(missing_ok=True)
        _reset_dir(paths.generated_figures_dir)
        paths.plotting_results.unlink(missing_ok=True)
        write_json(paths.figures_info, provided)

    def _section_writing(self) -> list[Path]:
        paths = self.ctx.paths
        paths.raw_draft_tex.unlink(missing_ok=True)
        _section_writing_agent(self.ctx).run()
        return [paths.raw_draft_tex]

    def _refinement(self) -> list[Path]:
        pdf = Path(_refinement_agent(self.ctx).run(self.ctx.paths.raw_draft_tex))
        return [pdf, pdf.with_suffix(".tex")]

    def _finalize(self) -> list[Path]:
        paths, harness = self.ctx.paths, self.ctx.harness
        refined = [paths.base_dir / rel for rel in self.state.record("refinement")["outputs"]]
        pdf = next((path for path in refined if path.suffix == ".pdf"), None)
        if pdf is None or not pdf.is_file():
            raise StageError("the refinement stage left no PDF to deliver")
        _atomic_copy(pdf, paths.final_pdf)
        _atomic_copy(pdf.with_suffix(".tex"), paths.final_tex)
        write_json(paths.time_breakdown, self._time_breakdown())

        report: dict[str, Any] = {
            "enabled": harness.enabled,
            "project": harness.project.stats() if harness.project is not None else None,
            "lab": harness.lab.stats() if harness.lab is not None else None,
            "reports": [],
        }
        try:
            report["reports"] = [dream.to_dict() for dream in harness.finish()]
        except Exception as exc:
            # The paper is already delivered; a failed consolidation must not undo that.
            log.warning("The end-of-run memory consolidation failed: %s", exc, exc_info=True)
            report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(self.memory_report, report)
        self._write_usage()
        return [
            paths.final_pdf,
            paths.final_tex,
            paths.time_breakdown,
            paths.usage,
            self.memory_report,
        ]

    def _time_breakdown(self) -> dict[str, float]:
        """Same keys as the original pipeline wrote, so existing tooling keeps working."""
        stages = self.state.stages()

        def seconds(name: str) -> float:
            return float(stages.get(name, {}).get("seconds") or 0.0)

        elapsed = time.time() - self.invocation_start
        return {
            "Start_Time_Absolute": self.start_time,
            "OutlineAgent": seconds("outline"),
            # The two stages overlap, so the slower one is the wall-clock time of the pair.
            "Parallel_Lit_and_Plot": max(seconds("literature"), seconds("plotting")),
            "SectionWritingAgent": seconds("section_writing"),
            "ContentRefinementAgent": seconds("refinement"),
            "Total_Time_Span": round(elapsed + self.carried_seconds, 3),
        }


def _fenced(text: str, language: str) -> str:
    """``text`` in a code fence longer than any backtick run it contains."""
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text.rstrip(chr(10))}\n{fence}"


def run_paper(
    cfg: PipelineConfig,
    *,
    harness: ContextHarness | None = None,
    compute: ComputeBackend | None = None,
) -> PaperResult:
    """Write one paper from ``<cfg.base_dir>/raw_materials``.

    Missing or invalid inputs raise right away (``FileNotFoundError``,
    ``ValueError``), and so does a compute backend that cannot be built
    (``ComputeError``).  Nothing else raises: a model without usable
    credentials, a missing TeX installation or a stage that keeps failing give
    a result with ``success=False``, and ``state.json`` says what went wrong.

    ``harness`` and ``compute`` default to what ``cfg`` asks for; a compute
    backend created here is closed before returning, one passed in is not.
    """
    paths = prepare_run_dir(cfg, None)
    if harness is None:
        harness = _build_harness(cfg)
    owns_compute = compute is None
    if compute is None:
        compute = _build_compute(cfg)
    ctx = RunContext(cfg=cfg, paths=paths, harness=harness, compute=compute, usage=harness.usage)
    try:
        return _Runner(ctx).run()
    finally:
        if owns_compute:
            try:
                compute.close()
            except Exception as exc:
                log.warning("Closing the %s compute backend failed: %s", compute.name, exc)
