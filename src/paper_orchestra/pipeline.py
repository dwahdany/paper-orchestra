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
after the last stage that completed.  That holds as long as the inputs and the
settings that shape the paper are the same and the research cutoff did not move
earlier; otherwise every stage runs again.
Agents are created through the module-level factory functions below, which
import lazily, so tests can swap in stubs and importing this module stays cheap.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import queue
import secrets
import shutil
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paper_orchestra.compute.base import ComputeError
from paper_orchestra.config import DEPENDENT_RESULT_SETTINGS, PipelineConfig, result_settings
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import LLMConfigError, Usage, UsageTracker
from paper_orchestra.memory.harness import RunCancelled
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

#: How long an interrupted run waits for the stages on other threads before the interrupt
#: is passed on.  They stop at their next model call; one in the middle of a long call or
#: job is left behind after this.
STOP_WAIT_SECONDS = 10.0
#: While stages run, ``usage.json`` is brought up to date at most this often, so a run that
#: is killed loses no more than the last moments of what it spent.
USAGE_FLUSH_SECONDS = 5.0

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
#: The error ``state.json`` gives for a run that was stopped.
_INTERRUPTED = "the run was interrupted"


#: Called as ``on_stage(name, record)`` when a stage has settled; see :func:`run_paper`.
StageCallback = Callable[[str, Mapping[str, Any]], None]


class StageError(RuntimeError):
    """A stage finished without producing the files the next stage needs."""


@dataclass
class PaperResult:
    success: bool
    pdf_path: Path | None
    base_dir: Path
    #: Per stage: ``status``, ``seconds``, ``attempts``, ``error`` (plus ``outputs``, ``reused``
    #: and, where they apply, ``tolerated`` and ``after_done``).
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
    return _backend_problem(cfg, compute)


def _backend_problem(cfg: PipelineConfig, compute: ComputeBackend) -> str | None:
    """Why the compute backend cannot do this run's work, when it says so."""
    try:
        problems = compute.check(python=cfg.use_plotting, latex=True)
    except ComputeError as exc:
        return str(exc)
    return problems[0] if problems else None


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


def _timestamp(epoch_seconds: float | None = None) -> str:
    """UTC time as written to ``state.json`` (now, unless an epoch time is given)."""
    if epoch_seconds is None:
        moment = datetime.now(UTC)
    else:
        moment = datetime.fromtimestamp(epoch_seconds, UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _describe(exc: BaseException) -> str:
    """``Type: message`` for state, notes and results; the log has the full traceback."""
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= _MAX_ERROR_CHARS:
        return text
    return text[:_MAX_ERROR_CHARS] + " ... (cut here; the log has the full error)"


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and, in turn, the exceptions it was raised from."""
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        yield link
        link = link.__cause__ or (None if link.__suppress_context__ else link.__context__)


def _fingerprint(cfg: PipelineConfig, paths: RunPaths) -> str:
    """Digest of what the authors supplied and the run uses: idea, log, figures, template."""
    digest = hashlib.sha256()

    def add(label: str, path: Path) -> None:
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())

    add("idea", paths.idea)
    add("experimental_log", paths.experimental_log)
    roots = [("template", cfg.template_dir)]
    if cfg.uses_provided_figures:
        # Figures that plotting replaces are not inputs: changing them changes nothing.
        roots.insert(0, ("figures", paths.provided_figures_dir))
    for label, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            add(f"{label}/{path.relative_to(root).as_posix()}", path)
    return digest.hexdigest()


def _cutoff_moved_earlier(previous: dict[str, Any], current: str) -> bool:
    """Whether the earlier run could have collected literature that ``current`` rules out."""
    from paper_orchestra.utils.scholar import parse_cutoff

    cutoff = parse_cutoff(current)
    if cutoff is None:
        return False
    config = previous.get("config")
    earlier = config.get("research_cutoff") if isinstance(config, dict) else None
    try:
        earlier_cutoff = parse_cutoff(earlier if isinstance(earlier, str) else None)
    except ValueError:
        return True
    return earlier_cutoff is None or cutoff < earlier_cutoff


def _changed_settings(previous: dict[str, Any], cfg: PipelineConfig) -> list[str]:
    """The settings that shape the paper and differ from the earlier run's (``name: old -> new``).

    A setting the earlier state does not record counts as changed: what that run
    did cannot be told.
    """
    recorded = previous.get("config")
    before = result_settings(recorded if isinstance(recorded, dict) else {})
    now = cfg.result_settings()
    # The settings of a part that was switched on or off go without saying.
    implied = {
        name
        for switch, names in DEPENDENT_RESULT_SETTINGS.items()
        if before.get(switch) != now.get(switch)
        for name in names
    }
    changes = []
    for name, value in now.items():
        if name in implied or (name in before and before[name] == value):
            continue
        old = repr(before[name]) if name in before else "not recorded"
        changes.append(f"{name}: {old} -> {value!r}")
    return changes


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


def _figures_in_use(cfg: PipelineConfig, paths: RunPaths) -> tuple[list[dict[str, Any]], list[str]]:
    """The provided figures that are part of this run, and what is off about them.

    With plotting every figure is generated and the provided ones stay out of the
    workspace, the figure list and the memory, unless the run is told to keep them.
    """
    if not cfg.uses_provided_figures:
        return [], []
    return _provided_figures(paths)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def _usage_of(row: dict[str, Any]) -> Usage:
    return Usage(**{name: int(row.get(name) or 0) for name in _USAGE_FIELDS})


def read_carried_usage(path: Path) -> list[dict[str, Any]]:
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


def usage_report(
    tracker: UsageTracker,
    carried: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """The run's usage ledger with an estimated cost, including earlier invocations.

    ``baseline`` is what ``tracker`` already held when this invocation started.  A harness
    that served an earlier invocation still carries that usage, which ``carried`` (read
    from ``usage.json``) accounts for, so only what was added since is counted.
    """
    before = {(row["label"], row["model"]): _usage_of(row) for row in baseline}
    merged = UsageTracker()
    for row in carried:
        merged.record(row["label"], row["model"], _usage_of(row))
    for row in tracker.to_dict()["by_label"]:
        spent = _usage_of(row)
        earlier = before.get((row["label"], row["model"]))
        if earlier is not None:
            spent = Usage(
                **{name: getattr(spent, name) - getattr(earlier, name) for name in _USAGE_FIELDS}
            )
        if any(spent.to_dict().values()):
            merged.record(row["label"], row["model"], spent)
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
        "started_at": None,
        "finished_at": None,
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
        self._closed = False
        self._data: dict[str, Any] = {**header, "status": "running", "error": None}
        self._data["stages"] = stages
        with self._lock:
            self._save()

    def _save(self) -> None:
        self._data["updated_at"] = _timestamp()
        write_json(self._path, self._data)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def update(self, stage: str, **fields: Any) -> None:
        """Record a transition of ``stage``; ignored once the run has been closed.

        A stage that is still winding down on another thread when the run is
        closed as interrupted must not turn up as completed afterwards: the
        resumed run would then reuse work nobody waited for.
        """
        with self._lock:
            if self._closed:
                return
            self._data["stages"][stage].update(fields)
            self._save()

    def record(self, stage: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._data["stages"][stage])

    def stages(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: dict(record) for name, record in self._data["stages"].items()}

    def interrupt(self, error: str) -> None:
        """Put on record that the run is being stopped.

        The state stays open, so a stage that stops in time can still say how it
        ended; :meth:`close` then makes the record final.
        """
        with self._lock:
            if self._closed:
                return
            self._set_status("interrupted", error)
            self._save()

    def close(self, status: str, error: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._set_status(status, error)
            self._save()

    def _set_status(self, status: str, error: str | None) -> None:
        self._data["status"] = status
        self._data["error"] = error
        if status == "interrupted":
            for record in self._data["stages"].values():
                if record["status"] == "running":
                    record["status"] = "interrupted"


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
    #: Follow-up that runs once after the stage completed.  The stage record says when it
    #: is done (``after_done``), so a resumed run makes up for one that is missing.
    after: Callable[[], None] | None = None
    #: Runs when the stage's result is reused: puts back what the stage keeps outside its
    #: recorded outputs, if that is gone.
    restore: Callable[[], None] | None = None


class _Runner:
    def __init__(self, ctx: RunContext, on_stage: StageCallback | None = None) -> None:
        self.ctx = ctx
        self.on_stage = on_stage
        middle = [_Stage("literature", self._literature, after=self._synthesize_literature)]
        if ctx.cfg.use_plotting:
            middle.append(_Stage("plotting", self._plotting, optional=True))
        self.groups: tuple[tuple[_Stage, ...], ...] = (
            (_Stage("ingest", self._ingest, restore=self._restore_source_pages),),
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
        #: What the harness's ledger already held when this invocation started.
        self.usage_baseline: list[dict[str, Any]] = []
        self.carried_seconds = 0.0
        self.invocation_start = 0.0
        self.start_time = 0.0
        #: Whether any stage is left to run in this invocation.
        self.pending = False
        #: Set when the run is over before its stages are: a stage that is still working on
        #: another thread starts nothing new.
        self._stop = threading.Event()
        #: Stages that neither another attempt nor going on without them can help, with what
        #: to add to the stage's own error (see :meth:`_blocker`).
        self._blocked: dict[str, str] = {}
        self._stage_threads: list[threading.Thread] = []
        self._usage_lock = threading.Lock()
        self.state: _State

    # -- orchestration ---------------------------------------------------------

    def run(self) -> PaperResult:
        ctx = self.ctx
        cfg, paths, harness = ctx.cfg, ctx.paths, ctx.harness
        self.invocation_start = time.time()
        self.usage_baseline = harness.usage.to_dict()["by_label"]
        fingerprint = _fingerprint(cfg, paths)
        previous = self._previous_state(fingerprint)
        self.reuse = self._plan_reuse(previous)
        names = [stage.name for group in self.groups for stage in group]
        to_run = [name for name in names if name not in self.reuse]
        self.pending = bool(to_run)

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
            self.carried_usage = read_carried_usage(paths.usage)
            for name in self.reuse:
                stages[name] = {**_blank_record(), **previous["stages"][name], "reused": True}
            # Stages that ran side by side overlapped, so their group took as long as its
            # slowest stage: the same rule as for Parallel_Lit_and_Plot in the breakdown.
            spent = {name: float(stages[name]["seconds"] or 0.0) for name in self.reuse}
            self.carried_seconds = sum(
                max((spent.get(stage.name, 0.0) for stage in group), default=0.0)
                for group in self.groups
            )
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
                "started_at": _timestamp(self.start_time),
                "start_time": self.start_time,
                "inputs": fingerprint,
                "config": cfg.to_dict(),
            },
            stages,
        )
        self._record(
            "run_start",
            models=cfg.to_dict()["models"],
            use_plotting=cfg.use_plotting,
            reused=sorted(self.reuse),
        )

        if failure is None:
            try:
                with self._usage_kept_current():
                    failure = self._run_groups()
            except BaseException:
                # The run is over.  Nothing may go on spending behind its back, and nothing
                # that ends from here on may be recorded as done.
                self._stop.set()
                self._guarded(harness.cancel, "stop of the run's model calls")
                self._guarded(
                    lambda: self.state.close("interrupted", _INTERRUPTED),
                    "update of state.json",
                )
                self._guarded(self._write_usage, "usage report")
                log.warning(
                    "The run in %s was interrupted; running the same command again resumes it.",
                    paths.base_dir,
                )
                raise

        status = "failed" if failure else "succeeded"
        self.state.close(status, failure)
        self._record("run_end", status=status, error=failure)
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
        changed = _changed_settings(previous, cfg)
        if changed:
            # The stages on disk were made under other settings.  Building on them would
            # deliver, and record as this configuration's work, a paper it did not write.
            log.warning(
                "The settings changed since the previous run in %s (%s), so every stage runs "
                "again and replaces that run's paper. The project memory there is kept; use a "
                "new output directory to keep both papers or for an independent run.",
                paths.base_dir,
                "; ".join(changed),
            )
            return None
        if previous.get("inputs") != fingerprint:
            log.warning(
                "The idea, experimental log, figures or template changed since the previous run "
                "in %s, so every stage runs again. The project memory there is kept; use a new "
                "output directory for an unrelated paper.",
                paths.base_dir,
            )
            return None
        if _cutoff_moved_earlier(previous, cfg.research_cutoff):
            # A later cutoff keeps what was collected valid; an earlier one does not.
            log.warning(
                "The research cutoff is now %s, earlier than in the previous run in %s, so "
                "every stage runs again: the literature collected then may be too recent.",
                cfg.research_cutoff,
                paths.base_dir,
            )
            return None
        return previous

    def _plan_reuse(self, previous: dict[str, Any] | None) -> set[str]:
        """Stages whose earlier result stands.

        A stage is reused when it completed, its recorded outputs still exist
        and every stage before it is reused too.  An optional stage whose
        failure the earlier run accepted is settled as well, provided that run
        went on to deliver its paper.  If it did not, nobody got a result
        without that stage's output, and what broke the stage may be what
        stopped the run: the stage is run again.
        """
        reuse: set[str] = set()
        if previous is None:
            return reuse
        base = self.ctx.paths.base_dir
        delivered = previous.get("status") == "succeeded"
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
                elif stage.optional and record.get("tolerated") and delivered:
                    done.append(stage.name)
            reuse.update(done)
            if len(done) < len(group):
                break
        return reuse

    def _run_groups(self) -> str | None:
        """Run every stage; return a description of the failure that stopped the run, if any."""
        for group in self.groups:
            outcomes = self._run_group(group)
            failed = [stage for stage in group if not outcomes[stage.name]]
            fatal = next((stage for stage in failed if not stage.optional), None)
            # Going on without an optional stage does not help when what broke it breaks
            # the stages after it as well.
            fatal = fatal or next((stage for stage in failed if stage.name in self._blocked), None)
            if fatal is None:
                for stage in failed:
                    record = self.state.record(stage.name)
                    self.state.update(stage.name, tolerated=True)
                    log.warning(
                        "Stage %s failed after %d attempt(s); continuing without its output. "
                        "Last error: %s",
                        stage.name,
                        record["attempts"],
                        record["error"],
                    )
            # Reported once the whole group has settled, so a callback (a volume commit,
            # say) never runs while a sibling stage is still writing files.
            for stage in group:
                self._notify(stage.name)
            if fatal is not None:
                record = self.state.record(fatal.name)
                detail = self._blocked.get(fatal.name)
                return (
                    f"stage '{fatal.name}' failed after {record['attempts']} attempt(s): "
                    f"{record['error']}" + (f" ({detail})" if detail else "")
                )
        return None

    def _run_group(self, group: tuple[_Stage, ...]) -> dict[str, bool]:
        """Run the stages of one group side by side; whether each one produced its output."""
        if len(group) == 1:
            return {group[0].name: self._run_stage(group[0])}
        results: queue.SimpleQueue[tuple[str, bool | BaseException]] = queue.SimpleQueue()

        def work(stage: _Stage) -> None:
            try:
                outcome: bool | BaseException = self._run_stage(stage)
            except BaseException as exc:  # passed to the main thread, which ends the run
                outcome = exc
            results.put((stage.name, outcome))

        # Daemon threads, so a stage that is slow to stop cannot keep an interrupted process
        # alive.  On the normal path every one of them is joined below.
        threads = [
            threading.Thread(target=work, args=(stage,), name=f"stage-{stage.name}", daemon=True)
            for stage in group
        ]
        self._stage_threads.extend(threads)
        outcomes: dict[str, bool] = {}
        try:
            for thread in threads:
                thread.start()
            for _ in threads:
                name, outcome = results.get()
                if isinstance(outcome, BaseException):
                    raise outcome
                outcomes[name] = outcome
        except BaseException:
            self._stop_stages(threads)
            raise
        # Every stage is awaited, so a failing one never leaves its sibling running behind
        # the pipeline's back.
        for thread in threads:
            thread.join()
        return outcomes

    def _stop_stages(self, threads: Sequence[threading.Thread]) -> None:
        """End the stages on other threads when the run is interrupted.

        A stage cannot be stopped in the middle of a model call or a compute job,
        but waiting for it to finish would keep the run going, and spending,
        through every call it has left.  The harness refuses every further model
        call instead, which ends a stage at its next one, and the stages get a
        bounded time to get there so that what they spent is still recorded.
        ``state.json`` and ``usage.json`` say that the run was interrupted before
        that wait begins.
        """
        self._stop.set()
        self._guarded(self.ctx.harness.cancel, "stop of the run's model calls")
        running = [thread for thread in threads if thread.is_alive()]
        if not running:
            return
        # On record before the wait, not after it: whoever stopped the run may end the
        # process sooner (a kill after a grace period, a second interrupt).
        self._guarded(lambda: self.state.interrupt(_INTERRUPTED), "update of state.json")
        self._guarded(self._write_usage, "usage report")
        log.warning(
            "Interrupted: stopping %s at the next model call (waiting up to %.0f s).",
            " and ".join(thread.name.removeprefix("stage-") for thread in running),
            STOP_WAIT_SECONDS,
        )
        deadline = time.monotonic() + STOP_WAIT_SECONDS
        for thread in running:
            thread.join(max(0.0, deadline - time.monotonic()))
        left = [thread.name for thread in running if thread.is_alive()]
        if left:
            log.warning(
                "Still busy %.0f s after the interrupt: %s. Not waiting any longer; what "
                "these do from here on is not recorded.",
                STOP_WAIT_SECONDS,
                ", ".join(left),
            )

    def running_stages(self) -> list[threading.Thread]:
        """Stage threads that outlived the run (it was interrupted and they were slow to stop)."""
        return [thread for thread in self._stage_threads if thread.is_alive()]

    def _stopping(self) -> bool:
        return self._stop.is_set() or self.ctx.harness.cancelled

    def _notify(self, name: str) -> None:
        callback = self.on_stage
        if callback is None:
            return
        record = self.state.record(name)
        self._guarded(lambda: callback(name, record), f"on_stage callback for the {name} stage")

    def _record(self, type: str, **fields: Any) -> None:
        """Add a milestone to the transcript, which is a record of the run, not a dependency."""
        self._guarded(lambda: self.ctx.harness.record(type, **fields), f"transcript entry '{type}'")

    def _run_stage(self, stage: _Stage) -> bool:
        name = stage.name
        if name in self.reuse:
            log.info("Stage %s: reusing the result of the earlier run.", name)
            self._make_up_for(stage)
            return True

        attempts = self.ctx.cfg.max_stage_attempts
        base = self.ctx.paths.base_dir
        seconds = 0.0
        started_at = _timestamp()
        for attempt in range(1, attempts + 1):
            if self._stopping():
                raise RunCancelled(f"the run was stopped before the {name} stage started")
            self.state.update(
                name,
                status="running",
                attempts=attempt,
                started_at=started_at,
                finished_at=None,
                error=None,
            )
            self._record("stage_start", stage=name, attempt=attempt)
            log.info("Stage %s started (attempt %d of %d).", name, attempt, attempts)
            started = time.monotonic()
            try:
                outputs = [Path(path) for path in stage.run()]
                if self._stopping():
                    # Agents carry on after a model call that failed in places, so a stage
                    # can come back from a stopped run with less than its work.
                    raise RunCancelled(f"the run was stopped during the {name} stage")
                missing = [str(path) for path in outputs if not path.is_file()]
                if missing:
                    raise StageError(
                        f"the {name} stage finished without producing {', '.join(missing)}"
                    )
            except Exception as exc:
                seconds += time.monotonic() - started
                if isinstance(exc, RunCancelled) or self._stopping():
                    # Not a failure of the stage: nothing to retry, to tolerate or to learn.
                    self._stage_interrupted(name, attempt, seconds)
                    if isinstance(exc, RunCancelled):
                        raise
                    raise RunCancelled(f"the run was stopped during the {name} stage") from exc
                error = _describe(exc)
                blocker = self._blocker(exc)
                retry = attempt < attempts and blocker is None
                log.error(
                    "Stage %s failed (attempt %d of %d): %s",
                    name,
                    attempt,
                    attempts,
                    error,
                    exc_info=True,
                )
                self.state.update(
                    name,
                    status="failed",
                    finished_at=_timestamp(),
                    seconds=round(seconds, 3),
                    error=error,
                )
                self._record(
                    "stage_end",
                    stage=name,
                    status="failed",
                    attempt=attempt,
                    seconds=round(seconds, 3),
                    error=error,
                )
                self._note_failure(stage, attempt, attempts, exc, retry, blocker is not None)
                if blocker is not None:
                    self._blocked[name] = blocker
                if not retry:
                    break
                continue

            seconds += time.monotonic() - started
            self.state.update(
                name,
                status="completed",
                finished_at=_timestamp(),
                seconds=round(seconds, 3),
                error=None,
                outputs=[_relative(path, base) for path in outputs],
            )
            self._record(
                "stage_end",
                stage=name,
                status="completed",
                attempt=attempt,
                seconds=round(seconds, 3),
            )
            log.info("Stage %s completed in %.1f s.", name, seconds)
            self._run_after(stage)
            self._settle(name)
            return True

        self._settle(name)
        return False

    def _stage_interrupted(self, name: str, attempt: int, seconds: float) -> None:
        self.state.update(
            name, status="interrupted", finished_at=_timestamp(), seconds=round(seconds, 3)
        )
        self._record(
            "stage_end",
            stage=name,
            status="interrupted",
            attempt=attempt,
            seconds=round(seconds, 3),
        )
        log.info("Stage %s was interrupted.", name)
        self._settle(name)

    def _blocker(self, exc: Exception) -> str | None:
        """Why neither another attempt nor going on without the stage can work, if so.

        A configuration problem (credentials, SDK, unsupported option) and a
        compute backend that cannot do its work are the same on every attempt,
        and the stages that follow depend on them as well.  A backend error can
        say so itself (an ``unusable`` attribute that is true).  Beyond that the
        backend is asked, because a stage may report a backend failure in its
        own words.  The result is what to add to the stage's own error, which
        may be nothing; ``None`` means the failure may be transient.
        """
        for link in _causes(exc):
            if isinstance(link, LLMConfigError) or (
                isinstance(link, ComputeError) and getattr(link, "unusable", False)
            ):
                return "" if link is exc else _describe(link)
        compute = self.ctx.compute
        try:
            problem = _backend_problem(self.ctx.cfg, compute)
        except Exception as error:
            log.warning("The %s compute backend could not be checked: %s", compute.name, error)
            return None
        if problem is None:
            return None
        return f"the {compute.name} compute backend cannot be used: {problem}"

    def _run_after(self, stage: _Stage) -> None:
        """Run the stage's follow-up and record that it is done.

        The stage itself is already recorded as completed.  The separate mark is
        what lets a resumed run tell a follow-up that ran from one that was cut
        off or did not complete, which it then makes up for.
        """
        if stage.after is None or self._stopping():
            return
        try:
            stage.after()
        except Exception as exc:
            # Memory upkeep: the paper can be written without it.
            log.warning(
                "The follow-up of the %s stage did not complete (%s). A resumed run tries it again.",
                stage.name,
                exc,
                exc_info=not isinstance(exc, StageError),
            )
            return
        self.state.update(stage.name, after_done=True)

    def _make_up_for(self, stage: _Stage) -> None:
        """Supply what a reused stage still owes the stages that run."""
        if stage.restore is not None:
            self._guarded(stage.restore, f"check of what the {stage.name} stage left in memory")
        # The follow-up prepares memory for later stages; when none of them runs, a model
        # session would be spent on pages that nothing reads.
        owed = stage.after is not None and not self.state.record(stage.name).get("after_done")
        if owed and self.pending:
            log.info("Stage %s: its follow-up is not recorded as done; running it now.", stage.name)
            self._run_after(stage)

    def _note_failure(
        self,
        stage: _Stage,
        attempt: int,
        attempts: int,
        exc: Exception,
        retry: bool,
        blocked: bool,
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
        elif stage.optional and not blocked:
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
        if self.state.closed:
            # A stage that outlived an interrupted run: the run's files are final.
            return
        self._guarded(self.ctx.harness.refresh, f"memory refresh after the {name} stage")
        self._guarded(self._write_usage, "usage report")

    def _write_usage(self) -> dict[str, Any]:
        # Stage threads and the periodic flush write this file too; one at a time, so the
        # report written last is never older than the one it replaces.
        with self._usage_lock:
            report = usage_report(self.ctx.harness.usage, self.carried_usage, self.usage_baseline)
            write_json(self.ctx.paths.usage, report)
        return report

    @contextlib.contextmanager
    def _usage_kept_current(self) -> Iterator[None]:
        """Bring ``usage.json`` up to date while stages run, not only when one settles.

        A stage can run for a long time, and a run that is killed in the middle
        of it would otherwise lose everything the stage spent.  The ledger does
        not announce new entries, so it is looked at every
        ``USAGE_FLUSH_SECONDS`` and written out when it has grown.
        """
        ledger = self.ctx.harness.usage
        done = threading.Event()

        def flush() -> None:
            written = ledger.total()
            while not done.wait(USAGE_FLUSH_SECONDS):
                total = ledger.total()
                if total != written:
                    self._guarded(self._write_usage, "usage report")
                    written = total

        thread = threading.Thread(target=flush, name="usage-flush", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    # -- stages ----------------------------------------------------------------

    def _ingest(self) -> list[Path]:
        cfg, paths, harness = self.ctx.cfg, self.ctx.paths, self.ctx.harness
        # Start from the template as shipped: drafts and figures of an earlier run must not
        # leak into this one.
        if paths.latex_dir.exists():
            shutil.rmtree(paths.latex_dir)
        shutil.copytree(cfg.template_dir, paths.latex_dir)
        paths.figures_dir.mkdir(exist_ok=True)
        if cfg.uses_provided_figures and paths.provided_figures_dir.is_dir():
            shutil.copytree(paths.provided_figures_dir, paths.figures_dir, dirs_exist_ok=True)
        # What an earlier run generated goes as well: without plotting no stage would clear
        # it, and it would pass for this run's figures.
        paths.plotting_results.unlink(missing_ok=True)
        if paths.generated_figures_dir.is_dir():
            shutil.rmtree(paths.generated_figures_dir)

        figures, problems = _figures_in_use(cfg, paths)
        for problem in problems:
            log.warning("Provided figures: %s.", problem)
        write_json(paths.figures_info, figures)
        if not cfg.uses_provided_figures:
            unused, _ = _provided_figures(paths)
            if unused:
                log.warning(
                    "The %d figure(s) in %s are not used: with plotting on, every figure is "
                    "generated. Pass --keep-provided-figures to give them to the writer as well.",
                    len(unused),
                    paths.provided_figures_dir,
                )
        elif not figures:
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
        # Figure pages describe files in the workspace, which was just rebuilt.  The pages
        # of an earlier run (figures it was given or generated) would announce files that
        # are not there, and captions this run is not meant to see.
        for stale in harness.list_pages("wiki/figures"):
            harness.delete_page(stale)
        self._write_source_pages(figures)
        return [paths.template_tex, paths.guidelines, paths.figures_info]

    def _source_pages(self) -> tuple[tuple[str, str, str, Path], ...]:
        """The inputs that are mirrored into the wiki: page name, title, summary, file."""
        cfg, paths = self.ctx.cfg, self.ctx.paths
        return (
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

    def _write_source_pages(self, figures: Sequence[dict[str, Any]]) -> None:
        """Mirror the inputs into the wiki as read-only ground truth."""
        paths, harness = self.ctx.paths, self.ctx.harness
        for name, title, summary, path in self._source_pages():
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

    def _restore_source_pages(self) -> None:
        """Write the source pages again when ingest is reused but the wiki lacks them.

        That happens when the memory directory was removed or another one is
        used now.  The source pages are what every session is told to trust and
        what keeps this paper's content out of the lab memory, so a run must not
        go on without them.  Writing them is plain file work.
        """
        harness = self.ctx.harness
        if not harness.enabled:
            return
        present = set(harness.list_pages("wiki/sources"))
        missing = [
            name for name, *_ in self._source_pages() if f"wiki/sources/{name}.md" not in present
        ]
        if not missing:
            return
        log.warning(
            "The project memory has no source page for: %s. Writing the source pages again.",
            ", ".join(missing),
        )
        figures, _ = _figures_in_use(self.ctx.cfg, self.ctx.paths)
        self._write_source_pages(figures)
        harness.refresh()

    def _outline(self) -> list[Path]:
        from paper_orchestra.agents.literature import CHECKPOINT_NAME

        cfg, paths = self.ctx.cfg, self.ctx.paths
        # A new outline starts the literature search over, even if it happens to ask for
        # the searches an earlier outline asked for.
        for stale in (paths.outline, self.pristine_outline, paths.literature_dir / CHECKPOINT_NAME):
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
        from paper_orchestra.agents.literature import CHECKPOINT_NAME

        cfg, paths = self.ctx.cfg, self.ctx.paths
        # A retried or resumed stage must see the inputs as they were before its own earlier
        # attempt filled in the template and injected citations into the outline.  The one
        # thing kept is the agent's record of the papers it has verified: the search is the
        # expensive half of the stage, and a failure while writing should not repeat it.
        # (The agent itself decides whether the record still matches the outline.)
        checkpoint = paths.literature_dir / CHECKPOINT_NAME
        verified = checkpoint.read_bytes() if checkpoint.is_file() else None
        _reset_dir(paths.literature_dir)
        if verified is not None:
            checkpoint.write_bytes(verified)
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
            raise StageError(f"literature synthesis: {report.error}")

    def _plotting(self) -> list[Path]:
        paths = self.ctx.paths
        self._reset_figure_workspace()
        _plotting_agent(self.ctx).run(self._outline_for_agents())
        return [paths.plotting_results, paths.figures_info]

    def _reset_figure_workspace(self) -> None:
        """Undo what an earlier plotting attempt left: only the figures ingest listed remain."""
        paths = self.ctx.paths
        provided, _ = _figures_in_use(self.ctx.cfg, paths)
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
    on_stage: StageCallback | None = None,
) -> PaperResult:
    """Write one paper from ``<cfg.base_dir>/raw_materials``.

    Missing or invalid inputs raise right away (``FileNotFoundError``,
    ``ValueError``), and so does a compute backend that cannot be built
    (``ComputeError``).  Nothing else raises: a model without usable
    credentials, a missing TeX installation or a stage that keeps failing give
    a result with ``success=False``, and ``state.json`` says what went wrong.

    ``harness`` and ``compute`` default to what ``cfg`` asks for; a compute
    backend created here is closed before returning, one passed in is not.

    An interrupt (``KeyboardInterrupt``, or ``harness.cancel()`` from another
    thread, which surfaces as ``RunCancelled``) is passed on after the run was
    recorded as interrupted in ``state.json`` and ``usage.json`` was written.
    Stages running on other threads are told to stop and waited for briefly; a
    harness that was cancelled stays cancelled, so resume with a new one.  If a
    stage is still busy after that wait, a backend created here is closed once
    the stage has ended, or when the process ends, whichever comes first.

    ``on_stage(name, record)`` is called from this thread each time a stage has
    settled (completed, failed for good, or reused from an earlier run), with
    the stage's ``state.json`` record.  Stages that run side by side are
    reported together once all of them have settled, so the callback is a safe
    place to checkpoint the run directory.  What it raises is logged and ignored.
    """
    paths = prepare_run_dir(cfg, None)
    if harness is None:
        harness = _build_harness(cfg)
    owns_compute = compute is None
    if compute is None:
        compute = _build_compute(cfg)
    ctx = RunContext(cfg=cfg, paths=paths, harness=harness, compute=compute, usage=harness.usage)
    runner = _Runner(ctx, on_stage)
    try:
        return runner.run()
    finally:
        if owns_compute:
            _close_when_idle(compute, runner.running_stages())


def _close_when_idle(compute: ComputeBackend, stages: Sequence[threading.Thread]) -> None:
    """Close a backend this module built, once no stage can be using it any more.

    After an interrupt a stage may still be in the middle of a job.  Closing the
    backend under it would let its next job start resources that nothing
    releases, so the close waits for those stages on a thread of its own.  A
    process that ends before they do closes the backend on its way out: the
    stages end with the process, and what the backend holds outside it (remote
    sandboxes, say) would otherwise be left running.
    """
    lock = threading.Lock()
    closed = False

    def close() -> None:
        nonlocal closed
        # Under the lock, so the end of the process waits for a close that is under way.
        with lock:
            if closed:
                return
            closed = True
            try:
                compute.close()
            except Exception as exc:
                log.warning("Closing the %s compute backend failed: %s", compute.name, exc)

    def close_after_stages() -> None:
        for stage in stages:
            stage.join()
        close()
        # Nothing is left for the end of the process to do, or to keep the backend for.
        atexit.unregister(close)

    if not stages:
        close()
        return
    atexit.register(close)
    threading.Thread(target=close_after_stages, name="compute-close", daemon=True).start()
