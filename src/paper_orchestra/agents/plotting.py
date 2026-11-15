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

"""Plotting agent: turns each entry of the outline's plotting plan into a figure.

For every planned figure the agent gathers the source material, optionally
retrieves few-shot examples from a PaperBanana checkout, plans the figure,
styles the plan, renders it, lets a critic look at the rendered image and
revise the description, and writes a caption.

Nothing is rendered in this process.  Plots, and diagrams when no
image-generation model is configured, are drawn by a model-written matplotlib
script that runs only through ``ctx.compute.run_python``; the script is
untrusted and its output is validated as an image before it is used.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import secrets
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image as PILImage
from pydantic import BaseModel

from paper_orchestra.config import PAPER_BANANA_ENV
from paper_orchestra.context import RunContext
from paper_orchestra.errors import stops_the_run
from paper_orchestra.llm.base import Image, LLMClient, LLMError, Part, Text
from paper_orchestra.llm.parsing import parse_python
from paper_orchestra.memory.harness import HarnessConversation, MemoryPolicy, RunCancelled
from paper_orchestra.prompts import plotting as prompts
from paper_orchestra.prompts.plotting import TaskName
from paper_orchestra.utils.common import read_text, write_json

log = logging.getLogger(__name__)

STAGE = "plotting"
AGENT = "PlottingAgent"

LABEL_RETRIEVE = "plotting.retrieve"
LABEL_PLAN = "plotting.plan"
LABEL_STYLE = "plotting.style"
LABEL_CODE = "plotting.code"
LABEL_CRITIC = "plotting.critic"
LABEL_CAPTION = "plotting.caption"

#: File a figure script leaves behind, relative to its working directory.
FIGURE_FILE = "figure.png"
FIGURE_DPI = 200
#: Longer side of a saved figure, in pixels, above which its resolution is lowered
#: (never below 72 dpi, so that an absurd figure size fails instead of being unreadable).
MAX_FIGURE_PIXELS = 4096
NO_FIGURE_EXIT_CODE = 3
DEFAULT_ASPECT_RATIO = "16:9"
#: Drawing area of a figure in square inches, whatever its aspect ratio.
FIGURE_AREA = 54.0
FEW_SHOT_EXAMPLES = 10
#: Candidates shown to the retriever, and how much of each candidate's content.
FEW_SHOT_POOL_LIMIT = 200
CANDIDATE_CONTENT_CHARS = 1500
NO_CHANGES = "No changes needed."

_MAX_ID_CHARS = 80
_OUTPUT_TAIL_CHARS = 2000
#: How many per-figure reasons a PlottingError quotes, and how much of each.
_REASONS_SHOWN = 3
_REASON_CHARS = 300

# Runs after the model's script, in the same interpreter inside the compute backend.  It
# looks pyplot up in sys.modules instead of importing it: a script that never imported
# pyplot cannot have an open figure.  It shares the script's global namespace, where names
# such as max or int may have been rebound (``from numpy import *`` does that), so every
# builtin it needs is reached through the builtins module.
SAVE_FIGURE_EPILOGUE = f'''

# Appended by PaperOrchestra: save the figure this script drew as {FIGURE_FILE}.
def _paper_orchestra_save_figure():
    import builtins
    import os
    import sys

    target = "{FIGURE_FILE}"
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None or not pyplot.get_fignums():
        sys.stderr.write(
            "No matplotlib figure was open when the script ended and it did not write "
            + target
            + ". Create the figure with matplotlib.pyplot and leave it open.\\n"
        )
        sys.exit({NO_FIGURE_EXIT_CODE})
    figure = pyplot.gcf()
    longest_side = builtins.max(figure.get_size_inches())
    dpi = builtins.max(
        72, builtins.min({FIGURE_DPI}, builtins.int({MAX_FIGURE_PIXELS} / longest_side))
    )
    figure.savefig(target, dpi=dpi, bbox_inches="tight")


_paper_orchestra_save_figure()
'''

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_ASPECT_RATIO = re.compile(r"^\s*(\d{1,3})\s*:\s*(\d{1,3})\s*$")
_DESCRIPTION_LABEL = re.compile(
    r"^\s*(?:#+\s*)?\**\s*Detailed Description\s*\**\s*(?::|\n)\s*\**\s*", re.IGNORECASE
)
_CAPTION_LABEL = re.compile(
    r"^\s*\**\s*(?:caption\s*\d*|(?:figure|fig\.?)\s*\d+)\s*\**\s*[:.]\s*\**\s*", re.IGNORECASE
)


class FewShotSelection(BaseModel):
    """Ids of the reference figures chosen as few-shot examples, best match first."""

    selected_ids: list[str]


class FigureCritique(BaseModel):
    """A critic's verdict on a rendered figure."""

    needs_revision: bool
    critic_suggestions: str = ""
    revised_description: str = ""


class PlottingError(RuntimeError):
    """Figures were planned and not one of them could be produced."""


class FigureRenderError(RuntimeError):
    """No usable image could be produced for a figure description."""

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        #: Tail of the failing script's output, when the failure came from figure code.
        self.detail = detail


class _CodeFailure(Exception):
    """One run of a figure script that did not yield a valid ``figure.png``."""

    def __init__(self, problem: str, output: str) -> None:
        super().__init__(problem)
        self.problem = problem
        self.output = output


@dataclass(frozen=True)
class FigureSpec:
    """One validated entry of the outline's plotting plan."""

    #: Filename-safe and unique within the run.
    figure_id: str
    title: str
    objective: str
    task: TaskName
    aspect_ratio: str
    data_source: str

    @property
    def intent(self) -> str:
        return f"Title: {self.title}\nObjective: {self.objective}"


@dataclass(frozen=True)
class _Finished:
    spec: FigureSpec
    description: str
    caption: str
    critic_history: list[dict[str, Any]]
    png: bytes
    method: str
    examples: int


@dataclass(frozen=True)
class _Skipped:
    spec: FigureSpec
    reason: str


class _Renderer(Protocol):
    method: str
    uses_code: bool

    def render(self, description: str, *, suggestions: str = "") -> bytes: ...


# -- small pure helpers ------------------------------------------------------------


def _as_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _non_empty(text: str) -> str | None:
    return text.strip() or None


def _one_line(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def _strip_description_label(text: str) -> str:
    return _DESCRIPTION_LABEL.sub("", text, count=1).strip()


def _clean_caption(text: str) -> str | None:
    """Drop a leading ``Caption:`` / ``Figure 1:`` label and line breaks; ``None`` if empty."""
    return " ".join(_CAPTION_LABEL.sub("", text, count=1).split()) or None


def _is_no_change(text: str) -> bool:
    return text.strip().strip("'\"").rstrip(".").casefold() == NO_CHANGES.rstrip(".").casefold()


def task_name(plot_type: str) -> TaskName:
    """``plot`` or ``diagram`` for a plan's ``plot_type`` (anything chart-like is a plot)."""
    kind = plot_type.strip().lower()
    if kind == "diagram":
        return "diagram"
    return "plot" if "chart" in kind or "plot" in kind else "diagram"


def normalize_aspect_ratio(value: object) -> str:
    match = _ASPECT_RATIO.match(_as_text(value))
    if match and int(match[1]) > 0 and int(match[2]) > 0:
        return f"{int(match[1])}:{int(match[2])}"
    return DEFAULT_ASPECT_RATIO


def figure_size(aspect_ratio: str) -> tuple[float, float]:
    """Width and height in inches for a normalised ``W:H`` ratio, at a constant area."""
    width, height = (int(part) for part in aspect_ratio.split(":"))
    ratio = min(4.0, max(0.25, width / height))
    return round(math.sqrt(FIGURE_AREA * ratio), 1), round(math.sqrt(FIGURE_AREA / ratio), 1)


def safe_figure_id(raw: object, index: int) -> str:
    """A figure id reduced to ``[A-Za-z0-9_-]`` so it is safe as a file and page name."""
    cleaned = _UNSAFE_ID_CHARS.sub("_", _as_text(raw)).strip("_-")
    return cleaned[:_MAX_ID_CHARS].strip("_-") or f"figure_{index}"


def plan_figures(plan: object, *, reserved: Iterable[str] = ()) -> list[FigureSpec]:
    """Validate the outline's plotting plan and give every figure a safe, unique id.

    ``reserved`` are file stems that already belong to other figures (the ones
    the user provided) and must not be reused.
    """
    if plan is None:
        return []
    if not isinstance(plan, list):
        raise ValueError(f"outline['plotting_plan'] must be a list, got {type(plan).__name__}")
    # Compared case-insensitively: the figures may land on a case-insensitive filesystem.
    taken = {name.lower() for name in reserved}
    specs: list[FigureSpec] = []
    for index, entry in enumerate(plan, start=1):
        if not isinstance(entry, Mapping):
            log.warning("Entry %d of the plotting plan is not an object and is ignored.", index)
            continue
        base = safe_figure_id(entry.get("figure_id"), index)
        figure_id, suffix = base, 2
        while figure_id.lower() in taken:
            figure_id, suffix = f"{base}_{suffix}", suffix + 1
        taken.add(figure_id.lower())
        if figure_id != entry.get("figure_id"):
            log.info("Figure id %r is stored as '%s'.", entry.get("figure_id"), figure_id)
        specs.append(
            FigureSpec(
                figure_id=figure_id,
                title=_as_text(entry.get("title")),
                objective=_as_text(entry.get("objective")),
                task=task_name(_as_text(entry.get("plot_type"))),
                aspect_ratio=normalize_aspect_ratio(entry.get("aspect_ratio")),
                data_source=_as_text(entry.get("data_source")),
            )
        )
    return specs


def to_png(data: bytes) -> bytes:
    """Check that ``data`` is a complete, decodable image and return it PNG-encoded.

    Raises ``ValueError`` otherwise.  PNG input is returned unchanged; other
    formats are converted, because the figure is stored and referenced as PNG.
    """
    media_type = Image.from_bytes(data).media_type
    try:
        with PILImage.open(io.BytesIO(data)) as picture:
            picture.load()
            if media_type == "image/png":
                return data
            has_alpha = "A" in picture.getbands() or "transparency" in picture.info
            buffer = io.BytesIO()
            picture.convert("RGBA" if has_alpha else "RGB").save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception as exc:
        # Pillow reports damaged files with many unrelated exception types; all of them
        # mean the same thing here.
        raise ValueError(f"the image data cannot be decoded ({exc})") from exc


def _write_bytes(path: Path, data: bytes) -> None:
    """Atomically write ``data``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run_figure_code(ctx: RunContext, code: str) -> bytes:
    """Run a model-written figure script in the compute backend; return the figure as PNG.

    Raises :class:`_CodeFailure` when the script exits non-zero, times out, or
    leaves no valid image behind.
    """
    timeout = ctx.cfg.figure_code_timeout
    result = ctx.compute.run_python(
        code.rstrip() + "\n" + SAVE_FIGURE_EPILOGUE,
        outputs=[FIGURE_FILE],
        timeout=timeout,
    )
    output = (result.stderr.strip() or result.stdout.strip())[-_OUTPUT_TAIL_CHARS:]
    if result.timed_out:
        raise _CodeFailure(f"It was stopped after the time limit of {timeout:g} seconds.", output)
    if result.returncode != 0:
        raise _CodeFailure(f"It exited with status {result.returncode}.", output)
    data = result.files.get(FIGURE_FILE)
    if not data:
        # The epilogue saves or fails loudly, so a clean exit without the file means it
        # never ran or saved somewhere else.
        raise _CodeFailure(
            f"It finished without producing {FIGURE_FILE}. The figure is saved only when the "
            "script runs to its last line in the directory it started in, so look for an "
            "early exit (sys.exit) or a changed working directory.",
            output,
        )
    try:
        return to_png(data)
    except ValueError as exc:
        raise _CodeFailure(
            f"The {FIGURE_FILE} it wrote is not a valid image: {exc}.", output
        ) from exc


# -- optional PaperBanana data -----------------------------------------------------


class FewShotLibrary:
    """Reference figures and style guides from an optional PaperBanana checkout.

    The checkout is located by the ``PAPER_BANANA_DIR`` environment variable.
    Without it every figure is planned zero-shot with the built-in style guide.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._pools: dict[str, list[dict[str, Any]]] = {}
        self._guides: dict[str, str] = {}

    @classmethod
    def from_env(cls) -> FewShotLibrary:
        raw = os.environ.get(PAPER_BANANA_ENV, "").strip()
        if not raw:
            log.info(
                "%s is not set: figures are planned zero-shot with the built-in style guide.",
                PAPER_BANANA_ENV,
            )
            return cls()
        root = Path(raw).expanduser()
        if not root.is_dir():
            log.warning(
                "%s points to %s, which is not a directory: figures are planned zero-shot "
                "with the built-in style guide.",
                PAPER_BANANA_ENV,
                root,
            )
            return cls()
        return cls(root)

    def _task_dir(self, task: str) -> Path:
        assert self.root is not None
        return self.root / "data" / "PaperBananaBench" / task

    def load(self, tasks: Iterable[str]) -> None:
        """Read the pools and style guides of ``tasks`` once, before worker threads start."""
        if self.root is None:
            return
        for task in sorted(set(tasks)):
            self._pools[task] = self._read_pool(task)
            guide = self.root / "style_guides" / f"neurips2025_{task}_style_guide.md"
            try:
                self._guides[task] = guide.read_text(encoding="utf-8")
            except (OSError, ValueError) as exc:
                log.warning(
                    "No usable %s style guide at %s (%s): using the built-in one.",
                    task,
                    guide,
                    exc,
                )

    def _read_pool(self, task: str) -> list[dict[str, Any]]:
        ref_file = self._task_dir(task) / "ref.json"
        if not ref_file.is_file():
            log.warning("No few-shot pool at %s: %ss are planned zero-shot.", ref_file, task)
            return []
        try:
            raw = json.loads(ref_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Cannot read %s (%s): %ss are planned zero-shot.", ref_file, exc, task)
            return []
        if not isinstance(raw, list):
            log.warning("%s does not contain a list: %ss are planned zero-shot.", ref_file, task)
            return []
        pool = [item for item in raw if isinstance(item, dict) and item.get("id") is not None]
        if len(pool) != len(raw):
            log.warning(
                "%s: %d of %d entries have no id and are ignored.",
                ref_file,
                len(raw) - len(pool),
                len(raw),
            )
        return pool[:FEW_SHOT_POOL_LIMIT]

    def candidates(self, task: str) -> list[dict[str, Any]]:
        return self._pools.get(task, [])

    def style_guide(self, task: str) -> str | None:
        return self._guides.get(task)

    def reference_image(self, task: str, item: Mapping[str, Any]) -> Image | None:
        """The reference figure of a pool item, or ``None`` when it cannot be used."""
        relative = item.get("path_to_gt_image")
        if not isinstance(relative, str) or not relative:
            return None
        task_dir = self._task_dir(task).resolve()
        path = (task_dir / relative).resolve()
        if not path.is_relative_to(task_dir) or not path.is_file():
            log.debug("Reference image of %s is missing or outside %s.", item.get("id"), task_dir)
            return None
        try:
            return Image.from_path(path)
        except (OSError, ValueError) as exc:
            log.debug("Reference image %s is unusable: %s", path, exc)
            return None


# -- renderers ---------------------------------------------------------------------


class _CodeRenderer:
    """Draws one figure with model-written matplotlib code.

    The script, its repairs and its later revisions live in one conversation,
    so the model edits its own code instead of starting over.
    """

    uses_code = True

    def __init__(self, ctx: RunContext, spec: FigureSpec) -> None:
        self._ctx = ctx
        self._spec = spec
        self._conversation: HarnessConversation | None = None
        self._has_script = False
        self.method = (
            f"matplotlib code written by {ctx.cfg.models.plotting} "
            f"and run by the '{ctx.compute.name}' compute backend"
        )

    def render(self, description: str, *, suggestions: str = "") -> bytes:
        ctx, spec = self._ctx, self._spec
        if self._conversation is None:
            self._conversation = ctx.harness.open(
                agent=AGENT,
                stage=STAGE,
                model=ctx.cfg.models.plotting,
                system=prompts.code_system(spec.task, timeout_seconds=ctx.cfg.figure_code_timeout),
                # The one plotting call that reads memory: lessons about figure-code
                # pitfalls from earlier figures and earlier runs are what it is for.
                memory=MemoryPolicy.readonly(),
                label=LABEL_CODE,
            )
        if self._has_script:
            message = prompts.code_revision_request(
                description=description, suggestions=suggestions
            )
        else:
            message = prompts.code_request(
                spec.task,
                description=description,
                aspect_ratio=spec.aspect_ratio,
                figsize=figure_size(spec.aspect_ratio),
            )

        attempts = 1 + max(0, ctx.cfg.max_repair_rounds)
        failures: list[_CodeFailure] = []
        for attempt in range(1, attempts + 1):
            code = self._conversation.send(message, parser=parse_python).parsed
            self._has_script = True
            try:
                png = run_figure_code(ctx, code)
            except _CodeFailure as failure:
                failures.append(failure)
                self._record(attempt, ok=False, problem=failure.problem, output=failure.output)
                log.info(
                    "Figure '%s': the figure script failed (attempt %d of %d). %s",
                    spec.figure_id,
                    attempt,
                    attempts,
                    failure.problem,
                )
                message = prompts.code_repair_request(
                    problem=failure.problem, output=failure.output
                )
                continue
            self._record(attempt, ok=True)
            if failures:
                ctx.harness.note(
                    f"The matplotlib script for figure '{spec.figure_id}' ({spec.task}) failed "
                    f"{len(failures)} time(s) before a corrected script ran.\n"
                    f"First failure: {failures[0].problem}\n\n{failures[0].output}",
                    stage=STAGE,
                    agent=AGENT,
                    kind="lesson",
                    title=f"Figure code repaired: {spec.figure_id}",
                )
            return png
        last = failures[-1]
        raise FigureRenderError(
            f"the figure script still failed after {attempts} attempt(s). {last.problem}",
            detail=last.output,
        )

    def _record(self, attempt: int, *, ok: bool, **fields: Any) -> None:
        self._ctx.harness.record(
            "figure_code",
            stage=STAGE,
            agent=AGENT,
            figure_id=self._spec.figure_id,
            attempt=attempt,
            ok=ok,
            **fields,
        )


class _ImageModelRenderer:
    """Draws a diagram with an image-generation model."""

    uses_code = False

    def __init__(self, ctx: RunContext, spec: FigureSpec, client: LLMClient, model: str) -> None:
        self._ctx = ctx
        self._spec = spec
        self._client = client
        self._model = model
        self.method = f"image model {model}"

    def render(self, description: str, *, suggestions: str = "") -> bytes:
        data = self._client.generate_image(
            prompts.image_model_request(description),
            model=self._model,
            aspect_ratio=self._spec.aspect_ratio,
            system=prompts.IMAGE_MODEL_SYSTEM,
        )
        try:
            if not data:
                raise ValueError("it returned no image")
            png = to_png(data)
        except ValueError as exc:
            self._record(ok=False, problem=str(exc))
            raise FigureRenderError(
                f"the image model '{self._model}' gave no usable image: {exc}"
            ) from exc
        self._record(ok=True)
        return png

    def _record(self, *, ok: bool, **fields: Any) -> None:
        self._ctx.harness.record(
            "image_generation",
            stage=STAGE,
            agent=AGENT,
            figure_id=self._spec.figure_id,
            image_model=self._model,
            ok=ok,
            **fields,
        )


# -- the agent ---------------------------------------------------------------------


class PlottingAgent:
    """Produces the figures of the outline's plotting plan (see the module docstring)."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self._library = FewShotLibrary()
        self._image_client: LLMClient | None = None
        self._vision = True

    def run(self, outline: dict[str, Any]) -> list[dict[str, Any]]:
        """Produce the planned figures; return (and write) the results of those that succeeded.

        A figure that fails is skipped.  When figures were planned and every one
        of them failed, the stage has no output and :class:`PlottingError` is raised.
        """
        cfg = self.ctx.cfg
        specs = plan_figures(outline.get("plotting_plan"), reserved=self._provided_stems())
        if not specs:
            log.info("The outline plans no figures: nothing to plot.")
            return self._write_outputs([])

        # Resolved here so that an unusable plotting model fails the stage at once instead
        # of failing every figure separately.
        self._vision = self.ctx.client(cfg.models.plotting).capabilities.image_input
        if not self._vision:
            log.warning(
                "The plotting model '%s' cannot read images: figures are not reviewed by a "
                "critic and are planned without reference images.",
                cfg.models.plotting,
            )
        self._image_client = self._resolve_image_client(specs)
        self._library = FewShotLibrary.from_env()
        self._library.load(spec.task for spec in specs)

        workers = max(1, min(cfg.figure_workers, len(specs)))
        log.info("Plotting %d figure(s) with %d worker(s).", len(specs), workers)
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="figure")
        interrupted = False
        try:
            outcomes = list(pool.map(self._figure_or_skip, specs))
        except KeyboardInterrupt:
            # Figures in flight stop at their next model call instead of running to the end.
            interrupted = True
            self.ctx.harness.cancel()
            raise
        finally:
            # After an error or an interrupt, figures that have not started are dropped.  An
            # interrupt is not held up by the calls in flight: they end on their own and,
            # the run being cancelled, start nothing new.
            pool.shutdown(wait=not interrupted, cancel_futures=True)
        finished = [outcome for outcome in outcomes if isinstance(outcome, _Finished)]
        skipped = [outcome for outcome in outcomes if isinstance(outcome, _Skipped)]
        results = self._write_outputs(finished)

        if not skipped:
            log.info("Plotting produced all %d figure(s).", len(specs))
            return results
        log.warning(
            "Plotting produced %d of %d figure(s); skipped: %s.",
            len(finished),
            len(specs),
            ", ".join(item.spec.figure_id for item in skipped),
        )
        if not finished:
            # Losing one figure is tolerated; losing all of them points at something every
            # figure shares (the model, the compute backend), which the caller must see.
            shown = "; ".join(
                f"{item.spec.figure_id}: {_one_line(item.reason, _REASON_CHARS)}"
                for item in skipped[:_REASONS_SHOWN]
            )
            more = len(skipped) - _REASONS_SHOWN
            raise PlottingError(
                f"none of the {len(specs)} planned figure(s) could be produced ({shown}"
                + (f"; and {more} more)" if more > 0 else ")")
            )
        return results

    # -- setup -----------------------------------------------------------------

    def _provided_stems(self) -> set[str]:
        """File stems that generated figures must not take: the provided figures in use."""
        provided = self.ctx.paths.provided_figures_dir
        if not self.ctx.cfg.uses_provided_figures or not provided.is_dir():
            return set()
        return {path.stem for path in provided.iterdir() if path.is_file()}

    def _resolve_image_client(self, specs: Sequence[FigureSpec]) -> LLMClient | None:
        """The client that renders diagrams as images, when one is configured and able to."""
        model = self.ctx.cfg.models.image
        if not model or all(spec.task != "diagram" for spec in specs):
            return None
        try:
            client = self.ctx.client(model)
        except LLMError as exc:
            log.warning(
                "The image model '%s' is unavailable (%s): diagrams are drawn with "
                "matplotlib code instead.",
                model,
                exc,
            )
            return None
        if not client.capabilities.image_generation:
            log.warning(
                "The image model '%s' cannot generate images: diagrams are drawn with "
                "matplotlib code instead.",
                model,
            )
            return None
        return client

    # -- one figure --------------------------------------------------------------

    def _figure_or_skip(self, spec: FigureSpec) -> _Finished | _Skipped:
        """Process one figure; a failure is logged and noted, and costs only this figure."""
        try:
            return self._process(spec)
        except Exception as exc:
            if stops_the_run(exc):
                # The run was stopped, or every other figure would fail the same way: this
                # is not one figure's problem, so it is not recorded as a skipped figure.
                raise
            log.warning("Figure '%s' failed and is skipped: %s", spec.figure_id, _describe(exc))
            log.debug("Traceback for figure '%s':", spec.figure_id, exc_info=True)
            harness = self.ctx.harness
            harness.note(
                f"Figure '{spec.figure_id}' ({spec.task}) was skipped: {_failure_text(exc)}",
                stage=STAGE,
                agent=AGENT,
                kind=_failure_kind(exc),
                title=f"Figure skipped: {spec.figure_id}",
            )
            # The outline still plans this figure; its page tells later stages that no
            # file exists, so the writer does not reference one.
            harness.write_page(
                f"wiki/figures/{spec.figure_id}.md",
                f"# {spec.title or spec.figure_id}\n\n"
                f"This planned {spec.task} was not produced: {_describe(exc)}\n\n"
                "No image file exists for it, so the paper cannot include it.",
                {
                    "title": spec.title or spec.figure_id,
                    "summary": f"Planned {spec.task} that was not produced: no file to include.",
                    "kind": "figure",
                    "figure_id": spec.figure_id,
                    "status": "failed",
                },
            )
            harness.record(
                "figure",
                stage=STAGE,
                agent=AGENT,
                figure_id=spec.figure_id,
                status="failed",
                error=_describe(exc),
            )
            return _Skipped(spec, _describe(exc))

    def _process(self, spec: FigureSpec) -> _Finished:
        log.info("Figure '%s': planning a %s (%s).", spec.figure_id, spec.task, spec.aspect_ratio)
        material = self._material(spec)
        examples = self._retrieve_examples(spec, material)
        description = self._plan(spec, material, examples)
        description = self._style(spec, material, description)
        renderer, png = self._first_render(spec, description)

        history: list[dict[str, Any]] = []
        rounds = self.ctx.cfg.plotting_max_critic_rounds if self._vision else 0
        for round_index in range(rounds):
            try:
                critique = self._critique(spec, material, description, png, renderer)
            except LLMError as exc:
                log.warning(
                    "Figure '%s': critic round %d failed (%s); keeping the current render.",
                    spec.figure_id,
                    round_index + 1,
                    exc,
                )
                break
            suggestions = critique.critic_suggestions.strip()
            if _is_no_change(suggestions):
                # Only an accepted figure is recorded with this text: sent along with a
                # revision it would tell the code model that nothing needs to change.
                suggestions = ""
            revised = _strip_description_label(critique.revised_description)
            if not critique.needs_revision or not revised or _is_no_change(revised):
                history.append(
                    _history_entry(round_index, suggestions or NO_CHANGES, description, "accepted")
                )
                log.info(
                    "Figure '%s': accepted in critic round %d.", spec.figure_id, round_index + 1
                )
                break
            try:
                revised_png = renderer.render(revised, suggestions=suggestions)
            except Exception as exc:
                if stops_the_run(exc):
                    raise
                # Whatever broke, the figure already has a valid render: a revision is an
                # improvement to try, never a reason to lose the figure.
                history.append(_history_entry(round_index, suggestions, revised, "render_failed"))
                log.warning(
                    "Figure '%s': the revision of critic round %d could not be rendered (%s); "
                    "keeping the previous render.",
                    spec.figure_id,
                    round_index + 1,
                    _describe(exc),
                )
                log.debug("Traceback for figure '%s':", spec.figure_id, exc_info=True)
                self.ctx.harness.note(
                    f"Figure '{spec.figure_id}' ({spec.task}): the description revised by the "
                    "critic could not be rendered, so the previous render was kept: "
                    f"{_failure_text(exc)}",
                    stage=STAGE,
                    agent=AGENT,
                    kind=_failure_kind(exc),
                    title=f"Figure revision not rendered: {spec.figure_id}",
                )
                break
            description, png = revised, revised_png
            history.append(_history_entry(round_index, suggestions, revised, "revised"))
            log.info("Figure '%s': revised in critic round %d.", spec.figure_id, round_index + 1)

        caption = self._caption(spec, material, description, png)
        return _Finished(
            spec=spec,
            description=description,
            caption=caption,
            critic_history=history,
            png=png,
            method=renderer.method,
            examples=len(examples),
        )

    def _drawn_with_code(self, spec: FigureSpec) -> bool:
        return spec.task == "plot" or self._image_client is None

    def _material(self, spec: FigureSpec) -> str:
        """The source files named by the figure's ``data_source``, each in its own tag."""
        paths = self.ctx.paths
        source = spec.data_source.lower()
        wants_idea = "idea" in source
        wants_log = "experiment" in source
        if not (wants_idea or wants_log):
            # "both", or a value the outline should not have produced: use everything.
            wants_idea = wants_log = True
        sections: list[str] = []
        if wants_idea:
            sections.append(f"<idea>\n{read_text(paths.idea).strip()}\n</idea>")
        if wants_log:
            body = read_text(paths.experimental_log).strip()
            sections.append(f"<experimental_log>\n{body}\n</experimental_log>")
        return "\n\n".join(sections)

    def _generate_text(self, parts: Part | Sequence[Part], *, system: str, label: str) -> str:
        response = self.ctx.harness.generate(
            parts,
            agent=AGENT,
            stage=STAGE,
            model=self.ctx.cfg.models.plotting,
            system=system,
            parser=_non_empty,
            memory=MemoryPolicy.none(),
            label=label,
        )
        return response.parsed

    def _retrieve_examples(
        self, spec: FigureSpec, material: str
    ) -> list[tuple[dict[str, Any], Image]]:
        """Few-shot examples for this figure: pool items with their reference images."""
        pool = self._library.candidates(spec.task)
        if not pool or not self._vision:
            return []
        cfg = self.ctx.cfg
        try:
            response = self.ctx.harness.generate(
                prompts.retrieval_request(
                    spec.task,
                    material=material,
                    intent=spec.intent,
                    candidates=pool,
                    max_content_chars=CANDIDATE_CONTENT_CHARS,
                ),
                agent=AGENT,
                stage=STAGE,
                model=cfg.models.fast,
                effort=cfg.models.fast_effort,
                system=prompts.retriever_system(spec.task),
                schema=FewShotSelection,
                memory=MemoryPolicy.none(),
                label=LABEL_RETRIEVE,
            )
        except LLMError as exc:
            log.warning(
                "Figure '%s': few-shot retrieval failed (%s); planning zero-shot.",
                spec.figure_id,
                exc,
            )
            return []
        by_id = {str(item["id"]): item for item in pool}
        examples: list[tuple[dict[str, Any], Image]] = []
        for ref_id in dict.fromkeys(response.parsed.selected_ids):
            item = by_id.get(ref_id)
            image = self._library.reference_image(spec.task, item) if item else None
            if item is not None and image is not None:
                examples.append((item, image))
            if len(examples) == FEW_SHOT_EXAMPLES:
                break
        log.info("Figure '%s': %d few-shot example(s).", spec.figure_id, len(examples))
        return examples

    def _plan(
        self, spec: FigureSpec, material: str, examples: Sequence[tuple[dict[str, Any], Image]]
    ) -> str:
        parts: list[Part] = []
        for index, (item, image) in enumerate(examples, start=1):
            parts += [Text(prompts.planner_example(spec.task, index, item)), image]
        parts.append(
            Text(
                prompts.planner_request(
                    spec.task,
                    material=material,
                    intent=spec.intent,
                    drawn_with_code=self._drawn_with_code(spec),
                )
            )
        )
        reply = self._generate_text(
            parts, system=prompts.planner_system(spec.task), label=LABEL_PLAN
        )
        return _strip_description_label(reply)

    def _style(self, spec: FigureSpec, material: str, description: str) -> str:
        guide = self._library.style_guide(spec.task) or prompts.default_style_guide(spec.task)
        reply = self._generate_text(
            prompts.stylist_request(
                spec.task,
                description=description,
                style_guide=guide,
                material=material,
                intent=spec.intent,
                drawn_with_code=self._drawn_with_code(spec),
            ),
            system=prompts.stylist_system(spec.task),
            label=LABEL_STYLE,
        )
        return _strip_description_label(reply)

    def _first_render(self, spec: FigureSpec, description: str) -> tuple[_Renderer, bytes]:
        if spec.task == "diagram" and self._image_client is not None:
            model = self.ctx.cfg.models.image
            assert model is not None
            image_renderer = _ImageModelRenderer(self.ctx, spec, self._image_client, model)
            try:
                return image_renderer, image_renderer.render(description)
            except Exception as exc:
                if isinstance(exc, RunCancelled):
                    raise
                # The code path below can draw any diagram, so no failure of the image
                # model (its provider's SDK included) is allowed to cost the figure.
                reason = (
                    str(exc)
                    if isinstance(exc, FigureRenderError)
                    else f"the image model '{model}' failed ({_describe(exc)})"
                )
                log.warning(
                    "Figure '%s': %s; drawing the diagram with matplotlib code instead.",
                    spec.figure_id,
                    reason,
                )
                log.debug("Traceback for figure '%s':", spec.figure_id, exc_info=True)
        code_renderer = _CodeRenderer(self.ctx, spec)
        return code_renderer, code_renderer.render(description)

    def _critique(
        self, spec: FigureSpec, material: str, description: str, png: bytes, renderer: _Renderer
    ) -> FigureCritique:
        response = self.ctx.harness.generate(
            [
                Text(prompts.critic_intro(spec.task)),
                Image(data=png, media_type="image/png"),
                Text(
                    prompts.critic_request(
                        spec.task,
                        description=description,
                        material=material,
                        intent=spec.intent,
                        drawn_with_code=renderer.uses_code,
                    )
                ),
            ],
            agent=AGENT,
            stage=STAGE,
            model=self.ctx.cfg.models.plotting,
            system=prompts.critic_system(spec.task),
            schema=FigureCritique,
            memory=MemoryPolicy.none(),
            label=LABEL_CRITIC,
        )
        return response.parsed

    def _caption(self, spec: FigureSpec, material: str, description: str, png: bytes) -> str:
        parts: list[Part] = [
            Text(
                prompts.caption_request(
                    spec.task, material=material, intent=spec.intent, description=description
                )
            )
        ]
        if self._vision:
            parts.insert(0, Image(data=png, media_type="image/png"))
        try:
            response = self.ctx.harness.generate(
                parts,
                agent=AGENT,
                stage=STAGE,
                model=self.ctx.cfg.models.plotting,
                system=prompts.CAPTION_SYSTEM,
                parser=_clean_caption,
                memory=MemoryPolicy.none(),
                label=LABEL_CAPTION,
            )
        except LLMError as exc:
            fallback = spec.title or spec.objective or spec.figure_id
            log.warning(
                "Figure '%s': no caption could be generated (%s); using its planned title.",
                spec.figure_id,
                exc,
            )
            return fallback
        return response.parsed

    # -- outputs -----------------------------------------------------------------

    def _write_outputs(self, finished: Sequence[_Finished]) -> list[dict[str, Any]]:
        """Write the images, ``plotting_results.json``, ``figures/info.json`` and wiki pages."""
        paths = self.ctx.paths
        results: list[dict[str, Any]] = []
        info: list[dict[str, str]] = []
        for item in finished:
            spec = item.spec
            name = f"{spec.figure_id}.png"
            target = paths.generated_figures_dir / name
            _write_bytes(target, item.png)
            _write_bytes(paths.figures_dir / name, item.png)
            image_path = target.relative_to(paths.base_dir).as_posix()
            results.append(
                {
                    "figure_id": spec.figure_id,
                    "title": spec.title,
                    "task_name": spec.task,
                    "description": item.description,
                    "caption": item.caption,
                    "aspect_ratio": spec.aspect_ratio,
                    "critic_history": item.critic_history,
                    "image_path": image_path,
                }
            )
            info.append({"name": name, "caption": item.caption})
            self._write_wiki_page(item, name)
            self.ctx.harness.record(
                "figure",
                stage=STAGE,
                agent=AGENT,
                figure_id=spec.figure_id,
                status="ok",
                file=image_path,
                method=item.method,
                critic_rounds=len(item.critic_history),
            )
        write_json(paths.plotting_results, results)
        write_json(paths.figures_info, self._merged_figures_info(info))
        return results

    def _merged_figures_info(self, generated: list[dict[str, str]]) -> list[dict[str, Any]]:
        """``figures/info.json``: the figures the user provided, then the generated ones.

        Entries left by an earlier plotting run are dropped; they are recognised
        by their file in the generated-figures directory.
        """
        paths = self.ctx.paths
        existing: object = []
        if paths.figures_info.is_file():
            try:
                existing = json.loads(paths.figures_info.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("Ignoring unreadable %s (%s).", paths.figures_info, exc)
        if not isinstance(existing, list):
            log.warning("Ignoring %s: it does not contain a list.", paths.figures_info)
            existing = []
        new_names = {entry["name"] for entry in generated}
        kept = [
            entry
            for entry in existing
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and entry["name"] not in new_names
            and not (paths.generated_figures_dir / Path(entry["name"]).name).is_file()
        ]
        return kept + generated

    def _write_wiki_page(self, item: _Finished, name: str) -> None:
        spec = item.spec
        if item.critic_history:
            review = "; ".join(
                f"round {entry['round'] + 1}: {entry['outcome'].replace('_', ' ')}"
                for entry in item.critic_history
            )
        else:
            review = "no critic round was recorded"
        if item.examples:
            planning = f"planned with {item.examples} PaperBanana few-shot example(s)"
        else:
            planning = "planned zero-shot"
        body = "\n".join(
            [
                f"# {spec.title or spec.figure_id}",
                "",
                f"- File: `figures/{name}`, as referenced from the LaTeX document",
                f"- Kind: {spec.task}, aspect ratio {spec.aspect_ratio}",
                f"- Produced by: {item.method}; {planning}",
                f"- Critic review: {review}",
                "",
                "## Caption",
                "",
                item.caption,
                "",
                "## Description",
                "",
                item.description,
            ]
        )
        self.ctx.harness.write_page(
            f"wiki/figures/{spec.figure_id}.md",
            body,
            {
                "title": spec.title or spec.figure_id,
                "summary": _one_line(f"{spec.task} in figures/{name}: {item.caption}", 200),
                "kind": "figure",
                "figure_id": spec.figure_id,
                "status": "produced",
                "file": f"figures/{name}",
            },
        )


def _failure_kind(exc: Exception) -> str:
    """Figure code that kept failing is a lesson for later runs; other failures are not."""
    return "lesson" if isinstance(exc, FigureRenderError) else "observation"


def _describe(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


def _failure_text(exc: Exception) -> str:
    """What went wrong, with the failing script's output when there is one."""
    detail = exc.detail if isinstance(exc, FigureRenderError) else ""
    return f"{_describe(exc)}\n\n{detail}" if detail else _describe(exc)


def _history_entry(
    round_index: int, suggestions: str, revised_description: str, outcome: str
) -> dict[str, Any]:
    return {
        "round": round_index,
        "suggestions": suggestions,
        "revised_description": revised_description,
        "outcome": outcome,
    }
