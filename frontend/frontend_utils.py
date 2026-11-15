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

"""Everything the Streamlit demo does that needs no Streamlit.

Building the ``paper-orchestra write`` command, preparing a run directory,
running the pipeline as a child process that can be stopped cleanly, reading
its progress from ``state.json``, browsing the run's wiki memory, packaging
the results and sending the notification e-mail.
"""

from __future__ import annotations

import atexit
import ctypes
import io
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import smtplib
import subprocess
import sys
import threading
import weakref
import zipfile
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.context import RunPaths
from paper_orchestra.memory import Wiki
from paper_orchestra.memory.wiki import INDEX_FILE, NOTES_DIR, WIKI_DIR

log = logging.getLogger("paper_orchestra.frontend")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
TEMPLATES_DIR = REPO_ROOT / "templates"
#: Overrides where demo runs are written (default: ``demo_logs`` next to the repository).
DEMO_DIR_ENV = "PAPER_ORCHESTRA_DEMO_DIR"

MODEL_CHOICES = (
    DEFAULT_MODEL,
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
)
#: Image models for diagrams.  Without one, diagrams are drawn by model-written plotting code.
IMAGE_MODEL_CHOICES = ("gemini-3-pro-image-preview",)
TEMPLATE_CHOICES = ("cvpr2025", "iclr2025")
COMPUTE_CHOICES = ("local", "modal")
DREAM_CHOICES = ("stages", "end", "off")

IDEA_FILENAME = "idea_sparse.md"
EXPERIMENTAL_LOG_FILENAME = "experimental_log.md"
PIPELINE_LOG_NAME = "pipeline.log"

#: Seconds between two refreshes of the page while a run is in progress.
POLL_SECONDS = 2.0
#: A mail server that does not answer must not keep a finished run "in progress".
SMTP_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    """What the sidebar selects for one run."""

    run_dir: Path
    template: str = TEMPLATE_CHOICES[0]
    model: str = DEFAULT_MODEL
    use_plotting: bool = True
    image_model: str | None = None
    compute: str = "local"
    memory: bool = True
    dream: str = "stages"

    def __post_init__(self) -> None:
        if self.compute not in COMPUTE_CHOICES:
            raise ValueError(f"unknown compute backend {self.compute!r}; use local or modal")
        if self.dream not in DREAM_CHOICES:
            raise ValueError(f"unknown dream mode {self.dream!r}; use off, end or stages")

    @property
    def paths(self) -> RunPaths:
        return RunPaths(Path(self.run_dir), IDEA_FILENAME, EXPERIMENTAL_LOG_FILENAME)


def build_write_command(
    options: RunOptions, *, python: str = sys.executable, templates_dir: Path = TEMPLATES_DIR
) -> list[str]:
    """The ``python -m paper_orchestra write ...`` invocation for ``options``.

    The selected model takes every role (``--model`` covers the ones without
    a flag of their own, so a run never needs a second provider's
    credentials).  An image model is only passed when one was chosen: without
    it the pipeline draws diagrams from model-written code.
    """
    paths = options.paths
    command = [
        python,
        "-m",
        "paper_orchestra",
        "write",
        "--raw-materials-dir",
        str(paths.materials_dir),
        "--output-dir",
        str(paths.base_dir),
        "--template",
        str(Path(templates_dir) / options.template),
        "--idea-filename",
        IDEA_FILENAME,
        "--experimental-log-filename",
        EXPERIMENTAL_LOG_FILENAME,
        "--model",
        options.model,
        "--writer-model",
        options.model,
        "--reflection-model",
        options.model,
        "--compute",
        options.compute,
    ]
    if options.use_plotting:
        command += ["--use-plotting", "--plotting-model", options.model]
        if options.image_model:
            command += ["--image-model", options.image_model]
    else:
        command.append("--no-use-plotting")
    if options.memory:
        command += ["--memory", "--dream", options.dream]
    else:
        command.append("--no-memory")
    return command


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


def new_run_directory(now: datetime | None = None) -> Path:
    """``<demo dir>/paper_orchestra_run_<timestamp>_<random id>`` (not created yet).

    The random part keeps two browser sessions that start in the same second
    from writing their inputs into, and running two pipelines in, one folder.
    """
    base = os.environ.get(DEMO_DIR_ENV)
    root = Path(base).expanduser() if base else REPO_ROOT.parent / "demo_logs"
    stamp = f"{(now or datetime.now()):%Y%m%d_%H%M%S}"
    return root / f"paper_orchestra_run_{stamp}_{secrets.token_hex(3)}"


_UNSAFE_FILE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_figure_name(name: str, taken: Sequence[str] = ()) -> str:
    """A browser-supplied figure name as a plain file name that is not in ``taken``.

    Directories are dropped, characters that are awkward in a LaTeX
    ``\\includegraphics`` path (spaces, ``%``, ``#`` ...) become ``_``, and a
    name already used by another figure gets a number, so no upload replaces
    another one.
    """
    base = Path(str(name).replace("\\", "/")).name
    cleaned = _UNSAFE_FILE_NAME_CHARS.sub("_", base).lstrip(".")
    stem, suffix = os.path.splitext(cleaned)
    if not any(character.isalnum() for character in stem) or cleaned.lower() == "info.json":
        raise ValueError(f"unusable figure file name: {name!r}")
    # Compared without case: the run directory may be on a case-insensitive file system.
    used = {other.lower() for other in taken}
    candidate, number = cleaned, 1
    while candidate.lower() in used:
        number += 1
        candidate = f"{stem}_{number}{suffix}"
    return candidate


def write_raw_materials(
    run_dir: Path, idea: str, experimental_log: str, figures: Sequence[Mapping[str, Any]] = ()
) -> Path:
    """Write the inputs of a run where the pipeline expects them.

    ``figures`` entries carry ``name``, ``caption`` and ``file`` (a path, or
    an upload exposing ``getvalue()``).  Each is stored under
    :func:`safe_figure_name`, and their captions go to ``figures/info.json``;
    without it the writer would never learn about them.  Returns the
    raw-materials directory.
    """
    paths = RunPaths(Path(run_dir), IDEA_FILENAME, EXPERIMENTAL_LOG_FILENAME)
    paths.provided_figures_dir.mkdir(parents=True, exist_ok=True)
    paths.idea.write_text(idea, encoding="utf-8")
    paths.experimental_log.write_text(experimental_log, encoding="utf-8")

    captions: dict[str, str] = {}
    for figure in figures:
        name = safe_figure_name(figure["name"], taken=list(captions))
        target = paths.provided_figures_dir / name
        source = figure["file"]
        if isinstance(source, (str, Path)):
            shutil.copyfile(source, target)
        else:
            target.write_bytes(source.getvalue())
        captions[name] = str(figure.get("caption") or "")
    if captions:
        info = [{"name": name, "caption": caption} for name, caption in captions.items()]
        (paths.provided_figures_dir / "info.json").write_text(
            json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return paths.materials_dir


def load_example(name: str = "cvpr_example.json") -> dict[str, Any]:
    """An example from ``frontend/examples``: idea, log and (optionally) one figure."""
    data = json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
    example: dict[str, Any] = {
        "idea": data.get("idea", ""),
        "log": data.get("log", ""),
        "figures": [],
    }
    figure_path = data.get("figure_path")
    if figure_path and (EXAMPLES_DIR / figure_path).is_file():
        example["figures"].append(
            {
                "file": str(EXAMPLES_DIR / figure_path),
                "name": figure_path,
                "caption": data.get("figure_caption", ""),
            }
        )
    return example


# ---------------------------------------------------------------------------
# The pipeline process
# ---------------------------------------------------------------------------

_SECRET_NAME_PARTS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD")
_NOT_SECRET_SUFFIXES = ("_FILE", "_PATH", "_DIR", "_URL")
_MIN_SECRET_LENGTH = 8
SECRET_MASK = "***"


def secret_values(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Values of the environment variables that hold credentials, longest first.

    Judged by name (``*_API_KEY``, ``*TOKEN*``, ``*PASSWORD*`` ...).  Short
    values are left out: masking them would blank ordinary words in the log.
    """
    values = set()
    for name, value in environ.items():
        upper = name.upper()
        named_secret = upper.endswith("_KEY") or any(part in upper for part in _SECRET_NAME_PARTS)
        if (
            named_secret
            and not upper.endswith(_NOT_SECRET_SUFFIXES)
            and len(value) >= _MIN_SECRET_LENGTH
        ):
            values.add(value)
    return tuple(sorted(values, key=lambda value: (-len(value), value)))


def mask_secrets(text: str, secrets_to_hide: Sequence[str]) -> str:
    """``text`` with every given secret replaced by ``***``."""
    for secret in secrets_to_hide:
        text = text.replace(secret, SECRET_MASK)
    return text


_PR_SET_PDEATHSIG = 1
_active_runs: weakref.WeakSet[PipelineRun] = weakref.WeakSet()
_cleanup_lock = threading.Lock()
_cleanup_registered = False


def _linux_death_signal() -> Callable[[], None] | None:
    """A ``preexec_fn`` that has the kernel kill the child when its parent thread dies.

    Linux only, best effort: it covers the case no exit handler can, the
    Streamlit server being killed outright.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        prctl = ctypes.CDLL(None).prctl
    except (OSError, AttributeError):
        return None

    def set_death_signal() -> None:
        prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)

    return set_death_signal


def _isolation_options() -> dict[str, Any]:
    """Popen options that give the child its own process group, so the whole tree can be stopped."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    options: dict[str, Any] = {"start_new_session": True}
    death_signal = _linux_death_signal()
    if death_signal is not None:
        options["preexec_fn"] = death_signal
    return options


def _signal_tree(process: subprocess.Popen[str], *, force: bool) -> None:
    """Interrupt (or kill) the child and everything it started."""
    try:
        if os.name == "nt":
            if force:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # The child leads its own session, so its pid is also its process group id.
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGINT)
    except OSError as exc:
        # Usually ProcessLookupError: the process group is already gone.
        log.debug("Could not signal the pipeline process %d: %s", process.pid, exc)


def _stop_active_runs() -> None:
    for run in list(_active_runs):
        run.stop(grace=2.0)


class PipelineRun:
    """One pipeline child process, its captured output and its outcome.

    The process is started from, and its output read by, a single worker
    thread that lives as long as the process: the Linux parent-death signal is
    tied to the thread that forked the child, so a short-lived thread (such as
    one Streamlit script run) must not be the one to start it.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_path: Path | None = None,
        on_finish: Callable[[PipelineRun], None] | None = None,
        max_log_lines: int = 2000,
    ) -> None:
        self.command = list(command)
        self.cwd = Path(cwd)
        self.log_path = log_path
        #: Set when the process could not be started.
        self.launch_error: str | None = None
        #: Set by :meth:`stop`: the run ended because the user asked for it.
        self.stopped = False
        #: Free-form result of ``on_finish`` (for example what happened to the e-mail).
        self.notification: str | None = None
        self._on_finish = on_finish
        self._lines: deque[str] = deque(maxlen=max_log_lines)
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._launched = threading.Event()
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._run, name="paper-orchestra-run", daemon=True)

    def start(self) -> None:
        """Launch the process; returns once it is running or has failed to start."""
        global _cleanup_registered
        with _cleanup_lock:
            if not _cleanup_registered:
                atexit.register(_stop_active_runs)
                _cleanup_registered = True
            _active_runs.add(self)
        self._thread.start()
        self._launched.wait()

    def _run(self) -> None:
        try:
            environment = dict(os.environ)
            try:
                self._process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=self.cwd,
                    # Unbuffered output, so the log tail follows the pipeline in real time.
                    env={**environment, "PYTHONUNBUFFERED": "1"},
                    **_isolation_options(),
                )
            except (OSError, ValueError) as exc:
                self.launch_error = f"The pipeline could not be started: {exc}"
                return
            finally:
                self._launched.set()
            self._pump_output(self._process, secret_values(environment))
            self._process.wait()
        finally:
            if self._on_finish is not None:
                try:
                    self._on_finish(self)
                except Exception:
                    log.exception("The completion handler of the pipeline run failed.")
            self._finished.set()
            _active_runs.discard(self)

    def _pump_output(self, process: subprocess.Popen[str], hidden: Sequence[str] = ()) -> None:
        """Read the child's output to its end, keeping the tail in memory and all of it on disk.

        Reading never stops early: a full pipe would block the pipeline.  A log
        file that cannot be written is given up on instead.  The values in
        ``hidden`` (the credentials the child inherited) are masked first: the
        tail is shown in the browser, and a library may echo a key in an error.
        """
        assert process.stdout is not None
        log_file = None
        try:
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = self.log_path.open("a", encoding="utf-8")
        except OSError as exc:
            log.warning("The pipeline log %s cannot be written: %s", self.log_path, exc)
        try:
            for line in process.stdout:
                line = mask_secrets(line, hidden)
                with self._lock:
                    self._lines.append(line)
                if log_file is None:
                    continue
                try:
                    log_file.write(line)
                    log_file.flush()
                except OSError as exc:
                    log.warning("Stopped writing the pipeline log %s: %s", self.log_path, exc)
                    log_file = None
        finally:
            if log_file is not None:
                log_file.close()

    @property
    def running(self) -> bool:
        return not self._finished.is_set()

    @property
    def returncode(self) -> int | None:
        """Exit status of the process; ``None`` while it runs or when it never started."""
        return None if self._process is None else self._process.poll()

    def log_tail(self, lines: int = 200) -> str:
        with self._lock:
            return "".join(list(self._lines)[-lines:])

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run (and its completion handler) is over; ``False`` on timeout."""
        return self._finished.wait(timeout)

    def stop(self, grace: float = 10.0) -> None:
        """Interrupt the process group, then kill whatever is left after ``grace`` seconds.

        The interrupt lets the pipeline record the run as interrupted in
        ``state.json``; the kill covers children that ignore it.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return
        self.stopped = True
        _signal_tree(process, force=False)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            log.warning("The pipeline did not stop within %.0f s; killing it.", grace)
        # Whether or not the leader has exited, nothing of its group may outlive the stop.
        _signal_tree(process, force=True)
        self._finished.wait(timeout=grace)


def run_succeeded(run: PipelineRun, run_dir: Path) -> bool:
    """A run succeeded when the process exited cleanly and the final PDF is there."""
    return run.returncode == 0 and RunPaths(Path(run_dir)).final_pdf.is_file()


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

#: Pipeline stages in execution order, with the labels the timeline shows.
STAGE_LABELS: dict[str, str] = {
    "ingest": "Preparing Inputs",
    "outline": "Outline Generation",
    "literature": "Literature Review",
    "plotting": "Figure Generation",
    "section_writing": "Section Writing",
    "refinement": "Content Refinement",
    "finalize": "Finalizing",
}
_STAGE_STATUSES = ("pending", "running", "completed", "failed", "interrupted", "skipped")
#: How a finished run is recorded: the pipeline writes ``completed``, its spec says ``succeeded``.
_RUN_STATUSES = {
    "completed": "completed",
    "succeeded": "completed",
    "failed": "failed",
    "interrupted": "interrupted",
}


@dataclass(frozen=True)
class StageProgress:
    name: str
    label: str
    status: str = "pending"
    attempts: int = 0
    seconds: float = 0.0
    error: str | None = None
    #: Result carried over from an earlier, interrupted run.
    reused: bool = False
    #: The stage failed but the run went on without its output.
    tolerated: bool = False


@dataclass(frozen=True)
class RunProgress:
    #: ``starting`` (no state yet), ``running``, ``completed``, ``failed`` or ``interrupted``.
    status: str
    stages: tuple[StageProgress, ...]
    error: str | None = None

    @property
    def done(self) -> int:
        """Number of stages that need no more work (completed or skipped)."""
        return sum(stage.status in ("completed", "skipped") for stage in self.stages)

    @property
    def fraction(self) -> float:
        """Share of the stages that are done, for a progress bar."""
        return self.done / len(self.stages) if self.stages else 0.0

    @property
    def active(self) -> tuple[str, ...]:
        """Labels of the stages running right now (literature and plotting run together)."""
        return tuple(stage.label for stage in self.stages if stage.status == "running")


def read_run_state(run_dir: Path) -> dict[str, Any] | None:
    """The run's ``state.json``, or ``None`` when it is absent or not readable yet."""
    path = RunPaths(Path(run_dir)).state
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    return state if isinstance(state, dict) and isinstance(state.get("stages"), dict) else None


def _number(value: Any) -> float:
    """A count or duration from ``state.json``; 0 for anything that is not a number."""
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _text(value: Any) -> str | None:
    """An error message from ``state.json`` as text; ``None`` when there is none."""
    return str(value) if value else None


def read_progress(
    run_dir: Path, *, use_plotting: bool = True, process_running: bool = True
) -> RunProgress:
    """Stage-by-stage progress of a run, read from its ``state.json``.

    Before the pipeline has written a state file every expected stage is
    pending.  With ``process_running=False`` anything still marked as running
    is reported as interrupted: the process is gone, so a killed run cannot
    look as if it were still working.
    """
    state = read_run_state(run_dir)
    if state is None:
        names = [name for name in STAGE_LABELS if use_plotting or name != "plotting"]
        stages = tuple(StageProgress(name, STAGE_LABELS[name]) for name in names)
        return RunProgress("starting" if process_running else "interrupted", stages)

    def settled(status: Any) -> str:
        status = status if status in _STAGE_STATUSES else "pending"
        return "interrupted" if status == "running" and not process_running else status

    records = state["stages"]
    ordered = [name for name in STAGE_LABELS if name in records]
    ordered += [name for name in records if name not in STAGE_LABELS]
    stages = []
    for name in ordered:
        record = records[name] if isinstance(records[name], dict) else {}
        stages.append(
            StageProgress(
                name=name,
                label=STAGE_LABELS.get(name, name.replace("_", " ").title()),
                status=settled(record.get("status")),
                attempts=int(_number(record.get("attempts"))),
                seconds=_number(record.get("seconds")),
                error=_text(record.get("error")),
                reused=bool(record.get("reused")),
                tolerated=bool(record.get("tolerated")),
            )
        )
    recorded = state.get("status")
    status = _RUN_STATUSES.get(recorded) if isinstance(recorded, str) else None
    if status is None:
        status = "running" if process_running else "interrupted"
    return RunProgress(status, tuple(stages), _text(state.get("error")))


# ---------------------------------------------------------------------------
# Wiki memory (read-only)
# ---------------------------------------------------------------------------

_HTML_COMMENT_LINE = re.compile(r"^\s*<!--.*-->\s*$")
#: The "!" that turns a Markdown link (inline or reference style) into an embedded image.
_IMAGE_EMBED_MARK = re.compile(r"!(?=\[[^\]\n]*\][(\[])")


@dataclass(frozen=True)
class WikiEntry:
    path: str
    title: str
    summary: str
    kind: str
    #: ``literature``, ``reviews``, ... for wiki pages; ``notes`` for pending notes.
    section: str

    @property
    def label(self) -> str:
        return f"{self.section} / {self.title}"


def open_run_wiki(run_dir: Path) -> Wiki | None:
    """The project wiki of a run, or ``None`` when the run has no memory (yet).

    The wiki is opened as a viewer: looking at a run never creates or changes
    anything in its memory directory.
    """
    memory_dir = RunPaths(Path(run_dir)).memory_dir
    if not (memory_dir / INDEX_FILE).is_file():
        return None
    return Wiki(memory_dir, create=False)


def without_embedded_images(markdown: str) -> str:
    """Turn ``![alt](url)`` image embeds into plain ``[alt](url)`` links.

    Memory pages are written by models, partly from web search results.
    Rendering an embed makes the viewer's browser fetch its URL at once; a
    link is only followed when the viewer chooses to.
    """
    return _IMAGE_EMBED_MARK.sub("", markdown)


def memory_index_markdown(wiki: Wiki) -> str:
    """``MEMORY.md`` as the model sees it, minus the marker comments.

    Empty when the index cannot be read right now (the run may be rewriting it).
    """
    try:
        text = wiki.read_text(INDEX_FILE)
    except (ValueError, OSError) as exc:
        log.warning("Could not read %s of %s: %s", INDEX_FILE, wiki.root, exc)
        return ""
    lines = (line for line in text.splitlines() if not _HTML_COMMENT_LINE.match(line))
    return without_embedded_images("\n".join(lines))


def list_wiki_pages(wiki: Wiki, *, include_notes: bool = True) -> list[WikiEntry]:
    """Every wiki page, then (optionally) the notes, each labelled with its section.

    The run rewrites, moves and archives pages while the viewer is looking,
    so each page is read on its own and one that vanished, cannot be decoded
    or is not addressable (:meth:`Wiki.normalize` decides, which also keeps
    links pointing out of the memory directory away) is left out.
    """
    entries = []
    for prefix in (WIKI_DIR, NOTES_DIR) if include_notes else (WIKI_DIR,):
        try:
            files = sorted((wiki.root / prefix).rglob("*.md"))
        except OSError as exc:
            log.warning("Could not list %s: %s", wiki.root / prefix, exc)
            continue
        for file in files:
            path = file.relative_to(wiki.root).as_posix()
            try:
                page = wiki.read(path)
            except (ValueError, OSError):
                continue
            parts = path.split("/")
            if prefix == NOTES_DIR:
                section = "notes" if len(parts) == 2 else "/".join(parts[:-1])
            else:
                section = parts[1] if len(parts) > 2 else "wiki"
            entries.append(WikiEntry(path, page.title, page.summary, page.kind, section))
    return entries


def read_wiki_page(wiki: Wiki, path: str) -> tuple[dict[str, Any], str] | None:
    """``(front matter, body)`` of a page; ``None`` when it cannot be shown (any more).

    The path is validated by :meth:`Wiki.normalize`, so nothing outside the
    memory directory can be opened.
    """
    try:
        page = wiki.read(path)
    except (ValueError, OSError):
        # WikiError (no such page, not addressable) and undecodable bytes are ValueErrors.
        return None
    return page.meta, without_embedded_images(page.body)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

#: Drafts that the final ``main.tex`` supersedes.
_DRAFT_TEX = ("template.tex", "raw_draft_paper.tex")


def final_tex_path(run_dir: Path) -> Path | None:
    """The final LaTeX source of a run, if the run got that far."""
    paths = RunPaths(Path(run_dir))
    for candidate in (paths.final_tex, paths.refinement_dir / "final_refined_paper.tex"):
        if candidate.is_file():
            return candidate
    return None


def latex_source_zip(run_dir: Path) -> bytes | None:
    """A zip of the LaTeX workspace, ready to compile; ``None`` when there is none.

    When the run produced a final paper it is included as ``main.tex`` and
    the intermediate drafts are left out.
    """
    latex_dir = RunPaths(Path(run_dir)).latex_dir
    if not latex_dir.is_dir():
        return None
    final_tex = final_tex_path(run_dir)
    skipped = set(_DRAFT_TEX) | {"main.tex"} if final_tex is not None else set()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(latex_dir.rglob("*")):
            relative = path.relative_to(latex_dir).as_posix()
            if path.is_file() and relative not in skipped:
                archive.write(path, relative)
        if final_tex is not None:
            archive.write(final_tex, "main.tex")
    return buffer.getvalue()


def version_number(version_key: str) -> int:
    """``"v3"`` -> ``3`` (worklog keys of the refinement stage)."""
    digits = "".join(ch for ch in version_key if ch.isdigit())
    return int(digits) if digits else 0


def last_accepted_version(worklog: Mapping[str, Any]) -> str | None:
    """Key of the last accepted iteration in a refinement worklog, ``None`` if all were rejected.

    Each accepted revision builds on the previous one, so the last accepted
    iteration is the version the stage hands on.
    """
    accepted = [
        key
        for key, entry in worklog.items()
        if isinstance(entry, dict) and str(entry.get("outcome", "")).startswith("ACCEPTED")
    ]
    return max(accepted, key=version_number) if accepted else None


def result_attachments(run_dir: Path) -> list[Path]:
    """The files a finished run sends out: the final PDF and the zipped LaTeX source."""
    run_dir = Path(run_dir)
    attachments = []
    final_pdf = RunPaths(run_dir).final_pdf
    if final_pdf.is_file():
        attachments.append(final_pdf)
    archive = latex_source_zip(run_dir)
    if archive is not None and final_tex_path(run_dir) is not None:
        zip_path = run_dir / "latex_source.zip"
        zip_path.write_bytes(archive)
        attachments.append(zip_path)
    return attachments


# ---------------------------------------------------------------------------
# Notification e-mail
# ---------------------------------------------------------------------------


_EMAIL_ADDRESS = re.compile(r"[^@\s,;<>\"']+@[^@\s,;<>\"']+\.[^@\s,;<>\"']+")


def valid_email_address(address: str) -> bool:
    """Whether ``address`` is one plain e-mail address: no list, display name or line break."""
    return bool(_EMAIL_ADDRESS.fullmatch(address))


def send_notification_email(
    recipient_email: str,
    paper_name: str,
    attachments: Sequence[str | Path] | None = None,
) -> tuple[bool, str]:
    """E-mail the results; returns ``(sent, message)``.

    The account comes from ``SMTP_EMAIL`` / ``SMTP_PASSWORD`` (optionally
    ``SMTP_HOST`` and ``SMTP_PORT``, default Gmail over SSL), read when the
    mail is sent.  The password is never logged or put into the message.
    """
    sender_email = os.environ.get("SMTP_EMAIL")
    sender_password = os.environ.get("SMTP_PASSWORD")
    if not sender_email or not sender_password:
        return False, "E-mail is not configured: set SMTP_EMAIL and SMTP_PASSWORD."
    # The address is typed into a public page: it must not be able to add recipients or headers.
    if not valid_email_address(recipient_email):
        return False, "No e-mail was sent: the address given is not a valid e-mail address."
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    try:
        port = int(os.environ.get("SMTP_PORT", "465"))
    except ValueError:
        return False, "SMTP_PORT must be a number."

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = f"Paper Orchestra: {paper_name} is Complete!"
    body = """
    <html>
    <body>
        <h2>Paper Orchestra Notification</h2>
        <p>Great news! Your paper has been successfully generated.</p>
        <p>Check the attachments for the final PDF and LaTeX source.</p>
        <hr>
        <p><small>Generated by Paper Orchestra AI</small></p>
    </body>
    </html>
    """
    message.attach(MIMEText(body, "html"))

    missing = []
    for file_path in map(Path, attachments or ()):
        if not file_path.is_file():
            missing.append(file_path.name)
            continue
        part = MIMEBase("application", "octet-stream")
        part.set_payload(file_path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=file_path.name)
        message.attach(part)

    try:
        with smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        return False, f"The e-mail could not be sent: {type(exc).__name__}: {exc}"
    if missing:
        return True, f"E-mail sent, without the missing attachment(s): {', '.join(missing)}."
    return True, "Email sent successfully!"


def notify_completion(run: PipelineRun, run_dir: Path, recipient_email: str | None) -> None:
    """Completion handler of a run: e-mail the results of a successful run.

    The outcome is stored in ``run.notification`` for the page to show.
    """
    if not recipient_email or run.stopped:
        return
    if not run_succeeded(run, run_dir):
        run.notification = "No e-mail was sent because the run did not produce a paper."
        return
    _, run.notification = send_notification_email(
        recipient_email, "Generated Paper", attachments=result_attachments(run_dir)
    )
