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

"""The ``paper-orchestra`` command line.

Exit codes: 0 on success, 1 when the pipeline (or a check) failed, 2 for a
usage or validation error, 130 when the command was interrupted (Ctrl-C or
SIGTERM).  For ``research advance`` that reads: 0 at a checkpoint, 1 when the
engine failed, 3 when the phase that would run needs a model and none can be
used (nothing was started; the project is as it was), 130 when it was stopped.

The research commands and ``ui`` import what they drive (the project store, the
engine, the web app) when they run, so that every other command starts without
them and without the optional extras.
"""

from __future__ import annotations

import collections
import contextlib
import importlib.util
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Collection, Iterator, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer
from typer.core import TyperCommand

from paper_orchestra import __version__
from paper_orchestra.batch import load_papers, run_batch, summarize
from paper_orchestra.compute.base import ComputeError
from paper_orchestra.config import (
    DEFAULT_MODEL,
    MemoryConfig,
    ModelConfig,
    PipelineConfig,
    load_env,
    resolve_template_dir,
)
from paper_orchestra.log import LOGGER_NAME, setup_logging
from paper_orchestra.modal_batch import run_batch_on_modal
from paper_orchestra.pipeline import prepare_run_dir, run_paper
from paper_orchestra.utils.common import read_json

if TYPE_CHECKING:
    from paper_orchestra.memory.harness import ContextHarness
    from paper_orchestra.memory.wiki import Wiki
    from paper_orchestra.pipeline import PaperResult
    from paper_orchestra.research.model import EditProposal, ResearchPage
    from paper_orchestra.research.project import Project

MODAL_MODULE = "paper_orchestra.modal_app"
MANUAL_DREAM_FOCUS = (
    "Consolidate the pending notes into the wiki pages they belong to, repair what the health "
    "check lists, and keep the standing notes short and durable."
)

#: Flag spellings of the original ``paper_writing_cli.py`` and what they are called now.
LEGACY_OPTIONS = {
    "--raw_materials_dir": "--raw-materials-dir",
    "--latex_template_dir": "--template",
    "--output_dir": "--output-dir",
    "--idea_filename": "--idea-filename",
    "--experimental_log_filename": "--experimental-log-filename",
    "--writer_model_name": "--writer-model",
    "--reflection_model_name": "--reflection-model",
    "--research_cutoff": "--research-cutoff",
    "--plotting_model_name": "--plotting-model",
    "--image_model_name": "--image-model",
    "--plotting_max_critic_rounds": "--plotting-max-critic-rounds",
}
#: The original ``--use_plotting`` took a value; it maps onto an on/off flag pair.
LEGACY_USE_PLOTTING = "--use_plotting"
#: Exit code of a command that was interrupted (Ctrl-C or SIGTERM).
INTERRUPTED = 130
#: Exit code of a research command that started nothing because the phase that would run
#: needs a model and none can be used.  It is no failure of the project: the web server
#: tells such an end of a worker from one that died by this number.
NEEDS_A_MODEL = 3
_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def translate_legacy_args(args: Sequence[str]) -> list[str]:
    """Rewrite the original underscore flags into today's options, leaving the rest alone."""
    out: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg == "--":
            out.append(arg)
            out.extend(args[index:])
            break
        name, equals, value = arg.partition("=")
        if name in LEGACY_OPTIONS:
            out.append(LEGACY_OPTIONS[name] + equals + value)
        elif name == LEGACY_USE_PLOTTING:
            if not equals:
                # ``--use_plotting true``: take the next word when it reads as a boolean,
                # otherwise treat the bare flag as "on".
                if index < len(args) and args[index].lower() in _TRUE | _FALSE:
                    value = args[index]
                    index += 1
                else:
                    value = "true"
            if value.lower() not in _TRUE | _FALSE:
                raise typer.BadParameter(
                    f"expected true or false, got {value!r}", param_hint=LEGACY_USE_PLOTTING
                )
            out.append("--use-plotting" if value.lower() in _TRUE else "--no-use-plotting")
        else:
            out.append(arg)
    return out


class _LegacyFlagsCommand(TyperCommand):
    """A command that also accepts the original CLI's flag spellings (not shown in help)."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        return super().parse_args(ctx, translate_legacy_args(args))


class _Compute(StrEnum):
    local = "local"
    modal = "modal"


class _Dream(StrEnum):
    off = "off"
    end = "end"
    stages = "stages"


class _Effort(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


app = typer.Typer(
    name="paper-orchestra",
    help="From an idea to a research wiki: agents propose sources, hypotheses, experiments "
    "and findings, you decide, and a paper can be written from it. `research` and `ui` are "
    "the workspace; `write` and `batch` turn an idea and an experimental log into a LaTeX "
    "paper directly.",
    no_args_is_help=True,
    add_completion=False,
    # A traceback with local variables could print credentials.
    pretty_exceptions_show_locals=False,
)
memory_app = typer.Typer(help="Inspect and maintain the wiki memory.", no_args_is_help=True)
modal_app = typer.Typer(help="Run the pipeline on Modal.", no_args_is_help=True)
research_app = typer.Typer(
    help="Research projects: give an idea, let agents fill a wiki, decide what stands.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")
app.add_typer(modal_app, name="modal")
app.add_typer(research_app, name="research")


#: Control characters other than the newline and the tab: what a terminal acts on.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _printable(text: object) -> str:
    """``text`` without control characters.

    Titles, summaries, errors and the output of runs are written by models and by the
    code they wrote.  Printed as they are, an escape sequence in one of them could
    rewrite what the terminal shows of the lines around it, in the very list the
    person decides from.
    """
    return _CONTROL.sub("", str(text))


def _say(message: object, *, err: bool = False, fg: str | None = None) -> None:
    """Print a line that may quote a page, an event, an error or a run."""
    if fg is None:
        typer.echo(_printable(message), err=err)
    else:
        typer.secho(_printable(message), fg=fg, err=err)


def _fail(message: str, code: int = 2) -> NoReturn:
    _say(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"paper-orchestra {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log debugging detail.")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    if verbose:
        logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG)


# -- options shared by `write` and `batch` -----------------------------------------

TemplateOption = Annotated[
    str,
    typer.Option(
        "--template",
        "--latex-template-dir",
        help="LaTeX template: a bundled name (for example cvpr2025) or a template directory.",
    ),
]
IdeaFilenameOption = Annotated[
    str, typer.Option("--idea-filename", help="Idea file inside the raw materials.")
]
LogFilenameOption = Annotated[
    str,
    typer.Option(
        "--experimental-log-filename", help="Experimental log file inside the raw materials."
    ),
]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        help="Model for every role that has no flag of its own.",
        show_default=DEFAULT_MODEL,
    ),
]
WriterModelOption = Annotated[
    str | None,
    typer.Option("--writer-model", help="Model for the outline, literature and writing agents."),
]
ReflectionModelOption = Annotated[
    str | None,
    typer.Option("--reflection-model", help="Model for reviewing, refinement and consolidation."),
]
PlottingModelOption = Annotated[
    str | None, typer.Option("--plotting-model", help="Model that writes and critiques figures.")
]
FastModelOption = Annotated[
    str | None,
    typer.Option(
        "--fast-model",
        help="Model for high-volume calls such as paper discovery. "
        "Default: --model when it is given, otherwise the writer model.",
    ),
]
ImageModelOption = Annotated[
    str | None,
    typer.Option(
        "--image-model",
        help="Image-generation model for diagrams. Default: diagrams are drawn with code.",
    ),
]
EffortOption = Annotated[_Effort, typer.Option("--effort", help="Reasoning effort.")]
CutoffOption = Annotated[
    str | None,
    typer.Option(
        "--research-cutoff",
        help="Ignore literature published after this month (YYYY-MM). Default: this month.",
    ),
]
PlottingOption = Annotated[
    bool,
    typer.Option(
        "--plotting/--no-plotting",
        "--use-plotting/--no-use-plotting",
        help="Generate the figures from the experimental log.",
    ),
]
KeepFiguresOption = Annotated[
    bool,
    typer.Option(
        "--keep-provided-figures",
        help="With --plotting, figures in raw_materials/figures are ignored unless this is given.",
    ),
]
CriticRoundsOption = Annotated[
    int,
    typer.Option(
        "--plotting-max-critic-rounds", min=0, help="Self-correction rounds per generated figure."
    ),
]
ComputeOption = Annotated[
    _Compute,
    typer.Option("--compute", help="Where figure code runs and LaTeX compiles."),
]
MemoryOption = Annotated[
    bool, typer.Option("--memory/--no-memory", help="Keep a wiki memory for the run.")
]
LabMemoryOption = Annotated[
    bool,
    typer.Option(
        "--lab-memory/--no-lab-memory",
        help="Read and extend the cross-run memory of process lessons.",
    ),
]
LabMemoryDirOption = Annotated[
    Path | None,
    typer.Option("--lab-memory-dir", help="Cross-run memory directory (default: per user)."),
]
DreamOption = Annotated[
    _Dream,
    typer.Option(
        "--dream",
        help="When a model consolidates memory: never, at the end, or after stages too.",
    ),
]
ResumeOption = Annotated[
    bool,
    typer.Option(
        "--resume/--no-resume", help="Continue an earlier run found in the output directory."
    ),
]
RootInputFolderOption = Annotated[
    Path | None,
    typer.Option(
        "--root-input-folder", help="Folder with one <paper_id>/raw_materials directory per paper."
    ),
]
PaperInfoOption = Annotated[
    Path | None,
    typer.Option(
        "--paper-info",
        help="JSON list of {paper_id, content_folder, pdf_path} instead of a folder.",
    ),
]
PaperCountOption = Annotated[
    int, typer.Option("--n-papers", help="Write only the first N papers (-1: all).")
]


def _role_models(
    model: str | None,
    writer_model: str | None,
    reflection_model: str | None,
    plotting_model: str | None,
    fast_model: str | None,
) -> dict[str, str]:
    """The model of each role, from ``--model`` and the flags of the single roles."""
    base = model or DEFAULT_MODEL
    writer = writer_model or base
    return {
        "writer": writer,
        "reflection": reflection_model or base,
        "plotting": plotting_model or base,
        # The original command line had no flag for this role and kept it with the writer's
        # provider.  Following the writer unless --model says otherwise keeps a run that
        # names its writer from needing a second provider's credentials.
        "fast": fast_model or model or writer,
    }


def build_config(
    *,
    base_dir: Path,
    template: str,
    idea_filename: str = "idea_sparse.md",
    experimental_log_filename: str = "experimental_log.md",
    model: str | None = None,
    writer_model: str | None = None,
    reflection_model: str | None = None,
    plotting_model: str | None = None,
    fast_model: str | None = None,
    image_model: str | None = None,
    effort: str = "high",
    research_cutoff: str | None = None,
    use_plotting: bool = False,
    keep_provided_figures: bool = False,
    plotting_max_critic_rounds: int = 3,
    compute: str = "local",
    memory: bool = True,
    memory_dir: Path | None = None,
    lab_memory: bool = True,
    lab_memory_dir: Path | None = None,
    dream: str = "stages",
    resume: bool = True,
) -> PipelineConfig:
    """The pipeline configuration the options of ``write`` stand for.

    The parameters are the command's options under the names they have there, with
    the same defaults.  It is the one place that turns them into a
    :class:`PipelineConfig`: ``write``, ``batch`` and the write-up of a research
    project (``research.writeup.write_paper``) all go through it.  Raises
    ``FileNotFoundError`` when ``template`` names no bundled template or directory.
    """
    cfg = PipelineConfig(
        base_dir=base_dir,
        template_dir=resolve_template_dir(template),
        idea_filename=idea_filename,
        experimental_log_filename=experimental_log_filename,
        models=ModelConfig(
            **_role_models(model, writer_model, reflection_model, plotting_model, fast_model),
            image=image_model or None,
            # ``str`` of a StrEnum member is its value, so both spellings are taken.
            effort=str(effort),
        ),
        memory=MemoryConfig(
            enabled=memory,
            project_dir=memory_dir,
            lab=lab_memory,
            lab_dir=lab_memory_dir,
            dream=str(dream),
        ),
        use_plotting=use_plotting,
        keep_provided_figures=keep_provided_figures,
        plotting_max_critic_rounds=plotting_max_critic_rounds,
        compute=str(compute),
        resume=resume,
    )
    if research_cutoff:
        cfg.research_cutoff = research_cutoff
    return cfg


def _config(**options: Any) -> PipelineConfig:
    """:func:`build_config` for a command: an unknown template is a usage error."""
    try:
        return build_config(**options)
    except FileNotFoundError as exc:
        _fail(str(exc))


@app.command(cls=_LegacyFlagsCommand)
def write(
    raw_materials_dir: Annotated[
        Path,
        typer.Option(
            "--raw-materials-dir",
            help="Directory with the idea file, the experimental log and optional figures/.",
        ),
    ],
    template: TemplateOption,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Run directory. Default: ./paper_output_<timestamp>."),
    ] = None,
    idea_filename: IdeaFilenameOption = "idea_sparse.md",
    experimental_log_filename: LogFilenameOption = "experimental_log.md",
    model: ModelOption = None,
    writer_model: WriterModelOption = None,
    reflection_model: ReflectionModelOption = None,
    plotting_model: PlottingModelOption = None,
    fast_model: FastModelOption = None,
    image_model: ImageModelOption = None,
    effort: EffortOption = _Effort.high,
    research_cutoff: CutoffOption = None,
    use_plotting: PlottingOption = False,
    keep_provided_figures: KeepFiguresOption = False,
    plotting_max_critic_rounds: CriticRoundsOption = 3,
    compute: ComputeOption = _Compute.local,
    memory: MemoryOption = True,
    memory_dir: Annotated[
        Path | None,
        typer.Option("--memory-dir", help="Project memory directory (default: <output>/memory)."),
    ] = None,
    lab_memory: LabMemoryOption = True,
    lab_memory_dir: LabMemoryDirOption = None,
    dream: DreamOption = _Dream.stages,
    resume: ResumeOption = True,
) -> None:
    """Write one paper from a directory of raw materials."""
    base_dir = output_dir or Path(f"paper_output_{datetime.now():%Y%m%d_%H%M%S}")
    cfg = _config(
        base_dir=base_dir,
        template=template,
        idea_filename=idea_filename,
        experimental_log_filename=experimental_log_filename,
        model=model,
        writer_model=writer_model,
        reflection_model=reflection_model,
        plotting_model=plotting_model,
        fast_model=fast_model,
        image_model=image_model,
        effort=effort,
        research_cutoff=research_cutoff,
        use_plotting=use_plotting,
        keep_provided_figures=keep_provided_figures,
        plotting_max_critic_rounds=plotting_max_critic_rounds,
        compute=compute,
        memory=memory,
        memory_dir=memory_dir,
        lab_memory=lab_memory,
        lab_memory_dir=lab_memory_dir,
        dream=dream,
        resume=resume,
    )
    try:
        prepare_run_dir(cfg, raw_materials_dir)
        result = run_paper(cfg)
    except (FileNotFoundError, ValueError, ComputeError) as exc:
        _fail(str(exc))
    _report_paper(result, cfg.base_dir)


def _report_paper(result: PaperResult, base_dir: Path) -> None:
    """Print where the paper is and what it cost; exit with 1 when it was not written."""
    if not result.success:
        # The log above has the whole story; repeat only its headline.
        headline = (result.error or "unknown error").splitlines()[0]
        _fail(
            f"the paper was not written: {headline} (details: {base_dir / 'state.json'})",
            code=1,
        )

    typer.echo(f"Paper written: {result.pdf_path}")
    total = result.usage.get("total", {})
    cost = result.usage.get("estimated_cost_usd")
    typer.echo(
        f"Usage: {total.get('requests', 0)} requests, {total.get('input_tokens', 0)} input and "
        f"{total.get('output_tokens', 0)} output tokens"
        + (f", about ${cost:.2f}" if cost is not None else "")
        + f" (details: {base_dir / 'usage.json'})"
    )


@app.command()
def batch(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Batch directory: one run directory per paper."),
    ],
    template: TemplateOption,
    root_input_folder: RootInputFolderOption = None,
    paper_info: PaperInfoOption = None,
    n_papers: PaperCountOption = -1,
    max_workers: Annotated[
        int, typer.Option("--max-workers", min=1, help="Papers written in parallel.")
    ] = 4,
    idea_filename: IdeaFilenameOption = "idea_sparse.md",
    experimental_log_filename: LogFilenameOption = "experimental_log.md",
    model: ModelOption = None,
    writer_model: WriterModelOption = None,
    reflection_model: ReflectionModelOption = None,
    plotting_model: PlottingModelOption = None,
    fast_model: FastModelOption = None,
    image_model: ImageModelOption = None,
    effort: EffortOption = _Effort.high,
    research_cutoff: CutoffOption = None,
    use_plotting: PlottingOption = False,
    keep_provided_figures: KeepFiguresOption = False,
    plotting_max_critic_rounds: CriticRoundsOption = 3,
    compute: ComputeOption = _Compute.local,
    memory: MemoryOption = True,
    lab_memory: LabMemoryOption = True,
    lab_memory_dir: LabMemoryDirOption = None,
    dream: DreamOption = _Dream.stages,
    resume: ResumeOption = True,
) -> None:
    """Write many papers, each in a worker process of its own; finished papers are skipped.

    Stopping the batch (Ctrl-C, or SIGTERM from kill, timeout or a scheduler) starts no
    further paper and stops the ones in flight; the same command continues it later.
    """
    cfg = _config(
        base_dir=output_dir,
        template=template,
        idea_filename=idea_filename,
        experimental_log_filename=experimental_log_filename,
        model=model,
        writer_model=writer_model,
        reflection_model=reflection_model,
        plotting_model=plotting_model,
        fast_model=fast_model,
        image_model=image_model,
        effort=effort,
        research_cutoff=research_cutoff,
        use_plotting=use_plotting,
        keep_provided_figures=keep_provided_figures,
        plotting_max_critic_rounds=plotting_max_critic_rounds,
        compute=compute,
        memory=memory,
        memory_dir=None,
        lab_memory=lab_memory,
        lab_memory_dir=lab_memory_dir,
        dream=dream,
        resume=resume,
    )
    try:
        papers = load_papers(
            root_input_folder=root_input_folder, paper_info_path=paper_info, n_papers=n_papers
        )
        log_path = run_batch(
            cfg,
            root_input_folder=root_input_folder,
            paper_info_path=paper_info,
            output_dir=output_dir,
            n_papers=n_papers,
            max_workers=max_workers,
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    _report_batch(papers, log_path)


def _report_batch(papers: Sequence[dict[str, Any]], log_path: Path) -> None:
    """Print how the papers of this batch fared; exit with 1 unless all of them were written."""
    # The log also holds papers of earlier batches in this directory; the summary and the
    # exit status are about the papers of this one.
    write_log = read_json(log_path)
    counts = summarize(
        {paper["paper_id"]: write_log.get(paper["paper_id"], {}) for paper in papers}
    )
    # Only a batch on Modal that was told to stop has these.
    cancelled = sum(
        write_log.get(paper["paper_id"], {}).get("status") == "cancelled" for paper in papers
    )
    typer.echo(
        f"{counts['success']} written, {counts['failed']} failed, {counts['error']} in error"
        + (f", {cancelled} cancelled" if cancelled else "")
        + f" (details: {log_path})"
    )
    if counts["failed"] or counts["error"]:
        raise typer.Exit(1)


@app.command()
def doctor(
    model: Annotated[
        str, typer.Option("--model", help="Check the provider of this model.")
    ] = DEFAULT_MODEL,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Check credentials, compute backends and memory locations. Sends nothing anywhere."""
    from paper_orchestra.doctor import collect, render

    report = collect(model)
    if as_json:
        typer.echo(json.dumps(report, indent=2))
    else:
        from rich.console import Console

        render(report, Console())
    if not report["ok"]:
        raise typer.Exit(1)


# -- research ----------------------------------------------------------------------

#: Where projects live unless a command is told otherwise; also the web app's default.
WORKSPACE = Path("research")
#: Name of the demonstration project's directory in the workspace.
DEMO_NAME = "demo-warmup"
#: How often the events of a working command are read to be printed, in seconds.
EVENT_POLL_SECONDS = 0.5
#: How many metrics of a run `research run` prints; the result record has them all.
RUN_LINE_METRICS = 6


class _Phase(StrEnum):
    """The engine's phases (``research.engine.PHASES``), spelled out so that reading the
    arguments of a command does not import the engine."""

    intake = "intake"
    research = "research"
    hypothesize = "hypothesize"
    design = "design"
    run = "run"
    analyze = "analyze"


ProjectArgument = Annotated[
    str,
    typer.Argument(
        metavar="PROJECT",
        help="The project: its directory, its name in ./research, or its id.",
        show_default=False,
    ),
]


def _quoted(path: Path | str) -> str:
    """A path as it can be pasted into a shell."""
    return shlex.quote(str(path))


def _sentence(text: str) -> str:
    text = " ".join(str(text).split())
    return text if text.endswith((".", "!", "?")) else text + "."


def _project_dirs(workspace: Path) -> list[Path]:
    """The project directories directly inside ``workspace``, by name."""
    from paper_orchestra.research.project import PROJECT_FILE

    if not workspace.is_dir():
        return []
    # A hidden directory is tool state (or a project being created), not a project.
    return sorted(
        entry
        for entry in workspace.iterdir()
        if not entry.name.startswith(".") and (entry / PROJECT_FILE).is_file()
    )


def _with_id(workspace: Path, project_id: str) -> list[Path]:
    """The project directories in ``workspace`` whose project.json carries ``project_id``."""
    from paper_orchestra.research.project import PROJECT_FILE

    found = []
    for candidate in _project_dirs(workspace):
        try:
            if read_json(candidate / PROJECT_FILE).get("id") == project_id:
                found.append(candidate)
        except (OSError, ValueError, AttributeError, RecursionError):
            continue
    return found


def _the_original(candidates: Sequence[Path]) -> Path | None:
    """Of several directories that carry one id, the one the project was created as: a
    copy of a project directory takes the id with it, but it is another directory.
    ``None`` when the directories do not tell (they are older than that record, or the
    original is gone)."""
    from paper_orchestra.research.model import ProjectError
    from paper_orchestra.research.project import Project

    originals = []
    for candidate in candidates:
        with contextlib.suppress(ProjectError):
            if Project.open(candidate).at_home():
                originals.append(candidate)
    return originals[0] if len(originals) == 1 else None


def _project_root(ref: str) -> Path | None:
    """The directory ``ref`` names: a path, a name in the workspace, or a project's id.

    An id names the project, not a copy of its directory: where several directories
    carry it, the one the project was created as is meant (and the others are named
    on stderr); where that cannot be told, the command ends and asks for a directory.
    """
    from paper_orchestra.research.project import PROJECT_FILE

    for candidate in (Path(ref), WORKSPACE / ref):
        if (candidate / PROJECT_FILE).is_file():
            return candidate
    candidates = _with_id(WORKSPACE, ref)
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    listed = ", ".join(str(candidate) for candidate in candidates)
    original = _the_original(candidates)
    if original is None:
        _fail(
            f"The id {ref} is carried by {len(candidates)} directories ({listed}): one was "
            "copied from another, and they do not say which is the original. Name the "
            "directory you mean instead of the id."
        )
    others = ", ".join(str(candidate) for candidate in candidates if candidate != original)
    _say(
        f"Note: {others} carries the id {ref} as well; it is a copy of {original}, which is "
        "the project this id names. Name a directory to work on a copy.",
        err=True,
    )
    return original


def _open_project(ref: str | Path) -> Project:
    """Open the project a command names, or end the command with what to do instead."""
    from paper_orchestra.research.model import ProjectError
    from paper_orchestra.research.project import Project

    root = _project_root(str(ref))
    if root is None:
        _fail(
            f"There is no research project at {ref}, and none with that name or id in "
            f"./{WORKSPACE}. `paper-orchestra research status` lists the projects there; "
            "`paper-orchestra research new` starts one."
        )
    try:
        return Project.open(root)
    except ProjectError as exc:
        _fail(str(exc))


def _free_dir(wanted: Path) -> Path:
    """``wanted``, or ``wanted-2`` and so on: a default location never lands in a directory
    that is already in use.  Nothing is created."""
    candidate, number = wanted, 1
    while candidate.exists():
        number += 1
        candidate = wanted.with_name(f"{wanted.name}-{number}")
    return candidate


def _engine_state(project: Project) -> dict[str, Any]:
    """``state.json`` as the engine last wrote it; empty before the first phase."""
    from paper_orchestra.research.project import STATE_FILE

    try:
        state = json.loads((project.root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return state if isinstance(state, dict) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    """A part of ``state.json`` that should be an object, read as one whatever a damaged
    or hand-edited file holds there."""
    return value if isinstance(value, Mapping) else {}


def _amount(value: Any) -> float:
    """A number of ``state.json`` for a summary; nothing for what is not a number."""
    usable = isinstance(value, int | float) and not isinstance(value, bool)
    return float(value) if usable and value == value and abs(value) != float("inf") else 0.0


def _worker_alive(state: Mapping[str, Any]) -> bool | None:
    """Whether the worker of a phase that reads as running is still there; ``None`` when
    nothing is said to be running."""
    if state.get("status") != "running":
        return None
    # Telling a live worker from a dead one is the engine's knowledge; it is only
    # loaded for a project that claims to have one.
    from paper_orchestra.research.engine import worker_alive

    return worker_alive(state)


def _state_as_it_is(project: Project) -> dict[str, Any]:
    """The engine's state for a summary: a phase whose worker died reads as stopped,
    which is what the engine makes of it when it next starts."""
    state = _engine_state(project)
    lost = project.state_lost() or _state_problem(project, state)
    if lost:
        # Phases ran and their account is gone or unreadable: the engine reads where the
        # project stands from its events, and so does this summary.
        state = {**_rebuilt_state(project), "state_lost": lost}
    if _worker_alive(state) is False:
        return {**state, "status": "stopped", "worker_gone": True}
    return state


def _state_problem(project: Project, state: Mapping[str, Any]) -> str | None:
    """What the engine says is wrong when the state holds no record of a phase: also the
    case the store cannot see, where the events are gone too while the wiki holds what
    agents wrote.  The engine is only loaded for a project without such records."""
    phases = state.get("phases")
    if isinstance(phases, list) and any(isinstance(record, Mapping) for record in phases):
        return None
    with contextlib.suppress(Exception):
        from paper_orchestra.research import engine

        return engine.state_problem(project)
    return None


def _rebuilt_state(project: Project) -> dict[str, Any]:
    """Where the project stands as the engine reads it from the events, when the engine
    can tell; empty otherwise."""
    with contextlib.suppress(Exception):
        from paper_orchestra.research import engine

        rebuilt = engine.read_state(project)
        if isinstance(rebuilt, Mapping):
            return dict(rebuilt)
    return {}


def _restore_interrupt() -> None:
    """Have an interrupt stop this command even where the process was started ignoring it.

    The web server stops a worker with SIGINT, so that the engine can record the phase
    as interrupted.  A server that was itself started in the background (``nohup``, ``&``,
    some launchers) ignores SIGINT, its children inherit that, and Python then installs
    no handler: the Stop button would do nothing.
    """
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)


class _EventPrinter:
    """Prints a project's new events to stderr, one line each, while a command works.

    It follows ``events.jsonl`` rather than hooking into the engine, so a worker's log
    shows everything that happened to the project in that time, whoever wrote it.
    """

    def __init__(self, project: Project, *, leave_out: Sequence[str] = ()) -> None:
        """``leave_out`` are the types of event the command says itself when it ends (the
        checkpoint, for ``advance``): printed here as well, a person would read the same
        sentence twice."""
        from paper_orchestra.research.project import EVENTS_FILE

        self._project = project
        self._leave_out = frozenset(leave_out)
        self._file = project.root / EVENTS_FILE
        earlier = project.events()
        self._seen = earlier[-1]["seq"] if earlier else 0
        self._size = self._file_size()
        self._done = threading.Event()
        #: What agents were last said to do: the log of a run an agent carries out
        #: repeats those sentences, and here a person would read each of them twice.
        self._told: collections.deque[str] = collections.deque(maxlen=64)
        #: The follower and the command both print; no event is printed twice.
        self._printing = threading.Lock()
        self._thread = threading.Thread(target=self._follow, name="events", daemon=True)

    def __enter__(self) -> _EventPrinter:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._done.set()
        self._thread.join(timeout=10)
        # Once more from this thread: what the last step wrote, an interrupt included.
        self.flush()

    def flush(self) -> None:
        """Print what has happened up to now, before the command says something itself."""
        self._print_new(force=True)

    def _file_size(self) -> int:
        try:
            return self._file.stat().st_size
        except OSError:
            return 0

    def _follow(self) -> None:
        while not self._done.wait(EVENT_POLL_SECONDS):
            self._print_new()

    def _print_new(self, *, force: bool = False) -> None:
        with self._printing:
            size = self._file_size()
            if size == self._size and not force:
                return  # the stream only grows; reading it again would find nothing
            self._size = size
            for event in self._project.events(after=self._seen):
                self._seen = event["seq"]
                if event.get("type") not in self._leave_out and not self._repeats(event):
                    _say(_event_line(event), err=True)

    def _repeats(self, event: Mapping[str, Any]) -> bool:
        """Whether ``event`` is the line of a run's log that repeats what the feed has
        just said of the agent that carries the run out."""
        from paper_orchestra.compute.base import NOTE_PREFIX

        if event.get("type") == "agent.activity":
            self._told.append(str(event.get("summary") or ""))
            return False
        if event.get("type") != "run.log":
            return False
        data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
        lines = [str(line) for line in data.get("lines") or []]
        last = lines[-1] if lines else ""
        if not last.startswith(NOTE_PREFIX):
            return False
        said = last.removeprefix(NOTE_PREFIX)
        return any(
            said == told or (len(said) > 40 and told.startswith(said)) for told in self._told
        )


def _limit_usd(amount: float) -> str:
    """A limit in dollars; one under a cent with the digits that tell it from nothing."""
    if 0 < amount < 0.01:
        return "$" + f"{amount:.4f}".rstrip("0")
    return f"${amount:.2f}"


def _event_line(event: Mapping[str, Any]) -> str:
    """``12:03:41 research (scout): Verified 12 of 15 candidate papers.``"""
    where = str(event.get("phase") or str(event.get("type") or "").split(".")[0])
    agent = f" ({event['agent']})" if event.get("agent") else ""
    clock = str(event.get("ts") or "")[11:19]
    return f"{clock} {where}{agent}: {event.get('summary', '')}"


@contextlib.contextmanager
def _working(project: Project, command: str = "advance") -> Iterator[_EventPrinter]:
    """A command that does the work of a worker: it can be stopped, its events are shown
    (by the printer this yields), and a refusal by the project, a model or a backend
    ends it with the reason and exit code 1.  ``command`` is the research command that
    is at work, which is the one that goes on after a stop."""
    from paper_orchestra.llm.base import LLMError
    from paper_orchestra.research.model import ProjectError

    _restore_interrupt()
    _sweep(project)
    _forget_stop(project)
    # `advance` ends by saying its checkpoint in full; `run` reaches none.
    said_at_the_end = ("checkpoint",) if command == "advance" else ()
    try:
        with _EventPrinter(project, leave_out=said_at_the_end) as events:
            yield events
    except KeyboardInterrupt:
        _stopped(project, command)
    except (ProjectError, ComputeError, LLMError) as exc:
        if _needs_a_model(exc):
            # Not an error of the project: said once, as the engine says it.
            _say(str(exc), fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(NEEDS_A_MODEL) from None
        _fail(str(exc), code=1)


def _needs_a_model(exc: BaseException) -> bool:
    """Whether the engine refused because the phase that would run needs a model and none
    can be used (``engine.ModelNeeded``).  Only the engine raises it, so it is loaded by
    the time one arrives."""
    engine = sys.modules.get("paper_orchestra.research.engine")
    refusal = getattr(engine, "ModelNeeded", None)
    return isinstance(refusal, type) and isinstance(exc, refusal)


def _unreported_spend(project: Project) -> tuple[float, list[str]]:
    """What of the model spend is a count from tokens for agents that never reported
    their cost (``unreported_usd`` of the usage ledger), and the models of such agents
    that have no list price and are in the sum with nothing (``unreported_unpriced``);
    nothing and none when there are none."""
    try:
        ledger = read_json(project.root / "usage.json")
        unpriced = [str(model) for model in ledger.get("unreported_unpriced") or []]
        return max(0.0, float(ledger.get("unreported_usd") or 0.0)), unpriced
    except (OSError, ValueError, TypeError, AttributeError, RecursionError):
        return 0.0, []


def _sweep(project: Project) -> None:
    """Remove the temporary files that writers which died left in the project.  Done when
    a worker starts: a crash must not leave litter that grows with every crash."""
    with contextlib.suppress(OSError):
        project.sweep_temporary_files()


def _stopped(project: Project, command: str = "advance") -> NoReturn:
    """Say that the command was stopped and what goes on from there; exit with 130.

    The stop is left on file first (``project.record_worker_stop``): the app's server
    starts the runs that wait when a worker has ended, and not after one that was
    stopped, which it cannot tell of a worker started here without that word."""
    _record_stop(project)
    root = _quoted(project.root)
    if command == "run":
        how = _after_a_stopped_run(project, root)
    else:
        how = (
            "What was in flight is on record as interrupted; "
            f"`paper-orchestra research advance {root}` continues from there."
        )
    _say(f"Stopped. {how}", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(INTERRUPTED)


def _record_stop(project: Project) -> None:
    from paper_orchestra.research.project import record_worker_stop

    record_worker_stop(project.root)


def _forget_stop(project: Project) -> None:
    """A command that starts work is not the worker that was stopped before it."""
    from paper_orchestra.research.project import forget_worker_stop

    forget_worker_stop(project.root)


def _after_a_stopped_run(project: Project, root: str) -> str:
    """What became of the experiments when `research run` was stopped, by their pages: a
    run that was in flight goes back to the queue; one that had just ended is on record;
    an approved experiment that was not started yet still waits for its run."""
    accepted = project.pages(kind="experiment", review="accepted")
    in_queue = [page for page in accepted if page.state in ("queued", "running")]
    untouched = [page for page in accepted if page.state == "approved"]
    # `advance` would start the project's next phase, which need not be the run.
    again = f"`paper-orchestra research run {root}`"
    if in_queue:
        return f"The run that was in flight is back in the queue; {again} runs it again."
    if untouched:
        one = len(untouched) == 1
        waiting = f"{len(untouched)} approved experiment{' still waits' if one else 's still wait'}"
        return (
            "No run was left in flight: a run that had finished is on record. "
            f"{waiting}: {again} runs {'it' if one else 'them'}."
        )
    # The stop came when the run had just ended: its outcome is on record.
    return (
        "No run is waiting to be run again: what had finished is on record. "
        f"`paper-orchestra research status {root}` shows the experiments."
    )


def _plural(count: int, kind: str) -> str:
    """``1 hypothesis``, ``3 hypotheses``, ``2 edit proposals``."""
    from paper_orchestra.research.model import SECTIONS

    if count == 1:
        return f"1 {kind}"
    many = SECTIONS.get(kind) or f"{kind}s"
    return f"{count} {many if many != kind else kind + ' pages'}"


def _waiting(counts: Mapping[str, Any]) -> str:
    """What is waiting for the person, by kind: ``2 hypotheses, 1 experiment``."""
    from paper_orchestra.research.model import KINDS

    pending = counts["pending"]
    parts = [
        _plural(pending["by_kind"][kind], kind) for kind in KINDS if pending["by_kind"].get(kind)
    ]
    if pending["proposals"]:
        parts.append(_plural(pending["proposals"], "edit proposal"))
    return ", ".join(parts)


def _where_it_stands(state: Mapping[str, Any]) -> str:
    """One line on the engine: the round, the phase and what became of it."""
    status = state.get("status")
    if status in (None, "", "idle"):
        if state.get("state_lost"):
            return (
                "Phases of this project have run, but where it stands is not on file "
                "(state.json is missing or cannot be read)."
            )
        return "Not started: no phase has run yet."
    from paper_orchestra.research.model import phase_name

    # A phase is named as the app names it, not by the word a file or an option uses.
    phase = phase_name(state["phase"]) if state.get("phase") else "no phase"
    line = f"Round {state.get('round', 0)}, "
    following = f"; next: {phase_name(state['next_phase'])}" if state.get("next_phase") else ""
    if status == "checkpoint":
        return line + f"checkpoint after {phase}{following}"
    if status == "running":
        worker = _mapping(state.get("worker"))
        return (
            line + f"at work on {phase}" + (f" (pid {worker['pid']})" if worker.get("pid") else "")
        )
    if status == "failed":
        return line + f"failed in {phase}: {_sentence(state.get('error') or 'no error recorded')}"
    if status == "stopped":
        gone = " (its worker is gone)" if state.get("worker_gone") else ""
        return line + f"stopped during {phase}{gone}"
    return line + f"{status}{following}"


def _held_runs(project: Project, state: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """The experiments that wait for the person although autopilot is on, as the project
    stands, each with its reason and whether the person can approve it (``kind``:
    ``research.autopilot.held``)."""
    from paper_orchestra.research import autopilot

    return autopilot.held(project, state)


def _held_in_words(held: Sequence[Mapping[str, str]]) -> list[str]:
    """What of autopilot's waits for the person, counted: the runs it left for them to
    approve, and those nobody can approve until what refuses them is dealt with."""
    from paper_orchestra.research import autopilot

    left = [item["id"] for item in autopilot.left(held)]
    refused = [item["id"] for item in autopilot.refused(held)]
    said = []
    if left:
        said.append(f"{_plural(len(left), 'run')} autopilot left for you ({_and_words(left)})")
    if refused:
        said.append(
            f"{_plural(len(refused), 'run')} that cannot be approved as things are "
            f"({_and_words(refused)})"
        )
    return said


def _and_words(names: Sequence[str]) -> str:
    names = [str(name) for name in names]
    return f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else "".join(names)


def _next_steps(
    project: Project,
    state: Mapping[str, Any],
    counts: Mapping[str, Any],
    held: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """What the person can do now, as commands they can paste.  ``held`` is what waits
    of autopilot's (``_held_runs``), where the caller has asked already."""
    root = _quoted(project.root)
    advance = f"`paper-orchestra research advance {root}`"
    status = state.get("status")
    if status in (None, "", "idle") and state.get("state_lost"):
        return (
            f"Next: {advance} --phase PHASE goes on with the phase you name (intake, "
            "research, hypothesize, design, run or analyze); nothing is started over."
        )
    unattended = project.settings.autonomy.mode == "autopilot"
    if status in (None, "", "idle") and unattended:
        return (
            f"Next: {advance} starts the research; on autopilot it goes on by itself until "
            "a cap is reached."
        )
    if status in (None, "", "idle"):
        return f"Next: {advance} starts the research and stops at the first checkpoint."
    if status == "running":
        return "A worker is at work on it; follow along in the web app (`paper-orchestra ui`)."
    if status == "failed":
        return f"Next: once the cause is dealt with, {advance} tries the phase again."
    if status == "stopped":
        return f"Next: {advance} continues the phase that was stopped."
    steps = []
    if counts["pending"]["total"]:
        steps.append(
            "decide what is waiting, in the web app (`paper-orchestra ui`) or with "
            f"`paper-orchestra research review {root}`"
        )
    planned = project.pages(kind="experiment", review="accepted", state="planned")
    if planned:
        ids = ", ".join(page.id or page.path for page in planned)
        steps.append(
            f"approve the experiments you want run ({ids} planned) with "
            f"`paper-orchestra research approve {root} {planned[0].id or planned[0].path}`"
        )
    if unattended:
        # What still waits is what autopilot left for the person; the rest it decides.
        steps = [step for step in steps if not step.startswith("approve the experiments")]
        from paper_orchestra.research import autopilot

        held = _held_runs(project, state) if held is None else held
        left = [item["id"] for item in autopilot.left(held)]
        refused = [item["id"] for item in autopilot.refused(held)]
        if left:
            # Without these a new stretch finds nothing it may run, and ends at once.
            steps.append(
                f"approve the runs autopilot left for you ({_and_words(left)}) with "
                f"`paper-orchestra research approve {root} {left[0]}`"
            )
        if refused:
            # Nobody can approve these as things are: the refusal says what to change,
            # and a stretch that starts before that finds the same and ends at once.
            steps.append(
                f"deal with what keeps {_and_words(refused)} from being approved (said "
                "with what waits for you); autopilot approves "
                f"{'it' if len(refused) == 1 else 'them'} in the stretch after that"
            )
        steps.append(
            f"{'then ' if steps else ''}{advance} starts a new stretch on autopilot "
            "(it goes on by itself until a cap is reached)"
        )
        return "Next: " + "; ".join(steps) + "."
    steps.append(f"{'then ' if steps else ''}{advance} goes on")
    return "Next: " + "; ".join(steps) + "."


#: What a checkpoint's sentence ends in: what comes next, and before it the count of what
#: waited (as the engine says it, and as the demonstration project does).
_SAID_NEXT = re.compile(r"(?:^|(?<=[.!?…]\s))Next: [^.]*\.\s*$")
_SAID_WAITING = re.compile(
    r"(?:^|(?<=[.!?…]\s))[^.!?]*\b(?:(?:is|are) waiting for you|awaits? your review)\.\s*$"
)
#: Between the two, where an agent had asked the person to choose something: how many
#: decisions waited (``decisions.waiting``; the demonstration project ends with it).
_SAID_UNDECIDED = re.compile(r"(?:^|(?<=[.!?…]\s))\d+ decisions? waits? for your answer\.\s*$")
_STAMP = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
_AT_STAMP = re.compile(rf"\b(at )?({_STAMP})")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _account(checkpoint: Mapping[str, Any]) -> str:
    """What the phases did, as their checkpoint tells it, without what was true at that
    moment only: how many things waited for the person, how many decisions, and what was
    next.  They are said from the project as it is now, on lines of their own.

    The engine keeps that account by itself (``did``).  A checkpoint without it (written
    by an earlier version, or rebuilt from the events after ``state.json`` was lost) has
    it in its summary, before those sentences.
    """
    did = checkpoint.get("did")
    if isinstance(did, str) and did.strip():
        return " ".join(did.split())
    summary = checkpoint.get("summary")
    told = " ".join(summary.split()) if isinstance(summary, str) else ""
    for stale in (_SAID_NEXT, _SAID_UNDECIDED, _SAID_WAITING):
        told = stale.sub("", told).rstrip()
    return told


def _readable(text: str) -> str:
    """A sentence of the project as a person reads it at a terminal: a time is said in
    local time (``on 16 Nov 2026 at 15:54``) where a record holds a stamp, and the
    author ``human`` is ``you``, as in the app."""
    from paper_orchestra.research.model import parse_stamp

    def said(match: re.Match[str]) -> str:
        moment = parse_stamp(match.group(2))
        if moment is None:
            return match.group(0)
        local = moment.astimezone()
        day = f"{local.day} {_MONTHS[local.month - 1]} {local.year}"
        clock = f"{local.hour:02d}:{local.minute:02d}"
        return f"on {day} at {clock}" if match.group(1) else f"{day}, {clock}"

    return re.sub(r"\bby (?:the )?human\b(?!-)", "by you", _AT_STAMP.sub(said, text))


def _status_lines(project: Project, state: Mapping[str, Any]) -> list[str]:
    """The compact summary of a project: where it stands, what waits, what it cost.

    Of a checkpoint it repeats what the phases did.  What waits is counted now: the
    count the checkpoint took is not said again once the person has decided things.
    """
    from paper_orchestra.research import autopilot

    counts = project.counts()
    lines = [f"  {_where_it_stands(state)}"]
    # With what autopilot did not approve: it waits for the person as a proposal does.
    held = _held_runs(project, state)
    waiting = [_waiting(counts)] if counts["pending"]["total"] else []
    waiting += _held_in_words(held)
    said: list[str] = []
    if waiting:
        said.append(f"  Waiting for you: {', '.join(waiting)}")
    elif state.get("status") == "checkpoint":
        said.append("  Nothing is waiting for you.")
    if held:
        # Why, once for all that share a reason, and right under it what to do: where a
        # stretch stopped for these, that is what the person came to read, so it goes
        # first and the lines below do not say the reason again.
        why = autopilot.grouped([(item["id"], item["reason"]) for item in held])
        said.append(f"  {_readable(_sentence(why))}")
        said.append(f"  {_next_steps(project, state, counts, held)}")
        lines += said
    lines += _autopilot_lines(project, state, held)
    checkpoint = _mapping(state.get("checkpoint"))
    if state.get("status") == "checkpoint":
        did = _account(checkpoint)
        if did:
            lines.append(f"  {_readable(_sentence(did))}")
    if not held:
        lines += said
    board = [f"{count} {name}" for name, count in counts["experiments"].items() if count]
    if board:
        lines.append(f"  Experiments: {', '.join(board)}")
        lines += _experiment_notes(project)
    if counts["requests"]["open"]:
        lines.append(f"  {_plural(counts['requests']['open'], 'open request')} for the agents")
    undecided = counts["decisions"]["open"]
    if undecided:
        lines.append(
            f"  {_plural(undecided, 'decision')} "
            f"{'waits' if undecided == 1 else 'wait'} for your answer: "
            f"`paper-orchestra research decide {_quoted(project.root)}`"
        )
    spent, limits = _mapping(state.get("spent")), project.settings.limits
    # The two budgets are totals; the time limit is per run, so the time used is only told.
    # A demonstration's model cost is part of what was written for it, not a bill.
    story = (
        " (the cost of the story this demonstration tells: no model was called)"
        if project.demonstration
        else ""
    )
    lines.append(
        f"  Spent: ${_amount(spent.get('model_usd')):.2f} of {_limit_usd(limits.model_usd)} on "
        f"models{story}; ${_amount(spent.get('experiment_usd')):.2f} of "
        f"{_limit_usd(limits.experiment_usd)} on experiments, which ran for "
        f"{_amount(spent.get('experiment_seconds')):.0f} s in all (one run may take up to "
        f"{limits.experiment_seconds:.0f} s)"
    )
    unreported, unpriced = _unreported_spend(project)
    if unreported:
        lines.append(
            f"  The model spend is at least that: ${unreported:.2f} of it stands for agents "
            "that were stopped before they reported their cost, counted from their tokens "
            "at the list price"
        )
    if unpriced:
        lines.append(
            "  The model spend is at least that: an agent on "
            f"{', '.join(repr(model) for model in unpriced)} was stopped before it "
            "reported its cost, and that model has no list price here, so its tokens are "
            "in usage.json and its cost is not in the sum"
        )
    lines += _problem_lines(project)
    if not held:
        lines.append(_next_steps(project, state, counts, held))
    return lines


def _has_live_worker(state: Mapping[str, Any]) -> bool:
    """Whether a worker that the state names is still there (between two phases of a
    stretch the state reads as a checkpoint, and the worker goes on all the same)."""
    if not isinstance(state.get("worker"), Mapping):
        return False
    from paper_orchestra.research.engine import worker_alive

    return bool(worker_alive(state))


def _compute_in_words(backend: str) -> str:
    """``settings.compute`` in a sentence: where an experiment runs unless its approval
    names another backend."""
    if backend == "modal":
        return (
            "Experiments: on Modal, each run in a sandbox of its own, unless an approval "
            "names another backend."
        )
    return (
        "Experiments: on this machine (the local backend: a run is a process of yours, "
        "not in a sandbox), unless an approval names another backend; one that asks for "
        "a GPU runs on Modal."
    )


def _autonomy_in_words(settings: Any, *, held: Sequence[Mapping[str, str]] = ()) -> str:
    """``settings.autonomy`` in a sentence: the mode, what a stretch is held to, and what
    it does not decide or count as these settings stand (the runs of a project whose
    experiments would run unsandboxed as the person, or on a Modal that cannot be used
    on this machine; what experiments cost).  ``held`` is what the caller has said
    waits, with the reasons: where the reason no run is approved is among them, it is
    not said again."""
    from paper_orchestra.research.autopilot import caps_in_words
    from paper_orchestra.research.experiments import (
        UNSANDBOXED_RUN,
        rule_cannot_run,
        rule_leaves_runs,
    )

    autonomy = settings.autonomy
    caps = caps_in_words(autonomy.to_dict())
    network = (
        "runs it approves get the network access their plans ask for"
        if autonomy.network
        else "runs it approves get no network access"
    )
    if autonomy.mode != "autopilot":
        return (
            "Autopilot: off (supervised: every proposal, edit, run and checkpoint waits for "
            f"you). Switched on, a stretch would be held to {caps}."
        )
    runs_wait = rule_leaves_runs(settings)
    # What the settings promise is said only where this machine can keep it.
    blocked = None if runs_wait else rule_cannot_run(settings)
    above = " (why is said with what waits for you, above)."
    reasons = [str(item.get("reason")) for item in held]
    if runs_wait:
        said = UNSANDBOXED_RUN in reasons
        decided = (
            "Pages are accepted, edits applied and checkpoints passed without you. Runs on "
            "this machine are not approved without you" + (above if said else f": {runs_wait}.")
        )
    elif blocked:
        said = any("Modal" in reason for reason in reasons)
        decided = (
            "Pages are accepted, edits applied and checkpoints passed without you. No run "
            "can be approved now, by autopilot or by you"
            + (above if said else f": {blocked}. Once it can be used, runs are approved.")
        )
    else:
        decided = (
            "Pages are accepted, edits applied, runs approved and checkpoints passed without you."
        )
    return (
        f"Autopilot: on. {decided} One stretch is held to {caps}; {network}. What the "
        "experiments cost is no part of these caps: the project's own limit holds it "
        f"({_limit_usd(settings.limits.experiment_usd)}, settings.limits.experiment_usd)."
    )


def _autopilot_lines(
    project: Project,
    state: Mapping[str, Any],
    held: Sequence[Mapping[str, str]] | None = None,
) -> list[str]:
    """What ``status`` says of autopilot: the mode, where the stretch stands or why the
    last one ended, what was decided without the person, and what waits for them all
    the same.  Nothing for a supervised project that never ran unattended.

    ``held`` is what waits of autopilot's where the caller says that itself, with the
    reasons (``_held_runs``): these lines then do not say the reasons a second time.
    """
    from paper_orchestra.research import autopilot

    autonomy = project.settings.autonomy
    found = autopilot.digest(project, state)
    if autonomy.mode != "autopilot" and found is None:
        return []
    lines = [f"  {_autonomy_in_words(project.settings, held=held or ())}"]
    if found is None:
        if autonomy.mode == "autopilot":
            lines.append(
                "  No stretch has run yet: "
                f"`paper-orchestra research advance {_quoted(project.root)}` starts one."
            )
        return lines
    caps = found["caps"]
    if found["running"] and _has_live_worker(state):
        lines.append(
            f"  The stretch is under way since {_readable(found['started'])}: "
            f"{found['rounds']} of at most {caps['rounds']} rounds completed, "
            f"${found['spent_usd']:.2f} of ${caps['model_usd']:.2f} spent on models."
        )
    else:
        why = found["said"] or (
            f"{(found['why'] or 'its worker ended without saying why')[0].upper()}"
            f"{(found['why'] or 'its worker ended without saying why')[1:]}."
        )
        if held and found["reason"] == "needs_you":
            why = "Nothing more can happen without you."  # what waits is said above
        ended = f" on {_readable(found['ended'])}" if found["ended"] else ""
        lines.append(f"  The last stretch ended{ended}: {_readable(_sentence(why))}")
    if held is not None and found.get("held"):
        found = {**found, "held": []}  # said by the caller
    lines.append(f"  {_readable(autopilot.summary(found))}")
    return lines


def _experiment_notes(project: Project) -> list[str]:
    """What the count of experiments by state does not say, one line each: an experiment
    that reads approved and will not run as it stands (it changed since the approval),
    and one that reads cancelled or failed although an earlier run of it succeeded."""
    from paper_orchestra.research.model import ProjectError

    waits, over = ("approved", "queued", "running"), ("cancelled", "failed")
    pages = [page for page in project.pages(kind="experiment") if page.state in waits + over]
    if not pages:
        return []
    # (Loaded for these experiments only: it is the module that runs them.)
    from paper_orchestra.research import experiments

    root, lines = _quoted(project.root), []
    for page in pages:
        name = page.id or page.path
        with contextlib.suppress(ProjectError, OSError):
            if page.state in over:
                found = experiments.last_success(project, page.path)
                if found:
                    lines.append(f"  {name} is {page.state}; its run {found['number']} succeeded.")
            elif page.review == "accepted":
                reason = experiments.will_not_run(project, page.path)
                if reason:
                    lines.append(
                        f"  {name} reads {page.state} but will not run: {_readable(reason)}. "
                        f"`paper-orchestra research approve {root} {name}` approves it as it "
                        "is now."
                    )
    return lines


#: How many of a project's problems ``status`` prints; ``--json`` has them all.
STATUS_PROBLEMS = 12


def _problem_lines(project: Project) -> list[str]:
    """What the project had to read leniently (``Project.problems()``): a hand-edited
    file is opened whatever state it is in, and this is where the person is told."""
    problems = project.problems()
    if not problems:
        return []
    count = len(problems)
    lines = [
        f"  {count} problem{'s' if count != 1 else ''} in the project's files (read as "
        "well as they could be):"
    ]
    lines += [f"    {item['path']}: {item['message']}" for item in problems[:STATUS_PROBLEMS]]
    if count > STATUS_PROBLEMS:
        lines.append(f"    ... and {count - STATUS_PROBLEMS} more; `status --json` lists them all")
    return lines


def _pending_item(item: ResearchPage | EditProposal) -> dict[str, Any]:
    """One thing waiting for a decision, as ``status --json`` and ``review`` list it."""
    from paper_orchestra.research.model import EditProposal

    if isinstance(item, EditProposal):
        return {
            "ref": item.id,
            "kind": "edit",
            "title": item.summary,
            "target": item.target,
            "author": item.author,
            "round": item.round,
        }
    return {
        "ref": item.id or item.path,
        "kind": item.kind,
        "title": item.title,
        "summary": item.summary,
        "path": item.path,
        "author": item.author,
        "round": item.round,
    }


def _status_data(project: Project) -> dict[str, Any]:
    """Everything ``status`` knows, for ``--json``.  ``state`` is ``state.json`` as it
    is; ``worker_alive`` says whether a phase that reads as running really is."""
    state = _engine_state(project)
    return {
        "project": {
            "id": project.id,
            "title": project.title,
            "root": str(project.root),
            "created": project.created,
        },
        "state": state,
        "worker_alive": _worker_alive(state),
        "counts": project.counts(),
        "pending": [_pending_item(item) for item in project.pending()],
        "limits": project.settings.limits.to_dict(),
        "autonomy": project.settings.autonomy.to_dict(),
        "autopilot": _digest_or_none(project, state),
        "problems": project.problems(),
    }


def _digest_or_none(project: Project, state: Mapping[str, Any]) -> dict[str, Any] | None:
    """The account of the latest unattended stretch (``research.autopilot.digest``);
    ``None`` for a project that never had one."""
    from paper_orchestra.research import autopilot

    return autopilot.digest(project, state)


def _advance(project: Project, *, phase: str | None, instruction: str | None) -> None:
    """Run the engine up to its next checkpoint and say how that ended.

    Returns at a checkpoint; exits with 1 when the engine failed and with 130 when it
    was stopped, by Ctrl-C, SIGTERM or a stop request that reached the engine itself.
    """
    with _working(project):
        from paper_orchestra.research.engine import Engine

        state = Engine(project).advance(phase=phase, instruction=instruction)
    status = state.get("status")
    root = _quoted(project.root)
    held = _held_runs(project, state)
    if _mapping(state.get("autopilot")).get("ended") and project.settings.autonomy.autopilot:
        # The unattended stretch this command was is over: why, and what was decided
        # without the person meanwhile.  Said before how the last phase ended.
        # (What waits is counted below; the checkpoint's own sentence has the reasons.)
        for line in _autopilot_lines(project, state, held)[1:]:
            _say(line.strip())
    if status == "failed":
        from paper_orchestra.research.engine import phase_in_words

        failed = phase_in_words(state["phase"]) if state.get("phase") else "the current phase"
        _fail(
            f"{failed[0].upper()}{failed[1:]} failed: "
            f"{_sentence(state.get('error') or 'no error was recorded')} The activity above "
            f"and {project.root / 'state.json'} have the details. Once the cause is dealt "
            f"with, `paper-orchestra research advance {root}` tries the phase again.",
            code=1,
        )
    if status == "stopped":
        _stopped(project)
    checkpoint = _mapping(state.get("checkpoint"))
    if status == "checkpoint" and checkpoint:
        from paper_orchestra.research.model import phase_name

        # Said at the moment of the pause, so the whole of it is true: what was done,
        # what waits and what comes next.
        _say(
            f"Checkpoint after {phase_name(checkpoint.get('phase') or state.get('phase'))} "
            f"(round {checkpoint.get('round', state.get('round', 0))}): "
            f"{_readable(_sentence(checkpoint.get('summary') or 'the phase is complete'))}"
        )
    else:
        _say(_where_it_stands(state))
    counts = project.counts()
    waiting = ([_waiting(counts)] if counts["pending"]["total"] else []) + _held_in_words(held)
    if waiting:
        _say(f"Waiting for you: {', '.join(waiting)}")
    _say(_next_steps(project, state, counts, held))


def _brief_text(brief: str | None, brief_file: Path | None) -> str:
    """The idea as the person gave it, from exactly one of the two options."""
    if (brief is None) == (brief_file is None):
        _fail(
            "Give the idea with --brief TEXT or --brief-file FILE. Exactly one of them is needed."
        )
    if brief_file is not None:
        try:
            brief = brief_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            _fail(f"The brief file {brief_file} does not exist.")
        except (OSError, UnicodeDecodeError) as exc:
            _fail(f"The brief file {brief_file} cannot be read as text ({exc}).")
    if not (brief or "").strip():
        _fail("The brief is empty: say in a few sentences what the idea is.")
    return brief or ""


def _default_project_dir(title: str | None, brief: str) -> Path:
    """``./research/<slug>``, named after the title or else the brief's first words."""
    from paper_orchestra.memory.wiki import slugify

    first_line = next((line for line in brief.splitlines() if line.strip()), "")
    name = title or " ".join(first_line.split()[:8])
    return _free_dir(WORKSPACE / slugify(name, 48))


AgentModelOption = Annotated[
    str | None,
    typer.Option(
        "--agent-model",
        "--model",
        metavar="MODEL",
        help="The model of the sdk agents (default: claude-fable-5-2).",
    ),
]
AgentFastOption = Annotated[
    bool | None,
    typer.Option(
        "--fast/--no-fast",
        help="Ask for fast mode (sdk agents). It costs several times the usual price and "
        "works with the `login` credential only.",
    ),
]
AgentUltracodeOption = Annotated[
    bool | None,
    typer.Option(
        "--ultracode/--no-ultracode",
        help="Let the sdk agents start helper agents and workflows.",
    ),
]
AgentCredentialOption = Annotated[
    str | None,
    typer.Option(
        "--credential",
        metavar="login|federation",
        help="Whose account the sdk agents work on: your Claude Code `login`, or the "
        "worker's Workload Identity `federation` settings.",
    ),
]
AgentSandboxOption = Annotated[
    bool | None,
    typer.Option(
        "--sandbox/--no-sandbox",
        help="Keep the sdk agents in their own folder, away from credentials and the "
        "project's files. Turn it off only if you accept that an agent can then reach "
        "everything you can.",
    ),
]


def _agent_changes(
    runtime: str | None,
    model: str | None,
    fast: bool | None,
    ultracode: bool | None,
    credential: str | None,
    sandbox: bool | None,
) -> dict[str, Any]:
    """The values of ``settings.agents`` that the options of a command name."""
    given = (
        ("runtime", runtime.strip().lower() if runtime is not None else None),
        ("model", model.strip() if model is not None else None),
        ("fast", fast),
        ("ultracode", ultracode),
        ("credential", credential.strip().lower() if credential is not None else None),
        ("sandbox", sandbox),
    )
    return {name: value for name, value in given if value is not None}


def _who_pays(agents: Any) -> str:
    """One line on whose account a project's sdk agents work, and at what speed: said
    before the first of them starts."""
    fast = bool(agents.fast) and agents.credential == "login"
    how = f"{agents.model}{' in fast mode' if fast else ''}"
    if agents.credential == "login":
        said = (
            f"The agents are Claude agents on {how}, signed in with your Claude Code login: "
            "what they use is billed to your own Claude account"
        )
        said += ", and fast mode costs several times the usual price." if fast else "."
    else:
        said = (
            f"The agents are Claude agents on {how}, signed in with the worker's Workload "
            "Identity Federation settings: what they use is billed to that organization."
        )
    return (
        f"{said} `paper-orchestra research settings` changes that (--credential, "
        "--agent-model, --fast/--no-fast)."
    )


@research_app.command("new")
def research_new(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(
            metavar="PATHS",
            help="Files and folders to attach: notes, data, figures, papers.",
            show_default=False,
        ),
    ] = None,
    brief: Annotated[
        str | None, typer.Option("--brief", help="The idea, in a few sentences.")
    ] = None,
    brief_file: Annotated[
        Path | None, typer.Option("--brief-file", help="A text file that holds the idea.")
    ] = None,
    title: Annotated[
        str | None,
        typer.Option("--title", help="The project's title. Default: the brief's first line."),
    ] = None,
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="The project's directory (new or empty). Default: ./research/<name taken "
            "from the title or the brief>.",
        ),
    ] = None,
    research_cutoff: Annotated[
        str | None,
        typer.Option(
            "--research-cutoff",
            metavar="YYYY-MM",
            help="Ignore work published from this month on (YYYY, YYYY-MM or YYYY-MM-DD). "
            "Default: no cutoff.",
        ),
    ] = None,
    runtime: Annotated[
        str | None,
        typer.Option(
            "--runtime",
            metavar="classic|sdk",
            help="How the project's agents are run: `classic` (this package's tool loop "
            "over a model API) or `sdk` (Claude Agent SDK agents with a shell, the web and "
            "the board). Default: classic, or what PAPER_ORCHESTRA_AGENT_RUNTIME says. "
            "`paper-orchestra research settings` changes it later.",
        ),
    ] = None,
    model: AgentModelOption = None,
    fast: AgentFastOption = None,
    ultracode: AgentUltracodeOption = None,
    credential: AgentCredentialOption = None,
    sandbox: AgentSandboxOption = None,
    compute: ExperimentComputeOption = None,
    autopilot: AutopilotOption = None,
    rounds: AutopilotRoundsOption = None,
    stretch_usd: AutopilotUsdOption = None,
    hours: AutopilotHoursOption = None,
    stretch_network: AutopilotNetworkOption = None,
    run: Annotated[
        bool,
        typer.Option(
            "--run/--no-run",
            help="Start the research right away and stop at the first checkpoint (on "
            "autopilot: when a cap is reached).",
        ),
    ] = True,
) -> None:
    """Start a research project from an idea, and advance it to its first checkpoint.

    The agents read what you gave, research prior work and fill the project's wiki.
    Everything they write is a suggestion until you accept it.
    """
    from paper_orchestra.research.model import ProjectError, settings_for_new_project
    from paper_orchestra.research.project import Project

    text = _brief_text(brief, brief_file)
    root = directory or _default_project_dir(title, text)
    given: dict[str, Any] = {}
    if research_cutoff is not None:
        given["research_cutoff"] = research_cutoff
    chosen = _agent_changes(runtime, model, fast, ultracode, credential, sandbox)
    if chosen:
        given["agents"] = chosen
    if compute is not None:
        given["compute"] = compute.value
    unattended = _autonomy_changes(autopilot, rounds, stretch_usd, hours, stretch_network)
    if unattended:
        given["autonomy"] = unattended
    # The runtime is written into the project when it is created: what the environment
    # says later changes nothing about a project that exists.
    try:
        settings = settings_for_new_project(given)
    except ProjectError as exc:
        _fail(str(exc))
    try:
        project = Project.create(
            root, title=title or "", brief=text, files=paths or (), settings=settings
        )
    except ProjectError as exc:
        _fail(str(exc))
    _say(f'Created the research project "{project.title}" in {root}')
    attached = len(project.attachments())
    if attached:
        _say(f"{attached} file{'s' if attached != 1 else ''} attached.")
    if project.settings.agents.runtime == "sdk":
        # Before the first agent starts: on whose account, and at what speed.
        _say(_who_pays(project.settings.agents))
    if project.settings.autonomy.autopilot:
        # Before anything is decided without them: that it will be, and up to where.
        _say(_autonomy_in_words(project.settings))
    shown = _quoted(root)
    if not run:
        stops = (
            "it goes on by itself until a cap of autopilot is reached"
            if project.settings.autonomy.autopilot
            else "it stops at the first checkpoint"
        )
        _say(
            "Next:\n"
            f"  paper-orchestra research advance {shown}   start the research; {stops}\n"
            "  paper-orchestra ui   open the web app"
        )
        return
    _say(
        "Starting the first round. Ctrl-C stops it; "
        f"`paper-orchestra research advance {shown}` continues."
    )
    _advance(project, phase=None, instruction=None)


ExperimentComputeOption = Annotated[
    _Compute | None,
    typer.Option(
        "--compute",
        help="Where the project's experiments run unless an approval says otherwise "
        "(settings.compute): local (this machine, as you, not a sandbox) or modal (a "
        "Modal sandbox for each run). Default: local.",
        show_default=False,
    ),
]
AutopilotOption = Annotated[
    bool | None,
    typer.Option(
        "--autopilot/--supervised",
        help="Autopilot: pages are accepted, edits applied, runs approved and checkpoints "
        "passed without you, up to the caps of one stretch. Supervised (the default): "
        "everything waits for you.",
        show_default=False,
    ),
]
AutopilotRoundsOption = Annotated[
    int | None,
    typer.Option(
        "--autopilot-rounds",
        min=1,
        metavar="N",
        help="Cap of one unattended stretch: rounds (default 3).",
        show_default=False,
    ),
]
AutopilotUsdOption = Annotated[
    float | None,
    typer.Option(
        "--autopilot-usd",
        metavar="USD",
        help="Cap of one unattended stretch: dollars of model spend (default 200).",
        show_default=False,
    ),
]
AutopilotHoursOption = Annotated[
    float | None,
    typer.Option(
        "--autopilot-hours",
        metavar="HOURS",
        help="Cap of one unattended stretch: hours (default 12).",
        show_default=False,
    ),
]
AutopilotNetworkOption = Annotated[
    bool | None,
    typer.Option(
        "--autopilot-network/--no-autopilot-network",
        help="Whether a run that autopilot approves gets the network access its plan asks "
        "for (default: no).",
        show_default=False,
    ),
]


def _autonomy_changes(
    autopilot: bool | None,
    rounds: int | None,
    usd: float | None,
    hours: float | None,
    network: bool | None,
) -> dict[str, Any]:
    """The values of ``settings.autonomy`` that the options of a command name."""
    given = (
        ("mode", None if autopilot is None else "autopilot" if autopilot else "supervised"),
        ("rounds", rounds),
        ("model_usd", usd),
        ("hours", hours),
        ("network", network),
    )
    return {name: value for name, value in given if value is not None}


@research_app.command("status")
def research_status(
    project: Annotated[
        str | None,
        typer.Argument(
            metavar="PROJECT",
            help="The project: its directory, its name in ./research, or its id. Default: "
            "the project in the current directory, or else every project in ./research.",
            show_default=False,
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Show where a project stands: round, phase, what waits for you, experiments, cost."""
    from paper_orchestra.research.project import PROJECT_FILE

    if project is None and not (Path.cwd() / PROJECT_FILE).is_file():
        _workspace_status(as_json)
        return
    opened = _open_project(project or Path.cwd())
    if as_json:
        typer.echo(json.dumps(_status_data(opened), indent=2, ensure_ascii=False, default=str))
        return
    _say(f"{opened.title} ({opened.id}) in {opened.root}")
    for line in _status_lines(opened, _state_as_it_is(opened)):
        _say(line)


def _workspace_status(as_json: bool) -> None:
    """One line per project in the default workspace.

    A project that cannot be opened is named on stderr with the reason and makes the
    exit code 1; the others are listed all the same.
    """
    from paper_orchestra.research.model import ProjectError
    from paper_orchestra.research.project import Project

    projects: list[Project] = []
    unreadable: list[str] = []
    for root in _project_dirs(WORKSPACE):
        try:
            projects.append(Project.open(root))
        except ProjectError as exc:
            unreadable.append(f"{root} cannot be opened: {exc}")
    if as_json:
        data = [_status_data(project) for project in projects]
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif not projects and not unreadable:
        _say(
            f"There are no research projects in ./{WORKSPACE}. Start one with "
            '`paper-orchestra research new --brief "..."`.'
        )
    else:
        for project in projects:
            counts = project.counts()
            waiting, undecided = counts["pending"]["total"], counts["decisions"]["open"]
            waits = [
                *([f"{waiting} waiting for you"] if waiting else []),
                *([f"{_plural(undecided, 'decision')} to answer"] if undecided else []),
            ]
            where = _where_it_stands(_state_as_it_is(project)).rstrip(".")
            _say(
                f'{project.root}  "{project.title}"  {where}'
                + (f"  ({', '.join(waits)})" if waits else "")
            )
    for problem in unreadable:
        _say(f"Error: {problem}", fg=typer.colors.RED, err=True)
    if unreadable:
        raise typer.Exit(1)


@research_app.command("advance")
def research_advance(
    project: ProjectArgument,
    phase: Annotated[
        _Phase | None,
        typer.Option(
            "--phase", help="Start from this phase instead of the next one.", show_default=False
        ),
    ] = None,
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", help="What you want the agents to do in this phase."),
    ] = None,
) -> None:
    """Let the agents work up to the next checkpoint.

    Prints one line per event to stderr as it goes.  Ctrl-C (or SIGTERM) stops at the
    next model call or ends the running job; the same command continues afterwards.
    What you approved runs without a model; the agents need one.

    Exit codes: 0 at a checkpoint, 1 when the engine failed, 3 when the next phase needs
    a model and none can be used (nothing was started), 130 when it was stopped.
    """
    _advance(_open_project(project), phase=phase.value if phase else None, instruction=instruction)


def _settled_in_words(done: Mapping[str, Sequence[str]], settings: Any) -> str:
    """What "also settle what waits now" did, in a sentence, and what becomes of the
    experiments that were plans before: said as the rule will treat them in this
    project (it approves none where they would run unsandboxed as the person)."""
    from paper_orchestra.research.experiments import rule_cannot_run, rule_leaves_runs

    if rule_leaves_runs(settings):
        planned = (
            "Experiments that were planned before still wait for your approval, as every "
            "run on this machine does (one that asks for a GPU is approved and run in the "
            "next stretch)."
        )
    elif rule_cannot_run(settings):
        planned = (
            "Experiments that were planned before are approved and run in the next stretch "
            "once Modal can be used on this machine."
        )
    else:
        planned = "Experiments that were planned before are approved and run in the next stretch."
    parts = [
        f"{len(done['accepted'])} page{'' if len(done['accepted']) == 1 else 's'} accepted",
        f"{len(done['applied'])} edit{'' if len(done['applied']) == 1 else 's'} applied",
        f"{len(done['defaulted'])} decision{'' if len(done['defaulted']) == 1 else 's'} left "
        "at the default",
    ]
    left = (
        f" {len(done['left'])} left for you (an edit to your own text, or one that cannot "
        "be applied as it stands)."
        if done["left"]
        else ""
    )
    return f"Autopilot settled what waited: {', '.join(parts)}.{left} {planned}"


def _kind_named(name: str) -> str:
    """A kind of page from how a person writes it: ``hypothesis`` or ``hypotheses``."""
    from paper_orchestra.research.model import KIND_OF_SECTION, KINDS

    kind = KIND_OF_SECTION.get(name.strip().lower(), name.strip().lower())
    if kind not in KINDS:
        _fail(f"'{name}' is not a kind of page; the kinds are: {', '.join(KINDS)}.")
    return kind


def _field_text(value: Any, name: str = "") -> str:
    """A front-matter value as one piece of a line, written as a page writes it."""
    if value is None or value == "":
        return "(not set)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        # A command is one line to read, not a list of words.
        joined = shlex.join(map(str, value)) if name == "command" else ", ".join(map(str, value))
        return joined or "(none)"
    return str(value)


def _edit_changes(project: Project, proposal: EditProposal) -> list[str]:
    """What accepting an edit proposal would change, line by line: every field from and
    to, every relation it adds, and the unified diff of the text.

    An agent's one-line summary is what it says the edit does.  The person decides on
    what it does.
    """
    import difflib

    from paper_orchestra.research.model import ProjectError

    try:
        page = project.page(proposal.target)
    except ProjectError:
        return ["the page it would change no longer exists; it can only be rejected"]
    lines = [
        f"{name}: {_field_text(page.meta.get(name), name)} -> {_field_text(value, name)}"
        for name, value in proposal.fields.items()
    ]
    lines += [
        f"{name}: adds {_field_text(targets)}" for name, targets in proposal.relations_add.items()
    ]
    if proposal.body is None or proposal.body == page.body:
        return [*lines, "text: unchanged"]
    diff = difflib.unified_diff(
        page.body.splitlines(),
        proposal.body.splitlines(),
        fromfile=f"{page.path} (as it is)",
        tofile=f"{page.path} (as {_author_in_words(proposal.author)} suggests it)",
        lineterm="",
    )
    return [*lines, "text:", *diff]


def _author_in_words(author: str) -> str:
    """``the scout`` for ``agent:scout``, ``you`` for the person, ``the engine`` for what
    code wrote: an author as the app names them."""
    from paper_orchestra.research.model import HUMAN, is_agent, role_of

    if author == HUMAN:
        return "you"
    return f"the {role_of(author)}" if is_agent(author) else "the engine"


def _say_changes(project: Project, proposal: EditProposal, indent: str) -> None:
    for line in _edit_changes(project, proposal):
        _say(f"{indent}{line}")


def _say_if_no_longer_approved(project: Project, target: str) -> None:
    """After an accepted edit to an experiment whose run was approved: say when the
    experiment is no longer what was approved, so that it is not found out at the run."""
    from paper_orchestra.research.model import ProjectError

    with contextlib.suppress(ProjectError):
        page = project.page(target)
        if page.kind != "experiment" or page.state not in ("approved", "queued"):
            return
        from paper_orchestra.research import experiments

        standing = experiments.approval(project, str(page.id))
        if not standing["matches"]:
            _say(
                f"  {page.id} is no longer what you approved ({'; '.join(standing['changed'])}). "
                "It will not run until you approve it again: "
                f"`paper-orchestra research approve {_quoted(project.root)} {page.id}`"
            )


def _number_lines(report: Mapping[str, Any] | None, lead: str = "numbers") -> list[str]:
    """What the number check found in a finding (or in a finding as an edit would leave
    it), for the person who decides: what was looked for, and which numbers are on no
    record or only on another one.  Nothing for a text without any number."""
    if not report or not (report["checked"] or report["skipped"]):
        return []
    lines = [f"{lead}: {report['summary']}"]
    if report["unverified"]:
        lines.append(f"  not found: {', '.join(report['unverified'])}")
    if report["elsewhere"]:
        lines.append(f"  found only elsewhere: {', '.join(report['elsewhere'])}")
    return lines


def _checked_numbers(project: Project, item: ResearchPage | EditProposal) -> list[str]:
    """The number check of a proposed finding, or of the finding a proposed edit changes
    as it would read with the edit applied."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research.model import EditProposal, ProjectError

    with contextlib.suppress(ProjectError):
        if isinstance(item, EditProposal):
            return _number_lines(
                experiments.edit_report(project, item), "numbers, as the finding would read"
            )
        if item.kind == "finding":
            return _number_lines(experiments.number_report(project, item))
    return []


def _page_named(project: Project, path: str) -> str:
    """A page as a line for the person names it: its id (``H2``) or its title in quotation
    marks; the path only for a page that is gone."""
    from paper_orchestra.research.model import ProjectError

    try:
        page = project.page(path)
    except ProjectError:
        return path
    return page.id or f'"{page.title}"'


def _list_pending(project: Project) -> None:
    from paper_orchestra.research.model import EditProposal

    pending = project.pending()
    waiting = [_pending_item(item) for item in pending]
    if not waiting:
        _say(f"Nothing is waiting for your decision in {project.root}.")
        return
    _say(f"{len(waiting)} waiting for your decision in {project.root}:")
    width = max(len(item["ref"]) for item in waiting)
    indent = " " * (width + 6)
    for item, source in zip(waiting, pending, strict=True):
        what = (
            f"edit of {_page_named(project, item['target'])}:"
            if item["kind"] == "edit"
            else item["kind"]
        )
        verb = "suggested" if item["kind"] == "edit" else "proposed"
        who = f"({verb} by {_author_in_words(item['author'])})"
        _say(f"  {item['ref'].ljust(width)}  {what} {item['title']}  {who}")
        if isinstance(source, EditProposal):
            # Shown here, before anything can be accepted: the summary is the agent's word.
            _say_changes(project, source, indent)
        elif item.get("summary") and item["summary"] != item["title"]:
            _say(f"{indent}{item['summary']}")
        for line in _checked_numbers(project, source):
            _say(f"{indent}{line}")
    _say(
        "Decide with --accept REF, --reject REF (with --reason TEXT) or --accept-all KIND, "
        "or in the web app (`paper-orchestra ui`)."
    )


def _decide(project: Project, ref: str, decision: str, reason: str | None) -> bool:
    """Make one decision and say what became of it; ``False`` when it could not be made."""
    from paper_orchestra.research.model import EditProposal, ProjectError

    changes: list[str] = []
    with contextlib.suppress(ProjectError):
        # Against the page as it is before the decision: afterwards there is no difference.
        if decision == "accept" and ref.lower().startswith("ep-"):
            proposal = project.proposal(ref.lower())
            changes = [*_edit_changes(project, proposal), *_checked_numbers(project, proposal)]
    try:
        outcome = project.review(ref, decision, by="human", reason=reason)
    except ProjectError as exc:
        # Also a conflict: its message says what was and was not applied.
        _say(f"{ref}: {exc}", fg=typer.colors.RED, err=True)
        return False
    verb = "Accepted" if decision == "accept" else "Rejected"
    if isinstance(outcome, EditProposal):
        target = _page_named(project, outcome.target)
        _say(f"{verb} {outcome.id}: {outcome.summary} (an edit of {target})")
        for line in changes:
            _say(f"    {line}")
        if decision == "accept":
            _say_if_no_longer_approved(project, outcome.target)
        return True
    name = outcome.id or outcome.path
    _say(f'{verb} {name}: {outcome.kind} "{outcome.title}"')
    if decision == "reject" and not outcome.decision_reason:
        # Said as it is: the agents are told why a page was turned down only if you say.
        _say(
            "  No reason was recorded: the agents see its title and that you turned it down, "
            "not why. `--reason TEXT` with the rejection tells them."
        )
    if decision == "accept" and outcome.kind == "experiment" and outcome.state == "planned":
        _say(
            f"  {name} is accepted as a plan; nothing runs until you approve it: "
            f"`paper-orchestra research approve {_quoted(project.root)} {name}`"
        )
    if decision == "reject" and outcome.kind == "experiment":
        for line in _what_rests_on(project, outcome):
            _say(f"  {line}")
    return True


def _what_rests_on(project: Project, experiment: ResearchPage) -> list[str]:
    """What the person should look at after rejecting an experiment: the findings that
    interpret its runs and the verdicts on hypotheses that came through them."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research.model import ProjectError

    try:
        resting = experiments.dependents(project, experiment.path)
    except ProjectError:
        return []
    name = resting["experiment"]
    lines = []
    for finding in resting["findings"]:
        how = f"rests on {name} alone" if finding["only"] else f"rests in part on {name}"
        lines.append(f'{finding["id"]} ({finding["review"]}) "{finding["title"]}" {how}.')
    for verdict in resting["hypotheses"]:
        through = ", ".join(str(item) for item in verdict["through"])
        lines.append(
            f'{verdict["id"]} "{verdict["title"]}" is {verdict["state"]} through {through}.'
        )
    if not lines:
        return []
    return [
        f"The records of {name}'s runs are no longer evidence for the number check, and a "
        f"paper does not report a finding that rests on {name} alone. Have a look at what "
        "rested on it:",
        *lines,
    ]


@research_app.command("review")
def research_review(
    project: ProjectArgument,
    accept: Annotated[
        list[str] | None,
        typer.Option(
            "--accept", metavar="REF", help="Accept a page (H2, E1 or a path) or an edit (ep-...)."
        ),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", metavar="REF", help="Reject a page or an edit proposal."),
    ] = None,
    accept_all: Annotated[
        list[str] | None,
        typer.Option(
            "--accept-all",
            metavar="KIND",
            help="Accept every proposed page of a kind (for example hypothesis) that is "
            "not rejected in the same command.",
        ),
    ] = None,
    reason: Annotated[
        str | None,
        typer.Option(
            "--reason",
            help="Why. Recorded with the rejections (the agents read it before they propose "
            "again), or with the acceptances when nothing is rejected.",
        ),
    ] = None,
) -> None:
    """Accept or reject what the agents proposed; without a decision, list what is waiting.

    Accepting an experiment accepts its plan only: it runs once you approve it
    (`research approve`).
    """
    opened = _open_project(project)
    accepts, rejects = list(accept or ()), list(reject or ())
    kinds = [_kind_named(name) for name in accept_all or ()]

    def named(ref: str) -> str:
        """What ``ref`` stands for: the page's path, or the id of an edit proposal."""
        return opened.resolve(ref) or ref.lower()

    both = {named(ref) for ref in accepts} & {named(ref) for ref in rejects}
    if both:
        twice = next(ref for ref in accepts if named(ref) in both)
        _fail(f"{twice} is named for both --accept and --reject.")
    if not (accepts or rejects or kinds):
        if reason:
            _fail("--reason goes with a decision: add --accept, --reject or --accept-all.")
        _list_pending(opened)
        return
    # Rejections first, so that "--accept-all hypothesis --reject H4" means all but H4.
    made = [_decide(opened, ref, "reject", reason) for ref in rejects]
    accept_reason = None if rejects else reason
    made += [_decide(opened, ref, "accept", accept_reason) for ref in accepts]
    for kind in kinds:
        proposed = opened.pages(kind=kind, review="proposed")
        if not proposed:
            _say(f"No {kind} is waiting for a decision.")
        made += [_decide(opened, page.path, "accept", accept_reason) for page in proposed]
    left = opened.counts()["pending"]["total"]
    _say(
        (f"{left} still waiting for your decision. " if left else "Nothing else is waiting. ")
        + f"`paper-orchestra research advance {_quoted(opened.root)}` goes on from here."
    )
    if not all(made):
        raise typer.Exit(1)


@research_app.command("approve")
def research_approve(
    project: ProjectArgument,
    experiment: Annotated[
        str,
        typer.Argument(
            metavar="EXPERIMENT", help="The experiment, for example E1.", show_default=False
        ),
    ],
    backend: Annotated[
        _Compute | None,
        typer.Option(
            "--backend",
            help="Where it runs. Default: the project's backend (Modal for a GPU), or what "
            "you chose when you approved this experiment before.",
        ),
    ] = None,
    gpu: Annotated[
        str | None, typer.Option("--gpu", help="GPU type on Modal, for example T4.")
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", min=1, help="Time limit of the run in seconds."),
    ] = None,
    network: Annotated[
        bool | None,
        typer.Option(
            "--network/--no-network",
            help="Let the run reach the network, or not. Default: not; a plan that asks for "
            "the network is approved only with one of the two.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Approve an experiment's run: your decision, and the only way one gets to run.

    The experiment's page must be accepted first.  The approval records exactly what may
    run (the code and how it runs); if either changes afterwards, the experiment needs
    your approval again.  Where it runs and whether it may reach the network are never
    taken from the plan, which an agent may have written: without --backend it runs on
    the project's backend, and a plan that asks for the network needs --network or
    --no-network.  The time limit and the GPU type left out keep what the plan says.
    """
    from paper_orchestra.research.model import ProjectError

    opened = _open_project(project)
    try:
        from paper_orchestra.research import experiments

        page = experiments.approve(
            opened,
            experiment,
            backend=backend.value if backend else None,
            gpu=gpu,
            timeout_seconds=timeout,
            network=network,
        )
    except (ProjectError, ComputeError) as exc:
        _fail(str(exc) + _how_to_accept(opened, experiment), code=1)
    name = page.id or page.path
    _say(f"{name} is approved to run{_approved_as(page)}. {_when_it_runs(opened, str(name))}")
    if _grants_model_access(page):
        _say(MODEL_ACCESS_SAID)
    for line in _what_was_approved(opened, page):
        _say(line)


@research_app.command("adopt-code")
def research_adopt_code(
    project: ProjectArgument,
    experiment: Annotated[
        str,
        typer.Argument(
            metavar="EXPERIMENT", help="The experiment, for example E1.", show_default=False
        ),
    ],
    run: Annotated[
        int | None,
        typer.Option(
            "--run",
            min=1,
            help="The run whose code to take. Default: the latest run an agent changed "
            "the code in.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Make the code an agent left after a run the experiment's code.

    Under the sdk runtime the agent that carries an experiment out repairs the code in
    its own folder, and the experiment's code stays what you approved: run again, it
    would fail the same way and be repaired again.  This writes the files the agent
    changed or added into the experiment's code as your own edit.  The earlier
    versions are kept, and the experiment must be approved again before it runs.
    """
    from paper_orchestra.research.model import ProjectError

    opened = _open_project(project)
    try:
        from paper_orchestra.research import experiments

        page = opened.page(experiment)
        adopted = experiments.adopt_run_code(opened, str(page.id or experiment), run)
    except (ProjectError, ComputeError) as exc:
        _fail(str(exc), code=1)
    name, root = page.id or page.path, _quoted(opened.root)
    _say(
        f"The code of run {adopted['run']} is now the code of {name}: "
        f"{', '.join(adopted['files'])}. The earlier versions are kept. `paper-orchestra "
        f"research approve {root} {name}` approves it to run again."
    )
    if adopted["removed"]:
        _say(f"Left in place, though the agent had removed it: {', '.join(adopted['removed'])}.")
    for problem in adopted["problems"]:
        _say(f"The check of the code found: {problem}", fg=typer.colors.YELLOW)


def _how_to_accept(project: Project, ref: str) -> str:
    """For an approval that was refused: when the experiment's page is not accepted, the
    command that accepts it (the refusal says that it has to be, not how); else nothing."""
    from paper_orchestra.research.model import ProjectError

    with contextlib.suppress(ProjectError, OSError):
        page = project.page(ref)
        if page.kind == "experiment" and page.review != "accepted":
            name = page.id or page.path
            return (
                f" `paper-orchestra research review {_quoted(project.root)} --accept {name}` "
                "accepts it."
            )
    return ""


def _what_was_approved(project: Project, page: ResearchPage) -> list[str]:
    """What the approval of ``page`` covers, for the person to hold against what they
    read: the command (and the repository it runs in), the code files with the
    approval's digest, and what a reading of the code found worth knowing."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research.model import ProjectError

    name = str(page.id or page.path)
    lines: list[str] = []
    with contextlib.suppress(ProjectError):
        lines.append(_what_runs(experiments.load_plan(project, name)))
    digest = str(page.meta.get("approved_digest") or "")
    stamped = f"approved digest {digest[:12]}" if digest else "no digest was recorded"
    files = experiments.code_files(project, name)
    if files:
        listed = ", ".join(
            f"{file} ({len(text.splitlines())} lines)" for file, text in files.items()
        )
        where = _quoted(experiments.code_dir(project, name))
        lines.append(f"Its code, in {where}: {listed}; {stamped}.")
        if digest:
            # Two digests are shown for one experiment, and they are of different things.
            code = experiments.code_digest(project, name)[:12]
            lines.append(
                "The approved digest covers the plan and the code together; the code "
                f"alone is sha256 {code}, which is what a result page names under Code."
            )
    elif digest:
        lines.append(f"It has no code files of its own; {stamped}.")
    with contextlib.suppress(ProjectError):
        lines += experiments.split_problems(experiments.check_code(project, name))[1]
    return lines


def _what_runs(plan: Any) -> str:
    """The command a run of ``plan`` executes and, in repo mode, where: the repository
    as it may be written down (never the token of its URL).  The packages the plan asks
    for are named with it: the approval pins them, and where they are installed for the
    run their code runs too."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research import repo as repositories

    told = f"It runs `{shlex.join(experiments.job_command(plan))}`"
    if plan.mode == "repo" and plan.repo:
        how = "a clone of" if repositories.is_url(plan.repo) else "a copy of your repository"
        told += f" in {how} {repositories.display_url(plan.repo)}"
        told += f" at {plan.repo_ref}" if plan.repo_ref else ""
    if plan.packages:
        told += f", with the packages {', '.join(plan.packages)}"
    return told + "."


def _asks_for_network(page: ResearchPage) -> bool:
    """Whether an experiment's page says that its run reaches the network."""
    value = page.meta.get("network")
    return value is True or str(value).strip().lower() in _TRUE


#: Said when a run is approved whose code will be able to call the Anthropic API: the
#: person learns of a spend that nothing here counts before they grant it, not after.
MODEL_ACCESS_SAID = (
    "With network access this run's code can call the Anthropic API with the worker's "
    "federation settings (PAPER_ORCHESTRA_JOB_MODEL_ACCESS is on): those calls are billed "
    "to that account and are counted against no limit of the project."
)


def _grants_model_access(page: ResearchPage) -> bool:
    """Whether the approval on ``page`` lets the run's code call the Anthropic API: a
    local run with network access while model access for jobs is on."""
    from paper_orchestra.compute.local import job_model_access

    local = str(page.meta.get("backend") or "local") == "local"
    return local and _asks_for_network(page) and job_model_access()


def _approved_as(page: ResearchPage) -> str:
    """What the approval recorded on the page: where the run goes, with what, for how
    long at most, and whether it may reach the network."""
    meta = page.meta
    told = f" on {meta['backend']}" if meta.get("backend") else ""
    gpu = str(meta.get("gpu") or "").strip()
    if gpu and gpu.lower() not in _FALSE | {"none", "null", "cpu"}:
        told += f" with a {gpu} GPU"
    limit = meta.get("timeout_seconds")
    if isinstance(limit, int | float) and not isinstance(limit, bool):
        told += f", for up to {limit:g} s"
    if _asks_for_network(page):
        told += ", and it may reach the network"
    elif "network" in meta:
        told += ", without network access"
    return told


def _run_line(project: Project, record: Mapping[str, Any]) -> str:
    """What became of one run, with where to read more."""
    from paper_orchestra.compute.base import LOG_NAME
    from paper_orchestra.research.model import page_path

    experiment, number = str(record.get("experiment")), record.get("number")
    status = str(record.get("status"))
    took = f"{float(record.get('seconds') or 0):.1f} s"
    head = f"{experiment} run {number}"
    if status == "succeeded":
        metrics = [f"{name}={value}" for name, value in (record.get("metrics") or {}).items()]
        more = len(metrics) - RUN_LINE_METRICS
        shown = ", ".join(metrics[:RUN_LINE_METRICS]) + (f" and {more} more" if more > 0 else "")
        result = page_path("result", name=f"{experiment}-run{number}")
        return (
            f"{head} succeeded in {took}."
            + (f" Metrics: {shown}." if metrics else "")
            + f" Result: {project.root / result}"
        )
    run_dir = project.experiment_dir(experiment) / "runs" / str(number)
    why = record.get("error") or (
        "it ran out of time" if record.get("timed_out") else f"exit code {record.get('exit_code')}"
    )
    verb = "was cancelled" if status == "cancelled" else status
    return f"{head} {verb} after {took}: {_sentence(str(why))} Log: {run_dir / LOG_NAME}"


@research_app.command("run")
def research_run(
    project: ProjectArgument,
    experiments: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="EXPERIMENT",
            help="Experiments to run, for example E1 E4. Default: every approved one.",
            show_default=False,
        ),
    ] = None,
    queue: Annotated[
        bool,
        typer.Option(
            "--queue",
            help="After the experiments named (or every approved one), go on with whatever "
            "is approved meanwhile, in the order of the approvals, until nothing waits. "
            "The app starts its runs this way.",
        ),
    ] = False,
) -> None:
    """Run approved experiments now and write their result records.

    Under the classic runtime no model is involved unless a run fails for a programming
    error and the code is repaired.  Under the sdk runtime an agent carries each run
    out; where the agents cannot start, the command says why and exits with code 3,
    as `research advance` does.  `research advance` then analyses the results first.
    One approval is one run: an experiment whose approved run has ended is not run again.

    Without --queue the command runs the experiments it was given (or the ones that
    were approved when it started) and ends.  With --queue it looks again after every
    run and goes on with every experiment that is approved by then, so that one
    approved while another runs needs no second command; each is tried once, and a
    stop ends the line.
    """
    from paper_orchestra.research.model import ProjectError

    opened = _open_project(project)
    root = _quoted(opened.root)
    _settle_what_ended(opened)
    # Also what reads as running: a worker that died leaves its experiment that way, and
    # the engine settles such a run and starts it again.  One that a live worker is
    # running is refused by the engine, which says so.
    waiting = opened.pages(
        kind="experiment", review="accepted", state=["approved", "queued", "running"]
    )
    refs = list(experiments or ()) or [page.id or page.path for page in waiting]
    if queue and not experiments:
        # The line as the app shows it: what a dead worker left running first (it is
        # settled and started again), then what waits, in the order of the approvals.
        left = [str(page.id or page.path) for page in waiting if page.state == "running"]
        refs = left + [name for name in _next_in_line(opened, set()) if name not in left]
    if not refs:
        _say(
            "No experiment is approved to run. Approve one with "
            f"`paper-orchestra research approve {root} E<n>`; "
            f"`paper-orchestra research status {root}` shows the experiments."
        )
        return
    outcomes: list[str | None] = []
    with _working(opened, "run") as events:
        from paper_orchestra.research.engine import Engine, WorkerBusy

        engine = Engine(opened)
        engine.drains = queue
        ahead: list[str] = list(refs)
        tried: set[tuple[str, str]] = set()
        while ahead:
            ref = ahead.pop(0)
            tried.add(_approval_tried(opened, ref))
            # Looked up again after every run: the person may approve one more while
            # another runs, and may take one back that was waiting.
            look_again = queue and not ahead
            try:
                record = engine.run_experiment(ref)
            except WorkerBusy:
                raise  # true of the project, so of every experiment: said once
            except (ProjectError, ComputeError) as exc:
                if _needs_a_model(exc):
                    # The agents that carry experiments out cannot start (the sdk
                    # runtime): true of every experiment, and no error of the project.
                    # Said once, with the exit code `advance` gives for the same thing.
                    raise
                # One experiment that cannot run does not keep the others from running.
                events.flush()
                _say(f"Error: {exc}", fg=typer.colors.RED, err=True)
                outcomes.append(None)
                if look_again:
                    ahead = _next_in_line(opened, tried)[:1]
                continue
            events.flush()
            _say(_run_line(opened, record))
            changed = _mapping(record.get("code_as_run"))
            if any(changed.get(verb) for verb in ("changed", "added", "removed")):
                # The result comes from other code than was approved: said where the
                # result is, with what makes that code the experiment's.  A plan that
                # came without code was given none to change: the agent wrote it.
                _say(
                    f"  The agent that carried it out {_did_to_the_code(changed)} "
                    f"(kept in runs/{record.get('number')}/code-as-run). `paper-orchestra "
                    f"research adopt-code {root} {record.get('experiment')}` makes it the "
                    "experiment's code; without that, a new run starts from the approved code."
                )
            outcomes.append(str(record.get("status")))
            if getattr(engine, "stop_requested", False):
                # A stop request (the run in flight went back to the queue) ends the
                # line: nothing else is started.  A run the person cancelled does not.
                _stopped(opened, "run")
            if look_again:
                ahead = _next_in_line(opened, tried)[:1]
                if ahead:
                    _say(f"{ahead[0]} was approved meanwhile and waits in line: it runs next.")
    # A run the person cancelled left nothing to analyse.
    if any(outcome in ("succeeded", "failed") for outcome in outcomes):
        _say(
            f"Next: `paper-orchestra research advance {root}` analyses the new results "
            "before anything else and stops at the checkpoint after the analysis "
            "(`--phase analyze` asks for just that step)."
        )
    if any(outcome != "succeeded" for outcome in outcomes):
        raise typer.Exit(1)


def _when_it_runs(project: Project, name: str) -> str:
    """What follows an approval, as things stand: a worker that runs experiments one
    after the other goes on with it (`research run --queue`, which the app starts), a
    worker in a phase may still reach it, and with nobody at work a command runs it."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research.model import ProjectError

    root = _quoted(project.root)
    run = f"`paper-orchestra research run {root} {name}`"
    alive = False
    try:
        state = _engine_state(project)
        worker = state.get("worker") if isinstance(state.get("worker"), Mapping) else None
        if worker is not None:
            from paper_orchestra.research.engine import worker_alive

            alive = worker_alive(state)
    except (ProjectError, OSError, ValueError):
        state, worker = {}, None
    if worker is not None and alive:
        if worker.get("queue"):
            try:
                after = experiments.line(project)["after"].get(name)
            except (ProjectError, OSError, ValueError):
                after = None
            return (
                f"It runs after {after}: " if after else "It runs next: "
            ) + "the worker that is running experiments goes on with it, and no command is needed."
        if state.get("status") == "running":
            return (
                "A worker is at work on this project: its run phase runs the experiment "
                f"if that phase is still to come. If it is not run by the time the worker "
                f"reaches its checkpoint, {run} runs it."
            )
        return (
            "A worker is running experiments it was started for and does not go on to "
            f"this one: when it has ended, {run} runs it."
        )
    return (
        f"{run} runs it now; `paper-orchestra research advance {root}` runs "
        "everything approved and analyses the results."
    )


def _approval_of(page: Any) -> tuple[str, str]:
    """What one try of an experiment is a try of: the experiment and the approval it
    has (its id and ``approved_at``).  One approval is one run, so a line of runs tries
    each approval once; approved again, the experiment is in line again."""
    return str(page.id or page.path), str(page.meta.get("approved_at") or "")


def _approval_tried(project: Project, ref: str) -> tuple[str, str]:
    """:func:`_approval_of` the experiment ``ref`` names, as it is now; ``ref`` as it is
    when it names none, which the engine says when it is asked to run it."""
    from paper_orchestra.research.model import ProjectError

    try:
        return _approval_of(project.page(ref))
    except (ProjectError, OSError, ValueError):
        return str(ref), ""


def _next_in_line(project: Project, tried: Collection[tuple[str, str]]) -> list[str]:
    """The experiments that wait for their run now and whose approval this command has
    not tried, in the order of their approvals.  The project is read again: another
    process (the app, a second terminal) is what approves one while this command runs."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research.model import ProjectError

    try:
        project.reload()
        waiting = experiments.waiting_in_line(project)
    except (ProjectError, OSError, ValueError):
        return []
    return [str(page.id) for page in waiting if _approval_of(page) not in tried]


def _did_to_the_code(as_run: Mapping[str, Any]) -> str:
    """What the agent that carried a run out did to the code, from the record's
    ``code_as_run``: it wrote the code where the plan came with none (``approved`` is
    the number of files that were approved), and changed what it was given otherwise."""
    wrote = "approved" in as_run and not as_run.get("approved") and not as_run.get("changed")
    return "wrote the code for this run" if wrote else "changed the code it was given"


def _settle_what_ended(project: Project) -> None:
    """Before anything is listed as waiting to run: an experiment whose approved run is
    over although its page does not say so (its worker stopped or died right after the
    job ended) is moved to how that run ended.  One approval is one run; such a page
    must not be offered, or run, again.  Nothing is started and nothing is waited for."""
    from paper_orchestra.research.model import ProjectError

    with contextlib.suppress(ProjectError, ComputeError, OSError):
        from paper_orchestra.research import experiments

        experiments.recover(project, follow=False)


@research_app.command("settings")
def research_settings(
    project: ProjectArgument,
    runtime: Annotated[
        str | None,
        typer.Option(
            "--runtime",
            metavar="classic|sdk",
            help="How the agents are run: `classic` (the tool loop over a model API) or "
            "`sdk` (Claude Agent SDK agents with a shell, the web and the board).",
        ),
    ] = None,
    model: AgentModelOption = None,
    fast: AgentFastOption = None,
    ultracode: AgentUltracodeOption = None,
    credential: AgentCredentialOption = None,
    sandbox: AgentSandboxOption = None,
    compute: ExperimentComputeOption = None,
    autopilot: AutopilotOption = None,
    rounds: AutopilotRoundsOption = None,
    stretch_usd: AutopilotUsdOption = None,
    hours: AutopilotHoursOption = None,
    stretch_network: AutopilotNetworkOption = None,
    settle: Annotated[
        bool,
        typer.Option(
            "--settle",
            help="With autopilot on: also settle what waits now. Proposed pages are "
            "accepted, suggested edits applied and open decisions left at their default, "
            "each recorded as autopilot's doing; the experiments that were planned before "
            "are the rule's from then on (approved and run in the next stretch, where it "
            "approves runs at all: not those that would run on this machine unsandboxed).",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the settings as JSON.")] = False,
) -> None:
    """Show how a project's agents are run and who decides, or change it.

    Without an option this prints `settings.agents` and `settings.autonomy`. With
    options it changes the values named and leaves the rest; the change holds from the
    next phase on. `--compute` sets where experiments run (`settings.compute`).

    Autopilot is yours alone to switch on: no agent can run this command, write the
    project's file or reach the app. What waits when you switch it on is left for you
    unless you say `--settle`. It is refused for a project whose agents work outside
    the sandbox.
    """
    from paper_orchestra.research.model import ProjectError

    opened = _open_project(project)
    changes = _agent_changes(runtime, model, fast, ultracode, credential, sandbox)
    unattended = _autonomy_changes(autopilot, rounds, stretch_usd, hours, stretch_network)
    settled: dict[str, list[str]] | None = None
    try:
        if changes or unattended or compute is not None:
            update = {
                **({"agents": changes} if changes else {}),
                **({"autonomy": unattended} if unattended else {}),
                **({"compute": compute.value} if compute is not None else {}),
            }
            opened.settings.update(update)
            opened.save_settings()
        if settle:
            settled = opened.settle_waiting()
    except ProjectError as exc:
        _fail(str(exc))
    agents = opened.settings.agents
    if as_json:
        shown = {
            **agents.to_dict(),
            "autonomy": opened.settings.autonomy.to_dict(),
            "compute": opened.settings.compute,
        }
        typer.echo(json.dumps(shown, indent=2, ensure_ascii=False, default=str))
        return
    if changes or unattended or compute is not None:
        named = [*changes, *(f"autonomy.{name}" for name in unattended)]
        named += ["compute"] if compute is not None else []
        _say(f"Changed {', '.join(named)} for {opened.title}.")
    if compute is not None:
        _say(_compute_in_words(opened.settings.compute))
    _say(_autonomy_in_words(opened.settings))
    if settled is not None:
        _say(_settled_in_words(settled, opened.settings))
    elif unattended.get("mode") == "autopilot":
        waiting = opened.counts()["pending"]["total"] + opened.counts()["decisions"]["open"]
        if waiting:
            _say(
                f"What waited before ({waiting} to decide) is left for you; "
                f"`paper-orchestra research settings {_quoted(opened.root)} --settle` lets "
                "autopilot settle it too."
            )
    if agents.runtime != "sdk":
        _say(
            "Agents: classic (this package's tool loop, with the models under "
            "settings.models). `--runtime sdk` makes them Claude Agent SDK agents."
        )
        return
    whose = "your Claude Code login" if agents.credential == "login" else "the worker's federation"
    _say(
        f"Agents: sdk (Claude Agent SDK agents on {agents.model}; fast mode "
        f"{'asked for' if agents.fast else 'off'}; helper agents and workflows "
        f"{'on' if agents.ultracode else 'off'}; credential: {whose}; sandbox "
        f"{'on' if agents.sandbox else 'OFF'})."
    )


@research_app.command("request")
def research_request(
    project: ProjectArgument,
    text: Annotated[
        str,
        typer.Argument(metavar="TEXT", help="What the agents should do.", show_default=False),
    ],
    role: Annotated[
        str | None,
        typer.Option(
            "--for",
            metavar="ROLE",
            help="Who it is for: scout, theorist, designer or analyst. Default: every agent.",
        ),
    ] = None,
    about: Annotated[
        str | None,
        typer.Option("--about", metavar="REF", help="The page it is about, for example H2."),
    ] = None,
) -> None:
    """Leave an instruction for the agents.

    A request for one role is given to that role's next phase and closed when the phase
    completes.  A request for every agent is shown to every phase that runs until the
    round's analysis has completed.  `research status` counts the open requests.
    """
    from paper_orchestra.research.model import FOR_ALL, ProjectError

    opened = _open_project(project)
    try:
        request = opened.add_request(text, target=about, role=role)
    except ProjectError as exc:
        _fail(str(exc))
    if request.role == FOR_ALL:
        whom = "for every agent"
        when = (
            "Every phase that runs is shown it until the analysis of the round has "
            "completed; then it is closed."
        )
    else:
        whom = f"for the {request.role}"
        when = (
            f"It is given to the next phase that is the {request.role}'s and closed when "
            "that phase completes."
        )
    target = f", about {request.target}" if request.target else ""
    _say(f"Left a request {whom}{target} ({request.id}). {when}")


#: How much of the person's own words the list of decisions repeats; one decision, asked
#: for by itself, is said whole.
_ANSWER_NAMED = 160


def _decision_lines(decision: Any, number: int | None = None) -> list[str]:
    """One decision as it is read at a terminal: the question, who asked, the options by
    number, what the agent would take, the default, and what was answered."""
    from paper_orchestra.research.decisions import answer_words

    asked = f"asked by the {decision.role}" if decision.role else "asked"
    where = f", round {decision.raised_by.get('round')}" if decision.raised_by.get("round") else ""
    lead = f"{number}. " if number is not None else ""
    lines = [f"{lead}{decision.id}  {decision.question}", f"   {asked}{where}"]
    if decision.page:
        lines[-1] += f"; about {decision.page}"
    for line in decision.context.splitlines():
        if line.strip():
            lines.append(f"   {line.strip()}")
    for index, option in enumerate(decision.options, start=1):
        mark = "  (the agent would take this" if index == decision.recommended else ""
        if mark:
            mark += f": {decision.reason})" if decision.reason else ")"
        meaning = f": {option.meaning}" if option.meaning else ""
        lines.append(f"     {index}  {option.label}{meaning}{mark}")
    if decision.default:
        lines.append(f"   Until you answer, the agents go on with: {decision.default}")
    said = answer_words(decision)
    if said:
        answer = decision.answer
        lines.append(_readable(f"   Your answer at {answer.at}: {said}"))
        if len(decision.answers) > 1:
            lines.append(f"   ({len(decision.answers) - 1} earlier answers are kept in the file.)")
    return lines


def _decision_numbers(project: Project) -> dict[str, int]:
    """The number each decision has at the terminal: its place among all the project's
    decisions, the oldest first.

    It is not the place among what still waits: that would move with every answer,
    and a person who answers 1 and then 2 from one listing would answer the third.
    A decision keeps its number once it is answered, so the number also finds it
    again, to look at the answer or to change it.
    """
    return {decision.id: number for number, decision in enumerate(project.decisions(), start=1)}


def _decision_named(project: Project, ref: str) -> Any:
    """The decision ``ref`` names: its number (:func:`_decision_numbers`), or its id or
    the beginning of it (``d-7c4``, ``7c41``)."""
    from paper_orchestra.research.model import NotFound

    known = project.decisions()
    text = ref.strip().lower()
    if text.isdigit() and 1 <= int(text) <= len(known):
        return known[int(text) - 1]
    # A few digits are a number and nothing else: "7" must not find d-7c41e0aa when
    # the list has no seventh decision.
    numbered = text.isdigit() and len(text) <= 3
    matches = [d for d in known if d.id == text] or [
        d
        for d in known
        if text and not numbered and d.id.startswith(text if text.startswith("d-") else f"d-{text}")
    ]
    if len(matches) > 1:
        raise NotFound(
            f"{ref!r} is the beginning of {len(matches)} decisions; give more of the id."
        )
    if not matches:
        count = f"the project has {_plural(len(known), 'decision')}, numbered from 1"
        raise NotFound(
            f"There is no decision {ref!r}"
            + (f": {count if known else 'the agents have asked none'}." if numbered else ".")
        )
    return matches[0]


def _what_a_new_answer_drops(before: Any, after: Any) -> str:
    """What to say when an answer replaced one that held more: a new answer is the whole
    answer, so the option or the words of the earlier one that it does not repeat no
    longer hold.  Empty when nothing was dropped."""
    from paper_orchestra.research.decisions import answer_words

    was, now = before.answer, after.answer
    if was is None or was.dismissed or now is None:
        return ""
    had_option = before.chosen is not None and now.option is None
    had_words = bool(was.text) and not now.text
    if not had_option and not had_words:
        return ""
    if had_option:
        gone = f"option {was.option} no longer holds"
        keep = f'give it with the words: `--option {was.option} --say "..."`'
    else:
        gone = "your earlier words are no longer part of it"
        keep = f'give them again with the option: `--option {now.option} --say "..."`'
    return (
        f" It replaces your earlier answer ({answer_words(before, _ANSWER_NAMED)}) as a "
        f"whole: {gone}. "
        f"To keep both, {keep}."
    )


@research_app.command("decide")
def research_decide(
    project: ProjectArgument,
    decision: Annotated[
        str | None,
        typer.Argument(
            metavar="D",
            help="The decision: its number in the list, or its id (d-1a2b3c4d). A decision "
            "keeps its number when it is answered. Without it, what waits is listed.",
            show_default=False,
        ),
    ] = None,
    option: Annotated[
        int | None,
        typer.Option("--option", metavar="N", help="Choose option N of the decision."),
    ] = None,
    say: Annotated[
        str | None,
        typer.Option(
            "--say",
            metavar="TEXT",
            help="Answer in your own words; alone, or together with --option.",
        ),
    ] = None,
    dismiss: Annotated[
        bool,
        typer.Option("--dismiss", help="Not for you to decide: the agents use their default."),
    ] = False,
) -> None:
    """Answer what the agents asked you to decide.

    An agent raises a decision when the answer is your choice and no evidence settles
    it: what to study, which reading of your idea is meant, a trade-off of cost or
    scope.  The project does not wait for you: the agents go on with the default the
    question states, and every later session is shown your answer.  An answer can be
    changed: the new one holds, as a whole (an option or words of the earlier answer
    that it does not repeat are dropped, and said to be), and the earlier ones are
    kept in the file.
    """
    from paper_orchestra.research.decisions import answer_words
    from paper_orchestra.research.model import ProjectError

    opened = _open_project(project)
    answering = option is not None or say is not None or dismiss
    if decision is None:
        if answering:
            _fail(
                "Say which decision: its id or its number in the list (`research decide PROJECT`)."
            )
        known = opened.decisions()
        numbers = _decision_numbers(opened)
        waiting = [item for item in known if item.state == "open"]
        if not waiting:
            _say("No decision is waiting for you.")
        else:
            _say(f"{_plural(len(waiting), 'decision')} waiting for you:")
        for item in waiting:
            _say("")
            for line in _decision_lines(item, numbers[item.id]):
                _say(line)
        settled = [item for item in known if item.state != "open"]
        if settled:
            _say("")
            _say("Decided:")
            for item in settled:
                # (named by its beginning: `research decide PROJECT N` says all of it)
                said = answer_words(item, _ANSWER_NAMED)
                _say(f"  {numbers[item.id]}. {item.id}  {item.question}  ->  {said}")
        if waiting:
            _say("")
            _say(
                f"Answer with `paper-orchestra research decide {_quoted(opened.root)} "
                f'{numbers[waiting[0].id]} --option N`, `--say "..."` or `--dismiss`. A '
                "decision keeps its number when it is answered."
            )
        return
    try:
        found = _decision_named(opened, decision)
        if dismiss and (option is not None or say is not None):
            _fail("--dismiss leaves the decision to the agents; it goes with no answer.")
        if not answering:
            for line in _decision_lines(found):
                _say(line)
            return
        if dismiss:
            done = opened.dismiss_decision(found.id)
            _say(
                f'Left {done.id}, "{done.question}", to the agents. They go on with: '
                f"{done.default or 'what they would have done'}"
            )
            return
        done = opened.answer_decision(found.id, option=option, text=say or "")
    except ProjectError as exc:
        _fail(str(exc))
    changed = "Changed your answer to" if found.state != "open" else "Answered"
    # The question is said with the id: an id alone does not tell which one was answered.
    # (As it was written: the question and the answer are not sentences of the project
    # whose "by the human" would be said as "by you".)
    said = _sentence(f'{changed} {done.id}, "{done.question}": {answer_words(done)}')
    _say(
        f"{said} Every later session of the agents is shown it."
        f"{_what_a_new_answer_drops(found, done)}"
    )


@research_app.command("cancel")
def research_cancel(
    project: ProjectArgument,
    experiment: Annotated[
        str,
        typer.Argument(
            metavar="EXPERIMENT", help="The experiment, for example E1.", show_default=False
        ),
    ],
) -> None:
    """Cancel an experiment's run, or take back its approval.

    An approved experiment that has not started becomes a plan again.  One that waits in
    the queue is cancelled at once; one that is running is told to stop, and its worker
    ends the job and records the run as cancelled within about a second.
    """
    from paper_orchestra.research.model import ProjectError

    opened = _open_project(project)
    root = _quoted(opened.root)
    try:
        from paper_orchestra.research import experiments

        page = opened.page(experiment)
        if page.kind != "experiment" or not page.id:
            _fail(f"{experiment} is {page.kind} {page.label}, not an experiment such as E1.")
        name, before = page.id, page.state
        withdrawn = before == "approved" and _withdrawn(opened, name)
        if not withdrawn:
            experiments.request_cancel(opened, name)
        after = opened.page(name).state
    except (ProjectError, ComputeError) as exc:
        _fail(str(exc))
    again = f"`paper-orchestra research approve {root} {name}` approves a run of it again."
    if withdrawn:
        _say(f"{name} is a plan again: its approval was taken back and nothing ran. {again}")
    elif before not in ("approved", "queued", "running"):
        _say(f"{name} is {before}: there is nothing to cancel.")
    elif after == "cancelled":
        _say(f"{name} is cancelled. {again}")
    elif after in ("approved", "queued", "running"):
        _say(
            f"{name} was told to stop. The worker that runs it ends the job and records the "
            f"run as cancelled; `paper-orchestra research status {root}` shows when it has."
        )
    else:
        # Its run had ended already; the page now says how.
        _say(f"{name} is {after}: its run had ended before it could be cancelled.")


def _withdrawn(project: Project, name: str) -> bool:
    """Take back the approval of an experiment that reads ``approved``; false when there
    was none to take back: the page only read that way (it was saved from an editor)
    while a process runs the experiment, or after its run had ended."""
    from paper_orchestra.research import experiments
    from paper_orchestra.research.model import ProjectError

    try:
        experiments.withdraw(project, name)
    except ProjectError:
        return False
    return True


@research_app.command("write")
def research_write(
    project: ProjectArgument,
    template: TemplateOption,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model for every role. Default: the project's models."),
    ] = None,
    writer_model: WriterModelOption = None,
    reflection_model: ReflectionModelOption = None,
    plotting_model: PlottingModelOption = None,
    fast_model: Annotated[
        str | None,
        typer.Option("--fast-model", help="Model for high-volume calls such as paper discovery."),
    ] = None,
    image_model: ImageModelOption = None,
    effort: Annotated[
        _Effort | None,
        typer.Option("--effort", help="Reasoning effort. Default: the project's."),
    ] = None,
    research_cutoff: Annotated[
        str | None,
        typer.Option(
            "--research-cutoff",
            help="Ignore literature published after this month (YYYY-MM). Default: the "
            "project's cutoff, or this month.",
        ),
    ] = None,
    use_plotting: Annotated[
        bool | None,
        typer.Option(
            "--plotting/--no-plotting",
            help="Also generate figures from the results. Default: only the figures the "
            "runs produced.",
        ),
    ] = None,
    compute: Annotated[
        _Compute | None,
        typer.Option(
            "--compute", help="Where LaTeX compiles and figure code runs. Default: the project's."
        ),
    ] = None,
    lab_memory: Annotated[
        bool | None,
        typer.Option(
            "--lab-memory/--no-lab-memory",
            help="Read and extend the cross-run memory of process lessons. Default: the "
            "project's setting.",
        ),
    ] = None,
    dream: Annotated[
        _Dream | None,
        typer.Option("--dream", help="When a model consolidates the paper run's memory."),
    ] = None,
    resume: Annotated[
        bool | None,
        typer.Option("--resume/--no-resume", help="Continue an earlier write-up in paper/."),
    ] = None,
) -> None:
    """Write a paper from the project's wiki (optional last step).

    Only accepted pages reach the paper: the idea, accepted hypotheses, the experiments
    that ran, their result records and the accepted findings.  The paper pipeline runs
    in the project's paper/ directory.
    """
    from paper_orchestra.research.model import ProjectError

    opened = _open_project(project)
    given = {
        "model": model,
        "writer_model": writer_model,
        "reflection_model": reflection_model,
        "plotting_model": plotting_model,
        "fast_model": fast_model,
        "image_model": image_model,
        "effort": effort.value if effort else None,
        "use_plotting": use_plotting,
        "compute": compute.value if compute else None,
        "lab_memory": lab_memory,
        "dream": dream.value if dream else None,
        "resume": resume,
        "research_cutoff": research_cutoff,
    }
    # What is not given is left to the project's own settings.
    options = {name: value for name, value in given.items() if value is not None}
    _restore_interrupt()
    _sweep(opened)
    _forget_stop(opened)
    try:
        from paper_orchestra.research import writeup

        with _EventPrinter(opened):
            result = writeup.write_paper(opened, template=template, **options)
    except KeyboardInterrupt:
        _record_stop(opened)
        _say(
            "Stopped. The same command continues the write-up.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(INTERRUPTED) from None
    except (ProjectError, FileNotFoundError, ValueError, ComputeError) as exc:
        _fail(str(exc))
    _report_paper(result, opened.paper_dir)


@research_app.command("demo")
def research_demo(
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help=f"Where to build it (new or empty). Default: ./research/{DEMO_NAME}.",
        ),
    ] = None,
    pages: Annotated[
        int | None,
        typer.Option(
            "--pages",
            min=1,
            help="Fill the wiki with generated pages up to this many, to see how the "
            "views hold up with a large project (for example 150).",
        ),
    ] = None,
) -> None:
    """Build a demonstration project without calling any model.

    A complete small project on learning-rate warmup: sources, topics, questions,
    hypotheses, experiments (one of them really run on this machine), a finding and
    things waiting for a decision.  It is for looking at the web app and for its tests.
    """
    from paper_orchestra.research.model import ProjectError

    root = directory or _free_dir(WORKSPACE / DEMO_NAME)
    try:
        from paper_orchestra.research import demo

        project = demo.build_demo_project(root, pages=pages)
    except (ProjectError, ComputeError) as exc:
        _fail(str(exc))
    workspace = root.parent
    flag = "" if workspace == WORKSPACE else f" --workspace {_quoted(workspace)}"
    _say(
        f'Built "{project.title}" in {root}: {project.counts()["pages"]} pages, '
        f"{project.counts()['pending']['total']} waiting for a decision."
    )
    _say(f"Look at it with `paper-orchestra ui{flag}`.")


# -- ui ----------------------------------------------------------------------------


@app.command()
def ui(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", help="The directory that holds the research projects."),
    ] = WORKSPACE,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Address to listen on. On every address, loopback included, the app is "
            "opened with the link that carries the access token.",
        ),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=0, max=65535, help="Port.")] = 8600,
    browser: Annotated[
        bool,
        typer.Option(
            "--browser/--no-browser",
            help="Open the app in the browser. Without it, open the link the server prints.",
        ),
    ] = True,
    stop_workers_on_exit: Annotated[
        bool,
        typer.Option(
            "--stop-workers-on-exit",
            help="Stop the agents and experiments this server started when the server "
            "ends. Default: they go on, and the next server finds them.",
        ),
    ] = False,
) -> None:
    """Open the web app: the wiki, its graph, the review inbox, experiments and activity.

    The app asks for an access token, so that nobody else on this machine can use it
    as you. The server prints the link that carries the token once, in this terminal,
    when it starts, and opens it in your browser; a page opened without the token says
    so. Every start makes a new token: set PAPER_ORCHESTRA_UI_TOKEN (16 characters or
    more) to keep one across restarts. Other programs send it as
    `Authorization: Bearer <token>`.

    Agents and experiments belong to their project, not to the server: when the server
    ends they keep running, and it prints which do and how to stop one.
    """
    try:
        from paper_orchestra.web.app import serve
        from paper_orchestra.web.security import TokenError
    except ModuleNotFoundError as exc:
        missing = exc.name or "a package"
        if missing.split(".")[0] == "paper_orchestra":
            raise
        _fail(f"The web app needs {missing}, which is not installed: run `uv sync --extra ui`.")
    options: dict[str, Any] = {"stop_workers_on_exit": True} if stop_workers_on_exit else {}
    try:
        serve(workspace, host=host, port=port, open_browser=browser, **options)
    except TokenError as exc:
        _fail(str(exc))


# -- memory ------------------------------------------------------------------------

MemoryDirOption = Annotated[
    Path | None,
    typer.Option("--memory-dir", help="A memory directory, for example <run directory>/memory."),
]
RunDirOption = Annotated[
    Path | None, typer.Option("--run-dir", help="A run directory; its memory/ folder is used.")
]


def _memory_root(memory_dir: Path | None, run_dir: Path | None) -> Path:
    if memory_dir is not None and run_dir is None:
        root = memory_dir
    elif run_dir is not None and memory_dir is None:
        root = run_dir / "memory"
    else:
        _fail("Give exactly one of --memory-dir and --run-dir.")
    # Checked first because opening a wiki creates it where there is none.
    if not (root / "MEMORY.md").is_file():
        _fail(f"No memory found at {root} (there is no {root / 'MEMORY.md'}).")
    return root


def _open_wiki(memory_dir: Path | None, run_dir: Path | None) -> Wiki:
    """The wiki at a memory directory, for reading."""
    from paper_orchestra.memory.harness import CODE_OWNED_SECTIONS
    from paper_orchestra.memory.wiki import Wiki

    # create=False: looking at a memory must not add directories to it.
    return Wiki(_memory_root(memory_dir, run_dir), read_only=CODE_OWNED_SECTIONS, create=False)


def _run_reflection_model(run_dir: Path) -> str | None:
    """The reflection model of the run in ``run_dir``, when its ``state.json`` names one."""
    try:
        model = read_json(run_dir / "state.json")["config"]["models"]["reflection"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return model if isinstance(model, str) and model else None


def _open_harness(root: Path, model: str) -> ContextHarness:
    """A harness over a run's memory, built exactly as a run builds its own."""
    from paper_orchestra.memory.harness import ContextHarness

    cfg = PipelineConfig(
        base_dir=root.parent,
        template_dir=root.parent,
        models=ModelConfig(reflection=model),
        memory=MemoryConfig(project_dir=root, lab=False),
    )
    return ContextHarness.create(cfg)


@memory_app.command("show")
def memory_show(
    page: Annotated[
        str | None,
        typer.Argument(help="A page to print instead of the index, e.g. literature/Smith2020."),
    ] = None,
    memory_dir: MemoryDirOption = None,
    run_dir: RunDirOption = None,
) -> None:
    """Print MEMORY.md, the index every session starts from (or one page)."""
    wiki = _open_wiki(memory_dir, run_dir)
    rel = "MEMORY.md" if page is None else wiki.resolve_link(page)
    if rel is None:
        _fail(f"There is no page {page!r} in {wiki.root}.")
    typer.echo(wiki.read_text(rel).rstrip("\n"))


@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Words to look for.")],
    memory_dir: MemoryDirOption = None,
    run_dir: RunDirOption = None,
    limit: Annotated[int, typer.Option("--limit", "-k", min=1, help="Number of hits.")] = 8,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Only search one wiki section, e.g. literature."),
    ] = None,
) -> None:
    """Search the wiki pages and pending notes."""
    wiki = _open_wiki(memory_dir, run_dir)
    hits = wiki.search(query, k=limit, prefix=f"wiki/{section.strip('/')}" if section else "")
    if not hits:
        typer.echo("No matches.")
        return
    for hit in hits:
        typer.echo(f"{hit.path}  [{hit.score:.2f}]  {hit.title}")
        typer.echo(f"    {hit.snippet}")


@memory_app.command("lint")
def memory_lint(
    memory_dir: MemoryDirOption = None,
    run_dir: RunDirOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Check the wiki's health: broken links, oversize pages, the index budget."""
    wiki = _open_wiki(memory_dir, run_dir)
    report = wiki.lint()
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(f"Memory at {wiki.root}: {'healthy' if report.ok else 'needs attention'}")
        for source, target in report.broken_links:
            typer.echo(f"  broken link: {source} -> [[{target}]]")
        for path, size in report.oversize:
            typer.echo(f"  oversize page: {path} ({size} bytes)")
        if report.index_over_budget:
            typer.echo(
                f"  MEMORY.md is over budget: {report.index_lines} lines, "
                f"{report.index_bytes} bytes"
            )
        for path in report.missing_summary:
            typer.echo(f"  no summary: {path}")
        typer.echo(
            f"  {len(report.orphans)} page(s) without inbound links, "
            f"{report.pending_notes} note(s) waiting to be consolidated"
        )
    if not report.ok:
        raise typer.Exit(1)


@memory_app.command("dream")
def memory_dream(
    memory_dir: MemoryDirOption = None,
    run_dir: RunDirOption = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model that does the consolidation. Default: the reflection model of the "
            f"run the memory belongs to, or {DEFAULT_MODEL} when that is not recorded.",
        ),
    ] = None,
    focus: Annotated[
        str | None,
        typer.Option("--focus", help="What this consolidation should concentrate on or build."),
    ] = None,
) -> None:
    """Consolidate a run's memory now: fold notes into pages, repair links, rebuild the index."""
    root = _memory_root(memory_dir, run_dir)
    if (root / "MEMORY.md").read_text(encoding="utf-8").startswith("# Memory index: lab"):
        _fail(
            f"{root} is the lab memory, which is consolidated at the end of every run. "
            "Point --memory-dir at a run's memory directory."
        )
    # A run consolidates its memory with its reflection model, which may belong to another
    # provider than the default model.
    model = model or _run_reflection_model(root.parent) or DEFAULT_MODEL
    from paper_orchestra.llm.base import LLMConfigError
    from paper_orchestra.llm.router import get_client

    try:
        # Before the memory is opened: a consolidation that cannot reach its model would
        # only add a failed session to the memory's records.
        get_client(model)
    except LLMConfigError as exc:
        # The message of a missing credential goes on with setup instructions, which
        # `doctor` prints in full; here its headline is enough.
        headline = (str(exc) or type(exc).__name__).splitlines()[0]
        _fail(
            f"The consolidation was not started: model '{model}' cannot be used: {headline} "
            f"(details: paper-orchestra doctor --model {model})",
            code=1,
        )
    harness = _open_harness(root, model)
    report = harness.dream(
        stage="manual", focus=focus or MANUAL_DREAM_FOCUS, build=focus is not None, force=True
    )
    if report is None:
        _fail(f"Memory is not available at {root}.", code=1)
    typer.echo(
        f"Notes: {report.notes_seen} seen, {report.notes_archived} consolidated. "
        f"Pages changed: {len(report.pages_changed)}. Links repaired: {report.links_repaired}."
    )
    for path in report.pages_changed:
        typer.echo(f"  {path}")
    if report.summary:
        typer.echo(report.summary)
    elif not report.error:
        typer.echo("No model session was needed; the index and links were refreshed.")
    if report.error:
        _fail(f"The consolidation did not complete: {report.error}", code=1)


# -- modal -------------------------------------------------------------------------

_PASS_THROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}
DetachOption = Annotated[
    bool,
    typer.Option("--detach", "-d", help="Keep the run alive when this terminal disconnects."),
]
PrintCommandOption = Annotated[
    bool, typer.Option("--print-command", help="Print the modal command instead of running it.")
]


#: The options of the Modal entrypoint (``modal_app.main``), as its command line spells
#: them, for ``modal run --help``.  A test keeps the list in step with the entrypoint.
MODAL_RUN_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--raw-materials-dir DIR", "local directory with the idea file and the experimental log"),
    ("--template NAME|DIR", "bundled template name (for example cvpr2025) or a directory"),
    ("--output-dir DIR", "where the paper is downloaded (default: ./paper_output_<run id>)"),
    ("--run-id ID", "resume the run with this id, or download its results again"),
    ("--model MODEL", "model for every role that has no flag of its own"),
    ("--writer-model MODEL", "model for the outline, literature and writing agents"),
    ("--reflection-model MODEL", "model for reviewing, refinement and consolidation"),
    ("--plotting-model MODEL", "model that writes and critiques figures"),
    ("--fast-model MODEL", "model for high-volume calls such as paper discovery"),
    ("--image-model MODEL", "image-generation model for diagrams"),
    ("--effort LEVEL", "reasoning effort: low, medium, high, xhigh or max"),
    ("--plotting", "generate the figures from the experimental log"),
    ("--keep-provided-figures", "with --plotting, also use the figures in the raw materials"),
    ("--plotting-max-critic-rounds N", "self-correction rounds per generated figure"),
    ("--idea-filename NAME", "idea file inside the raw materials"),
    ("--experimental-log-filename NAME", "experimental log file inside the raw materials"),
    ("--research-cutoff YYYY-MM", "ignore literature published after this month"),
    ("--no-memory", "do not keep a wiki memory for the run"),
    ("--no-lab-memory", "do not read or extend the cross-run memory"),
    ("--dream MODE", "when a model consolidates memory: off, end or stages"),
    ("--no-resume", "start over instead of continuing the run with this id"),
)


def _modal_run_help() -> str:
    width = max(len(option) for option, _ in MODAL_RUN_OPTIONS)
    options = "\n".join(f"{option.ljust(width)}  {text}" for option, text in MODAL_RUN_OPTIONS)
    return (
        "Write one paper on Modal.\n\n"
        "Uploads the raw materials, runs the pipeline in a Modal container and downloads the "
        "paper. Every argument except --detach and --print-command is handed to the Modal "
        "entrypoint, whose options are listed here; --raw-materials-dir and --template are "
        "required. The run id is printed when the run starts: pass it as --run-id to resume "
        "that run after an interruption, or to download its results again.\n\n"
        # \b keeps the list as it is laid out instead of reflowing it into a paragraph.
        "\b\n" + options
    )


def _modal(action: str, target: str, args: Sequence[str], *, detach: bool, show: bool) -> None:
    command = [sys.executable, "-m", "modal", action]
    if detach:
        command.append("--detach")
    command += ["-m", target, *args]
    if show:
        typer.echo(shlex.join(command))
        return
    if importlib.util.find_spec("modal") is None:
        _fail("The Modal SDK is not installed. Install it with `uv sync --extra modal`.")
    raise typer.Exit(subprocess.call(command))


@modal_app.command("run", context_settings=_PASS_THROUGH, help=_modal_run_help())
def modal_run(
    ctx: typer.Context, detach: DetachOption = False, print_command: PrintCommandOption = False
) -> None:
    """Run the Modal entrypoint with the remaining arguments (help: ``_modal_run_help``)."""
    _modal("run", f"{MODAL_MODULE}::main", ctx.args, detach=detach, show=print_command)


@modal_app.command("deploy", context_settings=_PASS_THROUGH)
def modal_deploy(ctx: typer.Context, print_command: PrintCommandOption = False) -> None:
    """Deploy the app so that write_paper and write_batch can be called by name."""
    _modal("deploy", MODAL_MODULE, ctx.args, detach=False, show=print_command)


@modal_app.command("batch")
def modal_batch(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Where each paper's PDF, LaTeX source and usage report are downloaded.",
        ),
    ],
    template: TemplateOption,
    root_input_folder: RootInputFolderOption = None,
    paper_info: PaperInfoOption = None,
    n_papers: PaperCountOption = -1,
    idea_filename: IdeaFilenameOption = "idea_sparse.md",
    experimental_log_filename: LogFilenameOption = "experimental_log.md",
    model: ModelOption = None,
    writer_model: WriterModelOption = None,
    reflection_model: ReflectionModelOption = None,
    plotting_model: PlottingModelOption = None,
    fast_model: FastModelOption = None,
    image_model: ImageModelOption = None,
    effort: EffortOption = _Effort.high,
    research_cutoff: CutoffOption = None,
    use_plotting: PlottingOption = False,
    keep_provided_figures: KeepFiguresOption = False,
    plotting_max_critic_rounds: CriticRoundsOption = 3,
    memory: MemoryOption = True,
    lab_memory: LabMemoryOption = True,
    dream: DreamOption = _Dream.stages,
    resume: ResumeOption = True,
    cancel: Annotated[
        bool,
        typer.Option(
            "--cancel", help="Stop the papers of this batch that are submitted and not finished."
        ),
    ] = False,
) -> None:
    """Write many papers with the deployed Modal app, one container per paper.

    Deploy the app first (paper-orchestra modal deploy). Each paper runs under a run id
    derived from its paper id and the options, so the same command always means the same
    runs: when a submission was interrupted, run the command again to re-attach to the
    papers still being written (nothing is started twice) and to resume the ones that did
    not finish. Other options are other runs. --cancel stops the submitted papers.
    """
    if importlib.util.find_spec("modal") is None:
        _fail("The Modal SDK is not installed. Install it with `uv sync --extra modal`.")
    import modal

    roles = _role_models(model, writer_model, reflection_model, plotting_model, fast_model)
    options = {
        "idea_filename": idea_filename,
        "experimental_log_filename": experimental_log_filename,
        "research_cutoff": research_cutoff,
        "use_plotting": use_plotting,
        "keep_provided_figures": keep_provided_figures,
        "plotting_max_critic_rounds": plotting_max_critic_rounds,
        "writer_model": roles["writer"],
        "reflection_model": roles["reflection"],
        "plotting_model": roles["plotting"],
        "fast_model": roles["fast"],
        "image_model": image_model,
        "effort": effort.value,
        "memory": memory,
        "lab_memory": lab_memory,
        "dream": dream.value,
        "resume": resume,
    }
    try:
        # Imported here: it defines the Modal app and needs the Modal SDK.
        from paper_orchestra import modal_app as deployed

        papers = load_papers(
            root_input_folder=root_input_folder, paper_info_path=paper_info, n_papers=n_papers
        )
        log_path = run_batch_on_modal(
            options,
            template_dir=resolve_template_dir(template),
            volume=deployed.volume,
            write_paper=modal.Function.from_name(deployed.SETTINGS.app_name, "write_paper"),
            lookup_call=modal.FunctionCall.from_id,
            root_input_folder=root_input_folder,
            paper_info_path=paper_info,
            output_dir=output_dir,
            n_papers=n_papers,
            cancel=cancel,
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    except modal.exception.NotFoundError as exc:
        _fail(f"{exc} Deploy the app first with `paper-orchestra modal deploy`.")
    except modal.exception.AuthError as exc:
        _fail(f"Modal is not set up on this machine ({exc}). Run `modal setup` first.")
    except modal.exception.Error as exc:
        _fail(f"Modal reported {type(exc).__name__}: {exc}", code=1)
    _report_batch(papers, log_path)


def _unwrap_descriptions(*apps: typer.Typer) -> None:
    """Give every command its description as paragraphs of one line each.

    A description is its function's docstring, whose lines break where the source is
    100 columns wide.  The help is wrapped to the terminal; kept as they are, those
    breaks would be breaks on the screen too, and a paragraph would be wrapped twice,
    with a word or two alone on every other line.  A help text that was given as such
    is left as it was written.
    """
    import inspect

    for group in apps:
        for command in group.registered_commands:
            if command.help:
                continue  # written out for the help as it is (the options of `modal run`)
            text = inspect.getdoc(command.callback) or ""
            paragraphs = [" ".join(paragraph.split()) for paragraph in text.split("\n\n")]
            command.help = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


_unwrap_descriptions(app, research_app, memory_app, modal_app)


def _send_logs_to_stderr() -> None:
    """Keep stdout for results (``doctor --json``, ``memory show``, the paper's path).

    ``setup_logging`` builds a Rich handler that prints to stdout, where a log
    line would end up in the middle of machine-readable output.
    """
    from rich.console import Console
    from rich.logging import RichHandler

    for handler in logging.getLogger(LOGGER_NAME).handlers:
        if isinstance(handler, RichHandler):
            handler.console = Console(stderr=True)


def _interrupt_on_sigterm() -> None:
    """Have ``kill``, ``timeout`` and job schedulers stop a command the way Ctrl-C does.

    Left alone, SIGTERM ends the process on the spot: a run's ``state.json`` stays at
    "running" and a batch leaves its worker processes behind.
    """
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    # Never restored: the handler is installed for the life of the command's process.
    signal.signal(signal.SIGTERM, interrupt)


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point: load ``.env``, set up logging, then dispatch."""
    load_env()
    setup_logging()
    _send_logs_to_stderr()
    _interrupt_on_sigterm()
    try:
        app(args=None if argv is None else list(argv))
    except SystemExit as exc:
        # Only as the program itself (no argv handed in): a caller that runs a command
        # in-process keeps its process.
        if exc.code == INTERRUPTED and argv is None:
            _leave_now(INTERRUPTED)
        raise


def _leave_now(code: int) -> None:
    """End the process after an interrupt without waiting for model calls in flight.

    By now the run is on record as interrupted.  What is left are worker threads
    inside a model call, which the interpreter would wait for at a normal exit,
    for as long as the longest call takes.  Exit handlers still run first, so a
    compute backend is closed and logs are flushed.
    """
    import atexit

    atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(code)
