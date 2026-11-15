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
SIGTERM).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Sequence
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
    help="Turn an idea and an experimental log into a LaTeX paper.",
    no_args_is_help=True,
    add_completion=False,
    # A traceback with local variables could print credentials.
    pretty_exceptions_show_locals=False,
)
memory_app = typer.Typer(help="Inspect and maintain the wiki memory.", no_args_is_help=True)
modal_app = typer.Typer(help="Run the pipeline on Modal.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")
app.add_typer(modal_app, name="modal")


def _fail(message: str, code: int = 2) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
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


def _config(
    *,
    base_dir: Path,
    template: str,
    idea_filename: str,
    experimental_log_filename: str,
    model: str | None,
    writer_model: str | None,
    reflection_model: str | None,
    plotting_model: str | None,
    fast_model: str | None,
    image_model: str | None,
    effort: _Effort,
    research_cutoff: str | None,
    use_plotting: bool,
    keep_provided_figures: bool,
    plotting_max_critic_rounds: int,
    compute: _Compute,
    memory: bool,
    memory_dir: Path | None,
    lab_memory: bool,
    lab_memory_dir: Path | None,
    dream: _Dream,
    resume: bool,
) -> PipelineConfig:
    try:
        template_dir = resolve_template_dir(template)
    except FileNotFoundError as exc:
        _fail(str(exc))
    cfg = PipelineConfig(
        base_dir=base_dir,
        template_dir=template_dir,
        idea_filename=idea_filename,
        experimental_log_filename=experimental_log_filename,
        models=ModelConfig(
            **_role_models(model, writer_model, reflection_model, plotting_model, fast_model),
            image=image_model or None,
            effort=effort.value,
        ),
        memory=MemoryConfig(
            enabled=memory,
            project_dir=memory_dir,
            lab=lab_memory,
            lab_dir=lab_memory_dir,
            dream=dream.value,
        ),
        use_plotting=use_plotting,
        keep_provided_figures=keep_provided_figures,
        plotting_max_critic_rounds=plotting_max_critic_rounds,
        compute=compute.value,
        resume=resume,
    )
    if research_cutoff:
        cfg.research_cutoff = research_cutoff
    return cfg


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
    if not result.success:
        # The log above has the whole story; repeat only its headline.
        headline = (result.error or "unknown error").splitlines()[0]
        _fail(
            f"the paper was not written: {headline} (details: {cfg.base_dir / 'state.json'})",
            code=1,
        )

    typer.echo(f"Paper written: {result.pdf_path}")
    total = result.usage.get("total", {})
    cost = result.usage.get("estimated_cost_usd")
    typer.echo(
        f"Usage: {total.get('requests', 0)} requests, {total.get('input_tokens', 0)} input and "
        f"{total.get('output_tokens', 0)} output tokens"
        + (f", about ${cost:.2f}" if cost is not None else "")
        + f" (details: {cfg.base_dir / 'usage.json'})"
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


#: Exit code of a command that was interrupted (Ctrl-C or SIGTERM).
INTERRUPTED = 130


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
