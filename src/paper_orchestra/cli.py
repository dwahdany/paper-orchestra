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
usage or validation error.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer
from typer.core import TyperCommand

from paper_orchestra import __version__
from paper_orchestra.batch import run_batch, summarize
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
        help="LaTeX template: a name under templates/ (for example cvpr2025) or a directory.",
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
    str, typer.Option("--model", help="Model for every role that has no flag of its own.")
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
    typer.Option("--fast-model", help="Model for high-volume calls such as paper discovery."),
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
        "--use-plotting/--no-use-plotting", help="Generate the figures from the experimental log."
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


def _config(
    *,
    base_dir: Path,
    template: str,
    idea_filename: str,
    experimental_log_filename: str,
    model: str,
    writer_model: str | None,
    reflection_model: str | None,
    plotting_model: str | None,
    fast_model: str | None,
    image_model: str | None,
    effort: _Effort,
    research_cutoff: str | None,
    use_plotting: bool,
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
            writer=writer_model or model,
            reflection=reflection_model or model,
            plotting=plotting_model or model,
            fast=fast_model or model,
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
    model: ModelOption = DEFAULT_MODEL,
    writer_model: WriterModelOption = None,
    reflection_model: ReflectionModelOption = None,
    plotting_model: PlottingModelOption = None,
    fast_model: FastModelOption = None,
    image_model: ImageModelOption = None,
    effort: EffortOption = _Effort.high,
    research_cutoff: CutoffOption = None,
    use_plotting: PlottingOption = False,
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
    root_input_folder: Annotated[
        Path | None,
        typer.Option(
            "--root-input-folder",
            help="Folder with one <paper_id>/raw_materials directory per paper.",
        ),
    ] = None,
    paper_info: Annotated[
        Path | None,
        typer.Option(
            "--paper-info",
            help="JSON list of {paper_id, content_folder, pdf_path} instead of a folder.",
        ),
    ] = None,
    n_papers: Annotated[
        int, typer.Option("--n-papers", help="Write only the first N papers (-1: all).")
    ] = -1,
    max_workers: Annotated[
        int, typer.Option("--max-workers", min=1, help="Papers written in parallel.")
    ] = 4,
    idea_filename: IdeaFilenameOption = "idea_sparse.md",
    experimental_log_filename: LogFilenameOption = "experimental_log.md",
    model: ModelOption = DEFAULT_MODEL,
    writer_model: WriterModelOption = None,
    reflection_model: ReflectionModelOption = None,
    plotting_model: PlottingModelOption = None,
    fast_model: FastModelOption = None,
    image_model: ImageModelOption = None,
    effort: EffortOption = _Effort.high,
    research_cutoff: CutoffOption = None,
    use_plotting: PlottingOption = False,
    plotting_max_critic_rounds: CriticRoundsOption = 3,
    compute: ComputeOption = _Compute.local,
    memory: MemoryOption = True,
    lab_memory: LabMemoryOption = True,
    lab_memory_dir: LabMemoryDirOption = None,
    dream: DreamOption = _Dream.stages,
    resume: ResumeOption = True,
) -> None:
    """Write many papers in parallel worker processes; finished papers are skipped."""
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
    counts = summarize(read_json(log_path))
    typer.echo(
        f"{counts['success']} written, {counts['failed']} failed, {counts['error']} in error "
        f"(details: {log_path})"
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

    return Wiki(_memory_root(memory_dir, run_dir), read_only=CODE_OWNED_SECTIONS)


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
        str, typer.Option("--model", help="Model that does the consolidation.")
    ] = DEFAULT_MODEL,
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


@modal_app.command("run", context_settings=_PASS_THROUGH)
def modal_run(
    ctx: typer.Context, detach: DetachOption = False, print_command: PrintCommandOption = False
) -> None:
    """Write one paper on Modal. Remaining arguments go to the Modal entrypoint, e.g.
    --raw-materials-dir DIR --template cvpr2025 --use-plotting.
    """
    _modal("run", f"{MODAL_MODULE}::main", ctx.args, detach=detach, show=print_command)


@modal_app.command("deploy", context_settings=_PASS_THROUGH)
def modal_deploy(ctx: typer.Context, print_command: PrintCommandOption = False) -> None:
    """Deploy the app so that write_paper and write_batch can be called by name."""
    _modal("deploy", MODAL_MODULE, ctx.args, detach=False, show=print_command)


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point: load ``.env``, set up logging, then dispatch."""
    load_env()
    setup_logging()
    app(args=None if argv is None else list(argv))
