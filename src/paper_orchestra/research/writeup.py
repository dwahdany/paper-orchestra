"""A paper from the wiki: the optional last step of a research project.

The paper pipeline writes from an idea file, an experimental log and figures.
:func:`export_materials` produces those three from the project's wiki, and
:func:`write_paper` runs the pipeline on them in ``<project>/paper``.

Only what the person accepted reaches a paper.  Proposed and rejected pages are
left out entirely, an experiment is described only once it has a run that
succeeded, and every number of a run is copied from its result record, which code
wrote from what the run produced.
"""

from __future__ import annotations

import contextlib
import inspect
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from paper_orchestra.compute.base import ComputeError, safe_relative
from paper_orchestra.pipeline import FIGURE_SUFFIXES, PaperResult, prepare_run_dir, run_paper
from paper_orchestra.research.model import IDEA_PATH, ProjectError, ResearchPage, parse_id
from paper_orchestra.utils.common import write_json, write_text

if TYPE_CHECKING:
    from paper_orchestra.config import PipelineConfig
    from paper_orchestra.research.project import Project

__all__ = ["export_materials", "write_paper"]

IDEA_FILE = "idea.md"
LOG_FILE = "experimental_log.md"
FIGURES_DIR = "figures"
FIGURES_INFO = "info.json"
#: Where the paper pipeline looks for its inputs inside its run directory.
MATERIALS_DIR = "raw_materials"

#: ``[[target#anchor|label]]``, the wiki's link syntax as in ``memory.wiki.WIKILINK``, with the
#: label captured without its bar.
_WIKI_LINK = re.compile(r"\[\[([^\[\]|#\n]+)(?:#[^\[\]|\n]*)?(?:\|([^\[\]\n]*))?\]\]")
_HEADING = re.compile(r"^(#{1,6})(\s+\S.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_RUN_NUMBER = re.compile(r"-run(\d+)$")
#: What a figure's file name must not contain: it ends up in a path that LaTeX reads.
_NOT_IN_A_NAME = re.compile(r"[^A-Za-z0-9._-]+")
#: Options of the paper run that are not the caller's to set: where it runs and what it reads.
_FIXED_OPTIONS = frozenset({"base_dir", "template", "idea_filename", "experimental_log_filename"})
#: Kinds the materials list under their ids, so that a reference by id can be followed.
_LISTED_BY_ID = frozenset({"hypothesis", "experiment", "finding"})
#: Front-matter fields of an experiment that describe how it is run.
_SETUP_FIELDS = ("mode", "entrypoint", "command", "repo", "gpu", "packages", "timeout_seconds")


#: Attached files handed to the paper's writers as they are: the person's own notes,
#: tables and logs.  (A PDF is a paper they attached, an image is a figure.)
PROVIDED_TEXT_SUFFIXES = frozenset(
    {".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".jsonl", ".log", ".yaml", ".yml", ".tex"}
)
PROVIDED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
#: Most text taken from one attached file, and from all of them together: the materials
#: are read whole by every stage of the paper pipeline.
MAX_PROVIDED_FILE_CHARS = 20_000
MAX_PROVIDED_CHARS = 60_000


@dataclass(frozen=True)
class _Provided:
    """A file the person attached that a paper may draw on."""

    name: str  # as listed under inputs/
    path: Path
    image: bool


@dataclass(frozen=True)
class _Materials:
    """What one export wrote, for the events that report it."""

    dest: Path
    runs: list[_Run]
    findings: list[ResearchPage]
    figures: list[str]
    provided: list[str]


@dataclass(frozen=True)
class _Run:
    """A run that succeeded: its accepted plan, its result record and its run record."""

    experiment: ResearchPage
    result: ResearchPage
    number: int
    record: Mapping[str, Any]

    @property
    def name(self) -> str:
        return f"{self.experiment.id} run {self.number}"


# ---------------------------------------------------------------------------
# What the wiki holds that a paper may use
# ---------------------------------------------------------------------------


def _run_number(result: ResearchPage) -> int | None:
    """``wiki/results/E1-run3.md`` -> 3."""
    match = _RUN_NUMBER.search(PurePosixPath(result.path).stem)
    return int(match.group(1)) if match else None


def _run_record(project: Project, experiment_id: str, number: int) -> Mapping[str, Any]:
    """The run's own record (``run.json``), through the module that writes it."""
    from paper_orchestra.research import experiments

    for record in experiments.runs(project, experiment_id):
        if record.get("number") == number:
            return record
    return {}


def _succeeded_runs(project: Project) -> list[_Run]:
    """Every run that succeeded of an experiment whose plan the person accepted."""
    plans = {page.path: page for page in project.pages(kind="experiment", review="accepted")}
    runs: list[_Run] = []
    for result in project.pages(kind="result", review="accepted", state="succeeded"):
        number = _run_number(result)
        owners = [project.resolve(target) for target in result.relations.get("run_of", [])]
        plan = next((plans[path] for path in owners if path in plans), None)
        if number is None or plan is None or not plan.id:
            continue
        runs.append(_Run(plan, result, number, _run_record(project, plan.id, number)))
    # E2 before E10, and the runs of an experiment in the order they were made.
    return sorted(runs, key=lambda run: (parse_id(run.experiment.id) or ("", 0), run.number))


def _unverified(project: Project, finding: ResearchPage) -> list[str]:
    """Numbers in a finding that no result record, source or input file supports."""
    from paper_orchestra.research import experiments

    return list(experiments.unverified_numbers(project, finding))


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _called(page: ResearchPage) -> str:
    """What the materials call an accepted page.

    Hypotheses, experiments and findings are listed in the materials under their
    ids, and a run under its experiment and number; any other page is not in the
    materials, so only its title says what is meant.
    """
    if page.kind in _LISTED_BY_ID and page.id:
        return page.id
    number = _run_number(page) if page.kind == "result" else None
    if number is not None:
        return f"{PurePosixPath(page.path).stem.rsplit('-run', 1)[0]} run {number}"
    return page.title


def _plain(project: Project, text: str) -> str:
    """``text`` with wiki links replaced by what they are called.

    The paper's writer does not know the wiki.  A link becomes its label, or else
    what the materials call the page it leads to (:func:`_called`) when that page
    is accepted; a link to anything else keeps only the name it was written with,
    so that no text of a page the person did not accept is carried over.
    """

    def name(match: re.Match[str]) -> str:
        target, label = match.group(1).strip(), (match.group(2) or "").strip()
        if label:
            return label
        path = project.resolve(target)
        page = project.page(path) if path else None
        if page is not None and page.review == "accepted":
            return _called(page)
        return PurePosixPath(target).name

    return _WIKI_LINK.sub(name, text)


def _demoted(body: str, levels: int) -> str:
    """``body`` with its headings moved ``levels`` down, so a page nests inside a file."""
    lines, fenced = [], False
    for line in body.strip().splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        match = None if fenced else _HEADING.match(line)
        if match:
            line = "#" * min(6, len(match.group(1)) + levels) + match.group(2)
        lines.append(line)
    return "\n".join(lines)


def _metrics_table(body: str) -> str:
    """The metrics table of a result record, exactly as the record has it: the first
    table under the heading that names the metrics.  Empty when there is none; a table
    found elsewhere on the page is not taken for it."""
    rows: list[str] = []
    under_metrics = False
    for line in body.splitlines():
        heading = _HEADING.match(line)
        if heading:
            if rows:
                break
            under_metrics = "metric" in heading.group(2).lower()
        elif under_metrics and line.lstrip().startswith("|"):
            rows.append(line.rstrip())
        elif rows:
            break
    return "\n".join(rows)


def _listed(project: Project, page: ResearchPage, relation: str) -> str:
    """The targets of one relation of ``page`` that are accepted pages, as they are called."""
    names = []
    for target in page.relations.get(relation, []):
        path = project.resolve(target)
        other = project.page(path) if path else None
        if other is not None and other.review == "accepted":
            names.append(_called(other))
    return ", ".join(names)


def _idea_text(project: Project, runs: list[_Run]) -> str:
    """The idea, the accepted hypotheses with where they stand, and the method that was run."""
    idea = project.page(IDEA_PATH)
    parts = [f"# {idea.title}", _plain(project, idea.body).strip()]
    hypotheses = project.pages(kind="hypothesis", review="accepted")
    if hypotheses:
        parts += [
            "## Hypotheses",
            "Each hypothesis is given with where it stands after the experiments. A refuted "
            "or inconclusive hypothesis is a result to report as such, not a claim to make.",
        ]
    for page in hypotheses:
        parts += [
            f"### {page.id}: {page.title} ({page.state})",
            _plain(project, _demoted(page.body, 2)),
        ]
    plans = list({run.experiment.path: run.experiment for run in runs}.values())
    if plans:
        parts += [
            "## Method",
            "The experiments that were run, as their accepted plans describe them.",
        ]
    for plan in plans:
        tested = _listed(project, plan, "tests")
        parts += [
            f"### {plan.id}: {plan.title}",
            *([f"Tests: {tested}."] if tested else []),
            _plain(project, _demoted(plan.body, 2)),
        ]
    return "\n\n".join(part for part in parts if part) + "\n"


def _setup(run: _Run) -> str:
    """How the run was made: the plan's settings and what the run record says of the run."""
    plan, record = run.experiment.meta, run.record
    facts = []
    for name in _SETUP_FIELDS:
        value = plan.get(name)
        if value:
            shown = ", ".join(map(str, value)) if isinstance(value, list) else value
            facts.append(f"- {name.replace('_', ' ')}: {shown}")
    for label, key in (
        ("backend", "backend"),
        ("finished", "finished"),
        ("duration in seconds", "seconds"),
        ("code digest", "code_digest"),
    ):
        if record.get(key) not in (None, ""):
            facts.append(f"- {label}: {record[key]}")
    facts.append(f"- result record: {run.result.path}")
    return "\n".join(facts)


def _log_text(
    project: Project,
    runs: list[_Run],
    findings: list[ResearchPage],
    figures: Mapping[str, list[str]],
    provided: Sequence[_Provided] = (),
) -> str:
    """Every succeeded run with its setup and metrics, what the person attached, then
    the accepted findings."""
    parts = ["# Experimental log"]
    if runs:
        parts.append(
            "Each run below succeeded. Its metrics table is copied from the result record, "
            "which was written by code from the files the run produced."
        )
    elif provided:
        parts.append(
            "No experiment of this project has a run that succeeded. The results are the "
            "material the researcher provided, below."
        )
    for run in runs:
        table = _metrics_table(run.result.body) or (
            f"_The result record has no metrics table; see {run.result.path}._"
        )
        parts += [
            f"## {run.name}: {run.experiment.title}",
            "### Setup",
            _setup(run),
            "### Metrics",
            table,
        ]
        if figures.get(run.result.path):
            parts += ["### Figures", "\n".join(f"- {name}" for name in figures[run.result.path])]
    parts += _provided_text(provided, figures.get(PROVIDED_KEY, []))
    if findings:
        parts += [
            "## Findings",
            "What the person accepted as the meaning of these results.",
        ]
    for page in findings:
        relations = [
            f"{label}: {names}."
            for label, relation in (
                ("Supports", "supports"),
                ("Refutes", "refutes"),
                ("Based on", "based_on"),
            )
            if (names := _listed(project, page, relation))
        ]
        parts += [
            f"### {page.id}: {page.title}",
            " ".join(relations),
            _plain(project, _demoted(page.body, 2)),
        ]
        unsupported = _unverified(project, page)
        if unsupported:
            parts.append(
                "Numbers in this finding that no result record, source or input file "
                f"supports, and that are not to be reported: {', '.join(unsupported)}."
            )
    return "\n\n".join(part for part in parts if part) + "\n"


# ---------------------------------------------------------------------------
# What the person attached
# ---------------------------------------------------------------------------

#: Key of the figure listing under which the attached images are found.
PROVIDED_KEY = "inputs"


def _provided(project: Project) -> list[_Provided]:
    """The attached files a paper may draw on: text and tables, and images as figures.

    They are the person's own material, so they may be reported; a paper they
    attached (a PDF) or a file of another kind is left to the wiki's pages.
    """
    found: list[_Provided] = []
    for entry in project.attachments():
        name = entry.get("name")
        file = _inside(project.inputs_dir, name) if isinstance(name, str) else None
        if file is None:
            continue
        suffix = file.suffix.lower()
        if suffix in PROVIDED_TEXT_SUFFIXES or suffix in PROVIDED_IMAGE_SUFFIXES:
            found.append(_Provided(str(name), file, suffix in PROVIDED_IMAGE_SUFFIXES))
    return found


def _fenced(text: str) -> str:
    """``text`` in a code fence that nothing in it can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text.rstrip()}\n{fence}"


def _provided_text(provided: Sequence[_Provided], figures: Sequence[str]) -> list[str]:
    """The section of the log that holds the attached text files, each as it is."""
    texts = [item for item in provided if not item.image]
    if not texts and not figures:
        return []
    parts = [
        "## Material the researcher provided",
        "These files were attached to the project by the researcher. They are the "
        "researcher's own material and were not produced by this project's runs.",
    ]
    room = MAX_PROVIDED_CHARS
    for item in texts:
        try:
            text = item.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            parts += [f"### inputs/{item.name}", f"_This file could not be read ({exc})._"]
            continue
        take = min(len(text), MAX_PROVIDED_FILE_CHARS, room)
        parts.append(f"### inputs/{item.name}")
        if take <= 0:
            parts.append(
                f"_Left out for lack of room: the file has {len(text)} characters and the "
                "files before it fill what the materials can hold._"
            )
            continue
        room -= take
        parts.append(_fenced(text[:take]))
        if take < len(text):
            parts.append(
                f"_Cut after {take} of {len(text)} characters; the rest is in the "
                "project's inputs._"
            )
    if figures:
        parts += ["### Figures provided", "\n".join(f"- {name}" for name in figures)]
    return parts


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _inside(base: Path, name: str) -> Path | None:
    """The regular file ``name`` leads to inside ``base``, reached through no symbolic
    link, or ``None``.  ``name`` is data: an entry of a run record or of a manifest."""
    try:
        relative = safe_relative(name)
    except ComputeError:
        return None
    file = base
    for part in relative.parts:
        file = file / part
        if file.is_symlink():
            return None
    return file if file.is_file() else None


def _artifact(run_dir: Path, name: str) -> Path | None:
    """The image file an artifact entry of a run record names, or ``None``: a regular
    file inside the run directory of a type the paper pipeline takes as a figure."""
    file = _inside(run_dir, name)
    if file is None or file.suffix.lower() not in FIGURE_SUFFIXES:
        return None
    return file


def _figure_name(run: _Run, artifact: PurePosixPath, taken: set[str]) -> str:
    """The file name a run's artifact is exported under.

    It says which run the figure is from, keeps apart artifacts of one run that
    share a name in different folders (``artifacts/a/plot.png`` and
    ``artifacts/b/plot.png``), and holds only characters that are safe in a path
    LaTeX reads: the name was chosen by the experiment's code.
    """
    parts = artifact.parts[1:] if len(artifact.parts) > 1 else artifact.parts
    stem = "-".join(parts)[: -len(artifact.suffix)]
    base = _NOT_IN_A_NAME.sub("-", f"{run.experiment.id}-run{run.number}-{stem}")
    name, number = base + artifact.suffix, 1
    while name.lower() in taken:
        number += 1
        name = f"{base}-{number}{artifact.suffix}"
    taken.add(name.lower())
    return name


def _export_figures(
    project: Project, runs: list[_Run], dest: Path, provided: Sequence[_Provided] = ()
) -> dict[str, list[str]]:
    """Copy the image artifacts of ``runs`` and the attached images to ``dest`` with an
    ``info.json``.

    Returns the figure names per result record path, and under ``PROVIDED_KEY``
    those of the attached images.  A caption says where the figure comes from:
    which experiment and run and what the run called it, or that the researcher
    provided it.
    """
    # Figures of an earlier export would pass for figures of this one.
    if dest.is_dir() and not dest.is_symlink():
        shutil.rmtree(dest)
    else:
        dest.unlink(missing_ok=True)
    entries: list[dict[str, str]] = []
    by_result: dict[str, list[str]] = {}
    taken: set[str] = set()
    for run in runs:
        run_dir = project.experiment_dir(str(run.experiment.id)) / "runs" / str(run.number)
        listed = [entry for entry in run.record.get("artifacts") or [] if isinstance(entry, str)]
        # Once each: a record that names a file twice still has one figure.
        for entry in dict.fromkeys(listed):
            source = _artifact(run_dir, entry)
            if source is None:
                continue
            artifact = PurePosixPath(source.relative_to(run_dir).as_posix())
            name = _figure_name(run, artifact, taken)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest / name)
            what = source.stem.replace("_", " ").replace("-", " ")
            entries.append(
                {"name": name, "caption": f"{run.experiment.title} ({run.name}): {what}."}
            )
            by_result.setdefault(run.result.path, []).append(name)
    for item in provided:
        if not item.image:
            continue
        base = _NOT_IN_A_NAME.sub("-", f"provided-{PurePosixPath(item.name).with_suffix('')}")
        suffix = item.path.suffix.lower()
        name, number = base + suffix, 1
        while name.lower() in taken:
            number += 1
            name = f"{base}-{number}{suffix}"
        taken.add(name.lower())
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.path, dest / name)
        what = item.path.stem.replace("_", " ").replace("-", " ")
        entries.append({"name": name, "caption": f"Provided by the researcher: {what}."})
        by_result.setdefault(PROVIDED_KEY, []).append(name)
    if entries:
        write_json(dest / FIGURES_INFO, entries)
    return by_result


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def export_materials(project: Project, dest: str | Path) -> Path:
    """Write what the paper pipeline reads, from the accepted pages of the wiki.

    In ``dest`` (created if needed; returned):

    * ``idea.md``: the idea page, the accepted hypotheses with their state, and the
      method as the accepted experiments that ran describe it;
    * ``experimental_log.md``: for each succeeded run its setup and the metrics
      table copied verbatim from the result record, then the text files the person
      attached to the project as they are (bounded in size), then the accepted
      findings;
    * ``figures/``: the image artifacts of those runs and the images the person
      attached, with ``info.json`` (``[{"name", "caption"}]``).  An earlier
      ``figures/`` is replaced.

    Proposed and rejected pages are never used, and neither is an experiment
    without a succeeded run.  Nothing in the project is changed.
    """
    return _export(project, Path(dest), *_reportable(project)).dest


def _reportable(project: Project) -> tuple[list[_Run], list[ResearchPage]]:
    """What a paper may report: the runs that succeeded and the accepted findings."""
    return _succeeded_runs(project), project.pages(kind="finding", review="accepted")


def _export(
    project: Project, dest: Path, runs: list[_Run], findings: list[ResearchPage]
) -> _Materials:
    provided = _provided(project)
    figures = _export_figures(project, runs, dest / FIGURES_DIR, provided)
    write_text(dest / IDEA_FILE, _idea_text(project, runs))
    write_text(dest / LOG_FILE, _log_text(project, runs, findings, figures, provided))
    names = [name for listed in figures.values() for name in listed]
    return _Materials(dest, runs, findings, names, [item.name for item in provided])


def _check_ready(project: Project, runs: list[_Run], findings: list[ResearchPage]) -> None:
    """Raise unless the project holds what a paper reports: results (a succeeded run, or
    results the person attached) and an accepted finding that says what they mean."""
    if not runs and not _provided(project):
        raise ProjectError(
            "There is nothing to write a paper from yet: no accepted experiment has a run "
            "that succeeded, and no results were attached to the project. Approve an "
            "experiment and run it (`paper-orchestra research approve`, then `research "
            "run`), then start the write-up again."
        )
    if not findings:
        raise ProjectError(
            "There is nothing to write a paper from yet: no finding is accepted. The "
            "results are in, but a paper reports what they mean: accept a finding (in the "
            "review inbox, or with `paper-orchestra research review`), or write one "
            "yourself, then start the write-up again."
        )


def _paper_config(project: Project, template: str, options: Mapping[str, Any]) -> PipelineConfig:
    """The pipeline's configuration for this project: its settings, then ``options``."""
    # The command line owns the reading of these options (it is what `write` takes);
    # the write-up must not grow a second one.
    from paper_orchestra.cli import build_config

    allowed = set(inspect.signature(build_config).parameters) - _FIXED_OPTIONS
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError(
            f"write_paper got {', '.join(unknown)}; the options are: {', '.join(sorted(allowed))}."
        )
    settings, models = project.settings, project.settings.models
    everywhere = options.get("model")
    chosen: dict[str, Any] = {
        "writer_model": everywhere or models.reasoning,
        "reflection_model": everywhere or models.reasoning,
        "plotting_model": everywhere or models.coding,
        "fast_model": everywhere or models.fast,
        "effort": models.effort,
        "research_cutoff": settings.research_cutoff,
        "compute": settings.compute,
        "lab_memory": settings.lab_memory,
        # The figures here are what the runs produced; generating more must not drop them.
        "keep_provided_figures": True,
    }
    chosen.update({name: value for name, value in options.items() if value is not None})
    chosen.pop("model", None)
    return build_config(
        base_dir=project.paper_dir,
        template=template,
        idea_filename=IDEA_FILE,
        experimental_log_filename=LOG_FILE,
        **chosen,
    )


@contextlib.contextmanager
def _as_the_worker(project: Project) -> Iterator[None]:
    """Be the project's one worker for the length of the write-up.

    Two write-ups at once would run two pipelines in the same ``paper`` directory,
    each replacing the other's materials, state and usage.  The write-up is on
    file as the project's worker like an engine is, so the web app shows one that
    was started from a terminal and starts nothing beside it.
    """
    # Imported here: the engine brings the agents and the model layer with it.
    from paper_orchestra.research.engine import as_worker

    with as_worker(project, kind="write"):
        yield


def _relative(project: Project, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_paper(project: Project, *, template: str, **options: Any) -> PaperResult:
    """Write a paper from the project's wiki with the paper pipeline, in ``<project>/paper``.

    ``template`` is a bundled template name or a template directory.  ``options``
    are the options of the ``write`` command under the names they have there
    (``model``, ``writer_model``, ``reflection_model``, ``plotting_model``,
    ``fast_model``, ``image_model``, ``effort``, ``research_cutoff``,
    ``use_plotting``, ``keep_provided_figures``, ``plotting_max_critic_rounds``,
    ``compute``, ``memory``, ``memory_dir``, ``lab_memory``, ``lab_memory_dir``,
    ``dream``, ``resume``); what is left out, or ``None``, follows the project's
    settings.  A run that was interrupted is continued by calling this again.

    Refuses with ``ProjectError`` when no accepted experiment has a succeeded run
    or no finding is accepted, and when the project has a worker at work (the
    write-up holds the project's worker lock while it runs: one worker at a time);
    raises ``FileNotFoundError`` for an unknown template and ``TypeError`` for an
    unknown option.  Emits ``paper.started`` and ``paper.finished`` on the project.
    As with ``pipeline.run_paper``, a paper that could not be written is a result
    with ``success=False``, and an interrupt is passed on after it was recorded.
    """
    runs, findings = _reportable(project)
    _check_ready(project, runs, findings)
    cfg = _paper_config(project, template, options)
    with _as_the_worker(project):
        return _write(project, template, cfg, runs, findings)


def _write(
    project: Project,
    template: str,
    cfg: PipelineConfig,
    runs: list[_Run],
    findings: list[ResearchPage],
) -> PaperResult:
    """Export the materials and run the pipeline on them, telling the project about it."""
    made = _export(project, cfg.base_dir / MATERIALS_DIR, runs, findings)
    prepare_run_dir(cfg, None)
    project.emit(
        "paper.started",
        f"Writing a paper from the wiki with the {Path(template).name} template: "
        f"{len(made.runs)} succeeded run{'s' if len(made.runs) != 1 else ''}, "
        f"{len(made.findings)} accepted finding{'s' if len(made.findings) != 1 else ''}"
        + (
            f", {len(made.provided)} attached file{'s' if len(made.provided) != 1 else ''}."
            if made.provided
            else "."
        ),
        refs=[_relative(project, made.dest / IDEA_FILE), _relative(project, made.dest / LOG_FILE)],
        data={
            "template": template,
            "dir": _relative(project, cfg.base_dir),
            "results": [run.result.path for run in made.runs],
            "findings": [page.path for page in made.findings],
            "figures": made.figures,
            "provided": made.provided,
        },
    )
    try:
        result = run_paper(cfg)
    except BaseException as exc:
        from paper_orchestra.memory.harness import RunCancelled

        stopped = isinstance(exc, (KeyboardInterrupt, RunCancelled))
        project.emit(
            "paper.finished",
            "The write-up was stopped before the paper was finished; starting it again "
            "continues it."
            if stopped
            else f"The write-up could not run: {exc}",
            data={"success": False, "interrupted": stopped, "error": str(exc) or None},
        )
        raise
    pdf = _relative(project, result.pdf_path)
    headline = (result.error or "unknown error").splitlines()[0]
    project.emit(
        "paper.finished",
        f"The paper is written: {pdf}." if result.success else f"No paper was written: {headline}",
        refs=[pdf] if result.success else [_relative(project, cfg.base_dir / "state.json")],
        data={
            "success": result.success,
            "pdf": pdf or None,
            "error": result.error,
            "cost_usd": result.usage.get("estimated_cost_usd"),
        },
    )
    return result
