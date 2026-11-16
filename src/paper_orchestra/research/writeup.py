"""A paper from the wiki: the optional last step of a research project.

The paper pipeline writes from an idea file, an experimental log and figures.
:func:`export_materials` produces those three from the project's wiki, and
:func:`write_paper` runs the pipeline on them in ``<project>/paper``.

Only what the person accepted reaches a paper.  Proposed and rejected pages are
left out entirely, an experiment is described only once it has a run that
succeeded, and every number of a run is copied from the run's own record
(``run.json``), which code wrote from what the run produced.  A finding that
rests only on an experiment the person rejected is left out and named; a number
of a finding that is on no record is taken out of the text the writers are
given.  The literature starts from the wiki as well: the accepted sources are
the papers the paper's literature stage begins with, the rejected ones are papers
not to cite, and the accepted topic pages are background.
"""

from __future__ import annotations

import contextlib
import inspect
import re
import shlex
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from paper_orchestra.compute.base import ComputeError, safe_relative
from paper_orchestra.pipeline import FIGURE_SUFFIXES, PaperResult, prepare_run_dir, run_paper
from paper_orchestra.research import repo as repositories
from paper_orchestra.research.model import IDEA_PATH, ProjectError, ResearchPage, parse_id
from paper_orchestra.research.sources import normalize_cutoff
from paper_orchestra.utils.common import write_json, write_text

if TYPE_CHECKING:
    from paper_orchestra.config import PipelineConfig
    from paper_orchestra.research.project import Project

__all__ = ["Missing", "export_materials", "left_out", "missing", "write_paper"]

IDEA_FILE = "idea.md"
LOG_FILE = "experimental_log.md"
FIGURES_DIR = "figures"
FIGURES_INFO = "info.json"
#: The papers the literature stage starts from and the ones it must not cite, as the
#: wiki has them: ``{"version": 1, "papers": [...], "excluded": [...]}``.  A paper has
#: the fields of ``agents.literature.PaperData``; an excluded one its citation key, its
#: title and the person's reason.
KNOWN_PAPERS_FILE = "known_papers.json"
#: Most text of the accepted topic pages that is appended to the idea as background,
#: and of one of them: the idea file is read whole by every stage of the paper pipeline.
MAX_BACKGROUND_CHARS = 30_000
MAX_TOPIC_CHARS = 8_000
#: Where the paper pipeline looks for its inputs inside its run directory.
MATERIALS_DIR = "raw_materials"

#: ``[[target#anchor|label]]``, the wiki's link syntax as in ``memory.wiki.WIKILINK``, with the
#: label captured without its bar.
_WIKI_LINK = re.compile(r"\[\[([^\[\]|#\n]+)(?:#[^\[\]|\n]*)?(?:\|([^\[\]\n]*))?\]\]")
_HEADING = re.compile(r"^(#{1,6})(\s+\S.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_RUN_NUMBER = re.compile(r"-run(\d+)$")
#: A day as Semantic Scholar gives a publication date and a source record keeps it.
_DATE = re.compile(r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])")
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
    #: Finding id -> the numbers that were taken out of its text: they are on no record.
    numbers_removed: dict[str, list[str]]
    #: Citation keys of the accepted sources the literature starts from, and of the
    #: rejected ones it must not cite.
    sources: list[str]
    excluded: list[str]


class Missing(str):
    """What a project still lacks for a paper (``results`` or ``finding``), as
    :func:`missing` returns it: the word, with ``detail`` saying which accepted findings
    were not counted and why (empty when none was left out)."""

    detail: str = ""

    def __new__(cls, what: str, detail: str = "") -> Missing:
        made = super().__new__(cls, what)
        made.detail = detail
        return made


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


def _stands(project: Project, target: str) -> bool:
    """Whether a record a finding names is one a paper may rest on: an accepted result
    record of an experiment that is there and that the person did not reject."""
    path = project.resolve(target)
    record = project.page(path) if path else None
    if record is None or record.kind != "result" or record.review != "accepted":
        return False
    for owner in record.relations.get("run_of", []):
        plan_path = project.resolve(owner)
        plan = project.page(plan_path) if plan_path else None
        if plan is not None and plan.kind == "experiment" and plan.review != "rejected":
            return True
    return False


def _named(project: Project, target: str) -> str:
    """A record a finding names, as a message calls it."""
    path = project.resolve(target)
    return _called(project.page(path)) if path else str(target)


def _sorted_out(project: Project) -> tuple[list[ResearchPage], list[dict[str, Any]]]:
    """The accepted findings a paper may report, and the ones it may not, with the reason.

    A finding interprets records.  One whose records all belong to experiments that
    were rejected or are gone rests on nothing the person stands behind, and is
    left out; one that names no record at all (the person's own reading of what
    they attached) is theirs to make.
    """
    kept: list[ResearchPage] = []
    left: list[dict[str, Any]] = []
    for page in project.pages(kind="finding", review="accepted"):
        based = page.relations.get("based_on", [])
        if not based or any(_stands(project, target) for target in based):
            kept.append(page)
            continue
        names = ", ".join(_named(project, target) for target in based)
        left.append(
            {
                "id": page.id,
                "path": page.path,
                "title": page.title,
                "reason": f"every record it is based on ({names}) belongs to an experiment "
                "that was rejected or is gone",
            }
        )
    return kept, left


def left_out(project: Project) -> list[dict[str, Any]]:
    """The accepted findings a paper would not report, each with why::

        [{"id": "F1", "path": "wiki/findings/F1-...md", "title": "...",
          "reason": "every record it is based on (E1 run 1) belongs to an experiment
                     that was rejected or is gone"}]

    For saying so before a write-up starts and when an experiment is rejected.
    """
    return _sorted_out(project)[1]


def _left_out_text(left: Sequence[Mapping[str, Any]]) -> str:
    """``F1: every record ...; F3: ...``."""
    return "; ".join(f"{item['id'] or item['path']}: {item['reason']}" for item in left)


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


def _basis(project: Project, finding: ResearchPage) -> tuple[str, str]:
    """What a reported finding is based on, as the materials say it: the records it
    names that stand (see :func:`_stands`), and a sentence about the ones that do not.

    A finding is reported when one of its records stands.  The others (runs of an
    experiment the person rejected since) are in no part of the materials; naming
    them as its basis would send the writer looking for a run that is not there.
    """
    stands, gone = [], []
    for target in finding.relations.get("based_on", []):
        (stands if _stands(project, target) else gone).append(_named(project, target))
    if not gone:
        return ", ".join(stands), ""
    several = len(gone) > 1
    what = "records" if several else "a record"
    are = "those records are" if several else "that record is"
    return ", ".join(stands), (
        f"It also names {', '.join(gone)}, {what} of an experiment that was rejected or is "
        f"gone: {are} not reported and {'support' if several else 'supports'} nothing here."
    )


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
    sorted_out = _sorted_out(project) if hypotheses else ([], [])
    for page in hypotheses:
        parts += [
            f"### {page.id}: {page.title} ({page.state})",
            _unsupported_verdict(project, page, *sorted_out),
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
    parts += _background(project)
    parts += _sources_text(project)
    return "\n\n".join(part for part in parts if part) + "\n"


def _unsupported_verdict(
    project: Project,
    hypothesis: ResearchPage,
    kept: Sequence[ResearchPage],
    left: Sequence[Mapping[str, Any]],
) -> str:
    """What to say of a hypothesis whose verdict came only through findings that a paper
    does not report (``left``: they rest on experiments that were rejected or are gone):
    the verdict is the person's and stays on the page, but nothing reportable carries it."""
    if hypothesis.state not in ("supported", "refuted", "inconclusive") or not left:
        return ""

    def judges(finding: ResearchPage) -> bool:
        targets = [*finding.relations.get("supports", []), *finding.relations.get("refutes", [])]
        return hypothesis.path in {project.resolve(target) for target in targets}

    carried = [page for page in kept if judges(page)]
    dropped = [str(item["id"]) for item in left if judges(project.page(item["path"]))]
    if carried or not dropped:
        return ""
    which = ", ".join(dropped)
    return (
        f"_This verdict came through {which}, which {'is' if len(dropped) == 1 else 'are'} "
        "not reported: every record behind it belongs to an experiment that was rejected or "
        "is gone. Treat the hypothesis as open._"
    )


def _background(project: Project) -> list[str]:
    """The accepted topic pages, as background for whoever writes the paper: what the
    research phase found and the person accepted.  Bounded, with a line for what did
    not fit."""
    topics = project.pages(kind="topic", review="accepted")
    if not topics:
        return []
    parts = [
        "## Background from the research wiki",
        "Syntheses of the literature that the researcher accepted. They are background "
        "for the paper, not results of this project.",
    ]
    room = MAX_BACKGROUND_CHARS
    for page in topics:
        text = _plain(project, _demoted(page.body, 2)).strip()
        take = min(len(text), MAX_TOPIC_CHARS, room)
        if take <= 0:
            parts.append(
                f"_{page.title}: left out for lack of room; it is in the project's wiki "
                f"({page.path})._"
            )
            continue
        room -= take
        parts += [f"### {page.title}", text[:take]]
        if take < len(text):
            parts.append(
                f"_Cut after {take} of {len(text)} characters; the rest is in {page.path}._"
            )
    return parts


def _abstract(page: ResearchPage) -> str:
    """The abstract a source record quotes (its lines that start with ``>``)."""
    quoted = [line[1:].strip() for line in page.body.splitlines() if line.startswith(">")]
    return " ".join(" ".join(quoted).split())


def _publication_date(published: Any) -> str | None:
    """The day a source record says the paper was published, when it says one.

    The literature stage holds the papers it starts from to the research cutoff.  The
    wiki admitted a source by this date; by its year alone, a paper of the cutoff's own
    year would be turned away there although the person accepted it here.
    """
    text = published.strip() if isinstance(published, str) else ""
    return text if _DATE.fullmatch(text) else None


def _known_papers(project: Project) -> dict[str, Any]:
    """The wiki's sources as the paper's literature stage can take them: the accepted
    ones as papers it starts from (in the shape it keeps verified papers in), the
    rejected ones as papers it must not cite."""
    papers = []
    for page in project.pages(kind="source", review="accepted"):
        meta = page.meta
        authors, year, cited = meta.get("authors"), meta.get("year"), meta.get("citations")
        papers.append(
            {
                "citation_key": page.id or PurePosixPath(page.path).stem,
                "title": page.title,
                "authors": [str(name) for name in authors] if isinstance(authors, list) else [],
                "venue": str(meta.get("venue") or ""),
                "year": year if isinstance(year, int) and not isinstance(year, bool) else None,
                "abstract": _abstract(page),
                "citation_count": cited if isinstance(cited, int) and cited >= 0 else 0,
                "found_in_section": "",
                "sections": [],
                "reason": "A source the researcher accepted in the project's research wiki.",
                "journal": None,
                "volume": None,
                "pages": None,
                "publication_date": _publication_date(meta.get("published")),
                "link": str(meta.get("link") or ""),
                "record": page.path,
            }
        )
    excluded = [
        {
            "citation_key": page.id or PurePosixPath(page.path).stem,
            "title": page.title,
            "reason": page.decision_reason or "rejected by the researcher",
        }
        for page in project.pages(kind="source", review="rejected")
    ]
    return {"version": 1, "papers": papers, "excluded": excluded}


def _sources_text(project: Project, known: Mapping[str, Any] | None = None) -> list[str]:
    """What the idea file says about the literature: the sources to start from and the
    papers not to cite, each by title, so that every stage that reads the idea knows."""
    known = _known_papers(project) if known is None else known
    parts: list[str] = []
    if known["papers"]:
        parts += [
            "## Sources the researcher accepted",
            "These papers were verified and accepted in the project's research wiki. The "
            "paper's related work starts from them; more may be found and cited.",
            "\n".join(
                f"- {paper['citation_key']}: {paper['title']}"
                + (f" ({', '.join(paper['authors'][:3])}" if paper["authors"] else " (")
                + ("" if len(paper["authors"]) <= 3 else " and others")
                + (f", {paper['year']})" if paper["year"] else ")")
                for paper in known["papers"]
            ).replace(" ()", ""),
        ]
    if known["excluded"]:
        parts += [
            "## Papers not to cite",
            "The researcher rejected these as sources. They are not to be cited or relied on.",
            "\n".join(f"- {paper['title']}: {paper['reason']}" for paper in known["excluded"]),
        ]
    return parts


def _setup_value(name: str, value: Any) -> str:
    """One setting of a run as the materials state it."""
    if name == "command" and isinstance(value, list):
        return shlex.join(map(str, value))
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if name == "repo":
        # Named, never with the password or token of its URL: the materials are read by
        # models and travel with the paper.
        return repositories.display_url(str(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _setup(run: _Run) -> str:
    """How the run was made: its settings and what the run record says of the run.

    What the run's own record holds of its plan (the command, the time limit, the
    packages, whether it could reach the network) is taken from there: the page of
    an experiment can be edited after the run, the record of what ran cannot.
    """
    record = run.record
    ran = record.get("plan") if isinstance(record.get("plan"), Mapping) else {}
    plan = {**run.experiment.meta, **ran}
    facts = []
    for name in _SETUP_FIELDS:
        value = plan.get(name)
        if value:
            facts.append(f"- {name.replace('_', ' ')}: {_setup_value(name, value)}")
    if "network" in ran:
        facts.append(f"- network access: {'allowed' if ran['network'] else 'not allowed'}")
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


def _metrics_of(project: Project, run: _Run) -> str:
    """The metrics of a run for the materials: from the run's own record, which code
    wrote and nobody edits through the wiki, never from a result page that may have been
    changed since.  Where the record is gone (run directories can be cleared), the page
    stands in, as long as it is still what code wrote."""
    from paper_orchestra.research import experiments

    edited = experiments.edited_record(project, run.result)
    table = experiments.metrics_table(run.record) if run.record else ""
    if table:
        note = (
            "\n\n_The result page of this run was edited after code wrote it; the values "
            "here are the run's own record._"
            if edited
            else ""
        )
        return table + note
    if run.record:
        return f"_The run recorded no metrics; see {run.result.path}._"
    if edited:
        return (
            "_The record of this run is not on file any more and its result page was "
            f"edited after code wrote it, so its metrics are not copied; see {run.result.path}._"
        )
    return _metrics_table(run.result.body) or (
        f"_The result record has no metrics table; see {run.result.path}._"
    )


def _log_text(
    project: Project,
    runs: list[_Run],
    findings: list[ResearchPage],
    figures: Mapping[str, list[str]],
    provided: Sequence[_Provided] = (),
    removed: dict[str, list[str]] | None = None,
) -> str:
    """Every succeeded run with its setup and metrics, what the person attached, then
    the accepted findings.  ``removed`` is filled with the numbers taken out of each
    finding because they are on no record."""
    from paper_orchestra.research import experiments

    parts = ["# Experimental log"]
    if runs:
        parts.append(
            "Each run below succeeded. Its metrics are the run's own record, written by "
            "code from the files the run produced."
        )
    elif provided:
        parts.append(
            "No run of an experiment the researcher accepted is reported here. The results "
            "are the material the researcher provided, below."
        )
    for run in runs:
        parts += [
            f"## {run.name}: {run.experiment.title}",
            "### Setup",
            _setup(run),
            "### Metrics",
            _metrics_of(project, run),
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
        based_on, not_counted = _basis(project, page)
        relations = [
            f"{label}: {names}."
            for label, names in (
                ("Supports", _listed(project, page, "supports")),
                ("Refutes", _listed(project, page, "refutes")),
                ("Based on", based_on),
            )
            if names
        ]
        relations += [not_counted] if not_counted else []
        # A number that is on no record does not reach the writers: it is taken out of
        # the text here, whatever a model would make of a sentence asking it not to.
        title, body, gone = experiments.reportable_text(project, page)
        parts += [
            f"### {page.id}: {title}",
            " ".join(relations),
            _plain(project, _demoted(body, 2)),
        ]
        if gone:
            if removed is not None:
                removed[str(page.id or page.path)] = gone
            count = f"{len(gone)} number{'s' if len(gone) != 1 else ''} of this finding"
            parts.append(
                f"{count} {'was' if len(gone) == 1 else 'were'} found in no run record, "
                "source or file of the researcher and "
                f"{'is' if len(gone) == 1 else 'are'} replaced above by "
                f"{experiments.NOT_ON_RECORD}. Do not supply a number in its place."
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
    * ``idea.md``, further: the accepted topic pages as background, the accepted
      sources by title as the papers the related work starts from, and the
      rejected sources as papers not to cite;
    * ``experimental_log.md``: for each succeeded run its setup and its metrics
      from the run's own record (``run.json``, not the result page, which can be
      edited), then the text files the person attached to the project as they are
      (bounded in size), then the accepted findings.  A finding whose records all
      belong to experiments that were rejected or are gone is left out
      (:func:`left_out` names it); in the text of the others, a number that the
      number check finds nowhere is replaced by "[a number not on record]";
    * ``known_papers.json``: the accepted sources in the shape the paper
      pipeline's literature stage keeps its verified papers in, and the rejected
      ones as an exclusion list;
    * ``figures/``: the image artifacts of those runs and the images the person
      attached, with ``info.json`` (``[{"name", "caption"}]``).  An earlier
      ``figures/`` is replaced.

    Proposed and rejected pages are never used, and neither is an experiment
    without a succeeded run.  Nothing in the project is changed.
    """
    runs, findings, _ = _reportable(project)
    return _export(project, Path(dest), runs, findings).dest


def _reportable(
    project: Project,
) -> tuple[list[_Run], list[ResearchPage], list[dict[str, Any]]]:
    """What a paper may report: the runs that succeeded and the accepted findings that
    rest on something; and the accepted findings that do not, with the reason."""
    findings, left = _sorted_out(project)
    return _succeeded_runs(project), findings, left


def _export(
    project: Project, dest: Path, runs: list[_Run], findings: list[ResearchPage]
) -> _Materials:
    provided = _provided(project)
    figures = _export_figures(project, runs, dest / FIGURES_DIR, provided)
    known = _known_papers(project)
    removed: dict[str, list[str]] = {}
    write_text(dest / IDEA_FILE, _idea_text(project, runs))
    write_text(dest / LOG_FILE, _log_text(project, runs, findings, figures, provided, removed))
    write_json(dest / KNOWN_PAPERS_FILE, known)
    names = [name for listed in figures.values() for name in listed]
    return _Materials(
        dest,
        runs,
        findings,
        names,
        [item.name for item in provided],
        removed,
        [paper["citation_key"] for paper in known["papers"]],
        [paper["citation_key"] for paper in known["excluded"]],
    )


def missing(project: Project) -> Missing | None:
    """What the project still lacks for a paper: ``"results"`` (no accepted experiment
    has a run that succeeded, and nothing the person attached can be reported),
    ``"finding"`` (no accepted finding can be reported), or ``None`` when a paper
    can be written.

    The rule :func:`write_paper` applies, for a caller that wants to say so before
    the person starts a write-up.  What is returned is the word (a ``str``); its
    ``detail`` names the accepted findings that do not count because every record
    they are based on belongs to a rejected or missing experiment, and why.
    """
    runs, findings, left = _reportable(project)
    detail = _left_out_text(left)
    if not runs and not _provided(project):
        return Missing("results", detail)
    return None if findings else Missing("finding", detail)


def _check_ready(
    project: Project,
    runs: list[_Run],
    findings: list[ResearchPage],
    left: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Raise unless the project holds what a paper reports: results (a succeeded run, or
    results the person attached) and an accepted finding that says what they mean."""
    not_counted = (
        f" Not counted: {_left_out_text(left)}. Accept the experiment again, or base the "
        "finding on a record that stands."
        if left
        else ""
    )
    if not runs and not _provided(project):
        raise ProjectError(
            "There is nothing to write a paper from yet: no accepted experiment has a run "
            "that succeeded, and no results were attached to the project. Approve an "
            "experiment and run it (`paper-orchestra research approve`, then `research "
            f"run`), then start the write-up again.{not_counted}"
        )
    if not findings:
        none = "no accepted finding can be reported" if left else "no finding is accepted"
        raise ProjectError(
            f"There is nothing to write a paper from yet: {none}. The results are in, but "
            "a paper reports what they mean: accept a finding (in the review inbox, or "
            "with `paper-orchestra research review`), or write one yourself, then start "
            f"the write-up again.{not_counted}"
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
    unreadable = settings.cutoff_problem()
    if unreadable and options.get("research_cutoff") is None:
        # Not "no cutoff": the paper would cite work the person meant to leave out.
        raise ProjectError(
            f"{unreadable}. Correct it in the project's settings (in the app, or in "
            "project.json), or give the write-up a cutoff of its own (--research-cutoff "
            "YYYY-MM); until then no paper is written."
        )
    everywhere = options.get("model")
    chosen: dict[str, Any] = {
        "writer_model": everywhere or models.reasoning,
        "reflection_model": everywhere or models.reasoning,
        "plotting_model": everywhere or models.coding,
        "fast_model": everywhere or models.fast,
        "effort": models.effort,
        "research_cutoff": None if unreadable else settings.research_cutoff,
        "compute": settings.compute,
        "lab_memory": settings.lab_memory,
        # The figures here are what the runs produced; generating more must not drop them.
        "keep_provided_figures": True,
    }
    chosen.update({name: value for name, value in options.items() if value is not None})
    chosen.pop("model", None)
    chosen["research_cutoff"] = _cutoff_for_the_paper(chosen["research_cutoff"])
    return build_config(
        base_dir=project.paper_dir,
        template=template,
        idea_filename=IDEA_FILE,
        experimental_log_filename=LOG_FILE,
        **chosen,
    )


def _cutoff_for_the_paper(cutoff: Any) -> str | None:
    """The research cutoff in the form the paper pipeline takes, read as the wiki reads it.

    The wiki admitted its sources by this cutoff, and there a year alone means "before
    that year" (``2024`` is ``2024-01``).  The pipeline takes a month only and would
    stop at a year, so the write-up hands it what the wiki applied.
    """
    try:
        return normalize_cutoff(cutoff)
    except ValueError:
        raise ProjectError(
            f"The research cutoff of the write-up is {cutoff!r}; it must be a date such as "
            "2024-06 (YYYY, YYYY-MM or YYYY-MM-DD)."
        ) from None


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


def _count(number: int, what: str) -> str:
    return f"{number} {what}{'s' if number != 1 else ''}"


def _started_summary(template: str, made: _Materials, left: Sequence[Mapping[str, Any]]) -> str:
    """What the person is told when a write-up starts: what it is written from, which
    accepted findings are not in it and why, and which numbers were taken out."""
    said = (
        f"Writing a paper from the wiki with the {Path(template).name} template: "
        f"{_count(len(made.runs), 'succeeded run')}, "
        f"{_count(len(made.findings), 'accepted finding')}"
        + (f", {_count(len(made.provided), 'attached file')}." if made.provided else ".")
    )
    if left:
        verb = "is" if len(left) == 1 else "are"
        said += (
            f" {_count(len(left), 'accepted finding')} {verb} left out ({_left_out_text(left)})."
        )
    removed = sum(len(numbers) for numbers in made.numbers_removed.values())
    if removed:
        which = "; ".join(
            f"{name}: {', '.join(numbers)}" for name, numbers in made.numbers_removed.items()
        )
        verb = "that is on no record was" if removed == 1 else "that are on no record were"
        said += f" {_count(removed, 'number')} {verb} taken out ({which})."
    return said


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
    or no accepted finding can be reported (the message names the findings that
    rest only on rejected or missing experiments), when the project's research
    cutoff cannot be read, and when the project has a worker at work (the
    write-up holds the project's worker lock while it runs: one worker at a time);
    raises ``FileNotFoundError`` for an unknown template and ``TypeError`` for an
    unknown option.  Emits ``paper.started`` and ``paper.finished`` on the project.
    As with ``pipeline.run_paper``, a paper that could not be written is a result
    with ``success=False``, and an interrupt is passed on after it was recorded.
    """
    runs, findings, left = _reportable(project)
    _check_ready(project, runs, findings, left)
    cfg = _paper_config(project, template, options)
    with _as_the_worker(project):
        return _write(project, template, cfg, runs, findings, left)


def _write(
    project: Project,
    template: str,
    cfg: PipelineConfig,
    runs: list[_Run],
    findings: list[ResearchPage],
    left: Sequence[Mapping[str, Any]] = (),
) -> PaperResult:
    """Export the materials and run the pipeline on them, telling the project about it."""
    made = _export(project, cfg.base_dir / MATERIALS_DIR, runs, findings)
    prepare_run_dir(cfg, None)
    project.emit(
        "paper.started",
        _started_summary(template, made, left),
        refs=[_relative(project, made.dest / IDEA_FILE), _relative(project, made.dest / LOG_FILE)],
        data={
            "template": template,
            "dir": _relative(project, cfg.base_dir),
            "results": [run.result.path for run in made.runs],
            "findings": [page.path for page in made.findings],
            "figures": made.figures,
            "provided": made.provided,
            "left_out": [dict(item) for item in left],
            "numbers_removed": made.numbers_removed,
            "sources": made.sources,
            "excluded_sources": made.excluded,
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
