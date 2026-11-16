"""Experiments: from an accepted plan to a result record.

An experiment is a wiki page (the plan) plus files under ``experiments/E<n>/``:
``code/`` holds what runs, ``runs/<k>/`` what each run left behind.  This module
is everything between the two:

* :func:`load_plan` reads the plan from the page's front matter and says what
  is wrong with it; :func:`write_code` and :func:`check_code` keep and vet the
  code (parsed, never executed);
* :func:`approve` is the person's decision to spend time and money on a run,
  :func:`options` tells them what each backend could do with the plan;
* :func:`run` executes one approved experiment on a compute backend, repairs a
  programming error a bounded number of times, and writes the run's records;
  :func:`request_cancel` and :func:`recover` end and settle runs;
* :func:`number_check` traces the numbers of a finding back to records.

Two rules shape all of it.  Nothing here approves anything: only :func:`approve`
moves an experiment to ``approved``, and it acts for the person.  And nothing
here is written by a model: ``run.json`` and the result page are made by code
from what the run produced, with the values as the run wrote them.
"""

from __future__ import annotations

import ast
import bisect
import contextlib
import difflib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as replaced
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from paper_orchestra.compute import ComputeError, JobResult, JobSpec, get_backend
from paper_orchestra.compute.base import (
    HANDLE_NAME,
    LOG_NAME,
    NOTE_PREFIX,
    JobStopped,
    human_size,
    spec_problems,
)
from paper_orchestra.errors import stops_the_run

# The scan that keeps a paper's results out of the shared memory already knows how
# numbers are written and which of them are specific enough to be someone's result;
# a finding is checked with the same notion, not with a second one.
from paper_orchestra.memory.leakscan import NUMBER, is_identifying, number_readings, numbers_in
from paper_orchestra.memory.lock import WikiLock
from paper_orchestra.research import repo as repositories
from paper_orchestra.research.model import (
    COMPUTE_BACKENDS,
    HUMAN,
    SYSTEM,
    Conflict,
    NotAllowed,
    NotFound,
    ProjectError,
    ResearchPage,
    actor_class,
    agent_author,
    compact_stamp,
    format_stamp,
    moved_on,
    normalize_author,
    parse_id,
    parse_stamp,
    stamp_after,
    with_article,
)
from paper_orchestra.research.project import EXPERIMENTS_DIR, INPUTS_DIR, MANIFEST_FILE, Project
from paper_orchestra.utils.common import write_json, write_text

__all__ = [
    "ExperimentPlan",
    "approve",
    "check_code",
    "code_authors",
    "code_digest",
    "code_dir",
    "code_files",
    "job_command",
    "job_spec",
    "load_plan",
    "number_check",
    "options",
    "recover",
    "request_cancel",
    "run",
    "runs",
    "split_problems",
    "supplies_missing_code",
    "unverified_numbers",
    "write_code",
]

MODES: tuple[str, ...] = ("script", "gpu", "repo")
DEFAULT_ENTRYPOINT = "main.py"
DEFAULT_TIMEOUT = 900.0

#: Under ``experiments/E<n>/``.
CODE_DIR = "code"
RUNS_DIR = "runs"
#: Earlier versions of code files, inside the code directory; never part of what runs.
HISTORY_DIR = ".history"
AUTHORS_FILE = "authors.json"
RUN_FILE = "run.json"
#: The prepared copy of a repository inside a run directory; removed when the run ends.
REPO_COPY = "repo"
#: The flag :func:`request_cancel` leaves for the process that runs the experiment.
CANCEL_FLAG = ".cancel"
#: Held, as a kernel lock, by the process that runs the experiment.
RUN_LOCK = ".run.lock"

#: Starts a problem of :func:`check_code` that the person should see but that does not
#: keep the code from running.
WARNING = "Warning: "
#: Where the web app serves a project's files from; result pages link artifacts there.
FILES_URL = "/files/"

MAX_CODE_FILE_BYTES = 1_000_000
MAX_CODE_FILES = 200
#: Bytes of an experiment's code directory that a run copies: it is copied for every run.
MAX_CODE_BYTES = 200 * 1024 * 1024
MAX_METRICS_BYTES = 1_000_000
MAX_METRICS = 200
MAX_METRIC_NAME_CHARS = 100
MAX_METRIC_TEXT_CHARS = 200
#: An artifact is a figure or a table for the wiki, not a dataset or a checkpoint.
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
MAX_ARTIFACTS = 200
#: Lines of the log on a result page, and characters of each.
LOG_TAIL_LINES = 40
LOG_LINE_CHARS = 500
#: Characters of a repair's diff kept in ``run.json``, and of all the diffs of a run
#: together on its result page (a page has a size limit; the record has them whole).
MAX_DIFF_CHARS = 60_000
MAX_DIFF_CHARS_ON_PAGE = 24_000
#: Characters of the reason a run failed that are kept: a rejected hunk can be long.
MAX_ERROR_CHARS = 4_000
#: Seconds between ``run.log`` events, and the lines one of them carries at most.
LOG_EVENT_SECONDS = 2.0
LOG_EVENT_LINES = 8
#: Seconds between looks for the handle of a remote job.
HANDLE_CHECK_SECONDS = 0.5
#: After this many runs in a row that ended because their worker died, the experiment is
#: no longer started again by itself.
MAX_WORKER_DEATHS = 3
#: Text files under ``inputs/`` larger than this are not searched for a finding's numbers.
MAX_EVIDENCE_BYTES = 2_000_000

_RUNNER = agent_author("runner")
_DESIGNER = agent_author("designer")
#: Failures a corrected file can cure: the code's own, as opposed to the backend's, the
#: clock's or the person's.
_REPAIRABLE = frozenset({"exit", "metrics", "code", "patch"})
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"})
_NAME_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
_RUN_NUMBER = re.compile(r"[0-9]{1,9}")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"", "false", "no", "off", "0", "none", "null"})
_NO_GPU = frozenset({"", "false", "no", "none", "null", "cpu", "0"})
_WORKER_STOPPED = "The worker stopped while this run was in progress."

#: Modules whose import means the code talks to the network.
_NETWORK_MODULES = frozenset(
    {
        "aiohttp",
        "boto3",
        "botocore",
        "ftplib",
        "gdown",
        "http.client",
        "httpx",
        "huggingface_hub",
        "kaggle",
        "kagglehub",
        "openml",
        "paramiko",
        "pycurl",
        "requests",
        "smtplib",
        "socket",
        "telnetlib",
        "urllib.request",
        "urllib3",
        "wandb",
        "websocket",
        "websockets",
        "xmlrpc.client",
    }
)
#: Calls that fetch data or models, whatever module they come from.
_DOWNLOAD_CALLS = frozenset(
    {
        "download_url",
        "fetch_openml",
        "from_pretrained",
        "hf_hub_download",
        "load_dataset",
        "snapshot_download",
        "urlopen",
        "urlretrieve",
    }
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return format_stamp(datetime.now(UTC))


def _one_line(text: Any) -> str:
    return " ".join(str("" if text is None else text).split())


def _clip(text: Any, limit: int) -> str:
    text = _one_line(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _text(value: Any) -> str:
    """A front-matter scalar as text; nothing for what is absent."""
    if value is None or isinstance(value, bool | list | tuple | dict):
        return ""
    return str(value).strip()


def _seconds_text(seconds: Any) -> str:
    """``42.1 s``.  One decimal: the duration of a run is not one of its results."""
    if not isinstance(seconds, int | float) or isinstance(seconds, bool):
        return "an unknown time"
    if seconds < 0.05:
        return "less than 0.1 s"
    return f"{seconds:.1f} s" if seconds < 100 else f"{seconds:.0f} s"


def _plain_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _experiment(project: Project, ref: Any) -> ResearchPage:
    """The experiment page ``ref`` names.  ``NotFound`` or ``ProjectError`` otherwise."""
    page = ref if isinstance(ref, ResearchPage) else project.page(str(ref))
    if page.kind != "experiment" or not page.id:
        raise ProjectError(
            f"{page.path} is {with_article(page.kind)} page, not an experiment. Experiments "
            "are the pages under wiki/experiments/, named E1, E2 and so on."
        )
    return page


def _experiment_id(project: Project, ref: Any) -> str:
    """``E3`` from an id in any case, or from any reference to the experiment's page."""
    parsed = parse_id(ref)
    if parsed is not None and parsed[0] == "E":
        return f"E{parsed[1]}"
    if parsed is not None:
        raise ProjectError(f"{ref} is not an experiment: experiments are named E1, E2 and so on.")
    return str(_experiment(project, ref).id)


def _page_or_none(project: Project, experiment_id: str) -> ResearchPage | None:
    try:
        page = project.page(experiment_id)
    except NotFound:
        return None
    return page if page.kind == "experiment" else None


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass
class ExperimentPlan:
    """What an experiment page asks to be run, as read from its front matter."""

    id: str
    path: str
    title: str
    #: ``script`` (CPU, minutes), ``gpu`` (Modal) or ``repo`` (in the person's repository).
    mode: str
    entrypoint: str = DEFAULT_ENTRYPOINT
    #: The command line as an argv; ``None`` runs ``python <entrypoint>``.
    command: list[str] | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT
    gpu: str | None = None
    network: bool = False
    packages: tuple[str, ...] = ()
    #: Repo mode: the directory or the URL of the repository, and the ref to run at.
    repo: str | None = None
    repo_ref: str | None = None
    #: Chosen at approval: ``local`` or ``modal``.
    backend: str | None = None
    est_minutes: float | None = None
    est_usd: float | None = None


@dataclass(frozen=True)
class _Limits:
    """What one run may spend (``settings.limits``, or what the caller allows instead)."""

    seconds: float
    usd: float
    repair_rounds: int


def _limits(project: Project, override: Mapping[str, Any] | None = None) -> _Limits:
    base = project.settings.limits
    chosen = {
        name: (override or {}).get(name, base[name])
        for name in ("experiment_seconds", "experiment_usd", "repair_rounds")
    }
    return _Limits(
        seconds=float(chosen["experiment_seconds"]),
        usd=float(chosen["experiment_usd"]),
        repair_rounds=int(chosen["repair_rounds"]),
    )


@dataclass
class _Reading:
    """A plan as far as it could be read, and what is wrong with it."""

    page: ResearchPage
    plan: ExperimentPlan
    problems: list[str] = field(default_factory=list)
    #: What was adjusted rather than refused (a time limit capped at the project's).
    notes: list[str] = field(default_factory=list)


def _flag(value: Any, name: str, problems: list[str]) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text in _TRUE:
        return True
    if text not in _FALSE:
        problems.append(f"{name} is {value!r}; it must be true or false.")
    return False


def _amount(value: Any, name: str, what: str, problems: list[str]) -> float | None:
    """A number that is zero or more, or ``None`` when the page gives none."""
    if value is None or value == "" or value == []:
        return None
    try:
        number = math.nan if isinstance(value, bool) else float(str(value).strip().lstrip("$"))
    except ValueError:
        number = math.nan
    if not 0 <= number < math.inf:
        problems.append(f"{name} is {value!r}; it must be a number ({what}), zero or more.")
        return None
    return number


def _items(value: Any) -> list[str]:
    """A list field, however it was written: a list, or one line with commas."""
    if isinstance(value, list | tuple):
        return [text for item in value if (text := str(item).strip())]
    text = _text(value)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    items: list[str] = []
    for piece in text.split(","):
        piece = piece.strip().strip("\"'").strip()
        if piece and items and piece[0] in "<>=!~":
            items[-1] += "," + piece  # the second half of `numpy>=1.26,<2`
        elif piece:
            items.append(piece)
    return items


def _command(value: Any, problems: list[str]) -> list[str] | None:
    if isinstance(value, list | tuple):
        return [str(part) for part in value] or None
    text = _text(value)
    if not text:
        return None
    try:
        return shlex.split(text) or None
    except ValueError as exc:
        problems.append(f"command is {text!r}, which cannot be read as a command line ({exc}).")
        return None


def _name_problem(name: Any) -> str | None:
    """Why ``name`` cannot name a file in a code directory, or ``None``."""
    if not isinstance(name, str) or not name:
        return "a file needs a name"
    if len(name) > 200 or len(name.split("/")) > 8:
        return "the name is too long"
    if not all(_NAME_PART.fullmatch(part) for part in name.split("/")):
        return (
            "use a relative path of plain names (letters, digits, '_', '-' and '.', none "
            "starting with a dot), such as main.py or lib/data.py"
        )
    return None


def _timeout(raw: Any, limits: _Limits, cap: bool, reading: _Reading) -> float:
    """The time limit of a run: the page's, never above the project's."""
    limit = limits.seconds
    if limit <= 0:
        reading.problems.append(
            "The project allows experiments 0 seconds (settings.limits.experiment_seconds), "
            "so none can run. Raise the limit in the project's settings."
        )
        return DEFAULT_TIMEOUT
    if raw is None or raw == "" or raw == []:
        return min(DEFAULT_TIMEOUT, limit)
    try:
        timeout = math.nan if isinstance(raw, bool) else float(str(raw).strip())
    except ValueError:
        timeout = math.nan
    if not 0 < timeout < math.inf:
        reading.problems.append(
            f"timeout_seconds is {raw!r}; it must be a positive number of seconds."
        )
        return min(DEFAULT_TIMEOUT, limit)
    if timeout <= limit:
        return timeout
    if cap:
        reading.notes.append(
            f"The time limit of {timeout:g} s on the page was capped to the {limit:g} s the "
            "project allows for experiments."
        )
        return limit
    reading.problems.append(
        f"Its time limit of {timeout:g} s is above the {limit:g} s the project allows for "
        "experiments (settings.limits.experiment_seconds). Lower timeout_seconds, or raise "
        "the limit in the project's settings."
    )
    return timeout


def _within(path: Path, base: Path) -> bool:
    with contextlib.suppress(OSError):
        return path.resolve().is_relative_to(base.resolve())
    return False


def _local_repository(project: Project, location: str) -> Path:
    path = Path(location).expanduser()
    return path if path.is_absolute() else project.root / path


def _repository(project: Project, reading: _Reading) -> None:
    """Fill in where a repo-mode plan runs, and vet it."""
    page, plan = reading.page, reading.plan
    given = project.settings.repo
    named = _text(page.meta.get("repo"))
    location = named or (given.path if given else "")
    if not location:
        reading.problems.append(
            "It runs in the person's repository (mode: repo), but no repository is named: "
            "set one in the project's settings (settings.repo), or name it on the page "
            "(repo: a path or a URL)."
        )
        return
    ref = _text(page.meta.get("repo_ref")) or (given.ref if given and not named else None)
    plan.repo_ref = ref or None
    if ref and (problem := repositories.ref_problem(ref)):
        reading.problems.append(problem)
    if repositories.is_url(location):
        plan.repo = location
        if problem := repositories.url_problem(location):
            reading.problems.append(problem)
        elif not plan.network:
            reading.problems.append(
                f"Its repository has to be cloned from "
                f"{repositories.display_url(location)}, which needs network "
                "access: set network: true on the page, so that the person approves the "
                "run with it, or point the project at a local copy of the repository."
            )
        return
    path = _local_repository(project, location)
    plan.repo = str(path)
    local = bool(given) and not repositories.is_url(given.path)
    theirs = _local_repository(project, given.path) if given and local else None
    # Which directory of this machine an experiment copies, and may send to a remote
    # backend, is the person's choice.  A page an agent wrote can only name theirs.
    if named and page.author != HUMAN and not (theirs and _within(path, theirs)):
        reading.problems.append(
            f"It names {path} as its repository, which is not the repository this project "
            f"was given ({theirs or 'none'}). Only the person chooses which directory an "
            "experiment may copy: set it in the project's settings (settings.repo), or "
            "write the experiment's page yourself."
        )
    elif not path.is_dir():
        reading.problems.append(
            f"Its repository {path} does not exist or is not a directory. Correct the path "
            "in the project's settings (settings.repo)."
        )


def _read(
    project: Project,
    page: ResearchPage,
    *,
    overrides: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
    cap: bool = False,
) -> _Reading:
    """Read the plan on ``page``.  Never raises: what is wrong is in ``problems``.

    ``overrides`` are the options chosen at approval (backend, gpu, timeout_seconds,
    network); ``None`` leaves the page's value.  With ``cap`` a time limit above the
    project's is lowered to it instead of being a problem: a run obeys the limit of
    today, whatever was approved under an earlier one.
    """
    meta = dict(page.meta)
    meta.update({key: value for key, value in (overrides or {}).items() if value is not None})
    plan = ExperimentPlan(id=str(page.id), path=page.path, title=page.title, mode="script")
    reading = _Reading(page=page, plan=plan)
    problems = reading.problems

    mode = _text(meta.get("mode")).lower() or "script"
    if mode in MODES:
        plan.mode = mode
    else:
        problems.append(f"mode is {meta.get('mode')!r}; the modes are: {', '.join(MODES)}.")
    plan.entrypoint = _text(meta.get("entrypoint")) or DEFAULT_ENTRYPOINT
    if _name_problem(plan.entrypoint):
        problems.append(
            f"entrypoint is {plan.entrypoint!r}; it must be the relative name of a file in "
            "the experiment's code, such as main.py."
        )
        plan.entrypoint = DEFAULT_ENTRYPOINT
    plan.command = _command(meta.get("command"), problems)
    plan.timeout_seconds = _timeout(
        meta.get("timeout_seconds"), _limits(project, limits), cap, reading
    )
    gpu = _text(meta.get("gpu"))
    plan.gpu = None if gpu.lower() in _NO_GPU else gpu
    plan.network = _flag(meta.get("network"), "network", problems)
    plan.packages = tuple(_items(meta.get("packages")))
    plan.backend = _text(meta.get("backend")).lower() or None
    plan.est_minutes = _amount(meta.get("est_minutes"), "est_minutes", "minutes", problems)
    plan.est_usd = _amount(meta.get("est_usd"), "est_usd", "US dollars", problems)

    if plan.mode == "gpu" and not plan.gpu:
        problems.append("A GPU experiment names its GPU type on the page, for example gpu: T4.")
    if plan.mode == "script" and plan.gpu:
        problems.append(
            f"It asks for a GPU ({plan.gpu}) but its mode is script, which runs on the CPU; "
            "a GPU needs mode: gpu."
        )
    if plan.backend and plan.backend not in COMPUTE_BACKENDS:
        problems.append(
            f"backend is {plan.backend!r}; the backends are: {', '.join(COMPUTE_BACKENDS)}."
        )
    elif plan.backend == "local" and plan.gpu:
        problems.append(
            f"It needs a GPU ({plan.gpu}), and the local backend runs on this machine "
            "without one. Run it on the Modal backend."
        )
    if plan.mode == "repo":
        _repository(project, reading)
    problems.extend(problem for problem in spec_problems(job_spec(plan)) if problem not in problems)
    return reading


def load_plan(project: Project, ref: str) -> ExperimentPlan:
    """The plan of the experiment ``ref`` names (its id, its path or a link to it).

    Raises ``NotFound`` when there is no such page and ``ProjectError`` when it
    is not an experiment or when the plan cannot be run as written; the message
    names every problem: an unknown mode, a GPU asked of a backend without one,
    repo mode without a repository, a time limit above the project's, values
    that are not what the field holds.  A time limit the page does not give is
    the default, never more than the project's limit.
    """
    reading = _read(project, _experiment(project, ref))
    if reading.problems:
        raise ProjectError(
            f"{reading.plan.id} cannot be run as it is written:\n- " + "\n- ".join(reading.problems)
        )
    return reading.plan


def job_command(plan: ExperimentPlan) -> list[str]:
    """The argv a run of ``plan`` executes in its working directory."""
    return list(plan.command) if plan.command else ["python", plan.entrypoint]


def job_spec(plan: ExperimentPlan, *, timeout: float | None = None) -> JobSpec:
    """What ``plan`` asks of a compute backend."""
    return JobSpec(
        command=job_command(plan),
        timeout=float(timeout if timeout is not None else plan.timeout_seconds),
        gpu=plan.gpu,
        network=plan.network,
        packages=tuple(plan.packages),
    )


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


def code_dir(project: Project, experiment_id: str) -> Path:
    """``experiments/E<n>/code``: where the files of the experiment live (not created)."""
    return project.experiment_dir(_experiment_id(project, experiment_id)) / CODE_DIR


def _code_paths(directory: Path) -> Iterator[tuple[str, Path]]:
    """``(relative posix name, path)`` of the files that are the experiment's code.

    Hidden names (the kept versions, anything a tool left) and links are not code.
    """
    for folder, dirs, names in os.walk(directory):
        here = Path(folder)
        dirs[:] = sorted(
            name
            for name in dirs
            if not name.startswith(".") and name != "__pycache__" and not (here / name).is_symlink()
        )
        for name in sorted(names):
            path = here / name
            if not name.startswith(".") and path.is_file() and not path.is_symlink():
                yield path.relative_to(directory).as_posix(), path


def _read_code(path: Path) -> str | None:
    """The text of a code file; ``None`` for what is not text or too large to show."""
    try:
        if path.stat().st_size > MAX_CODE_FILE_BYTES:
            return None
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def code_files(project: Project, experiment_id: str) -> dict[str, str]:
    """The experiment's code as ``{relative name: text}``, sorted by name.

    Files that are not text (data someone put there by hand) are left out here;
    a run still gets them.
    """
    files: dict[str, str] = {}
    for name, path in sorted(_code_paths(code_dir(project, experiment_id))):
        text = _read_code(path)
        if text is not None:
            files[name] = text
    return files


def code_digest(project: Project, experiment_id: str) -> str:
    """SHA-256 over the names and contents of the experiment's code files."""
    return _current_digest(project, experiment_id)


def code_authors(project: Project, experiment_id: str) -> dict[str, dict[str, str]]:
    """Who wrote the current version of each code file, and when:
    ``{name: {"author", "written"}}``, as :func:`write_code` records it.  A file that
    was put into the directory by hand has no entry."""
    return _authors(code_dir(project, _experiment_id(project, experiment_id)))


def _current_digest(project: Project, experiment_id: str) -> str:
    return _digest(_code_paths(code_dir(project, experiment_id)))


def _digest(files: Iterable[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files):
        data = path.read_bytes()
        # Name and length first, so that no two sets of files hash alike.
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def _code_name(name: Any) -> str:
    problem = _name_problem(name)
    if problem:
        raise ProjectError(f"{name!r} cannot name a file of an experiment's code: {problem}.")
    if name == "metrics.json" or name.split("/")[0] == "artifacts":
        raise ProjectError(
            f"{name} is what a run writes, so it cannot be part of the code: results come "
            "from running the experiment, not from a file put there beforehand."
        )
    return name


def _code_text(name: str, text: Any) -> str:
    if not isinstance(text, str) or "\0" in text:
        raise ProjectError(f"The content of {name} must be text, not {type(text).__name__}.")
    if len(text.encode("utf-8")) > MAX_CODE_FILE_BYTES:
        raise ProjectError(
            f"{name} is larger than {human_size(MAX_CODE_FILE_BYTES)}: a code file holds "
            "code. Generate data in the experiment instead of shipping it."
        )
    return text


def _check_may_write(
    project: Project, page: ResearchPage | None, author: str, experiment_id: str
) -> None:
    """Raise ``NotAllowed`` unless ``author`` may change the code of the experiment.

    The rule is the one for the page itself: an agent changes only what it
    proposed and nobody decided on.  Otherwise the plan would be locked once the
    person approved it while the code that runs was not.  The runner's repair
    is pipeline code, bounded and recorded in the run.

    One case lies outside that rule: an accepted plan that has no code yet (the
    person's own, or an analyst's next step they accepted).  The designer
    supplies that code; see :func:`supplies_missing_code`.
    """
    if page is None or actor_class(author) != "agent":
        return
    if moved_on(page):
        reason = f"is {page.state}, so its code stays as it was when that was decided"
    elif page.review == "rejected":
        reason = "was rejected by the person and is kept as it was"
    elif supplies_missing_code(project, page, author):
        return
    elif page.decided_by:
        reason = "was accepted by the person, so its code stays as they read it"
    elif page.author != author:
        reason = f"was written by {page.author}, and only its author changes its code"
    else:
        return
    raise NotAllowed(
        f"{experiment_id} {reason}. Say what should change in a proposal of your own, or "
        "leave the change to the person."
    )


def supplies_missing_code(project: Project, page: ResearchPage, author: str) -> bool:
    """Whether ``author`` is the designer writing code that nobody has decided on.

    True for an accepted plan that still waits for approval when every code
    file there was written by the designer after the plan became the person's
    (their decision, or their writing it), so also when there is none yet.
    Nothing the person read is changed by it: they read this code when they
    approve the run, and that approval closes it like any other.
    """
    if author != _DESIGNER or page.kind != "experiment" or not page.id:
        return False
    if page.review != "accepted" or moved_on(page):
        return False
    since = parse_stamp(page.meta.get("decided_at")) or parse_stamp(page.meta.get("created"))
    if since is None:
        return False
    directory = code_dir(project, str(page.id))
    authors = _authors(directory)
    for name, _path in _code_paths(directory):
        entry = authors.get(name, {})
        written = parse_stamp(entry.get("written"))
        if entry.get("author") != author or written is None or written <= since:
            return False
    return True


def _history_dir(directory: Path) -> Path:
    return directory / HISTORY_DIR


def _authors(directory: Path) -> dict[str, dict[str, str]]:
    """Who wrote the current version of each code file, and when."""
    data = _read_json(_history_dir(directory) / AUTHORS_FILE)
    return {
        str(name): entry
        for name, entry in (data.items() if isinstance(data, dict) else ())
        if isinstance(entry, dict)
    }


def _keep_version(directory: Path, name: str, data: bytes, entry: Mapping[str, str]) -> None:
    """File the version of ``name`` that is about to be replaced, as the project files
    the versions of a page: by when it was written and by whom."""
    folder = _history_dir(directory) / name
    folder.mkdir(parents=True, exist_ok=True)
    writer = re.sub(r"[^A-Za-z0-9_\-]", "-", entry.get("author") or HUMAN)
    stamp = entry.get("written") or _now()
    suffix = PurePosixPath(name).suffix
    while (target := folder / f"{compact_stamp(stamp)}-{writer}{suffix}").exists():
        stamp = stamp_after(stamp, stamp)
    target.write_bytes(data)


def write_code(
    project: Project, experiment_id: str, files: Mapping[str, str], *, author: str
) -> list[str]:
    """Write ``files`` (``{relative name: text}``) into the experiment's code directory
    and return what :func:`check_code` finds afterwards.

    Names are plain relative paths; nothing hidden, nothing outside the
    directory (not through a link either), and not the names a run writes its
    results to.  Every name and content is checked before the first file is
    written.  The version a file replaces is kept under
    ``code/.history/<name>/``.  An agent may write only the code of its own
    proposal that nobody has decided on, and the designer the code of a plan
    the person accepted without any (``NotAllowed`` otherwise); the person and
    the runner's repair may always.
    """
    author = normalize_author(author)
    experiment_id = _experiment_id(project, experiment_id)
    _check_may_write(project, _page_or_none(project, experiment_id), author, experiment_id)
    if not isinstance(files, Mapping) or not files:
        raise ProjectError("No files were given: pass the files to write as {name: text}.")
    if len(files) > MAX_CODE_FILES:
        raise ProjectError(f"An experiment's code is at most {MAX_CODE_FILES} files.")
    checked = {_code_name(name): _code_text(name, text) for name, text in files.items()}
    directory = code_dir(project, experiment_id)
    for name in checked:
        target = directory / name
        if target.is_symlink() or target.is_dir():
            raise ProjectError(f"{name} cannot be written: a link or a directory is there.")
        if not _within(target.parent, directory):
            # A folder of the code that is a link to somewhere else: writing "into" it
            # would write there.
            raise ProjectError(
                f"{name} cannot be written: a link in its path leads out of the experiment's "
                f"code directory {directory}. Replace the link with a real directory."
            )
    authors = _authors(directory)
    stamp = _now()
    try:
        for name, text in checked.items():
            target = directory / name
            data = text.encode("utf-8")
            previous = target.read_bytes() if target.is_file() else None
            if previous == data:
                continue
            if previous is not None:
                _keep_version(directory, name, previous, authors.get(name, {}))
            write_text(target, text)
            authors[name] = {"author": author, "written": stamp}
        write_json(_history_dir(directory) / AUTHORS_FILE, authors)
    except OSError as exc:
        raise ProjectError(
            f"The code of {experiment_id} could not be written ({exc}). Check that "
            f"{directory} can be written."
        ) from exc
    return check_code(project, experiment_id)


def split_problems(problems: Iterable[str]) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` of what :func:`check_code` returned."""
    errors = [problem for problem in problems if not problem.startswith(WARNING)]
    warnings = [problem for problem in problems if problem.startswith(WARNING)]
    return errors, warnings


def _parse(name: str, path: Path) -> tuple[ast.Module | None, str | None]:
    """The syntax tree of a Python file, or why it has none.  Nothing is executed."""
    try:
        source = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{name} cannot be read as UTF-8 text ({exc})."
    try:
        return ast.parse(source, filename=name), None
    except SyntaxError as exc:
        shown = f": {exc.text.strip()}" if exc.text and exc.text.strip() else ""
        return None, f"{name}, line {exc.lineno or 1}: {exc.msg}{shown}"
    except (ValueError, RecursionError, MemoryError) as exc:
        return None, f"{name} cannot be parsed ({exc})."


def _uses_network(module: str) -> str | None:
    """The network module ``module`` is or lies in, if any."""
    parts = module.split(".")
    for end in range(1, len(parts) + 1):
        if ".".join(parts[:end]) in _NETWORK_MODULES:
            return ".".join(parts[:end])
    return None


def _network_use(tree: ast.Module) -> Iterator[tuple[int, str, str]]:
    """``(line, key, what it does)`` for each obvious use of the network in ``tree``."""
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            for alias in node.names:
                if node.module == "sklearn.datasets" and alias.name.startswith("fetch_"):
                    yield node.lineno, alias.name, f"it imports {alias.name}, which downloads data"
        for module in modules:
            if found := _uses_network(module):
                yield node.lineno, found, f"it imports {found}, which uses the network"
        if isinstance(node, ast.Call):
            function = node.func
            called = (
                getattr(function, "attr", None) or getattr(function, "id", None) or "a function"
            )
            if called in _DOWNLOAD_CALLS:
                yield node.lineno, called, f"it calls {called}(), which fetches data or a model"
            for keyword in node.keywords:
                value = keyword.value
                if keyword.arg == "download" and isinstance(value, ast.Constant) and value.value:
                    yield node.lineno, "download", f"it calls {called}(download=True)"


def _network_warnings(name: str, tree: ast.Module) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for line, key, what in sorted(_network_use(tree)):
        if key not in seen:
            seen.add(key)
            warnings.append(
                f"{WARNING}{name}, line {line}: {what}, but the experiment runs without "
                "network access. Generate or bundle what it needs, or set network: true on "
                "the page so that the person can approve the run with it."
            )
    return warnings


def _entry_file(plan: ExperimentPlan) -> str | None:
    """The file the command starts, when the command is ``python <file> ...``."""
    command = job_command(plan)
    if len(command) >= 2 and command[0] in ("python", "python3") and command[1].endswith(".py"):
        return command[1]
    return None


def _repo_problems(plan: ExperimentPlan, files: Mapping[str, Path]) -> list[str]:
    """What is wrong with the code of a repo-mode experiment, seen against its repository."""
    problems: list[str] = []
    patch = files.get(repositories.PATCH_NAME)
    if patch is not None:
        try:
            repositories.parse_patch(patch.read_text(encoding="utf-8", errors="replace"))
        except repositories.PatchError as exc:
            problems.append(str(exc))
    local = Path(plan.repo) if plan.repo and not repositories.is_url(plan.repo) else None
    if local is None or not local.is_dir() or plan.repo_ref:
        return problems
    for name in files:
        if name != repositories.PATCH_NAME and os.path.lexists(local / name):
            problems.append(
                f"{name} is also a file of the repository. A file of the experiment is added "
                "next to the repository's files and never replaces one: change the "
                f"repository's file through {repositories.PATCH_NAME}, or rename this one."
            )
    entry = _entry_file(plan)
    if entry and entry not in files and not (local / entry).is_file():
        problems.append(
            f"The entry point {entry} is neither in the experiment's code nor in the "
            f"repository: the run starts with `{shlex.join(job_command(plan))}`."
        )
    return problems


def _plan_for_code(project: Project, experiment_id: str) -> ExperimentPlan:
    """The plan as far as the code check needs it; defaults where there is no page yet."""
    page = _page_or_none(project, experiment_id)
    if page is None:
        return ExperimentPlan(id=experiment_id, path="", title="", mode="script")
    return _read(project, page).plan


def check_code(project: Project, experiment_id: str) -> list[str]:
    """What is wrong with the experiment's code, found by reading it.  Nothing is run.

    Errors, which keep the experiment from being approved: no code at all, a
    missing entry point, a Python file that does not parse, a patch that is not
    one, a file that would replace one of the repository.  Then warnings, each
    starting with ``WARNING``, for what the person should know when approving:
    obvious use of the network by an experiment that runs without it, and code
    that never mentions ``metrics.json``.  :func:`split_problems` tells the two
    apart.
    """
    experiment_id = _experiment_id(project, experiment_id)
    plan = _plan_for_code(project, experiment_id)
    files = dict(_code_paths(code_dir(project, experiment_id)))
    in_repo = plan.mode == "repo"
    if not files and not in_repo:
        return [
            f"{experiment_id} has no code yet: write its files to "
            f"experiments/{experiment_id}/{CODE_DIR}/ (the entry point is {plan.entrypoint})."
        ]
    errors: list[str] = []
    warnings: list[str] = []
    mentions_metrics = False
    for name, path in files.items():
        if not name.endswith(".py"):
            continue
        tree, problem = _parse(name, path)
        if tree is None:
            errors.append(str(problem))
            continue
        mentions_metrics = mentions_metrics or "metrics.json" in (_read_code(path) or "")
        if not plan.network:
            warnings.extend(_network_warnings(name, tree))
    if in_repo:
        errors.extend(_repo_problems(plan, files))
    else:
        entry = _entry_file(plan)
        if entry and entry not in files:
            errors.append(
                f"The entry point {entry} is missing from the code: the run starts with "
                f"`{shlex.join(job_command(plan))}`."
            )
        if not mentions_metrics:
            warnings.append(
                f"{WARNING}no file of the code mentions metrics.json. A run that does not "
                "write its results there, as a flat JSON object, fails."
            )
        if entry is None:
            warnings.append(
                f"{WARNING}the run starts with `{shlex.join(job_command(plan))}`, which is "
                "not a Python file of the experiment. Read the command before approving it."
            )
    return errors + warnings[:20]


# ---------------------------------------------------------------------------
# Backends and budgets
# ---------------------------------------------------------------------------


@dataclass
class _Choice:
    """A backend someone may run on, or why it could not be made."""

    name: str
    backend: Any = None
    problem: str | None = None
    #: Whether it was built here and has to be closed here.
    owned: bool = False

    def close(self) -> None:
        if self.owned and self.backend is not None:
            with contextlib.suppress(Exception):
                self.backend.close()


def _build(name: str) -> _Choice:
    """The backend called ``name``, made with its defaults."""
    if name not in COMPUTE_BACKENDS:
        # The plan already says that this is not a backend.
        return _Choice(name)
    try:
        return _Choice(name, get_backend(name), owned=True)
    except (ComputeError, ValueError) as exc:
        return _Choice(name, problem=f"The {name} backend cannot be used: {exc}")


def _choices(backends: Any) -> list[_Choice]:
    """The backends in ``backends``: a mapping of name to backend, or names and backends."""
    if isinstance(backends, Mapping):
        return [
            _Choice(str(name), backend) if backend is not None else _build(str(name))
            for name, backend in backends.items()
        ]
    if isinstance(backends, str) or hasattr(backends, "run_job"):
        backends = [backends]
    return [
        _build(item) if isinstance(item, str) else _Choice(str(getattr(item, "name", "")), item)
        for item in backends or ()
    ]


def _chosen(name: str, backends: Any) -> _Choice:
    for choice in _choices(backends) if backends is not None else ():
        if choice.name == name:
            return choice
        choice.close()
    return _build(name)


def _default_backend(project: Project, page: ResearchPage, gpu: str | None = None) -> str:
    """Where an experiment runs when nobody said: whatever asks for a GPU on Modal, which
    is the only backend with GPUs, the others on the project's backend.

    ``gpu`` is the GPU chosen at approval; ``None`` leaves the page's.
    """
    named = _text(page.meta.get("backend")).lower()
    if named:
        return named
    asked = _text(page.meta.get("gpu") if gpu is None else gpu).lower()
    on_gpu = _text(page.meta.get("mode")).lower() == "gpu" or asked not in _NO_GPU
    return "modal" if on_gpu else project.settings.compute


def _cost_bound(plan: ExperimentPlan, backend: Any, seconds: float) -> float | None:
    """What ``seconds`` of the plan's resources cost on ``backend``; ``None`` when nothing
    is billed or no price is known."""
    if getattr(backend, "name", "") != "modal":
        return None
    from paper_orchestra.compute.modal_backend import estimate_cost

    return estimate_cost(
        seconds,
        gpu=plan.gpu,
        cpu=float(getattr(backend, "cpu", 2.0)),
        memory_mb=int(getattr(backend, "memory_mb", 4096)),
    )


def _spent_usd(project: Project) -> float:
    """What the runs of the project have cost so far, by their records.

    The cost limit is for all of them together (``settings.limits.experiment_usd``).
    A run that is not billed, or whose cost is not known, counts as nothing.
    """
    total = 0.0
    root = project.root / EXPERIMENTS_DIR
    for folder in root.iterdir() if root.is_dir() else ():
        parsed = parse_id(folder.name)
        if parsed is None or folder.name != f"E{parsed[1]}":
            continue  # not the directory of an experiment
        for _, record in _numbered_runs(project, folder.name):
            cost = record.get("cost_usd")
            if isinstance(cost, int | float) and not isinstance(cost, bool) and 0 < cost < math.inf:
                total += cost
    return total


def _budget_refusals(
    plan: ExperimentPlan, backend: Any, limits: _Limits, spent: float, *, remainder: bool = False
) -> list[str]:
    """Why the run would take the project past its cost limit, with the numbers.

    The limit is a total, so what a run may cost is what is left of it.
    ``spent`` is what the project's runs have cost so far.  With ``remainder``
    the caller has taken that off already: ``limits.usd`` is what is left, and
    ``spent`` what the earlier runs of this call cost.
    """
    refusals: list[str] = []
    if remainder:
        limit = (
            f"the ${limits.usd:.2f} that is left of what the project allows for its "
            "experiments (settings.limits.experiment_usd)"
        )
        so_far = f", on top of the ${spent:.2f} its earlier attempts cost" if spent else ""
    else:
        limit = (
            f"the ${limits.usd:.2f} the project allows for its experiments in total "
            "(settings.limits.experiment_usd)"
        )
        so_far = f", on top of the ${spent:.2f} its runs have cost so far" if spent else ""
    over = limits.usd + 1e-9  # sums of cents are not exact
    if plan.est_usd is not None and spent + plan.est_usd > over:
        refusals.append(
            f"Its estimated cost is ${plan.est_usd:.2f}{so_far}, above {limit}. Make the "
            "experiment smaller, or raise the limit in the project's settings."
        )
    bound = _cost_bound(plan, backend, plan.timeout_seconds)
    if bound is not None and spent + bound > over:
        refusals.append(
            f"Run to its time limit of {plan.timeout_seconds:g} s it could cost about "
            f"${bound:.2f} on Modal ({plan.gpu or 'CPU only'}){so_far}, above {limit}. "
            "Shorten the time limit, or raise the limit in the project's settings."
        )
    billed = getattr(backend, "name", "") == "modal"
    unknown = plan.est_usd is None and bound is None
    if billed and unknown and limits.usd - spent <= 1e-9:
        # Nothing says what this run would cost, and nothing is left to spend on it.
        used = "" if remainder else f": ${spent:.2f} of ${limits.usd:.2f} is spent"
        refusals.append(
            "Nothing is left of what the project allows for its experiments "
            f"(settings.limits.experiment_usd){used}, and this run would be billed. Raise "
            "the limit in the project's settings."
        )
    return refusals


def _refusals(
    project: Project, reading: _Reading, choice: _Choice, limits: _Limits, spent: float
) -> list[str]:
    """Every reason the plan cannot run on the chosen backend as things are."""
    reasons = list(reading.problems)
    reasons += split_problems(check_code(project, reading.plan.id))[0]
    if choice.problem:
        reasons.append(choice.problem)
    elif choice.backend is not None and not reading.problems:
        backend = choice.backend
        reasons += list(backend.supports(job_spec(reading.plan)))
        reasons += list(backend.check(python=True, latex=False))
        reasons += _budget_refusals(reading.plan, backend, limits, spent)
    return list(dict.fromkeys(reasons))


def _backend_notes(
    plan: ExperimentPlan, choice: _Choice, limits: _Limits, spent: float
) -> list[str]:
    """What the person should know about running the plan on this backend."""
    notes: list[str] = []
    backend = choice.backend
    if plan.network:
        notes.append(
            "The page asks for network access (network: true). Approving the run approves "
            "that too; approve it with network off to run it without."
        )
    if backend is None:
        return notes
    try:
        jobs = dict(backend.describe().get("jobs") or {})
    except Exception:  # a description is a courtesy; its absence decides nothing
        jobs = {}
    if not plan.network and jobs.get("network_isolation") is False:
        notes.append(
            f"The {choice.name} backend cannot block network access: the code runs on this "
            "machine as you, not in a sandbox."
        )
    bound = _cost_bound(plan, backend, plan.timeout_seconds)
    if bound is not None:
        used = f", of which ${spent:.2f} is spent" if spent else ""
        notes.append(
            f"Run to its time limit of {plan.timeout_seconds:g} s it would cost about "
            f"${bound:.2f}; the project allows ${limits.usd:.2f} for its experiments in "
            f"total{used}."
        )
    elif choice.name == "modal" and plan.gpu:
        notes.append(f"No price is known for the GPU type {plan.gpu}, so its cost is not bounded.")
    return notes


def options(project: Project, ref: str, backends: Any) -> list[dict[str, Any]]:
    """What each backend could do with the experiment's plan, for the approval form.

    ``backends`` is a mapping ``{name: backend}``, or a list of backends or of
    names (a name is built with its defaults).  One entry each, in that order::

        {"backend": "modal", "can_run": True, "reasons": [], "est_minutes": 10.0,
         "est_usd": 0.16, "notes": [...], "max_usd": 0.24, "timeout_seconds": 900.0}

    ``reasons`` are what keeps it from running there (the plan's problems, errors
    in the code, what ``ComputeBackend.supports`` and ``check`` say, and what is
    left of the project's cost limit).  ``est_minutes`` and ``est_usd`` are the
    plan's own; a missing
    ``est_usd`` is priced from ``est_minutes`` where the backend bills.
    ``max_usd`` is the cost of running to the time limit, when a price is known.
    ``notes`` are for the person: warnings about the code, and what the backend
    cannot guarantee.  Nothing is changed by asking.
    """
    page = _experiment(project, ref)
    limits = _limits(project)
    spent = _spent_usd(project)
    warnings = split_problems(check_code(project, str(page.id)))[1]
    found: list[dict[str, Any]] = []
    for choice in _choices(backends):
        try:
            reading = _read(project, page, overrides={"backend": choice.name})
            plan = reading.plan
            reasons = _refusals(project, reading, choice, limits, spent)
            est_usd = plan.est_usd
            if est_usd is None and plan.est_minutes is not None:
                est_usd = _cost_bound(plan, choice.backend, plan.est_minutes * 60)
            found.append(
                {
                    "backend": choice.name,
                    "can_run": not reasons,
                    "reasons": reasons,
                    "est_minutes": plan.est_minutes,
                    "est_usd": est_usd,
                    "notes": [
                        *warnings,
                        *reading.notes,
                        *_backend_notes(plan, choice, limits, spent),
                    ],
                    "max_usd": _cost_bound(plan, choice.backend, plan.timeout_seconds),
                    "timeout_seconds": plan.timeout_seconds,
                }
            )
        finally:
            choice.close()
    return found


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def _mark_testing(project: Project, page: ResearchPage) -> None:
    """Move the accepted hypotheses the experiment tests from ``open`` to ``testing``.

    A hypothesis under test is no longer its author's to rewrite: what is being
    tested must stay what the person read.
    """
    for target in page.relations.get("tests", []):
        with contextlib.suppress(ProjectError):
            hypothesis = project.page(target)
            waiting = hypothesis.review == "accepted" and hypothesis.state == "open"
            if hypothesis.kind == "hypothesis" and waiting:
                project.set_state(hypothesis.path, "testing", by=SYSTEM)


def _flag_path(project: Project, experiment_id: str) -> Path:
    return project.experiment_dir(experiment_id) / CANCEL_FLAG


def approve(
    project: Project,
    ref: str,
    *,
    backend: str | None = None,
    gpu: str | None = None,
    timeout_seconds: float | None = None,
    network: bool | None = None,
    base_updated: str | None = None,
    code_digest: str | None = None,
    backends: Any = None,
) -> ResearchPage:
    """The person approves a run of the experiment; returns its page, now ``approved``.

    This is the person's decision and nothing else calls it for them: it is what
    the command line and the web app do when the person says so.  The options
    are recorded on the page (``backend``, ``gpu``, ``timeout_seconds``,
    ``network``; ``None`` keeps what the page says, and an experiment without a
    backend gets the project's, one that asks for a GPU Modal).  Then the page moves
    ``planned -> approved`` (also from ``succeeded``, ``failed`` and
    ``cancelled``: one approval is one run) and the accepted hypotheses it tests
    move to ``testing``.

    Refused with ``ProjectError``, saying why, when the page is not accepted, is
    queued or running, when the plan or its code has errors, when the chosen
    backend cannot run it (``ComputeBackend.supports`` and ``check``), or when
    it could cost more than is left of what the project allows for all its runs
    together.  ``base_updated`` is the page's
    ``updated`` stamp as the person saw it and ``code_digest`` the digest
    (:func:`code_digest`) of the code they read: a page or code changed since
    is a ``Conflict``, because what is approved is what was read.  ``backends``
    (``{name: backend}``) supplies the backend to ask instead of one built with
    its defaults.
    """
    page = _experiment(project, ref)
    name = str(page.id)
    if code_digest is not None and code_digest != _current_digest(project, name):
        raise Conflict(
            f"The code of {name} changed after the version that was read. Look at the code "
            "as it is now, and approve again.",
            page=page,
        )
    if page.review != "accepted":
        raise ProjectError(
            f"{name} is {page.review}, not accepted. Accept the experiment's page before "
            "approving its run."
        )
    if page.state in ("queued", "running"):
        raise ProjectError(
            f"{name} is {page.state}. Cancel that run, or wait for it, before approving another."
        )
    chosen = (backend or "").strip().lower() or _default_backend(project, page, gpu)
    overrides = {
        "backend": chosen,
        "gpu": gpu,
        "timeout_seconds": timeout_seconds,
        "network": network,
    }
    reading = _read(project, page, overrides=overrides)
    # A plan that cannot run anywhere is refused without a backend being made for it.
    choice = _Choice(chosen) if reading.problems else _chosen(chosen, backends)
    try:
        reasons = _refusals(project, reading, choice, _limits(project), _spent_usd(project))
    finally:
        choice.close()
    if reasons:
        raise ProjectError(f"{name} cannot be approved:\n- " + "\n- ".join(reasons))
    plan = reading.plan
    fields = {
        "backend": chosen,
        "gpu": plan.gpu,
        "timeout_seconds": _plain_number(plan.timeout_seconds),
        "network": plan.network,
    }
    noted = project.update_page(page.path, author=HUMAN, fields=fields, base_updated=base_updated)
    # Once the options are on the page it is a new version; the person's is the one before.
    unchanged = noted.updated == page.updated
    project.set_state(
        page.path, "approved", by=HUMAN, base_updated=base_updated if unchanged else None
    )
    with contextlib.suppress(OSError):
        _flag_path(project, name).unlink(missing_ok=True)  # of a run cancelled earlier
    approved = project.page(page.path)
    _mark_testing(project, approved)
    return approved


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


def _runs_dir(project: Project, experiment_id: str) -> Path:
    return project.experiment_dir(experiment_id) / RUNS_DIR


def runs(project: Project, experiment_id: str) -> list[dict[str, Any]]:
    """The run records (``run.json``) of the experiment, by run number.

    A directory without a readable record is left out.
    """
    return [record for _, record in _numbered_runs(project, experiment_id)]


def _run_number(name: str) -> int | None:
    """The number a run directory is named by; ``None`` for any other name."""
    return int(name) if _RUN_NUMBER.fullmatch(name) else None


def _numbered_runs(project: Project, experiment_id: str) -> list[tuple[int, dict[str, Any]]]:
    """``(number of the run directory, its record)``, by number."""
    folder = _runs_dir(project, _experiment_id(project, experiment_id))
    found: list[tuple[int, dict[str, Any]]] = []
    for entry in folder.iterdir() if folder.is_dir() else ():
        number = _run_number(entry.name)
        record = _read_json(entry / RUN_FILE) if number is not None else None
        if number is not None and isinstance(record, dict):
            found.append((number, record))
    return sorted(found, key=lambda item: item[0])


def _recorded_runs(project: Project, experiment_id: str) -> list[int]:
    """The run numbers of the experiment that have a result record in the wiki."""
    named = re.compile(rf"{re.escape(experiment_id)}-run([0-9]{{1,9}})")
    stems = (PurePosixPath(page.path).stem for page in project.pages(kind="result"))
    return [int(match.group(1)) for stem in stems if (match := named.fullmatch(stem))]


def _new_run_dir(project: Project, experiment_id: str) -> tuple[int, Path]:
    """Make the directory of the next run and return its number and path.

    The number is one no run of the experiment has had: neither a directory nor a
    result record carries it.  Run directories may be cleared to save space, and
    a new run must not then replace the record of an old one, which findings
    may rest on.
    """
    folder = _runs_dir(project, experiment_id)
    folder.mkdir(parents=True, exist_ok=True)
    taken = [_run_number(entry.name) or 0 for entry in folder.iterdir()]
    number = max([*taken, *_recorded_runs(project, experiment_id)], default=0)
    while True:
        number += 1
        with contextlib.suppress(FileExistsError):
            (folder / str(number)).mkdir()
            return number, folder / str(number)


@dataclass
class _Attempt:
    """One run in flight: its directory, its record, and how the job ended."""

    number: int
    directory: Path
    record: dict[str, Any]
    #: ``time.monotonic()`` when it began; ``None`` for a run found, not started, here.
    began: float | None = None
    result: JobResult | None = None
    notes: list[str] = field(default_factory=list)

    def save(self) -> None:
        write_json(self.directory / RUN_FILE, self.record)


@dataclass
class _Outcome:
    """What became of a run."""

    status: str
    error: str | None = None
    #: Why it failed: ``exit``, ``metrics``, ``code``, ``patch`` (the code's own, which a
    #: repair can cure), ``command`` (it could not be started), ``timeout``, ``backend``,
    #: ``budget``, ``setup``, ``worker``, ``internal``.
    kind: str | None = None
    #: The run did not get to finish for a reason outside the experiment, and the
    #: experiment is back in the queue.
    requeued: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    #: The metric values as the run wrote them.
    metrics_text: dict[str, str] = field(default_factory=dict)


def _new_record(
    plan: ExperimentPlan, number: int, backend: str, repairs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "experiment": plan.id,
        "number": number,
        "status": "running",
        "backend": backend,
        "handle": None,
        "started": _now(),
        "finished": None,
        "seconds": None,
        "exit_code": None,
        "timed_out": False,
        "metrics": {},
        "artifacts": [],
        "error": None,
        "cost_usd": None,
        "code_digest": None,
        "repairs": [dict(repair) for repair in repairs],
        # Beyond the fields every reader relies on: why it failed, whether it will run
        # again by itself, what the person should know, and where its record page is.
        "error_kind": None,
        "requeued": False,
        "notes": [],
        "result": None,
        "plan": {
            "mode": plan.mode,
            "command": job_command(plan),
            "timeout_seconds": plan.timeout_seconds,
            "gpu": plan.gpu,
            "network": plan.network,
            "packages": list(plan.packages),
        },
        "worker": {"pid": os.getpid(), "host": socket.gethostname()},
    }


# -- what a run left behind ---------------------------------------------------------------


class _Number(str):
    """A number of ``metrics.json``, kept as the text the run wrote."""


class _NotANumber(str):
    """``NaN`` or an infinity, which JSON does not have and results should not."""


def _as_written(value: Any) -> str:
    """A parsed JSON value as text, with its numbers as the run wrote them."""
    if isinstance(value, _Number | _NotANumber):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_as_written(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{json.dumps(k)}: {_as_written(v)}" for k, v in value.items())
        return "{" + inner + "}"
    return json.dumps(value, ensure_ascii=False)


def _shown(value: Any, limit: int = 120) -> str:
    text = _as_written(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _metric(name: str, value: Any) -> tuple[Any, str]:
    """``(value, text as written)`` of one metric; ``ValueError`` says what is wrong."""
    label = json.dumps(name, ensure_ascii=False)
    if isinstance(value, _NotANumber):
        raise ValueError(f"{label} is {value}, which is not a number")
    if isinstance(value, _Number):
        number: Any = float(value) if re.search(r"[.eE]", value) else int(value)
        if isinstance(number, float) and not math.isfinite(number):
            raise ValueError(f"{label} is {value}, which is too large to be a number")
        return number, str(value)
    if isinstance(value, bool):
        return value, "true" if value else "false"
    if isinstance(value, str):
        if len(value) > MAX_METRIC_TEXT_CHARS:
            raise ValueError(
                f"{label} is a text of {len(value)} characters, longer than the "
                f"{MAX_METRIC_TEXT_CHARS} a metric may have ({_shown(value, 60)})"
            )
        return value, value
    kind = {type(None): "null", list: "a list", dict: "an object"}.get(type(value), "not a value")
    raise ValueError(f"{label} is {kind} ({_shown(value)})")


def _read_metrics(directory: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """The metrics of a run and their text as written.  ``ValueError`` says, quoting the
    file, why it does not hold metrics: it is read strictly, because what it says goes
    into the wiki as the result."""
    hint = (
        " It has to be a flat JSON object of names to numbers, true/false or short texts, "
        'such as {"accuracy": 0.91}; tables and curves belong in artifacts/.'
    )
    path = directory / "metrics.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("The run ended without writing metrics.json." + hint)
    if path.stat().st_size > MAX_METRICS_BYTES:
        size = human_size(path.stat().st_size)
        raise ValueError(f"metrics.json is {size}, too large to hold metrics." + hint)
    try:
        text = path.read_bytes().decode("utf-8")
        data = json.loads(text, parse_float=_Number, parse_int=_Number, parse_constant=_NotANumber)
    except UnicodeDecodeError as exc:
        raise ValueError(f"metrics.json is not UTF-8 text ({exc})." + hint) from None
    except (json.JSONDecodeError, RecursionError) as exc:
        start = _clip(text[:80], 80)
        raise ValueError(
            f"metrics.json is not valid JSON ({exc}); it starts with {start!r}." + hint
        ) from None
    if not isinstance(data, dict):
        raise ValueError(f"metrics.json holds {_shown(data)}, not an object." + hint)
    if not data:
        raise ValueError("metrics.json is an empty object: it reports no metric." + hint)
    if len(data) > MAX_METRICS:
        raise ValueError(
            f"metrics.json has {len(data)} entries; at most {MAX_METRICS} are metrics." + hint
        )
    metrics: dict[str, Any] = {}
    written: dict[str, str] = {}
    for name, value in data.items():
        if not name.strip() or len(name) > MAX_METRIC_NAME_CHARS:
            raise ValueError(
                f"metrics.json has the name {_clip(name, 60)!r}, which is empty or longer "
                f"than {MAX_METRIC_NAME_CHARS} characters." + hint
            )
        try:
            metrics[name], written[name] = _metric(name, value)
        except ValueError as exc:
            raise ValueError(f"metrics.json is not flat metrics: {exc}." + hint) from None
    return metrics, written


def _artifacts(directory: Path) -> tuple[list[str], list[str]]:
    """The artifacts of a run (``artifacts/...``, sorted) and notes on what was not kept.

    Only regular files under ``artifacts/`` count.  One above the size limit, or
    beyond the count, is removed from the run directory: the project is meant to
    stay small enough to copy.
    """
    root = directory / "artifacts"
    kept: list[str] = []
    dropped: list[str] = []
    for folder, dirs, names in os.walk(root) if root.is_dir() and not root.is_symlink() else ():
        dirs[:] = sorted(name for name in dirs if not (Path(folder) / name).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            if size > MAX_ARTIFACT_BYTES or len(kept) >= MAX_ARTIFACTS:
                dropped.append(f"{relative} ({human_size(size)})")
                path.unlink(missing_ok=True)
            else:
                kept.append(relative)
    notes = []
    if dropped:
        shown = ", ".join(dropped[:5]) + (f" and {len(dropped) - 5} more" if dropped[5:] else "")
        notes.append(
            f"Not kept (an artifact is at most {human_size(MAX_ARTIFACT_BYTES)}, and a run "
            f"keeps at most {MAX_ARTIFACTS}): {shown}."
        )
    return sorted(kept), notes


def _log_tail(directory: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    """The last lines of a run's log, cleaned of terminal control sequences."""
    try:
        with open(directory / LOG_NAME, "rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 256 * 1024))
            data = stream.read()
    except OSError:
        return []
    text = _CONTROL.sub("", _ANSI.sub("", data.decode("utf-8", errors="replace")))
    tail = [line.rstrip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    return [line[:LOG_LINE_CHARS] for line in tail[-lines:]]


def _last_words(tail: Sequence[str]) -> str:
    """The last line the job itself printed: where a traceback names the error."""
    for line in reversed(tail):
        if not line.startswith(NOTE_PREFIX):
            return _clip(line, 300)
    return ""


def _read_handle(directory: Path) -> str | None:
    try:
        return (directory / HANDLE_NAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# -- the result record --------------------------------------------------------------------


def _fenced(text: str, language: str = "text") -> list[str]:
    """``text`` as a code block whose fence is longer than any run of backticks in it."""
    longest = max((len(ticks) for ticks in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{language}", text.rstrip("\n"), fence]


def _cell(text: Any) -> str:
    return _one_line(text).replace("|", "\\|")


def _inert(text: Any) -> str:
    """Text a job, a page or a file name supplied, as one line that is only text: what
    would start a link, markup or a code span in Markdown is escaped."""
    return _cell(re.sub(r"([\\\[\]<`*])", r"\\\1", _one_line(text)))


def _decimal_form(written: str) -> str:
    """``0.00001`` for ``1e-05``: a reader, and the check of a finding's numbers, look for
    the value in plain digits."""
    if not re.search(r"[eE]", written):
        return ""
    with contextlib.suppress(InvalidOperation, ValueError):
        plain = format(Decimal(written), "f")
        if len(plain) <= 40:
            return f" ({plain})"
    return ""


def _cost_text(record: Mapping[str, Any]) -> str:
    cost = record.get("cost_usd")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        return f"about ${cost:.2f} (an estimate)"
    return "nothing is billed" if record.get("backend") == "local" else "not known"


def _repo_text(record: Mapping[str, Any]) -> str:
    info = record.get("repo") or {}
    text = _inert(info.get("source") or "")
    if info.get("ref"):
        text += f" at {_inert(info['ref'])}"
    if info.get("commit"):
        text += f", commit `{str(info['commit'])[:12]}`"
        if info.get("dirty"):
            text += " plus the changes that were not committed"
    if info.get("patched"):
        text += f", with {repositories.PATCH_NAME} applied"
    skipped = len(record.get("skipped") or []) + int(info.get("skipped_more") or 0)
    text += f"; {skipped} files were left out of the copy" if skipped else ""
    if info.get("method") == "git":
        # Data is often ignored, and a run that does not find it should say why.
        text += "; files that .gitignore excludes are not copied"
    return text


def _facts(record: Mapping[str, Any]) -> list[str]:
    rows = [
        ("Status", record.get("status")),
        ("Started", record.get("started")),
        ("Finished", record.get("finished")),
        ("Duration", _seconds_text(record.get("seconds"))),
        ("Backend", record.get("backend") or "none"),
        ("Exit status", "none" if record.get("exit_code") is None else record["exit_code"]),
        ("Estimated cost", _cost_text(record)),
    ]
    if record.get("handle"):
        rows.append(("Remote job", f"`{record['handle']}`"))
    if record.get("code_digest"):
        rows.append(("Code", f"sha256 `{str(record['code_digest'])[:12]}`"))
    if record.get("repo"):
        rows.append(("Repository", _repo_text(record)))
    return ["| | |", "|---|---|", *(f"| {name} | {_cell(value)} |" for name, value in rows)]


def _metrics_section(record: Mapping[str, Any], directory: Path) -> list[str]:
    metrics = record.get("metrics") or {}
    if not metrics:
        if record.get("status") != "succeeded" and (directory / "metrics.json").is_file():
            return [
                "The run wrote metrics.json, but its values are not reported here: the run "
                "did not finish, so they are not a result."
            ]
        return ["The run reported no metrics."]
    written = record.get("metrics_text") or {}
    lines = ["| metric | value |", "|---|---|"]
    for name, value in metrics.items():
        text = str(written.get(name, json.dumps(value, ensure_ascii=False)))
        lines.append(f"| {_inert(name)} | {_inert(text)}{_decimal_form(text)} |")
    return lines


def _artifact_lines(record: Mapping[str, Any], directory: Path) -> list[str]:
    base = f"{FILES_URL}experiments/{record['experiment']}/{RUNS_DIR}/{record['number']}/"
    lines: list[str] = []
    for path in record.get("artifacts") or []:
        # The name is the job's to choose.  On the page it is one line of link text and
        # nothing else: no line of its own, no bracket that ends the link early.
        name = _CONTROL.sub("", _one_line(path.removeprefix("artifacts/")))
        label = name.replace("\\", "\\\\").replace("[", "(").replace("]", ")")
        if PurePosixPath(path).suffix.lower() in _IMAGE_SUFFIXES:
            lines += [f"![{label}]({base}{quote(path)})", ""]
        else:
            with contextlib.suppress(OSError):
                label_size = human_size((directory / path).stat().st_size)
                lines.append(f"- [{label}]({base}{quote(path)}) ({label_size})")
    return lines


def _repair_lines(record: Mapping[str, Any]) -> list[str]:
    lines: list[str] = [
        "The runner changed the code before this run, to cure the failure of an earlier one.",
        "",
    ]
    repairs = record.get("repairs") or []
    room = MAX_DIFF_CHARS_ON_PAGE // max(1, len(repairs))
    for repair in repairs:
        diff = str(repair.get("diff") or "")
        if len(diff) > room:
            diff = diff[:room] + "\n… (the whole change is in run.json)"
        lines += [
            f"### Repair {repair.get('round')}, after run {repair.get('after_run')}",
            "",
            *_fenced(str(repair.get("reason") or "")),
            "",
            *_fenced(diff, "diff"),
            "",
        ]
    return lines


def _result_body(
    record: Mapping[str, Any], experiment: ResearchPage, directory: Path, tail: Sequence[str]
) -> str:
    """The text of a result page.  Every word of it follows from the run record, the
    run directory and the log; no model has a part in it."""
    status = record["status"]
    took = _seconds_text(record.get("seconds"))
    lines = [
        f"Run {record['number']} of [[{experiment.link}|{record['experiment']}]] "
        f"({_inert(experiment.title)}) {status} after {took} on the "
        f"{_inert(record.get('backend') or 'no')} backend.",
        "",
    ]
    if record.get("error"):
        lines += [*_fenced(str(record["error"])), ""]
    lines += [*_facts(record), "", "## Metrics", "", *_metrics_section(record, directory), ""]
    artifacts = _artifact_lines(record, directory)
    if artifacts:
        lines += ["## Artifacts", "", *artifacts, ""]
    if record.get("notes"):
        lines += ["## Notes", "", *(f"- {_inert(note)}" for note in record["notes"]), ""]
    if record.get("repairs"):
        lines += ["## Repairs", "", *_repair_lines(record)]
    lines += [f"## Log (last {LOG_TAIL_LINES} lines)", ""]
    lines += _fenced("\n".join(tail)) if tail else ["The run printed nothing."]
    return "\n".join(lines) + "\n"


def _result_summary(record: Mapping[str, Any]) -> str:
    status = record["status"]
    if status == "succeeded":
        written = record.get("metrics_text") or {}
        first = list((record.get("metrics") or {}).items())[:4]
        shown = ", ".join(f"{name}={written.get(name, value)}" for name, value in first)
        return _clip(f"succeeded in {_seconds_text(record.get('seconds'))}: {shown}", 200)
    return _clip(f"{status}: {record.get('error') or 'no reason recorded'}", 200)


def _write_result(
    project: Project, experiment: ResearchPage, attempt: _Attempt, tail: Sequence[str]
) -> ResearchPage:
    record = attempt.record
    return project.write_record(
        "result",
        f"{record['experiment']}-run{record['number']}",
        f"{record['experiment']} run {record['number']}: {experiment.title}",
        _result_body(record, experiment, attempt.directory, tail),
        summary=_result_summary(record),
        fields={"state": record["status"], "run": record["number"], "backend": record["backend"]},
        relations={"run_of": [record["experiment"]]},
    )


def _shorten(error: str | None) -> str | None:
    """The reason a run failed, cut to what a record and a page can carry."""
    if error is None or len(error) <= MAX_ERROR_CHARS:
        return error
    return error[:MAX_ERROR_CHARS].rstrip() + "\n… (cut short)"


def _emit(project: Project, type: str, summary: str, **details: Any) -> None:
    """Tell the activity feed; a feed that cannot be written must not end a run."""
    with contextlib.suppress(ProjectError, OSError):
        project.emit(type, summary, agent="runner", **details)


def _finish(
    project: Project, experiment: ResearchPage, attempt: _Attempt, outcome: _Outcome
) -> dict[str, Any]:
    """Settle a run: its record, its result page, the event.  Returns the record.

    The record is written first and by itself: whatever fails afterwards, the run
    no longer reads as running.
    """
    record, result = attempt.record, attempt.result
    artifacts, dropped = _artifacts(attempt.directory)
    seconds = result.seconds if result is not None and result.seconds else None
    if seconds is None and attempt.began is not None:
        seconds = time.monotonic() - attempt.began
    record.update(
        status=outcome.status,
        finished=_now(),
        seconds=round(seconds, 3) if seconds is not None else None,
        exit_code=result.returncode if result is not None else None,
        timed_out=bool(result is not None and result.timed_out),
        metrics=outcome.metrics,
        artifacts=artifacts,
        error=_shorten(outcome.error),
        cost_usd=result.cost_usd if result is not None else None,
        handle=record.get("handle") or (result.handle if result else None),
        error_kind=outcome.kind,
        requeued=outcome.requeued,
        notes=list(dict.fromkeys([*attempt.notes, *(result.notes if result else []), *dropped])),
    )
    # A record found on disk may be older than these fields, or written by hand.
    record.setdefault("repairs", [])
    record.setdefault("result", None)
    if outcome.metrics_text:
        record["metrics_text"] = outcome.metrics_text
    attempt.save()
    refs = [experiment.path]
    if not outcome.requeued:
        # A run that only ended because its worker did is no outcome of the experiment;
        # the run that follows it will have the record.
        try:
            page = _write_result(project, experiment, attempt, _log_tail(attempt.directory))
        except ProjectError as exc:
            record["notes"].append(f"The result page could not be written: {exc}")
        else:
            record["result"] = page.path
            refs.append(page.path)
        attempt.save()
    what = f"Run {record['number']} of experiment {record['experiment']}"
    took = _seconds_text(record["seconds"])
    why = f": {_clip(outcome.error, 200)}" if outcome.error else ""
    ending = "." if not why or why[-1] not in ".!?…" else ""
    _emit(
        project,
        "run.finished",
        f"{what} {outcome.status} after {took}{why}{ending}",
        refs=refs,
        data={
            "experiment": record["experiment"],
            "run": record["number"],
            "status": outcome.status,
            "seconds": record["seconds"],
            "error": outcome.error,
            "requeued": outcome.requeued,
            "result": record["result"],
        },
    )
    return record


# -- how a job ended ----------------------------------------------------------------------


def _judge(
    result: JobResult, directory: Path, plan: ExperimentPlan, *, cancelled_by_person: bool
) -> _Outcome:
    """Read a job's result the way the compute contract says: the backend's own failure
    first, then a stop or the clock, then the exit status, then what it wrote."""
    tail = _log_tail(directory)
    if result.error:
        return _Outcome("failed", result.error, kind="backend")
    if result.cancelled:
        if cancelled_by_person:
            return _Outcome("cancelled", "The run was cancelled at the person's request.")
        return _Outcome(
            "cancelled",
            "The run was stopped with the worker; it starts again when the work resumes.",
            requeued=True,
        )
    if result.timed_out:
        limit = plan.timeout_seconds
        return _Outcome(
            "failed", f"The run was stopped after its time limit of {limit:g} s.", kind="timeout"
        )
    if result.returncode is None:
        lost = "The backend lost track of the job: it reported no exit status."
        return _Outcome("failed", lost, kind="backend")
    if result.returncode != 0:
        cloning = repositories.clone_failure(result.returncode) if _clones(plan) else None
        if cloning is not None:
            said = " ".join(repositories.clone_lines(tail)[-3:])
            return _Outcome("failed", _one_line(f"{cloning[1]} {said}"), kind=cloning[0])
        said = _last_words(tail) or (tail[-1].removeprefix(NOTE_PREFIX) if tail else "")
        why = f"The command exited with status {result.returncode}" + (f": {said}" if said else ".")
        # 126 and 127 are what a shell gives a command it could not start.  The command is
        # the plan's, not the code's: no corrected file changes it.
        return _Outcome("failed", why, "command" if result.returncode in (126, 127) else "exit")
    try:
        metrics, written = _read_metrics(directory)
    except ValueError as exc:
        return _Outcome("failed", str(exc), kind="metrics")
    return _Outcome("succeeded", metrics=metrics, metrics_text=written)


def _clones(plan: ExperimentPlan) -> bool:
    return plan.mode == "repo" and bool(plan.repo) and repositories.is_url(str(plan.repo))


def _final_state(record: Mapping[str, Any]) -> tuple[str, str | None]:
    """The state a settled run leaves its experiment in, and the reason to note."""
    if record.get("requeued"):
        return "queued", record.get("error")
    if record["status"] == "succeeded":
        return "succeeded", None
    return ("cancelled" if record["status"] == "cancelled" else "failed"), record.get("error")


def _move(project: Project, path: str, state: str, reason: str | None = None) -> None:
    """Move the experiment along its run.  When the page is no longer where the run left
    it (the person changed or removed it meanwhile), theirs stands."""
    with contextlib.suppress(ProjectError):
        project.set_state(path, state, by=SYSTEM, reason=_clip(reason, 300) if reason else None)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


class _RunLock(WikiLock):
    """The kernel lock of ``memory.lock`` on a file of the experiment.

    Whoever runs the experiment holds it, so that no two processes run it at
    once and a run whose process is gone can be told from one that is alive: the
    kernel drops the lock with the process, however it died.
    """

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        super().__init__(directory, wait_seconds=0.0, poll=0.05)
        self.path = directory / RUN_LOCK


class _Setup(Exception):
    """A run could not be started; ``kind`` is the kind of failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _backend_of(compute: Any, plan: ExperimentPlan) -> Any:
    """``compute`` is a backend, or a function that chooses one for a plan."""
    return compute if hasattr(compute, "run_job") else compute(plan)


def _snapshot(source: Path, dest: Path) -> str:
    """Copy the experiment's code into the run directory and return its digest.

    A run starts from this copy: it is what ran, whatever is written to the code
    directory while the job is under way.
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name, path in _code_paths(source):
        copied += path.stat().st_size
        if copied > MAX_CODE_BYTES:
            raise _Setup(
                "setup",
                f"The code directory {source} holds more than {human_size(MAX_CODE_BYTES)}. "
                "It is copied for every run and holds code: keep data out of it.",
            )
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest / name)
    return _digest(_code_paths(dest))


def _unified_diff(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    """What changed between two versions of the code, as one unified diff."""
    pieces: list[str] = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old == new:
            continue
        pieces += difflib.unified_diff(
            (old or "").splitlines(keepends=True),
            (new or "").splitlines(keepends=True),
            fromfile=f"a/{name}" if old is not None else "/dev/null",
            tofile=f"b/{name}" if new is not None else "/dev/null",
        )
    text = "".join(line if line.endswith("\n") else line + "\n" for line in pieces)
    return text if len(text) <= MAX_DIFF_CHARS else text[:MAX_DIFF_CHARS] + "\n… (cut short)\n"


class _Session:
    """One call of :func:`run`: the attempts at one approved experiment.

    The caller holds the experiment's run lock for as long as this lives.
    """

    def __init__(
        self,
        project: Project,
        reading: _Reading,
        backend: Any,
        *,
        should_stop: Callable[[], bool] | None,
        repair: Callable[..., Mapping[str, str] | None] | None,
        on_log: Callable[[str], None] | None,
        limits: _Limits,
        usd_is_for_this_call: bool = False,
    ) -> None:
        self.project = project
        self.page = reading.page
        self.plan = reading.plan
        self.reading = reading
        self.backend = backend
        self.backend_name = str(getattr(backend, "name", "") or "")
        self.limits = limits
        #: Whether ``limits.usd`` is what this call may spend (its caller keeps the
        #: project's accounts) rather than the project's total.
        self._usd_is_for_this_call = usd_is_for_this_call
        #: What the runs settled in this call have cost.
        self._cost = 0.0
        self._user_stop = should_stop
        self._user_log = on_log
        self._repair = repair
        self._repairs: list[dict[str, Any]] = []
        self._rounds = 0
        self._current: _Attempt | None = None
        self._stopped = False
        #: Why a failure that could have been repaired was not: ``cancel`` or ``stop``.
        self._interrupted: str | None = None
        self._flag = _flag_path(project, self.plan.id)
        self._guard = threading.Lock()
        self._next_handle_check = 0.0
        self._pending: deque[str] = deque(maxlen=LOG_EVENT_LINES)
        self._last_log_event = -math.inf

    # -- stopping ----------------------------------------------------------------

    def _stop_reason(self) -> str | None:
        """``cancel`` when the person cancelled this experiment, ``stop`` when the worker
        was asked to stop, else ``None``."""
        if self._flag.exists():
            return "cancel"
        if not self._stopped and self._user_stop is not None and self._user_stop():
            self._stopped = True
        return "stop" if self._stopped else None

    def _should_stop(self) -> bool:
        # Asked about once a second whether or not the job prints: the one place to
        # notice the handle of a remote job that stays silent.
        self._watch_handle()
        return self._stop_reason() is not None

    def _stopped_outcome(self) -> _Outcome:
        result = JobResult(returncode=None, cancelled=True)
        return _judge(
            result,
            self._current.directory if self._current else Path(),
            self.plan,
            cancelled_by_person=self._flag.exists(),
        )

    # -- while the job runs ------------------------------------------------------

    def _watch_handle(self) -> None:
        """Copy the handle of a remote job into ``run.json`` as soon as there is one, so
        that a worker that dies leaves behind what is needed to pick the job up."""
        attempt = self._current
        now = time.monotonic()
        if attempt is None or attempt.record.get("handle") or now < self._next_handle_check:
            return
        with self._guard:
            self._next_handle_check = now + HANDLE_CHECK_SECONDS
            handle = _read_handle(attempt.directory)
            if handle and not attempt.record.get("handle"):
                attempt.record["handle"] = handle
                attempt.save()

    def _on_log(self, line: str) -> None:
        if self._user_log is not None:
            self._user_log(line)
        self._watch_handle()
        attempt = self._current
        if attempt is None:
            return
        self._pending.append(line[:LOG_LINE_CHARS])
        now = time.monotonic()
        if now - self._last_log_event < LOG_EVENT_SECONDS:
            return
        self._last_log_event = now
        lines = list(self._pending)
        self._pending.clear()
        _emit(
            self.project,
            "run.log",
            f"{self.plan.id} run {attempt.number}: {_clip(lines[-1], 160)}",
            refs=[self.page.path],
            data={"experiment": self.plan.id, "run": attempt.number, "lines": lines},
        )

    # -- one attempt -------------------------------------------------------------

    def _open(self) -> _Attempt:
        number, directory = _new_run_dir(self.project, self.plan.id)
        record = _new_record(self.plan, number, self.backend_name, self._repairs)
        record["code_digest"] = code_digest(self.project, self.plan.id)
        attempt = _Attempt(number, directory, record, began=time.monotonic())
        attempt.notes.extend(self.reading.notes)
        attempt.save()
        self._current = attempt
        self._pending.clear()
        self._last_log_event = -math.inf
        _move(self.project, self.page.path, "running")
        _emit(
            self.project,
            "run.started",
            f"Run {number} of experiment {self.plan.id} started on the "
            f"{self.backend_name or 'compute'} backend.",
            refs=[self.page.path],
            data={"experiment": self.plan.id, "run": number, "backend": self.backend_name},
        )
        return attempt

    def _refused(self) -> _Setup | None:
        """Why this attempt cannot start at all, checked before anything is copied."""
        plan = self.plan
        if self.reading.problems:
            return _Setup("setup", "The plan cannot be run: " + " ".join(self.reading.problems))
        errors = split_problems(check_code(self.project, plan.id))[0]
        if errors:
            return _Setup("code", "The code cannot be run: " + " ".join(errors))
        if plan.backend and self.backend_name and plan.backend != self.backend_name:
            return _Setup(
                "setup",
                f"{plan.id} was approved for the {plan.backend} backend, but this run was "
                f"given the {self.backend_name} backend. Run it on the backend it was "
                "approved for, or approve it again for the other one.",
            )
        over = _budget_refusals(
            plan, self.backend, self.limits, self._spent(), remainder=self._usd_is_for_this_call
        )
        if over:
            _emit(
                self.project,
                "budget.warning",
                f"Experiment {plan.id} was not run: {_clip(over[0], 300)}",
                refs=[self.page.path],
                data={"experiment": plan.id, "reasons": over},
            )
            return _Setup("budget", " ".join(over))
        return None

    def _spent(self) -> float:
        """What counts against the cost limit before the next attempt.

        The limit is the project's total, so it is what all its recorded runs
        have cost.  When the caller named the amount this call may spend, it has
        done that sum already, and only what this call's own runs cost (a run
        picked up, the attempts before a repair) is taken off.
        """
        return self._cost if self._usd_is_for_this_call else _spent_usd(self.project)

    def _count(self, record: Mapping[str, Any]) -> None:
        cost = record.get("cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool) and 0 < cost < math.inf:
            self._cost += cost

    def _prepare(self, attempt: _Attempt) -> tuple[Path, JobSpec]:
        """The directory the job is given and what it is asked to do; ``_Setup`` when the
        run cannot be made ready."""
        plan, directory, record = self.plan, attempt.directory, attempt.record
        code = directory / CODE_DIR
        record["code_digest"] = _snapshot(code_dir(self.project, plan.id), code)
        spec = job_spec(plan)
        if plan.mode != "repo":
            return code, spec
        files = {name: path.read_bytes() for name, path in _code_paths(code)}
        patch = files.pop(repositories.PATCH_NAME, None)
        patch_text = patch.decode("utf-8", errors="replace") if patch is not None else None
        copy = directory / REPO_COPY
        record["skipped"] = []
        try:
            if _clones(plan):
                # The record names the repository; a token in its URL stays in the settings.
                shown = repositories.display_url(str(plan.repo))
                made = repositories.RepoCopy(shown, "clone", ref=plan.repo_ref)
                record["repo"] = made.to_dict()
                command = repositories.clone_source(
                    copy,
                    url=str(plan.repo),
                    ref=plan.repo_ref,
                    patch=patch_text,
                    files=files,
                    command=spec.command,
                )
                return copy, replaced(spec, command=command)
            made = repositories.copy_local(
                str(plan.repo),
                copy,
                ref=plan.repo_ref,
                should_stop=self._should_stop,
                exclude=[self.project.root],
            )
            record["repo"] = made.to_dict()
            record["skipped"] = made.skipped
            if patch_text is not None:
                record["repo"]["patched"] = repositories.apply_patch(copy, patch_text)
            repositories.add_files(copy, files)
        except repositories.PatchError as exc:
            raise _Setup("patch", str(exc)) from None
        except repositories.RepoError as exc:
            raise _Setup("setup", str(exc)) from None
        return copy, spec

    def _execute(self, attempt: _Attempt) -> _Outcome:
        refused = self._refused()
        if refused is not None:
            return _Outcome("failed", str(refused), kind=refused.kind)
        try:
            try:
                source, spec = self._prepare(attempt)
            except _Setup as exc:
                return _Outcome("failed", str(exc), kind=exc.kind)
            except JobStopped:
                return self._stopped_outcome()
            except OSError as exc:
                # A full disk, a file that cannot be read: this run cannot start, which is
                # for the person to read on its record, not something that ends the worker.
                why = (
                    f"The files of the run could not be put in place: {exc}. Check that "
                    f"{attempt.directory} can be written and that the code can be read."
                )
                return _Outcome("failed", why, kind="setup")
            finally:
                attempt.save()
            if self._stop_reason():
                return self._stopped_outcome()
            attempt.result = self.backend.run_job(
                spec,
                source_dir=source,
                run_dir=attempt.directory,
                on_log=self._on_log,
                should_stop=self._should_stop,
            )
        finally:
            shutil.rmtree(attempt.directory / REPO_COPY, ignore_errors=True)
        return _judge(
            attempt.result,
            attempt.directory,
            self.plan,
            cancelled_by_person=self._flag.exists(),
        )

    def _attempt(self) -> dict[str, Any]:
        attempt = self._open()
        outcome = self._execute(attempt)
        record = _finish(self.project, self.page, attempt, outcome)
        self._current = None
        self._count(record)
        return record

    # -- repairs -----------------------------------------------------------------

    def _repairable(self, record: Mapping[str, Any]) -> bool:
        curable = (
            self._repair is not None
            and record["status"] == "failed"
            and record.get("error_kind") in _REPAIRABLE
            and self._rounds < self.limits.repair_rounds
        )
        if curable:
            self._interrupted = self._stop_reason()
        return curable and self._interrupted is None

    def _note(self, record: dict[str, Any], note: str) -> None:
        record["notes"].append(note)
        write_json(_runs_dir(self.project, self.plan.id) / str(record["number"]) / RUN_FILE, record)

    def _repaired(self, record: dict[str, Any]) -> bool:
        """Ask for corrected files after ``record`` failed and write them as the runner.
        True when the code changed, so that running again can end differently."""
        assert self._repair is not None
        self._rounds += 1
        directory = _runs_dir(self.project, self.plan.id) / str(record["number"])
        before = code_files(self.project, self.plan.id)
        try:
            files = self._repair(self.plan, dict(record), "\n".join(_log_tail(directory, 200)))
            if not files:
                return False
            write_code(self.project, self.plan.id, files, author=_RUNNER)
        except Exception as exc:
            if stops_the_run(exc):
                raise
            # A repairer that fails is no repair; the failed run stands as it is.
            self._note(record, f"The repair could not be made: {type(exc).__name__}: {exc}")
            return False
        diff = _unified_diff(before, code_files(self.project, self.plan.id))
        if not diff:
            self._note(record, "The repair changed nothing, so the run was not repeated.")
            return False
        self._repairs.append(
            {
                "round": self._rounds,
                "reason": record.get("error") or "",
                "diff": diff,
                "after_run": record["number"],
                "files": sorted(files),
            }
        )
        return True

    # -- the whole of it ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        except BaseException as exc:
            self._abandon(exc)
            raise

    def _run(self) -> dict[str, Any]:
        settled = _settle_stale(
            self.project,
            self.page,
            self.plan,
            lambda record: self.backend,
            should_stop=self._should_stop,
            on_log=self._user_log,
        )
        for record in settled:
            self._count(record)
        page = self.project.page(self.page.path)
        if page.state not in ("approved", "queued"):
            if settled:
                return settled[-1]
            raise ProjectError(_not_runnable(page))
        _move(self.project, page.path, "queued")
        _mark_testing(self.project, page)
        record = self._attempt()
        while self._repairable(record) and self._repaired(record):
            record = self._attempt()
        self._leave(record)
        return record

    def _leave(self, record: Mapping[str, Any]) -> None:
        """Put the page in the state the last run leaves it in."""
        state, reason = _final_state(record)
        if self._interrupted == "cancel":
            state, reason = "cancelled", "Cancelled at the person's request."
        elif self._interrupted == "stop":
            # Stopped between a failure and its repair: the repair is still owed.
            state, reason = "queued", "The worker was stopped before the failure was repaired."
        _move(self.project, self.page.path, state, reason)
        if state != "queued":
            with contextlib.suppress(OSError):
                self._flag.unlink(missing_ok=True)

    def _abandon(self, exc: BaseException) -> None:
        """Settle what an exception left in flight: neither ``run.json`` nor the page may
        go on reading ``running`` once nothing runs."""
        # A stop (an interrupt, the process being told to exit) or a backend that cannot
        # work is not the experiment's failure: it goes back to the queue.  Anything else
        # is a failure someone has to look at.
        requeue = stops_the_run(exc) or not isinstance(exc, Exception)
        said = f"{type(exc).__name__}: {exc}"
        attempt = self._current
        if attempt is not None and attempt.record.get("status") == "running":
            outcome = self._abandoned(exc, requeue, said)
            try:
                _finish(self.project, self.page, attempt, outcome)
            except Exception:
                # Even that could not be written in full: the bare minimum, then.
                attempt.record.update(status=outcome.status, error=outcome.error, finished=_now())
                with contextlib.suppress(Exception):
                    attempt.save()
            shutil.rmtree(attempt.directory / REPO_COPY, ignore_errors=True)
        with contextlib.suppress(Exception):
            if self.project.page(self.page.path).state == "running":
                state = "queued" if requeue else "failed"
                _move(self.project, self.page.path, state, None if requeue else said)

    @staticmethod
    def _abandoned(exc: BaseException, requeue: bool, said: str) -> _Outcome:
        if getattr(exc, "unusable", False) is True:
            return _Outcome(
                "failed", f"The compute backend cannot be used: {exc}", "backend", requeued=True
            )
        if requeue:
            return _Outcome(
                "cancelled",
                "The run was stopped with the worker; it starts again when the work resumes.",
                requeued=True,
            )
        return _Outcome(
            "failed", f"The run was ended by an error of the pipeline: {said}", "internal"
        )


def _not_runnable(page: ResearchPage) -> str:
    if page.review != "accepted":
        return (
            f"{page.id} is {page.review}, not accepted: only an accepted and approved "
            "experiment runs. Accept its page, then approve its run."
        )
    return (
        f"{page.id} is {page.state}; only an approved experiment runs, and one approval is "
        "one run. Approve it (`paper-orchestra research approve`, or the Experiments view) "
        "to run it."
    )


def run(
    project: Project,
    ref: str,
    *,
    compute: Any,
    should_stop: Callable[[], bool] | None = None,
    repair: Callable[[ExperimentPlan, dict[str, Any], str], Mapping[str, str] | None] | None = None,
    on_log: Callable[[str], None] | None = None,
    limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one approved experiment and return the record of its last run.

    The page moves ``approved -> queued -> running`` and ends ``succeeded``,
    ``failed`` or ``cancelled``; the hypotheses it tests move to ``testing``.
    Each run gets ``experiments/E<n>/runs/<k>/`` with ``run.json`` written at
    once (status ``running``) and again when the handle of a remote job is known
    and when the run ends; a copy of the code that ran; the log; ``metrics.json``
    and ``artifacts/`` as the job wrote them.  When a run ends, its result page
    ``wiki/results/E<n>-run<k>.md`` is written from those.  Events:
    ``experiment.state``, ``run.started``, ``run.log`` (at most one every two
    seconds), ``run.finished``.

    ``compute`` is a ``ComputeBackend`` or a function ``(plan) -> ComputeBackend``.
    ``on_log`` gets every line of output.  A run fails when the backend fails,
    when it exceeds its time limit (the page's, capped at the project's), exits
    with a non-zero status, or does not leave a valid ``metrics.json``.  When it
    failed for an error of its own code and ``repair(plan, run_record, log_tail)``
    returns corrected files (``{name: text}``), they are written as the runner,
    recorded with their diff under ``repairs`` of the next run, and the
    experiment runs again as that next run; at most ``limits.repair_rounds``
    times.

    ``limits`` overrides ``experiment_seconds``, ``experiment_usd`` and
    ``repair_rounds`` of the project's settings for this call.  The time limit
    of a run is capped at ``experiment_seconds``.  ``experiment_usd`` is the
    project's total for all its runs: a run is refused (``error_kind``
    ``budget``, with the numbers) when its estimate, or what it could cost at
    its time limit, does not fit into what is left of it, which is worked out
    from the cost of the runs on record.  A caller that keeps the project's
    accounts itself passes what is left as ``limits["experiment_usd"]``: an
    amount given this way is what this call may spend, on all its attempts
    together, and nothing recorded earlier is taken off it again.

    Two things end a run early, and they differ.  :func:`request_cancel` is the
    person's decision: the run and the experiment end ``cancelled``.
    ``should_stop`` turning true, an interrupt, or a backend that cannot be used
    mean that the worker stops: the run is recorded (``cancelled`` or ``failed``,
    with ``requeued`` true, and without a result page) and the experiment goes
    back to ``queued``, so that the same call resumes it later.

    Raises ``ProjectError`` when the experiment is not accepted and approved (or
    queued), or is being run by another process.  Whatever else is raised on the
    way (``KeyboardInterrupt``, ``BackendUnusable``, an error of a callback) is
    raised again after the run and the page have been settled: neither is ever
    left at ``running``.
    """
    page = _experiment(project, ref)
    if page.review != "accepted" or page.state not in ("approved", "queued", "running"):
        raise ProjectError(_not_runnable(page))
    reading = _read(project, page, limits=limits, cap=True)
    backend = _backend_of(compute, reading.plan)
    with _RunLock(project.experiment_dir(str(page.id))) as lock:
        if not lock.acquired:
            raise ProjectError(
                f"{page.id} cannot be run: {lock.error}."
                if lock.error
                else f"{page.id} is already being run by another process. Wait for that "
                "run, or cancel it."
            )
        session = _Session(
            project,
            reading,
            backend,
            should_stop=should_stop,
            repair=repair,
            on_log=on_log,
            limits=_limits(project, limits),
            usd_is_for_this_call=limits is not None and "experiment_usd" in limits,
        )
        return session.run()


# ---------------------------------------------------------------------------
# Cancelling and recovering
# ---------------------------------------------------------------------------


def _worker_deaths_before(project: Project, experiment_id: str, number: int) -> int:
    """How many runs right before run ``number`` ended because their worker died."""
    deaths = 0
    for earlier, record in reversed(_numbered_runs(project, experiment_id)):
        if earlier >= number:
            continue
        if record.get("error_kind") != "worker":
            break
        deaths += 1
    return deaths


def _can_reattach(backend: Any) -> bool:
    try:
        return bool((backend.describe().get("jobs") or {}).get("reattach"))
    except Exception:  # a backend that cannot say so cannot do it
        return False


def _pick_up(
    project: Project,
    plan: ExperimentPlan,
    attempt: _Attempt,
    backend_for: Callable[[Mapping[str, Any]], Any],
    should_stop: Callable[[], bool] | None,
    on_log: Callable[[str], None] | None,
) -> _Outcome | None:
    """Follow the remote job of a run whose worker died to its end, where that can be
    done; ``None`` when there is nothing to pick up.

    The person's cancel ends a job that is followed here as it ends one that was
    started here: it may run, and be billed, for hours yet.
    """
    handle = attempt.record.get("handle") or _read_handle(attempt.directory)
    if not handle:
        return None
    attempt.record["handle"] = handle
    flag = _flag_path(project, plan.id)

    def stop() -> bool:
        return flag.exists() or (should_stop is not None and should_stop())

    try:
        backend = backend_for(attempt.record)
        if backend is None or not _can_reattach(backend):
            return None
        attempt.result = backend.attach(
            handle, run_dir=attempt.directory, on_log=on_log, should_stop=stop
        )
    except ComputeError as exc:
        if exc.unusable:
            raise
        return _Outcome("failed", f"{_WORKER_STOPPED} {exc}", kind="worker")
    result = attempt.result
    if result.error:
        return _Outcome("failed", f"{_WORKER_STOPPED} {result.error}", kind="worker")
    return _judge(result, attempt.directory, plan, cancelled_by_person=flag.exists())


def _settle_stale(
    project: Project,
    page: ResearchPage,
    plan: ExperimentPlan,
    backend_for: Callable[[Mapping[str, Any]], Any],
    *,
    should_stop: Callable[[], bool] | None = None,
    on_log: Callable[[str], None] | None = None,
    cancel: bool = False,
) -> list[dict[str, Any]]:
    """Settle the runs of an experiment that read ``running`` although nothing runs them.

    The caller holds the experiment's run lock, which is what says that nothing
    does.  A remote job that outlived its worker is followed to its end; any
    other run is failed with "the worker stopped", and the experiment goes back
    to the queue unless that keeps happening.  With ``cancel`` the person has
    cancelled the experiment: the runs end ``cancelled`` instead.
    """
    settled: list[dict[str, Any]] = []
    for number, record in _numbered_runs(project, plan.id):
        if record.get("status") != "running":
            continue
        # The directory says which run it is, whatever a record edited by hand says.
        record.update(experiment=plan.id, number=number)
        attempt = _Attempt(number, _runs_dir(project, plan.id) / str(number), record)
        if cancel:
            _stop_remote(attempt, backend_for)
            outcome: _Outcome | None = _Outcome(
                "cancelled",
                "The run was cancelled at the person's request, after its worker had stopped.",
            )
        else:
            outcome = _pick_up(project, plan, attempt, backend_for, should_stop, on_log)
        if outcome is None:
            outcome = _Outcome("failed", _WORKER_STOPPED, kind="worker")
        if outcome.kind == "worker":
            deaths = _worker_deaths_before(project, plan.id, number) + 1
            outcome.requeued = deaths < MAX_WORKER_DEATHS
            if not outcome.requeued:
                outcome.error = (
                    f"{outcome.error} The worker has now stopped during {deaths} runs of this "
                    "experiment in a row, so it is not started again by itself. Approve it "
                    "again to retry."
                )
        settled.append(_finish(project, page, attempt, outcome))
    current = project.page(page.path)
    if cancel or current.state not in ("queued", "running"):
        return settled
    if settled and not settled[-1].get("requeued"):
        _move(project, page.path, *_final_state(settled[-1]))
        with contextlib.suppress(OSError):
            # The experiment is over: a cancel that was asked for has nothing left to end.
            _flag_path(project, plan.id).unlink(missing_ok=True)
    elif current.state == "running":
        reason = settled[-1].get("error") if settled else "The worker stopped before the run began."
        _move(project, page.path, "queued", reason)
    return settled


def _stop_remote(attempt: _Attempt, backend_for: Callable[[Mapping[str, Any]], Any]) -> None:
    """End the remote job of a cancelled run whose worker is gone, where there is one:
    left alone it would run, and bill, until its own time limit."""
    handle = attempt.record.get("handle") or _read_handle(attempt.directory)
    if not handle:
        return
    with contextlib.suppress(Exception):
        backend = backend_for(attempt.record)
        if backend is not None and _can_reattach(backend):
            backend.attach(handle, run_dir=attempt.directory, should_stop=lambda: True)


class _Backends:
    """The backends recovery needs, built once each and closed afterwards."""

    def __init__(self, compute: Any) -> None:
        self._compute = compute
        self._built: dict[str, _Choice] = {}
        #: The plan of the experiment being settled, for a ``compute`` that chooses by plan.
        self.plan: ExperimentPlan | None = None

    def for_record(self, record: Mapping[str, Any]) -> Any:
        if self._compute is not None:
            return _backend_of(self._compute, self.plan)
        name = str(record.get("backend") or "")
        if name not in self._built:
            self._built[name] = _build(name)
        return self._built[name].backend

    def close(self) -> None:
        for choice in self._built.values():
            choice.close()


def request_cancel(project: Project, ref: str) -> None:
    """The person cancels the experiment's run.

    A flag file is left for the process that runs the experiment, which looks
    for it about once a second, ends the job and records the run and the
    experiment as ``cancelled``.  When no process is running it (it waits to
    run, or its worker died), it is cancelled here and now.  An experiment that
    is neither approved, queued nor running has nothing to cancel, and nothing
    happens.
    """
    page = _experiment(project, ref)
    if page.state not in ("approved", "queued", "running"):
        return
    directory = project.experiment_dir(str(page.id))
    directory.mkdir(parents=True, exist_ok=True)
    flag = directory / CANCEL_FLAG
    write_text(flag, _now() + "\n")
    with _RunLock(directory) as lock:
        if not lock.acquired:
            return  # whoever holds the lock sees the flag
        backends = _Backends(None)
        try:
            # Whoever ran it may have finished between the look at the page and now.
            page = _experiment(project, page.path)
            if page.state not in ("approved", "queued", "running"):
                return
            plan = _read(project, page, cap=True).plan
            _settle_stale(project, page, plan, backends.for_record, cancel=True)
            project.set_state(
                page.path, "cancelled", by=HUMAN, reason="Cancelled at the person's request."
            )
        finally:
            backends.close()
            flag.unlink(missing_ok=True)


def recover(
    project: Project,
    *,
    compute: Any = None,
    should_stop: Callable[[], bool] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Settle the runs that a worker which died left at ``running``; returns their records.

    Call it before running experiments (the engine does, when its run phase
    starts).  For every experiment that is queued or running and that no live
    process is running: a remote job that outlived the worker is picked up with
    ``ComputeBackend.attach`` and followed to its end (this blocks for as long as
    the job takes, and ``should_stop`` and ``on_log`` work as in :func:`run`);
    any other run is recorded as failed because the worker stopped.  The
    experiment then goes back to ``queued``, to be run again, unless
    ``MAX_WORKER_DEATHS`` runs in a row ended that way: then it is ``failed`` and
    waits for the person.  A page left at ``running`` without any run goes back
    to ``queued``.

    ``compute`` is what :func:`run` takes; without it the backend named in a
    run's record is built with its defaults when a remote job has to be picked
    up.  Experiments that a live process is running are left alone.
    """
    settled: list[dict[str, Any]] = []
    backends = _Backends(compute)
    try:
        for page in project.pages(kind="experiment", state=("queued", "running")):
            if not page.id:
                continue
            with _RunLock(project.experiment_dir(page.id)) as lock:
                if not lock.acquired:
                    continue
                backends.plan = _read(project, page, cap=True).plan
                settled += _settle_stale(
                    project,
                    page,
                    backends.plan,
                    backends.for_record,
                    should_stop=should_stop,
                    on_log=on_log,
                )
    finally:
        backends.close()
    return settled


# ---------------------------------------------------------------------------
# The numbers of a finding
# ---------------------------------------------------------------------------


class _Evidence:
    """The numbers of one record, source or input, and whether a given number is among
    them: as written, rounded the way a person rounds when quoting, or as a percentage."""

    def __init__(self, path: str, text: str) -> None:
        self.path = path
        self._exact = numbers_in(text)
        self._values = sorted({float(number) for number in self._exact})

    def has(self, number: str) -> bool:
        if number in self._exact:
            return True
        try:
            value = Decimal(number)
        except InvalidOperation:
            return False
        return any(self._rounds_to(scaled) for scaled in (value, value / 100, value * 100))

    def _rounds_to(self, value: Decimal) -> bool:
        """Whether a number here, written with more digits, rounds to ``value``."""
        value = value.normalize()
        places = max(0, -int(value.as_tuple().exponent))
        half = Decimal(5).scaleb(-(places + 1))
        low = bisect.bisect_left(self._values, float(value - half) - 1e-12)
        for candidate in self._values[low:]:
            if candidate > float(value + half) + 1e-12:
                break
            exact = Decimal(repr(candidate))
            more_digits = -int(exact.normalize().as_tuple().exponent) > places
            if more_digits and abs(exact - value) <= half:
                return True
        return False


def _input_texts(project: Project) -> Iterator[tuple[str, str]]:
    """``(path, text)`` of the text files the person gave the project."""
    root = project.inputs_dir
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not (Path(folder) / name).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST_FILE or path.is_symlink() or not path.is_file():
                continue  # the manifest holds sizes and digests, not the person's material
            with contextlib.suppress(OSError, UnicodeDecodeError):
                if path.stat().st_size <= MAX_EVIDENCE_BYTES:
                    yield f"{INPUTS_DIR}/{relative}", path.read_bytes().decode("utf-8")


def _evidence(project: Project) -> list[_Evidence]:
    """Where a result may come from: accepted result and source records, and the inputs."""
    records = project.pages(kind=("result", "source"), review="accepted")
    found = [_Evidence(page.path, f"{page.title}\n{page.summary}\n{page.body}") for page in records]
    found.sort(key=lambda evidence: not evidence.path.startswith("wiki/results/"))
    return found + [_Evidence(path, text) for path, text in _input_texts(project)]


def number_check(project: Project, page: ResearchPage | str) -> list[dict[str, Any]]:
    """Where each number in a finding comes from.

    For every number in the page's text that is specific enough to be a result
    (the notion of ``memory.leakscan``: two or more significant digits; not a
    small count, a round figure or a year), in the order they appear and each
    once::

        {"number": "2.37", "found_in": ["wiki/results/E1-run1.md"]}

    ``number`` is the number as the page writes it.  ``found_in`` lists the
    accepted result records, accepted source records and text files under
    ``inputs/`` that contain it: as written, rounded to the digits the page
    gives (0.8765 is quoted as 0.88), or as a percentage of a fraction.  An
    empty list means that nothing on record supports the number.  ``page`` is a
    page or a reference to one.
    """
    if not isinstance(page, ResearchPage):
        page = project.page(str(page))
    evidence = _evidence(project)
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in NUMBER.findall(page.body):
        readings = number_readings(raw)
        number, parts = readings[0], readings[1:]
        if number in seen or not is_identifying(number):
            continue
        seen.add(number)
        found = [
            source.path
            for source in evidence
            # `12,345` is one number, or the two items of a list.
            if source.has(number) or (parts and all(source.has(part) for part in parts))
        ]
        checks.append({"number": raw, "found_in": found})
    return checks


def unverified_numbers(project: Project, page: ResearchPage | str) -> list[str]:
    """The numbers in a finding that occur in no result record, source or input file
    (see :func:`number_check`), as the page writes them."""
    return [check["number"] for check in number_check(project, page) if not check["found_in"]]
