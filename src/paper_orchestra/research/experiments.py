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
  :func:`request_cancel`, :func:`withdraw` and :func:`recover` end and settle
  runs;
* :func:`number_check` and :func:`number_report` say where the numbers of a
  finding occur on record, :func:`dependents` what rests on an experiment.

Four rules shape all of it.  Nothing here approves anything: only :func:`approve`
moves an experiment to ``approved``, and it acts for the person.  An approval is
a record of exactly what may run (:func:`plan_digest`), and a run starts only
while the experiment still is that.  One approval is one run: a run takes the
stamp of its approval with it, and an experiment whose approved run is over is
never started again, whatever became of its page.  And nothing here is written
by a model: ``run.json`` and the result page are made by code from what the run
produced, with the values as the run wrote them.  The files of a run are read
only inside the project and never through a symbolic link.
"""

from __future__ import annotations

import ast
import bisect
import contextlib
import difflib
import hashlib
import inspect
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
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as replaced
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote

from paper_orchestra import doctor
from paper_orchestra.compute import ComputeError, JobResult, JobSpec, get_backend
from paper_orchestra.compute.base import (
    HANDLE_NAME,
    LOG_NAME,
    NOTE_PREFIX,
    JobStopped,
    human_size,
    spec_problems,
    write_record,
)
from paper_orchestra.compute.local import job_model_access
from paper_orchestra.errors import stops_the_run

# The scan that keeps a paper's results out of the shared memory already knows how
# numbers are written and which of them are specific enough to be someone's result;
# a finding is checked with the same notion, not with a second one.
from paper_orchestra.memory.leakscan import NUMBER, is_identifying, number_readings
from paper_orchestra.memory.lock import WikiLock
from paper_orchestra.research import repo as repositories
from paper_orchestra.research.model import (
    AUTOPILOT,
    COMPUTE_BACKENDS,
    EDITED_BY_PERSON,
    HUMAN,
    RUN_FIELDS,
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
    quotable,
    stamp_after,
    with_article,
)
from paper_orchestra.research.project import EXPERIMENTS_DIR, INPUTS_DIR, MANIFEST_FILE, Project
from paper_orchestra.utils.common import write_json, write_text

__all__ = [
    "CODE_AS_RUN_DIR",
    "ExperimentPlan",
    "adopt_run_code",
    "approval",
    "approve",
    "approve_by_rule",
    "approved_by_rule",
    "check_code",
    "code_authors",
    "code_digest",
    "code_dir",
    "code_files",
    "dependents",
    "edit_report",
    "edited_record",
    "failures",
    "job_command",
    "job_spec",
    "keep_code_as_run",
    "line",
    "load_plan",
    "mask_credentials",
    "metrics_table",
    "number_check",
    "number_report",
    "options",
    "plan_digest",
    "read_metrics",
    "recover",
    "reportable_text",
    "request_cancel",
    "result_records",
    "rule_candidates",
    "rule_holds",
    "run",
    "run_code_changed",
    "runs",
    "split_problems",
    "supplies_missing_code",
    "unverified_numbers",
    "waiting_in_line",
    "withdraw",
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
#: Under ``experiments/E<n>/``: what the last approval pinned, field by field and file by
#: file, so that a later look can say what changed.  The digest on the page decides; this
#: only explains.
APPROVAL_FILE = "approval.json"
#: In a run directory, written by a backend whose jobs are billed: its estimate of what
#: the job has cost so far, ``{"cost_usd": float | null, "seconds": float, "final": bool}``.
COST_FILE = "cost.json"
#: In a run directory, written by a backend whose jobs are processes of this machine:
#: ``{"session": int, "session_started": str, "scratch": str}``, what to end and remove
#: when the worker died.
JOB_FILE = "job.json"
#: How long past a job's time limit a remote backend keeps its sandbox at most (Modal's
#: margin for upload and for fetching the outputs).  A sandbox that is gone when its run
#: is settled is charged for no longer than the time limit plus this.
REMOTE_MARGIN_SECONDS = 900.0
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
#: Bytes of a result page that its metrics table, its list of artifacts and its notes may
#: take: a page has a size limit, and the run record has all of them.
MAX_METRICS_BYTES_ON_PAGE = 60_000
MAX_ARTIFACT_LIST_BYTES = 60_000
MAX_NOTES_BYTES = 20_000
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
#: Pieces of evidence (records and input files) whose numbers are kept from one check of
#: a finding to the next, for as long as the file they came from does not change.
MAX_EVIDENCE_KEPT = 2_000
#: Values of the evidence that are looked at one by one for a number of a finding before
#: each piece of evidence is asked on its own instead: the bound on the work per number.
MAX_EVIDENCE_WALK = 256
#: The shortest password or token of a repository URL that is blanked out of logs: a
#: shorter one would blank ordinary words.
MIN_URL_SECRET_CHARS = 4
#: Bytes from the start of a log that are read for what the clone step of a job said
#: before it started the command.
CLONE_HEAD_BYTES = 16_384

_RUNNER = agent_author("runner")
_DESIGNER = agent_author("designer")
#: Failures a corrected file can cure: the code's own, as opposed to the backend's, the
#: clock's or the person's.
_REPAIRABLE = frozenset({"exit", "metrics", "code", "patch"})
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"})
_NAME_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
#: A value that reads as itself in a sentence: a word, a number, a path.
_PLAIN_VALUE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_/.:\-]*")
_RUN_NUMBER = re.compile(r"[0-9]{1,9}")
#: Earlier than any stamp: what a stamp that cannot be read counts as among others.
_EPOCH = datetime.min.replace(tzinfo=UTC)
#: The name ``utils.common.write_text`` gives a run record while it is being written.
_TEMPORARY_RECORD = re.compile(rf"\.{re.escape(RUN_FILE)}\.\d{{1,10}}\.[0-9a-f]{{8}}\.tmp")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"", "false", "no", "off", "0", "none", "null"})
_NO_GPU = frozenset({"", "false", "no", "none", "null", "cpu", "0"})
_WORKER_STOPPED = "The worker stopped while this run was in progress."
_URL_USERINFO = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*://)([^/@\s]+)@")
_HOW_TO_APPROVE = "`paper-orchestra research approve`, or Approve in the app"
#: Said on the record of a run that clones with a credential where nothing fences the job in.
_SECRET_WITHIN_REACH = (
    "The address of the repository carries a password or token. It is kept out of the "
    "job's working directory, its log and this record, but the local backend is not a "
    "sandbox: the command runs as you and can read the project's settings (project.json), "
    "where the address is. Run it on the Modal backend, or give git the credential another "
    "way (a credential helper, an ssh key), where that matters."
)

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
    except (OSError, ValueError, RecursionError):
        return None


def _real_dir(path: Path) -> bool:
    """Whether ``path`` is a directory of its own, not a link to one."""
    return not path.is_symlink() and path.is_dir()


def _own_file(directory: Path, name: str) -> Path | None:
    """The regular file ``name`` in a run directory, or ``None``: a link there is not a
    file of the run, whatever it points at.  A project directory can come from anywhere,
    and what a run left is read, ended and removed on the word of these files."""
    path = directory / name
    try:
        if path.is_symlink() or not path.is_file():
            return None
    except OSError:
        return None
    return path


def _read_own_json(directory: Path, name: str) -> Any:
    path = _own_file(directory, name)
    return _read_json(path) if path is not None else None


def _in_project(project: Project, path: Path) -> bool:
    """Whether ``path`` lies inside the project and no part of the way to it, from the
    project's root down, is a symbolic link."""
    try:
        relative = path.relative_to(project.root)
    except ValueError:
        return False
    here = project.root
    for part in relative.parts:
        here = here / part
        if here.is_symlink():
            return False
    return True


def _remove_copy(directory: Path) -> None:
    """Remove a run's prepared copy of the repository, where it is a directory of the
    run's own (a link there is left alone: what it points at is not the run's)."""
    copy = directory / REPO_COPY
    if _real_dir(copy):
        shutil.rmtree(copy, ignore_errors=True)


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


# -- what must never be written down ------------------------------------------------------


def _url_secrets(url: str) -> list[str]:
    """The password or token a repository URL carries, as the strings to keep out of
    logs: ``https://token@host`` and ``scheme://user:password@host``.  The user name of
    an ssh URL is no secret."""
    match = _URL_USERINFO.match(str(url or "").strip())
    if match is None:
        return []
    user, colon, password = match.group(2).partition(":")
    lone_token = not colon and match.group(1).lower().startswith("http")
    secret = password if colon else user if lone_token else ""
    found = {secret, unquote(secret)}  # git may print either form
    if colon and password:
        found.add(match.group(2))
    return [text for text in found if len(text) >= MIN_URL_SECRET_CHARS]


def _secrets(project: Project, plan: ExperimentPlan | None = None) -> list[str]:
    """The strings that must never reach a log, a record or an event of a run: the
    credentials in this process's environment (the rule of ``paper-orchestra doctor``)
    and what the URL of the repository carries.  Longest first.

    A job's own environment holds none of them.  It can still come by them: a local job
    runs as the same user as its worker, and a job that clones a private repository is
    handed its URL.
    """
    found = set(doctor.secret_values(os.environ))
    given = project.settings.repo
    for url in (given.path if given else "", plan.repo if plan and plan.repo else ""):
        found.update(_url_secrets(url))
    return sorted(found, key=len, reverse=True)


def _masked(text: str, secrets: Sequence[str]) -> str:
    return str(doctor.scrub(text, secrets)) if secrets else text


def mask_credentials(project: Project, text: str) -> str:
    """``text`` with the credentials of this process's environment and of the project's
    repository URL blanked out: for a log before it is shown or written anywhere."""
    return _masked(text, _secrets(project))


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
    #: Repo mode only: the command line as an argv.  ``None``, and every other mode,
    #: runs ``python <entrypoint>``.
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


def _same_repository(project: Project, one: str, other: str) -> bool:
    """Whether two locations name the same repository: the same URL, credentials and a
    trailing ``.git`` aside, or the same directory."""
    if repositories.is_url(one) != repositories.is_url(other):
        return False
    if repositories.is_url(one):
        bare = [
            _URL_USERINFO.sub(r"\1", text.strip()).rstrip("/").removesuffix(".git")
            for text in (one, other)
        ]
        return bare[0] == bare[1]
    with contextlib.suppress(OSError):
        first = _local_repository(project, one).resolve()
        return first == _local_repository(project, other).resolve()
    return False


def _foreign_repository(named: str, given: Any) -> str:
    theirs = repositories.display_url(given.path) if given else "none"
    return (
        f"It names {repositories.display_url(named)} as its repository, which is not the "
        f"repository this project was given ({theirs}). Only the person chooses which "
        "repository an experiment copies or clones and runs: set it in the project's "
        "settings (settings.repo), or write the experiment's page yourself."
    )


def _repository(project: Project, reading: _Reading) -> None:
    """Fill in where a repo-mode plan runs, and vet it."""
    page, plan = reading.page, reading.plan
    given = project.settings.repo
    named = _text(page.meta.get("repo"))
    # The project's own repository as a page can name it: the settings hold the form
    # that works (an agent never sees the token in its URL).
    own = bool(named) and bool(given) and _same_repository(project, named, given.path)
    location = given.path if given and (own or not named) else named
    if not location:
        reading.problems.append(
            "It runs in the person's repository (mode: repo), but no repository is named: "
            "set one in the project's settings (settings.repo), or name it on the page "
            "(repo: a path or a URL)."
        )
        return
    ref = _text(page.meta.get("repo_ref")) or (given.ref if given and (own or not named) else None)
    plan.repo_ref = ref or None
    if ref and (problem := repositories.ref_problem(ref)):
        reading.problems.append(problem)
    # Which repository an experiment copies or clones, runs, and may send to a remote
    # backend, is the person's choice.  A page an agent wrote can only name theirs.
    foreign = bool(named) and not own and page.author != HUMAN
    if repositories.is_url(location):
        plan.repo = location
        if foreign:
            reading.problems.append(_foreign_repository(named, given))
        elif problem := repositories.url_problem(location):
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
    if foreign and not (theirs and _within(path, theirs)):
        reading.problems.append(_foreign_repository(str(path), given))
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
    if plan.command and plan.mode != "repo":
        # What the person reads and approves of a script or gpu experiment is its code;
        # a command of the page's own would run instead of it.
        problems.append(
            f"It names a command (`{_clip(shlex.join(plan.command), 120)}`), which only an "
            f"experiment in the person's repository has (mode: repo). A {plan.mode} "
            f"experiment runs its entry point, `python {plan.entrypoint}`: remove command "
            "from the page."
        )
        plan.command = None
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
    repo mode without a repository (or, on a page an agent wrote, with another
    one than the project's), a ``command`` outside repo mode, a time limit above
    the project's, values that are not what the field holds.  A time limit the
    page does not give is the default, never more than the project's limit.
    """
    reading = _read(project, _experiment(project, ref))
    if reading.problems:
        raise ProjectError(
            f"{reading.plan.id} cannot be run as it is written:\n- " + "\n- ".join(reading.problems)
        )
    return reading.plan


def job_command(plan: ExperimentPlan) -> list[str]:
    """The argv a run of ``plan`` executes in its working directory: the plan's command
    in repo mode, and ``python <entrypoint>`` otherwise.  A script or gpu experiment
    starts its entry point whatever the plan holds as a command."""
    if plan.mode == "repo" and plan.command:
        return list(plan.command)
    return ["python", plan.entrypoint]


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
    """``experiments/E<n>/code``: where the files of the experiment live (not created).

    ``ProjectError`` when that place is reached through a symbolic link.  A project
    directory can come from anywhere; what a link there leads to is not the code of
    the experiment, to be shown as such, handed to a model, replaced by an agent or
    run: every reader and writer of the code comes through here.
    """
    name = _experiment_id(project, experiment_id)
    path = project.experiment_dir(name) / CODE_DIR
    if not _in_project(project, path):
        raise ProjectError(
            f"{EXPERIMENTS_DIR}/{name}/{CODE_DIR} is reached through a symbolic link, so "
            f"the code of {name} is not read, written or run: what the link leads to is "
            "not this project's. Remove the link and put the experiment's own directory "
            "in its place, inside the project."
        )
    return path


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


# -- what an approval pins ----------------------------------------------------------------


def _pinned_repository(plan: ExperimentPlan) -> str | None:
    """The repository of a plan as an approval pins it: a URL as it is written, a
    directory by the place where it really is.

    The same project is opened by its absolute path in the app and often by a relative
    one on the command line, and its repository may be given relative to it.  However
    the path was put together, the same directory is the same repository; another
    directory behind the same name (a link pointed elsewhere since) is another.
    """
    if not plan.repo or repositories.is_url(plan.repo):
        return plan.repo
    with contextlib.suppress(OSError, RuntimeError):
        return str(Path(plan.repo).resolve())
    return os.path.abspath(plan.repo)  # a loop of links: at least not the caller's spelling


def _run_fields(plan: ExperimentPlan) -> dict[str, Any]:
    """The fields of a plan that decide what a run executes, where and for how long
    (``model.RUN_FIELDS``), in a form that can be hashed and compared."""
    fields = {
        "mode": plan.mode,
        "entrypoint": plan.entrypoint,
        "command": job_command(plan),
        "packages": list(plan.packages),
        "repo": _pinned_repository(plan),
        "repo_ref": plan.repo_ref,
        "gpu": plan.gpu,
        "network": plan.network,
        "backend": plan.backend,
        "timeout_seconds": float(plan.timeout_seconds),
    }
    assert tuple(fields) == RUN_FIELDS
    return fields


def _digest_of(fields: Mapping[str, Any], code: str) -> str:
    """SHA-256 over the run-defining fields of a plan and the digest of its code."""
    text = json.dumps({"code": code, "plan": dict(fields)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plan_digest(project: Project, ref: str) -> str:
    """SHA-256 over what a run of the experiment would execute as things are now: its
    code files (as :func:`code_digest`) and the run-defining fields of its plan (mode,
    entrypoint, command, packages, repo, repo_ref, gpu, network, backend,
    timeout_seconds), as they are read from the page and the project's settings.  A
    repository that is a directory counts as the place where it really is, so the
    digest does not depend on how the path of the project was written when it was
    opened.

    An approval records it on the page as ``approved_digest``; a run starts only while
    the two are equal (:func:`approval` says whether they are).
    """
    page = _experiment(project, ref)
    code = _current_digest(project, str(page.id))
    return _digest_of(_run_fields(_read(project, page).plan), code)


def _shown_field(name: str, value: Any) -> str:
    """A run-defining field as the record of an approval and a message show it."""
    if value is None or value == []:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if name == "command":
        return shlex.join(value)
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if name == "repo":
        return repositories.display_url(str(value))  # never the token of its URL
    return f"{value:g}" if isinstance(value, float) else str(value)


def _as_said(value: Any) -> str:
    """A field value in the sentence that says what changed since an approval.  A plain
    one (``false``, ``900``, ``main.py``) stands as it is; anything else (a command, a
    list, whatever an agent wrote into the field) is a quotation, cut, with nothing in
    it that reads as its end: the sentence is shown at checkpoints as this module's."""
    text = _one_line(value)
    if len(text) <= 80 and _PLAIN_VALUE.fullmatch(text):
        return text
    return f'"{quotable(text, 160)}"'


def _file_digests(directory: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in _code_paths(directory)
    }


def _approval_path(project: Project, experiment_id: str) -> Path:
    return project.experiment_dir(experiment_id) / APPROVAL_FILE


def _approval_record(project: Project, page: ResearchPage) -> dict[str, Any] | None:
    """What the approval on ``page`` pinned, field by field and file by file; ``None``
    when the record on file is not that approval's."""
    record = _read_json(_approval_path(project, str(page.id)))
    approved = _text(page.meta.get("approved_digest"))
    if not isinstance(record, dict) or not approved or record.get("digest") != approved:
        return None
    return record


def _write_approval(
    project: Project,
    experiment_id: str,
    digest: str,
    stamp: str,
    fields: Mapping[str, Any],
    by: str = HUMAN,
) -> None:
    record = {
        "digest": digest,
        "at": stamp,
        "plan": {name: _shown_field(name, value) for name, value in fields.items()},
        "code": _file_digests(code_dir(project, experiment_id)),
        # Only where it was not the person: their own approvals are recorded as before.
        **({"by": by} if by != HUMAN else {}),
    }
    try:
        write_json(_approval_path(project, experiment_id), record)
    except OSError as exc:
        raise ProjectError(
            f"The approval of {experiment_id} could not be recorded ({exc}), so nothing was "
            f"approved. Check that {project.experiment_dir(experiment_id)} can be written."
        ) from exc


def _written_by(entry: Mapping[str, str] | None, approved_at: str) -> str:
    """Who put the current version of a code file there, as far as the project knows."""
    written = parse_stamp((entry or {}).get("written"))
    approved = parse_stamp(approved_at)
    if entry is None or written is None or (approved is not None and written < approved):
        # ``write_code`` records every write; this one went around it.
        return " (by hand, not through the project)"
    return f" (written by {entry.get('author') or 'someone'} at {entry['written']})"


def _changes_since(project: Project, page: ResearchPage, fields: Mapping[str, Any]) -> list[str]:
    """What differs from what was approved, one phrase per field and per code file."""
    name = str(page.id)
    record = _approval_record(project, page)
    if record is None:
        return [
            f"what changed cannot be named: the record of the approval "
            f"({EXPERIMENTS_DIR}/{name}/{APPROVAL_FILE}) is missing"
        ]
    then = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    # Outside repo mode the command follows from the entry point, which is named.
    derived = {"command"} if "repo" not in (then.get("mode"), fields.get("mode")) else set()
    changes = [
        f"{field}: {_as_said(then[field])} when approved, now "
        f"{_as_said(_shown_field(field, value))}"
        for field, value in fields.items()
        if field in then and field not in derived and then[field] != _shown_field(field, value)
    ]
    directory = code_dir(project, name)
    before = record.get("code") if isinstance(record.get("code"), dict) else {}
    now, authors = _file_digests(directory), _authors(directory)
    at = _text(page.meta.get("approved_at"))
    for file in sorted(set(before) | set(now)):
        if file not in now:
            changes.append(f"{file}: removed after the approval")
        elif file not in before:
            changes.append(f"{file}: added after the approval{_written_by(authors.get(file), at)}")
        elif before[file] != now[file]:
            changes.append(
                f"{file}: changed after the approval{_written_by(authors.get(file), at)}"
            )
    return changes or [
        "something the record of the approval does not show (for example the credentials "
        "in the address of the repository)"
    ]


def _departure(
    project: Project, page: ResearchPage, fields: Mapping[str, Any], code: str
) -> str | None:
    """Why the experiment is not what was approved, as the end of a sentence; ``None``
    when it is, or is the runner's recorded repair of it."""
    approved = _text(page.meta.get("approved_digest"))
    if not approved:
        return (
            "there is no record of what was approved (it was approved before approvals "
            "were recorded, or its state was set by hand)"
        )
    if _digest_of(fields, code) in (approved, _text(page.meta.get("repaired_digest"))):
        return None
    at = _text(page.meta.get("approved_at"))
    changes = "; ".join(_changes_since(project, page, fields))
    return f"it is not what was approved{f' at {at}' if at else ''} ({changes})"


def approval(project: Project, ref: str) -> dict[str, Any]:
    """What the person approved of the experiment, and whether it still is that::

        {"digest": "<sha256>", "at": "2026-11-15T12:00:00Z", "matches": True,
         "changed": [], "repaired": False}

    ``digest`` and ``at`` are the ``approved_digest`` and ``approved_at`` the last
    approval recorded (``None`` for an experiment without one).  ``matches`` says
    whether a run would start: the experiment is what was approved, or the
    runner's own recorded repair of it (``repaired``).  ``changed`` names what
    differs from the approval: each run-defining field with its value then and
    now, each code file that was added, removed or changed, and who wrote it.
    """
    page = _experiment(project, ref)
    approved = _text(page.meta.get("approved_digest"))
    if not approved:
        return {"digest": None, "at": None, "matches": False, "changed": [], "repaired": False}
    fields = _run_fields(_read(project, page).plan)
    current = _digest_of(fields, _current_digest(project, str(page.id)))
    repaired = current != approved and current == _text(page.meta.get("repaired_digest"))
    return {
        "digest": approved,
        "at": _text(page.meta.get("approved_at")) or None,
        "matches": current == approved or repaired,
        "changed": [] if current == approved else _changes_since(project, page, fields),
        "repaired": repaired,
    }


def will_not_run(project: Project, ref: str) -> str | None:
    """Why a run of the experiment would not start as it stands, as the rest of a
    sentence about it (``it is not what was approved at ... (main.py: changed ...)``);
    ``None`` when a run would start.

    This is the answer :func:`run` gives when it is asked, told beforehand to
    whoever lists what waits to run: the page is not accepted, is not approved
    (or queued, or left running), the approval on it was used by a run that is
    over (one approval is one run), nothing records what was approved, or the
    plan or the code is no longer that.  Nothing is written and no lock is
    taken; an experiment that another process is running right now is not told
    apart from one that waits.
    """
    page = _experiment(project, ref)
    if page.review != "accepted":
        return f"its page is {page.review}, not accepted"
    if page.state not in ("approved", "queued", "running"):
        return f"it is {page.state}, not approved"
    if _run_of_approval(project, page) is not None:
        return "the run it was approved for is over (one approval is one run)"
    fields = _run_fields(_read(project, page).plan)
    return _departure(project, page, fields, _current_digest(project, str(page.id)))


def approved_by_rule(page: ResearchPage) -> bool:
    """Whether the approval on ``page`` is the standing rule's (autopilot's) and not the
    person's own.  Such an approval holds for the unattended stretch it was made in:
    nothing but that stretch's worker runs it."""
    return _text(page.meta.get("approved_by")) == AUTOPILOT


def waiting_in_line(
    project: Project, *, skip: Collection[str] = (), rule: bool = False
) -> list[ResearchPage]:
    """The experiments that wait for their run, in the order they are run in.

    Those are the accepted experiments whose page reads ``approved`` or ``queued`` and
    that would run as they stand (see :func:`will_not_run`: one that is no longer what
    was approved needs the person again, and holds nobody up).  Nothing records that
    one experiment needs another's result, so the order is that of the approvals:
    whoever was approved first runs first, and two of the same moment go by their
    ids.  ``skip`` are page paths that are left out (what a worker has already tried).

    An experiment that autopilot approved (:func:`approved_by_rule`) is in the line only
    for the worker of the stretch it was approved in, which asks with ``rule``: for
    everyone else it does not wait to run, because nobody but that worker runs it.
    """
    waiting = []
    for page in project.pages(kind="experiment", review="accepted", state=["approved", "queued"]):
        if page.path in skip or not page.id:
            continue
        if approved_by_rule(page) and not rule:
            continue
        try:
            if will_not_run(project, page.path) is not None:
                continue
        except (ProjectError, OSError, ValueError):
            continue  # what cannot be read cannot be run; its own page says why
        waiting.append(page)

    def order(page: ResearchPage) -> tuple[bool, str, tuple[int, str]]:
        stamp = _text(page.meta.get("approved_at"))
        digits = "".join(ch for ch in str(page.id) if ch.isdigit())
        return (not stamp, stamp, (int(digits) if digits else 0, str(page.id)))

    return sorted(waiting, key=order)


def line(project: Project) -> dict[str, Any]:
    """Which experiment is being run and which wait for their run, in order:
    ``{"running": [ids], "waiting": [ids], "after": {id: id or None}}``.

    ``running`` are the accepted experiments whose page reads ``running`` (whether
    anything still runs them is for whoever watches the workers to say), ``waiting``
    is :func:`waiting_in_line`, and ``after`` names for each one that waits the
    experiment it comes after: the one before it in the line, for the first of the
    line the one being run, and ``None`` when nothing is ahead of it.
    """
    running = [
        str(page.id)
        for page in project.pages(kind="experiment", review="accepted", state=["running"])
        if page.id
    ]
    waiting = [str(page.id) for page in waiting_in_line(project)]
    ahead = running[-1] if running else None
    after: dict[str, str | None] = {}
    for name in waiting:
        after[name] = ahead
        ahead = name
    return {"running": running, "waiting": waiting, "after": after}


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
    elif page.decided_by_person:
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


def _check_not_the_persons(
    directory: Path, files: Mapping[str, str], authors: Mapping[str, Mapping[str, str]]
) -> None:
    """Raise ``NotAllowed`` when an agent's write would replace a file whose current
    version the person wrote, or put there by hand (no write of it is on record)."""
    for name, text in files.items():
        target = directory / name
        if not target.is_file() or target.read_bytes() == text.encode("utf-8"):
            continue  # a new file, or the same text: nothing of the person's is replaced
        if authors.get(name, {}).get("author", HUMAN) == HUMAN:
            raise NotAllowed(
                f"{name} is as the person wrote it, so it is not replaced. Say what should "
                "change in it (on the experiment's page, or with propose_edit), or put your "
                "code into a file of another name."
            )


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
    the person accepted without any; in neither case does it replace a file
    whose current version the person wrote (``NotAllowed`` otherwise).  The
    person and the runner's repair may always.

    Who may write is checked, and the files are written, under the project's
    lock: an approval that is under way has either seen this write or keeps it
    from happening.
    """
    author = normalize_author(author)
    experiment_id = _experiment_id(project, experiment_id)
    if not isinstance(files, Mapping) or not files:
        raise ProjectError("No files were given: pass the files to write as {name: text}.")
    if len(files) > MAX_CODE_FILES:
        raise ProjectError(f"An experiment's code is at most {MAX_CODE_FILES} files.")
    checked = {_code_name(name): _code_text(name, text) for name, text in files.items()}
    directory = code_dir(project, experiment_id)
    with project.locked():
        _check_may_write(project, _page_or_none(project, experiment_id), author, experiment_id)
        _check_targets(directory, checked)
        authors = _authors(directory)
        if actor_class(author) == "agent":
            _check_not_the_persons(directory, checked, authors)
        _write_files(directory, checked, authors, author, experiment_id, project.now())
    return check_code(project, experiment_id)


def _check_targets(directory: Path, files: Mapping[str, str]) -> None:
    for name in files:
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


def _write_files(
    directory: Path,
    files: Mapping[str, str],
    authors: dict[str, dict[str, str]],
    author: str,
    experiment_id: str,
    stamp: str,
) -> None:
    """Write the checked ``files``, keeping each version that is replaced and noting who
    wrote the new one and when (``stamp``, by the project's clock: it is held against
    the stamps of the experiment's page)."""
    try:
        for name, text in files.items():
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
            # Also refuses a patch that writes metrics.json or under artifacts/: results
            # come from running the experiment, not from a patch.
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
    one or that changes what a run writes (``metrics.json``, ``artifacts/``), a
    file that would replace one of the repository.  Then warnings, each
    starting with ``WARNING``, for what the person should know when approving:
    obvious use of the network by an experiment that runs without it, code
    that never mentions ``metrics.json``, and a command that does not start a
    Python file.  :func:`split_problems` tells the two apart.
    """
    experiment_id = _experiment_id(project, experiment_id)
    plan = _plan_for_code(project, experiment_id)
    files = dict(_code_paths(code_dir(project, experiment_id)))
    in_repo = plan.mode == "repo"
    # Under the ``sdk`` runtime an approved experiment is carried out by an agent that
    # writes and repairs code itself: what is wrong with the starting code is then
    # something the person should know when approving, not something that stops them.
    # (Only where an agent does: a run on a remote backend, and so every GPU run, is
    # executed there as it is, and nobody repairs it.)
    where = plan.backend or ("modal" if plan.mode == "gpu" else project.settings.compute)
    by_agent = bool(project.settings.agents.sdk) and not in_repo and where == "local"
    if not files and by_agent:
        return [
            f"{WARNING}{experiment_id} has no code yet. The agent that carries out the "
            "experiment writes it from the plan, so read the plan as what will run."
        ]
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
    entry = _entry_file(plan)
    if in_repo:
        errors.extend(_repo_problems(plan, files))
    elif entry and entry not in files:
        errors.append(
            f"The entry point {entry} is missing from the code: the run starts with "
            f"`{shlex.join(job_command(plan))}`."
        )
    if not in_repo and not mentions_metrics:
        warnings.append(
            f"{WARNING}no file of the code mentions metrics.json. A run that does not "
            "write its results there, as a flat JSON object, fails."
        )
    if entry is None:
        # Only a repo-mode plan can name such a command; the approval shows it.
        warnings.append(
            f"{WARNING}the run starts with `{shlex.join(job_command(plan))}`, which is "
            "not a Python file of the experiment or the repository. Read the command "
            "before approving it."
        )
    if by_agent and errors:
        said = "the agent that carries out the experiment has to put this right first."
        warnings = [f"{WARNING}{error} With the sdk runtime, {said}" for error in errors] + warnings
        errors = []
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
        # A library may quote the key it was refused with; the reason is shown and kept.
        why = _masked(str(exc), doctor.secret_values(os.environ))
        return _Choice(name, problem=f"The {name} backend cannot be used: {why}")


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
    """Where an experiment runs when the person did not say: whatever asks for a GPU on
    Modal, which is the only backend with GPUs; the others where an earlier approval of
    the same experiment put them, else on the project's backend.

    A ``backend`` that merely stands on the page is not a choice: an agent may have
    written it there (older projects), and it would move model-written code from the
    sandbox the person set for the project onto their machine.  ``gpu`` is the GPU
    chosen at approval; ``None`` leaves the page's.
    """
    asked = _text(page.meta.get("gpu") if gpu is None else gpu).lower()
    if _text(page.meta.get("mode")).lower() == "gpu" or asked not in _NO_GPU:
        return "modal"
    named = _text(page.meta.get("backend")).lower()
    record = _approval_record(project, page)
    plan_then = record.get("plan") if record else None
    if named and isinstance(plan_then, dict) and plan_then.get("backend") == named:
        return named  # the person's own choice, or default, of last time
    return project.settings.compute


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


def _priced_gpus() -> list[str]:
    """The GPU types the cost estimate knows a price for."""
    from paper_orchestra.compute.modal_backend import MODAL_PRICES_USD_PER_SECOND

    return sorted(MODAL_PRICES_USD_PER_SECOND["gpu"])


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
    if billed and plan.gpu and bound is None:
        # Without a price nothing holds the run to the limit, and its cost would count
        # as nothing: it is not run until its type has one.
        refusals.append(
            f"The GPU type {_as_said(plan.gpu)} has no price estimate, so what a run costs "
            "could not be held to what the project allows for its experiments "
            "(settings.limits.experiment_usd). Choose a type with a price: "
            f"{', '.join(_priced_gpus())} (gpu: on the page, or --gpu at the approval)."
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
        checks = backend.check(python=True, latex=False)
        reasons += [_for_an_experiment(str(reason), reading.plan) for reason in checks]
        reasons += _budget_refusals(reading.plan, backend, limits, spent)
    return list(dict.fromkeys(reasons))


def _for_an_experiment(reason: str, plan: ExperimentPlan) -> str:
    """What a backend's own check says, where it speaks of the paper pipeline (compiling
    the paper, ``--compute``), said for the approval of an experiment instead."""
    if "Modal is not authenticated" not in reason:
        return reason
    other = (
        "Modal is the only backend with GPUs, so this experiment cannot run on this machine."
        if plan.gpu
        else "Or approve the run for this machine: `--backend local` (in the app, choose "
        "the local backend)."
    )
    return (
        "Modal is not authenticated (no token is configured on this machine), so the "
        "sandbox this experiment would run in cannot be started. Run `modal setup` (or set "
        f"MODAL_TOKEN_ID and MODAL_TOKEN_SECRET) and approve it again. {other}"
    )


def _backend_notes(
    plan: ExperimentPlan, choice: _Choice, limits: _Limits, spent: float
) -> list[str]:
    """What the person should know about running the plan on this backend."""
    notes: list[str] = []
    backend = choice.backend
    if plan.network:
        notes.append(
            "The page asks for network access (network: true). That is yours to grant: "
            "approve the run with network access, or without it. An approval that does "
            "not say is refused."
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


def _network_is_theirs(experiment_id: str) -> str:
    return (
        f"The plan of {experiment_id} asks for network access (network: true on its page), "
        "and that is yours to grant, not the plan's. Approve it with --network to let the "
        "run reach the network, or with --no-network to run it without (in the app: tick "
        'or clear "Allow network access" before approving).'
    )


def _approval_options(
    project: Project,
    page: ResearchPage,
    *,
    backend: str | None,
    gpu: str | None,
    timeout_seconds: float | None,
    network: bool | None,
) -> dict[str, Any]:
    """What the person decided for this run, as overrides of the page's fields.

    The approval decides; it never inherits a decision from the page.  Whether the
    run may reach the network is the person's answer, and no answer is "no" unless the
    plan asks for it: then the approval is refused, because saying nothing would grant
    or deny it unseen.  The backend is the person's choice or the project's.
    """
    wrong: list[str] = []
    asks = _flag(page.meta.get("network"), "network", wrong)
    if network is None and asks:
        raise ProjectError(_network_is_theirs(str(page.id)))
    return {
        "backend": (backend or "").strip().lower() or _default_backend(project, page, gpu),
        "gpu": gpu,
        "timeout_seconds": timeout_seconds,
        # A value that is neither true nor false stays on the page, to be reported.
        "network": None if network is None and wrong else bool(network),
    }


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
    by: str = HUMAN,
) -> ResearchPage:
    """The person approves a run of the experiment; returns its page, now ``approved``.

    This is the person's decision and nothing else calls it for them: it is what
    the command line and the web app do when the person says so, and it is the
    only way an experiment becomes ``approved``.  One caller stands in for them
    where they said so beforehand: their standing rule (:func:`approve_by_rule`,
    ``by`` is then ``autopilot``), in a project whose settings switch it on.  The
    page, the record of the approval and the events then name the rule, never the
    person (``approved_by: autopilot``), and every check below is made all the same.

    The approval decides how the run may happen and never takes that from the
    page, which an agent may have written.  ``network``: ``True`` or ``False``
    is the person's answer; ``None`` means they said nothing, which approves
    the run without network access, unless the plan asks for the network
    (``network: true``): then it is refused, saying that this is theirs to
    grant.  ``backend``: ``None`` is the project's (``settings.compute``;
    Modal for what needs a GPU), or what an earlier approval of the same
    experiment chose, never a ``backend`` that merely stands on the page.
    ``gpu`` and ``timeout_seconds``: ``None`` keeps the plan's.

    What was approved is recorded on the page: the options (``backend``,
    ``gpu``, ``timeout_seconds``, ``network``), ``approved_at`` and
    ``approved_digest``, the :func:`plan_digest` of the code and the
    run-defining fields as approved.  :func:`run` starts only while the
    experiment still has that digest.  Then the page moves ``planned ->
    approved`` (also from ``succeeded``, ``failed`` and ``cancelled``: one
    approval is one run) and the accepted hypotheses it tests move to
    ``testing``.

    Refused with ``ProjectError``, saying why, when the page is not accepted, is
    queued or running, when the plan or its code has errors, when the chosen
    backend cannot run it (``ComputeBackend.supports`` and ``check``), when it
    could cost more than is left of what the project allows for all its runs
    together, or when it asks a billed backend for a GPU type that has no price
    (its cost could not be held to that limit).  ``base_updated`` is the page's ``updated`` stamp as the person
    saw it and ``code_digest`` the digest (:func:`code_digest`) of the code
    they read: a page or code changed since is a ``Conflict``, because what is
    approved is what was read; so is one that changes while it is being
    approved.  ``backends`` (``{name: backend}``) supplies the backend to ask
    instead of one built with its defaults.
    """
    page = _experiment(project, ref)
    name = str(page.id)
    read = _current_digest(project, name)
    if code_digest is not None and code_digest != read:
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
    if page.state == "queued":
        # A queued run waits for a worker; without one, waiting would never end.
        raise ProjectError(
            f"{name} is queued: its approved run has not started. Run it (Run in the app, "
            "`research run`, or the next `research advance`) or cancel it before approving "
            "another."
        )
    if page.state == "running":
        raise ProjectError(
            f"{name} is running. Cancel that run, or wait for it, before approving another."
        )
    overrides = _approval_options(
        project, page, backend=backend, gpu=gpu, timeout_seconds=timeout_seconds, network=network
    )
    chosen = overrides["backend"]
    reading = _read(project, page, overrides=overrides)
    # A plan that cannot run anywhere is refused without a backend being made for it.
    choice = _Choice(chosen) if reading.problems else _chosen(chosen, backends)
    try:
        reasons = _refusals(project, reading, choice, _limits(project), _spent_usd(project))
    finally:
        choice.close()
    if reasons:
        raise ProjectError(f"{name} cannot be approved:\n- " + "\n- ".join(reasons))
    approved = _record_approval(project, page, reading.plan, read, base_updated, by)
    with contextlib.suppress(OSError):
        _flag_path(project, name).unlink(missing_ok=True)  # of a run cancelled earlier
    _mark_testing(project, approved)
    return approved


def _record_approval(
    project: Project,
    page: ResearchPage,
    plan: ExperimentPlan,
    code: str,
    base_updated: str | None,
    by: str = HUMAN,
) -> ResearchPage:
    """Write what was approved onto the page and move it to ``approved``, as one step.

    ``page`` and ``code`` are what the checks of the approval looked at.  Under the
    project's lock, which every write of a page and of code takes, they are compared
    with what is there now: a change that landed meanwhile is a ``Conflict`` and
    nothing is approved.
    """
    name = str(page.id)
    fields = _run_fields(plan)
    digest = _digest_of(fields, code)
    with project.locked():
        current = project.page(page.path)
        # An approval is told from the one before it by its stamp, which a run takes
        # with it: two approvals within a second are still two.  It is later than
        # everything the runs so far recorded, so that none of them reads as its run.
        earlier = [_text(current.meta.get("approved_at"))]
        for record in runs(project, name):
            earlier += [_text(record.get(key)) for key in ("approval", "started", "finished")]
        stamp = stamp_after(_now(), max(earlier, key=lambda text: parse_stamp(text) or _EPOCH))
        if current.updated != page.updated or _current_digest(project, name) != code:
            raise Conflict(
                f"{name} changed while it was being approved (its page or its code was "
                "written in that moment). Look at it as it is now, and approve again.",
                page=current,
            )
        noted = project.record_approval(
            page.path,
            {
                "backend": plan.backend,
                "gpu": plan.gpu,
                "timeout_seconds": _plain_number(plan.timeout_seconds),
                "network": plan.network,
                "approved_at": stamp,
                "approved_digest": digest,
                "repaired_digest": None,  # of the run an earlier approval allowed
                # Who approved, where it was not the person (theirs takes the mark off).
                "approved_by": by if by != HUMAN else None,
            },
            by=by,
            base_updated=base_updated,
        )
        if _digest_of(_run_fields(_read(project, noted).plan), code) != digest:
            # The page does not read back as what was approved: better no approval
            # than one that the run would then refuse without anything having changed.
            raise ProjectError(
                f"The approval of {name} could not be recorded: its page reads differently "
                "once the approved options are written on it. Correct the page's fields "
                f"({', '.join(RUN_FIELDS)}) and approve again."
            )
        _write_approval(project, name, digest, stamp, fields, by)
        return project.set_state(page.path, "approved", by=by)


# ---------------------------------------------------------------------------
# The standing rule (autopilot)
# ---------------------------------------------------------------------------

#: Why the rule does not approve a run that the local backend would execute under the
#: classic runtime: code no person has read, run as the person, outside any sandbox.
UNSANDBOXED_RUN = (
    "it would run on this machine as you, outside a sandbox (the local backend under the "
    "classic runtime), and autopilot does not approve code nobody has read for that: "
    "approve it yourself, or run experiments on Modal or with sandboxed sdk agents"
)


def _ended_runs(project: Project, experiment_id: str) -> list[dict[str, Any]]:
    """The runs of the experiment that are over for good: settled, and not put back in
    the queue by a stop."""
    return [
        record
        for record in runs(project, experiment_id)
        if record.get("status") in ("succeeded", "failed", "cancelled")
        and not record.get("requeued")
    ]


def failures(project: Project, experiment_id: str, *, since: str = "") -> int:
    """How many approved runs of the experiment ended as failed, counted by approval (a
    run that was repaired and tried again is one), from the moment ``since`` on."""
    failed = {
        _text(record.get("approval")) or _text(record.get("started"))
        for record in _ended_runs(project, experiment_id)
        if record.get("status") == "failed"
        and (not since or _text(record.get("finished") or record.get("started")) >= since)
    }
    return len(failed)


def rule_holds(project: Project, ref: str) -> str | None:
    """Why the person's standing rule does not approve a run of the experiment as it
    stands, as the rest of a sentence about it; ``None`` when it would try.

    The rule approves an accepted experiment that is planned and has not run, and
    one whose only runs failed and whose plan or code was changed since (a second
    try, and no third).  It leaves to the person: what was planned before they
    switched autopilot on (unless they asked for that to be settled too), what
    they cancelled or what succeeded already, and a run that would execute unread
    code as them outside a sandbox (:data:`UNSANDBOXED_RUN`).  Nothing is written.
    """
    page = _experiment(project, ref)
    name = str(page.id)
    if page.review != "accepted":
        return f"its page is {page.review}, not accepted"
    if page.state not in ("planned", "failed"):
        return f"it is {page.state}"
    if page.path in project.autopilot_waited:
        return (
            "it was planned before autopilot was switched on, and what waited then is "
            "yours unless you ask for it to be settled as well"
        )
    ended = _ended_runs(project, name)
    if ended:
        last = ended[-1]
        if last.get("status") != "failed":
            return f"its last run {'was cancelled' if last.get('status') == 'cancelled' else last.get('status')}"
        if failures(project, name) >= 2:
            return "it has failed twice"
        fields = _run_fields(_read(project, page).plan)
        now = _digest_of(fields, _current_digest(project, name))
        if now in (
            _text(page.meta.get("approved_digest")),
            _text(page.meta.get("repaired_digest")),
        ):
            return "it failed, and neither its plan nor its code has changed since"
    elif page.state == "failed":
        return "it failed, and nothing records the run"
    agents = project.settings.agents
    sandboxed_agents = agents.runtime == "sdk" and agents.sandbox is True
    if _default_backend(project, page) == "local" and not sandboxed_agents:
        return UNSANDBOXED_RUN
    return None


def rule_candidates(project: Project) -> list[ResearchPage]:
    """The accepted experiments the standing rule would look at, by their ids: those
    that are planned, or failed (see :func:`rule_holds` for which of them it approves)."""
    pages = project.pages(kind="experiment", review="accepted", state=["planned", "failed"])

    def order(page: ResearchPage) -> tuple[int, str]:
        digits = "".join(ch for ch in str(page.id) if ch.isdigit())
        return (int(digits) if digits else 0, str(page.id))

    return sorted((page for page in pages if page.id), key=order)


def approve_by_rule(project: Project, ref: str, *, backends: Any = None) -> ResearchPage:
    """The person's standing rule approves a run of the experiment: through
    :func:`approve`, with every check it makes, recorded as the rule's.

    The options are the plan's, on the backend the person would get by default
    (the project's; never one that merely stands on the page).  Network access is
    granted only where the plan asks for it and ``settings.autonomy.network`` is
    on; a plan that asks without that is approved to run without the network.
    ``ProjectError`` says why nothing was approved: what :func:`rule_holds` says, or
    what the approval itself refuses (errors in the code, a backend that cannot
    run it, the cost limit).
    """
    if not project.settings.autonomy.autopilot:
        raise ProjectError(
            "Autopilot is off for this project (settings.autonomy.mode): nothing is "
            "approved without the person."
        )
    page = _experiment(project, ref)
    held = rule_holds(project, page.path)
    if held is not None:
        raise ProjectError(f"{page.id} was not approved by autopilot: {held}.")
    wrong: list[str] = []
    asks = _flag(page.meta.get("network"), "network", wrong)
    grant = bool(asks and project.settings.autonomy.network)
    return approve(project, page.path, network=grant, backends=backends, by=AUTOPILOT)


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


def _runs_dir(project: Project, experiment_id: str) -> Path:
    return project.experiment_dir(experiment_id) / RUNS_DIR


def runs(project: Project, experiment_id: str) -> list[dict[str, Any]]:
    """The run records (``run.json``) of the experiment, by run number.

    A directory without a readable record is left out, and so is anything that is
    reached through a symbolic link: only the project's own files are read.
    """
    return [record for _, record in _numbered_runs(project, experiment_id)]


def last_success(project: Project, ref: str) -> dict[str, Any] | None:
    """The latest run of the experiment that succeeded: ``{"number", "result_page"}``,
    with the path of its result record (``None`` when it has none); ``None`` when no
    run of it succeeded.

    An experiment's state says how its latest approved run ended.  One that reads
    ``cancelled`` or ``failed`` may still have a result from a run before that, and
    whoever shows the state can say so with this.
    """
    name = _experiment_id(project, ref)
    for number, record in reversed(_numbered_runs(project, name)):
        if record.get("status") == "succeeded":
            return {"number": number, "result_page": project.resolve(f"results/{name}-run{number}")}
    return None


def _run_number(name: str) -> int | None:
    """The number a run directory is named by; ``None`` for any other name."""
    return int(name) if _RUN_NUMBER.fullmatch(name) else None


def _run_dirs(project: Project, experiment_id: str) -> list[tuple[int, Path]]:
    """``(number, directory)`` of the experiment's run directories, by number: real
    directories inside the project.  A link in their place (or on the way to them) is
    not a run of this project."""
    folder = _runs_dir(project, _experiment_id(project, experiment_id))
    if not folder.is_dir() or not _in_project(project, folder):
        return []
    found = [
        (number, entry)
        for entry in folder.iterdir()
        if (number := _run_number(entry.name)) is not None and _real_dir(entry)
    ]
    return sorted(found, key=lambda item: item[0])


def _numbered_runs(project: Project, experiment_id: str) -> list[tuple[int, dict[str, Any]]]:
    """``(number of the run directory, its record)``, by number."""
    found: list[tuple[int, dict[str, Any]]] = []
    for number, directory in _run_dirs(project, experiment_id):
        record = _read_own_json(directory, RUN_FILE)
        if isinstance(record, dict):
            found.append((number, record))
    return found


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
    if _in_project(project, folder):  # looked at first: nothing is made through a link
        folder.mkdir(parents=True, exist_ok=True)
    if not _in_project(project, folder):
        raise ProjectError(
            f"{folder} is reached through a symbolic link, so no run is made there: a run "
            "writes and removes files in its directory. Replace the link with a real "
            "directory inside the project."
        )
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
    #: The plan the run was made from and the backend that ran it, where they are known:
    #: what a job costs whose result never came back is worked out from them.
    plan: ExperimentPlan | None = None
    backend: Any = None
    #: Strings that must not be written into the record, the result page or an event.
    secrets: Sequence[str] = ()
    #: Why the code could not be copied for the run, said when the attempt is executed.
    unready: Exception | None = None

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
        # The approval this run uses (its ``approved_at``): one approval is one run, and
        # a settled run that carries the stamp of the page's approval has used it.
        "approval": None,
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
        # What a job of this machine started (its process session and its working
        # directory), copied from the backend's job.json: recovery ends and removes it.
        "job": None,
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


def read_metrics(directory: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """The metrics a run left in ``directory`` (see :func:`_read_metrics`), for whoever
    has to know before the run is judged whether they are there."""
    return _read_metrics(directory)


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


def _log_tail(
    directory: Path, lines: int = LOG_TAIL_LINES, *, secrets: Sequence[str] = ()
) -> list[str]:
    """The last lines of a run's log, cleaned of terminal control sequences and of the
    ``secrets``: what is returned goes into records, pages, events and a repairer's
    prompt, whatever the job managed to print."""
    log = _own_file(directory, LOG_NAME)  # not what a link of that name points at
    if log is None:
        return []
    try:
        with open(log, "rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 256 * 1024))
            data = stream.read()
    except OSError:
        return []
    text = _CONTROL.sub("", _ANSI.sub("", data.decode("utf-8", errors="replace")))
    text = _masked(text, secrets)  # after the cleaning: colour codes do not hide one
    tail = [line.rstrip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    return [line[:LOG_LINE_CHARS] for line in tail[-lines:]]


def _last_words(tail: Sequence[str]) -> str:
    """The last line the job itself printed: where a traceback names the error."""
    for line in reversed(tail):
        if not line.startswith(NOTE_PREFIX):
            return _clip(line, 300)
    return ""


def _read_handle(directory: Path) -> str | None:
    path = _own_file(directory, HANDLE_NAME)
    try:
        return (path.read_text(encoding="utf-8").strip() or None) if path else None
    except (OSError, UnicodeDecodeError):
        return None


def _read_job(directory: Path) -> dict[str, Any] | None:
    """What the backend recorded of a job it started on this machine (``job.json``)."""
    job = _read_own_json(directory, JOB_FILE)
    return job if isinstance(job, dict) and job else None


def _is_amount(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value < math.inf


def _cost_without_result(attempt: _Attempt) -> float | None:
    """What a run cost whose job did not report it: stopped, interrupted, or gone when a
    dead worker's run was settled.  ``None`` when nothing was billed or nothing is known.

    A billed backend leaves its estimate next to the run (``cost.json``).  A final one
    is the cost.  Without it the job may have run on unattended, so the run is charged
    for the time since it started, at most its time limit plus the backend's margin.
    """
    record = attempt.record
    on_file = _read_own_json(attempt.directory, COST_FILE)
    on_file = on_file if isinstance(on_file, dict) else None
    known = float(on_file["cost_usd"]) if on_file and _is_amount(on_file.get("cost_usd")) else None
    if on_file and on_file.get("final") is True:
        return known
    handle = record.get("handle") or _read_handle(attempt.directory)
    started = parse_stamp(record.get("started"))
    if not (handle or on_file) or started is None or attempt.plan is None:
        return known  # no job was ever started for this run
    then = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    limit = then.get("timeout_seconds")
    limit = float(limit) if _is_amount(limit) else attempt.plan.timeout_seconds
    plan = replaced(attempt.plan, gpu=then.get("gpu", attempt.plan.gpu))
    elapsed = max(0.0, (datetime.now(UTC) - started).total_seconds())
    seconds = min(elapsed, limit + REMOTE_MARGIN_SECONDS)
    bound = _cost_bound(plan, attempt.backend, seconds)
    if bound is None:
        return known
    attempt.notes.append(
        f"What the job {handle or 'of this run'} cost is not known: it had not reported "
        f"its end when the run was settled. The run is charged for the {seconds:.0f} s "
        f"since it started (at most its time limit plus {REMOTE_MARGIN_SECONDS:g} s), "
        f"about ${bound:.2f}."
    )
    return max(bound, known or 0.0)


# -- the result record --------------------------------------------------------------------


def _fenced(text: str, language: str = "text") -> list[str]:
    """``text`` as a code block whose fence is longer than any run of backticks in it."""
    longest = max((len(ticks) for ticks in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{language}", text.rstrip("\n"), fence]


def _cell(text: Any) -> str:
    return _one_line(text).replace("|", "\\|")


def _plain(text: Any) -> str:
    """One line in which nothing starts a link, markup or a code span in Markdown."""
    return re.sub(r"([\\\[\]<`*])", r"\\\1", _one_line(text))


def _inert(text: Any) -> str:
    """Text a job, a page or a file name supplied, as one line that is only text: what
    would start a link, markup or a code span in Markdown is escaped."""
    return _cell(_plain(text))


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
    if record.get("backend") != "local":
        return "not known"
    said = "nothing is billed for the machine"
    if record.get("carried_out_by") == "agent":
        said += "; the model of the agent that carried it out counts as model spend"
    if record.get("model_access"):
        said += (
            "; its code could call the Anthropic API, and those calls are billed to the "
            "worker's account and counted nowhere here"
        )
    return said if said != "nothing is billed for the machine" else "nothing is billed"


def _code_cell(record: Mapping[str, Any]) -> str:
    """The code of a run for its result page: the digest of what was approved, and for a
    run an agent carried out what it made of that code."""
    approved = f"sha256 `{str(record['code_digest'])[:12]}`" if record.get("code_digest") else ""
    if record.get("carried_out_by") != "agent":
        return approved
    as_run = record.get("code_as_run")
    did = _code_changes(as_run)
    if not isinstance(as_run, Mapping):
        return f"{approved} as approved; what the agent made of it was not kept".strip()
    if not did:
        return f"{approved}, as approved: the agent left it as it was".strip()
    return f"sha256 `{str(as_run.get('digest'))[:12]}` as the agent left it: it {did}" + (
        f" (approved: {approved})" if approved else ""
    )


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
    by_agent = record.get("carried_out_by") == "agent"
    rows = [
        ("Status", record.get("status")),
        ("Started", record.get("started")),
        ("Finished", record.get("finished")),
        ("Duration", _seconds_text(record.get("seconds"))),
        ("Backend", record.get("backend") or "none"),
        (
            "Exit status",
            "not measured: an agent carried the run out, and its log says what it ran"
            if by_agent
            else "none"
            if record.get("exit_code") is None
            else record["exit_code"],
        ),
        ("Estimated cost", _cost_text(record)),
    ]
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    if plan.get("command"):
        # What was executed, and with what reach: the code digest alone does not say.
        command = _plain(shlex.join(map(str, plan["command"])))
        if by_agent:
            # The plan's command, which the agent was told to start from; nothing shows
            # that this, and only this, is what it ran.
            rows.append(("Command of the plan", f"{command} (the agent's log says what ran)"))
        else:
            rows.append(("Command", command))
    if "network" in plan:
        rows.append(("Network access", "allowed" if plan["network"] else "not allowed"))
    if record.get("handle"):
        rows.append(("Remote job", f"`{record['handle']}`"))
    if record.get("code_digest") or record.get("code_as_run"):
        rows.append(("Code", _code_cell(record)))
    if by_agent and record.get("approved_digest"):
        rows.append(
            (
                "Approval",
                f"sha256 `{str(record['approved_digest'])[:12]}`, the experiment page's "
                "`approved_digest`: it covers the plan and the code together, so it "
                "differs from the digest under Code, which is of the code files alone",
            )
        )
    adopted = record.get("code_adopted")
    if isinstance(adopted, Mapping) and adopted.get("to") == record.get("code_digest"):
        rows.append(
            (
                "Code history",
                f"before this run the researcher made the code that the agent left after "
                f"run {adopted.get('run')} the experiment's code (code sha256 "
                f"`{str(adopted.get('from'))[:12]}` before, `{str(adopted.get('to'))[:12]}` "
                "since) and approved the experiment again: the digests of the two runs "
                "differ for that reason",
            )
        )
    if record.get("repo"):
        rows.append(("Repository", _repo_text(record)))
    return ["| | |", "|---|---|", *(f"| {name} | {_cell(value)} |" for name, value in rows)]


def _fitting(lines: Sequence[str], room: int) -> tuple[list[str], int]:
    """As many of ``lines`` as fit into ``room`` bytes, and how many did not: a page has
    a size limit, and ``run.json`` holds everything."""
    kept: list[str] = []
    used = 0
    for line in lines:
        used += len(line.encode("utf-8")) + 1
        if used > room:
            break
        kept.append(line)
    return kept, len(lines) - len(kept)


def _and_more(more: int) -> str:
    return f"… and {more} more in run.json."


def _metric_rows(record: Mapping[str, Any]) -> list[str]:
    """One table row for each metric of a run record, with the value as the run wrote it."""
    metrics = record.get("metrics")
    written = record.get("metrics_text")
    written = written if isinstance(written, dict) else {}
    rows = []
    for name, value in metrics.items() if isinstance(metrics, dict) else ():
        text = str(written.get(name, json.dumps(value, ensure_ascii=False)))
        rows.append(f"| {_inert(name)} | {_inert(text)}{_decimal_form(text)} |")
    return rows


def metrics_table(record: Mapping[str, Any]) -> str:
    """The metrics of a run record (``run.json``) as a Markdown table, every one of them
    and each value as the run wrote it; empty when the record holds none.  It is the
    table of the run's result page without the page's limit on its size: for whoever
    reports the metrics and must not take them from a page that can be edited."""
    rows = _metric_rows(record)
    return "\n".join(["| metric | value |", "|---|---|", *rows]) if rows else ""


def _metrics_section(record: Mapping[str, Any], directory: Path, room: int) -> list[str]:
    rows = _metric_rows(record)
    if not rows:
        if record.get("status") != "succeeded" and (directory / "metrics.json").is_file():
            return [
                "The run wrote metrics.json, but its values are not reported here: the run "
                "did not finish, so they are not a result."
            ]
        return ["The run reported no metrics."]
    shown, more = _fitting(rows, room)
    return ["| metric | value |", "|---|---|", *shown, *(["", _and_more(more)] if more else [])]


def _artifact_lines(record: Mapping[str, Any], directory: Path) -> list[str]:
    base = f"{FILES_URL}experiments/{record['experiment']}/{RUNS_DIR}/{record['number']}/"
    entries: list[str] = []
    for path in record.get("artifacts") or []:
        # The name is the job's to choose.  On the page it is one line of link text and
        # nothing else: no line of its own, no bracket that ends the link early.
        name = _CONTROL.sub("", _one_line(path.removeprefix("artifacts/")))
        label = name.replace("\\", "\\\\").replace("[", "(").replace("]", ")")
        if PurePosixPath(path).suffix.lower() in _IMAGE_SUFFIXES:
            entries.append(f"![{label}]({base}{quote(path)})\n")
        else:
            with contextlib.suppress(OSError):
                label_size = human_size((directory / path).stat().st_size)
                entries.append(f"- [{label}]({base}{quote(path)}) ({label_size})")
    shown, more = _fitting(entries, MAX_ARTIFACT_LIST_BYTES)
    lines = [line for entry in shown for line in entry.split("\n")]
    return [*lines, *([f"- {_and_more(more)}"] if more else [])]


def _note_lines(record: Mapping[str, Any]) -> list[str]:
    notes = [f"- {_inert(_clip(note, 2000))}" for note in record.get("notes") or []]
    shown, more = _fitting(notes, MAX_NOTES_BYTES)
    return [*shown, *([f"- {_and_more(more)}"] if more else [])]


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
    record: Mapping[str, Any],
    experiment: ResearchPage,
    directory: Path,
    tail: Sequence[str],
    *,
    brief: bool = False,
) -> str:
    """The text of a result page.  Every word of it follows from the run record, the
    run directory and the log; no model has a part in it.

    What a job can make long (its artifacts, the notes, the metrics) is listed as far
    as it fits, with a line that says how many more ``run.json`` holds.  ``brief``
    leaves out everything but the facts and the metrics: the page that is written when
    even that did not fit.
    """
    status = record["status"]
    took = _seconds_text(record.get("seconds"))
    lines = [
        f"Run {record['number']} of [[{experiment.link}|{record['experiment']}]] "
        f"({_inert(_clip(experiment.title, 300))}) {status} after {took} on the "
        f"{_inert(record.get('backend') or 'no')} backend.",
        "",
    ]
    if record.get("error"):
        lines += [*_fenced(str(record["error"])), ""]
    room = MAX_METRICS_BYTES_ON_PAGE // (2 if brief else 1)
    lines += [*_facts(record), "", "## Metrics", "", *_metrics_section(record, directory, room)]
    lines.append("")
    if brief:
        where = f"{EXPERIMENTS_DIR}/{record['experiment']}/{RUNS_DIR}/{record['number']}"
        lines += [
            "The artifacts, notes, repairs and the log of this run are not shown: together "
            f"they did not fit on a page. They are in {where}/ ({RUN_FILE}, {LOG_NAME}).",
        ]
        return "\n".join(lines) + "\n"
    artifacts = _artifact_lines(record, directory)
    if artifacts:
        lines += ["## Artifacts", "", *artifacts, ""]
    if record.get("notes"):
        lines += ["## Notes", "", *_note_lines(record), ""]
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
    """Write the result record of a settled run.  When the page with everything on it
    is refused (a page has a size limit), the facts and the metrics are written alone:
    a run without a record would never be analysed."""
    record = attempt.record

    def write(body: str) -> ResearchPage:
        return project.write_record(
            "result",
            f"{record['experiment']}-run{record['number']}",
            f"{record['experiment']} run {record['number']}: {_clip(experiment.title, 300)}",
            body,
            summary=_result_summary(record),
            fields={
                "state": record["status"],
                "run": record["number"],
                "backend": record["backend"],
            },
            relations={"run_of": [record["experiment"]]},
        )

    try:
        return write(_result_body(record, experiment, attempt.directory, tail))
    except ProjectError:
        return write(_result_body(record, experiment, attempt.directory, tail, brief=True))


def _shorten(error: str | None) -> str | None:
    """The reason a run failed, cut to what a record and a page can carry."""
    if error is None or len(error) <= MAX_ERROR_CHARS:
        return error
    return error[:MAX_ERROR_CHARS].rstrip() + "\n… (cut short)"


def _emit(project: Project, type: str, summary: str, **details: Any) -> None:
    """Tell the activity feed; a feed that cannot be written must not end a run."""
    with contextlib.suppress(ProjectError, OSError):
        project.emit(type, summary, agent="runner", **details)


def _masked_entries(entries: Mapping[str, Any] | None, secrets: Sequence[str]) -> Any:
    """``entries`` (the metrics of a run, or their text as written) with every credential
    blanked out of the names and of the values that are text; numbers stay as they are.
    ``None`` stays ``None``."""
    if not entries or not secrets:
        return entries
    return {
        _masked(str(name), secrets): _masked(value, secrets) if isinstance(value, str) else value
        for name, value in entries.items()
    }


def _finish(
    project: Project, experiment: ResearchPage, attempt: _Attempt, outcome: _Outcome
) -> dict[str, Any]:
    """Settle a run: its record, its result page, the event.  Returns the record.

    The record is written first and by itself: whatever fails afterwards, the run
    no longer reads as running.  Nothing a job printed that is a credential gets
    into either (``attempt.secrets``).
    """
    record, result, secrets = attempt.record, attempt.result, attempt.secrets
    artifacts, dropped = _artifacts(attempt.directory)
    seconds = result.seconds if result is not None and result.seconds else None
    if seconds is None and attempt.began is not None:
        seconds = time.monotonic() - attempt.began
    cost = result.cost_usd if result is not None else None
    if cost is None and (result is None or result.error or result.cancelled):
        cost = _cost_without_result(attempt)  # the job's own account never came back
    error = _shorten(_masked(outcome.error, secrets) if outcome.error else outcome.error)
    # A text metric, and the name of any metric, are the job's words as much as a line
    # of its log: the record and the result page are written from the masked ones.
    metrics = _masked_entries(outcome.metrics, secrets)
    metrics_text = _masked_entries(outcome.metrics_text, secrets)
    cloned = _clone_notes(record, attempt.directory)
    notes = [*attempt.notes, *cloned, *(result.notes if result else []), *dropped]
    by_agent = record.get("carried_out_by") == "agent"
    if by_agent:
        as_run = _code_as_run(attempt.directory, attempt.plan)
        if as_run is not None:
            record["code_as_run"] = as_run
        did = _code_changes(as_run)
        if did:
            notes.append(
                f"The agent that carried out this run {did}. The code as the agent left "
                f"it is kept in runs/{record['number']}/{CODE_AS_RUN_DIR}; the experiment's "
                "own code is still what was approved."
            )
    record.update(
        status=outcome.status,
        finished=_now(),
        seconds=round(seconds, 3) if seconds is not None else None,
        # An agent may have run the code any number of times, or not at all: nobody
        # measured an exit status, so the record names none.
        exit_code=result.returncode if result is not None and not by_agent else None,
        timed_out=bool(result is not None and result.timed_out),
        metrics=metrics,
        artifacts=artifacts,
        error=error,
        cost_usd=cost,
        handle=record.get("handle")
        or (result.handle if result else None)
        or _read_handle(attempt.directory),
        error_kind=outcome.kind,
        requeued=outcome.requeued,
        notes=list(dict.fromkeys(_masked(str(note), secrets) for note in notes)),
        # The backend's file says the latest; the record's copy is from when it appeared.
        job=_read_job(attempt.directory) or record.get("job"),
    )
    # A record found on disk may be older than these fields, or written by hand.
    record.setdefault("repairs", [])
    record.setdefault("result", None)
    if metrics_text:
        record["metrics_text"] = metrics_text
    attempt.save()
    refs = [experiment.path]
    if not outcome.requeued:
        # A run that only ended because its worker did is no outcome of the experiment;
        # the run that follows it will have the record.
        page = _record_result(project, experiment, attempt)
        if page is not None:
            refs.append(page.path)
        attempt.save()
    what = f"Run {record['number']} of experiment {record['experiment']}"
    took = _seconds_text(record["seconds"])
    why = f": {_clip(error, 200)}" if error else ""
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
            "error": error,
            "requeued": outcome.requeued,
            "result": record["result"],
        },
    )
    return record


def _record_result(
    project: Project, experiment: ResearchPage, attempt: _Attempt
) -> ResearchPage | None:
    """Write the result record of a settled run and note its path in the run's record.

    A run without a record is one the analysis never sees.  When the page cannot be
    written even in its shortest form, the run's record says so and so does an
    ``error`` event: it must not pass in silence.
    """
    record = attempt.record
    tail = _log_tail(attempt.directory, secrets=attempt.secrets)
    try:
        page = _write_result(project, experiment, attempt, tail)
    except ProjectError as exc:
        record["notes"].append(f"The result page could not be written: {exc}")
        where = f"{EXPERIMENTS_DIR}/{record['experiment']}/{RUNS_DIR}/{record['number']}/{RUN_FILE}"
        _emit(
            project,
            "error",
            f"Run {record['number']} of experiment {record['experiment']} "
            f"{record['status']}, but its result record could not be written, so the "
            f"analysis will not see it: {_clip(exc, 300)} What the run produced is in {where}.",
            refs=[experiment.path],
            data={
                "experiment": record["experiment"],
                "run": record["number"],
                "error": str(exc),
            },
        )
        return None
    record["result"] = page.path
    # What code wrote, so that a later look can tell a page someone edited since.
    record["result_digest"] = _page_digest(page)
    return page


def _page_digest(page: ResearchPage) -> str:
    """SHA-256 over what a result page says: its title, summary and text."""
    said = f"{page.title}\0{page.summary}\0{page.body}"
    return hashlib.sha256(said.encode("utf-8")).hexdigest()


def _has_result(project: Project, record: Mapping[str, Any]) -> bool:
    """Whether the result page a settled run's record names is there."""
    path = record.get("result")
    return isinstance(path, str) and bool(path) and project.resolve(path) is not None


# -- how a job ended ----------------------------------------------------------------------


def _judge(
    result: JobResult,
    directory: Path,
    plan: ExperimentPlan,
    *,
    cancelled_by_person: bool,
    secrets: Sequence[str] = (),
) -> _Outcome:
    """Read a job's result the way the compute contract says: the backend's own failure
    first, then a stop or the clock, then the exit status, then what it wrote."""
    tail = _log_tail(directory, secrets=secrets)
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
    declared = getattr(result, "declared_failed", None)
    if declared:
        # An agent carried the run out and says that it failed (the result of a job
        # an agent did, ``research.sdk.experiment.AgentJobResult``).  Its words are a
        # quotation; the files it left are kept, and count as no result.
        said = quotable(declared, 600)
        return _Outcome(
            "failed",
            f'The agent that carried out the run says that it failed: "{said}".',
            kind="declared",
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
        # What the job printed last is the job's own words (its code may be a model's):
        # a quotation in the sentence, not a continuation of it.
        quoted = f': "{quotable(said, 300)}".' if said else "."
        why = f"The command exited with status {result.returncode}{quoted}"
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


def _move(
    project: Project, path: str, state: str, reason: str | None = None, *, limit: int = 300
) -> None:
    """Move the experiment along its run.  When the page is no longer where the run left
    it (the person changed or removed it meanwhile), theirs stands."""
    with contextlib.suppress(ProjectError):
        project.set_state(path, state, by=SYSTEM, reason=_clip(reason, limit) if reason else None)


# -- one approval is one run ------------------------------------------------------------------


def _run_of_approval(project: Project, page: ResearchPage) -> tuple[int, dict[str, Any]] | None:
    """The run that used the approval now on ``page``, as ``(number, record)``: the
    newest run of the experiment, when it is settled, will not be repeated by itself
    (it is not ``requeued`` and no repair of it is owed) and carries the stamp of that
    approval, or of a later one.  ``None`` when the approval has not been used.

    One approval is one run.  The record of a run is written before the experiment's
    page moves, so a worker that stops or dies in between leaves a page that still
    reads as waiting or running; this is what tells that its run is over.  Approvals
    are stamped in order (:func:`_record_approval`), so a page whose stamp is earlier
    than its newest run's shows an approval from before that run: a copy of the page
    saved from an editor that was open since then.  That approval was used long ago.
    """
    stamp = _text(page.meta.get("approved_at"))
    found = _numbered_runs(project, str(page.id)) if stamp and page.id else []
    if not found:
        return None
    number, record = found[-1]
    unsettled = record.get("status") in (None, "running")
    if unsettled or record.get("requeued") or record.get("repair_owed"):
        return None
    # A record from before runs carried their approval is told by when the run started.
    used = _text(record.get("approval")) or _text(record.get("started"))
    if used != stamp:
        of_page, of_run = parse_stamp(stamp), parse_stamp(used)
        if of_page is None or of_run is None or of_run < of_page:
            return None  # approved since that run (or the two cannot be compared)
    return number, record


def _conclude(project: Project, page: ResearchPage) -> dict[str, Any] | None:
    """Where the approval on ``page`` has been used by a run that is over: give the run
    the result page it is owed, move the page to the state that run left the experiment
    in, and return the run's record.  ``None``, and nothing done, when the experiment
    may still run.  The caller holds the experiment's run lock."""
    found = _run_of_approval(project, page)
    if found is None:
        return None
    number, record = found
    # The directory says which run it is, whatever a record edited by hand says.
    record.update(experiment=str(page.id), number=number)
    record.setdefault("notes", [])
    directory = _runs_dir(project, str(page.id)) / str(number)
    if not _has_result(project, record):
        attempt = _Attempt(number, directory, record, secrets=_secrets(project))
        if _record_result(project, page, attempt) is not None:
            with contextlib.suppress(OSError):
                attempt.save()
    if page.state in ("approved", "queued", "running"):
        _move(project, page.path, *_final_state(record))
        with contextlib.suppress(OSError):
            _flag_path(project, str(page.id)).unlink(missing_ok=True)
    return record


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


def _backend_of(compute: Any, plan: ExperimentPlan | None) -> Any:
    """``compute`` is a backend, or a function that chooses one for a plan."""
    return compute if hasattr(compute, "run_job") else compute(plan)


def _backend_of_record(compute: Any, plan: ExperimentPlan | None, record: Mapping[str, Any]) -> Any:
    """The backend that ran the job of ``record``: ``compute``'s choice for the plan as
    the record names its backend, whatever the page says today."""
    named = str(record.get("backend") or "")
    if plan is not None and named in COMPUTE_BACKENDS:
        plan = replaced(plan, backend=named)
    return _backend_of(compute, plan)


def _call(method: Callable[..., Any], *args: Any, **options: Any) -> Any:
    """Call a backend's ``run_job`` or ``attach`` with the options it takes.

    ``mask``, ``spec`` and ``started`` joined the contract later than the rest.  A
    backend written before them is called without: what it writes into the log is
    then masked here before any of it is kept, and a job it picks up is described by
    what the job's own side says.
    """
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return method(*args, **options)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return method(*args, **options)
    taken = {parameter.name for parameter in parameters}
    return method(*args, **{name: value for name, value in options.items() if name in taken})


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


#: Where the code lies as an agent left it when it had carried a run out, in the run
#: directory beside ``code/`` (the approved code the agent started from).
CODE_AS_RUN_DIR = "code-as-run"
#: Files an agent added to its folder that count as code of the run, by their ending.
#: Anything else it left there (data, outputs, notes) is not kept as code.
_SCRIPT_SUFFIXES = frozenset(
    {".py", ".sh", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".ipynb", ".r", ".jl"}
)
#: Directories of an agent's folder that hold no code of the run.
_NOT_CODE_DIRS = frozenset({"artifacts", "venv", "env", "node_modules", "site-packages", "inputs"})
#: How many files an agent added are kept as code of a run.
_MAX_ADDED_FILES = 200


def keep_code_as_run(
    work: Path, run_dir: Path, *, whole: bool = True, skip: Sequence[str] = ()
) -> None:
    """Keep the code as an agent left it in ``work`` when it had carried the run out, in
    ``run_dir/code-as-run``.

    The agent starts from the approved code and may change it: what the run's
    record names as its code must be what was there when the metrics were, not
    only what was approved.  Kept are the files of the approved code as they
    now are, and, when the folder was made from that code alone (``whole``), the
    scripts the agent added.  Links are not followed, hidden files, results and
    environments are not code, and the limits on code hold as for any run.
    ``skip`` names files in the folder that are neither (the launcher).
    """
    approved = {name for name, _ in _code_paths(Path(run_dir) / CODE_DIR)}
    dest = Path(run_dir) / CODE_AS_RUN_DIR
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    copied = added = 0
    for name, path in _code_paths(Path(work)):
        parts = name.split("/")
        if name in skip or name == "metrics.json" or _NOT_CODE_DIRS.intersection(parts[:-1]):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if name not in approved:
            fits = size <= MAX_CODE_FILE_BYTES and added < _MAX_ADDED_FILES
            script = path.suffix.lower() in _SCRIPT_SUFFIXES or parts[-1].startswith("requirements")
            if not (whole and script and fits):
                continue
            added += 1
        if copied + size > MAX_CODE_BYTES:
            continue
        copied += size
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            # Not through a link that appeared since the look: a link is copied as one,
            # and no reader of this directory takes a link for code.
            shutil.copyfile(path, dest / name, follow_symlinks=False)


def _code_as_run(directory: Path, plan: ExperimentPlan | None) -> dict[str, Any] | None:
    """What became of the approved code in the hands of the agent that carried the run
    out: the digest of the code as it left it (as :func:`code_digest` counts) and the
    names of the files it changed, added and removed.  ``None`` when nothing was kept."""
    kept = directory / CODE_AS_RUN_DIR
    if not kept.is_dir():
        return None
    approved = dict(_code_paths(directory / CODE_DIR))
    if plan is not None and plan.mode == "repo":
        approved.pop(repositories.PATCH_NAME, None)  # applied to the copy, not a file of it
    ran = dict(_code_paths(kept))

    def differs(name: str) -> bool:
        try:
            return approved[name].read_bytes() != ran[name].read_bytes()
        except OSError:
            return True

    return {
        "digest": _digest(ran.items()),
        # How many files the approved code had: none for a plan that came without code,
        # which the agent then wrote.
        "approved": len(approved),
        "changed": sorted(name for name in ran if name in approved and differs(name)),
        "added": sorted(name for name in ran if name not in approved),
        "removed": sorted(name for name in approved if name not in ran),
    }


def _code_changes(as_run: Mapping[str, Any] | None) -> str:
    """ "changed main.py, added util.py": what an agent did to the approved code, as the
    rest of a sentence; empty when it left it as it was."""
    if not isinstance(as_run, Mapping):
        return ""
    parts = []
    for verb in ("changed", "added", "removed"):
        names = [str(name) for name in as_run.get(verb) or []]
        if names:
            more = f" and {len(names) - 6} more" if len(names) > 6 else ""
            parts.append(f"{verb} {', '.join(_inert(name) for name in names[:6])}{more}")
    return ", ".join(parts)


def run_code_changed(record: Mapping[str, Any]) -> bool:
    """Whether the agent that carried ``record``'s run out left other code than the
    approved code it started from."""
    return bool(_code_changes(record.get("code_as_run")))


def adopt_run_code(
    project: Project, experiment_id: str, number: int | None = None, *, author: str = "human"
) -> dict[str, Any]:
    """Make the code an agent left after a run the experiment's code, as the person's
    own edit: ``{"run", "files", "removed", "problems"}``.

    The agent that carries an experiment out repairs the code in its own folder,
    and the experiment's code stays what was approved: run again, it would fail
    the same way and be repaired again.  This writes the files the agent
    changed or added in run ``number`` (default: the latest run that has any)
    into the experiment's code through :func:`write_code`, so every rule about
    code holds: earlier versions are kept, and an experiment whose approved
    code changes must be approved again.  A file the agent removed is left in
    place and named in ``removed``.
    """
    experiment_id = _experiment_id(project, experiment_id)
    wanted = [
        run
        for run in runs(project, experiment_id)
        if (number is None or run.get("number") == number) and run_code_changed(run)
    ]
    if not wanted:
        which = f"Run {number} of {experiment_id}" if number is not None else experiment_id
        raise ProjectError(f"{which} has no code that an agent changed: there is nothing to adopt.")
    run = wanted[-1]
    as_run = run["code_as_run"]
    kept = _runs_dir(project, experiment_id) / str(run["number"]) / CODE_AS_RUN_DIR
    files: dict[str, str] = {}
    for name in [*as_run.get("changed", []), *as_run.get("added", [])]:
        text = _read_code(kept / str(name)) if _within(kept / str(name), kept) else None
        if text is None:
            raise ProjectError(
                f"{name} of run {run['number']} cannot be adopted: it is not text, too "
                "large, or no longer there."
            )
        files[str(name)] = text
    before = code_digest(project, experiment_id)
    problems = write_code(project, experiment_id, files, author=author) if files else []
    after = code_digest(project, experiment_id)
    if after != before:
        # On the record of the run the code came from: a later run names it, so that
        # nobody takes two digests of one experiment for code of unknown origin.
        adopted = {"at": _now(), "by": author, "from": before, "to": after}
        path = _runs_dir(project, experiment_id) / str(run["number"]) / RUN_FILE
        with contextlib.suppress(OSError, ValueError):
            write_record(path, {**run, "adopted": adopted})
    return {
        "run": run["number"],
        "files": sorted(files),
        "removed": [str(name) for name in as_run.get("removed", [])],
        "problems": problems,
    }


def _adoption_before(project: Project, experiment_id: str) -> dict[str, Any] | None:
    """The latest time the person made the code an agent left after a run the
    experiment's code: ``{"run", "from", "to", "at"}``, or ``None``."""
    found = None
    for record in runs(project, experiment_id):
        adopted = record.get("adopted")
        if isinstance(adopted, dict) and adopted.get("to"):
            found = {"run": record.get("number"), **adopted}
    return found


def _clear_results(copy: Path) -> list[str]:
    """Remove what a run writes its results to (``metrics.json``, ``artifacts/``) from
    the copy of a repository, and return what was there.

    A result is what the command wrote.  A file of those names that came with the
    repository, from someone's earlier work, would be recorded as this run's.
    """
    removed: list[str] = []
    for name in ("metrics.json", "artifacts"):
        path = copy / name
        if not os.path.lexists(path):
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(f"{name}/")
        else:
            path.unlink(missing_ok=True)
            removed.append(name)
    return removed


def _clone_notes(record: Mapping[str, Any], directory: Path) -> list[str]:
    """What the clone step of a job removed before it started the command (a
    ``metrics.json`` or ``artifacts`` that came with the repository), as a note for the
    record of the run.

    The repository is cloned inside the job, so this side learns it from the log, where
    the clone step says it on a line of its own before the command prints anything.
    Only the names a run writes its results to are taken from that line: the command
    writes to the same log, and a line of its making must not become a statement of the
    record.
    """
    repo = record.get("repo")
    if not isinstance(repo, dict) or repo.get("method") != "clone":
        return []
    log = _own_file(directory, LOG_NAME)
    if log is None:
        return []
    try:
        with open(log, "rb") as stream:
            head = stream.read(CLONE_HEAD_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return []
    said = f"{repositories.CLONE_PREFIX}Removed "
    line = next((line for line in head.splitlines() if line.startswith(said)), "")
    listed = line[len(said) :].split(" that was there", 1)[0].split(",")
    names = [name.strip() for name in listed if name.strip() in repositories.RESULT_PATHS]
    if not names:
        return []
    return [
        f"The job that cloned the repository removed {', '.join(names)} before the command "
        "started (it came with the repository): only what the command itself writes is the "
        "result of the run."
    ]


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
        page: ResearchPage,
        compute: Any,
        *,
        should_stop: Callable[[], bool] | None,
        repair: Callable[..., Mapping[str, str] | None] | None,
        on_log: Callable[[str], None] | None,
        limits: Mapping[str, Any] | None,
        unrepaired: str | Callable[[Mapping[str, Any]], str] | None = None,
    ) -> None:
        self.project = project
        self._compute = compute
        self._limits_given = limits
        self.limits = _limits(project, limits)
        #: Whether ``limits.usd`` is what this call may spend (its caller keeps the
        #: project's accounts) rather than the project's total.
        self._usd_is_for_this_call = limits is not None and "experiment_usd" in limits
        #: The backend the attempts run on: chosen once the approval has been checked.
        self.backend: Any = None
        self.backend_name = ""
        #: What the runs settled in this call have cost.
        self._cost = 0.0
        self._user_stop = should_stop
        self._user_log = on_log
        self._repair = repair
        #: Why there is no repairer, for the notes of a run that one could have repaired.
        self._unrepaired = unrepaired or None
        self._repairs: list[dict[str, Any]] = []
        self._rounds = 0
        self._current: _Attempt | None = None
        self._stopped = False
        #: Why a failure that could have been repaired was not: ``cancel`` or ``stop``.
        self._interrupted: str | None = None
        self._flag = _flag_path(project, str(page.id))
        self._guard = threading.Lock()
        self._next_record_check = 0.0
        self._pending: deque[str] = deque(maxlen=LOG_EVENT_LINES)
        self._last_log_event = -math.inf
        self._adopt(page)

    def _adopt(self, page: ResearchPage) -> None:
        """Read from ``page`` what this session runs and what it holds against the
        approval."""
        self.page = page
        self.reading = _read(self.project, page, limits=self._limits_given, cap=True)
        self.plan = self.reading.plan
        #: The run-defining fields as an approval pins them: read like the plan that
        #: runs, but with the time limit the page has, before today's cap.
        self._fields = _run_fields(_read(self.project, page).plan)
        self._mask = _secrets(self.project, self.plan)

    # -- the approval ------------------------------------------------------------

    def _departure(self, code: str | None = None) -> str | None:
        """Why what this session is about to run is not what was approved, as the end
        of a sentence; ``None`` when it is.  ``code`` is the digest of the code that
        would run (the code directory's, unless it is given)."""
        page = self.project.page(self.page.path)  # the approval as the page records it now
        fields = _run_fields(_read(self.project, page).plan)
        if code is None:
            code = _current_digest(self.project, self.plan.id)
        reason = _departure(self.project, page, fields, code)
        if reason is None and fields != self._fields:
            reason = "its page changed while it was being run"
        return reason

    def _check_approval(self) -> None:
        """Raise ``ProjectError``, with the experiment back at ``planned``, unless it is
        still what the person approved (or the runner's recorded repair of that)."""
        reason = self._departure()
        if reason is not None:
            self._not_approved(reason)

    def _not_approved(self, reason: str) -> None:
        """Put the experiment back to ``planned`` with the reason and raise ``ProjectError``."""
        _move(
            self.project,
            self.page.path,
            "planned",
            f"Not run: {reason}. Approve it again.",
            limit=700,
        )
        raise ProjectError(
            f"{self.plan.id} was not run: {reason}. Look at the experiment as it is now and "
            f"approve it again ({_HOW_TO_APPROVE})."
        )

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
        # notice what a backend left next to a job that stays silent.
        self._watch_records()
        return self._stop_reason() is not None

    def _stopped_outcome(self) -> _Outcome:
        result = JobResult(returncode=None, cancelled=True)
        return _judge(
            result,
            self._current.directory if self._current else Path(),
            self.plan,
            cancelled_by_person=self._flag.exists(),
            secrets=self._mask,
        )

    # -- while the job runs ------------------------------------------------------

    def _watch_records(self) -> None:
        """Copy what the backend recorded of the job into ``run.json`` as soon as it is
        there: the handle of a remote job, or the process session and working directory
        of a local one.  A worker that dies then leaves behind what is needed to pick
        the job up, or to end it and remove what it left."""
        attempt = self._current
        now = time.monotonic()
        if attempt is None or now < self._next_record_check:
            return
        record = attempt.record
        if record.get("handle") or record.get("job"):
            return
        with self._guard:
            self._next_record_check = now + HANDLE_CHECK_SECONDS
            found = {
                "handle": _read_handle(attempt.directory),
                "job": _read_job(attempt.directory),
            }
            new = {name: value for name, value in found.items() if value and not record.get(name)}
            if new:
                record.update(new)
                attempt.save()

    def _on_log(self, line: str) -> None:
        # Whatever the backend let through: a credential the job printed stops here.
        line = _masked(line, self._mask)
        if self._user_log is not None:
            self._user_log(line)
        self._watch_records()
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
        record["approval"] = _text(self.page.meta.get("approved_at")) or None
        adopted = _adoption_before(self.project, self.plan.id)
        if adopted:
            record["code_adopted"] = adopted
        # Who does the job is a fact of this run, kept with it: the project's runtime
        # can be changed afterwards, and the run stays what it was.
        by_agent = bool(getattr(self.backend, "carried_out_by_agent", False))
        if by_agent:
            record["carried_out_by"] = "agent"
            # The digest of the approval (plan and code together), for the result page
            # to name beside the digest of the code as the agent left it: two digests
            # with one name read as a contradiction.  Only for a run by an agent: the
            # record and the result page of a classic run are what they always were.
            record["approved_digest"] = _text(self.page.meta.get("approved_digest")) or None
        if self.plan.network and self.backend_name == "local" and job_model_access():
            # The run's code can call the Anthropic API with the worker's federation
            # settings.  Nothing here counts those calls: the record says so.
            record["model_access"] = True
        attempt = _Attempt(
            number,
            directory,
            record,
            began=time.monotonic(),
            plan=self.plan,
            backend=self.backend,
            secrets=self._mask,
        )
        attempt.notes.extend(self.reading.notes)
        self._copy_code(attempt)
        # From its first save on the attempt is this session's to settle, whatever comes:
        # a stop between the two lines must not leave a record that reads running.
        self._current = attempt
        attempt.save()
        self._pending.clear()
        self._last_log_event = -math.inf
        _move(self.project, self.page.path, "running")
        _emit(
            self.project,
            "run.started",
            f"Run {number} of experiment {self.plan.id} started on the "
            f"{self.backend_name or 'compute'} backend.",
            refs=[self.page.path],
            data={
                "experiment": self.plan.id,
                "run": number,
                "backend": self.backend_name,
                **({"carried_out_by": "agent"} if by_agent else {}),
            },
        )
        return attempt

    def _copy_code(self, attempt: _Attempt) -> None:
        """Copy the code into the run directory and hold the copy against the approval.

        The copy is what runs.  A write that landed between the look at the code
        directory and the copy is not run: the run directory is removed again (no run
        took place, so none is on record), the experiment goes back to ``planned``
        and ``ProjectError`` says what changed, as at the first look.
        """
        record, directory = attempt.record, attempt.directory
        try:
            record["code_digest"] = _snapshot(
                code_dir(self.project, self.plan.id), directory / CODE_DIR
            )
        except (_Setup, OSError) as exc:
            # The run's own failure, on its record: said when the attempt is executed.
            attempt.unready = exc
            return
        changed = self._departure(record["code_digest"])
        if changed is not None:
            shutil.rmtree(directory, ignore_errors=True)
            self._not_approved(changed)

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
        if attempt.unready is not None:
            raise attempt.unready
        code = directory / CODE_DIR  # copied, and held against the approval, by _copy_code
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
                if self.backend_name == "local" and repositories.userinfo(str(plan.repo)):
                    # Nothing here can keep it from the command; the record says so.
                    attempt.notes.append(_SECRET_WITHIN_REACH)
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
        removed = _clear_results(copy)
        if removed:
            attempt.notes.append(
                f"Removed from the copy of the repository before the run: {', '.join(removed)}. "
                "It came with the repository; only what the run itself writes is its result."
            )
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
            attempt.result = _call(
                self.backend.run_job,
                spec,
                source_dir=source,
                run_dir=attempt.directory,
                on_log=self._on_log,
                should_stop=self._should_stop,
                mask=tuple(self._mask),
            )
        finally:
            _remove_copy(attempt.directory)
        return _judge(
            attempt.result,
            attempt.directory,
            self.plan,
            cancelled_by_person=self._flag.exists(),
            secrets=self._mask,
        )

    def _attempt(self) -> dict[str, Any]:
        attempt = self._open()
        outcome = self._execute(attempt)
        record = _finish(self.project, self.page, attempt, outcome)
        self._current = None
        self._count(record)
        return record

    # -- repairs -----------------------------------------------------------------

    def _repairable(self, record: dict[str, Any]) -> bool:
        wanted = (
            record["status"] == "failed"
            and record.get("error_kind") in _REPAIRABLE
            and self._rounds < self.limits.repair_rounds
        )
        if wanted and self._repair is None and self._unrepaired:
            # Nobody is there to repair it (no model can be used): the failure stands,
            # and the run says why nothing was tried.
            why = self._unrepaired(record) if callable(self._unrepaired) else self._unrepaired
            self._note(record, f"The failure was not repaired: {_one_line(str(why))}")
        curable = wanted and self._repair is not None
        if curable:
            self._interrupted = self._stop_reason()
            if self._interrupted == "stop":
                self._owe_repair(record, True)  # it goes back to the queue for it
        return curable and self._interrupted is None

    def _save(self, record: dict[str, Any]) -> None:
        write_json(_runs_dir(self.project, self.plan.id) / str(record["number"]) / RUN_FILE, record)

    def _note(self, record: dict[str, Any], note: str) -> None:
        # A note may quote an error, and an error what it was given (a key, a URL).
        record["notes"].append(_masked(note, self._mask))
        self._save(record)

    def _owe_repair(self, record: dict[str, Any], owed: bool) -> None:
        """Note on a failed run's record whether its repair is still to come.  While it
        is, the failure is not how the approved run ended: a worker that stops in the
        middle leaves the experiment to be run again, not failed."""
        if bool(record.get("repair_owed")) != owed:
            record["repair_owed"] = owed
            self._save(record)

    def _still_approved(self, record: dict[str, Any]) -> bool:
        """Whether the experiment is still what was approved; if not, the failed run's
        record says why its failure is not repaired."""
        changed = self._departure()
        if changed is not None:
            self._note(
                record,
                f"The failure was not repaired: {changed}. Approve the experiment again to run it.",
            )
        return changed is None

    def _repaired(self, record: dict[str, Any]) -> bool:
        """Ask for corrected files after ``record`` failed and write them as the runner.
        True when the code changed, so that running again can end differently.

        The repair is part of the run the person approved, and only of that: when
        the experiment changed since (before the repairer was asked, or while it
        worked), nothing is written and the run ends as it failed.
        """
        assert self._repair is not None
        if not self._still_approved(record):
            return False
        self._owe_repair(record, True)
        changed = self._repair_code(record)
        if not changed:
            self._owe_repair(record, False)  # the failure stands: that is how it ended
        return changed

    def _repair_code(self, record: dict[str, Any]) -> bool:
        """Ask the repairer and write what it returns; true when the code changed."""
        assert self._repair is not None
        self._rounds += 1
        directory = _runs_dir(self.project, self.plan.id) / str(record["number"])
        before = code_files(self.project, self.plan.id)
        try:
            tail = "\n".join(_log_tail(directory, 200, secrets=self._mask))
            files = self._repair(self.plan, dict(record), tail)
            if not files or not self._still_approved(record):
                return False
            with self.project.locked():
                write_code(self.project, self.plan.id, files, author=_RUNNER)
                self._record_repair()
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

    def _record_repair(self) -> None:
        """Note on the page what the experiment is after the runner's repair, so that the
        attempt that follows (also one resumed by another worker) finds it approved."""
        digest = _digest_of(self._fields, _current_digest(self.project, self.plan.id))
        self.project.record_approval(self.page.path, {"repaired_digest": digest}, by=SYSTEM)

    # -- the whole of it ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        except BaseException as exc:
            self._abandon(exc)
            raise

    def _backend_for(self, record: Mapping[str, Any]) -> Any:
        return _backend_of_record(self._compute, self.plan, record)

    def _run(self) -> dict[str, Any]:
        settled = _settle_stale(
            self.project,
            self.page,
            self.plan,
            self._backend_for,
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
        # From here on a run is started: of the page as it is now, and only if that is
        # what was approved.  This holds for a resumed, a recovered and a repeated run.
        self._adopt(page)
        self._check_approval()
        self.backend = _backend_of(self._compute, self.plan)
        self.backend_name = str(getattr(self.backend, "name", "") or "")
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
        # It goes onto the record and into the page's state, which every agent reads: an
        # error may quote what it was given (the address of the repository, a key).
        said = _masked(f"{type(exc).__name__}: {exc}", self._mask)
        attempt = self._current
        if attempt is not None and attempt.record.get("status") == "running":
            outcome = self._abandoned(exc, requeue, said)
            try:
                self._count(_finish(self.project, self.page, attempt, outcome))
            except Exception:
                # Even that could not be written in full: the bare minimum, then.
                attempt.record.update(status=outcome.status, error=outcome.error, finished=_now())
                with contextlib.suppress(Exception):
                    attempt.save()
            _remove_copy(attempt.directory)
        elif attempt is not None:
            # The job had ended and its outcome was being written down when this came.
            # The outcome stands: it is saved, and the run is not started a second time.
            with contextlib.suppress(Exception):
                attempt.save()
            _remove_copy(attempt.directory)
        with contextlib.suppress(Exception):
            page = self.project.page(self.page.path)
            if _conclude(self.project, page) is None and page.state == "running":
                state = "queued" if requeue else "failed"
                _move(self.project, self.page.path, state, None if requeue else said)

    @staticmethod
    def _abandoned(exc: BaseException, requeue: bool, said: str) -> _Outcome:
        if getattr(exc, "unusable", False) is True:
            # (An agent that could not work says so in a sentence of its own: the
            # compute backend is not what failed then.)
            whole = getattr(exc, "whole_sentence", False) is True
            said = str(exc) if whole else f"The compute backend cannot be used: {exc}"
            return _Outcome("failed", said, "backend", requeued=True)
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
    unrepaired: str | Callable[[Mapping[str, Any]], str] | None = None,
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

    A run starts only if the experiment is still what the person approved: the
    :func:`plan_digest` of its code and run-defining fields is compared with the
    ``approved_digest`` on the page (or with ``repaired_digest``, which the run
    itself writes when the runner's repair changed the code).  This holds for a
    resumed, a recovered and a repeated run alike, and it is checked twice: on
    the page and code directory, and once more on the copy of the code that is
    about to run.  On a mismatch, or without a recorded approval, nothing runs
    and nothing is recorded as a run: the experiment goes back to ``planned``
    with a ``state_reason`` naming what changed (which field, with its values as
    quotations, or which file and who wrote it), and ``ProjectError`` says the
    same and to approve it again.

    One approval is one run.  A run takes the stamp of the approval it uses
    (``run.json["approval"]``); an experiment whose newest run is over, is not
    to be repeated by itself (``requeued``, or a repair still owed) and carries
    the stamp of the approval on the page is not started again, whatever state
    its page shows: the page is moved to how that run ended, the run gets its
    result page if it has none, and ``ProjectError`` says that it needs a new
    approval.  The end of a run is recorded on the page from ``approved`` and
    ``queued`` as well as from ``running``, so a page saved from an editor
    during the run does not read as waiting afterwards.

    ``compute`` is a ``ComputeBackend`` or a function ``(plan) -> ComputeBackend``.
    ``on_log`` gets every line of output, with the credentials of this process's
    environment and of the repository's URL blanked out; the backend is given the
    same strings to keep out of the log.  A run fails when the backend fails,
    when it exceeds its time limit (the page's, capped at the project's), exits
    with a non-zero status, or does not leave a valid ``metrics.json``.  When it
    failed for an error of its own code and ``repair(plan, run_record, log_tail)``
    returns corrected files (``{name: text}``), they are written as the runner,
    recorded with their diff under ``repairs`` of the next run, and the
    experiment runs again as that next run; at most ``limits.repair_rounds``
    times.  Without ``repair`` a failure stands; ``unrepaired`` is then why there
    is no repairer (no model can be used), noted on the record of a run that
    failed for an error of its own code.

    ``limits`` overrides ``experiment_seconds``, ``experiment_usd`` and
    ``repair_rounds`` of the project's settings for this call.  The time limit
    of a run is capped at ``experiment_seconds``.  ``experiment_usd`` is the
    project's total for all its runs: a run is refused (``error_kind``
    ``budget``, with the numbers) when its estimate, or what it could cost at
    its time limit, does not fit into what is left of it, which is worked out
    from the cost of the runs on record; so is a run on a billed backend whose
    GPU type has no price, because nothing could hold it to the limit.  A
    caller that keeps the project's
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
            page,
            compute,
            should_stop=should_stop,
            repair=repair,
            on_log=on_log,
            limits=limits,
            unrepaired=unrepaired,
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


def _spec_of_record(record: Mapping[str, Any], plan: ExperimentPlan) -> JobSpec:
    """The job of a run as this side recorded it when the run started (the ``plan`` of
    ``run.json``); where the record does not say, as the page's plan has it."""
    then = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    try:
        return JobSpec(
            command=[str(part) for part in then.get("command") or job_command(plan)],
            timeout=float(then.get("timeout_seconds") or plan.timeout_seconds),
            gpu=then.get("gpu", plan.gpu),
            network=bool(then["network"]) if "network" in then else plan.network,
            packages=tuple(then["packages"]) if "packages" in then else tuple(plan.packages),
        )
    except (TypeError, ValueError):
        return job_spec(plan)  # a record edited by hand


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

    The job is described to the backend by what this side recorded of it (the plan
    it was started with, and when), never by what the job could have written where
    it ran.  The person's cancel ends a job that is followed here as it ends one that
    was started here: it may run, and be billed, for hours yet.
    """
    handle = attempt.record.get("handle") or _read_handle(attempt.directory)
    if not handle:
        return None
    attempt.record["handle"] = handle
    flag = _flag_path(project, plan.id)
    secrets = attempt.secrets

    def stop() -> bool:
        return flag.exists() or (should_stop is not None and should_stop())

    def log(line: str) -> None:
        if on_log is not None:
            on_log(_masked(line, secrets))

    try:
        backend = backend_for(attempt.record)
        if backend is None or not _can_reattach(backend):
            return None
        attempt.backend = backend
        attempt.result = _call(
            backend.attach,
            handle,
            run_dir=attempt.directory,
            spec=_spec_of_record(attempt.record, plan),
            started=attempt.record.get("started"),
            on_log=log if on_log is not None else None,
            should_stop=stop,
            mask=tuple(secrets),
        )
    except ComputeError as exc:
        if exc.unusable:
            raise
        return _Outcome("failed", f"{_WORKER_STOPPED} {exc}", kind="worker")
    result = attempt.result
    if result.error:
        return _Outcome("failed", f"{_WORKER_STOPPED} {result.error}", kind="worker")
    return _judge(
        result, attempt.directory, plan, cancelled_by_person=flag.exists(), secrets=secrets
    )


def _clean_up(attempt: _Attempt, backend_for: Callable[[Mapping[str, Any]], Any]) -> list[str]:
    """End and remove what the job of a run left on this machine when its worker died:
    its processes and its working directory (through the backend that started it, which
    recorded them in ``job.json``), and the run's copy of the repository.  Returns notes
    for the run's record."""
    notes: list[str] = []
    # The file first: a backend may add to it after the record took its copy.
    job = _read_job(attempt.directory) or attempt.record.get("job")
    if isinstance(job, dict) and job:
        attempt.record["job"] = job
        try:
            end = getattr(backend_for(attempt.record), "end_job", None)
            if callable(end):
                notes += [str(note) for note in end(dict(job)) or []]
        except Exception as exc:  # cleaning up must not keep the run from being settled
            notes.append(
                "What the job left on this machine could not be cleaned up "
                f"({type(exc).__name__}: {exc}). Its record names the process session and "
                f"the working directory: session {job.get('session')}, {job.get('scratch')}."
            )
    _remove_copy(attempt.directory)
    return notes


def _settle_stale(
    project: Project,
    page: ResearchPage,
    plan: ExperimentPlan,
    backend_for: Callable[[Mapping[str, Any]], Any],
    *,
    should_stop: Callable[[], bool] | None = None,
    on_log: Callable[[str], None] | None = None,
    cancel: bool = False,
    follow: bool = True,
) -> list[dict[str, Any]]:
    """Settle the runs of an experiment that read ``running`` although nothing runs them,
    and the page of an experiment whose run is over although the page does not say so.

    The caller holds the experiment's run lock, which is what says that nothing
    runs it.  A remote job that outlived its worker is followed to its end; any
    other run is failed with "the worker stopped", and the experiment goes back
    to the queue unless that keeps happening.  With ``cancel`` the person has
    cancelled the experiment: the runs end ``cancelled`` instead.  Either way
    what the dead worker's job left on this machine is ended and removed, and a
    job whose cost never came back is charged for the time it can have run.
    Without ``follow`` a run that has a remote job is left as it is, and its
    experiment with it: settling it could mean waiting for hours.

    One approval is one run: where the newest run is over and used the approval
    on the page, the page is moved to the state that run left (and the run gets
    its result page, if it has none) instead of going back to the queue.
    """
    settled: list[dict[str, Any]] = []
    secrets = _secrets(project, plan)
    left_alone = False
    for _, directory in _run_dirs(project, plan.id):
        _remove_unrecorded(directory)
    for number, record in _numbered_runs(project, plan.id):
        if record.get("status") != "running":
            continue
        # The directory says which run it is, whatever a record edited by hand says.
        record.update(experiment=plan.id, number=number)
        directory = _runs_dir(project, plan.id) / str(number)
        if not follow and not cancel and (record.get("handle") or _read_handle(directory)):
            left_alone = True  # its remote job may be alive: for a worker to follow
            continue
        attempt = _Attempt(number, directory, record, plan=plan, secrets=secrets)
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
        attempt.notes += _clean_up(attempt, backend_for)
        billed = record.get("handle") or _own_file(directory, COST_FILE) is not None
        if attempt.backend is None and billed:
            with contextlib.suppress(Exception):
                attempt.backend = backend_for(record)  # to price a job that is gone
        settled.append(_finish(project, page, attempt, outcome))
    current = project.page(page.path)
    if cancel or left_alone or current.state not in ("approved", "queued", "running"):
        return settled
    if _conclude(project, current) is not None or current.state == "approved":
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


def _remove_unrecorded(directory: Path) -> None:
    """Remove a run directory that never became a run: made by a worker that was killed
    before it wrote the run's record.  It holds nothing, or only the copy of the code
    and a record that was still being written.  Anything else in a directory without a
    record (a log, results, a file someone put there) is left as it is."""
    with contextlib.suppress(OSError):
        if os.path.lexists(directory / RUN_FILE):
            return
        entries = list(directory.iterdir())
        litter = [entry for entry in entries if _TEMPORARY_RECORD.fullmatch(entry.name)]
        others = [entry for entry in entries if entry not in litter]
        if any(entry.name != CODE_DIR or not _real_dir(entry) for entry in others):
            return
        for entry in litter:
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
        for entry in others:
            shutil.rmtree(entry, ignore_errors=True)
        directory.rmdir()


def _stop_remote(attempt: _Attempt, backend_for: Callable[[Mapping[str, Any]], Any]) -> None:
    """End the remote job of a cancelled run whose worker is gone, where there is one:
    left alone it would run, and bill, until its own time limit."""
    handle = attempt.record.get("handle") or _read_handle(attempt.directory)
    if not handle:
        return
    with contextlib.suppress(Exception):
        backend = backend_for(attempt.record)
        if backend is not None and _can_reattach(backend):
            attempt.backend = backend
            spec = _spec_of_record(attempt.record, attempt.plan) if attempt.plan else None
            attempt.result = _call(
                backend.attach,
                handle,
                run_dir=attempt.directory,
                spec=spec,
                started=attempt.record.get("started"),
                should_stop=lambda: True,
                mask=tuple(attempt.secrets),
            )


class _Backends:
    """The backends recovery needs, built once each and closed afterwards."""

    def __init__(self, compute: Any) -> None:
        self._compute = compute
        self._built: dict[str, _Choice] = {}
        #: The plan of the experiment being settled, for a ``compute`` that chooses by plan.
        self.plan: ExperimentPlan | None = None

    def for_record(self, record: Mapping[str, Any]) -> Any:
        if self._compute is not None:
            return _backend_of_record(self._compute, self.plan, record)
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
    happens; neither does one whose run had ended without its page saying so
    (the page is moved to how the run ended).  :func:`withdraw` takes back an
    approval without cancelling anything.
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
            if _conclude(project, page) is not None:
                return  # its run was over already: the page now says how it ended
            plan = _read(project, page, cap=True).plan
            _settle_stale(project, page, plan, backends.for_record, cancel=True)
            project.set_state(
                page.path, "cancelled", by=HUMAN, reason="Cancelled at the person's request."
            )
        finally:
            backends.close()
            flag.unlink(missing_ok=True)


def withdraw(project: Project, ref: str) -> ResearchPage:
    """The person takes back the approval of a run that has not started; returns the
    page, ``planned`` again.

    Nothing is cancelled and nothing is recorded as a run: the experiment is simply
    a plan again, to be approved when the person wants it run.  ``ProjectError``
    for an experiment that is not ``approved`` (a queued or running one is ended
    with :func:`request_cancel`), and for one whose page only reads that way: a
    process is running it at this moment, or the run it was approved for is over
    (the page is then moved to how that run ended, as :func:`recover` would).
    """
    page = _experiment(project, ref)
    cancel = " Cancel its run instead (`paper-orchestra research cancel`, or Cancel in the app)."
    if page.state == "approved":
        with _RunLock(project.experiment_dir(str(page.id))) as lock:
            if not lock.acquired:
                # The page was saved from an editor while the run was under way.
                raise ProjectError(
                    f"{page.id} is being run right now, so its approval is in use and "
                    f"cannot be withdrawn.{cancel}"
                )
            page = _experiment(project, page.path)
            if page.state == "approved" and _conclude(project, page) is None:
                return project.set_state(
                    page.path, "planned", by=HUMAN, reason="You withdrew the approval of its run."
                )
            page = _experiment(project, page.path)  # as its run, long over, left it
    how = cancel if page.state in ("queued", "running") else ""
    raise ProjectError(f"{page.id} is {page.state}; there is no approval to withdraw.{how}")


def recover(
    project: Project,
    *,
    compute: Any = None,
    should_stop: Callable[[], bool] | None = None,
    on_log: Callable[[str], None] | None = None,
    follow: bool = True,
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

    What the dead worker's job left on this machine is ended and removed as its
    run is settled: the processes of the job's session and its working directory
    (through the backend that started it, from the ``job`` its record holds) and
    the run's copy of the repository.  A remote job that is gone without having
    reported its cost is charged for the time since it started, at most its time
    limit plus ``REMOTE_MARGIN_SECONDS``, so that the project's budget counts it
    before the experiment is started again.

    One approval is one run.  An experiment whose page reads ``approved``,
    ``queued`` or ``running`` although the run it was approved for is over (the
    worker stopped or died after the run's record was written and before the
    page moved, or the page was saved from an editor meanwhile) is not put back
    in the queue: its page is moved to how that run ended, and the run gets the
    result page it is owed.

    With ``follow=False`` only what needs no waiting is settled: a run with a
    remote job is left at ``running`` for a worker to follow, and nothing is
    started or waited for.  That is the call for a process that must not block
    (the web server, after it saw a worker die).

    ``compute`` is what :func:`run` takes; without it the backend named in a
    run's record is built with its defaults when a remote job has to be picked
    up.  Experiments that a live process is running are left alone.
    """
    settled: list[dict[str, Any]] = []
    backends = _Backends(compute)
    try:
        waiting = project.pages(kind="experiment", state=("approved", "queued", "running"))
        for page in waiting:
            if not page.id:
                continue
            if page.state == "approved" and not _run_dirs(project, page.id):
                continue  # approved and never run: nothing a worker could have left
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
                    follow=follow,
                )
    finally:
        backends.close()
    return settled


# ---------------------------------------------------------------------------
# The numbers of a finding
# ---------------------------------------------------------------------------

#: The kinds of evidence a number can be found in, in the order they are given.
EVIDENCE_KINDS: tuple[str, ...] = ("metric", "source", "input")
#: The text that takes the place of a number of a finding that was found nowhere, where
#: the finding is handed on (the paper's materials).
NOT_ON_RECORD = "[a number not on record]"
#: A number as a text writes it: a minus sign where one can stand (not the hyphen of
#: ``0.1-0.2`` or ``E1-run1``), the digits as ``leakscan.NUMBER`` reads them, and an
#: exponent (``1.5e-9``, ``2.5 x 10^-7``).  Sign and exponent are part of the number:
#: ``-0.121`` is not 0.121 and ``1.5e-9`` is not 1.5.  The full stop that ends a sentence
#: ends the number too (``... was 1.5e-9.``); only a decimal after the exponent
#: (``1.5e-9.2``) says that it was no exponent.
_QUOTED_NUMBER = re.compile(
    rf"(?P<sign>(?<![\w.)\]%])[-\u2212])?(?P<digits>{NUMBER.pattern})"
    r"(?:[eE](?P<exp>[+\-\u2212]?\d{1,3})(?!\w|\.\d)"
    r"|\s*(?:\u00d7|\u00b7|\u22c5|x|\*|\\times|\\cdot)\s*10\s*\^\s*\{?\s*"
    r"(?P<pow>[+\-\u2212]?\d{1,3})\s*\}?)?"
)


def _plain_decimal(value: Decimal) -> str:
    """``0.0015`` for ``1.5E-3``: the canonical way a value is written here, as plain
    digits without a sign, an exponent, a thousands separator or zeros that say nothing."""
    whole, _, fraction = format(abs(value), "f").partition(".")
    whole, fraction = whole.lstrip("0") or "0", fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


@dataclass(frozen=True)
class _Quoted:
    """One number as a text writes it, and what it can stand for."""

    #: As written, sign and exponent included.
    raw: str
    #: Whether it is written with a minus sign.
    negative: bool
    #: The value in plain digits, without its sign; then, for ``12,345``, the items of
    #: the list it may also be.
    readings: tuple[str, ...]
    #: Whether it is written with an exponent.  Such a number says its own magnitude:
    #: it is not also read as a percentage of a fraction (``2.5e-9`` is not 2.5e-7).
    scaled: bool = False

    @property
    def value(self) -> str:
        return self.readings[0]

    @property
    def key(self) -> str:
        return f"-{self.value}" if self.negative else self.value


def _quoted_spans(text: str) -> Iterator[tuple[_Quoted, int, int]]:
    """Every number in ``text``, in order, each as it is written, with where it stands."""
    for match in _QUOTED_NUMBER.finditer(text):
        readings = number_readings(match.group("digits"))
        power = (match.group("exp") or match.group("pow") or "").replace("\u2212", "-")
        if power:
            try:
                readings = [_plain_decimal(Decimal(readings[0]).scaleb(int(power)))]
            except (InvalidOperation, ValueError):
                continue
        negative = bool(match.group("sign")) and readings[0] != "0"
        quoted = _Quoted(match.group(0), negative, tuple(readings), scaled=bool(power))
        yield quoted, match.start(), match.end()


def _quoted_numbers(text: str) -> Iterator[_Quoted]:
    return (quoted for quoted, _, _ in _quoted_spans(text))


def _numbers_in(text: str) -> frozenset[str]:
    """The numbers of a piece of evidence, in plain digits; a negative one with its sign."""
    found: set[str] = set()
    for quoted in _quoted_numbers(text):
        found.add(quoted.key)
        found.update(quoted.readings[1:])
    return frozenset(found)


class _Index:
    """The numbers of every piece of evidence, indexed once for a check: which pieces
    hold a given number as written, rounded the way a person rounds when quoting, or
    as a percentage of a fraction.

    One index over all the evidence instead of one search per piece: a finding with
    thousands of numbers is checked against dozens of records in one pass.
    """

    def __init__(self, evidence: Sequence[tuple[str, str, frozenset[str]]]) -> None:
        #: In the order the evidence was given: results, then sources, then inputs.
        self.paths = [path for path, _, _ in evidence]
        #: ``metric``, ``source`` or ``input`` for each of them.
        self.kinds = [kind for _, kind, _ in evidence]
        self._exact: dict[str, list[int]] = {}
        self._where: dict[float, list[int]] = {}
        for position, (_, _, numbers) in enumerate(evidence):
            for number in numbers:
                self._exact.setdefault(number, []).append(position)
                with contextlib.suppress(ValueError):
                    self._where.setdefault(float(number), []).append(position)
        self._values = sorted(self._where)
        self._any_negative = bool(self._values) and self._values[0] < 0
        #: value -> (the value as a decimal, how many decimal places it is written with).
        self._decimals: dict[float, tuple[Decimal, int]] = {}
        self._per_piece: list[list[float]] | None = None

    def holders(self, quoted: _Quoted) -> list[int]:
        """The positions of the pieces of evidence that hold the number.

        A number written with a minus sign is held only as a negative value.  One
        written without is the value or its size: "a drop of 0.25" quotes a recorded
        change of -0.25.  ``12,345`` is one number, or the two items of a list.  One
        written with an exponent is held at that magnitude only.
        """
        as_percent = not quoted.scaled
        if quoted.negative:
            return sorted(self._holding(quoted.key, as_percent))
        found = self._either_sign(quoted.value, as_percent)
        if quoted.readings[1:]:
            found |= set.intersection(*(self._either_sign(part) for part in quoted.readings[1:]))
        return sorted(found)

    def found_in(self, number: str, parts: Sequence[str] = ()) -> list[str]:
        """The pieces of evidence that hold ``number`` (in plain digits, with its sign if
        it is negative), or every one of ``parts``."""
        quoted = _Quoted(number, number.startswith("-"), (number.lstrip("-"), *parts))
        return [self.paths[position] for position in self.holders(quoted)]

    def _either_sign(self, number: str, as_percent: bool = True) -> set[int]:
        holders = self._holding(number, as_percent)
        if self._any_negative and number != "0":
            holders |= self._holding(f"-{number}", as_percent)
        return holders

    def _holding(self, number: str, as_percent: bool = True) -> set[int]:
        """The pieces that hold ``number``: as written, rounded, and (``as_percent``) with
        the decimal point moved two places either way."""
        holders = set(self._exact.get(number, ()))
        try:
            value = Decimal(number)
        except InvalidOperation:
            return holders
        for scaled in (value, value / 100, value * 100) if as_percent else (value,):
            if len(holders) == len(self.paths):
                break  # everything holds it already
            # The same value written another way (51.25% of a fraction 0.5125), or one
            # with more digits that rounds to it.
            holders.update(self._where.get(float(scaled), ()))
            self._add_rounding_to(scaled, holders)
        return holders

    def _rounds(self, candidate: float, value: Decimal, places: int, half: Decimal) -> bool:
        """Whether ``candidate`` is written with more digits than ``value`` and rounds to it."""
        known = self._decimals.get(candidate)
        if known is None:
            exact = Decimal(repr(candidate))
            known = (exact, -int(exact.normalize().as_tuple().exponent))
            self._decimals[candidate] = known
        return known[1] > places and abs(known[0] - value) <= half

    def _pieces(self) -> list[list[float]]:
        """The values of each piece of evidence on its own, sorted; made when first needed."""
        if self._per_piece is None:
            per_piece: list[list[float]] = [[] for _ in self.paths]
            for value in self._values:
                for position in self._where[value]:
                    per_piece[position].append(value)
            self._per_piece = per_piece
        return self._per_piece

    def _add_rounding_to(self, value: Decimal, holders: set[int]) -> None:
        """Add to ``holders`` the pieces that hold a number which, written with more
        digits, rounds to ``value``.

        The values in reach are walked across all the evidence at once.  Where there
        are very many of them (dense evidence, a number of few digits), the walk stops
        and each piece not yet found is asked on its own, so that the work for one
        number stays bounded however much evidence there is.
        """
        value = value.normalize()
        places = max(0, -int(value.as_tuple().exponent))
        half = Decimal(5).scaleb(-(places + 1))
        low, high = float(value - half) - 1e-12, float(value + half) + 1e-12
        values = self._values
        position = bisect.bisect_left(values, low)
        stop = position + MAX_EVIDENCE_WALK
        while position < len(values) and values[position] <= high:
            if position == stop or len(holders) == len(self.paths):
                break
            if self._rounds(values[position], value, places, half):
                holders.update(self._where[values[position]])
            position += 1
        else:
            return  # everything in reach was looked at
        for piece, own in enumerate(self._pieces()):
            at = bisect.bisect_left(own, low) if piece not in holders else len(own)
            while at < len(own) and own[at] <= high:
                if self._rounds(own[at], value, places, half):
                    holders.add(piece)
                    break
                at += 1


#: The numbers of a piece of evidence, by what identifies its content: a record's path
#: and version, a run record's or an input file's path, time of change and size.  A
#: finding is checked on every look at its page, three times in an analysis and once
#: more in a write-up; what did not change is not read and searched again.
_NUMBERS_KEPT: dict[tuple[Any, ...], frozenset[str]] = {}
#: The index last built for a project, with the evidence it was built from.
_INDEX_KEPT: dict[str, tuple[tuple[tuple[Any, ...], ...], _Index]] = {}
_KEPT_GUARD = threading.Lock()


def _numbers_of(key: tuple[Any, ...], text: Callable[[], str]) -> frozenset[str]:
    """The numbers in a piece of evidence; ``text`` is only called when ``key`` is new."""
    with _KEPT_GUARD:
        known = _NUMBERS_KEPT.get(key)
    if known is not None:
        return known
    numbers = _numbers_in(text())
    with _KEPT_GUARD:
        if len(_NUMBERS_KEPT) >= MAX_EVIDENCE_KEPT:
            _NUMBERS_KEPT.clear()  # rare, and cheaper than keeping an order
        _NUMBERS_KEPT[key] = numbers
    return numbers


def _measured(body: str) -> str:
    """The values of the metrics table of a result page, one per line: what the page says
    the run measured.  Nothing else on the page counts as a result: not its first line
    with the plan's title, not an error, a repair's diff or the log, which hold whatever
    a plan or a job chose to say.  (Read only where the run's own record is not on file
    any more.)"""
    values: list[str] = []
    fence: str | None = None
    in_metrics = False
    for line in body.splitlines():
        ticks = re.match(r"`{3,}", line)
        if fence is not None:
            fence = None if line.strip() == fence else fence
            continue
        if ticks:
            fence = ticks.group()  # a block of quoted text: an error, a diff, the log
            continue
        if line.startswith("## "):
            if in_metrics:
                break
            in_metrics = line.strip() == "## Metrics"
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
        if not in_metrics or len(cells) != 2:
            continue
        written = cells[1].split(" (", 1)[0]  # `1e-05 (0.00001)`: as written, then in digits
        with contextlib.suppress(ValueError):
            float(written)  # true/false and texts are not measurements
            values.append(written)
    return "\n".join(values)


def _metric_values(record: Mapping[str, Any]) -> str:
    """The measurements of a run record, one per line, as the run wrote them.  A metric
    that is true/false or a text is not a measurement."""
    metrics = record.get("metrics")
    written = record.get("metrics_text")
    written = written if isinstance(written, dict) else {}
    lines: list[str] = []
    for name, value in metrics.items() if isinstance(metrics, dict) else ():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        text = written.get(name)
        lines.append(text if isinstance(text, str) else json.dumps(value))
    return "\n".join(lines)


def _run_of_result(project: Project, page: ResearchPage) -> tuple[str, int] | None:
    """``("E1", 3)`` for the result record ``wiki/results/E1-run3.md``."""
    match = re.fullmatch(r"(E[0-9]{1,9})-run([0-9]{1,9})", PurePosixPath(page.path).stem)
    return (match.group(1), int(match.group(2))) if match else None


def _record_of_result(project: Project, page: ResearchPage) -> dict[str, Any] | None:
    """The run record (``run.json``) a result page was written from, where it is still
    on file: read inside the project, and not through a link."""
    named = _run_of_result(project, page)
    if named is None:
        return None
    directory = _runs_dir(project, named[0]) / str(named[1])
    if not _real_dir(directory) or not _in_project(project, directory):
        return None
    record = _read_own_json(directory, RUN_FILE)
    return record if isinstance(record, dict) else None


def _said_to_be_edited(page: ResearchPage) -> bool:
    """Whether the page itself says that someone other than pipeline code wrote it."""
    writer = str(page.meta.get("updated_by") or SYSTEM)
    return bool(page.meta.get(EDITED_BY_PERSON)) or actor_class_or_person(writer) != "code"


def actor_class_or_person(author: str) -> str:
    """``model.actor_class``, with an author that cannot be read taken for a person."""
    try:
        return actor_class(author)
    except ProjectError:
        return "person"


def edited_record(project: Project, page: ResearchPage | str) -> bool:
    """Whether a record (a result or a source page) is no longer only what code wrote.

    True when the person edited it through the project, and, for a result, when its
    text differs from the text the run's record says was written (an edit of the
    file itself).  The metrics of a run are always read from ``run.json``, so an
    edited record changes neither the number check nor a paper; this is for saying
    so where the page is shown.  ``False`` for a page that is not a record.
    """
    if not isinstance(page, ResearchPage):
        page = project.page(str(page))
    if not page.is_record:
        return False
    if _said_to_be_edited(page):
        return True
    record = _record_of_result(project, page) if page.kind == "result" else None
    digest = record.get("result_digest") if record else None
    return isinstance(digest, str) and digest != _page_digest(page)


def _run_evidence(
    project: Project, root: str, page: ResearchPage, plans: Mapping[str, ResearchPage]
) -> tuple[tuple[Any, ...], str, str, frozenset[str]] | None:
    """The measurements behind an accepted result record of a run that succeeded, or
    ``None`` when they are no evidence: the experiment was rejected or is gone, or the
    run's record says something else than the page."""
    owners = [project.resolve(target) for target in page.relations.get("run_of", [])]
    plan = next((plans[path] for path in owners if path in plans), None)
    if plan is None or plan.review == "rejected":
        return None
    record = _record_of_result(project, page)
    if record is not None:
        if record.get("status") != "succeeded":
            return None
        text = _metric_values(record)
        named = _run_of_result(project, page)
        file = _runs_dir(project, named[0]) / str(named[1]) / RUN_FILE if named else None
        try:
            status = file.stat() if file else None
        except OSError:
            status = None
        version = (status.st_mtime_ns, status.st_size) if status else (len(text),)
        key: tuple[Any, ...] = ("run", root, page.path, *version)
    elif _said_to_be_edited(page):
        return None  # its record is gone and the page is not what code wrote
    else:
        # Run directories may be cleared to save space; the page code wrote stands in.
        text = _measured(page.body)
        key = ("page", root, page.path, page.updated, len(text))
    return key, page.path, "metric", _numbers_of(key, lambda: text)


def _input_numbers(
    project: Project,
) -> Iterator[tuple[tuple[Any, ...], str, str, frozenset[str]]]:
    """``(key, path, kind, numbers)`` of the text files the person gave the project."""
    root = project.inputs_dir
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not (Path(folder) / name).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST_FILE or path.is_symlink() or not path.is_file():
                continue  # the manifest holds sizes and digests, not the person's material
            with contextlib.suppress(OSError, UnicodeDecodeError):
                status = path.stat()
                if status.st_size > MAX_EVIDENCE_BYTES:
                    continue
                key = ("file", str(path), status.st_mtime_ns, status.st_size)
                numbers = _numbers_of(key, lambda path=path: path.read_bytes().decode("utf-8"))
                yield key, f"{INPUTS_DIR}/{relative}", "input", numbers


def _evidence(project: Project) -> _Index:
    """Where a number of a finding may come from: the metrics of runs that succeeded, of
    experiments the person did not reject (read from the runs' own records), accepted
    source records, and the text files under ``inputs/``."""
    root = str(project.root.resolve())
    found: list[tuple[tuple[Any, ...], str, str, frozenset[str]]] = []
    plans = {page.path: page for page in project.pages(kind="experiment")}
    for page in project.pages(kind="result", review="accepted", state="succeeded"):
        entry = _run_evidence(project, root, page, plans)
        if entry is not None:
            found.append(entry)
    for page in project.pages(kind="source", review="accepted"):
        text = f"{page.title}\n{page.summary}\n{page.body}"
        key = ("page", root, page.path, page.updated, len(text))
        found.append((key, page.path, "source", _numbers_of(key, lambda text=text: text)))
    found += _input_numbers(project)
    keys = tuple((kind, *key) for key, _, kind, _ in found)
    with _KEPT_GUARD:
        kept = _INDEX_KEPT.get(root)
    if kept is not None and kept[0] == keys:
        return kept[1]
    index = _Index([(path, kind, numbers) for _, path, kind, numbers in found])
    with _KEPT_GUARD:
        if len(_INDEX_KEPT) >= 8:
            _INDEX_KEPT.clear()
        _INDEX_KEPT[root] = (keys, index)
    return index


def _is_reference(project: Project, text: str) -> bool:
    """Whether ``text`` is a reference to a page rather than text to check: one word
    (an id, a path, a link) that names a page of the project."""
    word = text.strip()
    return 0 < len(word) <= 300 and not re.search(r"\s", word) and project.resolve(word) is not None


def _checked_text(page: ResearchPage) -> str:
    """What is checked of a page: its title, the summary its author gave it, its text.
    (A summary that was only taken from the text can be a sentence cut short, and with
    it a number.)"""
    summary = page.meta.get("summary")
    return "\n".join([page.title, summary if isinstance(summary, str) else "", page.body])


def _look_up(
    project: Project, text: str, basis: Iterable[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """The check of every specific number of ``text``, and the numbers it did not look
    at because they are too unspecific, each as written, in order and once."""
    index = _evidence(project)
    own = {path for ref in basis if (path := project.resolve(str(ref)))}
    checks: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for quoted in _quoted_numbers(text):
        if quoted.key in seen:
            continue
        seen.add(quoted.key)
        if not is_identifying(quoted.value):
            skipped.append(quoted.raw)
            continue
        holders = index.holders(quoted)
        found_in = [index.paths[position] for position in holders]
        kinds = {index.kinds[position] for position in holders}
        checks.append(
            {
                "number": quoted.raw,
                "found_in": found_in,
                "in_basis": any(path in own for path in found_in),
                "kinds": [kind for kind in EVIDENCE_KINDS if kind in kinds],
            }
        )
    return checks, skipped


def _was(count: int) -> str:
    return f"{count} {'was' if count == 1 else 'were'}"


def _report_summary(checks: Sequence[Any], skipped: Sequence[Any], missing: int, far: int) -> str:
    """One sentence that says what the check looked at and what it found: what it
    proves, no more."""
    left = f" ({len(skipped)} here)" if skipped else ""
    tail = f"short and round numbers are not checked{left}."
    where = "in the run records, sources and your files"
    if not checks:
        return f"No number in it is specific enough to be looked for {where}; {tail}"
    looked = f"{len(checks)} number{'s' if len(checks) != 1 else ''}"
    parts = [f"{_was(missing)} not found" if missing else "all were found"]
    if far:
        parts.append(f"{_was(far)} found only outside the records this finding is based on")
    verb = "was" if len(checks) == 1 else "were"
    return f"{looked} {verb} looked for {where}: {', '.join(parts)}; {tail}"


def number_report(
    project: Project, page_or_text: ResearchPage | str, *, basis: Iterable[str] | None = None
) -> dict[str, Any]:
    """What the number check finds in a finding, or in any text, and what it did not
    look at::

        {"checked": [{"number": "0.91", "found_in": ["wiki/results/E1-run1.md"],
                      "in_basis": True, "kinds": ["metric"]}, ...],
         "skipped": ["5", "300"],
         "elsewhere": ["12.5"],
         "unverified": ["98765.4"],
         "summary": "4 numbers were looked for in the run records, ..."}

    ``page_or_text`` is a page, a reference to one (an id, a path), or text to
    check as it is: the text of a finding as it would read with a suggested edit
    applied, say.  ``basis`` names the records the text is based on (references to
    result records); for a page it defaults to the page's ``based_on``.

    ``checked`` is :func:`number_check`'s list.  ``skipped`` are the numbers that
    were not looked at because they are too unspecific to trace (small counts,
    round figures, years, one significant digit).  ``elsewhere`` are the numbers
    that were found, but in none of the records of the basis: in another run's
    metrics, a source or an input file.  ``unverified`` are those found nowhere.
    ``summary`` says all of it in one sentence, for whoever shows the report.

    The check proves that a number occurs on record, not that it is the number
    the sentence is about; ``in_basis`` and ``kinds`` say how close it came.
    """
    text = page_or_text
    if isinstance(text, str) and _is_reference(project, text):
        text = project.page(text.strip())
    if isinstance(text, ResearchPage):
        basis = text.relations.get("based_on", []) if basis is None else basis
        text = _checked_text(text)
    checks, skipped = _look_up(project, str(text), basis or ())
    unverified = [check["number"] for check in checks if not check["found_in"]]
    elsewhere = [c["number"] for c in checks if c["found_in"] and not c["in_basis"]]
    return {
        "checked": checks,
        "skipped": skipped,
        "elsewhere": elsewhere,
        "unverified": unverified,
        "summary": _report_summary(checks, skipped, len(unverified), len(elsewhere)),
    }


def edit_report(project: Project, proposal: Any) -> dict[str, Any] | None:
    """The :func:`number_report` of a finding as it would read with an edit proposal
    applied (its title, summary and text replaced where the proposal gives new ones, and
    the records it adds to ``based_on`` counted as its basis); ``None`` when the proposal
    is not about a finding or its page is gone.

    For whoever shows a suggested edit to the person: the numbers an edit brings are
    checked where the decision is taken, not after it.
    """
    try:
        page = project.page(str(proposal.target))
    except ProjectError:
        return None
    if page.kind != "finding":
        return None
    fields = proposal.fields if isinstance(proposal.fields, Mapping) else {}
    summary = fields.get("summary", page.meta.get("summary"))
    text = "\n".join(
        [
            str(fields.get("title") or page.title),
            summary if isinstance(summary, str) else "",
            page.body if proposal.body is None else str(proposal.body),
        ]
    )
    added = proposal.relations_add.get("based_on", []) if proposal.relations_add else []
    return number_report(project, text, basis=[*page.relations.get("based_on", []), *added])


def number_check(project: Project, page: ResearchPage | str) -> list[dict[str, Any]]:
    """Where each number in a finding comes from.

    For every number in the page's title, summary and text that is specific
    enough to be a result (the notion of ``memory.leakscan``: two or more
    significant digits; not a small count, a round figure or a year), in the
    order they appear and each once::

        {"number": "2.37", "found_in": ["wiki/results/E1-run1.md"],
         "in_basis": True, "kinds": ["metric"]}

    ``number`` is the number as the page writes it, with its sign and its
    exponent, which are part of it: ``-0.121`` is not found in a record of 0.121,
    nor ``1.5e-9`` in one of 1.5e-3.  (A number written without a sign is also
    found as the size of a negative value: "a drop of 0.25".)

    ``found_in`` lists what holds it, ``kinds`` of what kind those are:
    ``metric`` (the metrics of a run that succeeded, of an experiment the person
    did not reject, read from the run's own ``run.json``: not the result page's
    text, the plan's title, the log, an error or a repair's diff, and nothing of
    a run that failed), ``source`` (an accepted source record) and ``input`` (a
    text file under ``inputs/``).  ``in_basis`` says whether one of them is a
    record the page names in ``based_on``.  A number is held as written, rounded
    to the digits the page gives (0.8765 is quoted as 0.88), or with the decimal
    point moved two places (0.913 and 91.3%; not a number written with an
    exponent, which says its own magnitude).  An empty ``found_in`` means that
    nothing on record holds the number.  ``page`` is a page or a reference to one.

    This is an occurrence check: it does not tie a number to the metric or the
    run its sentence names, and it does not look at short and round numbers at
    all (:func:`number_report` lists those).  The evidence is indexed once per
    call and kept between calls for as long as it does not change, so the check
    stays quick on a project with many records.
    """
    if not isinstance(page, ResearchPage):
        page = project.page(str(page))
    return _look_up(project, _checked_text(page), page.relations.get("based_on", []))[0]


def unverified_numbers(project: Project, page: ResearchPage | str) -> list[str]:
    """The numbers in a finding that no metric of a succeeded run, no source and no input
    file holds (see :func:`number_check`), as the page writes them."""
    return [check["number"] for check in number_check(project, page) if not check["found_in"]]


def reportable_text(project: Project, page: ResearchPage | str) -> tuple[str, str, list[str]]:
    """A finding as it may be handed on: ``(title, text, removed)``.

    Every number of the title and the text that the number check finds nowhere
    (:func:`unverified_numbers`) is replaced by ``NOT_ON_RECORD``, wherever it
    occurs; ``removed`` lists those numbers as the page writes them.  Whether a
    number that is on no record reaches a paper then no longer depends on a
    writer obeying a sentence.  The page itself is not changed.
    """
    if not isinstance(page, ResearchPage):
        page = project.page(str(page))
    checks = _look_up(project, _checked_text(page), page.relations.get("based_on", []))[0]
    removed = [check["number"] for check in checks if not check["found_in"]]
    gone = {quoted.key for number in removed for quoted in _quoted_numbers(number)}

    def without(text: str) -> str:
        pieces: list[str] = []
        at = 0
        for quoted, start, end in _quoted_spans(text):
            if quoted.key in gone:
                pieces += [text[at:start], NOT_ON_RECORD]
                at = end
        return "".join([*pieces, text[at:]])

    return without(page.title), without(page.body), removed


# ---------------------------------------------------------------------------
# What rests on an experiment
# ---------------------------------------------------------------------------

#: The states of a hypothesis that are a verdict on it.
_VERDICTS = ("supported", "refuted", "inconclusive")


def result_records(project: Project, ref: str) -> list[str]:
    """The paths of the result records of the experiment's runs, in page order."""
    page = _experiment(project, ref)
    return [
        result.path
        for result in project.pages(kind="result")
        if page.path in {project.resolve(target) for target in result.relations.get("run_of", [])}
    ]


def dependents(project: Project, ref: str) -> dict[str, Any]:
    """What rests on an experiment, for saying so when the person rejects it::

        {"experiment": "E1",
         "results": ["wiki/results/E1-run1.md"],
         "findings": [{"id": "F1", "path": ..., "title": ..., "review": "accepted",
                       "only": True}],
         "hypotheses": [{"id": "H1", "path": ..., "title": ..., "state": "supported",
                         "through": ["F1"]}]}

    ``findings`` are the findings (not rejected ones) whose ``based_on`` names a
    result record of this experiment; ``only`` says that every record it names is
    one of this experiment's, so that nothing else carries it.  ``hypotheses``
    are the hypotheses with a verdict (supported, refuted, inconclusive) that
    such a finding supports or refutes, with the findings the verdict came
    ``through``.  Nothing is changed: a finding and a verdict are the person's to
    revisit.  The records of a rejected experiment are no evidence for the
    number check, and a paper does not report a finding that rests on nothing else.
    """
    page = _experiment(project, ref)
    results = result_records(project, page.path)
    own = set(results)
    findings: list[dict[str, Any]] = []
    verdicts: dict[str, dict[str, Any]] = {}
    for finding in project.pages(kind="finding"):
        based = [project.resolve(target) for target in finding.relations.get("based_on", [])]
        if finding.review == "rejected" or not own.intersection(based):
            continue
        findings.append(
            {
                "id": finding.id,
                "path": finding.path,
                "title": finding.title,
                "review": finding.review,
                "only": all(path in own for path in based),
            }
        )
        judged = [*finding.relations.get("supports", []), *finding.relations.get("refutes", [])]
        for target in judged:
            path = project.resolve(target)
            hypothesis = project.page(path) if path else None
            if hypothesis is None or hypothesis.kind != "hypothesis":
                continue
            if hypothesis.state not in _VERDICTS or hypothesis.review == "rejected":
                continue
            entry = verdicts.setdefault(
                hypothesis.path,
                {
                    "id": hypothesis.id,
                    "path": hypothesis.path,
                    "title": hypothesis.title,
                    "state": hypothesis.state,
                    "through": [],
                },
            )
            if finding.id not in entry["through"]:
                entry["through"].append(finding.id)
    return {
        "experiment": str(page.id),
        "results": results,
        "findings": findings,
        "hypotheses": list(verdicts.values()),
    }
