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

Three rules shape all of it.  Nothing here approves anything: only :func:`approve`
moves an experiment to ``approved``, and it acts for the person.  An approval is
a record of exactly what may run (:func:`plan_digest`), and a run starts only
while the experiment still is that.  And nothing here is written by a model:
``run.json`` and the result page are made by code from what the run produced,
with the values as the run wrote them.
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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
    stamp_after,
    with_article,
)
from paper_orchestra.research.project import EXPERIMENTS_DIR, INPUTS_DIR, MANIFEST_FILE, Project
from paper_orchestra.utils.common import write_json, write_text

__all__ = [
    "ExperimentPlan",
    "approval",
    "approve",
    "check_code",
    "code_authors",
    "code_digest",
    "code_dir",
    "code_files",
    "job_command",
    "job_spec",
    "load_plan",
    "mask_credentials",
    "number_check",
    "options",
    "plan_digest",
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
_RUN_NUMBER = re.compile(r"[0-9]{1,9}")
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
    project: Project, experiment_id: str, digest: str, stamp: str, fields: Mapping[str, Any]
) -> None:
    record = {
        "digest": digest,
        "at": stamp,
        "plan": {name: _shown_field(name, value) for name, value in fields.items()},
        "code": _file_digests(code_dir(project, experiment_id)),
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
        f"{field}: {then[field]} when approved, now {_shown_field(field, value)}"
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
        _write_files(directory, checked, authors, author, experiment_id)
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
) -> None:
    """Write the checked ``files``, keeping each version that is replaced and noting who
    wrote the new one."""
    stamp = _now()
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
) -> ResearchPage:
    """The person approves a run of the experiment; returns its page, now ``approved``.

    This is the person's decision and nothing else calls it for them: it is what
    the command line and the web app do when the person says so, and it is the
    only way an experiment becomes ``approved``.

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
    backend cannot run it (``ComputeBackend.supports`` and ``check``), or when
    it could cost more than is left of what the project allows for all its runs
    together.  ``base_updated`` is the page's ``updated`` stamp as the person
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
    if page.state in ("queued", "running"):
        raise ProjectError(
            f"{name} is {page.state}. Cancel that run, or wait for it, before approving another."
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
    approved = _record_approval(project, page, reading.plan, read, base_updated)
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
) -> ResearchPage:
    """Write what was approved onto the page and move it to ``approved``, as one step.

    ``page`` and ``code`` are what the checks of the approval looked at.  Under the
    project's lock, which every write of a page and of code takes, they are compared
    with what is there now: a change that landed meanwhile is a ``Conflict`` and
    nothing is approved.
    """
    name = str(page.id)
    stamp = _now()
    fields = _run_fields(plan)
    digest = _digest_of(fields, code)
    with project.locked():
        current = project.page(page.path)
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
            },
            by=HUMAN,
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
        _write_approval(project, name, digest, stamp, fields)
        return project.set_state(page.path, "approved", by=HUMAN)


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
    #: The plan the run was made from and the backend that ran it, where they are known:
    #: what a job costs whose result never came back is worked out from them.
    plan: ExperimentPlan | None = None
    backend: Any = None
    #: Strings that must not be written into the record, the result page or an event.
    secrets: Sequence[str] = ()

    def save(self) -> None:
        write_json(self.directory / RUN_FILE, self.record)


@dataclass
class _Outcome:
    """What became of a run."""

    status: str
    error: str | None = None
    #: Why it failed: ``exit``, ``metrics``, ``code``, ``patch`` (the code's own, which a
    #: repair can cure), ``command`` (it could not be started), ``timeout``, ``backend``,
    #: ``budget``, ``setup``, ``approval`` (the code changed between the check of the
    #: approval and the copy that would have run), ``worker``, ``internal``.
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
    try:
        with open(directory / LOG_NAME, "rb") as stream:
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
    try:
        return (directory / HANDLE_NAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _read_job(directory: Path) -> dict[str, Any] | None:
    """What the backend recorded of a job it started on this machine (``job.json``)."""
    job = _read_json(directory / JOB_FILE)
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
    on_file = _read_json(attempt.directory / COST_FILE)
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
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    if plan.get("command"):
        # What was executed, and with what reach: the code digest alone does not say.
        rows.append(("Command", _plain(shlex.join(map(str, plan["command"])))))
    if "network" in plan:
        rows.append(("Network access", "allowed" if plan["network"] else "not allowed"))
    if record.get("handle"):
        rows.append(("Remote job", f"`{record['handle']}`"))
    if record.get("code_digest"):
        rows.append(("Code", f"sha256 `{str(record['code_digest'])[:12]}`"))
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


def _metrics_section(record: Mapping[str, Any], directory: Path, room: int) -> list[str]:
    metrics = record.get("metrics") or {}
    if not metrics:
        if record.get("status") != "succeeded" and (directory / "metrics.json").is_file():
            return [
                "The run wrote metrics.json, but its values are not reported here: the run "
                "did not finish, so they are not a result."
            ]
        return ["The run reported no metrics."]
    written = record.get("metrics_text") or {}
    rows = []
    for name, value in metrics.items():
        text = str(written.get(name, json.dumps(value, ensure_ascii=False)))
        rows.append(f"| {_inert(name)} | {_inert(text)}{_decimal_form(text)} |")
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
    cloned = _clone_notes(record, attempt.directory)
    notes = [*attempt.notes, *cloned, *(result.notes if result else []), *dropped]
    record.update(
        status=outcome.status,
        finished=_now(),
        seconds=round(seconds, 3) if seconds is not None else None,
        exit_code=result.returncode if result is not None else None,
        timed_out=bool(result is not None and result.timed_out),
        metrics=outcome.metrics,
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
    if outcome.metrics_text:
        record["metrics_text"] = outcome.metrics_text
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
    return page


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
    try:
        with open(directory / LOG_NAME, "rb") as stream:
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
        if reason is None:
            return
        _move(
            self.project,
            self.page.path,
            "planned",
            f"Not run: {reason}. Approve it again.",
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
        record["code_digest"] = code_digest(self.project, self.plan.id)
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
        # The copy is what runs.  It is held against the approval once more: a write
        # that landed between the look at the code directory and the copy is not run.
        changed = self._departure(record["code_digest"])
        if changed is not None:
            raise _Setup(
                "approval",
                f"The run was not started: {changed}. Look at the experiment as it is now "
                f"and approve it again ({_HOW_TO_APPROVE}).",
            )
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
            shutil.rmtree(attempt.directory / REPO_COPY, ignore_errors=True)
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

    A run starts only if the experiment is still what the person approved: the
    :func:`plan_digest` of its code and run-defining fields is compared with the
    ``approved_digest`` on the page (or with ``repaired_digest``, which the run
    itself writes when the runner's repair changed the code).  This holds for a
    resumed, a recovered and a repeated run alike.  On a mismatch, or without a
    recorded approval, nothing runs: the experiment goes back to ``planned`` with
    a ``state_reason`` naming what changed (which field, or which file and who
    wrote it), and ``ProjectError`` says the same and to approve it again.

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
    shutil.rmtree(attempt.directory / REPO_COPY, ignore_errors=True)
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
) -> list[dict[str, Any]]:
    """Settle the runs of an experiment that read ``running`` although nothing runs them.

    The caller holds the experiment's run lock, which is what says that nothing
    does.  A remote job that outlived its worker is followed to its end; any
    other run is failed with "the worker stopped", and the experiment goes back
    to the queue unless that keeps happening.  With ``cancel`` the person has
    cancelled the experiment: the runs end ``cancelled`` instead.  Either way
    what the dead worker's job left on this machine is ended and removed, and a
    job whose cost never came back is charged for the time it can have run.
    """
    settled: list[dict[str, Any]] = []
    secrets = _secrets(project, plan)
    for number, record in _numbered_runs(project, plan.id):
        if record.get("status") != "running":
            continue
        # The directory says which run it is, whatever a record edited by hand says.
        record.update(experiment=plan.id, number=number)
        directory = _runs_dir(project, plan.id) / str(number)
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
        if attempt.backend is None:
            with contextlib.suppress(Exception):
                attempt.backend = backend_for(record)  # to price a job that is gone
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

    What the dead worker's job left on this machine is ended and removed as its
    run is settled: the processes of the job's session and its working directory
    (through the backend that started it, from the ``job`` its record holds) and
    the run's copy of the repository.  A remote job that is gone without having
    reported its cost is charged for the time since it started, at most its time
    limit plus ``REMOTE_MARGIN_SECONDS``, so that the project's budget counts it
    before the experiment is started again.

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


class _Index:
    """The numbers of every piece of evidence, indexed once for a check: which pieces
    hold a given number as written, rounded the way a person rounds when quoting, or
    as a percentage of a fraction.

    One index over all the evidence instead of one search per piece: a finding with
    thousands of numbers is checked against dozens of records in one pass.
    """

    def __init__(self, evidence: Sequence[tuple[str, frozenset[str]]]) -> None:
        #: In the order the evidence was given: results, then sources, then inputs.
        self.paths = [path for path, _ in evidence]
        self._exact: dict[str, list[int]] = {}
        self._where: dict[float, list[int]] = {}
        for position, (_, numbers) in enumerate(evidence):
            for number in numbers:
                self._exact.setdefault(number, []).append(position)
                with contextlib.suppress(ValueError):
                    self._where.setdefault(float(number), []).append(position)
        self._values = sorted(self._where)
        #: value -> (the value as a decimal, how many decimal places it is written with).
        self._decimals: dict[float, tuple[Decimal, int]] = {}
        self._per_piece: list[list[float]] | None = None

    def found_in(self, number: str, parts: Sequence[str] = ()) -> list[str]:
        """The pieces of evidence that hold ``number``, or every one of ``parts``
        (``12,345`` is one number, or the two items of a list)."""
        holders = self._holding(number)
        if parts:
            holders |= set.intersection(*(self._holding(part) for part in parts))
        return [self.paths[position] for position in sorted(holders)]

    def _holding(self, number: str) -> set[int]:
        holders = set(self._exact.get(number, ()))
        try:
            value = Decimal(number)
        except InvalidOperation:
            return holders
        for scaled in (value, value / 100, value * 100):
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
#: and version, an input file's path, time of change and size.  A finding is checked on
#: every look at its page, three times in an analysis and once more in a write-up; what
#: did not change is not read and searched again.
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
    numbers = frozenset(numbers_in(text()))
    with _KEPT_GUARD:
        if len(_NUMBERS_KEPT) >= MAX_EVIDENCE_KEPT:
            _NUMBERS_KEPT.clear()  # rare, and cheaper than keeping an order
        _NUMBERS_KEPT[key] = numbers
    return numbers


def _measured(body: str) -> str:
    """The values of the metrics table of a result page, one per line: what the run
    measured.  Nothing else on the page counts as a result: not its first line with the
    plan's title, not an error, a repair's diff or the log, which hold whatever a plan
    or a job chose to say."""
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
            values.append(cells[1])
    return "\n".join(values)


def _input_numbers(project: Project) -> Iterator[tuple[tuple[Any, ...], str, frozenset[str]]]:
    """``(key, path, numbers)`` of the text files the person gave the project."""
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
                yield key, f"{INPUTS_DIR}/{relative}", numbers


def _evidence(project: Project) -> _Index:
    """Where a number of a finding may come from: the metrics of runs that succeeded (as
    their accepted result records give them), accepted source records, and the inputs."""
    root = str(project.root.resolve())
    found: list[tuple[tuple[Any, ...], str, frozenset[str]]] = []
    for page in project.pages(kind=("result", "source"), review="accepted"):
        if page.kind == "result" and page.state != "succeeded":
            continue  # what a run that failed or was cancelled wrote is not a result
        if page.kind == "result":
            text = _measured(page.body)
        else:
            text = f"{page.title}\n{page.summary}\n{page.body}"
        key = ("page", root, page.path, page.updated, len(text))
        found.append((key, page.path, _numbers_of(key, lambda text=text: text)))
    found.sort(key=lambda item: not item[1].startswith("wiki/results/"))
    found += _input_numbers(project)
    keys = tuple(key for key, _, _ in found)
    with _KEPT_GUARD:
        kept = _INDEX_KEPT.get(root)
    if kept is not None and kept[0] == keys:
        return kept[1]
    index = _Index([(path, numbers) for _, path, numbers in found])
    with _KEPT_GUARD:
        if len(_INDEX_KEPT) >= 8:
            _INDEX_KEPT.clear()
        _INDEX_KEPT[root] = (keys, index)
    return index


def number_check(project: Project, page: ResearchPage | str) -> list[dict[str, Any]]:
    """Where each number in a finding comes from.

    For every number in the page's text that is specific enough to be a result
    (the notion of ``memory.leakscan``: two or more significant digits; not a
    small count, a round figure or a year), in the order they appear and each
    once::

        {"number": "2.37", "found_in": ["wiki/results/E1-run1.md"]}

    ``number`` is the number as the page writes it.  ``found_in`` lists what
    holds it: accepted result records of runs that succeeded, accepted source
    records and text files under ``inputs/``.  Of a result record only the
    metrics count (the values of its metrics table, which are the run's own
    ``metrics.json``): not the plan's title, the log, an error or a repair's
    diff, and nothing of a run that failed.  A number is held as written,
    rounded to the digits the page gives (0.8765 is quoted as 0.88), or as a
    percentage of a fraction.  An empty list means that nothing on record
    supports the number.  ``page`` is a page or a reference to one.

    The evidence is indexed once per call and kept between calls for as long as
    it does not change, so the check stays quick on a project with many records.
    """
    if not isinstance(page, ResearchPage):
        page = project.page(str(page))
    index = _evidence(project)
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in NUMBER.findall(page.body):
        readings = number_readings(raw)
        number, parts = readings[0], readings[1:]
        if number in seen or not is_identifying(number):
            continue
        seen.add(number)
        checks.append({"number": raw, "found_in": index.found_in(number, parts)})
    return checks


def unverified_numbers(project: Project, page: ResearchPage | str) -> list[str]:
    """The numbers in a finding that no metric of a succeeded run, no source and no input
    file holds (see :func:`number_check`), as the page writes them."""
    return [check["number"] for check in number_check(project, page) if not check["found_in"]]
