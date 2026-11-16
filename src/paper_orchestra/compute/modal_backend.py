"""Run model-written Python, LaTeX builds and experiment jobs in Modal Sandboxes.

Model-written figure code is untrusted, so it never runs next to credentials.
A backend owns up to two Sandboxes (one for Python jobs, one for LaTeX builds).
Each is created on first use with outbound networking blocked, no Secrets and
no Modal identity token, is reused for later jobs (every job gets its own
scratch directory) and is replaced transparently once it has expired or been
collected for idleness.

An experiment job (``run_job``) gets a Sandbox of its own instead: built from
the Python image plus the packages the job asks for, with the GPU it asks for,
and with the network blocked unless the job asks for it.  The job runs inside
the Sandbox independently of this process: its output goes to a file there,
which this process follows, and how it ended is recorded there too.  That is
what lets ``attach`` pick a job up after the process that started it has died.
The Sandbox is terminated as soon as ``run_job`` or ``attach`` returns or
raises; one whose orchestrator was killed ends by itself, at the latest
``idle_timeout`` seconds after its job has finished.

A job is followed, fetched and billed by what this process knows about it: the
``JobSpec`` and, next to the run, ``cost.json``.  That file holds the running
estimate of what the Sandbox has cost from the moment it exists, and the final
one once it has ended, so that a run whose result never reached its caller (it
was interrupted, or its worker died) still has a cost.  Nothing that decides
anything is read back from a file inside the Sandbox: the job could have
rewritten it.

``latex="local"`` keeps the Sandbox for Python but compiles with the TeX Live
install of the current machine.  ``paper_orchestra.modal_app`` offers that mode
(``PAPER_ORCHESTRA_MODAL_LATEX=local``) for speed; by default it compiles in a
Sandbox too, away from the credentials of its own container.

``PAPER_ORCHESTRA_MODAL_TEX_PACKAGES`` names extra Debian packages (separated
by spaces) to install in the TeX Live images, for templates that need more
than the list below.

The ``modal`` package is imported inside functions so that importing this
module, and the rest of the package, works without the ``modal`` extra.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from paper_orchestra.compute.base import (
    COST_NAME,
    HANDLE_NAME,
    LOG_NAME,
    BackendUnusable,
    CompileResult,
    ComputeBackend,
    ComputeError,
    ExecResult,
    JobLog,
    JobResult,
    JobSpec,
    JobStopped,
    SourceReport,
    choose_outputs,
    delivered_lines,
    human_size,
    leftover_notes,
    open_run,
    output_of,
    read_record,
    refuse_unsafe_document,
    safe_relative,
    scan_outputs_script,
    spec_problems,
    supervise,
    walk_source,
    write_record,
)
from paper_orchestra.compute.local import (
    DirectoryInUse,
    LocalBackend,
    holding,
    is_own_directory,
    lock_directory,
    remove_tree,
)
from paper_orchestra.utils.latex import latex_commands, summarize_build

if TYPE_CHECKING:
    from types import ModuleType

    import modal

log = logging.getLogger(__name__)

APP_ENV = "PAPER_ORCHESTRA_MODAL_APP"
TEX_PACKAGES_ENV = "PAPER_ORCHESTRA_MODAL_TEX_PACKAGES"
DEFAULT_APP_NAME = "paper-orchestra"
INSTALL_HINT = "Install the Modal SDK with `uv sync --extra modal`."

PYTHON_VERSION = "3.12"
PYTHON_PACKAGES = ("numpy>=1.26", "matplotlib>=3.8", "pandas", "seaborn", "scipy")

#: Covers the bundled templates and the usual conference styles.  ``plain-generic``
#: (soul, ulem) is only a Recommends of latex-extra, so it is listed explicitly.
#: ``fonts-extra`` (inconsolata, newtx, libertine, bbm, dsfont) and ``lmodern`` are what
#: ACL- and ACM-style templates and ``\mathbbm`` need; they add about 1.7 GiB to the layer.
TEXLIVE_PACKAGES = (
    "texlive-latex-base",
    "texlive-latex-recommended",
    "texlive-latex-extra",
    "texlive-fonts-recommended",
    "texlive-fonts-extra",
    "texlive-science",
    "texlive-bibtex-extra",
    "texlive-publishers",
    "texlive-plain-generic",
    "cm-super",
    "lmodern",
    "latexmk",
)
# Debian's rule for package names.  The names end up in a shell command of the image build.
_APT_PACKAGE = re.compile(r"[a-z0-9][a-z0-9+.-]+")


def extra_tex_packages(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The extra Debian packages asked for through ``PAPER_ORCHESTRA_MODAL_TEX_PACKAGES``.

    Raises ``ValueError`` for anything that is not a package name.
    """
    raw = (os.environ if environ is None else environ).get(TEX_PACKAGES_ENV, "")
    packages: list[str] = []
    for name in raw.split():
        if not _APT_PACKAGE.fullmatch(name):
            raise ValueError(
                f"{TEX_PACKAGES_ENV}={raw!r}: {name!r} is not a Debian package name. Give "
                "package names separated by spaces, for example 'texlive-lang-german fonts-lato'."
            )
        if name not in TEXLIVE_PACKAGES and name not in packages:
            packages.append(name)
    return tuple(packages)


def texlive_install_command(extra_packages: Sequence[str] = ()) -> str:
    """The image build step that installs TeX Live plus ``extra_packages``.

    ``Image.apt_install`` cannot pass ``--no-install-recommends``; with recommends
    the TeX Live layer roughly doubles in size.
    """
    return (
        "apt-get update && apt-get install -y --no-install-recommends "
        + " ".join((*TEXLIVE_PACKAGES, *extra_packages))
        + " && rm -rf /var/lib/apt/lists/*"
    )


SandboxKind = Literal["python", "latex"]
LatexMode = Literal["sandbox", "local"]

_ROOT = "/work"
_SCRIPT = "_script.py"
#: Hidden, so it can never collide with an uploaded file (hidden files are not uploaded).
_ARCHIVE = ".paper-orchestra-upload.tar"
_MAX_CAPTURE = 200_000
_MAX_FILE_BYTES = 50 * 1024 * 1024
#: Seconds between SIGTERM and SIGKILL when a job overruns its timeout.
_KILL_GRACE = 5
#: Modal's own exec deadline is only a backstop behind coreutils ``timeout``.
_EXEC_MARGIN = 15
#: Upload, download and round-trip time a job needs on top of its own timeout.
_JOB_OVERHEAD = 120
_UNPACK_TIMEOUT = 120
_MIN_LIFETIME = 300
_MAX_LIFETIME = 24 * 60 * 60
_LATEX_ARTIFACTS = (".log", ".blg")
#: Paranoid file access: TeX may only read and write inside the working directory
#: (and read its own distribution).
_TEX_ENV = {"openout_any": "p", "openin_any": "p"}

#: Where the one job of a job sandbox lives: ``src`` (its working directory), ``home``,
#: and next to them its record, its log and, once it has ended, its status.
_JOB_DIR = f"{_ROOT}/job"
_JOB_SOURCE = f"{_JOB_DIR}/src"
_JOB_HOME = f"{_JOB_DIR}/home"
_JOB_RECORD = "job.json"
_JOB_STARTED = "started"
_JOB_STATUS = "status"
#: Seconds a job sandbox may live beyond its job's time limit: the upload before the
#: job, the download after it, and the time a restarted orchestrator has to fetch the
#: outputs of a job that ended while nobody was attached.
_JOB_MARGIN = 900
#: How the directories are named that a job's packed source and fetched outputs pass
#: through on this machine.  Not as the local backend names the working directories of
#: its jobs: its clean-up after a dead worker goes by a record, and must never be
#: pointed at one of these.
_SCRATCH_PREFIX = "paper-orchestra-modal-"
#: Seconds such a directory is left alone after it was last written to.  A backend locks
#: its directory within an instant of making it; one that is older than this and that
#: nobody holds was left behind by a worker that died.
_SCRATCH_SETTLE_SECONDS = 60.0
#: What a sandbox that runs an experiment's job is started for (see ``_start``).
_A_JOB = "the job"
#: What a person can do when Modal is not usable and it is an experiment that needed it:
#: the paper pipeline's way out (``--compute local``) is not an option of an experiment.
_OR_LOCAL = (
    "An experiment that needs no GPU can also run on this machine: approve it for the "
    "local backend (`--backend local`; in the app, choose the local backend)."
)
_STARTED_AGAIN = f"then run the experiment again. {_OR_LOCAL}"
_PICKED_UP_AGAIN = (
    "then run the experiment again: its job on Modal is picked up where it is by then."
)
#: Bytes of source a job may bring along.  Beyond that the job is refused, not trimmed:
#: leaving files out would change what it computes.
_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
_COLLECT_TIMEOUT = 300
#: Entries and characters of the listing of a job's outputs.  The listing travels back
#: through the output of a command, which is only kept up to ``_MAX_CAPTURE``.
_SCAN_ENTRIES = 5000
_SCAN_CHARS = 150_000
#: How often in a row following a job's log may fail without any progress.
_FOLLOW_RETRIES = 3
#: Seconds between two writes of the running estimate while a job is followed: how far
#: behind the cost on file can be when the process that follows the job is killed.
_COST_EVERY = 30.0
#: What a job is asked to bring back when nobody says otherwise.
_DEFAULT_OUTPUTS = JobSpec(command=["job"]).outputs

# Runs the job, in its working directory.  The job's output goes to a file and its exit
# status and times to another, so that neither depends on this process staying alive.
# coreutils timeout keeps the time limit inside the sandbox for the same reason.  The
# first file tells a job that runs from one that was never started.
_RUNNER = """\
limit="$1"; grace="$2"; shift 2
: > ../started
started=$(date +%s.%N)
timeout --signal=TERM --kill-after="$grace" "$limit" "$@" > ../log.txt 2>&1 < /dev/null
code=$?
printf '%s %s %s\\n' "$code" "$started" "$(date +%s.%N)" > ../status.tmp
mv ../status.tmp ../status
"""

# Prints the job's log from byte $1 on as it grows, and ends once the job has ended and
# all of it was printed.  Whether the job has ended is looked at before the size is, so
# nothing written before the end is missed.
_FOLLOWER = """\
offset="$1"
while :; do
  if [ -e status ]; then last=1; else last=0; fi
  size=$(($(wc -c < log.txt)))
  if [ "$size" -gt "$offset" ]; then
    tail -c "+$((offset + 1))" log.txt | head -c "$((size - offset))"
    offset=$size
  fi
  if [ "$last" = 1 ]; then exit 0; fi
  sleep 0.3
done
"""

#: When the prices below were read from https://modal.com/pricing.
MODAL_PRICES_AS_OF = "2026-11-15"
#: US dollars per second.  A Sandbox pays the non-preemptible rate for CPU cores and
#: memory and the standard rate for a GPU.  Region surcharges and credits are not in
#: here, and a GPU type that is missing has no estimate rather than a wrong one.
MODAL_PRICES_USD_PER_SECOND: dict[str, Any] = {
    "cpu_core": 0.00003942,
    "memory_gib": 0.00000667,
    "gpu": {
        "T4": 0.000164,
        "L4": 0.000222,
        "A10": 0.000306,
        "A10G": 0.000306,
        "L40S": 0.000542,
        "A100": 0.000583,
        "A100-40GB": 0.000583,
        "A100-80GB": 0.000694,
        "H100": 0.001097,
        "H200": 0.001261,
        "B200": 0.001736,
        "B300": 0.001972,
    },
}

T = TypeVar("T")


def _import_modal() -> ModuleType:
    try:
        import modal
    except ModuleNotFoundError as exc:
        if exc.name != "modal":
            raise
        raise ComputeError(f"The Modal compute backend is not available. {INSTALL_HINT}") from exc
    return modal


def modal_token_configured() -> bool:
    """Whether the Modal SDK would find credentials.

    Reads the environment and ``~/.modal.toml`` only; nothing is sent anywhere.
    """
    try:
        from modal.config import config
    except ModuleNotFoundError:
        return False
    except Exception as exc:  # a malformed ~/.modal.toml raises at import
        log.warning("Could not read the Modal configuration: %s", exc)
        return False
    if config.get("token_id") and config.get("token_secret"):
        return True
    return bool(
        config.get("oauth_refresh_token")
        and config.get("oauth_client_id")
        and (config.get("oauth_client_secret") or config.get("oauth_jwt_key"))
    )


def python_image() -> modal.Image:
    """Image for model-written figure code: scientific Python and nothing else."""
    modal = _import_modal()
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .uv_pip_install(*PYTHON_PACKAGES)
        .env({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/opt/matplotlib"})
        # Importing pyplot once bakes the font cache, so no sandbox pays for rebuilding it.
        .run_commands("python -c 'import matplotlib.pyplot'")
    )


def latex_image(extra_packages: Sequence[str] | None = None) -> modal.Image:
    """Image with TeX Live for compiling papers.

    ``extra_packages`` defaults to what ``PAPER_ORCHESTRA_MODAL_TEX_PACKAGES`` asks for.
    """
    modal = _import_modal()
    extra = extra_tex_packages() if extra_packages is None else extra_packages
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .run_commands(texlive_install_command(extra))
        .env({"TEXMFVAR": "/tmp/texmf-var"})
    )


def pack_tree(root: Path, archive: Path, *, max_file_bytes: int) -> list[str]:
    """Write a tar of ``root`` to ``archive`` and return the relative paths left out.

    Hidden files and directories are omitted silently.  Files larger than
    ``max_file_bytes`` and symlinked directories are omitted and reported.
    """
    skipped: list[str] = []
    with tarfile.open(archive, "w", dereference=True) as tar:
        for current, dirs, files in os.walk(root):
            here = Path(current)
            visible = sorted(name for name in dirs if not name.startswith("."))
            skipped.extend(
                (here / name).relative_to(root).as_posix()
                for name in visible
                if (here / name).is_symlink()
            )
            dirs[:] = [name for name in visible if not (here / name).is_symlink()]
            for name in dirs:
                path = here / name
                tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
            for name in sorted(files):
                path = here / name
                if name.startswith(".") or not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                if path.stat().st_size > max_file_bytes:
                    skipped.append(relative)
                    continue
                tar.add(path, arcname=relative, recursive=False)
    return skipped


def estimate_cost(seconds: float, *, gpu: str | None, cpu: float, memory_mb: int) -> float | None:
    """What a Sandbox with these resources costs for ``seconds``, in US dollars.

    An estimate from ``MODAL_PRICES_USD_PER_SECOND``, on the high side for CPU and
    memory: Modal bills what was used or reserved, and this counts the limits.
    ``None`` when the GPU type has no price in the table.
    """
    prices = MODAL_PRICES_USD_PER_SECOND
    rate = cpu * prices["cpu_core"] + memory_mb / 1024 * prices["memory_gib"]
    if gpu:
        kind, _, count = gpu.partition(":")
        price = prices["gpu"].get(kind.rstrip("!").upper())
        if price is None:
            return None
        rate += price * int(count or 1)
    return round(max(0.0, seconds) * rate, 6)


def _anonymous(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Who owns a file on this machine is nobody's business in the sandbox."""
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def pack_source(
    source: Path,
    archive: Path,
    *,
    max_bytes: int,
    should_stop: Callable[[], bool] | None = None,
    outputs: Sequence[str] = (),
) -> list[str]:
    """Write the tar a job sandbox is filled from and return notes on what was left out.

    The archive unpacks to ``src`` (what :func:`walk_source` takes from ``source``)
    and an empty ``home``.  The paths in ``outputs`` are not packed: what the job is
    asked to bring back has to be written by its command, not brought along (the
    notes say when that removed something).  Raises :class:`ComputeError` when the
    files add up to more than ``max_bytes``, and :class:`JobStopped` when
    ``should_stop`` turns true on the way.
    """
    report = SourceReport()
    leftovers: set[str] = set()
    total = 0
    with tarfile.open(archive, "w") as tar:
        for name in ("src", "home"):
            directory = tarfile.TarInfo(name)
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            tar.addfile(directory)
        for relative, path, kind in walk_source(source, report, should_stop):
            output = output_of(relative, outputs)
            if output is not None:
                leftovers.add(output)
                continue
            if kind == "file":
                total += path.stat().st_size
                if total > max_bytes:
                    raise ComputeError(
                        f"the source directory holds more than {human_size(max_bytes)} (not "
                        "counting version-control data, caches and environments), and that is "
                        "all the Modal backend uploads. Remove what the job does not need, or "
                        "let the job download its data."
                    )
            tar.add(path, arcname=f"src/{relative}", recursive=False, filter=_anonymous)
    return [*report.notes(), *leftover_notes(leftovers)]


def sweep_scratch() -> list[str]:
    """Remove the directories that jobs of this backend left in the temporary directory
    when their worker was killed (the packed source of a job, up to a gigabyte, of which
    no record tells), and return the ones removed.

    Only what is provably nobody's any more: a real directory of this user's, named as
    this backend names them, not written to for a while, and whose lock can be taken.
    The backend of a job that runs holds that lock; where it cannot be told (a file
    system without such locks), everything is left as it is.
    """
    try:
        root = tempfile.gettempdir()
        names = [e.name for e in os.scandir(root) if e.name.startswith(_SCRATCH_PREFIX)]
    except OSError:
        return []
    removed: list[str] = []
    for name in names:
        path = os.path.join(root, name)
        try:
            settled = time.time() - os.lstat(path).st_mtime >= _SCRATCH_SETTLE_SECONDS
        except OSError:
            continue
        if not settled or not is_own_directory(path):
            continue
        try:
            descriptor = lock_directory(path)
        except (DirectoryInUse, OSError):  # held, or not to be opened: not known to be free
            continue
        if descriptor is None:
            continue
        try:
            if remove_tree(Path(path)):
                removed.append(path)
        finally:
            os.close(descriptor)
    if removed:
        log.info("Removed what earlier Modal jobs left behind here: %s", ", ".join(removed))
    return removed


@contextlib.contextmanager
def _scratch() -> Iterator[Path]:
    """A directory of this machine for what passes between it and a job's sandbox, held
    (see :func:`sweep_scratch`) until it has been removed again."""
    scratch = tempfile.TemporaryDirectory(prefix=_SCRATCH_PREFIX)
    with holding(scratch.name), scratch as name:
        yield Path(name)


def unpack_outputs(archive: Path, chosen: Mapping[str, int], run_dir: Path) -> list[str]:
    """Copy the ``chosen`` files (path to listed size) out of the tar a sandbox made of them.

    The tar is the work of untrusted code.  Only regular files that were chosen are
    taken, each at most as large as it was listed, whatever else the archive holds
    or claims; links and other paths never reach ``run_dir``.  A file with several
    names (hard links) is in a tar once, its other names referring to the first:
    such a name is taken as a copy of the file that was just taken under the first.
    Returns a note when files that were chosen did not arrive.
    """
    waiting = dict(chosen)
    taken: set[str] = set()
    try:
        with tarfile.open(archive, "r:") as tar:
            for member in tar:
                name = member.name.removeprefix("./")
                size = waiting.get(name)
                if size is None:
                    continue
                target = run_dir / name
                first = member.linkname.removeprefix("./") if member.islnk() else None
                if member.isreg() and member.size <= size:
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source, open(target, "wb") as copy:
                        shutil.copyfileobj(source, copy)
                elif first in taken and (run_dir / first).stat().st_size <= size:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(run_dir / first, target)
                else:
                    continue
                taken.add(name)
                del waiting[name]
    except tarfile.TarError as exc:
        return [f"The archive of the job's outputs could not be read ({exc})."]
    if not waiting:
        return []
    return [
        f"{len(waiting)} of the job's {len(chosen)} output files could not be fetched from the "
        "sandbox."
    ]


def _requested(path: str, kind: str, outputs: Sequence[str]) -> bool:
    """Whether ``path``, as a sandbox reported it, is one of the outputs or lies in one.

    A link may also be reported on the way to an output (``results`` for
    ``results/metrics.json``).  Links are never fetched, so accepting it only lets
    the person hear why the output did not arrive.
    """
    try:
        parts = safe_relative(path).parts
        wanted = [_parts(name) for name in outputs]
    except ComputeError:
        return False
    if any(parts[: len(name)] == name for name in wanted):
        return True
    return kind == "link" and any(name[: len(parts)] == parts for name in wanted)


def _parts(name: str) -> tuple[str, ...]:
    return safe_relative(name).parts


def _write_handle(run_dir: Path, handle: str) -> None:
    """Record a job's sandbox id next to its run, whole or not at all."""
    temporary = run_dir / f".{HANDLE_NAME}.tmp"
    temporary.write_text(handle + "\n", encoding="utf-8")
    os.replace(temporary, run_dir / HANDLE_NAME)


def _handle_on_record(run_dir: Path) -> str | None:
    """The sandbox id this backend wrote next to the run when it started its job."""
    try:
        return (run_dir / HANDLE_NAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _seconds(value: float) -> int:
    # Modal timeouts are uint32 protobuf fields: a float raises TypeError.
    return max(1, math.ceil(value))


def _ran(job: _Job, since: float) -> float:
    """Seconds a job has run that was stopped from here, and so left no record of its
    own: the time since ``since`` (a ``time.time()``), and no more than its limit allows."""
    return min(max(0.0, time.time() - since), float(job.timeout + _KILL_GRACE))


def _epoch(moment: object) -> float | None:
    """``moment`` as seconds since the epoch: a number is taken as that, a ``datetime``
    or an ISO 8601 stamp (``2026-11-16T02:25:00Z``) is converted, one without a time
    zone being read as UTC.  ``None`` for anything else."""
    if isinstance(moment, bool):
        return None
    if isinstance(moment, int | float):
        return float(moment) if 0 < moment < math.inf else None
    if isinstance(moment, str):
        try:
            moment = datetime.fromisoformat(moment.strip())
        except ValueError:
            return None
    if not isinstance(moment, datetime):
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).timestamp()


def _number(value: object, *, least: float = 0.0) -> bool:
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and least < value < math.inf
    )


def _kept_job(run_dir: Path) -> _Job | None:
    """The job as this backend wrote it down next to the run when it started it (in
    ``cost.json``); ``None`` when there is no such record or it is not whole.

    The run directory is this side's: a job in a sandbox can put nothing there but
    its outputs, and none of those may be called ``cost.json``.
    """
    kept = (read_record(run_dir / COST_NAME) or {}).get("job")
    if not isinstance(kept, dict):
        return None
    gpu = kept.get("gpu")
    whole = (
        _number(kept.get("created"))
        and _number(kept.get("timeout"))
        and _number(kept.get("cpu"))
        and _number(kept.get("memory_mb"))
        and (gpu is None or isinstance(gpu, str))
    )
    if not whole:
        return None
    return _Job(
        timeout=_seconds(kept["timeout"]),
        outputs=_DEFAULT_OUTPUTS,
        gpu=gpu,
        cpu=float(kept["cpu"]),
        memory_mb=int(kept["memory_mb"]),
        created=float(kept["created"]),
    )


def _read_tail(stream: Iterable[bytes]) -> bytes:
    """Drain an output stream of a sandbox process, holding on to its end only.

    Model-written code can print without bound, and what it prints must not
    exhaust the memory of this process.
    """
    budget = 4 * _MAX_CAPTURE  # bytes: room for _MAX_CAPTURE characters of any UTF-8 text
    chunks: deque[bytes] = deque()
    held = 0
    for chunk in stream:
        chunks.append(chunk)
        held += len(chunk)
        while held - len(chunks[0]) >= budget:
            held -= len(chunks.popleft())
    return b"".join(chunks)[-budget:]


def _decode(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    return text if len(text) <= _MAX_CAPTURE else text[-_MAX_CAPTURE:]


class _SandboxLost(Exception):
    """The sandbox stopped (lifetime, idle timeout, termination) while a job was using it."""


class _JobGone(ComputeError):
    """The sandbox of a job that was to be picked up has ended; when is not known."""


class _NotAJob(ComputeError):
    """The sandbox that was to be picked up holds no job of this backend."""


@dataclass
class _Completed:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass
class _Job:
    """What the backend goes by when it follows, fetches and bills a job.

    Always put together on this side: from the ``JobSpec``, or from what this backend
    wrote next to the run.  A copy lies in the sandbox as ``job.json``; it marks the
    sandbox as holding a job and is never read back, because the job could have
    rewritten it.
    """

    #: Time limit of the command in whole seconds.
    timeout: int
    outputs: tuple[str, ...]
    gpu: str | None
    cpu: float
    memory_mb: int
    #: ``time.time()`` from which the sandbox is billed; ``None`` when that, or what
    #: the sandbox was made of, is not known, and so there is no estimate.
    created: float | None = None

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    def billed(self) -> float:
        """Seconds the sandbox is billed for so far.  None is billed for longer than it
        can live, whatever the clocks say, and none for less than nothing."""
        if self.created is None:
            return 0.0
        return min(max(0.0, time.time() - self.created), self.longest())

    def longest(self) -> float:
        """Seconds the sandbox of the job can live at most."""
        return float(self.timeout + _JOB_MARGIN)


@dataclass
class _Slot:
    lock: threading.Lock = field(default_factory=threading.Lock)
    sandbox: Any = None
    #: ``time.monotonic()`` value at which the sandbox reaches its lifetime.
    expires_at: float = 0.0


class ModalBackend(ComputeBackend):
    """Compute backend backed by Modal Sandboxes.

    ``cpu`` (cores) and ``memory_mb`` (MiB) are hard limits of each sandbox;
    an experiment job may ask for its own.  ``sandbox_lifetime`` is the maximum
    age of a sandbox and ``idle_timeout`` how long it may sit unused before Modal
    collects it; both are in seconds and bound what a crashed orchestrator can
    leave running.  A job sandbox lives as long as its job's time limit asks for
    instead, up to Modal's maximum of 24 hours.
    """

    name = "modal"

    def __init__(
        self,
        *,
        app_name: str | None = None,
        latex: LatexMode = "sandbox",
        cpu: float = 2.0,
        memory_mb: int = 4096,
        sandbox_lifetime: float = 3600.0,
        idle_timeout: float = 600.0,
    ) -> None:
        if latex not in ("sandbox", "local"):
            raise ValueError(f"latex must be 'sandbox' or 'local', got {latex!r}")
        if cpu <= 0:
            raise ValueError(f"cpu must be positive, got {cpu!r}")
        if memory_mb < 256:
            raise ValueError(f"memory_mb must be at least 256, got {memory_mb!r}")
        if not _MIN_LIFETIME <= sandbox_lifetime <= _MAX_LIFETIME:
            raise ValueError(
                f"sandbox_lifetime must be between {_MIN_LIFETIME} and {_MAX_LIFETIME} seconds, "
                f"got {sandbox_lifetime!r}"
            )
        if not 0 < idle_timeout <= sandbox_lifetime:
            raise ValueError(
                f"idle_timeout must be positive and at most sandbox_lifetime, got {idle_timeout!r}"
            )
        if latex == "local" and not LocalBackend.latex_available():
            raise ComputeError(
                "latex='local' needs pdflatex and bibtex on PATH. Install TeX Live, or use "
                "latex='sandbox' to compile in a Modal sandbox."
            )
        self._modal = _import_modal()
        self.app_name = app_name or os.environ.get(APP_ENV) or DEFAULT_APP_NAME
        self.latex: LatexMode = latex
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.sandbox_lifetime = sandbox_lifetime
        self.idle_timeout = idle_timeout
        self._local = LocalBackend() if latex == "local" else None
        # Read now, so that a malformed value is reported before any work is done.
        self._tex_packages = extra_tex_packages() if latex == "sandbox" else ()
        self._slots: dict[SandboxKind, _Slot] = {"python": _Slot(), "latex": _Slot()}
        self._state_lock = threading.Lock()
        self._app: Any = None
        #: Sandboxes replaced before their lifetime ran out; other threads may still be
        #: finishing jobs on them, so they are only terminated in ``close``.
        self._retired: list[Any] = []
        #: Sandboxes of experiment jobs that ``run_job`` or ``attach`` is working on.
        self._jobs: list[Any] = []
        #: Sandboxes of jobs whose termination failed: they may still run, and bill.
        self._lingering: list[Any] = []

    # -- python ----------------------------------------------------------------

    def run_python(
        self,
        code: str,
        *,
        inputs: Mapping[str, bytes] | None = None,
        outputs: Sequence[str] = (),
        timeout: float = 120.0,
    ) -> ExecResult:
        files = {safe_relative(name).as_posix(): data for name, data in (inputs or {}).items()}
        wanted = [(name, safe_relative(name).as_posix()) for name in outputs]
        limit = _seconds(timeout)

        def job(sandbox: Any) -> ExecResult:
            job_dir = f"{_ROOT}/py-{uuid.uuid4().hex[:12]}"
            filesystem = sandbox.filesystem
            try:
                for relative, data in files.items():
                    filesystem.write_bytes(data, f"{job_dir}/{relative}")
                filesystem.write_text(code, f"{job_dir}/{_SCRIPT}")
                # -I: isolated mode, so PYTHON* variables and the user site are ignored.
                done = self._exec(
                    sandbox,
                    ["python", "-I", "-B", _SCRIPT],
                    cwd=job_dir,
                    limit=limit,
                    env={"HOME": job_dir, "TMPDIR": job_dir},
                )
                produced: dict[str, bytes] = {}
                notes: list[str] = []
                for name, relative in wanted:
                    self._collect(filesystem, f"{job_dir}/{relative}", name, produced, notes)
            finally:
                self._remove(sandbox, job_dir)
            return ExecResult(
                returncode=done.returncode,
                stdout=done.stdout,
                stderr=done.stderr + "".join(f"\n[paper-orchestra] {note}" for note in notes),
                timed_out=done.timed_out,
                files=produced,
            )

        return self._with_sandbox("python", limit, job)

    def _collect(
        self,
        filesystem: Any,
        remote: str,
        name: str,
        produced: dict[str, bytes],
        notes: list[str],
    ) -> None:
        """Fetch one requested output file if it exists and is small enough to hold in memory."""
        try:
            info = filesystem.stat(remote)
            if not info.is_file():
                return
            if info.size > _MAX_FILE_BYTES:
                notes.append(
                    f"output file {name!r} is {info.size / 1e6:.0f} MB, above the "
                    f"{_MAX_FILE_BYTES // (1024 * 1024)} MB limit, and was not returned."
                )
                return
            produced[name] = filesystem.read_bytes(remote)
        except self._modal.exception.SandboxFilesystemNotFoundError:
            return

    # -- latex -----------------------------------------------------------------

    def compile_latex(
        self,
        workdir: str | Path,
        tex_name: str,
        *,
        output_pdf: str | Path | None = None,
        timeout: float = 240.0,
    ) -> CompileResult:
        if self._local is not None:
            return self._local.compile_latex(
                workdir, tex_name, output_pdf=output_pdf, timeout=timeout
            )
        work = Path(workdir)
        source = work / safe_relative(f"{tex_name}.tex")
        if not source.is_file():
            raise ComputeError(f"{source} does not exist")
        produced = work / f"{tex_name}.pdf"
        target = Path(output_pdf) if output_pdf else produced
        limit = _seconds(timeout)
        # The sandbox holds no credentials, but both backends accept the same documents.
        refused = refuse_unsafe_document(work, tex_name)
        if refused is not None:
            produced.unlink(missing_ok=True)
            return refused

        def job(sandbox: Any) -> CompileResult:
            job_dir = f"{_ROOT}/tex-{uuid.uuid4().hex[:12]}"
            filesystem = sandbox.filesystem
            # A PDF left over from an earlier build must not pass for this one's output.
            produced.unlink(missing_ok=True)
            try:
                self._upload_tree(sandbox, work, job_dir)
                deadline = time.monotonic() + limit
                chunks: list[str] = []
                steps: list[tuple[str, str]] = []
                timed_out = False
                for command in latex_commands(tex_name):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    done = self._exec(
                        sandbox,
                        command,
                        cwd=job_dir,
                        limit=_seconds(remaining),
                        env=_TEX_ENV,
                    )
                    if done.returncode in (126, 127) and not done.timed_out:
                        raise ComputeError(
                            f"could not run {command[0]} in the Modal sandbox: "
                            f"{done.stderr.strip() or 'command not found'}"
                        )
                    steps.append((command[0], done.stdout + done.stderr))
                    chunks.append(f"$ {' '.join(command)}\n\n{done.stdout}{done.stderr}")
                    if done.timed_out:
                        timed_out = True
                        break
                pdf_path: Path | None = None
                if not timed_out and self._download(
                    filesystem, f"{job_dir}/{tex_name}.pdf", target
                ):
                    pdf_path = target
                for suffix in _LATEX_ARTIFACTS:
                    self._download(
                        filesystem, f"{job_dir}/{tex_name}{suffix}", work / f"{tex_name}{suffix}"
                    )
            finally:
                self._remove(sandbox, job_dir)
            output = "\n".join(chunks)
            # Same rule as the local backend: errors from every step, warnings from the
            # final pdflatex pass only.
            errors, warnings = summarize_build(steps)
            return CompileResult(
                ok=pdf_path is not None,
                pdf_path=pdf_path,
                errors=errors,
                warnings=warnings,
                log_tail=output[-6000:],
                timed_out=timed_out,
            )

        return self._with_sandbox("latex", limit, job)

    def _upload_tree(self, sandbox: Any, work: Path, job_dir: str) -> None:
        """Copy the visible files of ``work`` into ``job_dir`` with one upload."""
        with tempfile.TemporaryDirectory(prefix="paper-orchestra-upload-") as tmp:
            archive = Path(tmp) / "workdir.tar"
            skipped = pack_tree(work, archive, max_file_bytes=_MAX_FILE_BYTES)
            if skipped:
                log.warning(
                    "Not uploaded to the Modal sandbox (larger than %d MB or a symlinked "
                    "directory): %s",
                    _MAX_FILE_BYTES // (1024 * 1024),
                    ", ".join(skipped),
                )
            sandbox.filesystem.copy_from_local(archive, f"{job_dir}/{_ARCHIVE}")
        self._unpack(sandbox, job_dir, "the LaTeX workspace")

    def _unpack(self, sandbox: Any, job_dir: str, what: str) -> None:
        """Unpack the archive that was uploaded to ``job_dir`` there."""
        done = self._exec(
            sandbox, ["tar", "-xf", _ARCHIVE, "-C", "."], cwd=job_dir, limit=_UNPACK_TIMEOUT
        )
        if done.returncode != 0:
            raise ComputeError(
                f"could not unpack {what} in the Modal sandbox: "
                f"{(done.stderr or done.stdout).strip()[-500:] or f'tar exited {done.returncode}'}"
            )

    def _download(self, filesystem: Any, remote: str, local: Path) -> bool:
        try:
            filesystem.copy_to_local(remote, local)
        except self._modal.exception.SandboxFilesystemNotFoundError:
            return False
        return True

    # -- experiment jobs ---------------------------------------------------------

    def supports(self, spec: JobSpec) -> list[str]:
        reasons = spec_problems(spec)
        longest = _MAX_LIFETIME - _JOB_MARGIN
        if not reasons and spec.timeout > longest:
            reasons.append(
                f"Its time limit of {spec.timeout:g} s is more than a Modal sandbox allows: at "
                f"most {longest} s (24 hours, less the time to fetch the results)."
            )
        return reasons

    def run_job(
        self,
        spec: JobSpec,
        *,
        source_dir: Path,
        run_dir: Path,
        on_log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        mask: Sequence[str] = (),
    ) -> JobResult:
        source, run_dir = Path(source_dir), Path(run_dir)
        log_path = open_run(run_dir, spec.outputs)
        result = JobResult(returncode=None, backend=self.name, log_path=log_path)
        result.outputs_dir = run_dir
        # A worker that was killed while it packed or uploaded a source removed nothing.
        sweep_scratch()
        with JobLog(log_path, on_log, mask=mask) as job_log:
            reasons = self.supports(spec)
            if not reasons and not source.is_dir():
                reasons = [f"The source directory {source} does not exist."]
            if reasons:
                result.error = "The Modal backend cannot run this job. " + " ".join(reasons)
                job_log.note(result.error)
                return result
            try:
                with _scratch() as scratch:
                    self._run_job(spec, source, run_dir, scratch, job_log, should_stop, result)
            except BackendUnusable:
                raise
            except (ComputeError, _SandboxLost, OSError, *self._transport_errors) as exc:
                result.error = f"The Modal backend could not run the job: {self._reason(exc)}"
                job_log.note(result.error)
            result.notes.extend(job_log.notes())
        return result

    def attach(
        self,
        handle: str,
        *,
        run_dir: Path,
        spec: JobSpec | None = None,
        started: str | float | None = None,
        on_log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        mask: Sequence[str] = (),
    ) -> JobResult:
        """Pick up the job in sandbox ``handle`` whose orchestrator died, and finish it
        the way ``run_job`` would have: follow it to its end, fetch its outputs into
        ``run_dir``, terminate the sandbox.

        The job is what this side says it is.  ``spec`` gives its outputs, its time
        limit and its resources, and ``started`` the moment from which it is billed.
        Without them the backend goes by what it wrote next to the run when it started
        the job (``cost.json``), and by the default outputs; a job of which neither is
        known is followed and ended, and has no cost estimate.  Nothing is taken from
        the sandbox's own record of the job.

        The log in ``run_dir`` is rebuilt from the start of the job, and ``on_log``
        receives only the lines the log there had not delivered yet.  If the sandbox
        is gone (it ends ``idle_timeout`` seconds after a job nobody is attached to)
        the result has ``error`` set, the log is left as it was, and the run is
        charged for the time since it started, at most for as long as its sandbox
        could live; the notes say so.  A sandbox that cannot be reached is charged
        for all of that time, since it may still run.  A sandbox that holds no job
        of this backend is left alone.
        """
        run_dir = Path(run_dir)
        log_path = run_dir / LOG_NAME
        result = JobResult(returncode=None, backend=self.name, log_path=log_path, handle=handle)
        result.outputs_dir = run_dir
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ComputeError(
                f"The run directory {run_dir} cannot be used: {exc}. Check that it can be written."
            ) from exc
        # This is where a worker comes after one that died: what that one left in the
        # temporary directory goes now.
        sweep_scratch()
        # A job that this backend would not have started is not picked up as one either.
        problems = self.supports(spec) if spec is not None else []
        job = self._job_on_record(run_dir, None if problems else spec, started)
        try:
            if problems:
                raise _NotAJob("the job that was to be picked up is not one: " + problems[0])
            sandbox = self._find_job(handle, known=_handle_on_record(run_dir) == handle)
        except BackendUnusable:
            raise
        except (ComputeError, *self._transport_errors) as exc:
            result.error = f"The Modal backend could not pick the job up again: {self._reason(exc)}"
            with JobLog(log_path, on_log, append=True, mask=mask) as job_log:
                job_log.note(result.error)
                self._charge_lost(run_dir, job, exc, result, job_log)
            return result
        skip = delivered_lines(log_path)
        with JobLog(log_path, on_log, skip_lines=skip, mask=mask) as job_log:
            try:
                with _scratch() as scratch:
                    self._adopt(sandbox)
                    try:
                        # When the command was started is not known here; the run began
                        # shortly before it.
                        since = time.time() if job.created is None else job.created
                        self._finish(
                            sandbox,
                            job,
                            run_dir,
                            scratch,
                            job_log,
                            should_stop,
                            result,
                            since=since,
                        )
                    finally:
                        self._settle(sandbox, job, result, run_dir)
            except (ComputeError, _SandboxLost, OSError, *self._transport_errors) as exc:
                result.error = f"The Modal backend could not finish the job: {self._reason(exc)}"
                job_log.note(result.error)
            result.notes.extend(job_log.notes())
        return result

    def _job_of(self, spec: JobSpec, created: float | None) -> _Job:
        return _Job(
            timeout=_seconds(spec.timeout),
            outputs=tuple(spec.outputs),
            gpu=spec.gpu,
            cpu=spec.cpu or self.cpu,
            memory_mb=spec.memory_mb or self.memory_mb,
            created=created,
        )

    def _job_on_record(self, run_dir: Path, spec: JobSpec | None, started: object) -> _Job:
        """The job that is picked up, from what this side knows about it (see ``attach``)."""
        kept = _kept_job(run_dir)
        began = _epoch(started)
        if began is None and kept is not None:
            began = kept.created
        if spec is not None:
            return self._job_of(spec, began)
        if kept is not None:
            kept.created = began
            return kept
        # Nothing is known of it: it can be followed and ended, not priced.
        return _Job(
            timeout=_MAX_LIFETIME - _JOB_MARGIN,
            outputs=_DEFAULT_OUTPUTS,
            gpu=None,
            cpu=self.cpu,
            memory_mb=self.memory_mb,
        )

    def _record_cost(
        self, run_dir: Path, job: _Job, *, final: bool, seconds: float | None = None
    ) -> float | None:
        """Estimate what the job's sandbox has cost, write it next to the run (see
        ``COST_NAME``) and return it.  ``seconds`` is the time to charge when that is
        not simply the time since the sandbox was made."""
        record: dict[str, Any] = {"cost_usd": None, "seconds": 0.0, "final": final}
        if job.created is not None:
            seconds = job.billed() if seconds is None else seconds
            record["cost_usd"] = estimate_cost(
                seconds, gpu=job.gpu, cpu=job.cpu, memory_mb=job.memory_mb
            )
            record["seconds"] = round(seconds, 3)
            # What the estimate is of.  A later process that picks the job up without
            # being told goes by this, never by what the sandbox says of itself.
            record["job"] = {
                "created": job.created,
                "timeout": job.timeout,
                "gpu": job.gpu,
                "cpu": job.cpu,
                "memory_mb": job.memory_mb,
            }
        try:
            write_record(run_dir / COST_NAME, record)
        except OSError as exc:
            log.warning("Could not write the cost of the job to %s: %s", run_dir / COST_NAME, exc)
        return record["cost_usd"]

    def _charge_lost(
        self, run_dir: Path, job: _Job, exc: BaseException, result: JobResult, job_log: JobLog
    ) -> None:
        """Put a cost on a job that could not be picked up again.

        Its sandbox was billed whether or not anybody watched.  If the process that
        ran it wrote the final cost down, that stands.  Otherwise the sandbox is
        charged from the start of the run until now when it is known to be gone, and
        for as long as it can live when it could not be reached and may still run.
        """
        if isinstance(exc, _NotAJob):
            return
        kept = read_record(run_dir / COST_NAME) or {}
        if kept.get("final") is True:
            cost = kept.get("cost_usd")
            result.cost_usd = float(cost) if _number(cost, least=-1.0) else None
            return
        if job.created is None:
            return
        gone = isinstance(exc, _JobGone)
        seconds = job.billed() if gone else job.longest()
        result.cost_usd = self._record_cost(run_dir, job, final=True, seconds=seconds)
        note = (
            "When the sandbox ended is not known, so the run is charged for the time since "
            f"it started ({seconds:.0f} s; its sandbox could live for {job.longest():.0f} s)."
            if gone
            else "Whether the sandbox still runs is not known, so the run is charged for as "
            f"long as its sandbox can live ({seconds:.0f} s)."
        )
        job_log.note(note)
        result.notes.append(note)

    def _metered(
        self, should_stop: Callable[[], bool] | None, run_dir: Path, job: _Job
    ) -> Callable[[], bool]:
        """``should_stop`` for the following of a job, which is asked all the time: it
        also keeps the estimate on file up to date, so that little of the cost is
        missing if this process is killed."""
        next_write = time.monotonic() + _COST_EVERY

        def ask() -> bool:
            nonlocal next_write
            if time.monotonic() >= next_write:
                next_write = time.monotonic() + _COST_EVERY
                self._record_cost(run_dir, job, final=False)
            return should_stop is not None and should_stop()

        return ask

    def _run_job(
        self,
        spec: JobSpec,
        source: Path,
        run_dir: Path,
        scratch: Path,
        job_log: JobLog,
        should_stop: Callable[[], bool] | None,
        result: JobResult,
    ) -> None:
        archive = scratch / "source.tar"
        try:
            notes = pack_source(
                source,
                archive,
                max_bytes=_MAX_UPLOAD_BYTES,
                should_stop=should_stop,
                outputs=spec.outputs,
            )
        except JobStopped:
            result.cancelled = True
            job_log.note("The job was stopped on request, before it had started.")
            return
        for note in notes:
            job_log.note(note)
        result.notes.extend(notes)
        job_log.note(
            "Starting a Modal sandbox. The first job with a new set of packages builds an "
            "image, which can take several minutes."
        )
        job = self._job_of(spec, created=None)
        try:
            sandbox = self._await(
                lambda: self._create_job(spec, job), should_stop, undo=self._terminate
            )
        except JobStopped:
            result.cancelled = True
            job_log.note("The job was stopped on request, before it had started.")
            return
        job.created = time.time()
        self._adopt(sandbox)
        try:
            result.handle = sandbox.object_id
            # First of all: from here on an orchestrator that dies leaves a job behind,
            # and this file is how the next one finds it.
            _write_handle(run_dir, result.handle)
            # The sandbox costs money from now on, whatever becomes of this process.
            self._record_cost(run_dir, job, final=False)
            try:
                self._await(lambda: self._launch(sandbox, spec, job, archive), should_stop)
            finally:
                # A copy of the person's files, of no use any more.  The job may run for
                # hours, and a worker that is killed in that time removes nothing.
                with contextlib.suppress(OSError):
                    archive.unlink(missing_ok=True)
            self._finish(
                sandbox, job, run_dir, scratch, job_log, should_stop, result, since=time.time()
            )
        except JobStopped:
            result.cancelled = True
            job_log.note("The job was stopped on request, before it had started.")
        finally:
            self._settle(sandbox, job, result, run_dir)

    def _await(
        self,
        work: Callable[[], T],
        should_stop: Callable[[], bool] | None,
        undo: Callable[[T], None] | None = None,
    ) -> T:
        """Return ``work()``, a call to Modal that can block for minutes (an image is
        built, a source directory uploaded), without going deaf to the caller.

        The call runs on a thread of its own while this one keeps asking
        ``should_stop`` and stays open to Ctrl-C.  When either ends the wait
        (:class:`JobStopped` is raised for the former), the call is left to finish
        behind our back and ``undo`` is applied to what it returns: a sandbox that
        appears after its job was called off is terminated, not leaked.
        """
        lock = threading.Lock()
        outcome: dict[str, Any] = {}
        abandoned = False

        def run() -> None:
            try:
                value = work()
            except BaseException as exc:  # raised again by the waiting thread
                outcome["error"] = exc
                return
            with lock:
                outcome["value"] = value
                if abandoned and undo is not None:
                    undo(value)

        def abandon() -> None:
            nonlocal abandoned
            with lock:
                abandoned = True
                if "value" in outcome and undo is not None:
                    undo(outcome["value"])  # it finished just as we stopped waiting

        worker = threading.Thread(target=run, name="paper-orchestra-modal-call", daemon=True)
        worker.start()
        try:
            while True:
                worker.join(0.5)
                if not worker.is_alive():
                    break
                if should_stop is not None and should_stop():
                    raise JobStopped
        except BaseException:
            abandon()
            raise
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    def _create_job(self, spec: JobSpec, job: _Job) -> Any:
        image = python_image()
        if spec.packages:
            image = image.uv_pip_install(*spec.packages)
        options: dict[str, Any] = {"gpu": spec.gpu} if spec.gpu else {}
        return self._start(
            _A_JOB,
            image=image,
            timeout=_seconds(job.timeout + _JOB_MARGIN),
            idle_timeout=_seconds(self.idle_timeout),
            cpu=(min(0.5, job.cpu), job.cpu),
            memory=(min(1024, job.memory_mb), job.memory_mb),
            block_network=not spec.network,
            **options,
        )

    def _launch(self, sandbox: Any, spec: JobSpec, job: _Job, archive: Path) -> None:
        """Put the job's files into the sandbox and start its command there."""
        filesystem = sandbox.filesystem
        filesystem.write_bytes(job.to_json(), f"{_JOB_DIR}/{_JOB_RECORD}")
        filesystem.write_bytes(b"", f"{_JOB_DIR}/{LOG_NAME}")
        filesystem.copy_from_local(archive, f"{_JOB_DIR}/{_ARCHIVE}")
        self._unpack(sandbox, _JOB_DIR, "the source directory")
        # Not best effort, as other tidying is: the archive holds the source as it was
        # sent, and the command must not find in it what the first step of a job takes
        # out of its files before the command may read them (the token of a
        # repository's address).  A job whose archive cannot be removed is not started.
        filesystem.remove(f"{_JOB_DIR}/{_ARCHIVE}", recursive=True)
        # Not waited for and not read: the job reports through its files (see _RUNNER).
        sandbox.exec(
            "sh",
            "-c",
            _RUNNER,
            "job",
            str(job.timeout),
            str(_KILL_GRACE),
            *spec.command,
            workdir=_JOB_SOURCE,
            env={"HOME": _JOB_HOME, "TMPDIR": _JOB_HOME, "PYTHONUNBUFFERED": "1", **spec.env},
            timeout=job.timeout + _KILL_GRACE + _EXEC_MARGIN,
            text=False,
        )

    def _finish(
        self,
        sandbox: Any,
        job: _Job,
        run_dir: Path,
        scratch: Path,
        job_log: JobLog,
        should_stop: Callable[[], bool] | None,
        result: JobResult,
        *,
        since: float,
    ) -> None:
        """Follow a running job to its end, record how it ended and fetch its outputs.

        ``since`` is the ``time.time()`` at which the job was started, as well as this
        process knows it.  It times a job that is stopped from here; one that ends by
        itself records its own times.
        """
        # The time limit is kept inside the sandbox; this one is for a sandbox that
        # stopped answering.
        deadline = time.monotonic() + job.timeout + _KILL_GRACE + _EXEC_MARGIN + _JOB_OVERHEAD
        stalled = 0
        while True:
            seen = job_log.raw_bytes
            outcome = supervise(
                self._follow(sandbox, seen, job.timeout),
                job_log,
                timeout=max(0.0, deadline - time.monotonic()),
                should_stop=self._metered(should_stop, run_dir, job),
                interrupt=lambda: self._release(sandbox),
                grace=0,
            )
            if outcome.cancelled:
                result.cancelled = True
                result.seconds = _ran(job, since)
                job_log.note("The job was stopped on request.")
                return
            if outcome.timed_out:
                result.timed_out = True
                result.seconds = _ran(job, since)
                job_log.note(
                    "The sandbox was stopped because the job did not end within its time "
                    f"limit of {job.timeout} s."
                )
                return
            if outcome.failure is not None and not isinstance(
                outcome.failure, self._transport_errors
            ):
                raise outcome.failure
            code = None
            try:
                status = self._job_status(sandbox)
            except self._transport_errors:
                # A sandbox that is gone answers no better than a connection that is.
                code = self._exit_code(sandbox)
                if code is None:
                    raise
                status = None
            if status is not None:
                break
            code = self._exit_code(sandbox) if code is None else code
            if code is not None:
                raise _SandboxLost(
                    f"the sandbox stopped before the job finished (exit code {code}: it "
                    "reached its lifetime, was terminated or, with a GPU, was preempted). "
                    "Run the experiment again."
                )
            # The job is still running and only the reading of its log ended: read on
            # from where it stopped, unless that keeps failing.
            stalled = 0 if job_log.raw_bytes > seen else stalled + 1
            if stalled >= _FOLLOW_RETRIES:
                raise ComputeError(
                    "the job's output could not be read from the sandbox "
                    f"({self._reason(outcome.failure) if outcome.failure else 'the stream ended'})."
                )
            time.sleep(1.0)
        result.returncode, seconds = status
        result.seconds = seconds or 0.0
        # 124 and 137 are what coreutils timeout exits with when it had to stop the
        # command; a job may exit with them by itself, but not after exactly its limit.
        if result.returncode in (124, 137) and (seconds is None or seconds >= job.timeout - 1):
            result.timed_out = True
            job_log.note(
                f"The job was stopped because it ran longer than its time limit of {job.timeout} s."
            )
        notes = self._fetch_outputs(sandbox, job.outputs, run_dir, scratch)
        for note in notes:
            job_log.note(note)
        result.notes.extend(notes)

    def _follow(self, sandbox: Any, offset: int, limit: int) -> Iterator[bytes]:
        """The job's log from byte ``offset`` on, as it grows, until the job has ended."""
        process = sandbox.exec(
            "sh",
            "-c",
            _FOLLOWER,
            "follow",
            str(offset),
            workdir=_JOB_DIR,
            timeout=limit + _KILL_GRACE + _EXEC_MARGIN,
            text=False,
        )
        yield from process.stdout

    def _job_status(self, sandbox: Any) -> tuple[int, float | None] | None:
        """Exit status of the job and how long it ran (``None``: unknown), or ``None``
        while it has not ended."""
        try:
            data = sandbox.filesystem.read_bytes(f"{_JOB_DIR}/{_JOB_STATUS}")
        except self._modal.exception.SandboxFilesystemNotFoundError:
            return None
        fields = data.split()
        try:
            code = int(fields[0])
        except (IndexError, ValueError) as exc:
            raise ComputeError(
                f"the record of how the job ended is unreadable: {data[:80]!r}"
            ) from exc
        try:
            return code, max(0.0, float(fields[2]) - float(fields[1]))
        except (IndexError, ValueError):
            # A `date` that cannot print fractions of a second: the status still counts.
            return code, None

    def _exit_code(self, sandbox: Any) -> int | None:
        """The exit code of the sandbox itself; ``None`` while it is running.  A sandbox
        that cannot be asked any more counts as gone."""
        try:
            return sandbox.poll()
        except self._transport_errors:
            return -1

    def _fetch_outputs(
        self, sandbox: Any, outputs: Sequence[str], run_dir: Path, scratch: Path
    ) -> list[str]:
        """Copy the job's outputs from the sandbox to ``run_dir``; returns notes about
        what was not copied.  The sandbox only lists and packs; what is taken is
        decided here, by the rules the local backend applies."""
        if not outputs:
            return []
        limits = [str(_SCAN_ENTRIES), str(_SCAN_CHARS)]
        listed = self._exec(
            sandbox,
            # -I: nothing in the job's directory can stand in for a module of the script.
            ["python", "-I", "-c", scan_outputs_script(), *limits, *outputs],
            cwd=_JOB_SOURCE,
            limit=_COLLECT_TIMEOUT,
        )
        try:
            listing = json.loads(listed.stdout)
            entries = [(str(path), str(kind), int(size)) for path, kind, size in listing["entries"]]
            truncated = bool(listing["truncated"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ComputeError(
                "could not list the job's outputs in the sandbox: "
                f"{(listed.stderr or listed.stdout).strip()[-300:] or exc}"
            ) from exc
        # What the sandbox says is there is the word of untrusted code: nothing outside
        # the requested outputs is ever asked for or accepted.
        entries = [entry for entry in entries if _requested(entry[0], entry[1], outputs)]
        chosen, notes = choose_outputs(entries, truncated=truncated)
        if not chosen:
            return notes
        # "./" keeps a name that starts with a dash from being read as an option.
        names = b"".join(b"./" + os.fsencode(path) + b"\0" for path, _ in chosen)
        sandbox.filesystem.write_bytes(names, f"{_JOB_DIR}/outputs.list")
        packed = self._exec(
            sandbox,
            ["tar", "-cf", "../outputs.tar", "--no-recursion", "--null", "-T", "../outputs.list"],
            cwd=_JOB_SOURCE,
            limit=_COLLECT_TIMEOUT,
        )
        if packed.returncode not in (0, 1):  # 1: a file changed while it was read
            raise ComputeError(
                "could not pack the job's outputs in the sandbox: "
                f"{packed.stderr.strip()[-300:] or f'tar exited {packed.returncode}'}"
            )
        remote = f"{_JOB_DIR}/outputs.tar"
        expected = sum(size for _, size in chosen) + 4096 * len(chosen) + 1024 * 1024
        if sandbox.filesystem.stat(remote).size > expected:
            return [*notes, "The job's outputs grew while they were fetched and were not copied."]
        archive = scratch / "outputs.tar"
        sandbox.filesystem.copy_to_local(remote, archive)
        return [*notes, *unpack_outputs(archive, dict(chosen), run_dir)]

    def _find_job(self, handle: str, *, known: bool = False) -> Any:
        """The running sandbox ``handle``, which holds a job; :class:`ComputeError` if
        there is nothing to pick up.

        ``known`` says that this side has it on record as the sandbox of the run.  Any
        other sandbox has to carry the mark of a job, so that one which does something
        else (it runs figure code) is not taken for one and ended.
        """
        modal = self._modal
        try:
            sandbox = modal.Sandbox.from_id(handle)
        except modal.exception.AuthError as exc:
            raise self._unauthenticated(_PICKED_UP_AGAIN) from exc
        except modal.exception.NotFoundError as exc:
            raise _JobGone(
                f"Modal has no sandbox {handle}, so the job and its outputs cannot be "
                "recovered. Run the experiment again."
            ) from exc
        try:
            code = sandbox.poll()
            if code is not None:
                raise _JobGone(
                    f"the sandbox {handle} is no longer running (it ended with code {code}), "
                    "so the job's outputs cannot be recovered. Run the experiment again."
                )
            try:
                # Only that it is there counts, and only for a sandbox this side does
                # not know: the file is the job's to rewrite, and to remove.
                if not known:
                    sandbox.filesystem.stat(f"{_JOB_DIR}/{_JOB_RECORD}")
            except modal.exception.SandboxFilesystemNotFoundError as exc:
                raise _NotAJob(
                    f"the sandbox {handle} does not hold a job of this backend."
                ) from exc
            try:
                sandbox.filesystem.stat(f"{_JOB_DIR}/{_JOB_STARTED}")
            except modal.exception.SandboxFilesystemNotFoundError as exc:
                # Ours, and of no use: waiting for a job that never began would only
                # keep the sandbox, and the bill, running.
                self._terminate(sandbox)
                raise _JobGone(
                    f"the job in sandbox {handle} had not been started when its orchestrator "
                    "died. The sandbox was stopped; run the experiment again."
                ) from exc
            return sandbox
        except BaseException:
            # Not ours to end: only the connection to it is given up.
            with contextlib.suppress(*self._transport_errors):
                sandbox.detach()
            raise

    def _adopt(self, sandbox: Any) -> None:
        with self._state_lock:
            self._jobs.append(sandbox)

    def _release(self, sandbox: Any) -> bool:
        """Terminate the sandbox of a job, once: a stop request, ``close`` on another
        thread and the end of ``run_job`` may all come to it.  Returns whether it was
        ended, now or by an earlier call."""
        with self._state_lock:
            if sandbox not in self._jobs:
                return not any(sandbox is other for other in self._lingering)
            self._jobs.remove(sandbox)
        if self._terminate(sandbox):
            return True
        with self._state_lock:
            self._lingering.append(sandbox)
        return False

    def _settle(self, sandbox: Any, job: _Job, result: JobResult, run_dir: Path) -> None:
        """The end of every job, however it went: no sandbox left, and the bill estimated
        and on file, where it is found even if the result never reaches the caller.

        A sandbox that could not be ended lives, and is billed, until Modal ends it:
        the run is then charged for as long as its sandbox can live, and told so.
        """
        if self._release(sandbox):
            result.cost_usd = self._record_cost(run_dir, job, final=True)
            return
        seconds = job.longest()
        result.cost_usd = self._record_cost(run_dir, job, final=True, seconds=seconds)
        if job.created is not None:
            result.notes.append(
                "The sandbox could not be ended (Modal did not answer). It ends by itself, "
                f"at the latest {seconds:.0f} s after it was made, and the run is charged "
                "for all of that time. Look for it on your Modal dashboard to stop it sooner."
            )

    def _reason(self, exc: BaseException) -> str:
        if isinstance(exc, ComputeError | _SandboxLost):
            return str(exc)
        return f"{type(exc).__name__}: {exc}"

    # -- sandbox plumbing --------------------------------------------------------

    def _exec(
        self,
        sandbox: Any,
        command: Sequence[str],
        *,
        cwd: str,
        limit: int,
        env: Mapping[str, str] | None = None,
    ) -> _Completed:
        """Run ``command`` under coreutils ``timeout`` and classify how it ended."""
        started = time.monotonic()
        process = sandbox.exec(
            "timeout",
            "--signal=TERM",
            f"--kill-after={_KILL_GRACE}",
            str(limit),
            *command,
            workdir=cwd,
            env=dict(env) if env else None,
            timeout=limit + _KILL_GRACE + _EXEC_MARGIN,
            # Modal's text mode decodes strictly and LaTeX logs are not always UTF-8.
            text=False,
        )
        stdout = _decode(_read_tail(process.stdout))
        stderr = _decode(_read_tail(process.stderr))
        returncode = process.wait()
        elapsed = time.monotonic() - started
        # -1: Modal's exec deadline fired.  137: SIGKILL, from `timeout`, the OOM killer or
        # the sandbox itself going away, which must not be reported as the job's own failure.
        if returncode in (-1, 137) and self._is_dead(sandbox):
            raise _SandboxLost("the sandbox stopped while the command was running")
        timed_out = returncode == -1 or (returncode in (124, 137) and elapsed >= limit)
        return _Completed(returncode, stdout, stderr, timed_out)

    def _with_sandbox(self, kind: SandboxKind, limit: int, job: Callable[[Any], T]) -> T:
        """Run ``job`` on the sandbox for ``kind``; retry once if that sandbox is gone."""
        retried = False
        while True:
            sandbox = self._acquire(kind, limit + _JOB_OVERHEAD)
            try:
                return job(sandbox)
            except (_SandboxLost, *self._transport_errors) as exc:
                if isinstance(exc, _SandboxLost) or self._is_dead(sandbox):
                    self._discard(kind, sandbox)
                    if not retried:
                        retried = True
                        log.warning(
                            "The Modal sandbox for %s jobs is no longer running (%s); "
                            "retrying on a fresh one.",
                            kind,
                            exc,
                        )
                        continue
                raise ComputeError(
                    f"Modal sandbox {kind} job failed: {type(exc).__name__}: {exc}"
                ) from exc

    def _acquire(self, kind: SandboxKind, needed: int) -> Any:
        if needed > self.sandbox_lifetime:
            raise ComputeError(
                f"a {kind} job with this timeout needs up to {needed} s, more than the sandbox "
                f"lifetime of {self.sandbox_lifetime:.0f} s; raise sandbox_lifetime"
            )
        slot = self._slots[kind]
        with slot.lock:
            if slot.sandbox is not None and slot.expires_at - time.monotonic() < needed:
                log.info("Replacing the Modal sandbox for %s jobs before it expires.", kind)
                with self._state_lock:
                    self._retired.append(slot.sandbox)
                slot.sandbox = None
            if slot.sandbox is None:
                slot.sandbox = self._create(kind)
                slot.expires_at = time.monotonic() + self.sandbox_lifetime
            return slot.sandbox

    def _create(self, kind: SandboxKind) -> Any:
        log.info(
            "Starting a Modal sandbox for %s jobs (the first use builds its image, which "
            "can take several minutes).",
            kind,
        )
        return self._start(
            f"{kind} jobs",
            image=python_image() if kind == "python" else latex_image(self._tex_packages),
            timeout=_seconds(self.sandbox_lifetime),
            idle_timeout=_seconds(self.idle_timeout),
            cpu=(min(0.5, self.cpu), self.cpu),
            memory=(min(1024, self.memory_mb), self.memory_mb),
            # Untrusted code: no way out, and nothing worth taking.
            block_network=True,
        )

    def _start(self, what: str, **options: Any) -> Any:
        """Create a sandbox for ``what``.  Whatever it is for, it holds no credentials:
        no ``secrets=`` is ever passed and it gets no identity token."""
        modal = self._modal
        try:
            return modal.Sandbox.create(
                # Keeps the sandbox up between commands whatever the image's default is.
                "sleep",
                "infinity",
                app=self._lookup_app(),
                include_oidc_identity_token=False,
                **options,
            )
        except modal.exception.AuthError as exc:
            # check() only sees whether a token is configured; one that Modal turns down
            # shows here.  It is turned down on every attempt, so the run should stop.
            raise self._unauthenticated(_STARTED_AGAIN if what == _A_JOB else "") from exc
        except self._transport_errors as exc:
            raise ComputeError(
                f"could not start a Modal sandbox for {what}: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _unauthenticated(experiment: str = "") -> BackendUnusable:
        """Modal turned the token of this machine down.  ``experiment`` is what a person
        can do next when it was an experiment's job that needed Modal; without it the
        message is the paper pipeline's."""
        if not experiment:
            return BackendUnusable(
                "Modal is not authenticated. Run `modal token new`, or set MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET."
            )
        return BackendUnusable(
            "Modal is not authenticated: it did not accept the token that is configured on "
            "this machine, so the experiment cannot run there. Run `modal token new` (or set "
            f"MODAL_TOKEN_ID and MODAL_TOKEN_SECRET), {experiment}"
        )

    def _lookup_app(self) -> Any:
        """The App that owns the sandboxes; ``None`` selects the running Function's App."""
        if not self._modal.is_local():
            return None
        with self._state_lock:
            if self._app is None:
                self._app = self._modal.App.lookup(self.app_name, create_if_missing=True)
            return self._app

    @property
    def _transport_errors(self) -> tuple[type[BaseException], ...]:
        """What a Modal call raises when the sandbox or the connection to it fails."""
        return (self._modal.Error, TimeoutError, ConnectionError)

    def _is_dead(self, sandbox: Any) -> bool:
        try:
            return sandbox.poll() is not None
        except self._transport_errors:
            return True

    def _discard(self, kind: SandboxKind, sandbox: Any) -> None:
        slot = self._slots[kind]
        with slot.lock:
            if slot.sandbox is sandbox:
                slot.sandbox = None
        self._terminate(sandbox)

    def _terminate(self, sandbox: Any) -> bool:
        """End ``sandbox``; returns whether Modal took the request."""
        ended = True
        try:
            sandbox.terminate()
        except self._transport_errors as exc:
            log.warning("Could not terminate a Modal sandbox: %s", exc)
            ended = False
        # Releases the client-side connection; the sandbox itself is already going away.
        with contextlib.suppress(*self._transport_errors):
            sandbox.detach()
        return ended

    def _remove(self, sandbox: Any, job_dir: str) -> None:
        """Best-effort cleanup of a job directory; the sandbox is disposable anyway."""
        try:
            sandbox.filesystem.remove(job_dir, recursive=True)
        except self._transport_errors as exc:
            log.debug("Could not remove %s in the Modal sandbox: %s", job_dir, exc)

    # -- lifecycle ---------------------------------------------------------------

    def check(self, *, python: bool = True, latex: bool = True) -> list[str]:
        """Why no sandbox could be started, found without starting one or contacting Modal."""
        needs_sandbox = python or (latex and self._local is None)
        # Inside a Modal Function the container's own identity starts the sandboxes.
        if not needs_sandbox or not self._modal.is_local() or modal_token_configured():
            return []
        missing = "Modal is not authenticated (no token is configured on this machine)"
        setup = "Run `modal setup` (or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)"
        if latex:
            # Asked before a paper is written.
            return [
                f"{missing}, so the sandboxes that compile the paper and run figure code "
                f"cannot be started. {setup}, or use --compute local."
            ]
        # Asked for code alone: before an experiment is approved or run (see the base class).
        return [
            f"{missing}, so the sandbox that would run the experiment cannot be started. "
            f"{setup} and approve the experiment again. {_OR_LOCAL}"
        ]

    def close(self) -> None:
        sandboxes: list[Any] = []
        for slot in self._slots.values():
            with slot.lock:
                if slot.sandbox is not None:
                    sandboxes.append(slot.sandbox)
                    slot.sandbox = None
        with self._state_lock:
            sandboxes.extend(self._retired)
            self._retired.clear()
            # A job that is still running is ended too; its run_job reports the loss.
            jobs = list(self._jobs)
            self._jobs.clear()
        for sandbox in sandboxes:
            self._terminate(sandbox)
        for sandbox in jobs:
            if not self._terminate(sandbox):
                # Its run_job charges it for as long as it can live (see _settle).
                with self._state_lock:
                    self._lingering.append(sandbox)

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": self.name,
            "app": self.app_name,
            "modal_version": getattr(self._modal, "__version__", None),
            "token_configured": modal_token_configured(),
            "inside_modal_function": not self._modal.is_local(),
            "isolation": "Modal Sandbox (network blocked, no secrets, no identity token)",
            "latex": self.latex,
            "cpu_limit": self.cpu,
            "memory_limit_mb": self.memory_mb,
            "sandbox_lifetime_s": self.sandbox_lifetime,
            "idle_timeout_s": self.idle_timeout,
            "sandboxes": {
                kind: getattr(slot.sandbox, "object_id", None) for kind, slot in self._slots.items()
            },
            "jobs": {
                "supported": True,
                "gpu": True,
                "network_isolation": True,
                "packages": True,
                "reattach": True,
                "max_timeout_s": _MAX_LIFETIME - _JOB_MARGIN,
                "prices_as_of": MODAL_PRICES_AS_OF,
                "summary": (
                    "Runs in a Modal sandbox made for the job: the pip packages and the GPU "
                    "it asks for, no credentials, and no network unless it asks for it. "
                    "Billed by Modal; the cost shown is an estimate."
                ),
            },
        }
        if self._local is not None:
            local = self._local.describe()
            info["pdflatex"] = local["pdflatex"]
            info["bibtex"] = local["bibtex"]
        return info
