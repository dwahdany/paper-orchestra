"""Run the PaperOrchestra pipeline on Modal.

From a checkout of this repository (the image is built from its ``uv.lock``)::

    uv run modal run -m paper_orchestra.modal_app \\
        --raw-materials-dir ./raw_materials --template cvpr2025 --plotting
    uv run modal deploy -m paper_orchestra.modal_app

``modal run`` uploads the raw materials and the template, starts ``write_paper``,
waits for it and downloads ``final_paper.pdf``, ``final_paper.tex`` and
``usage.json``.  A deployed app exposes ``write_paper(job)``,
``write_batch(jobs)`` and ``consolidate_memory(model)`` to
``modal.Function.from_name``.

Volume layout (``PAPER_ORCHESTRA_MODAL_VOLUME``, default ``paper-orchestra``,
mounted at ``/vol``)::

    inputs/<run_id>/raw_materials/   uploaded by the local entrypoint
    inputs/<run_id>/template/        the LaTeX template of that run
    runs/<run_id>/                   the pipeline's working directory and outputs
    memory/                          the lab wiki shared by all runs

Credentials.  ``write_paper`` attaches the Modal Secret named by
``PAPER_ORCHESTRA_MODAL_SECRET`` (default ``paper-orchestra``; ``none`` attaches
no Secret).  Two setups are supported:

* API key: ``modal secret create paper-orchestra ANTHROPIC_API_KEY=sk-ant-...``
  (add ``SEMANTIC_SCHOLAR_API_KEY``, ``GEMINI_API_KEY`` or ``OPENAI_API_KEY`` as
  needed).
* Keyless, with Workload Identity Federation: Modal gives every Function an OIDC
  identity token (issuer ``https://oidc.modal.com``, audience ``oidc.modal.com``)
  and the Anthropic client exchanges it for short-lived access tokens, so no
  Anthropic secret is stored anywhere.  The container only needs the non-secret
  ids ``ANTHROPIC_FEDERATION_RULE_ID`` and ``ANTHROPIC_ORGANIZATION_ID``, plus
  ``ANTHROPIC_SERVICE_ACCOUNT_ID`` / ``ANTHROPIC_WORKSPACE_ID`` when the
  federation rule asks for them.  Put them in the Secret, or export them in the
  shell that runs ``modal run`` / ``modal deploy``: they are forwarded to the
  Function.  The federation rule should pin Modal's ``workspace_id``,
  ``app_name`` and ``function_name`` claims.

In keyless mode the container must not also hold ``ANTHROPIC_API_KEY``,
``ANTHROPIC_AUTH_TOKEN`` or ``ANTHROPIC_PROFILE``: the SDK would silently prefer
them over the federated identity.  ``write_paper`` refuses to start when both
are present.

Isolation.  The pipeline compiles LaTeX inside its own container, which holds
the credentials, so TeX runs with paranoid file access.  Model-written figure
code runs in a nested Sandbox with no network, no Secret and no identity token.

Preemption.  Modal may restart a Function at any time.  The volume is committed
in the background every few seconds, between stages and once more before a
summary is returned, and the pipeline resumes from
``runs/<run_id>/state.json``, so calling ``write_paper`` again with the same job
continues where the last attempt stopped.  A run id must not be in flight twice
at the same time.

Lab memory.  Runs share the lab wiki on the volume, where a lock file cannot keep
two containers apart and the last writer of a file wins.  So a run only adds its
lessons to the wiki as notes, and ``consolidate_memory``, which never runs in
more than one container at a time, folds the notes into pages.  ``write_paper``
calls it after a successful run.

Other settings, read when the app is run or deployed: ``PAPER_ORCHESTRA_MODAL_APP``
(App name, default ``paper-orchestra``), ``PAPER_ORCHESTRA_MODAL_TIMEOUT``
(seconds per attempt of ``write_paper``, default six hours) and
``PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS`` (papers written in parallel, default 4).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, get_args

from paper_orchestra.compute.base import ComputeError
from paper_orchestra.compute.modal_backend import (
    APP_ENV,
    DEFAULT_APP_NAME,
    INSTALL_HINT,
    PYTHON_VERSION,
    TEXLIVE_INSTALL_COMMAND,
    ModalBackend,
    python_image,
)
from paper_orchestra.config import (
    BUNDLED_TEMPLATES_DIR,
    DEFAULT_MODEL,
    LAB_MEMORY_ENV,
    TEMPLATES_ENV,
    DreamMode,
    MemoryConfig,
    ModelConfig,
    PipelineConfig,
    default_research_cutoff,
    resolve_template_dir,
)
from paper_orchestra.context import RunPaths
from paper_orchestra.llm.base import Effort, LLMConfigError
from paper_orchestra.log import setup_logging
from paper_orchestra.memory.harness import ContextHarness

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover - needs an environment without the extra
    if exc.name != "modal":
        raise
    raise ModuleNotFoundError(
        f"paper_orchestra.modal_app is not available. {INSTALL_HINT}", name="modal"
    ) from exc

log = logging.getLogger(__name__)

VOLUME_ENV = "PAPER_ORCHESTRA_MODAL_VOLUME"
SECRET_ENV = "PAPER_ORCHESTRA_MODAL_SECRET"
TIMEOUT_ENV = "PAPER_ORCHESTRA_MODAL_TIMEOUT"
MAX_CONTAINERS_ENV = "PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS"
DEFAULT_VOLUME_NAME = "paper-orchestra"
DEFAULT_SECRET_NAME = "paper-orchestra"
NO_SECRET = "none"

VOLUME_MOUNT = "/vol"
CONTAINER_TEMPLATES = "/root/templates"
INPUTS_ROOT = "inputs"
RUNS_ROOT = "runs"
LAB_MEMORY_ROOT = "memory"

#: Extras installed in the image, so the optional providers work on Modal too.
IMAGE_EXTRAS = ("modal", "gemini", "openai")
#: ``add_local_python_source`` ships only ``*.py`` unless told otherwise.
PACKAGE_FILES = (
    "**/*.py",
    "**/*.md",
    "**/*.txt",
    "**/*.json",
    "**/*.yaml",
    "**/*.toml",
    "**/*.tex",
    "**/*.sty",
    "**/*.cls",
    "**/*.bst",
    "**/*.bib",
)

WIF_ID_VARS = (
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
)
#: Non-secret settings copied from the launching shell into the Function.
FORWARDED_VARS = (*WIF_ID_VARS, "PAPER_ORCHESTRA_STRUCTURED_OUTPUTS")
_WIF_REQUIRED = WIF_ID_VARS[:2]
#: Each of these outranks the WIF variables in the Anthropic SDK's credential chain.
_STATIC_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
_IDENTITY_TOKEN_VARS = (
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_IDENTITY_TOKEN",
    "MODAL_IDENTITY_TOKEN_FILE",
    "MODAL_IDENTITY_TOKEN",
)

_DEFAULT_TIMEOUT = 6 * 60 * 60
_MAX_TIMEOUT = 24 * 60 * 60
_CONSOLIDATE_TIMEOUT = 60 * 60
#: Volumes (v1) recommend at most five concurrent writers: four papers and the consolidation.
_DEFAULT_MAX_CONTAINERS = 4
_COMMIT_ATTEMPTS = 3
_COMMIT_RETRY_DELAY = 2.0

# Modal only issues identity tokens to Apps whose name matches this rule.
_MODAL_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_TEMPLATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RESEARCH_CUTOFF = re.compile(r"\d{4}-(0[1-9]|1[0-2])")
_CONTAINER_OWNED_FIELDS = frozenset({"base_dir", "template_dir", "compute"})
_CONTAINER_OWNED_MEMORY = ("project_dir", "lab_dir", "lab_consolidate")


class JobError(ValueError):
    """The job cannot run as configured; running it again would fail the same way."""


# -- settings -------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    app_name: str
    volume_name: str
    #: ``None`` attaches no named Secret (keyless setups that forward the WIF ids).
    secret_name: str | None
    #: Seconds one attempt of ``write_paper`` may run.
    timeout: int
    #: How many papers are written at the same time.
    max_containers: int


def _modal_name(variable: str, value: str) -> str:
    if not _MODAL_NAME.fullmatch(value):
        raise ValueError(
            f"{variable}={value!r} is not a valid Modal name: use at most 64 letters, digits, "
            "'.', '_' or '-'"
        )
    return value


def _int_setting(
    environ: Mapping[str, str], variable: str, default: int, low: int, high: int
) -> int:
    raw = environ.get(variable)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{variable}={raw!r} is not an integer") from None
    if not low <= value <= high:
        raise ValueError(f"{variable}={value} must be between {low} and {high}")
    return value


def load_settings(environ: Mapping[str, str]) -> Settings:
    """Read the deployment settings; an unset variable selects its default.

    ``PAPER_ORCHESTRA_MODAL_SECRET`` set to ``none`` (or to nothing) attaches no
    named Secret.
    """
    secret = environ.get(SECRET_ENV, DEFAULT_SECRET_NAME).strip()
    attach_secret = bool(secret) and secret.lower() != NO_SECRET
    return Settings(
        app_name=_modal_name(APP_ENV, environ.get(APP_ENV) or DEFAULT_APP_NAME),
        volume_name=_modal_name(VOLUME_ENV, environ.get(VOLUME_ENV) or DEFAULT_VOLUME_NAME),
        secret_name=_modal_name(SECRET_ENV, secret) if attach_secret else None,
        timeout=_int_setting(environ, TIMEOUT_ENV, _DEFAULT_TIMEOUT, 60, _MAX_TIMEOUT),
        max_containers=_int_setting(environ, MAX_CONTAINERS_ENV, _DEFAULT_MAX_CONTAINERS, 1, 1000),
    )


def function_env(settings: Settings, environ: Mapping[str, str]) -> dict[str, str]:
    """Environment given to the Functions: fixed container settings plus forwarded values.

    The container imports this module again, and Modal requires the Function
    definitions to come out the same there (same number of Secrets and
    Volumes), so the settings they depend on travel with the Function.
    """
    env = {
        APP_ENV: settings.app_name,
        VOLUME_ENV: settings.volume_name,
        SECRET_ENV: settings.secret_name or NO_SECRET,
        TIMEOUT_ENV: str(settings.timeout),
        MAX_CONTAINERS_ENV: str(settings.max_containers),
        LAB_MEMORY_ENV: f"{VOLUME_MOUNT}/{LAB_MEMORY_ROOT}",
        TEMPLATES_ENV: CONTAINER_TEMPLATES,
    }
    env.update({name: environ[name] for name in FORWARDED_VARS if environ.get(name)})
    return env


def check_credentials(environ: Mapping[str, str]) -> str:
    """Name the Anthropic credential the container will use, without revealing it.

    Returns ``"workload_identity"``, ``"api_key"``, ``"auth_token"`` or
    ``"none"``.  Raises :class:`JobError` when Workload Identity Federation is
    configured next to a static credential, or without an identity token.
    """
    static = [name for name in _STATIC_CREDENTIALS if environ.get(name)]
    if all(environ.get(name) for name in _WIF_REQUIRED):
        if static:
            raise JobError(
                f"{' and '.join(static)} is set next to the Workload Identity Federation ids "
                f"({', '.join(_WIF_REQUIRED)}). The Anthropic SDK would silently use it instead "
                f"of the keyless identity. Remove it from the Modal Secret (or set {SECRET_ENV}="
                f"{NO_SECRET} to attach no Secret), or drop the federation ids, so that exactly "
                "one credential is configured."
            )
        if not any(environ.get(name) for name in _IDENTITY_TOKEN_VARS):
            raise JobError(
                "Workload Identity Federation ids are set but Modal provided no identity token "
                "(MODAL_IDENTITY_TOKEN_FILE and MODAL_IDENTITY_TOKEN are unset). Modal only "
                "issues tokens to Apps whose name consists of letters, digits, '.', '_' and '-'."
            )
        return "workload_identity"
    if environ.get("ANTHROPIC_API_KEY"):
        return "api_key"
    if environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "auth_token"
    return "none"


# -- jobs and paths -------------------------------------------------------------


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise JobError(
            f"run_id must be 1 to 80 letters, digits, '.', '_' or '-' and start with a letter "
            f"or digit; got {run_id!r}"
        )
    return run_id


def container_path(volume_path: object, mount: Path) -> Path:
    """Where a volume-relative path lives inside the container."""
    relative = PurePosixPath(volume_path) if isinstance(volume_path, str) else PurePosixPath()
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise JobError(f"not a path inside the volume: {volume_path!r}")
    return mount.joinpath(*relative.parts)


def volume_path(path: Path, mount: Path) -> str:
    """The volume-relative form of a container path under ``mount``."""
    return path.relative_to(mount).as_posix()


def build_job(
    *,
    run_id: str,
    template: str | None = None,
    template_dir: str | None = None,
    raw_materials_dir: str | None = None,
    idea_filename: str = "idea_sparse.md",
    experimental_log_filename: str = "experimental_log.md",
    research_cutoff: str | None = None,
    use_plotting: bool = False,
    plotting_max_critic_rounds: int | None = None,
    writer_model: str | None = None,
    reflection_model: str | None = None,
    plotting_model: str | None = None,
    fast_model: str | None = None,
    image_model: str | None = None,
    effort: str | None = None,
    memory: bool = True,
    lab_memory: bool = True,
    dream: str = "stages",
    resume: bool = True,
) -> dict[str, Any]:
    """Describe one paper as a JSON-serialisable job for ``write_paper``.

    ``template`` names a template bundled in the image; ``template_dir`` and
    ``raw_materials_dir`` are volume-relative directories.  Unset options are
    left out so the container applies the pipeline defaults, except the
    research cutoff, which is fixed now so a retry next month sees the same one.
    """
    if effort is not None and effort not in get_args(Effort):
        raise JobError(f"effort must be one of {', '.join(get_args(Effort))}; got {effort!r}")
    if dream not in get_args(DreamMode):
        raise JobError(f"dream must be one of {', '.join(get_args(DreamMode))}; got {dream!r}")
    cutoff = research_cutoff or default_research_cutoff()
    if not _RESEARCH_CUTOFF.fullmatch(cutoff):
        raise JobError(f"research_cutoff must look like 2024-11; got {cutoff!r}")
    models = {
        "writer": writer_model,
        "reflection": reflection_model,
        "plotting": plotting_model,
        "fast": fast_model,
        "image": image_model,
        "effort": effort,
    }
    job: dict[str, Any] = {
        "run_id": validate_run_id(run_id),
        "template": template,
        "template_dir": template_dir,
        "raw_materials_dir": raw_materials_dir,
        "idea_filename": idea_filename,
        "experimental_log_filename": experimental_log_filename,
        "research_cutoff": cutoff,
        "use_plotting": use_plotting,
        "plotting_max_critic_rounds": plotting_max_critic_rounds,
        "resume": resume,
        "models": {role: value for role, value in models.items() if value},
        "memory": {"enabled": memory, "lab": lab_memory, "dream": dream},
    }
    return {key: value for key, value in job.items() if value is not None}


@dataclass(frozen=True)
class JobSpec:
    run_id: str
    config: PipelineConfig
    #: Uploaded raw materials to copy into the run directory; ``None`` when they are
    #: already at ``runs/<run_id>/raw_materials``.
    raw_materials_dir: Path | None


def job_spec(job: Mapping[str, Any], mount: Path = Path(VOLUME_MOUNT)) -> JobSpec:
    """Turn a job into the pipeline configuration used inside the container.

    Besides ``run_id``, ``template`` / ``template_dir`` and ``raw_materials_dir``
    a job may set any :class:`PipelineConfig` field except the paths and the
    compute backend, which are decided here.
    """
    if not isinstance(job, Mapping):
        raise JobError(f"a job must be a dict, got {type(job).__name__}")
    options = dict(job)
    run_id = validate_run_id(options.pop("run_id", None))
    template = options.pop("template", None)
    template_dir = options.pop("template_dir", None)
    raw_materials_dir = options.pop("raw_materials_dir", None)

    allowed = {f.name for f in dataclasses.fields(PipelineConfig)} - _CONTAINER_OWNED_FIELDS
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise JobError(
            f"unknown job option(s): {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )
    if (template is None) == (template_dir is None):
        raise JobError(
            "a job needs exactly one of 'template' (a template bundled in the image) and "
            "'template_dir' (a directory on the volume)"
        )
    if template_dir is not None:
        resolved_template = container_path(template_dir, mount)
    elif isinstance(template, str) and _TEMPLATE_NAME.fullmatch(template):
        resolved_template = Path(CONTAINER_TEMPLATES) / template
    else:
        raise JobError(f"'template' must be the name of a bundled template; got {template!r}")

    for section in ("models", "memory"):
        if not isinstance(options.get(section, {}), Mapping):
            raise JobError(f"'{section}' must be a mapping, got {options[section]!r}")
    memory = dict(options.pop("memory", {}))
    fixed = sorted(key for key in _CONTAINER_OWNED_MEMORY if memory.get(key))
    if fixed:
        raise JobError(
            f"memory.{' and memory.'.join(fixed)} cannot be set: on Modal the wikis live on "
            "the volume and consolidate_memory maintains the shared one"
        )
    memory["lab_dir"] = str(mount / LAB_MEMORY_ROOT)
    # The run leaves its lessons as notes; consolidate_memory is the one writer of lab pages.
    memory["lab_consolidate"] = False
    try:
        config = PipelineConfig.from_dict(
            {
                **options,
                "base_dir": str(mount / RUNS_ROOT / run_id),
                "template_dir": str(resolved_template),
                "compute": "modal",
                "memory": memory,
            }
        )
    except TypeError as exc:  # an unknown key under "models" or "memory"
        raise JobError(f"invalid job options: {exc}") from exc
    return JobSpec(
        run_id=run_id,
        config=config,
        raw_materials_dir=(
            container_path(raw_materials_dir, mount) if raw_materials_dir is not None else None
        ),
    )


# -- running a job inside the container -------------------------------------------


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def failure_summary(run_id: object, error: str, seconds: float = 0.0) -> dict[str, Any]:
    return {
        "success": False,
        "run_id": run_id if isinstance(run_id, str) else None,
        "error": error,
        "volume": SETTINGS.volume_name,
        "run_dir": None,
        "files": {},
        "stages": {},
        "usage": {},
        "seconds": round(seconds, 1),
    }


def summarize(result: Any, spec: JobSpec, mount: Path, seconds: float) -> dict[str, Any]:
    """Describe a finished run (a ``PaperResult``) with volume-relative paths."""
    paths = RunPaths(spec.config.base_dir)
    outputs = {"pdf": paths.final_pdf, "tex": paths.final_tex, "usage": paths.usage}
    return _jsonable(
        {
            "success": bool(result.success),
            "run_id": spec.run_id,
            "error": result.error,
            "volume": SETTINGS.volume_name,
            "run_dir": volume_path(spec.config.base_dir, mount),
            "files": {
                kind: volume_path(path, mount) for kind, path in outputs.items() if path.is_file()
            },
            "stages": result.stages,
            "usage": result.usage,
            "seconds": round(seconds, 1),
        }
    )


class StageCheckpoint:
    """Commit the volume between the stages of the pipeline.

    Called whenever the pipeline refreshes its memory, which it does when a
    stage has settled.  An explicit commit may also reload the volume, which
    fails while files are open and hides the volume from other threads for a
    moment.  So only the thread that drives the pipeline commits: stages that
    run side by side settle on worker threads and are covered by the next
    checkpoint.  A checkpoint that fails is only logged; Modal's background
    commits cover the gap.
    """

    def __init__(self, commit: Callable[[], None]) -> None:
        self._commit = commit
        self._driver = threading.current_thread()

    def __call__(self) -> None:
        if threading.current_thread() is not self._driver:
            return
        try:
            self._commit()
        except Exception as exc:
            log.warning("Could not commit the volume between stages: %s", exc)
        else:
            log.info("Committed the volume between stages.")


class _CheckpointingHarness(ContextHarness):
    """The pipeline's harness, with a callback that runs after every memory refresh.

    ``run_paper`` reports no progress to its caller, but it refreshes the
    memory after every stage, so the refresh marks the stage boundaries.
    """

    after_refresh: Callable[[], None] | None = None

    def refresh(self) -> None:
        try:
            super().refresh()
        finally:
            if self.after_refresh is not None:
                self.after_refresh()


def commit_volume(commit: Callable[[], None]) -> None:
    """Make everything written so far durable and visible outside the container."""
    for attempt in range(1, _COMMIT_ATTEMPTS + 1):
        try:
            commit()
            return
        except Exception as exc:
            if attempt == _COMMIT_ATTEMPTS:
                raise
            # Typically a file on the volume was still open; give it a moment to close.
            log.warning("Committing the volume failed (%s); trying again.", exc)
            time.sleep(_COMMIT_RETRY_DELAY)


def _reload_volume(reload: Callable[[], None] | None) -> None:
    """Pick up what other containers committed since this one mounted the volume."""
    if reload is None:
        return
    try:
        reload()
    except RuntimeError as exc:
        log.warning("Could not reload the volume: %s", exc)


def run_job(
    job: Mapping[str, Any],
    *,
    commit: Callable[[], None],
    reload: Callable[[], None] | None = None,
    consolidate: Callable[[str], Any] | None = None,
    mount: Path = Path(VOLUME_MOUNT),
) -> dict[str, Any]:
    """Run one paper inside the container and return its summary.

    Problems that a retry cannot fix (a malformed job, missing inputs, unusable
    credentials) are returned as a failed summary.  Anything else propagates so
    that Modal retries the call, which resumes the run.  The summary is returned
    only after a final commit, so the files it lists can be downloaded at once.

    ``consolidate(model)`` is ``consolidate_memory``: after a successful run it
    folds the lessons the run left in the lab wiki into pages.
    """
    started = time.monotonic()
    run_id = job.get("run_id") if isinstance(job, Mapping) else None
    # Inputs may have been uploaded after this container mounted the volume.
    _reload_volume(reload)
    lessons_model: str | None = None
    try:
        spec = job_spec(job, mount)
        credential = check_credentials(os.environ)
        if credential == "none":
            log.warning(
                "No Anthropic credential is configured in this container (no API key in the "
                "Modal Secret and no Workload Identity Federation ids); Claude models will fail."
            )
        else:
            log.info("Anthropic credential in use: %s.", credential)
        if not spec.config.template_dir.is_dir():
            raise JobError(f"LaTeX template not found in the container: {spec.config.template_dir}")
        # Raises ComputeError here, before any model call, if the image lacks TeX Live.
        backend = ModalBackend(latex="local")
        summary = _run_pipeline(spec, backend, commit, mount, started)
        memory = spec.config.memory
        # Lessons reach the lab wiki in the last stage, and only a model can fold them in.
        if summary["success"] and memory.resolved_lab_dir() is not None and memory.dream != "off":
            lessons_model = spec.config.models.reflection
    except (FileNotFoundError, ValueError, LLMConfigError, ComputeError) as exc:
        log.error("Run %s cannot proceed: %s", run_id, exc)
        summary = failure_summary(run_id, str(exc), time.monotonic() - started)
    except Exception:
        try:
            commit()
        except Exception as exc:
            log.warning("Could not commit the volume after a failure: %s", exc)
        raise
    commit_volume(commit)
    if consolidate is not None and lessons_model is not None:
        _share_lessons(consolidate, lessons_model)
    return summary


def _run_pipeline(
    spec: JobSpec, backend: ModalBackend, commit: Callable[[], None], mount: Path, started: float
) -> dict[str, Any]:
    from paper_orchestra.pipeline import prepare_run_dir, run_paper

    config = spec.config
    os.environ[LAB_MEMORY_ENV] = str(mount / LAB_MEMORY_ROOT)
    # This container holds the credentials and compiles model-written LaTeX: paranoid input
    # keeps TeX from reading /proc/self/environ or the identity token file into the PDF.
    os.environ["openin_any"] = "p"  # noqa: SIM112 - kpathsea reads the lowercase name
    with backend:
        prepare_run_dir(config, spec.raw_materials_dir)
        harness = _CheckpointingHarness.create(config)
        harness.after_refresh = StageCheckpoint(commit)
        log.info("Writing paper %s in %s.", spec.run_id, config.base_dir)
        result = run_paper(config, harness=harness, compute=backend)
    return summarize(result, spec, mount, time.monotonic() - started)


def _share_lessons(consolidate: Callable[[str], Any], model: str) -> None:
    """Have the lab wiki consolidated; a failure there is not a failure of the paper."""
    try:
        report = consolidate(model)
    except Exception as exc:
        log.warning(
            "Could not consolidate the lab memory (%s: %s). The lessons of this run stay in "
            "it as notes until the next consolidation.",
            type(exc).__name__,
            exc,
        )
        return
    if report.get("error"):
        log.warning("The lab memory was not consolidated: %s", report["error"])
    else:
        log.info(
            "Lab memory consolidated: %d note(s) folded in, %d page(s) changed.",
            report.get("notes_archived", 0),
            len(report.get("pages_changed", ())),
        )


def consolidate_lab(
    model: str,
    *,
    commit: Callable[[], None],
    reload: Callable[[], None] | None = None,
    mount: Path = Path(VOLUME_MOUNT),
) -> dict[str, Any]:
    """Fold the notes that runs left in the lab wiki into its pages; return the report.

    ``model`` holds the consolidation session, which sees the lab wiki and
    nothing of any paper.  Only one of these may run at a time: it is the single
    writer of the lab's pages.
    """
    _reload_volume(reload)
    check_credentials(os.environ)
    with tempfile.TemporaryDirectory(prefix="paper-orchestra-lab-") as scratch:
        # A harness is built around a project wiki; no paper is involved here, so an empty
        # one in a scratch directory stands in.
        config = PipelineConfig(
            base_dir=Path(scratch),
            template_dir=Path(scratch),
            models=ModelConfig(reflection=model),
            memory=MemoryConfig(lab_dir=mount / LAB_MEMORY_ROOT, dream="end"),
        )
        report = ContextHarness.create(config).consolidate_lab_memory()
    commit_volume(commit)
    return _jsonable(report.to_dict()) if report is not None else {}


def check_batch(jobs: Sequence[Mapping[str, Any]]) -> None:
    """Reject a batch in which two jobs would write to the same run directory."""
    seen: set[str] = set()
    for job in jobs:
        run_id = validate_run_id(job.get("run_id") if isinstance(job, Mapping) else None)
        if run_id in seen:
            raise JobError(f"run_id {run_id!r} appears more than once in the batch")
        seen.add(run_id)


def batch_results(
    jobs: Sequence[Mapping[str, Any]], outcomes: Iterable[Any]
) -> list[dict[str, Any]]:
    """Pair each job with its summary; a call that raised becomes a failed summary."""
    results: list[dict[str, Any]] = []
    for job, outcome in zip(jobs, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            log.error("Run %s failed: %r", job.get("run_id"), outcome)
            outcome = failure_summary(job.get("run_id"), f"{type(outcome).__name__}: {outcome}")
        results.append(outcome)
    return results


# -- launching from this machine ---------------------------------------------------


def _visible_files(root: Path) -> Iterator[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(files):
            if not name.startswith("."):
                yield Path(current) / name


def upload_inputs(
    volume: Any, run_id: str, raw_materials_dir: Path, template_dir: Path
) -> tuple[str, str]:
    """Upload a run's inputs; return the volume paths of the raw materials and the template."""
    targets = (
        (raw_materials_dir, f"{INPUTS_ROOT}/{run_id}/raw_materials"),
        (template_dir, f"{INPUTS_ROOT}/{run_id}/template"),
    )
    # force: launching a run id again, to resume it, uploads the same files once more.
    with volume.batch_upload(force=True) as batch:
        for source, target in targets:
            for path in _visible_files(source):
                batch.put_file(path, f"/{target}/{path.relative_to(source).as_posix()}")
    return targets[0][1], targets[1][1]


def wait_for_result(call: Any, *, poll_seconds: float = 300.0) -> dict[str, Any]:
    """Block until a spawned ``write_paper`` call finishes, logging a heartbeat meanwhile.

    The run's own log lines are streamed by ``modal run``; the heartbeat only
    shows that this side is still waiting.
    """
    started = time.monotonic()
    while True:
        try:
            return call.get(timeout=poll_seconds)
        except TimeoutError:  # the builtin: FunctionCall.get raises it while the call runs
            minutes = (time.monotonic() - started) / 60
            log.info("The run is still in progress after %.0f min.", minutes)


def download_outputs(volume: Any, summary: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Copy the files a summary lists from the volume into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for kind, remote in summary.get("files", {}).items():
        target = output_dir / PurePosixPath(remote).name
        partial = target.with_name(f".{target.name}.part")
        try:
            with partial.open("wb") as handle:
                for chunk in volume.read_file(remote):
                    handle.write(chunk)
            os.replace(partial, target)
        except FileNotFoundError:
            log.warning("%s is not on the volume (yet); it was not downloaded.", remote)
            continue
        finally:
            partial.unlink(missing_ok=True)
        saved[kind] = target
    return saved


def launch(
    *,
    volume: Any,
    function: Any,
    raw_materials_dir: str | Path,
    template: str | Path,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    poll_seconds: float = 300.0,
    **options: Any,
) -> dict[str, Any]:
    """Upload the inputs, run ``function`` (``write_paper``) and download the results.

    ``options`` are the remaining :func:`build_job` arguments.  Returns the run
    summary with a ``downloaded`` entry mapping each fetched file to its local
    path.  Launching the same ``run_id`` again resumes that run.
    """
    raw = Path(raw_materials_dir).expanduser()
    if not raw.is_dir():
        raise FileNotFoundError(f"raw materials directory not found: {raw}")
    # Everything is validated here, before anything is uploaded or started.
    job = build_job(run_id=run_id or new_run_id(), **options)
    run = job["run_id"]
    for name in (job["idea_filename"], job["experimental_log_filename"]):
        if not (raw / name).is_file():
            raise FileNotFoundError(f"required file not found: {raw / name}")
    template_dir = resolve_template_dir(template)
    for name in ("template.tex", "guidelines.md"):
        if not (template_dir / name).is_file():
            raise FileNotFoundError(f"the LaTeX template {template_dir} has no {name}")

    log.info("Uploading %s and template %s for run %s.", raw, template_dir, run)
    job["raw_materials_dir"], job["template_dir"] = upload_inputs(volume, run, raw, template_dir)
    call = function.spawn(job)
    log.info(
        "Started run %s (function call %s). Launching this run id again resumes it.",
        run,
        getattr(call, "object_id", "unknown"),
    )
    summary = wait_for_result(call, poll_seconds=poll_seconds)

    target = Path(output_dir).expanduser() if output_dir else Path(f"paper_output_{run}")
    saved = download_outputs(volume, summary, target)
    if summary.get("success") and "pdf" in saved:
        log.info("Paper written to %s.", saved["pdf"])
    else:
        log.error(
            "Run %s did not produce a paper: %s",
            run,
            summary.get("error") or "final_paper.pdf is missing from the volume",
        )
    return {**summary, "downloaded": {kind: str(path) for kind, path in saved.items()}}


# -- Modal objects ------------------------------------------------------------------


def _pipeline_image() -> modal.Image:
    # In a checkout this file is src/paper_orchestra/modal_app.py.  Inside the container the
    # path points nowhere, which is fine: image steps are only evaluated when building.
    root = Path(__file__).resolve().parents[2]
    templates = os.environ.get(TEMPLATES_ENV) or str(BUNDLED_TEMPLATES_DIR)
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .run_commands(TEXLIVE_INSTALL_COMMAND)
        # Third-party dependencies exactly as locked; uv_sync does not install the project.
        .uv_sync(str(root), extras=list(IMAGE_EXTRAS), extra_options="--no-dev")
        .env({"PYTHONUNBUFFERED": "1"})
        .add_local_python_source(
            "paper_orchestra", ignore=~modal.FilePatternMatcher(*PACKAGE_FILES)
        )
        .add_local_dir(templates, CONTAINER_TEMPLATES)
    )


SETTINGS = load_settings(os.environ)

app = modal.App(SETTINGS.app_name, include_source=False)
volume = modal.Volume.from_name(SETTINGS.volume_name, create_if_missing=True)
image = _pipeline_image()
# Listed after the named Secret so that these values win.
_env_secret = modal.Secret.from_dict(function_env(SETTINGS, os.environ))
_named_secrets = [modal.Secret.from_name(SETTINGS.secret_name)] if SETTINGS.secret_name else []


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[*_named_secrets, _env_secret],
    timeout=SETTINGS.timeout,
    # A retry (or a restart after preemption) resumes from state.json on the volume.
    retries=modal.Retries(
        max_retries=2, initial_delay=10.0, backoff_coefficient=2.0, max_delay=60.0
    ),
    max_containers=SETTINGS.max_containers,
    # One paper per container: no state, open files or memory shared between papers.
    single_use_containers=True,
    cpu=2.0,
    memory=4096,
)
def write_paper(job: dict[str, Any]) -> dict[str, Any]:
    """Write one paper.  ``job`` is described by :func:`build_job` and :func:`job_spec`.

    Returns a JSON-serialisable summary: ``success``, ``run_id``, ``error``,
    ``volume``, ``run_dir``, ``files`` (volume paths of the PDF, the LaTeX
    source and the usage report), ``stages``, ``usage`` and ``seconds``.
    """
    setup_logging(rich=False)
    return run_job(
        job, commit=volume.commit, reload=volume.reload, consolidate=consolidate_memory.remote
    )


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[*_named_secrets, _env_secret],
    timeout=_CONSOLIDATE_TIMEOUT,
    # The single writer of the lab wiki's pages: calls queue up and run one at a time, each in
    # a fresh container that sees what the runs before it committed.
    max_containers=1,
    single_use_containers=True,
    memory=1024,
)
def consolidate_memory(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Fold the lessons that runs left as notes in the lab wiki into its pages.

    ``write_paper`` calls this after a successful run.  Returns the
    consolidation report (``notes_seen``, ``notes_archived``, ``pages_changed``,
    ``error``, ...).
    """
    setup_logging(rich=False)
    return consolidate_lab(model, commit=volume.commit, reload=volume.reload)


@app.function(
    image=image,
    secrets=[_env_secret],
    timeout=_MAX_TIMEOUT,
    memory=1024,
    # A restart after preemption would map every job again while runs of the first attempt
    # may still be in flight; this container only waits, so keeping it costs little.
    nonpreemptible=True,
)
def write_batch(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write several papers, at most ``PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS`` at a time.

    Returns one summary per job, in order; a job whose call raised is reported
    as a failed summary instead of failing the batch.
    """
    setup_logging(rich=False)
    check_batch(jobs)
    return batch_results(jobs, write_paper.map(jobs, return_exceptions=True))


@app.local_entrypoint()
def main(
    raw_materials_dir: str,
    template: str,
    output_dir: str | None = None,
    run_id: str | None = None,
    plotting: bool = False,
    idea_filename: str = "idea_sparse.md",
    experimental_log_filename: str = "experimental_log.md",
    research_cutoff: str | None = None,
    writer_model: str | None = None,
    reflection_model: str | None = None,
    plotting_model: str | None = None,
    fast_model: str | None = None,
    image_model: str | None = None,
    effort: str | None = None,
    plotting_max_critic_rounds: int | None = None,
    memory: bool = True,
    lab_memory: bool = True,
    dream: DreamMode = "stages",
    resume: bool = True,
) -> None:
    """Write one paper on Modal from a local raw-materials directory.

    The options mirror ``paper-orchestra write``.  ``template`` is a template
    name (``cvpr2025``) or a directory.  Pass the ``--run-id`` of an interrupted
    run to resume it.
    """
    setup_logging()
    if plotting:
        # Built here, where the build log is visible, rather than by the first figure of the run.
        python_image().build(app)
    summary = launch(
        volume=volume,
        function=write_paper,
        raw_materials_dir=raw_materials_dir,
        template=template,
        output_dir=output_dir,
        run_id=run_id,
        use_plotting=plotting,
        idea_filename=idea_filename,
        experimental_log_filename=experimental_log_filename,
        research_cutoff=research_cutoff,
        writer_model=writer_model,
        reflection_model=reflection_model,
        plotting_model=plotting_model,
        fast_model=fast_model,
        image_model=image_model,
        effort=effort,
        plotting_max_critic_rounds=plotting_max_critic_rounds,
        memory=memory,
        lab_memory=lab_memory,
        dream=dream,
        resume=resume,
    )
    if not summary["success"] or "pdf" not in summary["downloaded"]:
        raise SystemExit(1)
