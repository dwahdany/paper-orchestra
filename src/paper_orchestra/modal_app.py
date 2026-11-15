"""Run the PaperOrchestra pipeline on Modal.

From a checkout of this repository (the image is built from its ``pyproject.toml``
and ``uv.lock``, so the app cannot be run or deployed from an installed wheel)::

    uv run modal run -m paper_orchestra.modal_app \\
        --raw-materials-dir ./raw_materials --template cvpr2025 --plotting
    uv run modal deploy -m paper_orchestra.modal_app

``modal run`` uploads the raw materials and the template, starts ``write_paper``,
waits for it and downloads ``final_paper.pdf``, ``final_paper.tex`` and
``usage.json``.  A deployed app exposes ``write_paper(job)``,
``write_batch(jobs)`` and ``consolidate_memory(model)`` to
``modal.Function.from_name``; ``paper-orchestra modal batch`` calls
``write_paper`` once per paper (see :mod:`paper_orchestra.modal_batch`).

Volume layout (``PAPER_ORCHESTRA_MODAL_VOLUME``, default ``paper-orchestra``,
mounted at ``/vol``)::

    inputs/<run_id>/raw_materials/   uploaded by the local entrypoint
    inputs/<run_id>/template/        the LaTeX template of that run
    runs/<run_id>/                   the pipeline's working directory and outputs
    runs/<run_id>/.lease.json        present while a call is writing that run
    memory/                          the lab wiki shared by all runs
    memory/cache/                    what the pipeline keeps across runs, beside the wiki
    memory/.lease.json               present while the wiki is being consolidated

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
  Function.  The federation rule should pin Modal's ``workspace_id`` and
  ``app_name`` claims.  A rule that also pins ``function_name`` has to admit both
  ``write_paper`` and ``consolidate_memory``: each Function presents a token of
  its own.  When the token exchange is refused, the error says what the token
  of that container looked like and what to compare it with on the Anthropic
  side.

In keyless mode the container must not also hold ``ANTHROPIC_API_KEY``,
``ANTHROPIC_AUTH_TOKEN`` or ``ANTHROPIC_PROFILE``: the SDK would silently prefer
them over the federated identity.  ``write_paper`` refuses to start when both
are present.

Isolation.  The container of ``write_paper`` holds the credentials, so nothing
model-written runs in it: LaTeX is compiled, and figure code is run, in nested
Sandboxes with no network, no Secret and no identity token.  One LaTeX Sandbox
serves all the builds of a run.  ``PAPER_ORCHESTRA_MODAL_LATEX=local`` compiles
inside the container instead, which is faster; TeX then runs with a scrubbed
environment and paranoid file access, and documents that use file-reading
constructs are refused either way.

Preemption.  Modal may restart a Function at any time.  The volume is committed
in the background every few seconds, between stages and once more before a
summary is returned, and the pipeline resumes from
``runs/<run_id>/state.json``, so calling ``write_paper`` again with the same job
continues where the last attempt stopped.

One run per run id.  Two containers driving the same run directory would repeat
each other's stages and overwrite each other's files, so ``write_paper`` takes
a lease on the run directory: a second call for a run id that is in flight
returns a failed summary that says so instead of running.  A container that
was stopped (its call cancelled, ``modal run`` interrupted) leaves its lease
behind.  A call that finds a lease therefore watches it: a heartbeat that moves
belongs to a run in flight, one that has stopped is taken over once it is a few
minutes old, and the run is resumed.  A retry of the same call takes its lease
over at once.

Lab memory.  Runs share the lab wiki on the volume, where a lock file cannot keep
two containers apart and the last writer of a file wins.  So a run never writes
the wiki's pages or its index: it only adds files of its own, its lessons as
notes and what it looked up as cache entries, and ``consolidate_memory`` folds
the notes into pages and rewrites the index.  That Function never runs in more
than one container at a time, and it takes a lease on the wiki, so that two
Apps that share a volume (two ``modal run`` at once, say) do not consolidate it
together either.  ``write_paper`` calls it after a successful run, and once
before a run on a volume whose lab wiki does not exist yet.

Other settings, read when the app is run or deployed: ``PAPER_ORCHESTRA_MODAL_APP``
(App name, default ``paper-orchestra``), ``PAPER_ORCHESTRA_MODAL_TIMEOUT``
(seconds per attempt of ``write_paper``, default six hours),
``PAPER_ORCHESTRA_MODAL_MAX_CONTAINERS`` (papers written in parallel, default 4),
``PAPER_ORCHESTRA_MODAL_LATEX`` (``sandbox`` or ``local``, see above) and
``PAPER_ORCHESTRA_MODAL_TEX_PACKAGES`` (extra Debian packages for the TeX Live
images, separated by spaces).
"""

from __future__ import annotations

import base64
import contextlib
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
    TEX_PACKAGES_ENV,
    LatexMode,
    ModalBackend,
    extra_tex_packages,
    latex_image,
    python_image,
    texlive_install_command,
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
from paper_orchestra.memory.wiki import INDEX_FILE

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
LATEX_ENV = "PAPER_ORCHESTRA_MODAL_LATEX"
DEFAULT_VOLUME_NAME = "paper-orchestra"
DEFAULT_SECRET_NAME = "paper-orchestra"
NO_SECRET = "none"

#: Longest run id, in characters.  A run id is a directory name on the volume.
RUN_ID_MAX = 80

VOLUME_MOUNT = "/vol"
CONTAINER_TEMPLATES = "/root/templates"
INPUTS_ROOT = "inputs"
RUNS_ROOT = "runs"
LAB_MEMORY_ROOT = "memory"
#: In a directory of the volume while a call is writing it: who that is and when it last
#: said so.
LEASE_NAME = ".lease.json"

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
#: Seconds between the heartbeats of a lease.
_LEASE_REFRESH = 30.0
#: A lease whose heartbeat is older than this was left by a container that is gone.  Several
#: refresh periods: other containers only see a heartbeat once a background commit of the
#: volume has published it.
_LEASE_STALE = 5 * _LEASE_REFRESH
#: After taking a lease, wait this long before checking that it is still ours, so that a
#: container that took it in the same moment has had time to commit its own.
_LEASE_SETTLE = 3.0
#: Seconds between two looks at a lease that another call holds.
_LEASE_POLL = 5.0
#: How the Anthropic client words a credential it could not use (llm/anthropic_client.py).
_CREDENTIAL_FAILURE = re.compile(
    r"credentials could not be used|WorkloadIdentityError|IdentityTokenFileError"
)
#: Claims of Modal's identity token that a federation rule can match on.  None is a secret.
_TOKEN_CLAIMS = (
    "iss",
    "aud",
    "workspace_id",
    "environment_name",
    "app_name",
    "function_name",
)

# Modal only issues identity tokens to Apps whose name matches this rule.
_MODAL_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
_RUN_ID = re.compile(rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,{RUN_ID_MAX - 1}}}")
_TEMPLATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RESEARCH_CUTOFF = re.compile(r"\d{4}-(0[1-9]|1[0-2])")
_CONTAINER_OWNED_FIELDS = frozenset({"base_dir", "template_dir", "compute"})
_CONTAINER_OWNED_MEMORY = ("project_dir", "lab_dir", "lab_consolidate")


class JobError(ValueError):
    """The job cannot run as configured; running it again would fail the same way."""


class LeaseHeld(JobError):
    """Another call is writing this part of the volume right now."""


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
    #: Where ``write_paper`` compiles LaTeX: in a nested Sandbox or in its own container.
    latex: LatexMode = "sandbox"
    #: Extra Debian packages installed in the TeX Live images.
    tex_packages: tuple[str, ...] = ()


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


def latex_mode(environ: Mapping[str, str]) -> LatexMode:
    """Where ``write_paper`` compiles LaTeX, from ``PAPER_ORCHESTRA_MODAL_LATEX``."""
    raw = environ.get(LATEX_ENV) or ""
    mode = raw.strip().lower() or "sandbox"
    if mode not in get_args(LatexMode):
        raise ValueError(f"{LATEX_ENV}={raw!r} must be one of: {', '.join(get_args(LatexMode))}")
    return mode  # type: ignore[return-value]


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
        latex=latex_mode(environ),
        tex_packages=extra_tex_packages(environ),
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
        LATEX_ENV: settings.latex,
        LAB_MEMORY_ENV: f"{VOLUME_MOUNT}/{LAB_MEMORY_ROOT}",
        TEMPLATES_ENV: CONTAINER_TEMPLATES,
    }
    if settings.tex_packages:
        # The container builds the image of its LaTeX Sandbox from the same list.
        env[TEX_PACKAGES_ENV] = " ".join(settings.tex_packages)
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


def _identity_token(environ: Mapping[str, str]) -> str | None:
    for name in _IDENTITY_TOKEN_VARS:
        value = environ.get(name)
        if not value:
            continue
        if not name.endswith("_FILE"):
            return value.strip()
        try:
            return Path(value).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return None
    return None


def identity_token_claims(environ: Mapping[str, str]) -> dict[str, Any]:
    """What the identity token of this container says about itself, read without verifying.

    Only used to describe the token in logs and error messages; nothing here is
    trusted or secret.  Returns ``{}`` when there is no token or it is not a JWT.
    Besides the claims a federation rule can match on, the result has
    ``lifetime_seconds`` (``exp - iat``) and ``single_use`` (the token carries a
    ``jti``, so it can be exchanged once).
    """
    parts = (_identity_token(environ) or "").split(".")
    if len(parts) != 3:
        return {}
    try:
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    claims: dict[str, Any] = {
        name: payload[name] for name in _TOKEN_CLAIMS if isinstance(payload.get(name), str)
    }
    issued, expires = payload.get("iat"), payload.get("exp")
    if isinstance(issued, int | float) and isinstance(expires, int | float):
        claims["lifetime_seconds"] = int(expires - issued)
    claims["single_use"] = "jti" in payload
    return claims


def federation_advice(environ: Mapping[str, str]) -> str:
    """What to compare on the Anthropic side when the identity token was not accepted.

    The token endpoint refuses with the same opaque answer whatever the reason,
    so this names the usual ones next to what this container's token carries.
    """
    claims = identity_token_claims(environ)
    described = ", ".join(f"{name}={claims[name]}" for name in _TOKEN_CLAIMS if name in claims)
    lifetime = claims.get("lifetime_seconds")
    checks = [
        "the issuer registered there is https://oidc.modal.com with audience oidc.modal.com",
        (
            "the issuer's maximum token lifetime is at least that of Modal's token ("
            + (f"{lifetime} s for this one" if lifetime is not None else "it could not be read")
            + "; the default maximum is one hour, and Modal documents tokens valid for 48 h)"
        ),
        (
            "the conditions of the federation rule match this container"
            + (f" ({described})" if described else "")
            + "; a rule that pins function_name must admit both write_paper and "
            "consolidate_memory, which authenticate separately"
        ),
    ]
    if claims.get("single_use"):
        checks.append(
            "the token can be exchanged only once (it carries a jti), so the access token the "
            f"rule issues should last as long as one attempt of write_paper ({SETTINGS.timeout}"
            f" s, {TIMEOUT_ENV}): a run that outlives it needs a token Modal has refreshed"
        )
    return (
        "This container authenticates with Workload Identity Federation and its identity token "
        "was not accepted. The exact reason is in the federation history of the Anthropic "
        "Console. Check that " + "; that ".join(checks) + "."
    )


def explain_credential_failure(error: str | None, environ: Mapping[str, str]) -> str | None:
    """Add :func:`federation_advice` to an error that reports an unusable federated credential."""
    keyless = all(environ.get(name) for name in _WIF_REQUIRED)
    if not error or not keyless or not _CREDENTIAL_FAILURE.search(error):
        return error
    return f"{error}\n{federation_advice(environ)}"


def announce_credentials(environ: Mapping[str, str]) -> str:
    """Check the Anthropic credential of this container and log which one it is.

    Shared by ``write_paper`` and ``consolidate_memory``: both call a model, each
    with the identity of its own Function.  Returns what
    :func:`check_credentials` returns.
    """
    credential = check_credentials(environ)
    if credential == "none":
        log.warning(
            "No Anthropic credential is configured in this container (no API key in the "
            "Modal Secret and no Workload Identity Federation ids); Claude models will fail."
        )
        return credential
    log.info("Anthropic credential in use: %s.", credential)
    claims = identity_token_claims(environ) if credential == "workload_identity" else {}
    if claims:
        log.info(
            "Identity token: %s.",
            ", ".join(f"{name}={value}" for name, value in claims.items()),
        )
    return credential


# -- jobs and paths -------------------------------------------------------------


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise JobError(
            f"run_id must be 1 to {RUN_ID_MAX} letters, digits, '.', '_' or '-' and start with "
            f"a letter or digit; got {run_id!r}"
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
    keep_provided_figures: bool = False,
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
        "keep_provided_figures": keep_provided_figures or None,
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


class VolumeLease:
    """Marks a directory of the volume as being written by one call.

    The volume has no file locking and the last writer of a file wins, so the
    lease is a small file that names its holder and the time of its last
    heartbeat.  A call that finds the heartbeat of another holder moving leaves
    the directory alone.  The heartbeat is rewritten by a thread and reaches
    other containers with the background commits of the volume; this thread
    never commits itself, because an explicit commit must come from the thread
    that does the work (see :class:`StageCheckpoint`).
    """

    def __init__(self, path: Path, owner: str, subject: str) -> None:
        self.path = path
        self.owner = owner
        self.subject = subject
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def acquire(
        cls,
        directory: Path,
        owner: str,
        *,
        subject: str,
        commit: Callable[[], None],
        reload: Callable[[], None] | None = None,
    ) -> VolumeLease:
        """Take the lease of ``directory`` or raise :class:`LeaseHeld`.

        ``owner`` identifies the call; a retry of the same call (after a crash
        or a preemption) presents the same one and takes its own lease over at
        once.  ``subject`` names the directory in messages.  The caller has
        reloaded the volume, so a lease another container committed is visible
        here.  With ``reload``, a fresh lease of another call is watched before
        it is believed (see :meth:`_wait_for_holder`), which can take as long
        as a lease needs to go stale.
        """
        lease = cls(directory / LEASE_NAME, owner, subject)
        lease._wait_for_holder(reload)
        directory.mkdir(parents=True, exist_ok=True)
        lease._write()
        commit_volume(commit)
        if reload is not None:
            # Two calls that start together both find the directory free.  The later commit
            # wins the file, so look again once both have had time to commit.
            time.sleep(_LEASE_SETTLE)
            _reload_volume(reload)
            lease._refuse_if_held()
            lease._write()
        lease._thread = threading.Thread(target=lease._beat, name="volume-lease", daemon=True)
        lease._thread.start()
        return lease

    def _read(self) -> dict[str, Any] | None:
        try:
            held = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(held, dict) or not isinstance(held.get("at"), int | float):
            return None
        return held

    def _other_holder(self) -> dict[str, Any] | None:
        """The lease another call holds on the directory; ``None`` when it is free to take.

        A lease whose heartbeat is stale was left by a container that is gone,
        and counts as free.
        """
        held = self._read()
        if held is None or held.get("owner") == self.owner:
            return None
        age = time.time() - held["at"]
        if age >= _LEASE_STALE:
            log.warning(
                "Taking over %s: the call that held it (%s) last reported %.0f s ago.",
                self.subject,
                held.get("owner"),
                age,
            )
            return None
        return held

    def _held(self, held: Mapping[str, Any]) -> LeaseHeld:
        age = max(time.time() - held["at"], 0.0)
        return LeaseHeld(
            f"{self.subject} is already being written by another call ({held.get('owner')}, "
            f"last heartbeat {age:.0f} s ago), so this call left it alone. Wait for "
            "that call or cancel it. If its container was stopped, its hold ends once the "
            f"heartbeat is {_LEASE_STALE:.0f} s old."
        )

    def _refuse_if_held(self) -> None:
        held = self._other_holder()
        if held is not None:
            raise self._held(held)

    def _wait_for_holder(self, reload: Callable[[], None] | None) -> None:
        """Return once the directory is free to take; raise :class:`LeaseHeld` if it is in use.

        A container that was stopped (its call cancelled, its App stopped with
        Ctrl-C, killed) could not give its lease up, and until that heartbeat is
        stale it looks like one of a call at work.  Launching the run again is
        how such a run is resumed, so a fresh lease is not believed at once: it
        is watched.  A heartbeat that moves, or a lease that changes hands, is a
        call at work and this one is turned away; a heartbeat that stays as it
        is goes stale and the lease is taken over.  Without ``reload`` nothing
        another container commits can be seen, and a fresh lease is believed.
        """
        first = held = self._other_holder()
        if held is None:
            return
        if reload is None:
            raise self._held(held)
        # The watch never lasts longer than a lease takes to go stale, whatever the time in
        # the lease says: it was stamped by the clock of another container.
        watch = min(max(held["at"] + _LEASE_STALE - time.time(), 0.0), _LEASE_STALE)
        log.info(
            "Another call (%s) holds the lease on %s; watching whether that call is still "
            "at work, for up to %.0f s.",
            held.get("owner"),
            self.subject,
            watch,
        )
        deadline = time.monotonic() + watch
        while held == first:
            if time.monotonic() >= deadline:
                log.warning(
                    "Taking over %s: the call that held it (%s) is gone, its heartbeat did "
                    "not move.",
                    self.subject,
                    held.get("owner"),
                )
                return
            time.sleep(_LEASE_POLL)
            _reload_volume(reload)
            held = self._other_holder()
        if held is not None:
            raise self._held(held)

    def _write(self) -> None:
        record = json.dumps({"owner": self.owner, "at": time.time()})
        # Replaced in one step: a commit of the volume never catches a half-written lease.
        partial = self.path.with_name(f"{self.path.name}.{secrets.token_hex(3)}.tmp")
        try:
            partial.write_text(record, encoding="utf-8")
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)

    def _beat(self) -> None:
        while not self._stop.wait(_LEASE_REFRESH):
            try:
                self._write()
            except OSError as exc:  # for example while the volume is being reloaded
                log.warning("Could not refresh the lease of %s: %s", self.subject, exc)

    def release(self) -> None:
        """Stop the heartbeat and remove the lease; the next commit makes that visible."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        held = self._read()
        if held is None or held.get("owner") == self.owner:
            with contextlib.suppress(OSError):
                self.path.unlink(missing_ok=True)

    def __enter__(self) -> VolumeLease:
        return self

    def __exit__(self, kind: type[BaseException] | None, *exc: object) -> None:
        # Released when the work ended, well or badly.  Not when it was interrupted: after a
        # preemption the next attempt of the call may already hold this lease (it has the
        # same owner, and this container cannot see its file), and removing the file here
        # would remove it for that attempt too.  A lease left behind goes stale by itself.
        if kind is None or issubclass(kind, Exception):
            self.release()
        else:
            self._stop.set()


def _call_owner(owner: str | None) -> str:
    """The name a call holds leases under; calls made outside Modal get one of their own."""
    return owner or f"unnamed-{secrets.token_hex(6)}"


def run_job(
    job: Mapping[str, Any],
    *,
    commit: Callable[[], None],
    reload: Callable[[], None] | None = None,
    consolidate: Callable[[str], Any] | None = None,
    owner: str | None = None,
    mount: Path = Path(VOLUME_MOUNT),
) -> dict[str, Any]:
    """Run one paper inside the container and return its summary.

    Problems that a retry cannot fix (a malformed job, missing inputs, unusable
    credentials, a run id that another call is writing) are returned as a failed
    summary.  Anything else propagates so that Modal retries the call, which
    resumes the run.  The summary is returned only after a final commit, so the
    files it lists can be downloaded at once.

    ``consolidate(model)`` is ``consolidate_memory``, the one writer of the lab
    wiki: it creates the wiki before the first run on a volume and, after a
    successful run, folds the lessons the run left there into pages.  ``owner``
    identifies this call for the lease on the run directory (the Modal function
    call id, which a retry of the call keeps).
    """
    started = time.monotonic()
    run_id = job.get("run_id") if isinstance(job, Mapping) else None
    # Inputs may have been uploaded after this container mounted the volume.
    _reload_volume(reload)
    lessons_model: str | None = None
    try:
        spec = job_spec(job, mount)
        announce_credentials(os.environ)
        if not spec.config.template_dir.is_dir():
            raise JobError(f"LaTeX template not found in the container: {spec.config.template_dir}")
        # In local mode this raises ComputeError, before any model call, if the image lacks
        # TeX Live.
        backend = ModalBackend(latex=latex_mode(os.environ))
        _ensure_lab_wiki(spec, consolidate, reload)
        # Given up before the commits below, so that the run is free as soon as what it
        # wrote is visible to other containers.
        with VolumeLease.acquire(
            spec.config.base_dir,
            _call_owner(owner),
            subject=f"run {spec.run_id}",
            commit=commit,
            reload=reload,
        ):
            summary = _run_pipeline(spec, backend, commit, mount, started)
        summary["error"] = explain_credential_failure(summary["error"], os.environ)
        memory = spec.config.memory
        # Lessons reach the lab wiki in the last stage, and only a model can fold them in.
        if summary["success"] and memory.resolved_lab_dir() is not None and memory.dream != "off":
            lessons_model = spec.config.models.reflection
    except (FileNotFoundError, ValueError, LLMConfigError, ComputeError) as exc:
        log.error("Run %s cannot proceed: %s", run_id, exc)
        error = explain_credential_failure(str(exc), os.environ)
        summary = failure_summary(run_id, error or str(exc), time.monotonic() - started)
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


def _ensure_lab_wiki(
    spec: JobSpec, consolidate: Callable[[str], Any] | None, reload: Callable[[], None] | None
) -> None:
    """Have the lab wiki created by its one writer before this run opens it.

    Opening a wiki that does not exist yet writes its empty index.  Done from
    this container, that file could later replace an index the consolidation
    has written in the meantime, because the last commit of a file wins.
    """
    lab_dir = spec.config.memory.resolved_lab_dir()
    if consolidate is None or lab_dir is None or (lab_dir / INDEX_FILE).is_file():
        return
    log.info("The lab memory does not exist on this volume yet; having it created.")
    _share_lessons(consolidate, spec.config.models.reflection)
    _reload_volume(reload)


def _run_pipeline(
    spec: JobSpec, backend: ModalBackend, commit: Callable[[], None], mount: Path, started: float
) -> dict[str, Any]:
    from paper_orchestra.pipeline import prepare_run_dir, run_paper

    config = spec.config
    os.environ[LAB_MEMORY_ENV] = str(mount / LAB_MEMORY_ROOT)
    # This container holds the credentials.  When LaTeX is compiled here rather than in a
    # Sandbox, the local backend already gives TeX an environment of its own; this covers a
    # TeX process started any other way: paranoid input keeps it from reading
    # /proc/self/environ or the identity token file into the PDF.
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
    owner: str | None = None,
    mount: Path = Path(VOLUME_MOUNT),
) -> dict[str, Any]:
    """Fold the notes that runs left in the lab wiki into its pages; return the report.

    ``model`` holds the consolidation session, which sees the lab wiki and
    nothing of any paper.  Only one of these may run at a time: it is the single
    writer of the lab's pages and of its index.  Within one App the Function's
    container limit sees to that; a lease on the wiki keeps the consolidations
    of other Apps that use the same volume apart, and one that finds the wiki
    taken returns a report with an ``error`` and leaves the notes for the next.
    Credentials are checked as for ``write_paper``; this Function presents an
    identity token of its own.
    """
    _reload_volume(reload)
    announce_credentials(os.environ)
    lab_dir = mount / LAB_MEMORY_ROOT
    try:
        lease = VolumeLease.acquire(
            lab_dir, _call_owner(owner), subject="the lab memory", commit=commit, reload=reload
        )
    except LeaseHeld as exc:
        return {"scope": "lab", "error": str(exc), "notes_archived": 0, "pages_changed": []}
    with lease, tempfile.TemporaryDirectory(prefix="paper-orchestra-lab-") as scratch:
        # A harness is built around a project wiki; no paper is involved here, so an empty
        # one in a scratch directory stands in.
        config = PipelineConfig(
            base_dir=Path(scratch),
            template_dir=Path(scratch),
            models=ModelConfig(reflection=model),
            memory=MemoryConfig(lab_dir=lab_dir, dream="end"),
        )
        report = ContextHarness.create(config).consolidate_lab_memory()
    commit_volume(commit)
    if report is None:
        return {}
    described = _jsonable(report.to_dict())
    described["error"] = explain_credential_failure(described.get("error"), os.environ)
    return described


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


def visible_files(root: Path) -> Iterator[Path]:
    """The files under ``root`` that are uploaded: everything that is not hidden."""
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(files):
            if not name.startswith("."):
                yield Path(current) / name


def upload_inputs(
    volume: Any, run_id: str, raw_materials_dir: Path, template_dir: Path
) -> tuple[str, str]:
    """Upload a run's inputs; return the volume paths of the raw materials and the template.

    What an earlier launch of the run id uploaded is replaced, not added to: a
    figure that has since been removed from the raw materials must not reach
    the paper.  What is no longer among the inputs is removed after the upload
    and nothing else is, so the inputs are never missing from the volume: a run
    may be reading them (a second launch of a run in flight is turned away by
    the run itself, after this upload), and an upload that fails leaves the
    inputs of the earlier launch in place.
    """
    root = f"{INPUTS_ROOT}/{run_id}"
    targets = (
        (raw_materials_dir, f"{root}/raw_materials"),
        (template_dir, f"{root}/template"),
    )
    uploaded: set[str] = set()
    # force: launching a run id again, to resume it, uploads the same files once more.
    with volume.batch_upload(force=True) as batch:
        for source, target in targets:
            for path in visible_files(source):
                remote = f"{target}/{path.relative_to(source).as_posix()}"
                batch.put_file(path, f"/{remote}")
                uploaded.add(remote)
    _remove_stale_inputs(volume, root, uploaded)
    return targets[0][1], targets[1][1]


def _remove_stale_inputs(volume: Any, root: str, uploaded: set[str]) -> None:
    """Remove what is under ``root`` on the volume and was not part of this upload."""
    # The files just uploaded and the directories they are in.
    kept = uploaded | {
        parent.as_posix() for name in uploaded for parent in PurePosixPath(name).parents
    }
    removed: list[str] = []
    # Sorted, so that a directory comes before what it holds and goes in one step.
    for entry in sorted(volume.listdir(root, recursive=True), key=lambda entry: entry.path):
        name = entry.path.strip("/")
        # Entries are named from the root of the volume.  One that is not plainly inside this
        # run's inputs is not removed, whatever the listing meant by it.
        inside = name.startswith(f"{root}/") and ".." not in PurePosixPath(name).parts
        if not inside or name in kept or any(name.startswith(f"{gone}/") for gone in removed):
            continue
        with contextlib.suppress(FileNotFoundError):
            volume.remove_file(name, recursive=entry.type == modal.volume.FileEntryType.DIRECTORY)
        removed.append(name)
    if removed:
        log.info("Removed from the volume, as no longer among the inputs: %s", ", ".join(removed))


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
        "Started run %s (function call %s). Once it has stopped, launching this run id again "
        "resumes it; while it is running a second launch is refused.",
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


def project_root(package_file: Path) -> Path:
    """The source checkout that ``package_file`` (this module) belongs to.

    The image installs the third-party dependencies from the checkout's
    ``pyproject.toml`` and ``uv.lock``.  An installed wheel carries neither, and
    Modal would only say so in the middle of an image build, in terms of a path
    inside the virtual environment.
    """
    root = package_file.resolve().parents[2]
    missing = [name for name in ("pyproject.toml", "uv.lock") if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"The Modal app is run from a source checkout of paper-orchestra: its image is "
            f"built from the project's pyproject.toml and uv.lock, and {' and '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not in {root}. This copy of the package "
            "looks like an installed wheel. Clone the repository, run `uv sync --extra modal` "
            "in it, and use `uv run paper-orchestra modal ...` from there."
        )
    return root


def _pipeline_image(settings: Settings) -> modal.Image:
    # In a checkout this file is src/paper_orchestra/modal_app.py.  The container imports
    # this module too; there the image is never built and the path is not looked at.
    root = project_root(Path(__file__)) if modal.is_local() else Path(__file__).parents[2]
    templates = os.environ.get(TEMPLATES_ENV) or str(BUNDLED_TEMPLATES_DIR)
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        # For PAPER_ORCHESTRA_MODAL_LATEX=local.  It is also the first layer of the image of
        # the LaTeX Sandbox, so building this image leaves little of that one to build.
        .run_commands(texlive_install_command(settings.tex_packages))
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
image = _pipeline_image(SETTINGS)
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
    try:
        return run_job(
            job,
            commit=volume.commit,
            reload=volume.reload,
            consolidate=consolidate_memory.remote,
            # The same for every attempt of this call, so a retry takes over its own lease.
            owner=modal.current_function_call_id(),
        )
    except (modal.exception.Error, TimeoutError) as exc:
        # Whoever waits for this call must be able to tell "Modal could not be asked about
        # the call" and "the call is still running" (FunctionCall.get raises the builtin
        # TimeoutError for that) from "the call failed"; either error raised in here is the
        # third, and passed on as it is the caller would read it as one of the first two.
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


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

    ``write_paper`` calls this after a successful run, and before a run on a
    volume that has no lab wiki yet.  Returns the consolidation report
    (``notes_seen``, ``notes_archived``, ``pages_changed``, ``error``, ...).
    """
    setup_logging(rich=False)
    return consolidate_lab(
        model,
        commit=volume.commit,
        reload=volume.reload,
        owner=modal.current_function_call_id(),
    )


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

    For programmatic use with batches that finish within a day.  This call
    cannot outlive Modal's limit of 24 hours for one Function call: at that
    point it is stopped, the papers still being written are cancelled, and no
    summary is returned for any paper (the finished ones stay on the volume
    under ``runs/<run_id>``).  ``paper-orchestra modal batch`` does not use it:
    it calls ``write_paper`` once per paper and collects each paper as it
    finishes.
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
    keep_provided_figures: bool = False,
    idea_filename: str = "idea_sparse.md",
    experimental_log_filename: str = "experimental_log.md",
    research_cutoff: str | None = None,
    model: str | None = None,
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
    name (``cvpr2025``) or a directory.  ``model`` is the model of the writer,
    reflection, plotting and fast roles unless the option of a role names
    another; without ``model`` the fast role follows the writer, as it does for
    ``write``.  Pass the ``--run-id`` of an interrupted run to resume it.
    """
    setup_logging()
    # The Sandbox images are built here, where the build log is visible, rather than by the
    # first compile or the first figure of the run.
    if SETTINGS.latex == "sandbox":
        latex_image(SETTINGS.tex_packages).build(app)
    if plotting:
        python_image().build(app)
    summary = launch(
        volume=volume,
        function=write_paper,
        raw_materials_dir=raw_materials_dir,
        template=template,
        output_dir=output_dir,
        run_id=run_id,
        use_plotting=plotting,
        keep_provided_figures=keep_provided_figures,
        idea_filename=idea_filename,
        experimental_log_filename=experimental_log_filename,
        research_cutoff=research_cutoff,
        writer_model=writer_model or model,
        reflection_model=reflection_model or model,
        plotting_model=plotting_model or model,
        # Without --model, discovery stays on the writer's provider (see cli._role_models).
        fast_model=fast_model or model or writer_model,
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
