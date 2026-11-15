"""Run configuration.

Plain dataclasses, no I/O at import time.  ``load_env`` is called explicitly by
entry points (CLI, Modal app, frontend) and never by library modules.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from paper_orchestra.llm.base import Effort

DEFAULT_MODEL = "claude-fable-5-2"

ComputeName = Literal["local", "modal"]
DreamMode = Literal["off", "end", "stages"]

#: Environment variable overriding where the cross-run "lab" memory lives.
LAB_MEMORY_ENV = "PAPER_ORCHESTRA_MEMORY_DIR"
#: Optional checkout of PaperBanana used for few-shot figure examples.
PAPER_BANANA_ENV = "PAPER_BANANA_DIR"
TEMPLATES_ENV = "PAPER_ORCHESTRA_TEMPLATES"


def load_env(extra: str | Path | None = None) -> None:
    """Load ``.env`` from the working directory (and ``extra``) without overriding."""
    from dotenv import find_dotenv, load_dotenv

    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
    if extra and Path(extra).is_file():
        load_dotenv(extra, override=False)


def default_lab_memory_dir() -> Path:
    override = os.environ.get(LAB_MEMORY_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".paper_orchestra" / "memory"


def default_research_cutoff() -> str:
    return datetime.now().strftime("%Y-%m")


@dataclass
class ModelConfig:
    """Which model does what.

    Everything defaults to Claude Fable 5.2.  ``fast`` handles the high-volume
    calls (candidate discovery, title extraction) and runs at ``fast_effort``;
    point it at a smaller model to trade quality for cost.  ``image`` is only
    needed for image-generated diagrams (for example a Gemini image model);
    when unset, diagrams are rendered from model-written plotting code.
    """

    writer: str = DEFAULT_MODEL
    reflection: str = DEFAULT_MODEL
    plotting: str = DEFAULT_MODEL
    fast: str = DEFAULT_MODEL
    image: str | None = None
    effort: Effort = "high"
    fast_effort: Effort = "low"


@dataclass
class MemoryConfig:
    """The wiki memory harness.

    ``project_dir`` holds what the run learns about *this paper* (default
    ``<base_dir>/memory``).  ``lab_dir`` persists across runs and only receives
    process lessons (LaTeX pitfalls, venue rules, what revisions helped) plus
    public bibliographic metadata, never paper content, so the pipeline's
    knowledge-isolation requirement still holds.  Set ``lab=False`` for strict
    benchmark runs.
    """

    enabled: bool = True
    project_dir: Path | None = None
    lab: bool = True
    lab_dir: Path | None = None
    dream: DreamMode = "stages"

    def resolved_lab_dir(self) -> Path | None:
        if not (self.enabled and self.lab):
            return None
        return self.lab_dir or default_lab_memory_dir()


@dataclass
class PipelineConfig:
    base_dir: Path
    template_dir: Path
    idea_filename: str = "idea_sparse.md"
    experimental_log_filename: str = "experimental_log.md"
    models: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    research_cutoff: str = field(default_factory=default_research_cutoff)
    use_plotting: bool = False
    plotting_max_critic_rounds: int = 3
    max_reflections: int = 3
    max_formatting_rounds: int = 1
    review_ensemble: int = 3
    compute: ComputeName = "local"
    #: Skip stages whose outputs already exist from an earlier, interrupted run.
    resume: bool = True
    #: Attempts per stage before the run is declared failed.
    max_stage_attempts: int = 2
    #: Compile-and-repair rounds granted to a writing agent.
    max_repair_rounds: int = 3
    literature_workers: int = 6
    figure_workers: int = 4
    latex_timeout: float = 240.0

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.template_dir = Path(self.template_dir)

    @property
    def project_memory_dir(self) -> Path:
        return self.memory.project_dir or (self.base_dir / "memory")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return convert(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        data = dict(data)
        models = ModelConfig(**data.pop("models", {}))
        memory_raw = dict(data.pop("memory", {}))
        for key in ("project_dir", "lab_dir"):
            if memory_raw.get(key):
                memory_raw[key] = Path(memory_raw[key])
        memory = MemoryConfig(**memory_raw)
        return cls(models=models, memory=memory, **data)


def resolve_template_dir(template: str | Path) -> Path:
    """Resolve a template name (``cvpr2025``) or path to a template directory.

    Search order: the path itself, ``$PAPER_ORCHESTRA_TEMPLATES/<name>``,
    ``./templates/<name>``, then the ``templates/`` directory next to an
    editable checkout of this package.
    """
    candidate = Path(template).expanduser()
    if candidate.is_dir():
        return candidate
    roots: list[Path] = []
    if os.environ.get(TEMPLATES_ENV):
        roots.append(Path(os.environ[TEMPLATES_ENV]).expanduser())
    roots.append(Path.cwd() / "templates")
    roots.append(Path(__file__).resolve().parents[2] / "templates")
    for root in roots:
        if (root / str(template)).is_dir():
            return root / str(template)
    searched = ", ".join(str(r) for r in roots)
    raise FileNotFoundError(f"LaTeX template '{template}' not found (searched: {searched})")
