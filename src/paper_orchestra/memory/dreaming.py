"""Dreaming: consolidating memory between sessions.

Task sessions only jot notes; they are busy with the task.  A dream is a
separate session with one job: read the notes and transcripts of recent
sessions, check them, and fold what holds up into wiki pages that later
sessions can rely on.  Around the model's work, deterministic maintenance
keeps the wiki healthy: broken links are repointed, the index is regenerated
within its budget, and consolidated notes are archived.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from paper_orchestra.llm.base import LLMError
from paper_orchestra.memory import prompts
from paper_orchestra.memory.tools import virtual_path
from paper_orchestra.memory.wiki import LintReport, PageInfo, Wiki

if TYPE_CHECKING:
    from paper_orchestra.memory.harness import ContextHarness

log = logging.getLogger(__name__)

LOCK_FILE = ".dream.lock"
#: A lock older than this belongs to a session that died.
STALE_LOCK_SECONDS = 45 * 60
#: Notes listed in full in the dream prompt; the rest are named for the model to open.
MAX_INLINE_NOTES = 40
MAX_NOTE_PREVIEW_CHARS = 600

END_OF_RUN_FOCUS = (
    "The run is over. Consolidate everything it learned that would help if this paper were "
    "revised or the run resumed: which revisions the reviewers rewarded or punished and why, "
    "what broke during compilation and what fixed it, and decisions that were made along the "
    "way with their reasons."
)

LAB_FOCUS = (
    "Promote this run's process lessons into the lab memory so the next paper starts with "
    "them. Look at the project's lessons, reviews and notes about failures. Generalise each "
    "lesson away from this paper before writing it down, and merge it with what the lab "
    "memory already says."
)


@dataclass
class DreamReport:
    dream_id: str
    scope: str
    stage: str
    llm_used: bool = False
    notes_seen: int = 0
    notes_archived: int = 0
    pages_changed: list[str] = field(default_factory=list)
    links_repaired: int = 0
    lint: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _DreamLock:
    """Exclusive lock file so parallel runs sharing a wiki consolidate one at a time."""

    def __init__(self, root: Path, *, wait_seconds: float = 120.0, poll: float = 1.0) -> None:
        self.path = root / LOCK_FILE
        self.wait_seconds = wait_seconds
        self.poll = poll
        self.acquired = False

    def __enter__(self) -> _DreamLock:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                with contextlib.suppress(OSError):
                    if time.time() - self.path.stat().st_mtime > STALE_LOCK_SECONDS:
                        self.path.unlink(missing_ok=True)
                        continue
                if time.monotonic() >= deadline:
                    return self
                time.sleep(self.poll)
                continue
            with os.fdopen(fd, "w") as handle:
                handle.write(f"{os.getpid()}\n")
            self.acquired = True
            return self

    def __exit__(self, *exc: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def _note_listing(wiki: Wiki, notes: list[PageInfo]) -> str:
    lines: list[str] = []
    for info in notes[:MAX_INLINE_NOTES]:
        page = wiki.read(info.path)
        kind = page.kind or "observation"
        stage = page.meta.get("stage") or "-"
        preview = " ".join(page.body.split())[:MAX_NOTE_PREVIEW_CHARS]
        lines.append(f"- {virtual_path(info.path)} ({kind}, stage {stage}): {preview}")
    if len(notes) > MAX_INLINE_NOTES:
        rest = len(notes) - MAX_INLINE_NOTES
        lines.append(f"- ... and {rest} more under {virtual_path('notes')}")
    return "\n".join(lines)


def _lint_listing(report: LintReport) -> str:
    lines: list[str] = []
    for page, target in report.broken_links[:30]:
        lines.append(f"- {virtual_path(page)} links to [[{target}]], which does not exist")
    for page, size in report.oversize[:10]:
        lines.append(f"- {virtual_path(page)} is {size} bytes; consider splitting it")
    for page in report.missing_summary[:30]:
        lines.append(f"- {virtual_path(page)} has no `summary` in its front matter")
    if report.index_over_budget:
        lines.append(
            f"- MEMORY.md is over budget ({report.index_lines} lines, {report.index_bytes} "
            "bytes): shorten the standing notes"
        )
    return "\n".join(lines)


def _dream_prompt(
    *,
    scope: str,
    focus: str,
    wiki: Wiki,
    notes: list[PageInfo],
    lint: LintReport,
) -> str:
    stats = wiki.stats()
    parts = [f"Focus for this consolidation: {focus}", ""]
    target = "lab memory" if scope == "lab" else "project memory"
    parts.append(f"The {target} currently has {stats['pages']} wiki pages.")
    if scope == "project":
        if notes:
            parts += ["", f"Unconsolidated notes ({len(notes)}):", _note_listing(wiki, notes)]
        else:
            parts += ["", "There are no unconsolidated notes."]
    issues = _lint_listing(lint)
    if issues:
        parts += ["", "Wiki health problems found by the linter:", issues]
    return "\n".join(parts)


def _maintain(wiki: Wiki) -> tuple[int, LintReport]:
    repaired = len(wiki.repair_links())
    wiki.rebuild_index()
    return repaired, wiki.lint()


def dream(
    harness: ContextHarness,
    *,
    stage: str,
    focus: str,
    model: str | None,
    scope: Literal["project", "lab"] = "project",
    use_model: bool = True,
    build: bool = False,
    lock_wait: float = 120.0,
) -> DreamReport:
    """Consolidate one wiki.  Never raises: failures are logged and reported.

    Without ``build`` the model is only involved when there is something to
    consolidate or repair.  ``build`` runs it regardless, for a ``focus`` that
    asks for new pages (for example a synthesis across the collected papers).
    ``lock_wait`` is how long to wait for another process that is consolidating
    the same wiki before giving up (the notes stay pending for the next dream).
    """
    from paper_orchestra.memory.harness import MemoryPolicy

    wiki = harness.lab if scope == "lab" else harness.project
    report = DreamReport(
        dream_id=f"{time.strftime('%Y%m%dT%H%M%S')}-{stage}-{scope}-{secrets.token_hex(2)}",
        scope=scope,
        stage=stage,
    )
    if wiki is None or harness.project is None:
        return report

    with _DreamLock(wiki.root, wait_seconds=lock_wait) as lock:
        if not lock.acquired:
            report.error = "another consolidation of this wiki is in progress"
            log.warning("dream %s skipped: %s", report.dream_id, report.error)
            return report
        try:
            _run(
                harness,
                wiki,
                report,
                focus=focus,
                model=model,
                scope=scope,
                use_model=use_model,
                build=build,
                policy=MemoryPolicy.dream(lab=scope == "lab"),
            )
        except Exception as exc:  # a dream must never fail the run
            report.error = f"{type(exc).__name__}: {exc}"
            log.exception("dream %s failed", report.dream_id)
    fields = {key: value for key, value in report.to_dict().items() if key != "stage"}
    harness.record("dream", stage=stage, agent="dreamer", **fields)
    return report


def _run(
    harness: ContextHarness,
    wiki: Wiki,
    report: DreamReport,
    *,
    focus: str,
    model: str | None,
    scope: str,
    use_model: bool,
    build: bool,
    policy: Any,
) -> None:
    assert harness.project is not None
    repaired, lint = _maintain(wiki)
    report.links_repaired = repaired
    notes = harness.project.pending_notes() if scope == "project" else []
    report.notes_seen = len(notes)

    model_ok = False
    wants_model = use_model and model is not None
    if wants_model:
        if scope == "lab":
            # Promotion needs evidence: lessons, reviews or notes from this project.
            has_work = any(
                harness.project.list(prefix) for prefix in ("wiki/lessons", "wiki/reviews", "notes")
            )
        else:
            has_work = build or bool(notes) or bool(lint.broken_links) or lint.index_over_budget
        client = harness.client_for(model)
        if not client.capabilities.memory_tool:
            log.info("dream: model %s has no memory tool; maintenance only", model)
        elif has_work:
            conversation = harness.open(
                agent="dreamer",
                stage=report.stage,
                model=model,
                system=prompts.DREAM_LAB_SYSTEM if scope == "lab" else prompts.DREAM_SYSTEM,
                memory=policy,
                effort="medium",
                label=f"dream.{scope}",
            )
            try:
                response = conversation.send(
                    _dream_prompt(scope=scope, focus=focus, wiki=wiki, notes=notes, lint=lint),
                    require_parsed=False,
                )
            except LLMError as exc:
                report.error = f"{type(exc).__name__}: {exc}"
                log.warning("dream %s: model session failed: %s", report.dream_id, exc)
            else:
                model_ok = True
                report.llm_used = True
                report.summary = response.text.strip()
                report.usage = response.usage.to_dict()
            wanted = "lab" if scope == "lab" else "project"
            report.pages_changed = sorted(
                {rel for write_scope, rel in conversation.writes if write_scope == wanted}
            )

    # Notes are archived only once a model has actually consolidated them; without one they
    # stay pending for the next dream.
    if scope == "project" and model_ok and notes:
        report.notes_archived = len(
            harness.project.archive_notes([n.path for n in notes], report.dream_id)
        )
    repaired, lint = _maintain(wiki)
    report.links_repaired += repaired
    report.lint = lint.to_dict()
