"""Dreaming: consolidating memory between sessions.

Task sessions only jot notes; they are busy with the task.  A dream is a
separate session with one job: read the notes and transcripts of recent
sessions, check them, and fold what holds up into wiki pages that later
sessions can rely on.  Around the model's work, deterministic maintenance
keeps the wiki healthy: broken links are repointed, the index is regenerated
within its budget, numbers that no record supports are flagged, and
consolidated notes are archived.

Lessons cross from one paper's memory into the shared lab memory in two
steps, so the session that maintains the lab never sees a paper: pipeline
code copies each lesson marked portable into the lab as a note, after a
mechanical leak scan, and a lab-only session folds those notes into pages.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from paper_orchestra.llm.base import LLMError
from paper_orchestra.memory import prompts
from paper_orchestra.memory.tools import virtual_path
from paper_orchestra.memory.wiki import INDEX_FILE, LintReport, PageInfo, Wiki, tokenize

if TYPE_CHECKING:
    from paper_orchestra.memory.harness import ContextHarness

log = logging.getLogger(__name__)

LOCK_FILE = ".dream.lock"
PROMOTED_LEDGER = ".promoted.json"
#: A lock not refreshed for this long belongs to a session that died.
STALE_LOCK_SECONDS = 15 * 60
LOCK_REFRESH_SECONDS = 60
#: Notes listed with a preview in the dream prompt; the rest are named for the model to open.
MAX_INLINE_NOTES = 40
MAX_NOTE_PREVIEW_CHARS = 600
#: Pages that hold the paper's own content.  Nothing from them may reach the lab memory.
CONTENT_SOURCE_PREFIXES = ("wiki/sources/idea", "wiki/sources/experimental_log")
SHINGLE_TOKENS = 6

END_OF_RUN_FOCUS = (
    "The run is over. Consolidate everything it learned that would help if this paper were "
    "revised or the run resumed: which revisions the reviewers rewarded or punished and why, "
    "what broke during compilation and what fixed it, and decisions that were made along the "
    "way with their reasons. Mark each lesson that would help with a different paper as "
    "portable and state it without reference to this paper."
)

LAB_FOCUS = (
    "Fold the pending lesson notes into the lab memory so the next paper starts with them. "
    "Merge each one with what the lab memory already says about its topic."
)

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w]|\.\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_HOME_PATH = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)[\w.-]+")
_CREDENTIAL = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}|(?i:\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*\S{6,})"
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
    unverified_numbers: list[list[str]] = field(default_factory=list)
    summary: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


class _DreamLock:
    """Exclusive lock file so parallel runs sharing a wiki consolidate one at a time.

    The holder refreshes the file's timestamp while it works, so a long dream is
    not mistaken for a dead one.  This is mutual exclusion on a local file
    system; storage without atomic exclusive create (some network volumes)
    needs a single consolidating process instead.
    """

    def __init__(self, root: Path, *, wait_seconds: float = 120.0, poll: float = 1.0) -> None:
        self.path = root / LOCK_FILE
        self.wait_seconds = wait_seconds
        self.poll = poll
        self.acquired = False
        self._stop = threading.Event()
        self._refresher: threading.Thread | None = None

    def _refresh(self) -> None:
        while not self._stop.wait(LOCK_REFRESH_SECONDS):
            with contextlib.suppress(OSError):
                os.utime(self.path, None)

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
            self._refresher = threading.Thread(target=self._refresh, daemon=True)
            self._refresher.start()
            return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self.acquired:
            self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def _significant(number: str) -> bool:
    """Numbers specific enough to identify a result: decimals with three or more digits,
    or integers of four or more digits that are not calendar years."""
    digits = number.replace(".", "")
    if "." in number:
        return len(digits) >= 3
    return len(digits) >= 4 and not re.fullmatch(r"(19|20)\d\d", number)


def _contains_number(text: str, number: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)", text) is not None


def unverified_numbers(wiki: Wiki, *, limit: int = 40) -> list[tuple[str, str]]:
    """``(page, number)`` for numbers in model-written pages that no record contains.

    The read-only sections of a wiki are its records (inputs, bibliography,
    reviews).  A specific number that appears in a consolidated page or in the
    standing notes but in none of those records was either mistyped or made up,
    and every later session would trust it.
    """
    records: list[str] = []
    written: list[tuple[str, str]] = []
    for info in wiki.list():
        text = wiki.read_text(info.path)
        if wiki.is_read_only(info.path):
            records.append(text)
        elif not wiki.read(info.path).meta.get("auto"):
            written.append((info.path, wiki.read(info.path).body))
    written.append((INDEX_FILE, wiki.standing_notes()))
    evidence = "\n".join(records)
    findings: list[tuple[str, str]] = []
    for path, body in written:
        seen: set[str] = set()
        for number in _NUMBER.findall(body):
            if number in seen or not _significant(number):
                continue
            seen.add(number)
            if not _contains_number(evidence, number):
                findings.append((path, number))
                if len(findings) >= limit:
                    return findings
    return findings


def _shingles(text: str) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    return {tuple(tokens[i : i + SHINGLE_TOKENS]) for i in range(len(tokens) - SHINGLE_TOKENS + 1)}


def leak_reason(text: str, sources: Sequence[str]) -> str | None:
    """Why ``text`` must not enter the shared memory, or ``None`` if it may.

    Mechanical, so the rule does not depend on a model following instructions:
    no result numbers or runs of wording from this paper's own materials, and
    no e-mail addresses, home-directory paths or credentials.
    """
    match = _EMAIL.search(text)
    if match:
        return f"it contains an e-mail address ({match.group(0)})"
    match = _HOME_PATH.search(text)
    if match:
        return f"it contains a personal file-system path ({match.group(0)})"
    if _CREDENTIAL.search(text):
        return "it contains something that looks like a credential"
    if not sources:
        return None
    corpus = "\n".join(sources)
    for number in dict.fromkeys(_NUMBER.findall(text)):
        if _significant(number) and _contains_number(corpus, number):
            return f"it repeats a number from this paper's materials ({number})"
    shared = _shingles(text) & _shingles(corpus)
    if shared:
        return "it repeats wording from this paper's materials (\"" + " ".join(min(shared)) + '")'
    return None


def content_sources(project: Wiki) -> list[str]:
    """Text of the pages that hold the paper's own content (idea and experimental log)."""
    return [
        project.read_text(info.path)
        for info in project.list("wiki/sources")
        if info.path.startswith(CONTENT_SOURCE_PREFIXES)
    ]


def promote_lessons(harness: ContextHarness) -> list[str]:
    """Copy this run's portable lessons into the lab memory as notes.

    A lesson is portable when a consolidation session marked it
    ``portable: true``.  Each one is leak-scanned against the paper's own
    materials and, if clean, added to the lab as a uniquely named note: runs
    only ever *add* files to the shared memory, so parallel runs cannot
    overwrite each other.  Returns the lab note paths.
    """
    project, lab = harness.project, harness.lab
    if project is None or lab is None:
        return []
    ledger_path = project.root / PROMOTED_LEDGER
    try:
        promoted: set[str] = set(json.loads(ledger_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        promoted = set()
    sources = content_sources(project)
    added: list[str] = []
    for info in project.list("wiki/lessons"):
        page = project.read(info.path)
        if page.meta.get("portable") is not True:
            continue
        text = f"# {page.title}\n\n{page.body.strip()}\n"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in promoted:
            continue
        reason = leak_reason(text, sources)
        if reason:
            log.warning("lesson %s not shared with the lab memory: %s", info.path, reason)
            harness.record("promotion_blocked", stage="finalize", path=info.path, reason=reason)
            continue
        note = lab.add_note(
            text, stage="promotion", agent="pipeline", kind="lesson", title=page.title
        )
        promoted.add(digest)
        added.append(note)
        harness.record("promotion", stage="finalize", path=info.path, lab_note=note)
    if added:
        ledger_path.write_text(json.dumps(sorted(promoted)), encoding="utf-8")
    return added


# ---------------------------------------------------------------------------
# The dream itself
# ---------------------------------------------------------------------------


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


def _health_listing(report: LintReport, numbers: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for page, target in report.broken_links[:30]:
        lines.append(f"- {virtual_path(page)} links to [[{target}]], which does not exist")
    for page, size in report.oversize[:10]:
        lines.append(f"- {virtual_path(page)} is {size} bytes; consider splitting it")
    if report.standing_over_budget:
        lines.append(
            f"- The standing notes in MEMORY.md have {report.standing_lines} lines; keep them "
            "to the few rules every session needs and move the rest into pages"
        )
    if report.index_over_budget:
        lines.append(
            f"- MEMORY.md is over budget ({report.index_lines} lines, {report.index_bytes} bytes)"
        )
    for page, number in numbers[:30]:
        lines.append(
            f"- {virtual_path(page)} states {number}, which appears in no source, review or "
            "other record: check it and correct or remove it"
        )
    return "\n".join(lines)


def _dream_prompt(
    *,
    scope: str,
    focus: str,
    wiki: Wiki,
    notes: list[PageInfo],
    lint: LintReport,
    numbers: list[tuple[str, str]],
) -> str:
    stats = wiki.stats()
    target = "lab memory" if scope == "lab" else "project memory"
    parts = [
        f"Focus for this consolidation: {focus}",
        "",
        f"The {target} currently has {stats['pages']} wiki pages.",
    ]
    if notes:
        parts += ["", f"Unconsolidated notes ({len(notes)}):", _note_listing(wiki, notes)]
    else:
        parts += ["", "There are no unconsolidated notes."]
    issues = _health_listing(lint, numbers)
    if issues:
        parts += ["", "Problems found by the automatic checks:", issues]
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
    the same wiki before giving up; the notes then stay pending for the next
    dream, so nothing is lost.
    """
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
        else:
            try:
                _run(
                    harness,
                    wiki,
                    report,
                    focus=focus,
                    model=model,
                    use_model=use_model,
                    build=build,
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
    use_model: bool,
    build: bool,
) -> None:
    from paper_orchestra.memory.harness import MemoryPolicy

    scope = report.scope
    repaired, lint = _maintain(wiki)
    report.links_repaired = repaired
    notes = wiki.pending_notes()
    report.notes_seen = len(notes)
    numbers = unverified_numbers(wiki) if scope == "project" else []

    model_ok = False
    if use_model and model is not None:
        has_work = (
            build
            or bool(notes)
            or bool(lint.broken_links)
            or lint.index_over_budget
            or lint.standing_over_budget
            or bool(numbers)
        )
        client = harness.client_for(model)
        if not client.capabilities.memory_tool:
            log.info("dream: model %s has no memory tool; maintenance only", model)
        elif has_work:
            lab = scope == "lab"
            conversation = harness.open(
                agent="dreamer",
                stage=report.stage,
                model=model,
                system=prompts.DREAM_LAB_SYSTEM if lab else prompts.DREAM_SYSTEM,
                # The lab session works from the lab wiki alone: no project pages and no
                # transcripts, so no paper content is ever in front of it.
                memory=MemoryPolicy(write="all", transcripts=not lab),
                memory_root="lab" if lab else "project",
                effort="medium",
                label=f"dream.{scope}",
            )
            try:
                response = conversation.send(
                    _dream_prompt(
                        scope=scope,
                        focus=focus,
                        wiki=wiki,
                        notes=notes,
                        lint=lint,
                        numbers=numbers,
                    ),
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
            report.pages_changed = sorted(
                {rel for write_scope, rel in conversation.writes if write_scope == scope}
            )

    # Notes are archived only once a model has actually consolidated them; without one they
    # stay pending for the next dream.
    if model_ok and notes:
        report.notes_archived = len(wiki.archive_notes([n.path for n in notes], report.dream_id))
    repaired, lint = _maintain(wiki)
    report.links_repaired += repaired
    report.lint = lint.to_dict()
    if scope == "project":
        report.unverified_numbers = [list(pair) for pair in unverified_numbers(wiki)]
