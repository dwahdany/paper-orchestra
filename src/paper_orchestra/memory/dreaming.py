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
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from paper_orchestra.llm.base import LLMError
from paper_orchestra.memory import prompts
from paper_orchestra.memory.leakscan import (  # noqa: F401 - re-exported
    CONTENT_SECTIONS,
    NUMBER,
    LeakScanner,
    Materials,
    _significant,
    content_sources,
    leak_reason,
    number_readings,
    numbers_in,
    paper_materials,
    protected_files,
    venue_materials,
)
from paper_orchestra.memory.lock import LOCK_FILE, STALE_LOCK_SECONDS, WikiLock  # noqa: F401
from paper_orchestra.memory.tools import virtual_path
from paper_orchestra.memory.transcript import TranscriptLog, evidence_text
from paper_orchestra.memory.wiki import (
    INDEX_FILE,
    NOTES_DIR,
    WIKI_DIR,
    LintReport,
    PageInfo,
    Wiki,
    WikiError,
)

if TYPE_CHECKING:
    from paper_orchestra.memory.harness import ContextHarness

log = logging.getLogger(__name__)

PROMOTED_LEDGER = ".promoted.json"
RUN_TOKEN_FILE = ".run-token"
DREAM_STATE_FILE = ".dream-state.json"
#: A pending note this short is put into the dream prompt whole, until the budget is used.
MAX_INLINE_NOTE_CHARS = 4_000
INLINE_NOTES_BUDGET = 48_000
NOTE_PREVIEW_CHARS = 300
#: Notes named in the dream prompt; the rest are counted for the model to open.
MAX_LISTED_NOTES = 120
#: Unverified numbers listed in the dream prompt.
MAX_LISTED_NUMBERS = 30
#: The agent name of consolidation sessions.  Their events are not evidence about the run.
DREAMER = "dreamer"
#: Where an earlier version kept the bibliographic cache, inside the lab wiki.
LEGACY_CACHE_SECTION = "wiki/literature-cache"
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


#: A number, but not the factor of a LaTeX length (``0.48\textwidth``): layout, not a result.
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
    #: Lessons this run copied to the lab memory, and lessons the leak scan kept back.  A
    #: learning loop that has stopped sharing shows here, not only in the log.
    lessons_shared: int = 0
    lessons_withheld: int = 0
    summary: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _transcript_numbers(transcripts: TranscriptLog) -> set[str]:
    """Numbers the run itself put on record: compile results, review scores, tool output.

    Only records that cannot have come out of memory count.  Prompts repeat the
    memory index, notes are claims, memory-tool traffic and consolidation
    sessions quote the pages being checked, and a model's own answer may repeat
    a number it read in memory a moment before.  Usage counters are numbers
    about the run, not about the paper.
    """
    found: set[str] = set()
    for event in transcripts.events(exclude_agents=(DREAMER,)):
        if event.get("type") in ("request", "assistant", "usage", "note", "dream"):
            continue
        text = evidence_text(event)
        if text:
            found |= numbers_in(text)
    return found


def unverified_numbers(
    wiki: Wiki, *, transcripts: TranscriptLog | None = None, limit: int = 400
) -> list[tuple[str, str]]:
    """``(page, number)`` for numbers in model-written pages that no record contains.

    The read-only sections of a wiki are its records (inputs, bibliography,
    reviews), and so is what the run's transcripts recorded (compile output,
    review scores).  A specific number that appears in a consolidated page or
    in the standing notes but in none of those records was either mistyped or
    made up, and every later session would trust it.
    """
    evidence: set[str] = set()
    written: list[tuple[str, str]] = []
    for info in wiki.list():
        if wiki.is_read_only(info.path):
            evidence |= numbers_in(wiki.read_text(info.path))
        elif not wiki.read(info.path).meta.get("auto"):
            written.append((info.path, wiki.read(info.path).body))
    written.append((INDEX_FILE, wiki.standing_notes()))
    if transcripts is not None:
        evidence |= _transcript_numbers(transcripts)
    findings: list[tuple[str, str]] = []
    for path, body in written:
        seen: set[str] = set()
        for raw in NUMBER.findall(body):
            readings = number_readings(raw)
            number = readings[0]
            if number in seen or not _significant(number):
                continue
            seen.add(number)
            supported = number in evidence or (
                len(readings) > 1 and all(part in evidence for part in readings[1:])
            )
            if not supported:
                findings.append((path, raw))
                if len(findings) >= limit:
                    return findings
    return findings


# ---------------------------------------------------------------------------
# Sharing lessons with the lab memory
# ---------------------------------------------------------------------------


def run_token(project: Wiki) -> str:
    """An opaque identifier of this project memory, so the lab can tell runs apart.

    It says nothing about the paper; it only lets a consolidation see that two
    lesson notes came from the same run and are not independent evidence.
    """
    path = project.root / RUN_TOKEN_FILE
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if not re.fullmatch(r"[0-9a-f]{12}", token):
        token = secrets.token_hex(6)
        path.write_text(token + "\n", encoding="utf-8")
    return token


def _load_ledger(path: Path) -> tuple[dict[str, str], set[str]]:
    """``(page path -> digest promoted, digests promoted without a page path)``.

    The second set holds what an earlier format of the ledger recorded (digests
    only); it is carried along so those lessons are not shared a second time.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, set()
    if isinstance(data, list):
        return {}, {str(item) for item in data}
    if isinstance(data, dict) and isinstance(data.get("pages"), dict):
        digests = data.get("digests")
        return (
            {str(k): str(v) for k, v in data["pages"].items()},
            {str(item) for item in digests} if isinstance(digests, list) else set(),
        )
    return {}, set()


def _is_portable(value: Any) -> bool:
    """``portable: true`` in the spellings a model writes it (True, yes, "true", 1)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().strip("\"'").lower() in ("true", "yes", "1")


def promote_lessons(harness: ContextHarness) -> list[str]:
    """Copy this run's portable lessons into the lab memory as notes.

    A lesson is portable when a consolidation session marked its page
    ``portable: true``.  Each one is leak-scanned against the paper's own
    materials and, if clean, added to the lab as a uniquely named note: runs
    only ever *add* files to the shared memory, so parallel runs cannot
    overwrite each other.  Nothing is shared when there is nothing to scan
    against.  Returns the lab note paths.
    """
    project, lab = harness.project, harness.lab
    if project is None or lab is None:
        return []
    portable = []
    for info in project.list(WIKI_DIR):
        if project.is_read_only(info.path):
            continue
        page = project.read(info.path)
        if _is_portable(page.meta.get("portable")) and not page.meta.get("auto"):
            portable.append(page)
    harness.last_promotion = {"shared": 0, "withheld": 0}
    if not portable:
        return []
    scanner = harness.leak_scanner()
    if not scanner.has_sources:
        reason = "the paper's materials are not available to scan against"
        log.warning("%d lesson(s) not shared with the lab memory: %s", len(portable), reason)
        harness.record("promotion_blocked", stage="finalize", path="*", reason=reason)
        harness.last_promotion["withheld"] = len(portable)
        return []

    ledger_path = project.root / PROMOTED_LEDGER
    promoted, legacy = _load_ledger(ledger_path)
    token = run_token(project)
    added: list[str] = []
    for page in portable:
        text = f"# {page.title}\n\n{page.body.strip()}\n"
        # The lesson is its body: a page that was only renamed (its title may follow its
        # path) is the same lesson.
        digest = hashlib.sha256(page.body.strip().encode("utf-8")).hexdigest()
        with_title = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Already shared: under this path, under another, or by a ledger that recorded
        # digests of the whole note and no paths.
        if digest in promoted.values() or {digest, with_title} & legacy:
            continue
        reason = scanner.reason(text)
        if reason:
            log.warning("lesson %s not shared with the lab memory: %s", page.path, reason)
            harness.record("promotion_blocked", stage="finalize", path=page.path, reason=reason)
            harness.last_promotion["withheld"] += 1
            continue
        note = lab.add_note(
            text,
            stage="promotion",
            agent="pipeline",
            kind="lesson",
            title=page.title,
            meta={"run": token},
        )
        promoted[page.path] = digest
        added.append(note)
        harness.record("promotion", stage="finalize", path=page.path, lab_note=note)
    if added:
        ledger = {"pages": promoted, "digests": sorted(legacy)}
        ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    harness.last_promotion["shared"] = len(added)
    return added


# ---------------------------------------------------------------------------
# The dream itself
# ---------------------------------------------------------------------------


def _note_listing(wiki: Wiki, notes: list[PageInfo]) -> tuple[str, list[str]]:
    """The pending notes as prompt text, and the paths of those included whole.

    Short notes are included in full until the budget is used.  A note that is
    cut says so, because a consolidation may only archive what it has read.
    """
    lines: list[str] = []
    shown: list[str] = []
    used = 0
    for index, info in enumerate(notes):
        if index >= MAX_LISTED_NOTES:
            lines.append(
                f"- ... and {len(notes) - index} more under {virtual_path(NOTES_DIR)}: "
                "open them to consolidate them"
            )
            break
        page = wiki.read(info.path)
        body = page.body.strip()
        header = (
            f"- {virtual_path(info.path)} "
            f"({page.kind or 'observation'}, stage {page.meta.get('stage') or '-'}"
        )
        if page.meta.get("run"):
            header += f", run {page.meta['run']}"
        if page.meta.get("presented"):
            header += (
                ", already shown to an earlier consolidation: check whether the pages "
                "hold it before counting it as new evidence"
            )
        if len(body) <= MAX_INLINE_NOTE_CHARS and used + len(body) <= INLINE_NOTES_BUDGET:
            used += len(body)
            shown.append(info.path)
            lines.append(header + "):")
            lines.extend(f"    {line}" for line in body.splitlines())
        else:
            preview = " ".join(body.split())[:NOTE_PREVIEW_CHARS]
            lines.append(
                f"{header}, {len(body)} characters, only the beginning is shown here, "
                f"open the file to read it): {preview} ..."
            )
    return "\n".join(lines), shown


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
    for page, number in numbers[:MAX_LISTED_NUMBERS]:
        lines.append(
            f"- {virtual_path(page)} states {number}, which appears in no source, review or "
            "transcript record: check it, and correct or remove it if nothing supports it"
        )
    return "\n".join(lines)


def _dream_prompt(
    *,
    scope: str,
    focus: str,
    wiki: Wiki,
    notes: list[PageInfo],
    listing: str,
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
        parts += ["", f"Unconsolidated notes ({len(notes)}):", listing]
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


def _drop_legacy_cache_pages(wiki: Wiki) -> None:
    """Remove bibliographic cache pages that an earlier version kept inside the lab wiki.

    The cache now lives beside the wiki, where no session can read it.  Pages
    left from before would still be searchable by every paper's sessions.
    """
    legacy = wiki.root / LEGACY_CACHE_SECTION
    if legacy.is_dir():
        shutil.rmtree(legacy, ignore_errors=True)
        log.info("Removed the bibliographic cache pages an earlier version kept in the lab wiki.")


def _reported_numbers(wiki: Wiki) -> set[tuple[str, str]]:
    """Unverified numbers that a consolidation session has already been asked about."""
    try:
        data = json.loads((wiki.root / DREAM_STATE_FILE).read_text(encoding="utf-8"))
        return {(str(page), str(number)) for page, number in data.get("numbers", [])}
    except (OSError, ValueError, TypeError, AttributeError):
        return set()


def _save_reported_numbers(wiki: Wiki, numbers: list[tuple[str, str]]) -> None:
    with contextlib.suppress(OSError):
        (wiki.root / DREAM_STATE_FILE).write_text(
            json.dumps({"numbers": [list(pair) for pair in numbers]}), encoding="utf-8"
        )


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

    from paper_orchestra.memory.harness import RunCancelled

    try:
        with WikiLock(wiki.root, wait_seconds=lock_wait) as lock:
            if lock.acquired:
                _run(
                    harness,
                    wiki,
                    report,
                    focus=focus,
                    model=model,
                    use_model=use_model,
                    build=build,
                )
            else:
                report.error = lock.error or "another consolidation of this wiki is in progress"
                log.warning("dream %s skipped: %s", report.dream_id, report.error)
    except (LLMError, RunCancelled) as exc:
        # No credentials, an unknown model, or the run was stopped: expected, no traceback.
        report.error = f"{type(exc).__name__}: {exc}"
        log.warning("dream %s did not complete: %s", report.dream_id, exc)
    except Exception as exc:  # a dream must never fail the run
        report.error = f"{type(exc).__name__}: {exc}"
        log.exception("dream %s failed", report.dream_id)
    fields = {key: value for key, value in report.to_dict().items() if key != "stage"}
    harness.record("dream", stage=stage, agent=DREAMER, **fields)
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
    lab = scope == "lab"
    transcripts = None if lab else harness.transcripts
    if lab:
        _drop_legacy_cache_pages(wiki)
    repaired, lint = _maintain(wiki)
    report.links_repaired = repaired
    notes = wiki.pending_notes()
    report.notes_seen = len(notes)
    numbers = [] if lab else unverified_numbers(wiki, transcripts=transcripts)

    handled: list[str] = []
    shown: list[str] = []
    listed: list[tuple[str, str]] = []
    model_ok = False
    vetoes = harness.lab_vetoes
    # A number that an earlier session was already asked about, and kept, is not a reason
    # to open another session on its own.  New ones come first, so they are always listed.
    asked = _reported_numbers(wiki)
    numbers.sort(key=lambda pair: pair in asked)
    if use_model and model is not None:
        has_work = (
            build
            or bool(notes)
            or bool(lint.broken_links)
            or lint.index_over_budget
            or lint.standing_over_budget
            or any(pair not in asked for pair in numbers)
        )
        client = harness.client_for(model)
        if not client.capabilities.memory_tool:
            log.info("dream: model %s has no memory tool; maintenance only", model)
        elif has_work:
            listing, shown = _note_listing(wiki, notes)
            listed = numbers[:MAX_LISTED_NUMBERS]
            conversation = harness.open(
                agent=DREAMER,
                stage=report.stage,
                model=model,
                system=prompts.DREAM_LAB_SYSTEM if lab else prompts.DREAM_SYSTEM,
                # The lab session works from the lab wiki alone: no project pages and no
                # transcripts, so no paper content is ever in front of it.  The project
                # session in turn does not see the lab: what it writes is material for this
                # paper, which must come from this run alone.
                memory=MemoryPolicy(write="all", transcripts=not lab, lab=False),
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
                        listing=listing,
                        lint=lint,
                        numbers=listed,
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
            # Only what the session had in front of it counts as dealt with: the notes
            # included whole in its prompt and the ones it opened or edited.
            touched = {
                rel
                for touched_scope, rel in [*conversation.reads, *conversation.writes]
                if touched_scope == scope
            }
            handled = [n.path for n in notes if n.path in shown or n.path in touched]

    if lab and harness.lab_vetoes > vetoes:
        # A lesson the session tried to write was turned down, so its note has not reached
        # a page.  Everything stays pending: the next consolidation, run for another paper,
        # will not collide with this one.
        log.warning(
            "dream %s: %d write(s) to the lab memory were refused; its notes stay pending",
            report.dream_id,
            harness.lab_vetoes - vetoes,
        )
        handled = []
        # The next consolidation will be shown these notes again.  It is told so, because
        # some of them may already be in the pages.
        for rel in shown:
            with contextlib.suppress(WikiError):
                page = wiki.read(rel)
                wiki.write(rel, page.body, {**page.meta, "presented": report.dream_id})
    # Notes are archived only once a model has actually consolidated them; without one they
    # stay pending for the next dream.
    if model_ok and handled:
        report.notes_archived = len(wiki.archive_notes(handled, report.dream_id))
    repaired, lint = _maintain(wiki)
    report.links_repaired += repaired
    report.lint = lint.to_dict()
    if not lab:
        remaining = unverified_numbers(wiki, transcripts=transcripts)
        report.unverified_numbers = [list(pair) for pair in remaining]
        if model_ok:
            # Only what a session was shown and kept is settled; a number the session
            # itself introduced has not been looked at by anyone.
            settled = asked | set(listed)
            _save_reported_numbers(wiki, [pair for pair in remaining if pair in settled])
