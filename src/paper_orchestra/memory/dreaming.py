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
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from paper_orchestra.llm.base import LLMError
from paper_orchestra.memory import prompts
from paper_orchestra.memory.tools import virtual_path
from paper_orchestra.memory.transcript import TranscriptLog, event_text
from paper_orchestra.memory.wiki import (
    INDEX_FILE,
    NOTES_DIR,
    WIKI_DIR,
    LintReport,
    PageInfo,
    Wiki,
    tokenize,
)

if TYPE_CHECKING:
    from paper_orchestra.memory.harness import ContextHarness

log = logging.getLogger(__name__)

LOCK_FILE = ".dream.lock"
PROMOTED_LEDGER = ".promoted.json"
RUN_TOKEN_FILE = ".run-token"
DREAM_STATE_FILE = ".dream-state.json"
#: A lock not refreshed for this long belongs to a session that died.
STALE_LOCK_SECONDS = 15 * 60
LOCK_REFRESH_SECONDS = 60
#: A pending note this short is put into the dream prompt whole, until the budget is used.
MAX_INLINE_NOTE_CHARS = 4_000
INLINE_NOTES_BUDGET = 48_000
NOTE_PREVIEW_CHARS = 300
#: Notes named in the dream prompt; the rest are counted for the model to open.
MAX_LISTED_NOTES = 120
#: The agent name of consolidation sessions.  Their events are not evidence about the run.
DREAMER = "dreamer"
#: Sections of the project wiki that hold the paper's own content.  Nothing from them may
#: reach the lab memory.  (The literature section is public bibliographic data.)
CONTENT_SECTIONS = (
    "wiki/sources/",
    "wiki/paper/",
    "wiki/figures/",
    "wiki/reviews/",
    "wiki/drafts/",
)
#: Source pages that describe the venue rather than the paper.  Lessons may quote them.
VENUE_SOURCE_PREFIXES = ("wiki/sources/guidelines", "wiki/sources/template")
VENUE_FILE_NAMES = frozenset({"template.tex", "guidelines.md"})
_TEXT_SUFFIXES = frozenset({".md", ".txt", ".tex", ".json", ".csv", ".tsv", ".yaml", ".yml"})
MAX_PROTECTED_FILE_BYTES = 2_000_000
SHINGLE_TOKENS = 5

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

_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)")
_YEAR = re.compile(r"(?:19|20)\d\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_HOME_PATH = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)[\w.-]+")
_CREDENTIAL = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}|(?i:\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*\S{6,})"
)
#: Words with a capital after the first letter and at least one lower-case letter (FooNet).
_CAMEL = re.compile(r"\b(?=[A-Za-z0-9]*[a-z])[A-Za-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")
#: Upper-case names of three or more characters, with optional suffixes (CIFAR-10).
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")

#: Names that say nothing about a particular paper: tools, formats, venues and the common
#: vocabulary of the field.  Any other distinctive name found in a paper's materials is
#: treated as identifying that paper.
COMMON_TERMS = frozenset(
    """
    TeX LaTeX BibTeX BibLaTeX pdfTeX pdfLaTeX XeTeX XeLaTeX LuaTeX LuaLaTeX TikZ PGFPlots PGF
    CTAN OT1 UTF PDF DVI EPS PNG JPG JPEG SVG GIF DPI RGB CMYK URL DOI ISBN ISSN HTML XML JSON
    YAML CSV TSV API CLI CPU GPU RAM OOM ASCII arXiv OpenReview GitHub ORCID
    IEEE ACM CVPR ICCV ECCV WACV ICLR ICML NeurIPS NIPS AAAI IJCAI ACL EMNLP NAACL COLING
    KDD SIGIR SIGGRAPH TMLR JMLR PMLR TPAMI IJCV
    NumPy SciPy PyTorch TensorFlow JAX CUDA OpenCV
    SOTA MLP CNN RNN LSTM GRU GAN VAE SGD NLP LLM AUC ROC MSE MAE RMSE
    TODO FIXME NOTE WARNING ERROR EVIDENCE CONFLICT RULE WHY
    """.split()  # noqa: SIM905
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


def _process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


class _DreamLock:
    """Exclusive lock file so parallel runs sharing a wiki consolidate one at a time.

    The holder refreshes the file's timestamp while it works, so a long dream is
    not mistaken for a dead one.  The file names the holder's process and host:
    a lock left by a process that no longer exists on this host is taken over at
    once, any other lock after ``STALE_LOCK_SECONDS`` without a refresh.  This
    is mutual exclusion on a local file system; storage without atomic exclusive
    create (some network volumes) needs a single consolidating process instead.
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

    def _abandoned(self) -> bool:
        """Whether the existing lock file belongs to a holder that is certainly gone."""
        try:
            if time.time() - self.path.stat().st_mtime > STALE_LOCK_SECONDS:
                return True
            fields = self.path.read_text(encoding="utf-8").split()
        except OSError:
            return False
        if len(fields) >= 2 and fields[1] == socket.gethostname() and fields[0].isdigit():
            pid = int(fields[0])
            return pid != os.getpid() and _process_is_gone(pid)
        return False

    def __enter__(self) -> _DreamLock:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._abandoned():
                    with contextlib.suppress(OSError):
                        self.path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    return self
                time.sleep(self.poll)
                continue
            with os.fdopen(fd, "w") as handle:
                handle.write(f"{os.getpid()} {socket.gethostname()}\n")
            self.acquired = True
            self._refresher = threading.Thread(target=self._refresh, daemon=True)
            self._refresher.start()
            return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self.acquired:
            self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _canonical(raw: str) -> str:
    """``2,908`` -> ``2908``, ``87.30`` -> ``87.3``, ``007`` -> ``7``."""
    whole, _, fraction = raw.replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def _readings(raw: str) -> list[str]:
    """Canonical forms a matched number can stand for.

    ``12,345`` is either one number with a thousands separator or two list
    items, so both readings are returned, the joined one first.
    """
    readings = [_canonical(raw)]
    if "," in raw:
        whole, _, fraction = raw.partition(".")
        parts = whole.split(",")
        if fraction:
            parts[-1] = f"{parts[-1]}.{fraction}"
        readings.extend(_canonical(part) for part in parts)
    return readings


def numbers_in(text: str) -> set[str]:
    """Every number in ``text`` in canonical form, under each possible reading."""
    found: set[str] = set()
    for raw in _NUMBER.findall(text):
        found.update(_readings(raw))
    return found


def _significant(number: str) -> bool:
    """Numbers specific enough that a wrong one matters: decimals with three or more
    digits, or integers of four or more digits that are not calendar years."""
    whole, _, fraction = number.partition(".")
    if fraction:
        return len(whole + fraction) >= 3
    return len(whole) >= 4 and not _YEAR.fullmatch(whole)


def _identifying(number: str) -> bool:
    """Numbers that could identify a result: at least two significant digits, and for
    integers at least three digits that are not a round figure or a calendar year."""
    whole, _, fraction = number.partition(".")
    if fraction:
        return len((whole + fraction).lstrip("0")) >= 2
    return len(whole) >= 3 and len(whole.rstrip("0")) >= 2 and not _YEAR.fullmatch(whole)


def _transcript_numbers(transcripts: TranscriptLog) -> set[str]:
    """Numbers recorded by the run itself: outputs, tool results, compile and review records.

    Prompts are left out because they repeat the memory index, notes are left
    out because they are claims, and memory-tool traffic and consolidation
    sessions are left out because they quote the pages being checked.
    """
    found: set[str] = set()
    for event in transcripts.events(exclude_agents=(DREAMER,)):
        kind = event.get("type")
        if kind in ("request", "note", "memory_read", "dream"):
            continue
        if kind in ("tool_call", "tool_result") and event.get("name") in (
            "memory",
            "wiki_search",
            "transcript_search",
        ):
            continue
        found |= numbers_in(event_text(event))
    return found


def unverified_numbers(
    wiki: Wiki, *, transcripts: TranscriptLog | None = None, limit: int = 40
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
        for raw in _NUMBER.findall(body):
            readings = _readings(raw)
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
# Keeping paper content out of the shared memory
# ---------------------------------------------------------------------------


def _shingles(text: str) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    return {tuple(tokens[i : i + SHINGLE_TOKENS]) for i in range(len(tokens) - SHINGLE_TOKENS + 1)}


def _names(text: str) -> tuple[set[str], set[str]]:
    """Distinctive names in ``text``: mixed-case words (lower-cased) and upper-case names."""
    return {match.lower() for match in _CAMEL.findall(text)}, set(_ACRONYM.findall(text))


class LeakScanner:
    """Decides mechanically whether a text may enter the memory shared between papers.

    Built from a paper's own materials (``sources``) and from texts that
    describe the venue rather than the paper (``allowed``: the template and the
    guidelines, which lessons legitimately quote).  A text is turned down when
    it contains an e-mail address, a home-directory path or a credential, or
    when it repeats something from the paper's materials: a result-like number,
    a distinctive name (a method, dataset or system), or a run of wording.

    This is a filter on form, not on meaning.  It stops the copies that happen
    in practice (a lesson that carries its evidence along); it cannot stop a
    paraphrase that shares no name, number or phrase with the paper.
    """

    def __init__(self, sources: Sequence[str], allowed: Sequence[str] = ()) -> None:
        corpus = "\n".join(sources)
        permitted = "\n".join(allowed)
        self.has_sources = bool(corpus.strip())
        self._numbers = {n for n in numbers_in(corpus) - numbers_in(permitted) if _identifying(n)}
        mixed, upper = _names(corpus)
        permitted_mixed, permitted_upper = _names(permitted)
        self._mixed = mixed - permitted_mixed - {term.lower() for term in COMMON_TERMS}
        self._upper = upper - permitted_upper - COMMON_TERMS
        self._shingles = _shingles(corpus) - _shingles(permitted)

    def reason(self, text: str) -> str | None:
        """Why ``text`` must not enter the shared memory, or ``None`` if it may."""
        match = _EMAIL.search(text)
        if match:
            return f"it contains an e-mail address ({match.group(0)})"
        match = _HOME_PATH.search(text)
        if match:
            return f"it contains a personal file-system path ({match.group(0)})"
        if _CREDENTIAL.search(text):
            return "it contains something that looks like a credential"
        if not self.has_sources:
            return None
        for raw in dict.fromkeys(_NUMBER.findall(text)):
            for number in _readings(raw):
                if number in self._numbers:
                    return f"it repeats a number from this paper's materials ({raw})"
        for name in dict.fromkeys(_ACRONYM.findall(text)):
            if name in self._upper:
                return f"it names something from this paper's materials ({name})"
        for word in dict.fromkeys(_WORD.findall(text)):
            if word.lower() in self._mixed:
                return f"it names something from this paper's materials ({word})"
        shared = _shingles(text) & self._shingles
        if shared:
            return (
                "it repeats wording from this paper's materials (\"" + " ".join(min(shared)) + '")'
            )
        return None


def leak_reason(text: str, sources: Sequence[str], allowed: Sequence[str] = ()) -> str | None:
    """One-off form of :class:`LeakScanner`: why ``text`` may not be shared, or ``None``."""
    return LeakScanner(sources, allowed).reason(text)


def protected_files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file() and not file.is_symlink():
                    yield file


def _read_protected(file: Path) -> str | None:
    try:
        if file.stat().st_size > MAX_PROTECTED_FILE_BYTES:
            return None
        return file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def content_sources(project: Wiki, paths: Sequence[Path] = ()) -> list[str]:
    """Every text that holds the paper's own content.

    From the project wiki: the inputs, the outline, figure captions, reviews
    and draft records.  From ``paths`` (the run's files and directories that
    the harness was told to protect): the raw materials and every draft of the
    paper.  The venue's template and guidelines are not paper content.
    """
    texts = [
        project.read_text(info.path)
        for section in CONTENT_SECTIONS
        for info in project.list(section.rstrip("/"))
        if not info.path.startswith(VENUE_SOURCE_PREFIXES)
    ]
    for file in protected_files(paths):
        if file.suffix.lower() in _TEXT_SUFFIXES and file.name not in VENUE_FILE_NAMES:
            text = _read_protected(file)
            if text:
                texts.append(text)
    return texts


def venue_materials(project: Wiki, paths: Sequence[Path] = ()) -> list[str]:
    """The venue's template and guidelines: shared by every paper for that venue."""
    texts = [
        project.read_text(info.path)
        for info in project.list("wiki/sources")
        if info.path.startswith(VENUE_SOURCE_PREFIXES)
    ]
    for file in protected_files(paths):
        if file.name in VENUE_FILE_NAMES:
            text = _read_protected(file)
            if text:
                texts.append(text)
    return texts


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
    """``(page path -> digest promoted, digests promoted by earlier versions)``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, set()
    if isinstance(data, list):
        return {}, {str(item) for item in data}
    if isinstance(data, dict) and isinstance(data.get("pages"), dict):
        return {str(k): str(v) for k, v in data["pages"].items()}, set()
    return {}, set()


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
        if page.meta.get("portable") is True and not page.meta.get("auto"):
            portable.append(page)
    if not portable:
        return []
    scanner = harness.leak_scanner()
    if not scanner.has_sources:
        reason = "the paper's materials are not available to scan against"
        log.warning("%d lesson(s) not shared with the lab memory: %s", len(portable), reason)
        harness.record("promotion_blocked", stage="finalize", path="*", reason=reason)
        return []

    ledger_path = project.root / PROMOTED_LEDGER
    promoted, legacy = _load_ledger(ledger_path)
    token = run_token(project)
    added: list[str] = []
    for page in portable:
        text = f"# {page.title}\n\n{page.body.strip()}\n"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if promoted.get(page.path) == digest or digest in legacy:
            continue
        reason = scanner.reason(text)
        if reason:
            log.warning("lesson %s not shared with the lab memory: %s", page.path, reason)
            harness.record("promotion_blocked", stage="finalize", path=page.path, reason=reason)
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
        ledger_path.write_text(json.dumps({"pages": promoted}, sort_keys=True), encoding="utf-8")
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
    for page, number in numbers[:30]:
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
            except LLMError as exc:  # no credentials, unknown model: expected, no traceback
                report.error = f"{type(exc).__name__}: {exc}"
                log.warning("dream %s failed: %s", report.dream_id, exc)
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
    repaired, lint = _maintain(wiki)
    report.links_repaired = repaired
    notes = wiki.pending_notes()
    report.notes_seen = len(notes)
    numbers = [] if lab else unverified_numbers(wiki, transcripts=transcripts)

    handled: list[str] = []
    model_ok = False
    if use_model and model is not None:
        # A number that an earlier session was already asked about, and kept, is not a
        # reason to open another session on its own.
        asked = _reported_numbers(wiki)
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
            # Only what the session had in front of it counts as dealt with: the notes
            # included whole in its prompt and the ones it opened or edited.
            touched = {
                rel
                for touched_scope, rel in [*conversation.reads, *conversation.writes]
                if touched_scope == scope
            }
            handled = [n.path for n in notes if n.path in shown or n.path in touched]

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
            _save_reported_numbers(wiki, remaining)
