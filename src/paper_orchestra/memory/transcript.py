"""The transcript layer: an append-only, searchable event log.

Transcripts are the bottom of the memory hierarchy.  They record what actually
happened in each session (requests, model output, tool calls, compile results,
review scores) so that notes and wiki pages can later be checked against them.
One JSON object per line; one file per session.
"""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_orchestra.llm.base import EventSink
from paper_orchestra.memory.wiki import tokenize

#: Longest string kept per field.  Full prompts repeat their inputs many times
#: over a run; the head and tail are what later verification needs.
MAX_FIELD_CHARS = 60_000


@dataclass(frozen=True)
class TranscriptHit:
    session: str
    seq: int
    ts: str
    type: str
    stage: str
    agent: str
    score: float
    snippet: str


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        half = MAX_FIELD_CHARS // 2
        dropped = len(value) - 2 * half
        return f"{value[:half]}\n[... {dropped} characters omitted ...]\n{value[-half:]}"
    if isinstance(value, dict):
        return {str(k): _clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return f"[{len(value)} bytes]"
    return value


def _event_text(event: dict[str, Any]) -> str:
    """The searchable text of an event (its string-valued payload)."""
    skip = {"ts", "seq", "session", "type", "stage", "agent", "model", "label"}
    chunks: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            chunks.append(str(value))

    for key, value in event.items():
        if key not in skip:
            walk(value)
    return "\n".join(chunks)


class TranscriptLog:
    """Append-only JSONL log for one session, searchable across all sessions."""

    def __init__(
        self,
        directory: str | Path,
        *,
        session_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self.session_id = session_id or (
            self._clock().strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        )
        self.path = self.directory / f"{self.session_id}.jsonl"
        self._lock = threading.Lock()
        self._seq = self._last_seq()

    def _last_seq(self) -> int:
        if not self.path.is_file():
            return 0
        last = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    last = max(last, int(json.loads(line).get("seq", 0)))
                except (ValueError, TypeError):
                    continue
        return last

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append ``event`` (stamped with ``ts``, ``seq`` and ``session``)."""
        with self._lock:
            self._seq += 1
            stamped = {
                "ts": self._clock().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seq": self._seq,
                "session": self.session_id,
                **_clip(event),
            }
            line = json.dumps(stamped, ensure_ascii=False, default=str)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return stamped

    def sink(self, **context: Any) -> EventSink:
        """An event sink that tags every event with ``context`` (stage, agent...)."""

        def emit(event: dict[str, Any]) -> None:
            self.record({**context, **event})

        return emit

    def events(self, *, session: str | None = None) -> Iterator[dict[str, Any]]:
        """Events of one session, or of every session in the directory, in order."""
        if session:
            files = [self.directory / f"{session}.jsonl"]
        else:
            files = sorted(self.directory.glob("*.jsonl"))
        for file in files:
            if not file.is_file():
                continue
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        yield event

    def grep(
        self,
        needle: str,
        *,
        k: int = 10,
        stage: str | None = None,
        session: str | None = None,
    ) -> list[TranscriptHit]:
        """Events containing ``needle`` verbatim (case-insensitive), most recent first."""
        wanted = needle.lower().strip()
        if not wanted:
            return []
        hits: list[TranscriptHit] = []
        for event in self.events(session=session):
            if stage and event.get("stage") != stage:
                continue
            text = _event_text(event)
            position = text.lower().find(wanted)
            if position < 0:
                continue
            hits.append(
                TranscriptHit(
                    session=str(event.get("session", "")),
                    seq=int(event.get("seq", 0)),
                    ts=str(event.get("ts", "")),
                    type=str(event.get("type", "")),
                    stage=str(event.get("stage", "")),
                    agent=str(event.get("agent", "")),
                    score=1.0,
                    snippet=_literal_snippet(text, position),
                )
            )
        hits.sort(key=lambda h: (h.ts, h.seq), reverse=True)
        return hits[: max(1, k)]

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        stage: str | None = None,
        types: Sequence[str] | None = None,
        session: str | None = None,
    ) -> list[TranscriptHit]:
        """Rank events by how many query terms they contain (rarer terms first).

        Ties go to the most recent event.  This is a verification aid, not a
        retrieval engine: it answers "did this actually happen, and what did
        the output say" for a note or page under review.
        """
        terms = set(tokenize(query))
        if not terms:
            return []
        candidates: list[tuple[dict[str, Any], str, set[str]]] = []
        doc_freq: dict[str, int] = dict.fromkeys(terms, 0)
        for event in self.events(session=session):
            if stage and event.get("stage") != stage:
                continue
            if types and event.get("type") not in types:
                continue
            text = _event_text(event)
            present = terms.intersection(tokenize(text))
            if not present:
                continue
            for term in present:
                doc_freq[term] += 1
            candidates.append((event, text, present))
        total = len(candidates) or 1
        hits: list[TranscriptHit] = []
        for event, text, present in candidates:
            score = sum(1.0 + (1.0 - doc_freq[t] / total) for t in present)
            hits.append(
                TranscriptHit(
                    session=str(event.get("session", "")),
                    seq=int(event.get("seq", 0)),
                    ts=str(event.get("ts", "")),
                    type=str(event.get("type", "")),
                    stage=str(event.get("stage", "")),
                    agent=str(event.get("agent", "")),
                    score=round(score, 4),
                    snippet=_snippet(text, present),
                )
            )
        hits.sort(key=lambda h: (h.score, h.ts, h.seq), reverse=True)
        return hits[: max(1, k)]


def _literal_snippet(text: str, position: int, width: int = 320) -> str:
    start = max(0, position - width // 4)
    chunk = " ".join(text[start : start + width].split())
    return ("…" if start > 0 else "") + chunk + ("…" if start + width < len(text) else "")


def _snippet(text: str, terms: set[str], width: int = 320) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms]
    positions = [p for p in positions if p >= 0]
    start = max(0, (min(positions) if positions else 0) - width // 4)
    chunk = " ".join(text[start : start + width].split())
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + width < len(text) else ""
    return f"{prefix}{chunk}{suffix}"
