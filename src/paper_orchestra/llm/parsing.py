"""Parsers that turn a model's text into JSON / LaTeX.

These are provider-independent: every model is asked for the same fenced
blocks, and a client that supports native structured output bypasses them.
"""

from __future__ import annotations

import json
import re
from typing import Any

import json_repair

_FENCE = r"```[ \t]*{lang}[ \t]*\r?\n(.*?)```"
_JSON_FENCE = re.compile(_FENCE.format(lang="json"), re.DOTALL | re.IGNORECASE)
_LATEX_FENCE = re.compile(_FENCE.format(lang="(?:latex|tex)"), re.DOTALL | re.IGNORECASE)
_PYTHON_FENCE = re.compile(_FENCE.format(lang="(?:python|py)"), re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```[^\n`]*\r?\n(.*?)```", re.DOTALL)


def identity(text: str) -> str:
    return text


def _loads_lenient(candidate: str) -> Any:
    """``json.loads`` with a repair fallback; returns ``None`` when hopeless."""
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        repaired = json_repair.loads(candidate)
    except Exception:
        return None
    # json_repair returns "" (or the bare string) when nothing JSON-like is found.
    return repaired if isinstance(repaired, (dict, list)) else None


def _outermost(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    end = text.rfind(close_ch)
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_json(text: str | None) -> Any:
    """Extract the first JSON object/array from ``text``.

    Prefers ```` ```json ```` fences, then the outermost ``{...}`` or ``[...]``
    span (whichever starts first).  Returns ``None`` when nothing parses.
    """
    if not text or not isinstance(text, str):
        return None

    for match in _JSON_FENCE.finditer(text):
        parsed = _loads_lenient(match.group(1))
        if parsed is not None:
            return parsed

    spans = [s for s in (_outermost(text, "{", "}"), _outermost(text, "[", "]")) if s]
    spans.sort(key=lambda s: text.find(s))
    for span in spans:
        parsed = _loads_lenient(span)
        if parsed is not None:
            return parsed
    return None


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Like :func:`parse_json` but only accepts a JSON object."""
    parsed = parse_json(text)
    return parsed if isinstance(parsed, dict) else None


def parse_latex(text: str | None) -> str | None:
    """Return the LaTeX document in ``text``.

    Takes the longest ```` ```latex ```` block (models sometimes show a short
    snippet before the full document); falls back to the whole response when it
    is unfenced LaTeX.  Returns ``None`` for empty input.
    """
    if not text or not text.strip():
        return None
    blocks = [m.group(1) for m in _LATEX_FENCE.finditer(text)]
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    stripped = text.strip()
    if "\\documentclass" in stripped or "\\begin{document}" in stripped:
        fenced = _ANY_FENCE.search(stripped)
        if fenced and "\\documentclass" in fenced.group(1):
            return fenced.group(1).strip() + "\n"
        return stripped + "\n"
    return None


def parse_json_and_latex(text: str | None) -> dict[str, Any] | None:
    """Parse a response made of a JSON worklog block followed by a LaTeX block.

    Returns ``{"json": <object or None>, "latex": <str>}``, or ``None`` when no
    LaTeX document is present (the caller cannot proceed without one).
    """
    latex = parse_latex(text)
    if latex is None:
        return None
    worklog: Any = None
    if text:
        for match in _JSON_FENCE.finditer(text):
            worklog = _loads_lenient(match.group(1))
            if worklog is not None:
                break
    return {"json": worklog, "latex": latex}


def parse_python(text: str | None) -> str | None:
    """Return the Python source in ``text`` (fenced, or the whole response)."""
    if not text or not text.strip():
        return None
    blocks = [m.group(1) for m in _PYTHON_FENCE.finditer(text)]
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    fenced = _ANY_FENCE.search(text)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return text.strip() + "\n"
