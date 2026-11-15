"""Small file helpers that fail loudly instead of returning ``None``."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file; a missing file is an error, not an empty string."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"required file not found: {p}")
    return p.read_text(encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON; return ``default`` only when the file does not exist."""
    p = Path(path)
    if not p.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(f"required file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> Path:
    """Atomically write UTF-8 text, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> Path:
    return write_text(path, json.dumps(data, indent=4, ensure_ascii=False, default=str) + "\n")
