"""A small on-disk cache kept beside the lab memory, out of every model's reach.

Pipeline code uses it for data that is worth keeping across runs but must not
be readable by a model session, such as bibliographic records fetched for
another paper.  It lives outside the wiki tree, so no memory tool, search or
index ever shows its contents.

Each entry is its own file, written atomically and never read back in order to
write.  Runs that share the directory (parallel local runs, or containers on a
last-write-wins volume) therefore cannot lose each other's entries.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class KeyValueCache:
    """``namespace`` + ``key`` -> a JSON value, one file per entry."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, namespace: str, key: str) -> Path:
        if not _NAMESPACE.match(namespace):
            raise ValueError(f"invalid cache namespace: {namespace!r}")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / digest[:2] / f"{digest[:40]}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        """The stored value, or ``None`` when there is none or it cannot be read."""
        path = self._path(namespace, key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("ignoring the unreadable cache entry %s: %s", path, exc)
            return None
        if not isinstance(entry, dict) or entry.get("key") != key:
            return None
        return entry.get("value")

    def put(self, namespace: str, key: str, value: Any) -> bool:
        """Store ``value``; returns False (and logs) when the entry could not be written."""
        path = self._path(namespace, key)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps({"key": key, "value": value}, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning("could not write the cache entry %s: %s", path, exc)
            with contextlib.suppress(OSError):
                temporary.unlink()
            return False
        return True

    def count(self, namespace: str) -> int:
        """Number of entries stored under ``namespace``."""
        if not _NAMESPACE.match(namespace):
            raise ValueError(f"invalid cache namespace: {namespace!r}")
        directory = self.root / namespace
        return sum(1 for _ in directory.glob("*/*.json")) if directory.is_dir() else 0
