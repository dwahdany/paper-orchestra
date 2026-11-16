"""The web app: an HTTP API over research projects, live events, and the static frontend.

``paper-orchestra ui`` serves it.  The server never runs the engine itself: it
starts ``paper-orchestra research ...`` as a separate process per project
(:mod:`paper_orchestra.web.workers`) and learns everything from the project's
files.

Importing this package does not import the web stack (FastAPI, uvicorn,
markdown-it-py), which is an optional extra; :func:`create_app` and
:func:`serve` are loaded on first use.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "serve"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from paper_orchestra.web import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
