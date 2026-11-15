"""Compute backends: where model-written Python and LaTeX builds run.

``LocalBackend`` and ``ModalBackend`` are loaded on first access, so importing
this package never pulls in the Modal SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paper_orchestra.compute.base import CompileResult, ComputeBackend, ComputeError, ExecResult

if TYPE_CHECKING:
    from paper_orchestra.compute.local import LocalBackend
    from paper_orchestra.compute.modal_backend import ModalBackend

__all__ = [
    "BACKEND_NAMES",
    "CompileResult",
    "ComputeBackend",
    "ComputeError",
    "ExecResult",
    "LocalBackend",
    "ModalBackend",
    "get_backend",
]

BACKEND_NAMES = ("local", "modal")

_LAZY_EXPORTS = {
    "LocalBackend": "paper_orchestra.compute.local",
    "ModalBackend": "paper_orchestra.compute.modal_backend",
}


def get_backend(name: str, **options: Any) -> ComputeBackend:
    """Build the compute backend called ``name`` (``"local"`` or ``"modal"``).

    ``options`` are passed to the backend's constructor.  Asking for ``"modal"``
    without the ``modal`` extra installed raises :class:`ComputeError` with the
    install command.
    """
    if name == "local":
        from paper_orchestra.compute.local import LocalBackend

        return LocalBackend(**options)
    if name == "modal":
        from paper_orchestra.compute.modal_backend import ModalBackend

        return ModalBackend(**options)
    raise ComputeError(
        f"unknown compute backend {name!r}; choose one of: {', '.join(BACKEND_NAMES)}"
    )


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module), name)
