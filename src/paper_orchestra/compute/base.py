"""Compute backend contract.

Two kinds of work leave the orchestrator process:

* running model-written Python (figure code), which is untrusted, and
* compiling model-written LaTeX, which needs a TeX distribution.

``LocalBackend`` runs both as subprocesses on this machine.  ``ModalBackend``
runs them in Modal Sandboxes, so nothing model-written executes locally and no
TeX install is needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ComputeError(RuntimeError):
    """The backend itself failed (as opposed to the job exiting non-zero)."""


@dataclass
class ExecResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    #: Requested output files that existed after the run, keyed by relative path.
    files: dict[str, bytes] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass
class CompileResult:
    ok: bool
    pdf_path: Path | None = None
    #: LaTeX error messages (``! ...`` blocks with their ``l.<n>`` context).
    errors: list[str] = field(default_factory=list)
    #: Undefined citations / references and other warnings worth fixing.
    warnings: list[str] = field(default_factory=list)
    #: Tail of the combined tool output, for humans and for repair prompts.
    log_tail: str = ""
    timed_out: bool = False

    def feedback(self, max_chars: int = 6000) -> str:
        """Compact description of what went wrong, suitable for a repair prompt."""
        lines: list[str] = []
        if self.timed_out:
            lines.append("Compilation timed out.")
        if self.errors:
            lines.append("LaTeX errors:")
            lines.extend(f"- {e}" for e in self.errors[:40])
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {w}" for w in self.warnings[:40])
        if not self.ok and not self.errors and self.log_tail:
            lines.append("Log tail:")
            lines.append(self.log_tail[-2500:])
        text = "\n".join(lines)
        return text[:max_chars]


def refuse_unsafe_document(workdir: str | Path, tex_name: str) -> CompileResult | None:
    """A failed result for a document that must not be compiled, else ``None``.

    Every backend calls this before running TeX.  TeX's own file-access limits do
    not cover every primitive (pdfTeX can embed any readable file in the PDF), so
    a document that uses file-reading, file-writing or code-running constructs is
    turned down and the reasons are returned as errors for the author to fix.
    """
    from paper_orchestra.utils.latex import unsafe_constructs

    source = Path(workdir) / f"{tex_name}.tex"
    try:
        problems = unsafe_constructs(source.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    if not problems:
        return None
    return CompileResult(
        ok=False,
        errors=problems,
        log_tail="The document was not compiled because it uses constructs that are not allowed.",
    )


class ComputeBackend(ABC):
    name: str = ""

    @abstractmethod
    def run_python(
        self,
        code: str,
        *,
        inputs: Mapping[str, bytes] | None = None,
        outputs: Sequence[str] = (),
        timeout: float = 120.0,
    ) -> ExecResult:
        """Run ``code`` as a script in a fresh, empty working directory.

        ``inputs`` are written into that directory first.  Each relative path in
        ``outputs`` that exists afterwards is returned in ``ExecResult.files``.
        The code must be treated as untrusted: no credentials in its
        environment, and no network where the backend can enforce it.
        """

    @abstractmethod
    def compile_latex(
        self,
        workdir: str | Path,
        tex_name: str,
        *,
        output_pdf: str | Path | None = None,
        timeout: float = 240.0,
    ) -> CompileResult:
        """Compile ``<workdir>/<tex_name>.tex`` (pdflatex, bibtex, pdflatex x2).

        Shell escape is always disabled.  On success the PDF is at
        ``output_pdf`` (default ``<workdir>/<tex_name>.pdf``).  A PDF produced
        despite recoverable LaTeX errors still counts as ``ok`` with the errors
        listed, matching what a human author would get from ``nonstopmode``.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release remote resources.  Safe to call more than once."""

    def describe(self) -> dict[str, Any]:
        """Non-secret diagnostics for ``paper-orchestra doctor``."""
        return {"backend": self.name}

    def __enter__(self) -> ComputeBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
