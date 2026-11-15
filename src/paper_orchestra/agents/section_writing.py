# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This agent takes in a latex file with partially filled sections and fills in the missing sections.

Input (all read from the run directory, see ``RunPaths``):
- outline.json: the outline of the paper, including section headings and key points to cover in each section.
- template.tex: a LaTeX file with some sections (intro & related work) filled and others missing.
- the idea file: a detailed summary of the methodology, core contributions, and theoretical framework.
- the experimental log: experimental results, including raw data points, ablation studies, and performance metrics.
- guidelines.md: the submission guidelines for the target conference.
- citation_map.json: the reference library collected by the literature agent.
- figures/info.json: the figures to include, each with a file name and a caption.

Output:
- raw_draft_paper.tex: the completed LaTeX file with all sections filled in.

The returned document is checked statically and compiled on a scratch copy of the
LaTeX workspace.  Problems go back to the writer as the next turn of the same
conversation until the document builds cleanly or the repair rounds are used up;
the best version that compiles is kept.
"""

from __future__ import annotations

import dataclasses
import difflib
import logging
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_orchestra.compute.base import CompileResult, ComputeBackend
from paper_orchestra.context import RunContext
from paper_orchestra.llm.base import Image, LLMError, LLMParseError, Part, Text
from paper_orchestra.llm.parsing import parse_latex
from paper_orchestra.prompts.section_writing import (
    NO_LATEX_IN_REPLY,
    SECTION_WRITING_SYSTEM_PROMPT,
    repair_prompt,
    section_writing_user_prompt,
)
from paper_orchestra.utils.common import read_json, read_text, write_text
from paper_orchestra.utils.latex import (
    GRAPHICS_EXTENSIONS,
    cite_keys,
    parse_latex_log,
    resolve_graphic,
    static_check,
)
from paper_orchestra.utils.pdf import pdf_page_images

log = logging.getLogger(__name__)

AGENT = "SectionWritingAgent"
STAGE = "section_writing"

DRAFT_PAGE = "wiki/drafts/raw_draft_paper.md"

_REFERENCE_WARNINGS = ("undefined citation", "undefined reference")
_BLOCKING_WARNINGS = (*_REFERENCE_WARNINGS, "file not found")
_BUILD_SUFFIXES = (".aux", ".bbl", ".blg", ".log", ".out", ".toc", ".pdf")
_BUILD_PRODUCTS = shutil.ignore_patterns(
    "*.aux", "*.blg", "*.log", "*.out", "*.toc", "*.fls", "*.fdb_latexmk", "*.synctex.gz"
)
_NO_PDF = "LaTeX did not produce a PDF"
NO_LATEX_PROBLEM = "the reply contained no LaTeX document"


class SectionWritingError(RuntimeError):
    """The writer never returned a document that compiles."""


@dataclass(frozen=True)
class DraftCheck:
    """What the static checks and a real build say about one LaTeX document."""

    tex: str
    static_problems: tuple[str, ...]
    compile: CompileResult

    @property
    def compiles(self) -> bool:
        return self.compile.ok

    @property
    def pdf(self) -> Path | None:
        return self.compile.pdf_path if self.compile.ok else None

    @property
    def problems(self) -> list[str]:
        """Everything a repair turn has to fix (empty: the document is clean)."""
        found = list(self.static_problems)
        if not self.compile.ok:
            found.append(_NO_PDF + (" (the build timed out)" if self.compile.timed_out else ""))
        found.extend(self.compile.errors)
        found.extend(w for w in self.compile.warnings if w.startswith(_BLOCKING_WARNINGS))
        return found

    @property
    def clean(self) -> bool:
        return not self.problems

    def feedback(self) -> str:
        """The problems, worded for the model that has to fix them."""
        lines: list[str] = []
        if self.static_problems:
            lines.append("Problems found by checking the source:")
            lines.extend(f"- {problem}" for problem in self.static_problems)
        if not self.compile.ok:
            lines.append(f"{_NO_PDF}.")
        build = self.compile.feedback()
        if build:
            lines.append(build)
        return "\n".join(lines)


def _settled(result: CompileResult, workdir: Path, tex_name: str, tex: str) -> CompileResult:
    """Drop findings that describe an intermediate build step, not the finished document.

    The backends parse the output of all four build steps together.  On the first
    pdflatex pass every citation and cross-reference is still undefined, so a correct
    document comes back with a warning for each of them; only the log of the last pass
    says which ones remain.  BibTeX also reports "no \\citation commands" as an error
    for a document that cites nothing, which is not something a rewrite can fix.
    """
    warnings = list(result.warnings)
    final_log = workdir / f"{tex_name}.log"
    if not result.ok:
        # Nothing can resolve before the document builds: the errors are what needs fixing.
        warnings = [w for w in warnings if not w.startswith(_REFERENCE_WARNINGS)]
    elif final_log.is_file():
        _, final = parse_latex_log(final_log.read_text(encoding="utf-8", errors="replace"))
        warnings = [w for w in warnings if not w.startswith(_REFERENCE_WARNINGS) or w in final]
    errors = list(result.errors)
    if not cite_keys(tex):
        errors = [e for e in errors if "I found no \\citation commands" not in e]
    return dataclasses.replace(result, errors=errors, warnings=warnings)


def check_latex(
    compute: ComputeBackend, workdir: Path, tex_name: str, tex: str, *, timeout: float
) -> DraftCheck:
    """Write ``tex`` to ``<workdir>/<tex_name>.tex``, check it statically and build it.

    Build products of an earlier attempt under the same name are removed first: a
    corrupt ``.aux`` file can break a build whose source is fine, and a stale PDF
    must not pass for the output of a build that failed.
    """
    for suffix in _BUILD_SUFFIXES:
        (workdir / f"{tex_name}{suffix}").unlink(missing_ok=True)
    write_text(workdir / f"{tex_name}.tex", tex)
    static = static_check(tex, workdir=workdir)
    result = compute.compile_latex(workdir, tex_name, timeout=timeout)
    return DraftCheck(
        tex=tex,
        static_problems=tuple(static),
        compile=_settled(result, workdir, tex_name, tex),
    )


def copy_workspace(source: Path, destination: Path) -> None:
    """Copy a LaTeX workspace without the build products of earlier compiles."""
    if not source.is_dir():
        raise FileNotFoundError(f"LaTeX workspace not found: {source}")
    shutil.copytree(source, destination, ignore=_BUILD_PRODUCTS, dirs_exist_ok=True)


def source_diff(before: str, after: str, *, max_lines: int = 60, max_width: int = 240) -> str:
    """The changed lines between two versions of a document, clipped for a note."""
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
    # The first two lines are the file header; "@@" lines only locate the hunks.
    changed = [line[:max_width] for line in list(diff)[2:] if not line.startswith("@@")]
    if len(changed) > max_lines:
        changed = [*changed[:max_lines], f"... ({len(changed) - max_lines} more changed lines)"]
    return "\n".join(changed)


def repair_lesson(
    what: str, problems: Sequence[str], before_tex: str | None, after: DraftCheck | None
) -> str:
    """Text of a ``lesson`` note: a build failure, what was changed, and how it ended."""
    if after is None:
        verdict = "the model's reply contained no LaTeX document"
    else:
        fixed = [problem for problem in problems if problem not in after.problems]
        verdict = f"{len(fixed)} of {len(problems)} reported problems were fixed" + (
            " and the document now builds cleanly" if after.clean else ""
        )
    lines = [f"LaTeX repair of {what}: {verdict}.", "", "Problems reported to the model:"]
    lines.extend(f"- {problem}" for problem in problems)
    if after is not None:
        if before_tex is not None:
            diff = source_diff(before_tex, after.tex)
            if diff:
                lines += ["", "What the fix changed in the source:", "```diff", diff, "```"]
        if after.problems:
            lines += ["", "Still wrong after the fix:"]
            lines.extend(f"- {problem}" for problem in after.problems)
    return "\n".join(lines)


def format_citation_library(citation_map: dict[str, Any]) -> str:
    """Every collected paper with its key, title, authors, year and abstract."""
    if not citation_map:
        return "No citation data provided."
    entries = ["Reference library (use these keys with \\cite{}):", ""]
    for key, data in citation_map.items():
        data = data if isinstance(data, dict) else {}
        authors_list = [str(author) for author in data.get("authors") or []]
        authors = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
        entries += [
            f"--- Key: {key} ---",
            f"Title: {data.get('title') or 'Unknown Title'}",
            f"Authors: {authors or 'Unknown'} ({data.get('year') or 'N/A'})",
            f"Abstract: {data.get('abstract') or 'No abstract available.'}",
            "",
        ]
    return "\n".join(entries)


@dataclass(frozen=True)
class _Figure:
    path: Path
    #: The path to use in ``\includegraphics``, relative to the LaTeX workspace.
    tex_path: str
    caption: str


@dataclass(frozen=True)
class _Attempt:
    round: int
    #: ``None`` when the reply contained no LaTeX document.
    check: DraftCheck | None

    @property
    def problems(self) -> list[str]:
        return self.check.problems if self.check is not None else [NO_LATEX_PROBLEM]


class SectionWritingAgent:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    # -- inputs ----------------------------------------------------------------

    def _figure_entries(self) -> list[dict[str, Any]]:
        """``figures/info.json``, or the files of the figures directory when it is absent."""
        paths = self.ctx.paths
        if paths.figures_info.is_file():
            try:
                entries = read_json(paths.figures_info)
            except ValueError as exc:
                raise ValueError(f"{paths.figures_info} is not valid JSON: {exc}") from exc
            if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                raise ValueError(
                    f'{paths.figures_info} must be a JSON list of {{"name", "caption"}} objects'
                )
            return entries
        files = (
            sorted(
                p for p in paths.figures_dir.iterdir() if p.suffix.lower() in GRAPHICS_EXTENSIONS
            )
            if paths.figures_dir.is_dir()
            else []
        )
        if files:
            log.warning(
                "%s does not exist; using the %d figure file(s) in %s without captions.",
                paths.figures_info,
                len(files),
                paths.figures_dir,
            )
        return [{"name": file.name} for file in files]

    def _figures(self) -> list[_Figure]:
        """The listed figures that exist in a format pdflatex can include."""
        paths = self.ctx.paths
        figures: list[_Figure] = []
        for index, entry in enumerate(self._figure_entries(), start=1):
            name = str(entry.get("name") or "")
            found = resolve_graphic(paths.figures_dir, name) if name else None
            if found is None or found.suffix.lower() not in GRAPHICS_EXTENSIONS:
                log.warning(
                    "Figure %d (%r) is left out of the paper: %s.",
                    index,
                    name,
                    f"no such file in {paths.figures_dir}"
                    if found is None
                    else f"pdflatex cannot include {found.suffix} files",
                )
                continue
            figures.append(
                _Figure(
                    path=found,
                    tex_path=found.relative_to(paths.latex_dir).as_posix(),
                    caption=str(entry.get("caption") or "No caption provided."),
                )
            )
        return figures

    @staticmethod
    def _figures_listing(figures: Sequence[_Figure]) -> str:
        if not figures:
            return "No figures provided."
        lines: list[str] = []
        for index, figure in enumerate(figures, start=1):
            lines += [
                f"Item {index}:",
                f"  - File: {figure.tex_path}",
                f"  - Caption: {figure.caption}",
            ]
        return "\n".join(lines)

    @staticmethod
    def _figure_parts(figures: Sequence[_Figure]) -> list[Part]:
        """The figures as images, each introduced by the path used to include it.

        Models take raster images only, so a PDF figure is shown as a rendering of its
        first page.
        """
        parts: list[Part] = []
        for figure in figures:
            try:
                if figure.path.suffix.lower() == ".pdf":
                    image = Image.from_bytes(pdf_page_images(figure.path, max_pages=1)[0])
                else:
                    image = Image.from_path(figure.path)
            except (OSError, ValueError, RuntimeError, IndexError) as exc:
                log.warning(
                    "Figure %s is not shown to the writer (%s: %s).",
                    figure.path,
                    type(exc).__name__,
                    exc,
                )
                continue
            parts += [Text(f"Image of {figure.tex_path}:"), image]
        return parts

    def _first_turn(self) -> list[Part]:
        paths = self.ctx.paths
        figures = self._figures()
        prompt = section_writing_user_prompt(
            outline=read_text(paths.outline),
            citation_library=format_citation_library(read_json(paths.citation_map)),
            idea=read_text(paths.idea),
            experimental_log=read_text(paths.experimental_log),
            figures=self._figures_listing(figures),
            guidelines=read_text(paths.guidelines),
            template=read_text(paths.template_tex),
        )
        return [*self._figure_parts(figures), Text(prompt)]

    # -- checking --------------------------------------------------------------

    def _check(self, tex: str, round_no: int) -> DraftCheck:
        """Check and build ``tex`` in a throw-away copy of the LaTeX workspace."""
        paths = self.ctx.paths
        with tempfile.TemporaryDirectory(prefix="paper-orchestra-draft-") as tmp:
            scratch = Path(tmp) / paths.latex_dir.name
            copy_workspace(paths.latex_dir, scratch)
            check = check_latex(
                self.ctx.compute,
                scratch,
                paths.raw_draft_tex.stem,
                tex,
                timeout=self.ctx.cfg.latex_timeout,
            )
        # The PDF went away with the scratch directory.
        check = dataclasses.replace(
            check, compile=dataclasses.replace(check.compile, pdf_path=None)
        )
        self.ctx.harness.record(
            "compile",
            stage=STAGE,
            agent=AGENT,
            document=paths.raw_draft_tex.name,
            round=round_no,
            ok=check.compiles,
            clean=check.clean,
            static_problems=list(check.static_problems),
            errors=check.compile.errors,
            warnings=check.compile.warnings,
        )
        return check

    # -- main ------------------------------------------------------------------

    def run(self) -> Path:
        cfg, paths, harness = self.ctx.cfg, self.ctx.paths, self.ctx.harness
        log.info(
            "Agent %s starting: writing the missing sections with %s.", AGENT, cfg.models.writer
        )
        message: Sequence[Part] | str = self._first_turn()
        conversation = harness.open(
            agent=AGENT,
            stage=STAGE,
            model=cfg.models.writer,
            system=SECTION_WRITING_SYSTEM_PROMPT,
            temperature=0.7,
        )

        attempts: list[_Attempt] = []
        for round_no in range(cfg.max_repair_rounds + 1):
            try:
                tex = conversation.send(message, parser=parse_latex).parsed
            except LLMParseError:
                attempt = _Attempt(round_no, None)
            except LLMError as exc:
                if not any(a.check is not None and a.check.compiles for a in attempts):
                    raise
                # A draft that compiles is worth more than a repair that could not be asked for.
                log.warning(
                    "%s: repair round %d failed (%s: %s); keeping the best draft so far.",
                    AGENT,
                    round_no,
                    type(exc).__name__,
                    exc,
                )
                break
            else:
                attempt = _Attempt(round_no, self._check(tex, round_no))
            if attempts:
                self._note_repair(attempts[-1], attempt)
            attempts.append(attempt)
            if not attempt.problems:
                break
            log.warning(
                "%s: draft %d has %d problem(s): %s",
                AGENT,
                round_no,
                len(attempt.problems),
                "; ".join(attempt.problems)[:600],
            )
            message = (
                repair_prompt(attempt.check.feedback())
                if attempt.check is not None
                else NO_LATEX_IN_REPLY
            )

        compiled = [a for a in attempts if a.check is not None and a.check.compiles]
        if not compiled:
            last = attempts[-1]
            harness.note(
                f"Section writing failed: none of {len(attempts)} drafts compiled.\n\n"
                "Problems of the last draft:\n" + "\n".join(f"- {p}" for p in last.problems),
                stage=STAGE,
                agent=AGENT,
                kind="lesson",
            )
            raise SectionWritingError(
                f"none of the {len(attempts)} drafts written by {cfg.models.writer} compiled; "
                f"problems of the last one: {'; '.join(last.problems)[:1500]}"
            )
        # Clean beats everything, then the fewest problems, then the latest repair.
        best = max(compiled, key=lambda a: (not a.problems, -len(a.problems), a.round))
        assert best.check is not None
        write_text(paths.raw_draft_tex, best.check.tex)
        self._write_draft_page(best, len(attempts))
        if best.problems:
            log.warning(
                "%s: keeping draft %d, which compiles but still has %d problem(s).",
                AGENT,
                best.round,
                len(best.problems),
            )
        log.info("%s: draft saved to %s.", AGENT, paths.raw_draft_tex)
        return paths.raw_draft_tex

    # -- memory ----------------------------------------------------------------

    def _note_repair(self, before: _Attempt, after: _Attempt) -> None:
        self.ctx.harness.note(
            repair_lesson(
                f"the section writing draft (repair round {after.round})",
                before.problems,
                before.check.tex if before.check is not None else None,
                after.check,
            ),
            stage=STAGE,
            agent=AGENT,
            kind="lesson",
        )

    def _write_draft_page(self, best: _Attempt, attempts: int) -> None:
        paths = self.ctx.paths
        relative = paths.raw_draft_tex.relative_to(paths.base_dir).as_posix()
        status = "builds cleanly" if not best.problems else f"{len(best.problems)} open problem(s)"
        lines = [
            "# Raw draft",
            "",
            f"The complete first draft of the paper is `{relative}` in the run directory "
            "(the LaTeX itself is not kept in memory).",
            "",
            f"- Written by: {AGENT} with `{self.ctx.cfg.models.writer}`",
            f"- Drafts written: {attempts} (the one kept is draft {best.round})",
            f"- Build status: {status}",
        ]
        if best.problems:
            lines += ["", "## Open problems", ""]
            lines.extend(f"- {problem}" for problem in best.problems)
        self.ctx.harness.write_page(
            DRAFT_PAGE,
            "\n".join(lines),
            {
                "title": "Raw draft",
                "summary": f"First full draft at {relative}: {status} after {attempts} draft(s)",
                "kind": "draft",
            },
        )
