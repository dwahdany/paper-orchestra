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

"""Content refinement: review, revise, accept or reject; then a formatting pass.

The agent works in ``RunPaths.refinement_dir``, a copy of the LaTeX workspace:

1. Compile the draft (``initial_draft``), repairing it first if it does not build,
   and have it peer-reviewed.
2. Up to ``max_reflections`` times: revise the paper against the current review,
   build and review the revision, and keep it only if the scores did not get worse.
3. Up to ``max_formatting_rounds`` times: review the rendered pages for layout
   problems and apply a formatting-only fix.
4. Save the surviving version as ``final_refined_paper.tex`` / ``.pdf``.

Once the initial draft compiles there is always a valid version to fall back on, so
a failure in a later step is logged and recorded and the run continues with the last
version that was accepted.
"""

from __future__ import annotations

import functools
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from paper_orchestra.agents.section_writing import (
    DraftCheck,
    check_latex,
    copy_workspace,
    repair_lesson,
)
from paper_orchestra.context import RunContext
from paper_orchestra.eval.agent_review import perform_review_agentreview
from paper_orchestra.llm.base import Image, LLMParseError, Part, Pdf, Text
from paper_orchestra.llm.parsing import parse_json_and_latex, parse_latex
from paper_orchestra.memory.harness import HarnessConversation, MemoryPolicy
from paper_orchestra.prompts.format import (
    FORMAT_FIX_SYSTEM_PROMPT,
    FORMAT_REVIEW_SYSTEM_PROMPT,
    format_fix_user_prompt,
    format_review_user_prompt,
)
from paper_orchestra.prompts.refinement import (
    CONTENT_REFINEMENT_SYSTEM_PROMPT,
    LATEX_REPAIR_SYSTEM_PROMPT,
    REFINEMENT_DREAM_FOCUS,
    latex_repair_prompt,
    revision_repair_prompt,
    revision_user_prompt,
)
from paper_orchestra.prompts.section_writing import NO_LATEX_IN_REPLY, repair_prompt
from paper_orchestra.utils.common import read_text, write_json, write_text
from paper_orchestra.utils.latex import cite_keys
from paper_orchestra.utils.pdf import load_paper, page_count, pdf_to_grid_images

log = logging.getLogger(__name__)

AGENT = "ContentRefinementAgent"
STAGE = "refinement"

CONTENT_WORKLOG = "content_refinement_worklog.json"
FORMAT_WORKLOG = "format_refinement_worklog.json"

SCORE_AXES = (
    "Originality",
    "Quality",
    "Clarity",
    "Significance",
    "Soundness",
    "Presentation",
    "Contribution",
    "Overall",
    "Confidence",
)
_NOT_COMPARED = ("Overall", "Confidence")
_PAGES_PER_SHEET = 4
#: Longest side of a page sheet sent to the formatting reviewer.  Larger images are
#: downscaled by the vision models anyway; this keeps a 2x2 sheet of pages legible.
_SHEET_MAX_SIDE = 2576
_NO_LATEX = "the reply contained no LaTeX document"


class RefinementError(RuntimeError):
    """The refinement stage has no compilable paper to work with."""


# -- formatting review schema -------------------------------------------------


def _is_issue(text: str) -> bool:
    return text.strip().lower().rstrip(".") not in ("", "none", "n/a")


class FigureTableFinding(BaseModel):
    element: str = Field(description='The figure or table, for example "Figure 1" or "Table 2".')
    detected_issue: str = Field(description='The formatting problem, or "None" if there is none.')
    suggested_fix: str = Field(description='How to fix it in the LaTeX source, or "None".')


class LayoutIssue(BaseModel):
    page: int = Field(description="Page number, starting from 1.")
    element: str = Field(description="'Section x, Paragraph y' or another specific location.")
    detected_issue: str = Field(description="The layout or text problem.")
    suggested_fix: str = Field(description="How to fix it.")


class FormattingReview(BaseModel):
    figure_and_tables: list[FigureTableFinding] = Field(
        description="One entry for every figure and every table of the paper."
    )
    other_issues: list[LayoutIssue] = Field(description="Any other layout or text problem.")

    def has_issues(self) -> bool:
        findings = [*self.figure_and_tables, *self.other_issues]
        return any(_is_issue(finding.detected_issue) for finding in findings)


# -- acceptance logic ---------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """How a revision's review compares with the review of the version it revised."""

    outcome: str
    accepted: bool
    scores_before: dict[str, float]
    scores_after: dict[str, float]
    deltas: dict[str, float]
    total_gain: float
    total_drop: float

    def movement(self) -> str:
        """One line: the overall score and every sub-axis that moved."""
        before, after = self.scores_before["Overall"], self.scores_after["Overall"]
        moved = ", ".join(f"{axis} {delta:+g}" for axis, delta in self.deltas.items() if delta)
        return (
            f"Overall {before:g} -> {after:g}; sub-axis gain {self.total_gain:g}, "
            f"drop {self.total_drop:g} ({moved or 'no sub-axis moved'})"
        )


def extract_scores(review: Mapping[str, Any]) -> dict[str, float]:
    """The numeric scores of a review; a missing or malformed score counts as 0."""
    scores: dict[str, float] = {}
    for axis in SCORE_AXES:
        try:
            scores[axis] = float(review.get(axis, 0))
        except (TypeError, ValueError):
            scores[axis] = 0.0
    return scores


def judge_revision(old_review: Mapping[str, Any], new_review: Mapping[str, Any]) -> Verdict:
    """Accept a revision unless the reviewers like it less than the version before.

    A higher overall score is accepted and a lower one rejected.  When the overall
    score is unchanged, the sub-axes decide: the revision is accepted if their total
    gain is at least their total drop.
    """
    before, after = extract_scores(old_review), extract_scores(new_review)
    deltas = {axis: after[axis] - before[axis] for axis in SCORE_AXES if axis not in _NOT_COMPARED}
    total_gain = sum(delta for delta in deltas.values() if delta > 0)
    total_drop = sum(-delta for delta in deltas.values() if delta < 0)
    if after["Overall"] > before["Overall"]:
        outcome = "ACCEPTED_SCORE_INCREASE"
    elif after["Overall"] < before["Overall"]:
        outcome = "REJECTED_SCORE_DECREASE"
    elif total_drop > total_gain:
        outcome = "REJECTED_DEGRADATION"
    else:
        outcome = "ACCEPTED_NEUTRAL_IMPROVEMENT"
    return Verdict(
        outcome=outcome,
        accepted=outcome.startswith("ACCEPTED"),
        scores_before=before,
        scores_after=after,
        deltas=deltas,
        total_gain=total_gain,
        total_drop=total_drop,
    )


# -- state --------------------------------------------------------------------


@dataclass(frozen=True)
class _Version:
    """A version of the paper that compiled: the fallback for everything after it."""

    name: str
    tex: str
    pdf: Path
    static_problems: frozenset[str]


@dataclass(frozen=True)
class _Reflection:
    entry: dict[str, Any]
    #: The revision and its review when it was accepted.
    version: _Version | None = None
    review: dict[str, Any] | None = None
    #: The scores went down: stop revising.
    stop: bool = False


def _plan_lines(plan: Any) -> list[str]:
    """The reviser's own worklog as note text."""
    if not isinstance(plan, dict) or not plan:
        return ["- (the revision came without a worklog)"]
    lines: list[str] = []
    for key, value in plan.items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"- {key}: {value}")
    return lines


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class ContentRefinementAgent:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.workdir = ctx.paths.refinement_dir
        self.model = ctx.cfg.models.reflection
        self.worklogs: dict[str, dict[str, Any]] = {}
        self.format_worklogs: dict[str, dict[str, Any]] = {}
        self._guidelines = ""
        self._experimental_log = ""
        self._citation_map = ""
        self._reviews: dict[str, dict[str, Any]] = {}
        #: Per reviewed version: what happened to it and to the revisions answering it,
        #: as ``(short, full)`` sentences.
        self._outcomes: dict[str, list[tuple[str, str]]] = {}

    # -- main ------------------------------------------------------------------

    def run(self, tex_path: Path) -> Path:
        """Refine the draft at ``tex_path``; return the final PDF in the refinement directory."""
        paths = self.ctx.paths
        tex = read_text(tex_path)
        self._guidelines = read_text(paths.guidelines)
        self._experimental_log = read_text(paths.experimental_log)
        self._citation_map = read_text(paths.citation_map)
        self._prepare_workspace()
        log.info("%s: working in %s with %s.", AGENT, self.workdir, self.model)

        current = self._initial_version(tex)
        self._screenshots(current.pdf, self._shots("content_refinement", "v0"))
        current = self._content_loop(current)
        current = self._formatting_pass(current)
        return self._finalize(current)

    def _prepare_workspace(self) -> None:
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        copy_workspace(self.ctx.paths.latex_dir, self.workdir)
        (self.workdir / "peer_reviews").mkdir(parents=True, exist_ok=True)
        for phase in ("content_refinement", "format_refinement"):
            (self.workdir / "pdf_screenshots" / phase).mkdir(parents=True, exist_ok=True)

    def _shots(self, phase: str, label: str) -> Path:
        return self.workdir / "pdf_screenshots" / phase / label

    # -- building and reviewing ------------------------------------------------

    def _build(self, name: str, tex: str) -> DraftCheck:
        check = check_latex(
            self.ctx.compute, self.workdir, name, tex, timeout=self.ctx.cfg.latex_timeout
        )
        self.ctx.harness.record(
            "compile",
            stage=STAGE,
            agent=AGENT,
            document=f"{name}.tex",
            ok=check.compiles,
            clean=check.clean,
            static_problems=list(check.static_problems),
            errors=check.compile.errors,
            warnings=check.compile.warnings,
        )
        return check

    @staticmethod
    def _version(name: str, check: DraftCheck) -> _Version:
        if check.pdf is None or not check.pdf.is_file():
            raise RefinementError(f"the build of {name}.tex reported success but left no PDF")
        return _Version(name, check.tex, check.pdf, frozenset(check.static_problems))

    def _screenshots(self, pdf: Path, out_dir: Path, *, max_side: int | None = None) -> list[Path]:
        """Page sheets of ``pdf`` for the frontend (and the formatting reviewer)."""
        try:
            return [Path(p) for p in pdf_to_grid_images(pdf, out_dir, max_side=max_side)]
        except Exception as exc:
            log.warning(
                "%s: could not render page images of %s (%s).", AGENT, pdf.name, _error_text(exc)
            )
            return []

    def _review(self, pdf: Path, label: str) -> dict[str, Any]:
        """Peer-review ``pdf``.  The reviewers get no memory: they judge the paper alone."""
        review = perform_review_agentreview(
            load_paper(pdf),
            self.model,
            generate=functools.partial(
                self.ctx.harness.generate, agent=AGENT, stage=STAGE, memory=MemoryPolicy.none()
            ),
            num_review_ensemble=self.ctx.cfg.review_ensemble,
            label=f"{STAGE}.review",
        )
        write_json(self.workdir / "peer_reviews" / f"review_{label}.json", review)
        log.info("%s: review of %s: Overall %s/10.", AGENT, label, review.get("Overall"))
        return review

    def _failed(self, what: str, exc: Exception, kept: _Version) -> None:
        """Log and record a failure that the run survives by keeping ``kept``."""
        log.error(
            "%s: %s failed (%s); continuing with %s.",
            AGENT,
            what,
            _error_text(exc),
            kept.name,
            exc_info=exc,
        )
        self.ctx.harness.record(
            "error", stage=STAGE, agent=AGENT, error=f"{what}: {_error_text(exc)}", kept=kept.name
        )
        self.ctx.harness.note(
            f"Refinement: {what} failed with {_error_text(exc)}. "
            f"The run continued with {kept.name}.",
            stage=STAGE,
            agent=AGENT,
        )

    # -- initial draft ---------------------------------------------------------

    def _initial_version(self, tex: str) -> _Version:
        name = "initial_draft"
        check = self._build(name, tex)
        if not check.compiles:
            check = self._repair_initial(name, check)
        if check.static_problems:
            log.warning(
                "%s: the initial draft compiles but has open problems: %s",
                AGENT,
                "; ".join(check.static_problems),
            )
        return self._version(name, check)

    def _repair_initial(self, name: str, check: DraftCheck) -> DraftCheck:
        """Repair turns for a draft that does not build; raises when they do not help."""
        rounds = self.ctx.cfg.max_repair_rounds
        log.warning(
            "%s: the initial draft does not compile (%s); trying to repair it.",
            AGENT,
            "; ".join(check.problems)[:600],
        )
        conversation = self.ctx.harness.open(
            agent=AGENT,
            stage=STAGE,
            model=self.model,
            system=LATEX_REPAIR_SYSTEM_PROMPT,
            label=f"{STAGE}.initial_repair",
        )
        message = latex_repair_prompt(tex=check.tex, feedback=check.feedback())
        for round_no in range(1, rounds + 1):
            try:
                tex = conversation.send(message, parser=parse_latex).parsed
            except LLMParseError:
                message = NO_LATEX_IN_REPLY
                continue
            repaired = self._build(name, tex)
            self.ctx.harness.note(
                repair_lesson(
                    f"the initial draft (repair round {round_no})",
                    check.problems,
                    check.tex,
                    repaired,
                ),
                stage=STAGE,
                agent=AGENT,
                kind="lesson",
            )
            if repaired.compiles:
                return repaired
            check = repaired
            message = repair_prompt(repaired.feedback())
        raise RefinementError(
            f"the initial draft does not compile and {rounds} repair round(s) with "
            f"{self.model} did not fix it; problems: {'; '.join(check.problems)[:1500]}"
        )

    # -- content loop ----------------------------------------------------------

    def _content_loop(self, current: _Version) -> _Version:
        cfg, harness = self.ctx.cfg, self.ctx.harness
        try:
            review = self._review(current.pdf, "v0")
            self._reviews["v0"] = review
            self._write_review_page("v0")
            harness.record(
                "review",
                stage=STAGE,
                agent=AGENT,
                version="v0",
                outcome="BASELINE",
                scores=extract_scores(review),
            )
        except Exception as exc:
            self._failed("the baseline review (the paper is not revised)", exc, current)
            write_json(self.workdir / CONTENT_WORKLOG, self.worklogs)
            return current

        base = "v0"
        for round_no in range(1, cfg.max_reflections + 1):
            label = f"v{round_no}"
            log.info("%s: refinement iteration %d of %d.", AGENT, round_no, cfg.max_reflections)
            try:
                result = self._reflect(round_no, current, review, base)
            except Exception as exc:
                self.worklogs[label] = {
                    "round": round_no,
                    "outcome": "ERROR",
                    "error": _error_text(exc),
                }
                self._failed(f"refinement iteration {round_no}", exc, current)
                break
            self.worklogs[label] = result.entry
            write_json(self.workdir / CONTENT_WORKLOG, self.worklogs)
            log.info("%s: revision %s: %s.", AGENT, label, result.entry["outcome"])
            if result.version is not None and result.review is not None:
                current, review, base = result.version, result.review, label
            if result.stop:
                break
            if round_no < cfg.max_reflections:
                # Turn the outcomes so far into rules the next revision sees in its index.
                harness.dream(stage=STAGE, focus=REFINEMENT_DREAM_FOCUS)
        write_json(self.workdir / CONTENT_WORKLOG, self.worklogs)
        return current

    def _reflect(
        self, round_no: int, current: _Version, review: dict[str, Any], base: str
    ) -> _Reflection:
        """One iteration: revise, build, review, and judge against the current review."""
        label = f"v{round_no}"
        plan, check = self._revise(round_no, f"refined_paper_{label}", current, review)
        entry: dict[str, Any] = {"round": round_no, "agent_plan": plan}

        problems = self._build_problems(check, current)
        if check is None or problems:
            outcome = (
                "REJECTED_STATIC_CHECK"
                if check is not None and check.compiles
                else "REJECTED_COMPILE_FAILURE"
            )
            entry.update(outcome=outcome, problems=problems)
            self._remember_unbuildable(label, base, entry)
            return _Reflection(entry)

        version = self._version(f"refined_paper_{label}", check)
        self._screenshots(version.pdf, self._shots("content_refinement", label))
        new_review = self._review(version.pdf, label)
        verdict = judge_revision(review, new_review)
        entry.update(
            scores_before=verdict.scores_before,
            scores_after=verdict.scores_after,
            deltas=verdict.deltas,
            total_gain=verdict.total_gain,
            total_drop=verdict.total_drop,
            outcome=verdict.outcome,
        )
        self._remember_verdict(label, base, plan, new_review, verdict)
        if verdict.accepted:
            return _Reflection(entry, version, new_review)
        return _Reflection(entry, stop=True)

    @staticmethod
    def _build_problems(check: DraftCheck | None, current: _Version) -> list[str]:
        """Why a revision cannot be reviewed (empty: it can).

        Static problems the current version already has do not disqualify a revision:
        it is no worse for keeping them, and the scores decide the rest.
        """
        if check is None:
            return [_NO_LATEX]
        if not check.compiles:
            return check.problems
        return [p for p in check.static_problems if p not in current.static_problems]

    def _revise(
        self, round_no: int, name: str, current: _Version, review: dict[str, Any]
    ) -> tuple[Any, DraftCheck | None]:
        """Ask for a revision and build it; one repair turn if it does not hold up.

        Returns the reviser's worklog and the check of the last document it returned
        (``None`` if it never returned one).
        """
        prompt = revision_user_prompt(
            iteration=round_no,
            previous_log=json.dumps(self.worklogs, indent=4) if self.worklogs else "None",
            review=json.dumps(review, indent=2, ensure_ascii=False),
            score=extract_scores(review)["Overall"],
            guidelines=self._guidelines,
            experimental_log=self._experimental_log,
            citation_map=self._citation_map,
            tex=current.tex,
            known_problems=sorted(current.static_problems),
        )
        parts: list[Part] = [Text(prompt)]
        if self.ctx.client(self.model).capabilities.pdf_input:
            parts.insert(0, Pdf.from_path(current.pdf))
        conversation = self.ctx.harness.open(
            agent=AGENT,
            stage=STAGE,
            model=self.model,
            system=CONTENT_REFINEMENT_SYSTEM_PROMPT,
            label=f"{STAGE}.revise",
        )
        plan, check = self._ask_revision(conversation, parts, name)
        if not self._build_problems(check, current):
            return plan, check

        problems = check.problems if check is not None else [_NO_LATEX]
        log.warning("%s: %s needs a repair turn: %s", AGENT, name, "; ".join(problems)[:600])
        message = (
            revision_repair_prompt(check.feedback()) if check is not None else NO_LATEX_IN_REPLY
        )
        repaired_plan, repaired = self._ask_revision(conversation, message, name)
        self.ctx.harness.note(
            repair_lesson(
                f"revision {name}", problems, check.tex if check is not None else None, repaired
            ),
            stage=STAGE,
            agent=AGENT,
            kind="lesson",
        )
        return plan or repaired_plan, repaired if repaired is not None else check

    def _ask_revision(
        self, conversation: HarnessConversation, message: Sequence[Part] | str, name: str
    ) -> tuple[Any, DraftCheck | None]:
        try:
            parsed = conversation.send(message, parser=parse_json_and_latex).parsed
        except LLMParseError:
            log.warning("%s: the reply for %s contained no LaTeX document.", AGENT, name)
            return None, None
        return parsed["json"], self._build(name, parsed["latex"])

    # -- memory of the loop ----------------------------------------------------

    def _write_review_page(self, label: str) -> None:
        review = self._reviews[label]
        outcomes = self._outcomes.get(label, [])
        body = [
            f"# Review of {label}",
            "",
            f"Peer review of version {label} of the paper by `{self.model}` "
            f"({self.ctx.cfg.review_ensemble} reviewer(s)); also saved as "
            f"`peer_reviews/review_{label}.json` in the refinement directory.",
            "",
            "```json",
            json.dumps(review, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Outcome",
            "",
            *(f"- {full}" for _, full in outcomes),
        ]
        if not outcomes:
            body.append("- No revision has been judged against this review yet.")
        summary = f"Overall {review.get('Overall')}/10 ({review.get('Decision')})"
        self.ctx.harness.write_page(
            f"wiki/reviews/{label}.md",
            "\n".join(body),
            {
                "title": f"Review of {label}",
                "summary": "; ".join([summary, *(short for short, _ in outcomes)]),
                "kind": "review",
            },
        )

    def _add_outcome(self, label: str, short: str, full: str) -> None:
        """Attach an outcome to a reviewed version's page (``short`` goes into the index)."""
        self._outcomes.setdefault(label, []).append((short, full))
        if label in self._reviews:
            self._write_review_page(label)

    def _remember_verdict(
        self, label: str, base: str, plan: Any, review: dict[str, Any], verdict: Verdict
    ) -> None:
        harness = self.ctx.harness
        decided = "accepted" if verdict.accepted else "rejected"
        headline = f"{decided} ({verdict.outcome}) against {base}: {verdict.movement()}"
        self._reviews[label] = review
        self._add_outcome(label, f"{decided} against {base}", f"This revision was {headline}.")
        self._add_outcome(
            base,
            f"revision {label} {decided}",
            f"Revision {label}, written in response, was {headline}.",
        )
        harness.record(
            "review",
            stage=STAGE,
            agent=AGENT,
            version=label,
            compared_with=base,
            outcome=verdict.outcome,
            scores_before=verdict.scores_before,
            scores_after=verdict.scores_after,
            deltas=verdict.deltas,
            total_gain=verdict.total_gain,
            total_drop=verdict.total_drop,
        )
        harness.note(
            "\n".join(
                [
                    f"Revision {label} was {headline}.",
                    "",
                    "What the revision changed (its own worklog):",
                    *_plan_lines(plan),
                ]
            ),
            stage=STAGE,
            agent=AGENT,
            kind="lesson",
        )

    def _remember_unbuildable(self, label: str, base: str, entry: dict[str, Any]) -> None:
        harness = self.ctx.harness
        headline = f"rejected before review ({entry['outcome']})"
        self._add_outcome(
            base,
            f"revision {label} rejected before review",
            f"Revision {label}, written in response, was {headline}.",
        )
        harness.record(
            "review",
            stage=STAGE,
            agent=AGENT,
            version=label,
            compared_with=base,
            outcome=entry["outcome"],
            problems=entry["problems"],
        )
        harness.note(
            "\n".join(
                [
                    f"Revision {label} was {headline}: it did not hold up after its repair turn.",
                    "",
                    "Problems:",
                    *(f"- {problem}" for problem in entry["problems"]),
                    "",
                    "What the revision changed (its own worklog):",
                    *_plan_lines(entry["agent_plan"]),
                ]
            ),
            stage=STAGE,
            agent=AGENT,
            kind="lesson",
        )

    # -- formatting pass -------------------------------------------------------

    def _formatting_pass(self, current: _Version) -> _Version:
        cfg = self.ctx.cfg
        write_text(self.workdir / "format_candidate_v0.tex", current.tex)
        sheets = self._screenshots(
            current.pdf, self._shots("format_refinement", "v0"), max_side=_SHEET_MAX_SIDE
        )
        for round_no in range(1, cfg.max_formatting_rounds + 1):
            label = f"v{round_no}"
            if not sheets:
                log.warning(
                    "%s: no page images of %s; skipping the formatting pass.",
                    AGENT,
                    current.name,
                )
                break
            try:
                review = self._format_review(current.pdf, sheets)
                if not review.has_issues():
                    log.info("%s: the formatting review found nothing to fix.", AGENT)
                    break
                entry, accepted = self._format_fix(label, current, review)
            except Exception as exc:
                self.format_worklogs[label] = {"outcome": "ERROR", "error": _error_text(exc)}
                self._failed(f"formatting round {round_no}", exc, current)
                continue
            self.format_worklogs[label] = entry
            write_json(self.workdir / FORMAT_WORKLOG, self.format_worklogs)
            log.info("%s: formatting candidate %s: %s.", AGENT, label, entry["outcome"])
            if accepted is not None:
                current = accepted
                sheets = self._screenshots(
                    current.pdf, self._shots("format_refinement", label), max_side=_SHEET_MAX_SIDE
                )
        write_json(self.workdir / FORMAT_WORKLOG, self.format_worklogs)
        return current

    def _format_review(self, pdf: Path, sheets: Sequence[Path]) -> FormattingReview:
        pages = page_count(pdf)
        parts: list[Part] = []
        for index, sheet in enumerate(sheets):
            first = index * _PAGES_PER_SHEET + 1
            last = min(first + _PAGES_PER_SHEET - 1, pages)
            parts += [
                Text(f"Pages {first} to {last} (left to right, then top to bottom):"),
                Image.from_path(sheet),
            ]
        parts.append(Text(format_review_user_prompt(self._guidelines)))
        response = self.ctx.harness.generate(
            parts,
            agent=AGENT,
            stage=STAGE,
            model=self.model,
            system=FORMAT_REVIEW_SYSTEM_PROMPT,
            schema=FormattingReview,
            memory=MemoryPolicy.none(),
            label=f"{STAGE}.format_review",
        )
        return FormattingReview.model_validate(response.parsed)

    def _format_fix(
        self, label: str, current: _Version, review: FormattingReview
    ) -> tuple[dict[str, Any], _Version | None]:
        """Apply a formatting-only fix; the candidate is kept only if it is safe."""
        name = f"formatted_candidate_{label}"
        feedback = review.model_dump()
        response = self.ctx.harness.generate(
            format_fix_user_prompt(
                feedback=json.dumps(feedback, indent=2, ensure_ascii=False),
                guidelines=self._guidelines,
                tex=current.tex,
            ),
            agent=AGENT,
            stage=STAGE,
            model=self.model,
            system=FORMAT_FIX_SYSTEM_PROMPT,
            parser=parse_latex,
            memory=MemoryPolicy.readonly(),
            label=f"{STAGE}.format_fix",
        )
        tex: str = response.parsed
        entry: dict[str, Any] = {"outcome": "", "formatting_feedback": feedback}

        before, after = cite_keys(current.tex), cite_keys(tex)
        problems: list[str] = []
        accepted: _Version | None = None
        if before != after:
            # A formatting fix has no business touching citations: do not even build it.
            write_text(self.workdir / f"{name}.tex", tex)
            entry["outcome"] = "REJECTED_CITATIONS_CHANGED"
            problems = [
                f"citation keys removed: {', '.join(sorted(before - after)) or 'none'}",
                f"citation keys added: {', '.join(sorted(after - before)) or 'none'}",
            ]
        else:
            check = self._build(name, tex)
            if check.compiles:
                entry["outcome"] = "ACCEPTED_COMPILE_SUCCESS"
                accepted = self._version(name, check)
            else:
                entry["outcome"] = "REJECTED_COMPILE_FAILURE"
                problems = check.problems

        self.ctx.harness.record(
            "format", stage=STAGE, agent=AGENT, version=label, outcome=entry["outcome"]
        )
        if problems:
            entry["problems"] = problems
            self.ctx.harness.note(
                "\n".join(
                    [
                        f"Formatting candidate {label} was rejected ({entry['outcome']}); "
                        f"{current.name} was kept.",
                        "",
                        "Problems:",
                        *(f"- {problem}" for problem in problems),
                    ]
                ),
                stage=STAGE,
                agent=AGENT,
                kind="lesson",
            )
        return entry, accepted

    # -- final output ----------------------------------------------------------

    def _finalize(self, current: _Version) -> Path:
        final_tex = self.workdir / "final_refined_paper.tex"
        final_pdf = self.workdir / "final_refined_paper.pdf"
        write_text(final_tex, current.tex)
        shutil.copyfile(current.pdf, final_pdf)

        relative = final_pdf.relative_to(self.ctx.paths.base_dir).as_posix()
        revisions = [
            f"- {label}: {entry['outcome']}" for label, entry in self.worklogs.items()
        ] or ["- (no revision was judged)"]
        formatting = [
            f"- {label}: {entry['outcome']}" for label, entry in self.format_worklogs.items()
        ] or ["- (no formatting fix was applied)"]
        self.ctx.harness.write_page(
            "wiki/drafts/final_refined_paper.md",
            "\n".join(
                [
                    "# Final refined paper",
                    "",
                    f"The refined paper is `{relative}` (source next to it as "
                    f"`{final_tex.name}`). It is a copy of `{current.name}`.",
                    "",
                    "## Content revisions",
                    "",
                    *revisions,
                    "",
                    "## Formatting candidates",
                    "",
                    *formatting,
                ]
            ),
            {
                "title": "Final refined paper",
                "summary": f"Refined paper at {relative}, a copy of {current.name}",
                "kind": "draft",
            },
        )
        log.info("%s: final paper saved to %s (from %s).", AGENT, final_pdf, current.name)
        return final_pdf
