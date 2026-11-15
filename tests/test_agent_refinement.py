from __future__ import annotations

import io
import json
import logging
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import pymupdf
import pytest
from PIL import Image as PILImage

from paper_orchestra.agents.refinement import (
    ContentRefinementAgent,
    FormattingReview,
    RefinementError,
    extract_scores,
    judge_revision,
)
from paper_orchestra.compute.base import CompileResult, ComputeBackend
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.config import PipelineConfig
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.eval.agent_review import ReviewError
from paper_orchestra.llm.base import Capabilities, LLMConfigError, LLMError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.llm.parsing import parse_json
from paper_orchestra.memory import ContextHarness, RunCancelled, TranscriptLog, Wiki
from paper_orchestra.prompts.format import (
    FORMAT_FIX_SYSTEM_PROMPT,
    FORMAT_REVIEW_SYSTEM_PROMPT,
    format_fix_user_prompt,
)
from paper_orchestra.prompts.no_leakage import UNIVERSAL_NO_LEAKAGE_PROMPT
from paper_orchestra.prompts.refinement import (
    CONTENT_REFINEMENT_SYSTEM_PROMPT,
    LATEX_REPAIR_SYSTEM_PROMPT,
    REFINEMENT_DREAM_FOCUS,
    revision_user_prompt,
)
from paper_orchestra.utils.pdf import load_paper, page_count

REVISE = "refinement.revise"
REVIEW = "refinement.review"
META = "refinement.review.meta"
FORMAT_REVIEW = "refinement.format_review"
FORMAT_FIX = "refinement.format_fix"
INITIAL_REPAIR = "refinement.initial_repair"
DREAM = "dream.project"

BIB = """@inproceedings{Vaswani2017Attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
"""
NO_ISSUES = json.dumps(
    {
        "figure_and_tables": [
            {"element": "Table 1", "detected_issue": "None", "suggested_fix": "None"}
        ],
        "other_issues": [],
    }
)
ISSUES = json.dumps(
    {
        "figure_and_tables": [
            {
                "element": "Table 1",
                "detected_issue": "The table spills into the right margin.",
                "suggested_fix": "Apply \\small to the table.",
            }
        ],
        "other_issues": [
            {
                "page": 1,
                "element": "Section 4, Paragraph 1",
                "detected_issue": "None",
                "suggested_fix": "None",
            }
        ],
    }
)


def paper(repo_root: Path, cue: str, *, extra: str = "") -> str:
    """The cvpr2025 template, written out.  ``cue`` tells the scripted reviewers what to score."""
    template = (
        repo_root / "src" / "paper_orchestra" / "templates" / "cvpr2025" / "template.tex"
    ).read_text(encoding="utf-8")
    written = {
        "My Title": "Sparse Routing for Tiny Models",
        "Abstract here.": "We study sparse routing for small models.",
        "Introduction here.": "Attention is widely used~\\cite{Vaswani2017Attention}. "
        "Our method is described in \\cref{sec:method}.",
        "Related work here.": "Routing has a long history.",
        "Methodology here.": "We route each token to few experts." + extra,
        "Experimental setup, results, and analysis here.": "Accuracy improves by 2.1 points."
        f"\n\nReviewer cue: {cue}.",
        "Conclusion here.": "Sparse routing works.",
    }
    for placeholder, text in written.items():
        template = template.replace(placeholder, text)
    return template.strip() + "\n"


def review_json(**overrides) -> str:
    review = {
        "Summary": "A paper about sparse routing.",
        "Strengths": ["clear idea"],
        "Weaknesses": ["thin ablations"],
        "Originality": 2,
        "Quality": 2,
        "Clarity": 2,
        "Significance": 2,
        "Questions": ["What is the latency?"],
        "Limitations": ["small scale"],
        "EthicalConcerns": False,
        "Soundness": 2,
        "Presentation": 2,
        "Contribution": 2,
        "Overall": 5,
        "Confidence": 4,
        "Decision": "Reject",
    }
    review.update(overrides)
    return json.dumps(review)


def review_by_cue(request: FakeRequest) -> str:
    """A reviewer whose scores are whatever the paper's cue line says."""
    cue = re.search(r"Reviewer cue: ([^.]*)\.", " ".join(request.text.split())).group(1)
    scores = {axis.capitalize(): int(value) for axis, value in re.findall(r"(\w+)=(\d+)", cue)}
    return review_json(**scores)


def fence(tex: str) -> str:
    return f"```latex\n{tex}```\n"


def revision(tex: str, **plan) -> str:
    plan = plan or {"addressed_weaknesses": ["added ablations"], "actions_taken": ["rewrote 4.2"]}
    return f"```json\n{json.dumps(plan)}\n```\n```latex\n{tex}```\n"


class Reviser:
    """Answers reflection iteration ``n`` with ``rounds[n - 1]``: one reply per turn."""

    def __init__(self, *rounds: str | Sequence[str]) -> None:
        self.rounds = [[r] if isinstance(r, str) else list(r) for r in rounds]

    def __call__(self, request: FakeRequest) -> str:
        first_turn = request.history[0][0] if request.history else request.text
        iteration = int(re.search(r"reflection iteration (\d+)", first_turn).group(1))
        return self.rounds[iteration - 1][request.turn]


def script(**overrides) -> dict:
    responders = {
        REVIEW: review_by_cue,
        META: review_json(Summary="Meta-review: the reviewers agree."),
        FORMAT_REVIEW: NO_ISSUES,
        DREAM: "Nothing to consolidate.",
    }
    responders.update(overrides)
    return responders


def write_pdf(path: Path, tex: str) -> None:
    """A one-page PDF carrying the reviewer cue of ``tex``."""
    cue = re.search(r"Reviewer cue: [^\n]*", tex)
    lines = [
        "Sparse Routing for Tiny Models.",
        cue.group(0) if cue else "No cue.",
        "We route each token to few experts and accuracy improves by 2.1 points.",
        "The compiled paper has enough text for the reviewers to read.",
    ]
    document = pymupdf.open()
    page = document.new_page(width=300, height=200)
    for index, line in enumerate(lines):
        page.insert_text((10, 20 + 10 * index), line, fontsize=6)
    document.save(path)
    document.close()


def latex_error(tex_name: str, tex: str, marker: str) -> str:
    """The error pdflatex reports for an undefined ``marker``, with file name and line."""
    number, line = next((n, ln) for n, ln in enumerate(tex.splitlines(), 1) if marker in ln)
    quoted = line[: line.index(marker) + len(marker)][-40:]
    return f"./{tex_name}.tex:{number}: Undefined control sequence. [l.{number} {quoted}]"


class ScriptedCompute(ComputeBackend):
    """Builds nothing: a document "compiles" unless it contains one of the failing markers.

    A flawed marker stands for the usual LaTeX error: pdflatex reports it and writes a PDF
    all the same.
    """

    name = "scripted"

    def __init__(
        self,
        failing: Sequence[str] = ("\\badmacro",),
        flawed: Sequence[str] = ("\\flawedmacro", "\\otherflaw"),
    ) -> None:
        self.failing = tuple(failing)
        self.flawed = tuple(flawed)
        self.builds: list[tuple[str, str]] = []

    def run_python(self, code, *, inputs=None, outputs=(), timeout=120.0):
        raise AssertionError("the writing agents never run Python")

    def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0):
        work = Path(workdir)
        tex = (work / f"{tex_name}.tex").read_text(encoding="utf-8")
        self.builds.append((tex_name, tex))
        for marker in self.failing:
            if marker in tex:
                return CompileResult(
                    ok=False, errors=[f"Undefined control sequence. [l.40 {marker}]"]
                )
        pdf = Path(output_pdf) if output_pdf else work / f"{tex_name}.pdf"
        write_pdf(pdf, tex)
        errors = [latex_error(tex_name, tex, marker) for marker in self.flawed if marker in tex]
        return CompileResult(ok=True, pdf_path=pdf, errors=errors)


def make_ctx(
    tmp_path: Path,
    repo_root: Path,
    client: FakeLLM,
    draft: str,
    compute: ComputeBackend | None = None,
    *,
    memory: bool = True,
    **overrides,
) -> RunContext:
    base = tmp_path / "run"
    template_dir = repo_root / "src" / "paper_orchestra" / "templates" / "cvpr2025"
    cfg = PipelineConfig(base_dir=base, template_dir=template_dir, **overrides)
    paths = RunPaths(base)
    paths.materials_dir.mkdir(parents=True)
    paths.idea.write_text("# Idea\nRoute each token to a few experts.", encoding="utf-8")
    paths.experimental_log.write_text(
        "# Log\nBaseline 71.3, ours 73.4 (+2.1 points).", encoding="utf-8"
    )
    shutil.copytree(template_dir, paths.latex_dir)
    paths.references_bib.write_text(BIB, encoding="utf-8")
    paths.figures_dir.mkdir()
    buffer = io.BytesIO()
    PILImage.new("RGB", (24, 16), "white").save(buffer, format="PNG")
    (paths.figures_dir / "accuracy.png").write_bytes(buffer.getvalue())
    paths.literature_dir.mkdir()
    paths.citation_map.write_text(
        json.dumps({"Vaswani2017Attention": {"title": "Attention Is All You Need"}}),
        encoding="utf-8",
    )
    paths.raw_draft_tex.write_text(draft, encoding="utf-8")
    # Left over from the section writing stage: must not leak into the refinement workspace.
    (paths.latex_dir / "raw_draft_paper.aux").write_text("stale", encoding="utf-8")
    if memory:
        harness = ContextHarness(
            project=Wiki(base / "memory"),
            transcripts=TranscriptLog(base / "memory" / "transcripts"),
            client_for=lambda model: client,
        )
    else:
        harness = ContextHarness(client_for=lambda model: client)
    return RunContext(cfg=cfg, paths=paths, harness=harness, compute=compute or ScriptedCompute())


def run(ctx: RunContext) -> Path:
    return ContentRefinementAgent(ctx).run(ctx.paths.raw_draft_tex)


def worklog(ctx: RunContext, name: str = "content_refinement_worklog.json") -> dict:
    return json.loads((ctx.paths.refinement_dir / name).read_text(encoding="utf-8"))


def text_of(ctx: RunContext, name: str) -> str:
    return (ctx.paths.refinement_dir / name).read_text(encoding="utf-8")


def events(ctx: RunContext, kind: str) -> list[dict]:
    return [e for e in ctx.harness.transcripts.events() if e["type"] == kind]


def all_notes(ctx: RunContext) -> list[str]:
    """Bodies of every note, including the ones a dream has already archived."""
    wiki = ctx.harness.project
    return [wiki.read(info.path).body for info in wiki.list("notes")]


def requests(client: FakeLLM, label: str) -> list[FakeRequest]:
    return [request for request in client.requests if request.label == label]


# -- acceptance logic ----------------------------------------------------------


def test_judge_revision_branches():
    base = json.loads(review_json(Overall=5))

    up = judge_revision(base, json.loads(review_json(Overall=6, Clarity=1, Quality=1)))
    assert (up.outcome, up.accepted) == ("ACCEPTED_SCORE_INCREASE", True)
    assert (up.total_gain, up.total_drop) == (0, 2)  # sub-axes do not matter when overall rose

    down = judge_revision(base, json.loads(review_json(Overall=4, Clarity=4, Quality=4)))
    assert (down.outcome, down.accepted) == ("REJECTED_SCORE_DECREASE", False)

    even = judge_revision(base, json.loads(review_json(Clarity=3, Soundness=1)))
    assert (even.outcome, even.accepted) == ("ACCEPTED_NEUTRAL_IMPROVEMENT", True)
    assert (even.total_gain, even.total_drop) == (1, 1)
    assert even.deltas == {
        "Originality": 0,
        "Quality": 0,
        "Clarity": 1,
        "Significance": 0,
        "Soundness": -1,
        "Presentation": 0,
        "Contribution": 0,
    }

    unchanged = judge_revision(base, base)
    assert unchanged.outcome == "ACCEPTED_NEUTRAL_IMPROVEMENT"

    worse = judge_revision(base, json.loads(review_json(Clarity=3, Soundness=1, Quality=1)))
    assert (worse.outcome, worse.accepted) == ("REJECTED_DEGRADATION", False)
    assert "Overall 5 -> 5; sub-axis gain 1, drop 2" in worse.movement()
    assert "Clarity +1" in worse.movement() and "Soundness -1" in worse.movement()
    # Confidence never takes part in the comparison.
    assert judge_revision(base, json.loads(review_json(Confidence=1))).total_drop == 0


def test_extract_scores_tolerates_malformed_values():
    scores = extract_scores({"Overall": "7", "Clarity": None, "Quality": "high", "Soundness": 3})
    assert scores["Overall"] == 7.0 and scores["Soundness"] == 3.0
    assert scores["Clarity"] == 0.0 and scores["Quality"] == 0.0 and scores["Originality"] == 0.0


def test_formatting_review_issue_detection():
    assert not FormattingReview.model_validate_json(NO_ISSUES).has_issues()
    assert FormattingReview.model_validate_json(ISSUES).has_issues()
    only_other = {
        "figure_and_tables": [],
        "other_issues": [
            {"page": "2", "element": "Sec. 3", "detected_issue": "Orphan", "suggested_fix": "x"}
        ],
    }
    assert FormattingReview.model_validate(only_other).has_issues()
    blank = {
        "figure_and_tables": [
            {"element": "Figure 1", "detected_issue": " none. ", "suggested_fix": ""},
            {"element": "Figure 2", "detected_issue": "N/A", "suggested_fix": ""},
            {"element": "Figure 3", "detected_issue": "", "suggested_fix": ""},
        ],
        "other_issues": [],
    }
    assert not FormattingReview.model_validate(blank).has_issues()
    for name, spec in FormattingReview.model_json_schema()["$defs"].items():
        assert all(field.get("type") != "object" for field in spec["properties"].values()), name


# -- the content loop ----------------------------------------------------------


def test_accepted_revisions_and_every_output_file(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=6", extra=" We add an ablation.")
    v2 = paper(repo_root, "overall=6 clarity=3", extra=" We add an ablation and a diagram.")
    lesson_page = (
        "---\ntitle: Revision rules\nsummary: Ablations raised Overall (v1)\nkind: lesson\n---\n"
        "RULE: answer reviewer questions with ablations."
    )

    def dream(request: FakeRequest) -> str:
        request.memory.create("/memories/wiki/lessons/revision-rules.md", lesson_page)
        return "Wrote one rule."

    client = FakeLLM(script(**{REVISE: Reviser(revision(v1), revision(v2)), DREAM: dream}))
    compute = ScriptedCompute()
    ctx = make_ctx(tmp_path, repo_root, client, v0, compute, max_reflections=2)

    final_pdf = run(ctx)

    work = ctx.paths.refinement_dir
    assert final_pdf == work / "final_refined_paper.pdf"
    assert text_of(ctx, "final_refined_paper.tex") == v2
    assert final_pdf.read_bytes() == (work / "refined_paper_v2.pdf").read_bytes()
    assert "overall=6 clarity=3" in load_paper(final_pdf)

    # Every file of the output contract.
    produced = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
    assert produced >= {
        "initial_draft.tex",
        "initial_draft.pdf",
        "refined_paper_v1.tex",
        "refined_paper_v1.pdf",
        "refined_paper_v2.tex",
        "refined_paper_v2.pdf",
        "peer_reviews/review_v0.json",
        "peer_reviews/review_v1.json",
        "peer_reviews/review_v2.json",
        "content_refinement_worklog.json",
        "format_candidate_v0.tex",
        "format_refinement_worklog.json",
        "pdf_screenshots/content_refinement/v0/grid_1.png",
        "pdf_screenshots/content_refinement/v1/grid_1.png",
        "pdf_screenshots/content_refinement/v2/grid_1.png",
        "pdf_screenshots/format_refinement/v0/grid_1.png",
        "final_refined_paper.tex",
        "final_refined_paper.pdf",
        # The workspace is a copy of the LaTeX directory ...
        "cvpr.sty",
        "references.bib",
        "figures/accuracy.png",
        "raw_draft_paper.tex",
    }
    # ... without build products, and nothing was formatted because nothing was wrong.
    assert "raw_draft_paper.aux" not in produced
    assert not any(name.startswith("formatted_candidate") for name in produced)
    assert text_of(ctx, "initial_draft.tex") == v0
    assert text_of(ctx, "format_candidate_v0.tex") == v2
    assert [name for name, _ in compute.builds] == [
        "initial_draft",
        "refined_paper_v1",
        "refined_paper_v2",
    ]
    assert worklog(ctx, "format_refinement_worklog.json") == {}

    log = worklog(ctx)
    assert list(log) == ["v1", "v2"]
    assert list(log["v1"]) == [
        "round",
        "agent_plan",
        "scores_before",
        "scores_after",
        "deltas",
        "total_gain",
        "total_drop",
        "outcome",
    ]
    assert log["v1"]["round"] == 1 and log["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert log["v1"]["agent_plan"]["addressed_weaknesses"] == ["added ablations"]
    assert log["v1"]["scores_before"]["Overall"] == 5 and log["v1"]["scores_after"]["Overall"] == 6
    assert log["v2"]["outcome"] == "ACCEPTED_NEUTRAL_IMPROVEMENT"
    assert log["v2"]["deltas"]["Clarity"] == 1
    assert (log["v2"]["total_gain"], log["v2"]["total_drop"]) == (1, 0)

    # Reviews: the ensemble plus a meta-review per version, with no access to our memory.
    review_v1 = json.loads(text_of(ctx, "peer_reviews/review_v1.json"))
    assert review_v1["Overall"] == 6 and review_v1["Summary"].startswith("Meta-review")
    assert "Ethical Concerns" in review_v1
    assert len(requests(client, REVIEW)) == 9 and len(requests(client, META)) == 3
    for request in requests(client, REVIEW) + requests(client, META):
        assert request.model == ctx.cfg.models.reflection
        assert request.memory is None and not request.tools and "# Memory" not in request.system

    # The reviser works from the current version: its PDF, its source, its review.
    first, second = requests(client, REVISE)
    assert first.system.startswith("Role: Senior AI Researcher.") and "# Memory" in first.system
    assert first.memory is not None
    assert [pdf.title for pdf in first.pdfs] == ["initial_draft.pdf"]
    assert [pdf.title for pdf in second.pdfs] == ["refined_paper_v1.pdf"]
    assert "This is reflection iteration 1." in first.text
    assert "<previous_iterations_log>\nNone\n</previous_iterations_log>" in first.text
    assert '<reviewer_feedback score="5">' in first.text and '"Overall": 5' in first.text
    assert "# CVPR 2025 Paper Writing Guidelines" in first.text
    assert "Baseline 71.3, ours 73.4 (+2.1 points)." in first.text
    assert '"Vaswani2017Attention"' in first.text
    assert f"<paper_tex>\n{v0.strip()}\n</paper_tex>" in first.text
    assert "<known_problems>" not in first.text
    assert '<reviewer_feedback score="6">' in second.text
    assert '"outcome": "ACCEPTED_SCORE_INCREASE"' in second.text
    assert f"<paper_tex>\n{v1.strip()}\n</paper_tex>" in second.text

    # One dream between the two iterations; the rule it wrote reaches the next reviser
    # through the index in its system prompt.
    [dreaming] = requests(client, DREAM)
    assert REFINEMENT_DREAM_FOCUS in dreaming.text
    assert "Revision v1 was accepted (ACCEPTED_SCORE_INCREASE) against v0" in dreaming.text
    assert "[[lessons/revision-rules]]" not in first.system
    assert "[[lessons/revision-rules]]: Ablations raised Overall (v1)" in second.system

    # Memory: review pages, transcript records and one lesson note per judged revision.
    wiki = ctx.harness.project
    page_v0 = wiki.read("wiki/reviews/v0.md")
    assert page_v0.meta["kind"] == "review"
    assert page_v0.meta["summary"] == "Overall 5/10 (Reject); revision v1 accepted"
    assert '"Weaknesses": [' in page_v0.body and '"Questions": [' in page_v0.body
    assert (
        "Revision v1, written in response, was accepted (ACCEPTED_SCORE_INCREASE) against v0: "
        "Overall 5 -> 6" in page_v0.body
    )
    page_v1 = wiki.read("wiki/reviews/v1.md")
    assert page_v1.meta["summary"] == (
        "Overall 6/10 (Reject); accepted against v0; revision v2 accepted"
    )
    assert "This revision was accepted (ACCEPTED_SCORE_INCREASE) against v0" in page_v1.body
    assert "Clarity +1" in wiki.read("wiki/reviews/v2.md").body
    final_page = wiki.read("wiki/drafts/final_refined_paper.md")
    assert "content_refinement_workdir/final_refined_paper.pdf" in final_page.body
    assert "- v2: ACCEPTED_NEUTRAL_IMPROVEMENT" in final_page.body

    reviews = events(ctx, "review")
    assert [(e["version"], e["outcome"]) for e in reviews] == [
        ("v0", "BASELINE"),
        ("v1", "ACCEPTED_SCORE_INCREASE"),
        ("v2", "ACCEPTED_NEUTRAL_IMPROVEMENT"),
    ]
    assert all(e["stage"] == "refinement" for e in reviews)
    assert reviews[2]["deltas"]["Clarity"] == 1 and reviews[2]["compared_with"] == "v1"
    assert [e["document"] for e in events(ctx, "compile")] == [
        "initial_draft.tex",
        "refined_paper_v1.tex",
        "refined_paper_v2.tex",
    ]
    lessons = [note for note in all_notes(ctx) if note.startswith("Revision v")]
    assert len(lessons) == 2
    assert any(
        "Revision v1 was accepted (ACCEPTED_SCORE_INCREASE) against v0: Overall 5 -> 6" in note
        and "- addressed_weaknesses:\n  - added ablations" in note
        for note in lessons
    )


def test_score_decrease_is_rejected_and_the_previous_version_restored(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=4 clarity=4", extra=" A bolder claim.")
    client = FakeLLM(script(**{REVISE: Reviser(revision(v1))}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=3, retry_after_rejection=False)

    final_pdf = run(ctx)

    work = ctx.paths.refinement_dir
    assert worklog(ctx)["v1"]["outcome"] == "REJECTED_SCORE_DECREASE"
    assert list(worklog(ctx)) == ["v1"]
    # Without retry_after_rejection the loop stops at the first rejection and falls back
    # to the version before it.
    assert len(requests(client, REVISE)) == 1 and not requests(client, DREAM)
    assert text_of(ctx, "final_refined_paper.tex") == v0
    assert text_of(ctx, "format_candidate_v0.tex") == v0
    assert final_pdf.read_bytes() == (work / "initial_draft.pdf").read_bytes()
    # The rejected revision stays on disk for inspection.
    assert text_of(ctx, "refined_paper_v1.tex") == v1 and (work / "refined_paper_v1.pdf").is_file()
    assert (work / "peer_reviews" / "review_v1.json").is_file()

    wiki = ctx.harness.project
    assert "revision v1 rejected" in wiki.read("wiki/reviews/v0.md").meta["summary"]
    assert "rejected (REJECTED_SCORE_DECREASE) against v0: Overall 5 -> 4" in (
        wiki.read("wiki/reviews/v1.md").body
    )
    assert any(
        note.startswith("Revision v1 was rejected (REJECTED_SCORE_DECREASE)")
        for note in all_notes(ctx)
    )
    assert "a copy of `initial_draft`" in wiki.read("wiki/drafts/final_refined_paper.md").body


def test_after_a_rejection_the_loop_retries_from_the_last_accepted_version(tmp_path, repo_root):
    """The default: a rejected revision is discarded, its lesson is consolidated, and the
    remaining iterations revise the last accepted version again."""
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=4 clarity=4", extra=" A bolder claim.")
    v2 = paper(repo_root, "overall=6", extra=" A better supported claim.")
    client = FakeLLM(script(**{REVISE: Reviser(revision(v1), revision(v2))}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=2)
    assert ctx.cfg.retry_after_rejection

    run(ctx)

    log = worklog(ctx)
    assert [log[v]["outcome"] for v in ("v1", "v2")] == [
        "REJECTED_SCORE_DECREASE",
        "ACCEPTED_SCORE_INCREASE",
    ]
    # The second revision started from v0, not from the rejected v1 ...
    second = requests(client, REVISE)[1].text
    assert "A bolder claim." not in second.split("<paper>")[-1]
    # ... was judged against v0's review ...
    assert log["v2"]["scores_before"]["Overall"] == 5
    # ... and the rejection was consolidated before it began.
    assert len(requests(client, DREAM)) == 1
    assert text_of(ctx, "final_refined_paper.tex") == v2


@pytest.mark.parametrize(
    ("cue", "outcome", "kept"),
    [
        ("overall=5 clarity=3 soundness=1", "ACCEPTED_NEUTRAL_IMPROVEMENT", "v1"),
        ("overall=5", "ACCEPTED_NEUTRAL_IMPROVEMENT", "v1"),
        ("overall=5 clarity=3 soundness=1 quality=1", "REJECTED_DEGRADATION", "v0"),
    ],
)
def test_equal_overall_is_decided_by_the_sub_axes(tmp_path, repo_root, cue, outcome, kept):
    versions = {
        "v0": paper(repo_root, "overall=5"),
        "v1": paper(repo_root, cue, extra=" Revised."),
    }
    client = FakeLLM(script(**{REVISE: Reviser(revision(versions["v1"]))}))
    ctx = make_ctx(tmp_path, repo_root, client, versions["v0"], max_reflections=1)

    run(ctx)

    assert worklog(ctx)["v1"]["outcome"] == outcome
    assert text_of(ctx, "final_refined_paper.tex") == versions[kept]


def test_revision_that_does_not_build_gets_one_repair_turn(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    broken = paper(repo_root, "overall=9", extra=" \\badmacro")
    fixed = paper(repo_root, "overall=6", extra=" Fixed macro.")
    plan = {"actions_taken": ["introduced a macro"]}
    client = FakeLLM(script(**{REVISE: Reviser([revision(broken, **plan), fence(fixed)])}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=1)

    run(ctx)

    first, repair = requests(client, REVISE)
    assert (first.turn, repair.turn) == (0, 1)
    assert "LaTeX did not produce a PDF." in repair.text
    assert "Undefined control sequence. [l.40 \\badmacro]" in repair.text
    assert "without undoing your revision" in repair.text and not repair.pdfs
    entry = worklog(ctx)["v1"]
    assert entry["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert entry["agent_plan"] == plan  # the worklog of the first reply survives the repair
    assert text_of(ctx, "refined_paper_v1.tex") == fixed
    assert text_of(ctx, "final_refined_paper.tex") == fixed
    [lesson] = [n for n in all_notes(ctx) if n.startswith("LaTeX repair of revision")]
    assert "2 of 2 reported problems were fixed and the document now builds cleanly" in lesson
    assert "\\badmacro" in lesson
    assert [e["ok"] for e in events(ctx, "compile")] == [True, False, True]


@pytest.mark.parametrize(
    ("defect", "outcome", "problem"),
    [
        (" \\badmacro", "REJECTED_COMPILE_FAILURE", "LaTeX did not produce a PDF"),
        (
            " See~\\cite{Ghost2099}.",
            "REJECTED_STATIC_CHECK",
            "1 citation key(s) are not in the bibliography: Ghost2099",
        ),
        # pdflatex wrote a PDF, but with an error the current version does not have.
        (" \\flawedmacro", "REJECTED_LATEX_ERRORS", "Undefined control sequence."),
    ],
)
def test_unrepairable_revision_is_rejected_and_the_loop_goes_on(
    tmp_path, repo_root, defect, outcome, problem
):
    v0 = paper(repo_root, "overall=5")
    bad = paper(repo_root, "overall=9", extra=defect)
    v2 = paper(repo_root, "overall=6", extra=" Second attempt.")
    client = FakeLLM(script(**{REVISE: Reviser([revision(bad), revision(bad)], revision(v2))}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=2)

    run(ctx)

    work = ctx.paths.refinement_dir
    log = worklog(ctx)
    assert log["v1"]["outcome"] == outcome
    assert any(problem in found for found in log["v1"]["problems"])
    assert "scores_after" not in log["v1"]
    assert log["v2"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    # The broken revision was never reviewed, and v2 was judged against v0.
    assert not (work / "peer_reviews" / "review_v1.json").exists()
    assert log["v2"]["scores_before"]["Overall"] == 5
    assert text_of(ctx, "final_refined_paper.tex") == v2
    # The next iteration starts again from v0 and is told what went wrong.
    retry = requests(client, REVISE)[2]
    assert (
        f'"outcome": "{outcome}"' in retry.text
        and "Ghost2099" not in retry.text.split("<paper_tex>")[1]
    )
    assert [pdf.title for pdf in retry.pdfs] == ["initial_draft.pdf"]
    assert len(requests(client, DREAM)) == 1

    page = ctx.harness.project.read("wiki/reviews/v0.md")
    assert f"Revision v1, written in response, was rejected before review ({outcome})" in page.body
    assert page.meta["summary"].endswith("revision v1 rejected before review; revision v2 accepted")
    assert not ctx.harness.project.exists("wiki/reviews/v1.md")
    assert any(
        note.startswith(f"Revision v1 was rejected before review ({outcome})")
        for note in all_notes(ctx)
    )
    assert ("v1", outcome) in [(e["version"], e["outcome"]) for e in events(ctx, "review")]


def test_reply_without_latex_counts_as_a_failed_revision(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    client = FakeLLM(script(**{REVISE: Reviser(["I would change nothing.", "Still nothing."])}))
    compute = ScriptedCompute()
    ctx = make_ctx(tmp_path, repo_root, client, v0, compute, max_reflections=1)

    run(ctx)

    entry = worklog(ctx)["v1"]
    assert entry["outcome"] == "REJECTED_COMPILE_FAILURE"
    assert entry["problems"] == ["the reply contained no LaTeX document"]
    assert len(requests(client, REVISE)) == 2
    assert [name for name, _ in compute.builds] == ["initial_draft"]
    assert text_of(ctx, "final_refined_paper.tex") == v0


def test_known_problems_of_the_current_version_do_not_disqualify_a_revision(tmp_path, repo_root):
    # The draft arrives with a citation that is not in the bibliography.  A revision that
    # keeps it is no worse; one that adds a second unknown key is.
    v0 = paper(repo_root, "overall=5", extra=" See~\\cite{Ghost2099}.")
    v1 = paper(repo_root, "overall=6", extra=" See~\\cite{Ghost2099}. Revised.")
    client = FakeLLM(script(**{REVISE: Reviser(revision(v1))}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=1)

    run(ctx)

    [request] = requests(client, REVISE)
    assert (
        "<known_problems>\n- 1 citation key(s) are not in the bibliography: Ghost2099\n"
        "</known_problems>" in request.text
    )
    assert worklog(ctx)["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert text_of(ctx, "final_refined_paper.tex") == v1


def test_revision_with_latex_errors_is_repaired_although_a_pdf_was_written(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    flawed = paper(repo_root, "overall=9", extra=" \\flawedmacro")
    fixed = paper(repo_root, "overall=6", extra=" Fixed macro.")
    client = FakeLLM(script(**{REVISE: Reviser([revision(flawed), fence(fixed)])}))
    compute = ScriptedCompute()
    ctx = make_ctx(tmp_path, repo_root, client, v0, compute, max_reflections=1)

    run(ctx)

    _, repair = requests(client, REVISE)
    assert "Undefined control sequence." in repair.text and "\\flawedmacro]" in repair.text
    assert "LaTeX did not produce a PDF" not in repair.text
    # The flawed build was never reviewed: only v0 and the repaired v1 were.
    assert len(requests(client, REVIEW)) == 6
    assert [(e["ok"], e["clean"]) for e in events(ctx, "compile")] == [
        (True, True),
        (True, False),
        (True, True),
    ]
    assert worklog(ctx)["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert text_of(ctx, "final_refined_paper.tex") == fixed


def test_problems_inherited_from_the_current_version_survive_an_edit(tmp_path, repo_root):
    # The draft arrives with a LaTeX error and two unknown citation keys.  The revision
    # moves the error to another line (and another file name) and drops one of the keys:
    # it is better than the draft, so neither a repair turn nor a rejection is due.
    flaw = "\nWe write \\flawedmacro for the routing matrix."
    v0 = paper(repo_root, "overall=5", extra=" See~\\cite{Ghost2099, Phantom2098}." + flaw)
    v1 = paper(
        repo_root, "overall=6", extra="\n\nA new paragraph.\n\nSee~\\cite{Ghost2099}." + flaw
    )
    worse = v1.replace("Sparse routing works.", "Sparse routing works. \\otherflaw")
    client = FakeLLM(
        script(**{REVISE: Reviser(revision(v1)), FORMAT_REVIEW: ISSUES, FORMAT_FIX: fence(worse)})
    )
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=1)

    run(ctx)

    [request] = requests(client, REVISE)
    known = request.text.split("<known_problems>")[1].split("</known_problems>")[0]
    assert "- 2 citation key(s) are not in the bibliography: Ghost2099, Phantom2098" in known
    assert "- ./initial_draft.tex:" in known and "Undefined control sequence." in known
    first, second = (e["errors"] for e in events(ctx, "compile")[:2])
    assert first != second  # the same defect, reported at another place
    assert worklog(ctx)["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    # A formatting candidate that adds an error of its own is still turned down.
    entry = worklog(ctx, "format_refinement_worklog.json")["v1"]
    assert entry["outcome"] == "REJECTED_LATEX_ERRORS"
    [problem] = entry["problems"]
    assert "\\otherflaw]" in problem
    assert text_of(ctx, "final_refined_paper.tex") == v1


# -- the initial draft ---------------------------------------------------------


def test_initial_draft_is_repaired_before_it_is_reviewed(tmp_path, repo_root):
    broken = paper(repo_root, "overall=5", extra=" \\badmacro")
    still_broken = paper(repo_root, "overall=5", extra=" \\badmacro again")
    repaired = paper(repo_root, "overall=5", extra=" Repaired.")
    client = FakeLLM(
        script(
            **{INITIAL_REPAIR: lambda request: [fence(still_broken), fence(repaired)][request.turn]}
        )
    )
    ctx = make_ctx(tmp_path, repo_root, client, broken, max_reflections=0)

    final_pdf = run(ctx)

    first, second = requests(client, INITIAL_REPAIR)
    assert first.system.startswith(LATEX_REPAIR_SYSTEM_PROMPT) and "# Memory" in first.system
    assert first.model == ctx.cfg.models.reflection
    assert "Undefined control sequence. [l.40 \\badmacro]" in first.text
    assert f"<paper_tex>\n{broken.strip()}\n</paper_tex>" in first.text
    assert second.turn == 1 and "<paper_tex>" not in second.text
    assert text_of(ctx, "initial_draft.tex") == repaired
    assert text_of(ctx, "final_refined_paper.tex") == repaired
    assert final_pdf.is_file() and worklog(ctx) == {}
    assert json.loads(text_of(ctx, "peer_reviews/review_v0.json"))["Overall"] == 5
    repairs = [n for n in all_notes(ctx) if n.startswith("LaTeX repair of the initial draft")]
    assert len(repairs) == 2 and any("now builds cleanly" in note for note in repairs)


def test_initial_draft_that_cannot_be_repaired_raises(tmp_path, repo_root):
    broken = paper(repo_root, "overall=5", extra=" \\badmacro")
    client = FakeLLM(script(**{INITIAL_REPAIR: fence(broken)}))
    ctx = make_ctx(tmp_path, repo_root, client, broken, max_repair_rounds=2)

    with pytest.raises(RefinementError) as info:
        run(ctx)

    message = str(info.value)
    assert "initial draft does not compile" in message and "2 repair round(s)" in message
    assert "Undefined control sequence" in message
    assert len(requests(client, INITIAL_REPAIR)) == 2
    assert not requests(client, REVIEW)
    assert not (ctx.paths.refinement_dir / "final_refined_paper.pdf").exists()


def test_missing_draft_raises_before_anything_is_touched(tmp_path, repo_root):
    client = FakeLLM(script())
    ctx = make_ctx(tmp_path, repo_root, client, paper(repo_root, "overall=5"))
    ctx.paths.raw_draft_tex.unlink()
    with pytest.raises(FileNotFoundError, match=r"raw_draft_paper\.tex"):
        run(ctx)
    assert not ctx.paths.refinement_dir.exists() and client.requests == []


def test_rerun_starts_from_a_fresh_workspace(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    client = FakeLLM(script())
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=0)
    stale = ctx.paths.refinement_dir / "refined_paper_v3.pdf"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"from an earlier attempt")

    run(ctx)

    assert not stale.exists() and text_of(ctx, "final_refined_paper.tex") == v0


# -- the formatting pass -------------------------------------------------------


def test_formatting_fix_is_applied_when_it_compiles_and_keeps_the_citations(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    formatted = v0.replace("We route each token", "{\\small We route each token").replace(
        "few experts.", "few experts.}"
    )
    reviews = iter([ISSUES, NO_ISSUES])
    client = FakeLLM(
        script(**{FORMAT_REVIEW: lambda request: next(reviews), FORMAT_FIX: fence(formatted)})
    )
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=0, max_formatting_rounds=2)

    final_pdf = run(ctx)

    work = ctx.paths.refinement_dir
    assert text_of(ctx, "format_candidate_v0.tex") == v0
    assert text_of(ctx, "formatted_candidate_v1.tex") == formatted
    assert text_of(ctx, "final_refined_paper.tex") == formatted
    assert final_pdf.read_bytes() == (work / "formatted_candidate_v1.pdf").read_bytes()
    for version in ("v0", "v1"):
        assert (work / "pdf_screenshots" / "format_refinement" / version / "grid_1.png").is_file()
    assert not (work / "formatted_candidate_v2.tex").exists()

    log = worklog(ctx, "format_refinement_worklog.json")
    assert list(log) == ["v1"]
    assert list(log["v1"]) == ["outcome", "formatting_feedback"]
    assert log["v1"]["outcome"] == "ACCEPTED_COMPILE_SUCCESS"
    assert log["v1"]["formatting_feedback"] == json.loads(ISSUES)

    # The reviewer sees the labelled page sheets and the guidelines, and none of our memory.
    first_review, second_review = requests(client, FORMAT_REVIEW)
    assert first_review.schema is FormattingReview
    assert first_review.system == FORMAT_REVIEW_SYSTEM_PROMPT and first_review.memory is None
    assert len(first_review.images) == page_count(work / "initial_draft.pdf") == 1
    assert "Pages 1 to 1 (left to right, then top to bottom):" in first_review.text
    assert "# CVPR 2025 Paper Writing Guidelines" in first_review.text
    sheet = work / "pdf_screenshots" / "format_refinement" / "v0" / "grid_1.png"
    assert first_review.images[0].data == sheet.read_bytes()
    # The second round reviews the pages of the accepted candidate.
    sheet = work / "pdf_screenshots" / "format_refinement" / "v1" / "grid_1.png"
    assert second_review.images[0].data == sheet.read_bytes()

    [fix] = requests(client, FORMAT_FIX)
    assert fix.system.startswith(FORMAT_FIX_SYSTEM_PROMPT) and "# Memory" in fix.system
    assert "Memory is read-only in this session." in fix.system
    assert '"detected_issue": "The table spills into the right margin."' in fix.text
    assert f"<paper_tex>\n{v0.strip()}\n</paper_tex>" in fix.text
    assert [(e["version"], e["outcome"]) for e in events(ctx, "format")] == [
        ("v1", "ACCEPTED_COMPILE_SUCCESS")
    ]
    assert "- v1: ACCEPTED_COMPILE_SUCCESS" in (
        ctx.harness.project.read("wiki/drafts/final_refined_paper.md").body
    )


@pytest.mark.parametrize(
    ("change", "outcome", "problem", "built"),
    [
        (
            lambda tex: tex.replace("~\\cite{Vaswani2017Attention}", ""),
            "REJECTED_CITATIONS_CHANGED",
            "citation keys removed: Vaswani2017Attention",
            False,
        ),
        (
            lambda tex: tex.replace("Routing has", "Routing~\\cite{Other2020Key} has"),
            "REJECTED_CITATIONS_CHANGED",
            "citation keys added: Other2020Key",
            False,
        ),
        (
            lambda tex: tex.replace("Routing has", "\\badmacro Routing has"),
            "REJECTED_COMPILE_FAILURE",
            "LaTeX did not produce a PDF",
            True,
        ),
        (
            lambda tex: tex.replace(
                "Routing has", "\\includegraphics{figures/gone.png} Routing has"
            ),
            "REJECTED_STATIC_CHECK",
            "figure file(s) not found: figures/gone.png",
            True,
        ),
        (
            lambda tex: tex.replace("Routing has", "\\flawedmacro Routing has"),
            "REJECTED_LATEX_ERRORS",
            "Undefined control sequence.",
            True,
        ),
    ],
)
def test_unsafe_formatting_candidate_is_rejected(
    tmp_path, repo_root, change, outcome, problem, built
):
    v0 = paper(repo_root, "overall=5")
    candidate = change(v0)
    client = FakeLLM(script(**{FORMAT_REVIEW: ISSUES, FORMAT_FIX: fence(candidate)}))
    compute = ScriptedCompute()
    ctx = make_ctx(tmp_path, repo_root, client, v0, compute, max_reflections=0)

    final_pdf = run(ctx)

    work = ctx.paths.refinement_dir
    entry = worklog(ctx, "format_refinement_worklog.json")["v1"]
    assert entry["outcome"] == outcome
    assert any(problem in found for found in entry["problems"])
    assert entry["formatting_feedback"] == json.loads(ISSUES)
    assert ("formatted_candidate_v1" in [name for name, _ in compute.builds]) is built
    # The candidate is kept for inspection, but the paper is the version before it.
    assert text_of(ctx, "formatted_candidate_v1.tex") == candidate
    compiled = outcome in ("REJECTED_STATIC_CHECK", "REJECTED_LATEX_ERRORS")
    assert (work / "formatted_candidate_v1.pdf").exists() is compiled
    assert text_of(ctx, "final_refined_paper.tex") == v0
    assert final_pdf.read_bytes() == (work / "initial_draft.pdf").read_bytes()
    assert any(
        note.startswith(f"Formatting candidate v1 was rejected ({outcome}); initial_draft was kept")
        for note in all_notes(ctx)
    )


# -- failures after a valid version exists -------------------------------------


def test_failure_in_an_iteration_costs_that_iteration_only(tmp_path, repo_root, caplog):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=6", extra=" Revised.")
    v3 = paper(repo_root, "overall=7", extra=" Revised again.")
    reviser = Reviser(revision(v1), "(never sent: the call fails)", revision(v3))

    def revise(request: FakeRequest) -> str:
        if "reflection iteration 2" in request.text:
            raise LLMError("request req_42 failed: overloaded")
        return reviser(request)

    client = FakeLLM(script(**{REVISE: revise, FORMAT_REVIEW: ISSUES, FORMAT_FIX: fence(v3)}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=3)

    with caplog.at_level(logging.ERROR, logger="paper_orchestra.agents.refinement"):
        final_pdf = run(ctx)

    log = worklog(ctx)
    assert log["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert log["v2"] == {
        "round": 2,
        "outcome": "ERROR",
        "error": "LLMError: request req_42 failed: overloaded",
    }
    assert "refinement iteration 2 failed" in caplog.text
    assert "continuing with refined_paper_v1" in caplog.text
    [error] = events(ctx, "error")
    assert "req_42" in error["error"] and error["kept"] == "refined_paper_v1"

    # The remaining iteration was used: it revised the last accepted version and was judged
    # against that version's review.
    assert log["v3"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert log["v3"]["scores_before"]["Overall"] == 6
    third = requests(client, REVISE)[-1]
    assert "reflection iteration 3" in third.text
    assert third.text.split("<paper_tex>")[1].split("</paper_tex>")[0].strip() == v1.strip()
    # The failed iteration left no revision behind, so it is not part of the revision
    # history the next one learns from.
    history = third.text.split("<previous_iterations_log>")[1].split("</previous_")[0]
    assert set(json.loads(history)) == {"v1"} and "req_42" not in third.text
    # Nothing new to consolidate after a failure: one consolidation, after iteration 1.
    assert len(requests(client, DREAM)) == 1

    # The formatting pass ran on the last accepted version.
    assert worklog(ctx, "format_refinement_worklog.json")["v1"]["outcome"] == (
        "ACCEPTED_COMPILE_SUCCESS"
    )
    assert final_pdf.is_file() and text_of(ctx, "final_refined_paper.tex") == v3


def test_failure_in_the_last_iteration_keeps_the_last_accepted_version(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=6", extra=" Revised.")
    reviser = Reviser(revision(v1))

    def revise(request: FakeRequest) -> str:
        if "reflection iteration 2" in request.text:
            raise LLMError("request req_42 failed: overloaded")
        return reviser(request)

    client = FakeLLM(script(**{REVISE: revise}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=2)

    final_pdf = run(ctx)

    assert [entry["outcome"] for entry in worklog(ctx).values()] == [
        "ACCEPTED_SCORE_INCREASE",
        "ERROR",
    ]
    assert final_pdf.is_file() and text_of(ctx, "final_refined_paper.tex") == v1


def test_a_configuration_error_is_not_tried_again_in_every_iteration(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")

    def revise(request: FakeRequest) -> str:
        raise LLMConfigError("model 'fake' does not accept PDF input")

    client = FakeLLM(script(**{REVISE: revise}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=3)

    run(ctx)

    # The same request would be refused the same way, so the loop stops instead of
    # collecting one identical error per iteration.
    assert len(requests(client, REVISE)) == 1
    assert list(worklog(ctx)) == ["v1"] and worklog(ctx)["v1"]["outcome"] == "ERROR"
    assert text_of(ctx, "final_refined_paper.tex") == v0


def test_failed_review_of_a_revision_is_never_scored_as_zero(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "no scores here", extra=" Revised.")
    v2 = paper(repo_root, "overall=6", extra=" Revised with care.")

    def review(request: FakeRequest) -> str:
        return (
            "The reviewer is unavailable."
            if "no scores here" in request.text
            else review_by_cue(request)
        )

    reviser = Reviser(revision(v1), revision(v2))
    client = FakeLLM(script(**{REVISE: reviser, REVIEW: review}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=2)

    run(ctx)

    log = worklog(ctx)
    entry = log["v1"]
    assert entry["outcome"] == "ERROR" and entry["error"].startswith("ReviewError: all 3 review")
    # The unreviewed revision was dropped; the next one started from v0 and was judged
    # against v0's review, not against a review that never happened.
    assert log["v2"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert log["v2"]["scores_before"]["Overall"] == 5
    assert "Revised." not in requests(client, REVISE)[1].text.split("<paper_tex>")[1]
    assert text_of(ctx, "final_refined_paper.tex") == v2


def test_failed_baseline_review_fails_the_stage_so_that_it_is_retried(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=6", extra=" Revised.")
    outage = {"over": False}

    def review(request: FakeRequest) -> str:
        if not outage["over"]:
            raise LLMError("request req_7 failed: rate limited")
        return review_by_cue(request)

    client = FakeLLM(script(**{REVIEW: review, REVISE: Reviser(revision(v1))}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=1)
    work = ctx.paths.refinement_dir

    with pytest.raises(RefinementError, match="baseline review") as failure:
        run(ctx)

    # The error names its cause, for state.json and the log ...
    assert "ReviewError: all 3 review" in str(failure.value) and "req_7" in str(failure.value)
    cause = failure.value.__cause__
    assert isinstance(cause, ReviewError) and isinstance(cause.__cause__, LLMError)
    # ... and the unrevised draft is not passed off as the refined paper.
    assert not requests(client, REVISE) and not requests(client, FORMAT_REVIEW)
    assert not (work / "final_refined_paper.pdf").exists()
    assert not (work / "final_refined_paper.tex").exists()
    assert not (work / "peer_reviews" / "review_v0.json").exists()

    # The next attempt of the stage, once the outage is over, refines the paper.
    outage["over"] = True
    final_pdf = run(ctx)
    assert worklog(ctx)["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert final_pdf.is_file() and text_of(ctx, "final_refined_paper.tex") == v1


def test_an_unusable_baseline_review_fails_the_stage_too(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    client = FakeLLM(script(**{REVIEW: "The reviewer is unavailable."}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=0)

    # Also when no revision is planned: the review of the draft is part of what the stage
    # delivers.
    with pytest.raises(RefinementError, match="baseline review"):
        run(ctx)
    assert not (ctx.paths.refinement_dir / "final_refined_paper.pdf").exists()


@pytest.mark.parametrize("during", ["revision", "formatting"])
def test_a_stopped_run_ends_the_stage_instead_of_being_survived(tmp_path, repo_root, during):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=6", extra=" Revised.")
    holder: dict[str, RunContext] = {}

    def stop_then(reply: str):
        def responder(request: FakeRequest) -> str:
            holder["ctx"].harness.cancel()  # the call in flight still finishes
            return reply

        return responder

    responders = (
        {REVISE: stop_then(revision(v1))}
        if during == "revision"
        else {REVISE: Reviser(revision(v1)), FORMAT_REVIEW: stop_then(ISSUES)}
    )
    client = FakeLLM(script(**responders))
    ctx = holder["ctx"] = make_ctx(
        tmp_path,
        repo_root,
        client,
        v0,
        max_reflections=3 if during == "revision" else 1,
        max_formatting_rounds=3,
    )

    with pytest.raises(RunCancelled):
        run(ctx)

    # No further iteration or round was started, and no error was recorded as survived.
    stopped_at = REVISE if during == "revision" else FORMAT_REVIEW
    assert len(requests(client, stopped_at)) == 1 and not requests(client, FORMAT_FIX)
    assert events(ctx, "error") == []
    assert not (ctx.paths.refinement_dir / "final_refined_paper.pdf").exists()


def test_a_run_stopped_during_the_baseline_review_is_not_a_failed_review(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    holder: dict[str, RunContext] = {}

    def review(request: FakeRequest) -> str:
        holder["ctx"].harness.cancel()  # the call in flight still finishes
        return review_by_cue(request)

    client = FakeLLM(script(**{REVIEW: review}))
    ctx = holder["ctx"] = make_ctx(tmp_path, repo_root, client, v0, max_reflections=1)

    # A stop is passed on as a stop.  Wrapped as a failed baseline review it would be a
    # failure of the stage, the kind of error that a caller retries.
    with pytest.raises(RunCancelled):
        run(ctx)
    assert not requests(client, META) and not requests(client, REVISE)
    assert not (ctx.paths.refinement_dir / "final_refined_paper.pdf").exists()


def test_failed_formatting_review_is_not_read_as_no_issues(tmp_path, repo_root, caplog):
    v0 = paper(repo_root, "overall=5")
    client = FakeLLM(script(**{FORMAT_REVIEW: "[]"}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=0)

    with caplog.at_level(logging.ERROR, logger="paper_orchestra.agents.refinement"):
        run(ctx)

    entry = worklog(ctx, "format_refinement_worklog.json")["v1"]
    assert entry["outcome"] == "ERROR" and "LLMParseError" in entry["error"]
    assert "formatting round 1 failed" in caplog.text
    assert text_of(ctx, "final_refined_paper.tex") == v0


# -- provider differences ------------------------------------------------------


def test_pdf_is_only_attached_for_models_that_read_pdfs(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=6", extra=" Revised.")
    text_only = Capabilities(
        tools=False, memory_tool=False, pdf_input=False, structured_output=False
    )
    fenced_reviews = script(
        **{
            REVISE: Reviser(revision(v1)),
            REVIEW: lambda request: f"```json\n{review_by_cue(request)}\n```",
            META: f"```json\n{review_json()}\n```",
            FORMAT_REVIEW: f"```json\n{NO_ISSUES}\n```",
        }
    )
    client = FakeLLM(fenced_reviews, capabilities=text_only)
    ctx = make_ctx(tmp_path, repo_root, client, v0, max_reflections=1)

    run(ctx)

    [request] = requests(client, REVISE)
    assert not request.pdfs and request.memory is None
    # Push mode: the index is still injected, and schemas travel as prompt text.
    assert "# Memory" in request.system
    assert all(r.schema is None for r in requests(client, REVIEW))
    assert worklog(ctx)["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"


def test_runs_with_memory_disabled(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    v1 = paper(repo_root, "overall=6", extra=" Revised.")
    v2 = paper(repo_root, "overall=7", extra=" Revised twice.")
    client = FakeLLM(script(**{REVISE: Reviser(revision(v1), revision(v2))}))
    ctx = make_ctx(tmp_path, repo_root, client, v0, memory=False, max_reflections=2)

    final_pdf = run(ctx)

    assert final_pdf.is_file() and text_of(ctx, "final_refined_paper.tex") == v2
    assert not requests(client, DREAM)
    assert all("# Memory" not in (request.system or "") for request in client.requests)


# -- prompts -------------------------------------------------------------------


def test_prompts_carry_literal_json_and_latex():
    for prompt in (
        CONTENT_REFINEMENT_SYSTEM_PROMPT,
        LATEX_REPAIR_SYSTEM_PROMPT,
        FORMAT_REVIEW_SYSTEM_PROMPT,
        FORMAT_FIX_SYSTEM_PROMPT,
    ):
        assert "{{" not in prompt and "}}" not in prompt

    example = parse_json(CONTENT_REFINEMENT_SYSTEM_PROMPT)
    assert set(example) == {"addressed_weaknesses", "integrated_answers", "actions_taken"}
    assert "`\\end{figure*}` (not `\\end{figure}`)" in CONTENT_REFINEMENT_SYSTEM_PROMPT
    assert "do not remove the \\documentclass" in CONTENT_REFINEMENT_SYSTEM_PROMPT
    assert CONTENT_REFINEMENT_SYSTEM_PROMPT.endswith(UNIVERSAL_NO_LEAKAGE_PROMPT)
    assert LATEX_REPAIR_SYSTEM_PROMPT.endswith(UNIVERSAL_NO_LEAKAGE_PROMPT)
    assert FORMAT_FIX_SYSTEM_PROMPT.endswith(UNIVERSAL_NO_LEAKAGE_PROMPT)
    assert "`\\usepackage[capitalize]{cleveref}`" in FORMAT_FIX_SYSTEM_PROMPT
    assert "`width=1.0\\linewidth`" in FORMAT_FIX_SYSTEM_PROMPT
    assert "`\\vspace{-5pt}`" in FORMAT_FIX_SYSTEM_PROMPT
    assert "only make formatting, layout, and spacing adjustments" in FORMAT_FIX_SYSTEM_PROMPT
    assert "`[width=\\linewidth]` in `\\includegraphics`" in FORMAT_REVIEW_SYSTEM_PROMPT

    latex = "\\title{{A} {B}}\n\\begin{document}{x}\\end{document}"
    prompt = revision_user_prompt(
        iteration=2,
        previous_log='{"v1": {"round": 1}}',
        review='{"Overall": 5}',
        score=5.0,
        guidelines="rules",
        experimental_log="log",
        citation_map='{"K": {}}',
        tex=latex,
        known_problems=["environment 'figure*': 1 \\begin vs 0 \\end"],
    )
    assert f"<paper_tex>\n{latex}\n</paper_tex>" in prompt
    assert '<previous_iterations_log>\n{"v1": {"round": 1}}\n</previous_iterations_log>' in prompt
    assert "improve the reviewer score (currently 5)" in prompt
    assert "<known_problems>\n- environment 'figure*': 1 \\begin vs 0 \\end\n" in prompt
    fix = format_fix_user_prompt(feedback='{"other_issues": []}', guidelines="rules", tex=latex)
    assert '<formatting_feedback>\n{"other_issues": []}\n</formatting_feedback>' in fix


# -- the real thing ------------------------------------------------------------


@pytest.mark.latex
def test_real_compile_review_and_revision(tmp_path, repo_root):
    v0 = paper(repo_root, "overall=5")
    # pdflatex writes a PDF for this one too, with an error and a "??" in it.
    flawed = paper(repo_root, "overall=9", extra=" See \\undefinedmacro and \\cref{fig:gone}.")
    v1 = paper(repo_root, "overall=6", extra=" We also report an ablation.")
    client = FakeLLM(script(**{REVISE: Reviser([revision(flawed), revision(v1)])}))
    ctx = make_ctx(
        tmp_path, repo_root, client, v0, LocalBackend(), max_reflections=1, review_ensemble=1
    )

    final_pdf = run(ctx)

    work = ctx.paths.refinement_dir
    assert worklog(ctx)["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    assert text_of(ctx, "final_refined_paper.tex") == v1
    assert final_pdf.read_bytes() == (work / "refined_paper_v1.pdf").read_bytes()
    # Undo line breaks and end-of-line hyphenation of the two-column layout.
    text = " ".join(load_paper(final_pdf).split()).replace("- ", "")
    assert "Sparse Routing for Tiny Models" in text and "We also report an ablation." in text
    assert "??" not in text  # the citation and the cross-reference resolved
    initial, rejected, repaired = events(ctx, "compile")
    assert initial["document"] == "initial_draft.tex" and initial["ok"] and initial["clean"]
    assert rejected["document"] == repaired["document"] == "refined_paper_v1.tex"
    assert rejected["ok"] and not rejected["clean"]
    assert "Undefined control sequence." in rejected["errors"][0]
    assert "undefined reference 'fig:gone'" in rejected["warnings"]
    assert repaired["ok"] and repaired["clean"]
    # One repair turn for the build problems; the flawed build was never reviewed, and the
    # single reviewer wrote each review without a meta pass.
    _, repair = requests(client, REVISE)
    assert "\\undefinedmacro]" in repair.text and "undefined reference 'fig:gone'" in repair.text
    assert len(requests(client, REVIEW)) == 2 and not requests(client, META)
    assert (work / "pdf_screenshots" / "content_refinement" / "v1" / "grid_1.png").is_file()
    [sheet] = requests(client, FORMAT_REVIEW)[0].images
    assert max(PILImage.open(io.BytesIO(sheet.data)).size) <= 2576
