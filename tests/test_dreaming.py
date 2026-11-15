from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from paper_orchestra.llm.base import Capabilities, LLMConfigError, LLMError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, TranscriptLog, Wiki
from paper_orchestra.memory.dreaming import (
    LOCK_FILE,
    MAX_INLINE_NOTE_CHARS,
    PROMOTED_LEDGER,
    STALE_LOCK_SECONDS,
    LeakScanner,
    content_sources,
    dream,
    leak_reason,
    numbers_in,
    promote_lessons,
    unverified_numbers,
    venue_materials,
)


def make_harness(tmp_path, client, *, lab=False, dream_mode="stages") -> ContextHarness:
    project = Wiki(tmp_path / "project", name="project")
    project.write("wiki/sources/experimental_log.md", "Accuracy 91.2", force=True)
    lab_wiki = Wiki(tmp_path / "lab", name="lab", read_only=()) if lab else None
    return ContextHarness(
        project=project,
        lab=lab_wiki,
        transcripts=TranscriptLog(project.root / "transcripts", session_id="s1"),
        client_for=lambda model: client,
        dream_mode=dream_mode,
        dream_model="fake",
    )


def consolidating_responder(request: FakeRequest) -> str:
    assert "No paper is being written in this session" in request.system
    assert sorted(request.tools) == ["transcript_search", "wiki_search"]
    evidence = request.call_tool("transcript_search", {"query": "rejected overall"})
    assert "refinement" in evidence
    request.memory.create(
        "/memories/wiki/lessons/refinement.md",
        "---\nsummary: What reviewers rewarded and punished\nkind: lesson\n---\n"
        "Rule: do not add limitations paragraphs late. Evidence: v2 rejected, Overall 6 -> 5. "
        "See [[reviews/v2]] and [[sources/experimental_log]].",
    )
    request.memory.str_replace(
        "/memories/MEMORY.md", "_None yet._", "- Check every number against the log."
    )
    return "Folded two notes into lessons/refinement."


def test_project_dream_consolidates_notes_and_maintains_the_wiki(tmp_path):
    client = FakeLLM(consolidating_responder)
    harness = make_harness(tmp_path, client)
    harness.write_page(
        "wiki/reviews/v2.md", "Overall 5. See [[Experimental_Log]].", {"summary": "v2"}
    )
    harness.note(
        "v2 REJECTED: Overall 6 -> 5 after adding limitations", stage="refinement", kind="lesson"
    )
    harness.note("compile failed on cleverref", stage="section_writing", kind="lesson")
    harness.record("review", stage="refinement", outcome="rejected", overall=5)

    report = harness.dream(stage="refinement", focus="Turn revision outcomes into rules.")

    assert report.llm_used and report.error is None
    assert report.notes_seen == 2 and report.notes_archived == 2
    assert report.pages_changed == ["MEMORY.md", "wiki/lessons/refinement.md"]
    assert report.links_repaired == 1  # [[Experimental_Log]] -> sources/experimental_log
    assert report.summary.startswith("Folded two notes")
    assert harness.project.pending_notes() == []
    assert len(harness.project.list("notes/consolidated")) == 2
    index = harness.project.injected_context()
    assert "- Check every number against the log." in index
    assert "[[lessons/refinement]]: What reviewers rewarded and punished" in index
    assert report.lint["ok"] and report.lint["pending_notes"] == 0
    prompt = client.requests[0].text
    assert "Turn revision outcomes into rules." in prompt
    assert "v2 REJECTED: Overall 6 -> 5" in prompt and "Unconsolidated notes (2)" in prompt
    assert (
        next(e for e in harness.transcripts.events() if e["type"] == "dream")["notes_archived"] == 2
    )
    assert not (harness.project.root / LOCK_FILE).exists()


def test_stage_dreams_are_maintenance_only_unless_mode_is_stages(tmp_path):
    client = FakeLLM(lambda request: "should not be called")
    harness = make_harness(tmp_path, client, dream_mode="end")
    harness.note("something happened", stage="outline")
    report = harness.dream(stage="outline", focus="f")
    assert not report.llm_used and report.notes_archived == 0
    assert client.requests == [] and len(harness.project.pending_notes()) == 1
    forced = harness.dream(stage="outline", focus="f", force=True)
    assert forced.llm_used and forced.notes_archived == 1


def test_no_model_session_when_there_is_nothing_to_do(tmp_path):
    client = FakeLLM(lambda request: "unused")
    harness = make_harness(tmp_path, client)
    harness.write_page("wiki/a/page.md", "body", {"summary": "s"})
    harness.write_page("wiki/a/other.md", "[[a/page]]", {"summary": "s"})
    assert not harness.dream(stage="s", focus="f").llm_used
    assert client.requests == []
    # A focus that asks for new pages runs the model even with no notes pending.
    assert harness.dream(stage="s", focus="Build a synthesis.", build=True).llm_used
    assert "Build a synthesis." in client.requests[0].text


def test_lab_promotion_is_skipped_without_evidence(tmp_path):
    client = FakeLLM(lambda request: "unused")
    harness = make_harness(tmp_path, client, lab=True, dream_mode="end")
    reports = harness.finish()
    assert [r.llm_used for r in reports] == [False, False] and client.requests == []


def test_failed_model_session_keeps_notes_pending(tmp_path):
    def failing(request: FakeRequest) -> str:
        raise LLMError("overloaded")

    harness = make_harness(tmp_path, FakeLLM(failing))
    harness.note("important", stage="refinement")
    report = harness.dream(stage="refinement", focus="f")
    assert not report.llm_used and "overloaded" in report.error
    assert report.notes_archived == 0 and len(harness.project.pending_notes()) == 1
    assert report.lint  # the deterministic post-pass still ran
    assert not (harness.project.root / LOCK_FILE).exists()


def test_unexpected_errors_never_escape(tmp_path):
    def exploding(request: FakeRequest) -> str:
        raise RuntimeError("boom")

    harness = make_harness(tmp_path, FakeLLM(exploding))
    harness.note("n", stage="s")
    report = harness.dream(stage="s", focus="f")
    assert "RuntimeError: boom" in report.error
    assert not (harness.project.root / LOCK_FILE).exists()


def test_models_without_a_memory_tool_get_maintenance_only(tmp_path):
    client = FakeLLM(lambda r: "unused", capabilities=Capabilities(tools=False, memory_tool=False))
    harness = make_harness(tmp_path, client)
    harness.note("n", stage="s")
    report = harness.dream(stage="s", focus="f")
    assert not report.llm_used and client.requests == []
    assert len(harness.project.pending_notes()) == 1


def test_finish_promotes_portable_lessons_through_a_lab_only_session(tmp_path):
    seen: list[str] = []

    def responder(request: FakeRequest) -> str:
        seen.append(request.label)
        if request.label == "dream.project":
            request.memory.create(
                "/memories/wiki/lessons/latex.md",
                "---\nsummary: LaTeX pitfalls\nportable: true\n---\n"
                "RULE: load cleveref, never cleverref. EVIDENCE: the build failed until fixed.",
            )
            request.memory.create(
                "/memories/wiki/lessons/this-paper.md",
                "---\nsummary: paper-specific\n---\nReviewers liked the FooNet ablation.",
            )
            return "Consolidated."
        # The lab session: the lab wiki is its whole memory.
        assert "shared by every paper" in request.system
        assert sorted(request.tools) == ["wiki_search"]
        assert "cleveref" in request.text and "FooNet" not in request.text
        listing = request.memory.view("/memories")
        assert "sources" not in listing and "/memories/notes" in listing
        request.memory.create(
            "/memories/wiki/lessons/latex.md",
            "---\nsummary: LaTeX pitfalls\n---\nUse cleveref (the package), never cleverref.",
        )
        return "Folded one lesson."

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True, dream_mode="end")
    harness.write_page("wiki/sources/idea.md", "We propose FooNet for long video understanding.")
    harness.note("cleverref typo broke the build", stage="section_writing", kind="lesson")
    project_report, lab_report = harness.finish()

    assert seen == ["dream.project", "dream.lab"]
    assert project_report.notes_archived == 1 and project_report.scope == "project"
    assert lab_report.scope == "lab" and lab_report.llm_used
    assert lab_report.notes_seen == 1 and lab_report.notes_archived == 1
    assert lab_report.pages_changed == ["wiki/lessons/latex.md"]
    assert "[[lessons/latex]]: LaTeX pitfalls" in harness.lab.injected_context()
    assert harness.lab.pending_notes() == []
    promoted = [e for e in harness.transcripts.events() if e["type"] == "promotion"]
    assert [e["path"] for e in promoted] == ["wiki/lessons/latex.md"]
    # Finishing again (a resumed run) does not promote the same lesson twice.
    harness.finish()
    assert harness.lab.pending_notes() == [] and len(harness.lab.list("notes/consolidated")) == 1


def test_lessons_that_carry_paper_content_are_not_promoted(tmp_path):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "unused"), lab=True, dream_mode="off")
    harness.write_page(
        "wiki/sources/idea.md",
        "We propose FooNet, a sliding window transformer for long video question answering.",
    )
    harness.write_page("wiki/sources/experimental_log.md", "FooNet reaches 87.35 accuracy.")
    lessons = {
        "number": "Adding the 87.35 result to the abstract raised Clarity.",
        "wording": "Describe the sliding window transformer for long video question answering early.",
        "path": "The build failed in /home/alice/paper until cleveref was loaded.",
        "email": "Ask reviewer@example.org about the template.",
        "clean": "Answer a reviewer question inside the section it concerns, not in a new paragraph.",
        "not-portable": "Reviewers liked the ablation.",
    }
    for name, text in lessons.items():
        meta = {"summary": name}
        if name != "not-portable":
            meta["portable"] = True
        harness.project.write(f"wiki/lessons/{name}.md", text, meta)

    added = promote_lessons(harness)
    assert len(added) == 1
    assert "inside the section it concerns" in harness.lab.read(added[0]).body
    blocked = {
        e["path"]: e["reason"]
        for e in harness.transcripts.events()
        if e["type"] == "promotion_blocked"
    }
    assert set(blocked) == {f"wiki/lessons/{n}.md" for n in ("number", "wording", "path", "email")}
    assert "87.35" in blocked["wiki/lessons/number.md"]
    assert "wording" in blocked["wiki/lessons/wording.md"]


def test_leak_reason_ignores_years_and_small_numbers():
    sources = ["Trained for 100 epochs in 2024 with learning rate 0.1; accuracy 0.9125."]
    assert (
        leak_reason("Use dpi 100 and width 0.5 for figures in the 2024 template.", sources) is None
    )
    assert "0.9125" in leak_reason("Reached 0.9125.", sources)
    assert leak_reason("Reached 10.9125 elsewhere.", sources) is None
    assert "credential" in leak_reason("api_key = abcdef123456", [])


def test_numbers_no_record_supports_are_flagged_and_shown_to_the_dream(tmp_path):
    prompts_seen: list[str] = []

    def responder(request: FakeRequest) -> str:
        prompts_seen.append(request.text)
        request.memory.str_replace("/memories/wiki/lessons/claims.md", "87.9", "87.3")
        return "Corrected the number."

    harness = make_harness(tmp_path, FakeLLM(responder))
    harness.write_page("wiki/sources/experimental_log.md", "| Ours | 87.3 |\n| Baseline | 85.0 |")
    harness.project.write(
        "wiki/lessons/claims.md", "Ours reaches 87.9 against 85.0 (2 points).", {"summary": "s"}
    )
    assert unverified_numbers(harness.project) == [("wiki/lessons/claims.md", "87.9")]
    report = harness.dream(stage="section_writing", focus="check the numbers")
    assert report.llm_used and "states 87.9, which appears in no source" in prompts_seen[0]
    assert report.unverified_numbers == []


def test_finish_with_dreaming_off_only_maintains(tmp_path):
    client = FakeLLM(lambda r: "unused")
    harness = make_harness(tmp_path, client, lab=True, dream_mode="off")
    harness.note("n", stage="s")
    reports = harness.finish()
    assert [r.llm_used for r in reports] == [False, False] and client.requests == []


def test_lock_serialises_dreams_and_recovers_from_stale_locks(tmp_path):
    client = FakeLLM(lambda r: "ok")
    harness = make_harness(tmp_path, client)
    harness.note("n", stage="s")
    lock = harness.project.root / LOCK_FILE
    lock.write_text("12345\n")
    busy = dream(harness, stage="s", focus="f", model="fake", use_model=True, lock_wait=0)
    assert "in progress" in busy.error and not busy.llm_used
    assert lock.exists() and len(harness.project.pending_notes()) == 1
    old = time.time() - STALE_LOCK_SECONDS - 10
    lock.write_text("12345\n")
    os.utime(lock, (old, old))
    recovered = dream(harness, stage="s", focus="f", model="fake", use_model=True)
    assert recovered.error is None and not lock.exists()


# -- what reaches the shared memory -------------------------------------------


def test_numbers_are_compared_by_value_not_by_spelling():
    assert numbers_in("2,908 steps, 87.30 and 007") >= {"2908", "87.3", "7"}
    assert numbers_in("version 3.5.1") == set()  # not a number
    sources = ["The corpus has 2,908 documents; ours scores 87.30 and 4.2 BLEU on 908 items."]
    for text in ("We kept 2908 documents.", "Scored 87.3.", "A 4.2 gain.", "Only 908 items."):
        assert "number" in leak_reason(text, sources), text
    sources = ["Trained on 2908 documents."]
    assert "2,908" in leak_reason("Mention the 2,908 documents early.", sources)
    # Round figures, small integers, years and one-digit decimals identify nothing.
    sources = ["100 epochs, 12 layers, batch 5000, lr 0.5, in 2024."]
    assert (
        leak_reason("Use 100 dpi, 12 pt, 5000 px, width 0.5, per the 2024 template.", sources)
        is None
    )


def test_distinctive_names_from_the_paper_are_refused_however_they_are_spelled():
    scanner = LeakScanner(
        ["We propose FooNet, evaluated on CIFAR-10 and WidgetBench with BLEU, in LaTeX at CVPR."],
        allowed=["Submissions to the XYZConf use the xyzconf.sty LaTeX style."],
    )
    assert "FooNet" in scanner.reason("Reviewers rewarded the FooNet ablation.")
    assert "foonet" in scanner.reason("Put the foonet ablation first.")
    assert "CIFAR-10" in scanner.reason("Report CIFAR-10 before the ablation.")
    assert "WidgetBench" in scanner.reason("WidgetBench tables overflow the column.")
    # Tools, venues and the field's vocabulary are not paper content, nor is the venue's
    # own material; an upper-case word is only a name when written in upper case.
    assert scanner.reason("LaTeX at CVPR needs the review option.") is None
    assert scanner.reason("The XYZConf style redefines \\cite.") is None
    assert scanner.reason("We bleu it; cvpr rules apply; add a note.") is None
    assert scanner.has_sources and not LeakScanner([]).has_sources


def test_the_scan_does_not_mistake_ordinary_writing_for_paper_content():
    scanner = LeakScanner(
        [
            "RESULTS\nOur results do NOT depend on the seed, and we do not tune it. "
            "We train on 8 GPUs and release PDFs. BERT is the encoder.\n"
            "\\includegraphics[width=0.48\\textwidth]{fig} reaches 0.48 F1 and 87.3 \\pm 0.2."
        ]
    )
    # Words written loudly, plurals of common abbreviations, and layout factors.
    assert scanner.reason("Do NOT move the RESULTS table; GPUs and PDFs are fine.") is None
    assert scanner.reason("Use 0.48\\textwidth for two figures side by side.") is None
    # The same digits as a result are still the paper's.
    assert "0.48" in scanner.reason("The method reaches 0.48.")
    assert "87.3" in scanner.reason("Reported as 87.3 \\pm 0.2.")
    assert "BERT" in scanner.reason("Mention BERT in the abstract.")


def test_the_scan_covers_everything_that_holds_paper_content(tmp_path):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "unused"), lab=True, dream_mode="off")
    harness.write_page("wiki/sources/idea.md", "A sparse method for long documents.")
    harness.write_page(
        "wiki/sources/guidelines.md", "Papers are limited to eight pages excluding references."
    )
    harness.write_page(
        "wiki/paper/outline.md", "Title: Sparse Routing Beats Dense Attention On Long Inputs"
    )
    harness.write_page(
        "wiki/figures/fig-main.md", "Caption: latency grows linearly while quality stays flat"
    )
    harness.write_page(
        "wiki/reviews/v1.md",
        "Weaknesses: the comparison against retrieval baselines omits memory cost entirely.",
    )
    shipped = "\\usepackage[review]{venue} % camera ready uses the final option instead\n"
    template = tmp_path / "template"
    template.mkdir()
    (template / "template.tex").write_text(shipped, encoding="utf-8")
    (template / "macros.tex").write_text("\\newcommand{\\tO}{\\mathbb{O}}", encoding="utf-8")
    workspace = tmp_path / "run" / "latex_writeup"
    workspace.mkdir(parents=True)
    (workspace / "macros.tex").write_text("\\newcommand{\\tO}{\\mathbb{O}}", encoding="utf-8")
    (workspace / "raw_draft_paper.tex").write_text(
        "\\section{Method} Our router assigns each token bucket a learned sparsity budget.",
        encoding="utf-8",
    )
    # After the literature stage the workspace's template.tex holds this paper's
    # Introduction: whatever its name, it is no longer the venue's file.
    (workspace / "template.tex").write_text(
        shipped + "\\section{Introduction} We introduce GlimmerNet, which reaches 54.58 on "
        "average while halving the latency of dense routing.",
        encoding="utf-8",
    )
    harness.protect(workspace)
    harness._venue_dirs.append(template)

    sources = "\n".join(content_sources(harness.project, [workspace], [template]))
    for fragment in ("Sparse Routing", "latency grows", "retrieval baselines", "token bucket"):
        assert fragment in sources
    assert "GlimmerNet" in sources and "eight pages" not in sources
    assert "mathbb" not in sources  # the workspace copy of a shipped file is the venue's
    venue = "\n".join(venue_materials(harness.project, [workspace], [template]))
    assert "eight pages" in venue and "camera ready" in venue and "GlimmerNet" not in venue

    lessons = {
        "title": "For Sparse Routing Beats Dense Attention On Long Inputs, lead with speed.",
        "caption": "Captions such as latency grows linearly while quality stays flat read well.",
        "review": "When the comparison against retrieval baselines omits memory cost, add it.",
        "draft": "Say that the router assigns each token bucket a learned sparsity budget.",
        "intro-name": "Reviewers rewarded naming GlimmerNet in the first sentence.",
        "intro-number": "Leading with the 54.58 average raised the Clarity score.",
        "intro-wording": "Say early that it works while halving the latency of dense routing.",
        "venue": "Papers are limited to eight pages excluding references, so cut early.",
        "template": "The camera ready uses the final option instead of review.",
        "words": "Go to the appendix so the main text stays short; my advice is to cut, etc.",
    }
    for name, text in lessons.items():
        harness.project.write(f"wiki/lessons/{name}.md", text, {"summary": name, "portable": True})
    added = promote_lessons(harness)
    shared = "\n".join(harness.lab.read(note).body for note in added)
    assert len(added) == 3, shared
    assert "eight pages" in shared and "camera ready" in shared and "my advice" in shared
    blocked = {e["path"] for e in harness.transcripts.events() if e["type"] == "promotion_blocked"}
    assert blocked == {
        f"wiki/lessons/{n}.md"
        for n in (
            "title",
            "caption",
            "review",
            "draft",
            "intro-name",
            "intro-number",
            "intro-wording",
        )
    }
    # A draft written later is protected too: the scanner follows the files.
    (workspace / "v2.tex").write_text("We call the method QuillFeather routing.")
    assert "QuillFeather" in harness.leak_scanner().reason("QuillFeather needs an ablation.")


def test_a_template_directory_that_contains_the_run_does_not_whitelist_the_paper(tmp_path):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "unused"), lab=True, dream_mode="off")
    workspace = tmp_path / "run" / "latex_writeup"
    workspace.mkdir(parents=True)
    (workspace / "raw_draft_paper.tex").write_text("We introduce GlimmerNet.", encoding="utf-8")
    (tmp_path / "template.tex").write_text("\\usepackage{venue}", encoding="utf-8")
    harness.protect(workspace)
    harness._venue_dirs.append(tmp_path)  # a template given as "." or a parent directory
    assert "GlimmerNet" in harness.leak_scanner().reason("GlimmerNet tables overflow.")


def test_nothing_is_shared_when_there_is_nothing_to_scan_against(tmp_path):
    project = Wiki(tmp_path / "project", name="project")
    lab = Wiki(tmp_path / "lab", name="lab", read_only=())
    harness = ContextHarness(
        project=project,
        lab=lab,
        transcripts=TranscriptLog(project.root / "transcripts"),
        client_for=lambda model: FakeLLM(["unused"]),
    )
    project.write("wiki/lessons/a.md", "Load cleveref last.", {"summary": "s", "portable": True})
    assert promote_lessons(harness) == [] and lab.pending_notes() == []
    (event,) = [e for e in harness.transcripts.events() if e["type"] == "promotion_blocked"]
    assert "not available to scan against" in event["reason"]


def test_portable_pages_are_promoted_from_any_section_and_tagged_with_their_run(tmp_path):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "unused"), lab=True, dream_mode="off")
    harness.project.write(
        "wiki/process/figures.md",
        "Save figures as vector graphics so text stays sharp.",
        {"summary": "s", "portable": True},
    )
    (first,) = promote_lessons(harness)
    token = harness.lab.read(first).meta["run"]
    assert len(str(token)) == 12 and "project" not in str(token)
    assert promote_lessons(harness) == []  # unchanged: not shared twice
    # The same run revising its lesson adds a note with the same run value, so the lab
    # consolidation can tell it is one run and not independent evidence.
    harness.project.write(
        "wiki/process/figures.md",
        "Save figures as vector graphics; raster only for photographs.",
        {"summary": "s", "portable": True},
    )
    (second,) = promote_lessons(harness)
    assert harness.lab.read(second).meta["run"] == token and second != first
    other = make_harness(tmp_path / "other", FakeLLM(lambda r: "unused"), lab=True)
    other.project.write("wiki/lessons/x.md", "Check overfull boxes.", {"portable": True})
    (third,) = promote_lessons(other)
    assert other.lab.read(third).meta["run"] != token


def test_lab_dream_prompt_names_the_run_of_each_note(tmp_path):
    prompts_seen: list[str] = []

    def responder(request: FakeRequest) -> str:
        prompts_seen.append(request.text)
        assert "count once as evidence" in request.system
        return "ok"

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True, dream_mode="end")
    harness.lab.add_note("Load cleveref last.", kind="lesson", meta={"run": "abc123abc123"})
    harness.consolidate_lab_memory()
    assert "run abc123abc123" in prompts_seen[0]


# -- checking claims against the record ---------------------------------------


def test_a_dream_never_finds_its_own_questions_in_the_transcript(tmp_path):
    answers: list[str] = []

    def responder(request: FakeRequest) -> str:
        # The note under review is quoted in this session's own prompt, and each search is
        # recorded before it runs.  Neither may come back as evidence.
        answers.append(
            request.call_tool("transcript_search", {"query": "zebra stripes", "exact": True})
        )
        answers.append(request.call_tool("transcript_search", {"query": "zebra stripes"}))
        answers.append(request.call_tool("transcript_search", {"query": "undefined cleverref"}))
        answers.append(
            request.call_tool("transcript_search", {"query": "zebra stripes", "exact": True})
        )
        return "Nothing supports the zebra note."

    harness = make_harness(tmp_path, FakeLLM(responder))
    harness.record("compile", stage="section_writing", errors=["Undefined cleverref package"])
    harness.project.add_note("the zebra stripes fix worked", stage="refinement", kind="lesson")
    report = harness.dream(stage="refinement", focus="check")
    assert report.llm_used
    assert answers[0] == answers[1] == answers[3] == "No matching transcript events."
    assert "Undefined cleverref package" in answers[2] and "dreamer" not in answers[2]
    # A second dream of the same run does not find the first one's traffic either.
    harness.project.add_note("another zebra stripes claim", stage="refinement", kind="lesson")
    answers.clear()
    harness.dream(stage="finalize", focus="check")
    assert answers[0] == "No matching transcript events."


def test_only_notes_the_session_actually_read_are_archived(tmp_path):
    long_fix = "Problem list. " + "x " * (MAX_INLINE_NOTE_CHARS // 2 + 10) + "THE FIX: use cleveref"
    opened: list[str] = []

    def responder(request: FakeRequest) -> str:
        assert "short note about tables" in request.text
        assert "THE FIX" not in request.text  # cut, and the prompt says so
        assert "only the beginning is shown here" in request.text
        for line in request.text.splitlines():
            if "second-long" in line:
                path = line.split()[1]
                opened.append(path)
                assert "THE FIX: use cleveref" in request.memory.view(path)
        return "done"

    harness = make_harness(tmp_path, FakeLLM(responder))
    harness.project.add_note("short note about tables", stage="s", title="short")
    harness.project.add_note(long_fix, stage="s", title="first-long")
    harness.project.add_note(long_fix, stage="s", title="second-long")
    report = harness.dream(stage="s", focus="f")
    assert report.notes_seen == 3 and report.notes_archived == 2 and len(opened) == 1
    (pending,) = harness.project.pending_notes()
    assert "first-long" in pending.path  # never read, so it waits for the next dream


def test_numbers_the_run_recorded_are_evidence_and_checked_numbers_do_not_reopen(tmp_path):
    client = FakeLLM(lambda request: "Looked at it; the number stays.")
    harness = make_harness(tmp_path, client)
    harness.record("compile", stage="section_writing", errors=["l.1423 Undefined control sequence"])
    harness.note("line 9999 is where it broke", stage="section_writing")  # a claim, not a record
    harness.project.write(
        "wiki/lessons/compile.md",
        "The build stopped at l.1423; someone claimed line 9999; the delta was 1,234.5.",
        {"summary": "s"},
    )
    flagged = unverified_numbers(harness.project, transcripts=harness.transcripts)
    assert flagged == [("wiki/lessons/compile.md", "9999"), ("wiki/lessons/compile.md", "1,234.5")]
    first = harness.dream(stage="section_writing", focus="check")  # one note pending: a session
    assert first.llm_used and len(first.unverified_numbers) == 2
    # The session saw those numbers and kept them.  They alone do not open another one.
    second = harness.dream(stage="refinement", focus="check")
    assert not second.llm_used and len(client.requests) == 1
    harness.project.write("wiki/lessons/new.md", "A fresh claim: 77.77.", {"summary": "s"})
    third = harness.dream(stage="refinement", focus="check")
    assert third.llm_used and "states 77.77" in client.requests[-1].text


def test_project_dreams_do_not_see_the_lab_memory(tmp_path):
    def responder(request: FakeRequest) -> str:
        assert "lab_memory_index" not in request.system and "/memories/lab" not in request.system
        assert "only what the source pages" in request.system  # the isolation rule
        assert "/memories/lab" not in request.memory.view("/memories")
        with pytest.raises(Exception, match="no lab memory"):
            request.memory.view("/memories/lab/wiki/lessons/other.md")
        return "ok"

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True)
    harness.lab.write("wiki/lessons/other.md", "From another paper.", {"summary": "s"})
    harness.note("n", stage="s")
    assert harness.dream(stage="s", focus="f").error is None


def test_a_lock_left_by_a_dead_process_is_taken_over_at_once(tmp_path):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "ok"))
    harness.note("n", stage="s")
    finished = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True
    )
    dead_pid = int(finished.stdout)
    lock = harness.project.root / LOCK_FILE
    lock.write_text(f"{dead_pid} {socket.gethostname()}\n")
    report = dream(harness, stage="s", focus="f", model="fake", use_model=True, lock_wait=0)
    assert report.error is None and report.llm_used and not lock.exists()
    # A live holder, or one on another machine, is respected until the lock goes stale.
    for holder in (f"{os.getppid()} {socket.gethostname()}", f"{dead_pid} some-other-host"):
        harness.note("n", stage="s")
        lock.write_text(holder + "\n")
        busy = dream(harness, stage="s", focus="f", model="fake", use_model=True, lock_wait=0)
        assert "in progress" in busy.error
        lock.unlink()


def test_a_missing_credential_is_reported_without_a_traceback(tmp_path, caplog):
    def no_client(model: str):
        raise LLMConfigError("ANTHROPIC_API_KEY is not set")

    project = Wiki(tmp_path / "project", name="project")
    harness = ContextHarness(
        project=project,
        transcripts=TranscriptLog(project.root / "transcripts"),
        client_for=no_client,
        dream_model="fake",
    )
    harness.note("n", stage="s")
    report = harness.dream(stage="s", focus="f")
    assert "ANTHROPIC_API_KEY is not set" in report.error
    assert "Traceback" not in caplog.text and "ANTHROPIC_API_KEY is not set" in caplog.text


def test_notes_written_while_a_dream_runs_stay_pending(tmp_path):
    holder: dict = {}

    def responder(request: FakeRequest) -> str:
        # What the plotting stage does while the literature synthesis dream is running.
        holder["late"] = holder["harness"].note("figure 3 failed twice", stage="plotting")
        return "Consolidated."

    project = Wiki(tmp_path / "project", name="project")
    harness = ContextHarness(
        project=project,
        transcripts=TranscriptLog(project.root / "transcripts", session_id="s1"),
        client_for=lambda model: FakeLLM(responder),
        dream_model="fake",
    )
    holder["harness"] = harness
    harness.note("an earlier observation", stage="literature")
    report = harness.dream(stage="literature", focus="f")
    assert report.llm_used and report.notes_seen == 1 and report.notes_archived == 1
    assert [info.path for info in project.pending_notes()] == [holder["late"]]


def test_leak_scan_details(tmp_path):
    assert "credential" in leak_reason("The run used sk-ant-api03-abcdefghijklmnop.", [])
    assert "path" in leak_reason("The build ran in /Users/alice/paper.", [])

    # An experimental log too large for one page is stored in parts; the scan must read them.
    project = Wiki(tmp_path / "project", name="project", max_page_bytes=6_000)
    lab = Wiki(tmp_path / "lab", name="lab", read_only=())
    harness = ContextHarness(
        project=project,
        lab=lab,
        transcripts=TranscriptLog(project.root / "transcripts", session_id="s1"),
        client_for=lambda model: FakeLLM(lambda r: "unused"),
        dream_mode="off",
    )
    log_text = "".join(f"| run {i} | {i * 1.375:.3f} |\n" for i in range(900))
    harness.write_page("wiki/sources/experimental_log.md", log_text, {"summary": "raw results"})
    assert any("-part-" in info.path for info in project.list("wiki/sources"))
    project.write(
        "wiki/lessons/number.md",
        "Quoting 1234.750 in the abstract raised Clarity.",
        {"summary": "s", "portable": True},
    )
    assert "1234.750" in log_text
    assert promote_lessons(harness) == []


# -- the scan, spelling by spelling ------------------------------------------------------


def test_numbers_are_found_in_the_spellings_results_are_written_in():
    sources = [
        "2.3x faster with 110M parameters at 145ms; $87.3_{\\pm 0.2}$ and an F1 of .568; "
        "a 9.23pp gain."
    ]
    for text in (
        "A 2.3 times speed-up.",
        "With 110 million parameters.",
        "Latency of 145 ms.",
        "Report 87.3 in the abstract.",
        "An F1 of 0.568.",
        "A gain of 9.23 points.",
        "The 9.23pp gain.",
        "Write $87.3_{\\pm 0.2}$ in the table.",
    ):
        assert "number" in leak_reason(text, sources), text
    # The same result as a fraction or a percentage, rounded, or with a decimal comma.
    results = ["Seen Jaccard 43.43, F-score 0.568."]
    for text in (
        "A Jaccard of 0.4343 belongs in the abstract.",
        "An F-score of 56.8% belongs in the abstract.",
        "A Jaccard of 43.4 is the headline.",
        "Putting 43,43 in the abstract raised Clarity.",
    ):
        assert "number" in leak_reason(text, results), text
    assert leak_reason("See sections 3,4 and use 43 points of space, or 0.5.", results) is None
    # Lengths and layout factors are not results, even when the digits coincide.
    layout = "Overfull \\hbox (12.3pt too wide); use \\vspace{2.3em}, 110mm and 0.568\\linewidth."
    assert leak_reason(layout, ["12.3 and 2.3 and 110 and 0.568"]) is None


def test_names_are_recognised_in_identifiers_plurals_and_with_suffixes():
    scanner = LeakScanner(
        [
            "We evaluate TSAM on CIFAR-10 with PromptWeaver and report mAP; the attention map "
            "is sparse. \\FloatBarrier \\DeclareMathOperator{\\argmax}{argmax}"
        ]
    )
    for text in (
        "See TSAM_overview.png for the layout.",
        "The label fig:tsam_overview was undefined.",
        "TSAMs differ in depth.",
        "TSAM-B is the larger variant.",
        "CIFAR-100 needs another table.",
        "PromptWeavers were compared.",
        "Reviewers asked how mAP was computed.",
        "Reviewers rewarded the tsam ablation.",
        "Reviewers rewarded the Tsam ablation.",
        "Explain the Prompt Weaver before the experiments.",
        "The prompt_weaver figure was too small.",
        "Typeset \\textsc{Prompt}Weaver consistently.",
        "Explain TSAM2 before the method.",
    ):
        assert "names something" in scanner.reason(text), text
    # Ordinary words, and LaTeX commands the paper happens to use, are not names.
    assert scanner.reason("Map each figure to a section; a bit of care helps.") is None
    assert scanner.reason("Put \\FloatBarrier before the appendix.") is None
    assert scanner.reason("Use \\DeclareMathOperator for upright operators.") is None


def test_names_a_paper_coins_in_plain_capitals_are_protected():
    scanner = LeakScanner(
        [
            "\\title{Falcon: Fast Routing for Long Inputs}\n"
            "We propose Kestrel, a sparse router. We present Table 2 and call our method Osprey. "
            "The table lists every run."
        ]
    )
    assert "Falcon" in scanner.reason("Reviewers rewarded the Falcon ablation.")
    assert "Kestrel" in scanner.reason("Introduce Kestrel in the first sentence.")
    assert "Osprey" in scanner.reason("Osprey needs a runtime table.")
    assert scanner.reason("Table captions go above the table; Fast builds help.") is None


def test_the_pipelines_own_record_formats_are_not_paper_wording(tmp_path):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "unused"), lab=True, dream_mode="off")
    harness.write_page(
        "wiki/reviews/v2.md",
        "Summary: the submission studies sparse routing for very long input sequences.\n\n"
        "## Outcome\n\nOverall 5 -> 6; sub-axis gain 1, drop 0 (Clarity +1)",
    )
    workdir = tmp_path / "run" / "content_refinement_workdir"
    workdir.mkdir(parents=True)
    (workdir / "content_refinement_worklog.json").write_text(
        json.dumps(
            {
                "iterations": [
                    {
                        "problems": ["Undefined control sequence in the second results table"],
                        "scores": {
                            "Originality": 3,
                            "Quality": 3,
                            "Clarity": 2,
                            "Significance": 3,
                            "Soundness": 3,
                            "Presentation": 2,
                            "Contribution": 3,
                        },
                        "summary": "Reviewers want the routing budget assumption stated early.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    harness.protect(workdir)
    lessons = {
        "outcome": "Answering one question moved Overall 5 -> 6; sub-axis gain 1, drop 0.",
        "scores": "Track Originality, Quality, Clarity, Significance, Soundness, Presentation "
        "and Contribution separately.",
        "error": "Undefined control sequence in the second results table means a lost package.",
        "worklog-prose": "Reviewers want the routing budget assumption stated early, so do it.",
        "review-prose": "The submission studies sparse routing for very long input sequences.",
    }
    for name, text in lessons.items():
        harness.project.write(f"wiki/lessons/{name}.md", text, {"summary": name, "portable": True})
    added = promote_lessons(harness)
    shared = "\n".join(harness.lab.read(note).body for note in added)
    assert len(added) == 3 and "sub-axis gain" in shared and "Soundness" in shared
    blocked = {e["path"] for e in harness.transcripts.events() if e["type"] == "promotion_blocked"}
    assert blocked == {"wiki/lessons/worklog-prose.md", "wiki/lessons/review-prose.md"}
    assert harness.last_promotion == {"shared": 3, "withheld": 2}


# -- promotion bookkeeping ----------------------------------------------------------------


def test_a_lesson_is_shared_once_whatever_happens_to_its_page_or_the_ledger(tmp_path):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "unused"), lab=True, dream_mode="off")
    project = harness.project
    project.write("wiki/lessons/a.md", "Load cleveref last.", {"summary": "s", "portable": True})
    assert len(promote_lessons(harness)) == 1
    # A consolidation renames the page: still the same lesson.
    project.rename("wiki/lessons/a.md", "wiki/lessons/cleveref.md")
    assert promote_lessons(harness) == []
    # A ledger written by an earlier version held digests only; they are not forgotten
    # when a new lesson is recorded.
    ledger = project.root / PROMOTED_LEDGER
    digests = list(json.loads(ledger.read_text())["pages"].values())
    ledger.write_text(json.dumps(digests))
    project.write("wiki/lessons/b.md", "Check overfull boxes.", {"summary": "s", "portable": True})
    assert len(promote_lessons(harness)) == 1
    assert promote_lessons(harness) == [] and len(harness.lab.pending_notes()) == 2


@pytest.mark.parametrize("flag", ["True", "yes", "TRUE", '"true"', "1"])
def test_portable_is_understood_however_the_model_spells_it(tmp_path, flag):
    harness = make_harness(tmp_path, FakeLLM(lambda r: "unused"), lab=True, dream_mode="off")
    harness.project.write_text(
        "wiki/lessons/a.md", f"---\nsummary: s\nportable: {flag}\n---\nLoad cleveref last."
    )
    harness.project.write_text(
        "wiki/lessons/b.md", "---\nsummary: s\nportable: false\n---\nOnly for this paper."
    )
    (note,) = promote_lessons(harness)
    assert "cleveref" in harness.lab.read(note).body


def test_finish_reports_how_many_lessons_were_shared_and_withheld(tmp_path):
    def responder(request: FakeRequest) -> str:
        if request.label == "dream.project":
            request.memory.create(
                "/memories/wiki/lessons/good.md",
                "---\nsummary: s\nportable: true\n---\nLoad cleveref last.",
            )
            request.memory.create(
                "/memories/wiki/lessons/bad.md",
                "---\nsummary: s\nportable: true\n---\nLead with the 91.2 accuracy.",
            )
        return "ok"

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True, dream_mode="end")
    harness.note("n", stage="s")
    project_report = harness.finish()[0]
    assert (project_report.lessons_shared, project_report.lessons_withheld) == (1, 1)
    assert project_report.to_dict()["lessons_withheld"] == 1


# -- the lab session's guard --------------------------------------------------------------


def test_a_refused_lab_write_keeps_the_notes_pending_and_other_runs_lessons_pass(tmp_path):
    attempts: list[str] = []

    def responder(request: FakeRequest) -> str:
        if request.label != "dream.lab":
            return "ok"
        # Folding another run's lesson, which names something this paper also uses.
        request.memory.create(
            "/memories/wiki/lessons/optimisers.md",
            "---\nsummary: optimiser notes\n---\nFooNet-style warmup tables need a units row.",
        )
        attempts.append("foreign lesson written")
        # Merging into a page by rewriting it: the lines already there are not on trial.
        request.memory.create(
            "/memories/wiki/lessons/old.md",
            "---\nsummary: old\n---\nFooNet was the running example in 2023.\nUse booktabs.",
        )
        attempts.append("existing page extended")
        # But this paper's content is still refused, in the text and in the file name.
        with pytest.raises(Exception, match="names something"):
            request.memory.create("/memories/wiki/lessons/new.md", "FooNet ablations go first.")
        with pytest.raises(Exception, match="names something"):
            request.memory.rename(
                "/memories/wiki/lessons/optimisers.md", "/memories/wiki/lessons/foonet-tips.md"
            )
        return "done"

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True, dream_mode="end")
    harness.write_page("wiki/sources/idea.md", "We propose FooNet for long video understanding.")
    harness.lab.write(
        "wiki/lessons/old.md", "FooNet was the running example in 2023.", {"summary": "old"}
    )
    foreign = harness.lab.add_note(
        "FooNet-style warmup tables need a units row.",
        kind="lesson",
        meta={"run": "0123456789ab"},
    )
    report = harness.consolidate_lab_memory()
    assert attempts == ["foreign lesson written", "existing page extended"]
    assert report.llm_used and harness.lab_vetoes == 2
    # A write was refused, so nothing is archived: the lesson waits for a consolidation
    # run for another paper, which will not collide with this one.
    assert report.notes_archived == 0
    assert [info.path for info in harness.lab.pending_notes()] == [foreign]
    assert [e["type"] for e in harness.transcripts.events()].count("lab_write_blocked") == 2
