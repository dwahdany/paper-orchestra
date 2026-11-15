from __future__ import annotations

import os
import time

from paper_orchestra.llm.base import Capabilities, LLMError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, TranscriptLog, Wiki
from paper_orchestra.memory.dreaming import LOCK_FILE, STALE_LOCK_SECONDS, dream


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


def test_finish_promotes_process_lessons_to_the_lab(tmp_path):
    seen: list[str] = []

    def responder(request: FakeRequest) -> str:
        seen.append(request.label)
        if request.label == "dream.lab":
            assert "shared by every paper" in request.system
            assert "generalise" in request.text.lower()
            try:
                request.memory.create("/memories/wiki/lessons/x.md", "project is read-only here")
            except Exception as exc:
                assert "read-only" in str(exc)
            request.memory.create(
                "/memories/lab/wiki/lessons/latex.md",
                "---\nsummary: LaTeX pitfalls\n---\nUse cleveref (the package), never cleverref.",
            )
            return "Promoted one LaTeX lesson."
        request.memory.create("/memories/wiki/lessons/compile.md", "---\nsummary: compile\n---\nx")
        return "Consolidated."

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True, dream_mode="end")
    harness.note("cleverref typo broke the build", stage="section_writing", kind="lesson")
    project_report, lab_report = harness.finish()
    assert seen == ["dream.project", "dream.lab"]
    assert project_report.notes_archived == 1 and project_report.scope == "project"
    assert lab_report.scope == "lab" and lab_report.pages_changed == ["wiki/lessons/latex.md"]
    assert "[[lessons/latex]]: LaTeX pitfalls" in harness.lab.injected_context()
    assert not harness.project.exists("wiki/lessons/x.md")


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
